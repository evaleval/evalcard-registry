"""Every id in the flat seed files is defined exactly once.

The seed loader upserts by id (`queries.upsert_entity`), so a second block
with the same id silently overwrites the first, and a duplicated mapping key
in families.yaml / composites.yaml is silently last-wins at YAML load time.
Neither raises. Two rebases in one week each produced such a duplicate
(`length-controlled-win-rate` in #55, the perplexity family in #60) because
the colliding blocks sat thousands of lines apart and merged cleanly. The
existing duplicate gates cover canonical model and org ids only; this one
covers the flat seeds, the model enrichment layers and the generated
benchmark files. Reads the YAML directly so it runs without built fixtures.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pytest
import yaml

SEED = Path(__file__).resolve().parent.parent / "seed"

# Files whose top level is a list of {id: ...} entries.
LIST_SEEDS = ("metrics.yaml", "benchmarks.yaml", "harnesses.yaml", "orgs.yaml",
              "inference_platforms.yaml", "models/enrichments/aliases.yaml",
              "models/enrichments/parents.yaml",
              *sorted(p.relative_to(SEED).as_posix()
                      for p in (SEED / "benchmarks_generated").glob("*.yaml")))
# Files whose top level is a {slug: {...}} mapping.
MAPPING_SEEDS = ("families.yaml", "composites.yaml")


class _DuplicateKey(Exception):
    pass


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that refuses a mapping with a repeated key instead of
    keeping the last value."""


def _construct_mapping(loader, node, deep=False):
    seen = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise _DuplicateKey(f"{key!r} (line {key_node.start_mark.line + 1})")
        seen[key] = loader.construct_object(value_node, deep=deep)
    return seen


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def _loader_key(value: str) -> str:
    """The seed loader's collision key (`_check_benchmark_collisions._norm`):
    NFKD, strip combining marks, casefold, drop non-alphanumerics. Two ids
    that agree under it (`micro-f1` / `micro_f1` / `MicroF1`) are one entity
    split in two, which the exact-string check would miss."""
    decomposed = unicodedata.normalize("NFKD", str(value))
    base = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]", "", base.casefold())


@pytest.mark.parametrize("name", LIST_SEEDS)
def test_list_seed_ids_are_unique(name):
    path = SEED / name
    if not path.exists():
        pytest.skip(f"{name} not present")
    entries = yaml.safe_load(path.read_text()) or []
    assert isinstance(entries, list), f"{name}: expected a top-level list"
    ids = [str(e.get("id")) for e in entries if isinstance(e, dict)]
    exact = sorted(k for k, v in Counter(ids).items() if v > 1)
    assert exact == [], f"{name}: id defined more than once: {exact}"
    by_key = defaultdict(list)
    for i in ids:
        by_key[_loader_key(i)].append(i)
    split = sorted(v for v in by_key.values() if len(v) > 1)
    assert split == [], f"{name}: ids that differ only by case/separators: {split}"


@pytest.mark.parametrize("name", MAPPING_SEEDS)
def test_mapping_seed_keys_are_unique(name):
    path = SEED / name
    if not path.exists():
        pytest.skip(f"{name} not present")
    try:
        yaml.load(path.read_text(), Loader=_StrictLoader)
    except _DuplicateKey as exc:
        pytest.fail(f"{name}: key defined more than once: {exc}")
