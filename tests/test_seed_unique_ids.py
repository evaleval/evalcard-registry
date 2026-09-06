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

from collections import Counter, defaultdict
from pathlib import Path

import pytest
import yaml

from eval_card_registry.cli import seed_collision_key

SEED = Path(__file__).resolve().parent.parent / "seed"

# Files whose top level is a list of {id: ...} entries.
LIST_SEEDS = ("metrics.yaml", "benchmarks.yaml", "harnesses.yaml", "orgs.yaml",
              "inference_platforms.yaml", "models/enrichments/aliases.yaml",
              "models/enrichments/parents.yaml",
              "models/enrichments/upstream_corrections.yaml",
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
        by_key[seed_collision_key(i)].append(i)
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


# Flat entity seeds whose entries carry review_status and (for metrics) bounds.
# models/core.yaml is not gated: 15 of its entries declare review_status: null
# on purpose (generator-owned rows awaiting a decision) and the generated
# benchmark files use a third value, `auto`; both need their own rule.
ENTITY_SEEDS = ("metrics.yaml", "benchmarks.yaml", "harnesses.yaml", "orgs.yaml")


@pytest.mark.parametrize("name", ENTITY_SEEDS)
def test_every_entity_declares_a_review_status(name):
    """The flat-seed loader stores a missing review_status as NULL (only the
    families / composites loaders default). A NULL row matches neither
    `?review_status=draft` nor `=reviewed`, so it drops out of the review
    queue and of stale-removal alike; `ter` reached a branch that way."""
    entries = yaml.safe_load((SEED / name).read_text()) or []
    missing = sorted(str(e.get("id")) for e in entries
                     if isinstance(e, dict) and e.get("review_status") not in ("draft", "reviewed"))
    assert missing == [], f"{name}: entries without a valid review_status: {missing}"


def test_metric_bounds_are_well_formed():
    """`.inf` / `-.inf` mean unbounded by definition, null means not stated
    (seed/metrics.yaml header). NaN is never a bound, +inf is never a lower
    bound, -inf never an upper one, and a stated pair is ordered."""
    import math

    entries = yaml.safe_load((SEED / "metrics.yaml").read_text()) or []
    bad = []
    for e in entries:
        lo, hi = e.get("min_score"), e.get("max_score")
        for k, v in (("min_score", lo), ("max_score", hi)):
            if isinstance(v, float) and math.isnan(v):
                bad.append((e["id"], k, "NaN"))
        if isinstance(lo, float) and lo == math.inf:
            bad.append((e["id"], "min_score", "+inf"))
        if isinstance(hi, float) and hi == -math.inf:
            bad.append((e["id"], "max_score", "-inf"))
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and lo > hi:
            bad.append((e["id"], "bounds", f"{lo} > {hi}"))
    assert bad == [], f"malformed metric bounds: {bad}"
