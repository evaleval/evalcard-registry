"""Shared seed-YAML I/O helpers for the generator/refresh scripts.

These two helpers were duplicated verbatim across the model-source generators
(refresh_from_hub_stats, freeze_hub_stats_cache, fold_modelsdev_dupes,
generate_hf_oracle_seed, generate_tier3_inferred_seed). Single-sourcing them
here — in the installed package, so the scripts import them with no sys.path
manipulation — keeps the candidate-loading and org-map construction
byte-identical across all of them.
"""
from __future__ import annotations

from pathlib import Path

import yaml

# Scalar fields a generated enrich record may donate under its `weak:` map
# (written by scripts/refresh_from_modelsdev.py when a mint is suppressed,
# folded, or core-skipped). The seed loader applies weak values LAST: a weak
# value fills a field only when every full entry left it empty AND core.yaml
# does not explicitly carry the key (even as null). Single-sourced here so the
# generator and the loader cannot drift on the field set.
WEAK_SCALAR_FIELDS = (
    "release_date",
    "open_weights",
    "params_billions",
    "input_modalities",
    "output_modalities",
    "architecture",
)


def safe_load_yaml(text: str):
    """Parse trusted seed YAML with libyaml acceleration when available.

    ``yaml.safe_load`` always selects the Python ``SafeLoader``. These source
    files are read dozens of times by the refresh regression suite, so using
    the API-equivalent ``CSafeLoader`` cuts minutes from cron verification.
    Environments without libyaml retain the same safe Python fallback.
    """
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    return yaml.load(text, Loader=loader)


def resolve_oracle_path(name: str = "hf_model_id_resolution.json") -> Path:
    """Absolute path to a frozen curation input (the HF oracle JSON by default).

    Prefers the in-repo committed copy under ``curation/`` so the generators and
    the gate find it in a single-repo CI checkout (where the evaleval workspace
    root is not part of the registry checkout); falls back to the workspace-parent
    location for local dev where that copy is the shared source of truth. This is
    the single source of the resolution order — the gate suite and the generator
    scripts both call it (so they cannot drift apart)."""
    repo_root = Path(__file__).resolve().parents[3]
    tracked = repo_root / "curation" / name
    return tracked if tracked.exists() else repo_root.parent / name


def load_entries_from_yaml(path: Path) -> list[dict]:
    """Entry list from a seed-model YAML file. Handles both shapes — a flat
    list, or a ``{skip_ids, skip_source_ids, entries}`` dict (core.yaml).
    Returns ``[]`` for a missing file so callers can iterate over optional
    generated sources without a per-path existence check."""
    if not path.exists():
        return []
    raw = safe_load_yaml(path.read_text()) or []
    return (raw.get("entries") if isinstance(raw, dict) else raw) or []


def build_hf_to_dev_from_orgs_yaml(orgs_path: Path) -> dict[str, str]:
    """HF-org-lowercase -> curated developer slug, via the single shared builder
    ``eval_entity_resolver.fold.build_curated_org_map`` (``_ORG_ALIASES`` UNION
    every curated org's id / hf_org / ``aliases``). Reading the ALIAS tier (not
    just ``hf_org``) is what folds minimaxai->minimax, EnnoAi->Enno-Ai,
    ai2->allenai, etc. — so the generators, resolver, and gate all agree on one
    org map."""
    from eval_entity_resolver.fold import build_curated_org_map

    orgs = safe_load_yaml(orgs_path.read_text()) or [] if orgs_path.exists() else []
    return build_curated_org_map(orgs)
