from dataclasses import dataclass
from typing import Literal, Optional


ResolutionStrategy = Literal["exact", "normalized", "fuzzy", "no_match"]
# Resolution mode: "resolve" runs the full chain incl. fuzzy inference;
# "exact" stops after the exact/normalized alias and HF-id-check steps and
# returns no_match instead of guessing (and callers must not auto-create).
ResolveMode = Literal["resolve", "exact"]
# `composite` and `family` are first-class resolvable entity types
# (they resolve against canonical_composites / canonical_families).
# `slice`/subset is deliberately NOT a type — it stays a parent-only
# alias-fold onto its parent benchmark; a slice match is surfaced
# as resolution detail, not as its own entity.
EntityType = Literal[
    "model", "benchmark", "metric", "harness", "org", "composite", "family"
]


def looks_like_hf_id(raw_value: str) -> bool:
    """HF id heuristic: contains a single `/` with non-empty parts on both
    sides. Conservative — won't trigger HF-id checks for bare model names or
    paths with multiple slashes (which are likely malformed)."""
    if not raw_value or raw_value.count("/") != 1:
        return False
    org, name = raw_value.split("/", 1)
    return bool(org.strip()) and bool(name.strip())


@dataclass
class HfIdHit:
    """A confirmed HF repo id from an injected `hf_id_checker`.

    - `hf_id`: the repo id in HF-true casing.
    - `verbatim`: True when the raw value equals `hf_id` case-insensitively
      (HF repo ids are case-insensitively unique, so this identifies the repo
      exactly); False when the match was a separator-collapse (normalized).
    - `source`: where the confirmation came from — `hub_stats_index` (the
      cron-built local index) or `hf_live` (a live Hub API check). Maps
      directly onto `resolution_source` when the hit wins for an id the
      registry doesn't know.
    """
    hf_id: str
    verbatim: bool
    source: str


@dataclass
class ResolutionResult:
    """Outcome of one `Resolver.resolve` call.

    Core matching fields (always populated):
      - `raw_value`, `entity_type`, `source_config`: echo of the inputs
      - `canonical_id`: the matched canonical (None on no_match). For
        models with a `root_model_id` set, this is the IDENTITY ROOT —
        i.e. the unquantized base — so callers reasoning about
        same-identity quantizations get one canonical instead of N.
      - `strategy`: which matcher fired (or "no_match")
      - `confidence`: 0.0–1.0

    Enrichment fields (populated when the `Resolver` is constructed
    with a `CanonicalStore`; otherwise None):
      - `review_status`: review state of the matched canonical
      - `parent_canonical_id`: family/variant parent (for models, the
        `variant` edge in `parents`; for benchmarks/orgs, the
        `parent_*_id` scalar column).
      - `resolved_leaf_id`: the originally-matched canonical before
        any root-collapse. Equals `canonical_id` when no quantized
        chain. Models only.
      - `root_model_id`: identity root via quantized-only walk. NULL
        when the matched leaf IS the root. Models only. DEPRECATED
        output alias — equals `model_group_id`; drop once the producer
        is live.
      - `model_group_id`: identity-group root (fold {version, quantized,
        mode}); the rename target of `root_model_id`. Models only.
      - `model_family_id`: family-release root (fold the versioned
        release line). Models only.
      - `lineage_origin_model_id`: deepest non-variant ancestor's id
        (what it was built from). Models only.
      - `lineage_origin_org_id`: deepest non-variant ancestor's
        org_id. Models only. DEPRECATED output alias — equals
        `lineage_origin_model_org_id`.
      - `lineage_origin_model_org_id`: deepest non-variant ancestor's
        org_id; the rename target of `lineage_origin_org_id`. Models only.
      - `inference_platform`: serving platform (FK→inference_platforms.id).
        Models only.
      - `resolution_source`: enum {hf|models_dev|curated|inferred|none}.
      - `resolution_granularity`: enum {variant|group|family}.
      - `parents`: full typed-edge list of the matched leaf. Models only.
      - `open_weights`: True/False/None. Models only.
      - `release_date`: YYYY-MM or YYYY-MM-DD. Models only.
      - `params_billions`: approximate parameter count. Models only.
      - `family_key`: canonical_families.id this benchmark belongs to.
        Defaults to the benchmark's own id for singleton families
        (when no curated multi-benchmark family covers it). Benchmarks
        only.
      - `composite_keys`: canonical_composites.id values where this
        benchmark appears (via the composite's source_configs ↔ EEE
        folders chain). Benchmarks only; empty list when none.
      - `category`: curated single-valued category from the family
        (general / agentic / reasoning / knowledge / multimodal /
        tool-use / math / security / factuality / reward-modelling /
        safety / code / instruction-following / other). Benchmarks
        only; None when no category curated.
    """
    raw_value: str
    entity_type: EntityType
    source_config: Optional[str]
    canonical_id: Optional[str]
    strategy: ResolutionStrategy
    confidence: float
    # Enrichment fields — None when no CanonicalStore is attached.
    review_status: Optional[str] = None
    parent_canonical_id: Optional[str] = None
    resolved_leaf_id: Optional[str] = None
    root_model_id: Optional[str] = None
    lineage_origin_org_id: Optional[str] = None
    # Extended lineage / provenance fields (all Optional[str]=None).
    # `model_group_id` / `lineage_origin_model_org_id` are the rename targets
    # of the deprecated `root_model_id` / `lineage_origin_org_id` (kept above
    # for compat).
    model_group_id: Optional[str] = None
    model_family_id: Optional[str] = None
    lineage_origin_model_id: Optional[str] = None
    lineage_origin_model_org_id: Optional[str] = None
    inference_platform: Optional[str] = None
    resolution_source: Optional[str] = None
    resolution_granularity: Optional[str] = None
    parents: Optional[list[dict]] = None
    open_weights: Optional[bool] = None
    release_date: Optional[str] = None
    params_billions: Optional[float] = None
    # Benchmark-only enrichment.
    family_key: Optional[str] = None
    composite_keys: Optional[list[str]] = None
    category: Optional[str] = None
    # --- Hierarchy contract (type-agnostic ancestry + typed detail) ---
    # `ancestry`: ordered list of `{canonical_id, level}` from the matched
    # entity's IMMEDIATE PARENT up to the root. `[]` when self is a root.
    #   model     -> e.g. [{group}, {family}]
    #   benchmark -> e.g. [{family}, {composite}]
    #   family    -> e.g. [{composite}]
    #   composite/metric/harness/org -> [] (roots)
    # Computed by `CanonicalStore.compute_ancestry` from the existing
    # graph tables (model group/family walk; benchmark→family via
    # canonical_families.benchmark_ids; family→composite via
    # canonical_families.composite_keys / canonical_composites.family_id).
    ancestry: Optional[list[dict]] = None
    # `resolution_detail`: typed sub-object keyed by entity_type.
    #   model     -> {"granularity": variant|group|family,
    #                 "hf_repo_id": HF repo id when runtime-attested (oracle/
    #                 hub-stats), else None — null is "not attested", not "not on HF"}
    #   benchmark -> {"level": composite|family|benchmark|slice,
    #                 "matched_subset": str|None}
    #   harness   -> {"harness_version_stripped": str, "bare_name": str,
    #                 "bare_tier": "exact"|"normalized"} — ONLY when the
    #                 resolver stripped a trailing version token to reach the
    #                 match (`lm_eval 0.4.12` -> `lm-evaluation-harness`);
    #                 `{}` for a harness hit on the alias tiers and on a
    #                 no-match.
    #   composite|family|metric|org -> {} (reserved)
    resolution_detail: Optional[dict] = None
    # The repo id the injected HF id checker confirmed during this resolve
    # (HF-true casing), when it did. Set on checker-won results AND on
    # alias-won results the checker agreed with. The service layer uses it to
    # let checker-backed results outrank a stored alias without guessing from
    # strategy/source fields. Never serialized over HTTP.
    hf_attestation: Optional[str] = None
    # True when the match came from the HF id check and the confirmed repo id
    # is NOT a registered canonical. The service layer uses this to route the
    # result into auto-create (adopting the HF-true id) in write mode, and to
    # avoid emitting FK references (aliases, parent edges, eval_results rows)
    # to an entity that doesn't exist. Never serialized over HTTP.
    hf_attested_unregistered: bool = False


@dataclass
class ResolverConfig:
    threshold: float = 0.85
