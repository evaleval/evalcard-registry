"""Wraps the canonical entity tables (models / benchmarks / metrics /
harnesses / orgs) for the resolver to enrich its results with metadata
beyond just the matched canonical_id.

The resolver package is intentionally small — alias matching is its core
job. But callers consistently want richer return values: the matched
entity's `review_status`, parent edges, model-specific lineage fields,
quantized-chain root collapse. Putting that lookup logic here means any
caller of the bare `Resolver` gets the same response shape as the HTTP
API, without duplicating logic in the service wrapper.

Structure mirrors `AliasStore`: lazy loading from parquet/HF, an empty
fallback when the underlying file is missing, and read-only lookup
methods. Writes are out of scope — entity creation is a service-side
concern (the resolver doesn't auto-draft anything)."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Per-entity-type parquet filenames (matches the eval-card-registry
# fixtures layout / HF Dataset config naming).
_TABLES = {
    "model": "canonical_models",
    "benchmark": "canonical_benchmarks",
    "metric": "canonical_metrics",
    "harness": "eval_harnesses",
    "org": "canonical_orgs",
    # families and composites are first-class registry entities.
    # Resolution lookups don't query them directly, but the resolver
    # enrichment for a `benchmark` consults `canonical_families` to
    # populate `ResolutionResult.family_key` and `category`.
    "family": "canonical_families",
    "composite": "canonical_composites",
}

# The `parent_*` column for each entity type that carries the
# in-family parent id (used for non-model types). Models use the typed
# `parents` JSON list instead — see `decode_parents`.
_PARENT_FIELD = {
    "benchmark": "parent_benchmark_id",
    "org": "parent_org_id",
}


# ---------------------------------------------------------------------------
# Helpers (pure, exported for reuse — service wrapper imports these)
# ---------------------------------------------------------------------------

def _is_na(value) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _na_to_none(value: Any) -> Any:
    return None if _is_na(value) else value


def _safe_json_load(s: str, default: Any = None) -> Any:
    """Tolerant `json.loads`: returns `default` on any decode/type error
    instead of raising. Used wherever a JSON-encoded parquet column may
    hold malformed or unexpected content."""
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return default


def _hf_repo_id_of(ent: Optional[dict], canonical_id: Optional[str]) -> Optional[str]:
    """The matched model's Hugging Face repo id when the registry can ATTEST it
    at runtime, else None.

    Attested = sourced from the HF oracle (`resolution_source == "hf"`) OR
    hub-stats confirmed that exact id (`metadata.hf_id == canonical_id`) — the
    runtime-available half of the gate's `has_real_hf_id` predicate. A shadow slug
    (e.g. a `kimi/...` slug whose `metadata.hf_id` points at a DIFFERENT
    `moonshotai/...` repo) is correctly excluded — its own id is not the real repo.

    None is CONSERVATIVE: it means "not runtime-attested", NOT "not on HF". A
    genuine HF repo whose provenance isn't verifiable at runtime (a models.dev-only
    or name-inferred entry carrying no `hf_id`) also returns None. So a non-null
    value is reliable; a null is the absence of attestation, not a negative claim."""
    if not ent or not canonical_id:
        return None
    if _na_to_none(ent.get("resolution_source")) == "hf":
        return canonical_id
    md = _na_to_none(ent.get("metadata"))
    if isinstance(md, str) and md.strip():
        # metadata is free-form JSON — guard against a value that parses to a
        # non-dict (list/str/number) so `.get` can't raise and 500 the resolve.
        md_obj = _safe_json_load(md, {})
        if isinstance(md_obj, dict) and md_obj.get("hf_id") == canonical_id:
            return canonical_id
    return None


def decode_parents(value) -> list[dict]:
    """Decode `canonical_models.parents` (JSON-encoded list-of-edges) to a
    Python list. Tolerant of NA/NaN, None, empty strings, and pre-decoded
    lists. Returns [] for any unparseable input."""
    if _is_na(value) or value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s or s in ("[]", "null"):
            return []
        decoded = _safe_json_load(s, default=[])
        return list(decoded) if isinstance(decoded, list) else []
    return []


def variant_parent_id(parents: list[dict]) -> Optional[str]:
    """Return the id of the first `variant` edge in a parents list, or
    None. Feeds the ResolutionResult `parent_canonical_id` field, which
    exposes a model's immediate family / variant parent."""
    for edge in parents:
        if isinstance(edge, dict) and edge.get("relationship") == "variant":
            pid = edge.get("id")
            if pid:
                return pid
    return None


def _kwarg_for(entity_type: str) -> str:
    """Map an entity_type to its CanonicalStore constructor kwarg name.
    Constructor uses canonical English plurals (`models_df`,
    `benchmarks_df`, `harnesses_df`, `families_df`, `composites_df`),
    not the simpler `<type>s_df` rule that breaks on `harness`/`family`."""
    return {
        "model": "models_df",
        "benchmark": "benchmarks_df",
        "metric": "metrics_df",
        "harness": "harnesses_df",
        "org": "orgs_df",
        "family": "families_df",
        "composite": "composites_df",
    }[entity_type]


# ---------------------------------------------------------------------------
# CanonicalStore
# ---------------------------------------------------------------------------

class CanonicalStore:
    """Read-only access to the canonical entity tables. Holds one
    DataFrame per entity type; provides `lookup(entity_type, id)` for
    O(1) row retrieval. Empty tables are valid — `lookup` just returns
    None."""

    def __init__(
        self,
        models_df: Optional[pd.DataFrame] = None,
        benchmarks_df: Optional[pd.DataFrame] = None,
        metrics_df: Optional[pd.DataFrame] = None,
        harnesses_df: Optional[pd.DataFrame] = None,
        orgs_df: Optional[pd.DataFrame] = None,
        families_df: Optional[pd.DataFrame] = None,
        composites_df: Optional[pd.DataFrame] = None,
    ) -> None:
        self._tables: dict[str, pd.DataFrame] = {
            "model": models_df if models_df is not None else pd.DataFrame(),
            "benchmark": benchmarks_df if benchmarks_df is not None else pd.DataFrame(),
            "metric": metrics_df if metrics_df is not None else pd.DataFrame(),
            "harness": harnesses_df if harnesses_df is not None else pd.DataFrame(),
            "org": orgs_df if orgs_df is not None else pd.DataFrame(),
            "family": families_df if families_df is not None else pd.DataFrame(),
            "composite": composites_df if composites_df is not None else pd.DataFrame(),
        }
        # Per-table id-indexed cache for O(1) lookups
        self._index: dict[str, dict[str, dict]] = {}
        # Lazy reverse index: benchmark_id → family_id (built from
        # canonical_families.benchmark_ids on first access). Used by
        # benchmark-side enrichment to populate ResolutionResult.family_key
        # without scanning the families table per resolve call.
        self._benchmark_to_family: Optional[dict[str, str]] = None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_parquet(cls, path: str | Path) -> "CanonicalStore":
        """Load all five canonical tables from `<path>/<table>.parquet`.
        Missing files become empty tables — matches the AliasStore
        fallback so a partial fixtures directory still works."""
        p = Path(path)
        kwargs: dict[str, pd.DataFrame] = {}
        for entity_type, fname in _TABLES.items():
            file = p / f"{fname}.parquet"
            if not file.exists():
                logger.info(
                    "CanonicalStore.from_parquet: %s not found; using empty table",
                    file,
                )
                continue
            try:
                df = pd.read_parquet(file)
            except (OSError, ValueError) as exc:
                logger.warning(
                    "CanonicalStore.from_parquet: failed to read %s (%s: %s); "
                    "falling back to empty table",
                    file, type(exc).__name__, exc,
                )
                continue
            kwargs[_kwarg_for(entity_type)] = df
        return cls(**kwargs)

    @classmethod
    def from_hf(cls, repo_id: str) -> "CanonicalStore":
        """Load all five canonical tables from a HF Dataset repo. Each
        table lives at `<table>/part-0.parquet`. Missing tables fall
        back to empty (matches AliasStore's behavior)."""
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import (
            EntryNotFoundError,
            HfHubHTTPError,
            RepositoryNotFoundError,
        )

        kwargs: dict[str, pd.DataFrame] = {}
        for entity_type, fname in _TABLES.items():
            try:
                local = hf_hub_download(
                    repo_id=repo_id,
                    filename=f"{fname}/part-0.parquet",
                    repo_type="dataset",
                )
                df = pd.read_parquet(local)
            except (
                RepositoryNotFoundError,
                EntryNotFoundError,
                HfHubHTTPError,
                FileNotFoundError,
                OSError,
                ValueError,
            ) as exc:
                logger.warning(
                    "CanonicalStore.from_hf: failed to load %s from %r (%s: %s); "
                    "using empty table",
                    fname, repo_id, type(exc).__name__, exc,
                )
                continue
            kwargs[_kwarg_for(entity_type)] = df
        return cls(**kwargs)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def _ensure_index(self, entity_type: str) -> dict[str, dict]:
        if entity_type in self._index:
            return self._index[entity_type]
        df = self._tables.get(entity_type)
        idx: dict[str, dict] = {}
        if df is not None and not df.empty and "id" in df.columns:
            for _, row in df.iterrows():
                cid = row["id"]
                if isinstance(cid, str):
                    idx[cid] = {k: _na_to_none(v) for k, v in row.items()}
        self._index[entity_type] = idx
        return idx

    def lookup(self, entity_type: str, canonical_id: str) -> Optional[dict]:
        """Return the canonical row as a dict (with NaN coerced to None),
        or None when the id isn't present. O(1)."""
        if not canonical_id:
            return None
        return self._ensure_index(entity_type).get(canonical_id)

    @property
    def org_dev_map(self) -> dict[str, str]:
        """The curated HF-namespace -> developer-org map built from the LOADED
        `canonical_orgs` table (its `id` / `hf_org` / `aliases`), unioned with
        the hardcoded `_ORG_ALIASES`. This is how the resolver gets the orgs.yaml
        ALIAS tier (e.g. `AlephAlpha`->`aleph-alpha`, `MiniMaxAI`->`minimax`,
        `kimi`->`moonshotai`) that the bare `_ORG_ALIASES` lacks — without the
        resolver needing to read orgs.yaml. Threaded into the fuzzy org-agreement
        guard so a fuzzy stem match folds org-equivalent namespaces correctly.
        Cached after first build."""
        cached = getattr(self, "_org_dev_map", None)
        if cached is not None:
            return cached
        from eval_entity_resolver.fold import build_curated_org_map

        df = self._tables.get("org")
        records: list[dict] = []
        if df is not None and not df.empty:
            for rec in df.to_dict("records"):
                rec = {k: _na_to_none(v) for k, v in rec.items()}
                # `aliases` is VARCHAR (JSON-encoded list) in the parquet tables;
                # build_curated_org_map expects a list, so decode it.
                al = rec.get("aliases")
                if isinstance(al, str):
                    rec["aliases"] = _safe_json_load(al, default=[])
                records.append(rec)
        self._org_dev_map = build_curated_org_map(records)
        return self._org_dev_map

    # ------------------------------------------------------------------
    # Enrichment — used by `Resolver` to populate the rich response
    # fields. Pure functions of (entity, optional root entity); no
    # access to any state outside what's passed in.
    # ------------------------------------------------------------------

    def benchmark_family_enrichment(
        self, benchmark_id: Optional[str]
    ) -> dict:
        """For a matched benchmark canonical_id, return the family/category
        fields that populate the benchmark side of `ResolutionResult`.

        Output shape (dict; consumed by Resolver._enrich):
          - `family_key`: id of the canonical_families row whose
            benchmark_ids contains `benchmark_id`. Falls back to
            `benchmark_id` itself for singleton families (`family.id ==
            benchmark.id` when no curated family covers it).
          - `category`: family's curated category, or None.
          - `composite_keys`: empty list at the resolver layer. The
            producer's view layer is the right place to compute which
            composites a benchmark appears in (it has the facts), so the
            resolver leaves this empty and downstream callers fill it.
        """
        if not benchmark_id:
            return {"family_key": None, "category": None, "composite_keys": []}

        if self._benchmark_to_family is None:
            self._benchmark_to_family = self._build_benchmark_to_family_index()

        # 1. Curated family directly listing this benchmark id.
        family_key = self._benchmark_to_family.get(benchmark_id)

        # 2. Slice inherits its parent's family. A benchmark with
        #    parent_benchmark_id != self is a slice; walk up to find the
        #    root, then look that root up in the curated families. Cycle-
        #    safe via visited set; terminates at a root or a missing entry.
        if family_key is None:
            visited: set[str] = {benchmark_id}
            cur = benchmark_id
            while True:
                bench_row = self.lookup("benchmark", cur)
                if bench_row is None:
                    break
                parent = _na_to_none(bench_row.get("parent_benchmark_id"))
                if not parent or parent == cur or parent in visited:
                    break
                visited.add(parent)
                cur = parent
                fam = self._benchmark_to_family.get(parent)
                if fam:
                    family_key = fam
                    break
            # When no curated family covers this id or any of its parents,
            # the family root IS the parent walk's terminus (or the id
            # itself for true root benchmarks). That's the singleton-family
            # default.
            if family_key is None:
                family_key = cur

        family_row = self.lookup("family", family_key)
        category = (
            _na_to_none(family_row.get("category"))
            if family_row is not None
            else None
        )
        return {
            "family_key": family_key,
            "category": category,
            "composite_keys": [],
        }

    # ------------------------------------------------------------------
    # Hierarchy: ancestry (type-agnostic) + typed resolution_detail.
    # Pure functions of the loaded tables. `ancestry` lists the matched
    # entity's immediate parent UP to the root; `resolution_detail` is a
    # typed sub-object keyed by entity_type.
    # ------------------------------------------------------------------

    def _family_to_composite(self, family_id: Optional[str]) -> Optional[str]:
        """Return the composite a family rolls up into (the family's first
        `composite_keys` entry), or the first composite whose `family_id`
        points back at this family. None when the family is a root."""
        if not family_id:
            return None
        family_row = self.lookup("family", family_id)
        if family_row is not None:
            keys = family_row.get("composite_keys")
            if isinstance(keys, str):
                keys = _safe_json_load(keys, default=[])
            if isinstance(keys, list):
                for k in keys:
                    if isinstance(k, str) and k:
                        return k
        # Fall back to the reverse pointer (composite.family_id == family).
        comp_df = self._tables.get("composite")
        if comp_df is not None and not comp_df.empty and "family_id" in comp_df.columns:
            hit = comp_df[comp_df["family_id"] == family_id]
            if not hit.empty:
                cid = hit.iloc[0].get("id")
                if isinstance(cid, str):
                    return cid
        return None

    def compute_ancestry(
        self, entity_type: str, canonical_id: Optional[str],
        matched_entity: Optional[dict] = None,
    ) -> list[dict]:
        """Ordered `[{canonical_id, level}]` from the matched entity's
        IMMEDIATE PARENT up to the root. `[]` when self is a root.

        - model: group (model_group_id, when it differs from the leaf) then
          family (model_family_id, when distinct from leaf+group).
        - benchmark: family (family_key, when != self) then that family's
          composite.
        - family: its composite.
        - composite/metric/harness/org: [] (roots).
        """
        if not canonical_id:
            return []
        out: list[dict] = []
        if entity_type == "model":
            ent = matched_entity if matched_entity is not None else self.lookup("model", canonical_id)
            if not ent:
                return []
            group = _na_to_none(ent.get("model_group_id"))
            family = _na_to_none(ent.get("model_family_id"))
            if group and group != canonical_id:
                out.append({"canonical_id": group, "level": "group"})
            if family and family != canonical_id and family != group:
                out.append({"canonical_id": family, "level": "family"})
            return out
        if entity_type == "benchmark":
            fam = self.benchmark_family_enrichment(canonical_id)
            family_key = fam.get("family_key")
            if family_key and family_key != canonical_id:
                out.append({"canonical_id": family_key, "level": "family"})
            composite = self._family_to_composite(family_key)
            if composite and composite != canonical_id:
                out.append({"canonical_id": composite, "level": "composite"})
            return out
        if entity_type == "family":
            composite = self._family_to_composite(canonical_id)
            if composite and composite != canonical_id:
                out.append({"canonical_id": composite, "level": "composite"})
            return out
        # composite, metric, harness, org are roots in this graph.
        return out

    def resolution_detail(
        self, entity_type: str, canonical_id: Optional[str],
        raw_value: Optional[str] = None,
        matched_entity: Optional[dict] = None,
    ) -> dict:
        """Typed resolution-detail sub-object keyed by entity_type.

        - model:     {"granularity": variant|group|family}
        - benchmark: {"level": composite|family|benchmark|slice,
                      "matched_subset": str|None}
        - harness:   {} here — the RESOLVER (not the store) fills
                     {harness_version_stripped, bare_name, bare_tier} on a
                     match it reached by stripping a trailing version token.
        - others:    {}
        """
        if entity_type == "model":
            ent = matched_entity if matched_entity is not None else self.lookup("model", canonical_id)
            gran = _na_to_none((ent or {}).get("resolution_granularity")) if ent else None
            return {"granularity": gran, "hf_repo_id": _hf_repo_id_of(ent, canonical_id)}
        if entity_type == "benchmark":
            ent = matched_entity if matched_entity is not None else self.lookup("benchmark", canonical_id)
            level = "benchmark"
            matched_subset: Optional[str] = None
            if ent:
                parent = _na_to_none(ent.get("parent_benchmark_id"))
                if parent and parent != canonical_id:
                    # The matched canonical is itself a decomposed slice of a
                    # parent benchmark (a parent-only alias-fold, not its own
                    # entity): surface as a slice match.
                    level = "slice"
            # A subset/alias-fold match (e.g. "Anatomy" -> mmlu) is surfaced
            # via `matched_subset` when the raw value differs from the
            # canonical's own surface forms. We carry the raw value through;
            # downstream forensics maps it to the folded subset.
            if raw_value and canonical_id and raw_value.strip().lower() != canonical_id.lower():
                matched_subset = raw_value
            return {"level": level, "matched_subset": matched_subset}
        return {}

    def _build_benchmark_to_family_index(self) -> dict[str, str]:
        """Walk canonical_families and produce a benchmark_id → family_id
        index. `benchmark_ids` is JSON-encoded on the parquet column;
        decode tolerantly. Returns an empty index when the families table
        is absent, so a deployment without it still resolves benchmarks
        (just without family ancestry)."""
        out: dict[str, str] = {}
        df = self._tables.get("family")
        if df is None or df.empty or "id" not in df.columns:
            return out
        for _, row in df.iterrows():
            family_id = row.get("id")
            if not isinstance(family_id, str):
                continue
            raw = row.get("benchmark_ids")
            if _is_na(raw) or raw is None:
                continue
            if isinstance(raw, list):
                items = raw
            elif isinstance(raw, str):
                s = raw.strip()
                if not s or s in ("[]", "null"):
                    continue
                # A malformed value decodes to [], so the row contributes
                # no benchmark→family entries (same effect as skipping it).
                items = _safe_json_load(s, default=[])
            else:
                continue
            for bid in items:
                if isinstance(bid, str):
                    # Validation has already rejected multi-family
                    # benchmarks at seed time, so first-write-wins is
                    # safe (and deterministic by family load order).
                    out.setdefault(bid, family_id)
        return out

    def parent_canonical_id(
        self, entity_type: str, entity: Optional[dict]
    ) -> Optional[str]:
        """Family/variant parent id. For models: the first `variant` edge
        in the typed parents list. For benchmarks/orgs: the
        `parent_*_id` scalar column."""
        if not entity:
            return None
        if entity_type == "model":
            return variant_parent_id(decode_parents(entity.get("parents")))
        field = _PARENT_FIELD.get(entity_type)
        if not field:
            return None
        return _na_to_none(entity.get(field))

    def model_metadata_fields(
        self, matched_canonical_id: str, matched_entity: Optional[dict]
    ) -> dict:
        """Compute the model-specific response fields.

        `canonical_id` is the exact matched LEAF (the precise artifact
        evaluated — snapshot, precision, mode all distinct). `model_group_id`
        carries the identity-GROUP id, which is GROUP MEMBERSHIP — a total
        partition, so it is ALWAYS present: it equals the group root for a
        member of a non-trivial group, and equals the leaf (== canonical_id)
        for a singleton (a group of one whose id is itself). NOT null at the
        root. `resolved_leaf_id == canonical_id` (both the leaf), retained
        for compat. The deprecated `root_model_id` output key keeps its
        null-at-root semantics: it carries the group root ONLY when the leaf
        actually collapses into a larger group (`model_group_id != leaf_id`),
        else None.

        `model_family_id` and `lineage_origin_model_id` are read straight off
        the matched (leaf) entity row — `derive_model_lineage_fields` already
        materialised them at seed time. Metadata fields (`open_weights`,
        `release_date`, `params_billions`) come from the matched LEAF row —
        the response identifies the leaf, so its own row is the consistent
        source for these per-artifact values."""
        if not matched_entity:
            return {
                "canonical_id": matched_canonical_id,
                "resolved_leaf_id": matched_canonical_id,
                "root_model_id": None,
                "lineage_origin_org_id": None,
                # Extended lineage / provenance fields — None when there is no
                # matched entity row to read them from.
                "model_group_id": None,
                "model_family_id": None,
                "lineage_origin_model_id": None,
                "lineage_origin_model_org_id": None,
                "inference_platform": None,
                "resolution_source": None,
                "resolution_granularity": None,
                "parents": None,
                "open_weights": None,
                "release_date": None,
                "params_billions": None,
            }

        # `model_group_id` is the identity-GROUP id (GROUP MEMBERSHIP — a
        # total partition), so it is ALWAYS set on the column: equal to the
        # group root for a member of a larger group, equal to SELF (the leaf)
        # for a singleton. canonical_id stays the matched LEAF; for a
        # singleton model_group_id == canonical_id.
        group_id = _na_to_none(matched_entity.get("model_group_id"))
        leaf_id = matched_canonical_id
        parents_decoded = decode_parents(matched_entity.get("parents")) or None
        # The three walk fields are read straight off the matched LEAF row —
        # materialised at seed by derive_model_lineage_fields.
        leaf_family = _na_to_none(matched_entity.get("model_family_id"))
        leaf_lineage_model = _na_to_none(matched_entity.get("lineage_origin_model_id"))
        leaf_lineage_org = _na_to_none(matched_entity.get("lineage_origin_model_org_id"))

        # The leaf collapses into a LARGER group iff `model_group_id !=
        # leaf_id`. The deprecated `root_model_id` compat key keeps its
        # null-at-root semantics — it carries the group only on a real
        # collapse, else None (a singleton, whose group is itself, reports
        # root_model_id == None, matching the producer's null-at-root
        # contract).
        collapses = group_id is not None and group_id != leaf_id
        return {
            "canonical_id": leaf_id,
            "resolved_leaf_id": leaf_id,
            "root_model_id": group_id if collapses else None,
            "lineage_origin_org_id": leaf_lineage_org,
            # ALWAYS present (self at root) — group membership is total.
            "model_group_id": group_id,
            "model_family_id": leaf_family,
            "lineage_origin_model_id": leaf_lineage_model,
            "lineage_origin_model_org_id": leaf_lineage_org,
            "inference_platform": None,
            # Provenance fields read straight off the matched LEAF row (set at
            # seed from the YAML, or at live auto-create by resolution_service).
            "resolution_source": _na_to_none(matched_entity.get("resolution_source")),
            "resolution_granularity": _na_to_none(matched_entity.get("resolution_granularity")),
            "parents": parents_decoded,
            "open_weights": _na_to_none(matched_entity.get("open_weights")),
            "release_date": _na_to_none(matched_entity.get("release_date")),
            "params_billions": _na_to_none(matched_entity.get("params_billions")),
        }
