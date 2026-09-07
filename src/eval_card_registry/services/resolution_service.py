"""
resolution_service: wraps the eval-entity-resolver package.

Responsibilities:
- Call the resolver
- Auto-create draft canonical entities when resolver returns no_match
- Write aliases for every resolution (add on first resolve, update on rerun)
- Append to the resolution log
"""
from __future__ import annotations

import json
import re
import threading
import uuid
from dataclasses import replace as _dc_replace
from datetime import datetime, timezone
from typing import Optional

from eval_entity_resolver import AliasStore, CanonicalStore, Resolver, ResolverConfig, ResolutionResult
from eval_entity_resolver.display import humanize_model_slug

from eval_card_registry.config import settings
from eval_card_registry.store.hf_store import RegistryStore
from eval_card_registry.store import queries


# Tier-3 base-family lexicon (mirrors scripts/generate_tier3_inferred_seed.py).
# Used only to DETECT an inferable base; an edge is emitted solely when the
# candidate base alias-confirms to an existing canonical.
_TIER3_BASE_TOKENS = (
    "tinyllama", "openhermes", "openchat", "mixtral",
    "llama", "mistral", "qwen", "gemma", "yi", "phi", "gpt", "claude",
    "gemini", "falcon", "bloom", "deepseek", "baichuan", "cohere", "command",
    "neural", "solar", "nous", "zephyr", "orca", "dolphin",
)
# `unknown` and bare hosts are placeholder org prefixes — an id like
# `unknown/foo` has NO extractable org: treat it as org-less, not as org
# `unknown`.
_PLACEHOLDER_ORG_PREFIXES = {"unknown", "none", "null", "model", "models", "local"}


# Map entity_type to table name
_ENTITY_TABLE = {
    "model": "canonical_models",
    "benchmark": "canonical_benchmarks",
    "metric": "canonical_metrics",
    "harness": "eval_harnesses",
    "org": "canonical_orgs",
}

def _slugify(value: str) -> str:
    """
    Produce a lowercase slug for auto-created entity IDs.
    Falls back to a UUID-derived ID if the input reduces to nothing (e.g. all punctuation).
    """
    slug = value.lower().strip()
    slug = re.sub(r"[^\w\s\-/]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = slug.strip("-")  # trim leading/trailing dashes
    if not slug:
        slug = f"auto-{str(uuid.uuid4())[:8]}"
    return slug


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _table_with_pending(registry_store: RegistryStore, name: str) -> "pd.DataFrame":
    """Return a table DataFrame with pending-buffer rows appended.

    `_auto_create_entity` writes drafts with `buffered=True`, so they sit
    in `store._pending[<table>]` until `flush_pending` runs at the end of
    a sync. Without overlaying pending here, the resolver's
    `CanonicalStore` snapshot can't see the just-created row, and
    `build_result` for an auto-created entity returns null for every
    metadata field that hub-stats just enriched.

    Concat is safe because `upsert_entity` enforces id-uniqueness across
    base + pending (existing rows go to in-place update; only genuinely
    new ids land in pending), so no duplicate keys end up in the
    CanonicalStore index.
    """
    import pandas as pd
    base_df = registry_store.table(name) if registry_store.has_table(name) else pd.DataFrame()
    pending = getattr(registry_store, "_pending", {}).get(name, [])
    if not pending:
        return base_df
    pending_df = pd.DataFrame(pending)
    if base_df.empty:
        return pending_df
    return pd.concat([base_df, pending_df], ignore_index=True)


def _build_alias_store(registry_store: RegistryStore) -> AliasStore:
    """Build an AliasStore from the registry's in-memory aliases table."""
    aliases_df = registry_store.table("aliases")
    return AliasStore(aliases_df, read_only=True)


def _build_canonical_store(registry_store: RegistryStore) -> CanonicalStore:
    """Build a CanonicalStore from the registry's in-memory canonical
    tables. Lets the bare resolver enrich its results with the same
    metadata fields the HTTP API exposes — including benchmark
    `family_key` / `category` (which need families_df + composites_df
    to populate; otherwise they fall back to the benchmark's own id).

    Pending-buffer rows are overlaid so the resolver sees auto-created
    drafts before `flush_pending` runs. See `_table_with_pending`."""
    return CanonicalStore(
        models_df=_table_with_pending(registry_store, "canonical_models"),
        benchmarks_df=_table_with_pending(registry_store, "canonical_benchmarks"),
        metrics_df=_table_with_pending(registry_store, "canonical_metrics"),
        harnesses_df=_table_with_pending(registry_store, "eval_harnesses"),
        orgs_df=_table_with_pending(registry_store, "canonical_orgs") if registry_store.has_table("canonical_orgs") else None,
        families_df=registry_store.table("canonical_families") if registry_store.has_table("canonical_families") else None,
        composites_df=registry_store.table("canonical_composites") if registry_store.has_table("canonical_composites") else None,
    )


_RESPONSE_FIELDS = (
    "canonical_id", "strategy", "confidence", "review_status",
    "parent_canonical_id", "resolved_leaf_id", "root_model_id",
    "lineage_origin_org_id",
    # `root_model_id` / `lineage_origin_org_id` above are deprecated compat
    # aliases for `model_group_id` / `lineage_origin_model_org_id`; both names
    # are emitted so older consumers keep working.
    "model_group_id", "model_family_id", "lineage_origin_model_id",
    "lineage_origin_model_org_id", "inference_platform",
    "resolution_source", "resolution_granularity",
    "parents", "open_weights",
    "release_date", "params_billions",
    "family_key", "composite_keys", "category",
    # Hierarchy contract — type-agnostic ancestry + typed detail. Carried
    # on the rich service dict; the HTTP route projects the lean shape from
    # these (see api/routes_resolve.py::_project_response).
    "ancestry", "resolution_detail",
)


def _result_to_dict(result: ResolutionResult, *, created_new: bool) -> dict:
    """Convert a `ResolutionResult` dataclass to the dict shape the
    service contract returns. The dataclass already carries every
    response field (computed by the resolver via `CanonicalStore`);
    this just selects the public fields and tacks on `created_new`,
    which is service-state the resolver doesn't know about."""
    return {field: getattr(result, field) for field in _RESPONSE_FIELDS} | {
        "created_new": created_new,
    }


def _no_match_result() -> dict:
    """Stable no-match dict — used by the empty-input guard before any
    resolver call happens."""
    return {field: None for field in _RESPONSE_FIELDS} | {
        "strategy": "no_match",
        "confidence": 0.0,
        "created_new": False,
    }


class ResolutionService:
    def __init__(self, registry_store: RegistryStore) -> None:
        import threading
        self.store = registry_store
        self._resolver: Optional[Resolver] = None
        # Cache: (raw_value, entity_type, source_config) → resolve result dict.
        # Avoids re-running the full strategy chain for duplicate strings
        # (e.g. "Accuracy" appears in every record).
        self._resolve_cache: dict[tuple[str, str, Optional[str]], dict] = {}
        # Hub-stats live-lookup state (built lazily on first use). The
        # indices snapshot the aliases / orgs tables; both get invalidated
        # by `invalidate_resolver()` whenever a new entity is auto-created
        # so subsequent lookups can resolve baseModels against the just-
        # added canonical. Lock guards the lazy build under FastAPI's
        # threadpool executor.
        self._hub_stats_client = None
        self._hub_stats_indices: Optional[tuple[dict[str, str], dict[str, str]]] = None
        self._hub_stats_indices_lock = threading.Lock()
        # HF id confirmation index: normalized HF id -> HF-true id, built
        # lazily from the `hub_stats_index` table (both store modes). Feeds
        # the HfIdVerifier, which the resolver consults as the HF-id-check
        # step of the chain. Invalidated alongside the hub-stats indices.
        self._hf_id_index: Optional[dict[str, str]] = None
        # The runtime HF repo-id verifier (index + rate-limit-guarded live
        # Hub API fallback). Outlives resolver invalidations so its result
        # caches and breaker state survive entity churn within the process.
        from eval_card_registry.services.hf_id_verifier import HfIdVerifier
        self._hf_id_verifier = HfIdVerifier(index_provider=self._build_hf_id_index)

    def _get_resolver(self) -> Resolver:
        if self._resolver is None:
            alias_store = _build_alias_store(self.store)
            canonical_store = _build_canonical_store(self.store)
            config = ResolverConfig(threshold=settings.resolver_auto_merge_threshold)
            # The resolver returns a fully-enriched `ResolutionResult` —
            # same fields the HTTP API exposes. Parent-decode / root-collapse /
            # metadata lookup all live in the resolver; this service just
            # converts the dataclass to a dict and adds `created_new`
            # (auto-draft state the resolver doesn't track).
            self._resolver = Resolver(
                alias_store, config, canonical_store=canonical_store,
                hf_id_checker=self._hf_id_verifier.check,
            )
        return self._resolver

    def invalidate_resolver(self) -> None:
        """Call after alias or entity changes to force resolver rebuild.
        Also clears the hub-stats indices cache so subsequent live lookups
        can resolve `baseModels` parents against just-added canonicals
        (e.g. when EEE sync creates a parent draft, then sees a child
        whose baseModels references that parent in the same run)."""
        self._resolver = None
        with self._hub_stats_indices_lock:
            self._hub_stats_indices = None
            self._hf_id_index = None

    def resolve(
        self,
        raw_value: str,
        entity_type: str,
        source_config: Optional[str],
        source_field: Optional[str],
        sync_run_id: Optional[str] = None,
        rerun: bool = False,
        mode: str = "resolve",
    ) -> dict:
        """Resolve a raw value to a canonical entity. Returns a dict with
        the full enriched response shape — same fields as
        `eval_entity_resolver.ResolutionResult` plus `created_new`. The
        keys are the values in `_RESPONSE_FIELDS` plus `"created_new"`;
        every match (including auto-drafts and no_match) emits the same
        shape so callers don't need to branch on missing keys.

        `mode="exact"` disables fuzzy inference AND all side effects on
        entity data (no auto-created drafts, no alias writes) — a miss is
        an honest no_match.

        See `api/schemas.py::ResolveResponse` for the field documentation
        and the README "API" section for an example response."""
        if not raw_value or not raw_value.strip():
            return _no_match_result()

        # Fast path: return cached result for duplicate
        # (raw_value, entity_type, source_config, mode).
        cache_key = (raw_value, entity_type, source_config, mode)
        if not rerun and cache_key in self._resolve_cache:
            return self._resolve_cache[cache_key]

        # Read-only mode: resolve only, no side effects on entity data.
        # The HF id check runs inside the resolver chain (injected checker),
        # so a real-but-unminted HF id resolves to itself with no minting —
        # verbatim confirmations outrank alias hits, normalized ones rank
        # between the normalized-alias and fuzzy steps.
        if settings.read_only:
            resolver = self._get_resolver()
            result: ResolutionResult = resolver.resolve(
                raw_value, entity_type, source_config, mode=mode
            )
            result_dict = (
                _result_to_dict(result, created_new=False)
                if result.canonical_id is not None else _no_match_result()
            )
            self._memoize(cache_key, result_dict)
            return result_dict

        # Check if alias already exists (skip resolver on rerun=False).
        # Re-run the strategy chain so the response carries the correct
        # `resolved_leaf_id` — the alias table only stores the
        # root-collapsed `canonical_id`, so reconstructing the response
        # via `build_result(root, ...)` would clobber the leaf to the
        # root (model_metadata_fields can't recover leaf identity from
        # a root row alone — there's no back-pointer). The strategy
        # chain re-derives leaf cleanly; perf cost is one alias-index
        # lookup since exact-match hits in O(1) for already-aliased
        # values. Audit fields are overlaid from the alias entry so
        # callers still see the original strategy/confidence.
        prefetched: Optional[ResolutionResult] = None
        if not rerun:
            existing = queries.get_alias(self.store, raw_value, entity_type, source_config)
            if existing:
                resolver = self._get_resolver()
                fresh = resolver.resolve(raw_value, entity_type, source_config, mode=mode)
                checker_backed = (
                    fresh.hf_attestation is not None
                    and fresh.canonical_id == fresh.hf_attestation
                )
                if fresh.canonical_id == existing["canonical_id"]:
                    # Agreement. Overlay the stored audit fields — unless
                    # the fresh result is checker-backed, whose runtime
                    # attestation (strategy/confidence/detail) must never
                    # be overwritten by stale stored audit fields.
                    enriched = fresh if checker_backed else _dc_replace(
                        fresh,
                        strategy=existing["strategy"],
                        confidence=existing["confidence"],
                    )
                elif (
                    checker_backed
                    and fresh.canonical_id.lower() == existing["canonical_id"].lower()
                ):
                    # Case-only disagreement — the same repo (HF ids are
                    # case-insensitively unique). Keep the registered row;
                    # no flag, no mint.
                    enriched = resolver.build_result(
                        raw_value, entity_type, source_config,
                        existing["canonical_id"], fresh.strategy, fresh.confidence,
                    )
                elif checker_backed:
                    if mode == "exact":
                        # Exact mode is side-effect-free: serve the
                        # attestation, no flag, no mint.
                        enriched = fresh
                    else:
                        # A runtime HF attestation disagrees with the stored
                        # alias. The verbatim HF id wins (owner decision);
                        # the alias is flagged for review, never silently
                        # repointed. Fall through to the main path so an
                        # unregistered attested id still gets minted.
                        self._flag_alias_hf_disagreement(existing, fresh.hf_attestation)
                        prefetched = fresh
                else:
                    # Rare: registry restructure has moved the canonical
                    # for this raw_value since the alias was written.
                    # The alias entry is the source of truth for "what
                    # this raw resolved to" — accept the leaf clobber.
                    enriched = resolver.build_result(
                        raw_value, entity_type, source_config,
                        existing["canonical_id"], existing["strategy"], existing["confidence"],
                    )
                if prefetched is None:
                    result_dict = _result_to_dict(enriched, created_new=False)
                    self._memoize(cache_key, result_dict)
                    return result_dict

        resolver = self._get_resolver()
        result = (
            prefetched if prefetched is not None
            else resolver.resolve(raw_value, entity_type, source_config, mode=mode)
        )

        # Exact-only mode is fully side-effect-free: no drafts, no org
        # mints, no alias writes, no store log rows — on a match AND on a
        # miss. (An unregistered attestation is served as-is, read-only
        # style.)
        if mode == "exact":
            result_dict = (
                _result_to_dict(result, created_new=False)
                if result.canonical_id is not None else _no_match_result()
            )
            self._memoize(cache_key, result_dict)
            return result_dict

        created_new = False
        alias_status = "auto"

        if result.canonical_id is not None and result.hf_attested_unregistered:
            # HF-attested but not a registered entity. If a case-variant
            # canonical already exists (e.g. a lowercased draft minted
            # before the index knew the repo), reuse it instead of minting
            # a shadow duplicate — HF ids are case-insensitively unique.
            case_variant = self._find_case_variant_model(result.canonical_id)
            if case_variant is not None:
                canonical_id = case_variant
                result = resolver.build_result(
                    raw_value, entity_type, source_config,
                    case_variant, result.strategy, result.confidence,
                )
            else:
                # Mint it, adopting the checker's HF-true id (never a
                # slugified re-derivation), so no alias / eval_results row
                # ever points at a nonexistent canonical.
                canonical_id = self._auto_create_entity(
                    entity_type, raw_value, preset_hf_id=result.canonical_id
                )
                created_new = True
        elif result.canonical_id is not None:
            # Match found above threshold
            canonical_id = result.canonical_id
            alias_status = "auto"
        else:
            # No match — auto-create draft entity
            canonical_id = self._auto_create_entity(entity_type, raw_value)
            alias_status = "uncertain"
            created_new = True

        strategy_used = result.strategy if result.canonical_id else "auto_draft"

        # A harness match the resolver reached by stripping a trailing version
        # token carries its provenance in `resolution_detail`; keep it on the
        # `auto` alias row so the minted per-version rows stay auditable.
        detail = result.resolution_detail or {}
        strip_note = None
        if isinstance(detail, dict) and detail.get("harness_version_stripped"):
            strip_note = (
                f"harness version stripped: {detail['harness_version_stripped']} "
                f"(bare {detail.get('bare_name')}, {detail.get('bare_tier')} tier)"
            )

        # Write alias (buffered during sync for performance)
        alias_data = {
            "raw_value": raw_value,
            "entity_type": entity_type,
            "canonical_id": canonical_id,
            "source_config": source_config,
            "source_field": source_field,
            "status": alias_status,
            "strategy": strategy_used,
            "confidence": result.confidence,
            "notes": strip_note,
        }
        if rerun:
            existing_row = queries.get_alias(self.store, raw_value, entity_type, source_config)
            existing_alias_id = existing_row.get("id") if existing_row else None
            if existing_alias_id and (
                existing_row.get("status") == "confirmed"
                and existing_row.get("canonical_id") != canonical_id
            ):
                # Never auto-repoint or demote a CONFIRMED alias, even on a
                # rerun. Record the disagreement for review instead; the
                # response still carries the fresh resolution.
                queries.update_alias(
                    self.store,
                    alias_id=existing_alias_id,
                    updates={
                        "notes": (
                            f"rerun disagreement: resolver returned "
                            f"{canonical_id} ({strategy_used})"
                        ),
                    },
                )
            elif existing_alias_id:
                updates = {
                    "canonical_id": canonical_id,
                    "status": alias_status,
                    "strategy": strategy_used,
                    "confidence": result.confidence,
                }
                if strip_note is not None:
                    # The rerun path must carry the marker too, or the alias
                    # row stops being the audit trail after the first rerun.
                    updates["notes"] = strip_note
                queries.update_alias(
                    self.store,
                    alias_id=existing_alias_id,
                    updates=updates,
                )
            else:
                try:
                    queries.add_alias(self.store, alias_data, buffered=True)
                except ValueError:
                    pass
        else:
            try:
                queries.add_alias(self.store, alias_data, buffered=True)
            except ValueError:
                pass  # alias already exists (from prior resolution in this run)

        # Log
        if sync_run_id:
            queries.append_resolution_log(
                self.store,
                {
                    "sync_run_id": sync_run_id,
                    "raw_value": raw_value,
                    "entity_type": entity_type,
                    "source_config": source_config,
                    "strategy": strategy_used,
                    "confidence": result.confidence,
                    "canonical_id": canonical_id,
                    "created_new": created_new,
                },
            )

        # Only invalidate when a new entity was created — its alias could
        # help future fuzzy matches AND the resolver/canonical-store cache
        # snapshot needs to refresh so the just-added entity is visible.
        if created_new:
            self.invalidate_resolver()

        # Build the enriched response. Two cases:
        #   1. Match found — the original `result` already carries the
        #      correct canonical_id (root-collapsed), resolved_leaf_id
        #      (the matched leaf), parents, and metadata. Don't re-run
        #      `build_result` here: it would call `model_metadata_fields`
        #      with the ROOT id, which can't recover the leaf and ends
        #      up returning resolved_leaf_id = canonical_id. The alias
        #      write earlier doesn't change canonical_models — `result`
        #      stays accurate.
        #   2. Auto-create — `result.canonical_id` was None, the new
        #      `canonical_id` came from `_auto_create_entity`. The new
        #      canonical IS the leaf (its parents may point at family
        #      via the inferred version-axis edge), so `build_result`
        #      with the new id correctly preserves leaf info via
        #      `model_metadata_fields`. The `invalidate_resolver()`
        #      above ensures the canonical_store snapshot sees the new
        #      row, but the entity may still sit in the pending-write
        #      buffer; on lookup miss the review_status falls back to
        #      None and we override to 'draft' below.
        if created_new:
            resolver = self._get_resolver()
            enriched = resolver.build_result(
                raw_value, entity_type, source_config,
                canonical_id, strategy_used, result.confidence,
            )
        else:
            enriched = result
        result_dict = _result_to_dict(enriched, created_new=created_new)
        if created_new and result_dict.get("review_status") is None:
            result_dict["review_status"] = "draft"
        self._memoize(cache_key, result_dict)
        return result_dict

    def _memoize(self, cache_key: tuple, result_dict: dict) -> None:
        """Memoize a resolve result for duplicate raw values within the
        process. A no_match for an HF-shaped model value is NOT memoized:
        the HF id checker may confirm it later (negative-TTL expiry, index
        refresh, circuit-breaker close), and a permanent memo would pin the
        miss for the process lifetime."""
        raw_value, entity_type = cache_key[0], cache_key[1]
        if (
            result_dict.get("canonical_id") is None
            and entity_type == "model"
            and self._looks_like_hf_id(raw_value)
        ):
            return
        self._resolve_cache[cache_key] = result_dict

    def _find_case_variant_model(self, hf_id: str) -> Optional[str]:
        """A registered canonical model id equal to `hf_id` up to casing,
        or None. Linear scan — only reached on the rare attested-but-
        unregistered mint path."""
        df = self.store.table("canonical_models")
        if df is None or df.empty:
            return None
        lower = hf_id.lower()
        for existing_id in df["id"]:
            if isinstance(existing_id, str) and existing_id.lower() == lower:
                return existing_id
        return None

    def _flag_alias_hf_disagreement(self, existing: dict, hf_id: str) -> None:
        """A runtime HF attestation disagreed with a stored alias. The alias
        is never silently repointed: a non-confirmed row drops to `uncertain`
        so it lands in the review bucket; a confirmed row keeps its status.
        Both get a note recording the disagreement."""
        alias_id = existing.get("id")
        if not alias_id:
            return
        updates: dict = {"notes": f"hf-id-check disagreement: attested {hf_id}"}
        if existing.get("status") != "confirmed":
            updates["status"] = "uncertain"
        queries.update_alias(self.store, alias_id=alias_id, updates=updates)

    def _auto_create_entity(
        self, entity_type: str, raw_value: str, preset_hf_id: Optional[str] = None,
    ) -> str:
        table = _ENTITY_TABLE[entity_type]

        # Hub-stats live enrichment: when a model raw value looks like an
        # HF id, query hub-stats FIRST (before id finalisation) for
        # release_date / params / parents / lineage_origin_model_org_id and,
        # crucially, the HF-true repo `hf_id`. Best-effort — `enrichment` is
        # `{}` on lookup miss or any error.
        #
        # HF SOURCE-OF-TRUTH CASING: if hub-stats confirms the repo,
        # mint the canonical id + display_name from HF-true casing via the
        # two-tier org rule (do NOT `_slugify`-lowercase an HF-confirmed model
        # name) and mark `resolution_source="hf"`, `review_status="reviewed"`.
        # Otherwise fall back to the lowercase slug as before.
        enrichment: dict = {}
        if entity_type == "model" and (preset_hf_id or self._looks_like_hf_id(raw_value)):
            # Provisional id only for the hub-stats family-version self-edge
            # suppression; the real id is finalised from `hf_id` below.
            enrichment = self._lookup_hub_stats(
                preset_hf_id or raw_value, target_canonical=_slugify(raw_value)
            ) or {}
        if preset_hf_id and not (
            isinstance(enrichment.get("hf_id"), str) and enrichment["hf_id"].strip()
        ):
            # The runtime HF id checker already attested this repo id — adopt
            # it even when the (shard-0-limited) hub-stats parquet misses it.
            # Metadata enrichment stays best-effort; existence is settled.
            enrichment["hf_id"] = preset_hf_id

        hf_confirmed = entity_type == "model" and isinstance(
            enrichment.get("hf_id"), str
        ) and enrichment["hf_id"].strip()

        now = _now()
        if hf_confirmed:
            from eval_card_registry.services.hub_stats import (
                hf_id_to_canonical_cased,
            )
            hf_id = enrichment["hf_id"]
            candidate_id, _hf_org_id = hf_id_to_canonical_cased(
                hf_id, self._build_hf_to_dev()
            )
            # Display name = the HF model-name part (HF casing preserved).
            display = hf_id.split("/", 1)[1] if "/" in hf_id else hf_id
        else:
            candidate_id = _slugify(raw_value)
            # Models get a humanized display name (`gpt-5-2025-08-07` ->
            # `GPT-5 (2025-08-07)`); other entity types pass `raw_value`
            # through — benchmark/metric/harness/org names are usually
            # already in their preferred display form.
            if entity_type == "model":
                display = humanize_model_slug(raw_value) or raw_value
            else:
                display = raw_value

        # Ensure uniqueness (case-sensitive id collision).
        df = self.store.table(table)
        if (df["id"] == candidate_id).any():
            candidate_id = f"{candidate_id}-{str(uuid.uuid4())[:8]}"

        base = {
            "id": candidate_id,
            "display_name": display,
            "metadata": "{}",
            "review_status": "reviewed" if hf_confirmed else "draft",
            "created_at": now,
            "updated_at": now,
        }
        if entity_type == "model":
            # When HF-confirmed, the org_id comes from the two-tier rule on the
            # HF-true repo (so it matches the seeded HF-cased canonical's org);
            # otherwise resolve the raw prefix through the org resolver.
            if hf_confirmed:
                org_id = _hf_org_id
                # Materialise the HF org row if it does not exist yet so the FK
                # holds (community kind, like the seeded orgs.generated.yaml).
                self._ensure_hf_org(org_id)
            else:
                org_id = self._resolve_model_org_id(raw_value)
            # Tier-3: when NOT HF-confirmed and NOT models.dev-rescued (this
            # slug-fallback branch), the draft is a name-based INFERENCE. Mark
            # it `resolution_source="inferred"`,
            # `resolution_granularity="variant"`. If the raw value carries NO
            # extractable org (no `/`, or an `unknown`/host placeholder prefix),
            # do NOT auto-guess one: org_id stays None and we tag `org-unknown`
            # for the review bucket.
            tier3 = (entity_type == "model") and not hf_confirmed
            org_unknown = tier3 and not self._has_extractable_org(raw_value)
            if org_unknown:
                org_id = None
            base.update({
                "developer": None,
                "org_id": org_id,
                "family": None,
                "architecture": None,
                "params_billions": None,
                "parents": "[]",
                "model_group_id": None,
                "lineage_origin_model_org_id": None,
                "open_weights": None,
                "tags": '["org-unknown"]' if org_unknown else "[]",
                "resolution_source": "hf" if hf_confirmed else (
                    "inferred" if tier3 else None
                ),
                "resolution_granularity": "variant" if tier3 else None,
            })
            # Apply hub-stats enrichment last so its non-empty values
            # override the defaults we just set. The enrichment dict
            # only contains keys hub-stats actually had data for; other
            # defaults (None / "[]") survive. `hf_id` is provenance only,
            # not a column — skip it.
            for k, v in enrichment.items():
                if k == "hf_id":
                    continue
                if v is not None:
                    base[k] = v
            # Family-version inference fallback: when hub-stats misses
            # (parquet stale, lookup disabled, rate-limited, or row
            # absent), the snapshot still has its shape — try to infer a
            # version-axis parent from just the alias index. The
            # inference is alias-lookup-only, so it never manufactures
            # a false parent. Idempotent with the inference inside
            # enrich_draft_from_row: only fires when no version-axis
            # edge is already present.
            if self._looks_like_hf_id(raw_value):
                self._maybe_infer_family_parent(base, raw_value, candidate_id)
            # Tier-3 base inference: for an inferred draft with no parent yet,
            # detect a base family token and emit a finetune edge ONLY if the
            # base alias-confirms to an existing canonical (NEVER an invented
            # edge). Org-less inferred drafts can still link to a base
            # (lineage is independent of the missing org); we just don't guess
            # the org from it.
            if tier3:
                self._maybe_infer_tier3_base(base, raw_value, candidate_id)
        elif entity_type == "benchmark":
            base.update({"description": None, "dataset_repo": None, "parent_benchmark_id": None, "tags": "[]"})
        elif entity_type == "metric":
            base.update({"score_type": None, "lower_is_better": False, "min_score": None, "max_score": None})
        elif entity_type == "harness":
            base.update({"version": None, "fork_url": None})
        elif entity_type == "org":
            base.update({
                "parent_org_id": None,
                "website": None,
                "logo_url": None,
                "hf_org": None,
                "kind": "unknown",
                "tags": "[]",
            })

        queries.upsert_entity(self.store, table, base, buffered=True)
        return candidate_id

    def _maybe_infer_family_parent(
        self, base: dict, raw_value: str, candidate_id: str,
    ) -> None:
        """Mutate `base['parents']` to add a `{variant, axis: version}`
        edge when the raw value's snapshot shape resolves to an existing
        family canonical via the alias index. Runs independently of
        hub-stats so brand-new releases not yet in the parquet still
        get linked into the lineage graph."""
        try:
            existing = json.loads(base.get("parents") or "[]")
        except (ValueError, TypeError):
            existing = []
        if any(
            p.get("relationship") == "variant" and p.get("axis") == "version"
            for p in existing
            if isinstance(p, dict)
        ):
            return
        from eval_card_registry.services.hub_stats import infer_family_parent_edge
        try:
            aliases_to_canonical, _ = self._build_hub_stats_indices()
        except Exception:
            return
        edge = infer_family_parent_edge(
            raw_value, aliases_to_canonical, target_canonical=candidate_id,
        )
        if edge is None:
            return
        existing.append(edge)
        base["parents"] = json.dumps(existing)

    @staticmethod
    def _has_extractable_org(raw_value: str) -> bool:
        """True iff the raw value carries an extractable org prefix: a single
        `/` with a non-placeholder left side. `unknown/foo`, `foo` (no slash),
        and free-text labels all return False (org-less)."""
        if not raw_value or raw_value.count("/") != 1:
            return False
        org, name = (p.strip() for p in raw_value.split("/", 1))
        if not org or not name:
            return False
        return org.lower() not in _PLACEHOLDER_ORG_PREFIXES

    def _maybe_infer_tier3_base(
        self, base: dict, raw_value: str, candidate_id: str,
    ) -> None:
        """Add a single `{finetune}` edge to `base['parents']` when the raw
        value carries a recognized base-family token AND a shorter stem of that
        name alias-confirms (exact/normalized, NEVER fuzzy) to an existing
        canonical that is NOT the same identity as `candidate_id`. No-op when a
        parent already exists (the version-axis inference ran), when no base
        token is found, or when nothing alias-confirms — NEVER an invented edge.
        Mirrors scripts/generate_tier3_inferred_seed.py."""
        try:
            existing = json.loads(base.get("parents") or "[]")
        except (ValueError, TypeError):
            existing = []
        if existing:  # version-axis or hub-stats edge already present
            return

        name = raw_value.split("/", 1)[1] if "/" in raw_value else raw_value
        toks = [t for t in re.split(r"[\s\-_/.:]+", name.lower()) if t]
        tokset = set(toks)
        base_tok = None
        for fam in _TIER3_BASE_TOKENS:
            if fam in tokset or any(
                t.startswith(fam) and t[len(fam):][:1].isdigit() for t in toks
            ):
                base_tok = fam
                break
        if base_tok is None:
            return

        from eval_card_registry.services.hub_stats import normalize as _nz
        raw_name_nz = _nz(name)
        org_prefix = (
            raw_value.split("/", 1)[0] if self._has_extractable_org(raw_value) else None
        )
        try:
            start = next(
                i for i, t in enumerate(toks)
                if t == base_tok or (t.startswith(base_tok) and t[len(base_tok):][:1].isdigit())
            )
        except StopIteration:
            start = 0
        tail = toks[start:]
        resolver = self._get_resolver()
        seen: set[str] = set()
        for end in range(len(tail), 0, -1):
            stem = "-".join(tail[:end])
            candidates = []
            if org_prefix:
                hf_to_dev = self._build_hf_to_dev()
                dev = hf_to_dev.get(org_prefix.lower(), org_prefix)
                candidates.append(f"{dev}/{stem}")
            candidates.append(stem)
            for cand in candidates:
                if cand in seen or _nz(cand) == _nz(raw_value):
                    continue
                seen.add(cand)
                # check_hf=False: stem candidates are alias-index probes;
                # routing them through the HF checker would draw live-lookup
                # budget per stem AND could attest an unregistered repo.
                res = resolver.resolve(cand, "model", check_hf=False)
                hit = res.canonical_id
                if not hit or res.strategy not in ("exact", "normalized"):
                    continue
                # Never emit an edge to an HF-attested-but-unregistered id —
                # that would be a dangling parent FK (the checker confirms
                # existence on the Hub, not existence in the registry).
                if res.hf_attested_unregistered:
                    continue
                hit_name = hit.split("/", 1)[1] if "/" in hit else hit
                # Reject a "base" that is really the same identity as the draft.
                if _nz(hit) == _nz(raw_value) or _nz(hit_name) == raw_name_nz:
                    continue
                if _nz(hit) == _nz(candidate_id):
                    continue
                existing.append({"id": hit, "relationship": "finetune"})
                base["parents"] = json.dumps(existing)
                return

    @staticmethod
    def _looks_like_hf_id(raw_value: str) -> bool:
        """HF id heuristic: contains a single `/` with non-empty parts on
        both sides. Conservative — won't trigger hub-stats lookups for
        bare model names or paths with multiple slashes (which are likely
        malformed)."""
        if not raw_value or raw_value.count("/") != 1:
            return False
        org, name = raw_value.split("/", 1)
        return bool(org.strip()) and bool(name.strip())

    def _lookup_hub_stats(
        self, hf_id: str, target_canonical: Optional[str] = None,
    ) -> Optional[dict]:
        """Query hub-stats live for `hf_id` and return a partial draft
        dict (release_date, params_billions, parents, lineage_origin_model_org_id,
        tags, metadata) ready to merge. Returns None on miss or any error.
        Uses the `aliases` table to resolve baseModels parents to our
        canonical ids, and `canonical_orgs` HF aliases to map authors.

        `target_canonical` is the candidate canonical id of the draft
        being created — passed through to enrich_draft_from_row so the
        family-version inference can suppress a self-edge."""
        if not settings.hub_stats_lookup_enabled:
            return None
        try:
            client = self._get_hub_stats_client()
            row = client.lookup(hf_id)
        except Exception:
            return None
        if row is None:
            return None
        from eval_card_registry.services import hub_stats as _hs
        try:
            aliases_to_canonical, org_alias_map = self._build_hub_stats_indices()
            return _hs.enrich_draft_from_row(
                row, aliases_to_canonical, org_alias_map,
                target_canonical=target_canonical,
            )
        except Exception:
            return None

    def _get_hub_stats_client(self):
        """Lazy-init the hub-stats client. Reused across lookups."""
        if self._hub_stats_client is None:
            from eval_card_registry.services.hub_stats import HubStatsClient
            self._hub_stats_client = HubStatsClient()
        return self._hub_stats_client

    def _build_hub_stats_indices(self) -> tuple[dict[str, str], dict[str, str]]:
        """Cache + return the indices `enrich_draft_from_row` needs:
        - normalized canonical-alias → canonical_id (so baseModels parents
          can resolve to our registry's ids)
        - normalized HF org alias → canonical org_id (so author-org
          mapping picks the right slug)
        Built lazily, cached until `invalidate_resolver()` clears it.
        Lock-guarded so two concurrent threads (FastAPI threadpool) don't
        race the lazy build."""
        # Fast path: check without taking the lock to avoid the contention
        # cost on the hot path where the cache is already populated.
        cached = self._hub_stats_indices
        if cached is not None:
            return cached
        with self._hub_stats_indices_lock:
            # Double-check after acquiring — another thread may have built it.
            if self._hub_stats_indices is not None:
                return self._hub_stats_indices
            from eval_card_registry.services.hub_stats import normalize as _hsnorm

            aliases_df = self.store.table("aliases")
            models_df = self.store.table("canonical_models")

            a2c: dict[str, str] = {}
            # PASS 1 — canonical ids FIRST. A canonical's own id is a STRONGER
            # claim on its normalized form than being another canonical's alias,
            # so ids win on collision. This ordering MUST match the generator
            # (refresh_from_hub_stats.load_existing_canonical_aliases, ids-first):
            # if the live auto-create path built aliases-first it could resolve an
            # identical baseModels edge to a DIFFERENT parent id than the generator,
            # silently diverging the offline and live lineage graphs.
            for _, row in models_df.iterrows():
                cid = row.get("id")
                if isinstance(cid, str):
                    a2c.setdefault(_hsnorm(cid), cid)
            # PASS 2 — aliases fill in only forms no canonical id already claimed.
            for _, row in aliases_df.iterrows():
                if row.get("entity_type") != "model":
                    continue
                raw = row.get("raw_value")
                cid = row.get("canonical_id")
                if isinstance(raw, str) and "/" in raw and isinstance(cid, str):
                    a2c.setdefault(_hsnorm(raw), cid)

            # Use the same complete, store-backed org map as every other live
            # resolver path.  Building a reduced id/hf_org-only map here drops
            # curated package aliases such as CohereLabs -> cohere and
            # ibm-granite -> ibm, causing hub-stats lineage enrichment to emit
            # different org ids from the bulk refresh for the very same row.
            org_map = self._build_hf_to_dev()

            self._hub_stats_indices = (a2c, org_map)
            return self._hub_stats_indices

    def _build_hf_id_index(self) -> dict[str, str]:
        """Cache + return `normalized HF id -> HF-true id` built from the
        `hub_stats_index` table. Lets the read-only resolve path confirm an
        exact HF model id that never landed in the registry. Returns `{}` when
        the table is absent or empty. Lock-guarded double-checked lazy build
        (mirrors `_build_hub_stats_indices`); cleared by `invalidate_resolver`."""
        cached = self._hf_id_index
        if cached is not None:
            return cached
        with self._hub_stats_indices_lock:
            if self._hf_id_index is not None:
                return self._hf_id_index
            from eval_card_registry.services.hub_stats import normalize as _hsnorm

            index: dict[str, str] = {}
            if not self.store.has_table("hub_stats_index"):
                self._hf_id_index = index
                return index
            df = self.store.table("hub_stats_index")
            if df is None or df.empty:
                self._hf_id_index = index
                return index
            has_norm = "id_norm" in df.columns
            for _, row in df.iterrows():
                hf_id = row.get("id")
                if not isinstance(hf_id, str) or not hf_id.strip():
                    continue
                key = row.get("id_norm") if has_norm else None
                if not isinstance(key, str) or not key:
                    key = _hsnorm(hf_id)
                # Two DISTINCT repos collapsing to one id_norm make
                # normalized matching ambiguous — the entry degrades to a
                # verbatim-only dict {lower_id: true_id} that the verifier
                # answers only case-insensitively (anything else falls
                # through to the live tier).
                existing = index.get(key)
                if existing is None:
                    index[key] = hf_id
                elif isinstance(existing, str):
                    if existing.lower() != hf_id.lower():
                        index[key] = {
                            existing.lower(): existing,
                            hf_id.lower(): hf_id,
                        }
                else:
                    existing.setdefault(hf_id.lower(), hf_id)
            self._hf_id_index = index
            return index

    def _build_hf_to_dev(self) -> dict[str, str]:
        """Two-tier org map (HF-org-lowercase -> developer/community slug) for live
        auto-create casing. Single source: the shared store-backed dev-org map
        (`canonical_orgs` id/hf_org + the org ALIAS rows — canonical_orgs has no
        aliases column — so the curated alias tier reaches the live path the same
        way it reaches the generators + resolver). Matches the seeded HF casing."""
        from eval_entity_resolver.fold import build_org_dev_map_from_store

        adf = self.store.table("aliases")
        org_alias_pairs = (
            zip(adf[adf["entity_type"] == "org"]["raw_value"],
                adf[adf["entity_type"] == "org"]["canonical_id"])
            if adf is not None and not adf.empty else []
        )
        return build_org_dev_map_from_store(
            self.store.table("canonical_orgs").to_dict("records"), org_alias_pairs
        )

    def _ensure_hf_org(self, org_id: Optional[str]) -> None:
        """Create a community `canonical_orgs` row for an HF-derived org id
        if it does not already exist, so a freshly-minted HF-confirmed model's
        `org_id` FK resolves. No-op in read-only mode or when the row exists.
        Mirrors the seeded `orgs.generated.yaml` shape (kind community,
        review_status reviewed, hf_org set)."""
        if not org_id or settings.read_only:
            return
        orgs_df = self.store.table("canonical_orgs")
        if (orgs_df["id"] == org_id).any():
            return
        now = _now()
        queries.upsert_entity(
            self.store,
            "canonical_orgs",
            {
                "id": org_id,
                "display_name": org_id,
                "parent_org_id": None,
                "website": None,
                "logo_url": None,
                "hf_org": org_id,
                "kind": "community",
                "tags": "[]",
                "metadata": "{}",
                "review_status": "reviewed",
                "created_at": now,
                "updated_at": now,
            },
            buffered=True,
        )

    def _resolve_model_org_id(self, raw_value: str) -> Optional[str]:
        """Map an HF-shaped `org/name` raw value to a canonical org id.

        On no-match, recurse into `self.resolve()` for the org part so an
        org draft + alias get auto-created — same machinery as for model
        auto-creates. Without this, a freshly-drafted model gets
        `org_id=None` and the FK to `canonical_orgs.id` silently breaks.
        Read-only mode skips the auto-create (returns the resolver's own
        canonical_id, possibly None) since draft writes are forbidden.
        """
        if "/" not in raw_value:
            return None
        raw_org = raw_value.split("/", 1)[0].strip()
        if not raw_org:
            return None
        if settings.read_only:
            return self._get_resolver().resolve(raw_org, "org", None).canonical_id
        # `source_field=None` because this is an implicit resolve from a
        # model raw value, not a top-level request from a caller.
        result_dict = self.resolve(raw_org, "org", None, None)
        return result_dict.get("canonical_id")

    def _find_alias_id(
        self,
        raw_value: str,
        entity_type: str,
        source_config: Optional[str],
    ) -> Optional[str]:
        df = self.store.table("aliases")
        mask = (df["raw_value"] == raw_value) & (df["entity_type"] == entity_type)
        if source_config:
            mask = mask & (df["source_config"] == source_config)
        else:
            mask = mask & df["source_config"].isna()
        rows = df[mask]
        return rows.iloc[0]["id"] if not rows.empty else None
