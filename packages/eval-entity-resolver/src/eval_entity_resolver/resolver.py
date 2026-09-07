"""The bare resolver. Matches a raw value to a canonical id via the
strategy chain (exact → normalized → fuzzy → no_match), and — when
given a `CanonicalStore` — enriches the result with the matched
canonical's metadata, parent edges, model-specific lineage fields,
and quantized-chain root collapse.

The enrichment matches the HTTP API's response shape exactly. Callers
using the resolver standalone get the same `ResolutionResult` they'd
get back from `POST /api/v1/resolve`."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from eval_entity_resolver.alias_store import AliasStore
from eval_entity_resolver.canonical_store import CanonicalStore, _hf_repo_id_of
from eval_entity_resolver.eee import _keyword_extract, prepare_eval_name_segments
from eval_entity_resolver.normalization import normalize
from eval_entity_resolver.models import (
    HfIdHit,
    ResolutionResult,
    ResolverConfig,
    looks_like_hf_id,
)
from eval_entity_resolver.strategies.exact import exact_match
from eval_entity_resolver.strategies.normalized import normalized_match
from eval_entity_resolver.strategies.fuzzy import fuzzy_match

# Confidence assigned to normalized-match results. Below 1.0 (exact) and
# above _STEM_CONFIDENCE (0.90, fuzzy) so the provenance is clear in the
# resolution log.
_NORMALIZED_CONFIDENCE = 0.95

# Two-character segments are language codes and version fragments, never a
# benchmark, and the registry does hold a handful of two-letter canonicals
# (`if`, `mc`, `nq`) a bare segment would otherwise collide with.
_MIN_BENCHMARK_SEGMENT_LEN = 3


@dataclass
class StructuredBenchmark:
    """One dotted `evaluation_name` resolved against the benchmark
    vocabulary. `benchmark_raw` is the surface form to record downstream:
    the winning segment plus any subset segments, which is what the
    producer's slice machinery reads."""

    canonical_id: str
    benchmark_raw: str
    subset: Optional[str]


class Resolver:
    def __init__(
        self,
        store: AliasStore,
        config: Optional[ResolverConfig] = None,
        canonical_store: Optional[CanonicalStore] = None,
        hf_id_checker: Optional[Callable[[str], Optional[HfIdHit]]] = None,
    ) -> None:
        """`store` is required (alias matching is the resolver's core job).
        `canonical_store` is optional — when provided, results are
        enriched with parent / lineage / metadata fields. Without it,
        only the basic match fields (canonical_id, strategy, confidence)
        are populated.

        `hf_id_checker` is optional — when provided, HF-shaped model raw
        values are checked against it as a resolution step: a verbatim
        (case-insensitive) confirmation outranks any alias match, and a
        normalized (separator-collapse) confirmation ranks between the
        normalized-alias and fuzzy steps. Without it, the chain is
        identical to the classic exact → normalized → fuzzy order."""
        self.store = store
        self.config = config or ResolverConfig()
        self.canonical_store = canonical_store
        self.hf_id_checker = hf_id_checker

    @classmethod
    def from_parquet(
        cls,
        path: str | Path,
        config: Optional[ResolverConfig] = None,
    ) -> "Resolver":
        """Load both alias and canonical stores from a parquet directory
        (e.g. `./fixtures/`) and return a fully-enriching resolver. This
        is the recommended convenience for callers who want the same
        response shape as the HTTP API."""
        return cls(
            AliasStore.from_parquet(path),
            config=config,
            canonical_store=CanonicalStore.from_parquet(path),
        )

    @classmethod
    def from_hf(
        cls,
        repo_id: str,
        config: Optional[ResolverConfig] = None,
    ) -> "Resolver":
        """Load both stores from a HF Dataset repo and return a
        fully-enriching resolver."""
        return cls(
            AliasStore.from_hf(repo_id),
            config=config,
            canonical_store=CanonicalStore.from_hf(repo_id),
        )

    def resolve(
        self,
        raw_value: str,
        entity_type: str,
        source_config: Optional[str] = None,
        mode: str = "resolve",
        check_hf: bool = True,
    ) -> ResolutionResult:
        """`check_hf=False` disables the injected HF id check for this call.
        Internal inference resolves (e.g. tier-3 stem probing) use it so a
        batch of stem candidates never draws live-lookup budget."""
        # 1. Exact
        canonical_id = exact_match(raw_value, entity_type, source_config, self.store)

        # 2. HF id check (models only, HF-shaped only, checker injected).
        # Called at most once per resolve; the hit is reused by the
        # normalized-tier step below.
        hf_hit = (
            self._maybe_check_hf_id(raw_value, entity_type, canonical_id)
            if check_hf else None
        )
        if hf_hit is not None and hf_hit.verbatim:
            if canonical_id is not None and canonical_id == hf_hit.hf_id:
                # Agreement — the registry result is richer; stamp the
                # runtime attestation onto it.
                result = self._enrich(
                    raw_value, entity_type, source_config, canonical_id, "exact", 1.0
                )
                return self._stamp_hf_attestation(result, hf_hit)
            # Disagreement or registry miss — the verbatim HF id wins over
            # any alias mapping (owner decision: a string that IS a real HF
            # repo id resolves to itself).
            return self._hf_checker_result(
                raw_value, entity_type, source_config, hf_hit, "exact", 1.0
            )

        if canonical_id is not None:
            result = self._enrich(
                raw_value, entity_type, source_config, canonical_id, "exact", 1.0
            )
            if hf_hit is not None and canonical_id == hf_hit.hf_id:
                result = self._stamp_hf_attestation(result, hf_hit)
            return result

        # 3. Normalized alias (confidence 0.95 — only return if above
        # threshold). A curated normalized alias outranks a merely
        # separator-collapsed HF index hit.
        if _NORMALIZED_CONFIDENCE >= self.config.threshold:
            canonical_id = normalized_match(raw_value, entity_type, self.store, source_config)
            if canonical_id is not None:
                result = self._enrich(
                    raw_value, entity_type, source_config,
                    canonical_id, "normalized", _NORMALIZED_CONFIDENCE,
                )
                if hf_hit is not None and canonical_id == hf_hit.hf_id:
                    result = self._stamp_hf_attestation(result, hf_hit)
                return result

        # 4. Normalized-tier HF id check hit (separator-collapse match).
        if hf_hit is not None and not hf_hit.verbatim:
            return self._hf_checker_result(
                raw_value, entity_type, source_config,
                hf_hit, "normalized", _NORMALIZED_CONFIDENCE,
            )

        # Exact-only mode stops here: no fuzzy inference.
        if mode == "exact":
            return ResolutionResult(
                raw_value=raw_value,
                entity_type=entity_type,
                source_config=source_config,
                canonical_id=None,
                strategy="no_match",
                confidence=0.0,
            )

        # 5. Fuzzy — thread the store-backed curated org map (incl. orgs.yaml
        # alias tier) into the org-agreement guard so org-equivalent namespaces
        # (AlephAlpha/aleph-alpha, MiniMaxAI/minimax, Alibaba-NLP/alibaba) fold
        # and match.
        canonical_id, confidence, inferred_platform = fuzzy_match(
            raw_value, entity_type, self.config.threshold, self.store, source_config,
            org_dev_map=self._org_fold_map(), known_orgs=self._known_orgs(),
        )
        if canonical_id is not None:
            result = self._enrich(
                raw_value, entity_type, source_config,
                canonical_id, "fuzzy", confidence,
            )
            # Thread the captured inference_platform onto the result. This is
            # the per-run platform read off an EXPLICIT host token in the raw
            # id (a `together/`-prefix or `-bedrock`-suffix), which WINS — an
            # explicit host token in the id is the strongest per-run platform
            # fact. Only set it when a token was actually present (None
            # otherwise), so non-host ids leave the field untouched.
            if inferred_platform is not None:
                result.inference_platform = inferred_platform
            return result

        # 6. No match
        return ResolutionResult(
            raw_value=raw_value,
            entity_type=entity_type,
            source_config=source_config,
            canonical_id=None,
            strategy="no_match",
            confidence=0.0,
        )

    def resolve_structured_metric_id(
        self,
        raw_id: Optional[str],
        source_config: Optional[str] = None,
        catch_all_ids: frozenset = frozenset(),
    ) -> Optional[str]:
        """Positionless structural resolution for a namespaced
        `metric_config.metric_id` (`lmarena.elo.overall`,
        `vals_ai.mgsm.mgsm_de.accuracy`, `openeval.bbq.exact-match`).

        Segment roles vary by adapter — the metric can sit last
        (vals_ai), in the middle (lmarena), or be absent entirely
        (artificial_analysis) — so no positional parse is trusted.
        Instead every segment after the adapter namespace is resolved
        against the registry's metric vocabulary (exact → normalized
        tiers only; a fuzzy near-miss on a short token means a different
        metric, not a spelling variant), and registry membership decides
        which segment is the metric.

        `catch_all_ids` marks no-information buckets (registry metric
        entries whose metadata carries `"catch_all": true`, e.g. the
        generic `score` that raw score-field names resolve to). A
        catch-all hit is treated as "no disclosure" so a field name never
        outranks prose that names the real metric.

        Returns the canonical id when exactly one distinct non-catch-all
        metric is disclosed; None otherwise (no hits, only catch-all
        hits, or conflicting hits) — callers fall back to their existing
        description/name path unchanged.
        """
        if not raw_id or not isinstance(raw_id, str):
            return None
        raw_id = raw_id.strip()
        if ("." not in raw_id and "/" not in raw_id) or any(c.isspace() for c in raw_id):
            return None
        segments = [s for s in re.split(r"[./]", raw_id) if s]
        if len(segments) < 2:
            return None
        hits: list[str] = []
        for segment in segments[1:]:  # segments[0] is the adapter namespace
            canonical = exact_match(
                segment, "metric", source_config, self.store
            ) or normalized_match(segment, "metric", self.store, source_config)
            if canonical is not None:
                hits.append(canonical)
        specific = {h for h in hits if h not in catch_all_ids}
        if len(specific) == 1:
            return next(iter(specific))
        return None

    def resolve_structured_benchmark(
        self,
        raw_name: Optional[str],
        source_config: Optional[str] = None,
    ) -> Optional["StructuredBenchmark"]:
        """Positionless structural resolution for a dotted `evaluation_name`
        (`bbq.bbq.overall`, `MMLU.MMLU-Pro.overall`, `vals_ai.mmlu_pro.biology`).

        The benchmark-side counterpart to `resolve_structured_metric_id`, and
        for the same reason: segment roles vary by adapter. The benchmark can
        sit first (`bbq.bbq.overall`), second (`vals_ai.mmlu_pro.biology`) or
        third (`{composite}.{family}.{benchmark}.{split}`), so no positional
        parse is trusted. Every segment is probed against the registry's
        benchmark vocabulary and membership decides.

        Before probing: identical adjacent segments collapse, aggregate
        markers (`overall`) and purely numeric segments are dropped, a
        leading segment naming the row's own `source_config` is dropped, and
        a trailing segment that names a metric is dropped (the legacy
        `bfcl.live.live_accuracy` contract, where the last segment is the
        metric).

        Each hit is then re-tried with its trailing segments attached, from
        the shallowest hit down, and the first surface form the registry
        knows wins. That keeps the MOST SPECIFIC reading: a leading source /
        composite / family namespace loses to the benchmark it qualifies
        (`MMLU.MMLU-Pro` is MMLU-Pro), while a subset that only means
        something under its parent stays with it (`vals_ai.mmlu_pro.math` is
        MMLU-Pro's math subject, not the MATH dataset). Segments after the
        winner are the subset, kept on `benchmark_raw` so the producer's
        slice machinery still sees them.

        Returns None when no segment resolves, so callers fall back to
        `clean_eval_name` unchanged.
        """
        segments = prepare_eval_name_segments(raw_name)
        if segments is None:
            return None
        if (
            len(segments) > 1
            and source_config
            and normalize(segments[0]) == normalize(source_config)
        ):
            segments = segments[1:]      # leading source namespace, not a benchmark
        if len(segments) > 1 and self._segment_is_metric(segments[-1], source_config):
            segments = segments[:-1]

        hits: list[tuple[int, str]] = []
        for i, segment in enumerate(segments):
            probe = (
                self.resolve(segment, "benchmark", source_config, check_hf=False)
                if len(segment) >= _MIN_BENCHMARK_SEGMENT_LEN
                else None
            )
            if probe is not None and probe.canonical_id is not None:
                hits.append((i, probe.canonical_id))
                continue
            nxt = segments[i + 1] if i + 1 < len(segments) else None
            if not self._is_namespace_segment(segment, nxt, source_config):
                # An unrecognized segment ends the identity path: anything
                # deeper is a subset of a benchmark the registry doesn't know
                # (`vals_ai.programbench.strict`), and matching it on its own
                # would report the subset as the benchmark.
                break
        if not hits:
            return None

        def _spell(parts: list[str]) -> str:
            return " ".join(p.replace("_", " ") for p in parts)

        for index, canonical in hits:
            benchmark_raw = _spell(segments[index:])
            if index == len(segments) - 1:
                return StructuredBenchmark(canonical, benchmark_raw, None)
            joined = self.resolve(
                benchmark_raw, "benchmark", source_config, check_hf=False
            )
            if joined.canonical_id is not None:
                return StructuredBenchmark(
                    joined.canonical_id, benchmark_raw, _spell(segments[index + 1:])
                )

        index, canonical = hits[-1]
        return StructuredBenchmark(
            canonical, _spell(segments[index:]), _spell(segments[index + 1:]) or None
        )

    def _is_namespace_segment(
        self, segment: str, next_segment: Optional[str], source_config: Optional[str]
    ) -> bool:
        """A leading segment that qualifies the benchmark rather than being
        one, so the probe may look past it: a composite or family the registry
        knows, or a stem the next segment extends (`alpaca_eval` before
        `alpaca_eval_v2`)."""
        for entity_type in ("composite", "family"):
            if exact_match(
                segment, entity_type, source_config, self.store
            ) or normalized_match(segment, entity_type, self.store, source_config):
                return True
        return next_segment is not None and normalize(next_segment).startswith(
            normalize(segment) + " "
        )

    def _segment_is_metric(self, segment: str, source_config: Optional[str]) -> bool:
        """The documented dotted-name contract is that the last segment is
        the metric (`bfcl.live.live_accuracy`). Honour it when the segment
        resolves as a metric or reads as one to the metric extractor, so a
        metric tail is never mistaken for a subset."""
        spelled = segment.replace("_", " ")
        if exact_match(segment, "metric", source_config, self.store) or normalized_match(
            segment, "metric", self.store, source_config
        ):
            return True
        return _keyword_extract(spelled.lower()) is not None

    # ------------------------------------------------------------------
    # HF id check (injected)
    # ------------------------------------------------------------------

    def _maybe_check_hf_id(
        self,
        raw_value: str,
        entity_type: str,
        exact_canonical_id: Optional[str],
    ) -> Optional[HfIdHit]:
        """Run the injected HF id checker when it can change the outcome.

        Skipped when: no checker, non-model, not HF-shaped, or the exact
        alias hit is byte-equal to the raw value AND that canonical row
        either is already HF-attested (oracle `resolution_source == "hf"`
        or hub-stats-confirmed — the checker could only agree) or carries a
        curated/models.dev provenance claim (`models_dev`/`NA`/`curated` —
        a registered off-HF canonical like an OpenRouter or closed-API id,
        where a live check would just 404 on every resolve). A case-only or
        name-inferred (tier-3) agreement still runs the checker, because
        the checker is what recovers HF-true casing for lowercased
        drafts."""
        if self.hf_id_checker is None or entity_type != "model":
            return None
        if not looks_like_hf_id(raw_value):
            return None
        if exact_canonical_id is not None and exact_canonical_id == raw_value:
            if self._hf_check_skippable(exact_canonical_id):
                return None
        return self.hf_id_checker(raw_value)

    def _hf_check_skippable(self, canonical_id: str) -> bool:
        if self.canonical_store is None:
            return False
        ent = self.canonical_store.lookup("model", canonical_id)
        if ent is None:
            return False
        if _hf_repo_id_of(ent, canonical_id) is not None:
            return True
        src = ent.get("resolution_source")
        return isinstance(src, str) and src in ("models_dev", "NA", "curated")

    def _hf_checker_result(
        self,
        raw_value: str,
        entity_type: str,
        source_config: Optional[str],
        hit: HfIdHit,
        strategy: str,
        confidence: float,
    ) -> ResolutionResult:
        """Build the result for a winning HF id check hit. A registered
        canonical gets the full registry enrichment; an unregistered one
        gets a bare result flagged `hf_attested_unregistered` so the
        service layer can auto-create it (write mode) or serve it without
        minting (read-only)."""
        known = (
            self.canonical_store is not None
            and self.canonical_store.lookup("model", hit.hf_id) is not None
        )
        if known:
            result = self._enrich(
                raw_value, entity_type, source_config, hit.hf_id, strategy, confidence
            )
            return self._stamp_hf_attestation(result, hit)
        return ResolutionResult(
            raw_value=raw_value,
            entity_type=entity_type,
            source_config=source_config,
            canonical_id=hit.hf_id,
            strategy=strategy,
            confidence=confidence,
            resolution_source=hit.source,
            ancestry=[],
            resolution_detail={"granularity": None, "hf_repo_id": hit.hf_id},
            hf_attestation=hit.hf_id,
            hf_attested_unregistered=True,
        )

    @staticmethod
    def _stamp_hf_attestation(result: ResolutionResult, hit: HfIdHit) -> ResolutionResult:
        """The checker just runtime-attested the repo id, so surface it in
        `resolution_detail.hf_repo_id` even when the canonical row itself
        carries no HF provenance. Only stamped when the response canonical
        IS the attested id (root-collapse may have moved it)."""
        if result.canonical_id == hit.hf_id:
            result.hf_attestation = hit.hf_id
            if isinstance(result.resolution_detail, dict):
                result.resolution_detail["hf_repo_id"] = hit.hf_id
        return result

    def _org_fold_map(self) -> dict:
        """The org-fold map threaded into the fuzzy org-agreement guard, built
        ONCE from BOTH stores and cached:
          - canonical_store.org_dev_map: `_ORG_ALIASES` ∪ every canonical_orgs
            `id`/`hf_org` (lowercased -> curated id);
          - the org ALIAS rows in the alias table (`iter_alias_pairs("org")`),
            which carry the curated orgs.yaml alias tier (`AI2`->`allenai`,
            `ai21labs`->`ai21`) that is NOT a canonical_orgs column.
        Alias rows win on key collision (they ARE the curated seed). Keyed
        lowercase; `_fold_org` separator-strips on lookup so case/separator
        variants fold too."""
        cached = getattr(self, "_org_fold_map_cache", None)
        if cached is not None:
            return cached
        from eval_entity_resolver.fold import _ORG_ALIASES

        m: dict = dict(self.canonical_store.org_dev_map) if self.canonical_store else dict(_ORG_ALIASES)
        if self.store is not None:
            for raw, cid in self.store.iter_alias_pairs("org"):
                m[raw.lower()] = cid
        self._org_fold_map_cache = m
        return m

    def _known_orgs(self) -> frozenset:
        """The set of canonical_orgs ids (a developer that has a real row). The
        fuzzy org-agreement guard never separator-strip-merges two prefixes that
        resolve to DIFFERENT ids in this set — so distinct uploaders kept apart in
        canonical_orgs (the orgs_distinct_allowlist contract) stay apart at resolve
        time too. Cached; empty when no canonical_store is attached."""
        cached = getattr(self, "_known_orgs_cache", None)
        if cached is not None:
            return cached
        ids: set[str] = set()
        if self.canonical_store is not None:
            df = self.canonical_store._tables.get("org")
            if df is not None and not df.empty and "id" in df.columns:
                ids = {str(i) for i in df["id"] if isinstance(i, str)}
        self._known_orgs_cache = frozenset(ids)
        return self._known_orgs_cache

    # ------------------------------------------------------------------
    # Enrichment (no-op when no canonical_store is attached)
    # ------------------------------------------------------------------

    def build_result(
        self,
        raw_value: str,
        entity_type: str,
        source_config: Optional[str],
        canonical_id: str,
        strategy: str,
        confidence: float,
    ) -> ResolutionResult:
        """Construct an enriched `ResolutionResult` for a canonical_id
        the caller already knows — useful for callers that bypass the
        strategy chain (e.g. an alias-table cache hit, an auto-created
        draft) but want the same rich response shape. Identical to the
        enrichment that happens inside `resolve()`."""
        return self._enrich(raw_value, entity_type, source_config, canonical_id, strategy, confidence)

    def _enrich(
        self,
        raw_value: str,
        entity_type: str,
        source_config: Optional[str],
        matched_canonical_id: str,
        strategy: str,
        confidence: float,
    ) -> ResolutionResult:
        """Look up the matched canonical's row and populate the rich
        response fields. When no canonical_store is attached, the rich
        fields stay None and the result has just the basic match info."""
        if self.canonical_store is None:
            return ResolutionResult(
                raw_value=raw_value,
                entity_type=entity_type,
                source_config=source_config,
                canonical_id=matched_canonical_id,
                strategy=strategy,
                confidence=confidence,
            )

        cs = self.canonical_store
        matched_entity = cs.lookup(entity_type, matched_canonical_id)
        review_status = (matched_entity or {}).get("review_status") if matched_entity else None

        if entity_type == "model":
            fields = cs.model_metadata_fields(matched_canonical_id, matched_entity)
            # If the response collapses to a different canonical (root),
            # surface THAT canonical's review_status — keeps the response
            # internally consistent.
            if fields["canonical_id"] != matched_canonical_id:
                root_entity = cs.lookup("model", fields["canonical_id"])
                if root_entity:
                    review_status = root_entity.get("review_status") or review_status
            return ResolutionResult(
                raw_value=raw_value,
                entity_type=entity_type,
                source_config=source_config,
                canonical_id=fields["canonical_id"],
                strategy=strategy,
                confidence=confidence,
                review_status=review_status,
                parent_canonical_id=cs.parent_canonical_id("model", matched_entity),
                resolved_leaf_id=fields["resolved_leaf_id"],
                root_model_id=fields["root_model_id"],
                lineage_origin_org_id=fields["lineage_origin_org_id"],
                # Extended lineage / provenance fields. None-safe .get so a
                # store predating these keys still works.
                model_group_id=fields.get("model_group_id"),
                model_family_id=fields.get("model_family_id"),
                lineage_origin_model_id=fields.get("lineage_origin_model_id"),
                lineage_origin_model_org_id=fields.get("lineage_origin_model_org_id"),
                inference_platform=fields.get("inference_platform"),
                resolution_source=fields.get("resolution_source"),
                resolution_granularity=fields.get("resolution_granularity"),
                parents=fields["parents"],
                open_weights=fields["open_weights"],
                release_date=fields["release_date"],
                params_billions=fields["params_billions"],
                ancestry=cs.compute_ancestry("model", fields["canonical_id"], matched_entity),
                resolution_detail=cs.resolution_detail(
                    "model", fields["canonical_id"], matched_entity=matched_entity
                ),
            )

        # Benchmark: fill in hierarchy-alignment fields (family_key,
        # category) by walking canonical_families. composite_keys stays
        # empty here — see CanonicalStore.benchmark_family_enrichment for
        # why composite computation belongs in the producer.
        if entity_type == "benchmark":
            fam = cs.benchmark_family_enrichment(matched_canonical_id)
            return ResolutionResult(
                raw_value=raw_value,
                entity_type=entity_type,
                source_config=source_config,
                canonical_id=matched_canonical_id,
                strategy=strategy,
                confidence=confidence,
                review_status=review_status,
                parent_canonical_id=cs.parent_canonical_id(entity_type, matched_entity),
                family_key=fam["family_key"],
                category=fam["category"],
                composite_keys=fam["composite_keys"],
                ancestry=cs.compute_ancestry("benchmark", matched_canonical_id, matched_entity),
                resolution_detail=cs.resolution_detail(
                    "benchmark", matched_canonical_id,
                    raw_value=raw_value, matched_entity=matched_entity,
                ),
            )

        # Other non-model types (metric, harness, org, family, composite):
        # parent_canonical_id + review_status, plus ancestry/detail (family
        # carries a composite parent; the rest are roots with empty detail).
        return ResolutionResult(
            raw_value=raw_value,
            entity_type=entity_type,
            source_config=source_config,
            canonical_id=matched_canonical_id,
            strategy=strategy,
            confidence=confidence,
            review_status=review_status,
            parent_canonical_id=cs.parent_canonical_id(entity_type, matched_entity),
            ancestry=cs.compute_ancestry(entity_type, matched_canonical_id, matched_entity),
            resolution_detail=cs.resolution_detail(
                entity_type, matched_canonical_id,
                raw_value=raw_value, matched_entity=matched_entity,
            ),
        )
