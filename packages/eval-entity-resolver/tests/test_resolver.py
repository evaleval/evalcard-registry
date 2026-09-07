import pandas as pd
import pytest

from eval_entity_resolver import AliasStore, Resolver, ResolverConfig


def _store_with_aliases(*rows) -> AliasStore:
    """Build an AliasStore from (raw_value, entity_type, canonical_id, source_config, status) tuples."""
    from datetime import datetime, timezone
    import uuid

    now = datetime.now(timezone.utc).isoformat()
    records = []
    for raw_value, entity_type, canonical_id, source_config, status in rows:
        records.append(
            {
                "id": str(uuid.uuid4()),
                "raw_value": raw_value,
                "entity_type": entity_type,
                "canonical_id": canonical_id,
                "source_config": source_config,
                "source_field": None,
                "status": status,
                "strategy": "confirmed",
                "confidence": 1.0,
                "notes": None,
                "created_at": now,
                "updated_at": now,
            }
        )
    from eval_entity_resolver.alias_store import _empty_df
    df = pd.DataFrame(records) if records else _empty_df()
    return AliasStore(df)


class TestExactStrategy:
    def test_exact_match(self):
        store = _store_with_aliases(("IFEval", "benchmark", "ifeval", None, "confirmed"))
        resolver = Resolver(store)
        result = resolver.resolve("IFEval", "benchmark")
        assert result.canonical_id == "ifeval"
        assert result.strategy == "exact"
        assert result.confidence == 1.0

    def test_config_scoped_before_global(self):
        store = _store_with_aliases(
            ("MATH", "benchmark", "math-global", None, "confirmed"),
            ("MATH", "benchmark", "math-helm", "helm_lite", "confirmed"),
        )
        resolver = Resolver(store)
        result = resolver.resolve("MATH", "benchmark", source_config="helm_lite")
        assert result.canonical_id == "math-helm"

    def test_falls_back_to_global(self):
        store = _store_with_aliases(("MATH", "benchmark", "math-global", None, "confirmed"))
        resolver = Resolver(store)
        result = resolver.resolve("MATH", "benchmark", source_config="some_other_config")
        assert result.canonical_id == "math-global"

    def test_rejected_alias_skipped(self):
        store = _store_with_aliases(("IFEval", "benchmark", "ifeval", None, "rejected"))
        resolver = Resolver(store)
        result = resolver.resolve("IFEval", "benchmark")
        assert result.canonical_id is None
        assert result.strategy == "no_match"


class TestScopedAliasIsolation:
    """Scoped aliases (source_config != None) must not leak into unrelated lookups."""

    def test_scoped_isolated_from_other_config(self):
        store = _store_with_aliases(
            ("Overall", "benchmark", "ace", "ace", "confirmed"),
            ("Overall", "benchmark", "apex-v1", "apex-v1", "confirmed"),
        )
        resolver = Resolver(store)
        # Different config — scoped alias must not match.
        result = resolver.resolve("Overall", "benchmark", source_config="hfopenllm_v2")
        assert result.canonical_id is None
        assert result.strategy == "no_match"

    def test_scoped_isolated_without_source_config(self):
        store = _store_with_aliases(
            ("Arabic", "benchmark", "global-mmlu-lite", "global-mmlu-lite", "confirmed"),
        )
        resolver = Resolver(store)
        # No source_config provided — scoped alias must not match.
        result = resolver.resolve("Arabic", "benchmark")
        assert result.canonical_id is None

    def test_scoped_normalized_match_respects_scope(self):
        store = _store_with_aliases(
            ("Abstract Algebra", "benchmark", "mmlu", "helm_mmlu", "confirmed"),
        )
        resolver = Resolver(store)
        # Same scope, different casing — normalized strategy must match.
        result = resolver.resolve("abstract algebra", "benchmark", source_config="helm_mmlu")
        assert result.canonical_id == "mmlu"
        assert result.strategy == "normalized"
        # Different scope — must NOT match via normalized either.
        result = resolver.resolve("abstract algebra", "benchmark", source_config="helm_lite")
        assert result.canonical_id is None

    def test_global_alias_still_matches_any_scope(self):
        store = _store_with_aliases(("MMLU", "benchmark", "mmlu", None, "confirmed"))
        resolver = Resolver(store)
        for sc in [None, "helm_mmlu", "helm_lite", "some_other"]:
            result = resolver.resolve("MMLU", "benchmark", source_config=sc)
            assert result.canonical_id == "mmlu", f"failed for source_config={sc}"


class TestNormalizedStrategy:
    def test_normalized_match(self):
        store = _store_with_aliases(("MATH Level 5", "benchmark", "math-level-5", None, "confirmed"))
        resolver = Resolver(store)
        result = resolver.resolve("math level 5", "benchmark")
        assert result.canonical_id == "math-level-5"
        assert result.strategy == "normalized"

    def test_punctuation_stripped(self):
        store = _store_with_aliases(("GPQA!", "benchmark", "gpqa", None, "confirmed"))
        resolver = Resolver(store)
        result = resolver.resolve("gpqa", "benchmark")
        assert result.canonical_id == "gpqa"
        assert result.strategy in ("exact", "normalized")


class TestFuzzyStrategy:
    def test_suffix_strip_matches_base(self):
        """model-name-fc should match model-name via suffix stripping."""
        store = _store_with_aliases(("writer/palmyra-x-004", "model", "writer/palmyra-x-004", None, "confirmed"))
        resolver = Resolver(store)
        result = resolver.resolve("writer/palmyra-x-004-fc", "model")
        assert result.canonical_id == "writer/palmyra-x-004"
        assert result.strategy == "fuzzy"

    def test_org_normalization_matches(self):
        """deepseek-ai/model should match deepseek/model via org alias."""
        store = _store_with_aliases(("deepseek/deepseek-r1-0528", "model", "deepseek/deepseek-r1-0528", None, "confirmed"))
        resolver = Resolver(store)
        result = resolver.resolve("deepseek-ai/deepseek-r1-0528", "model")
        assert result.canonical_id == "deepseek/deepseek-r1-0528"
        assert result.strategy == "fuzzy"

    def test_distinct_versions_not_merged(self):
        """gpt-5-mini should NOT fuzzy-match to gpt-5 — they are different models."""
        store = _store_with_aliases(("openai/gpt-5-2025-08-07", "model", "openai/gpt-5-2025-08-07", None, "confirmed"))
        resolver = Resolver(store)
        result = resolver.resolve("openai/gpt-5-mini-2025-08-07", "model")
        assert result.canonical_id is None
        assert result.strategy == "no_match"

    def test_distinct_benchmarks_not_merged(self):
        """fibble2 should NOT fuzzy-match to fibble1."""
        store = _store_with_aliases(("fibble1_arena_win_rate", "benchmark", "fibble1-arena-win-rate", None, "confirmed"))
        resolver = Resolver(store)
        result = resolver.resolve("fibble2_arena_win_rate", "benchmark")
        assert result.canonical_id is None
        assert result.strategy == "no_match"

    def test_no_match_on_unrelated_string(self):
        store = _store_with_aliases(("completely-different", "harness", "x", None, "confirmed"))
        resolver = Resolver(store)
        result = resolver.resolve("unrelated string xyz", "harness")
        assert result.canonical_id is None
        assert result.strategy == "no_match"

    def test_thinking_budget_suffix_stripped(self):
        """claude-opus-4-5-thinking-16k should match claude-opus-4-5 (card_backend pattern)."""
        store = _store_with_aliases(
            ("anthropic/claude-opus-4-5", "model", "anthropic/claude-opus-4-5", None, "confirmed")
        )
        resolver = Resolver(store)
        result = resolver.resolve("anthropic/claude-opus-4-5-thinking-16k", "model")
        assert result.canonical_id == "anthropic/claude-opus-4-5"
        assert result.strategy == "fuzzy"

    def test_thinking_budget_prefers_thinking_canonical_when_aliased(self):
        """When a thinking-mode canonical IS aliased, `-thinking-Nk` should
        peel only the `-Nk` budget and resolve to the thinking variant —
        NOT collapse to the bare base (which would lose the thinking-mode
        signal in eval results). Drop-thinking behavior remains the
        fallback when no thinking-mode canonical exists (covered by the
        prior test)."""
        store = _store_with_aliases(
            ("anthropic/claude-haiku-4-5-20251001",
             "model", "anthropic/claude-haiku-4-5-20251001", None, "confirmed"),
            ("anthropic/claude-haiku-4-5-20251001-thinking",
             "model", "anthropic/claude-haiku-4-5-20251001-thinking", None, "confirmed"),
        )
        resolver = Resolver(store)
        for budget in ("1k", "8k", "16k", "32k"):
            raw = f"anthropic/claude-haiku-4-5-20251001-thinking-{budget}"
            result = resolver.resolve(raw, "model")
            assert result.canonical_id == "anthropic/claude-haiku-4-5-20251001-thinking", (
                f"{raw!r} expected to peel just the budget and stay on the "
                f"thinking-mode canonical; got {result.canonical_id!r}"
            )
            assert result.strategy == "fuzzy"

    def test_thinking_none_suffix_stripped(self):
        store = _store_with_aliases(
            ("anthropic/claude-opus-4-5", "model", "anthropic/claude-opus-4-5", None, "confirmed")
        )
        resolver = Resolver(store)
        result = resolver.resolve("anthropic/claude-opus-4-5-thinking-none", "model")
        assert result.canonical_id == "anthropic/claude-opus-4-5"
        assert result.strategy == "fuzzy"

    def test_date_version_suffix_no_longer_strips_to_family(self):
        """Trailing 8-digit YYYYMMDD must NOT silently collapse to the
        family pointer — that loses per-snapshot release_date. The
        resolver returns no_match so the caller's auto-create + hub-stats
        path produces a properly-linked snapshot canonical with a
        version-axis parent edge."""
        store = _store_with_aliases(
            ("anthropic/claude-opus-4-5", "model", "anthropic/claude-opus-4-5", None, "confirmed")
        )
        resolver = Resolver(store)
        result = resolver.resolve("anthropic/claude-opus-4-5-20251101", "model")
        assert result.canonical_id is None
        assert result.strategy == "no_match"

    def test_date_version_resolves_when_snapshot_canonical_exists(self):
        """When the snapshot canonical IS aliased in the registry,
        normalized/exact match wins before fuzzy ever tries to strip —
        the resolver returns the snapshot directly. This is the desired
        behavior: snapshot canonicals carry their own release_date and
        the family-version edge is on the canonical itself."""
        store = _store_with_aliases(
            ("anthropic/claude-opus-4-5", "model", "anthropic/claude-opus-4-5", None, "confirmed"),
            ("anthropic/claude-opus-4-5-20251101", "model",
             "anthropic/claude-opus-4-5-20251101", None, "confirmed"),
        )
        resolver = Resolver(store)
        result = resolver.resolve("anthropic/claude-opus-4-5-20251101", "model")
        assert result.canonical_id == "anthropic/claude-opus-4-5-20251101"

    def test_date_plus_thinking_no_longer_strips_to_family(self):
        """Compound date+mode (`-20251101-thinking-16k`) used to double-
        strip down to the family. Now: thinking-budget peel runs first
        (`...-thinking`); when no aliased mode-promoted snapshot
        canonical exists, the strip ladder runs out without producing
        the bare-family candidate. Auto-create owns the rest."""
        store = _store_with_aliases(
            ("anthropic/claude-opus-4-5", "model", "anthropic/claude-opus-4-5", None, "confirmed")
        )
        resolver = Resolver(store)
        result = resolver.resolve("anthropic/claude-opus-4-5-20251101-thinking-16k", "model")
        assert result.canonical_id is None
        assert result.strategy == "no_match"

    def test_dot_version_normalizes_to_hyphen(self):
        """claude-opus-4.5 should normalize the same as claude-opus-4-5."""
        store = _store_with_aliases(
            ("anthropic/claude-opus-4-5", "model", "anthropic/claude-opus-4-5", None, "confirmed")
        )
        resolver = Resolver(store)
        result = resolver.resolve("anthropic/claude-opus-4.5", "model")
        assert result.canonical_id == "anthropic/claude-opus-4-5"
        assert result.strategy == "normalized"

    def test_meta_llama_org_alias(self):
        """meta-llama/ → meta/ via expanded org alias map."""
        store = _store_with_aliases(("meta/llama-3-70b", "model", "meta/llama-3-70b", None, "confirmed"))
        resolver = Resolver(store)
        result = resolver.resolve("meta-llama/llama-3-70b", "model")
        assert result.canonical_id == "meta/llama-3-70b"
        assert result.strategy == "fuzzy"

    def test_qwen_org_alias_to_alibaba(self):
        """`Qwen/<model>` (HF-namespace upload form) → `alibaba/<model>`
        via the qwen → alibaba org alias. The reverse direction
        (alibaba → qwen) was rejected because of the non-Qwen
        `alibaba/mineru2-pipeline` entry; this direction has no analogous
        collision."""
        store = _store_with_aliases(
            ("alibaba/qwen2-vl-7b-instruct", "model", "alibaba/qwen2-vl-7b-instruct", None, "confirmed")
        )
        resolver = Resolver(store)
        result = resolver.resolve("Qwen/Qwen2-VL-7B-Instruct", "model")
        assert result.canonical_id == "alibaba/qwen2-vl-7b-instruct"

    def test_year_only_no_longer_strips_to_family(self):
        """`openai/gpt-5-2024` used to peel the trailing year to
        `openai/gpt-5`. Now: year-only is exclusively a bare-family
        peel and the auto-create path owns it. Returns no_match."""
        store = _store_with_aliases(("openai/gpt-5", "model", "openai/gpt-5", None, "confirmed"))
        resolver = Resolver(store)
        result = resolver.resolve("openai/gpt-5-2024", "model")
        assert result.canonical_id is None
        assert result.strategy == "no_match"

    def test_iso_date_strip_does_not_apply_to_non_openai(self):
        """The OpenAI date-peel is org-scoped: `meta/llama-3-2024` must NOT
        strip to `meta/llama-3` because Meta's release cadence doesn't use
        the OpenAI YYYY-MM-DD truncated-month convention."""
        store = _store_with_aliases(("meta/llama-3", "model", "meta/llama-3", None, "confirmed"))
        resolver = Resolver(store)
        result = resolver.resolve("meta/llama-3-2024", "model")
        assert result.canonical_id is None
        assert result.strategy == "no_match"

    def test_iso_date_strip_rejects_non_year_4digit_tail(self):
        """The year-range guard (2015–2035) prevents arbitrary 4-digit
        tails like `-1024` (a number, not a year) from triggering the
        peel even on OpenAI-shaped raws."""
        store = _store_with_aliases(("openai/gpt-5", "model", "openai/gpt-5", None, "confirmed"))
        resolver = Resolver(store)
        result = resolver.resolve("openai/gpt-5-1024", "model")
        assert result.canonical_id is None
        assert result.strategy == "no_match"

    def test_diverse_date_shapes_all_no_match_when_only_family_aliased(self):
        """Regression coverage for the three real-world snapshot shapes
        called out during planning. Each should fall through fuzzy
        without collapsing to its family — the auto-create path then
        owns producing a snapshot canonical with proper parents.

          - google/gemini-exp-1114        — trailing MMDD (4-digit)
          - stepfun/step-2-16k-202411     — trailing YYYYMM (6-digit)
          - tencent/hunyuan-turbos-20250313 — trailing YYYYMMDD (8-digit)

        The 4-digit and 6-digit cases never had a fuzzy strip in the
        first place; they're tested here together with the 8-digit
        case so the behavior is documented in one place."""
        store = _store_with_aliases(
            ("google/gemini-exp", "model", "google/gemini-exp", None, "confirmed"),
            ("stepfun/step-2-16k", "model", "stepfun/step-2-16k", None, "confirmed"),
            ("tencent/hunyuan-turbos", "model", "tencent/hunyuan-turbos", None, "confirmed"),
        )
        resolver = Resolver(store)
        for raw in [
            "google/gemini-exp-1114",
            "stepfun/step-2-16k-202411",
            "tencent/hunyuan-turbos-20250313",
        ]:
            result = resolver.resolve(raw, "model")
            assert result.canonical_id is None, f"{raw!r} unexpectedly matched"
            assert result.strategy == "no_match"

    def test_iso_full_date_no_longer_collapses_to_family(self):
        """`openai/gpt-5-2025-08-07` used to peel through `-2025-08`
        and `-2025` and finally land on `openai/gpt-5` (family). Now:
        the strip ladder still tries `-2025-08` and `-2025` as
        intermediate snapshot candidates, but stops short of the bare
        family. When NO intermediate snapshot canonical is aliased,
        no_match is returned; auto-create produces the snapshot."""
        store = _store_with_aliases(("openai/gpt-5", "model", "openai/gpt-5", None, "confirmed"))
        resolver = Resolver(store)
        result = resolver.resolve("openai/gpt-5-2025-08-07", "model")
        assert result.canonical_id is None
        assert result.strategy == "no_match"

    def test_iso_date_strip_prefers_truncated_month_canonical(self):
        """When the registry has both the truncated-month canonical and
        the family root, the peel stops at the first hit (truncated
        month) — preserves snapshot identity instead of over-collapsing."""
        store = _store_with_aliases(
            ("openai/gpt-5", "model", "openai/gpt-5", None, "confirmed"),
            ("openai/gpt-5-2025-08", "model", "openai/gpt-5-2025-08", None, "confirmed"),
        )
        resolver = Resolver(store)
        result = resolver.resolve("openai/gpt-5-2025-08-07", "model")
        assert result.canonical_id == "openai/gpt-5-2025-08"
        assert result.strategy == "fuzzy"

    def test_iso_date_with_unknown_host_prefix_no_longer_collapses(self):
        """`unknown/gpt-5-2025-08-07` — host strip drops `unknown/` and
        the OpenAI ISO-date peel runs. With the bare-family candidate
        no longer emitted by the peel, no candidate hits when only the
        family is aliased. Auto-create owns the snapshot."""
        store = _store_with_aliases(
            ("openai/gpt-5", "model", "openai/gpt-5", None, "confirmed"),
            ("gpt-5",        "model", "openai/gpt-5", None, "confirmed"),
        )
        resolver = Resolver(store)
        result = resolver.resolve("unknown/gpt-5-2025-08-07", "model")
        assert result.canonical_id is None
        assert result.strategy == "no_match"


class TestNoMatch:
    def test_empty_store(self):
        store = _store_with_aliases()
        resolver = Resolver(store)
        result = resolver.resolve("anything", "benchmark")
        assert result.canonical_id is None
        assert result.strategy == "no_match"
        assert result.confidence == 0.0


class TestPromotedVariantResolution:
    """After the alias-promotion pass, instruct/chat/quantized/snapshot
    variants are first-class canonicals with their own aliases. Resolving
    a raw value matching one of those variants must NOT collapse to the
    base — eval scores on `Llama-3-8B-Instruct` aren't comparable to
    scores on the base `Llama-3-8B`. Regression coverage: there is no
    `_FAMILY_STAGE_SUFFIXES` strip in fuzzy.py — the resolver relies on
    explicit alias entries for instruct/chat/etc., and stays out of the
    way for unknown post-training suffixes."""

    def _registry_like_store(self):
        """Mini fixture mimicking the post-promotion registry: base + promoted
        instruct + promoted instruct-quant, each with their own surface-form
        aliases."""
        return _store_with_aliases(
            # Base
            ("Llama-3-8B", "model", "meta/llama-3-8b", None, "confirmed"),
            ("meta/llama-3-8b", "model", "meta/llama-3-8b", None, "confirmed"),
            # Promoted instruct (variant/mode of the base)
            ("Llama-3-8B-Instruct", "model", "meta/llama-3-8b-instruct", None, "confirmed"),
            ("meta-llama/Meta-Llama-3-8B-Instruct", "model", "meta/llama-3-8b-instruct", None, "confirmed"),
            ("meta/llama-3-8b-instruct", "model", "meta/llama-3-8b-instruct", None, "confirmed"),
            # Promoted instruct-turbo (quantized of the instruct)
            ("Llama-3-8B-Instruct-Turbo", "model", "meta/llama-3-8b-instruct-turbo", None, "confirmed"),
            ("meta/llama-3-8b-instruct-turbo", "model", "meta/llama-3-8b-instruct-turbo", None, "confirmed"),
            # Promoted snapshot (variant/version of a base)
            ("gpt-4-0613", "model", "openai/gpt-4-0613", None, "confirmed"),
            ("openai/gpt-4-0613", "model", "openai/gpt-4-0613", None, "confirmed"),
            ("openai/gpt-4", "model", "openai/gpt-4", None, "confirmed"),
        )

    def test_instruct_resolves_to_instruct_canonical(self):
        resolver = Resolver(self._registry_like_store())
        result = resolver.resolve("Llama-3-8B-Instruct", "model")
        assert result.canonical_id == "meta/llama-3-8b-instruct"
        assert result.strategy == "exact"

    def test_base_still_resolves_to_base(self):
        resolver = Resolver(self._registry_like_store())
        result = resolver.resolve("Llama-3-8B", "model")
        assert result.canonical_id == "meta/llama-3-8b"
        assert result.strategy == "exact"

    def test_doubled_org_prefix_instruct_resolves_to_instruct(self):
        """HF-style `meta-llama/Meta-Llama-3-8B-Instruct` (org form duplicated
        inside the bare model id) should land on the instruct canonical."""
        resolver = Resolver(self._registry_like_store())
        result = resolver.resolve("meta-llama/Meta-Llama-3-8B-Instruct", "model")
        assert result.canonical_id == "meta/llama-3-8b-instruct"

    def test_quantized_variant_resolves_to_quantized_canonical(self):
        resolver = Resolver(self._registry_like_store())
        result = resolver.resolve("Llama-3-8B-Instruct-Turbo", "model")
        assert result.canonical_id == "meta/llama-3-8b-instruct-turbo"

    def test_snapshot_resolves_to_snapshot_canonical(self):
        resolver = Resolver(self._registry_like_store())
        result = resolver.resolve("gpt-4-0613", "model")
        assert result.canonical_id == "openai/gpt-4-0613"
        # Sanity check that the snapshot is NOT collapsed onto the base
        assert result.canonical_id != "openai/gpt-4"

    def test_quant_falls_through_to_nearest_parent(self):
        """When a specific quantization isn't a canonical, the fuzzy stem
        strip drops the `-fp8` suffix and lands on the next-up canonical
        (the unquantized instruct, in this case). This is the
        precision-loss policy we explicitly opted into."""
        resolver = Resolver(self._registry_like_store())
        result = resolver.resolve("meta/llama-3-8b-instruct-fp8", "model")
        # -fp8 is in _STRIP_SUFFIXES, so fuzzy strips and lands on instruct
        assert result.canonical_id == "meta/llama-3-8b-instruct"
        assert result.strategy == "fuzzy"

    def test_unknown_finetune_suffix_does_not_collapse_to_base(self):
        """If a raw value has an unrecognized suffix (a community finetune
        we haven't catalogued), the resolver must NOT silently strip it
        and land on the base — that would misattribute scores. Returns
        no_match so the caller can auto-draft a separate canonical."""
        resolver = Resolver(self._registry_like_store())
        result = resolver.resolve("Llama-3-8B-Instruct-CommunityFinetune", "model")
        assert result.canonical_id is None
        assert result.strategy == "no_match"

    def test_unknown_instruct_does_not_collapse_to_base(self):
        """Specifically: there is no resolver-side `-instruct` strip. An
        instruct variant we haven't promoted yet does NOT silently fall
        through to the base."""
        store = _store_with_aliases(
            ("Llama-3-8B", "model", "meta/llama-3-8b", None, "confirmed"),
            ("meta/llama-3-8b", "model", "meta/llama-3-8b", None, "confirmed"),
            # NB: no -instruct alias / canonical seeded
        )
        resolver = Resolver(store)
        result = resolver.resolve("Llama-3-8B-Instruct", "model")
        assert result.canonical_id is None
        assert result.strategy == "no_match"


class TestSnakeCaseEquivalence:
    """Snake_case forms of seeded display-form aliases resolve via normalized matcher,
    without requiring the snake_case alias to be listed explicitly."""

    @pytest.mark.parametrize(
        "raw,expected_canonical",
        [
            ("easy_problems", "livecodebench-pro"),
            ("medium_problems", "livecodebench-pro"),
            ("hard_problems", "livecodebench-pro"),
        ],
    )
    def test_lcb_pro_difficulty_tiers_normalize(self, raw, expected_canonical):
        store = _store_with_aliases(
            ("Easy Problems", "benchmark", "livecodebench-pro", "livecodebenchpro", "confirmed"),
            ("Medium Problems", "benchmark", "livecodebench-pro", "livecodebenchpro", "confirmed"),
            ("Hard Problems", "benchmark", "livecodebench-pro", "livecodebenchpro", "confirmed"),
        )
        resolver = Resolver(store)
        result = resolver.resolve(raw, "benchmark", source_config="livecodebenchpro")
        assert result.canonical_id == expected_canonical
        assert result.strategy == "normalized"


class TestResolveStructuredMetricId:
    """Positionless, registry-driven segment resolution for namespaced
    metric_config ids, with catch-all deferral."""

    CATCH_ALL = frozenset({"score"})

    def _resolver(self):
        return Resolver(
            _store_with_aliases(
                ("elo", "metric", "elo", None, "confirmed"),
                ("accuracy", "metric", "accuracy", None, "confirmed"),
                ("exact-match", "metric", "exact-match", None, "confirmed"),
                ("score", "metric", "score", None, "confirmed"),
            )
        )

    def test_metric_in_last_segment(self):
        assert (
            self._resolver().resolve_structured_metric_id(
                "vals_ai.mgsm.mgsm_de.accuracy", catch_all_ids=self.CATCH_ALL
            )
            == "accuracy"
        )

    def test_metric_in_middle_segment(self):
        # lmarena.elo.overall — the metric is NOT last; position must not matter.
        assert (
            self._resolver().resolve_structured_metric_id(
                "lmarena.elo.overall", catch_all_ids=self.CATCH_ALL
            )
            == "elo"
        )

    def test_slash_separator(self):
        assert (
            self._resolver().resolve_structured_metric_id(
                "openeval/bbq/exact-match", catch_all_ids=self.CATCH_ALL
            )
            == "exact-match"
        )

    def test_catch_all_hit_defers(self):
        # llm_stats.gdpval-aa.score — "score" is a raw field name, not a
        # disclosure; the catch-all flag makes this defer to the prose path.
        assert (
            self._resolver().resolve_structured_metric_id(
                "llm_stats.gdpval-aa.score", catch_all_ids=self.CATCH_ALL
            )
            is None
        )

    def test_catch_all_unflagged_wins(self):
        # Without the registry flag, "score" is an ordinary vocabulary hit.
        assert (
            self._resolver().resolve_structured_metric_id("llm_stats.gdpval-aa.score")
            == "score"
        )

    def test_no_segment_resolves(self):
        # artificial_analysis.mmlu_pro — no segment is a metric; falls through.
        assert (
            self._resolver().resolve_structured_metric_id(
                "artificial_analysis.mmlu_pro", catch_all_ids=self.CATCH_ALL
            )
            is None
        )

    def test_conflicting_specific_hits_defer(self):
        assert (
            self._resolver().resolve_structured_metric_id(
                "adapter.elo.accuracy", catch_all_ids=self.CATCH_ALL
            )
            is None
        )

    def test_adapter_namespace_segment_excluded(self):
        # First segment never contributes, even when it matches the vocabulary.
        assert (
            self._resolver().resolve_structured_metric_id(
                "elo.some-benchmark", catch_all_ids=self.CATCH_ALL
            )
            is None
        )

    def test_duplicate_hits_are_one_disclosure(self):
        assert (
            self._resolver().resolve_structured_metric_id(
                "adapter.accuracy.accuracy", catch_all_ids=self.CATCH_ALL
            )
            == "accuracy"
        )

    def test_bare_id_never_routes_through(self):
        assert self._resolver().resolve_structured_metric_id("accuracy") is None

    def test_whitespace_text_never_routes_through(self):
        # Prose containing a dot (a sentence) must not be parsed as an id.
        assert (
            self._resolver().resolve_structured_metric_id(
                "reports performance as an Elo score."
            )
            is None
        )

    def test_none_and_blank_inputs(self):
        r = self._resolver()
        assert r.resolve_structured_metric_id(None) is None
        assert r.resolve_structured_metric_id("  ") is None


class TestResolveStructuredBenchmark:
    """Segment-wise, registry-driven resolution for dotted evaluation_names."""

    def _resolver(self):
        return Resolver(
            _store_with_aliases(
                ("bbq", "benchmark", "bbq", None, "confirmed"),
                ("agieval_lsat_lr", "benchmark", "agieval", None, "confirmed"),
                ("MMLU", "benchmark", "mmlu", None, "confirmed"),
                ("MMLU-Pro", "benchmark", "mmlu-pro", None, "confirmed"),
                ("mmlu_pro", "benchmark", "mmlu-pro", None, "confirmed"),
                ("Mmlu pro math", "benchmark", "mmlu-pro", None, "confirmed"),
                ("math", "benchmark", "math", None, "confirmed"),
                ("bfcl", "benchmark", "bfcl", None, "confirmed"),
                ("live_accuracy", "metric", "accuracy", None, "confirmed"),
                ("mmlu_college_physics", "benchmark", "mmlu", None, "confirmed"),
            )
        )

    def test_doubled_segments_collapse(self):
        match = self._resolver().resolve_structured_benchmark("bbq.bbq.overall")
        assert (match.canonical_id, match.benchmark_raw, match.subset) == ("bbq", "bbq", None)

    def test_doubled_segments_collapse_with_underscores(self):
        match = self._resolver().resolve_structured_benchmark(
            "agieval_lsat_lr.agieval_lsat_lr.overall"
        )
        assert match.canonical_id == "agieval"
        assert match.benchmark_raw == "agieval lsat lr"

    def test_most_specific_segment_wins(self):
        # The family slot must not outrank the benchmark it qualifies.
        match = self._resolver().resolve_structured_benchmark("MMLU.MMLU-Pro.overall")
        assert match.canonical_id == "mmlu-pro"
        assert match.benchmark_raw == "MMLU-Pro"

    def test_subset_stays_with_its_parent(self):
        # `math` is a benchmark of its own, but here it is MMLU-Pro's subject.
        match = self._resolver().resolve_structured_benchmark(
            "vals_ai.mmlu_pro.math", source_config="vals-ai"
        )
        assert match.canonical_id == "mmlu-pro"
        assert match.subset == "math"

    def test_leading_source_namespace_is_not_the_benchmark(self):
        match = self._resolver().resolve_structured_benchmark(
            "vals_ai.mmlu_pro.biology", source_config="vals-ai"
        )
        assert match.canonical_id == "mmlu-pro"
        assert match.benchmark_raw == "mmlu pro biology"
        assert match.subset == "biology"

    def test_trailing_metric_segment_is_dropped(self):
        match = self._resolver().resolve_structured_benchmark("bfcl.live.live_accuracy")
        assert match.canonical_id == "bfcl"
        assert match.subset == "live"

    def test_run_config_tokens_resolve_through_the_strip(self):
        match = self._resolver().resolve_structured_benchmark(
            "mmlu_flan_cot_zeroshot_college_physics."
            "mmlu_flan_cot_zeroshot_college_physics.overall"
        )
        assert match.canonical_id == "mmlu"

    def test_bare_name_never_routes_through(self):
        assert self._resolver().resolve_structured_benchmark("gsm8k") is None

    def test_no_segment_resolves_falls_back(self):
        assert self._resolver().resolve_structured_benchmark("nope.nothing.overall") is None

    def test_whitespace_text_never_routes_through(self):
        assert (
            self._resolver().resolve_structured_benchmark("NaturalQuestions (open-book).")
            is None
        )

    def test_none_and_blank_inputs(self):
        r = self._resolver()
        assert r.resolve_structured_benchmark(None) is None
        assert r.resolve_structured_benchmark("  ") is None
