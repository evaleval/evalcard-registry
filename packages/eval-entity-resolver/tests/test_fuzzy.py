"""Unit tests for the fuzzy resolution strategy primitives + integration tests
for the new ``_ORG_ALIASES`` entries (zhipu/z-ai/moonshot families) and the
``_drop_duplicated_org_prefix`` heuristic.

The existing ``test_resolver.py`` covers fuzzy_match behavior end-to-end.
This file focuses on the pure-function helpers and on the new aliases /
behaviors landed alongside them.
"""
from __future__ import annotations

import pandas as pd
import pytest

from eval_entity_resolver import AliasStore, Resolver
from eval_entity_resolver.strategies.fuzzy import (
    _drop_duplicated_org_prefix,
    _drop_host_prefix,
    _normalize_org,
    _strip_and_capture_platform_suffix,
    fuzzy_match,
)


# ---------------------------------------------------------------------------
# _drop_duplicated_org_prefix unit tests
# ---------------------------------------------------------------------------


class TestDropDuplicatedOrgPrefix:
    def test_slash_dash_form(self):
        assert (
            _drop_duplicated_org_prefix("moonshotai/moonshotai-kimi-k2-instruct")
            == "moonshotai/kimi-k2-instruct"
        )

    def test_slug_form_dash(self):
        # Slug input → slug output. The pipeline already uses `__` as the
        # org/model separator, so we keep that shape; a downstream step
        # can normalize to slash form if needed.
        assert (
            _drop_duplicated_org_prefix("moonshotai__moonshotai-kimi-k2-instruct")
            == "moonshotai__kimi-k2-instruct"
        )

    def test_slash_double_slash_form(self):
        assert (
            _drop_duplicated_org_prefix("moonshotai/moonshotai/kimi-k2-instruct")
            == "moonshotai/kimi-k2-instruct"
        )

    def test_slug_double_underscore_form(self):
        assert (
            _drop_duplicated_org_prefix("moonshotai__moonshotai__kimi-k2-instruct")
            == "moonshotai__kimi-k2-instruct"
        )

    def test_does_not_collapse_distinct_tokens(self):
        # gpt-4 and gpt-4-turbo share a substring but are different model
        # paths, so collapsing would drop information. The hyphen-in-org
        # guard keeps this from firing.
        assert _drop_duplicated_org_prefix("gpt-4/gpt-4-turbo") is None

    def test_real_case_openai_o1(self):
        assert _drop_duplicated_org_prefix("openai/openai-o1") == "openai/o1"

    def test_single_prefix_returns_none(self):
        assert _drop_duplicated_org_prefix("anthropic/claude") is None

    def test_no_separator_returns_none(self):
        assert _drop_duplicated_org_prefix("singleton") is None

    def test_empty_returns_none(self):
        assert _drop_duplicated_org_prefix("") is None

    def test_underscore_separator_in_slash_form(self):
        # `<org>/<org>_<rest>`
        assert (
            _drop_duplicated_org_prefix("moonshotai/moonshotai_kimi-k2")
            == "moonshotai/kimi-k2"
        )

    def test_case_insensitive_token_match(self):
        # Token equality is case-insensitive; original casing is preserved
        # in the returned string.
        assert (
            _drop_duplicated_org_prefix("Moonshotai/MOONSHOTAI-Kimi")
            == "Moonshotai/Kimi"
        )

    def test_substring_overlap_does_not_match(self):
        # `openai` is not a strict prefix of `openaicommunity` followed by
        # a separator, so this must NOT fire.
        assert _drop_duplicated_org_prefix("openai/openaicommunity-x") is None


# ---------------------------------------------------------------------------
# _ORG_ALIASES integration tests (new entries)
# ---------------------------------------------------------------------------


def _store_with_aliases(*rows) -> AliasStore:
    """Mirror of the helper in test_resolver.py — builds an AliasStore from
    (raw_value, entity_type, canonical_id, source_config, status) tuples."""
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


class TestNewOrgAliases:
    @pytest.mark.parametrize(
        "raw_value,canonical",
        [
            ("zhipu/glm-4.5", "zai/glm-4.5"),
            ("zhipu-ai/glm-4.5", "zai/glm-4.5"),
            ("z-ai/glm-4.5", "zai/glm-4.5"),
            ("zai-org/glm-4.5", "zai/glm-4.5"),
        ],
    )
    def test_zhipu_family_to_zai(self, raw_value, canonical):
        store = _store_with_aliases(("zai/glm-4.5", "model", "zai/glm-4.5", None, "confirmed"))
        resolver = Resolver(store)
        result = resolver.resolve(raw_value, "model")
        assert result.canonical_id == canonical
        assert result.strategy == "fuzzy"

    @pytest.mark.parametrize(
        "raw_value,canonical",
        [
            ("moonshot/kimi-k2", "moonshotai/kimi-k2"),
            ("moonshot-ai/kimi-k2", "moonshotai/kimi-k2"),
        ],
    )
    def test_moonshot_family_to_moonshotai(self, raw_value, canonical):
        store = _store_with_aliases(
            ("moonshotai/kimi-k2", "model", "moonshotai/kimi-k2", None, "confirmed")
        )
        resolver = Resolver(store)
        result = resolver.resolve(raw_value, "model")
        assert result.canonical_id == canonical
        assert result.strategy == "fuzzy"


class TestDuplicatedOrgIntegration:
    """End-to-end check that fuzzy_match wires _drop_duplicated_org_prefix
    in the right place — the deduped candidate must go through alias
    lookup AND through the org-alias pass."""

    def test_duplicated_prefix_resolves_via_alias(self):
        store = _store_with_aliases(
            ("moonshotai/kimi-k2-instruct", "model", "moonshotai/kimi-k2-instruct", None, "confirmed")
        )
        resolver = Resolver(store)
        result = resolver.resolve("moonshotai/moonshotai-kimi-k2-instruct", "model")
        assert result.canonical_id == "moonshotai/kimi-k2-instruct"
        assert result.strategy == "fuzzy"

    def test_duplicated_prefix_plus_org_alias(self):
        """Combined: `moonshot/moonshot-kimi-k2` → dedup → `moonshot/kimi-k2`
        → org-alias → `moonshotai/kimi-k2`."""
        store = _store_with_aliases(
            ("moonshotai/kimi-k2", "model", "moonshotai/kimi-k2", None, "confirmed")
        )
        resolver = Resolver(store)
        result = resolver.resolve("moonshot/moonshot-kimi-k2", "model")
        assert result.canonical_id == "moonshotai/kimi-k2"
        assert result.strategy == "fuzzy"


# ---------------------------------------------------------------------------
# _drop_host_prefix unit tests
# ---------------------------------------------------------------------------


class TestDropHostPrefix:
    """Unit coverage for the host/gateway/placeholder prefix stripper.

    The helper recognizes 39 known hosting platforms plus placeholders,
    and accepts both ``host/model`` and ``host.model`` separators. It now
    returns a ``(bare_suffix, platform_id)`` tuple — the captured
    ``platform_id`` is the canonical
    inference_platform id when the host token carries one in the
    single-source seed map, else None. Returns None (not a tuple) when the
    prefix is not in the known set.
    """

    # ------- slash-form positive cases -------

    def test_strips_unknown_slash_prefix(self):
        # `unknown/` is the alphaxiv leaderboard placeholder for missing
        # developer field — drop it and resolve on the bare suffix. The
        # sentinel carries NO platform.
        assert _drop_host_prefix("unknown/openai-o1") == ("openai-o1", None)

    def test_strips_bedrock_slash_prefix(self):
        assert _drop_host_prefix("bedrock/anthropic-claude")[0] == "anthropic-claude"

    def test_strips_amazon_bedrock_slash_prefix(self):
        assert _drop_host_prefix("amazon-bedrock/claude-3")[0] == "claude-3"

    def test_strips_aws_bedrock_slash_prefix(self):
        assert _drop_host_prefix("aws-bedrock/claude-3")[0] == "claude-3"

    def test_strips_azure_slash_prefix(self):
        assert _drop_host_prefix("azure/gpt-4")[0] == "gpt-4"

    def test_strips_azure_openai_slash_prefix(self):
        assert _drop_host_prefix("azure-openai/gpt-4")[0] == "gpt-4"

    def test_strips_azure_cognitive_services_slash_prefix(self):
        assert _drop_host_prefix("azure-cognitive-services/gpt-4")[0] == "gpt-4"

    def test_strips_vertex_slash_prefix(self):
        assert _drop_host_prefix("vertex/gemini-2.0")[0] == "gemini-2.0"

    def test_strips_google_vertex_slash_prefix(self):
        assert _drop_host_prefix("google-vertex/gemini-2.0")[0] == "gemini-2.0"

    def test_strips_vertex_anthropic_slash_prefix(self):
        assert _drop_host_prefix("vertex-anthropic/claude-3")[0] == "claude-3"

    def test_strips_fireworks_slash_prefix(self):
        assert _drop_host_prefix("fireworks/llama-3")[0] == "llama-3"

    def test_strips_fireworks_ai_slash_prefix(self):
        assert _drop_host_prefix("fireworks-ai/llama-3")[0] == "llama-3"

    def test_strips_groq_slash_prefix(self):
        assert _drop_host_prefix("groq/llama-3")[0] == "llama-3"

    def test_strips_together_slash_prefix(self):
        assert _drop_host_prefix("together/llama-3")[0] == "llama-3"

    def test_strips_togetherai_slash_prefix(self):
        assert _drop_host_prefix("togetherai/llama-3")[0] == "llama-3"

    def test_strips_together_ai_slash_prefix(self):
        assert _drop_host_prefix("together-ai/llama-3")[0] == "llama-3"

    def test_strips_openrouter_slash_prefix(self):
        assert _drop_host_prefix("openrouter/anthropic-claude")[0] == "anthropic-claude"

    def test_strips_perplexity_agent_slash_prefix(self):
        assert _drop_host_prefix("perplexity-agent/sonar")[0] == "sonar"

    def test_strips_deepinfra_slash_prefix(self):
        assert _drop_host_prefix("deepinfra/llama-3")[0] == "llama-3"

    def test_strips_anyscale_slash_prefix(self):
        assert _drop_host_prefix("anyscale/llama-3")[0] == "llama-3"

    def test_strips_novita_slash_prefix(self):
        assert _drop_host_prefix("novita/llama-3")[0] == "llama-3"

    def test_strips_novita_ai_slash_prefix(self):
        assert _drop_host_prefix("novita-ai/llama-3")[0] == "llama-3"

    def test_strips_replicate_slash_prefix(self):
        assert _drop_host_prefix("replicate/llama-3")[0] == "llama-3"

    def test_strips_ollama_slash_prefix(self):
        assert _drop_host_prefix("ollama/llama-3")[0] == "llama-3"

    def test_strips_ollama_cloud_slash_prefix(self):
        assert _drop_host_prefix("ollama-cloud/llama-3")[0] == "llama-3"

    def test_strips_github_models_slash_prefix(self):
        assert _drop_host_prefix("github-models/gpt-4")[0] == "gpt-4"

    def test_strips_github_copilot_slash_prefix(self):
        assert _drop_host_prefix("github-copilot/gpt-4")[0] == "gpt-4"

    def test_strips_lambda_slash_prefix(self):
        assert _drop_host_prefix("lambda/llama-3")[0] == "llama-3"

    def test_strips_baseten_slash_prefix(self):
        assert _drop_host_prefix("baseten/llama-3")[0] == "llama-3"

    def test_strips_modal_slash_prefix(self):
        assert _drop_host_prefix("modal/llama-3")[0] == "llama-3"

    def test_strips_runpod_slash_prefix(self):
        assert _drop_host_prefix("runpod/llama-3")[0] == "llama-3"

    def test_strips_cerebras_slash_prefix(self):
        assert _drop_host_prefix("cerebras/llama-3")[0] == "llama-3"

    def test_strips_sap_ai_core_slash_prefix(self):
        assert _drop_host_prefix("sap-ai-core/gpt-4")[0] == "gpt-4"

    def test_strips_cloudflare_ai_gateway_slash_prefix(self):
        assert _drop_host_prefix("cloudflare-ai-gateway/llama-3")[0] == "llama-3"

    def test_strips_aihubmix_slash_prefix(self):
        assert _drop_host_prefix("aihubmix/gpt-4")[0] == "gpt-4"

    def test_strips_kilo_slash_prefix(self):
        assert _drop_host_prefix("kilo/gpt-4")[0] == "gpt-4"

    def test_strips_vercel_slash_prefix(self):
        assert _drop_host_prefix("vercel/gpt-4")[0] == "gpt-4"

    def test_strips_llmgateway_slash_prefix(self):
        assert _drop_host_prefix("llmgateway/gpt-4")[0] == "gpt-4"

    def test_strips_poe_slash_prefix(self):
        assert _drop_host_prefix("poe/gpt-4")[0] == "gpt-4"

    # ------- platform-capture cases: tokens in the single-source map -------

    def test_captures_fireworks_platform(self):
        # `fireworks/` is in the seed map → captures `fireworks-ai`.
        assert _drop_host_prefix("fireworks/llama-3") == ("llama-3", "fireworks-ai")

    def test_captures_together_platform(self):
        assert _drop_host_prefix("together/llama-3") == ("llama-3", "togetherai")

    def test_captures_groq_platform(self):
        assert _drop_host_prefix("groq/llama-3") == ("llama-3", "groq")

    def test_captures_azure_platform(self):
        assert _drop_host_prefix("azure/gpt-4") == ("gpt-4", "azure")

    def test_unmapped_host_token_captures_none(self):
        # `vertex` is stripped for matching but has no `vertex/` token in
        # the single-source seed map yet → platform None.
        assert _drop_host_prefix("vertex/gemini-2.0") == ("gemini-2.0", None)

    # ------- dot-form positive cases -------

    def test_strips_bedrock_dot_prefix(self):
        # Bedrock model IDs use dots: `bedrock.anthropic.claude-3-5`. The
        # helper splits at the first dot only, so the rest can carry its
        # own dotted segments.
        assert (
            _drop_host_prefix("bedrock.anthropic-claude-3-5")[0]
            == "anthropic-claude-3-5"
        )

    def test_strips_vertex_dot_prefix(self):
        assert _drop_host_prefix("vertex.google-gemini-2.0")[0] == "google-gemini-2.0"

    def test_dot_form_only_first_dot_consumed(self):
        # `bedrock.anthropic.claude-3-5` → first dot is the separator;
        # anything after it (including subsequent dots) is the rest.
        assert (
            _drop_host_prefix("bedrock.anthropic.claude-3-5")[0]
            == "anthropic.claude-3-5"
        )

    # ------- case-insensitivity -------

    def test_uppercase_unknown_slash(self):
        # The map is lowercased; the helper lowercases the prefix before
        # the membership check.
        assert _drop_host_prefix("UNKNOWN/openai-o1")[0] == "openai-o1"

    def test_mixed_case_bedrock_slash(self):
        assert _drop_host_prefix("Bedrock/Claude-3")[0] == "Claude-3"

    def test_mixed_case_dot_form(self):
        assert _drop_host_prefix("BEDROCK.claude-3")[0] == "claude-3"

    # ------- negative cases (still return None, not a tuple) -------

    def test_unknown_org_returns_none(self):
        # `random` is not a hosting platform.
        assert _drop_host_prefix("random/openai-o1") is None

    def test_real_org_slash_returns_none(self):
        # `openai` is a real org, not a host — the helper must NOT strip.
        assert _drop_host_prefix("openai/gpt-4") is None

    def test_anthropic_dot_form_returns_none(self):
        # `anthropic` is a real org, not in the host set, even though
        # Bedrock IDs sometimes look like `anthropic.claude-3-5`.
        assert _drop_host_prefix("anthropic.claude-3-5-sonnet") is None

    def test_no_separator_returns_none(self):
        # Bare token with no `/` or `.` — nothing to strip.
        assert _drop_host_prefix("unknown") is None

    def test_empty_string_returns_none(self):
        assert _drop_host_prefix("") is None

    def test_empty_suffix_after_slash_returns_none(self):
        # `unknown/` with empty rest — no model name to fall back to.
        assert _drop_host_prefix("unknown/") is None

    def test_empty_suffix_after_dot_returns_none(self):
        # `bedrock.` with empty rest — no model name to fall back to.
        assert _drop_host_prefix("bedrock.") is None

    def test_slash_takes_precedence_over_dot(self):
        # When both separators exist, slash is checked first. The prefix
        # before the slash (`groq` here) is what's tested against the set.
        assert (
            _drop_host_prefix("groq/anthropic.claude-3-5")[0]
            == "anthropic.claude-3-5"
        )

    def test_substring_of_host_name_does_not_match(self):
        # `bedrocky` is not the same token as `bedrock`. The helper
        # compares the full prefix, not a substring.
        assert _drop_host_prefix("bedrocky/claude-3") is None


# ---------------------------------------------------------------------------
# _normalize_org direct unit tests
# ---------------------------------------------------------------------------


class TestNormalizeOrg:
    """Direct unit coverage of `_normalize_org`. The integration tests in
    `TestNewOrgAliases` exercise this through `Resolver.resolve`; these tests
    pin the helper's contract independent of the resolver wiring."""

    def test_deepseek_ai_to_deepseek(self):
        assert (
            _normalize_org("deepseek-ai/deepseek-v3")
            == "deepseek/deepseek-v3"
        )

    def test_meta_llama_to_meta(self):
        assert (
            _normalize_org("meta-llama/llama-3-8b")
            == "meta/llama-3-8b"
        )

    def test_zhipu_ai_to_zai(self):
        assert _normalize_org("zhipu-ai/glm-4.5") == "zai/glm-4.5"

    def test_moonshot_ai_to_moonshotai(self):
        assert (
            _normalize_org("moonshot-ai/kimi-k2")
            == "moonshotai/kimi-k2"
        )

    def test_unknown_org_returns_none(self):
        # `openai` is not in `_ORG_ALIASES` (already canonical).
        assert _normalize_org("openai/gpt-4") is None

    def test_no_slash_returns_none(self):
        # Helper requires a slash — bare tokens are out of scope.
        assert _normalize_org("deepseek-ai") is None

    def test_empty_string_returns_none(self):
        assert _normalize_org("") is None

    def test_case_insensitive_org_match(self):
        # Map keys are lowercased; the helper lowercases the org token
        # before lookup. Casing of the rest is preserved on output.
        assert (
            _normalize_org("DeepSeek-AI/DeepSeek-V3")
            == "deepseek/DeepSeek-V3"
        )


# ---------------------------------------------------------------------------
# Integration: suffix-strip after duplicated-org-prefix collapse
# ---------------------------------------------------------------------------


class TestDuplicatedOrgPlusSuffixIntegration:
    """The dedup pass runs AFTER suffix stripping in `fuzzy_match`. So a
    raw value with BOTH a duplicated org prefix AND a known suffix can
    only resolve if the suffix-stripped form ALSO carries the duplicated
    prefix (so dedup fires on it) — or if dedup runs once and the result
    happens to match an alias as-is.

    These tests pin the current behavior. If a future refactor re-runs
    `_strip_suffix` on derived candidates produced by the dedup pass,
    `test_double_prefix_plus_fc_suffix_resolves` will start passing.
    """

    def test_dedup_alone_produces_expected_candidate(self):
        # Pure-helper sanity check: dedup on the raw value yields a
        # string that still carries the `-fc` suffix.
        assert (
            _drop_duplicated_org_prefix("moonshotai/moonshotai-kimi-k2-fc")
            == "moonshotai/kimi-k2-fc"
        )

    def test_double_prefix_plus_fc_suffix_resolves(self):
        """End-to-end: `moonshotai/moonshotai-kimi-k2-fc` should resolve
        to `moonshotai/kimi-k2` (suffix `-fc` stripped, double prefix
        deduped). Today this works because suffix-stripping fires first
        on the original, producing `moonshotai/moonshotai-kimi-k2`, which
        the dedup pass then collapses to `moonshotai/kimi-k2`."""
        store = _store_with_aliases(
            ("moonshotai/kimi-k2", "model", "moonshotai/kimi-k2", None, "confirmed")
        )
        resolver = Resolver(store)
        result = resolver.resolve("moonshotai/moonshotai-kimi-k2-fc", "model")
        assert result.canonical_id == "moonshotai/kimi-k2"
        assert result.strategy == "fuzzy"

    def test_suffix_first_then_dedup_path(self):
        """The order of operations in `fuzzy_match` is: suffix-strip first,
        then host-strip, then dedup, then org-alias. This test exercises
        the path where suffix-strip fires on the raw value (yielding
        `<org>/<org>-...`), and dedup then runs on the suffix-stripped form.
        """
        store = _store_with_aliases(
            ("openai/o1", "model", "openai/o1", None, "confirmed")
        )
        resolver = Resolver(store)
        # `-prompt` is a known suffix → `openai/openai-o1` → dedup → `openai/o1`.
        result = resolver.resolve("openai/openai-o1-prompt", "model")
        assert result.canonical_id == "openai/o1"
        assert result.strategy == "fuzzy"

    def test_dedup_then_suffix_path_not_currently_covered(self):
        """KNOWN GAP: when the deduped-then-org-aliased candidate still
        carries a known suffix that wasn't on the original, the resolver
        does NOT re-apply `_strip_suffix` to derived candidates.

        Construct a case where this matters: a raw value where the dedup
        pass produces a string carrying `-fc`, but the original raw value
        does NOT end in `-fc` (so suffix-strip never fires on it).

        This is not currently constructible from the existing helpers
        because dedup only TRIMS leading tokens — it can't introduce new
        trailing characters. So in practice, the order
        (suffix → host → dedup → org-alias → lookup) is sufficient for
        all observed corpus cases. This test pins that observation: a
        synthetic suffix on the original IS picked up by the
        suffix-first pass, so the integration works.
        """
        store = _store_with_aliases(
            ("moonshotai/kimi-k2", "model", "moonshotai/kimi-k2", None, "confirmed")
        )
        resolver = Resolver(store)
        # Suffix `-fc` is on the original; suffix-strip fires first
        # producing `moonshotai/moonshotai-kimi-k2`, then dedup collapses.
        result = resolver.resolve("moonshotai/moonshotai-kimi-k2-fc", "model")
        assert result.canonical_id == "moonshotai/kimi-k2"


class TestQuantizationAndHfStripping:
    """Quantization / HF-upload suffix stripping. See PRECISION-LOSS POLICY in
    fuzzy.py module docstring — these collapses sacrifice precision intentionally."""

    def test_hf_suffix_strips(self):
        store = _store_with_aliases(
            ("meta/llama-2-7b", "model", "meta/llama-2-7b", None, "confirmed")
        )
        resolver = Resolver(store)
        result = resolver.resolve("meta/llama-2-7b-hf", "model")
        assert result.canonical_id == "meta/llama-2-7b"
        assert result.strategy == "fuzzy"

    def test_fp8_suffix_strips(self):
        store = _store_with_aliases(
            ("deepseek/deepseek-v4-flash", "model", "deepseek/deepseek-v4-flash", None, "confirmed")
        )
        resolver = Resolver(store)
        result = resolver.resolve("deepseek/deepseek-v4-flash-fp8", "model")
        assert result.canonical_id == "deepseek/deepseek-v4-flash"

    def test_int4_awq_strips_as_one_unit(self):
        # Compound suffix: -int4-awq is in the strip list before -int4 / -awq alone,
        # so it should strip in one step (not stripping -awq first then leaving
        # -int4 behind to fail).
        store = _store_with_aliases(
            ("meta/llama-3-70b", "model", "meta/llama-3-70b", None, "confirmed")
        )
        resolver = Resolver(store)
        result = resolver.resolve("meta/llama-3-70b-int4-awq", "model")
        assert result.canonical_id == "meta/llama-3-70b"

    def test_gguf_suffix_strips(self):
        store = _store_with_aliases(
            ("alibaba/qwen2-72b", "model", "alibaba/qwen2-72b", None, "confirmed")
        )
        resolver = Resolver(store)
        result = resolver.resolve("alibaba/qwen2-72b-gguf", "model")
        assert result.canonical_id == "alibaba/qwen2-72b"


class TestLetterDigitDashCollapse:
    """Letter-digit boundary dashes: ``qwen-2-72b`` should match ``qwen2-72b``."""

    def test_qwen_dash_form_resolves(self):
        store = _store_with_aliases(
            ("alibaba/qwen2-72b", "model", "alibaba/qwen2-72b", None, "confirmed")
        )
        resolver = Resolver(store)
        result = resolver.resolve("alibaba/qwen-2-72b", "model")
        assert result.canonical_id == "alibaba/qwen2-72b"
        assert result.strategy == "fuzzy"

    def test_dash_form_with_instruct_suffix(self):
        # alibaba/qwen-2-72b-instruct (real route_id form) must reach
        # alibaba/qwen2-72b after both the digit-dash collapse AND
        # whatever path handles -instruct.
        store = _store_with_aliases(
            ("alibaba/qwen2-72b-instruct", "model", "alibaba/qwen2-72b", None, "confirmed")
        )
        resolver = Resolver(store)
        result = resolver.resolve("alibaba/qwen-2-72b-instruct", "model")
        assert result.canonical_id == "alibaba/qwen2-72b"

    def test_no_change_when_no_letter_digit_boundary(self):
        # qwen2-72b already has no boundary dash → collapse no-op.
        store = _store_with_aliases(
            ("alibaba/qwen2-72b", "model", "alibaba/qwen2-72b", None, "confirmed")
        )
        resolver = Resolver(store)
        result = resolver.resolve("alibaba/qwen2-72b", "model")
        assert result.canonical_id == "alibaba/qwen2-72b"
        # exact path, not fuzzy
        assert result.strategy == "exact"

    def test_distinct_models_stay_distinct(self):
        # gpt-4 and gpt-4-mini are distinct models; the collapse should
        # not merge them. Both have letter-digit at "t-4" boundary but
        # the -mini suffix preserves identity.
        store = _store_with_aliases(
            ("openai/gpt4", "model", "openai/gpt4", None, "confirmed"),
            ("openai/gpt4-mini", "model", "openai/gpt4-mini", None, "confirmed"),
        )
        resolver = Resolver(store)
        # gpt-4 → gpt4 (matches gpt4 canonical)
        assert resolver.resolve("openai/gpt-4", "model").canonical_id == "openai/gpt4"
        # gpt-4-mini → gpt4-mini (matches gpt4-mini canonical, NOT gpt4)
        assert resolver.resolve("openai/gpt-4-mini", "model").canonical_id == "openai/gpt4-mini"


# ---------------------------------------------------------------------------
# platform-capture suffix + fuzzy_match 3-tuple shape + threading
# ---------------------------------------------------------------------------


class TestStripAndCapturePlatformSuffix:
    """`_strip_and_capture_platform_suffix` — moved the 3 host suffixes out of
    the generic `_STRIP_SUFFIXES` so they're CAPTURED, not silently dropped."""

    def test_together_suffix_captures_platform(self):
        assert _strip_and_capture_platform_suffix("llama-3-8b-together") == (
            "llama-3-8b",
            "togetherai",
        )

    def test_bedrock_suffix_captures_platform(self):
        assert _strip_and_capture_platform_suffix("claude-3-bedrock") == (
            "claude-3",
            "amazon-bedrock",
        )

    def test_openrouter_suffix_captures_platform(self):
        assert _strip_and_capture_platform_suffix("gpt-4o-openrouter") == (
            "gpt-4o",
            "openrouter",
        )

    def test_no_platform_suffix_returns_none(self):
        # `-fp8` is a quant suffix handled by the generic strip, NOT a
        # platform suffix → this helper returns None.
        assert _strip_and_capture_platform_suffix("llama-3-8b-fp8") is None

    def test_case_insensitive_suffix(self):
        stem, platform = _strip_and_capture_platform_suffix("Claude-3-BEDROCK")
        assert stem == "Claude-3"
        assert platform == "amazon-bedrock"


class TestFuzzyMatchPlatformCapture:
    """`fuzzy_match` now returns a 3-tuple ``(canonical_id, confidence,
    inference_platform)``. Capture happens ONCE; suffix beats prefix; the
    captured platform is a SIDE VALUE — it never changes the resolved id."""

    def test_returns_three_tuple_on_match(self):
        # Bare `gpt-4o` alias so the suffix-stripped stem (`gpt-4o`) matches.
        store = _store_with_aliases(
            ("gpt-4o", "model", "openai/gpt-4o", None, "confirmed")
        )
        result = fuzzy_match("gpt-4o-openrouter", "model", 0.85, store)
        assert len(result) == 3
        canonical_id, confidence, platform = result
        assert canonical_id == "openai/gpt-4o"
        assert platform == "openrouter"

    def test_returns_three_tuple_on_no_match(self):
        store = _store_with_aliases()
        assert fuzzy_match("nonexistent-model-xyz", "model", 0.85, store) == (
            None,
            0.0,
            None,
        )

    def test_non_model_entity_returns_three_tuple(self):
        store = _store_with_aliases()
        assert fuzzy_match("anything", "benchmark", 0.85, store) == (None, 0.0, None)

    def test_prefix_host_captured_when_no_suffix(self):
        store = _store_with_aliases(
            ("meta/llama-3-8b", "model", "meta/llama-3-8b", None, "confirmed")
        )
        canonical_id, _, platform = fuzzy_match(
            "together/meta/llama-3-8b", "model", 0.85, store
        )
        assert canonical_id == "meta/llama-3-8b"
        assert platform == "togetherai"

    def test_suffix_beats_prefix_capture(self):
        # Both a `fireworks/` prefix AND a `-bedrock` suffix present: the
        # SUFFIX wins (explicit model-name host token is stronger).
        store = _store_with_aliases(
            ("meta/llama-3-8b", "model", "meta/llama-3-8b", None, "confirmed")
        )
        canonical_id, _, platform = fuzzy_match(
            "fireworks/meta/llama-3-8b-bedrock", "model", 0.85, store
        )
        assert canonical_id == "meta/llama-3-8b"
        assert platform == "amazon-bedrock"

    def test_capture_does_not_change_canonical(self):
        # The same model with and without a host suffix must resolve to the
        # SAME canonical_id — capture is a pure side-value. Use the full
        # resolver chain so the plain id hits the exact-match path (fuzzy
        # alone never sees the raw value as a candidate).
        store = _store_with_aliases(
            ("gpt-4o", "model", "openai/gpt-4o", None, "confirmed")
        )
        resolver = Resolver(store)
        plain = resolver.resolve("gpt-4o", "model").canonical_id
        hosted = resolver.resolve("gpt-4o-openrouter", "model").canonical_id
        assert plain == hosted == "openai/gpt-4o"

    def test_no_host_token_captures_none(self):
        store = _store_with_aliases(
            ("gpt-4o", "model", "openai/gpt-4o", None, "confirmed")
        )
        _, _, platform = fuzzy_match("gpt-4o-fc", "model", 0.85, store)
        assert platform is None


class TestResolverThreadsPlatform:
    """`Resolver.resolve` threads the fuzzy-captured platform onto the
    ResolutionResult (the explicit-host-token-in-raw fact that WINS)."""

    def test_resolve_sets_inference_platform_on_host_suffix(self):
        store = _store_with_aliases(
            ("gpt-4o", "model", "openai/gpt-4o", None, "confirmed")
        )
        result = Resolver(store).resolve("gpt-4o-openrouter", "model")
        assert result.canonical_id == "openai/gpt-4o"
        assert result.inference_platform == "openrouter"

    def test_resolve_leaves_platform_none_on_plain_id(self):
        store = _store_with_aliases(
            ("openai/gpt-4o", "model", "openai/gpt-4o", None, "confirmed")
        )
        # Exact-match path: no fuzzy capture, platform stays None.
        result = Resolver(store).resolve("openai/gpt-4o", "model")
        assert result.inference_platform is None


class TestBenchmarkRunConfigStrip:
    """Benchmarks get a run-configuration-only strip vocabulary. Eval
    methodology is not benchmark identity, but answer format and metric
    tokens are held back deliberately."""

    def _store(self):
        return _store_with_aliases(
            ("mmlu", "benchmark", "mmlu", None, "confirmed"),
            ("mmlu_college_physics", "benchmark", "mmlu", None, "confirmed"),
            ("bbh_snarks", "benchmark", "bbh", None, "confirmed"),
            ("global-mmlu", "benchmark", "global-mmlu", None, "confirmed"),
            ("truthfulqa", "benchmark", "truthfulqa", None, "confirmed"),
            ("acp_bench", "benchmark", "acpbench", None, "confirmed"),
        )

    def test_strips_cot_zeroshot_infix(self):
        result = Resolver(self._store()).resolve(
            "mmlu flan cot zeroshot college physics", "benchmark"
        )
        assert result.canonical_id == "mmlu"
        assert result.strategy == "fuzzy"

    def test_strips_cot_fewshot_infix(self):
        assert (
            Resolver(self._store()).resolve("bbh_cot_fewshot_snarks", "benchmark").canonical_id
            == "bbh"
        )

    def test_strips_gen_and_shot_suffixes(self):
        assert (
            Resolver(self._store()).resolve("global_mmlu_gen_0shot", "benchmark").canonical_id
            == "global-mmlu"
        )

    def test_holds_back_metric_tokens(self):
        # `_mc2` is metric identity: it must not strip onto the English parent.
        assert (
            Resolver(self._store()).resolve("truthfulqa_de_mc2", "benchmark").canonical_id
            is None
        )

    def test_holds_back_answer_format_tokens(self):
        # `_bool` / `_mcq` are subsets the registry models separately.
        assert (
            Resolver(self._store()).resolve("acp_bench_bool", "benchmark").canonical_id is None
        )

    def test_no_run_config_token_is_a_no_op(self):
        assert (
            Resolver(self._store()).resolve("some_unknown_benchmark", "benchmark").canonical_id
            is None
        )
