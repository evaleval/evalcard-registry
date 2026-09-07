"""
Fuzzy matching strategy.

Instead of generic string similarity (which falsely merges distinct versions,
sizes, and variants), this uses two targeted approaches:

1. **Stem matching** — strip known non-semantic suffixes (evaluation mode,
   hosting provider, effort level) and check if the stem has an exact or
   normalized match.  This catches real duplicates like
   ``model-name-fc`` → ``model-name`` without collapsing ``gpt-5-mini`` into
   ``gpt-5``.

2. **Org normalization** — handle cases where the org prefix differs between
   configs (``deepseek-ai/model`` vs ``deepseek/model``).

If neither approach produces a match the strategy returns None so the resolver
can fall through to auto-draft.

PRECISION-LOSS POLICY for stem stripping
=========================================
Some suffixes are stripped at the cost of conflating variants the registry
considers "the same model" but a precision-sensitive consumer might not.
The current strip list collapses:

- ``-hf``: HuggingFace re-uploaded copy of an official model release. The
  weights are the same; the upload path differs.
- ``-fp8`` / ``-fp16`` / ``-fp4`` / ``-bf16`` / ``-int4`` / ``-int8`` /
  ``-q4`` / ``-q8`` / ``-quant`` / ``-gguf`` / ``-awq`` / ``-gptq``:
  quantization variants. The architecture is the same; numerical precision
  differs. **Benchmark scores can differ measurably between quantization
  levels** — e.g., a 70B model at FP16 may outperform the same model at
  INT4. We collapse them anyway so the registry treats them as one canonical
  identity, but this means a row labeled "Llama-3.1-70B" in the catalog may
  represent any quantization that was reported.

Consumers needing per-quantization precision should NOT use the canonical_id
alone — they need the original `model_route_id` (which preserves the suffix
verbatim) and the `generation_args` payload.

This collapse is intentional: it's far more useful to match a quantized
inference run to the model family it represents than to leave it unresolved.
But it's a deliberate precision sacrifice and downstream code that compares
scores must respect that.
"""
from __future__ import annotations

import re
from typing import Optional

from eval_entity_resolver.normalization import normalize
from eval_entity_resolver.strategies._platform_map import get_host_token_platform


# Suffixes stripped only from the *end* of the raw value.  Order matters:
# longer suffixes first to avoid partial stripping.
#
# Many of these patterns were identified from raw_model_ids in the
# evaleval/card_backend dataset, where a single canonical model family
# (e.g. ``anthropic/claude-opus-4-5``) has variants like
# ``claude-opus-4-5-20251101-thinking-16k``, ``-fc``, ``-prompt``, etc.
_STRIP_SUFFIXES = [
    # Evaluation-mode suffixes (BFCL, etc.)
    "-fc",
    "-prompt",
    # NB: hosting-provider suffixes (`-together`, `-bedrock`, `-openrouter`)
    # are NOT stripped here — they live in `_SUFFIX_PLATFORM_MAP` below so the
    # platform is CAPTURED as an `inference_platform` side-value rather than
    # discarded. The same stem is produced and matched either way.
    # Reasoning-effort suffixes. Longer compound forms first. `-max` IS an
    # effort tier in the wild (arc-agi's low/high/max ladder); genuine "-max"
    # PRODUCTS (qwen3-max, qwen3.8-max) are safe because exact/normalized
    # matching wins before fuzzy ever strips — only an uncatalogued new -max
    # product could transiently fold onto its base, and that beats a broken
    # no_match entity for the same gap window.
    "-xhigh-effort",
    "-high-effort",
    "-medium-effort",
    "-low-effort",
    "-minimal-effort",
    "-non-reasoning",
    "-xhigh",
    "-high",
    "-medium",
    "-low",
    "-minimal",
    "-max",
    # Thinking-style suffixes
    "-highthinking",
    "-nothinking",
    "-nothink",
    "-thinking-none",
    # HuggingFace re-upload variants — same weights as official release.
    "-hf",
    # Quantization variants — see PRECISION-LOSS POLICY in module docstring.
    # Order: longer first so e.g. ``-int8`` doesn't pre-empt ``-int8-awq``.
    "-int4-awq",
    "-int8-awq",
    "-int4-gptq",
    "-int8-gptq",
    "-fp4",
    "-fp8",
    "-fp16",
    "-bf16",
    "-int4",
    "-int8",
    "-q4",
    "-q8",
    "-awq",
    "-gptq",
    "-gguf",
    "-quant",
]

# Regex-based suffix patterns applied after the literal suffixes.
# These capture variants with numeric parameters (thinking budgets, dates).
# Each pattern must anchor with $ and match only the tail of the string.
_STRIP_SUFFIX_PATTERNS: list[re.Pattern[str]] = [
    # Thinking-budget suffix: "-thinking-8k", "-thinking-16k", "-thinking-64k"
    # Strips the WHOLE thinking-budget tail to the bare base, used as a
    # fallback when the thinking-mode canonical isn't aliased. Paired with
    # `_THINKING_BUDGET_PRESERVE_RE` below, which produces a "preserve
    # thinking" candidate that gets tried FIRST so promoted mode-variant
    # canonicals (e.g. `claude-haiku-4-5-20251001-thinking`) win when they
    # exist; only when they don't does this strip's drop-thinking behavior
    # take over.
    re.compile(r"-thinking-\d+k$", re.IGNORECASE),
    # NB: trailing 8-digit date suffix (`-20251101`) is NOT stripped here.
    # Stripping a packed YYYYMMDD ALWAYS produces the bare-family form,
    # which silently aliases dated snapshots into their family pointer
    # and loses the snapshot's `release_date`. The auto-create +
    # hub-stats path produces a properly-linked snapshot canonical
    # instead. See `infer_family_parent_edge` in
    # services/hub_stats.py for the family-version edge inference.
    # When a snapshot canonical is already aliased (exact / normalized
    # match wins before fuzzy), the resolver returns it directly.
]

# Strip just the `-Nk` budget tail, leaving `-thinking` intact. Used by
# fuzzy_match to produce a "preserve thinking" candidate alongside the
# fallback `prefix` (drop thinking). Anchored to require `-thinking`
# precedes the budget so non-thinking numeric tails are unaffected.
_THINKING_BUDGET_PRESERVE_RE = re.compile(r"(.*-thinking)-\d+k$", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Benchmark run-configuration vocabulary
# ---------------------------------------------------------------------------
# Harness task names weld the run configuration into the benchmark name
# (`mmlu_flan_cot_zeroshot_college_physics`, `bbh_cot_fewshot_snarks`,
# `global_mmlu_gen_0shot`). Eval methodology is not benchmark identity
# (specs/entity-modeling.md (c)), so these tokens are dropped anywhere in
# the name and what remains is matched against the registry.
#
# Deliberately NOT here: `mc1`/`mc2` (metric identity) and `bool`/`mcq`
# (answer format, which the registry models as distinct subsets).
_BENCHMARK_RUN_CONFIG_TOKENS = frozenset({
    "cot", "zeroshot", "fewshot", "0shot", "flan", "gen",
})
_BENCHMARK_SHOT_TOKEN_RE = re.compile(r"^\d+shot$")


def _strip_run_config_tokens(value: str) -> str | None:
    """Drop run-configuration tokens from a benchmark name. Returns the
    remaining tokens space-joined (the normalized tier collapses separators,
    so the join character is immaterial), or None when nothing was dropped
    or nothing would survive."""
    tokens = re.split(r"[\s_\-]+", value.strip())
    kept = [
        t for t in tokens
        if t.lower() not in _BENCHMARK_RUN_CONFIG_TOKENS
        and not _BENCHMARK_SHOT_TOKEN_RE.match(t.lower())
    ]
    if not kept or len(kept) == len(tokens):
        return None
    return " ".join(kept)


def _benchmark_stem_match(
    raw_value: str, alias_store, source_config: Optional[str]
) -> tuple[Optional[str], float, None]:
    """Benchmark counterpart to the model stem match: one candidate, built
    by removing run-configuration tokens. Runs after exact and normalized,
    so a canonical that genuinely carries such a token in its name (an
    `-x-shot` benchmark) still wins on its own spelling."""
    candidate = _strip_run_config_tokens(raw_value)
    if candidate is None:
        return None, 0.0, None
    canonical_id = (
        alias_store.lookup(candidate, "benchmark", source_config)
        or alias_store.get_normalized_lookup("benchmark", source_config).get(
            normalize(candidate)
        )
    )
    if canonical_id is None:
        return None, 0.0, None
    return canonical_id, _STEM_CONFIDENCE, None

# Known org aliases: {variant_prefix: canonical_prefix}
# Convention: simplify HF org names (e.g. "deepseek-ai" → "deepseek") to the
# shorter form used as canonical in this registry.
#
# Zhipu/Z.ai cluster: the GLM-family canonical org is `zai` (short form used
# in this registry for canonical_ids like `zai/glm-4.5`). HF and various
# leaderboards spell it as `zhipu`, `zhipu-ai`, `z-ai`, or `zai-org` — all
# refer to the same Beijing AI startup behind GLM.
#
# Moonshot AI cluster: canonical org is `moonshotai` (matches HF
# `moonshotai/Kimi-*` namespace); aliases cover `moonshot` and `moonshot-ai`
# spellings seen in the corpus.
#
# `alibaba` → `qwen` was considered but skipped: the corpus has 1
# non-Qwen entry (`alibaba__mineru2-pipeline`) which would be wrongly
# rewritten. Qwen models under `alibaba/` are handled via explicit
# overrides instead.
# _ORG_ALIASES now lives in eval_entity_resolver.fold (single owner, shared
# with the seed generators). Imported below.
from eval_entity_resolver.fold import _ORG_ALIASES, _norm_org_key, dev_org_of_prefix  # noqa: E402,F401


def _canon_org(prefix: str, org_map: dict) -> str:
    """The canonical developer org for an id-prefix: curated id / alias (by
    lowercase, then separator-stripped key), else the prefix itself."""
    return org_map.get(prefix.lower()) or org_map.get(_norm_org_key(prefix)) or prefix


def _orgs_agree(
    raw_value: str,
    matched_id: str,
    org_map: Optional[dict] = None,
    known_orgs: Optional[frozenset] = None,
) -> bool:
    """True unless `raw_value` and `matched_id` carry org prefixes that fold to
    DIFFERENT developers (after the curated remap + case/separator fold).

    The org-agreement contract every other dedup surface (fold.decide_fold,
    the generators' _hf_defer_target / by_org_name buckets) already enforces —
    applied here so the resolver's FUZZY stem match never merges across genuinely
    different developers (e.g. `DevQuasar/...` must not match `DevQuasar-3/...`).
    Same-developer alternate namespaces (meta-llama vs facebook, THUDM vs zai-org)
    fold equal and still match; org-less / single-token candidates have no org to
    disagree on and are unaffected.

    `org_map` is the full HF-namespace -> developer-org map the resolver builds
    from BOTH stores (canonical_orgs id/hf_org + the org ALIAS rows in the
    aliases table incl. the orgs.yaml tier `AI2`->`allenai`, `ai21labs`->`ai21`)
    unioned with `_ORG_ALIASES`. When None, falls back to the hardcoded
    `_ORG_ALIASES`.

    `known_orgs` is the set of canonical_orgs ids. Two prefixes that resolve to
    DIFFERENT registered orgs are NEVER separator-strip-merged — this honours the
    distinct-uploader contract (seed/orgs_distinct_allowlist.yaml) directly from
    the data: a genuinely-distinct uploader survives as its own org row, so
    `Enno-Ai` vs `EnnoAi` (both real rows) stay distinct. The case/separator fold
    only fires for the UNREGISTERED tail (e.g. an `aleph-alpha` spelling whose only
    registered twin is `AlephAlpha`)."""
    if "/" not in raw_value or "/" not in matched_id:
        return True
    m = org_map if org_map is not None else _ORG_ALIASES
    ca = _canon_org(raw_value.split("/", 1)[0], m)
    cm = _canon_org(matched_id.split("/", 1)[0], m)
    if ca == cm:
        return True
    # Two DISTINCT registered developers must never be strip-merged.
    if known_orgs is not None and ca in known_orgs and cm in known_orgs:
        return False
    # Unregistered tail: fold case + separator (aleph-alpha == AlephAlpha).
    return _norm_org_key(ca) == _norm_org_key(cm)

# Host / gateway / placeholder prefixes that should be DROPPED entirely
# (not rewritten to a canonical org). These are not model authors —
# they're hosting platforms, gateways, or placeholders for missing
# developer fields. When raw_value uses one of these as the org prefix,
# the resolver tries the bare suffix in addition to the full string.
#
# Identified from corpus surveys: alphaxiv leaderboard uses `unknown/`
# when developer field is absent; Bedrock/Vertex/Azure/Fireworks/etc.
# are inference platforms re-hosting other companies' models.
# The KEYS here are the matching contract — the host-org spellings that, when
# they appear as a developer prefix, are dropped so the bare model body is
# tried for a match. This set is intentionally BROADER than the host tokens
# carried in the single-source `inference_platforms` seed: stripping a host
# org for matching is a resolution heuristic, independent of whether we can
# attribute a canonical platform id to it.
#
# The VALUE for each key is the captured `inference_platform` id, looked up
# from the SINGLE SOURCE (`_platform_map.get_host_token_platform`, which reads
# `seed/inference_platforms.yaml`) by the token's prefix spelling (`<org>/`).
# Tokens absent from the seed map (e.g. `vertex`, `deepinfra`) capture `None`
# — still stripped for matching, just with no platform attribution until the
# seed grows. Building the values programmatically (never a hand-copied
# literal) keeps this in lock-step with the inference_platforms seed.
_HOST_PREFIX_TOKENS: tuple[str, ...] = (
    "unknown",
    "bedrock", "amazon-bedrock", "aws-bedrock",
    "azure", "azure-openai", "azure-cognitive-services",
    "vertex", "google-vertex", "vertex-anthropic",
    "fireworks", "fireworks-ai",
    "groq",
    "together", "togetherai", "together-ai",
    "openrouter",
    "perplexity-agent",
    "deepinfra", "anyscale", "novita", "novita-ai", "replicate",
    "ollama", "ollama-cloud",
    "github-models", "github-copilot",
    "lambda", "baseten", "modal", "runpod", "cerebras",
    "sap-ai-core", "cloudflare-ai-gateway", "aihubmix",
    "kilo", "vercel", "llmgateway", "poe",
)

_HOST_PREFIXES_TO_STRIP: dict[str, Optional[str]] = {
    # `unknown` is the missing-developer sentinel → never a real platform.
    token: (None if token == "unknown" else get_host_token_platform(f"{token}/"))
    for token in _HOST_PREFIX_TOKENS
}

# Suffixes that signal an inference platform (NOT model semantics). Captured
# (mapped to a platform_id) rather than stripped-and-discarded, so the platform
# is preserved. Maps the literal suffix → platform_id from the single host map.
_SUFFIX_PLATFORM_MAP: dict[str, Optional[str]] = {
    suffix: get_host_token_platform(suffix)
    for suffix in ("-together", "-bedrock", "-openrouter")
}


def _drop_duplicated_org_prefix(value: str) -> str | None:
    """Detect and collapse a repeated-org-prefix typo.

    Recognized shapes (token equality is case-insensitive, but the
    returned string preserves the original casing of `value` so the
    downstream lookups can still match exact aliases):

      - ``<org>/<org>-<rest>``           → ``<org>/<rest>``
      - ``<org>/<org>_<rest>``           → ``<org>/<rest>``
      - ``<org>/<org>/<rest>``           → ``<org>/<rest>`` (literal double slash)
      - ``<org>__<org>-<rest>``          → ``<org>__<rest>`` (slug form;
        the pipeline rewrites ``/`` → ``__`` for route_ids and the resolver
        may receive either)
      - ``<org>__<org>__<rest>``         → ``<org>__<rest>`` (slug form
        of the literal double-slash variant)

    Returns ``None`` when the prefix is not duplicated, or when the
    repeated-prefix slug shape is followed by something that doesn't
    cleanly separate (e.g. ``gpt-4/gpt-4-turbo`` — the second ``gpt-4``
    is the START of the model name, not a duplicated prefix).

    The match requires exact token equality of the two leading tokens.
    A substring overlap (``gpt-4`` ⊂ ``gpt-4-turbo``) is intentionally
    NOT enough — that's a real two-segment HF path, not a typo.

    To disambiguate the org-typo case (``openai/openai-o1``) from the
    model-family-prefix case (``gpt-4/gpt-4-turbo``): the heuristic
    only fires when the leading org token has no internal hyphen.
    Real org names (``openai``, ``moonshotai``, ``anthropic``) are
    single tokens; model-family prefixes (``gpt-4``, ``llama-3``,
    ``claude-opus-4-5``) contain hyphens. This is imperfect — a
    hyphenated org like ``mistral-ai`` would slip through — but
    those are already captured upstream by the org-alias pass.
    """
    if not value:
        return None

    # Slash forms first (canonical HF path style).
    if "/" in value:
        first_slash = value.index("/")
        org = value[:first_slash]
        rest = value[first_slash + 1:]
        if not org or not rest:
            return None
        # Skip when the leading token contains a hyphen — likely a
        # model-family prefix (e.g. `gpt-4/gpt-4-turbo`), not a
        # duplicated-org typo. Hyphenated orgs like `mistral-ai` are
        # canonicalized via the org-alias pass first.
        if "-" in org:
            return None
        org_lower = org.lower()
        # `<org>/<org>/<rest>` literal double slash
        if "/" in rest:
            second, after = rest.split("/", 1)
            if second.lower() == org_lower and after:
                return f"{org}/{after}"
        # `<org>/<org>-<rest>` and `<org>/<org>_<rest>`
        for sep in ("-", "_"):
            prefix = org_lower + sep
            if rest.lower().startswith(prefix) and len(rest) > len(prefix):
                return f"{org}/{rest[len(prefix):]}"

    # Slug forms (route_id style with `__`).
    if "__" in value:
        first = value.index("__")
        org = value[:first]
        rest = value[first + 2:]
        if not org or not rest:
            return None
        # Same hyphen-in-org guard (see slash branch above).
        if "-" in org:
            return None
        org_lower = org.lower()
        # `<org>__<org>__<rest>`
        if "__" in rest:
            second, after = rest.split("__", 1)
            if second.lower() == org_lower and after:
                return f"{org}__{after}"
        # `<org>__<org>-<rest>` (and `_<rest>` — note we already consumed `__`,
        # so the next separator is a single `-` or `_`).
        for sep in ("-", "_"):
            prefix = org_lower + sep
            if rest.lower().startswith(prefix) and len(rest) > len(prefix):
                return f"{org}__{rest[len(prefix):]}"

    return None


def _drop_host_prefix(value: str) -> tuple[str, Optional[str]] | None:
    """If value's developer prefix is a known hosting platform, return the
    bare suffix portion (everything after the first separator) AND the
    captured ``inference_platform`` id. Otherwise None.

    Handles both `host/model` and `host.model` separators.

    Returns:
        ``(rest, platform_id)`` where ``platform_id`` is the canonical
        inference_platform id (or None when the host token has no platform
        attribution in the single-source seed map). The casing of the raw
        value is preserved in ``rest`` (lookup is case-insensitive).
    """
    if "/" in value:
        org, rest = value.split("/", 1)
        org_lower = org.lower()
        if org_lower in _HOST_PREFIXES_TO_STRIP and rest:
            return (rest, _HOST_PREFIXES_TO_STRIP[org_lower])
    if "." in value:
        # Bedrock-style: "anthropic.claude-3-5-sonnet" → "anthropic.claude-3-5-sonnet"
        # is itself a host format, but the prefix BEFORE the dot is the host.
        # Only strip if everything-before-first-dot is a host name.
        first_dot = value.index(".")
        org = value[:first_dot]
        rest = value[first_dot + 1:]
        org_lower = org.lower()
        if org_lower in _HOST_PREFIXES_TO_STRIP and rest:
            return (rest, _HOST_PREFIXES_TO_STRIP[org_lower])
    return None

# Confidence assigned to stem-match results.  Below 1.0 (exact) and 0.95
# (normalized) so the provenance is clear in the resolution log.
_STEM_CONFIDENCE = 0.90


def _strip_suffix(value: str) -> str | None:
    """Strip a single known suffix.  Returns the stem or None if no suffix matched."""
    lower = value.lower()
    for suffix in _STRIP_SUFFIXES:
        if lower.endswith(suffix):
            return value[: len(value) - len(suffix)]
    for pattern in _STRIP_SUFFIX_PATTERNS:
        m = pattern.search(value)
        if m:
            return value[: m.start()]
    return None


def _strip_and_capture_platform_suffix(value: str) -> tuple[str, Optional[str]] | None:
    """Strip a known platform-capture suffix (`-together` / `-bedrock` /
    `-openrouter`) from the END, returning ``(stem, platform_id)``.

    Returns None when no platform suffix matched. Run BEFORE the generic
    `_strip_suffix()` so the host suffix is captured as a platform side-value
    rather than discarded. The stem is the value with the suffix removed, so
    the candidate that goes to matching is the same as a plain suffix strip —
    the platform is a SIDE value, never embedded in the candidate."""
    lower = value.lower()
    for suffix, platform in _SUFFIX_PLATFORM_MAP.items():
        if lower.endswith(suffix):
            stem = value[: len(value) - len(suffix)]
            return (stem, platform)
    return None


# Letter-then-dash-then-digit pattern: ``qwen-2`` → ``qwen2``. Models commonly
# appear in two spellings: ``qwen-2-72b`` (pipeline route_id form) vs
# ``qwen2-72b`` (registry canonical form). Collapsing the boundary dash lets
# them resolve to the same canonical without having to enumerate every variant
# as an alias. Safe because real distinguishing tokens are digits or words on
# the OTHER side of separators (e.g. ``gpt-4`` vs ``gpt-4-mini`` stays
# distinct because the ``-mini`` separator survives).
_LETTER_DIGIT_DASH = re.compile(r"([a-zA-Z])-(\d)")


def _collapse_letter_digit_dashes(value: str) -> str | None:
    """Return value with letter-digit boundary dashes removed, or None if no change."""
    collapsed = _LETTER_DIGIT_DASH.sub(r"\1\2", value)
    if collapsed == value:
        return None
    return collapsed


def _normalize_org(value: str) -> str | None:
    """Replace a known org-alias prefix.  Returns the rewritten string or None."""
    if "/" not in value:
        return None
    org, rest = value.split("/", 1)
    canonical_org = _ORG_ALIASES.get(org.lower())
    if canonical_org is None:
        return None
    return f"{canonical_org}/{rest}"


# OpenAI body detection — applied AFTER any host-prefix drop so
# `unknown/gpt-5-2025-08-07` and `openai/gpt-5-2025-08-07` both qualify.
# Word-boundary anchored so we don't match e.g. `gptq-int4` (a quant
# suffix elsewhere) or `o3rganization`.
_OPENAI_MODEL_BODY_RE = re.compile(
    r"^(?:gpt|chatgpt|o\d|davinci|babbage|curie|ada)(?:-|$)",
    re.IGNORECASE,
)


def _is_openai_shaped(value: str) -> bool:
    """True if `value` looks like an OpenAI model handle.

    Scoped because OpenAI's release cadence emits dated daily snapshots
    (`gpt-5-2025-08-07`) while the registry typically aliases the
    truncated-month form (`gpt-5-2025-08`) and root-collapses that to the
    moving family pointer (`gpt-5`). Other orgs use different conventions
    (Anthropic compresses to `YYYYMMDD`; Allen-AI uses YYMM tags like
    `1124`); applying ISO-date peeling broadly would either over-match
    (false-conflate distinct snapshots) or under-match (different format).
    """
    if value.lower().startswith("openai/"):
        return True
    body = value.split("/", 1)[1] if "/" in value else value
    return bool(_OPENAI_MODEL_BODY_RE.match(body))


# ISO-date tail capture, anchored. Strict component widths so we don't
# accidentally peel non-date numeric tokens (e.g. context length `-32k`
# isn't matched because `\d{4}` requires 4 digits). Year range guard is
# applied at strip-time, not in regex, so future-dated snapshots stay
# strippable without a regex bump.
_ISO_DATE_FULL_RE = re.compile(r"^(.+)-(\d{4})-(\d{2})-(\d{2})$")
_ISO_DATE_MONTH_RE = re.compile(r"^(.+)-(\d{4})-(\d{2})$")
_ISO_DATE_YEAR_RE = re.compile(r"^(.+)-(\d{4})$")


def _strip_openai_iso_date(value: str) -> list[str]:
    """For OpenAI-shaped values ending in an ISO-format date, return
    progressively-truncated candidates that STILL retain at least one
    date component. The bare-family candidate (everything stripped) is
    intentionally omitted: collapsing a dated snapshot all the way to
    its family pointer drops the per-snapshot identity and silently
    loses the snapshot's `release_date`. The auto-create + hub-stats
    path is the right home for that case — it creates a snapshot
    canonical with a `variant axis=version` parent edge to the family.

    When an INTERMEDIATE snapshot canonical is aliased in the registry
    (e.g. `openai/gpt-5-2025-08`), this function still returns it as a
    candidate so a more-specific raw value (`openai/gpt-5-2025-08-07`)
    can resolve to the existing snapshot rather than auto-creating a
    duplicate.

    Examples (registry contents shape what hits — this just emits the
    candidates that are tried in order):
        openai/gpt-5-2025-08-07 → [openai/gpt-5-2025-08, openai/gpt-5-2025]
        openai/o3-mini-2025-01-31 → [openai/o3-mini-2025-01, openai/o3-mini-2025]
        openai/gpt-4o-mini-2024 → []       (year-only has no intermediate;
                                            handled via auto-create path)
        meta/llama-3-2024-04-18 → []       (not OpenAI-shaped)
    """
    if not _is_openai_shaped(value):
        return []

    # Year sanity guard — only peel when the year looks like a real
    # release-snapshot year. Avoids stripping arbitrary 4-digit tokens
    # that aren't dates (e.g. parameter sizes, batch numbers).
    def _is_release_year(s: str) -> bool:
        try:
            y = int(s)
        except ValueError:
            return False
        return 2015 <= y <= 2035

    candidates: list[str] = []
    m = _ISO_DATE_FULL_RE.match(value)
    if m:
        prefix, y, mo, d = m.groups()
        if _is_release_year(y) and 1 <= int(mo) <= 12 and 1 <= int(d) <= 31:
            candidates.append(f"{prefix}-{y}-{mo}")
            candidates.append(f"{prefix}-{y}")
            return candidates

    m = _ISO_DATE_MONTH_RE.match(value)
    if m:
        prefix, y, mo = m.groups()
        if _is_release_year(y) and 1 <= int(mo) <= 12:
            candidates.append(f"{prefix}-{y}")
            return candidates

    # Year-only case (`-YYYY`) intentionally produces no candidates: the
    # only possible peel is to bare family, which the auto-create path
    # owns. Returning empty falls through to no_match cleanly.
    return candidates


def fuzzy_match(
    raw_value: str,
    entity_type: str,
    threshold: float,  # kept for API compat; not used by stem matching
    alias_store,
    source_config: Optional[str] = None,
    org_dev_map: Optional[dict] = None,
    known_orgs: Optional[frozenset] = None,
) -> tuple[Optional[str], float, Optional[str]]:
    """
    Attempt targeted fuzzy resolution.

    Returns ``(canonical_id, confidence, inference_platform)``; canonical_id is
    None on no match. ``inference_platform`` is a captured host token (prefix)
    or suffix (or None) — it is a SIDE VALUE and is NEVER embedded in the
    candidate string used for matching, so capture cannot alter the resolved
    canonical_id. The platform is captured exactly ONCE: a SUFFIX capture
    (`-bedrock`/`-together`/`-openrouter`) takes priority over a PREFIX capture
    (`together/`/`fireworks/`/…) because an explicit model-name suffix host
    token is the stronger per-run signal.
    """
    # Benchmarks get their own, much narrower vocabulary: run-configuration
    # tokens only.
    if entity_type == "benchmark":
        return _benchmark_stem_match(raw_value, alias_store, source_config)

    # The heuristics below are intentionally model-specific: they strip
    # hosting prefixes, org aliases, dated model snapshots, and inference-mode
    # suffixes. Applying them to metrics/harnesses can merge unrelated
    # entities that merely share a host-like prefix or model-ish tail.
    if entity_type != "model":
        return None, 0.0, None

    candidates_to_try: list[str] = []
    # The captured inference_platform side-value (first non-None source wins;
    # suffix wins over prefix because we attempt suffix capture first).
    captured_platform: Optional[str] = None

    # 1a. Thinking-budget "preserve" pass — runs BEFORE the generic
    # suffix strip so `model-thinking-16k` → `model-thinking` is tried
    # before `model-thinking-16k` → `model` (the latter drops the
    # thinking-mode signal). When a thinking-mode canonical exists, the
    # exact match on the preserved form wins; otherwise the lookup falls
    # through to the drop-thinking candidate produced by `_strip_suffix`.
    preserve_match = _THINKING_BUDGET_PRESERVE_RE.match(raw_value)
    if preserve_match:
        candidates_to_try.append(preserve_match.group(1))

    # 1a-host. Platform-capture suffix — runs BEFORE the generic suffix strip
    # so `-together`/`-bedrock`/`-openrouter` are CAPTURED (as the platform
    # side-value) rather than silently discarded. The produced stem is added
    # as a candidate and itself fed through the generic strip (handles e.g.
    # `model-fc-together` → capture `together`, then strip `-fc`). Capture has
    # priority over the prefix capture below.
    suffix_capture = _strip_and_capture_platform_suffix(raw_value)
    if suffix_capture:
        stem, platform = suffix_capture
        candidates_to_try.append(stem)
        if platform is not None:
            captured_platform = platform
        stem_stripped = _strip_suffix(stem)
        if stem_stripped:
            candidates_to_try.append(stem_stripped)

    # 1b. Suffix stripping (may produce multiple stems: strip one, strip two, etc.)
    stripped = _strip_suffix(raw_value)
    if stripped:
        candidates_to_try.append(stripped)
        # Try double-strip (e.g. "model-fc-prompt" — unlikely but cheap)
        double = _strip_suffix(stripped)
        if double:
            candidates_to_try.append(double)

    # 2. Host-prefix dropping — if raw_value's developer prefix is a known
    # hosting platform / gateway / placeholder, also try the bare suffix.
    # Apply on the original AND any suffix-stripped forms. The captured
    # platform is taken ONLY if a suffix capture didn't already set one
    # (suffix priority).
    for val in [raw_value] + candidates_to_try[:]:
        host = _drop_host_prefix(val)
        if host:
            bare, platform = host
            candidates_to_try.append(bare)
            if captured_platform is None and platform is not None:
                captured_platform = platform
            # The bare form might itself need suffix stripping
            stripped_bare = _strip_suffix(bare)
            if stripped_bare:
                candidates_to_try.append(stripped_bare)

    # 3. Duplicated-org-prefix collapse — catches typos like
    # `moonshotai/moonshotai-kimi-k2-instruct` (and the slug-form
    # `moonshotai__moonshotai-kimi-k2-instruct`). Runs AFTER suffix /
    # host strip so the deduped form goes through the rest of the
    # pipeline (org alias + lookup), and BEFORE org alias so the
    # collapsed string can pick up `_ORG_ALIASES` rewriting on the
    # next step.
    for val in [raw_value] + candidates_to_try[:]:
        deduped = _drop_duplicated_org_prefix(val)
        if deduped:
            candidates_to_try.append(deduped)

    # 4. Org normalization — on original, suffix-stripped, host-stripped,
    # and duplicate-org-collapsed forms.
    for val in [raw_value] + candidates_to_try[:]:
        rewritten = _normalize_org(val)
        if rewritten:
            candidates_to_try.append(rewritten)

    # 5. Letter-digit dash collapse — try every candidate with the
    # ``letter-digit`` boundary dash removed (e.g. ``qwen-2-72b`` →
    # ``qwen2-72b``). Run last so it composes with all earlier rewrites.
    for val in [raw_value] + candidates_to_try[:]:
        collapsed = _collapse_letter_digit_dashes(val)
        if collapsed:
            candidates_to_try.append(collapsed)

    # 6. OpenAI ISO-date suffix peel — progressively truncate
    # `-YYYY-MM-DD` → `-YYYY-MM` → `-YYYY` → bare. Scoped to OpenAI-shaped
    # raws because the cadence (daily snapshot of a monthly pointer that
    # version-collapses to the family root) is OpenAI-specific. Other
    # orgs' dated snapshots use different conventions and are handled by
    # `-\d{8}$` (Anthropic-style YYYYMMDD) or already-aliased canonicals.
    # Lookup-verified: each truncated candidate must hit an existing
    # alias to count, so this can never invent a mapping.
    for val in [raw_value] + candidates_to_try[:]:
        for peeled in _strip_openai_iso_date(val):
            candidates_to_try.append(peeled)

    # 7. Check each candidate against exact then normalized lookups.
    # Scoped-aware: config-scoped aliases for ``source_config`` count as
    # candidates; unrelated scoped aliases are excluded.
    norm_lookup = alias_store.get_normalized_lookup(entity_type, source_config)

    for candidate in candidates_to_try:
        # Org-agree against the CANDIDATE (host/account scaffolding already
        # stripped), NOT raw_value — a hosted raw like `together/meta-llama/
        # Llama-3-8B` carries the HOST as its prefix, which is not the model's
        # developer. The candidate's prefix is the real model org.
        exact_id = alias_store.lookup(candidate, entity_type, source_config)
        if exact_id is not None and _orgs_agree(candidate, exact_id, org_dev_map, known_orgs):
            return exact_id, _STEM_CONFIDENCE, captured_platform

        norm = normalize(candidate)
        canonical_id = norm_lookup.get(norm)
        if canonical_id is not None and _orgs_agree(candidate, canonical_id, org_dev_map, known_orgs):
            return canonical_id, _STEM_CONFIDENCE, captured_platform

    return None, 0.0, None
