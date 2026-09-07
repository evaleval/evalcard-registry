#!/usr/bin/env python3
"""
Generate seed/models/sources/models_dev.generated.yaml from models.dev.

This script writes ONE data source — pure models.dev output, no curated
overlays. The seed CLI loader applies `seed/models/core.yaml` (curated
canonicals) and `seed/models/enrichments/aliases.yaml` (optional alias
additions) at load time.

models.dev is strong on hosted-API model catalogs (Anthropic, OpenAI, xAI,
Google Gemini), weaker on open-weight families released directly to
HuggingFace (Meta Llama, Mistral / Mixtral, Qwen open weights, Gemma, Phi,
Yi, OLMo, Falcon, Granite, etc.). Curated entries in `core.yaml` cover
those and win at load time on id collision.

The right policy: prioritize correct expected coverage of what EEE actually
contains over the bounds of any single upstream catalog. When a refresh PR
introduces an unexpected drop or a too-coarse family, prefer adding/keeping
a `core.yaml` entry over chasing the upstream catalog.

This script fetches https://models.dev/api.json, filters to known
model-author providers (labs that release their own models, not re-hosting
inference providers), collapses models to family granularity, and writes
the generated YAML. Curated entries in core.yaml are NOT merged here — the
output is pure data-source.

Usage:
    python scripts/refresh_from_modelsdev.py              # fetch + write
    python scripts/refresh_from_modelsdev.py --no-fetch   # use /tmp cache
    python scripts/refresh_from_modelsdev.py --dry-run    # diff vs current

Re-running this is safe: it overwrites the generated YAML. The seed CLI
(`uv run eval-card-registry seed --local`) is idempotent over the result.

Source: https://models.dev (MIT, (c) 2025 models.dev)
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable

import yaml


class UnionFind:
    """Disjoint-set with path compression. Shared by the two grouping passes
    (catalog-wide dedup and intra-output reconciliation) so the merge semantics
    stay identical between them."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self._parent.setdefault(x, x)
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        self._parent[self.find(a)] = self.find(b)

from eval_card_registry.lib.collision_fold import _bsizes, collision_key
from eval_card_registry.lib.seed_io import (
    WEAK_SCALAR_FIELDS,
    resolve_oracle_path,
    safe_load_yaml,
)

# Resolver lives in the workspace package; this script runs from the repo
# root via `uv run`, so the import resolves through pyproject's path dep.
from eval_entity_resolver.display import humanize_model_slug
from eval_entity_resolver.strategies.fuzzy import _ORG_ALIASES

# Strip trailing date suffixes and `-latest` to collapse a model down to its
# major-version family slug. Mirrors the resolver's fuzzy stem so per-snapshot
# entries from models.dev (e.g. `gpt-4o-2024-05-13`, `claude-opus-4-5-20251101`)
# fold into a single canonical with the dated snapshots as aliases.
_FAMILY_DATE_RES = [
    re.compile(r"-\d{8}$"),                  # YYYYMMDD: -20251101
    re.compile(r"-\d{4}-\d{2}-\d{2}$"),      # YYYY-MM-DD: -2024-05-13
    re.compile(r"-preview-\d{2}-\d{2}$"),    # -preview-05-06 (Google Gemini preview snapshots)
    re.compile(r"-preview-\d{4}-\d{2}-\d{2}$"),  # -preview-2024-05-13 (rare)
    re.compile(r"-preview$"),                # bare -preview
    re.compile(r"-latest$"),                 # -latest hosting tag
    re.compile(r"-v\d+(\.\d+)*$", re.IGNORECASE),  # version suffix: -v0.3, -v1, -v1.0.0
]
# Legacy alias names kept for any external callers (tests etc.)
_FAMILY_LATEST_RE = _FAMILY_DATE_RES[-1]
_FAMILY_PREVIEW_RE = _FAMILY_DATE_RES[-2]

# Training-stage suffixes — matches the resolver's _STRIP_SUFFIXES. Stripped
# from canonical so base / instruct / chat / it variants share one entry.
_FAMILY_STAGE_SUFFIXES = ("-instruct", "-chat", "-it", "-base")

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED_PATH = REPO_ROOT / "seed" / "models" / "sources" / "models_dev.generated.yaml"
ORGS_SEED_PATH = REPO_ROOT / "seed" / "orgs.yaml"
CACHE_PATH = Path("/tmp/modelsdev_api.json")
SOURCE_URL = "https://models.dev/api.json"

# Map models.dev provider slug -> our canonical_orgs.id.
# Most match by name; a few need translation. Providers not listed here are
# skipped (most are inference re-hosts, gateways, regional duplicates).
PROVIDER_TO_ORG: dict[str, str] = {
    "anthropic": "anthropic",
    "openai": "openai",
    "google": "google",
    "xai": "xai",
    "cohere": "cohere",
    "mistral": "mistralai",
    "deepseek": "deepseek",
    "alibaba": "alibaba",
    "moonshotai": "moonshotai",
    "stepfun": "stepfun",
    "minimax": "minimax",
    "zai": "zai",
    "inception": "inception",
    "upstage": "upstage",
    "perplexity": "perplexity",
    "nvidia": "nvidia",
    # Add more here as we extend the allowlist; each must have a matching
    # entry in seed/orgs.yaml or the validator will fail.
}

# --- inference_platforms single-source --------------------------------------
# Path to the curated platform catalog (source of truth). The same file
# seeds seed/inference_platforms.yaml; we read its `models_dev_provider` field
# to build PROVIDER_TO_INFERENCE_PLATFORM so the host-token→platform map stays
# byte-identical between seed generation here and runtime capture in fuzzy.py
# (single-source mandate — DO NOT hand-maintain a parallel dict).
INFERENCE_PLATFORMS_JSON = (
    REPO_ROOT  / "curation" / "inference_platforms.proposed.json"
)


def _load_provider_to_inference_platform(
    path: Path = INFERENCE_PLATFORMS_JSON,
) -> dict[str, str]:
    """Build {models.dev provider slug -> inference_platforms.id} from the
    curated catalog. Every platform built from a models.dev provider declares
    its `models_dev_provider` (null only for non-provider platforms, e.g.
    prodia); this maps all of them, so every provider with a catalog entry is
    processed (PROVIDER_TO_ORG, narrower, only governs which providers can
    anchor a group's authorship)."""
    data = json.loads(path.read_text())
    mapping: dict[str, str] = {}
    for plat in data.get("platforms", []):
        prov = plat.get("models_dev_provider")
        pid = plat.get("id")
        if not prov or not pid:
            continue
        mapping[prov] = pid
    return mapping


# Module-level singleton. Falls back to an empty dict if the curated catalog
# file is not present; in that case source the map from
# seed/inference_platforms.yaml instead.
try:
    PROVIDER_TO_INFERENCE_PLATFORM: dict[str, str] = _load_provider_to_inference_platform()
except FileNotFoundError:  # pragma: no cover - catalog file absent
    PROVIDER_TO_INFERENCE_PLATFORM = {}


# --- Author-lab classification + org inference -----------------------------

# Author-lab provider slugs, sourced from the SAME curated catalog (kind ==
# "author_lab"). These are the providers whose spelling can anchor a group's
# authorship (but only when their org matches the family-implied org — a lab
# can re-host others' models too).
def _load_strict_author(path: Path = INFERENCE_PLATFORMS_JSON) -> set[str]:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:  # pragma: no cover
        return set()
    return {
        p["models_dev_provider"]
        for p in data.get("platforms", [])
        if p.get("kind") == "author_lab" and p.get("models_dev_provider")
    }


STRICT_AUTHOR: set[str] = _load_strict_author()

# Author-lab provider id -> HF-style org slug (used when that provider's
# spelling anchors a group).
AUTHOR_PROV_ORG: dict[str, str | None] = {
    "anthropic": "anthropic", "openai": "openai", "google": "google",
    "mistral": "mistralai", "cohere": "cohere", "zai": "zai-org",
    "zhipuai": "zai-org", "alibaba": "qwen", "deepseek": "deepseek-ai",
    "llama": "meta-llama", "minimax": "minimaxai", "moonshotai": "moonshotai",
    "nvidia": "nvidia", "xai": "xai", "xiaomi": "xiaomi",
    "stepfun": "stepfun-ai", "stepfun-ai": "stepfun-ai", "upstage": "upstage",
    "venice": None, "perplexity": "perplexity", "perplexity-agent": "perplexity",
    "nova": "amazon", "sarvam": "sarvam-ai", "inception": "inceptionai",
    "poolside": "poolside", "morph": "morph", "v0": "vercel",
    "lucidquery": None, "inceptron": None,
}

# org inference from family / name tokens (HF-style slugs).
ORG_BY_FAMILY_PREFIX: dict[str, str] = {
    "claude": "anthropic", "gpt": "openai", "o-": "openai", "o": "openai", "gpt-": "openai",
    "gemini": "google", "gemma": "google", "imagen": "google", "learnlm": "google",
    "qwen": "qwen", "qwen3": "qwen", "qwen3.": "qwen",
    "llama": "meta-llama",
    "glm": "zai-org", "glm-": "zai-org",
    "deepseek": "deepseek-ai",
    "minimax": "minimaxai", "mimo": "xiaomi",
    "kimi": "moonshotai",
    "grok": "xai",
    "mistral": "mistralai", "ministral": "mistralai", "devstral": "mistralai",
    "codestral": "mistralai", "mistral-": "mistralai", "mixtral": "mistralai",
    "phi": "microsoft",
    "nemotron": "nvidia", "nova": "amazon", "command": "cohere",
    "command-r": "cohere", "command-a": "cohere",
    "ernie": "baidu", "hunyuan": "tencent", "seed": "bytedance", "doubao": "bytedance",
    "flux": "black-forest-labs", "voyage": "voyageai", "ling": "inclusionai",
    "gpt-oss": "openai",
}


def org_from_family(fam: str | None) -> str | None:
    """Infer an HF-style org slug from a models.dev `family` token."""
    if not fam:
        return None
    f = fam.lower()
    base = f.split("-")[0]
    for key in (f, base):
        if key in ORG_BY_FAMILY_PREFIX:
            return ORG_BY_FAMILY_PREFIX[key]
    for pref, org in ORG_BY_FAMILY_PREFIX.items():
        if f.startswith(pref):
            return org
    return None


# Map HF-style org slugs onto the registry's CURATED developer-org slugs
# (org identity & casing model: curated dev org ids keep their authored slug;
# HF namespaces are recorded as aliases and RESOLVE to the dev org). Built once
# from seed/orgs.yaml alias index, with a few explicit overrides for HF slugs
# that aren't already aliased.
_ORG_SLUG_OVERRIDES: dict[str, str] = {
    "meta-llama": "meta",
    "qwen": "alibaba",
    "deepseek-ai": "deepseek",
    "zai-org": "zai",
}


def _build_org_alias_index() -> dict[str, str]:
    """{lowercased org id / hf_org / alias / _ORG_ALIASES key -> curated org id}.

    Built by the shared eval_entity_resolver.fold.build_curated_org_map:
    `_ORG_ALIASES` UNION every curated org's id + hf_org + `aliases`. Using the
    shared builder (rather than reading orgs.yaml directly) is what folds e.g.
    `minimaxai`->minimax via `_ORG_ALIASES`, and keeps this map identical to the
    one the resolver uses."""
    from eval_entity_resolver.fold import build_curated_org_map
    data = safe_load_yaml(ORGS_SEED_PATH.read_text()) if ORGS_SEED_PATH.exists() else []
    return build_curated_org_map(data or [])


_DEV_ALIAS_INDEX: dict[str, str] | None = None


def _dev_alias_index() -> dict[str, str]:
    """Module-cached org alias index (orgs.yaml parsed once, not per group)."""
    global _DEV_ALIAS_INDEX
    if _DEV_ALIAS_INDEX is None:
        _DEV_ALIAS_INDEX = _build_org_alias_index()
    return _DEV_ALIAS_INDEX


def normalize_org_slug(hf_org: str | None, alias_index: dict[str, str]) -> str | None:
    """Map an HF-style org slug to the registry's curated org id when one
    exists; else return the HF slug unchanged (a new HF-derived org row will be
    auto-created with HF casing). Returns None for None in."""
    if not hf_org:
        return None
    if hf_org in _ORG_SLUG_OVERRIDES:
        return _ORG_SLUG_OVERRIDES[hf_org]
    mapped = alias_index.get(hf_org.lower())
    return mapped if mapped else hf_org


# --- Developer (org) derivation: PREFIX-authoritative ----------------------
# The model id's org prefix is the developer. The model NAME only says what a
# model is DERIVED FROM (base lineage), not who MADE it, so name-matching is
# used ONLY for bare ids / serving-hosted ids (no genuine prefix), NEVER to
# override a prefix (`3rd-Degree-Burn/Llama-...` is by 3rd-Degree-Burn, NOT
# meta; `nvidia/llama-nemotron` is nvidia's, NOT meta). Serving hosts are
# stripped; a curated-prefix model whose name disagrees (a possible re-host,
# e.g. nvidia/whisper) is FLAGGED for curation, not auto-flipped.
# Serving / gateway platforms (where a model is SERVED, not DEVELOPED) — a model
# id prefixed with one of these is stripped and the developer taken from the
# name. Single-sourced from the curated inference_platforms catalog: every
# models.dev provider that is NOT an author_lab (those re-host others' models —
# fireworks, together, volcengine, fal-ai, openrouter, ...), plus host
# scaffolding tokens. (nvidia IS an author_lab, so it stays out and its genuine
# re-hosts are handled by explicit curation, not stripped here.)
_SERVING_HOSTS = {
    "accounts", "clarifai", "route", "orcarouter", "workers-ai", "openrouter",
    "stealth", "hf", "cf", "@cf",
    # serving-brand id namespaces (not top-level providers, but appear as id
    # prefixes): ByteDance's Volcano Engine cloud, fal's image-serving, etc.
    "volcengine", "fal-ai", "fal", "kilo", "kilo-auto",
    # nano-gpt's trusted-execution serving namespace (`TEE/gemma4-31b`) — a
    # serving mode like the `-tee` suffix, NOT an uploader org.
    "tee",
} | ({p.lower() for p in PROVIDER_TO_INFERENCE_PLATFORM} - {a.lower() for a in STRICT_AUTHOR})
# model NAME starts with TOKEN -> HF-style developer slug (normalize_org_slug
# maps to the curated org). Longest-prefix-first.
_NAME_VENDOR_MAP: list[tuple[str, str]] = sorted([
    ("meta-llama", "meta-llama"), ("codellama", "meta-llama"), ("llama", "meta-llama"),
    ("ministral", "mistralai"), ("mixtral", "mistralai"), ("pixtral", "mistralai"),
    ("codestral", "mistralai"), ("devstral", "mistralai"), ("magistral", "mistralai"),
    ("mistral", "mistralai"),
    ("qwen", "qwen"), ("qwq", "qwen"), ("qvq", "qwen"),
    ("gpt-oss", "openai"), ("gpt", "openai"), ("o1", "openai"), ("o3", "openai"),
    ("o4", "openai"), ("whisper", "openai"), ("chatgpt", "openai"), ("codex", "openai"),
    ("dall-e", "openai"), ("text-embedding", "openai"),
    ("claude", "anthropic"),
    ("gemini", "google"), ("paligemma", "google"), ("gemma", "google"),
    ("imagen", "google"), ("learnlm", "google"),
    ("deepseek", "deepseek-ai"), ("grok", "xai"), ("chatglm", "zai-org"), ("glm", "zai-org"),
    ("kimi", "moonshotai"), ("moonshot", "moonshotai"), ("minimax", "minimax"),
    ("phi", "microsoft"), ("nemotron", "nvidia"), ("nvlm", "nvidia"), ("mimo", "xiaomi"),
    ("hunyuan", "tencent"), ("ernie", "baidu"), ("doubao", "bytedance"), ("seed", "bytedance"),
    ("command", "cohere"), ("aya", "cohere"), ("nova", "amazon"), ("titan", "amazon"),
    ("solar", "upstage"), ("jamba", "ai21"), ("jurassic", "ai21"), ("sonar", "perplexity"),
    ("hermes", "nousresearch"), ("granite", "ibm"), ("flux", "black-forest-labs"),
    ("voyage", "voyageai"), ("cogito", "deepcogito"), ("falcon", "tiiuae"),
    ("olmo", "allenai"), ("tulu", "allenai"), ("bge", "baai"), ("inflection", "inflection"),
], key=lambda kv: -len(kv[0]))


def developer_from_name(name: str | None) -> str | None:
    """HF-style developer slug from a LEADING vendor token in the model name.
    Leading-token only, so a derivative's base token mid-name can't hijack it.
    For BARE / serving-hosted ids only — never to override a real prefix."""
    if not name:
        return None
    s = re.sub(r"^[a-z]+\.", "", str(name).strip().lower())
    s = s.split("/")[-1]
    s = re.sub(r"[_\s]+", "-", s)
    for tok, dev in _NAME_VENDOR_MAP:
        if re.match(re.escape(tok) + r"([0-9._:\-]|$)", s):
            return dev
    return None


def _derive_group_org(
    recs: list[dict], alias_index: dict[str, str]
) -> tuple[str | None, str | None]:
    """Developer org for a models.dev underlying group (prefix-authoritative).

    Returns (hf_org_slug | None, rehost_review | None). `rehost_review` is the
    disagreeing name-vendor when a CURATED-org prefix's model name points to a
    different vendor (a possible re-host to curate, e.g. nvidia/whisper)."""
    prefix_orgs: list[str] = []
    name_orgs: list[str] = []
    for r in recs:
        # Strip provider prefixes the same way normalize_modelsdev_id does BEFORE
        # taking the org, so an id like `hf:nvidia/...` yields org `nvidia`, not
        # `hf:nvidia` (which would mint a malformed `hf:nvidia/...` canonical + a
        # dangling org FK). Mirrors the `^hf:` strip on the normalize path.
        raw = re.sub(r"^hf:", "", (r.get("raw") or "").lstrip("~"))
        if "/" in raw and raw.split("/")[0].lower() not in _SERVING_HOSTS:
            prefix_orgs.append(raw.split("/")[0])          # uploader OR curated (raw spelling)
        else:                                              # bare or serving-hosted
            no = developer_from_name(r.get("name"))
            if no:
                name_orgs.append(no)
    if prefix_orgs:
        # Take the prefix VERBATIM (most common spelling present). We do NOT
        # reconcile spelling variants (e.g. 'TheDrummer 2' vs 'thedrummer') — we
        # have no authoritative basis to assert they're the same uploader, so
        # picking/cleaning one would be an arbitrary, unverified identity claim.
        # Curated orgs.yaml is the place to assert such equivalences explicitly.
        low = Counter(p.lower() for p in prefix_orgs).most_common(1)[0][0]
        org = next(p for p in prefix_orgs if p.lower() == low)
        rehost = None
        if alias_index.get(org.lower()) and name_orgs:    # curated prefix + name signal
            nd = Counter(name_orgs).most_common(1)[0][0]
            if normalize_org_slug(nd, alias_index) != normalize_org_slug(org, alias_index):
                rehost = nd
        return org, rehost
    if name_orgs:
        return Counter(name_orgs).most_common(1)[0][0], None
    return None, None


# --- Dedup key + head-pick --------------------------------------------------
_DATE8_RE = re.compile(r"^\d{8}$")
_DATE6_RE = re.compile(r"^\d{6}$")
_NUM_TOKEN_RE = re.compile(r"^\d+[a-z]?$")


def normalize_modelsdev_id(raw: str, strip_variants: bool = True) -> str:
    """Normalize a models.dev model id to an underlying-model spelling — strips
    provider/host/region scaffolding, unifies separators.

    `strip_variants` (default True): also collapse IDENTITY variant suffixes
    (-turbo/-thinking/-reasoner/-fp8/...) so a variant groups with its base for
    routing/dedup (build_underlying_groups). Pass False for ALIAS derivation so
    a variant keeps its own identity and never emits the base id as an alias —
    otherwise a variant (e.g. `gpt-4-turbo`) would claim the base canonical's id
    (`gpt-4`) as one of its aliases, double-claiming it and aborting the seed.
    Serving TAGS (`:free`, `-fast`, ...) are stripped regardless — they are
    scaffolding, not a distinct model."""
    s = raw.strip()
    s = s.lstrip("~")  # openrouter '~' latest marker
    s = re.sub(r"^accounts/[^/]+/models/", "", s)
    s = re.sub(r"^hf:", "", s)
    s = re.sub(r"^@cf/", "", s)
    s = re.sub(r"^clarifai/[^/]+/models/", "", s)
    s = re.sub(r"^route/", "", s)
    s = re.sub(r"^orcarouter/", "", s)
    s = s.replace("--", "/")  # sap style
    s = re.sub(r"^databricks-", "", s)
    s = re.sub(r"^azure-", "", s)
    s = re.sub(r"^aws-", "", s)
    s = re.sub(r"^openai-", "", s)
    s = re.sub(r"^anthropic-", "", s)
    s = re.sub(r"^ai21-", "ai21/", s)
    s = re.sub(r"^stealth/", "", s)
    s = _strip_host_region_prefixes(s)
    if "/" in s:
        s = s.split("/")[-1]
    s = _strip_host_region_prefixes(s)
    s = s.lower()
    s = s.replace("@default", "")
    s = re.sub(r"@(\d{8})$", r"-\1", s)
    s = re.sub(r"@.*$", "", s)
    s = _BEDROCK_VER_RE.sub("", s)
    # A trailing `:NNN<unit>` is a SIZE spec (e.g. `gpt-oss:120b`), not a serving
    # tag — convert to `-NNN<unit>` so it survives the `:.*$` tag strip and keeps
    # distinct sizes in distinct groups (gpt-oss:20b vs gpt-oss:120b). The size
    # UNIT (b/m/k/t) is REQUIRED so a bare `:1024` / `:1.0` (thinking budget /
    # version serving param) is still stripped as a tag, not kept as a size.
    s = re.sub(r":(\d+(?:\.\d+)?[bmkt])$", r"-\1", s)
    prev = None
    while prev != s:
        prev = s
        s = _TAG_SUFFIX_RE.sub("", s)           # scaffolding tags: always
        if strip_variants:
            s = _IDENTITY_VARIANT_RE.sub("", s)  # identity variants: routing/dedup only
    s = re.sub(r"(\d)\.(\d)", r"\1-\2", s)
    s = re.sub(r"[_\s]+", "-", s)
    s = re.sub(r"\(.*$", "", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


_HOST_PREFIXES = ["amazon.", "anthropic.", "qwen.", "meta.", "google.", "cohere.",
                  "ai21.", "deepseek.", "mistral."]
_REGION_PREFIXES = ["us.", "eu.", "jp.", "au.", "apac.", "global."]
# Serving/routing TAGS — pure scaffolding (a hosting tier / serving mode /
# provider routing marker, NOT a distinct model). Always stripped, even for alias
# derivation: `gpt-4o:free` and `gpt-4o` are the SAME canonical served at
# different tiers. `-tee` (trusted-execution serving) and `-maas` (model-as-a-
# service tier) are serving modes, not distinct models. NOTE: a trailing `:NNNb`
# is a SIZE spec (gpt-oss:120b), NOT a tag — it is converted to `-NNNb` in
# normalize_modelsdev_id BEFORE the `:.*$` strip fires, so it survives as a size.
_TAG_SUFFIX_RE = re.compile(r"(:.*$)|(-fast$)|(-precision$)|(-free$)|(-maas$)|(-tee$)")
# The PREFIX spelling of the trusted-execution serving marker (`TEE/gemma4-31b`)
# — stripped like the `-tee` suffix; the raw form is kept as an alias on the
# stripped target.
_TEE_PREFIX_RE = re.compile(r"^TEE/", re.IGNORECASE)
# IDENTITY variants — a genuinely DIFFERENT canonical (a decoding mode, a quant).
# Stripped ONLY for routing/dedup grouping (strip_variants=True), NEVER for alias
# derivation: collapsing `gpt-4-turbo`->`gpt-4` would emit the BASE id as one of
# the VARIANT's aliases, stealing the base canonical's id (a double-claim that
# aborts the seed).
_IDENTITY_VARIANT_RE = re.compile(
    r"(-thinking$)|(-think$)|(-reasoner$)|(-reasoning$)"
    r"|(-turbo$)"
    r"|(-fp8$)|(-bf16$)|(-int8$)|(-awq$)|(-gptq$)"
)
# Back-compat union (some external callers/tests referenced the old name).
_VARIANT_SUFFIX_RE = re.compile(
    _TAG_SUFFIX_RE.pattern + "|" + _IDENTITY_VARIANT_RE.pattern
)
_BEDROCK_VER_RE = re.compile(r"-v\d+:\d+$")


def _strip_host_region_prefixes(s: str) -> str:
    changed = True
    while changed:
        changed = False
        for rp in _REGION_PREFIXES:
            if s.startswith(rp):
                s = s[len(rp):]
                changed = True
        for hp in _HOST_PREFIXES:
            if s.startswith(hp):
                s = s[len(hp):]
                changed = True
    return s


def canon_key_ordered(norm: str) -> str:
    """Underlying-model key (order-preserving)."""
    s = re.sub(r"-(latest|old|new)$", "", norm)
    s = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", s)
    s = re.sub(r"-\d{2}-\d{2}$", "", s)
    toks = [t for t in s.split("-") if not _DATE8_RE.match(t) and not _DATE6_RE.match(t)]
    if toks and re.match(r"^v\d+$", toks[-1]):
        toks.pop()
    return "-".join(toks)


def safe_sig(key: str) -> str:
    """Token-multiset signature for the merge pass. Words sorted, version
    numbers order-preserved (so `gpt-4-5` != `gpt-5-4`)."""
    toks = key.split("-")
    words = sorted(t for t in toks if not _NUM_TOKEN_RE.match(t))
    nums = [t for t in toks if _NUM_TOKEN_RE.match(t)]
    return "|".join(words) + "#" + "-".join(nums)


def build_underlying_groups(api_json: dict) -> dict[str, list[dict]]:
    """Group all (provider, model) records across the catalog into underlying
    groups keyed by a canonical root via a union-find merge pass. Returns
    {root_key -> [records]} where each record is
    {provider, raw, norm, key, family, release, name, open_weights}."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for prov, pdata in api_json.items():
        for mid, mr in (pdata.get("models") or {}).items():
            n = normalize_modelsdev_id(mid)
            k = canon_key_ordered(n)
            if not k:
                continue
            groups[k].append(dict(
                provider=prov, raw=mid, norm=n, key=k,
                family=mr.get("family"), release=mr.get("release_date"),
                name=mr.get("name"), open_weights=mr.get("open_weights"),
                record=mr,
            ))

    # Union-find: merge ordered keys sharing the same token-multiset signature,
    # but only when their families don't CONFLICT (guards anagram merges).
    uf = UnionFind()

    for k in groups:
        uf.find(k)
    mset: dict[str, list[str]] = defaultdict(list)
    for k in groups:
        mset[safe_sig(k)].append(k)
    for _ms, ks in sorted(mset.items()):
        if len(ks) > 1:
            # Deterministic base (lexicographically first key, not models.dev's
            # API iteration order) so grouping can't flip when upstream reorders
            # keys. fam_a ACCUMULATES the families of every key already merged in,
            # making the merge transitive: k1{A}, k2{A,B}, k3{B} all land in one
            # group (k2 bridges k1 and k3) instead of splitting k3 off the base.
            ks = sorted(ks)
            base = ks[0]
            fam_a = {r["family"] for r in groups[base] if r["family"]}
            for o in ks[1:]:
                fam_b = {r["family"] for r in groups[o] if r["family"]}
                if fam_a & fam_b or not fam_a or not fam_b:
                    uf.union(o, base)
                    fam_a |= fam_b

    merged: dict[str, list[dict]] = defaultdict(list)
    for k, recs in groups.items():
        merged[uf.find(k)].extend(recs)
    return merged


def pick_underlying(root: str, recs: list[dict]) -> dict:
    """Choose the authoritative (org, display_name, release, open_weights) and
    head spelling for an underlying group. Returns a dict with a
    `head_spelling` (the cleanest spelling to mint from).

    Org returned is HF-style; callers normalize via normalize_org_slug() to the
    curated dev org."""
    fams = [r["family"] for r in recs if r["family"]]
    fam_org = org_from_family(Counter(fams).most_common(1)[0][0]) if fams else None
    author_recs = [r for r in recs if r["provider"] in STRICT_AUTHOR]
    # Sort by cleanest spelling (shortest normalized form, then lexicographic)
    # so the head_spelling / display_name pick is deterministic — NOT models.dev's
    # provider iteration order, which would otherwise flip the minted id when
    # upstream reorders providers.
    true_author_recs = sorted(
        (r for r in author_recs
         if fam_org is None or AUTHOR_PROV_ORG.get(r["provider"]) == fam_org),
        key=lambda r: (len(r["norm"]), r["raw"]),
    )
    # Developer org: PREFIX-authoritative (the id's namespace is the developer);
    # name only for bare/serving ids; never name-override a prefix. Re-host
    # disagreements (curated prefix vs name) are flagged for curation, not flipped.
    org, rehost_review = _derive_group_org(recs, _dev_alias_index())
    has_author = bool(true_author_recs)  # gates family-tree vs single mint downstream

    if true_author_recs:
        disp = true_author_recs[0]["name"] or root
        head_spelling = true_author_recs[0]["raw"]
    else:
        names = [r["name"] for r in recs if r["name"]]
        disp = Counter(names).most_common(1)[0][0] if names else root
        # Head spelling for a re-host-only group: the cleanest normalised
        # spelling = the one whose normalized form is shortest / equals root.
        head_spelling = _cleanest_spelling(root, recs)

    rels = [r["release"] for r in recs if r["release"]]
    release = min(rels) if rels else None
    ow = any(r["open_weights"] for r in recs)
    return {
        "author_org": org,
        "display_name": disp,
        "release_date": release,
        "open_weights": ow,
        "has_author_lab_entry": has_author,
        "head_spelling": head_spelling,
        "rehost_review": rehost_review,
    }


def _cleanest_spelling(root: str, recs: list[dict]) -> str:
    """Pick the cleanest raw spelling to mint a canonical from when no author
    lab anchors the group: prefer a record whose normalized form equals the
    canon root, else the shortest normalized form; tie-break alphabetically by
    raw for determinism."""
    exact = [r for r in recs if r["norm"] == root]
    pool = exact or recs
    return sorted(pool, key=lambda r: (len(r["norm"]), r["raw"]))[0]["raw"]


def _fetch(use_cache: bool) -> dict:
    if use_cache and CACHE_PATH.exists():
        print(f"[refresh] using cache: {CACHE_PATH}", file=sys.stderr)
        return json.loads(CACHE_PATH.read_text())
    print(f"[refresh] fetching {SOURCE_URL}", file=sys.stderr)
    # Send an identifiable User-Agent — models.dev's CDN rejects the default
    # Python-urllib UA with a 403, and a generic UA is a good citizen anyway
    # since it lets the data source see who's hitting the API.
    req = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "evalcard-registry-refresh/1.0 (+https://github.com/evaleval/evalcard-registry)"},
    )
    with urllib.request.urlopen(req) as r:
        raw = r.read().decode()
    CACHE_PATH.write_text(raw)
    return json.loads(raw)


def _slugify(value: str) -> str:
    """Lowercase + collapse whitespace/underscores to single hyphens.
    Preserves dots (for version readability), hyphens, slashes. Drops
    parens, brackets, and other display-only punctuation that would
    otherwise leak into canonical ids."""
    s = value.strip().lower()
    # Drop punctuation that's display-only (e.g. "Claude 4.5 (latest)")
    s = re.sub(r"[()\[\]{}]", "", s)
    s = re.sub(r"[\s_]+", "-", s)   # spaces/underscores → hyphen
    s = re.sub(r"-+", "-", s)        # collapse multiple hyphens
    return s.strip("-")


def _family_for(model: dict) -> str:
    """Group key for a model record (= the lineage canonical slug).

    Prefer the `name` field over `id` because models.dev's `id` slugs
    sometimes mangle separators (e.g. Alibaba's `qwen2-5-14b-instruct` for
    what HF calls `Qwen2.5-14B-Instruct`). The `name` field carries the
    lab's spelling with dots intact.

    Strip date suffixes (snapshot collapse), `-latest`/`-preview` markers,
    and training-stage suffixes (`-instruct`, `-chat`, `-it`, `-base`) so
    base + instruct + chat variants share one canonical. This mirrors the
    resolver's fuzzy stem; doing it here means the seed is consistent with
    the resolution rule.

    We do NOT use models.dev's `family` field — it's too coarse (groups all
    Claude Opus major versions 4.0/4.5/4.7 under one slug). The pipeline's
    `family_slug` works at "Opus 4.5" granularity.
    """
    raw = _slugify(model.get("name") or model.get("id", ""))
    # Loop until no suffix matches — date / preview / latest / training-stage
    # patterns can stack (e.g. "-preview-05-06-instruct"). Each strip is a
    # monotonic shrink, so the fixpoint is reached and the loop terminates.
    while True:
        before = raw
        for pat in _FAMILY_DATE_RES:
            raw = pat.sub("", raw)
        for suffix in sorted(_FAMILY_STAGE_SUFFIXES, key=len, reverse=True):
            if raw.endswith(suffix):
                raw = raw[: -len(suffix)]
                break
        if raw == before:
            break
    return raw


# --- Suffix → (relationship, axis) classification --------------------------
# Used to translate the diff between a snapshot's id and its family slug
# into a typed `parents` edge. The classification table uses the same enum
# values as curated core.yaml entries so models.dev refresh and curated
# entries land in identical shapes.
#
# Axis semantics (closed enum):
#   version  — dated snapshot / vN marker: same API identity, different release.
#   training_stage — base / instruct / chat / it: a post-training stage of the
#                    same pretrained model.
#   tier     — branded sibling (haiku/sonnet/opus, mini/nano, flash/pro,
#              small/medium/large): a DIFFERENT product in the same family that
#              makes NO disclosed-scale claim. Never emit a `size` edge for these.
#   size     — a genuinely-disclosed scale: an open-weight name size token
#              (7b/70b/405b/8x7b) or a MoE active-param token (a16b). See
#              `_is_size_token`. NEVER assert for a branded tier.
#   mode     — runtime/decoding mode that is not a training stage (thinking,
#              reasoning, guard, …).
#   modality / domain — unchanged.
_TOKEN_CLASSIFICATIONS: dict[str, tuple[str, str | None]] = {
    # Training-stage post-training axis.
    "instruct": ("variant", "training_stage"),
    "it": ("variant", "training_stage"),
    "chat": ("variant", "training_stage"),
    "base": ("variant", "training_stage"),
    "pt": ("variant", "training_stage"),
    "sft": ("variant", "training_stage"),
    # Branded tiers — sibling products, NO scale claim (axis=tier).
    "haiku": ("variant", "tier"),
    "sonnet": ("variant", "tier"),
    "opus": ("variant", "tier"),
    "mini": ("variant", "tier"),
    "nano": ("variant", "tier"),
    "micro": ("variant", "tier"),
    "flash": ("variant", "tier"),
    "pro": ("variant", "tier"),
    "small": ("variant", "tier"),
    "medium": ("variant", "tier"),
    "large": ("variant", "tier"),
    "lite": ("variant", "tier"),
    # Runtime / decoding modes (NOT training stages).
    "thinking": ("variant", "mode"),
    "reasoning": ("variant", "mode"),
    "nothink": ("variant", "mode"),
    "guard": ("variant", "mode"),
    "safeguard": ("variant", "mode"),
    "moderation": ("variant", "mode"),
    # Modality / domain.
    "vision": ("variant", "modality"),
    "vl": ("variant", "modality"),
    "coder": ("variant", "domain"),
    "code": ("variant", "domain"),
    "math": ("variant", "domain"),
    # Precision / serving quantization.
    "turbo": ("quantized", None),
    "fp8": ("quantized", None),
    "fp16": ("quantized", None),
    "bf16": ("quantized", None),
    "int4": ("quantized", None),
    "int8": ("quantized", None),
    "awq": ("quantized", None),
    "gptq": ("quantized", None),
    "gguf": ("quantized", None),
}
# The set of branded-tier tokens — used to GUARD against ever emitting a size
# edge for a tier (branded tiers never carry a scale claim).
_TIER_TOKENS = frozenset(
    k for k, (_rel, axis) in _TOKEN_CLASSIFICATIONS.items() if axis == "tier"
)
_VERSION_RE = re.compile(r"^v\d+(\.\d+)*$", re.IGNORECASE)
_DATE_8_RE = re.compile(r"^\d{8}$")
_DATE_4_RE = re.compile(r"^\d{4}$")
_DATE_3_RE = re.compile(r"^\d{3}$")
_MOE_ACTIVE_RE = re.compile(r"^a\d+b$", re.IGNORECASE)
# Disclosed open-weight scale tokens: 7b, 70b, 405b, 1.5b (dot already slugged
# to dash so we also accept a leading number-dash-number — handled in the
# size-aware classifier), and MoE expert tokens like 8x7b / 8x22b.
_SIZE_TOKEN_RE = re.compile(r"^\d+(\.\d+)?b$", re.IGNORECASE)
_MOE_EXPERT_RE = re.compile(r"^\d+x\d+(\.\d+)?b$", re.IGNORECASE)


def _is_size_token(token: str) -> bool:
    """True iff `token` is a GENUINELY-DISCLOSED scale token from an open-weight
    family name: a bare param count (`7b`, `70b`, `405b`, `1.5b`), a MoE
    expert spec (`8x7b`, `8x22b`), or a MoE active-param spec (`a16b`).

    This is the ONLY route to a `size` edge from models.dev (which carries no
    params field). A branded tier token is NEVER a size token (guarded by
    classification order in `_classify_token`)."""
    t = token.lower()
    return bool(_SIZE_TOKEN_RE.match(t) or _MOE_EXPERT_RE.match(t) or _MOE_ACTIVE_RE.match(t))


def _classify_token(token: str) -> tuple[str, str | None] | None:
    t = token.lower()
    # Branded tiers and named tokens take priority over the size regex so a
    # tier is never mis-read as a scale claim.
    if t in _TOKEN_CLASSIFICATIONS:
        return _TOKEN_CLASSIFICATIONS[t]
    if _VERSION_RE.match(t) or _DATE_8_RE.match(t) or _DATE_4_RE.match(t) or _DATE_3_RE.match(t):
        return ("variant", "version")
    # Disclosed scale: open-weight size token / MoE spec. Tier tokens already
    # returned above, so a `size` axis here always reflects a real scale.
    if _is_size_token(t):
        return ("variant", "size")
    return None


def _classify_suffix_segments(suffix: str) -> list[tuple[str, str | None, str]]:
    """Greedy left-to-right parse of a hyphen/dot-separated suffix.

    Returns a list of (relationship, axis, token) segments. When a single
    token can't be classified, falls back to a single (variant, version, suffix)
    segment so we always emit at least one parent edge — consumer can refine
    via curated core.yaml entries that override on collision.
    """
    if not suffix:
        return []
    # Normalize separators within the suffix for token splitting (mirrors
    # what _slugify does to ids — but apply also to dot for v0.1 → v0-1).
    norm = re.sub(r"[._]+", "-", suffix.lower()).strip("-")
    if not norm:
        return []
    tokens = norm.split("-")
    segments: list[tuple[str, str | None, str]] = []
    i = 0
    while i < len(tokens):
        # YYYY-MM-DD across 3 tokens
        if i + 3 <= len(tokens):
            window = "-".join(tokens[i:i + 3])
            if re.match(r"^\d{4}-\d{2}-\d{2}$", window):
                segments.append(("variant", "version", window))
                i += 3
                continue
        # vN-N across 2 tokens (slugified v0.3 → v0-3)
        if i + 2 <= len(tokens):
            window = "-".join(tokens[i:i + 2])
            if re.match(r"^v\d+-\d+$", window) or re.match(r"^\d{4}-\d{2}$", window):
                segments.append(("variant", "version", window))
                i += 2
                continue
        cls = _classify_token(tokens[i])
        if cls is None:
            # Unknown token mid-suffix — bail out and emit the rest as a
            # single version segment so at least the outer parent edge is set.
            tail = "-".join(tokens[i:])
            segments.append(("variant", "version", tail))
            return segments
        relationship, axis = cls
        segments.append((relationship, axis, tokens[i]))
        i += 1
    return segments


def _build_family_entries(
    org_id: str,
    family_slug: str,
    models: list[dict],
    group_recs: list[dict] | None = None,
    alias_index: dict[str, str] | None = None,
) -> list[dict]:
    """Emit canonical entries for a family.

    `group_recs` (the full underlying group's provider records) enables the
    G1 OpenRouter id adoption over the emitted entries; None (legacy callers)
    skips adoption.

    Returns a list:
      [0]   family root canonical (parents=[])
      [1..] one child per snapshot/variant whose slugified id != family_slug,
            each with a typed `parents` edge. Compound suffixes (e.g.
            `mistral-7b-instruct-v0-3`) materialize their intermediate
            canonicals so models.dev output matches the post-promotion
            shape of core.yaml — this matters because the seed loader's
            parents-merge is union-by-id, so disagreement on the parent
            id between source and core would produce a spurious second edge.
    """
    family_canonical_id = f"{org_id}/{family_slug}"

    # ---- Family root display_name + aggregated metadata ----
    # Prefer the lab's preferred name (vendor casing like `GPT-4o`); fall
    # back to our humanizer when models.dev didn't supply a name.
    display_name = ""
    for m in models:
        if _slugify(m.get("id", "")) == family_slug:
            display_name = m.get("name") or humanize_model_slug(family_slug)
            break
    if not display_name:
        display_name = humanize_model_slug(family_slug)

    open_weights = any(m.get("open_weights") for m in models)
    release_dates = sorted({m["release_date"] for m in models if m.get("release_date")})
    release_date = release_dates[0] if release_dates else None
    snapshot_ids = sorted({_slugify(m["id"]) for m in models if m.get("id")})

    # Aggregate modalities across all snapshots in the family (union).
    # Per-snapshot modalities still flow through to leaf children below;
    # the family root surfaces the superset so any snapshot's modality is
    # represented at the parent identity.
    family_input_modalities: set[str] = set()
    family_output_modalities: set[str] = set()
    for m in models:
        mods = m.get("modalities") or {}
        for v in (mods.get("input") or []):
            if isinstance(v, str) and v.strip():
                family_input_modalities.add(v.strip())
        for v in (mods.get("output") or []):
            if isinstance(v, str) and v.strip():
                family_output_modalities.add(v.strip())
    family_input_modalities_list = sorted(family_input_modalities) or None
    family_output_modalities_list = sorted(family_output_modalities) or None

    metadata: dict = {"snapshots": snapshot_ids}
    if release_dates:
        metadata["release_dates"] = release_dates
    if any(m.get("knowledge") for m in models):
        metadata["knowledge_cutoffs"] = sorted({
            m["knowledge"] for m in models if m.get("knowledge")
        })

    # Family-root aliases — surface forms of the family slug only. Snapshot
    # ids no longer go on the root; they're emitted as separate canonicals.
    root_aliases = [f"{org_id}/{family_slug}"] if f"{org_id}/{family_slug}" != family_canonical_id else []

    family_root_entry = {
        "id": family_canonical_id,
        "display_name": display_name,
        "org_id": org_id,
        "family": family_slug,
        "architecture": None,
        "params_billions": None,
        "parents": [],
        "open_weights": open_weights,
        "release_date": release_date,
        "input_modalities": family_input_modalities_list,
        "output_modalities": family_output_modalities_list,
        "tags": ["open-weight"] if open_weights else [],
        "aliases": root_aliases,
        "metadata": json.dumps(metadata, sort_keys=True),
        "review_status": "reviewed",
    }

    # ---- Child entries: one per snapshot/variant whose id != family_slug ----
    out_entries: list[dict] = [family_root_entry]
    seen_ids: dict[str, dict] = {family_canonical_id: family_root_entry}

    # `family_slug` may carry dots from the lab's display name
    # (`qwen2.5-7b`); slugified ids use dashes (`qwen2-5-7b-instruct`).
    # Compare on the dashed form, but build chain canonical ids from the
    # dotted form so children inherit the lab's preferred spelling.
    family_slug_dashed = re.sub(r"\.", "-", family_slug)

    for m in models:
        snap_dashed = _slugify(m.get("id", ""))
        if not snap_dashed:
            continue
        # If the model's id already matches the family slug (in either form),
        # it IS the family root — no child entry needed.
        if snap_dashed == family_slug_dashed or snap_dashed == family_slug:
            continue
        # Snapshot ids that don't share the family-slug prefix are unusual
        # (mirror entries, etc.) — skip rather than emit a malformed child.
        if not snap_dashed.startswith(family_slug_dashed + "-"):
            continue
        suffix = snap_dashed[len(family_slug_dashed) + 1:]
        segments = _classify_suffix_segments(suffix)
        if not segments:
            continue

        # Walk the chain, materializing intermediates as anchor entries.
        current_id = family_canonical_id
        for idx, (relationship, axis, token) in enumerate(segments):
            new_id = f"{current_id}-{token}"
            is_leaf = (idx == len(segments) - 1)
            parent_edge = {"id": current_id, "relationship": relationship}
            if axis:
                parent_edge["axis"] = axis

            if new_id in seen_ids:
                # Intermediate already emitted — walk through.
                current_id = new_id
                continue

            # Build the entry. Leaf gets the source model's metadata + the
            # original models.dev id as an alias (so dashed-form raw values
            # like `qwen2-5-7b-instruct` resolve via exact match even when
            # the canonical uses the dotted spelling). Intermediates are
            # anchor-only — humanized name, no release_date, no aliases.
            child_aliases: list[str] = []
            child_release: str | None = None
            child_open_weights = open_weights
            child_input_modalities: list[str] | None = None
            child_output_modalities: list[str] | None = None
            if is_leaf:
                child_aliases = sorted({snap_dashed, f"{org_id}/{snap_dashed}"})
                if m.get("release_date"):
                    child_release = m["release_date"]
                child_open_weights = bool(m.get("open_weights")) or open_weights
                # Per-snapshot modalities — narrower than the family aggregate.
                mods = m.get("modalities") or {}
                _ci = sorted({v.strip() for v in (mods.get("input") or []) if isinstance(v, str) and v.strip()})
                _co = sorted({v.strip() for v in (mods.get("output") or []) if isinstance(v, str) and v.strip()})
                child_input_modalities = _ci or None
                child_output_modalities = _co or None

            entry = {
                "id": new_id,
                "display_name": (m.get("name") or humanize_model_slug(new_id)) if is_leaf else humanize_model_slug(new_id),
                "org_id": org_id,
                "family": family_slug,
                "architecture": None,
                "params_billions": None,
                "parents": [parent_edge],
                "open_weights": child_open_weights,
                "release_date": child_release,
                "input_modalities": child_input_modalities,
                "output_modalities": child_output_modalities,
                "tags": ["open-weight"] if child_open_weights else [],
                "aliases": child_aliases,
                "metadata": "{}",
                "review_status": "reviewed",
            }
            seen_ids[new_id] = entry
            out_entries.append(entry)
            current_id = new_id

    # G1: adopt eligible OpenRouter keys over invented family ids (per-entry
    # identity match — a variant/dated key can only rename the entity it
    # names, never the family root). Children of a renamed root keep their
    # invented ids; their parent edges are repointed inside the adoption pass.
    if group_recs:
        _adopt_openrouter_ids(
            out_entries, group_recs, org_id, alias_index or _dev_alias_index()
        )
    return out_entries


def _provider_alias_forms(
    raw: str, org_id: str | None, provider: str | None = None
) -> list[str]:
    """Surface forms a provider's raw spelling should resolve through.

    Emits clean, resolvable forms only:
      - the raw spelling AS-IS when it carries no host/account scaffolding
        (no leading `@cf/`, no `accounts/...`, no embedded slash) — a provider's
        own bare spelling is worth an exact alias;
      - for the OpenRouter provider, the raw `org/slug` catalog key VERBATIM
        (G2: OpenRouter ids are resolvable everywhere, not only where they are
        canonical) plus its tag-stripped cleaned form — router pseudo-endpoints
        (`openrouter/*`) excluded;
      - the models.dev-normalized form (host/region/account scaffolding stripped),
        which is what the resolver sees post-host-capture;
      - the org-prefixed form of the normalized slug (last path segment), so both
        bare and `org/`-prefixed spellings resolve.

    Gnarly multi-segment host-scaffolded raws (`@cf/qwen/qwen3-30b-a3b-fp8`,
    `workers-ai/@cf/...`) are NOT emitted verbatim — only their normalized form
    is, so the alias list stays clean and free of double-prefix ids."""
    forms: set[str] = set()
    if not raw:
        return []
    # Keep the raw spelling only when it's a clean single-token id.
    if "/" not in raw and not raw.startswith("@") and not raw.startswith("~"):
        forms.add(raw)
        slug_raw = _slugify(raw)
        if slug_raw:
            forms.add(slug_raw)
    # A `TEE/`-prefixed raw (nano-gpt's trusted-execution serving namespace) is
    # a clean resolvable surface form: keep it verbatim on the stripped target.
    elif _TEE_PREFIX_RE.match(raw):
        forms.add(raw)
    # OpenRouter's `org/slug` catalog keys are real external ids: emit the raw
    # key verbatim (incl. a `~`/tagged spelling) AND the cleaned tag-stripped
    # key, so both resolve wherever the entity lives (canonical or alias).
    if (
        provider == _OPENROUTER_PROVIDER
        and "/" in raw
        and not _is_router_pseudo_endpoint(raw)
    ):
        forms.add(raw)
        cleaned = _clean_openrouter_key(raw)
        if cleaned:
            forms.add(cleaned)
    # Always emit the normalized form + its org-prefixed variant. Use
    # strip_variants=False: alias forms must PRESERVE a variant's identity
    # (-turbo/-thinking/-reasoner/-fp8/...). Stripping them here would emit the
    # BASE model's id as one of the variant's aliases (e.g. `gpt-4-turbo` -> alias
    # `gpt-4`), which steals the base canonical's id and aborts the seed.
    norm = normalize_modelsdev_id(raw, strip_variants=False)
    slug = _slugify(norm)
    if slug:
        leaf = slug.rsplit("/", 1)[-1]
        forms.add(leaf)
        if org_id:
            forms.add(f"{org_id}/{leaf}")
    return sorted(f for f in forms if f and f.count("/") <= 1)


def _variant_identity(s: str) -> str:
    """Variant-PRESERVING identity of a spelling (serving tags stripped, but
    size/mode/quant/training-stage/version variants KEPT), leaf only. Two
    spellings with the same identity are the same canonical; a different identity
    means a different model. Shared by _attach_provider_aliases AND the reconcile
    merge so neither attaches an `-instruct`/`-fp8`/`-120b` form onto a base/sibling."""
    return _slugify(normalize_modelsdev_id(s, strip_variants=False)).rsplit("/", 1)[-1]


# ---------------------------------------------------------------------------
# OpenRouter id adoption (specs/model-id-resolution/PLAN.md G1/G2).
# The id ladder: a real HF id is adopted first, then an OpenRouter catalog key,
# and only a model on neither gets an invented `{org}/{slug}` id. `org_id`
# stays the curated developer for all three rungs.
# ---------------------------------------------------------------------------
_OPENROUTER_PROVIDER = "openrouter"

# Curated re-host repoint oracle: `junk` ids are spellings curation has ruled
# must NEVER be canonical (the model's real identity is the `target`, e.g. a
# real HF repo the frozen oracle predates — `prime-intellect/intellect-3` ->
# `PrimeIntellect/INTELLECT-3`). Adoption must respect that ruling: an
# OpenRouter key that equals a junk id is not adopted (the reconcile passes
# still fold the mint onto the real target as before).
_REHOST_REPOINT_JSON = REPO_ROOT / "curation" / "rehost_repoint.json"
_REHOST_JUNK_IDS: frozenset[str] | None = None


def _rehost_junk_ids() -> frozenset[str]:
    global _REHOST_JUNK_IDS
    if _REHOST_JUNK_IDS is None:
        junk: set[str] = set()
        if _REHOST_REPOINT_JSON.exists():
            for e in json.loads(_REHOST_REPOINT_JSON.read_text()) or []:
                j = e.get("junk") if isinstance(e, dict) else None
                if isinstance(j, str) and j:
                    junk.add(j)
        _REHOST_JUNK_IDS = frozenset(junk)
    return _REHOST_JUNK_IDS


def _is_router_pseudo_endpoint(raw: str) -> bool:
    """`openrouter/*` keys (`auto`, `free`, `bodybuilder`, `owl-alpha`,
    `pareto-code`) are routing PRODUCTS, not models: never ADOPTED as a
    canonical id (`_clean_openrouter_key` returns None) and never emitted as
    an `openrouter/*` alias form (`_provider_alias_forms` skips them). The
    BARE raws behind them stay minted, though: other providers (e.g. kilo)
    list `auto`/`bodybuilder`/… directly, those are real EEE surface forms,
    and the nothing-is-removed floor keeps them resolvable as bare org-less
    mints. Gate 4 (tests/test_gate_invariants.py) pins the bare-id set so
    growth in that class is visible."""
    return raw.lstrip("~").lower().startswith("openrouter/")


def _clean_openrouter_key(raw: str) -> str | None:
    """The adoptable form of an OpenRouter catalog key: the `~` latest marker
    and `_TAG_SUFFIX_RE` serving tags stripped, `org/leaf` shape enforced — a
    key is never adopted with a tag in it. Returns None when the key is not
    adoptable at all: a router pseudo-endpoint, a tag-only leaf, or a moving
    `…-latest` pointer (names no fixed release; stays an alias)."""
    key = raw.lstrip("~")
    if "/" not in key or _is_router_pseudo_endpoint(raw):
        return None
    org, _, leaf = key.partition("/")
    prev = None
    while prev != leaf:
        prev = leaf
        leaf = _TAG_SUFFIX_RE.sub("", leaf)
    if not org or not leaf or leaf.endswith("-latest"):
        return None
    return f"{org}/{leaf}"


def _identity_sig(s: str) -> str:
    """Order-insensitive, variant-preserving identity signature: `safe_sig`
    over `_variant_identity`. Two spellings with the same sig name the same
    canonical even when token ORDER differs (`claude-haiku-3` vs OpenRouter's
    `claude-3-haiku`); any added/removed token (`-instruct`, `-fp8`, `-think`,
    a date) is a different identity, so a variant key can never match a base."""
    return safe_sig(_variant_identity(s))


def _entity_identity_sigs(
    cid: str, org_id: str | None, alias_index: dict[str, str]
) -> set[str]:
    """Identity sigs an entry answers to: its own, plus the brand-stripped one
    when the leaf's LEADING token is a curated alias of the entry's dev org
    (`perplexity/perplexity-sonar-…` == OpenRouter's `perplexity/sonar-…`) —
    the same brand-strip rule `_candidate_name_norms` applies for HF defer."""
    ident = _variant_identity(cid)
    sigs = {safe_sig(ident)}
    if org_id:
        dev = alias_index.get(org_id.lower(), org_id)
        toks = ident.split("-")
        if len(toks) > 1 and alias_index.get(toks[0]) == dev:
            sigs.add(safe_sig("-".join(toks[1:])))
    return sigs


def _openrouter_org_agrees(
    key_org: str, org_id: str | None, alias_index: dict[str, str]
) -> bool:
    """Adoption org guard: the OpenRouter key's org prefix must fold to the
    SAME developer as the entry's org_id (curated fold, then case/separator
    collapse for the unregistered community tail). A key under a different
    developer's namespace (e.g. a `google/…` key on an `unsloth` re-upload
    group) must never re-attribute the entry via adoption."""
    if not org_id:
        return False
    from eval_entity_resolver.fold import _norm_org_key

    fold = lambda o: alias_index.get(o.lower(), o)  # noqa: E731
    a, b = fold(key_org), fold(org_id)
    return a == b or _norm_org_key(a) == _norm_org_key(b)


def _adopt_openrouter_ids(
    entries: list[dict],
    group_recs: list[dict],
    org_id: str | None,
    alias_index: dict[str, str],
) -> None:
    """G1 adoption, shared by BOTH generator paths (`_mint_or_defer_rehost`
    and `_build_family_entries`): an entry that would carry an INVENTED
    `{org}/{slug}` id, whose group has an eligible OpenRouter key naming the
    SAME variant-preserving identity, takes the cleaned OpenRouter key as its
    canonical id VERBATIM (prefix included); the invented id becomes an alias.

    Guards: HF-deferred entries never rename (rung 1 of the ladder); identity
    match (a variant/dated/version-only key aliases the entity it names, never
    the base); org agreement; tag/`~` cleanup; tie-break shortest tag-stripped
    key then alphabetical; `openrouter/*` router endpoints excluded."""
    cands = sorted(
        {
            k
            for r in group_recs
            if r.get("provider") == _OPENROUTER_PROVIDER
            for k in (_clean_openrouter_key(r.get("raw") or ""),)
            if k and k not in _rehost_junk_ids()
        },
        key=lambda k: (len(k), k),
    )
    if not cands:
        return
    taken = {e["id"] for e in entries if e.get("id")}
    renames: dict[str, str] = {}
    for e in entries:
        cid = e.get("id")
        if not cid or "/" not in cid:
            continue  # org-less mints keep their invented slug
        if cid in cands:
            # Already carries an OpenRouter spelling verbatim: no rename here.
            # The entry is still tagged metadata.openrouter_adopted — but by
            # `_tag_openrouter_key_ids` AFTER the intra-output reconcile, not
            # here: `_pick_winner` ranks the flag as "renamed to an external
            # key" (rung 2 beats an invented twin), and tagging an
            # equal-spelling coincidence at adoption time would flip a
            # reviewed author-family entry into its re-host shell twin.
            continue
        if _entry_meta(e).get("hf_deferred"):
            continue  # canonical is the real HF id — never demoted
        sigs = _entity_identity_sigs(cid, e.get("org_id"), alias_index)
        for key in cands:
            if key in taken:
                continue
            if _identity_sig(key) not in sigs:
                continue
            if not _openrouter_org_agrees(
                key.split("/", 1)[0], e.get("org_id"), alias_index
            ):
                continue
            renames[cid] = key
            taken.add(key)
            e["id"] = key
            e["aliases"] = sorted(set(e.get("aliases") or []) | {cid})
            meta = _entry_meta(e)
            meta["openrouter_adopted"] = True
            e["metadata"] = json.dumps(meta, sort_keys=True)
            break
    if renames:
        # Repoint intra-group parent edges (family children of a renamed root).
        for e in entries:
            for edge in e.get("parents") or []:
                if isinstance(edge, dict) and edge.get("id") in renames:
                    edge["id"] = renames[edge["id"]]


def _tag_openrouter_key_ids(entries: list[dict], api_json: dict) -> list[dict]:
    """Tag `metadata.openrouter_adopted` on every FULL entry whose canonical id
    IS a cleaned OpenRouter catalog key. `_adopt_openrouter_ids` tags renames at
    adoption time; a mint whose invented id already EQUALS the key carries the
    same external-id fact, and the duplicate-identity gate's "adoption-touched"
    scoping must not depend on that spelling coincidence. Runs AFTER
    `_generate_models` (so `_pick_winner`, which ranks the flag as a rung-2
    rename, is not perturbed) and before the write. HF-deferred entries stay
    untagged: their id is rung 1 (HF). In-place; returns `entries`."""
    or_keys = {
        k
        for m in ((api_json.get(_OPENROUTER_PROVIDER) or {}).get("models") or {})
        for k in (_clean_openrouter_key(m),)
        if k
    }
    for e in entries:
        cid = e.get("id")
        if not cid or "display_name" not in e or cid not in or_keys:
            continue
        meta = _entry_meta(e)
        if meta.get("hf_deferred") or meta.get("openrouter_adopted") is True:
            continue
        meta["openrouter_adopted"] = True
        e["metadata"] = json.dumps(meta, sort_keys=True)
    return entries


def _attach_provider_aliases(
    entries: list[dict],
    group_recs: list[dict],
    org_id: str | None,
) -> None:
    """Union every provider spelling in the underlying group onto the matching
    emitted entry as a provider-tagged alias.

    Each models.dev record is routed to the entry whose canonical/alias set
    already contains its slugified family/leaf spelling; the spelling is added
    with its `inference_platform` (from PROVIDER_TO_INFERENCE_PLATFORM). Tagged
    aliases are accumulated under entry['alias_platforms'] (a {alias->platform}
    map) which the writer flattens into the alias list while preserving the
    platform provenance in metadata. Plain aliases (no platform) still go on
    entry['aliases']."""
    # Index entries by every id/alias surface form -> entry.
    by_form: dict[str, dict] = {}
    for e in entries:
        by_form[e["id"]] = e
        for a in e.get("aliases", []):
            by_form.setdefault(a, e)
    # Fallback target: the family-root entry (parents == []), else the first.
    root = next((e for e in entries if not e.get("parents")), entries[0] if entries else None)

    _identity = _variant_identity  # module-level (shared with reconcile's merge)
    for r in group_recs:
        platform = PROVIDER_TO_INFERENCE_PLATFORM.get(r["provider"])
        raw = r["raw"]
        # Route via the NORMALIZED leaf slug (host/account scaffolding stripped)
        # so a host-prefixed mirror still lands on the right entry. Fall to the
        # raw form, the org-prefixed slug, then the family-root entry.
        norm_slug = _slugify(normalize_modelsdev_id(raw)).rsplit("/", 1)[-1]
        target = (
            by_form.get(raw)
            or by_form.get(norm_slug)
            or (by_form.get(f"{org_id}/{norm_slug}") if org_id else None)
            or root
        )
        if target is None:
            continue
        # IDENTITY GUARD: only attach a provider spelling to a target whose
        # variant-preserving identity MATCHES the raw's. A `-fp8`/`-120b`/`-instruct`
        # /`-v0.3` record routed (via tag/variant collapse) onto a base or sibling
        # target would otherwise contaminate it with the variant's id as an alias
        # and abort the seed (a models.dev mint shadowing a distinct canonical's id).
        # Serving tags are already stripped by _identity, so legit `:free`/`-maas`
        # /`-tee` spellings still match their target and attach. Compared via the
        # order-insensitive `safe_sig` (token multiset, numbers order-preserved):
        # an OpenRouter-adopted target (`claude-3-haiku`) still matches the other
        # providers' spellings of the same identity (`claude-haiku-3`), while any
        # added/removed variant token keeps mismatching.
        if safe_sig(_identity(raw)) != safe_sig(_identity(target["id"])):
            continue
        ap = target.setdefault("alias_platforms", {})
        for form in _provider_alias_forms(raw, org_id, r.get("provider")):
            if form == target["id"]:
                continue
            # Intra-group steal-guard: never attach a form that is already the
            # id/alias of a DIFFERENT entry in this family group — that would make
            # two canonicals claim the same alias and abort the seed (e.g. a
            # `-turbo` record's form landing on the base). Merge such forms onto
            # their rightful owner; never duplicate a claim.
            owner = by_form.get(form)
            if owner is not None and owner is not target:
                continue
            # Record the platform provenance; a form seen from multiple
            # providers keeps the first non-null platform.
            if form not in ap or (ap.get(form) is None and platform):
                ap[form] = platform


# ---------------------------------------------------------------------------
# Mint-decision rule. Before minting an off-HF {org}/{slug} canonical we
# ask: is this underlying group already a real HF repo? The authority is the
# frozen HF oracle (hf_model_id_resolution.json). We DEFER (no mint; the
# canonical IS the real HF id) only on a normalized-identity match CORROBORATED
# BY ORG AGREEMENT after the curated two-tier dev-org remap — never a loose
# name-only match across different developers. Default to MINT when unsure.
# ---------------------------------------------------------------------------

# The frozen HF oracle — in-repo at curation/ (CI), workspace-parent fallback (dev).
HF_ORACLE_JSON = resolve_oracle_path()

_HF_AUTHORITY: dict[str, dict[str, str]] | None = None

# --- shared org-aware fold inputs (used by reconcile_generated_against_existing) ---
_HF_TO_DEV: dict[str, str] | None = None


def _hf_to_dev() -> dict[str, str]:
    """HF-org-lowercase -> curated developer slug. The SINGLE shared curated map
    (eval_entity_resolver.fold.build_curated_org_map): `_ORG_ALIASES` UNION every
    curated org's id + hf_org + `aliases`. Same map every generator + the resolver
    use, so the org-aware fold here agrees with them."""
    global _HF_TO_DEV
    if _HF_TO_DEV is None:
        from eval_entity_resolver.fold import build_curated_org_map
        data = safe_load_yaml(ORGS_SEED_PATH.read_text()) if ORGS_SEED_PATH.exists() else []
        _HF_TO_DEV = build_curated_org_map(data or [])
    return _HF_TO_DEV


_ORACLE_FIXED_IDS_CACHE: frozenset[str] | None = None


def _oracle_fixed_ids() -> frozenset[str]:
    """Real HF repo ids from the frozen oracle (fixed_exact/near_miss), so the
    fold index recognises HF repos even before they're written to a source."""
    global _ORACLE_FIXED_IDS_CACHE
    if _ORACLE_FIXED_IDS_CACHE is None:
        ids: set[str] = set()
        if HF_ORACLE_JSON.exists():
            oracle = json.loads(HF_ORACLE_JSON.read_text()).get("resolutions", {})
            for v in oracle.values():
                if v.get("resolution_status") in ("fixed_exact", "fixed_near_miss"):
                    fx = v.get("fixed_hf_model_id")
                    if isinstance(fx, str) and "/" in fx:
                        ids.add(fx)
        _ORACLE_FIXED_IDS_CACHE = frozenset(ids)
    return _ORACLE_FIXED_IDS_CACHE


def _build_hf_authority(
    oracle_path: Path = HF_ORACLE_JSON,
    alias_index: dict[str, str] | None = None,
) -> dict[str, dict[str, str]]:
    """Build the "is this on HF" authority from the frozen oracle.

    Returns {dev_org: {normalized_name: real_hf_model_id}} over every oracle
    entry with resolution_status in {fixed_exact, fixed_near_miss} carrying a
    fixed_hf_model_id. The HF org of each repo is remapped through the SAME
    curated two-tier dev-org map used everywhere else (`_build_org_alias_index`
    / orgs.yaml hf_org + _ORG_ALIASES), so the bucket key is the developer org
    (`Qwen`->`alibaba`, `meta-llama`->`meta`, ...). The name is normalized via
    the resolver's `normalize` (case + all separators collapsed to a space)."""
    from eval_entity_resolver.normalization import normalize as _norm

    ai = alias_index if alias_index is not None else _dev_alias_index()
    out: dict[str, dict[str, str]] = defaultdict(dict)
    if not oracle_path.exists():
        return out
    oracle = json.loads(oracle_path.read_text()).get("resolutions", {})
    for _raw, meta in oracle.items():
        if meta.get("resolution_status") not in ("fixed_exact", "fixed_near_miss"):
            continue
        fixed = meta.get("fixed_hf_model_id")
        if not isinstance(fixed, str) or "/" not in fixed:
            continue
        hf_org, hf_name = fixed.split("/", 1)
        dev_org = ai.get(hf_org.lower(), hf_org.lower())
        out[dev_org].setdefault(_norm(hf_name), fixed)
    return out


def _hf_authority() -> dict[str, dict[str, str]]:
    global _HF_AUTHORITY
    if _HF_AUTHORITY is None:
        _HF_AUTHORITY = _build_hf_authority()
    return _HF_AUTHORITY


def _candidate_name_norms(
    spellings: list[str], dev_org: str | None, alias_index: dict[str, str]
) -> set[str]:
    """Normalized NAME forms a models.dev group's spellings should match HF on.

    For each spelling we take its leaf (post org/host strip), normalize it, AND
    — because a models.dev key can carry the developer's brand as a prefix
    (`qwen-qwq-32b` for `Qwen/QwQ-32B`) — also emit the form with a leading
    brand token stripped when that token is a curated alias of THIS group's dev
    org. This collapses `qwen-qwq-32b`->`qwq-32b` without a bespoke fuzzy
    matcher: it reuses the same org alias index used for org resolution."""
    from eval_entity_resolver.normalization import normalize as _norm

    norms: set[str] = set()
    for sp in spellings:
        if not sp:
            continue
        leaf = sp.rsplit("/", 1)[-1]
        n = _norm(leaf)
        if not n:
            continue
        norms.add(n)
        # Strip a leading brand token that is a curated alias of the dev org.
        toks = n.split(" ")
        if dev_org and len(toks) > 1:
            stripped = " ".join(toks[1:])
            if alias_index.get(toks[0]) == dev_org and stripped:
                norms.add(stripped)
    return norms


def _hf_defer_target(
    candidate_id: str,
    org_id: str | None,
    spellings: list[str],
    alias_index: dict[str, str],
    authority: dict[str, dict[str, str]] | None = None,
) -> str | None:
    """Decide DEFER vs MINT for a models.dev underlying group.

    Returns the real HF id to defer to when the group resolves to an HF repo
    with org agreement (after dev-org remap); returns None to MINT otherwise.

    Confident == normalized-identity match WITHIN THE SAME dev-org bucket. A
    group with no org, or whose normalized names match only under a DIFFERENT
    developer, always MINTS (no cross-developer false merges)."""
    if org_id is None:
        return None
    auth = authority if authority is not None else _hf_authority()
    bucket = auth.get(org_id)
    if not bucket:
        return None
    forms = list(spellings)
    if candidate_id:
        forms.append(candidate_id)
    # _candidate_name_norms returns a SET, so iterate it in a deterministic
    # total order — otherwise the chosen HF target depends on Python's
    # string-hash randomization (PYTHONHASHSEED) and the cron churns the file.
    # Prefer the SHORTEST norm (fewest tokens), alpha tiebreak: the shortest
    # norm is the group's base identity, so a stray decorated spelling (e.g. a
    # models.dev `name` of "Llama 3.2 1B Instruct" on a base `llama-3.2-1b`
    # record) never pulls a base group onto a variant repo.
    for name_norm in sorted(
        _candidate_name_norms(forms, org_id, alias_index),
        key=lambda n: (n.count(" "), len(n), n),
    ):
        hit = bucket.get(name_norm)
        if hit is not None:
            return hit
    return None


def _hf_deferred_entry(
    hf_id: str,
    org_id: str | None,
    head: dict,
    group_recs: list[dict],
    mint_id: str,
    display_name: str,
) -> dict:
    """Build an HF-deferred record: canonical id IS the real HF repo id, with
    models.dev metadata (providers / open_weights / release_date) merged on and
    the models.dev spellings (mint id + display name) added as aliases. Mirrors
    the hand-folds for Qwen/QwQ-32B and LiquidAI/LFM2-24B-A2B (no new mint)."""
    aliases = []
    for a in (mint_id, display_name):
        if a and a != hf_id and a not in aliases:
            aliases.append(a)
    return {
        "id": hf_id,
        "display_name": head["display_name"] or humanize_model_slug(hf_id.split("/", 1)[-1]),
        "org_id": org_id,
        "family": None,
        "architecture": None,
        "params_billions": None,
        "parents": [],
        "open_weights": head["open_weights"],
        "release_date": head["release_date"],
        "input_modalities": None,
        "output_modalities": None,
        "tags": ["open-weight"] if head["open_weights"] else [],
        "aliases": aliases,
        "metadata": json.dumps(
            {
                "underlying_key": head.get("root_key", hf_id),
                "providers": sorted({r["provider"] for r in group_recs}),
                "hf_deferred": True,
            },
            sort_keys=True,
        ),
        # The canonical is the HF id; this record only enriches it, so it must
        # never override the HF source's reviewed status — it's an enrichment.
        "review_status": "reviewed",
        "resolution_source": "models_dev",
    }


def _mint_rehost_entry(
    root_key: str,
    org_id: str | None,
    head: dict,
    group_recs: list[dict],
) -> list[dict]:
    """Mint a single canonical for a re-host-only / closed-API group that has
    no author-lab family tree. The canonical id is `{org}/{Model-Name}` when an
    org is known, else a bare slug (org-less,
    flagged for curation). Returns [entry] (provider aliases attached by the
    caller via _attach_provider_aliases)."""
    head_raw = head["head_spelling"] or root_key
    # Mint slug from the head spelling, but FIRST run it through the models.dev
    # normalizer so provider/host/account scaffolding (`@cf/`, `accounts/.../`,
    # `org/`) is stripped — otherwise a mirror spelling leaks slashes into the
    # canonical id. Fall back to the canon root key when normalization empties.
    slug = _slugify(normalize_modelsdev_id(head_raw)) or _slugify(root_key)
    # Defensive: never let a multi-segment spelling produce a 2-slash id.
    slug = slug.rsplit("/", 1)[-1]
    canonical_id = f"{org_id}/{slug}" if org_id else slug
    display_name = head["display_name"] or humanize_model_slug(slug)
    open_weights = head["open_weights"]
    release_date = head["release_date"]
    tags = ["open-weight"] if open_weights else []
    if org_id is None:
        tags = tags + ["org-unknown"]
    entry = {
        "id": canonical_id,
        "display_name": display_name,
        "org_id": org_id,
        "family": slug,
        "architecture": None,
        "params_billions": None,
        "parents": [],
        "open_weights": open_weights,
        "release_date": release_date,
        "input_modalities": None,
        "output_modalities": None,
        "tags": tags,
        "aliases": [],
        "metadata": json.dumps(
            {"underlying_key": root_key, "providers": sorted({r["provider"] for r in group_recs})},
            sort_keys=True,
        ),
        # Re-host-only / minted-from-models.dev groups are NOT author-confirmed.
        "review_status": "draft" if not head["has_author_lab_entry"] else "reviewed",
        "resolution_source": "models_dev",
    }
    return [entry]


def _mint_or_defer_rehost(
    root_key: str,
    org_id: str | None,
    head: dict,
    group_recs: list[dict],
    alias_index: dict[str, str],
) -> list[dict]:
    """Mint-decision wrapper for the re-host path: if the underlying group
    already resolves to a real HF repo (normalized-identity match with
    dev-org agreement against the frozen oracle), DEFER — emit an HF-deferred
    record keyed by the real HF id with the models.dev spellings as aliases.
    Otherwise MINT the off-HF {org}/{slug} canonical (via _mint_rehost_entry)."""
    # The prospective mint id + display, mirroring _mint_rehost_entry's slug.
    head_raw = head["head_spelling"] or root_key
    slug = (_slugify(normalize_modelsdev_id(head_raw)) or _slugify(root_key)).rsplit("/", 1)[-1]
    mint_id = f"{org_id}/{slug}" if org_id else slug
    display_name = head["display_name"] or humanize_model_slug(slug)

    # Candidate spellings to check against HF: the mint id, the head spelling,
    # every raw provider spelling, and the display name (org/host scaffolding is
    # stripped to the leaf inside _candidate_name_norms).
    spellings = [mint_id, head_raw, display_name, slug]
    spellings += [r["raw"] for r in group_recs if r.get("raw")]

    hf_id = _hf_defer_target(mint_id, org_id, spellings, alias_index)
    if hf_id is not None:
        head = {**head, "root_key": root_key}
        return [_hf_deferred_entry(hf_id, org_id, head, group_recs, mint_id, display_name)]
    entries = _mint_rehost_entry(root_key, org_id, head, group_recs)
    # Rung 2 of the id ladder: no HF defer -> adopt an eligible OpenRouter key
    # over the invented `{org}/{slug}` id (G1).
    _adopt_openrouter_ids(entries, group_recs, org_id, alias_index)
    return entries


def _generate_models(api_json: dict, known_org_ids: set[str]) -> tuple[list[dict], list[str]]:
    """Provider-preserving group -> mint -> alias over the FULL models.dev
    catalog. Every provider that maps to an inference_platform is
    processed (no author-only gate); each underlying group yields one canonical
    family (author-lab tree when the author lab is present, else a minted
    re-host canonical), and every provider spelling in the group is aliased in
    carrying its inference_platform.

    Returns (entries, skipped_no_org). A non-empty skipped_no_org is a hard
    error (an author-lab provider mapped to an org missing from seed/orgs.yaml).
    """
    out: list[dict] = []
    skipped_providers: list[str] = []
    skipped_no_org: list[str] = []
    alias_index = _build_org_alias_index()

    # 1. Dedup the whole catalog into underlying groups.
    groups = build_underlying_groups(api_json)

    for root_key, recs in sorted(groups.items()):
        # Drop records whose provider isn't a known inference_platform (none
        # today — all 137 map — but keep the guard for forward-compat).
        recs = [r for r in recs if r["provider"] in PROVIDER_TO_INFERENCE_PLATFORM]
        if not recs:
            # No record survived the platform filter; nothing to mint or track.
            continue

        head = pick_underlying(root_key, recs)
        hf_org = head["author_org"]
        org_id = normalize_org_slug(hf_org, alias_index)

        # If a provider in the curated PROVIDER_TO_ORG allowlist authored this
        # group, prefer ITS curated org id (validated against seed/orgs.yaml) over
        # the HF-style slug derived above (from AUTHOR_PROV_ORG / family name) —
        # the two can diverge (e.g. AUTHOR_PROV_ORG says `inceptionai`,
        # PROVIDER_TO_ORG says `inception`).
        curated_author_recs = [
            r for r in recs
            if r["provider"] in PROVIDER_TO_ORG
            and "/" not in r["raw"]
            and head["has_author_lab_entry"]
            # only treat as author when its curated org agrees with the family-org
            and (
                org_id is None
                or PROVIDER_TO_ORG[r["provider"]] == org_id
                or normalize_org_slug(AUTHOR_PROV_ORG.get(r["provider"]), alias_index) == org_id
            )
        ]
        if curated_author_recs:
            org_id = PROVIDER_TO_ORG[curated_author_recs[0]["provider"]]

        # Records belonging to the author lab (their provider's org matches the
        # group org) drive the family tree; the rest are re-host aliases.
        author_recs = curated_author_recs

        if head["has_author_lab_entry"] and author_recs and org_id:
            if org_id not in known_org_ids:
                skipped_no_org.append(
                    f"{author_recs[0]['provider']} -> {org_id} (group {root_key})"
                )
                continue
            # Build the author-lab family tree from the author records, grouped
            # by family slug (a group may span a couple of stage/size siblings).
            by_family: dict[str, list[dict]] = defaultdict(list)
            for r in author_recs:
                by_family[_family_for(r["record"])].append(r["record"])
            group_entries: list[dict] = []
            for family_slug, models in sorted(by_family.items()):
                if not family_slug:
                    continue
                group_entries.extend(_build_family_entries(
                    org_id, family_slug, models,
                    group_recs=recs, alias_index=alias_index,
                ))
            if not group_entries:
                group_entries = _mint_or_defer_rehost(root_key, org_id, head, recs, alias_index)
        else:
            # Re-host-only / closed-API group with no usable author tree: mint
            # UNLESS this group is already a real HF repo (defer instead).
            group_entries = _mint_or_defer_rehost(root_key, org_id, head, recs, alias_index)

        # 2. Alias every provider spelling in the group into the entries.
        _attach_provider_aliases(group_entries, recs, org_id)
        out.extend(group_entries)

    if skipped_providers:
        print(
            f"[refresh] skipped {len(skipped_providers)} provider records not in "
            f"PROVIDER_TO_INFERENCE_PLATFORM",
            file=sys.stderr,
        )
    # Dedup entries by id (a model that appears in two dedup groups, e.g. via
    # different snapshots, would otherwise emit twice). Merge aliases on collide.
    # Then reconcile within this output: merge same-model dups (org-aware) and
    # strip alias-steals so no emitted alias claims another canonical's id.
    return _reconcile_intra_output(_dedup_entries(out)), skipped_no_org


def _pick_winner(group: list[dict]) -> dict:
    """Pick the authoritative entry among same-model dups: prefer an HF-deferred
    real repo, then an OpenRouter-adopted external id (rung 2 of the id ladder —
    an adopted key must never lose to an invented twin), then an org-qualified
    id, then an HF-true-cased id (has uppercase — e.g. `Qwen/...`), then a
    reviewed entry, then the shortest id (drops doubled-brand mints like
    `cohere/cohere-command-a` in favour of `cohere/command-a`), then alphabetical
    for determinism."""
    def _meta_flag(e: dict, flag: str) -> bool:
        try:
            return json.loads(e.get("metadata") or "{}").get(flag) is True
        except (ValueError, TypeError):
            return False

    def key(e: dict):
        cid = e.get("id") or ""
        return (
            0 if _meta_flag(e, "hf_deferred") else 1,
            0 if _meta_flag(e, "openrouter_adopted") else 1,
            0 if "/" in cid else 1,            # org-qualified beats bare org-less
            0 if any(c.isupper() for c in cid) else 1,  # HF-true casing
            0 if e.get("review_status") == "reviewed" else 1,
            len(cid),
            cid,
        )

    return sorted(group, key=key)[0]


def _reconcile_intra_output(entries: list[dict]) -> list[dict]:
    """Reconcile dups/alias-steals WITHIN one generation's output (after id-dedup):

    1. MERGE same-model dups — two entries are the same model iff they share an
       org_id AND their org/brand-aware name norms intersect (reusing
       `_candidate_name_norms`, the same brand-strip the HF-defer uses). This
       collapses a dev-org-slug mint into the HF-deferred real repo
       (`alibaba/qwen3-32b` -> `Qwen/Qwen3-32B`), a doubled-brand mint into its
       clean sibling (`cohere/cohere-command-a` -> `cohere/command-a`), and a
       dot/case spelling twin (`moonshotai/kimi-k2.6` <-> `…/Kimi-K2.6`). The
       authoritative entry (`_pick_winner`) keeps id+casing; losers' aliases are
       merged on, their ids added as aliases, and parent edges repointed.
    2. STRIP alias-steals — a variant that wrongly carries a DIFFERENT model's id
       as an alias (`…-thinking` aliasing its base; a deferred base aliasing its
       `-it` variant) has that alias removed (the canonical id always wins). This
       is NOT a merge — the two are genuinely different models.

    The result: no emitted alias equals (exact or resolver-normalized) another
    canonical's id, so the seed never aborts on a double-claim."""
    from eval_entity_resolver.normalization import normalize as _norm

    alias_index = _dev_alias_index()

    # ---- pass 1: org-aware same-model union-find ----
    uf = UnionFind()

    for e in entries:
        uf.find(e["id"])

    # (a) Two canonicals with the SAME normalized id collide at seed time no
    # matter what, so they MUST merge (regardless of org) — catches a bare
    # org-less mint vs its org-qualified twin (`xiaomi-mimo-v2-5` vs
    # `xiaomi/mimo-v2-5`) and dot/case spelling variants.
    norm_id_bucket: dict[str, list[str]] = defaultdict(list)
    for e in entries:
        norm_id_bucket[_norm(e["id"])].append(e["id"])
    for ids in norm_id_bucket.values():
        for other in ids[1:]:
            uf.union(other, ids[0])

    # (b) Org-aware same-model union: same org_id AND intersecting org/brand-aware
    # name norms (reusing the HF-defer brand-strip). Merges a dev-org-slug mint
    # into the HF-deferred real repo and a doubled-brand mint into its sibling.
    bucket: dict[tuple, list[str]] = defaultdict(list)
    for e in entries:
        org = e.get("org_id")
        if not org:
            continue  # org-less tail only auto-merges via the normalized-id rule above
        for nm in _candidate_name_norms([e["id"]], org, alias_index):
            bucket[(org, nm)].append(e["id"])
    for ids in bucket.values():
        for other in ids[1:]:
            uf.union(other, ids[0])

    groups: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        groups[uf.find(e["id"])].append(e)

    merged: list[dict] = []
    loser_to_winner: dict[str, str] = {}
    for grp in groups.values():
        if len(grp) == 1:
            merged.append(grp[0])
            continue
        winner = _pick_winner(grp)
        walias = set(winner.get("aliases") or [])
        wap = winner.setdefault("alias_platforms", {})
        # Scalar fields to fill from a loser when the winner's value is empty
        # (the winner is often an hf_deferred entry with family/params/modalities
        # None and parents []). Mirrors _dedup_entries / loader _merge_into so a
        # merge is identity-correct WITHOUT discarding the richer loser's metadata.
        _PREFER_NONEMPTY = (
            "params_billions", "family", "architecture",
            "input_modalities", "output_modalities",
        )
        wparents = {edge.get("id"): edge for edge in (winner.get("parents") or []) if isinstance(edge, dict)}
        for e in grp:
            if e is winner:
                continue
            loser_to_winner[e["id"]] = winner["id"]
            walias.update(e.get("aliases") or [])
            if e["id"] != winner["id"]:
                walias.add(e["id"])
            for k, v in (e.get("alias_platforms") or {}).items():
                if k not in wap or (wap.get(k) is None and v):
                    wap[k] = v
            winner["open_weights"] = bool(winner.get("open_weights")) or bool(e.get("open_weights"))
            if e.get("review_status") == "reviewed":
                winner["review_status"] = "reviewed"
            # Prefer-non-empty scalar fill from the loser.
            for f in _PREFER_NONEMPTY:
                if not winner.get(f) and e.get(f):
                    winner[f] = e[f]
            # Earliest release_date wins (matches _dedup_entries).
            rds = [d for d in (winner.get("release_date"), e.get("release_date")) if d]
            if rds:
                winner["release_date"] = min(rds)
            # Union parent edges by id (don't drop the loser's lineage).
            for edge in (e.get("parents") or []):
                if isinstance(edge, dict) and edge.get("id") not in wparents:
                    wparents[edge["id"]] = edge
        if wparents:
            winner["parents"] = list(wparents.values())
        winner["aliases"] = sorted(walias)
        merged.append(winner)

    # Repoint parent edges that referenced a merged-away loser.
    for e in merged:
        for edge in (e.get("parents") or []):
            if isinstance(edge, dict) and edge.get("id") in loser_to_winner:
                edge["id"] = loser_to_winner[edge["id"]]

    # ---- pass 2: strip alias-steals (alias == a DIFFERENT canonical's id) ----
    # Clean BOTH `aliases` and the working `alias_platforms` map (the latter is
    # flattened back into aliases by _finalize_entries, so a stolen form left
    # there would reappear).
    ids = {e["id"] for e in merged}
    id_norm: dict[str, str] = {}
    for cid in ids:
        id_norm.setdefault(_norm(cid), cid)

    def _steals_id(form: str, cid: str) -> bool:
        if not form or form == cid:
            return False
        if form in ids:  # exact: another canonical's id
            return True
        owner = id_norm.get(_norm(form))
        return owner is not None and owner != cid  # normalized: another canonical's id

    for e in merged:
        cid = e["id"]
        e["aliases"] = sorted({a for a in (e.get("aliases") or []) if not _steals_id(a, cid)})
        ap = e.get("alias_platforms")
        if isinstance(ap, dict):
            e["alias_platforms"] = {k: v for k, v in ap.items() if not _steals_id(k, cid)}

    # ---- pass 3: ambiguous claims (alias OR display_name claimed by 2+ DISTINCT
    # surviving canonicals) abort the seed (the loader promotes display_name to a
    # global alias). Keep each contested form on ONE canonical — the "natural"
    # owner whose id-name normalizes to it, else the first by id — and drop it
    # from the others (cross-org bare names like `gemma-3-1b-it`, shared display
    # names like `Claude Opus 4.5`).
    claimers: dict[str, set[str]] = defaultdict(set)
    for e in merged:
        cid = e["id"]
        for a in e.get("aliases") or []:
            claimers[a].add(cid)
        for k in (e.get("alias_platforms") or {}):
            claimers[k].add(cid)
        dn = e.get("display_name")
        if isinstance(dn, str) and dn:
            claimers[dn].add(cid)

    keeper: dict[str, str] = {}
    for form, owners in claimers.items():
        if len(owners) < 2:
            continue
        natural = sorted(c for c in owners if _norm(c.split("/", 1)[-1]) == _norm(form))
        keeper[form] = (natural or sorted(owners))[0]

    if keeper:
        for e in merged:
            cid = e["id"]
            e["aliases"] = sorted(
                a for a in (e.get("aliases") or []) if a not in keeper or keeper[a] == cid
            )
            ap = e.get("alias_platforms")
            if isinstance(ap, dict):
                e["alias_platforms"] = {k: v for k, v in ap.items() if k not in keeper or keeper[k] == cid}
            dn = e.get("display_name")
            if isinstance(dn, str) and dn in keeper and keeper[dn] != cid:
                # Re-derive a non-colliding display_name from the id tail.
                e["display_name"] = cid.split("/", 1)[-1]

    return sorted(merged, key=lambda e: e["id"])


def _dedup_entries(entries: list[dict]) -> list[dict]:
    """Collapse entries that share a canonical id (can happen when two
    underlying groups mint the same family root). Union aliases / tags /
    alias_platforms; prefer reviewed over draft; keep first non-null scalars."""
    by_id: dict[str, dict] = {}
    for e in entries:
        cur = by_id.get(e["id"])
        if cur is None:
            by_id[e["id"]] = e
            continue
        # Union list/dict fields.
        cur["aliases"] = sorted(set(cur.get("aliases", [])) | set(e.get("aliases", [])))
        cur["tags"] = sorted(set(cur.get("tags", [])) | set(e.get("tags", [])))
        ap = cur.setdefault("alias_platforms", {})
        for k, v in (e.get("alias_platforms") or {}).items():
            if k not in ap or (ap.get(k) is None and v):
                ap[k] = v
        # open_weights any(True); reviewed wins.
        cur["open_weights"] = bool(cur.get("open_weights")) or bool(e.get("open_weights"))
        if e.get("review_status") == "reviewed":
            cur["review_status"] = "reviewed"
        # Keep earliest release_date.
        rd = [d for d in (cur.get("release_date"), e.get("release_date")) if d]
        cur["release_date"] = min(rd) if rd else None
    return sorted(by_id.values(), key=lambda e: e["id"])


def _load_known_org_ids() -> set[str]:
    if not ORGS_SEED_PATH.exists():
        return set()
    data = safe_load_yaml(ORGS_SEED_PATH.read_text()) or []
    return {e["id"] for e in data if "id" in e}


_HEADER = """# Generated from models.dev (https://models.dev) — DO NOT EDIT BY HAND.
# To update: edit seed/models/core.yaml (curated canonicals win at load
# time), then run `python scripts/refresh_from_modelsdev.py` to regenerate
# this file.
#
# Source: https://models.dev/api.json (MIT, (c) 2025 models.dev)
# Last refresh date is in git history — see
# `git log -1 -- seed/models/sources/models_dev.generated.yaml`.
#
# This file is one data source among potentially several under
# `seed/models/sources/`. It contains pure models.dev output — no curated
# overlays. The seed CLI loader merges sources → core → enrichments at
# load time (field-level merge: aliases / tags UNION).
#
# Each entry collapses all snapshots / dated variants of a model family
# into one canonical id (`<org>/<family-slug>`). The resolver's fuzzy stem
# step (in eval_entity_resolver/strategies/fuzzy.py) strips date suffixes,
# thinking budgets, hosting provider tags, etc., so per-snapshot raw IDs
# resolve to this family canonical without needing per-snapshot entries.
#
# `aliases` lists snapshot IDs we observed in models.dev for this family.
# Enrich records ({id, aliases} onto an existing canonical) may carry a
# `weak:` scalar map donated from the suppressed/folded mint; the seed
# loader applies those only to fields every full entry left empty and
# core.yaml does not explicitly carry.
"""


def _finalize_entries(entries: list[dict]) -> list[dict]:
    """Flatten the in-progress `alias_platforms` map into the persisted shape:
    union its keys into `aliases`, and fold the {alias -> inference_platform}
    provenance into metadata['alias_platforms'] so the loader can wire the
    platform FK per alias. Drops the working `alias_platforms` key. Returns a
    new list; does not mutate inputs in place beyond the working key."""
    finalized: list[dict] = []
    for e in entries:
        ap = e.pop("alias_platforms", None) or {}
        aliases = set(e.get("aliases", []))
        aliases.update(ap.keys())
        # Remove self-id from aliases.
        aliases.discard(e["id"])
        e["aliases"] = sorted(aliases)
        if ap:
            meta = json.loads(e.get("metadata") or "{}")
            # Only non-null platform tags carry FK provenance.
            meta["alias_platforms"] = {k: v for k, v in sorted(ap.items()) if v}
            e["metadata"] = json.dumps(meta, sort_keys=True)
        finalized.append(e)
    return finalized


def _write_yaml(entries: list[dict], path: Path) -> str:
    body = yaml.safe_dump(_finalize_entries(entries), sort_keys=False, allow_unicode=True, width=200)
    return _HEADER + "\n" + body


# ---------------------------------------------------------------------------
# Full-catalog split/dedup + org de-orphan. The daily cron regenerates
# models_dev_catalog.generated.yaml directly so it never goes stale, and HF
# source-of-truth wins every collision: an HF-present model becomes an
# ALIAS-ONLY enrichment onto the existing HF-cased canonical (no lowercase twin
# minted); only genuinely models.dev-only (not-on-HF) models are minted fresh.
# ---------------------------------------------------------------------------

CATALOG_OUT_PATH = REPO_ROOT / "seed" / "models" / "sources" / "models_dev_catalog.generated.yaml"
HF_ORACLE_PATH = REPO_ROOT / "seed" / "models" / "sources" / "hf_oracle.generated.yaml"
HUB_STATS_PATH = REPO_ROOT / "seed" / "models" / "sources" / "hub_stats.generated.yaml"
TIER3_PATH = REPO_ROOT / "seed" / "models" / "sources" / "tier3_inferred.generated.yaml"
CORE_PATH = REPO_ROOT / "seed" / "models" / "core.yaml"
ENRICH_ALIASES_PATH = REPO_ROOT / "seed" / "models" / "enrichments" / "aliases.yaml"
ORGS_GENERATED_PATH = REPO_ROOT / "seed" / "orgs.generated.yaml"

# All EXISTING model sources whose id+alias surface forms the catalog must not
# clash with. HF/curated WIN id+casing. models_dev.generated.yaml (the
# re-cased pure source) is included so the catalog stays purely additive.
_CATALOG_EXISTING_SOURCES = (
    HF_ORACLE_PATH,
    SEED_PATH,
    HUB_STATS_PATH,
    TIER3_PATH,
    CORE_PATH,
    ENRICH_ALIASES_PATH,
)

# Existing-sources set the NON-catalog (full re-cased) write path reconciles
# against. SAME set as the catalog path EXCEPT SEED_PATH itself is dropped —
# the non-catalog write REWRITES SEED_PATH, so it must not dedup against its own
# (stale) prior output. CORE_PATH IS included, which is the whole point: a fresh
# re-cased mint whose normalized form collides with a curated core canonical
# under a DIFFERENT id is suppressed/repointed instead of clobbering core.
_NONCATALOG_EXISTING_SOURCES = (
    HF_ORACLE_PATH,
    HUB_STATS_PATH,
    TIER3_PATH,
    CORE_PATH,
    ENRICH_ALIASES_PATH,
)


def _load_core_skip_ids() -> set[str]:
    """Ids core.yaml suppresses (`skip_ids` + `skip_source_ids`). A committed
    entry with one of these ids is NOT carried forward — curation explicitly
    removed it from the source universe."""
    if not CORE_PATH.exists():
        return set()
    doc = safe_load_yaml(CORE_PATH.read_text())
    if not isinstance(doc, dict):
        return set()
    return set(doc.get("skip_ids") or []) | set(doc.get("skip_source_ids") or [])


def _entry_meta(e: dict) -> dict:
    """Parse an entry's metadata JSON into a dict. Malformed COMMITTED
    metadata degrades to {} (with a warning) instead of crashing the cron."""
    try:
        meta = json.loads(e.get("metadata") or "{}")
    except (ValueError, TypeError):
        print(f"[refresh] WARN: malformed metadata on {e.get('id')!r}; treating as empty", file=sys.stderr)
        return {}
    return meta if isinstance(meta, dict) else {}


def _entry_alias_platforms(e: dict) -> dict:
    """The {alias -> inference_platform} provenance map from a persisted
    entry's metadata JSON ({} when absent/malformed)."""
    ap = _entry_meta(e).get("alias_platforms")
    return ap if isinstance(ap, dict) else {}


def _donor_scalars(e: dict) -> dict:
    """Non-empty weak-scalar fields a suppressed/folded/skipped mint donates
    onto its owner's enrich record (under `weak:`; the seed loader applies them
    only to fields every full entry left empty and core.yaml does not
    explicitly carry). Reads a full entry's top-level fields or an enrich
    record's existing `weak:` map. `parents` edges (dangling-edge risk) and
    mint-specific metadata (`providers`, `underlying_key`) are deliberately
    NOT donated."""
    weak = e.get("weak")
    weak = weak if isinstance(weak, dict) else {}
    out: dict = {}
    for f in WEAK_SCALAR_FIELDS:
        v = e.get(f)
        if v in (None, "", [], {}):
            v = weak.get(f)
        if v not in (None, "", [], {}):
            out[f] = v
    return out


def _merge_weak(rec: dict, scalars: dict) -> None:
    """Merge donated scalars into rec['weak'] (existing keys win), keeping the
    canonical WEAK_SCALAR_FIELDS key order so output stays byte-deterministic."""
    if not scalars:
        return
    weak = rec.get("weak")
    weak = dict(weak) if isinstance(weak, dict) else {}
    for k, v in scalars.items():
        weak.setdefault(k, v)
    rec["weak"] = {k: weak[k] for k in WEAK_SCALAR_FIELDS if k in weak}


def _make_form_owner_lookup(sources: tuple[Path, ...]) -> Callable[[str], str | None]:
    """form -> owning canonical id over `sources` (exact, then resolver-
    normalized) — the same claims/alias index the reconcile passes use."""
    from eval_entity_resolver.normalization import normalize as _norm

    exact, normed = _build_existing_index(sources)

    def _owner(form: str) -> str | None:
        return exact.get(form) or normed.get(_norm(form))

    return _owner


def _union_alias_platforms(entry: dict, donor_ap: dict) -> None:
    """Union `donor_ap` into entry's metadata.alias_platforms for forms the
    entry carries as aliases, so a committed alias that the carry-forward
    re-added keeps its platform-FK provenance. Existing keys win."""
    aliases = set(entry.get("aliases") or [])
    add = {k: v for k, v in donor_ap.items() if v and k in aliases}
    if not add:
        return
    meta = _entry_meta(entry)
    ap = meta.get("alias_platforms") or {}
    meta["alias_platforms"] = dict(sorted({**add, **ap}.items()))
    entry["metadata"] = json.dumps(meta, sort_keys=True)


def _twin_key(cid: str) -> str:
    """Org-aware seed-collision key for a model id: the org prefix folded
    through the curated dev-org map, the name part keyed exactly like the
    seed loader's collision fold (`collision_key`). A committed id and a
    fresh respelling of it (`google/veo-3-1` vs `google/veo3-1`) share a key,
    so the carry-forward can absorb the reappearance instead of emitting
    normalized twins for the seed's collision_fold / gate to trip on."""
    if "/" in cid:
        org, name = cid.split("/", 1)
        dev = _hf_to_dev().get(org.lower())
        if dev:
            return f"{dev}/{collision_key(name)}"
    return collision_key(cid)


def _version_marker_sig(name: str) -> str:
    """Sorted `vN`-marker tokens of a RAW leaf (split on separators BEFORE any
    tag/Bedrock strip). `nova-premier-v1:0` carries `v1`; the bare
    `nova-premier` carries none — the two must never twin (a `-v1`/Bedrock
    version key keeps separate identity per the adoption plan)."""
    toks = re.split(r"[-_.:/\s]+", name.lower())
    return "+".join(sorted(t for t in toks if re.fullmatch(r"v\d+", t)))


def _extended_twin_keys(cid: str) -> set[str]:
    """Order/brand-insensitive FALLBACK twin keys for the respellings
    `_twin_key`'s ordered `collision_key` cannot match — the OpenRouter
    adoption class (`anthropic/claude-haiku-3` vs adopted
    `anthropic/claude-3-haiku`, `sao10K/l3-8b-lunaris` vs
    `sao10k/l3-lunaris-8b`, `perplexity/perplexity-sonar-…` vs
    `perplexity/sonar-…`). Keys are `dev-org/identity-sig#vN-markers`
    (identity sig = variant-preserving token multiset; the vN-marker sig keeps
    a Bedrock `-v1:0` spelling from twinning its base), so a
    variant/size/version difference still never twins. Used only after a
    `_twin_key` miss, with the same `_bsizes` guard at the call sites."""
    if "/" not in cid:
        return {f"{_identity_sig(cid)}#{_version_marker_sig(cid)}"}
    org, name = cid.split("/", 1)
    dev = _hf_to_dev().get(org.lower()) or org.lower()
    vsig = _version_marker_sig(name)
    keys = {f"{dev}/{_identity_sig(name)}#{vsig}"}
    toks = _variant_identity(name).split("-")
    if len(toks) > 1 and _hf_to_dev().get(toks[0]) == dev:
        keys.add(f"{dev}/{safe_sig('-'.join(toks[1:]))}#{vsig}")
    return keys


def _pick_twin_candidate(cid: str, candidates: list[dict]) -> dict | None:
    """Pick a stable carry-forward twin without conflating dotted releases.

    ``_twin_key`` deliberately collapses separators so a harmless respelling
    such as ``veo-3-1`` -> ``veo3-1`` retains its canonical id.  That broad key
    can also contain distinct model spellings: ``qwen3.8`` and ``qwen-3-8``
    both fold to ``qwen38`` but are not the same identity under the registry's
    variant-preserving identity rules.  Prefer an exact identity match whenever
    there is one.  Preserve the historical separator-only fallback only when
    the collision bucket has exactly one size-compatible candidate.

    This mirrors the gate's identity rule while retaining the stability rule
    for unambiguous letter/digit respellings that ``_identity_sig`` intentionally
    keeps conservative.
    """
    from eval_card_registry.lib.collision_fold import _bsizes

    size_matches = [e for e in candidates if _bsizes(cid) == _bsizes(e["id"])]
    if not size_matches:
        return None

    def exact_identity(e: dict) -> bool:
        return (
            _identity_sig(cid) == _identity_sig(e["id"])
            and _version_marker_sig(cid) == _version_marker_sig(e["id"])
        )

    identity_matches = [e for e in size_matches if exact_identity(e)]
    if identity_matches:
        return sorted(identity_matches, key=lambda e: e["id"])[0]
    if len(size_matches) == 1:
        return size_matches[0]
    return None


def _carry_forward_committed(
    fresh: list[dict], committed: list[dict], adopt_migration: bool = False
) -> tuple[list[dict], dict[str, str]]:
    """Merge the COMMITTED generated file into a fresh wholesale rewrite so an
    upstream (provider,model) removal never deletes a resolvable surface form:

      * surviving id (still emitted today): union the committed aliases back
        in — an alias-level upstream removal must not regress resolution —
        keeping platform provenance for re-added forms. The committed
        display_name counts too (the loader promotes it to a global alias):
        when upstream re-spells it, the old one is unioned back as an alias;
      * respelled reappearance / twin (STABILITY RULE): a removed id whose
        org-aware twin key matches a FRESH mint (upstream re-emitted the model
        under a new spelling, `google/veo-3-1` -> `google/veo3-1`, or an
        OpenRouter listing change renamed the adopted key) is unified with
        that mint. The COMMITTED id wins: the fresh entry is renamed to the
        committed id and today's spelling becomes an alias — so the daily cron
        can never thrash canonical ids on upstream respells/relistings.
        EXCEPT with `adopt_migration=True` (the one-shot
        `--adopt-openrouter-ids-migration` run): the FRESH (externally-
        adopted) id wins and the committed id + display_name + aliases become
        aliases on it — never retained as a normalized twin that the seed's
        collision_fold / gate would trip on;
      * removed id: retain the committed entry verbatim (full entries tagged
        `metadata.upstream_status: removed`) UNLESS core.yaml skips it.
        Retained entries then run through the same core-aware reconciliation
        as fresh mints, so an entry that has since been curated/folded is
        suppressed rather than resurrected;
      * core-skipped id whose surface form an existing canonical claims (e.g.
        core aliasing the skipped mint): its scalars still flow as a weak
        enrich record onto that owner — only the aliases stay suppressed
        (that is what skip_source_ids curates away). Unclaimed: pure skip;
      * returns (batch, committed_claims): every committed id/alias mapped to
        its surviving owner. reconcile_generated_against_existing pre-seeds
        its form-hygiene `claimed` map with these, so a NEW mint can never
        steal a carried-forward entry's alias claims.
    """
    by_id = {e["id"]: e for e in fresh if e.get("id")}
    by_twin: dict[str, list[dict]] = defaultdict(list)
    by_twin_ext: dict[str, list[dict]] = defaultdict(list)
    by_form: dict[str, dict] = {}
    for e in sorted(fresh, key=lambda x: x.get("id") or ""):
        if e.get("id"):
            by_twin[_twin_key(e["id"])].append(e)
            for k in _extended_twin_keys(e["id"]):
                by_twin_ext[k].append(e)
            for a in (e.get("aliases") or []):
                if a:
                    by_form.setdefault(a, e)
    skip = _load_core_skip_ids()
    claims: dict[str, str] = {}
    out = list(fresh)
    retained = 0
    absorbed = 0
    stabilized: dict[str, str] = {}  # fresh-spelling id -> committed id it took
    promoted: dict[str, str] = {}    # committed id absorbed -> fresh id that outranked it
    pending_rung2 = 0                # OpenRouter-key twins deferred by the stability rule
    skip_owner: Callable[[str], str | None] | None = None
    for c in committed:
        cid = c.get("id")
        if not cid:
            continue
        cur = by_id.get(cid)
        dn = c.get("display_name")
        # Respelled-reappearance twin: FULL committed entries only (an enrich
        # record's id is ANOTHER canonical — never donatable as an alias), and
        # only when the b-size signature agrees (same guard as fold_collisions:
        # opt-1.3b vs opt-13b key-collide but are different models). The
        # order/brand-insensitive `_extended_twin_keys` fallback catches the
        # OpenRouter-adoption respellings the ordered collision key cannot.
        twin = None
        twin_via_form = False
        if cur is None and cid not in skip and "display_name" in c:
            t = _pick_twin_candidate(cid, by_twin.get(_twin_key(cid), []))
            if t is None:
                t = _pick_twin_candidate(
                    cid,
                    [
                        e
                        for k in sorted(_extended_twin_keys(cid))
                        for e in by_twin_ext.get(k, [])
                    ],
                )
            if t is None and "/" in cid and cid.split("/", 1)[0].lower() in _SERVING_HOSTS:
                # A committed mint under a serving-host prefix (the pre-strip
                # `TEE/...` uploader-org bug): the fresh batch carries that id
                # as an alias on the stripped target — absorb onto it (a
                # malformed serving-prefixed id never wins the stability rule).
                t = by_form.get(cid)
                twin_via_form = t is not None
            twin = t
        # STABILITY RULE, RUNG-MONOTONE: on a twin match the COMMITTED id wins
        # among EQUAL rungs of the id ladder (the fresh entry is renamed
        # below), so the cron never thrashes ids on an upstream
        # respell/relisting. A fresh twin from a STRICTLY HIGHER rung
        # overrides it: an hf_deferred twin (a REAL HF repo id, rung 1) is
        # never renamed back to a committed non-HF id — that would demote the
        # HF id to an alias, violating the adoption path's own "canonical is
        # the real HF id — never demoted" rule. The committed id + forms
        # become aliases on the HF id instead (the absorb branch below), with
        # parent-edge repointing like the rename path's. Rung 2 over 3
        # (a fresh OpenRouter-key twin over a committed invented id) stays
        # gated behind the one-shot migration flag per PLAN G3; the deferred
        # debt is counted and logged so the daily cron shows it. The one-shot
        # migration flag inverts the rule wholesale (the fresh
        # externally-adopted id wins). A twin that already took an earlier
        # committed id this run is never renamed twice — later committed
        # twins absorb onto it instead.
        promote_twin = (
            twin is not None
            and not twin_via_form
            and "display_name" in twin
            and _entry_meta(twin).get("hf_deferred") is True
            and _entry_meta(c).get("hf_deferred") is not True
        )
        rename_twin = (
            twin is not None
            and not adopt_migration
            and not promote_twin
            and not twin_via_form
            and twin["id"] not in stabilized
            and "display_name" in twin
        )
        if (
            twin is not None
            and not adopt_migration
            and not promote_twin
            and _entry_meta(twin).get("openrouter_adopted") is True
            and _entry_meta(c).get("openrouter_adopted") is not True
            and _entry_meta(c).get("hf_deferred") is not True
        ):
            # Counted for EVERY deferred-debt shape — the plain rename_twin
            # case, a twin already stabilized onto an earlier committed id,
            # and a form-level absorb — each one defers a rung-2 promotion.
            pending_rung2 += 1
        owner = (
            cur["id"] if cur is not None
            else (cid if rename_twin else (twin["id"] if twin is not None else cid))
        )
        claims.setdefault(cid, owner)
        for a in (c.get("aliases") or []):
            if a:
                claims.setdefault(a, owner)
        if isinstance(dn, str) and dn:
            claims.setdefault(dn, owner)
        if cur is not None:
            back = set(c.get("aliases") or []) - {cid}
            if isinstance(dn, str) and dn and dn not in (cid, cur.get("display_name")):
                back.add(dn)
            if back:
                cur["aliases"] = sorted(set(cur.get("aliases") or []) | back)
                _union_alias_platforms(cur, _entry_alias_platforms(c))
            continue
        if cid in skip:
            scalars = _donor_scalars(c)
            if scalars:
                if skip_owner is None:
                    skip_owner = _make_form_owner_lookup(_NONCATALOG_EXISTING_SOURCES)
                fold_owner = skip_owner(cid)
                if fold_owner is not None and fold_owner != cid:
                    out.append({"id": fold_owner, "weak": scalars})
            continue
        if twin is not None:
            if rename_twin:
                fresh_spelling = twin["id"]
                twin["id"] = cid
                stabilized[cid] = fresh_spelling
                by_id[cid] = twin
                back = (set(c.get("aliases") or []) | {fresh_spelling}) - {cid}
            else:
                # The committed id lost the twin match (migration run, a
                # higher-rung fresh id, or an already-stabilized twin): it
                # becomes an alias below, so any parent edge naming it must
                # be repointed onto the surviving fresh id (mirror of the
                # `stabilized` repoint for the opposite direction).
                promoted[cid] = twin["id"]
                back = (set(c.get("aliases") or []) | {cid}) - {twin["id"]}
            if isinstance(dn, str) and dn and dn not in (twin["id"], twin.get("display_name")):
                back.add(dn)
            twin["aliases"] = sorted(set(twin.get("aliases") or []) | back)
            _union_alias_platforms(twin, _entry_alias_platforms(c))
            absorbed += 1
            continue
        rc = dict(c)
        if "display_name" in rc:  # full entry, not an alias-only enrich record
            meta = _entry_meta(rc)
            meta["upstream_status"] = "removed"
            rc["metadata"] = json.dumps(meta, sort_keys=True)
            retained += 1  # enrich records are carried too but not counted here
        out.append(rc)
    if stabilized or promoted:
        # Repoint parent edges that referenced a fresh spelling the stability
        # rule renamed back to its committed id — and, symmetrically, edges
        # that referenced a committed id absorbed onto a surviving fresh id.
        renamed = {old: new for new, old in stabilized.items()}
        renamed.update(promoted)
        # Path-compress the union: a promote-then-rename chain (committed
        # invented id -> fresh twin -> committed HF id) must compose to the
        # FINAL survivor — a one-hop repoint would land edges and claims on
        # a renamed-away id (a dangling canonical reference).
        for old in list(renamed):
            seen = {old}
            target = renamed[old]
            while target in renamed and target not in seen:
                seen.add(target)
                target = renamed[target]
            renamed[old] = target
        for e in out:
            for edge in e.get("parents") or []:
                if isinstance(edge, dict) and edge.get("id") in renamed:
                    edge["id"] = renamed[edge["id"]]
        # Claims values must name SURVIVING ids — a renamed-away value would
        # poison the committed_claims escape downstream in reconcile.
        for k, v in list(claims.items()):
            if v in renamed:
                claims[k] = renamed[v]
        if promoted:
            # Diagnosability: a rung-1 promotion on a plain cron run demotes
            # a committed id. If a non-regenerated source (hub_stats/tier3/
            # curated parents) still references the demoted id, the dedup
            # gate fails closed and the day's commit aborts — this line
            # names the ids so the wedge is diagnosable from the cron log.
            print(
                "[refresh] carry-forward: promoted higher-rung fresh id(s): "
                + ", ".join(
                    f"{old} -> {renamed[old]}" for old in sorted(promoted)
                ),
                file=sys.stderr,
            )
    if stabilized:
        print(
            f"[refresh] carry-forward: kept {len(stabilized)} committed id(s) "
            f"over a respelled fresh twin (stability rule)",
            file=sys.stderr,
        )
    if not adopt_migration:
        # Owner-visible debt line (always printed on a plain cron run): how
        # many rung-2 promotions — a fresh OpenRouter-key twin outranking a
        # committed invented id — the stability rule deferred. Promoting them
        # is a DELIBERATE `--adopt-openrouter-ids-migration` run, never
        # automatic (PLAN G3).
        print(
            f"[refresh] stability rule: {pending_rung2} pending rung-2 id "
            f"promotion(s) (OpenRouter key over invented id) awaiting the next "
            f"--adopt-openrouter-ids-migration run",
            file=sys.stderr,
        )
    if retained:
        print(
            f"[refresh] carry-forward: retained {retained} committed entr(ies) "
            f"absent from today's upstream",
            file=sys.stderr,
        )
    if absorbed:
        print(
            f"[refresh] carry-forward: absorbed {absorbed} respelled committed "
            f"id(s) onto today's {'fresh spelling' if adopt_migration else 'surviving entry'}",
            file=sys.stderr,
        )
    return out, claims


def _iter_unskipped_source_entries(
    sources: tuple[Path, ...],
) -> Iterable[tuple[Path, dict]]:
    """Yield active source records, excluding generated ids suppressed by core.

    ``skip_source_ids`` removes a generated definition from the merged seed
    universe. Treating that stale record as an owner during reconciliation can
    create an enrichment keyed by an entity that the loader will immediately
    discard, making the refresh path-dependent. Curated definitions in
    ``core.yaml`` itself remain authoritative even when they override a source
    id with the same spelling.
    """
    skipped = _load_core_skip_ids()
    for path in sources:
        for entry in _catalog_load_list(path):
            cid = entry.get("id") if isinstance(entry, dict) else None
            if path != CORE_PATH and cid in skipped:
                continue
            yield path, entry


def _build_existing_index(
    sources: tuple[Path, ...],
) -> tuple[dict[str, str], dict[str, str]]:
    """Build (existing_exact, existing_norm) maps {form -> owning canonical id}
    over every id / display_name / alias surface form in `sources`.

    `existing_exact` keys on the raw form; `existing_norm` keys on the resolver's
    `normalize` (case + all separators collapsed) — the same form-matching the
    catalog path and the seed validator's normalized_match steal-guard use.
    First writer wins on a duplicate form (setdefault), mirroring the catalog
    path's accumulation order."""
    from eval_entity_resolver.normalization import normalize as _norm

    existing_exact: dict[str, str] = {}
    existing_norm: dict[str, str] = {}

    def _add_exact(form: str, cid: str) -> None:
        if form:
            existing_exact.setdefault(form, cid)
            existing_norm.setdefault(_norm(form), cid)

    for _path, e in _iter_unskipped_source_entries(sources):
        cid = e.get("id")
        if not cid:
            continue
        _add_exact(cid, cid)
        dn = e.get("display_name")
        if dn:
            _add_exact(dn, cid)
        for a in (e.get("aliases") or []):
            _add_exact(a, cid)
    return existing_exact, existing_norm


def _make_steal_guard(
    existing_exact: dict[str, str], existing_norm: dict[str, str]
) -> Callable[[str, str], bool]:
    """Return a `_steals(form, cid)` predicate: True iff `form` (exact OR
    normalized) is already owned by a DIFFERENT canonical id. Identical
    semantics to regenerate_catalog's inner `_steals`."""
    from eval_entity_resolver.normalization import normalize as _norm

    def _steals(form: str, cid: str) -> bool:
        owner = existing_exact.get(form)
        if owner is not None and owner != cid:
            return True
        nowner = existing_norm.get(_norm(form))
        return nowner is not None and nowner != cid

    return _steals


def reconcile_generated_against_existing(
    entries: list[dict],
    sources: tuple[Path, ...] = _NONCATALOG_EXISTING_SOURCES,
    committed_claims: dict[str, str] | None = None,
) -> list[dict]:
    """Core-aware reconciliation for the NON-catalog write path — a true MERGE,
    not a drop: on an alias collision, move aliases onto the owner, drop the
    source row, and rewrite parent edges to point at the owner.

    For every minted entry whose id (exact OR normalized) collides with an
    EXISTING canonical (from `sources`, incl. CORE_PATH) under a DIFFERENT id:
      1. drop the colliding mint (the existing curated/HF id wins id+casing);
      2. MERGE: emit an enrich record {id: <existing owner>, aliases: [the mint
         id + its non-stealing aliases]} so those surface forms resolve to the
         owner (the seed loader unions aliases by id across sources) — mirrors
         the catalog path's `_enrich_target`. Aliases are NOT silently dropped;
      3. rewrite every SURVIVING entry's `parents[].id` that pointed at a dropped
         mint to the owner instead, so no surviving lineage edge dangles.

    Reuses the steal-guard machinery (`_build_existing_index`/`_make_steal_guard`).
    `sources` is injectable for unit testing. `committed_claims` (from
    `_carry_forward_committed`) maps every previously-committed id/alias to its
    committed owner; the form-hygiene pass below treats those claims as already
    taken, so a NEW mint can never steal a carried-forward entry's alias.
    Returns a new id-sorted list of survivors + enrich records.
    """
    from eval_entity_resolver.fold import build_hf_index, decide_fold
    from eval_entity_resolver.normalization import normalize as _norm

    existing_exact, existing_norm = _build_existing_index(sources)
    _steals = _make_steal_guard(existing_exact, existing_norm)

    # Ids DEFINED (not merely enriched) by an existing source: only a record
    # with a display_name defines a canonical. An alias-only enrich record
    # (e.g. an enrichments/aliases.yaml bridge keyed by the id) must not make
    # the re-emit branch below suppress the batch's own DEFINITION of that id
    # — that would leave the canonical defined nowhere (a bare row).
    defining_ids: set[str] = set()
    for _path, e in _iter_unskipped_source_entries(sources):
        if e.get("id") and "display_name" in e:
            defining_ids.add(e["id"])
    # Valid suppression OWNERS additionally include ids defined by the derived
    # broad sources and by this batch itself — but NEVER an id that exists
    # only as an enrich-record key (suppressing a definition onto a
    # non-defining id would materialize a bare canonical).
    owner_defining: set[str] = set(defining_ids)
    for _path, e in _iter_unskipped_source_entries(
        (CATALOG_OUT_PATH, TIER3_PATH)
    ):
        if e.get("id") and "display_name" in e:
            owner_defining.add(e["id"])
    for e in entries:
        if isinstance(e, dict) and e.get("id") and "display_name" in e:
            owner_defining.add(e["id"])

    def _owner_of(form: str) -> str | None:
        """The existing canonical id that owns `form` (exact then normalized)."""
        return existing_exact.get(form) or existing_norm.get(_norm(form))

    # Broader owner index for the MERGE-donation + survivor HYGIENE (NOT the
    # partition steal-guard, which stays narrow): a form may be owned by an
    # established canonical in the models_dev_catalog / tier3 sources too. Those
    # are excluded from `sources` so the partition does not dedup against
    # derived/own output, but they ARE authoritative for "this form already
    # belongs to X" — so a base spelling like `meta/llama-3-1-70b` (owned by the
    # base in the catalog) is not donated onto / kept on the `…-70B-Instruct`
    # canonical.
    _hy_exact, _hy_norm = _build_existing_index(sources + (CATALOG_OUT_PATH, TIER3_PATH))

    def _owner_broad(form: str) -> str | None:
        return _hy_exact.get(form) or _hy_norm.get(_norm(form))

    # Defined-owner lookup: like _owner_broad, but only claims made by records
    # whose id is itself DEFINED anywhere count. Used to repoint an enrich
    # record keyed by a renamed-away id onto the entity that now carries that
    # id as an alias (a dead record's self-claim must not keep it alive).
    _def_exact: dict[str, str] = {}
    _def_norm: dict[str, str] = {}
    for _path, e2 in _iter_unskipped_source_entries(
        sources + (CATALOG_OUT_PATH, TIER3_PATH)
    ):
        cid2 = e2.get("id")
        if not cid2 or cid2 not in owner_defining:
            continue
        for form2 in [cid2, e2.get("display_name"), *(e2.get("aliases") or [])]:
            if form2:
                _def_exact.setdefault(form2, cid2)
                _def_norm.setdefault(_norm(form2), cid2)
    for e2 in entries:
        cid2 = e2.get("id") if isinstance(e2, dict) else None
        if not cid2 or "display_name" not in e2:
            continue
        for form2 in [cid2, e2.get("display_name"), *(e2.get("aliases") or [])]:
            if form2:
                _def_exact.setdefault(form2, cid2)
                _def_norm.setdefault(_norm(form2), cid2)

    def _defined_owner_of(form: str) -> str | None:
        return _def_exact.get(form) or _def_norm.get(_norm(form))

    # ORG-AWARE fold index over the existing sources (+ frozen oracle HF ids): a
    # mint that refers to the SAME model as a real HF repo under a DIFFERENT id
    # (the org-decoupled dev-org-slug case the plain-normalize steal-guard misses,
    # e.g. `alibaba/qwen-2-5-14b-instruct` -> `Qwen/Qwen2.5-14B-Instruct`) must
    # DEFER to the HF id. Uses eval_entity_resolver.fold.decide_fold so this path
    # and the resolver agree on what counts as the same model (no drift).
    existing_entries = [
        entry for _path, entry in _iter_unskipped_source_entries(sources)
    ]
    hf_to_dev = _hf_to_dev()
    _hf_ids, _alias_to_hf, _by_org_name, _ = build_hf_index(
        existing_entries, hf_to_dev, _oracle_fixed_ids()
    )

    def _fold_target(e: dict) -> str | None:
        f = decide_fold(e, _hf_ids, _alias_to_hf, _by_org_name, hf_to_dev)
        if f is None:
            return None
        tgt = f["hf_target"]
        return tgt if tgt and tgt != e.get("id") else None

    # First pass: partition into survivors vs suppressed. A mint is suppressed
    # when its id steals an existing canonical (plain-normalized, different id) OR
    # it org-aware-folds to a real HF repo. Record each suppressed id -> owner.
    survivors: list[dict] = []
    suppressed: list[dict] = []
    suppressed_owner: dict[str, str] = {}
    for e in entries:
        cid = e.get("id")
        owner: str | None = None
        if cid and _steals(cid, cid):
            o = _owner_of(cid)
            if o and o != cid and o in owner_defining:
                owner = o
        # A mint whose id EXACTLY equals an existing canonical id IS that canonical
        # (a re-emit, e.g. an HF-present model models.dev also serves). MERGE its
        # forms onto that existing canonical via an enrich record — do NOT keep it
        # as a models_dev-source SURVIVOR (that would be a redundant duplicate of
        # the HF canonical, and any contaminating variant spelling it carries would
        # then read as a models_dev mint shadowing a real HF id). And NEVER run
        # decide_fold on it: decide_fold's fuzzy tier strips variant markers
        # (-instruct/-it/…) and would drag a distinct variant that is itself a real
        # canonical onto the base (e.g. Llama-3.2-1B-Instruct -> the base ...-3.2-1B).
        if owner is None and cid and cid in defining_ids:
            suppressed.append(e)
            suppressed_owner[cid] = cid   # owner == self: enrich onto the existing id
            continue
        if owner is None and cid:
            owner = _fold_target(e)  # org-aware same-model fold
        if owner and owner != cid:
            suppressed.append(e)
            suppressed_owner[cid] = owner
            continue
        # A carried-forward enrich record (no display_name) keyed by an id no
        # source DEFINES would otherwise survive as a bare canonical (no
        # display_name/org) forever. Repoint it onto the DEFINED entity that
        # carries the id as an alias (the post-adoption owner) when one
        # exists; enriching nothing at all is meaningless — drop LOUDLY, so a
        # real resolution loss fails the oracle gates instead of hiding
        # behind a silent bare canonical.
        if cid and "display_name" not in e and cid not in owner_defining:
            ob = _defined_owner_of(cid)
            if ob is not None and ob != cid:
                suppressed.append(e)
                suppressed_owner[cid] = ob
                continue
            print(
                f"[refresh] WARNING: dropping orphaned enrich record {cid!r} "
                f"(owner vanished from every source); forms dropped: "
                f"{[cid, *(e.get('aliases') or [])]}",
                file=sys.stderr,
            )
            continue
        survivors.append(e)

    # NOTE: do NOT early-return when `suppressed` is empty — the surviving-mint
    # cross-source hygiene below must run regardless (a survivor can carry a
    # foreign alias/display_name even when no mint id collides).

    # MERGE: accumulate enrich records keyed by owner. The dropped mint id plus
    # its non-stealing aliases (forms not owned by yet another canonical) are
    # carried onto the owner so resolvable spellings survive; its non-empty
    # scalars are carried as WEAK values (under `weak:`) so the suppressed
    # mint's metadata isn't dropped on the floor either. Donors iterate in id
    # order so the first donor wins per scalar field, deterministically.
    sibling_ids = {e["id"] for e in survivors}
    _committed = committed_claims or {}
    enrich_aliases: dict[str, set[str]] = defaultdict(set)
    enrich_scalars: dict[str, dict] = defaultdict(dict)
    for e in sorted(suppressed, key=lambda x: x["id"]):
        cid = e["id"]
        owner = suppressed_owner[cid]
        # cid is ITSELF a real canonical owned by a DIFFERENT id: a distinct
        # model an org-aware fold wrongly dragged here (e.g. the base repo
        # `google/gemma-2-9b` folded onto the variant `…-9b-it` via a mangled
        # models.dev key alias) — donating its id OR its scalars would attach
        # one model's identity/metadata to another.
        #
        # Ownership evidence is TIERED. The INDEPENDENT sources (`_owner_of`:
        # hf_oracle / hub_stats / core / enrichments) are authoritative — a
        # claim there vetoes the donation, with the one standing exception of
        # a stale claim by a non-defined id the carry-forward absorbed onto
        # `owner` (recorded in committed_claims). The DERIVED files (catalog /
        # tier3 — the extra layers in `_owner_broad`) are NOT authoritative
        # about the suppressed mint itself: a derived record keyed by cid, or
        # by an id committed_claims assigns to this owner, is the PREVIOUS
        # cron's echo of the very mint being suppressed/absorbed (the catalog
        # is rewritten right after this step). Treating that echo as a
        # foreign owner blocks the donation, and the subsequent catalog
        # rewrite then drops its own orphaned record — deleting the surface
        # forms from every source (the `deepseek-chat-v3-1` regression
        # class). A derived claim by any OTHER id still vetoes: if tier3
        # genuinely defines the mint id as its own canonical, the donation
        # goes ahead and the seed's collision fold / gate suite fails LOUDLY
        # on a wrong merge rather than silently dropping forms.
        ind_cid = _owner_of(cid)
        cid_owner = _owner_broad(cid)
        foreign_cid = (
            ind_cid is not None
            and ind_cid != owner
            and (ind_cid in owner_defining or _committed.get(ind_cid) != owner)
        ) or (
            cid_owner is not None
            and cid_owner not in (owner, cid)
            and _committed.get(cid_owner) != owner
        )
        if not _steals(cid, owner) or cid != owner:
            # The dropped mint id resolves to the owner: keep it as an alias —
            # UNLESS it normalized-equals the owner's own id form, OR it is a
            # foreign canonical (above). Donating it would double-claim the
            # form and merge two distinct canonicals. The real owner already
            # supplies it, so drop it. Same foreign-owner guard the loser's
            # OTHER aliases get (via `_owner_broad`) — applied symmetrically
            # to the loser id.
            if _norm(cid) != _norm(owner) and not foreign_cid:
                enrich_aliases[owner].add(cid)
        if not foreign_cid:
            for k, v in _donor_scalars(e).items():
                enrich_scalars[owner].setdefault(k, v)
        # Carry the dropped mint's display_name too (it was a resolvable form: a
        # raw value can NORMALIZED-match a canonical via its display_name, e.g.
        # `command-r+` -> the folded `cohere/command-r+` whose display was
        # `Command R+`). Without donating it, that bare-form resolvability is lost
        # when the mint folds onto the real HF id. Same foreign-owner guard.
        for a in [e.get("display_name"), *(e.get("aliases") or [])]:
            if not a or a == owner:
                continue
            # Drop an alias only if it is owned by a DIFFERENT canonical than the
            # owner we are merging onto (would double-claim); else carry it over.
            # A surviving batch mint's id counts as such an owner too — donating
            # it would alias-claim a sibling canonical's id. Likewise a form the
            # COMMITTED file assigns to a third entity (carry-forward keeps it
            # there; donating it here would double-claim). Same TIERED evidence
            # as `foreign_cid` above: an INDEPENDENT-source claim is
            # authoritative (with the absorbed-non-defined-id escape); a claim
            # that exists only in the DERIVED catalog/tier3 layer, made by the
            # suppressed mint itself (cid) or by an id committed_claims
            # assigns to this owner, is the previous cron's echo — its forms
            # transfer with the merge.
            other_ind = _owner_of(a)
            if other_ind is not None and other_ind != owner and (
                other_ind in owner_defining or _committed.get(other_ind) != owner
            ):
                continue
            other = _owner_broad(a)
            if other is not None and other not in (owner, cid) and (
                _committed.get(other) != owner
            ):
                continue
            if a in sibling_ids:
                continue
            co = _committed.get(a)
            if co is not None and co not in (owner, cid):
                continue
            enrich_aliases[owner].add(a)

    # Rewrite surviving parent edges that pointed at a dropped mint -> the owner,
    # so no surviving lineage edge dangles.
    for e in survivors:
        parents = e.get("parents")
        if not isinstance(parents, list):
            continue
        for edge in parents:
            if isinstance(edge, dict) and edge.get("id") in suppressed_owner:
                edge["id"] = suppressed_owner[edge["id"]]

    # SURVIVING-mint cross-source + INTRA-BATCH form hygiene: a survivor (its id
    # does NOT steal) can still carry an ALIAS / DISPLAY_NAME owned by a DIFFERENT
    # canonical — a double-claim the seed rejects as nondeterministic. The owner
    # can be either:
    #   * an EXISTING source canonical (cross-source) — e.g. models_dev
    #     `mistralai/mixtral-8x7b` keeping the bare alias `mixtral-8x7b` owned by
    #     hf_oracle's `mistralai/Mixtral-8x7B-Instruct-v0.1`; cross-org
    #     `unsloth/...` vs `google/...` sharing a bare name; OR
    #   * a SIBLING SURVIVOR in this same rewrite batch (intra-batch) — the
    #     non-catalog path REWRITES models_dev.generated.yaml wholesale, so a base
    #     mint (`meta-llama/Llama-3.1-70B`) aliasing a variant that is ALSO its own
    #     batch entry (`…-70B-Instruct`) double-claims it. The cross-source check
    #     alone misses this (neither is in the EXISTING sources).
    # Resolution (deterministic): an entry's own id always wins; a sibling's id
    # beats any alias (distinct canonicals — drop the alias, not a merge); a
    # non-id form shared by >1 survivor goes to the lexicographically-first
    # claimant. Mirrors the catalog path's fresh_form_owner ownership.
    # Non-id form -> owning claimant. Pre-seeded with the committed file's
    # claims (carry-forward), so a contested form stays with its committed
    # owner regardless of sort order — a NEW mint cannot steal it.
    claimed: dict[str, str] = dict(committed_claims or {})
    for e in sorted(survivors, key=lambda x: x["id"]):
        cid = e["id"]

        def _foreign(form: str) -> bool:
            # INDEPENDENT-source claim (hf_oracle / hub_stats / core /
            # enrichments): authoritative — a veto, unless the claiming id is
            # NOT itself a defined canonical and was just absorbed onto cid by
            # the carry-forward (a stale record in a not-yet-regenerated
            # source file). A DEFINED owner always keeps its forms.
            o_ind = _owner_of(form)
            if o_ind is not None and o_ind != cid and (
                o_ind in owner_defining or claimed.get(o_ind) != cid
            ):
                return True
            # Claim exists only in the DERIVED layer (catalog / tier3). When
            # the carry-forward assigned this form — or the claiming id
            # itself — to THIS survivor (committed_claims: a twin-absorbed
            # respelling like committed `google/veo3-1` absorbed onto fresh
            # `google/veo-3-1`, in either stability direction), the derived
            # claim is the PREVIOUS cron's echo of the absorbed id: the
            # catalog is rewritten right after this step, so stripping on its
            # strength deletes the absorbed forms from every source
            # (normalize does not equate letter-digit-boundary twins, so
            # `veo3-1`-class forms exist ONLY as explicit aliases).
            o = _owner_broad(form)
            if o is not None and o != cid and (
                claimed.get(o) != cid and claimed.get(form) != cid
            ):
                return True                       # owned by an established canonical
            if form in sibling_ids and form != cid:
                return True                       # is a different sibling's id
            c = claimed.get(form)
            return c is not None and c != cid     # already claimed by an earlier sibling

        kept: list[str] = []
        for a in (e.get("aliases") or []):
            if not a or a == cid or _foreign(a):
                continue
            claimed.setdefault(a, cid)
            kept.append(a)
        e["aliases"] = sorted(set(kept))
        ap = e.get("alias_platforms")
        if isinstance(ap, dict):
            e["alias_platforms"] = {k: v for k, v in ap.items() if k != cid and not _foreign(k)}
        # Post-finalize entries carry alias_platforms in METADATA (the working
        # key is gone). Keep that map in lockstep with the alias drop above —
        # a key whose alias form was just stripped as foreign must not survive
        # in metadata, or the NEXT run's carry-forward (which unions provenance
        # only for forms still in `aliases`) silently drops it: the persisted
        # output would violate its own `ap keys ⊆ aliases` invariant and the
        # pipeline would not be idempotent over its own output.
        meta_ap = _entry_alias_platforms(e)
        if meta_ap:
            alias_set = set(e["aliases"])
            kept_ap = {k: v for k, v in meta_ap.items() if k in alias_set}
            if kept_ap != meta_ap:
                meta = _entry_meta(e)
                if kept_ap:
                    meta["alias_platforms"] = dict(sorted(kept_ap.items()))
                else:
                    meta.pop("alias_platforms", None)
                e["metadata"] = json.dumps(meta, sort_keys=True)
        dn = e.get("display_name")
        if isinstance(dn, str) and _foreign(dn):
            cand = cid.split("/", 1)[-1]
            e["display_name"] = cid if _foreign(cand) else cand

    enrich_records: list[dict] = []
    for owner in sorted(set(enrich_aliases) | set(enrich_scalars)):
        rec: dict = {"id": owner}
        if enrich_aliases.get(owner):
            rec["aliases"] = sorted(enrich_aliases[owner])
        _merge_weak(rec, enrich_scalars.get(owner) or {})
        if len(rec) > 1:
            enrich_records.append(rec)
    out = survivors + enrich_records

    # Consolidate records sharing an id: carried-forward enrich records can
    # duplicate each other (or a fresh enrich record for the same owner) —
    # without this merge each regen would carry ONE MORE copy forever. A
    # display_name-bearing record wins the base slot; aliases set-union; weak
    # scalars first-wins per field (the loader's tie-break order).
    by_id: dict[str, dict] = {}
    consolidated: list[dict] = []
    for e in out:
        cid = e.get("id")
        if not cid:
            consolidated.append(e)
            continue
        cur = by_id.get(cid)
        if cur is None:
            by_id[cid] = e
            consolidated.append(e)
            continue
        if "display_name" in e and "display_name" not in cur:
            # keep the DEFINING record as the base; fold `cur` into it
            e, cur = cur, e
            idx = consolidated.index(e)
            consolidated[idx] = cur
            by_id[cid] = cur
        if e.get("aliases"):
            cur["aliases"] = sorted(set(cur.get("aliases") or []) | set(e["aliases"]))
        if "display_name" not in cur:
            # weak scalars only flow between enrich records; a full entry
            # keeps its own strong fields and never absorbs weak ones.
            _merge_weak(cur, e.get("weak") or {})
    return sorted(consolidated, key=lambda e: e.get("id") or "")

_CATALOG_HEADER = """# AUTO-GENERATED by scripts/refresh_from_modelsdev.py (catalog split) — DO NOT HAND-EDIT.
# models.dev full-catalog seed. Two record kinds:
#   * fresh canonical mints: models.dev-only (not-on-HF) models — closed-API
#     families (Claude/GPT/Gemini/Grok) + the re-host/community tail. No HF
#     collision, so HF source-of-truth is not violated.
#   * alias-only enrichments {id, aliases}: a models.dev model that IS HF-present
#     (already a canonical). The existing HF-cased canonical wins; only the
#     provider-spelling aliases (carrying inference_platform in
#     metadata.alias_platforms) union onto it, plus the donor's scalars under
#     `weak:` (the seed loader fills only still-empty fields with those). No
#     duplicate canonical is minted.
# The re-cased seed/models/sources/models_dev.generated.yaml is left intact;
# this file is purely additive. Regenerated by the daily refresh-models cron.
"""


def _catalog_load_list(path: Path) -> list[dict]:
    if not path.exists():
        return []
    d = safe_load_yaml(path.read_text())
    if isinstance(d, dict):
        d = d.get("entries", [])
    return d or []


def regenerate_catalog(full: list[dict], adopt_migration: bool = False) -> None:
    """Split the finalized full models.dev catalog (`full`) against the existing
    canonical universe and write models_dev_catalog.generated.yaml + reconcile
    HF-derived community orgs into orgs.generated.yaml. `full` is the output of
    `_finalize_entries(_generate_models(...))`. Dedup/steal-guard semantics:
    HF wins; mint only models.dev-only."""
    # Use the resolver's normalize (collapses case + all separators + digit-dots)
    # to mirror the seed validator's normalized_match steal-guard.
    from eval_entity_resolver.normalization import normalize as _norm

    existing_form_to_cid: dict[str, str] = {}
    existing_exact: dict[str, str] = {}
    existing_norm: dict[str, str] = {}

    def _add_form(form: str, cid: str) -> None:
        if form:
            existing_form_to_cid.setdefault(form.lower(), cid)

    def _add_exact(form: str, cid: str) -> None:
        if form:
            existing_exact.setdefault(form, cid)
            existing_norm.setdefault(_norm(form), cid)

    # Only DEFINED canonicals (a display_name-bearing record somewhere) may own
    # forms: an enrich record keyed by a renamed-away id must never become a
    # fold/donation target — a catalog record keyed by it would materialize a
    # bare canonical that splits the entity from its adopted definition.
    defined_anywhere: set[str] = set()
    active_existing = list(
        _iter_unskipped_source_entries(_CATALOG_EXISTING_SOURCES)
    )
    for _path, e in active_existing:
        if e.get("id") and "display_name" in e:
            defined_anywhere.add(e["id"])

    for _path, e in active_existing:
        cid = e.get("id")
        if not cid or cid not in defined_anywhere:
            continue
        _add_form(cid, cid)
        _add_exact(cid, cid)
        dn = e.get("display_name")
        if dn:
            _add_exact(dn, cid)
        for a in (e.get("aliases") or []):
            _add_form(a, cid)
            _add_exact(a, cid)

    def _steals(form: str, cid: str) -> bool:
        owner = existing_exact.get(form)
        if owner is not None and owner != cid:
            return True
        nowner = existing_norm.get(_norm(form))
        return nowner is not None and nowner != cid

    # ORG-AWARE fold index (the same eval_entity_resolver.fold.decide_fold
    # reconcile_generated_against_existing uses, so both paths agree): a
    # catalog mint that refers to the SAME model as a real HF repo under a
    # different id (org-decoupled, e.g. `zai/GLM-5` -> `zai-org/GLM-5`) must
    # enrich onto the HF id, NOT mint a fresh shadow that aborts the seed.
    from eval_entity_resolver.fold import build_hf_index as _bhi, decide_fold as _df

    _existing_entries = [e for _path, e in active_existing]
    _hf_to_dev_map = _hf_to_dev()
    _chf_ids, _calias, _cby_org, _ = _bhi(_existing_entries, _hf_to_dev_map, _oracle_fixed_ids())

    def _catalog_fold_target(e: dict) -> str | None:
        f = _df(e, _chf_ids, _calias, _cby_org, _hf_to_dev_map)
        return f["hf_target"] if f and f["hf_target"] != e.get("id") else None

    fresh: list[dict] = []
    enrich: list[dict] = []
    # The single enrich record per owner id (first emission wins the slot).
    # EVERY donor for an owner consolidates onto this record — aliases
    # set-union, alias_platforms per-key union (first writer wins), weak
    # per-field first-wins (the loader's tie-break order) — so one owner never
    # carries duplicate records in the written catalog.
    enrich_by_id: dict[str, dict] = {}
    fresh_seen_lc: dict[str, dict] = {}
    fresh_seen_twin: dict[str, dict] = {}
    fresh_form_owner: dict[str, str] = {}

    # Carry-forward inputs: the COMMITTED catalog's claims (incl. its
    # display_names — loader-promoted global aliases) pre-seed
    # fresh_form_owner, so today's split can never steal (or duplicate-claim) a
    # committed record's surface form; after the split the committed records
    # are unioned back / re-emitted below.
    committed = _catalog_load_list(CATALOG_OUT_PATH)
    core_skip = _load_core_skip_ids()
    for c in committed:
        ccid = c.get("id")
        if not ccid or ccid in core_skip:
            continue
        # A committed ENRICH record keyed by an id no source defines is dead
        # (a renamed-away key) — its claims must not block the forms' real
        # owners. Committed FULL mints define themselves and always pre-claim.
        if "display_name" not in c and ccid not in defined_anywhere:
            continue
        fresh_form_owner.setdefault(ccid, ccid)
        for a in (c.get("aliases") or []):
            if a:
                fresh_form_owner.setdefault(a, ccid)
        cdn = c.get("display_name")
        if isinstance(cdn, str) and cdn:
            fresh_form_owner.setdefault(cdn, ccid)

    def _enrich_target(
        cid: str,
        aliases: list[str],
        ap: dict | None,
        donor_id: str | None = None,
        scalars: dict | None = None,
    ) -> None:
        # `donor_id`: a committed record being DONATED onto `cid` (its id has
        # since been absorbed/folded) — forms it pre-claimed are donatable.
        # `scalars`: the donor's non-empty weak scalars (_donor_scalars),
        # carried under `weak:` for the seed loader's empty-fields-only fill.
        ok = (None, cid) if donor_id is None else (None, cid, donor_id)
        keep = sorted({
            a for a in aliases
            if a and a != cid and not _steals(a, cid)
            and fresh_form_owner.get(a) in ok
        })
        for a in keep:
            fresh_form_owner[a] = cid
        ap2: dict = {}
        if ap:
            ap2 = {
                k: v for k, v in ap.items()
                if k != cid and not _steals(k, cid)
                and fresh_form_owner.get(k) in ok
            }
        rec = enrich_by_id.get(cid)
        if rec is None:
            rec = {"id": cid}
            if keep:
                rec["aliases"] = keep
            if ap2:
                rec["metadata"] = json.dumps({"alias_platforms": ap2}, sort_keys=True)
            _merge_weak(rec, scalars or {})
            if len(rec) > 1:
                enrich.append(rec)
                enrich_by_id[cid] = rec
            return
        if keep:
            rec["aliases"] = sorted(set(rec.get("aliases") or []) | set(keep))
        if ap2:
            meta = _entry_meta(rec)
            cur_ap = meta.get("alias_platforms")
            cur_ap = cur_ap if isinstance(cur_ap, dict) else {}
            meta["alias_platforms"] = dict(sorted({**ap2, **cur_ap}.items()))
            rec["metadata"] = json.dumps(meta, sort_keys=True)
        _merge_weak(rec, scalars or {})

    def _forms_of(e: dict) -> list[str]:
        forms = [e["id"]]
        if e.get("display_name"):
            forms.append(e["display_name"])
        forms.extend(a for a in (e.get("aliases") or []) if a)
        return forms

    for e in full:
        cid = e["id"]
        cid_low = cid.lower()
        meta = json.loads(e.get("metadata") or "{}")
        ap = meta.get("alias_platforms") or {}

        cased = existing_form_to_cid.get(cid_low)
        if cased is not None:
            _enrich_target(cased, e.get("aliases", []), ap, scalars=_donor_scalars(e))
            continue
        owner_exact = (
            existing_exact.get(cid)
            or existing_norm.get(_norm(cid))
            or (existing_exact.get(e.get("display_name")) if e.get("display_name") else None)
        )
        if owner_exact is not None:
            _enrich_target(owner_exact, [cid] + list(e.get("aliases", [])), ap, scalars=_donor_scalars(e))
            continue
        fold_tgt = _catalog_fold_target(e)
        if fold_tgt is not None:
            _enrich_target(fold_tgt, [cid] + list(e.get("aliases", [])), ap, scalars=_donor_scalars(e))
            continue
        if cid_low in fresh_seen_lc:
            prior = fresh_seen_lc[cid_low]
            cur = set(prior.get("aliases", []))
            for a in (a for a in e.get("aliases", []) if a):
                if a == prior["id"] or existing_exact.get(a) is not None:
                    continue
                owner = fresh_form_owner.get(a)
                if owner is not None and owner != prior["id"]:
                    continue
                cur.add(a)
                fresh_form_owner[a] = prior["id"]
            prior["aliases"] = sorted(cur)
            continue
        peer = fresh_form_owner.get(cid)
        if peer is not None and peer != cid:
            continue

        clean: list[str] = []
        for a in e.get("aliases", []):
            if not a or a == cid or _steals(a, cid):
                continue
            owner = fresh_form_owner.get(a)
            if owner is not None and owner != cid:
                continue
            clean.append(a)
        e["aliases"] = sorted(set(clean))
        dn = e.get("display_name")

        def _claimed(form: str) -> bool:
            return existing_exact.get(form) is not None or (
                fresh_form_owner.get(form) not in (None, cid))

        if dn and _claimed(dn):
            cand = cid.split("/", 1)[-1]
            e["display_name"] = cand if not _claimed(cand) else cid
        for form in _forms_of(e):
            fresh_form_owner.setdefault(form, cid)
            existing_exact.setdefault(form, cid)
            existing_norm.setdefault(_norm(form), cid)
        fresh_seen_lc[cid_low] = e
        fresh_seen_twin.setdefault(_twin_key(cid), e)
        for k in _extended_twin_keys(cid):
            fresh_seen_twin.setdefault(k, e)
        if ap:
            ap2 = {k: v for k, v in ap.items() if k in e["aliases"]}
            if ap2:
                meta["alias_platforms"] = ap2
            else:
                meta.pop("alias_platforms", None)
            e["metadata"] = json.dumps(meta, sort_keys=True)
        fresh.append(e)

    # Carry the committed catalog forward: an upstream removal must not delete
    # a resolvable surface form. A surviving record gets its committed aliases
    # unioned back; a record whose id vanished from today's split is re-emitted
    # against its still-existing canonical, donated onto the canonical that has
    # since absorbed it, or (for a full mint with no surviving target) retained
    # verbatim tagged `metadata.upstream_status: removed`.
    def _committed_forms(c: dict) -> list[str]:
        """A committed record's donatable surface forms: aliases plus its
        display_name (a loader-promoted global alias — dropping it on a fold
        deletes a resolvable form, e.g. the Veo `Veo-3.1-Fast` regression)."""
        forms = list(c.get("aliases") or [])
        cdn = c.get("display_name")
        if isinstance(cdn, str) and cdn and cdn not in forms:
            forms.append(cdn)
        return forms

    def _union_back(rec: dict, c: dict, donor_id: str | None = None) -> None:
        # `donor_id`: c's committed id when it differs from rec's (a respelled
        # reappearance) — it and the forms it pre-claimed become aliases.
        rcid = rec["id"]
        ok = (None, rcid) if donor_id is None else (None, rcid, donor_id)
        cur = set(rec.get("aliases") or [])
        forms = _committed_forms(c) + ([donor_id] if donor_id else [])
        for a in forms:
            if not a or a in (rcid, rec.get("display_name")) or a in cur or _steals(a, rcid):
                continue
            if fresh_form_owner.get(a) not in ok:
                continue
            cur.add(a)
            fresh_form_owner[a] = rcid
        # Don't materialize `aliases: []` on a weak-only enrich record — the
        # fresh emission omits the key, and the carry-forward must round-trip
        # byte-identically.
        if cur or "aliases" in rec:
            rec["aliases"] = sorted(cur)
        _union_alias_platforms(rec, _entry_alias_platforms(c))
        # Weak scalars on a committed enrich record persist through the
        # carry-forward (today's values win per field); a FULL mint keeps its
        # own strong fields and never absorbs weak ones.
        if "display_name" not in rec:
            _merge_weak(rec, _donor_scalars(c))

    retained = 0
    stabilized_cat: dict[str, str] = {}  # committed id kept -> fresh spelling renamed away
    promoted_cat: dict[str, str] = {}    # committed id absorbed -> higher-rung fresh id
    pending_rung2_cat = 0                # OpenRouter-key twins deferred by the stability rule
    for c in committed:
        ccid = c.get("id")
        if not ccid:
            continue
        if ccid in core_skip:
            # Same skip rule as _carry_forward_committed: a core-skipped
            # committed record whose surface form an existing canonical claims
            # still donates its scalars weakly onto that owner; its aliases
            # stay suppressed (skip_source_ids curates them away).
            scalars = _donor_scalars(c)
            if scalars:
                sk_owner = existing_exact.get(ccid) or existing_norm.get(_norm(ccid))
                if sk_owner is not None and sk_owner != ccid:
                    _enrich_target(sk_owner, [], None, scalars=scalars)
            continue
        prior = fresh_seen_lc.get(ccid.lower())
        if prior is None and "display_name" in c:
            # Respelled reappearance (org-aware collision key + the
            # fold_collisions size guard, with the order/brand-insensitive
            # extended-twin fallback) — full committed mints only.
            t = fresh_seen_twin.get(_twin_key(ccid)) or next(
                (fresh_seen_twin[k] for k in sorted(_extended_twin_keys(ccid))
                 if k in fresh_seen_twin),
                None,
            )
            if t is not None and _bsizes(ccid) == _bsizes(t["id"]):
                prior = t
        if prior is not None:
            # STABILITY RULE, RUNG-MONOTONE (same as _carry_forward_committed):
            # a committed id beats a respelled fresh twin among EQUAL rungs —
            # including a CASE-ONLY respell (committed casing wins, matching
            # the non-catalog path) — except during the one-shot adoption
            # migration. A fresh hf_deferred twin (a REAL HF repo id, rung 1)
            # is never renamed back to a committed non-HF id: the committed id
            # is absorbed as an alias instead (`_union_back` with donor) and
            # its parent-edge references are repointed below. Rung 2 over 3
            # (OpenRouter key over invented) stays migration-gated; the
            # deferred debt is counted and logged. Renamed spellings are
            # repointed below.
            promote = (
                prior["id"] != ccid
                and _entry_meta(prior).get("hf_deferred") is True
                and _entry_meta(c).get("hf_deferred") is not True
            )
            if (
                prior["id"] != ccid
                and not adopt_migration
                and not promote
                and _entry_meta(prior).get("openrouter_adopted") is True
                and _entry_meta(c).get("openrouter_adopted") is not True
                and _entry_meta(c).get("hf_deferred") is not True
            ):
                # Counted for every deferred-debt shape, incl. a twin that
                # already stabilized onto an earlier committed id (mirrors
                # the non-catalog counter).
                pending_rung2_cat += 1
            if (
                prior["id"] != ccid
                and not adopt_migration
                and not promote
                and "display_name" in prior
                and prior["id"] not in stabilized_cat  # already took a committed id
            ):
                old = prior["id"]
                prior["id"] = ccid
                stabilized_cat[ccid] = old
                fresh_seen_lc[ccid.lower()] = prior
                fresh_seen_twin.setdefault(_twin_key(ccid), prior)
                for _m in (fresh_form_owner, existing_exact):
                    for k, v in list(_m.items()):
                        if v == old:
                            _m[k] = ccid
                for k, v in list(existing_norm.items()):
                    if v == old:
                        existing_norm[k] = ccid
                prior["aliases"] = sorted(set(prior.get("aliases") or []) | {old})
                fresh_form_owner[old] = ccid
                _union_back(prior, c, donor_id=old)
                continue
            if prior["id"] != ccid:
                # The committed id lost the twin match (migration run, a
                # higher-rung fresh id, or an already-stabilized twin): it
                # becomes an alias via `_union_back`, so parent edges naming
                # it are repointed below (mirror of the stabilized repoint).
                promoted_cat[ccid] = prior["id"]
            _union_back(prior, c, donor_id=ccid if prior["id"] != ccid else None)
            continue
        target = enrich_by_id.get(ccid)
        if target is not None:
            _union_back(target, c)
            continue
        owner = existing_exact.get(ccid) or existing_norm.get(_norm(ccid))
        if owner == ccid:
            _enrich_target(ccid, _committed_forms(c), _entry_alias_platforms(c), scalars=_donor_scalars(c))
            continue
        if owner is not None:
            _enrich_target(owner, [ccid, *_committed_forms(c)], _entry_alias_platforms(c), donor_id=ccid, scalars=_donor_scalars(c))
            continue
        if "display_name" in c:  # full mint with no surviving target anywhere
            rc = dict(c)
            meta = _entry_meta(rc)
            meta["upstream_status"] = "removed"
            rc["metadata"] = json.dumps(meta, sort_keys=True)
            fresh.append(rc)
            fresh_seen_lc[ccid.lower()] = rc
            fresh_seen_twin.setdefault(_twin_key(ccid), rc)
            for k in _extended_twin_keys(ccid):
                fresh_seen_twin.setdefault(k, rc)
            for form in _forms_of(rc):
                fresh_form_owner.setdefault(form, ccid)
            retained += 1
    if stabilized_cat or promoted_cat:
        renamed = {old: new for new, old in stabilized_cat.items()}
        renamed.update(promoted_cat)
        # Path-compress (mirror of _carry_forward_committed): a
        # promote-then-rename chain must compose to the final survivor.
        for old in list(renamed):
            seen = {old}
            target = renamed[old]
            while target in renamed and target not in seen:
                seen.add(target)
                target = renamed[target]
            renamed[old] = target
        for e in fresh:
            for edge in e.get("parents") or []:
                if isinstance(edge, dict) and edge.get("id") in renamed:
                    edge["id"] = renamed[edge["id"]]
        if promoted_cat:
            print(
                "[refresh] catalog carry-forward: promoted higher-rung fresh "
                "id(s): "
                + ", ".join(
                    f"{old} -> {renamed[old]}" for old in sorted(promoted_cat)
                ),
                file=sys.stderr,
            )
    if stabilized_cat:
        print(
            f"[refresh] catalog carry-forward: kept {len(stabilized_cat)} committed "
            f"id(s) over a respelled fresh twin (stability rule)",
            file=sys.stderr,
        )
    if not adopt_migration:
        # Owner-visible debt line (always printed on a plain cron run): rung-2
        # promotions (OpenRouter key over invented id) the stability rule
        # deferred to the next deliberate migration run (PLAN G3).
        print(
            f"[refresh] catalog stability rule: {pending_rung2_cat} pending "
            f"rung-2 id promotion(s) (OpenRouter key over invented id) awaiting "
            f"the next --adopt-openrouter-ids-migration run",
            file=sys.stderr,
        )
    if retained:
        print(
            f"[refresh] catalog carry-forward: retained {retained} committed "
            f"mint(s) absent from today's upstream",
            file=sys.stderr,
        )

    # Org-FK canonicalization (merge NEW upstream models against the EXISTING org
    # universe): snap each fresh mint's org_id to the canonical developer /
    # HF-true community casing BEFORE writing the catalog
    # and computing `missing`. Without this, a new models.dev model whose org
    # already exists as a community org under HF-true casing (e.g. `Sao10K`) but
    # arrives lowercased (`sao10k`) mints a case-variant TWIN org (split
    # identity) via the case-sensitive set-difference below. Source-local: only
    # the catalog's own fresh entries + orgs.generated.yaml are written here; the
    # whole-universe rewrite stays the separate one-shot. Same authority closure
    # as canonicalize_model_org_ids -> the two never diverge.
    _canon_org = _build_org_canonicalizer()
    for e in fresh:
        for field in ("org_id", "lineage_origin_model_org_id"):
            if e.get(field):
                e[field] = _canon_org(e[field])

    # Stable record key order regardless of which donor reached the owner
    # first (a later donor may add `aliases` to a weak-only record).
    out_entries = fresh + [
        {k: r[k] for k in ("id", "aliases", "metadata", "weak") if k in r}
        for r in enrich
    ]
    body = yaml.safe_dump(out_entries, sort_keys=False, allow_unicode=True, width=200)
    CATALOG_OUT_PATH.write_text(_CATALOG_HEADER + "\n" + body)

    # --- Org reconciliation (two-tier rule) --------------------------------
    curated_org_ids = {e["id"] for e in _catalog_load_list(ORGS_SEED_PATH) if "id" in e}
    gen_orgs = _catalog_load_list(ORGS_GENERATED_PATH)
    gen_org_ids = {e["id"] for e in gen_orgs if "id" in e}
    referenced = {e.get("org_id") for e in fresh if e.get("org_id")}
    missing = sorted(referenced - curated_org_ids - gen_org_ids)
    if missing:
        for oid in missing:
            gen_orgs.append({
                "id": oid, "display_name": oid, "hf_org": oid,
                "kind": "community", "tags": "[]", "metadata": "{}",
                "review_status": "reviewed",
            })
        gen_header = (
            ORGS_GENERATED_PATH.read_text().split("\n- ", 1)[0].rstrip()
            if ORGS_GENERATED_PATH.exists() else ""
        )
        if not gen_header.startswith("#"):
            gen_header = "# AUTO-GENERATED — HF-derived community orgs."
        ORGS_GENERATED_PATH.write_text(
            gen_header + "\n"
            + yaml.safe_dump(gen_orgs, sort_keys=False, allow_unicode=True, width=200)
        )
        print(f"[refresh] catalog: reconciled {len(missing)} missing community org(s): {missing}", file=sys.stderr)
    print(
        f"[refresh] catalog: {len(fresh)} fresh mint(s) (not-on-HF), "
        f"{len(enrich)} alias-only enrichment(s) (HF-present) -> {CATALOG_OUT_PATH}",
        file=sys.stderr,
    )


# All model source files whose org_ids must have a canonical_orgs row.
_ALL_MODEL_SOURCES = (
    HF_ORACLE_PATH, SEED_PATH, HUB_STATS_PATH, CATALOG_OUT_PATH, TIER3_PATH, CORE_PATH,
)
# The real-HF casing authority: org_ids minted from real HF repos (HF-true casing).
_HF_TRUE_CASING_SOURCES = (HF_ORACLE_PATH, HUB_STATS_PATH)
ORGS_DISTINCT_ALLOWLIST_PATH = REPO_ROOT / "seed" / "orgs_distinct_allowlist.yaml"


def _load_distinct_org_allowlist() -> set[str]:
    if not ORGS_DISTINCT_ALLOWLIST_PATH.exists():
        return set()
    return {
        x
        for x in (safe_load_yaml(ORGS_DISTINCT_ALLOWLIST_PATH.read_text()) or [])
        if isinstance(x, str)
    }


def _write_source_entries(path: Path, entries: list[dict]) -> None:
    """Rewrite a generated/core source file's entries, preserving its `# header`
    and the `{skip_ids,...,entries}` dict shape where present."""
    text = path.read_text() if path.exists() else ""
    header = "\n".join(ln for ln in text.splitlines() if ln.startswith("#"))
    doc = safe_load_yaml(text) if text else None
    if isinstance(doc, dict) and "entries" in doc:
        out = {**doc, "entries": entries}
    else:
        out = entries
    body = yaml.safe_dump(out, sort_keys=False, allow_unicode=True, width=200)
    path.write_text((header + "\n" if header else "") + body)


def _build_org_canonicalizer() -> Callable[[str | None], str | None]:
    """The single org-canonicalization closure shared by the whole-universe
    one-shot (`canonicalize_model_org_ids`) and the source-local --catalog cron
    (`regenerate_catalog`), so the two NEVER diverge on casing (which would
    oscillate across runs). Folds an HF org spelling to: curated developer id ->
    authoritative HF-true community casing -> verbatim, honoring the distinct-org
    allowlist. Authority: PRIMARY from real-HF sources (HF-true), FALLBACK across
    all model sources + the existing canonical_orgs rows (so a new upstream model
    snaps to an EXISTING org row even when no model yet references that casing).
    Returns a `_canon(org_spelling) -> canonical_org_id` callable."""
    from eval_entity_resolver.fold import build_curated_org_map, build_community_casing, canonicalize_org

    curated_map = build_curated_org_map(_catalog_load_list(ORGS_SEED_PATH))
    distinct = _load_distinct_org_allowlist()

    primary = build_community_casing([
        e.get("org_id") for p in _HF_TRUE_CASING_SOURCES for e in _catalog_load_list(p)
        if isinstance(e.get("org_id"), str)
    ])
    fallback = build_community_casing(
        [e.get("org_id") for p in _ALL_MODEL_SOURCES for e in _catalog_load_list(p)
         if isinstance(e.get("org_id"), str)]
        + [e["id"] for p in (ORGS_SEED_PATH, ORGS_GENERATED_PATH)
           for e in _catalog_load_list(p) if isinstance(e.get("id"), str)]
    )
    community = {**fallback, **primary}  # primary (real-HF) wins

    def _canon(org):
        if not isinstance(org, str) or not org:
            return org
        return canonicalize_org(org, curated_map, community, distinct)

    return _canon


def canonicalize_model_org_ids(write_core: bool = False) -> int:
    """Canonicalize EVERY model's org_id + lineage_origin_model_org_id across all
    sources to ONE spelling per developer, via the single shared
    eval_entity_resolver.fold.canonicalize_org: curated developer id when the org
    folds to one, else the authoritative HF-true community casing (from the
    hf_oracle/hub_stats real-HF org_ids), honoring the distinct-org allowlist.

    This is the single org-canonicalization pass — replaces relying on each
    generator to emit a consistent casing (they see different raw spellings). A
    refresh re-runs it deterministically. Returns the number of fields rewritten.

    `write_core` gates rewriting the hand-curated core.yaml: the cron path
    NEVER writes it (`_write_source_entries`'s YAML re-dump hoists core's
    inline comments and reformats the whole file) — core stays in the READ
    set for the canonicalization authority, and any core org_id spelling
    that would need rewriting is reported loudly for a manual fix instead.
    The one-shot regenerate_sources.sh opts in via --reconcile-orgs-write-core."""
    _canon = _build_org_canonicalizer()

    rewritten = 0
    for path in _ALL_MODEL_SOURCES:
        entries = _catalog_load_list(path)
        if not entries:
            continue
        core_readonly = path == CORE_PATH and not write_core
        changed = False
        for e in entries:
            if not isinstance(e, dict):
                continue
            for field in ("org_id", "lineage_origin_model_org_id"):
                old = e.get(field)
                new = _canon(old)
                if new != old:
                    if core_readonly:
                        print(
                            f"::warning::core.yaml {e.get('id')}: {field} {old!r} "
                            f"should be {new!r} — fix core.yaml by hand (the cron "
                            f"never rewrites the curated file)"
                        )
                        continue
                    e[field] = new
                    rewritten += 1
                    changed = True
        if changed:
            _write_source_entries(path, entries)
    print(f"[refresh] org-canonicalize: rewrote {rewritten} org field(s) to the "
          f"canonical developer spelling", file=sys.stderr)
    return rewritten


def reconcile_all_orgs(write_core: bool = False) -> None:
    """Ensure EVERY org_id referenced by ANY model (across all sources + core)
    has a canonical_orgs row — mint a community org for those that don't.

    The per-generator org reconciliation only covers each generator's OWN mints
    (hf_oracle its targets; the --catalog split its fresh mints), so org_ids that
    appear ONLY in the non-catalog models_dev or tier3 mints dangle. This runs as
    the LAST regen step (after tier3) over the UNION of all sources, so it is the
    single authoritative org reconciler. Additive-only: never deletes/renames a
    row, preserves exact HF org spelling (no separator/case collapse, so it can't
    collapse genuinely-distinct orgs), excludes org_id=None and any org already
    CLAIMED by a curated org (id / hf_org / alias) so a community row never shadows
    a curated lab."""
    # FIRST canonicalize every model's org_id to one spelling per developer
    # (curated id / HF-true community casing), so the rows minted below are over
    # canonical org_ids and no case/separator twins are created.
    canonicalize_model_org_ids(write_core=write_core)

    # Curated CLAIMS (not just ids) so a referenced org that a curated lab already
    # owns as an hf_org/alias is remapped, not minted as a community twin.
    curated_claims: set[str] = set()
    for e in _catalog_load_list(ORGS_SEED_PATH):
        for form in (e.get("id"), e.get("hf_org"), *(e.get("aliases") or [])):
            if isinstance(form, str) and form:
                curated_claims.add(form.lower())
    gen_orgs = _catalog_load_list(ORGS_GENERATED_PATH)
    gen_org_ids = {e["id"] for e in gen_orgs if isinstance(e.get("id"), str)}

    referenced: set[str] = set()
    for path in _ALL_MODEL_SOURCES:
        for e in _catalog_load_list(path):
            oid = e.get("org_id")
            if isinstance(oid, str) and oid:
                referenced.add(oid)

    # PRUNE stale case/separator TWINS: a generated org row that NO model
    # references AND whose case/separator-insensitive key matches a DIFFERENT,
    # referenced org is a leftover from a prior community casing (e.g. an old
    # lowercase `madeagents` after canonicalize flipped every model to the
    # HF-true `MadeAgents`). It would trip test_no_case_split_orgs. Dropping it
    # is safe — it's unreferenced and a genuine twin (NOT a distinct uploader,
    # which would still be referenced by its own models). Curated orgs are never
    # touched (only seed/orgs.generated.yaml rows). Honors the distinct allowlist.
    from eval_entity_resolver.fold import _norm_org_key
    distinct = _load_distinct_org_allowlist()
    ref_keys = {_norm_org_key(o): o for o in referenced}
    pruned: list[str] = []
    kept_orgs = []
    for e in gen_orgs:
        oid = e.get("id")
        if (isinstance(oid, str) and oid and oid not in referenced
                and oid not in distinct
                and _norm_org_key(oid) in ref_keys
                and ref_keys[_norm_org_key(oid)] != oid):
            pruned.append(oid)
            continue
        kept_orgs.append(e)
    gen_orgs = kept_orgs
    gen_org_ids = {e["id"] for e in gen_orgs if isinstance(e.get("id"), str)}

    missing = sorted(
        oid for oid in referenced
        if oid not in gen_org_ids and oid.lower() not in curated_claims
    )
    for oid in missing:
        gen_orgs.append({
            "id": oid, "display_name": oid, "hf_org": oid,
            "kind": "community", "tags": "[]", "metadata": "{}",
            "review_status": "reviewed",
        })
    if not missing and not pruned:
        print("[refresh] org-reconcile: no dangling org_ids", file=sys.stderr)
        return
    gen_header = (
        ORGS_GENERATED_PATH.read_text().split("\n- ", 1)[0].rstrip()
        if ORGS_GENERATED_PATH.exists() else ""
    )
    if not gen_header.startswith("#"):
        gen_header = "# AUTO-GENERATED — HF-derived community orgs."
    ORGS_GENERATED_PATH.write_text(
        gen_header + "\n" + yaml.safe_dump(gen_orgs, sort_keys=False, allow_unicode=True, width=200)
    )
    print(f"[refresh] org-reconcile: minted {len(missing)} community org(s), pruned "
          f"{len(pruned)} stale twin(s) (e.g. mint {missing[:4]}, prune {pruned[:4]})", file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--no-fetch", action="store_true", help="use cached /tmp/modelsdev_api.json")
    p.add_argument("--dry-run", action="store_true", help=f"print diff vs current {SEED_PATH}; don't write")
    p.add_argument(
        "--preview-out",
        type=Path,
        default=None,
        help="write to this PREVIEW path instead of the committed generated YAML "
        "(for inspection; leaves the committed file untouched)",
    )
    p.add_argument(
        "--catalog",
        action="store_true",
        help="ONLY (re)generate the models_dev_catalog.generated.yaml "
        "split (de-orphan) + reconcile orgs.generated.yaml; does NOT rewrite "
        "models_dev.generated.yaml. The cron runs this as a second step after "
        "the source write so the catalog splits against the settled re-cased "
        "models_dev.",
    )
    p.add_argument(
        "--reconcile-orgs",
        action="store_true",
        help="ONLY run the universal org reconcile: mint a community canonical_orgs "
        "row for every org_id referenced by ANY model source (incl. tier3) that "
        "lacks one. Run as the LAST regen step (after tier3) — does not fetch or "
        "rewrite model sources. Never WRITES core.yaml (reads it for authority; "
        "warns when a core org_id spelling needs a manual fix).",
    )
    p.add_argument(
        "--adopt-openrouter-ids-migration",
        action="store_true",
        help="ONE-SHOT migration switch for the OpenRouter id adoption "
        "(specs/model-id-resolution PLAN.md G3): on a committed/fresh twin "
        "match the FRESH externally-adopted id wins and the committed id "
        "becomes an alias. Without it the stability rule holds — the "
        "committed id wins — so the daily cron can never thrash ids on "
        "OpenRouter listing changes.",
    )
    p.add_argument(
        "--reconcile-orgs-write-core",
        action="store_true",
        help="like --reconcile-orgs, but ALSO rewrites core.yaml org_id "
        "spellings (a full YAML re-dump that hoists core's inline comments). "
        "Manual/one-shot use only (regenerate_sources.sh) — the cron must "
        "never write the curated file.",
    )
    args = p.parse_args()

    # --reconcile-orgs: standalone, no network / no model-source rewrite.
    if args.reconcile_orgs or args.reconcile_orgs_write_core:
        reconcile_all_orgs(write_core=args.reconcile_orgs_write_core)
        return 0

    api = _fetch(use_cache=args.no_fetch)
    known_orgs = _load_known_org_ids()
    if not known_orgs:
        print(f"[refresh] ERROR: {ORGS_SEED_PATH} not found or empty. Seed orgs first.", file=sys.stderr)
        return 1

    # --catalog: skip the models_dev source rewrite entirely; only split the
    # full author-lab catalog against the EXISTING on-disk sources.
    if args.catalog:
        if args.preview_out is not None:
            p.error(
                "--preview-out is not supported with --catalog: the catalog "
                "split always writes the committed catalog + orgs files"
            )
        generated, skipped_no_org = _generate_models(api, known_orgs)
        if skipped_no_org:
            print(f"[refresh] ERROR: {len(skipped_no_org)} provider(s) -> unknown org_id", file=sys.stderr)
            return 1
        regenerate_catalog(
            _tag_openrouter_key_ids(_finalize_entries(generated), api),
            adopt_migration=args.adopt_openrouter_ids_migration,
        )
        return 0

    generated, skipped_no_org = _generate_models(api, known_orgs)
    if skipped_no_org:
        print(
            f"[refresh] ERROR: {len(skipped_no_org)} provider(s) mapped to unknown org_id "
            f"(must exist in {ORGS_SEED_PATH}):",
            file=sys.stderr,
        )
        for entry in skipped_no_org:
            print(f"  - {entry}", file=sys.stderr)
        print(
            "[refresh] Add the missing orgs to seed/orgs.yaml or fix the "
            "PROVIDER_TO_ORG mapping in this script, then re-run.",
            file=sys.stderr,
        )
        return 1
    # Finalize BEFORE reconciliation (same order as the --catalog path, which
    # dedups `_finalize_entries(generated)`). _finalize_entries flattens the
    # working `alias_platforms` map into `aliases`; without it the reconcile's
    # steal/fold/donate logic reads only `aliases` and is BLIND to the provider
    # spellings still buried in alias_platforms (e.g. a `-Instruct` mint's
    # `meta/llama-3-1-8b-instruct` provider forms), so those forms leak past the
    # dedup and only surface — on the wrong canonical — once _write_yaml flattens
    # them, aborting the seed with a base/variant alias collision. Idempotent, so
    # _write_yaml's re-finalize below is a no-op.
    generated = _tag_openrouter_key_ids(_finalize_entries(generated), api)
    # Carry the committed file forward through the wholesale rewrite: an
    # upstream (provider,model) removal retains the committed entry, an
    # alias-level removal unions the committed aliases back, and the returned
    # claims map stops a fresh mint from stealing a carried entry's forms.
    generated, committed_claims = _carry_forward_committed(
        generated, _catalog_load_list(SEED_PATH),
        adopt_migration=args.adopt_openrouter_ids_migration,
    )
    # Core-aware reconciliation: suppress/repoint any mint whose normalized id
    # collides with an existing canonical (incl. core.yaml) under a DIFFERENT id,
    # so the full re-cased rewrite is ADDITIVE rather than clobbering curated
    # fixes. Same steal-guard the --catalog path uses; excludes SEED_PATH (we are
    # rewriting it). Carried-forward entries run through the same pass, so one
    # that has since been curated/folded is suppressed, not resurrected.
    before = len(generated)
    generated = reconcile_generated_against_existing(
        generated, committed_claims=committed_claims
    )
    if len(generated) != before:
        print(
            f"[refresh] reconciliation: suppressed {before - len(generated)} "
            f"mint(s) colliding (normalized) with an existing canonical",
            file=sys.stderr,
        )
    # Org-FK canonicalization (same closure + authority as the --catalog path and
    # the one-shot): snap each surviving mint's org_id to the existing developer /
    # HF-true community casing so the full re-cased rewrite never mints a
    # case-variant TWIN org for a developer that already exists under HF-true
    # casing. Without it, a model whose org arrives lowercased (`sao10k`) would
    # split-identity against the existing `Sao10K` row.
    _canon_org = _build_org_canonicalizer()
    for e in generated:
        for field in ("org_id", "lineage_origin_model_org_id"):
            if e.get(field):
                e[field] = _canon_org(e[field])
    out_path = args.preview_out or SEED_PATH
    new_text = _write_yaml(generated, out_path)

    if args.preview_out is not None and not args.dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(new_text)
        print(
            f"[refresh] PREVIEW: wrote {len(generated)} model entries to {out_path} "
            f"(committed {SEED_PATH} untouched)",
            file=sys.stderr,
        )
        return 0

    if args.dry_run:
        if SEED_PATH.exists():
            old = SEED_PATH.read_text()
            if old == new_text:
                print("[refresh] no changes")
            else:
                import difflib
                diff = difflib.unified_diff(
                    old.splitlines(keepends=True),
                    new_text.splitlines(keepends=True),
                    fromfile=str(SEED_PATH),
                    tofile=f"{SEED_PATH} (generated)",
                )
                sys.stdout.writelines(diff)
        else:
            print(new_text)
        return 0

    SEED_PATH.parent.mkdir(parents=True, exist_ok=True)
    SEED_PATH.write_text(new_text)
    print(
        f"[refresh] wrote {len(generated)} model entries to {SEED_PATH}",
        file=sys.stderr,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
