"""Unit tests for the models.dev dedup / group / mint / alias logic.

OFFLINE: drives the committed reference sidecars under
curation/ plus the PINNED models.dev API snapshot at
tests/fixtures/modelsdev_api.snapshot.json. No network, no seed --local.
The snapshot and the underlying index are frozen from the SAME models.dev pull,
so the dedup count and the group roots reproduce EXACTLY (no drift band).

Validates:
  - the ported dedup (normalize -> canon_key_ordered -> safe_sig union-find)
    reproduces the committed 1,274-group underlying index;
  - head selection honours author-lab > re-host > null-org precedence;
  - the 663 EEE rescues map to their recorded underlying groups;
  - edge axis classification: training_stage / tier (no scale claim) /
    size (only on disclosed open-weight tokens);
  - every PROVIDER_TO_INFERENCE_PLATFORM target id exists in the curated
    inference_platforms catalog.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest
import yaml

from conftest import load_script_module

from eval_card_registry.lib.seed_io import resolve_oracle_path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC_DIR = REPO_ROOT  / "curation"
API_CACHE = REPO_ROOT / "tests" / "fixtures" / "modelsdev_api.snapshot.json"
UNDERLYING_INDEX = SPEC_DIR / "modelsdev_underlying_index.json"
RESCUES = SPEC_DIR / "eee_modelsdev_rescues.json"
PLATFORMS = SPEC_DIR / "inference_platforms.proposed.json"

# Frozen-pull dedup floor: distinct underlying groups in curation/modelsdev_underlying_index.json.
EXPECTED_UNDERLYING_GROUPS = 1_279
# EEE rescue floor: rows in curation/eee_modelsdev_rescues.json.
EXPECTED_EEE_RESCUES = 662
# Curated inference-platform count: entries in curation/inference_platforms.proposed.json.
EXPECTED_PLATFORM_COUNT = 138


@pytest.fixture(scope="module")
def mod():
    return load_script_module("refresh_from_modelsdev")


@pytest.fixture(scope="module")
def api():
    if not API_CACHE.exists():
        pytest.skip(f"pinned models.dev API snapshot missing at {API_CACHE}")
    return json.loads(API_CACHE.read_text())


@pytest.fixture(scope="module")
def underlying_index():
    if not UNDERLYING_INDEX.exists():
        pytest.skip(f"underlying index sidecar missing at {UNDERLYING_INDEX}")
    return json.loads(UNDERLYING_INDEX.read_text())


# ---------------------------------------------------------------------------
# 1. Dedup reproduces the committed 1,283 underlying groups.
# ---------------------------------------------------------------------------
def test_dedup_reproduces_underlying_group_count(mod, api, underlying_index):
    groups = mod.build_underlying_groups(api)
    n = len(groups)
    # The snapshot and the committed index are frozen from the SAME models.dev
    # pull, so the count is EXACT — no drift band. Re-pinned 1283->1279 when the
    # normalizer was corrected (`-maas` serving-tier collapse + `:NNNb` size specs
    # kept distinct, e.g. gemma3:12b / gpt-oss:120b); the index was regenerated
    # from the same snapshot with the corrected normalize.
    assert n == EXPECTED_UNDERLYING_GROUPS, f"expected {EXPECTED_UNDERLYING_GROUPS} underlying groups, got {n}"
    assert len(underlying_index) == EXPECTED_UNDERLYING_GROUPS


def test_dedup_group_roots_match_committed_index(mod, api, underlying_index):
    """The generator's group roots must reproduce the committed reference index
    EXACTLY — proving the ported normalize/canon_key/safe_sig logic is unchanged.

    The snapshot (tests/fixtures/modelsdev_api.snapshot.json) and the reference
    `modelsdev_underlying_index.json` are frozen from the SAME pull, so there is
    no snapshot drift to tolerate: any difference in the root set is a real
    normalize/canon_key/safe_sig regression. To re-pin against a refreshed API,
    replace the snapshot fixture AND regenerate the index from it with the same
    normalize/canon_key/safe_sig logic."""
    my_roots = set(mod.build_underlying_groups(api).keys())
    idx_roots = set(underlying_index.keys())
    assert my_roots == idx_roots, (
        f"root set diverged from the frozen index — "
        f"new={sorted(my_roots - idx_roots)[:8]} "
        f"gone={sorted(idx_roots - my_roots)[:8]}"
    )


@pytest.mark.parametrize(
    "root,expect_author,expect_org_hf",
    [
        ("claude-3-5-sonnet", True, "anthropic"),
        ("gpt-4o", True, "openai"),
        ("yi-large", False, None),          # re-host-only, org-unknown
        ("abliterated-model", False, None),  # community re-host, no author lab
    ],
)
def test_dedup_spotcheck_known_groups(mod, api, underlying_index, root, expect_author, expect_org_hf):
    groups = mod.build_underlying_groups(api)
    assert root in groups, f"group {root!r} not produced"
    head = mod.pick_underlying(root, groups[root])
    assert head["has_author_lab_entry"] is expect_author
    assert head["author_org"] == expect_org_hf
    # The committed index agrees with our head pick.
    idx_entry = underlying_index[root]
    assert idx_entry["has_author_lab_entry"] is expect_author
    assert idx_entry["author_org"] == expect_org_hf


def _reorder_api(api: dict) -> dict:
    """Return the same catalog with provider order AND per-provider model order
    reversed — a non-trivial deterministic permutation of models.dev's key order.
    Used to prove grouping/head-pick don't depend on upstream iteration order."""
    out = {}
    for prov in reversed(list(api.keys())):
        pdata = dict(api[prov])
        models = pdata.get("models")
        if isinstance(models, dict):
            pdata["models"] = {k: models[k] for k in reversed(list(models.keys()))}
        out[prov] = pdata
    return out


def test_grouping_invariant_to_modelsdev_key_order(mod, api):
    """GATE: build_underlying_groups must produce the SAME roots and the SAME
    group membership regardless of models.dev's key order. Guards the determinism
    fix — the union-find base is the lexicographically-first key (not API order),
    so a provider/model reordering upstream cannot flip a group head."""
    a = mod.build_underlying_groups(api)
    b = mod.build_underlying_groups(_reorder_api(api))
    assert set(a.keys()) == set(b.keys()), (
        f"roots changed under key reorder: "
        f"new={sorted(set(b) - set(a))[:8]} gone={sorted(set(a) - set(b))[:8]}"
    )
    membership = lambda g: {root: sorted(r["raw"] for r in recs) for root, recs in g.items()}
    assert membership(a) == membership(b), "group membership changed under key reorder"


def test_head_pick_invariant_to_modelsdev_key_order(mod, api):
    """GATE: pick_underlying's head_spelling must be stable under a key reorder —
    the author-rec sort keys on (len(norm), raw), not provider iteration order, so
    the minted id can't churn when models.dev reorders providers."""
    a = mod.build_underlying_groups(api)
    b = mod.build_underlying_groups(_reorder_api(api))
    for root in a:
        ha = mod.pick_underlying(root, a[root])
        hb = mod.pick_underlying(root, b[root])
        assert ha["head_spelling"] == hb["head_spelling"], (
            f"head_spelling for {root!r} flipped under key reorder: "
            f"{ha['head_spelling']!r} != {hb['head_spelling']!r}"
        )


# ---------------------------------------------------------------------------
# 2. Head selection precedence: author-lab > re-host > null.
# ---------------------------------------------------------------------------
def test_head_pick_author_lab_beats_rehost(mod):
    """A small synthetic group: the author-lab provider's org/name anchor wins
    over re-host providers, even when re-hosts disagree on date/name."""
    root = "claude-3-5-sonnet"
    recs = [
        # Re-host with a (wrong) earlier date and a different name.
        dict(provider="venice", raw="claude-3-5-sonnet", norm="claude-3-5-sonnet",
             key=root, family="claude", release="2023-01-01", name="Sonnet (venice)",
             open_weights=False, record={}),
        # The true author lab.
        dict(provider="anthropic", raw="claude-3-5-sonnet-20240620",
             norm="claude-3-5-sonnet-20240620", key=root, family="claude",
             release="2024-06-20", name="Claude 3.5 Sonnet", open_weights=False, record={}),
        # Another re-host.
        dict(provider="openrouter", raw="anthropic/claude-3.5-sonnet",
             norm="claude-3-5-sonnet", key=root, family="claude", release="2024-06-21",
             name="Claude 3.5 Sonnet", open_weights=False, record={}),
    ]
    head = mod.pick_underlying(root, recs)
    assert head["has_author_lab_entry"] is True
    assert head["author_org"] == "anthropic"
    # Display name comes from the author-lab record, not the re-host's vanity name.
    assert head["display_name"] == "Claude 3.5 Sonnet"
    # Head spelling is the author-lab's own spelling.
    assert head["head_spelling"] == "claude-3-5-sonnet-20240620"


def test_head_pick_rehost_only_infers_org_from_family(mod):
    """No author-lab provider in the group, but the family token implies an org
    (re-host > null): org inferred, has_author_lab_entry stays False."""
    root = "llama-3-1-8b"
    recs = [
        dict(provider="togetherai", raw="meta-llama/Llama-3.1-8B",
             norm="llama-3-1-8b", key=root, family="llama", release="2024-07-23",
             name="Llama 3.1 8B", open_weights=True, record={}),
        dict(provider="fireworks-ai", raw="llama-v3p1-8b",
             norm="llama-3-1-8b", key=root, family="llama", release=None,
             name="Llama 3.1 8B", open_weights=True, record={}),
    ]
    head = mod.pick_underlying(root, recs)
    assert head["has_author_lab_entry"] is False
    assert head["author_org"] == "meta-llama"   # inferred from family, HF-style
    assert head["open_weights"] is True          # any(True)


def test_head_pick_null_org_when_no_signal(mod):
    """No author lab and no family-org signal -> org stays null (the org-less
    bucket); head spelling falls back to the cleanest normalised spelling."""
    root = "abliterated-model"
    recs = [
        dict(provider="abliteration-ai", raw="abliterated-model",
             norm="abliterated-model", key=root, family=None, release="2026-01-06",
             name="Abliterated Model", open_weights=True, record={}),
    ]
    head = mod.pick_underlying(root, recs)
    assert head["has_author_lab_entry"] is False
    assert head["author_org"] is None
    assert head["head_spelling"] == "abliterated-model"


# ---------------------------------------------------------------------------
# 3. The 663 EEE rescues map to their recorded underlying groups.
# ---------------------------------------------------------------------------
def test_rescues_map_to_underlying_groups(mod, underlying_index):
    if not RESCUES.exists():
        pytest.skip(f"rescues sidecar missing at {RESCUES}")
    rescues = json.loads(RESCUES.read_text())
    assert len(rescues) == EXPECTED_EEE_RESCUES

    key_to_root = {r: r for r in underlying_index}
    mset_to_root: dict[str, list[str]] = defaultdict(list)
    for r in underlying_index:
        mset_to_root[mod.safe_sig(r)].append(r)

    def lookup(eee_id: str) -> str | None:
        n = mod.normalize_modelsdev_id(eee_id)
        k = mod.canon_key_ordered(n)
        if not k:
            return None
        if k in key_to_root:
            return k
        ms = mod.safe_sig(k)
        if ms in mset_to_root:
            return mset_to_root[ms][0]
        return None

    matched = sum(1 for eee_id, rec in rescues.items()
                  if lookup(eee_id) == rec["underlying_model_id"])
    # The ported normalize/canon must reproduce every recorded match.
    assert matched == EXPECTED_EEE_RESCUES, (
        f"only {matched}/{EXPECTED_EEE_RESCUES} rescues reproduced"
    )


def test_rescues_sample_spotcheck(mod, underlying_index):
    """Spot-check a handful of named rescues resolve to the expected group."""
    if not RESCUES.exists():
        pytest.skip("rescues sidecar missing")
    rescues = json.loads(RESCUES.read_text())
    sample = [eee for eee in ("01-ai/yi-large", "01-ai/Yi-1.5-34B-Chat") if eee in rescues]
    assert sample, "expected at least one known sample id in the rescues"
    for eee_id in sample:
        n = mod.normalize_modelsdev_id(eee_id)
        k = mod.canon_key_ordered(n)
        root = k if k in underlying_index else None
        if root is None:
            ms = mod.safe_sig(k)
            for r in underlying_index:
                if mod.safe_sig(r) == ms:
                    root = r
                    break
        assert root == rescues[eee_id]["underlying_model_id"]


# ---------------------------------------------------------------------------
# 4. Edge axis classification.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "token,expected",
    [
        ("instruct", ("variant", "training_stage")),
        ("chat", ("variant", "training_stage")),
        ("it", ("variant", "training_stage")),
        ("base", ("variant", "training_stage")),
        ("sonnet", ("variant", "tier")),
        ("haiku", ("variant", "tier")),
        ("opus", ("variant", "tier")),
        ("mini", ("variant", "tier")),
        ("nano", ("variant", "tier")),
        ("flash", ("variant", "tier")),
        ("pro", ("variant", "tier")),
        ("7b", ("variant", "size")),
        ("70b", ("variant", "size")),
        ("405b", ("variant", "size")),
        ("8x7b", ("variant", "size")),
        ("a16b", ("variant", "size")),
        ("thinking", ("variant", "mode")),
        ("vision", ("variant", "modality")),
        ("coder", ("variant", "domain")),
    ],
)
def test_classify_token_axis(mod, token, expected):
    assert mod._classify_token(token) == expected


def test_branded_tier_never_gets_size_edge(mod):
    """A branded tier (no disclosed scale) must classify as `tier`, NEVER
    `size`. Every tier token is guarded."""
    for tok in ("haiku", "sonnet", "opus", "mini", "nano", "flash", "pro",
                "small", "medium", "large"):
        rel, axis = mod._classify_token(tok)
        assert axis == "tier", f"{tok} mis-classified as {axis}"
        assert axis != "size"
        assert tok in mod._TIER_TOKENS


def test_size_token_only_for_disclosed_open_weight_tokens(mod):
    """A `size` edge is asserted ONLY for a genuinely-disclosed scale token
    (open-weight name params / MoE specs) — never for a branded tier or an
    unrelated token."""
    assert mod._is_size_token("70b") is True
    assert mod._is_size_token("8x7b") is True
    assert mod._is_size_token("a16b") is True
    assert mod._is_size_token("1.5b") is True
    assert mod._is_size_token("mini") is False
    assert mod._is_size_token("sonnet") is False
    assert mod._is_size_token("instruct") is False


def test_suffix_chain_llama_size_then_stage(mod):
    """A compound open-weight suffix `70b-instruct` classifies as a size edge
    followed by a training_stage edge (size first because the family root has no
    size token here)."""
    segments = mod._classify_suffix_segments("70b-instruct")
    assert segments == [
        ("variant", "size", "70b"),
        ("variant", "training_stage", "instruct"),
    ]


def test_instruct_suffix_is_training_stage_in_family_tree(mod):
    """End-to-end: a models.dev `-instruct` snapshot emits a variant/
    training_stage edge (NOT axis=mode) in the built family tree."""
    api = {
        "mistral": {
            "id": "mistral", "name": "Mistral", "models": {
                "mistral-7b-instruct": {
                    "id": "mistral-7b-instruct", "name": "Mistral 7B Instruct",
                    "release_date": "2023-09-27", "open_weights": True, "family": "mistral",
                },
            },
        },
    }
    out, _missing = mod._generate_models(api, {"mistralai"})
    by_id = {e["id"]: e for e in out}
    instruct = by_id["mistralai/mistral-7b-instruct"]
    assert instruct["parents"] == [{
        "id": "mistralai/mistral-7b", "relationship": "variant", "axis": "training_stage",
    }]


# ---------------------------------------------------------------------------
# 5. PROVIDER_TO_INFERENCE_PLATFORM ids all exist in the curated catalog.
# ---------------------------------------------------------------------------
def test_provider_to_inference_platform_ids_exist(mod):
    if not PLATFORMS.exists():
        pytest.skip(f"inference platforms sidecar missing at {PLATFORMS}")
    catalog = json.loads(PLATFORMS.read_text())
    valid_ids = {p["id"] for p in catalog["platforms"]}
    valid_providers = {p["models_dev_provider"] for p in catalog["platforms"]}

    assert mod.PROVIDER_TO_INFERENCE_PLATFORM, "map is empty"
    # Every models.dev provider (inference platforms + author labs) maps to a
    # catalog platform; counts are kept in lockstep with the curated catalog.
    # Platforms without a models_dev_provider (EEE/prefix-only, e.g. prodia)
    # are catalog entries but not provider-map entries.
    assert len(catalog["platforms"]) == EXPECTED_PLATFORM_COUNT
    provider_backed = [p for p in catalog["platforms"] if p.get("models_dev_provider")]
    assert len(mod.PROVIDER_TO_INFERENCE_PLATFORM) == len(provider_backed)

    for provider, platform_id in mod.PROVIDER_TO_INFERENCE_PLATFORM.items():
        assert platform_id in valid_ids, f"platform id {platform_id!r} not in catalog"
        assert provider in valid_providers, f"provider {provider!r} not a known slug"


def test_strict_author_sourced_from_catalog(mod):
    """STRICT_AUTHOR is the kind=author_lab subset of the same catalog."""
    if not PLATFORMS.exists():
        pytest.skip("inference platforms sidecar missing")
    catalog = json.loads(PLATFORMS.read_text())
    author_providers = {
        p["models_dev_provider"] for p in catalog["platforms"]
        if p["kind"] == "author_lab"
    }
    assert mod.STRICT_AUTHOR == author_providers
    assert len(mod.STRICT_AUTHOR) == 29


# ---------------------------------------------------------------------------
# 6. Full-catalog generation: provider-preserving group -> mint -> alias.
# ---------------------------------------------------------------------------
def test_full_generation_seeds_closed_api_and_rehosts(mod, api):
    """The refactor seeds the FULL catalog (closed-API + re-hosts), not just
    author labs; re-host-only / minted groups are draft, author trees reviewed;
    provider spellings are aliased carrying their platform."""
    known = mod._load_known_org_ids()
    out, skipped = mod._generate_models(api, known)
    assert skipped == [], f"unexpected missing-org skips: {skipped[:5]}"

    by_id = {e["id"]: e for e in out}
    # Closed-API author lab is present and reviewed.
    assert "openai/gpt-4o" in by_id
    assert by_id["openai/gpt-4o"]["review_status"] == "reviewed"
    # A re-host-only / org-less group is minted as draft + org-unknown.
    assert "yi-large" in by_id
    yi = by_id["yi-large"]
    assert yi["org_id"] is None
    assert yi["review_status"] == "draft"
    assert "org-unknown" in yi["tags"]

    # No malformed (multi-slash) canonical ids.
    assert all(e["id"].count("/") <= 1 for e in out)

    # Provider-tagged aliases: a multi-provider author group carries platform
    # provenance folded into metadata.alias_platforms after finalize.
    fin = mod._finalize_entries([dict(e) for e in out])
    fin_by_id = {e["id"]: e for e in fin}
    gpt4o = fin_by_id["openai/gpt-4o"]
    meta = json.loads(gpt4o["metadata"])
    assert meta.get("alias_platforms"), "expected provider platform provenance"
    # Every tagged platform id is a real catalog id.
    catalog = json.loads(PLATFORMS.read_text())
    valid_ids = {p["id"] for p in catalog["platforms"]}
    for plat in meta["alias_platforms"].values():
        assert plat in valid_ids
    # No gnarly aliases survived finalize.
    for e in fin:
        for a in e["aliases"]:
            assert a.count("/") <= 1
            assert not a.startswith("@")


# ---------------------------------------------------------------------------
# 7. Mint-decision rule: defer HF-resolvable groups, mint off-HF ones.
# ---------------------------------------------------------------------------
HF_ORACLE_JSON = resolve_oracle_path()


def test_mint_decision_defers_on_hf_group_and_mints_off_hf(mod):
    """A models.dev group that IS a real HF repo defers to the HF id (spellings
    become aliases, no shadow mint); a genuinely off-HF group still mints.

    Uses a SYNTHETIC authority so the test is deterministic regardless of the
    live oracle contents, and exercises the hard `qwen-qwq-32b` vs `Qwen/QwQ-32B`
    case (differs in BOTH org spelling and brand-prefixed name)."""
    from eval_entity_resolver.normalization import normalize as _nz

    alias_index = mod._build_org_alias_index()
    # Synthetic authority: {dev_org: {normalized_name: real_hf_id}}.
    authority = {
        "alibaba": {_nz("QwQ-32B"): "Qwen/QwQ-32B"},
    }

    # (a) On-HF, hard case: models.dev key carries the brand prefix
    # (alibaba/qwen-qwq-32b), HF org is Qwen->alibaba, name qwen-qwq-32b->qwq-32b.
    deferred = mod._hf_defer_target(
        "alibaba/qwen-qwq-32b",
        "alibaba",
        ["alibaba/qwen-qwq-32b", "qwen-qwq-32b", "Qwen QwQ 32B"],
        alias_index,
        authority,
    )
    assert deferred == "Qwen/QwQ-32B", "on-HF brand-prefixed group must DEFER"

    # (b) On-HF, easy case: name already matches without prefix-strip.
    assert (
        mod._hf_defer_target(
            "alibaba/qwq-32b", "alibaba", ["QwQ-32B", "Qwen/QwQ-32B"], alias_index, authority
        )
        == "Qwen/QwQ-32B"
    )

    # (c) Off-HF closed-API group: nothing in authority -> MINT (None).
    assert (
        mod._hf_defer_target(
            "anthropic/claude-opus-4-5",
            "anthropic",
            ["claude-opus-4-5", "Claude Opus 4.5"],
            alias_index,
            authority,
        )
        is None
    )

    # (d) No false merge across DIFFERENT developers: a group named qwq-32b but
    # authored by openai must NOT defer to the alibaba HF id.
    assert (
        mod._hf_defer_target(
            "openai/qwq-32b", "openai", ["qwq-32b"], alias_index, authority
        )
        is None
    )

    # (e) Org-less group never defers (no org agreement possible).
    assert (
        mod._hf_defer_target("qwq-32b", None, ["qwq-32b"], alias_index, authority)
        is None
    )


def test_mint_decision_emits_hf_deferred_record_in_generation(mod, api):
    """End-to-end against the pinned snapshot + frozen oracle: the QwQ-32B shadow
    group defers to the real `Qwen/QwQ-32B` (an hf_deferred record carrying the
    models.dev spellings as aliases, NOT a shadow `alibaba/qwen-qwq-32b` mint),
    while closed-API families (Claude/Grok) still mint off-HF canonicals."""
    if not HF_ORACLE_JSON.exists():
        pytest.skip(f"HF oracle missing at {HF_ORACLE_JSON}")
    # Reset the module-cached authority so it builds from the real oracle.
    mod._HF_AUTHORITY = None

    known = mod._load_known_org_ids()
    out, skipped = mod._generate_models(api, known)
    assert skipped == [], f"unexpected missing-org skips: {skipped[:5]}"
    by_id = {e["id"]: e for e in out}

    def _is_deferred(e):
        return json.loads(e.get("metadata") or "{}").get("hf_deferred") is True

    # DEFER: the real HF id is present as an hf_deferred record; the shadow mint
    # id is NOT a canonical, and is carried as an alias on the HF entry instead.
    assert "Qwen/QwQ-32B" in by_id, "expected the real HF id to be emitted"
    qwq = by_id["Qwen/QwQ-32B"]
    assert _is_deferred(qwq)
    assert qwq["resolution_source"] == "models_dev"
    assert "alibaba/qwen-qwq-32b" not in by_id, "shadow dupe must NOT be minted"
    assert "alibaba/qwen-qwq-32b" in qwq["aliases"], "shadow spelling must be an alias"

    # MINT: genuinely off-HF closed-API families still mint {org}/{slug}.
    grok = [e for e in out if e["id"].startswith("xai/grok-3") and not _is_deferred(e)]
    assert grok, "off-HF Grok family must still mint"
    claude = [
        e for e in out if e["id"].startswith("anthropic/claude") and not _is_deferred(e)
    ]
    assert claude, "off-HF Claude family must still mint"

    # The defer path produced at least the known hand-fold shape.
    deferred_ids = {e["id"] for e in out if _is_deferred(e)}
    assert "Qwen/QwQ-32B" in deferred_ids


# ---------------------------------------------------------------------------
# Core-aware reconciliation of the NON-catalog write path: a mint must defer to
# an existing curated canonical it normalized-collides with, never emit a twin.
# ---------------------------------------------------------------------------
REHOST_REPOINT = SPEC_DIR / "rehost_repoint.json"


def test_reconciliation_suppresses_normalized_collision_with_core(mod, tmp_path):
    """The non-catalog reconciliation pass must NOT emit a mint whose NORMALIZED
    id collides with an existing core canonical under a DIFFERENT id — it
    suppresses/repoints to the curated id instead.

    Uses a SYNTHETIC injected existing-sources set (a tmp core.yaml with a
    curated `Foo/Bar-7B`) so no real seed file is touched, exercising the
    injectable `sources` parameter."""
    from eval_entity_resolver.normalization import normalize as _nz

    core = tmp_path / "core.yaml"
    core.write_text(
        yaml.safe_dump(
            [
                {
                    "id": "Foo/Bar-7B",
                    "display_name": "Foo Bar 7B",
                    "aliases": ["foo-bar-7b"],
                }
            ]
        )
    )

    generated = [
        # Lowercase twin of the curated canonical — normalized form collides
        # (`foo bar 7b`) under a DIFFERENT id. Must be suppressed.
        {
            "id": "foo/bar-7b",
            "display_name": "foo bar 7b",
            "aliases": ["foo/bar-7b", "foo-bar-7b-extra-alias"],
        },
        # A genuinely-novel mint with no collision — must survive untouched.
        {
            "id": "openai/gpt-4o",
            "display_name": "GPT-4o",
            "aliases": ["gpt-4o"],
        },
    ]

    out = mod.reconcile_generated_against_existing(generated, sources=(core,))
    by_id = {e["id"]: e for e in out}
    emitted_ids = set(by_id)

    # The colliding lowercase twin is gone as a CANONICAL; the novel mint survives.
    assert "foo/bar-7b" not in emitted_ids, "normalized-colliding twin must be suppressed"
    assert "openai/gpt-4o" in emitted_ids, "non-colliding mint must survive"

    # MERGE, not drop: the suppressed mint's non-stealing alias must be
    # carried onto an enrich record for the curated OWNER — not lost.
    assert "Foo/Bar-7B" in by_id, "expected an enrich record merging onto the owner"
    assert "foo-bar-7b-extra-alias" in by_id["Foo/Bar-7B"]["aliases"], (
        "suppressed mint's unique alias must survive on the owner (merge, not drop)"
    )

    # Core invariant: NO emitted id collides (normalized) with the core canonical
    # under a DIFFERENT id.
    core_canon = "Foo/Bar-7B"
    core_norm = _nz(core_canon)
    for e in out:
        if _nz(e["id"]) == core_norm:
            assert e["id"] == core_canon, (
                f"emitted id {e['id']!r} normalized-collides with core "
                f"canonical {core_canon!r} under a different id"
            )


def test_tier3_is_an_existing_source_for_both_refresh_paths(mod, tmp_path):
    """A models.dev mint must not create a normalized twin of a Tier-3
    canonical. Both the wholesale and catalog refresh paths therefore include
    Tier-3 in their owner index; the generated data is donated to the existing
    owner as enrichment instead of becoming a second full canonical.

    This pins the live ``seed-2.0-pro`` / ``seed-2-0-pro`` failure class without
    relying on a dated live snapshot.
    """
    assert mod.TIER3_PATH in mod._NONCATALOG_EXISTING_SOURCES
    assert mod.TIER3_PATH in mod._CATALOG_EXISTING_SOURCES

    tier3 = tmp_path / "tier3.yaml"
    tier3.write_text(
        yaml.safe_dump(
            [
                {
                    "id": "bytedance/seed-2.0-pro",
                    "display_name": "Seed 2.0 Pro",
                    "aliases": ["bytedance/seed-2.0-pro"],
                }
            ]
        )
    )
    generated = [
        {
            "id": "bytedance/seed-2-0-pro",
            "display_name": "Seed 2.0 Pro",
            "aliases": ["seed-2-0-pro", "seed-2.0-pro"],
            "release_date": "2026-02-14",
        }
    ]

    out = mod.reconcile_generated_against_existing(generated, sources=(tier3,))
    full_ids = {e["id"] for e in out if "display_name" in e}
    assert "bytedance/seed-2-0-pro" not in full_ids
    (enrichment,) = [e for e in out if e["id"] == "bytedance/seed-2.0-pro"]
    assert "seed-2-0-pro" in enrichment["aliases"]
    assert enrichment["weak"]["release_date"] == "2026-02-14"


def test_reconciliation_rewrites_surviving_parent_edges_off_suppressed_mints(mod, tmp_path):
    """Rewrite parent edges: when a mint is suppressed (collides with a core
    canonical), a SURVIVING entry whose parents[].id pointed at that suppressed
    mint must be repointed to the owner — never left dangling."""
    core = tmp_path / "core.yaml"
    core.write_text(yaml.safe_dump([{
        "id": "Acme/Base-7B", "display_name": "Acme Base 7B",
        "aliases": ["acme-base-7b"],
    }]))
    generated = [
        # Collides (normalized) with the curated Acme/Base-7B -> suppressed.
        {"id": "acme/base-7b", "display_name": "Acme Base 7B v0", "aliases": []},
        # A surviving child whose parent edge points at the suppressed mint id.
        {
            "id": "acme/base-7b-instruct",
            "display_name": "Acme Base 7B Instruct",
            "aliases": [],
            "parents": [{"id": "acme/base-7b", "relationship": "variant", "axis": "training_stage"}],
        },
    ]
    out = mod.reconcile_generated_against_existing(generated, sources=(core,))
    by_id = {e["id"]: e for e in out}
    assert "acme/base-7b" not in by_id, "colliding base mint suppressed"
    child = by_id["acme/base-7b-instruct"]
    assert child["parents"][0]["id"] == "Acme/Base-7B", (
        "surviving child's parent edge must be repointed to the curated owner, "
        f"not left dangling: {child['parents']}"
    )


def test_reconciliation_drops_intra_batch_sibling_id_alias(mod, tmp_path):
    """INTRA-BATCH hygiene: the non-catalog path REWRITES models_dev.generated.yaml
    wholesale, so a base mint that aliases a variant which is ALSO its own batch
    entry would double-claim the form and abort the seed. The sibling's id wins
    (distinct canonicals — drop the alias, not a merge); non-colliding aliases on
    the base survive. The cross-source check alone misses this (neither sibling is
    in the EXISTING sources)."""
    core = tmp_path / "core.yaml"
    core.write_text("entries: []\n")
    # Fully-synthetic ids (absent from the real frozen oracle / sources) so the
    # only collision under test is the intra-batch sibling-id one, not an
    # incidental org-aware fold to a real upstream id.
    generated = [
        {"id": "acmecorp/Foo-70B", "display_name": "Acmecorp Foo 70B",
         "aliases": ["acmecorp/Foo-70B-Instruct", "foo-70b-bare"]},
        {"id": "acmecorp/Foo-70B-Instruct", "display_name": "Acmecorp Foo 70B Instruct",
         "aliases": []},
    ]
    out = mod.reconcile_generated_against_existing([dict(e) for e in generated], sources=(core,))
    by_id = {e["id"]: e for e in out}
    assert "acmecorp/Foo-70B" in by_id and "acmecorp/Foo-70B-Instruct" in by_id
    base_aliases = by_id["acmecorp/Foo-70B"].get("aliases") or []
    assert "acmecorp/Foo-70B-Instruct" not in base_aliases, (
        "base must not claim the sibling variant's id as an alias (would abort seed)"
    )
    assert "foo-70b-bare" in base_aliases, "a non-colliding alias must survive"


def test_reconciliation_dedups_intra_batch_shared_nonid_alias(mod, tmp_path):
    """Two sibling survivors sharing a NON-id alias (neither owns it as its id)
    must not both claim it (the seed owner would be order-dependent). The
    lexicographically-first claimant keeps it; the other drops it — deterministic
    across cron runs."""
    core = tmp_path / "core.yaml"
    core.write_text("entries: []\n")
    generated = [
        {"id": "zzz/model-b", "display_name": "Model B", "aliases": ["shared-free-alias"]},
        {"id": "aaa/model-a", "display_name": "Model A", "aliases": ["shared-free-alias"]},
    ]
    out = mod.reconcile_generated_against_existing([dict(e) for e in generated], sources=(core,))
    owners = sorted(e["id"] for e in out if "shared-free-alias" in (e.get("aliases") or []))
    assert owners == ["aaa/model-a"], f"shared non-id alias not deterministically owned: {owners}"


def test_reconciliation_does_not_donate_suppressed_id_owned_elsewhere(mod, tmp_path):
    """When an org-aware fold suppresses a mint whose id is ITSELF a real canonical
    owned by a DIFFERENT id (a distinct model wrongly dragged onto a variant via a
    mangled alias — e.g. the base google/gemma-2-9b folded onto google/gemma-2-9b-it),
    the suppressed id must NOT be donated as an alias onto the fold owner. The real
    owner already supplies that form; donating it would double-claim and merge two
    distinct canonicals. Mirrors the foreign-owner guard the loser's OTHER aliases
    already get."""
    # Anchored on the REAL frozen sources (default _NONCATALOG_EXISTING_SOURCES):
    # google/gemma-2-9b (base) and google/gemma-2-9b-it (instruct) are both real
    # HF canonicals. A models.dev mint whose id is the base but which carries the
    # instruct's mangled key alias (google/gemma2-9b-it) gets org-aware-folded onto
    # the instruct. The base id must NOT then be donated as an alias onto it
    # (build_hf_index only treats real-HF ids as fold targets, so a synthetic
    # fixture cannot reproduce this — it needs real entries).
    generated = [
        {"id": "google/gemma-2-9b", "display_name": "Gemma 2 9B", "aliases": ["google/gemma2-9b-it"]},
    ]
    out = mod.reconcile_generated_against_existing([dict(e) for e in generated])
    donated_onto = [e["id"] for e in out if "google/gemma-2-9b" in (e.get("aliases") or [])]
    assert donated_onto == [], (
        f"the base canonical google/gemma-2-9b was wrongly donated as an alias onto "
        f"{donated_onto} — a distinct model merged into a variant"
    )


def test_reconciliation_no_op_when_no_collision(mod, tmp_path):
    """When no mint collides (normalized) with the injected existing set, the
    reconciliation is a pure pass-through (id-sorted): the common no-collision case
    leaves the generated rows unchanged apart from ordering."""
    core = tmp_path / "core.yaml"
    core.write_text(yaml.safe_dump([{"id": "Acme/Widget-3B", "aliases": []}]))
    generated = [
        {"id": "openai/gpt-4o", "display_name": "GPT-4o", "aliases": ["gpt-4o"]},
        {"id": "anthropic/claude-opus-4-5", "display_name": "Claude Opus 4.5",
         "aliases": ["claude-opus-4-5"]},
    ]
    out = mod.reconcile_generated_against_existing(generated, sources=(core,))
    assert [e["id"] for e in out] == sorted(e["id"] for e in generated)


def test_rehost_mint_never_uses_base_vendor_prefix(mod, api):
    """Re-host id rule: a minted canonical id must NEVER be the base-vendor
    `{junk}` id from curation/rehost_repoint.json. Drive
    `_generate_models` over the pinned snapshot and assert that for every
    REKEY_REAL / CLOSED entry the generator does NOT emit the `junk`
    (base-vendor) id as a canonical.

    Entries whose underlying group is NOT present in the pinned snapshot can't
    be reproduced offline and are skipped (counted); the assertion runs on the
    snapshot-reachable subset, which is the meaningful guard against recurrence.
    """
    if not REHOST_REPOINT.exists():
        pytest.skip(f"rehost repoint oracle missing at {REHOST_REPOINT}")
    repoint = json.loads(REHOST_REPOINT.read_text())

    # Reset cached HF authority so it builds from the real frozen oracle.
    mod._HF_AUTHORITY = None
    known = mod._load_known_org_ids()
    out, skipped = mod._generate_models(api, known)
    assert skipped == [], f"unexpected missing-org skips: {skipped[:5]}"
    emitted_ids = {e["id"] for e in out}

    # Group roots reachable from the pinned snapshot.
    reachable_roots = set(mod.build_underlying_groups(api).keys())

    # Oracle-real ids: a `junk` entry that is in fact a real HF repo (fixed_exact/
    # near_miss) is NO LONGER junk — the generator correctly DEFERS to it (HF-true
    # casing). Some repoint entries predate a model becoming a real repo (e.g.
    # MiniMaxAI/MiniMax-M2.1 was closed-API when curated, is now on HF). Such a
    # defer is correct, not a base-vendor mint, so exclude oracle-real junk ids.
    HF_ORACLE_JSON = resolve_oracle_path()
    oracle_real: set[str] = set()
    if HF_ORACLE_JSON.exists():
        for _r, _m in json.loads(HF_ORACLE_JSON.read_text()).get("resolutions", {}).items():
            if _m.get("resolution_status") in ("fixed_exact", "fixed_near_miss"):
                fx = _m.get("fixed_hf_model_id")
                if isinstance(fx, str):
                    oracle_real.add(fx)

    checked = 0
    skipped_unreachable = 0
    offenders: list[str] = []
    for e in repoint:
        if e.get("kind") not in ("REKEY_REAL", "CLOSED"):
            continue
        junk = e.get("junk")
        if not junk or junk in oracle_real:  # a real HF repo -> deferring to it is correct
            continue
        root = mod.canon_key_ordered(mod.normalize_modelsdev_id(junk))
        if root not in reachable_roots:
            skipped_unreachable += 1
            continue
        checked += 1
        if junk in emitted_ids:
            offenders.append(f"{e['kind']}:{junk}")

    assert checked > 0, "expected at least one snapshot-reachable REKEY/CLOSED case"
    assert not offenders, (
        f"generator minted {len(offenders)} base-vendor junk id(s) as canonicals: "
        f"{offenders[:10]}"
    )
    # Sanity: a meaningful share of the oracle is actually exercised offline.
    assert checked >= 50, (
        f"only {checked} REKEY/CLOSED cases reachable in the pinned snapshot "
        f"({skipped_unreachable} skipped) — coverage unexpectedly low"
    )


# ---------------------------------------------------------------------------
# A variant must never emit its variant-suffix-stripped BASE form as one of its
# own aliases: that base form is owned by another canonical, so donating it
# would double-claim the form and abort the seed (e.g. gpt-4-turbo -> gpt-4).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "variant_raw,stolen_base",
    [
        ("gpt-4-turbo", "gpt-4"),
        ("glm-5-turbo", "glm-5"),
        ("deepseek-reasoner", "deepseek"),
        ("qwen3-next-80b-a3b-thinking", "qwen3-next-80b-a3b"),
        ("Phi-4-reasoning", "phi-4"),
        ("model-x-fp8", "model-x"),
    ],
)
def test_provider_alias_forms_never_emits_variant_stripped_base(mod, variant_raw, stolen_base):
    """A variant spelling's alias forms must PRESERVE its identity — never the
    base id/slug that another canonical owns. Donating the base form (gpt-4-turbo
    -> alias gpt-4) double-claims it and aborts the seed."""
    forms = mod._provider_alias_forms(variant_raw, "anyorg")
    assert stolen_base not in forms, f"{variant_raw} emitted base {stolen_base!r}: {forms}"
    assert f"anyorg/{stolen_base}" not in forms, f"{variant_raw} emitted org/base: {forms}"
    # The variant's OWN form is still emitted (identity preserved).
    own = mod._slugify(mod.normalize_modelsdev_id(variant_raw, strip_variants=False)).rsplit("/", 1)[-1]
    assert own in forms, f"{variant_raw} lost its own form {own!r}: {forms}"


def test_serving_tags_still_stripped_in_alias_forms(mod):
    """Serving TAGS (`:free`, `-fast`) are scaffolding and stay stripped even in
    alias forms — only IDENTITY variants are preserved."""
    assert "gpt-4o" in mod._provider_alias_forms("gpt-4o:free", "openai")
    forms = mod._provider_alias_forms("some-model-fast", "x")
    assert "some-model" in forms


def test_intra_output_merge_preserves_loser_scalars_and_parents(mod):
    """The winner-merge must FILL empty winner scalars from a loser and UNION
    parent edges — not silently drop the richer loser's modalities / params /
    lineage (the winner is often an hf_deferred entry with those fields empty)."""
    entries = [
        # Winner: hf_deferred real repo, empty enrichment fields.
        {
            "id": "Foo/Bar-7B",
            "org_id": "foo",
            "aliases": [],
            "parents": [],
            "input_modalities": None,
            "params_billions": None,
            "metadata": json.dumps({"hf_deferred": True}),
        },
        # Loser: same model (same normalized id), carries real metadata + lineage.
        {
            "id": "foo/bar-7b",
            "org_id": "foo",
            "aliases": ["foo-bar-7b"],
            "parents": [{"id": "foo/base-7b", "relationship": "finetune"}],
            "input_modalities": ["text", "image"],
            "params_billions": 7.0,
            "metadata": "{}",
        },
    ]
    out = mod._reconcile_intra_output(entries)
    by_id = {e["id"]: e for e in out}
    assert "Foo/Bar-7B" in by_id and "foo/bar-7b" not in by_id, "same-model dup merged into HF-true winner"
    w = by_id["Foo/Bar-7B"]
    assert w["input_modalities"] == ["text", "image"], "loser modalities must be inherited"
    assert w["params_billions"] == 7.0, "loser params must be inherited"
    assert any(edge["id"] == "foo/base-7b" for edge in w.get("parents") or []), "loser parent edge must survive"


def test_reconcile_strips_surviving_mint_foreign_alias_and_displayname(mod, tmp_path):
    """A SURVIVING mint (its id does not steal) must not keep an alias OR
    display_name owned by a DIFFERENT existing canonical — that double-claim aborts
    the seed. Strip both."""
    core = tmp_path / "core.yaml"
    core.write_text(
        yaml.safe_dump(
            [{"id": "Existing/Canon-7B", "display_name": "Existing Canon", "aliases": ["shared-bare"]}]
        )
    )
    generated = [
        {
            "id": "other/distinct-model",   # id does NOT collide -> survives
            "display_name": "Existing Canon",   # but display_name is owned by core
            "aliases": ["shared-bare", "own-clean-alias"],  # shared-bare owned by core
        }
    ]
    out = mod.reconcile_generated_against_existing(generated, sources=(core,))
    surv = next(e for e in out if e["id"] == "other/distinct-model")
    assert "shared-bare" not in surv["aliases"], "foreign alias must be stripped from survivor"
    assert "own-clean-alias" in surv["aliases"], "clean alias must be kept"
    assert surv["display_name"] != "Existing Canon", "foreign display_name must be re-derived"


def test_intra_output_dedupes_ambiguous_bare_alias_across_distinct_canonicals(mod):
    """A bare alias claimed by 2+ DISTINCT surviving canonicals (cross-org
    `gemma-3-1b-it`) double-claims the form and aborts the seed. Keep it on one
    (the natural/first owner), drop from the others."""
    entries = [
        {"id": "google/gemma-3-1b-it", "org_id": "google", "aliases": ["gemma-3-1b-it"]},
        {"id": "unsloth/gemma-3-1b-it", "org_id": "unsloth", "aliases": ["gemma-3-1b-it"]},
    ]
    out = mod._reconcile_intra_output(entries)
    claimers = [e["id"] for e in out if "gemma-3-1b-it" in (e.get("aliases") or [])]
    assert len(claimers) == 1, f"ambiguous bare alias must be kept on exactly one canonical, got {claimers}"


def test_no_emitted_alias_steals_another_canonical_id(mod, api):
    """End-to-end over the pinned snapshot: NO emitted entry may carry another
    emitted canonical's id (exact OR resolver-normalized) as an alias — that is
    the exact double-claim the seed loader aborts on."""
    from collections import defaultdict

    from eval_entity_resolver.normalization import normalize as _nz

    mod._HF_AUTHORITY = None
    out, skipped = mod._generate_models(api, mod._load_known_org_ids())
    assert skipped == [], f"unexpected missing-org skips: {skipped[:5]}"
    fin = mod._finalize_entries([dict(e) for e in out])

    ids = {e["id"] for e in fin}
    id_norms = {_nz(i): i for i in ids}
    offenders = []
    for e in fin:
        for a in e.get("aliases", []):
            # An alias that is exactly a DIFFERENT canonical's id.
            if a in ids and a != e["id"]:
                offenders.append((e["id"], "exact", a))
            # ...or normalized-collides with a different canonical's id.
            owner = id_norms.get(_nz(a))
            if owner is not None and owner != e["id"] and a not in ids:
                offenders.append((e["id"], "norm", a, owner))
    assert not offenders, (
        f"{len(offenders)} alias(es) steal another canonical's id "
        f"(seed would abort): {offenders[:12]}"
    )


# ---------------------------------------------------------------------------
# SOURCE-SHAPE GUARD: a catalog regen is deterministic and never DROPS a
# committed canonical id. The resolution gates (tests/test_gate_invariants.py)
# validate resolution OUTCOMES, not the shape of *.generated.yaml — so a cron
# regen that silently drops / renames / re-cases a mint would pass them. This
# guards the 'only ADD, never regress' invariant at the SOURCE layer, which is
# what makes the daily --catalog cron safe.
# ---------------------------------------------------------------------------
def test_catalog_regen_deterministic_and_no_drop(mod, api, tmp_path, monkeypatch):
    import copy
    import yaml as _y

    known = mod._load_known_org_ids()
    full = mod._finalize_entries(mod._generate_models(api, known)[0])

    def _regen(catalog_path):
        # Write catalog + orgs to TMP (not the committed files). regenerate_catalog
        # dedups against _CATALOG_EXISTING_SOURCES (committed, read-only) and writes
        # the catalog BEFORE the org reconcile, so the catalog output is unaffected
        # by pointing ORGS_GENERATED_PATH at an empty tmp file.
        monkeypatch.setattr(mod, "CATALOG_OUT_PATH", catalog_path)
        monkeypatch.setattr(mod, "ORGS_GENERATED_PATH", tmp_path / (catalog_path.stem + "_orgs.yaml"))
        mod.regenerate_catalog(copy.deepcopy(full))
        return catalog_path.read_text()

    text1 = _regen(tmp_path / "cat1.yaml")
    text2 = _regen(tmp_path / "cat2.yaml")
    assert text1 == text2, "catalog regen is NON-DETERMINISTIC (would oscillate across cron runs)"

    regen_ids = {e["id"] for e in (_y.safe_load(text1) or []) if isinstance(e, dict) and e.get("id")}
    committed = _y.safe_load((REPO_ROOT / "seed" / "models" / "sources" / "models_dev_catalog.generated.yaml").read_text())
    committed_ids = {e["id"] for e in (committed or []) if isinstance(e, dict) and e.get("id")}

    # A regen may legitimately DROP a committed catalog id IF another source still
    # covers it — a stale catalog standalone of an HF-present model correctly folds
    # to an alias-only enrichment on regen (e.g. meta-llama/Llama-3.1-70B, which
    # hf_oracle owns). The regression we guard against is a drop that NO other
    # source covers -> a real coverage loss the resolution gates wouldn't localize.
    other_source_ids = set()
    for f in ("hf_oracle", "hub_stats", "models_dev"):
        for e in _y.safe_load((REPO_ROOT / "seed" / "models" / "sources" / f"{f}.generated.yaml").read_text()) or []:
            if isinstance(e, dict) and e.get("id"):
                other_source_ids.add(e["id"])
    _core = _y.safe_load((REPO_ROOT / "seed" / "models" / "core.yaml").read_text()) or {}
    for e in (_core.get("entries") if isinstance(_core, dict) else _core) or []:
        if isinstance(e, dict) and e.get("id"):
            other_source_ids.add(e["id"])

    uncovered = (committed_ids - regen_ids) - other_source_ids
    assert not uncovered, (
        f"catalog regen DROPS {len(uncovered)} committed canonical id(s) NOT covered by "
        f"any other source — the daily --catalog cron would regress coverage: "
        f"{sorted(uncovered)[:12]}"
    )


# ---------------------------------------------------------------------------
# FROZEN-SNAPSHOT DETERMINISM: this test's input is intentionally pinned to the
# June snapshot. A committed source is refreshed from live models.dev, so
# comparing it byte-for-byte to that historical input is invalid and keeps both
# main and the cron permanently red. Instead, run the complete non-catalog
# pipeline twice from the same pinned input and committed carry-forward state.
# The fixture remains a strong generator-regression corpus while live-refresh
# validation is handled by the source-shape and identity invariants.
# ---------------------------------------------------------------------------
def test_noncatalog_regen_is_deterministic_from_pinned_snapshot(mod, api):
    def run() -> str:
        # Hermetic reset: an earlier test may have primed the module cache with
        # a monkeypatched HF authority index.
        mod._HF_AUTHORITY = None
        gen, skipped = mod._generate_models(api, mod._load_known_org_ids())
        assert skipped == []
        gen = mod._tag_openrouter_key_ids(
            mod._finalize_entries([dict(e) for e in gen]), api
        )
        committed = mod._catalog_load_list(mod.SEED_PATH)
        gen, claims = mod._carry_forward_committed(gen, committed)
        gen = mod.reconcile_generated_against_existing(gen, committed_claims=claims)
        canonicalize_org = mod._build_org_canonicalizer()
        for entry in gen:
            for field in ("org_id", "lineage_origin_model_org_id"):
                if entry.get(field):
                    entry[field] = canonicalize_org(entry[field])
        return mod._write_yaml(gen, mod.SEED_PATH)

    assert run() == run(), "non-catalog regen is non-deterministic for the pinned snapshot"


# ---------------------------------------------------------------------------
# MERGE CORRECTNESS: a NEW upstream model whose author org already exists in our
# registry as a community org under HF-true casing (e.g. Sao10K) but arrives
# lowercased (sao10k) must FK to the EXISTING org row — NOT mint a case-variant
# twin (split identity). This is the "only enhance new entities, never regress
# merge" invariant at the org-FK layer. The daily --catalog cron regenerates
# against live models.dev, so a freshly-added model under an existing org in a
# different casing is the exact path this guards.
# ---------------------------------------------------------------------------
def test_catalog_new_model_snaps_to_existing_community_org_casing(mod, tmp_path, monkeypatch):
    import yaml as _y

    gen_orgs = _y.safe_load((REPO_ROOT / "seed" / "orgs.generated.yaml").read_text()) or []
    # An existing community org with non-lowercase HF-true casing and no '/'.
    existing = next(
        e["id"] for e in gen_orgs
        if isinstance(e, dict) and e.get("id") and "/" not in e["id"]
        and e["id"].lower() != e["id"]
    )
    variant = existing.lower()
    assert variant != existing

    # A synthetic NOT-on-HF model id (stays a fresh mint; not folded to an HF
    # target) whose org_id arrives in the lowercased variant casing.
    synthetic = {
        "id": f"{variant}/Zzz-Synthetic-New-Model-9000",
        "display_name": "Zzz-Synthetic-New-Model-9000",
        "org_id": variant,
        "resolution_source": "models_dev",
        "review_status": "auto",
        "aliases": [],
        "metadata": "{}",
    }

    cat_out = tmp_path / "cat.yaml"
    orgs_out = tmp_path / "orgs.yaml"
    orgs_out.write_text((REPO_ROOT / "seed" / "orgs.generated.yaml").read_text())
    monkeypatch.setattr(mod, "CATALOG_OUT_PATH", cat_out)
    monkeypatch.setattr(mod, "ORGS_GENERATED_PATH", orgs_out)
    mod.regenerate_catalog([dict(synthetic)])

    cat = _y.safe_load(cat_out.read_text()) or []
    rec = next((e for e in cat if isinstance(e, dict) and e.get("id", "").startswith(f"{variant}/")), None)
    assert rec is not None, "synthetic fresh mint missing from catalog output"
    assert rec["org_id"] == existing, (
        f"org_id not snapped to existing community casing: {rec['org_id']!r} != {existing!r} "
        f"— a new upstream model would split-identity its org"
    )

    after = _y.safe_load(orgs_out.read_text()) or []
    after_ids = {e["id"] for e in after if isinstance(e, dict) and e.get("id")}
    assert variant not in after_ids, (
        f"minted a case-variant twin org {variant!r} alongside existing {existing!r}"
    )


# ---------------------------------------------------------------------------
# Carry-forward: a wholesale rewrite never deletes committed surface forms.
# ---------------------------------------------------------------------------

@pytest.fixture
def cf_mod(tmp_path):
    """Fresh module instance (not the module-scoped `mod`) with CORE_PATH
    pointed at a tmp file, so each test controls the core skip lists."""
    m = load_script_module("refresh_from_modelsdev", "refresh_modelsdev_carryforward")
    m.CORE_PATH = tmp_path / "core.yaml"
    return m


def _cf_entry(cid: str, aliases=(), metadata: str = "{}") -> dict:
    return {
        "id": cid,
        "display_name": cid.split("/")[-1],
        "org_id": cid.split("/")[0] if "/" in cid else None,
        "aliases": list(aliases),
        "metadata": metadata,
        "review_status": "draft",
    }


def test_carry_forward_retains_upstream_removed_entry(cf_mod):
    """A committed entry whose id vanished from today's upstream is retained
    verbatim, tagged `metadata.upstream_status: removed`, and its forms are
    recorded in the returned claims map."""
    committed = [_cf_entry("groq/llama3-9b-9192", ["llama3-9b-9192"])]
    fresh = [_cf_entry("openai/gpt-4o-test", ["gpt-4o-test"])]
    batch, claims = cf_mod._carry_forward_committed(fresh, committed)
    by_id = {e["id"]: e for e in batch}
    kept = by_id["groq/llama3-9b-9192"]
    assert kept["aliases"] == ["llama3-9b-9192"]
    assert json.loads(kept["metadata"])["upstream_status"] == "removed"
    assert claims["llama3-9b-9192"] == "groq/llama3-9b-9192"
    assert by_id["openai/gpt-4o-test"]["aliases"] == ["gpt-4o-test"]


def test_carry_forward_respects_core_skip(cf_mod):
    """An id core.yaml suppresses (skip_ids / skip_source_ids) is NOT carried
    forward — curation removed it from the source universe on purpose."""
    cf_mod.CORE_PATH.write_text(
        yaml.safe_dump({"skip_source_ids": ["groq/llama3-9b-9192"], "entries": []})
    )
    committed = [_cf_entry("groq/llama3-9b-9192", ["llama3-9b-9192"])]
    batch, _claims = cf_mod._carry_forward_committed([], committed)
    assert batch == []


def test_carry_forward_unions_committed_aliases_back(cf_mod):
    """An alias-level upstream removal (cerebras dropping `llama3.1-8b`) must
    not regress resolution: the committed alias is unioned back onto the
    surviving entry, keeping its platform provenance."""
    committed = [_cf_entry(
        "meta/llama3.1-8b-cftest",
        ["llama3.1-8b-cftest"],
        metadata=json.dumps(
            {"alias_platforms": {"llama3.1-8b-cftest": "cerebras"}}, sort_keys=True
        ),
    )]
    fresh = [_cf_entry("meta/llama3.1-8b-cftest", ["llama-3.1-8b-cftest-new"])]
    batch, _claims = cf_mod._carry_forward_committed(fresh, committed)
    (e,) = batch
    assert e["aliases"] == ["llama-3.1-8b-cftest-new", "llama3.1-8b-cftest"]
    assert json.loads(e["metadata"])["alias_platforms"]["llama3.1-8b-cftest"] == "cerebras"


def test_new_mint_cannot_steal_retained_alias(cf_mod, tmp_path):
    """The TEE/gemma4-31b class: upstream re-spells a model under a serving-org
    prefix and the old entry drops out. The fresh mint must NOT capture the
    carried-forward entry's alias claims — the committed owner keeps them
    through reconcile's pre-seeded `claimed` map."""
    core = tmp_path / "sources-core.yaml"
    core.write_text(yaml.safe_dump([]))
    committed = [_cf_entry("google/gemma-9-31B", ["gemma9-31b", "gemma9:31b"])]
    fresh = [_cf_entry("TEE/gemma9-31b", ["gemma9-31b", "gemma9:31b"])]
    batch, claims = cf_mod._carry_forward_committed(fresh, committed)
    out = cf_mod.reconcile_generated_against_existing(
        batch, sources=(core,), committed_claims=claims
    )
    by_id = {e["id"]: e for e in out}
    assert set(by_id["google/gemma-9-31B"]["aliases"]) >= {"gemma9-31b", "gemma9:31b"}
    assert not set(by_id["TEE/gemma9-31b"].get("aliases") or []) & {"gemma9-31b", "gemma9:31b"}


def test_carry_forward_absorbs_serving_prefixed_committed_mint(cf_mod):
    """A committed mint under a serving-host prefix (the pre-strip `TEE/...`
    uploader-org bug) is absorbed onto the fresh entry that now carries that
    id as an alias on the stripped target — never retained as a `TEE/`
    canonical."""
    committed = [_cf_entry("TEE/zmodel9-70b", ["zmodel9-70b"])]
    fresh = [_cf_entry("meta/zmodel9-70b", ["TEE/zmodel9-70b", "zmodel9-70b"])]
    batch, claims = cf_mod._carry_forward_committed(fresh, committed)
    assert [e["id"] for e in batch] == ["meta/zmodel9-70b"]
    (e,) = batch
    assert {"TEE/zmodel9-70b", "zmodel9-70b"} <= set(e["aliases"])
    assert claims["TEE/zmodel9-70b"] == "meta/zmodel9-70b"
    assert claims["zmodel9-70b"] == "meta/zmodel9-70b"


def test_tee_stripped_mint_folds_onto_claims_owner_with_weak_donation(cf_mod, tmp_path):
    """The live `TEE/gemma4-31b` class: after the serving-prefix strip the
    group mints under the name-derived org; that mint folds onto the existing
    canonical that claims the form, the TEE/ raw form rides along as an alias,
    and the mint's scalars donate weakly."""
    core = tmp_path / "sources-core.yaml"
    core.write_text(yaml.safe_dump([{
        "id": "google/gemma-9X-31B", "display_name": "Gemma 9X 31B",
        "aliases": ["google/gemma9x-31b"],
    }]))
    fresh = [{
        "id": "google/gemma9x-31b", "display_name": "Gemma 9X 31B",
        "org_id": "google", "aliases": ["TEE/gemma9x-31b", "gemma9x-31b"],
        "release_date": "2026-04-04", "open_weights": True,
        "metadata": "{}", "review_status": "draft", "resolution_source": "models_dev",
    }]
    out = cf_mod.reconcile_generated_against_existing(fresh, sources=(core,))
    by_id = {e["id"]: e for e in out}
    assert "google/gemma9x-31b" not in by_id
    rec = by_id["google/gemma-9X-31B"]
    assert {"TEE/gemma9x-31b", "gemma9x-31b", "google/gemma9x-31b"} <= set(rec["aliases"])
    assert rec["weak"] == {"release_date": "2026-04-04", "open_weights": True}


def test_carry_forward_unions_replaced_display_name_back(cf_mod):
    """A committed display_name is a loader-promoted global alias: when today's
    upstream re-spells it, the old display must survive as an alias on the
    surviving entry (e.g. `IBM: Granite 4.0 Micro` after openrouter renamed)."""
    committed = [dict(_cf_entry("ibm/granite-cftest-micro", ["granite-cftest-micro"]),
                      display_name="IBM: Granite CFTest Micro")]
    fresh = [dict(_cf_entry("ibm/granite-cftest-micro", ["granite-cftest-micro"]),
                  display_name="Granite CFTest Micro")]
    batch, claims = cf_mod._carry_forward_committed(fresh, committed)
    (e,) = batch
    assert "IBM: Granite CFTest Micro" in e["aliases"]
    assert claims["IBM: Granite CFTest Micro"] == "ibm/granite-cftest-micro"


def test_carry_forward_reappearing_id_clears_removed_tag(cf_mod):
    """A previously-retained entry (`upstream_status: removed`) whose id
    reappears upstream is REPLACED by the fresh entry: exactly one record,
    tag cleared, committed aliases unioned back."""
    committed = [dict(
        _cf_entry("groq/llama3-9b-9192", ["llama3-9b-9192"]),
        metadata=json.dumps({"upstream_status": "removed"}, sort_keys=True),
    )]
    fresh = [_cf_entry("groq/llama3-9b-9192")]
    batch, claims = cf_mod._carry_forward_committed(fresh, committed)
    assert [e["id"] for e in batch] == ["groq/llama3-9b-9192"]
    (e,) = batch
    assert "llama3-9b-9192" in e["aliases"]
    assert "upstream_status" not in json.loads(e["metadata"])
    assert claims["llama3-9b-9192"] == "groq/llama3-9b-9192"


def test_carry_forward_respelled_reappearance_committed_id_wins(cf_mod, tmp_path):
    """The Veo class under the STABILITY RULE: upstream drops
    `zorgvid/Zeo-3-1` and re-emits the model respelled (`zorgvid/zeo3-1`).
    The COMMITTED id wins — the fresh entry is renamed back and today's
    spelling becomes an alias — so the daily cron can never thrash canonical
    ids on upstream respells. Never a normalized twin for the seed's
    collision_fold / gate to trip on. Deterministic."""
    core = tmp_path / "sources-core.yaml"
    core.write_text("entries: []\n")
    committed = [dict(_cf_entry("zorgvid/Zeo-3-1", ["zeo-3-1-alias"]),
                      display_name="Zeo-3.1-Fast")]
    fresh = [_cf_entry("zorgvid/zeo3-1")]
    batch, claims = cf_mod._carry_forward_committed([dict(e) for e in fresh], committed)
    assert [e["id"] for e in batch] == ["zorgvid/Zeo-3-1"], "expected a single survivor"
    (e,) = batch
    assert {"zorgvid/zeo3-1", "Zeo-3.1-Fast", "zeo-3-1-alias"} <= set(e["aliases"])
    assert "upstream_status" not in json.loads(e["metadata"])
    # Claims point at the SURVIVOR, so reconcile's hygiene keeps the donated forms.
    assert claims["zorgvid/Zeo-3-1"] == "zorgvid/Zeo-3-1"
    assert claims["Zeo-3.1-Fast"] == "zorgvid/Zeo-3-1"
    out = cf_mod.reconcile_generated_against_existing(
        batch, sources=(core,), committed_claims=claims
    )
    (surv,) = [x for x in out if x["id"] == "zorgvid/Zeo-3-1"]
    assert {"zorgvid/zeo3-1", "Zeo-3.1-Fast", "zeo-3-1-alias"} <= set(surv["aliases"])
    batch2, claims2 = cf_mod._carry_forward_committed([dict(e) for e in fresh], committed)
    assert batch2 == batch and claims2 == claims


def test_carry_forward_respelled_reappearance_migration_flag_inverts(cf_mod, tmp_path):
    """With the one-shot `--adopt-openrouter-ids-migration` flag the FRESH
    (externally-adopted) spelling wins the twin match and the committed id,
    display_name AND aliases all become aliases on it — the pre-stability
    behavior, reserved for the deliberate migration run. The absorbed forms
    must then survive THROUGH the reconcile hygiene pass too (never delete a
    resolvable surface form on a migration run)."""
    core = tmp_path / "sources-core.yaml"
    core.write_text("entries: []\n")
    committed = [dict(_cf_entry("zorgvid/Zeo-3-1", ["zeo-3-1-alias"]),
                      display_name="Zeo-3.1-Fast")]
    fresh = [_cf_entry("zorgvid/zeo3-1")]
    batch, claims = cf_mod._carry_forward_committed(
        [dict(e) for e in fresh], committed, adopt_migration=True
    )
    assert [e["id"] for e in batch] == ["zorgvid/zeo3-1"], "expected a single survivor"
    (e,) = batch
    assert {"zorgvid/Zeo-3-1", "Zeo-3.1-Fast", "zeo-3-1-alias"} <= set(e["aliases"])
    assert claims["zorgvid/Zeo-3-1"] == "zorgvid/zeo3-1"
    assert claims["Zeo-3.1-Fast"] == "zorgvid/zeo3-1"
    out = cf_mod.reconcile_generated_against_existing(
        batch, sources=(core,), committed_claims=claims
    )
    (surv,) = [x for x in out if x["id"] == "zorgvid/zeo3-1"]
    assert {"zorgvid/Zeo-3-1", "Zeo-3.1-Fast", "zeo-3-1-alias"} <= set(surv["aliases"]), (
        f"absorbed committed forms must survive migration hygiene: {surv['aliases']}"
    )


def test_carry_forward_stability_repoints_parent_edges(cf_mod):
    """When the stability rule renames a fresh twin back to the committed id,
    sibling entries' parent edges that pointed at the fresh spelling are
    repointed to the surviving committed id (no dangling edge)."""
    committed = [dict(_cf_entry("zorgvid/Zeo-3-1"), display_name="Zeo 3.1")]
    child = dict(_cf_entry("zorgvid/zeo3-1-mini"))
    child["parents"] = [{"id": "zorgvid/zeo3-1", "relationship": "variant", "axis": "tier"}]
    fresh = [_cf_entry("zorgvid/zeo3-1"), child]
    batch, _claims = cf_mod._carry_forward_committed(
        [dict(e) for e in fresh], committed
    )
    by_id = {e["id"]: e for e in batch}
    assert "zorgvid/Zeo-3-1" in by_id and "zorgvid/zeo3-1" not in by_id
    assert by_id["zorgvid/zeo3-1-mini"]["parents"][0]["id"] == "zorgvid/Zeo-3-1"


def test_carry_forward_size_guard_blocks_false_absorb(cf_mod):
    """The twin key strips separators, so `opt-1.3b`/`opt-13b` collide — the
    fold_collisions b-size guard must keep them apart (different models):
    the committed entry is retained, not absorbed."""
    committed = [_cf_entry("acme/opt-1.3b")]
    fresh = [_cf_entry("acme/opt-13b")]
    batch, _claims = cf_mod._carry_forward_committed(fresh, committed)
    by_id = {e["id"]: e for e in batch}
    assert set(by_id) == {"acme/opt-13b", "acme/opt-1.3b"}
    assert json.loads(by_id["acme/opt-1.3b"]["metadata"])["upstream_status"] == "removed"


def test_twin_picker_prefers_variant_preserving_identity(cf_mod):
    """The broad collision key folds dotted and dashed number sequences into
    one bucket. When that bucket contains both ``qwen3.8`` and ``qwen-3-8``,
    carry-forward must choose the candidate matching the committed identity,
    independent of lexical ordering.
    """
    candidates = [
        _cf_entry("alibaba/qwen-3-8-max"),
        _cf_entry("qwen/qwen3.8-max"),
    ]
    picked = cf_mod._pick_twin_candidate("alibaba/qwen3.8-max", candidates)
    assert picked is not None
    assert picked["id"] == "qwen/qwen3.8-max"


def test_carry_forward_does_not_absorb_enrich_record_onto_mint(cf_mod):
    """An alias-only enrich record's id is ANOTHER canonical — even if a fresh
    mint twin-matches it, it must NOT be absorbed (its id would become a mint's
    alias, stealing the canonical). It is carried verbatim instead."""
    committed = [{"id": "Realorg/Real-Canon-9X", "aliases": ["real-canon-9x-form"]}]
    fresh = [_cf_entry("realorg/real-canon-9x")]
    batch, _claims = cf_mod._carry_forward_committed(fresh, committed)
    by_id = {e["id"]: e for e in batch}
    assert set(by_id) == {"realorg/real-canon-9x", "Realorg/Real-Canon-9X"}
    assert "Realorg/Real-Canon-9X" not in (by_id["realorg/real-canon-9x"].get("aliases") or [])


# ---------------------------------------------------------------------------
# Orphaned enrich records: enriching a vanished owner is meaningless — drop
# loudly rather than survive as a bare canonical forever.
# ---------------------------------------------------------------------------
def test_reconcile_drops_orphaned_enrich_record_loudly(mod, tmp_path, capsys):
    core = tmp_path / "core.yaml"
    core.write_text(yaml.safe_dump({"entries": [{
        "id": "Kept/Owner-X", "display_name": "Owner X", "aliases": [],
    }]}))
    generated = [
        # Owner id exists nowhere (sources, catalog, tier3, core) -> drop + warn.
        {"id": "ghostorg/vanished-owner-xyz", "aliases": ["ghost-form-1"]},
        # Owner still in core -> the enrich record keeps donating onto it.
        {"id": "Kept/Owner-X", "aliases": ["kept-form-1"]},
    ]
    out = mod.reconcile_generated_against_existing([dict(e) for e in generated], sources=(core,))
    by_id = {e["id"]: e for e in out}
    assert "ghostorg/vanished-owner-xyz" not in by_id, "bare canonical must not survive"
    assert "kept-form-1" in by_id["Kept/Owner-X"]["aliases"]
    err = capsys.readouterr().err
    assert "vanished-owner-xyz" in err and "ghost-form-1" in err, (
        "dropping an orphaned enrich record must be loud (forms listed)"
    )


# ---------------------------------------------------------------------------
# Catalog carry-forward: display-name donation on fold, retention/union-back/
# dup-record consolidation, and idempotence over its own output.
# ---------------------------------------------------------------------------
def test_catalog_carry_forward_donates_display_name_on_respelled_fold(mod, tmp_path, monkeypatch):
    """The Veo regression shape under the STABILITY RULE: the committed
    catalog mint `…/zeo-3-1` (display_name `Zeo-3.1-Fast`) vanished upstream
    and reappeared respelled as `…/zeo3-1`. The COMMITTED id wins — the fresh
    spelling and its display all stay resolvable as aliases on it — and no
    normalized twin survives."""
    committed = [{
        "id": "zorgvid/zeo-3-1", "display_name": "Zeo-3.1-Fast", "org_id": None,
        "aliases": ["zeo-3-1"],
        "metadata": json.dumps({"alias_platforms": {"zeo-3-1": "poe"}}, sort_keys=True),
        "review_status": "draft", "resolution_source": "models_dev",
    }]
    cat = tmp_path / "cat.yaml"
    cat.write_text(yaml.safe_dump(committed, sort_keys=False))
    monkeypatch.setattr(mod, "CATALOG_OUT_PATH", cat)
    monkeypatch.setattr(mod, "ORGS_GENERATED_PATH", tmp_path / "orgs.yaml")
    fresh = [{
        "id": "zorgvid/zeo3-1", "display_name": "Zeo 3.1", "org_id": None,
        "aliases": [], "metadata": "{}",
        "review_status": "draft", "resolution_source": "models_dev",
    }]
    mod.regenerate_catalog([dict(e) for e in fresh])
    out = yaml.safe_load(cat.read_text()) or []
    assert "zorgvid/zeo3-1" not in {e["id"] for e in out}, "respelled twin must not survive"
    (surv,) = [e for e in out if e["id"] == "zorgvid/zeo-3-1"]
    assert {"zorgvid/zeo3-1", "zeo-3-1"} <= set(surv["aliases"]), (
        f"surface forms lost on fold: {surv['aliases']}"
    )
    # Platform provenance for the re-added committed alias survives too.
    assert json.loads(surv["metadata"])["alias_platforms"]["zeo-3-1"] == "poe"


def test_catalog_carry_forward_respelled_fold_migration_flag_inverts(mod, tmp_path, monkeypatch):
    """Same shape with `adopt_migration=True` (the one-shot adoption run): the
    FRESH spelling wins and the committed id + display_name become aliases —
    display_name was the one form the fold used to delete."""
    committed = [{
        "id": "zorgvid/zeo-3-1", "display_name": "Zeo-3.1-Fast", "org_id": None,
        "aliases": ["zeo-3-1"],
        "metadata": json.dumps({"alias_platforms": {"zeo-3-1": "poe"}}, sort_keys=True),
        "review_status": "draft", "resolution_source": "models_dev",
    }]
    cat = tmp_path / "cat.yaml"
    cat.write_text(yaml.safe_dump(committed, sort_keys=False))
    monkeypatch.setattr(mod, "CATALOG_OUT_PATH", cat)
    monkeypatch.setattr(mod, "ORGS_GENERATED_PATH", tmp_path / "orgs.yaml")
    fresh = [{
        "id": "zorgvid/zeo3-1", "display_name": "Zeo 3.1", "org_id": None,
        "aliases": [], "metadata": "{}",
        "review_status": "draft", "resolution_source": "models_dev",
    }]
    mod.regenerate_catalog([dict(e) for e in fresh], adopt_migration=True)
    out = yaml.safe_load(cat.read_text()) or []
    assert "zorgvid/zeo-3-1" not in {e["id"] for e in out}, "respelled twin must not survive"
    (surv,) = [e for e in out if e["id"] == "zorgvid/zeo3-1"]
    assert {"zorgvid/zeo-3-1", "Zeo-3.1-Fast", "zeo-3-1"} <= set(surv["aliases"]), (
        f"committed surface forms lost on fold: {surv['aliases']}"
    )
    assert json.loads(surv["metadata"])["alias_platforms"]["zeo-3-1"] == "poe"


def test_catalog_carry_forward_retention_unionback_and_dup_consolidation(mod, tmp_path, monkeypatch):
    """Direct test of the catalog carry-forward block: a committed file with
    TWO records for one id (mint + enrich dup) consolidates onto the single
    fresh mint (union of all aliases, exactly one record); a committed mint
    with no surviving target is retained tagged `upstream_status: removed`."""
    committed = [
        {"id": "zorg/zmodel-cf", "display_name": "ZModel CF", "org_id": None,
         "aliases": ["zmodel-cf-a1"], "metadata": "{}",
         "review_status": "draft", "resolution_source": "models_dev"},
        {"id": "zorg/zmodel-cf", "aliases": ["zmodel-cf-a2"]},  # dup record, same id
        {"id": "zorg/zgone-cf", "display_name": "ZGone CF", "org_id": None,
         "aliases": ["zgone-cf-alias"], "metadata": "{}",
         "review_status": "draft", "resolution_source": "models_dev"},
    ]
    cat = tmp_path / "cat.yaml"
    cat.write_text(yaml.safe_dump(committed, sort_keys=False))
    monkeypatch.setattr(mod, "CATALOG_OUT_PATH", cat)
    monkeypatch.setattr(mod, "ORGS_GENERATED_PATH", tmp_path / "orgs.yaml")
    fresh = [{
        "id": "zorg/zmodel-cf", "display_name": "ZModel CF", "org_id": None,
        "aliases": ["zmodel-cf-a3"], "metadata": "{}",
        "review_status": "draft", "resolution_source": "models_dev",
    }]
    mod.regenerate_catalog(fresh)
    out = yaml.safe_load(cat.read_text()) or []
    recs = [e for e in out if e["id"] == "zorg/zmodel-cf"]
    assert len(recs) == 1, f"dup committed records must consolidate, got {len(recs)}"
    assert {"zmodel-cf-a1", "zmodel-cf-a2", "zmodel-cf-a3"} <= set(recs[0]["aliases"])
    (gone,) = [e for e in out if e["id"] == "zorg/zgone-cf"]
    assert json.loads(gone["metadata"])["upstream_status"] == "removed"
    assert "zgone-cf-alias" in gone["aliases"]


def test_catalog_consolidates_enrich_records_per_owner(mod, tmp_path, monkeypatch):
    """Two donors folding onto the SAME owner emit exactly ONE enrich record:
    aliases set-union, alias_platforms per-key union (first writer wins), weak
    per-field first-wins — the loader's tie-break order."""
    src = tmp_path / "src-core.yaml"
    src.write_text(yaml.safe_dump([{
        "id": "zorg/ZModel-9B", "display_name": "ZModel 9B",
        "aliases": ["zorg/zmodel-9b", "zmodel-9b-v0"],
    }]))
    monkeypatch.setattr(mod, "_CATALOG_EXISTING_SOURCES", (src,))
    cat = tmp_path / "cat.yaml"
    cat.write_text(yaml.safe_dump([]))
    monkeypatch.setattr(mod, "CATALOG_OUT_PATH", cat)
    monkeypatch.setattr(mod, "ORGS_GENERATED_PATH", tmp_path / "orgs.yaml")
    full = [
        {"id": "zorg/zmodel-9b", "display_name": "ZModel 9B A", "org_id": "zorg",
         "aliases": ["zmodel-9b-alias1"],
         "metadata": json.dumps({"alias_platforms": {"zmodel-9b-alias1": "poe"}}),
         "release_date": "2024-01-01", "open_weights": True,
         "review_status": "draft", "resolution_source": "models_dev"},
        {"id": "zmodel-9b-v0", "display_name": "ZModel 9B B", "org_id": "zorg",
         "aliases": ["zmodel-9b-alias2"],
         "metadata": json.dumps({"alias_platforms": {"zmodel-9b-alias2": "novita-ai"}}),
         "release_date": "2024-02-02",
         "review_status": "draft", "resolution_source": "models_dev"},
    ]
    mod.regenerate_catalog(full)
    out = yaml.safe_load(cat.read_text()) or []
    recs = [e for e in out if e["id"] == "zorg/ZModel-9B"]
    assert len(recs) == 1, f"expected one consolidated enrich record, got {len(recs)}"
    rec = recs[0]
    assert {"zmodel-9b-alias1", "zmodel-9b-alias2"} <= set(rec["aliases"])
    ap = json.loads(rec["metadata"])["alias_platforms"]
    assert ap == {"zmodel-9b-alias1": "poe", "zmodel-9b-alias2": "novita-ai"}
    assert rec["weak"]["release_date"] == "2024-01-01"  # first donor wins per field
    assert rec["weak"]["open_weights"] is True


def test_catalog_pipeline_idempotent_over_committed_output(mod, api, tmp_path, monkeypatch):
    """Run the --catalog step over the COMMITTED catalog, then AGAIN over its
    own output: byte-identical. This is the daily-cron regression class
    (oscillating carry-forward/split output)."""
    import copy

    mod._HF_AUTHORITY = None
    full = mod._finalize_entries(mod._generate_models(api, mod._load_known_org_ids())[0])
    cat = tmp_path / "cat.yaml"
    cat.write_text((REPO_ROOT / "seed" / "models" / "sources" / "models_dev_catalog.generated.yaml").read_text())
    orgs = tmp_path / "orgs.yaml"
    orgs.write_text((REPO_ROOT / "seed" / "orgs.generated.yaml").read_text())
    monkeypatch.setattr(mod, "CATALOG_OUT_PATH", cat)
    monkeypatch.setattr(mod, "ORGS_GENERATED_PATH", orgs)
    mod.regenerate_catalog(copy.deepcopy(full))
    text1 = cat.read_text()
    mod.regenerate_catalog(copy.deepcopy(full))
    assert cat.read_text() == text1, "catalog regen not idempotent over its own output"


def test_noncatalog_pipeline_idempotent_over_own_output(mod, api):
    """Run the full non-catalog pipeline (carry-forward + reconcile + org-canon
    + write) over the COMMITTED file, then AGAIN over its own output:
    byte-identical."""
    mod._HF_AUTHORITY = None

    def run(committed):
        gen, skipped = mod._generate_models(api, mod._load_known_org_ids())
        assert skipped == []
        gen = mod._finalize_entries([dict(e) for e in gen])
        batch, claims = mod._carry_forward_committed(gen, committed)
        out = mod.reconcile_generated_against_existing(batch, committed_claims=claims)
        _canon = mod._build_org_canonicalizer()
        for e in out:
            for f in ("org_id", "lineage_origin_model_org_id"):
                if e.get(f):
                    e[f] = _canon(e[f])
        return mod._write_yaml(out, mod.SEED_PATH)  # returns text; does not write

    text1 = run(mod._catalog_load_list(mod.SEED_PATH))
    text2 = run(yaml.safe_load(text1))
    assert text1 == text2, "non-catalog regen not idempotent over its own output"


# ---------------------------------------------------------------------------
# --reconcile-orgs must never WRITE core.yaml (a YAML re-dump hoists its inline
# comments); core stays in the READ set and mis-spellings are warned loudly.
# ---------------------------------------------------------------------------
def test_reconcile_orgs_never_writes_core_by_default(mod, tmp_path, monkeypatch, capsys):
    gen_orgs = yaml.safe_load((REPO_ROOT / "seed" / "orgs.generated.yaml").read_text()) or []
    _canon = mod._build_org_canonicalizer()
    # An existing community org whose lowercased spelling the shared
    # canonicalizer maps back to it — so a core entry carrying the lowercase
    # variant genuinely NEEDS the rewrite.
    existing = next(
        e["id"] for e in gen_orgs
        if isinstance(e, dict) and e.get("id") and "/" not in e["id"]
        and e["id"].lower() != e["id"] and _canon(e["id"].lower()) == e["id"]
    )
    variant = existing.lower()
    core = tmp_path / "core.yaml"
    core_text = (
        "# hand-curated header comment that a re-dump would hoist\n"
        + yaml.safe_dump(
            {"skip_ids": [], "entries": [
                {"id": f"{variant}/zzz-core-org-write-test", "org_id": variant}
            ]},
            sort_keys=False,
        )
    )
    core.write_text(core_text)
    monkeypatch.setattr(mod, "CORE_PATH", core)
    monkeypatch.setattr(mod, "_ALL_MODEL_SOURCES", (core,))

    n = mod.canonicalize_model_org_ids()  # cron default: no core write
    assert core.read_text() == core_text, "the cron must never rewrite core.yaml"
    assert n == 0
    out = capsys.readouterr().out
    assert "::warning::" in out and variant in out, "needed core fix must be loud"

    n2 = mod.canonicalize_model_org_ids(write_core=True)  # one-shot opt-in
    assert n2 == 1
    rewritten = yaml.safe_load(core.read_text())
    assert rewritten["entries"][0]["org_id"] == existing


# ---------------------------------------------------------------------------
# Scalar enrichment flow: a suppressed/folded/skipped mint's non-empty scalars
# travel onto the owner's enrich record under `weak:` (the seed loader fills
# only still-empty, non-core-claimed fields with them) instead of being
# dropped on the floor. Parents edges and mint-specific metadata stay behind.
# ---------------------------------------------------------------------------
def test_reconciliation_donates_suppressed_mint_scalars(mod, tmp_path):
    core = tmp_path / "core.yaml"
    core.write_text(yaml.safe_dump([
        {"id": "Foo/Bar-7B", "display_name": "Foo Bar 7B", "aliases": ["foo-bar-7b"]},
    ]))
    generated = [
        {
            "id": "foo/bar-7b",
            "display_name": "foo bar 7b",
            "aliases": ["foo-bar-7b-extra-alias"],
            "release_date": "2025-12-01",
            "open_weights": True,
            "params_billions": None,
            "input_modalities": ["text"],
            "output_modalities": ["video"],
            "parents": [{"id": "foo/bar", "relationship": "variant", "axis": "size"}],
            "metadata": json.dumps({"providers": ["fastrouter"], "underlying_key": "bar-7b"}),
        },
    ]
    out = mod.reconcile_generated_against_existing(generated, sources=(core,))
    by_id = {e["id"]: e for e in out}
    assert "foo/bar-7b" not in by_id
    rec = by_id["Foo/Bar-7B"]
    assert rec["weak"] == {
        "release_date": "2025-12-01",
        "open_weights": True,
        "input_modalities": ["text"],
        "output_modalities": ["video"],
    }
    assert "display_name" not in rec, "enrich records must stay display_name-less"
    assert "parents" not in rec, "parents edges must NOT be donated (dangling-edge risk)"
    assert "metadata" not in rec, "mint-specific metadata must NOT be donated"


def test_carry_forward_skip_donates_scalars_onto_claiming_owner(cf_mod, tmp_path):
    """skip_source_ids fold: a core-skipped committed entry whose surface form
    an existing canonical claims (here: core aliasing the skipped mint) donates
    its scalars as a weak enrich record onto that owner — its aliases stay
    suppressed (that is what skip_source_ids curates away)."""
    cf_mod.CORE_PATH.write_text(
        yaml.safe_dump({"skip_source_ids": ["wanx/wan-v2-6"], "entries": []})
    )
    existing = tmp_path / "existing.yaml"
    existing.write_text(yaml.safe_dump([
        {"id": "alibaba/wan-v2-6", "display_name": "Wan 2.6",
         "aliases": ["wanx/wan-v2-6", "wan-v2-6"]},
    ]))
    cf_mod._NONCATALOG_EXISTING_SOURCES = (existing,)
    committed = [{
        "id": "wanx/wan-v2-6",
        "display_name": "Wan 2.6",
        "aliases": ["wan-v2-6"],
        "release_date": "2025-12-01",
        "open_weights": True,
        "metadata": "{}",
    }]
    batch, _claims = cf_mod._carry_forward_committed([], committed)
    assert batch == [
        {"id": "alibaba/wan-v2-6",
         "weak": {"release_date": "2025-12-01", "open_weights": True}}
    ]


def test_carry_forward_skip_unclaimed_stays_pure_skip(cf_mod, tmp_path):
    """A core-skipped entry whose surface form nothing claims keeps the current
    behavior: pure skip, even when it carries scalars."""
    cf_mod.CORE_PATH.write_text(
        yaml.safe_dump({"skip_source_ids": ["lonely/mint-1b"], "entries": []})
    )
    empty = tmp_path / "existing.yaml"
    empty.write_text("[]\n")
    cf_mod._NONCATALOG_EXISTING_SOURCES = (empty,)
    committed = [{
        "id": "lonely/mint-1b", "display_name": "Lonely Mint 1B",
        "release_date": "2024-01-01", "metadata": "{}",
    }]
    batch, _claims = cf_mod._carry_forward_committed([], committed)
    assert batch == []


def test_catalog_enrich_record_carries_donor_scalars(mod, tmp_path, monkeypatch):
    """The --catalog split's HF-present fold (`_enrich_target`) donates the
    folded record's scalars under `weak:` on the alias-only enrichment."""
    core = tmp_path / "core.yaml"
    core.write_text(yaml.safe_dump({
        "skip_source_ids": [],
        "entries": [
            {"id": "alibaba/zzz-wan-v9", "display_name": "Zzz Wan V9",
             "aliases": ["wanx/zzz-wan-v9", "zzz-wan-v9"], "open_weights": None},
        ],
    }))
    cat_out = tmp_path / "cat.yaml"
    orgs_out = tmp_path / "orgs.yaml"
    orgs_out.write_text((REPO_ROOT / "seed" / "orgs.generated.yaml").read_text())
    monkeypatch.setattr(mod, "CORE_PATH", core)
    monkeypatch.setattr(mod, "_CATALOG_EXISTING_SOURCES", (core,))
    monkeypatch.setattr(mod, "CATALOG_OUT_PATH", cat_out)
    monkeypatch.setattr(mod, "ORGS_GENERATED_PATH", orgs_out)
    full = [{
        "id": "wanx/zzz-wan-v9",
        "display_name": "Zzz Wan V9",
        "org_id": "wanx",
        "aliases": ["zzz-wan-v9"],
        "release_date": "2025-12-01",
        "open_weights": True,
        "metadata": "{}",
        "review_status": "draft",
    }]
    mod.regenerate_catalog([dict(e) for e in full])
    cat = yaml.safe_load(cat_out.read_text()) or []
    rec = next(e for e in cat if e.get("id") == "alibaba/zzz-wan-v9")
    assert rec["weak"] == {"release_date": "2025-12-01", "open_weights": True}
    assert "display_name" not in rec


def test_catalog_committed_skip_donates_scalars(mod, tmp_path, monkeypatch):
    """The --catalog carry-forward applies the same skip_source_ids rule: a
    core-skipped committed mint donates its scalars (no aliases) weakly onto
    the canonical claiming its surface form."""
    core = tmp_path / "core.yaml"
    core.write_text(yaml.safe_dump({
        "skip_source_ids": ["wanx/zzz-wan-v9"],
        "entries": [
            {"id": "alibaba/zzz-wan-v9", "display_name": "Zzz Wan V9",
             "aliases": ["wanx/zzz-wan-v9", "zzz-wan-v9"], "open_weights": None},
        ],
    }))
    cat_out = tmp_path / "cat.yaml"
    cat_out.write_text(yaml.safe_dump([
        {"id": "wanx/zzz-wan-v9", "display_name": "Zzz Wan V9",
         "aliases": ["zzz-wan-v9"], "release_date": "2025-11-30",
         "open_weights": True, "metadata": "{}"},
    ]))
    orgs_out = tmp_path / "orgs.yaml"
    orgs_out.write_text((REPO_ROOT / "seed" / "orgs.generated.yaml").read_text())
    monkeypatch.setattr(mod, "CORE_PATH", core)
    monkeypatch.setattr(mod, "_CATALOG_EXISTING_SOURCES", (core,))
    monkeypatch.setattr(mod, "CATALOG_OUT_PATH", cat_out)
    monkeypatch.setattr(mod, "ORGS_GENERATED_PATH", orgs_out)
    mod.regenerate_catalog([])
    cat = yaml.safe_load(cat_out.read_text()) or []
    assert cat == [
        {"id": "alibaba/zzz-wan-v9",
         "weak": {"release_date": "2025-11-30", "open_weights": True}}
    ], "skipped committed mint's scalars must flow weakly onto the owner, aliases suppressed"


# ---------------------------------------------------------------------------
# Idempotence / donation-hole pins (P0 alignment pass): the sites that made
# the non-catalog pipeline lose forms or desync provenance, each pinned in
# isolation. Ownership evidence is TIERED — independent sources (hf_oracle /
# hub_stats / core / enrichments) are authoritative; a claim whose only
# evidence is a derived file (catalog / tier3), made by the suppressed mint
# itself or by an id committed_claims assigns to the owner, is the previous
# cron's echo and never vetoes a donation.
# ---------------------------------------------------------------------------

def test_hygiene_strips_metadata_alias_platforms_with_foreign_alias(mod, tmp_path, monkeypatch):
    """Metadata/alias desync pin: when the survivor hygiene strips a
    foreign-owned alias, the matching metadata.alias_platforms key must go
    with it. Persisting an ap key without its alias violates the
    `ap keys SUBSET-OF aliases` invariant and made run N+1 differ from run N
    (the next carry-forward unions provenance only for forms in `aliases`)."""
    src = tmp_path / "src.yaml"
    src.write_text(yaml.safe_dump([
        {"id": "zother/ZOwner-1B", "display_name": "ZOwner 1B",
         "aliases": ["zshared-form"]},
    ]))
    batch = [{
        "id": "zorg/zmine-model", "display_name": "ZMine Model", "org_id": "zorg",
        "aliases": ["zmine-a", "zshared-form"],
        "metadata": json.dumps(
            {"alias_platforms": {"zmine-a": "kilo", "zshared-form": "poe"}},
            sort_keys=True),
        "review_status": "draft", "resolution_source": "models_dev",
    }]
    out = mod.reconcile_generated_against_existing(batch, sources=(src,))
    (surv,) = [e for e in out if e["id"] == "zorg/zmine-model"]
    assert surv["aliases"] == ["zmine-a"], "foreign alias must be stripped"
    ap = json.loads(surv["metadata"])["alias_platforms"]
    assert ap == {"zmine-a": "kilo"}, (
        f"metadata.alias_platforms must stay in lockstep with aliases: {ap}"
    )
    assert set(ap) <= set(surv["aliases"])


def test_suppressed_mint_donates_forms_despite_stale_catalog_echo(mod, tmp_path, monkeypatch):
    """Donation-hole pin (the `deepseek-chat-v3-1` regression shape): a mint
    suppressed onto an independent-source canonical must donate its surface
    forms EVEN WHEN the stale committed catalog still echoes the mint under
    its own id (the previous cron's output, rewritten right after). Under the
    un-tiered guard the echo read as a foreign owner, nothing was donated, and
    the subsequent catalog rewrite dropped its own orphaned record — deleting
    the forms from every source."""
    src = tmp_path / "src.yaml"
    src.write_text(yaml.safe_dump([
        {"id": "zorg/zchat-v3.1", "display_name": "ZChat v3.1"},
    ]))
    echo = tmp_path / "stale-catalog.yaml"
    echo.write_text(yaml.safe_dump([
        {"id": "zorg/zchat-v3-1", "aliases": ["zchat-v3-1"]},
    ]))
    monkeypatch.setattr(mod, "CATALOG_OUT_PATH", echo)
    batch = [{
        "id": "zorg/zchat-v3-1", "display_name": "zchat-v3-1", "org_id": "zorg",
        "aliases": ["zchat-v3-1"], "metadata": "{}",
        "release_date": "2026-05-28", "open_weights": False,
        "review_status": "draft", "resolution_source": "models_dev",
    }]
    out = mod.reconcile_generated_against_existing(batch, sources=(src,))
    ids = {e["id"] for e in out}
    assert "zorg/zchat-v3-1" not in ids, "normalized-colliding mint must be suppressed"
    (rec,) = [e for e in out if e["id"] == "zorg/zchat-v3.1"]
    assert "zchat-v3-1" in (rec.get("aliases") or []), (
        "suppressed mint's forms must be donated onto the owner despite the "
        f"stale catalog echo: {rec}"
    )
    assert (rec.get("weak") or {}).get("release_date") == "2026-05-28", (
        "suppressed mint's scalars must flow weakly onto the owner"
    )


def test_suppressed_mint_donates_despite_full_stale_catalog_echo(mod, tmp_path, monkeypatch):
    """V2b: same as the donation-hole pin but the stale echo is a FULL record
    (display_name-bearing) — a full echo lands in `owner_defining`, which is
    exactly the poisoned evidence the tiered guard must ignore."""
    src = tmp_path / "src.yaml"
    src.write_text(yaml.safe_dump([
        {"id": "zorg/zchat-v3.1", "display_name": "ZChat v3.1"},
    ]))
    echo = tmp_path / "stale-catalog.yaml"
    echo.write_text(yaml.safe_dump([
        {"id": "zorg/zchat-v3-1", "display_name": "zchat-v3-1",
         "aliases": ["zchat-v3-1"]},
    ]))
    monkeypatch.setattr(mod, "CATALOG_OUT_PATH", echo)
    batch = [{
        "id": "zorg/zchat-v3-1", "display_name": "zchat-v3-1", "org_id": "zorg",
        "aliases": ["zchat-v3-1"], "metadata": "{}",
        "release_date": "2026-05-28", "open_weights": False,
        "review_status": "draft", "resolution_source": "models_dev",
    }]
    out = mod.reconcile_generated_against_existing(batch, sources=(src,))
    ids = {e["id"] for e in out}
    assert "zorg/zchat-v3-1" not in ids, "normalized-colliding mint must be suppressed"
    (rec,) = [e for e in out if e["id"] == "zorg/zchat-v3.1"]
    assert "zchat-v3-1" in (rec.get("aliases") or []), (
        f"donation must survive a FULL stale echo: {rec}"
    )


def test_suppressed_mint_donates_with_claims_escape(mod, tmp_path, monkeypatch):
    """V2c: committed_claims maps the echoed id onto the owner (the
    carry-forward absorbed it this run) — the claims escape hatch must be
    reachable, not short-circuited by `owner_defining`."""
    src = tmp_path / "src.yaml"
    src.write_text(yaml.safe_dump([
        {"id": "zorg/zchat-v3.1", "display_name": "ZChat v3.1"},
    ]))
    echo = tmp_path / "stale-catalog.yaml"
    echo.write_text(yaml.safe_dump([
        {"id": "zorg/zchat-v3-1", "aliases": ["zchat-v3-1"]},
    ]))
    monkeypatch.setattr(mod, "CATALOG_OUT_PATH", echo)
    batch = [{
        "id": "zorg/zchat-v3-1", "display_name": "zchat-v3-1", "org_id": "zorg",
        "aliases": ["zchat-v3-1"], "metadata": "{}",
        "release_date": "2026-05-28", "open_weights": False,
        "review_status": "draft", "resolution_source": "models_dev",
    }]
    claims = {"zorg/zchat-v3-1": "zorg/zchat-v3.1", "zchat-v3-1": "zorg/zchat-v3.1"}
    out = mod.reconcile_generated_against_existing(
        batch, sources=(src,), committed_claims=claims
    )
    ids = {e["id"] for e in out}
    assert "zorg/zchat-v3-1" not in ids
    (rec,) = [e for e in out if e["id"] == "zorg/zchat-v3.1"]
    assert "zchat-v3-1" in (rec.get("aliases") or []), (
        f"claims escape hatch must allow donation: {rec}"
    )


def test_steady_state_absorbed_fresh_forms_survive_hygiene(mod, tmp_path, monkeypatch):
    """V3s, steady-state direction (committed id wins): the stale catalog
    echoes the COMMITTED id (previous cron). The fresh spelling folds in as an
    alias and every form must survive the hygiene pass.

    NOTE: this pin is a DIRECTION pin, not a discriminating regression pin —
    in this shape the echo's claiming id equals the survivor id, so it passed
    even before the tiered-evidence fix (the discriminating pins for that fix
    are P2/V2b/V2c/V3m). Kept for the steady-state contract it documents."""
    src = tmp_path / "src.yaml"
    src.write_text(yaml.safe_dump([]))
    echo = tmp_path / "stale-catalog.yaml"
    echo.write_text(yaml.safe_dump([
        {"id": "zorgvid/zeo3-1", "aliases": ["zeo3-1", "zeo3-1-x"]},
    ]))
    monkeypatch.setattr(mod, "CATALOG_OUT_PATH", echo)
    committed = [{
        "id": "zorgvid/zeo3-1", "display_name": "zeo3-1", "org_id": None,
        "aliases": ["zeo3-1-x"],
        "metadata": json.dumps({"alias_platforms": {"zeo3-1-x": "fastrouter"}},
                               sort_keys=True),
        "review_status": "draft", "resolution_source": "models_dev",
    }]
    fresh = [{
        "id": "zorgvid/zeo-3-1", "display_name": "Zeo 3.1", "org_id": None,
        "aliases": ["zeo-3-1"], "metadata": "{}",
        "review_status": "draft", "resolution_source": "models_dev",
    }]
    batch, claims = mod._carry_forward_committed(
        [dict(e) for e in fresh], committed
    )
    assert claims.get("zorgvid/zeo3-1") == "zorgvid/zeo3-1", (
        f"steady state: committed id must win: {claims}"
    )
    out = mod.reconcile_generated_against_existing(
        batch, sources=(src,), committed_claims=claims
    )
    (surv,) = [e for e in out if e["id"] == "zorgvid/zeo3-1"]
    assert {"zorgvid/zeo-3-1", "zeo-3-1", "zeo3-1-x"} <= set(surv["aliases"]), (
        f"fresh spelling + committed aliases must survive: {surv['aliases']}"
    )
    ap = json.loads(surv["metadata"])["alias_platforms"]
    assert ap.get("zeo3-1-x") == "fastrouter", "platform provenance must survive"
    assert set(ap) <= set(surv["aliases"])


def test_migration_absorbed_committed_forms_survive_hygiene(mod, tmp_path, monkeypatch):
    """V3m, migration direction (adopt_migration=True; fresh id wins): the
    stale catalog carries a FULL echo of the committed id. The absorbed
    committed forms must survive hygiene on any FUTURE migration run — the
    `owner_defining` veto must not short-circuit the claims escape."""
    src = tmp_path / "src.yaml"
    src.write_text(yaml.safe_dump([]))
    echo = tmp_path / "stale-catalog.yaml"
    echo.write_text(yaml.safe_dump([
        {"id": "zorgvid/zeo3-1", "display_name": "zeo3-1",
         "aliases": ["zeo3-1-x"]},
    ]))
    monkeypatch.setattr(mod, "CATALOG_OUT_PATH", echo)
    committed = [{
        "id": "zorgvid/zeo3-1", "display_name": "zeo3-1", "org_id": None,
        "aliases": ["zeo3-1-x"],
        "metadata": json.dumps({"alias_platforms": {"zeo3-1-x": "fastrouter"}},
                               sort_keys=True),
        "review_status": "draft", "resolution_source": "models_dev",
    }]
    fresh = [{
        "id": "zorgvid/zeo-3-1", "display_name": "Zeo 3.1", "org_id": None,
        "aliases": ["zeo-3-1"], "metadata": "{}",
        "review_status": "draft", "resolution_source": "models_dev",
    }]
    batch, claims = mod._carry_forward_committed(
        [dict(e) for e in fresh], committed, adopt_migration=True
    )
    assert claims.get("zorgvid/zeo3-1") == "zorgvid/zeo-3-1", (
        f"migration: fresh id must win: {claims}"
    )
    out = mod.reconcile_generated_against_existing(
        batch, sources=(src,), committed_claims=claims
    )
    (surv,) = [e for e in out if e["id"] == "zorgvid/zeo-3-1"]
    assert {"zorgvid/zeo3-1", "zeo3-1-x"} <= set(surv["aliases"]), (
        f"absorbed committed forms must survive migration hygiene: {surv['aliases']}"
    )
    ap = json.loads(surv["metadata"])["alias_platforms"]
    assert ap.get("zeo3-1-x") == "fastrouter", "platform provenance must survive"


# ---------------------------------------------------------------------------
# Pure-generator alias emission (synthetic snapshot): the fixpoint/confluence
# contract lets committed aliases self-ratify through carry-forward, so an
# alias-EMISSION regression in the generator would not fail it. Pin the
# emission itself: provider spellings of a grouped serving land as aliases
# with platform provenance, without any carry-forward in the loop. The leaf
# canonical is the ADOPTED OpenRouter key (rung 2 — this generator adopts
# external ids); the invented spelling stays a resolvable alias.
# ---------------------------------------------------------------------------

def test_generator_emits_provider_alias_forms_from_snapshot(mod):
    api = {
        "mistral": {"id": "mistral", "name": "Mistral", "models": {
            "mistral-7b-instruct-v0.2": {
                "id": "mistral-7b-instruct-v0.2", "name": "Mistral 7B Instruct v0.2",
                "release_date": "2023-12-11", "open_weights": True, "family": "mistral",
            },
        }},
        "openrouter": {"id": "openrouter", "name": "OpenRouter", "models": {
            "mistralai/mistral-7b-instruct-v0.2": {
                "id": "mistralai/mistral-7b-instruct-v0.2",
                "name": "Mistral 7B Instruct v0.2",
                "release_date": "2023-12-11", "open_weights": True,
            },
        }},
    }
    out, missing = mod._generate_models(api, {"mistralai"})
    assert missing == []
    out = mod._finalize_entries(out)
    by_id = {e["id"]: e for e in out}
    leaf = by_id["mistralai/mistral-7b-instruct-v0.2"]
    # Every provider spelling of the grouped serving survives as an alias:
    # the slugified bare form, the dotted raw key, and the invented
    # org-prefixed id the adoption renamed away.
    assert leaf["aliases"] == [
        "mistral-7b-instruct-v0-2",
        "mistral-7b-instruct-v0.2",
        "mistralai/mistral-7b-instruct-v0-2",
    ]
    meta = json.loads(leaf["metadata"])
    assert meta["openrouter_adopted"] is True
    ap = meta["alias_platforms"]
    assert ap == {
        "mistral-7b-instruct-v0-2": "mistral",
        "mistral-7b-instruct-v0.2": "mistral",
        "mistralai/mistral-7b-instruct-v0-2": "mistral",
    }


# ---------------------------------------------------------------------------
# RUNG-MONOTONE stability: a fresh twin from a strictly higher rung of the id
# ladder (hf_deferred = real HF id, rung 1) is never renamed back to a
# committed non-HF id; same-rung twins keep the committed id. Both carry-
# forward paths (non-catalog + catalog) enforce it; rung 2 over 3 (OpenRouter
# key over invented) stays migration-gated with a visible pending counter.
# ---------------------------------------------------------------------------

def test_carry_forward_hf_deferred_twin_never_renamed_back(cf_mod):
    committed = [
        dict(_cf_entry("zorgx/zmod-3-1"), display_name="ZMod 3.1"),
        # A retained committed child pointing at the absorbed id: its parent
        # edge must be repointed onto the surviving HF id.
        dict(_cf_entry("zorgx/zmod-3-1-mini"),
             parents=[{"id": "zorgx/zmod-3-1", "relationship": "variant",
                       "axis": "tier"}]),
    ]
    fresh = [dict(
        _cf_entry("zorgx/ZMod3-1"),
        aliases=["zmod3-1"],
        metadata=json.dumps({"hf_deferred": True}, sort_keys=True),
    )]
    batch, claims = cf_mod._carry_forward_committed([dict(e) for e in fresh], committed)
    by_id = {e["id"]: e for e in batch}
    assert "zorgx/ZMod3-1" in by_id, "HF-deferred id must survive as canonical"
    assert "zorgx/zmod-3-1" not in by_id, "committed invented twin must be absorbed"
    surv = by_id["zorgx/ZMod3-1"]
    assert "zorgx/zmod-3-1" in surv["aliases"], "absorbed committed id stays resolvable"
    assert json.loads(surv["metadata"])["hf_deferred"] is True
    assert claims["zorgx/zmod-3-1"] == "zorgx/ZMod3-1"
    # Parent-edge repointing, mirror of the rename path's.
    assert by_id["zorgx/zmod-3-1-mini"]["parents"][0]["id"] == "zorgx/ZMod3-1"


def test_carry_forward_promote_then_rename_chain_composes(cf_mod):
    """Two committed entries twin-match the SAME fresh entry: the invented
    one is promoted onto the fresh id, then the committed HF one renames the
    fresh twin back (same-rung stability). The repoint map must COMPOSE the
    chain (invented -> fresh -> committed HF) so edges and claims land on
    the final survivor, never the renamed-away intermediate."""
    committed = [
        dict(_cf_entry("zorgx/zmod-3-1"), display_name="ZMod 3.1"),
        dict(
            _cf_entry("zorgx/ZMod-3.1"),
            metadata=json.dumps({"hf_deferred": True}, sort_keys=True),
        ),
        dict(_cf_entry("zorgx/zmod-3-1-mini"),
             parents=[{"id": "zorgx/zmod-3-1", "relationship": "variant",
                       "axis": "tier"}]),
    ]
    fresh = [dict(
        _cf_entry("zorgx/ZMod3-1"),
        metadata=json.dumps({"hf_deferred": True}, sort_keys=True),
    )]
    batch, claims = cf_mod._carry_forward_committed([dict(e) for e in fresh], committed)
    by_id = {e["id"]: e for e in batch}
    assert "zorgx/ZMod-3.1" in by_id, "committed HF id wins the same-rung twin"
    assert "zorgx/ZMod3-1" not in by_id
    assert "zorgx/zmod-3-1" not in by_id
    # The child edge must compose through the chain to the FINAL survivor.
    edge = by_id["zorgx/zmod-3-1-mini"]["parents"][0]["id"]
    assert edge == "zorgx/ZMod-3.1", f"edge landed on renamed-away id: {edge}"
    # Claims too — a renamed-away value poisons the committed_claims escape.
    assert claims["zorgx/zmod-3-1"] == "zorgx/ZMod-3.1"


def test_carry_forward_same_rung_hf_twins_keep_committed_id(cf_mod):
    """Two hf_deferred twins (both rung 1): the stability rule holds — the
    committed id wins and today's respelling becomes an alias."""
    committed = [dict(
        _cf_entry("zorgx/ZMod-1B"),
        metadata=json.dumps({"hf_deferred": True}, sort_keys=True),
    )]
    fresh = [dict(
        _cf_entry("zorgx/Z-Mod-1B"),
        metadata=json.dumps({"hf_deferred": True}, sort_keys=True),
    )]
    batch, claims = cf_mod._carry_forward_committed([dict(e) for e in fresh], committed)
    assert [e["id"] for e in batch] == ["zorgx/ZMod-1B"]
    assert "zorgx/Z-Mod-1B" in batch[0]["aliases"]
    assert claims["zorgx/ZMod-1B"] == "zorgx/ZMod-1B"


def test_carry_forward_pending_rung2_counter_logged(cf_mod, capsys):
    """A fresh OpenRouter-key twin (rung 2) deferred by the stability rule is
    counted and the debt line prints on a plain (non-migration) run."""
    committed = [dict(_cf_entry("zorgx/zmod-large"), display_name="ZMod Large")]
    fresh = [dict(
        _cf_entry("zorgx/zmodlarge"),
        metadata=json.dumps({"openrouter_adopted": True}, sort_keys=True),
    )]
    batch, _claims = cf_mod._carry_forward_committed([dict(e) for e in fresh], committed)
    by_id = {e["id"]: e for e in batch}
    assert "zorgx/zmod-large" in by_id and "zorgx/zmodlarge" not in by_id, (
        "rung-2 promotion must stay migration-gated (committed id wins)"
    )
    err = capsys.readouterr().err
    assert "1 pending rung-2 id promotion(s)" in err, (
        f"pending rung-2 debt must be visible in the cron log: {err!r}"
    )


def test_catalog_carry_forward_hf_deferred_twin_not_renamed_back(mod, tmp_path, monkeypatch):
    """Catalog mirror of the rung-monotone rule: a committed invented catalog
    mint whose model reappears as a fresh hf_deferred entry (real HF id) is
    absorbed onto the HF id — never the reverse."""
    committed = [{
        "id": "zorgvid/zeo-9", "display_name": "Zeo 9", "org_id": None,
        "aliases": ["zeo-9"], "metadata": "{}",
        "review_status": "draft", "resolution_source": "models_dev",
    }]
    cat = tmp_path / "cat.yaml"
    cat.write_text(yaml.safe_dump(committed, sort_keys=False))
    monkeypatch.setattr(mod, "CATALOG_OUT_PATH", cat)
    monkeypatch.setattr(mod, "ORGS_GENERATED_PATH", tmp_path / "orgs.yaml")
    fresh = [{
        "id": "zorgvid/Zeo9", "display_name": "Zeo 9", "org_id": None,
        "aliases": [], "metadata": json.dumps({"hf_deferred": True}, sort_keys=True),
        "review_status": "draft", "resolution_source": "models_dev",
    }]
    mod.regenerate_catalog([dict(e) for e in fresh])
    out = yaml.safe_load(cat.read_text()) or []
    by_id = {e["id"]: e for e in out}
    assert "zorgvid/Zeo9" in by_id, "HF-deferred id must survive as canonical"
    assert "zorgvid/zeo-9" not in by_id, "committed invented twin must be absorbed"
    surv = by_id["zorgvid/Zeo9"]
    assert {"zorgvid/zeo-9", "zeo-9"} <= set(surv.get("aliases") or []), (
        f"absorbed committed forms must stay resolvable: {surv.get('aliases')}"
    )


def test_catalog_carry_forward_case_only_respell_keeps_committed_casing(mod, tmp_path, monkeypatch):
    """Case-only respell, SAME rung: the committed casing wins in the catalog
    path too (unified with the non-catalog path's behavior); the fresh casing
    stays resolvable as an alias."""
    committed = [{
        "id": "zorgvid/zeo-x1", "display_name": "Zeo X1", "org_id": None,
        "aliases": [], "metadata": "{}",
        "review_status": "draft", "resolution_source": "models_dev",
    }]
    cat = tmp_path / "cat.yaml"
    cat.write_text(yaml.safe_dump(committed, sort_keys=False))
    monkeypatch.setattr(mod, "CATALOG_OUT_PATH", cat)
    monkeypatch.setattr(mod, "ORGS_GENERATED_PATH", tmp_path / "orgs.yaml")
    fresh = [{
        "id": "zorgvid/Zeo-X1", "display_name": "Zeo X1", "org_id": None,
        "aliases": [], "metadata": "{}",
        "review_status": "draft", "resolution_source": "models_dev",
    }]
    mod.regenerate_catalog([dict(e) for e in fresh])
    out = yaml.safe_load(cat.read_text()) or []
    by_id = {e["id"]: e for e in out}
    assert "zorgvid/zeo-x1" in by_id, "committed casing must win among equals"
    assert "zorgvid/Zeo-X1" not in by_id
    assert "zorgvid/Zeo-X1" in (by_id["zorgvid/zeo-x1"].get("aliases") or [])
