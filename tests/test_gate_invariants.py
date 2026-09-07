"""Acceptance-gate invariants for model-resolution / canonical-graph integrity.

These are PERMANENT regression tests: they encode the acceptance gate as pytest
cases that must hold across all future model-reconcile work. They run OFFLINE against
the committed `fixtures/` parquet warehouse and the frozen live-HF oracle
`hf_model_id_resolution.json` (at the evaleval workspace root).

The gate has two complementary surfaces:
  - the resolver behaviour (does every EEE id resolve, and do the 4,074 HF ids
    resolve ORG-AWARE to their HF-true `fixed_hf_model_id`); and
  - the canonical graph itself (dedup, no dangling parent edges, total
    group/family membership, null-at-origin lineage, Tier-3 honesty,
    no name-only cross-org merges).

Heavy cases (the full 6,720-id resolve sweep + the 4,074 oracle sweep) are
marked `@pytest.mark.slow` so they can be selected/deselected, but they stay
runnable in CI (the sweep is sub-second against the parquet fixtures).

The numbers asserted are the gate floor: COVERAGE must be 6,720/6,720 non-null;
ORACLE must be 4,074/4,074 org-aware-correct; case-insensitive canonical dups
must be 0; dangling parent edges must be 0. A regression that breaks any of
these MUST fail this module.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import pytest
import yaml

from eval_card_registry.lib.seed_io import resolve_oracle_path

from eval_entity_resolver.resolver import Resolver
from eval_entity_resolver.strategies.fuzzy import _ORG_ALIASES


# --------------------------------------------------------------------------
# Locations (offline; everything is committed)
# --------------------------------------------------------------------------
REGISTRY_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = REGISTRY_ROOT / "fixtures"
ORGS_YAML = REGISTRY_ROOT / "seed" / "orgs.yaml"
# The frozen live-HF oracle. Tracked IN-REPO (curation/) so the oracle gates
# actually run in CI (the evaleval workspace root is not part of the registry
# checkout); the shared helper falls back to the workspace-root copy for local
# dev. Single-sourced so the gate and the generator scripts resolve it identically.
ORACLE_PATH = resolve_oracle_path()

# Gate floor numbers (the registry's measured baseline).
EXPECTED_TOTAL = 6720   # total EEE ids in the frozen hf_model_id_resolution.json oracle
EXPECTED_ORACLE = 4074  # fixed_exact + fixed_near_miss subset of that oracle


# --------------------------------------------------------------------------
# Module-scoped fixtures (load the warehouse + oracle once)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def resolver() -> Resolver:
    """Resolver wired to the committed parquet fixtures (canonical graph +
    alias index). Same construction path the oracle-comparison script uses."""
    assert FIXTURES_DIR.exists(), f"missing fixtures dir: {FIXTURES_DIR}"
    return Resolver.from_parquet(str(FIXTURES_DIR))


@pytest.fixture(scope="module")
def models_df() -> pd.DataFrame:
    """The canonical_models table as materialised in fixtures (post-seed:
    the derived walk columns are already populated)."""
    df = pd.read_parquet(FIXTURES_DIR / "canonical_models.parquet")
    assert not df.empty
    return df


@pytest.fixture(scope="module")
def oracle() -> dict[str, dict]:
    assert ORACLE_PATH.exists(), f"missing oracle: {ORACLE_PATH}"
    return json.loads(ORACLE_PATH.read_text())["resolutions"]


@pytest.fixture(scope="module")
def hf_to_dev() -> dict[str, str]:
    """The HF-namespace → developer-org map for the ORG-AWARE oracle comparison.
    Single source: the shared `build_curated_org_map` (incl. the orgs.yaml ALIAS
    tier), so the gate folds orgs the SAME way the resolver does — a stricter
    hf_org-only map here would make the gate diverge from real resolution."""
    from eval_entity_resolver.fold import build_curated_org_map
    return build_curated_org_map(yaml.safe_load(ORGS_YAML.read_text()) or [])


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _org_of(cid: str) -> str:
    return cid.split("/", 1)[0] if "/" in cid else ""


def _name_of(cid: str) -> str:
    return cid.split("/", 1)[1] if "/" in cid else cid


def _parse_parents(value) -> list[dict]:
    """`parents` is stored as a JSON string; may also arrive as a list or
    pandas NA. Normalise to a list of edge dicts."""
    if value is None:
        return []
    if isinstance(value, str):
        try:
            return json.loads(value) or []
        except (ValueError, TypeError):
            return []
    # array-like (pyarrow list) or scalar NA
    try:
        if pd.isna(value):  # scalar NA
            return []
    except (ValueError, TypeError):
        pass  # array-like → not a scalar NA
    return [e for e in list(value) if isinstance(e, dict)]


def _parse_tags(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            return json.loads(value) or []
        except (ValueError, TypeError):
            return []
    try:
        if pd.isna(value):
            return []
    except (ValueError, TypeError):
        pass
    return list(value)


# --------------------------------------------------------------------------
# Sanity: the oracle has the expected shape (guards against a swapped/empty file)
# --------------------------------------------------------------------------
def test_oracle_has_expected_population(oracle):
    assert len(oracle) == EXPECTED_TOTAL, (
        f"oracle key count drifted: {len(oracle)} != {EXPECTED_TOTAL}"
    )
    n_oracle = sum(
        1 for v in oracle.values()
        if v.get("resolution_status") in ("fixed_exact", "fixed_near_miss")
    )
    assert n_oracle == EXPECTED_ORACLE, (
        f"fixed_exact+fixed_near_miss count drifted: {n_oracle} != {EXPECTED_ORACLE}"
    )


# --------------------------------------------------------------------------
# 1. COVERAGE — every one of the 6,720 EEE ids resolves to a non-null canonical
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_coverage_all_eee_ids_resolve_non_null(resolver, oracle):
    """GATE part 1 (quantitative floor): every EEE id (incl. the 2,646
    not-on-HF tail that must gain non-null via Tier-2/3) resolves to a
    non-null canonical. A single no_match is a gate break."""
    no_match = [raw for raw in oracle if resolver.resolve(raw, "model").canonical_id is None]
    assert no_match == [], (
        f"{len(no_match)}/{len(oracle)} EEE ids resolved to no_match (gate floor "
        f"is 0). First few: {no_match[:10]}"
    )


# --------------------------------------------------------------------------
# 2. ORACLE — the 4,074 HF ids resolve ORG-AWARE to their fixed_hf_model_id
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_oracle_org_aware_match(resolver, oracle, hf_to_dev):
    """GATE part 1: each fixed_exact / fixed_near_miss id resolves to a
    canonical C where model-name(C) == model-name(fixed_hf_model_id)
    CASE-SENSITIVELY, and org(C) is ORG-AWARE-equal to the HF org (equal
    case-insensitively for community ids, or a curated namespace-alias of the
    resolved developer org for big-dev ids, e.g. meta-llama ↔ meta).

    The model-name comparison is case-sensitive ON PURPOSE — HF repo names are
    case-sensitive and the canonical id preserves HF casing."""
    fixed = {
        raw: meta for raw, meta in oracle.items()
        if meta.get("resolution_status") in ("fixed_exact", "fixed_near_miss")
    }

    # Raws whose oracle `fixed_near_miss` redirect must NOT be followed (a wrong
    # HF "did-you-mean": a genuine size change or a cross-uploader migration).
    # They resolve to their own/corrected canonical, so each is checked against
    # THAT (must resolve away from the bad redirect) rather than against it.
    #
    # SINGLE SOURCE OF TRUTH: the exemption set is derived from the SAME
    # `nearmiss_changes_identity` rule the generator (generate_hf_oracle_seed)
    # uses to decide which near-miss redirects to refuse — so the gate and the
    # generator can never disagree about which redirects are "bad", and there is
    # no hand-maintained list to drift. The curated `audit_bad_nearmiss.json` is
    # one INPUT to that rule (audit-confirmed specific cases), not a parallel list.
    _nearmiss_changes_identity, _hf_oracle_hf_to_dev = _import_nearmiss_rule()
    nm_hf_to_dev = _hf_oracle_hf_to_dev()
    AUDIT_CORRECTED = frozenset(
        raw for raw, meta in fixed.items()
        if meta.get("resolution_status") == "fixed_near_miss"
        and isinstance(meta.get("fixed_hf_model_id"), str)
        and _nearmiss_changes_identity(raw, meta["fixed_hf_model_id"], nm_hf_to_dev)
    )

    def org_aware_equal(resolved_org: str, hf_org: str) -> bool:
        # The resolved canonical's id-prefix is the REAL HF org (not a folded
        # dev slug). Two HF orgs of the same developer are both
        # valid (e.g. THUDM and zai-org both publish GLM); the invariant is
        # that they fold to the SAME canonical developer.
        if resolved_org.lower() == hf_org.lower():
            return True
        fold = lambda x: hf_to_dev.get(x.lower(), x.lower())
        return fold(resolved_org) == fold(hf_org)

    # Each audit-corrected raw must resolve to a DIFFERENT canonical than its bad
    # redirect does (i.e. it was actually re-pointed away from the wrong model).
    # Compared by resolved canonical, not name — the cross-uploader corrections
    # share the redirect's model NAME but differ in org.
    not_corrected = []
    for raw in (AUDIT_CORRECTED & set(fixed)):
        bad_id = fixed[raw]["fixed_hf_model_id"]
        cid = resolver.resolve(raw, "model").canonical_id
        bad_cid = resolver.resolve(bad_id, "model").canonical_id
        if cid is None or cid == bad_cid:
            not_corrected.append((raw, bad_id, cid))
    assert not_corrected == [], (
        f"{len(not_corrected)} audit-corrected raw(s) still resolve to the oracle's "
        f"bad redirect (the bad-redirect correction regressed): {not_corrected[:10]}"
    )

    # Every other oracle id must org-aware-match its fixed_hf_model_id.
    checkable = {raw: meta for raw, meta in fixed.items() if raw not in AUDIT_CORRECTED}
    passed = 0
    buckets: Counter = Counter()
    samples: list[tuple] = []
    for raw, meta in checkable.items():
        hf_id = meta["fixed_hf_model_id"]
        hf_org, hf_name = _org_of(hf_id), _name_of(hf_id)
        cid = resolver.resolve(raw, "model").canonical_id
        if cid is None:
            buckets["no_match"] += 1
            if len(samples) < 12:
                samples.append((raw, hf_id, None))
            continue
        name_ok = _name_of(cid) == hf_name           # case-sensitive
        org_ok = org_aware_equal(_org_of(cid), hf_org)
        if name_ok and org_ok:
            passed += 1
        else:
            key = "name+org" if (not name_ok and not org_ok) else ("name" if not name_ok else "org")
            buckets[f"{key}_mismatch"] += 1
            if len(samples) < 12:
                samples.append((raw, hf_id, cid))

    assert passed == len(checkable) == EXPECTED_ORACLE - len(AUDIT_CORRECTED & set(fixed)), (
        f"ORACLE org-aware match: {passed}/{len(checkable)} (floor "
        f"{EXPECTED_ORACLE - len(AUDIT_CORRECTED & set(fixed))} = {EXPECTED_ORACLE} - "
        f"{len(AUDIT_CORRECTED & set(fixed))} audit-corrected). Failure buckets: "
        f"{dict(buckets)}. Samples (raw, hf, resolved): {samples}"
    )


# --------------------------------------------------------------------------
# 3. DEDUP — 0 case-insensitive duplicate canonical ids
# --------------------------------------------------------------------------
def test_no_case_insensitive_duplicate_canonical_ids(models_df):
    """No lowercase+HF-cased duplicate pair may exist: the HF-cased id wins and
    any lowercase spelling is an alias on it. Two canonical rows whose ids differ
    only by case are a dedup failure."""
    lc = Counter(i.lower() for i in models_df["id"])
    dups = {k: v for k, v in lc.items() if v > 1}
    # Surface the actual colliding ids for a useful failure message.
    detail = {}
    if dups:
        by_lower = defaultdict(list)
        for i in models_df["id"]:
            if i.lower() in dups:
                by_lower[i.lower()].append(i)
        detail = dict(by_lower)
    assert dups == {}, f"case-insensitive duplicate canonical ids: {detail}"


# --------------------------------------------------------------------------
# 3b. NO ENRICHMENT-MINT — enrichments/*.yaml bridge, never create a canonical
# --------------------------------------------------------------------------
def test_alias_enrichments_never_mint_canonicals():
    """`enrichments/*.yaml` records (aliases.yaml AND parents.yaml, etc.) must
    attach to an EXISTING canonical, never mint one. The loader (`cli.py::_absorb`)
    creates a bare canonical for ANY enrichment-record id with no backing entity
    def — a metadata-less row that silently shadows any real canonical it
    normalize-duplicates. Every enrichment id must be a def id in core.yaml or
    sources/*.generated.yaml.

    `skip_source_ids` is subtracted: an id core deliberately skips is DROPPED by the
    loader (`_absorb(..., extra_skip=skip_source_ids)`), never minted, so it can't
    create a phantom. (A stale record pointing at a skipped id merely loses its
    surface forms — a separate, lesser cleanup, not the mint bug this gate guards.)"""
    seed_models = REGISTRY_ROOT / "seed" / "models"

    def _entry_ids(obj) -> set[str]:
        entries = obj.get("entries", []) if isinstance(obj, dict) else obj
        return {e["id"] for e in (entries or []) if isinstance(e, dict) and e.get("id")}

    core = yaml.safe_load((seed_models / "core.yaml").read_text()) or {}
    def_ids = _entry_ids(core)
    for src in sorted((seed_models / "sources").glob("*.generated.yaml")):
        def_ids |= _entry_ids(yaml.safe_load(src.read_text()) or [])
    skip = set(core.get("skip_source_ids") or []) if isinstance(core, dict) else set()

    minted = {}
    for enr in sorted((seed_models / "enrichments").glob("*.yaml")):
        bad = sorted(_entry_ids(yaml.safe_load(enr.read_text()) or []) - def_ids - skip)
        if bad:
            minted[enr.name] = bad
    assert minted == {}, (
        f"enrichment record(s) MINT a new canonical (id absent from core/sources "
        f"and not in skip_source_ids): {minted}"
    )


# --------------------------------------------------------------------------
# 4. NO DANGLING — every parents[].id references an existing canonical id
# --------------------------------------------------------------------------
def test_no_dangling_parent_edges(models_df):
    """Every `parents[].id` edge across canonical_models must point at a row
    that exists as a canonical id. A dangling parent edge breaks the
    group/family/lineage walks and the producer's lineage derivation."""
    idset = set(models_df["id"])
    dangling: list[tuple[str, str]] = []
    for cid, parents in zip(models_df["id"], models_df["parents"]):
        for edge in _parse_parents(parents):
            target = edge.get("id")
            if target is not None and target not in idset:
                dangling.append((cid, target))
    assert dangling == [], (
        f"{len(dangling)} dangling parent edges (parent id not a canonical). "
        f"First few: {dangling[:10]}"
    )


# --------------------------------------------------------------------------
# 5. MEMBERSHIP — group/family non-null (self for singletons);
#    lineage_origin_model_id null-at-origin (NOT self)
# --------------------------------------------------------------------------
def test_group_and_family_membership_is_total(models_df):
    """`model_group_id` and `model_family_id` are a TOTAL partition: NON-NULL
    for every model, equal to SELF for singletons / roots (NOT null at root,
    via the self-fallback). A null group/family is a membership break."""
    null_group = models_df[models_df["model_group_id"].isna()]
    null_family = models_df[models_df["model_family_id"].isna()]
    assert null_group.empty, (
        f"{len(null_group)} models have null model_group_id "
        f"(must be self at root). e.g. {null_group['id'].head(10).tolist()}"
    )
    assert null_family.empty, (
        f"{len(null_family)} models have null model_family_id "
        f"(must be self at root). e.g. {null_family['id'].head(10).tolist()}"
    )
    # And the self-fallback must actually fire for at least the obvious
    # singletons — guard against "non-null" being satisfied trivially by some
    # other value while self-at-root regressed. Many rows are their own group.
    self_group = (models_df["model_group_id"] == models_df["id"]).sum()
    assert self_group > 0, "no model is its own group root — self-fallback regressed"


def test_lineage_origin_model_id_is_null_at_origin_not_self(models_df):
    """`lineage_origin_model_id` (the id of the deepest non-variant ancestor)
    is NULL when self is the origin — NO self-fallback (unlike the org_id
    variant, which DOES self-fall-back). A row whose lineage_origin_model_id
    equals its own id is a regression of the null-at-origin semantics."""
    self_pointing = models_df[models_df["lineage_origin_model_id"] == models_df["id"]]
    assert self_pointing.empty, (
        f"{len(self_pointing)} models have lineage_origin_model_id == self "
        f"(must be null at origin). e.g. {self_pointing['id'].head(10).tolist()}"
    )
    # Non-trivial: SOME rows must actually carry a (non-null, non-self) origin
    # pointer — otherwise the column is uniformly null and the assertion above
    # is vacuous. Finetunes/quants of upstream weights have one.
    has_origin = models_df["lineage_origin_model_id"].notna().sum()
    assert has_origin > 0, (
        "no model carries a lineage_origin_model_id — the lineage walk produced "
        "nothing, so the null-at-origin check is trivially green"
    )


# --------------------------------------------------------------------------
# 6. NO NAME-ONLY CROSS-ORG MERGE — a same model-name under two orgs stays distinct
# --------------------------------------------------------------------------
def test_no_name_only_cross_org_merge(resolver, models_df):
    """Org-aware identity: a model NAME shared by two different orgs must NOT
    collapse to a single canonical. Concrete known case: the `Llama-3-Instruct-
    8B-SimPO` finetune exists independently under both `haoranxu` and
    `princeton-nlp`. Both ids must (a) exist as distinct canonicals and (b)
    resolve to THEMSELVES — not to each other or a shared merged id."""
    a = "haoranxu/Llama-3-Instruct-8B-SimPO"
    b = "princeton-nlp/Llama-3-Instruct-8B-SimPO"
    idset = set(models_df["id"])
    assert a in idset and b in idset, (
        "spot-check canonicals missing from fixtures — pick a fresh known "
        f"multi-org name. have a={a in idset} b={b in idset}"
    )
    # Same model name, different org → genuinely distinct identities.
    assert _name_of(a) == _name_of(b)
    assert _org_of(a) != _org_of(b)

    res_a = resolver.resolve(a, "model").canonical_id
    res_b = resolver.resolve(b, "model").canonical_id
    assert res_a == a, f"{a} resolved to {res_a}, not itself (cross-org merge?)"
    assert res_b == b, f"{b} resolved to {res_b}, not itself (cross-org merge?)"
    assert res_a != res_b, "two distinct-org same-name models merged to one canonical"


# --------------------------------------------------------------------------
# 7. TIER-3 HONESTY — inferred rows are draft; org-less inferred are org-unknown
# --------------------------------------------------------------------------
def test_tier3_inferred_rows_are_draft(models_df):
    """Every Tier-3 mint (`resolution_source=inferred`) must be `review_status
    = draft` — the registry never silently promotes a name-inferred guess to
    reviewed."""
    inferred = models_df[models_df["resolution_source"] == "inferred"]
    assert not inferred.empty, "no inferred rows in fixtures — Tier-3 mints vanished"
    non_draft = inferred[inferred["review_status"] != "draft"]
    assert non_draft.empty, (
        f"{len(non_draft)} inferred rows are not review_status=draft "
        f"(Tier-3 honesty break). e.g. {non_draft['id'].head(10).tolist()}"
    )


def test_orgless_inferred_rows_are_tagged_org_unknown(models_df):
    """Free-text mints WITHOUT an extractable org go to the org-less bucket and
    the registry never auto-guesses an org. The bucket is keyed by the `unknown`
    sentinel org (a `null` org_id materialises to it from the `unknown/` prefix
    at seed time). Every such inferred row MUST carry the `org-unknown` tag, so
    the bucket is honestly surfaced for review whether a consumer filters on the
    org FK or the tag."""
    inferred = models_df[models_df["resolution_source"] == "inferred"]
    orgless = inferred[inferred["org_id"].isna() | (inferred["org_id"] == "unknown")]
    assert not orgless.empty, (
        "no org-less inferred rows in fixtures — the org-less bucket vanished; "
        "if intentional, update this test"
    )
    missing_tag = [
        cid for cid, tags in zip(orgless["id"], orgless["tags"])
        if "org-unknown" not in _parse_tags(tags)
    ]
    assert missing_tag == [], (
        f"{len(missing_tag)} org-less inferred rows lack the 'org-unknown' tag "
        f"(org-less honesty break). e.g. {missing_tag[:10]}"
    )


# --------------------------------------------------------------------------
# ORG-INTEGRITY (org-canonicalization gate) — every model's org FK resolves,
# orgs are one-per-developer (no case-splits), and each oracle id is its own
# real-HF-repo canonical (not a folded org-slug form).
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def orgs_df() -> pd.DataFrame:
    df = pd.read_parquet(FIXTURES_DIR / "canonical_orgs.parquet")
    assert not df.empty
    return df


def test_no_dangling_org_fk(models_df, orgs_df):
    """Every `org_id` AND `lineage_origin_model_org_id` on a model must name a
    real `canonical_orgs` row. A dangling org FK = a model whose developer
    can't be resolved downstream (null `developer` in the producer)."""
    org_ids = set(orgs_df["id"].astype(str))
    org_fk = {x for x in models_df["org_id"].astype("string").dropna().unique() if x not in org_ids}
    lin_fk = {
        x for x in models_df["lineage_origin_model_org_id"].astype("string").dropna().unique()
        if x not in org_ids
    }
    assert org_fk == set(), f"{len(org_fk)} dangling org_id FK(s): {sorted(org_fk)[:10]}"
    assert lin_fk == set(), f"{len(lin_fk)} dangling lineage-org FK(s): {sorted(lin_fk)[:10]}"


def test_no_case_split_orgs(orgs_df):
    """One canonical_orgs row per developer: no two org ids may differ only by
    case (`Tencent` vs `tencent`), which would fragment the developer in every
    downstream org list/facet."""
    ci: Counter = Counter(str(i).lower() for i in orgs_df["id"])
    split = {lo: [i for i in orgs_df["id"] if str(i).lower() == lo] for lo, n in ci.items() if n > 1}
    assert split == {}, f"{len(split)} case-split org(s): {list(split.values())[:8]}"


@pytest.mark.slow
def test_oracle_canonical_id_is_real_hf_repo_id(resolver, oracle):
    """Real-HF-repo invariant: each oracle `fixed_hf_model_id` (the real HF repo
    id) must resolve to a canonical that IS that real id (directly), OR — for the
    rare oracle near-miss whose registry canonical sits under the original
    publisher's namespace — reach it via a confirmed alias. canonical_ids are
    never the synthetic org-folded form (`meta/Llama-…`) anymore."""
    fixed = [
        m["fixed_hf_model_id"] for m in oracle.values()
        if isinstance(m.get("fixed_hf_model_id"), str)
    ]
    unreachable = [
        fx for fx in fixed if resolver.resolve(fx, "model").canonical_id is None
    ]
    assert unreachable == [], (
        f"{len(unreachable)}/{len(fixed)} oracle real-HF ids unreachable: {unreachable[:10]}"
    )


def test_canonical_id_equals_metadata_hf_id(models_df):
    """Real-HF-repo invariant for the hub-stats tail: a model that records a real
    HF repo id in `metadata.hf_id` must USE it as its `canonical_id`. A mismatch
    where the real id is ABSENT means the canonical is a synthetic/slugified
    form that 404s on HF."""
    ids = set(models_df["id"].astype(str))
    viol = []
    for r in models_df.itertuples():
        md = getattr(r, "metadata", None)
        if not isinstance(md, str):
            continue
        try:
            h = json.loads(md).get("hf_id")
        except Exception:
            h = None
        if isinstance(h, str) and "/" in h and h != str(r.id) and h not in ids:
            viol.append((str(r.id), h))
    assert viol == [], (
        f"{len(viol)} models whose canonical_id != metadata.hf_id (real absent): {viol[:10]}"
    )


@pytest.mark.slow
def test_old_folded_form_resolves_to_real_hf_id(resolver, models_df):
    """Safety net (also guards the models.dev refresh cron): the org-folded id
    spelling (`alibaba/Qwen2.5-7B`) must resolve to the real-HF canonical
    (`Qwen/Qwen2.5-7B`) via an alias on it. That alias is what makes
    `regenerate_catalog` (refresh_from_modelsdev) fold a folded-slug mint onto the
    real canonical instead of minting a duplicate. A regression here = the cron
    would re-introduce folded canonicals."""
    ids = set(models_df["id"].astype(str))
    pairs = [
        ("alibaba/Qwen2.5-7B", "Qwen/Qwen2.5-7B"),
        ("meta/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"),
        ("deepseek/DeepSeek-V3", "deepseek-ai/DeepSeek-V3"),
    ]
    checked = 0
    for folded, real in pairs:
        if real not in ids:
            continue  # not in this fixture snapshot — skip, don't false-fail
        checked += 1
        got = resolver.resolve(folded, "model").canonical_id
        assert got == real, f"{folded!r} resolved to {got!r}, expected {real!r}"
    assert checked > 0, "no folded->real pairs present in fixtures to check"


def test_no_separator_split_orgs(orgs_df):
    """No two org ids differ ONLY by a separator/case (`prime-intellect` vs
    `PrimeIntellect`) — the plain case-insensitive check misses these, but they
    still fragment one developer into two rows downstream.

    EXCEPTION: a curated allowlist (seed/orgs_distinct_allowlist.yaml) of pairs
    verified to be GENUINELY DISTINCT real HF uploaders — merging those would be a
    false same-uploader assertion the generators refuse. (A pair that IS the same
    uploader is instead merged via a curated orgs.yaml alias, not allowlisted.)"""
    import yaml as _yaml
    allow_path = REGISTRY_ROOT / "seed" / "orgs_distinct_allowlist.yaml"
    allow = set(_yaml.safe_load(allow_path.read_text()) or []) if allow_path.exists() else set()
    norm = lambda x: re.sub(r"[^a-z0-9]", "", str(x).lower())
    by: dict = defaultdict(list)
    for oid in orgs_df["id"]:
        by[norm(oid)].append(str(oid))
    split = {k: v for k, v in by.items() if len(v) > 1 and not set(v) <= allow}
    assert split == {}, f"{len(split)} separator/case-split org group(s): {list(split.values())[:8]}"


def test_no_real_hf_id_duplicated_by_slug(models_df):
    """A real HF id (`Qwen/Qwen2.5-7B-Instruct`, oracle/hf-sourced) must never be
    duplicated by a NON-real folded/slug variant (`alibaba/qwen-2-5-7b-instruct`)
    — the dup-merge collapses those. Two distinct REAL repos that
    happen to normalize alike (NAPS v-0.1.0 vs v0.1.0) are allowed; 0-real
    API/spelling dups are a known residual."""
    fixed = set()
    for v in json.loads(ORACLE_PATH.read_text())["resolutions"].values():
        fx = v.get("fixed_hf_model_id")
        if isinstance(fx, str) and "/" in fx:
            fixed.add(fx)
    from eval_entity_resolver.fold import build_curated_org_map
    hf_to_dev = build_curated_org_map(yaml.safe_load(ORGS_YAML.read_text()) or [])

    def fold(org):
        return hf_to_dev.get(org.lower(), org)

    def ndup(s):
        s = s.lower()
        s = re.sub(r"([a-z])[-_ /]+(\d)", r"\1\2", s)
        s = re.sub(r"(\d)\.(\d)(?![bmkt])", r"\1-\2", s)
        return re.sub(r"[-_ /]+", "-", s)

    def has_real_hf_id(row):
        """True when the row's canonical_id is a verified real HF repo id —
        an oracle fixed id, an hf-sourced mint, or == its own metadata.hf_id."""
        cid = str(row.id)
        if cid in fixed:
            return True
        if isinstance(row.resolution_source, str) and row.resolution_source == "hf":
            return True
        md = getattr(row, "metadata", None)
        if isinstance(md, str):
            try:
                return json.loads(md).get("hf_id") == cid
            except Exception:
                return False
        return False

    clusters: dict = defaultdict(lambda: {"real": [], "nonreal": []})
    for row in models_df.itertuples():
        cid = str(row.id)
        if "/" not in cid:
            continue
        org, name = cid.split("/", 1)
        bucket = "real" if has_real_hf_id(row) else "nonreal"
        clusters[(fold(org), ndup(name))][bucket].append(cid)
    # Violation = a real id sharing a cluster with a NON-real slug variant.
    bad = [
        sorted(c["real"] + c["nonreal"])
        for c in clusters.values()
        if c["real"] and c["nonreal"]
    ]
    assert bad == [], f"{len(bad)} real-HF id(s) duplicated by a non-real slug: {bad[:8]}"


# --------------------------------------------------------------------------
# No minted models.dev canonical still shadows a real HF id.
#
# This is the RULE-AS-ASSERTION gate: it re-runs the SAME confident-match
# predicate the fold (scripts/fold_modelsdev_dupes.py) used — exact id / alias
# linkage / normalized-name-with-ORG-AGREEMENT / brand-prefix-stripped-with-org
# — over the materialised fixtures. If the fold cleaned everything, the predicate
# now finds NOTHING; a reintroduced mint-dupe makes it find a fold and fail.
#
# Why this catches what test_no_real_hf_id_duplicated_by_slug misses: that test
# clusters by a STRING normalisation of (folded-org, ndup(name)), so it cannot
# group a dupe that differs in BOTH org spelling AND name across the brand
# prefix (e.g. `alibaba/qwen-qwq-32b` vs real `Qwen/QwQ-32B` → dev `alibaba`):
# the names `qwen-qwq-32b` and `qwq-32b` land in different ndup() buckets. The
# fold predicate defers via the resolver's brand-prefix stripping + org
# agreement, so re-running it as the gate closes that exact blind spot.
# --------------------------------------------------------------------------
def _import_nearmiss_rule():
    """The generator's near-miss block rule + its hf_to_dev builder — so the gate
    derives its AUDIT_CORRECTED exemption from the SAME rule the generator uses
    (no parallel hand-maintained list)."""
    scripts_dir = REGISTRY_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import generate_hf_oracle_seed as gho  # noqa: E402

    def _hf_to_dev():
        return gho.build_hf_to_dev(gho._load_curated_orgs())

    return gho.nearmiss_changes_identity, _hf_to_dev


def _import_fold_module():
    scripts_dir = REGISTRY_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import fold_modelsdev_dupes as fold  # noqa: E402

    return fold


def _model_aliases_by_canonical() -> dict[str, list[str]]:
    """raw_value spellings declared for each model canonical, from the alias
    table — so the gate feeds the fold predicate the same alias surface the
    fold script reads off `core.yaml` entries."""
    apath = FIXTURES_DIR / "aliases.parquet"
    out: dict[str, list[str]] = defaultdict(list)
    if not apath.exists():
        return out
    adf = pd.read_parquet(apath)
    adf = adf[adf["entity_type"] == "model"]
    for row in adf.itertuples():
        cid = getattr(row, "canonical_id", None)
        rv = getattr(row, "raw_value", None)
        if isinstance(cid, str) and isinstance(rv, str) and rv:
            out[cid].append(rv)
    return out


# Allowlist of models.dev mints that the fold PREDICATE flags but which are NOT
# true dupes: the predicate's normalized/brand-stripped tier can over-reach (a
# base vs a specific dated/quant/size variant, a family pointer vs a leaf, or a
# not-yet-real placeholder). Entries here are intentionally NOT folded so the gate
# stays green for the legitimately-not-folded state while any NEW confident
# mint-dupe still fails. Keep this list TIGHT — every entry is a reviewed
# exception, not a blanket waiver; the gate fails on a dead (unused) waiver.
# Currently EMPTY: org-canonicalization plus the shared decide_fold predicate fold
# every known case correctly.
_KNOWN_NON_DUPE_MINTS: frozenset[str] = frozenset()


def test_no_minted_modelsdev_canonical_shadows_real_hf_id(models_df):
    """GATE: NO canonical with resolution_source=models_dev still has a CONFIDENT
    real-HF match — i.e. the fold found every mint-dupe and none can
    silently return. Uses the fold script's own predicate (shared, not a third
    matcher), so the gate and the cleanup agree by construction. A tight
    allowlist (`_KNOWN_NON_DUPE_MINTS`) carries the reviewed predicate
    over-reaches; any NEW confident mint-dupe outside it fails the gate."""
    fold = _import_fold_module()

    # reconstruct the entry-list shape decide_fold expects, from the fixtures
    alias_map = _model_aliases_by_canonical()
    entries: list[dict] = []
    for row in models_df.itertuples():
        cid = str(row.id)
        org_id = getattr(row, "org_id", None)
        dn = getattr(row, "display_name", None)
        rsrc = getattr(row, "resolution_source", None)
        md = getattr(row, "metadata", None)
        entries.append({
            "id": cid,
            "org_id": org_id if isinstance(org_id, str) else None,
            "display_name": dn if isinstance(dn, str) else None,
            "resolution_source": rsrc if isinstance(rsrc, str) else None,
            "aliases": alias_map.get(cid, []),
            "metadata": md if isinstance(md, str) else None,
        })

    hf_to_dev = fold.build_hf_to_dev()
    hf_ids, alias_to_hf, by_org_name, _ = fold.build_hf_targets(entries, hf_to_dev)

    mints = [e for e in entries if e["resolution_source"] == "models_dev"]
    assert mints, "no models_dev canonicals in fixtures — gate cannot run (reseed?)"

    leftover = []
    for m in mints:
        if m["id"] in _KNOWN_NON_DUPE_MINTS:
            continue
        f = fold.decide_fold(m, hf_ids, alias_to_hf, by_org_name, hf_to_dev)
        if f is not None:
            leftover.append((f["mint_id"], f["hf_target"], f["match_type"]))

    assert leftover == [], (
        f"{len(leftover)} models.dev-minted canonical(s) still confidently match "
        f"a real HF id (mint-dupe shadow not folded): {leftover[:12]}"
    )

    # Self-policing: every allowlist entry must still be a live, flagged mint —
    # an entry that no longer matches the predicate (renamed/removed/refolded) is
    # a DEAD waiver and must be removed, so the list can't silently rot.
    by_mint = {e["id"]: e for e in mints}
    dead = [
        a for a in _KNOWN_NON_DUPE_MINTS
        if a not in by_mint
        or fold.decide_fold(by_mint[a], hf_ids, alias_to_hf, by_org_name, hf_to_dev) is None
    ]
    assert dead == [], (
        f"{len(dead)} dead _KNOWN_NON_DUPE_MINTS waiver(s) (no longer a flagged "
        f"models_dev mint-dupe) — remove them to keep the allowlist tight: {sorted(dead)}"
    )


# --------------------------------------------------------------------------
# org-conditional quant grouping: model_group_id never crosses a developer
# --------------------------------------------------------------------------
def test_model_group_id_does_not_cross_developer_org(models_df):
    """A model and its `model_group_id` root MUST share the same developer
    `org_id`. A first-party precision variant (same org) folds into the base
    group; a THIRD-PARTY / community quant (e.g. `unsloth/...-bnb-4bit`
    quantizing `microsoft/phi-4`) keeps its OWN group — its `quantized` edge
    still records the link via `lineage_origin_model_id`, but its scores never
    merge into the base lab's model. The same guard drops a spurious cross-org
    version edge. A group root under a different developer is the
    cross-org-identity-grouping bug this enforces against."""
    org_by_id = dict(
        zip(models_df["id"].astype(str), models_df["org_id"].astype("string"))
    )
    bad = []
    for cid, grp in zip(
        models_df["id"].astype(str), models_df["model_group_id"].astype("string")
    ):
        if grp is None or pd.isna(grp) or grp == cid:
            continue
        o_self = org_by_id.get(cid)
        o_grp = org_by_id.get(grp)
        if (
            o_self is not None and o_grp is not None
            and not pd.isna(o_self) and not pd.isna(o_grp)
            and o_self != o_grp
        ):
            bad.append((cid, grp, o_self, o_grp))
    assert bad == [], (
        f"{len(bad)} model(s) whose model_group_id root is a DIFFERENT developer "
        f"org (cross-org identity grouping is forbidden; a community quant must "
        f"keep its own group). First few: {bad[:10]}"
    )


# --------------------------------------------------------------------------
# BEHAVIORAL ORACLE SNAPSHOT — resolution must not regress an already-seen
# canonical/alias. The HF-id oracle gates above only police the ~4074
# HF-resolvable ids in hf_model_id_resolution.json; they are BLIND to the
# closed-API, NA-source, and reviewed curated canonicals the curated floor exists
# to preserve. This gate guards the FULL snapshot at curation/oracle_snapshot/
# (canonical_models.parquet + aliases.parquet, 7196 canonicals / 28900 aliases):
# every snapshot canonical id (and every snapshot alias raw) must still resolve to
# a NON-NULL canonical whose developer is ORG-AWARE-equal to the snapshot's. Names
# are NOT required equal — id spellings may differ from the snapshot
# (meta/llama-3-70b -> meta-llama/Meta-Llama-3-70B); the invariant is "same
# developer, still resolves". Outcomes that intentionally differ from the snapshot
# are enumerated in the exemption sets (kept tight + justified) so drift stays
# visible.
# --------------------------------------------------------------------------
ORACLE_SNAPSHOT_DIR = REGISTRY_ROOT / "curation" / "oracle_snapshot"

# Oracle-snapshot canonical ids whose resolution intentionally lands on a
# different developer in a way the same-model leaf rule below can't auto-detect
# (the snapshot id was MALFORMED — an embedded org or a placeholder host — so the
# corrected leaf legitimately differs). Each is a reviewed exception.
_ORACLE_CANON_EXEMPT: frozenset = frozenset({
    "openai/aion-labs-aion-2-0",           # oracle embedded the org in the leaf -> aion-labs/aion-2-0
    "unknown/perplexity-sonar-reasoning",  # `unknown/` placeholder host -> perplexity/sonar-reasoning
    "unknown/cohere-embed-v-4-0",          # `unknown/` placeholder host -> cohere/embed-v4-0 (collision fold)
    "unknown/writer.palmyra-x4",           # `unknown/` placeholder host, org in the leaf -> writer/palmyra-x4
    # ai21 == ai21-labs (same developer; HF org is ai21-labs). The leaf also
    # gains an `ai21-` brand prefix in the real repo so the same-model-leaf rule
    # can't auto-detect it; resolving to ai21-labs/ai21-jamba-* is correct.
    "ai21/jamba-1.5-large",
    "ai21/jamba-1.5-mini",
    # EVA finetune mis-filed under alibaba (it's a Qwen finetune) with the `eva`
    # brand as a leaf SUFFIX. Real HF repo is EVA-UNIT-01/EVA-Qwen2.5-32B-v0.2 —
    # right org, brand moves to a PREFIX, so the same-model-leaf rule can't match
    # the two spellings. Resolving to EVA-UNIT-01/eva-qwen2-5-32b-v0-2 is correct.
    "alibaba/qwen2-5-32b-eva-v0-2",
    # Lunaris is Sao10K's model; the snapshot's `meta/` attribution was wrong
    # (same class as the `meta/l3-euryale` example above). The corrected id
    # also swaps the size token's position, so the same-model-leaf rule can't
    # auto-detect it. Resolving to sao10k/l3-lunaris-8b is correct.
    "meta/l3-8b-lunaris",
})

# Oracle-snapshot aliases intentionally NOT resolved — the snapshot's attribution
# was itself wrong. Reviewed exceptions only.
# Keyed either by bare raw_value or by (entity_type, raw_value).
_ORACLE_ALIAS_EXEMPT: frozenset = frozenset({
    # `vercel` is an inference platform / AI gateway (seed/inference_platforms.yaml
    # + the resolver host-prefix strip list), NOT a model developer. The snapshot's
    # `vercel` org (3 models mis-attributed to the host) is a host-mislabel: the
    # host is stripped, so resolving `vercel` as a developer org is correctly
    # no_match.
    ("org", "vercel"),
})


def _org_aware_equal(resolved_org: str, oracle_org: str, hf_to_dev: dict) -> bool:
    if resolved_org.lower() == oracle_org.lower():
        return True
    fold = lambda x: hf_to_dev.get(x.lower(), x.lower())
    return fold(resolved_org) == fold(oracle_org)


def _leaf_norm(cid: str) -> str:
    from eval_entity_resolver.normalization import normalize
    return normalize(cid.split("/", 1)[1] if "/" in cid else cid).replace(" ", "")


@pytest.mark.slow
def test_phase0_oracle_canonicals_preserved(resolver, hf_to_dev):
    """GATE: every oracle-snapshot canonical id still resolves NON-NULL to an
    org-aware-equal developer (no curated/closed-API/NA entity silently orphaned
    to no_match or re-attributed to a different developer).

    A resolution to a DIFFERENT developer is permitted ONLY when the model leaf
    is identical (same model, corrected org) — that corrects wrong developer
    attributions in the snapshot (the models.dev provider mislabels
    `meta/aion-1-0` -> real `aion-labs/aion-1-0`, `amazon/manta-*` ->
    `meganova-ai/manta-*`, `meta/l3-euryale` -> `sao10k/...`). A different-developer
    resolution with a DIFFERENT leaf is a genuine re-attribution and fails (unless
    explicitly exempted as a malformed snapshot id)."""
    import pandas as pd
    oc = pd.read_parquet(ORACLE_SNAPSHOT_DIR / "canonical_models.parquet")
    no_match, wrong_dev = [], []
    for row in oc.itertuples():
        oid = str(row.id)
        if oid in _ORACLE_CANON_EXEMPT:
            continue
        cid = resolver.resolve(oid, "model").canonical_id
        if cid is None:
            no_match.append(oid)
        elif ("/" in oid and "/" in cid
              and not _org_aware_equal(_org_of(cid), _org_of(oid), hf_to_dev)
              and _leaf_norm(cid) != _leaf_norm(oid)):  # not a same-model org correction
            wrong_dev.append((oid, cid))
    assert not no_match, (
        f"{len(no_match)} oracle-snapshot canonical(s) now resolve to no_match "
        f"(already-seen resolution regressed). First few: {sorted(no_match)[:15]}"
    )
    assert not wrong_dev, (
        f"{len(wrong_dev)} oracle-snapshot canonical(s) now resolve to a DIFFERENT "
        f"developer AND a different model leaf (silent re-attribution). First few: {wrong_dev[:15]}"
    )


@pytest.mark.slow
def test_phase0_oracle_aliases_preserved(resolver):
    """GATE: every oracle-snapshot alias raw still resolves NON-NULL — for ITS OWN
    entity_type (model / benchmark / org / metric / harness / composite / family),
    with the alias's source_config — so no already-seen resolution regresses for
    ANY entity type, not just models. The coverage backstop the HF-id coverage
    gate cannot provide for non-EEE / non-model aliases."""
    import pandas as pd
    apath = ORACLE_SNAPSHOT_DIR / "aliases.parquet"
    adf = pd.read_parquet(apath)
    adf = adf[adf["status"] != "rejected"]
    no_match = set()
    for row in adf.itertuples():
        rv = getattr(row, "raw_value", None)
        et = getattr(row, "entity_type", None)
        if not isinstance(rv, str) or not rv or not isinstance(et, str):
            continue
        if rv in _ORACLE_ALIAS_EXEMPT or (et, rv) in _ORACLE_ALIAS_EXEMPT:
            continue
        sc = getattr(row, "source_config", None)
        sc = sc if isinstance(sc, str) and sc else None
        if resolver.resolve(rv, et, sc).canonical_id is None:
            no_match.add((et, rv))
    assert not no_match, (
        f"{len(no_match)} oracle-snapshot alias(es) now resolve to no_match for their "
        f"entity_type (already-seen resolution regressed). By type: "
        f"{ {t: sum(1 for tt, _ in no_match if tt == t) for t in sorted({tt for tt, _ in no_match})} }. "
        f"First few: {sorted(no_match)[:15]}"
    )


def test_model_display_names_are_humanized(models_df):
    """GATE: every canonical model carries a real, human-facing display_name —
    non-empty and never just the canonical id. The seed loader humanizes the
    tail at load time (tier-3 `inferred` raws, empty rows, and id-placeholder
    rows where a source set display_name to the id itself), so a regression here
    means a model would render as a raw slug in the frontend."""
    dn = models_df["display_name"].fillna("").astype(str).str.strip()
    ids = models_df["id"].astype(str)
    empty = sorted(models_df.loc[dn == "", "id"].tolist())
    id_placeholder = sorted(models_df.loc[(dn != "") & (dn == ids), "id"].tolist())
    assert not empty, f"{len(empty)} model(s) have an empty display_name: {empty[:15]}"
    assert not id_placeholder, (
        f"{len(id_placeholder)} model(s) have display_name == id (a raw slug, not a "
        f"label) — the load-time humanizer should have replaced it: {id_placeholder[:15]}"
    )


# Snapshot (oracle_id, parent_id) edges that are allowed not to land. Each is an
# enumerated, justified exception — any OTHER lost edge fails the gate.
#
# Dated-snapshot reclassification (finetune -> variant/version): each of these is
# a pinned release snapshot of its SAME-org base (e.g. gpt-5.4-pro-2026-03-05 of
# gpt-5.4-pro). The snapshot recorded the edge as `finetune`, which is not
# identity-preserving — it broke the model-group walk so every snapshot surfaced
# as a SEPARATE model page from its base. They are now `variant/version` edges
# (same release along the version line) so they fold into the base's group; the
# old `finetune` tuple intentionally no longer lands.
_ORACLE_EDGE_EXEMPT: frozenset = frozenset({
    # The snapshot's tier3 name-inference produced an INVERTED edge (the base
    # Qwen coder as a "finetune" of Intel's int4 quant). The raw now folds
    # onto the real HF base repo; the junk edge is deliberately dropped.
    ("alibaba/qwen-3-coder-480b", "Intel/Qwen3-Coder-480B-A35B-Instruct-int4-mixed-AutoRound"),
    # Snapshot holds the tier3 draft edge; the real parent is
    # deepseek-ai/DeepSeek-Coder-V2-Base (finetune, per hub-stats baseModels).
    ("deepseek-ai/DeepSeek-Coder-V2-Instruct", "deepseek/deepseek-coder-v2"),
    # Snapshot edge is inverted: Lite-Base is the pretrained ORIGIN (parents: []),
    # not a finetune of the Lite umbrella (which folds to Lite-Instruct) — keeping
    # it would make Lite-Base a finetune of its own instruct sibling (a cycle).
    ("deepseek/deepseek-coder-v2-lite-base", "deepseek/deepseek-coder-v2-lite"),
    # Dated snapshots re-typed from the snapshot's conservative `finetune` to the
    # correct identity-preserving `variant/version` (same parent) — a relationship
    # refinement, not a lost edge.
    ("google/gemini-1.5-pro-0514", "google/gemini-1.5-pro"),
    ("google/gemini-1.5-pro-0924", "google/gemini-1.5-pro"),
    ("google/gemini-exp-1114", "google/gemini-exp"),
    ("google/gemini-exp-1121", "google/gemini-exp"),
    ("alibaba/qwen3-vl-plus-2025-09-23", "alibaba/qwen3-vl-plus"),
    ("anthropic/claude-sonnet-4-5-thinking-20250929", "anthropic/claude-sonnet-4.5-thinking"),
    ("openai/gpt-4-5-2025-02-27", "openai/gpt-4.5"),
    ("openai/gpt-4-turbo-2024-04-09", "openai/gpt-4-turbo"),
    ("openai/gpt-4.1-2025-04-14", "openai/gpt-4.1"),
    ("openai/gpt-4.1-nano-2025-04-14", "openai/gpt-4.1-nano"),
    ("openai/gpt-4.5-preview-2025-02-27", "openai/gpt-4.5"),
    ("openai/gpt-5-mini-2025-08-07", "openai/gpt-5-mini"),
    ("openai/gpt-5-nano-2025-08-07", "openai/gpt-5-nano"),
    ("openai/gpt-5-pro-2025-10-06", "openai/gpt-5-pro"),
    ("openai/gpt-5.1-high-2025-11-12", "openai/gpt-5.1"),
    ("openai/gpt-5.1-instant-2025-11-12", "openai/gpt-5.1"),
    ("openai/gpt-5.1-medium-2025-11-12", "openai/gpt-5.1"),
    ("openai/gpt-5.1-thinking-2025-11-12", "openai/gpt-5.1-thinking"),
    ("openai/gpt-5.2-2025-12-11", "openai/gpt-5.2"),
    ("openai/gpt-5.2-pro-2025-12-11", "openai/gpt-5.2-pro"),
    ("openai/gpt-5.4-2026-03-05", "openai/gpt-5.4"),
    ("openai/gpt-5.4-pro-2026-03-05", "openai/gpt-5.4-pro"),
})


@pytest.mark.slow
def test_phase0_oracle_edges_preserved(resolver, models_df):
    """GATE: every oracle-snapshot TYPED parent edge (relationship + axis:
    variant/finetune/quantized/merge/adapter, all axes incl. training_stage) on
    a surviving canonical is preserved — repointed to the parent's surviving
    canonical. Guards the full typed lineage GRAPH, not just the derived
    group/family/lineage_origin fields. Edges whose parent model is genuinely
    absent from the registry (no surviving canonical to link to) are allowed."""
    import json as _json
    oc = pd.read_parquet(ORACLE_SNAPSHOT_DIR / "canonical_models.parquet")

    def _edges(v):
        if isinstance(v, str):
            try:
                return _json.loads(v) or []
            except Exception:
                return []
        return v if isinstance(v, list) else []

    def _edge_key(target, rel, axis):
        # `axis` is a variant-only concept (the closed enum lives on variant
        # edges); a stray axis on a finetune/quantized/merge/adapter edge —
        # e.g. the field-merge of two sources typing the same edge
        # differently — must not read as a lost edge.
        return (target, rel, axis if rel == "variant" else None)

    cur: dict[str, set] = {}
    for r in models_df.itertuples():
        cur[str(r.id)] = {
            _edge_key(e.get("id"), e.get("relationship"), e.get("axis"))
            for e in _edges(getattr(r, "parents", None)) if isinstance(e, dict)
        }

    home: dict[str, str] = {}

    def _home(i: str):
        if i not in home:
            home[i] = resolver.resolve(i, "model").canonical_id
        return home[i]

    missing, parent_gone = [], 0
    for row in oc.itertuples():
        oes = _edges(getattr(row, "parents", None))
        if not oes:
            continue
        t = _home(str(row.id))
        if t is None:
            continue  # canonical gone -> covered by the snapshot-canonicals gate above
        tedges = cur.get(t, set())
        for e in oes:
            if not isinstance(e, dict) or not e.get("id"):
                continue
            tp = _home(str(e["id"]))
            if tp is None:
                parent_gone += 1
                continue
            if tp == t:
                continue
            key = _edge_key(tp, e.get("relationship"), e.get("axis"))
            if (str(row.id), str(e["id"])) in _ORACLE_EDGE_EXEMPT:
                continue
            if key not in tedges:
                missing.append((str(row.id), t, str(e["id"]), tp, e.get("relationship"), e.get("axis")))
    assert not missing, (
        f"{len(missing)} oracle-snapshot typed parent edge(s) lost on the surviving "
        f"canonical (relationship/axis link not preserved). First few: {missing[:15]}"
    )


@pytest.mark.slow
def test_resolve_surfaces_typed_edges_and_ancestry(resolver, models_df):
    """GATE: the typed edge graph is queryable AT RESOLVE TIME, not merely stored.
    For every model canonical with stored parents, resolve(id) surfaces a
    non-empty `parents` list; and when the model belongs to a group/family (root
    != id), resolve(id) surfaces a non-empty `ancestry` chain. Guards that edge
    preservation (the parents-enrichment) actually flows through resolution +
    ancestry computation — not just sits in the fixtures."""
    missing_parents, missing_ancestry = [], []
    for r in models_df.itertuples():
        cid = str(r.id)
        if not _parse_parents(getattr(r, "parents", None)):
            continue
        res = resolver.resolve(cid, "model")
        if not res.parents:
            missing_parents.append(cid)
        roots = {
            x for x in (getattr(r, "model_family_id", None), getattr(r, "model_group_id", None))
            if isinstance(x, str) and x != cid
        }
        if roots and not res.ancestry:
            missing_ancestry.append(cid)
    assert not missing_parents, (
        f"{len(missing_parents)} model(s) with stored parents whose resolve() surfaces "
        f"NO typed edges. First few: {missing_parents[:15]}"
    )
    assert not missing_ancestry, (
        f"{len(missing_ancestry)} grouped/family model(s) whose resolve() surfaces NO "
        f"ancestry chain. First few: {missing_ancestry[:15]}"
    )


# Snapshot (field, id) lineage memberships that cannot be preserved: the group/
# family ROOT model is not a distinct canonical in the registry (its base was
# coarsened into a variant or is absent), so the variant has no root to group
# under. Enumerated, justified — each variant resolves correctly but as a
# singleton root.
_ORACLE_LINEAGE_EXEMPT: frozenset = frozenset({
    # Gemini dated snapshots re-typed finetune -> variant/version (a dated
    # snapshot is the SAME identity as its family, not a finetune of it), so
    # they correctly no longer carry a separate lineage origin.
    ("lineage_origin_model_id", "google/gemini-1.5-pro-0514"),
    ("lineage_origin_model_id", "google/gemini-1.5-pro-0924"),
    ("lineage_origin_model_id", "google/gemini-exp-1114"),
    ("lineage_origin_model_id", "google/gemini-exp-1121"),
    # DeepSeek-Coder-V2-Instruct family root shifted when its parent edge was
    # corrected from the tier3 draft `deepseek/deepseek-coder-v2` to the
    # authoritative deepseek-ai/DeepSeek-Coder-V2-Base (see _ORACLE_EDGE_EXEMPT).
    ("model_family_id", "deepseek-ai/DeepSeek-Coder-V2-Instruct"),
    # Folds into the curated HF-true DeepSeek-Coder-V2-Lite-Base (a clean root);
    # its snapshot lineage_origin was the deepseek/deepseek umbrella.
    ("lineage_origin_model_id", "deepseek/deepseek-coder-v2-lite-base"),
    ("model_group_id", "Qwen/Qwen3-235B-A22B-Instruct-2507"),
    ("model_group_id", "Qwen/Qwen3-VL-32B-Thinking"),
    ("model_group_id", "Qwen/Qwen3-VL-8B-Thinking"),
    ("model_group_id", "microsoft/Phi-4-mini-reasoning"),
    ("model_family_id", "Qwen/Qwen3-30B-A3B"),
    ("model_family_id", "Qwen/Qwen3-VL-32B-Thinking"),
    ("model_family_id", "Qwen/Qwen3-VL-8B-Thinking"),
    ("model_family_id", "microsoft/Phi-4-mini-reasoning"),
    ("model_family_id", "xiaomi/mimo-v2-flash"),
    # The models.dev SLUG `nvidia/nemotron-3-super-120b-a12b` folds into the real
    # HF canonical nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 (curated, recent
    # model absent from the frozen HF oracle). The snapshot's non-trivial family
    # root for the slug was an artifact of the slug arrangement (a size edge to a
    # non-existent `…-120b` base; A12B is the MoE active-param notation, not a
    # separate size); the real canonical is correctly a singleton root.
    ("model_family_id", "nvidia/nemotron-3-super-120b-a12b"),
    # Dated-snapshot reclassification (finetune -> variant/version): a pinned
    # release snapshot of its same-org base is the SAME model along the version
    # line, so it correctly has NO finetune/quant lineage origin. The snapshot's
    # (buggy) finetune edge walked to a lineage_origin; that origin is
    # intentionally dropped now the edge is variant/version. (model_group_id is
    # GAINED here, not lost — these now fold into the base group.)
    ("lineage_origin_model_id", "alibaba/qwen3-vl-plus-2025-09-23"),
    ("lineage_origin_model_id", "anthropic/claude-sonnet-4-5-thinking-20250929"),
    ("lineage_origin_model_id", "openai/gpt-4-5-2025-02-27"),
    ("lineage_origin_model_id", "openai/gpt-4-turbo-2024-04-09"),
    ("lineage_origin_model_id", "openai/gpt-4.1-2025-04-14"),
    ("lineage_origin_model_id", "openai/gpt-4.1-nano-2025-04-14"),
    ("lineage_origin_model_id", "openai/gpt-4.5-preview-2025-02-27"),
    ("lineage_origin_model_id", "openai/gpt-5-mini-2025-08-07"),
    ("lineage_origin_model_id", "openai/gpt-5-nano-2025-08-07"),
    ("lineage_origin_model_id", "openai/gpt-5-pro-2025-10-06"),
    ("lineage_origin_model_id", "openai/gpt-5.1-high-2025-11-12"),
    ("lineage_origin_model_id", "openai/gpt-5.1-instant-2025-11-12"),
    ("lineage_origin_model_id", "openai/gpt-5.1-medium-2025-11-12"),
    ("lineage_origin_model_id", "openai/gpt-5.1-thinking-2025-11-12"),
    ("lineage_origin_model_id", "openai/gpt-5.2-2025-12-11"),
    ("lineage_origin_model_id", "openai/gpt-5.2-pro-2025-12-11"),
    ("lineage_origin_model_id", "openai/gpt-5.4-2026-03-05"),
    ("lineage_origin_model_id", "openai/gpt-5.4-pro-2026-03-05"),
    # Normalize-collision fold (lib/collision_fold.py): these dated Claude 3.5
    # Sonnet snapshots are the surviving canonical after folding the dashed/
    # compact-date spelling in; as version snapshots they correctly carry a
    # variant/version edge (no finetune lineage origin). The oracle recorded a
    # finetune-derived origin on the folded spelling; intentionally dropped.
    ("lineage_origin_model_id", "anthropic/claude-3.5-sonnet-2024-06-20"),
    ("lineage_origin_model_id", "anthropic/claude-3.5-sonnet-2024-10-22"),
    # The snapshot origin came from the inverted tier3 edge to Intel's quant
    # (see _ORACLE_EDGE_EXEMPT); the raw now folds onto the real HF base repo,
    # which is correctly at origin.
    ("lineage_origin_model_id", "alibaba/qwen-3-coder-480b"),
    # Bare tier3 scaffolds (`meta/llama4-maverick`, `…-scout`) carried a
    # finetune edge to the real Llama-4 canonicals; they are now curated
    # ALIASES of those canonicals (core skip + enrichment bridge), which are
    # themselves at origin — the scaffold's self-referential origin is
    # intentionally gone.
    ("lineage_origin_model_id", "meta/llama4-maverick"),
    ("lineage_origin_model_id", "meta/llama4-scout"),
    # The `microsoft/phi-4-multimodal` mint (the API spelling) merged into the
    # real repo Phi-4-multimodal-instruct; the snapshot's family root WAS that
    # mint, so the surviving canonical is now correctly its own family root.
    ("model_family_id", "microsoft/Phi-4-multimodal-instruct"),
    # The curated edge between the Phi-4-mini siblings now points the right
    # way: Phi-4-mini-REASONING is the mode variant derived from
    # Phi-4-mini-instruct (the served "phi-4-mini"), not vice versa. The
    # snapshot's family root on instruct walked the inverted edge; instruct is
    # correctly its own family root now.
    ("model_family_id", "microsoft/Phi-4-mini-instruct"),
    # Curated fold (core.yaml skip_source_ids): the tier3 `…-reasoning` mint and
    # the models.dev slug stub both fold into the real HF repo
    # Qwen/Qwen3-Omni-30B-A3B-Thinking ("reasoning" IS the Thinking repo, not a
    # finetune of a base). The snapshot's lineage origin was an artifact of the
    # tier3-inferred finetune edge to the slug stub; the real canonical is
    # correctly at origin (no base Qwen/Qwen3-Omni-30B-A3B canonical exists).
    ("lineage_origin_model_id", "alibaba/qwen3-omni-30b-a3b-reasoning"),
})


@pytest.mark.slow
def test_phase0_oracle_lineage_no_loss(resolver, models_df):
    """GATE: no oracle-snapshot canonical LOSES its lineage. For every surviving
    snapshot canonical that had a non-trivial model_group_id / model_family_id /
    lineage_origin_model_id (root != self, or non-null lineage origin), the
    canonical it now resolves to must ALSO have that field non-trivial.

    Checks PRESENCE, not exact value: the union of snapshot + generator edges may
    walk to a deeper/truer origin (an ENRICHMENT, not a regression). The exact raw
    edges are guarded by test_phase0_oracle_edges_preserved; this guards the
    DERIVED lineage fields against being dropped (so e.g. a change to the org-map
    or lineage-derivation can't silently un-group a model)."""
    oc = pd.read_parquet(ORACLE_SNAPSHOT_DIR / "canonical_models.parquet")
    cur = {str(x.id): x for x in models_df.itertuples()}
    home: dict = {}

    def _home(i: str):
        if not isinstance(i, str):
            return None
        if i not in home:
            home[i] = resolver.resolve(i, "model").canonical_id
        return home[i]

    def _v(x, f):
        v = getattr(x, f, None)
        return v if isinstance(v, str) and v else None

    lost = []
    for ob in oc.itertuples():
        oid = str(ob.id)
        t = _home(oid)
        if t is None or t not in cur:
            continue  # canonical gone -> guarded elsewhere
        T = cur[t]
        for f in ("model_group_id", "model_family_id", "lineage_origin_model_id"):
            ov = _v(ob, f)
            nontrivial = ov is not None and (f == "lineage_origin_model_id" or ov != oid)
            if not nontrivial or (f, oid) in _ORACLE_LINEAGE_EXEMPT:
                continue
            tv = _v(T, f)
            t_nontrivial = tv is not None and (f == "lineage_origin_model_id" or tv != t)
            if not t_nontrivial:
                lost.append((f, oid, t))
    assert not lost, (
        f"{len(lost)} oracle canonical(s) LOST a lineage field (group/family/origin "
        f"dropped to trivial/null). By field: "
        f"{ {f: sum(1 for ff, _, _ in lost if ff == f) for f in sorted({ff for ff, _, _ in lost})} }. "
        f"First few: {lost[:12]}"
    )


# --------------------------------------------------------------------------
# BARE-NAME OWNERSHIP (prefer-official gate) — when several models share a
# normalized leaf name, the un-namespaced alias must belong to the owner the
# seed precedence selects (lab first, else the unique confirmed entry), not
# to whichever entry a regen happens to seed first. Guards the redirect in
# cli.py `seed` (lab_leaf_owner / strong_leaf_owner) against silent flips
# from the daily source refreshes.
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def aliases_df() -> pd.DataFrame:
    df = pd.read_parquet(FIXTURES_DIR / "aliases.parquet")
    assert not df.empty
    return df


def _nz(v) -> str:
    """NA/None-safe str."""
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    return str(v)


@pytest.fixture(scope="module")
def leaf_ownership(models_df, orgs_df):
    """Recompute the leaf-ownership maps from the OUTPUT tables (the seed
    builds them from the merged YAML; org_id in fixtures is the derived
    superset of the load-time signal, so this is the post-hoc view of the
    same precedence)."""
    from eval_entity_resolver.normalization import normalize

    kind = dict(zip(orgs_df["id"].astype(str), orgs_df["kind"].astype(str)))
    lab_claimants: dict[str, set[str]] = {}
    strong_claimants: dict[str, set[str]] = {}
    strong_ids: set[str] = set()
    lab_ids: set[str] = set()
    for r in models_df.itertuples():
        cid = _nz(r.id)
        is_lab = kind.get(_nz(r.org_id)) == "lab"
        if is_lab:
            lab_ids.add(cid)
        if "/" not in cid:
            continue
        confirmed = _nz(r.resolution_source) == "hf" or _nz(r.review_status) == "reviewed"
        if not confirmed:
            continue
        norm = normalize(cid.split("/", 1)[1])
        strong_ids.add(cid)
        strong_claimants.setdefault(norm, set()).add(cid)
        if is_lab:
            lab_claimants.setdefault(norm, set()).add(cid)
    return {
        "lab_claimants": lab_claimants,
        "strong_claimants": strong_claimants,
        "strong_ids": strong_ids,
        "lab_ids": lab_ids,
    }


def test_lab_leaf_ownership_is_unambiguous(leaf_ownership):
    """No two confirmed LAB models may share a normalized leaf — the seed's
    lab_leaf_owner keeps the first claimant, so a second would make the bare
    name's owner load-order-dependent across regens."""
    multi = {n: sorted(c) for n, c in leaf_ownership["lab_claimants"].items() if len(c) > 1}
    assert multi == {}, f"{len(multi)} normalized leaf(s) owned by 2+ lab models: {multi}"


def test_bare_alias_points_at_lab_leaf_owner(aliases_df, leaf_ownership):
    """Every global model alias whose normalized form IS a lab-confirmed
    model's leaf must resolve to that lab model (prefer-official, tier 1)."""
    from eval_entity_resolver.normalization import normalize

    lab_claimants = leaf_ownership["lab_claimants"]
    mal = aliases_df[
        (aliases_df["entity_type"] == "model") & aliases_df["source_config"].isna()
    ]
    bad = []
    for raw, cid in zip(mal["raw_value"].astype(str), mal["canonical_id"].astype(str)):
        owners = lab_claimants.get(normalize(raw))
        if owners and len(owners) == 1 and cid not in owners:
            bad.append((raw, cid, next(iter(owners))))
    assert bad == [], (
        f"{len(bad)} bare alias(es) not pointing at their lab leaf owner "
        f"(raw, points_at, lab_owner): {bad[:10]}"
    )


def test_bare_alias_points_at_unique_confirmed_owner(aliases_df, leaf_ownership):
    """Prefer-official tier 2: an alias whose normalized form matches exactly
    ONE confirmed model's leaf (and no lab's) must resolve to it — unless the
    alias's own entry is itself confirmed or lab-owned (the seed never takes
    a name from a real repo or a lab entry, only from unconfirmed drafts)."""
    from eval_entity_resolver.normalization import normalize

    lab_claimants = leaf_ownership["lab_claimants"]
    strong_claimants = leaf_ownership["strong_claimants"]
    strong_ids = leaf_ownership["strong_ids"]
    lab_ids = leaf_ownership["lab_ids"]
    mal = aliases_df[
        (aliases_df["entity_type"] == "model") & aliases_df["source_config"].isna()
    ]
    bad = []
    for raw, cid in zip(mal["raw_value"].astype(str), mal["canonical_id"].astype(str)):
        norm = normalize(raw)
        if norm in lab_claimants:
            continue  # tier 1 governs
        owners = strong_claimants.get(norm)
        if not owners or len(owners) != 1:
            continue  # no unique confirmed owner -> no expectation
        owner = next(iter(owners))
        if cid == owner or cid in strong_ids or cid in lab_ids:
            continue  # exempt: alias belongs to a real repo / lab entry
        bad.append((raw, cid, owner))
    assert bad == [], (
        f"{len(bad)} draft-owned alias(es) shadowing a unique confirmed repo "
        f"(raw, points_at, confirmed_owner): {bad[:10]}"
    )


def test_every_model_id_is_a_resolvable_raw(models_df, aliases_df):
    """Every canonical model id must appear as an alias raw_value — the
    anti-shadow display drop must never eat an entity's own id alias (it did
    once: an id-placeholder display colliding with a lab norm discarded the
    shared string from the alias set)."""
    raws = set(
        aliases_df[aliases_df["entity_type"] == "model"]["raw_value"].astype(str)
    )
    missing = [i for i in models_df["id"].astype(str) if i not in raws]
    assert missing == [], f"{len(missing)} model id(s) with no alias row: {missing[:10]}"


# --------------------------------------------------------------------------
# OPENROUTER ID ADOPTION gates (specs/model-id-resolution PLAN.md G3).
# --------------------------------------------------------------------------
MODELSDEV_SNAPSHOT = REGISTRY_ROOT / "tests" / "fixtures" / "modelsdev_api.snapshot.json"
OPENROUTER_BEFORE_SNAPSHOT = REGISTRY_ROOT / "curation" / "openrouter_adoption_before.parquet"
SEED_MODELS_DIR = REGISTRY_ROOT / "seed" / "models"


def _import_refresh_module():
    scripts_dir = REGISTRY_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import refresh_from_modelsdev as rfm  # noqa: E402

    return rfm


def _load_source_entries() -> list[tuple[str, dict]]:
    """(source-name, entry) for every record in core.yaml + sources/*.generated.yaml."""
    out: list[tuple[str, dict]] = []
    paths = [SEED_MODELS_DIR / "core.yaml"] + sorted(
        (SEED_MODELS_DIR / "sources").glob("*.generated.yaml")
    )
    for p in paths:
        doc = yaml.safe_load(p.read_text())
        entries = doc.get("entries", doc) if isinstance(doc, dict) else doc
        for e in entries or []:
            if isinstance(e, dict) and e.get("id"):
                out.append((p.name, e))
    return out


def _core_skip_ids() -> set[str]:
    doc = yaml.safe_load((SEED_MODELS_DIR / "core.yaml").read_text())
    if not isinstance(doc, dict):
        return set()
    return set(doc.get("skip_ids") or []) | set(doc.get("skip_source_ids") or [])


def test_no_parent_edge_names_a_renamed_away_id():
    """GATE (adoption 1/5): no `parents[].id` in any source or in core names a
    renamed-away id — every edge target must be a FULL entry's canonical id in
    the merged source universe, never a form that survives only as an ALIAS on
    some entry (which is what a renamed-away id becomes)."""
    recs = _load_source_entries()
    skip = _core_skip_ids()
    full_ids = {e["id"] for _, e in recs if "display_name" in e and e["id"] not in skip}
    alias_owner: dict[str, str] = {}
    for _, e in recs:
        for a in e.get("aliases") or []:
            if a:
                alias_owner.setdefault(a, e["id"])
    bad = []
    for src, e in recs:
        if e["id"] in skip:
            continue
        for edge in e.get("parents") or []:
            t = edge.get("id") if isinstance(edge, dict) else None
            if t and t not in full_ids:
                bad.append((src, e["id"], t, alias_owner.get(t)))
    assert bad == [], (
        f"{len(bad)} parent edge(s) point at a non-canonical id (renamed-away "
        f"ids resolve only as aliases; 4th field = the alias's owner). "
        f"First few: {bad[:12]}"
    )


# Pinned PRE-EXISTING non-adoption duplicate-identity clusters (gate 2's
# ratchet baseline): spelling twins that predate the adoption gates and are
# follow-up curation, not adoption regressions. The tail may only SHRINK —
# any cluster not listed here fails the gate as a NEW duplicate.
_PREEXISTING_DUP_TAIL: frozenset = frozenset({
    ("ai2/molmo-7b-d", "allenai/molmo-7b-d"),
    ("ai2/olmo-3-7b-instruct", "allenai/olmo-3-7b-instruct"),
    ("ai2/olmo-3-7b-think", "allenai/olmo-3-7b-think"),
    ("ai21-labs/ai21-jamba-1-5-large", "ai21-labs/jamba-1-5-large"),
    ("ai21-labs/ai21-jamba-1-5-mini", "ai21-labs/jamba-1-5-mini"),
    ("ai21/jamba-1.6-large", "ai21/jamba-large-1-6", "ai21labs/jamba-large-1.6"),
    ("ai21/jamba-1.6-mini", "ai21/jamba-mini-1-6", "ai21labs/jamba-mini-1.6"),
    ("ai21/jamba-1.7-mini", "ai21/jamba-mini-1-7"),
    ("alibaba/qwen-14b-chat", "alibaba/qwen-chat-14b"),
    ("alibaba/qwen-72b-chat", "alibaba/qwen-chat-72b"),
    ("alibaba/qwen3-4b-2507-instruct", "qwen/qwen3-4b-instruct-2507"),
    ("amazon/amazon.nova-lite-v1:0", "amazon/nova-lite", "amazon/nova-lite-v1:0", "aws/nova-lite"),
    ("amazon/amazon.nova-micro-v1:0", "amazon/nova-micro", "amazon/nova-micro-v1:0", "aws/nova-micro"),
    ("amazon/amazon.nova-pro-v1:0", "amazon/nova-pro", "amazon/nova-pro-v1:0", "aws/nova-pro"),
    ("amazon/nova-premier", "amazon/nova-premier-v1:0", "aws/nova-premier"),
    ("anthropic/Claude-3.5-Sonnet(Oct)", "anthropic/claude-3.5-sonnet", "anthropic/claude-sonnet-3.5"),
    ("anthropic/anthropic/claude-3-7-sonnet-20250219", "anthropic/claude-3-7-sonnet-20250219"),
    ("anthropic/claude-4-opus-20250514", "anthropic/claude-opus-4-20250514"),
    ("anthropic/claude-4-opus-thinking", "anthropic/claude-opus-4-thinking"),
    ("anthropic/claude-4-sonnet-20250514", "anthropic/claude-sonnet-4-20250514"),
    ("anthropic/claude-4-sonnet-thinking", "anthropic/claude-sonnet-4-thinking"),
    ("anthropic/claude-4.5-opus-thinking", "anthropic/claude-opus-4-5-thinking"),
    ("anthropic/claude-4.5-sonnet-thinking", "anthropic/claude-sonnet-4.5-thinking"),
    ("anthropic/claude-4.6-opus-thinking", "anthropic/claude-opus-4-6-thinking"),
    ("google/gemini-1.5-flash", "google/gemini-flash-1-5"),
    # Veo 3.1 vs Veo 3.1 Fast are DISTINCT video products; they cluster only
    # because the tag-strip fold eats the `fast` token. False positive, kept
    # apart deliberately (2026-08-13 models.dev catch-up).
    ("google/veo-3-1", "google/veo-3.1-fast"),
    ("google/text-bison", "google/text-bison@001"),
    ("google/text-unicorn", "google/text-unicorn@001"),
    ("lmms-lab/llava-video-7b", "lmms-lab/video-llava-7b"),
    ("meta-llama/Llama-3.1-70B-Instruct", "meta/llama-3-1-instruct-70b"),
    ("meta-llama/Llama-3.1-8B-Instruct", "meta/llama-3-1-instruct-8b"),
    ("meta-llama/Llama-3.2-1B-Instruct", "meta/llama-3-2-instruct-1b"),
    ("meta-llama/Llama-3.2-3B-Instruct", "meta/llama-3-2-instruct-3b"),
    ("meta-llama/Llama-3.2-90B-Vision-Instruct", "meta/llama-3-2-instruct-90b-vision"),
    ("meta-llama/Llama-3.3-70B-Instruct", "meta/llama-3-3-instruct-70b"),
    ("meta-llama/llama-3.1", "meta/Llama 3.1"),
    ("meta/Llama-2-13b-chat", "meta/llama-2-chat-13b"),
    ("meta/Llama-2-70b-chat", "meta/llama-2-chat-70b"),
    ("meta/Llama-2-7b-chat", "meta/llama-2-chat-7b"),
    ("meta/llama-3-1-405b-instruct", "meta/llama-3-1-instruct-405b"),
    ("mosaicml/MPT-Instruct-30B", "mosaicml/mpt-30b-instruct"),
    ("nvidia/Llama-3.1-Nemotron-70B-Instruct", "nvidia/llama-3-1-nemotron-instruct-70b"),
    ("openai/Lingma Agent + Lingma SWE-GPT 72b (v0918)", "openai/Lingma Agent + Lingma SWE-GPT 72b (v0925)"),
    ("openai/Lingma Agent + Lingma SWE-GPT 7b (v0918)", "openai/Lingma Agent + Lingma SWE-GPT 7b (v0925)"),
    ("tiiuae/Falcon-Instruct-40B", "tiiuae/falcon-40b-instruct"),
    ("tiiuae/Falcon-Instruct-7B", "tiiuae/falcon-7b-instruct"),
    ("xai/grok-3-mini", "xai/grok-3-mini-fast"),
    ("xai/grok-4", "xai/grok-4-fast"),
    ("xai/grok-4-1", "xai/grok-4-1-fast"),
})


def test_no_two_full_entries_share_folded_identity(hf_to_dev):
    """GATE (adoption 2/5): no ADOPTION-TOUCHED cluster of FULL entries shares
    the same curated-org-folded, tag-stripped, order-insensitive
    variant-preserving identity — the duplicate class the carry-forward would
    otherwise retain forever when an id is renamed (`anthropic/claude-haiku-3`
    vs the adopted `anthropic/claude-3-haiku`). A cluster counts as
    adoption-touched when at least one member is an OpenRouter-adopted entry
    (`metadata.openrouter_adopted`) OR carries an id present in the frozen
    OpenRouter snapshot key set — the flag alone must not scope the gate,
    because an invented mint that already EQUALS the key is the same id-thrash
    class. The registry carries a separate pre-existing tail of non-adoption
    dup clusters (jamba spelling twins, nova sentinel twins, …) that predate
    this gate: that tail is PINNED below and may only SHRINK — a new
    non-adoption dup cluster fails the gate too. Clusters made ONLY of
    real-HF-attested repos are allowed (two genuinely distinct repos may
    normalize alike); a size-signature difference also keeps entries apart
    (opt-1.3b vs opt-13b)."""
    rfm = _import_refresh_module()
    from eval_card_registry.lib.collision_fold import _bsizes

    oracle_fixed = set()
    for v in json.loads(ORACLE_PATH.read_text())["resolutions"].values():
        fx = v.get("fixed_hf_model_id")
        if isinstance(fx, str) and "/" in fx:
            oracle_fixed.add(fx)

    def _attested(src: str, e: dict) -> bool:
        if src in ("hf_oracle.generated.yaml", "hub_stats.generated.yaml"):
            return True
        if e.get("resolution_source") == "hf" or e["id"] in oracle_fixed:
            return True
        md = e.get("metadata")
        if isinstance(md, str):
            try:
                m = json.loads(md)
                if m.get("hf_deferred") or m.get("hf_id") == e["id"]:
                    return True
            except (ValueError, TypeError):
                pass
        return False

    snap = json.loads(MODELSDEV_SNAPSHOT.read_text())
    or_keys: set = set()
    for k in (snap.get("openrouter", {}).get("models") or {}):
        or_keys.add(k)
        or_keys.add(k.lstrip("~"))

    skip = _core_skip_ids()
    clusters: dict[tuple, list[tuple[str, dict]]] = defaultdict(list)
    for src, e in _load_source_entries():
        if "display_name" not in e or e["id"] in skip or "/" not in e["id"]:
            continue
        org, name = e["id"].split("/", 1)
        dev = hf_to_dev.get(org.lower(), org.lower())
        clusters[(dev, rfm._identity_sig(name), _bsizes(name))].append((src, e))

    def _adopted(e: dict) -> bool:
        md = e.get("metadata")
        if isinstance(md, str):
            try:
                return json.loads(md).get("openrouter_adopted") is True
            except (ValueError, TypeError):
                return False
        return False

    dups, tail = [], []
    for key, members in clusters.items():
        ids = sorted({e["id"] for _, e in members})
        if len(ids) < 2:
            continue
        if all(_attested(src, e) for src, e in members):
            continue  # real-HF cluster — allowed
        touched = any(_adopted(e) for _, e in members) or any(
            i in or_keys for i in ids
        )
        (dups if touched else tail).append(ids)
    assert dups == [], (
        f"{len(dups)} folded-identity duplicate cluster(s) involving an "
        f"OpenRouter-adopted / OpenRouter-key id (adoption id-thrash class): "
        f"{dups[:10]}"
    )
    # RATCHET on the pre-existing non-adoption dup tail: it may only SHRINK.
    # A cluster not in the pinned set is a NEW duplicate (fails, visible); a
    # pinned cluster that disappears just shrinks the tail (fine). Re-pin
    # deliberately when curating one away.
    new_tail = [c for c in sorted(map(tuple, tail)) if c not in _PREEXISTING_DUP_TAIL]
    assert new_tail == [], (
        f"{len(new_tail)} NEW non-adoption folded-identity dup cluster(s) not in "
        f"the pinned pre-existing tail: {new_tail[:10]}"
    )
    assert len(tail) <= len(_PREEXISTING_DUP_TAIL), (
        f"pre-existing dup tail grew: {len(tail)} > {len(_PREEXISTING_DUP_TAIL)}"
    )


@pytest.mark.slow
def test_every_openrouter_snapshot_key_resolves(resolver):
    """GATE (adoption 3/5): every OpenRouter model key in the frozen snapshot
    resolves — to an HF canonical through the crossmap, to its own adopted
    canonical, or to another owning canonical (all count as success).
    `openrouter/*` router pseudo-endpoints are routing products, excluded."""
    snap = json.loads(MODELSDEV_SNAPSHOT.read_text())
    keys = [
        k for k in (snap.get("openrouter", {}).get("models") or {})
        if not k.lstrip("~").lower().startswith("openrouter/")
    ]
    assert keys, "frozen snapshot lost its openrouter provider"
    unresolved = [
        k for k in keys if resolver.resolve(k, "model").canonical_id is None
    ]
    assert unresolved == [], (
        f"{len(unresolved)}/{len(keys)} OpenRouter snapshot key(s) resolve to "
        f"no_match: {unresolved[:15]}"
    )


_BARE_ID_CEILING = 250


def test_bare_canonical_org_ambiguity_is_explicit(models_df):
    """Bare upstream ids are legitimate but lack a namespace from which the
    generator can safely infer a developer. Their count naturally grows with
    models.dev, so a frozen size ceiling turns every upstream addition into a
    manual cron outage. Pin the correctness property instead: a bare row has
    ``org-unknown`` exactly when its ``org_id`` remains unresolved.

    This keeps the ambiguous tail visible without inventing org ownership. The
    ceiling retains the existing growth ratchet with bounded headroom for a
    normal upstream refresh.
    """
    bad = []
    for row in models_df.itertuples():
        cid = str(row.id)
        if "/" in cid:
            continue
        org_id = getattr(row, "org_id", None)
        has_org = isinstance(org_id, str) and bool(org_id)
        tagged_unknown = "org-unknown" in _parse_tags(getattr(row, "tags", None))
        if has_org == tagged_unknown:
            bad.append((cid, org_id if has_org else None, tagged_unknown))
    assert bad == [], (
        f"{len(bad)} bare canonical id(s) have inconsistent org visibility; "
        f"expected org-unknown iff org_id is null: {bad[:15]}"
    )
    bare = [str(i) for i in models_df["id"].astype(str) if "/" not in str(i)]
    assert len(bare) <= _BARE_ID_CEILING, (
        f"bare (org-less) canonical id count grew: {len(bare)} > "
        f"{_BARE_ID_CEILING}; inspect new upstream mints before re-pinning"
    )


def test_canonical_id_prefix_is_external_or_folds_to_org(models_df, hf_to_dev):
    """GATE (adoption 4/5, the amended id-space rule): every canonical id
    prefix is a real external namespace adopted verbatim (HF-attested or an
    OpenRouter catalog key whose prefix org AGREES with the entry's curated
    developer — the same `_openrouter_org_agrees` fold the generator's
    adoption guard applies) OR folds through the curated org map to the
    entry's `org_id` (case/separator-insensitively). Invented ids may only use
    the curated developer prefix. Bare ids are checked separately by
    ``test_bare_canonical_org_ambiguity_is_explicit``."""
    from eval_entity_resolver.fold import _norm_org_key

    oracle_fixed = set()
    for v in json.loads(ORACLE_PATH.read_text())["resolutions"].values():
        fx = v.get("fixed_hf_model_id")
        if isinstance(fx, str) and "/" in fx:
            oracle_fixed.add(fx)
    snap = json.loads(MODELSDEV_SNAPSHOT.read_text())
    or_keys = set()
    for k in (snap.get("openrouter", {}).get("models") or {}):
        or_keys.add(k)
        or_keys.add(k.lstrip("~"))

    def fold(o: str) -> str:
        return hf_to_dev.get(o.lower(), o)

    # Curated core.yaml defs are explicit judgment calls: a curated entry may
    # deliberately keep a raw's mis-prefixed spelling as its id while carrying
    # the corrected developer org (e.g. `alibaba/dolphin-2-9.2-qwen2-72b`,
    # org cognitivecomputations).
    core_doc = yaml.safe_load((SEED_MODELS_DIR / "core.yaml").read_text())
    core_defs = {
        e["id"] for e in (core_doc.get("entries") or [])
        if isinstance(e, dict) and e.get("id") and "display_name" in e
    }

    bad = []
    for row in models_df.itertuples():
        cid = str(row.id)
        if "/" not in cid:
            continue
        org_id = getattr(row, "org_id", None)
        org_id = org_id if isinstance(org_id, str) and org_id else None
        prefix = cid.split("/", 1)[0]
        # External namespaces adopted verbatim; curated core judgment calls.
        if cid in oracle_fixed or cid in core_defs:
            continue
        if cid in or_keys:
            # or_keys exemption requires prefix-org agreement (the generator's
            # `_openrouter_org_agrees` fold): a key under a different
            # developer's namespace must not launder a mis-attributed row.
            a0, b0 = fold(prefix), fold(org_id) if org_id else None
            if org_id and (a0 == b0 or _norm_org_key(a0) == _norm_org_key(b0)):
                continue
        rsrc = getattr(row, "resolution_source", None)
        if isinstance(rsrc, str) and rsrc == "hf":
            continue
        md = getattr(row, "metadata", None)
        if isinstance(md, str):
            try:
                m = json.loads(md)
                if m.get("hf_id") == cid or m.get("hf_deferred") or m.get("openrouter_adopted"):
                    continue
            except (ValueError, TypeError):
                pass
        # Invented / community ids: prefix must fold to the entry's org.
        if org_id is None:
            if prefix.lower() == "unknown":
                continue
            bad.append((cid, None))
            continue
        a, b = fold(prefix), fold(org_id)
        if a == b or _norm_org_key(a) == _norm_org_key(b):
            continue
        bad.append((cid, org_id))
    assert bad == [], (
        f"{len(bad)} canonical id(s) whose prefix neither is an adopted external "
        f"namespace nor folds to the entry's org_id: {bad[:15]}"
    )


# Deliberate curated CORRECTIONS during the adoption migration: raws whose
# pre-migration resolution was itself a mis-attribution (a turbo serving
# spelling bridged onto the base product, a base spelling bridged onto a
# sibling variant). Reviewed exceptions only.
_ADOPTION_REGRESSION_EXEMPT: frozenset = frozenset({
    "z-ai-glm-5-turbo",     # was bridged onto zai-org/GLM-5 (base); turbo is a sibling product
    "z-ai-glm-5v-turbo",    # was bridged onto the 5V base record; same correction
    # 2026-08-13 catch-up corrections: raw now reaches the TRUE product entity
    # that didn't exist before the refresh (previous home was a normalized/
    # fuzzy near-miss, not this model).
    "Veo-3.1-Fast",                        # was google/veo3-1; now the real google/veo-3.1-fast
    "deepseek/deepseek-v4-pro-lightning",  # was fuzzy-folded onto DeepSeek-V4-Pro; Lightning is its own product
    # Sentinel-org scaffold raw: tier3 keeps it on its org-less draft
    # (`unknown/amazon.nova-premier-v10`); the org-present twin draft
    # `amazon/nova-premier-v1:0` still resolves by its own id. Same model,
    # split across the pre-existing tier3 sentinel drafts — reviewed.
    "unknown/amazon.nova-premier-v1:0",
    # The 5V-Turbo split's inverse leg: the BASE spellings had been absorbed
    # by the turbo entity; both now resolve to the 5V base `zai/z-ai-glm-5v`.
    "glm-5v",
    "zai/glm-5v",
    # The served "phi-4-mini" IS microsoft/Phi-4-mini-instruct (API names
    # drop -instruct); the reasoning attribution named a different model.
    "Phi-4 Mini",
    "microsoft/phi-4-mini",
})


@pytest.mark.slow
def test_openrouter_adoption_no_resolution_regression(resolver, aliases_df):
    """GATE (adoption 5/5, no-regression): every canonical id and alias that
    resolved before the adoption migration (snapshotted in
    curation/openrouter_adoption_before.parquet) still resolves to the same
    entity — an identity shift counts as the same entity only when the old
    canonical id is carried as an alias on the new entry.

    Provenance of the snapshot: a one-shot resolve sweep over ALL seeded
    aliases + canonical ids (18,999 raws) at the pre-adoption state — the
    parent of commit 1104fc5 ("chore(gates): snapshot pre-OpenRouter-adoption
    model resolution state"), which committed the parquet."""
    before = pd.read_parquet(OPENROUTER_BEFORE_SNAPSHOT)
    alias_pairs = set(
        zip(
            aliases_df[aliases_df["entity_type"] == "model"]["raw_value"].astype(str),
            aliases_df[aliases_df["entity_type"] == "model"]["canonical_id"].astype(str),
        )
    )
    old_home: dict[str, str | None] = {}

    def _old_home(old: str):
        # The old-id-follows rescue accepts EXACT/NORMALIZED resolution ONLY.
        # A fuzzy hit must not count: it would launder a dropped draft whose
        # raw degraded from an exact alias (1.0) to a fuzzy stem match (0.9)
        # as a "rename" — the tier3 carry-forward floor exists precisely so
        # every dropped draft's forms stay exact/normalized-resolvable.
        if old not in old_home:
            res = resolver.resolve(old, "model")
            old_home[old] = (
                res.canonical_id
                if res.strategy in ("exact", "normalized")
                else None
            )
        return old_home[old]

    no_match, moved = [], []
    for raw, old in zip(before["raw_value"].astype(str), before["canonical_id"].astype(str)):
        if raw in _ADOPTION_REGRESSION_EXEMPT:
            continue
        got = resolver.resolve(raw, "model").canonical_id
        if got is None:
            no_match.append(raw)
            continue
        if got == old or (old, got) in alias_pairs:
            continue
        # Same entity iff the OLD canonical id itself follows to the entity
        # the raw now resolves to (rename/fold with the old id resolvable),
        # via alias or normalized match.
        if _old_home(old) == got:
            continue
        moved.append((raw, old, got))
    assert no_match == [], (
        f"{len(no_match)} previously-resolving raw(s) now no_match: {no_match[:15]}"
    )
    assert moved == [], (
        f"{len(moved)} raw(s) resolved to a DIFFERENT entity whose aliases do NOT "
        f"carry the old id (identity break, not a rename): {moved[:15]}"
    )


def test_prefer_official_pinned_examples(aliases_df):
    """Concrete regression pins for the ownership precedence — each one is a
    case that WAS wrong (or luck-dependent) before the redirect existed."""
    mal = aliases_df[
        (aliases_df["entity_type"] == "model") & aliases_df["source_config"].isna()
    ]
    lookup = dict(zip(mal["raw_value"].astype(str), mal["canonical_id"].astype(str)))
    pins = {
        # fork's bare display must not shadow the official repo (tier 1)
        "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
        # models.dev draft with a mis-attributed org prefix loses the bare
        # name to the real hf repo (tier 2)
        "dolphin-2-9.2-qwen2-72b": "dphn/dolphin-2.9.2-qwen2-72b",
        # tier3 bare-slug draft loses its id-shaped claim to the curated
        # official entry that lists it as an alias
        "llama-4-maverick-17b-128e-instruct": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        # a name-inferred `meta/...` draft must NOT own the bare name of a
        # real community repo (confirmed-only ownership)
        "Reflection-Llama-3.1-70B": "mattshumer/Reflection-Llama-3.1-70B",
        # the entity healed by the display==id guard resolves by its own id
        "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL": "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL",
    }
    wrong = {
        raw: (lookup[raw], want)
        for raw, want in pins.items()
        if raw in lookup and lookup[raw] != want
    }
    gone = [raw for raw in pins if raw not in lookup]
    assert wrong == {}, f"pinned bare-name owner(s) regressed (raw: (got, want)): {wrong}"
    assert gone == [], (
        f"pinned raw(s) no longer seeded at all — the contested surface form "
        f"disappeared from the seed (upstream refresh dropped it?), not an "
        f"ownership regression: {gone}"
    )


# ---------------------------------------------------------------------------
# Benchmark entity invariants
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def benchmarks_df() -> pd.DataFrame:
    path = REGISTRY_ROOT / "fixtures" / "canonical_benchmarks.parquet"
    if not path.exists():
        pytest.skip("fixtures/canonical_benchmarks.parquet not present (run seed --local)")
    return pd.read_parquet(path)


def test_no_separator_split_benchmarks(benchmarks_df):
    """No two benchmark ids differ only by separators/case (`wild-bench` vs
    `wildbench`) — each such pair fragments one benchmark's results across
    two entities and two eval pages. The seed loader enforces the same rule
    at seed time (incl. aliases); this gate keeps the committed fixtures
    honest. Genuinely distinct pairs go in
    seed/benchmarks_distinct_allowlist.yaml."""
    allow_path = REGISTRY_ROOT / "seed" / "benchmarks_distinct_allowlist.yaml"
    allow: list[set] = []
    if allow_path.exists():
        for group in yaml.safe_load(allow_path.read_text()) or []:
            allow.append(set(group))
    norm = lambda x: re.sub(r"[^a-z0-9]", "", str(x).lower())
    by: dict = defaultdict(list)
    for bid in benchmarks_df["id"]:
        by[norm(bid)].append(str(bid))
    split = {
        k: v for k, v in by.items()
        if len(v) > 1 and not any(set(v) <= grp for grp in allow)
    }
    assert split == {}, f"{len(split)} separator/case-split benchmark group(s): {list(split.values())[:8]}"


def test_no_slash_in_benchmark_ids(benchmarks_df):
    """'/' in a benchmark id would let a merged-view single-segment eval id
    collide with per-source composite/benchmark ids."""
    bad = [b for b in benchmarks_df["id"] if "/" in str(b)]
    assert bad == [], f"benchmark ids containing '/': {bad[:8]}"
