#!/usr/bin/env python3
"""
Generate the HF source-of-truth seed from the frozen HuggingFace oracle
(`hf_model_id_resolution.json`).

OFFLINE. The only HF data source is the oracle JSON at the evaleval root,
which carries, for each raw EEE `model_id`, the HF-true repo id
(`fixed_hf_model_id`) and a `resolution_status` in
{fixed_exact, fixed_near_miss, unresolved_not_found_or_inaccessible}.
We materialise the 4,074 HF-present ids (fixed_exact + fixed_near_miss).

What this script does (deterministic, re-runnable):

1. Builds a SINGLE-SOURCED two-tier org map (`hf_to_dev`) from
   `canonical_orgs.hf_org` AND `strategies/fuzzy.py:_ORG_ALIASES` — no third
   hardcoded copy. Applies the canon-id rule:
     - curated developer namespace (meta-llama->meta, qwen->alibaba, ...) =>
       canonical id `{dev_slug}/{HF-NAME}`, org_id = dev_slug;
     - otherwise => `{HF-ORG}/{HF-NAME}` verbatim (HF org casing preserved),
       org_id = HF-ORG.
   The model-NAME part ALWAYS keeps HF casing, case-sensitive.

2. Computes each oracle entry's target canonical:
     - fixed_exact:      target = canon_id(fixed_hf_model_id)  (== raw)
     - fixed_near_miss:  target = canon_id(fixed_hf_model_id)  (the HF REDIRECT
       repo, possibly a different org); the raw EEE id becomes an ALIAS -> target.
   In all cases the raw EEE id is a confirmed alias of the target.

3. Partitions targets against the CURRENT canonical_models fixtures:
     - present exact (HF-cased id already there)  -> ok, nothing minted;
     - present case-insensitively (lowercase id)  -> RE-KEY in place;
     - brand new                                  -> MINT.

4. Writes/overwrites:
   (a) seed/models/sources/hf_oracle.generated.yaml — the brand-new canonicals
       (id HF-cased, org_id, display_name, resolution_source: hf,
       review_status: reviewed, aliases incl raw EEE id(s) + lowercase form).
       Sparse metadata; NO invented parents.
   (b) RE-KEYS the colliding lowercase canonicals IN-PLACE wherever they live
       (core.yaml / sources/models_dev.generated.yaml /
       sources/hub_stats.generated.yaml): id lowercase->HF-cased, old lowercase
       appended to that entry's aliases, raw EEE id(s) appended to aliases.
   (c) Builds a GLOBAL rename map {old_lower -> hf_cased} and rewrites EVERY
       parents[].id reference AND model_group_id / model_family_id /
       lineage_origin_model_id references across ALL model entries in ALL three
       files so no edge dangles.
   (d) seed/orgs.generated.yaml — the new HF-cased orgs (kind community,
       review_status reviewed, hf_org set). Curated orgs.yaml wins on collision.

Usage:
    LOCAL_MODE=true uv run python scripts/generate_hf_oracle_seed.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from eval_card_registry.lib.seed_io import (
    build_hf_to_dev_from_orgs_yaml,
    resolve_oracle_path,
)

from eval_entity_resolver.strategies.fuzzy import _ORG_ALIASES

REGISTRY_ROOT = Path(__file__).resolve().parents[1]            # eval-card-registry/
ORACLE = resolve_oracle_path()

SEED = REGISTRY_ROOT / "seed"
FIXTURES = REGISTRY_ROOT / "fixtures"
MODELS_DIR = SEED / "models"
CORE_YAML = MODELS_DIR / "core.yaml"
SOURCES_DIR = MODELS_DIR / "sources"
HF_ORACLE_YAML = SOURCES_DIR / "hf_oracle.generated.yaml"
MODELS_DEV_YAML = SOURCES_DIR / "models_dev.generated.yaml"
HUB_STATS_YAML = SOURCES_DIR / "hub_stats.generated.yaml"
ORGS_YAML = SEED / "orgs.yaml"
ORGS_GENERATED_YAML = SEED / "orgs.generated.yaml"

# Files whose model entries may carry edge references that must be rewritten on
# re-key (in-place candidates + their parents/group/family pointers).
EDIT_FILES = [CORE_YAML, MODELS_DEV_YAML, HUB_STATS_YAML]
EDGE_ID_KEYS = ("model_group_id", "model_family_id", "lineage_origin_model_id")


def _load_yaml_entries(path: Path) -> tuple[object, list[dict]]:
    """Return (raw_doc, entries-list-view). core.yaml uses a
    {skip_ids, skip_source_ids, entries} dict; sources are flat lists."""
    if not path.exists():
        return None, []
    with open(path) as f:
        doc = yaml.safe_load(f) or []
    if isinstance(doc, dict):
        return doc, doc.get("entries", []) or []
    return doc, doc


def _entries_view(doc: object) -> list[dict]:
    if isinstance(doc, dict):
        return doc.get("entries", []) or []
    return doc if isinstance(doc, list) else []


def _load_curated_orgs() -> list[dict]:
    """Curated orgs from seed/orgs.yaml (the source of truth — fixtures are a
    build artifact and may be stale/polluted, so we never read them here)."""
    if not ORGS_YAML.exists():
        return []
    with open(ORGS_YAML) as f:
        return [e for e in (yaml.safe_load(f) or []) if isinstance(e, dict)]


def build_hf_to_dev(curated_orgs: list[dict]) -> dict[str, str]:
    """HF-org-lowercase -> curated developer slug (see
    `eval_card_registry.lib.seed_io.build_hf_to_dev_from_orgs_yaml`). Reading the alias tier lets
    ai2->allenai / aws->amazon / kimi->moonshotai / prime-intellect->PrimeIntellect
    resolve instead of dangling. `curated_orgs` is accepted for call-site
    compatibility; the org map is rebuilt from `ORGS_YAML` (identical result)."""
    return build_hf_to_dev_from_orgs_yaml(ORGS_YAML)


def canon_id(hf_model_id: str, hf_to_dev: dict[str, str]) -> tuple[str, str]:
    """Returns (canonical_id, org_id). canonical_id = the real HF repo id
    verbatim (org is never folded into the id); org_id = the curated parent
    if the HF org maps to one, else the HF org itself."""
    org_part, name_part = hf_model_id.split("/", 1)
    org_id = hf_to_dev.get(org_part.lower(), org_part)
    return f"{org_part}/{name_part}", org_id


# --- near-miss identity guard ----------------------------------------------
# HF `fixed_near_miss` redirects are "did-you-mean" matches that sometimes map a
# raw EEE string to a DIFFERENT model (size/generation/uploader swap). Such a
# redirect must NOT become a confirmed alias (it mis-resolves real evals). A
# curated denylist (audit_bad_nearmiss.json) confirmed a set of these; we also
# block any near-miss that changes a genuine size token or crosses a developer
# org. Conservative on purpose — never flags version-vs-size
# (`Yi-1.5-34B`) or MoE active-param (`A22B`), which are the same model.
_NM_SIZE_RE = re.compile(r'(?<![a-z0-9.])(\d+(?:\.\d+)?)b(?![a-z])', re.I)
_NM_ACTIVE_RE = re.compile(r'(?<![a-z0-9])a\d+(?:\.\d+)?b', re.I)


def _nm_sizes(name: str) -> set[str]:
    return set(_NM_SIZE_RE.findall(_NM_ACTIVE_RE.sub(" ", name.lower())))


def _load_audit_bad_nearmiss() -> frozenset[str]:
    p = REGISTRY_ROOT  / "curation" / "audit_bad_nearmiss.json"
    try:
        return frozenset(json.loads(p.read_text()).get("raws", []))
    except FileNotFoundError:  # pragma: no cover
        return frozenset()


_AUDIT_BAD_NEARMISS = _load_audit_bad_nearmiss()


def _fold_org(org: str, hf_to_dev: dict[str, str]) -> str:
    """Developer identity of an HF org spelling: apply the curated two-tier remap
    THEN collapse separators + case to one token. So a pure separator/case rename
    of the SAME uploader folds equal (`DeepAutoAI` == `DeepAuto-AI`), while a
    genuine cross-uploader move stays distinct (`ai4free` != `helpingai`)."""
    return re.sub(r"[^a-z0-9]", "", hf_to_dev.get(org.lower(), org.lower()))


def nearmiss_changes_identity(raw: str, fixed: str, hf_to_dev: dict[str, str]) -> str | None:
    """Reason a fixed_near_miss raw->fixed redirect must NOT be aliased (changes
    model identity), else None. Signals: audit denylist; genuine size change;
    cross-developer org.

    The org check is SEPARATOR/CASE aware (via `_fold_org`): an owner RENAME of
    the same uploader (`DeepAutoAI/X` -> `DeepAuto-AI/X`, same model name) is the
    same developer and SHOULD be aliased onto the HF-true canonical, so it is NOT
    flagged. A genuine cross-uploader migration (`AI4free/X` -> `HelpingAI/X`) is
    a different developer and stays flagged (handled via audit_bad_nearmiss so it
    resolves to its own canonical, never mis-aliased cross-developer)."""
    if raw in _AUDIT_BAD_NEARMISS:
        return "audit-confirmed wrong redirect"
    rn, fn = raw.split("/", 1)[-1], fixed.split("/", 1)[-1]
    rs, fs = _nm_sizes(rn), _nm_sizes(fn)
    if rs and fs and rs.isdisjoint(fs):
        return f"size change {sorted(rs)}->{sorted(fs)}"
    if "/" in raw and "/" in fixed:
        ro = raw.split("/", 1)[0]
        if ro.lower() != "unknown":
            rd = _fold_org(ro, hf_to_dev)
            fd = _fold_org(fixed.split("/", 1)[0], hf_to_dev)
            if rd != fd:
                return f"cross-dev-org {rd}!={fd}"
    return None


# --- authoritative-repo marker --------------------------------------------
# Every hf_oracle mint id IS the real HF repo id (canon_id returns it verbatim),
# so the mint is AUTHORITATIVE: it must WIN id+casing over any colliding curated
# slug, never be suppressed in favor of one. We record `metadata.hf_id == id` so
# consumers (and the seed loader's merge) recognise the id must not be rewritten.
# When a real HF repo and a dev-org slug name the same model, the slug canonical
# is the one removed — never the authoritative HF repo.


def authoritative_metadata(hf_repo_id: str) -> str:
    """metadata JSON marking a mint as the authoritative real HF repo
    (`hf_id == id`). An id whose metadata.hf_id equals itself must never be
    rewritten/suppressed in favor of a different (e.g. dev-org-slug) id."""
    return json.dumps({"hf_id": hf_repo_id}, sort_keys=True)


def main() -> None:
    oracle = json.loads(ORACLE.read_text())["resolutions"]
    curated_orgs = _load_curated_orgs()

    hf_to_dev = build_hf_to_dev(curated_orgs)

    # "Existing" = every model id authored in the seed YAML source files (the
    # SOURCE OF TRUTH). We deliberately do NOT read fixtures/*.parquet: those
    # are a regenerated build artifact and become polluted after a `seed`
    # materialises the mints, which would make a re-run see everything as
    # already-present. Reading only the git-tracked YAML keeps the generator
    # deterministic and re-runnable from a clean checkout.
    existing_ids: set[str] = set()
    for path in EDIT_FILES:
        _doc, _entries = _load_yaml_entries(path)
        for e in _entries:
            # Only DEFINING records count as existing: an enrich record (no
            # display_name — e.g. every hub_stats row, which enriches a
            # canonical defined elsewhere) defines nothing, and counting it
            # would make a non-reset re-run classify every oracle target as
            # "already present" and gut this file down to the residual mints.
            if (
                isinstance(e, dict)
                and isinstance(e.get("id"), str)
                and "display_name" in e
            ):
                existing_ids.add(e["id"])
    existing_lower: dict[str, str] = {}
    for cid in sorted(existing_ids):
        existing_lower.setdefault(cid.lower(), cid)

    existing_org_ids: set[str] = {
        e["id"] for e in curated_orgs if isinstance(e.get("id"), str)
    }

    # --- Incumbent index: which canonical does each HF/raw id resolve to TODAY?
    # The fixtures lag the YAML, so we read the seed YAML aliases (the truth)
    # plus each entry's own id. Two lookups: exact raw_value and normalized.
    from eval_card_registry.services.hub_stats import normalize as _nz

    incumbent_exact: dict[str, str] = {}
    incumbent_norm: dict[str, str] = {}

    def _claim_incumbent(raw_value: str, cid: str) -> None:
        if not isinstance(raw_value, str) or not raw_value:
            return
        incumbent_exact.setdefault(raw_value, cid)
        incumbent_norm.setdefault(_nz(raw_value), cid)

    for path in EDIT_FILES:
        _doc, _entries = _load_yaml_entries(path)
        for e in _entries:
            if not isinstance(e, dict):
                continue
            cid = e.get("id")
            if not isinstance(cid, str):
                continue
            _claim_incumbent(cid, cid)
            dn = e.get("display_name")
            if isinstance(dn, str):
                _claim_incumbent(dn, cid)
            for a in e.get("aliases") or []:
                if isinstance(a, str):
                    _claim_incumbent(a, cid)

    def incumbent_for(*ids: str) -> Optional[str]:
        for i in ids:
            if i in incumbent_exact:
                return incumbent_exact[i]
        for i in ids:
            n = _nz(i)
            if n in incumbent_norm:
                return incumbent_norm[n]
        return None

    # --- Compute targets ---------------------------------------------------
    # target_canonical -> {"org_id","name","raws":set,"fixed":set}
    targets: dict[str, dict] = {}
    skipped_nearmiss: list[tuple[str, str, str]] = []
    for raw, meta in oracle.items():
        status = meta.get("resolution_status")
        if status not in ("fixed_exact", "fixed_near_miss"):
            continue
        fixed = meta.get("fixed_hf_model_id")
        if not isinstance(fixed, str) or "/" not in fixed:
            continue
        tgt, org_id = canon_id(fixed, hf_to_dev)
        name = fixed.split("/", 1)[1]
        bucket = targets.setdefault(
            tgt, {"org_id": org_id, "name": name, "raws": set(), "fixed": set()}
        )
        bucket["fixed"].add(fixed)
        # The raw EEE id is always a confirmed alias of the target. For
        # fixed_exact raw == fixed (a no-op self-alias the seed CLI emits);
        # for fixed_near_miss raw differs -> real alias — UNLESS the redirect
        # changes model identity (a bad HF "did-you-mean"); those are NOT aliased
        # (the raw resolves to its own/corrected canonical via Tier-3 instead).
        if raw != tgt:
            reason = (nearmiss_changes_identity(raw, fixed, hf_to_dev)
                      if status == "fixed_near_miss" else None)
            if reason:
                skipped_nearmiss.append((raw, fixed, reason))
            else:
                bucket["raws"].add(raw)
    if skipped_nearmiss:
        print(f"[hf-oracle] guarded {len(skipped_nearmiss)} identity-changing near_miss "
              f"redirect(s) (NOT aliased): e.g. {skipped_nearmiss[:5]}")

    # --- Partition: ok / re-key / mint ------------------------------------
    # A target's INCUMBENT is the canonical its raw/fixed ids resolve to today.
    #   - incumbent == target            -> ok (already correct casing).
    #   - incumbent is a casing-variant  -> RE-KEY incumbent id -> target
    #       (same model, name-token-equal under normalize). The HF casing wins.
    #   - incumbent token-different OR none -> MINT a new HF-cased canonical.
    #       When an incumbent (a curated short-slug like meta/llama-3-8b) still
    #       claims the HF/raw id as an alias, we STRIP that alias off the
    #       incumbent so resolve(raw) reaches the new HF canonical (no collision).
    ok_targets: dict[str, dict] = {}
    rekey_targets: dict[str, dict] = {}   # target -> info (+ old_id)
    mint_targets: dict[str, dict] = {}
    # aliases to remove from incumbents, keyed by incumbent canonical id:
    incumbent_alias_strip: dict[str, set[str]] = {}

    for tgt, info in targets.items():
        hf_name = tgt.split("/", 1)[1]
        lookup_ids = sorted(info["fixed"]) + sorted(info["raws"]) + [tgt]
        inc = incumbent_for(*lookup_ids)

        # When a token-equal incumbent exists, PRESERVE its org identity (a
        # curated dev org like `cohere` must not be re-orged to a raw HF org
        # like `CohereLabs`). The re-key only adopts the HF NAME casing under
        # the incumbent's existing org. Re-key key becomes {inc_org}/{hf_name}.
        if inc is not None and _nz(inc.split("/", 1)[-1]) == _nz(hf_name):
            inc_org = inc.split("/", 1)[0]
            # The re-key adopts the HF NAME casing but KEEPS the incumbent's id
            # prefix (the real HF repo namespace, e.g. `zai-org`, or a curated
            # dev slug). The curated developer is recorded on `org_id` ONLY —
            # canonical_id stays the real HF repo id, decoupled from the org (the
            # two-tier rule). Folding the prefix into the dev slug here
            # (`zai-org/x` -> `zai/x`) would re-mint a shadow id that collides
            # with the committed `zai/x` -> `zai-org/x` enrichment alias and abort
            # the seed.
            dev_org = hf_to_dev.get(inc_org.lower(), inc_org)
            rekey_tgt = f"{inc_org}/{hf_name}"
            if rekey_tgt == inc:
                # Name already correctly cased under the incumbent's id prefix.
                # "ok" ONLY when a DEFINING record elsewhere carries the id —
                # an incumbency claimed solely by enrich records (hub_stats
                # rows / this generator's own prior output reflected there)
                # means THIS file is the definer and must re-emit the entry,
                # or a non-reset re-run would gut the Tier-1 source.
                if rekey_tgt in existing_ids:
                    ok_targets[rekey_tgt] = info
                else:
                    mint_targets[rekey_tgt] = info
                continue
            # Guard: one re-key per incumbent. If another target already claims
            # this incumbent, fold the current ids in as aliases on that re-key
            # (don't double-rename one entry).
            existing_rk = next(
                (t for t, i in rekey_targets.items() if i["old_id"] == inc),
                None,
            )
            if existing_rk is not None:
                rekey_targets[existing_rk]["raws"] |= info["raws"]
                rekey_targets[existing_rk]["fixed"] |= info["fixed"]
                continue
            rk = dict(
                info,
                old_id=inc,
                org_id=dev_org,
                raws=set(info["raws"]),
                fixed=set(info["fixed"]),
            )
            rekey_targets[rekey_tgt] = rk
            continue

        if tgt in existing_ids:
            ok_targets[tgt] = info
            continue
        if tgt.lower() in existing_lower and existing_lower[tgt.lower()] != tgt:
            # Pure case-variant id exists (no alias incumbent) -> re-key it.
            rekey_targets[tgt] = dict(info, old_id=existing_lower[tgt.lower()])
            continue

        # near_miss where the incumbent's ID *is* the raw repo (HF renamed the
        # repo entirely, e.g. `allenai/olmo-1.7-7b` -> `allenai/OLMo-7B-0424`).
        # resolve(raw) hits the incumbent by exact id, so stripping aliases is
        # not enough — MERGE the incumbent into the redirect target by re-keying
        # its id (the raw becomes an alias). Skip if that incumbent is already a
        # re-key source.
        all_src_ids = {x.lower() for x in (info["fixed"] | info["raws"])}
        if (
            inc is not None
            and inc.lower() in all_src_ids
            and not any(i["old_id"] == inc for i in rekey_targets.values())
            and tgt not in rekey_targets
        ):
            rekey_targets[tgt] = dict(
                info, old_id=inc, raws=set(info["raws"]), fixed=set(info["fixed"])
            )
            continue

        # Mint. Record the HF/raw ids to strip off a token-different incumbent
        # (e.g. a curated short-slug) so resolution moves to the new canonical.
        if inc is not None and inc != tgt:
            strip = incumbent_alias_strip.setdefault(inc, set())
            for x in info["fixed"] | info["raws"]:
                strip.add(x)
        mint_targets[tgt] = info

    # Global rename map for edge rewriting: old_id -> hf_cased target.
    rename_map: dict[str, str] = {
        info["old_id"]: tgt for tgt, info in rekey_targets.items()
    }

    # Case-collision consolidation: a re-key/mint may produce an HF-cased id
    # that case-collides with a DIFFERENT existing id (e.g. a re-key
    # `minimaxai/MiniMax-M2.1`->`minimax/MiniMax-M2.1` clashing with a curated
    # `minimax/minimax-m2.1`). Fold those stragglers into the HF-cased target so
    # no case-duplicate pair survives (the dedup obligation: the HF-cased id
    # wins, the old lowercase becomes an alias).
    final_targets = (
        set(rekey_targets) | set(mint_targets) | set(ok_targets)
    )
    final_by_lower = {t.lower(): t for t in final_targets}
    for cid in existing_ids:
        if cid in final_targets:
            continue
        winner = final_by_lower.get(cid.lower())
        if winner is not None and winner != cid and cid not in rename_map:
            rename_map[cid] = winner
            # carry its old id as an alias on the winner where possible
            for tgt, info in rekey_targets.items():
                if tgt == winner:
                    info.setdefault("raws", set()).add(cid)
                    break

    # --- (b)+(c) Re-key in place across all edit files --------------------
    # First pass: collect which entry id lives in which file; then rewrite.
    rekey_by_old = {info["old_id"]: (tgt, info) for tgt, info in rekey_targets.items()}
    rekeyed_count = 0

    import copy as _copy

    docs: dict[Path, object] = {}
    docs_before: dict[Path, object] = {}
    for path in EDIT_FILES:
        doc, _ = _load_yaml_entries(path)
        docs[path] = doc
        docs_before[path] = _copy.deepcopy(doc)

    def _rewrite_edges_in_entry(entry: dict) -> None:
        # parents[].id
        parents = entry.get("parents")
        if isinstance(parents, list):
            for p in parents:
                if isinstance(p, dict) and isinstance(p.get("id"), str):
                    if p["id"] in rename_map:
                        p["id"] = rename_map[p["id"]]
        # scalar edge id pointers
        for key in EDGE_ID_KEYS:
            v = entry.get(key)
            if isinstance(v, str) and v in rename_map:
                entry[key] = rename_map[v]

    stripped_alias_count = 0
    for path in EDIT_FILES:
        entries = _entries_view(docs[path])
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            cid = entry.get("id")
            # Re-key this entry's own id if it is a collision target.
            if isinstance(cid, str) and cid in rekey_by_old:
                tgt, info = rekey_by_old[cid]
                entry["id"] = tgt
                # Append old id + raw EEE id(s) + the HF-true id(s) to aliases
                # (dedup) so resolution still hits the old id and the redirect
                # sources continue to resolve.
                aliases = list(entry.get("aliases") or [])
                seen = {a.lower() for a in aliases if isinstance(a, str)}
                for extra in [cid, *sorted(info["raws"]), *sorted(info["fixed"])]:
                    if extra and extra != tgt and extra.lower() not in seen:
                        aliases.append(extra)
                        seen.add(extra.lower())
                entry["aliases"] = aliases
                if isinstance(entry.get("org_id"), str):
                    entry["org_id"] = info["org_id"]
                rekeyed_count += 1
            # Case-collision straggler: id case-collides with an HF-cased target
            # owned by another entry — fold this one into it (old id -> alias).
            elif isinstance(cid, str) and cid in rename_map:
                winner = rename_map[cid]
                entry["id"] = winner
                aliases = list(entry.get("aliases") or [])
                seen = {a.lower() for a in aliases if isinstance(a, str)}
                if cid.lower() not in seen and cid != winner:
                    aliases.append(cid)
                entry["aliases"] = aliases
                if isinstance(entry.get("org_id"), str):
                    win_org = winner.split("/", 1)[0]
                    entry["org_id"] = hf_to_dev.get(win_org.lower(), win_org)
                rekeyed_count += 1
            # Strip HF/raw ids that a MINT is taking over from this incumbent
            # (token-different curated short-slugs) so the alias doesn't collide
            # and resolution moves to the new HF canonical.
            elif isinstance(cid, str) and cid in incumbent_alias_strip:
                strip = incumbent_alias_strip[cid]
                aliases = list(entry.get("aliases") or [])
                kept = [
                    a for a in aliases
                    if not (isinstance(a, str) and a in strip)
                ]
                stripped_alias_count += len(aliases) - len(kept)
                entry["aliases"] = kept
            # Always rewrite edge references (some entries point at a re-keyed
            # parent without themselves being a target).
            _rewrite_edges_in_entry(entry)

    # (Edit files persisted at the very end, after the global de-collision pass.)

    # --- Global claimed-alias set ----------------------------------------
    # Every alias the seed CLI will emit from SURVIVING (non-mint) entries:
    # their final id + display_name + explicit aliases. The seed run aborts if
    # two canonicals declare the same alias, so a mint must never re-emit one of
    # these, and the mints must not collide with each other. Keyed by EXACT
    # string (the CLI's collision check is exact, case-sensitive).
    claimed: dict[str, str] = {}  # alias -> owning canonical id

    def _claim(alias: str, owner: str) -> None:
        if isinstance(alias, str) and alias:
            claimed.setdefault(alias, owner)

    for path in EDIT_FILES:
        for e in _entries_view(docs[path]):
            if not isinstance(e, dict):
                continue
            cid = e.get("id")
            if not isinstance(cid, str):
                continue
            _claim(cid, cid)
            dn = e.get("display_name")
            if isinstance(dn, str):
                _claim(dn, cid)
            for a in e.get("aliases") or []:
                _claim(a, cid)

    # --- (a) Mint brand-new canonicals into hf_oracle.generated.yaml ------
    mint_entries: list[dict] = []
    for tgt in sorted(mint_targets):
        info = mint_targets[tgt]
        name = info["name"]
        org_id = info["org_id"]

        # display_name = HF NAME, unless that bare name is already claimed by a
        # different canonical (cross-org name clash, e.g. multiple
        # `zephyr-7b-beta`). Then qualify with the org so the auto-emitted
        # display alias stays unique. The id is always unique (HF-cased).
        if name in claimed and claimed[name] != tgt:
            display = tgt  # org-qualified, equals the id (no extra alias)
        else:
            display = name
            _claim(name, tgt)

        aliases: list[str] = []
        seen: set[str] = {tgt.lower(), display.lower()}

        def _add_alias(a: str) -> None:
            nonlocal aliases
            if not a or a.lower() in seen:
                return
            if a in claimed and claimed[a] != tgt:
                return  # already owned by another canonical — skip
            aliases.append(a)
            seen.add(a.lower())
            _claim(a, tgt)

        # raw EEE id(s) + HF-true id(s) + lowercase canonical form
        for a in sorted(info["raws"]) + sorted(info["fixed"]):
            _add_alias(a)
        low = tgt.lower()
        if low != tgt:
            _add_alias(low)

        entry = {
            "id": tgt,
            "display_name": display,
            "org_id": org_id,
            "resolution_source": "hf",
            "review_status": "reviewed",
            # tgt IS the real HF repo id -> mark it authoritative: never to
            # be rewritten/suppressed in favor of a colliding dev-org slug.
            "metadata": authoritative_metadata(tgt),
        }
        if aliases:
            entry["aliases"] = aliases
        mint_entries.append(entry)
        _claim(tgt, tgt)

    # --- Global de-collision pass ----------------------------------------
    # The seed CLI emits {id, display_name} ∪ aliases for every entry and ABORTS
    # if two canonicals declare the same exact alias. A few pre-existing seed
    # duplicates (e.g. competing minimax/skywork entries) plus the casing
    # re-keys can leave residual clashes. Deterministically keep the FIRST
    # declarer of each exact alias (priority: core > models_dev > hub_stats >
    # hf_oracle mints) and drop the duplicate alias from later entries. An
    # entry's own `id` is never dropped; on an id-vs-alias clash the alias loses.
    decollide_owner: dict[str, str] = {}
    dropped_collisions = 0
    all_entry_groups = [_entries_view(docs[p]) for p in EDIT_FILES] + [mint_entries]

    # Pass 1: every canonical `id` claims itself FIRST, globally. An id is
    # immutable, so it must win over any other entry's display_name/alias.
    for grp in all_entry_groups:
        for e in grp:
            if isinstance(e, dict) and isinstance(e.get("id"), str):
                decollide_owner.setdefault(e["id"], e["id"])

    # Pass 2: display_names then aliases, keeping the first non-id declarer.
    def _decollide(entries: list[dict]) -> None:
        nonlocal dropped_collisions
        for e in entries:
            if not isinstance(e, dict):
                continue
            cid = e.get("id")
            dn = e.get("display_name")
            if isinstance(dn, str) and dn != cid:
                owner = decollide_owner.get(dn)
                if owner is not None and owner != cid:
                    e["display_name"] = cid  # fall back to the (unique) id
                else:
                    decollide_owner.setdefault(dn, cid)
            new_aliases = []
            for a in e.get("aliases") or []:
                if not isinstance(a, str) or not a:
                    continue
                owner = decollide_owner.get(a)
                if owner is not None and owner != cid:
                    dropped_collisions += 1
                    continue
                decollide_owner.setdefault(a, cid)
                new_aliases.append(a)
            if "aliases" in e:
                e["aliases"] = new_aliases

    for grp in all_entry_groups:      # core, models_dev, hub_stats, then mints
        _decollide(grp)

    # Persist edited edit-files (preserve original doc shape). Only files whose
    # entries actually CHANGED are rewritten — a yaml re-dump of an untouched
    # file would still destroy its comments (hand-curated core.yaml carries
    # documentation comments a wholesale dump hoists).
    for path in EDIT_FILES:
        if docs[path] != docs_before[path]:
            with open(path, "w") as f:
                yaml.safe_dump(
                    docs[path], f, sort_keys=False, allow_unicode=True, width=10_000
                )

    header = (
        "# AUTO-GENERATED by scripts/generate_hf_oracle_seed.py — DO NOT HAND-EDIT.\n"
        "# HF source-of-truth canonicals (Tier-1), minted from the frozen oracle\n"
        "# hf_model_id_resolution.json. Ids carry HF-true casing per the two-tier\n"
        "# org rule. Metadata is sparse (params/release/open_weights\n"
        "# enriched later online); NO invented parents.\n"
    )
    with open(HF_ORACLE_YAML, "w") as f:
        f.write(header)
        yaml.safe_dump(
            mint_entries, f, sort_keys=False, allow_unicode=True, width=10_000
        )

    # --- (d) New HF-cased orgs -------------------------------------------
    # Union of org_ids referenced by mint + re-key targets that are not curated
    # and not already in canonical_orgs. Only auto-org (non-dev-remap) orgs
    # need creating — a dev-remap target's org_id is a curated slug already.
    needed_orgs: set[str] = set()
    for info in list(mint_targets.values()) + list(rekey_targets.values()):
        oid = info["org_id"]
        if oid and oid not in existing_org_ids:
            needed_orgs.add(oid)

    # Curated orgs.yaml wins on id collision AND on alias collision: the seed
    # CLI emits {id, display_name, hf_org, *aliases} as org aliases and aborts
    # if a new org id duplicates one (e.g. curated `huggingface` already claims
    # `HuggingFaceTB` as an alias). Exclude any needed org whose id is already
    # spoken for by a curated org's id/display_name/hf_org/alias.
    curated_org_claims: set[str] = set()
    if ORGS_YAML.exists():
        with open(ORGS_YAML) as f:
            for e in yaml.safe_load(f) or []:
                if not isinstance(e, dict):
                    continue
                for key in ("id", "display_name", "hf_org"):
                    v = e.get(key)
                    if isinstance(v, str) and v:
                        curated_org_claims.add(v)
                for a in e.get("aliases") or []:
                    if isinstance(a, str) and a:
                        curated_org_claims.add(a)
    needed_orgs -= curated_org_claims

    org_entries = []
    for oid in sorted(needed_orgs):
        org_entries.append(
            {
                "id": oid,
                "display_name": oid,
                "hf_org": oid,
                "kind": "community",
                "tags": "[]",
                "metadata": "{}",
                "review_status": "reviewed",
            }
        )
    org_header = (
        "# AUTO-GENERATED by scripts/generate_hf_oracle_seed.py — DO NOT HAND-EDIT.\n"
        "# HF-derived orgs (Tier-1 community namespaces). Curated seed/orgs.yaml\n"
        "# wins on id collision (loader merge in cli.py:_load_orgs_merged).\n"
    )
    with open(ORGS_GENERATED_YAML, "w") as f:
        f.write(org_header)
        yaml.safe_dump(
            org_entries, f, sort_keys=False, allow_unicode=True, width=10_000
        )

    # --- Report -----------------------------------------------------------
    print("=== generate_hf_oracle_seed ===")
    print(f"oracle fixed_exact+near_miss entries: "
          f"{sum(1 for v in oracle.values() if v.get('resolution_status') in ('fixed_exact','fixed_near_miss'))}")
    print(f"distinct target canonicals: {len(targets)}")
    print(f"  already present (ok, no work): {len(ok_targets)}")
    print(f"  re-keyed in place (collision): {len(rekey_targets)} "
          f"(entries rewritten: {rekeyed_count})")
    print(f"  brand-new canonicals (mint):   {len(mint_targets)}")
    print(f"wrote {HF_ORACLE_YAML.relative_to(REGISTRY_ROOT)}: {len(mint_entries)} entries")
    print(f"global rename map (old_id -> hf_cased): {len(rename_map)} entries")
    print(f"incumbent aliases stripped (taken over by mints): {stripped_alias_count}")
    print(f"residual collision aliases dropped (de-collision pass): {dropped_collisions}")
    print(f"wrote {ORGS_GENERATED_YAML.relative_to(REGISTRY_ROOT)}: "
          f"{len(org_entries)} new HF orgs")


if __name__ == "__main__":
    main()
