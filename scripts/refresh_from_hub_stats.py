#!/usr/bin/env python3
"""
Generate seed/models/sources/hub_stats.generated.yaml from
HuggingFace's `cfahlgren1/hub-stats` dataset.

Scope: BACKFILL ONLY. For HF ids (`org/name`) already aliased to one of
our canonicals, enrich the existing canonical with hub-stats metadata —
`release_date` from `createdAt`, `params_billions` approximated from
the safetensors total, license from cardData, useful tags from the
tag list. The seed loader's field-level merge fills in missing scalars
without overriding curated values in `seed/models/core.yaml`.

Lineage descendant pre-load (community quants/finetunes whose
`baseModels` chain points at our covered models) is intentionally
NOT done here — initial scoping showed ~177k descendants of our
4k HF ids, mostly LoRA adapters. The on-demand enrichment at draft
creation handles those when EEE actually encounters them.

Re-run efficiency: an etag-watermark in
`seed/models/sources/hub_stats.state.json` records the parquet's
content ETag and the candidate HF ids checked against it. When the
upstream parquet hasn't republished AND the candidate set is unchanged,
the script exits without querying — most cron cycles are no-ops.

Output:
    seed/models/sources/hub_stats.generated.yaml — enrichment entries
    that merge into existing canonicals at seed time.
    seed/models/sources/hub_stats.state.json     — etag watermark.

Reproducibility: by DEFAULT this reads the committed offline cache
`curation/hub_stats_frozen.parquet` (the candidate subset of the upstream
parquet incl. the `baseModels` lineage column), so a clean regen reproduces
lineage without depending on flaky live HF. Refresh that cache with
`scripts/freeze_hub_stats_cache.py` (live); pass `--live` here to bypass the
cache and query HF directly.

Usage:
    python scripts/refresh_from_hub_stats.py             # offline cache (reproducible)
    python scripts/refresh_from_hub_stats.py --live      # query live HF directly
    python scripts/refresh_from_hub_stats.py --dry-run   # preview only
    python scripts/refresh_from_hub_stats.py --force     # ignore etag (live only)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import duckdb
import httpx
import yaml

from eval_card_registry.lib.seed_io import build_hf_to_dev_from_orgs_yaml, load_entries_from_yaml

# Shared hub-stats helpers live in the package so the runtime resolver
# (live lookup at draft creation) and this bulk refresh script stay
# consistent on row-shape parsing.
from eval_card_registry.services.hub_stats import (
    HUB_STATS_LOCAL_PARQUET_ENV,
    PARQUET_URL,
    QUERY_COLUMNS,
    approx_params_billions as _approx_params_billions,
    coerce_date as _coerce_date,
    enrich_draft_from_row,
    extract_license as _extract_license,
    filter_useful_tags as _filter_useful_tags,
    hf_id_to_canonical_cased,
    is_local_parquet,
    normalize as _normalize,
    resolve_parquet_source,
    slugify as _slugify,
)

# Single-sourced two-tier org map (same as `_auto_create_entity._build_hf_to_dev`
# and `generate_hf_oracle_seed.build_hf_to_dev`): HF-org-lowercase -> developer
# slug, from `strategies/fuzzy.py:_ORG_ALIASES` + `canonical_orgs.hf_org`.
from eval_entity_resolver.strategies.fuzzy import _ORG_ALIASES

REPO_ROOT = Path(__file__).resolve().parent.parent
ORGS_PATH = REPO_ROOT / "seed" / "orgs.yaml"
MODELS_OUT_PATH = REPO_ROOT / "seed" / "models" / "sources" / "hub_stats.generated.yaml"
STATE_PATH = REPO_ROOT / "seed" / "models" / "sources" / "hub_stats.state.json"
CORE_PATH = REPO_ROOT / "seed" / "models" / "core.yaml"
MODELS_DEV_GENERATED = REPO_ROOT / "seed" / "models" / "sources" / "models_dev.generated.yaml"
HF_ORACLE_GENERATED = REPO_ROOT / "seed" / "models" / "sources" / "hf_oracle.generated.yaml"
MODELS_DEV_CATALOG_GENERATED = REPO_ROOT / "seed" / "models" / "sources" / "models_dev_catalog.generated.yaml"

# Durable OFFLINE enrichment cache: the candidate subset of cfahlgren1/hub-stats
# (incl. the `baseModels` lineage column) frozen by scripts/freeze_hub_stats_cache.py.
# Reading it makes a regen REPRODUCIBLE without depending on flaky live HF. The
# default below points HUB_STATS_LOCAL_PARQUET at it unless overridden or --live
# is passed.
FROZEN_CACHE_PATH = REPO_ROOT / "curation" / "hub_stats_frozen.parquet"

# Every model source whose canonicals/aliases can be enriched from hub-stats.
# hf_oracle (the HF-present mints) and the models.dev catalog are included so a
# refresh backfills their params/release_date/open_weights/baseModels parents.
_MODEL_SOURCES = (
    CORE_PATH,
    MODELS_DEV_GENERATED,
    HF_ORACLE_GENERATED,
    MODELS_DEV_CATALOG_GENERATED,
)


def build_hf_to_dev() -> dict[str, str]:
    """HF-org-lowercase -> developer/community slug (see
    `eval_card_registry.lib.seed_io.build_hf_to_dev_from_orgs_yaml`). Reading the ALIAS tier lets a
    refresh honour a curated same-uploader merge (e.g. `EnnoAi` -> `Enno-Ai`)
    instead of re-emitting the community-twin spelling the org-fold merged."""
    return build_hf_to_dev_from_orgs_yaml(ORGS_PATH)


# ---------------------------------------------------------------------------
# Etag watermark — short-circuits the daily cron on no-op cycles.
# ---------------------------------------------------------------------------

def fetch_parquet_etag() -> Optional[str]:
    """HEAD the hub-stats parquet and return its content ETag (a stable
    SHA-256 of the file body). Returns None on any HTTP/network failure
    so the caller can fall back to the unconditional re-fetch path."""
    try:
        with httpx.Client(follow_redirects=True, timeout=30.0) as c:
            r = c.head(PARQUET_URL)
            r.raise_for_status()
            etag = r.headers.get("etag")
            return etag.strip('"') if etag else None
    except Exception as e:
        print(f"[refresh] WARN: parquet HEAD failed ({e}); skipping etag check", file=sys.stderr)
        return None


def load_state() -> dict:
    """Read the watermark state file. Returns an empty dict on missing,
    unreadable, or schema-invalid input — degrades to "no state known,
    re-check everything" so a corrupted/deleted file never makes the
    script behave worse than today's unconditional behaviour."""
    if not STATE_PATH.exists():
        return {}
    try:
        data = json.loads(STATE_PATH.read_text())
        if not isinstance(data, dict):
            return {}
        if not isinstance(data.get("rows_checked_at_etag"), list):
            return {}
        return data
    except (OSError, ValueError):
        return {}


def write_state(etag: str, rows_checked: list[str]) -> None:
    """Write the watermark. Sorted for diff-clean commits. Called LAST,
    after the YAML write succeeds — a crash before this line just means
    next cron run redoes the work, never that we mark rows checked
    without their data having landed in the YAML."""
    payload = {
        "parquet_etag": etag,
        "rows_checked_at_etag": sorted(set(rows_checked)),
    }
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Loading the existing registry's index — both org aliases and canonical
# model HF aliases. We use these to gate which hub-stats rows we look up.
# ---------------------------------------------------------------------------

def load_org_alias_to_canonical() -> dict[str, str]:
    """Map normalized HF org alias → our canonical org id.
    Example entry: `'meta-llama' → 'meta'`, `'qwen' → 'alibaba'`."""
    with open(ORGS_PATH) as f:
        orgs = yaml.safe_load(f) or []
    out: dict[str, str] = {}
    for o in orgs:
        cid = o.get("id")
        if not cid:
            continue
        out[_normalize(cid)] = cid
        if o.get("hf_org"):
            out[_normalize(o["hf_org"])] = cid
        for a in (o.get("aliases") or []):
            if isinstance(a, str):
                out[_normalize(a)] = cid
    return out


def load_existing_canonical_aliases() -> dict[str, str]:
    """Walk core.yaml + models_dev.generated.yaml and map every
    HF-shaped surface form (any string containing `/`) to the canonical
    id it belongs to. Keyed by normalized form so case/separator drift
    doesn't matter at lookup time."""
    out: dict[str, str] = {}

    # PASS 1: register every canonical `id` first, across ALL sources. A
    # canonical's own id is a STRONGER claim than being listed as another
    # canonical's alias — so when the same normalized form is both (a known
    # cross-source inconsistency, e.g. `DeepSeek-V3.1-Terminus` is an alias of
    # the fuzzy-collapsed `deepseek-v3.1` AND the id of a distinct HF canonical),
    # the id-claim wins. This stops an enrichment from emitting an alias that the
    # seed validator would see as double-claimed.
    for path in _MODEL_SOURCES:
        for e in load_entries_from_yaml(path):
            cid = e.get("id")
            if cid:
                out.setdefault(_normalize(cid), cid)
    # PASS 2: aliases fill in only forms no canonical id already claimed.
    for path in _MODEL_SOURCES:
        for e in load_entries_from_yaml(path):
            cid = e.get("id")
            if not cid:
                continue
            for a in (e.get("aliases") or []):
                if isinstance(a, str) and "/" in a:
                    out.setdefault(_normalize(a), cid)
    return out


def load_ambiguous_canonicals() -> set[str]:
    """Canonical ids whose id/name/alias normalized form is ALSO claimed by a
    DIFFERENT canonical — a pre-existing near-duplicate pair (e.g.
    `alibaba/QwQ-32B` carries the core alias `qwen-qwq-32b`, which is the
    name of the separate canonical `alibaba/qwen-qwq-32b`). For such pairs the
    resolver's normalized-match tie-break is finely balanced, so merely adding
    an enrichment row can flip an oracle id. We skip enriching them (a handful)
    rather than risk a resolution regression; curation should de-dup the pair.
    Computed once over all sources."""
    norm_owners: dict[str, set[str]] = {}

    def _claim(form: str, cid: str) -> None:
        if isinstance(form, str) and form:
            norm_owners.setdefault(_normalize(form), set()).add(cid)

    for path in _MODEL_SOURCES:
        for e in load_entries_from_yaml(path):
            cid = e.get("id")
            if not isinstance(cid, str):
                continue
            _claim(cid, cid)
            if "/" in cid:
                _claim(cid.split("/", 1)[1], cid)
            for a in (e.get("aliases") or []):
                _claim(a, cid)
    ambiguous: set[str] = set()
    for owners in norm_owners.values():
        if len(owners) > 1:
            ambiguous.update(owners)
    return ambiguous


def load_canonical_id_norms() -> dict[str, str]:
    """Map -> the canonical id for every NORMALIZED claim a distinct canonical
    makes: its full id AND its bare model-name part. Used to reject an
    enrichment alias whose normalized form collides with a DIFFERENT
    canonical's id OR name — that double-claim both trips the seed validator
    AND lets the resolver's normalized-match intercept another canonical's raw
    id (the oracle-steal class, e.g. adding `Qwen/QwQ-32B` (norm `qwen-qwq-32b`)
    as an alias of `alibaba/QwQ-32B` would collide with the distinct
    `alibaba/qwen-qwq-32b` canonical and flip `qwen/qwq-32b`'s resolution)."""
    out: dict[str, str] = {}
    for path in _MODEL_SOURCES:
        for e in load_entries_from_yaml(path):
            cid = e.get("id")
            if not isinstance(cid, str):
                continue
            out.setdefault(_normalize(cid), cid)
            if "/" in cid:
                out.setdefault(_normalize(cid.split("/", 1)[1]), cid)
    return out


def query_hub_stats(
    con: duckdb.DuckDBPyConnection,
    candidate_hf_ids: set[str],
) -> list[dict]:
    """Fetch hub-stats rows whose `id` is in `candidate_hf_ids`. Single
    pass — bounded by the size of our HF-aliased canonical set."""
    if not candidate_hf_ids:
        return []
    quoted = ", ".join(f"'{i.replace(chr(39), chr(39)*2)}'" for i in candidate_hf_ids)
    source = resolve_parquet_source(PARQUET_URL)
    sql = f"""
        SELECT {QUERY_COLUMNS}
        FROM read_parquet('{source}')
        WHERE id IN ({quoted})
    """
    cursor = con.execute(sql)
    cols = [d[0] for d in cursor.description]
    rows = cursor.fetchall()
    return [dict(zip(cols, r)) for r in rows]


def build_entry(
    row: dict,
    org_alias_map: dict[str, str],
    aliases_to_canonical: dict[str, str],
    hf_to_dev: Optional[dict[str, str]] = None,
    canonical_id_norms: Optional[dict[str, str]] = None,
    ambiguous_canonicals: Optional[set[str]] = None,
) -> Optional[dict]:
    """Construct a seed entry from a hub-stats row. Returns None when
    the row's HF id isn't in our existing aliases — backfill only
    operates on canonicals we already cover.

    All hub-stats-derived data fields (release_date, params_billions,
    open_weights, tags, metadata, parents, lineage_origin_model_org_id) come
    from `enrich_draft_from_row` so this script and the live-lookup
    path stay byte-identical on extraction. The seed loader's
    `_merge_into` unions parents by id across sources — generated
    parents from hub-stats compose with any curated parents in
    core.yaml rather than overriding them.
    """
    hf_id = row["id"]
    # Derive the canonical id with HF casing preserved (two-tier org rule), so it
    # matches the SAME HF-cased ids the HF-oracle generator mints and live
    # auto-create use — never a lowercase duplicate. The gate below is on the
    # normalized (case-insensitive) form, so an HF-cased canonical that exists in
    # the seed still matches.
    if hf_to_dev is None:
        hf_to_dev = build_hf_to_dev()
    canonical_id, _ = hf_id_to_canonical_cased(hf_id, hf_to_dev)
    norm_canon = _normalize(canonical_id)
    if norm_canon not in aliases_to_canonical:
        return None
    # Use the registry's actual canonical id (may differ in dot/dash from
    # our slugify of the HF id — e.g., `meta/llama-3.1-8b` vs `meta/llama-3-1-8b`).
    canonical_id = aliases_to_canonical[norm_canon]

    # Ambiguous-pair guard: skip enriching a canonical that shares a normalized
    # id/name/alias form with a DIFFERENT canonical (a pre-existing
    # near-duplicate pair, e.g. `alibaba/QwQ-32B` carries the core alias
    # `qwen-qwq-32b` = the name of the separate `alibaba/qwen-qwq-32b`). For
    # such pairs the resolver's normalized-match tie-break is finely balanced;
    # merely adding an enrichment row perturbs it and can flip an oracle id.
    # Backfilling these few is not worth a resolution regression — curation
    # should de-duplicate the pair first.
    if ambiguous_canonicals and canonical_id in ambiguous_canonicals:
        return None

    # Candidate aliases = the HF repo id + its slug. ONLY keep a candidate when
    # it isn't already owned (by normalized form) by a DIFFERENT canonical — the
    # seed validator rejects an alias claimed by two canonicals. Now that the
    # candidate set spans every source (incl. the hf_oracle mints), a
    # snapshot HF id can be both an enrichment alias here AND its own separate
    # canonical (e.g. `…-Chat-3B-v1` is a distinct canonical, not an alias of
    # the fuzzy-collapsed `redpajama-incite-3b` base). Dropping the steal keeps
    # the seed valid without weakening resolution — the owning canonical already
    # resolves the form.
    cid_norms = canonical_id_norms or {}

    # Candidate aliases: ONLY the raw HF repo id (case-preserved). We do NOT
    # emit a lowercased `_slugify(hf_id)` form — the resolver's normalized-match
    # already collapses case/separators, and a fabricated lowercase slug tends
    # to collide with a DIFFERENT canonical's existing alias (e.g.
    # `berkeley-nest/starling-lm-7b-alpha` is already a core alias of
    # `nexusflow/starling-rm`), which the seed validator rejects as a double
    # claim. Drop the HF id when it equals the canonical (no-op) or when any
    # other canonical already owns that exact string OR its normalized form.
    def _alias_ok(a: str) -> bool:
        if a == canonical_id:
            return False
        n = _normalize(a)
        if aliases_to_canonical.get(n, canonical_id) != canonical_id:
            return False
        # Reject if the alias's full-normalized OR bare-name-normalized form is a
        # distinct canonical's id/name (would steal that canonical's resolution).
        if cid_norms.get(n, canonical_id) != canonical_id:
            return False
        if "/" in a:
            name_n = _normalize(a.split("/", 1)[1])
            if cid_norms.get(name_n, canonical_id) != canonical_id:
                return False
        return True

    aliases = sorted(a for a in {hf_id} if _alias_ok(a))

    # Pass the resolved registry canonical so the family-version
    # inference inside enrich_draft_from_row can suppress a self-edge
    # when the HF id is aliased directly to its family pointer (rather
    # than a separate snapshot canonical). Without target_canonical,
    # `Olmo-3-1125-32B` aliased to `allenai/olmo-3-32b` would gain a
    # parent edge to itself and corrupt the lineage walker.
    # The lightweight YAML map above does not include every curated alias from
    # the resolver's shared org table (notably ``CohereLabs`` and
    # ``ibm-granite``). Union in the complete HF-prefix map so this cron emits
    # the same canonical lineage-origin org ids as the models.dev reconcile
    # step; otherwise the two daily crons rewrite the same rows back and forth.
    lineage_org_map = {**org_alias_map, **hf_to_dev}
    enrichment = enrich_draft_from_row(
        row, aliases_to_canonical, lineage_org_map,
        target_canonical=canonical_id,
    )

    # `metadata.hf_id` claims "this canonical IS that real HF repo". When the
    # enrichment lands on a registry canonical under a DIFFERENT spelling
    # (org-slug mint, e.g. `deepseek/deepseek-vl2`) and the real id is not
    # itself a canonical anywhere, keeping the claim trips the
    # canonical-id-equals-hf-id gate (the canonical id would 404 on HF) and
    # stalls the cron. The HF surface form still resolves via the alias —
    # drop only the metadata claim.
    if "metadata" in enrichment and hf_id != canonical_id:
        if aliases_to_canonical.get(_normalize(hf_id)) != hf_id:
            try:
                meta = json.loads(enrichment["metadata"])
            except (ValueError, TypeError):
                meta = None
            if isinstance(meta, dict) and "hf_id" in meta:
                meta.pop("hf_id")
                enrichment["metadata"] = json.dumps(meta, sort_keys=True)

    # Decode tags from JSON-encoded string (helper output) back to a YAML
    # list. Loader accepts either form; list-form keeps generated YAML
    # diffs reviewable.
    if "tags" in enrichment:
        try:
            enrichment["tags"] = json.loads(enrichment["tags"])
        except (ValueError, TypeError):
            pass

    # Preserve the existing YAML key order so the diff after this change
    # is dominated by NEW fields (parents / lineage_origin_model_org_id) rather
    # than wholesale reformatting.
    # NO display_name here: this is a backfill enrichment that MERGES onto an
    # existing canonical (hf_oracle/core/models_dev), which already carries the
    # authoritative HF-cased display_name. Emitting a humanized, org-stripped
    # name (e.g. `QwenStock1 14B`) would (a) collide across orgs — the seed
    # auto-derives display_name as a global alias — and (b) needlessly churn the
    # diff. The loader's field-merge keeps the existing display_name.
    # NO org_id either: the owning source / seed-time org-from-prefix rule
    # derives it (and auto-creates the org row — which only fires when org_id
    # is None). Stamping the raw HF prefix here both overrode deliberately
    # org-less curated canonicals via the prefer-non-empty field merge and left
    # dangling org FKs for prefixes this cron never writes a row for.
    entry: dict = {
        "id": canonical_id,
    }
    if "release_date" in enrichment:
        entry["release_date"] = enrichment["release_date"]
    if "tags" in enrichment:
        entry["tags"] = enrichment["tags"]
    entry["aliases"] = [a for a in aliases if a != canonical_id]
    if "metadata" in enrichment:
        entry["metadata"] = enrichment["metadata"]
    entry["review_status"] = "reviewed"
    if "params_billions" in enrichment:
        entry["params_billions"] = enrichment["params_billions"]
    if "open_weights" in enrichment:
        entry["open_weights"] = enrichment["open_weights"]
    if "parents" in enrichment:
        entry["parents"] = json.loads(enrichment["parents"])
    if "lineage_origin_model_org_id" in enrichment:
        entry["lineage_origin_model_org_id"] = enrichment["lineage_origin_model_org_id"]
    return entry


def _carry_forward_committed(fresh: list[dict], committed: list[dict]) -> list[dict]:
    """Merge the COMMITTED generated file into a fresh wholesale rewrite so a
    model that fell out of today's fetch (upstream parquet shrank, repo gated/
    deleted) never silently loses its committed enrichment at the next seed.
    Mirrors refresh_from_modelsdev's carry-forward:

      * id still in the fresh output: the fresh entry wins outright (and a
        prior `upstream_status: removed` tag is thereby cleared);
      * id absent: retain the committed entry, tagged
        `metadata.upstream_status: removed`. The committed `org_id` field
        (pre-dating the no-stamping rule) is dropped — the seed-time
        org-from-prefix rule recovers the org.

    Malformed committed metadata degrades to {} (with a warning) rather than
    crashing the cron. Output is id-sorted (deterministic)."""
    fresh_ids = {e["id"] for e in fresh if e.get("id")}
    out = list(fresh)
    retained = 0
    for c in committed:
        cid = c.get("id")
        if not cid or cid in fresh_ids:
            continue
        rc = {k: v for k, v in c.items() if k != "org_id"}
        try:
            meta = json.loads(rc.get("metadata") or "{}")
            if not isinstance(meta, dict):
                meta = {}
        except (ValueError, TypeError):
            print(f"[refresh] WARN: malformed committed metadata on {cid}; treating as empty", file=sys.stderr)
            meta = {}
        meta["upstream_status"] = "removed"
        rc["metadata"] = json.dumps(meta, sort_keys=True)
        out.append(rc)
        retained += 1
    if retained:
        print(
            f"[refresh] carry-forward: retained {retained} committed entr(ies) "
            f"absent from today's fetch",
            file=sys.stderr,
        )
    return sorted(out, key=lambda e: e["id"])


# ---------------------------------------------------------------------------
# Output writing.
# ---------------------------------------------------------------------------

_HEADER = """# Generated from cfahlgren1/hub-stats — DO NOT EDIT BY HAND.
# To update: run `python scripts/refresh_from_hub_stats.py`.
#
# Source: https://huggingface.co/datasets/cfahlgren1/hub-stats (Apache-2.0)
# Last refresh date is in git history.
#
# Backfill-only: every entry here MERGES into an existing canonical_models
# row at seed time. Fills in release_date, params_billions, license, tags,
# etc. via the seed loader's field-level merge (curated values in
# core.yaml take precedence on conflict).
#
# Entries tagged `metadata.upstream_status: removed` are carry-forwards:
# their id fell out of the upstream fetch, so the committed enrichment is
# retained rather than deleted. A reappearing id replaces the carried entry.
#
# Lineage descendant pre-load (community quants/finetunes of our covered
# models) is not done here. EEE drafts get on-demand enrichment via the
# live hub-stats lookup at draft creation.
"""


def write_models_yaml(entries: list[dict]) -> None:
    MODELS_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(entries, sort_keys=False, allow_unicode=True, width=200)
    MODELS_OUT_PATH.write_text(_HEADER + "\n" + body)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="print summary; don't write files")
    p.add_argument("--force", action="store_true", help="ignore the etag watermark; re-fetch unconditionally")
    p.add_argument(
        "--live",
        action="store_true",
        help="query live cfahlgren1/hub-stats directly instead of the committed "
        "offline cache (curation/hub_stats_frozen.parquet). Use to refresh; the "
        "default is the reproducible offline cache. Refresh the cache itself with "
        "scripts/freeze_hub_stats_cache.py.",
    )
    args = p.parse_args()

    # Reproducible-by-default: read the committed offline cache unless the caller
    # passed --live or set HUB_STATS_LOCAL_PARQUET explicitly. This decouples
    # "fetch from HF" (the freeze tool / --live) from "derive enrichment" (this
    # script), so a clean regen reproduces lineage without flaky live HF.
    if (
        not args.live
        and not os.environ.get(HUB_STATS_LOCAL_PARQUET_ENV)
        and FROZEN_CACHE_PATH.exists()
    ):
        os.environ[HUB_STATS_LOCAL_PARQUET_ENV] = str(FROZEN_CACHE_PATH)
        print(
            f"[refresh] reading frozen offline cache {FROZEN_CACHE_PATH} "
            f"(pass --live to query HF directly)",
            file=sys.stderr,
        )

    org_alias_map = load_org_alias_to_canonical()
    aliases_to_canonical = load_existing_canonical_aliases()
    canonical_id_norms = load_canonical_id_norms()
    ambiguous_canonicals = load_ambiguous_canonicals()
    hf_to_dev = build_hf_to_dev()
    if not org_alias_map:
        print("[refresh] ERROR: seed/orgs.yaml is empty or missing.", file=sys.stderr)
        return 1
    if not aliases_to_canonical:
        print("[refresh] ERROR: no canonical models found in seed/. Seed first.", file=sys.stderr)
        return 1

    # Reverse map: developer slug -> the HF org spellings that remap to it
    # (`meta` -> {`meta-llama`, `facebook`}). Used to reconstruct the true HF
    # repo id for a big-dev canonical (`meta/Llama-3.1-8B`) whose id is NOT the
    # HF repo (`meta-llama/Llama-3.1-8B`) — so the parquet lookup finds the row.
    dev_to_hf_orgs: dict[str, set[str]] = {}
    for hf_org, dev in hf_to_dev.items():
        dev_to_hf_orgs.setdefault(dev, set()).add(hf_org)

    # Initial candidates: every HF-shaped alias + canonical id on a known
    # canonical, across ALL model sources (including the hf_oracle mints
    # and the models.dev catalog). We pass the original (non-normalized) form to DuckDB
    # since the parquet `id` column carries case-sensitive original strings. For
    # big-dev canonicals we also enqueue the reconstructed HF-org repo id.
    initial: set[str] = set()
    for path in _MODEL_SOURCES:
        for e in load_entries_from_yaml(path):
            for a in (e.get("aliases") or []):
                if isinstance(a, str) and "/" in a:
                    initial.add(a)
            cid = e.get("id")
            if isinstance(cid, str) and "/" in cid:
                initial.add(cid)
                org_part, name_part = cid.split("/", 1)
                for hf_org in dev_to_hf_orgs.get(org_part, ()):  # big-dev re-map
                    initial.add(f"{hf_org}/{name_part}")
    print(f"[refresh] HF-id candidates to look up: {len(initial)}", file=sys.stderr)

    # Etag short-circuit: if the parquet hasn't republished AND we've
    # already checked every current candidate against this etag AND the
    # YAML is still on disk, exit without doing any DuckDB work. Etag
    # fetch failure → fall through to unconditional re-fetch (degrades
    # to pre-watermark behaviour). The YAML-existence guard catches the
    # case where someone deleted the YAML but the state file survived;
    # without it we'd silently leave hub-stats enrichment missing from
    # the seed merge.
    # The etag watermark HEADs the live parquet URL — irrelevant (and a
    # needless network call) when reading a LOCAL parquet. In offline mode we
    # always re-run the local read; it's cheap and has no rate limit.
    offline = is_local_parquet()
    current_etag = None if offline else fetch_parquet_etag()
    state = load_state()
    can_short_circuit = (
        not offline
        and not args.force
        and not args.dry_run
        and current_etag is not None
        and state.get("parquet_etag") == current_etag
        and MODELS_OUT_PATH.exists()
        and initial.issubset(set(state.get("rows_checked_at_etag", [])))
    )
    if can_short_circuit:
        print(
            f"[refresh] no-op: parquet etag unchanged ({current_etag[:12]}...) "
            f"and all {len(initial)} candidates already checked",
            file=sys.stderr,
        )
        return 0

    con = duckdb.connect()
    if offline:
        print(
            f"[refresh] OFFLINE: reading local parquet {resolve_parquet_source(PARQUET_URL)}",
            file=sys.stderr,
        )
    else:
        con.execute("INSTALL httpfs; LOAD httpfs;")
        # Authenticate parquet fetches when HF_TOKEN is in the environment.
        # Unauth limit is 500 requests / 5min; one DuckDB read_parquet streams
        # the file via many range requests and can brush that ceiling on its
        # own. With auth the limit is ~30k / 5min.
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            escaped = hf_token.replace("'", "''")
            con.execute(
                f"CREATE SECRET hf_auth (TYPE HTTP, BEARER_TOKEN '{escaped}', "
                f"SCOPE 'https://huggingface.co');"
            )
    rows = query_hub_stats(con, initial)
    print(f"[refresh] hub-stats rows fetched: {len(rows)}", file=sys.stderr)

    entries: list[dict] = []
    for row in rows:
        e = build_entry(
            row, org_alias_map, aliases_to_canonical, hf_to_dev,
            canonical_id_norms, ambiguous_canonicals,
        )
        if e is not None:
            entries.append(e)
    entries.sort(key=lambda e: e["id"])

    # Dedupe by canonical id: when multiple HF ids collapse to the same
    # canonical via aliases_to_canonical (e.g. -v0.1/-v0.2/-v0.3 all
    # mapping to mistralai/mistral-7b-instruct because the resolver's
    # fuzzy strip removes the version suffix), pick the entry with the
    # latest release_date as the winner and UNION the alias lists across
    # all dupes so every HF id stays addressable.
    by_id: dict[str, dict] = {}
    collisions = 0
    for e in entries:
        eid = e["id"]
        prev = by_id.get(eid)
        if prev is None:
            by_id[eid] = e
            continue
        collisions += 1
        # Pick winner by max release_date (None sorts before any date).
        prev_date = prev.get("release_date") or ""
        new_date = e.get("release_date") or ""
        winner = e if new_date > prev_date else prev
        loser = prev if winner is e else e
        merged_aliases = sorted(set(winner.get("aliases", [])) | set(loser.get("aliases", [])))
        winner["aliases"] = merged_aliases
        by_id[eid] = winner
    if collisions:
        print(
            f"[refresh] deduped {collisions} collision(s) by canonical id "
            f"(latest release_date wins; aliases unioned)",
            file=sys.stderr,
        )
    entries = sorted(by_id.values(), key=lambda e: e["id"])

    # Carry the committed file forward: an id that fell out of today's fetch
    # keeps its committed enrichment (tagged `upstream_status: removed`)
    # instead of losing release_date/params/metadata at the next seed.
    committed = (
        yaml.safe_load(MODELS_OUT_PATH.read_text()) or []
        if MODELS_OUT_PATH.exists() else []
    )
    entries = _carry_forward_committed(entries, committed)
    print(f"[refresh] enrichment entries: {len(entries)}", file=sys.stderr)

    if args.dry_run:
        print("[refresh] [dry-run] no files written")
        if entries:
            print("\nFirst 5 entries:")
            for e in entries[:5]:
                print(f"  {e['id']}  release_date={e.get('release_date')}  params={e.get('params_billions')}")
        return 0

    write_models_yaml(entries)
    print(f"[refresh] wrote {MODELS_OUT_PATH}", file=sys.stderr)

    # State write LAST: if anything above crashed, next run sees old
    # state and redoes the work — safe. Skip when etag fetch failed so
    # we don't pin a bogus watermark; next run will re-attempt cleanly.
    if current_etag is not None:
        write_state(current_etag, sorted(initial))
        print(f"[refresh] wrote {STATE_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
