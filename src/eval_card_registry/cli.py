"""
eval-card-registry CLI.

Commands:
  seed      Load known entities from seed/ YAML files
  stats     Print registry summary
  sync      Batch sync one or all EEE configs → eval_results table
"""
import json
import re
import unicodedata
from pathlib import Path
from typing import Optional

import typer
import yaml


def seed_collision_key(s: str) -> str:
    """The key under which two seed surface forms count as one entity:
    NFKD + combining-mark strip (so "racé" and "race" collide before the
    alphanumeric strip would silently drop the accent), casefold, and drop
    everything but [a-z0-9]. Used by the benchmark collision guard at seed
    time and by tests/test_seed_unique_ids.py for every flat seed file.
    Residual: cross-script confusables (Cyrillic 'а') still evade."""
    decomposed = unicodedata.normalize("NFKD", str(s))
    base = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]", "", base.casefold())


def _json_encode_if_needed(value):
    """Encode lists/dicts as JSON strings; pass through anything else.

    seed/models.yaml uses YAML-native lists for `tags` (e.g. `["open-weight"]`)
    while seed/benchmarks.yaml stores them pre-encoded as strings (e.g.
    `'["instruction-following"]'`). The canonical_* parquet columns are all
    VARCHAR, so we coerce on the way in to keep both formats supported.
    """
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return value


def _legacy_parent_model_id_to_parents(entry: dict) -> None:
    """Translate a legacy `parent_model_id: X` field to the typed `parents`
    list shape. Mutates the entry in place.

    Legacy core.yaml / sources/*.generated.yaml use a single scalar
    `parent_model_id` to express a family/variant relationship (e.g.
    Llama-3-8B → Llama-3). The new schema replaces this with a typed list
    of parent edges. This shim converts on load so existing YAML keeps
    working until each file is migrated to emit `parents` natively.

    No-op when `parents` is already present (new shape wins) or when neither
    field is set.
    """
    if "parents" in entry and entry["parents"] is not None:
        entry.pop("parent_model_id", None)
        return
    legacy = entry.pop("parent_model_id", None)
    if legacy:
        entry["parents"] = [{"id": legacy, "relationship": "variant", "axis": "size"}]

from eval_card_registry.store.hf_store import get_store
from eval_card_registry.store import queries, schemas
from eval_card_registry.lib import collision_fold, org_attribution
from eval_card_registry.lib.seed_io import WEAK_SCALAR_FIELDS, build_hf_to_dev_from_orgs_yaml
from eval_card_registry.store.queries import _derive_release_date_from_id, _is_na
from eval_entity_resolver.normalization import normalize as _normalize_alias
from eval_entity_resolver.display import humanize_model_slug


def _humanized_display(entry: dict) -> str:
    """Presentation display_name for a model, humanized at load time.

    This shapes ONLY the stored display column — NOT alias promotion — so display
    coverage stays fully DECOUPLED from resolution: the seed loop promotes the
    ORIGINAL display_name (pre-humanize) as an alias exactly as before, leaving
    every raw's resolved canonical unchanged.

    Rule: a display is a label if it contains a SPACE; otherwise it is a slug.
    - Empty display: humanize the id.
    - Slug display (no spaces — a raw id, a bare leaf like `miqu-1-70b-sf`, an
      org-qualified id like `meta-llama/Llama-3.1-8B-Instruct`, or a tier-3 raw
      `EVA-UNIT-01/eva-qwen2-5-32b-v0-2`): humanize it. In a slug every hyphen is
      a separator, so this is safe.
    - Label display (has spaces — curated/generator human names like
      `Jamba Large 1.6` or `Lingma Agent + Lingma SWE-GPT 72b (v0918)`): keep
      as-is. Humanizing here would mangle proper-noun hyphens (`SWE-GPT` ->
      `SWE GPT`).

    A leading `unknown/` placeholder-org prefix is dropped from the rendered
    label (`unknown` is the "real org not known" sentinel, never a display org).

    KNOWN LIMITATION: `humanize_model_slug` is a heuristic, so a handful of
    slug displays with intentional casing lose it (`QwQ-32B` -> `QWQ 32B`,
    `...-128E-Instruct` -> `...128e...`). These stay readable; fixing them means
    extending the shared humanizer, which also ripples into generator-output
    confluence, so it is deferred.
    """
    cid = str(entry.get("id") or "")
    dn = str(entry.get("display_name") or "").strip()
    name = cid.split("/", 1)[1] if "/" in cid else cid  # org-stripped id
    if not dn or dn == cid:
        # No real label — derive from the id. An org-stripped name that already
        # reads as a label (has spaces, e.g. `Amazon Q Developer Agent (...)`)
        # is used verbatim; a slug name is humanized.
        out = name if " " in name else humanize_model_slug(cid)
    elif " " not in dn:
        out = humanize_model_slug(dn)
    else:
        out = dn
    # Drop a leading `unknown/` placeholder-org prefix. The slug/id branches
    # already strip it via the org split; this catches the space-containing
    # curated labels (e.g. `unknown/Lingma Agent + ...`).
    if out.startswith("unknown/"):
        out = out[len("unknown/"):]
    return out

app = typer.Typer(help="eval-card-registry CLI")


def _load_store():
    store = get_store()
    if not store.loaded:
        store.load()
    return store


# ------------------------------------------------------------------
# seed
# ------------------------------------------------------------------

@app.command()
def seed(
    local: bool = typer.Option(False, "--local", help="Write to fixtures/ instead of HF Hub"),
    seed_dir: str = typer.Option("./seed", "--seed-dir"),
    prune_stale: bool = typer.Option(
        False,
        "--prune-stale/--no-prune-stale",
        help="Remove reviewed seed entities and seed aliases absent from the current YAML snapshot.",
    ),
):
    """Load known canonical entities from seed YAML files."""
    import os
    if local:
        os.environ["LOCAL_MODE"] = "true"

    store = _load_store()
    seed_path = Path(seed_dir)

    # ------------------------------------------------------------------
    # Models — three-layer load from seed/models/:
    #   sources/*.generated.yaml  → external catalog data (e.g. models.dev),
    #                               flat lists, never hand-edited
    #   core.yaml                 → curated canonicals (the source of truth),
    #                               flat list OR {skip_ids, entries} dict
    #   enrichments/aliases.yaml  → optional alias-only entries ({id, aliases})
    #                               that union onto whatever exists
    #
    # Merge order: sources → core → enrichments. Field-level merge per entry
    # (aliases / tags UNION; other scalars prefer non-empty, last-write-wins).
    # `skip_ids` from core drops generated entries we don't want.
    # Source enrich records may carry a `weak:` scalar map (donated from a
    # suppressed/folded mint); weak values are applied LAST, filling only
    # fields every strong merge left empty — see the weak-fill block below.
    # ------------------------------------------------------------------
    def _load_models_merged() -> list[dict]:
        models_dir = seed_path / "models"
        sources_dir = models_dir / "sources"
        core_file = models_dir / "core.yaml"
        enrichments_dir = models_dir / "enrichments"

        source_entries: list[dict] = []
        core_entries: list[dict] = []
        enrichment_entries: list[dict] = []
        skip_ids: set[str] = set()

        def _is_empty(v) -> bool:
            if v is None:
                return True
            if isinstance(v, (list, dict)) and len(v) == 0:
                return True
            if isinstance(v, str) and v.strip() in ("", "[]", "{}"):
                return True
            return False

        # WEAK scalar contributions: a generated enrich record may carry a
        # `weak: {field: value}` map (scalars donated from a suppressed/folded
        # mint by scripts/refresh_from_modelsdev.py). Pulled out at load so
        # they never enter the strong field merge; applied after ALL strong
        # merges (see the weak-fill block below). Collected in deterministic
        # order: source files in sorted-filename order, records in file order.
        weak_candidates: list[tuple[str, str, object]] = []
        if sources_dir.is_dir():
            for src_path in sorted(sources_dir.glob("*.generated.yaml")):
                with open(src_path) as f:
                    loaded = yaml.safe_load(f) or []
                if not isinstance(loaded, list):
                    raise typer.BadParameter(f"{src_path} must be a flat list")
                for entry in loaded:
                    if not isinstance(entry, dict) or not entry.get("id"):
                        continue
                    weak = entry.pop("weak", None)
                    if isinstance(weak, dict):
                        for field in WEAK_SCALAR_FIELDS:
                            if not _is_empty(weak.get(field)):
                                weak_candidates.append((entry["id"], field, weak[field]))
                source_entries.extend(loaded)

        skip_source_ids: set[str] = set()
        if core_file.exists():
            with open(core_file) as f:
                loaded = yaml.safe_load(f) or {}
            if isinstance(loaded, list):
                core_entries = loaded
            elif isinstance(loaded, dict):
                core_entries = loaded.get("entries", []) or []
                skip_ids = set(loaded.get("skip_ids", []) or [])
                # `skip_source_ids` drops these ids from sources/enrichments only,
                # leaving core entries authoritative. Used when models.dev (or any
                # auto-generated source) ships bad aliases for a model that core.yaml
                # curates correctly — otherwise the loader's UNION-merge would
                # re-introduce the bad aliases on every refresh.
                skip_source_ids = set(loaded.get("skip_source_ids", []) or [])
            else:
                raise typer.BadParameter(f"{core_file} unexpected shape {type(loaded)}")

        # Keys EXPLICITLY present on each core entry — even with a null value.
        # An explicit core null (e.g. `open_weights: null` where the upstream
        # catalog's value is known-wrong) is a curated "unknown" and blocks
        # weak fill on that field; weak fills only keys core leaves ABSENT.
        core_explicit: dict[str, set[str]] = {}
        for entry in core_entries:
            if isinstance(entry, dict) and entry.get("id"):
                core_explicit.setdefault(entry["id"], set()).update(entry.keys())

        # Enrichment overlays: every enrichments/*.yaml (flat list of {id, ...}),
        # field-merged onto the matching canonical (aliases + parents UNION; see
        # _merge_into). Separate files keep concerns apart — aliases.yaml carries
        # alias bridges, parents.yaml the curated typed-edge graph (lineage edges
        # the bulk generators can't reconstruct from their source data).
        if enrichments_dir.is_dir():
            for enr_path in sorted(enrichments_dir.glob("*.yaml")):
                with open(enr_path) as f:
                    loaded = yaml.safe_load(f) or []
                if not isinstance(loaded, list):
                    raise typer.BadParameter(f"{enr_path} must be a flat list")
                enrichment_entries.extend(loaded)

        def _merge_into(target: dict, src: dict) -> dict:
            """Merge two entries with the same canonical_id.

            Field-level merge policy:
            - `aliases`: UNION (case-insensitive dedup).
            - `tags`: UNION (case-insensitive dedup). Both YAML-list and
              JSON-encoded-string forms supported. Protects against session
              additions overwriting `[open-weight, moe]` with `[open-weight]`.
            - `metadata`: per-KEY merge of the two JSON objects (later source
              wins per key, no key ever dropped). Protects against e.g.
              models_dev_catalog's `{alias_platforms}` wiping hub_stats'
              `{hf_id, downloads_all_time, ...}`. Falls back to the scalar
              rule when either side isn't a JSON object.
            - Other scalars: prefer non-empty across the pair; when both
              sides have a non-empty value, last-write-wins. Protects against
              session-batch entries that omit `architecture` /
              `params_billions` from silently overwriting earlier rich entries.

            "Empty" means: None, "", [], {}, or default-looking '{}' / '[]'.
            """
            import json as _json

            existing_aliases = list(target.get("aliases") or [])
            existing_lc = {a.lower() for a in existing_aliases if a}
            new_aliases = list(src.get("aliases") or [])
            for a in new_aliases:
                if a and a.lower() not in existing_lc:
                    existing_aliases.append(a)
                    existing_lc.add(a.lower())

            def _decode_list_field(v):
                """tags / metadata may be either YAML-list or JSON-encoded
                string. Return a list (best-effort) and a boolean indicating
                whether to re-encode on write."""
                if v is None:
                    return [], False
                if isinstance(v, list):
                    return list(v), False
                if isinstance(v, str):
                    s = v.strip()
                    if not s or s in ("[]", "null"):
                        return [], True
                    try:
                        d = _json.loads(s)
                        if isinstance(d, list):
                            return list(d), True
                    except (ValueError, TypeError):
                        pass
                return [v], False

            # Union tags (handles both list and JSON-string formats)
            tgt_tags, tgt_was_json = _decode_list_field(target.get("tags"))
            src_tags, src_was_json = _decode_list_field(src.get("tags"))
            seen_tags_lc = {str(t).lower() for t in tgt_tags}
            for t in src_tags:
                if t is not None and str(t).lower() not in seen_tags_lc:
                    tgt_tags.append(t)
                    seen_tags_lc.add(str(t).lower())
            # Re-encode if either source was a JSON string (the parquet column
            # is VARCHAR; _json_encode_if_needed downstream handles either).
            tags_merged = _json.dumps(tgt_tags) if (tgt_was_json or src_was_json) else tgt_tags

            # Union `parents` by id. For an edge present in both, field-merge
            # within the edge so a later source can fill in `axis` (or correct
            # `relationship`) without duplicating the edge. Edges from the
            # target preserve their order; new edges from src are appended.
            tgt_parents, tgt_p_was_json = _decode_list_field(target.get("parents"))
            src_parents, src_p_was_json = _decode_list_field(src.get("parents"))
            parents_by_id: dict[str, dict] = {}
            parents_order: list[str] = []
            for p in tgt_parents:
                if isinstance(p, dict) and p.get("id"):
                    pid = p["id"]
                    if pid not in parents_by_id:
                        parents_order.append(pid)
                        parents_by_id[pid] = dict(p)
            for p in src_parents:
                if not isinstance(p, dict) or not p.get("id"):
                    continue
                pid = p["id"]
                if pid in parents_by_id:
                    merged_edge = dict(parents_by_id[pid])
                    for k, v in p.items():
                        if _is_empty(v):
                            continue
                        merged_edge[k] = v
                    parents_by_id[pid] = merged_edge
                else:
                    parents_order.append(pid)
                    parents_by_id[pid] = dict(p)
            parents_list = [parents_by_id[pid] for pid in parents_order]
            parents_merged = (
                _json.dumps(parents_list)
                if (tgt_p_was_json or src_p_was_json)
                else parents_list
            )

            # Per-key metadata merge. Both sides may be a dict or a
            # JSON-encoded string; malformed/non-dict input opts that pair out
            # (handled by the scalar last-non-empty-wins loop below instead).
            def _decode_dict_field(v):
                """Return (dict, was_json, is_dict)."""
                if v is None:
                    return {}, False, True
                if isinstance(v, dict):
                    return dict(v), False, True
                if isinstance(v, str):
                    s = v.strip()
                    if not s or s in ("{}", "null"):
                        return {}, True, True
                    try:
                        d = _json.loads(s)
                        if isinstance(d, dict):
                            return d, True, True
                    except (ValueError, TypeError):
                        pass
                return {}, False, False

            tgt_meta, tgt_m_json, tgt_m_ok = _decode_dict_field(target.get("metadata"))
            src_meta, src_m_json, src_m_ok = _decode_dict_field(src.get("metadata"))
            metadata_handled = tgt_m_ok and src_m_ok
            if metadata_handled:
                meta_merged = {**tgt_meta, **src_meta}  # later source wins per key
                metadata_merged = (
                    _json.dumps(meta_merged, sort_keys=True)
                    if (tgt_m_json or src_m_json)
                    else meta_merged
                )

            merged = dict(target)
            for k, v in src.items():
                if k in ("aliases", "tags", "parents"):
                    continue  # handled separately
                if k == "metadata" and metadata_handled:
                    continue  # handled separately
                if _is_empty(v):
                    continue
                merged[k] = v
            if metadata_handled and ("metadata" in target or "metadata" in src):
                merged["metadata"] = metadata_merged
            merged["aliases"] = existing_aliases
            merged["tags"] = tags_merged
            # Only emit `parents` if at least one side had any (avoids creating
            # a spurious empty list on entries that never had a parents field).
            if tgt_parents or src_parents:
                merged["parents"] = parents_merged
            return merged

        by_id: dict[str, dict] = {}

        def _absorb(entries: list[dict], extra_skip: set[str] = frozenset()) -> None:
            drop = skip_ids | extra_skip
            for e in entries:
                if "id" not in e:
                    raise typer.BadParameter(f"models seed entry missing id: {e!r}")
                if e["id"] in drop:
                    continue
                # Translate legacy `parent_model_id` scalar to the typed
                # `parents` list before any merge / column-filter step.
                _legacy_parent_model_id_to_parents(e)
                if e["id"] in by_id:
                    by_id[e["id"]] = _merge_into(by_id[e["id"]], e)
                else:
                    by_id[e["id"]] = e

        # Sources/enrichments respect both skip_ids and skip_source_ids;
        # core entries respect only skip_ids so curated overrides always apply.
        _absorb(source_entries, extra_skip=skip_source_ids)
        _absorb(core_entries)
        _absorb(enrichment_entries, extra_skip=skip_source_ids)
        merged = list(by_id.values())

        # Attribute the developer org to malformed-org draft ids (no '/', org
        # glued by -/.) so genuine models aren't orphaned from their developer
        # (cohere-march-2024 -> org cohere; nvidia.nemotron-* -> nvidia/...).
        # Runs BEFORE the fold so any normalised id can collapse with its twin.
        merged, _org_merges = org_attribution.attribute_orgs(
            merged, build_hf_to_dev_from_orgs_yaml(seed_path / "orgs.yaml")
        )

        # Fold normalize-collisions: the SAME model minted under different
        # separator spellings (gemini-1.5-pro vs gemini-1-5-pro, gpt-5.2 vs the
        # venice relabel gpt-52) collapses into ONE canonical so it surfaces as
        # one page. Guarded against false size merges (opt-1.3b != opt-13b); a
        # curated collision_overrides.yaml carries the do-not-fold list + winner
        # pins. (See lib/collision_fold.py.)
        ov_file = models_dir / "collision_overrides.yaml"
        never_fold, prefer, curated_merge, non_lineage_bases = [], {}, {}, set()
        if ov_file.is_file():
            ov = yaml.safe_load(ov_file.read_text()) or {}
            never_fold = ov.get("never_fold") or []
            prefer = ov.get("prefer") or {}
            # `merge`: curated {loser_id -> HF-true winner_id} for same-model
            # cross-namespace spellings the automatic fold can't key together
            # (different org prefix, or a `1-2b`-vs-`1.2B` size-guard false block).
            curated_merge = ov.get("merge") or {}
            # `non_lineage_bases`: underspecified umbrella ids that stay resolvable
            # but may never be a lineage PARENT (see collision_overrides.yaml).
            non_lineage_bases = set(ov.get("non_lineage_bases") or [])
        merged, _remap = collision_fold.fold_collisions(
            merged, never_fold, prefer,
            force_merge={**_org_merges, **curated_merge},
            non_lineage_bases=non_lineage_bases,
        )

        # WEAK FILL — LAST, after every strong merge INCLUDING the collision
        # fold (a folded-in full entry's scalars are strong too), so a value
        # from any full entry beats a weak one regardless of file load order.
        # A weak value lands only when the merged field is still empty AND the
        # key is not explicitly present on the core entry (an explicit core
        # null is a curated unknown — see core_explicit above). Two weak values
        # for the same (id, field): the FIRST contribution wins — source files
        # in sorted-filename order, records in file order (weak_candidates).
        weak_scalars: dict[tuple[str, str], object] = {}
        for eid, field, value in weak_candidates:
            if eid in skip_ids or eid in skip_source_ids:
                continue
            weak_scalars.setdefault((_remap.get(eid, eid), field), value)
        by_merged_id = {e["id"]: e for e in merged}
        for (eid, field), value in weak_scalars.items():
            target = by_merged_id.get(eid)
            if target is None or field in core_explicit.get(eid, ()):
                continue
            # A dated-snapshot id's parsed date outranks weak data: the
            # seed-time derive fallback (`release_date_derived_from_id`) would
            # fill it anyway, and the id stamp is the snapshot's own identity.
            if field == "release_date" and _derive_release_date_from_id(eid) is not None:
                continue
            if _is_empty(target.get(field)):
                target[field] = value
        return merged

    # ------------------------------------------------------------------
    # Benchmarks — two-source load:
    #   seed/benchmarks.yaml                 → curated canonicals (the
    #                                          source of truth, hand-edited)
    #   seed/benchmarks_generated/*.yaml     → bulk auto-generated entries
    #                                          (e.g. AIR-Bench 2024's 373
    #                                          categories, derived from
    #                                          scripts/data/air_bench_2024_raw_strings.txt)
    #
    # Merge order: generated → curated. Field-level merge per id (aliases
    # union; other scalars prefer non-empty, last-write-wins) so curated
    # entries can refine an auto-generated row without losing its aliases.
    # Generator scripts must use stable canonical_ids so refreshes are
    # idempotent.
    # ------------------------------------------------------------------
    def _load_benchmarks_merged() -> list[dict]:
        curated_path = seed_path / "benchmarks.yaml"
        generated_dir = seed_path / "benchmarks_generated"

        generated_entries: list[dict] = []
        if generated_dir.is_dir():
            for src_path in sorted(generated_dir.glob("*.yaml")):
                with open(src_path) as f:
                    loaded = yaml.safe_load(f) or []
                if not isinstance(loaded, list):
                    raise typer.BadParameter(f"{src_path} must be a flat list")
                generated_entries.extend(loaded)

        curated_entries: list[dict] = []
        if curated_path.exists():
            with open(curated_path) as f:
                loaded = yaml.safe_load(f) or []
            if not isinstance(loaded, list):
                raise typer.BadParameter(f"{curated_path} must be a flat list")
            curated_entries = loaded

        def _merge_benchmark(generated: dict, curated: dict) -> dict:
            """Curated wins on every field it specifies; aliases are
            unioned (case-insensitive dedup) so generator-emitted aliases
            survive even when curated narrows the entry."""
            merged = dict(generated)
            for k, v in curated.items():
                if k == "aliases":
                    continue
                merged[k] = v
            existing = list(generated.get("aliases") or [])
            existing_lc = {a.lower() for a in existing if a}
            for a in (curated.get("aliases") or []):
                if a and a.lower() not in existing_lc:
                    existing.append(a)
                    existing_lc.add(a.lower())
            merged["aliases"] = existing
            return merged

        by_id: dict[str, dict] = {}
        for entry in generated_entries:
            if "id" not in entry:
                raise typer.BadParameter(f"benchmarks generated entry missing id: {entry!r}")
            by_id[entry["id"]] = entry
        for entry in curated_entries:
            if "id" not in entry:
                raise typer.BadParameter(f"benchmarks seed entry missing id: {entry!r}")
            if entry["id"] in by_id:
                by_id[entry["id"]] = _merge_benchmark(by_id[entry["id"]], entry)
            else:
                by_id[entry["id"]] = entry
        # `preferred_metric` (seed key) -> `preferred_metric_id` (column),
        # validated against metrics.yaml so a typo can't silently null the
        # merged-view default.
        metrics_path = seed_path / "metrics.yaml"
        metric_ids: set[str] = set()
        if metrics_path.exists():
            with open(metrics_path) as f:
                metric_ids = {m["id"] for m in (yaml.safe_load(f) or [])}
        for entry in by_id.values():
            pm = entry.pop("preferred_metric", None)
            if pm is not None:
                if pm not in metric_ids:
                    raise typer.BadParameter(
                        f"benchmark {entry['id']!r}: preferred_metric {pm!r} "
                        f"is not a canonical metric id"
                    )
                entry["preferred_metric_id"] = pm

        _check_benchmark_collisions(list(by_id.values()))
        return list(by_id.values())

    def _check_benchmark_collisions(entries: list[dict]) -> None:
        """Seed-time guard: two benchmarks whose ids, display names, or
        global aliases collapse to the same normalized key (casefold +
        strip non-alphanumerics — stricter than collision_fold's
        separator-only key, so spaced llm-stats-mint aliases like
        "Vita Bench" are covered) are one entity split in two — fold them
        (specs/benchmark-relationships-audit.md). Flag-only, never
        auto-folds. Groups verified genuinely distinct go in
        seed/benchmarks_distinct_allowlist.yaml (a list of id groups).
        Residual: semantic dupes with different normalized keys
        (mmmu-val vs mmmu-validation) are invisible to this check.
        Also rejects '/' in benchmark ids — reserved by the merged-view
        URL scheme (single-segment merged eval ids stay collision-free
        with two-segment per-source ids only while benchmark ids are
        slash-free)."""
        _norm = seed_collision_key
        allow_path = seed_path / "benchmarks_distinct_allowlist.yaml"
        allowed: list[frozenset[str]] = []
        if allow_path.exists():
            with open(allow_path) as f:
                for group in yaml.safe_load(f) or []:
                    allowed.append(frozenset(group))

        # Typo guard: every allowlisted id must name a real benchmark,
        # else a stale/misspelled allowlist entry silently stops guarding.
        all_ids = {str(e["id"]) for e in entries}
        stale = sorted({bid for grp in allowed for bid in grp} - all_ids)
        if stale:
            raise typer.BadParameter(
                f"benchmarks_distinct_allowlist.yaml names unknown benchmark id(s): {stale}"
            )

        key_owners: dict[str, set[str]] = {}
        for e in entries:
            bid = str(e["id"])
            if "/" in bid:
                raise typer.BadParameter(
                    f"benchmark id {bid!r} contains '/' (reserved for merged-view routing)"
                )
            # id + display_name + aliases: all three become resolver
            # aliases at seed time (display_name is auto-promoted), so all
            # three can carry a dupe in.
            keys = {_norm(bid)}
            if e.get("display_name"):
                keys.add(_norm(e["display_name"]))
            keys |= {_norm(a) for a in (e.get("aliases") or []) if a}
            keys.discard("")
            for k in keys:
                key_owners.setdefault(k, set()).add(bid)
        conflicts = [
            sorted(ids)
            for ids in key_owners.values()
            if len(ids) > 1 and not any(ids <= grp for grp in allowed)
        ]
        if conflicts:
            listing = "; ".join(str(g) for g in sorted(conflicts))
            raise typer.BadParameter(
                f"{len(conflicts)} separator/case-collision group(s) among benchmark "
                f"ids+aliases — fold them in the seed or allowlist in "
                f"benchmarks_distinct_allowlist.yaml: {listing}"
            )

    # ------------------------------------------------------------------
    # Families — translate seed/families.yaml's nested {slug: {fields}}
    # shape into flat dicts ready for upsert. The YAML uses the slug as
    # the mapping key for human friendliness (`mmlu:` reads as a header);
    # the table needs `id` as a column.
    #
    # Output schema mirrors `canonical_families`: list-valued fields
    # (`benchmark_ids`, `folder_aliases`, `composite_keys`) are
    # JSON-encoded so they round-trip through the parquet StringDtype
    # column without losing structure.
    # ------------------------------------------------------------------
    def _load_families_seed() -> list[dict]:
        path = seed_path / "families.yaml"
        if not path.exists():
            return []
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise typer.BadParameter(f"{path} must be a top-level mapping {{slug: {{...}}}}")

        out: list[dict] = []
        # Validation: each benchmark may only appear in one curated family.
        seen_benchmarks: dict[str, str] = {}
        for slug, fields in raw.items():
            if not isinstance(fields, dict):
                raise typer.BadParameter(f"family {slug!r} entry must be a mapping, got {type(fields).__name__}")
            benchmark_ids = list(fields.get("benchmarks") or [])
            for bid in benchmark_ids:
                if bid in seen_benchmarks and seen_benchmarks[bid] != slug:
                    raise typer.BadParameter(
                        f"benchmark {bid!r} listed in two families: "
                        f"{seen_benchmarks[bid]!r} and {slug!r}"
                    )
                seen_benchmarks[bid] = slug
            entry = {
                "id": slug,
                "display_name": fields.get("display") or slug,
                "category": fields.get("category"),
                "benchmark_ids": benchmark_ids,
                "primary_benchmark_key": fields.get("primary_benchmark_key"),
                "folder_aliases": list(fields.get("folder_aliases") or []),
                "composite_keys": list(fields.get("composite_keys") or []),
                "tags": fields.get("tags") or [],
                "metadata": fields.get("metadata") or {},
                "review_status": fields.get("review_status") or "reviewed",
            }
            out.append(entry)
        return out

    # ------------------------------------------------------------------
    # Composites — same translation as families. YAML shape:
    #   {slug: {display, configs: [...], category?, family_id?}}
    # ------------------------------------------------------------------
    def _load_composites_seed() -> list[dict]:
        path = seed_path / "composites.yaml"
        if not path.exists():
            return []
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise typer.BadParameter(f"{path} must be a top-level mapping {{slug: {{...}}}}")

        out: list[dict] = []
        for slug, fields in raw.items():
            if not isinstance(fields, dict):
                raise typer.BadParameter(f"composite {slug!r} entry must be a mapping, got {type(fields).__name__}")
            raw_configs = fields.get("configs")
            if raw_configs is None:
                # Display-only override (no explicit `configs:`): implicit
                # single source_config equal to the slug. Some upstream
                # EEE folders are kebab (`arc-agi`), others snake
                # (`helm_classic`); ship both forms so the producer's
                # composite_config_map JOIN matches whichever the data
                # uses. De-dup when slug has no `-`.
                kebab = slug
                snake = slug.replace("-", "_")
                source_configs = [kebab] if kebab == snake else [kebab, snake]
            else:
                # Members are bare config strings or scoped mappings
                # {config, org[, source]} (composite-partition spec).
                # Scoped mappings pass through as objects — the mixed
                # list is JSON-encoded below by _json_encode_if_needed;
                # str() would ship Python-repr garbage.
                source_configs = []
                for c in raw_configs:
                    if isinstance(c, dict):
                        unknown = sorted(set(c) - {"config", "org", "source"})
                        if unknown or "config" not in c:
                            raise typer.BadParameter(
                                f"composite {slug!r}: scoped configs member "
                                f"{c!r} must have keys {{config, org[, source]}}"
                            )
                        if c.get("source") is not None and c.get("org") is None:
                            raise typer.BadParameter(
                                f"composite {slug!r}: scoped member {c!r} "
                                f"sets `source` without `org`"
                            )
                        member = {"config": str(c["config"])}
                        if c.get("org") is not None:
                            member["org"] = str(c["org"])
                        if c.get("source") is not None:
                            member["source"] = str(c["source"])
                        source_configs.append(member)
                    else:
                        source_configs.append(str(c))
            entry = {
                "id": slug,
                "display_name": fields.get("display") or slug,
                "category": fields.get("category"),
                "source_configs": source_configs,
                "family_id": fields.get("family_id"),
                "tags": fields.get("tags") or [],
                "metadata": fields.get("metadata") or {},
                "review_status": fields.get("review_status") or "reviewed",
            }
            out.append(entry)
        return out

    # ------------------------------------------------------------------
    # Orgs — two-file load:
    #   seed/orgs.yaml            → curated first-party labs (the source
    #                               of truth, hand-edited)
    #   seed/orgs.generated.yaml  → auto-created orgs from hub-stats refresh
    #                               (HF authors that aren't curated labs)
    #
    # Curated wins on id collision. Unlike the models merge (field-level),
    # orgs use a simple "drop generated entry if id is in curated" policy:
    # curated entries are deliberate and richer; auto-created entries are
    # thin (just id, display_name, kind=unknown), so a partial overlay
    # would never improve the curated record.
    # ------------------------------------------------------------------
    def _load_orgs_merged() -> list[dict]:
        curated_path = seed_path / "orgs.yaml"
        generated_path = seed_path / "orgs.generated.yaml"

        curated: list[dict] = []
        if curated_path.exists():
            with open(curated_path) as f:
                loaded = yaml.safe_load(f) or []
            if not isinstance(loaded, list):
                raise typer.BadParameter(f"{curated_path} must be a flat list")
            curated = loaded

        generated: list[dict] = []
        if generated_path.exists():
            with open(generated_path) as f:
                loaded = yaml.safe_load(f) or []
            if not isinstance(loaded, list):
                raise typer.BadParameter(f"{generated_path} must be a flat list")
            generated = loaded

        curated_ids = {e["id"] for e in curated if "id" in e}
        out = list(curated)
        for e in generated:
            if "id" not in e:
                raise typer.BadParameter(f"orgs.generated.yaml entry missing id: {e!r}")
            if e["id"] not in curated_ids:
                out.append(e)
        return out

    # table name, yaml file, label, entity_type (for alias creation)
    seed_specs = [
        # Orgs: load via merge helper to combine curated + auto-generated.
        ("canonical_orgs", "__merged_orgs__", "orgs", "org"),
        # Benchmarks: load via merge helper. Curated entries live in
        # seed/benchmarks.yaml; bulk-generated entries (e.g. AIR-Bench
        # 2024's 373 categories from the refresh script) live in
        # seed/benchmarks_generated/*.yaml. Sentinel path triggers the
        # _load_benchmarks_merged() helper.
        ("canonical_benchmarks", "__merged_benchmarks__", "benchmarks", "benchmark"),
        ("canonical_metrics", seed_path / "metrics.yaml", "metrics", "metric"),
        ("eval_harnesses", seed_path / "harnesses.yaml", "harnesses", "harness"),
        # Families & composites are first-class registry entities. Their
        # YAML uses {slug: {...}} shape, so we need translation loaders
        # rather than the flat-list path.
        # entity_type='family'/'composite' aliases are emitted for
        # consistency but aren't consulted by the resolver today.
        ("canonical_families", "__nested_families__", "families", "family"),
        ("canonical_composites", "__nested_composites__", "composites", "composite"),
        # Models: load via the merge helper; pass a sentinel path that
        # signals the loop below to invoke _load_models_merged() instead of
        # reading a single YAML file.
        ("canonical_models", "__merged_models__", "models", "model"),
    ]

    alias_count = 0
    # Track all seed entity IDs and alias keys so we can remove stale ones.
    # Alias key: (raw_value, entity_type, canonical_id, source_config)
    seed_snapshot: list[tuple[str, str, set[str], set[tuple[str, str, str, Optional[str]]]]] = []

    # Detect same-run alias collisions: two DIFFERENT canonicals declaring the
    # same (raw_value, entity_type, source_config) within ONE seed pass. The
    # add_alias/repoint path below silently last-write-wins on these, so the
    # owner is seed-order-dependent (nondeterministic) — a real data bug. We
    # record them here and raise after the loop so dirty seed data can't ship
    # silently. NOTE: a legitimate YAML *rename* (the persisted store had the
    # alias on a canonical that no longer claims it) is NOT a collision — only
    # same-pass double-claims are, which is why this is keyed on this run only.
    # Values/rows carry (owner, declarer) pairs: `owner` is the post-redirect
    # target the row will point at; `declarer` is the entity whose alias list
    # actually claimed the raw (what the operator must edit to resolve it).
    run_alias_owners: dict[tuple[str, str, Optional[str]], tuple[str, str]] = {}
    alias_collisions: list[tuple[str, str, Optional[str], str, str, str, str]] = []

    # Build the alias index once so add_alias collision checks are O(1) instead
    # of O(N) DataFrame mask scans. Combined with buffered=True below, this
    # avoids the O(N²) pd.concat-per-row cost on ~1k entities + ~13k aliases.
    queries._rebuild_alias_index(store)

    for table, yaml_file, label, entity_type in seed_specs:
        table_columns = set(schemas.empty(table).columns)
        if yaml_file == "__merged_models__":
            items = _load_models_merged()
            if not items:
                typer.echo(f"  [skip] no model entries found in seed/models.yaml or _overrides/")
                continue
        elif yaml_file == "__merged_orgs__":
            items = _load_orgs_merged()
            if not items:
                typer.echo(f"  [skip] no org entries found in seed/orgs.yaml or seed/orgs.generated.yaml")
                continue
        elif yaml_file == "__merged_benchmarks__":
            items = _load_benchmarks_merged()
            if not items:
                typer.echo(f"  [skip] no benchmark entries found in seed/benchmarks.yaml or seed/benchmarks_generated/")
                continue
        elif yaml_file == "__nested_families__":
            items = _load_families_seed()
            if not items:
                typer.echo(f"  [skip] no family entries found in seed/families.yaml")
                continue
        elif yaml_file == "__nested_composites__":
            items = _load_composites_seed()
            if not items:
                typer.echo(f"  [skip] no composite entries found in seed/composites.yaml")
                continue
        else:
            if not yaml_file.exists():
                typer.echo(f"  [skip] {yaml_file} not found")
                continue
            with open(yaml_file) as f:
                items = yaml.safe_load(f) or []

        yaml_ids: set[str] = set()
        yaml_alias_keys: set[tuple[str, str, str, Optional[str]]] = set()

        # Anti-shadowing (models): a community/non-lab model's display_name is a
        # noisy identity signal. When a fork's bare display_name (e.g.
        # "aya-expanse-8b") normalizes to the same form as a real LAB model
        # ("Aya Expanse 8B"), auto-seeding it as a global alias makes the fork
        # WIN the bare name at confidence 1.0 and shadow the lab. Build the set
        # of lab-owned normalized forms so we can drop only the COLLIDING
        # display_name aliases off non-lab models (non-colliding ones still seed,
        # so coverage is unaffected). org kind comes from canonical_orgs, which
        # seeds before models.
        lab_norms: set[str] = set()
        # Prefer-official: normalized bare LEAF -> owning canonical id, decided
        # by precedence instead of seed load order:
        #   1. lab_leaf_owner — a LAB model owns its leaf. A lab rarely seeds
        #      its own bare leaf as an alias (its forms are org-prefixed), so a
        #      non-lab FORK sharing the leaf (jebish7/Llama-3.1-8B-Instruct,
        #      whose display IS the bare "Llama-3.1-8B-Instruct") would
        #      otherwise win the un-namespaced name.
        #   2. strong_leaf_owner — else the leaf's UNIQUE confirmed entry
        #      (hf-sourced or reviewed) owns it: a machine-minted draft
        #      (models.dev / tier3 name inference, where org mis-attributions
        #      live) never beats a real repo (e.g. the models.dev draft
        #      `alibaba/dolphin-2-9.2-qwen2-72b` vs the real hf
        #      `dphn/dolphin-2.9.2-qwen2-72b`). Applied only when the alias's
        #      own entry is neither confirmed nor lab-owned — evidence can rank
        #      a draft below a real repo, but not two real repos against each
        #      other, and a lab draft (e.g. `microsoft/WizardLM-2-8x22B`, repo
        #      deleted upstream) keeps its name on developer-attribution
        #      grounds.
        # Leaves with 2+ confirmed non-lab claimants stay unowned: those are
        # duplicate re-uploads to deduplicate, not a name-preference call.
        lab_leaf_owner: dict[str, str] = {}
        lab_owner_dev: dict[str, str] = {}
        strong_leaf_owner: dict[str, str] = {}
        strong_ids: set[str] = set()
        org_kind: dict[str, str] = {}
        hf_to_dev: dict[str, str] = {}
        org_names: dict[str, str] = {}

        def _entry_dev(it: dict) -> str:
            # Resolved developer org id for a seed entry. A set org_id is
            # authoritative (curated remaps put the TRUE developer there —
            # e.g. the mis-prefixed draft `alibaba/dolphin-2-9.2-qwen2-72b`
            # carries org_id `cognitivecomputations`, and trusting its
            # `alibaba/` prefix would launder the mis-attribution into lab
            # status). Only when org_id is unset at load time (derived later,
            # e.g. dated snapshots) fall back to the id's org prefix, resolved
            # through the curated org map and then the merged-orgs name map —
            # the same surface forms `derive_model_lineage_fields` uses to
            # fill org_id, so the seed's lab decision and the seeded org_id
            # agree (a prefix-shaped org display_name like "Scale" resolves
            # identically in both).
            org_id = it.get("org_id")
            if org_id:
                return str(org_id)
            cid = str(it.get("id") or "")
            if "/" not in cid:
                return ""
            prefix = cid.split("/", 1)[0]
            return hf_to_dev.get(prefix.lower()) or org_names.get(prefix.lower()) or prefix

        def _entry_is_lab(it: dict) -> bool:
            return org_kind.get(_entry_dev(it)) == "lab"

        if entity_type == "model":
            # Read kinds from the YAML source, not store.table — orgs above were
            # upserted buffered=True and aren't flushed yet. Curated labs live in
            # orgs.yaml; any org absent here (or kind!=lab) is treated as non-lab.
            merged_orgs = [
                o for o in (_load_orgs_merged() or []) if isinstance(o, dict) and o.get("id")
            ]
            org_kind = {str(o["id"]): str(o.get("kind") or "") for o in merged_orgs}
            hf_to_dev = build_hf_to_dev_from_orgs_yaml(seed_path / "orgs.yaml")
            for o in merged_orgs:
                oid = str(o["id"])
                for name in [oid, o.get("hf_org"), o.get("display_name"), *(o.get("aliases") or [])]:
                    if isinstance(name, str) and name:
                        org_names.setdefault(name.lower(), oid)

            strong_claimants: dict[str, set[str]] = {}
            for it in items:
                if not isinstance(it, dict):
                    continue
                cid = str(it.get("id") or "")
                # "Confirmed" = a real repo (hf-sourced) or a reviewed entry —
                # never a tier3 name-inferred draft, which is where org
                # mis-attributions live (e.g. `meta/reflection-llama-3.1-70b`,
                # inferred, would otherwise steal `Reflection-Llama-3.1-70B`
                # from the real hf `mattshumer/Reflection-Llama-3.1-70B`).
                confirmed = "/" in cid and (
                    it.get("resolution_source") == "hf"
                    or it.get("review_status") == "reviewed"
                )
                if confirmed:
                    strong_ids.add(cid)
                    strong_claimants.setdefault(
                        _normalize_alias(cid.split("/", 1)[1]), set()
                    ).add(cid)
                if not _entry_is_lab(it):
                    continue
                # Only the aliases a lab actually SEEDS (display_name + explicit
                # aliases) can catch a raw via normalized match — so only drop a
                # community display that collides with one of THOSE. (name-part /
                # id are NOT bare aliases, so dropping on them would orphan the raw.)
                for s in [it.get("display_name"), *(it.get("aliases") or [])]:
                    if isinstance(s, str) and s:
                        lab_norms.add(_normalize_alias(s))
                if confirmed:
                    # First claimant wins on a shared lab leaf; the gate
                    # (test_lab_leaf_ownership_is_unambiguous) enforces that
                    # no two confirmed lab models actually share one.
                    _leaf_norm = _normalize_alias(cid.split("/", 1)[1])
                    if lab_leaf_owner.setdefault(_leaf_norm, cid) == cid:
                        lab_owner_dev[_leaf_norm] = _entry_dev(it)
            strong_leaf_owner = {
                norm: next(iter(cids))
                for norm, cids in strong_claimants.items()
                if len(cids) == 1
            }

        for original_item in items:
            item = dict(original_item)
            # Pop 'aliases' / 'scoped_aliases' before upserting — not table columns.
            extra_aliases = item.pop("aliases", []) or []
            scoped_aliases = item.pop("scoped_aliases", {}) or {}
            # Normalize list/dict columns: YAML may have native lists/dicts,
            # but the canonical_* parquet columns are VARCHAR, so encode if
            # needed. `parents` is a list-of-edges on canonical_models.
            # `benchmark_ids` / `folder_aliases` / `composite_keys` are
            # list-valued on canonical_families. `source_configs` is
            # list-valued on canonical_composites.
            for col in (
                "tags", "metadata", "parents",
                "input_modalities", "output_modalities",
                "benchmark_ids", "folder_aliases", "composite_keys",
                "source_configs",
            ):
                if col in item:
                    item[col] = _json_encode_if_needed(item[col])
            entity_item = {k: v for k, v in item.items() if k in table_columns}
            unknown_keys = sorted(set(item.keys()) - table_columns)
            if unknown_keys:
                typer.echo(
                    f"  [warn] {label} entry {item.get('id', '?')!r} has unknown "
                    f"key(s) {unknown_keys} — silently dropped. Check for typos."
                )
            if "id" not in entity_item:
                raise typer.BadParameter(f"{label} seed entry is missing required id: {original_item!r}")
            # Presentation-only: humanize the STORED display column for label-less
            # model rows (tier-3 raws, empty, id-placeholder). Alias promotion
            # below uses `display_name` = the ORIGINAL value, so resolution is
            # unchanged — display coverage is decoupled from resolution.
            display_name = entity_item.get("display_name", "")
            if entity_type == "model":
                humanized = _humanized_display(entity_item)
                if humanized:
                    entity_item["display_name"] = humanized
            queries.upsert_entity(store, table, entity_item, buffered=True)
            canonical_id = entity_item["id"]
            yaml_ids.add(canonical_id)

            # Global aliases (source_config=None): matched regardless of caller's source_config.
            # Scoped aliases (source_config=<name>): matched only when the caller passes that
            # source_config — lets short tokens ("Overall", "Arabic") map to different
            # benchmarks depending on which EEE config they came from.
            global_aliases = {canonical_id, display_name} | set(extra_aliases)
            # Drop a non-lab model's display_name alias when it would shadow a
            # lab model (see lab_norms above). canonical_id + explicit aliases
            # are kept — only the auto-promoted display_name is suppressed.
            # global_aliases is a set, so when display_name IS the canonical_id
            # (id-placeholder display) or an explicitly curated alias,
            # discarding it would destroy that stronger claim too (an entity
            # unresolvable by its own id) — never drop those strings.
            item_dev = _entry_dev(entity_item) if entity_type == "model" else ""
            item_is_lab = org_kind.get(item_dev) == "lab"
            if (
                entity_type == "model"
                and display_name
                and display_name != canonical_id
                and display_name not in extra_aliases
                and not item_is_lab
                and _normalize_alias(display_name) in lab_norms
            ):
                global_aliases.discard(display_name)

            alias_specs: list[tuple[str, Optional[str]]] = [
                (raw, None) for raw in global_aliases if raw
            ]
            for source_cfg, raw_values in scoped_aliases.items():
                for raw in raw_values or []:
                    if raw:
                        alias_specs.append((raw, source_cfg))

            for raw_value, source_cfg in alias_specs:
                # Prefer-official redirect: an un-namespaced leaf alias (a fork's
                # bare display "Llama-3.1-8B-Instruct", or a short/dated form) is
                # pointed at the leaf's owner (see the precedence above
                # lab_leaf_owner / strong_leaf_owner) so the official model wins
                # the bare name, not whichever entry sorts first. Global aliases
                # only (source_config=None). The lab tier takes names from
                # non-lab entries and from SAME-developer lab entries (a lab's
                # models.dev/casing mint folding onto its confirmed repo) but
                # never across distinct labs — a lab entry keeps its name on
                # developer-attribution grounds even against another lab's
                # colliding leaf. The confirmed tier only takes names FROM
                # unconfirmed non-lab drafts.
                target_cid = canonical_id
                if entity_type == "model" and source_cfg is None:
                    _norm = _normalize_alias(raw_value)
                    _lab = lab_leaf_owner.get(_norm)
                    if _lab and _lab != canonical_id and (
                        not item_is_lab or lab_owner_dev.get(_norm) == item_dev
                    ):
                        target_cid = _lab
                    elif not item_is_lab and canonical_id not in strong_ids:
                        _strong = strong_leaf_owner.get(_norm)
                        if _strong and _strong != canonical_id:
                            target_cid = _strong
                # Index stale-removal by (raw_value, entity_type, canonical_id, source_config)
                yaml_alias_keys.add((raw_value, entity_type, target_cid, source_cfg))
                # Same-run collision check (see run_alias_owners above): if a
                # different canonical already claimed this exact alias in this
                # pass, the owner would be nondeterministic — record it.
                _claim_key = (raw_value, entity_type, queries._source_config_key(source_cfg))
                _prev = run_alias_owners.get(_claim_key)
                if _prev is None:
                    run_alias_owners[_claim_key] = (target_cid, canonical_id)
                elif _prev[0] != target_cid:
                    alias_collisions.append(
                        (raw_value, entity_type, _claim_key[2],
                         _prev[0], target_cid, _prev[1], canonical_id)
                    )
                try:
                    queries.add_alias(store, {
                        "raw_value": raw_value,
                        "entity_type": entity_type,
                        "canonical_id": target_cid,
                        "source_config": source_cfg,
                        "source_field": "seed",
                        "status": "confirmed",
                        "strategy": "seed",
                        "confidence": 1.0,
                        "notes": None,
                    }, buffered=True)
                    alias_count += 1
                except ValueError:
                    # add_alias raises on uniqueness collision: an alias row
                    # already exists for (entity_type, raw_value, source_config).
                    # YAML is the source of truth, so if the existing row points
                    # at a different canonical_id, this is a YAML rename and we
                    # must REPOINT the existing row — NOT silently swallow it.
                    # Without this, stale-removal at the end of seed would then
                    # delete the row (its old key is no longer in
                    # yaml_alias_keys), causing total alias loss.
                    aliases_df = store.table("aliases")
                    mask = (
                        (aliases_df["raw_value"] == raw_value)
                        & (aliases_df["entity_type"] == entity_type)
                        & (aliases_df["status"] != "rejected")
                    )
                    if source_cfg is not None:
                        mask = mask & (aliases_df["source_config"] == source_cfg)
                    else:
                        mask = mask & aliases_df["source_config"].isna()
                    existing = aliases_df[mask]
                    if existing.empty:
                        # Collision came from the pending buffer (this run added
                        # the same key earlier). For same-canonical re-adds this
                        # is a no-op; for different-canonical we must mutate the
                        # pending dict in place so the rename isn't lost on
                        # flush. _alias_index points at the same dict, so
                        # updating it here keeps the index consistent.
                        # Repoint to target_cid, NOT canonical_id: a redirected
                        # alias (see target_cid above) must keep its redirect
                        # target even when a duplicate same-run claim lands here
                        # — repointing to the raw claimant would silently undo
                        # prefer-official for exactly the contested names it
                        # exists to protect.
                        for p in queries._get_pending(store, "aliases"):
                            if (p.get("entity_type") == entity_type
                                    and p.get("raw_value") == raw_value
                                    and queries._source_config_key(p.get("source_config")) == queries._source_config_key(source_cfg)
                                    and p.get("status") != "rejected"):
                                if p["canonical_id"] != target_cid:
                                    prev = p["canonical_id"]
                                    p["canonical_id"] = target_cid
                                    p["source_field"] = "seed"
                                    p["status"] = "confirmed"
                                    p["strategy"] = "seed"
                                    p["confidence"] = 1.0
                                    typer.echo(
                                        f"  [rename] alias {raw_value!r} ({entity_type}) "
                                        f"moved {prev!r} -> {target_cid!r} (pending)"
                                    )
                                    alias_count += 1
                                break
                        continue
                    row = existing.iloc[0]
                    if row["canonical_id"] != target_cid:
                        # Rename: repoint the existing row at the new canonical.
                        queries.update_alias(store, row["id"], {
                            "canonical_id": target_cid,
                            "source_field": "seed",
                            "status": "confirmed",
                            "strategy": "seed",
                            "confidence": 1.0,
                        })
                        typer.echo(
                            f"  [rename] alias {raw_value!r} ({entity_type}) "
                            f"moved {row['canonical_id']!r} -> {target_cid!r}"
                        )
                        alias_count += 1
                    # else: identical re-seed of an existing alias — no-op.

        seed_snapshot.append((table, entity_type, yaml_ids, yaml_alias_keys))
        typer.echo(f"  {label}: {len(items)}")

    # Fail fast on same-run alias collisions (nondeterministic owner). Don't
    # flush — dirty data must not persist. Each line names the contended alias
    # and the two canonicals; fix by removing the duplicate declaration from
    # the non-owning entity (per project rule: the later/more-specific entity
    # owns the name).
    if alias_collisions:
        deduped = sorted(set(alias_collisions))
        lines = "\n".join(
            f"    {rv!r} ({et}{', cfg=' + ck if ck else ''}): "
            f"would point at both {a!r} (declared by {da!r}) "
            f"and {b!r} (declared by {db!r})"
            for (rv, et, ck, a, b, da, db) in deduped
        )
        raise typer.BadParameter(
            f"{len(deduped)} alias collision(s) — the same alias is declared by "
            "more than one canonical in this seed pass, so the owner would be "
            "seed-order-dependent (nondeterministic). Each alias must belong to "
            "exactly one canonical:\n" + lines
        )

    # Flush all buffered upserts (entities + aliases) into their tables in a
    # single pd.concat per table. prune_stale below reads store.table(...)
    # directly, so this must happen before that block.
    queries.flush_pending(store)

    # ------------------------------------------------------------------
    # Inference platforms — flat dim table. Loaded directly into
    # canonical_inference_platforms rather than through the seed_specs loop:
    # its `aliases` column is a stored JSON list of host-token spellings (the
    # single source for lib/inference_platforms_map.py), NOT alias-table rows.
    # ------------------------------------------------------------------
    def _load_inference_platforms_merged() -> list[dict]:
        path = seed_path / "inference_platforms.yaml"
        if not path.exists():
            return []
        with open(path) as f:
            loaded = yaml.safe_load(f) or []
        if not isinstance(loaded, list):
            raise typer.BadParameter(f"{path} must be a flat list")
        return loaded

    inf_plat_entries = _load_inference_platforms_merged()
    if inf_plat_entries:
        import pandas as pd

        now = queries._now()
        inf_cols = list(schemas._SCHEMAS["canonical_inference_platforms"].keys())
        inf_plat_rows = []
        for e in inf_plat_entries:
            if "id" not in e or "display_name" not in e or "kind" not in e:
                raise typer.BadParameter(
                    f"inference_platforms entry missing required field: {e!r}"
                )
            row = {
                "id": e.get("id"),
                "display_name": e.get("display_name"),
                "kind": e.get("kind"),
                "aliases": _json_encode_if_needed(e.get("aliases")),
                "canonical_org": e.get("canonical_org"),
                "variant_of": e.get("variant_of"),
                "homepage": e.get("homepage"),
                "created_at": now,
                "updated_at": now,
            }
            inf_plat_rows.append(row)
        inf_plat_df = pd.DataFrame(inf_plat_rows)
        # Schema-pad any missing columns and order to match the schema.
        for col in inf_cols:
            if col not in inf_plat_df.columns:
                inf_plat_df[col] = None
        inf_plat_df = inf_plat_df[inf_cols]
        store.set_table("canonical_inference_platforms", inf_plat_df)
        typer.echo(f"  inference_platforms: {len(inf_plat_rows)}")

    # ------------------------------------------------------------------
    # Benchmark metric folds — flat dim table (no entity ids, so it skips
    # the seed_specs/upsert path). Entries state that `from_metric` on
    # `benchmark` is the preferred measurement under a generic name.
    # ------------------------------------------------------------------
    folds_path = seed_path / "metric_folds.yaml"
    if folds_path.exists():
        import pandas as pd

        with open(folds_path) as f:
            fold_entries = yaml.safe_load(f) or []
        if not isinstance(fold_entries, list):
            raise typer.BadParameter(f"{folds_path} must be a flat list")
        bench_ids = {e["id"] for e in _load_benchmarks_merged()}
        fold_metrics_path = seed_path / "metrics.yaml"
        metric_ids = set()
        if fold_metrics_path.exists():
            with open(fold_metrics_path) as f:
                metric_ids = {m["id"] for m in (yaml.safe_load(f) or [])}
        seen_pairs: set[tuple[str, str]] = set()
        fold_rows = []
        for e in fold_entries:
            b, fm, tm = e.get("benchmark"), e.get("from_metric"), e.get("to_metric")
            if not (b and fm and tm):
                raise typer.BadParameter(f"metric_folds entry needs benchmark/from_metric/to_metric: {e!r}")
            if b not in bench_ids:
                raise typer.BadParameter(f"metric_folds: unknown benchmark {b!r}")
            missing = [m for m in (fm, tm) if m not in metric_ids]
            if missing:
                raise typer.BadParameter(f"metric_folds ({b}): unknown metric id(s) {missing}")
            if fm == tm:
                raise typer.BadParameter(f"metric_folds ({b}): self-fold {fm!r}")
            if (b, fm) in seen_pairs:
                raise typer.BadParameter(f"metric_folds: duplicate fold for ({b}, {fm})")
            seen_pairs.add((b, fm))
            factor = e.get("scale_factor")
            if factor is not None and (not isinstance(factor, (int, float)) or factor <= 0):
                raise typer.BadParameter(
                    f"metric_folds ({b}): scale_factor must be a positive number, got {factor!r}"
                )
            fold_rows.append({
                "benchmark_id": b, "from_metric_id": fm,
                "to_metric_id": tm, "scale_factor": factor, "note": e.get("note"),
            })
        # No chains: a fold target must not itself be folded on the same benchmark.
        targets = {(r["benchmark_id"], r["to_metric_id"]) for r in fold_rows}
        chained = sorted(targets & seen_pairs)
        if chained:
            raise typer.BadParameter(f"metric_folds: chained folds {chained}")
        folds_df = pd.DataFrame(
            fold_rows, columns=list(schemas._SCHEMAS["benchmark_metric_folds"].keys())
        )
        store.set_table("benchmark_metric_folds", folds_df)
        typer.echo(f"  metric_folds: {len(fold_rows)}")

    # Derive denormalized parent-walk caches now that all canonical_models
    # rows are present. `model_group_id` and `lineage_origin_model_org_id`
    # are computed from `parents` and need the full graph to be in place.
    lineage_counts = queries.derive_model_lineage_fields(store)
    typer.echo(
        f"  derived: model_group_id={lineage_counts['group_set']}, "
        f"model_family_id={lineage_counts['family_set']}, "
        f"lineage_origin_model_id={lineage_counts['lineage_model_set']}, "
        f"lineage_origin_model_org_id={lineage_counts['lineage_org_set']}, "
        f"open_weights_inherited={lineage_counts['open_weights_inherited']}, "
        f"release_date_from_id={lineage_counts['release_date_derived_from_id']}"
    )

    removed_entities = 0
    removed_aliases = 0
    if prune_stale:
        # Remove seed-originated entities and aliases that are no longer in the YAML.
        # Only touches rows that were created by seed (strategy == "seed"), never
        # sync-created aliases or auto-draft entities.
        for table, entity_type, yaml_ids, yaml_alias_keys in seed_snapshot:
            # Remove stale seed aliases for this entity type.
            aliases_df = store.table("aliases")
            seed_mask = (aliases_df["strategy"] == "seed") & (aliases_df["entity_type"] == entity_type)
            if seed_mask.any():
                seed_aliases = aliases_df[seed_mask]
                stale_alias_mask = seed_mask.copy()
                for idx in seed_aliases.index:
                    row = seed_aliases.loc[idx]
                    sc = row.get("source_config")
                    if _is_na(sc):
                        sc = None
                    key = (row["raw_value"], row["entity_type"], row["canonical_id"], sc)
                    if key in yaml_alias_keys:
                        stale_alias_mask[idx] = False
                n_stale = stale_alias_mask.sum()
                if n_stale > 0:
                    store.set_table("aliases", aliases_df[~stale_alias_mask].reset_index(drop=True))
                    removed_aliases += int(n_stale)

            # Remove stale seed entities — only those with review_status "reviewed"
            # that came from seed and are no longer in the YAML.
            entity_df = store.table(table)
            if len(entity_df) > 0:
                stale = entity_df["id"].isin(yaml_ids)
                stale_entities = entity_df[~stale & (entity_df["review_status"] == "reviewed")]
                # FK guard: never prune an org still referenced by a surviving
                # FK. A model whose org_id is DERIVED from its id-prefix (no
                # curated remap) gets an org row auto-created at seed time that
                # isn't in the orgs YAML; without this guard prune would drop it
                # and orphan the model (dangling org_id FK). Orgs also reference
                # other orgs via parent_org_id, so an org that is any other org's
                # parent must be kept too.
                if entity_type == "org" and len(stale_entities) > 0:
                    referenced: set[str] = set()
                    models = store.table("canonical_models")
                    if models is not None and len(models) > 0:
                        for col in ("org_id", "lineage_origin_model_org_id"):
                            if col in models.columns:
                                referenced |= {
                                    str(x) for x in models[col].dropna().astype(str)
                                }
                    if "parent_org_id" in entity_df.columns:
                        referenced |= {
                            str(x) for x in entity_df["parent_org_id"].dropna().astype(str)
                        }
                    stale_entities = stale_entities[
                        ~stale_entities["id"].astype(str).isin(referenced)
                    ]
                # Only remove if every alias for this entity is also seed-originated,
                # meaning it wasn't referenced by sync data.
                current_aliases = store.table("aliases")
                for eid in stale_entities["id"]:
                    entity_aliases = current_aliases[
                        (current_aliases["canonical_id"] == eid)
                        & (current_aliases["entity_type"] == entity_type)
                    ]
                    if len(entity_aliases) == 0 or (entity_aliases["strategy"] == "seed").all():
                        entity_df = entity_df[entity_df["id"] != eid]
                        # Also remove any remaining aliases pointing to it.
                        current_aliases = current_aliases[
                            ~((current_aliases["canonical_id"] == eid)
                              & (current_aliases["entity_type"] == entity_type))
                        ]
                        removed_entities += 1
                store.set_table(table, entity_df.reset_index(drop=True))
                store.set_table("aliases", current_aliases.reset_index(drop=True))

    typer.echo(f"  aliases: {alias_count} added, {removed_aliases} removed")
    if removed_entities:
        typer.echo(f"  stale entities removed: {removed_entities}")

    store.push_to_hub()
    typer.echo("Seed complete.")


# ------------------------------------------------------------------
# stats
# ------------------------------------------------------------------

@app.command()
def stats(
    local: bool = typer.Option(False, "--local", help="Read from fixtures/ instead of HF Hub"),
):
    """Print registry entity counts and pending review summary."""
    import os
    if local:
        os.environ["LOCAL_MODE"] = "true"

    store = _load_store()

    def _row(table):
        df = store.table(table)
        total = len(df)
        draft = int((df["review_status"] == "draft").sum()) if "review_status" in df.columns else 0
        return total, draft

    for label, table in [
        ("models    ", "canonical_models"),
        ("benchmarks", "canonical_benchmarks"),
        ("metrics   ", "canonical_metrics"),
        ("harnesses ", "eval_harnesses"),
    ]:
        total, draft = _row(table)
        typer.echo(f"  {label}  total={total}  draft={draft}")

    aliases_df = store.table("aliases")
    uncertain = int((aliases_df["status"] == "uncertain").sum()) if "status" in aliases_df.columns else 0
    typer.echo(f"\n  aliases        total={len(aliases_df)}  uncertain={uncertain}")
    typer.echo(f"  eval_results   total={len(store.table('eval_results'))}")
    typer.echo(f"  resolution_log total={len(store.table('resolution_log'))}")
    typer.echo(f"  sync_runs      total={len(store.table('sync_runs'))}")


# ------------------------------------------------------------------
# sync
# ------------------------------------------------------------------

@app.command()
def sync(
    config: Optional[str] = typer.Option(None, "--config", help="EEE config name"),
    all_configs: bool = typer.Option(False, "--all", help="Sync all EEE configs"),
    rerun: bool = typer.Option(False, "--rerun", help="Re-resolve all raw strings even if already aliased"),
    local: bool = typer.Option(False, "--local"),
):
    """
    Batch sync EEE config(s) → writes resolved results to eval_results table.
    Each result row is one (model × benchmark × metric) combination with resolved canonical IDs.
    """
    import os
    if local:
        os.environ["LOCAL_MODE"] = "true"

    if not config and not all_configs:
        typer.echo("Specify --config <name> or --all", err=True)
        raise typer.Exit(1)

    from eval_card_registry.services.ingestion import run_sync
    import datasets as ds_lib

    store = _load_store()

    configs_to_run: list[str] = []
    if all_configs:
        configs_to_run = ds_lib.get_dataset_config_names("evaleval/EEE_datastore")
    else:
        configs_to_run = [config]

    failed = []
    for cfg in configs_to_run:
        typer.echo(f"Syncing {cfg}...")
        try:
            counts = run_sync(cfg, store, rerun=rerun)
            typer.echo(f"  {cfg}: {counts}")
        except Exception as e:
            typer.echo(f"  {cfg}: FAILED — {e}", err=True)
            failed.append(cfg)

    typer.echo("Persisting tables...")
    store.push_to_hub()

    if failed:
        typer.echo(f"Done with {len(failed)} failed config(s): {', '.join(failed)}")
    else:
        typer.echo("Done.")
