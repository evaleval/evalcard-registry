"""Change C at the SERVICE level: a versioned harness string resolves via its
bare name, the `auto` alias row the service mints records the strip in `notes`,
exact mode stays an honest `no_match` on a clean store, and an existing
`uncertain` draft alias masks the retry (drafts are never repointed).

One `_fresh_store()` per test on purpose: the cases are order-dependent
because a resolve-mode call PERSISTS an alias for the raw string, and every
later lookup of that string is served from the alias row.
"""
import pytest

from eval_card_registry.store import queries
from eval_card_registry.store import schemas as s
from eval_card_registry.store.hf_store import RegistryStore
from eval_card_registry.services.resolution_service import ResolutionService


def _fresh_store() -> RegistryStore:
    store = RegistryStore()
    store._tables = {name: s.empty(name) for name in [
        "canonical_orgs",
        "canonical_models", "canonical_benchmarks", "canonical_metrics",
        "eval_harnesses", "aliases", "resolution_log", "eval_results", "sync_runs",
    ]}
    store._loaded = True
    return store


def _seed_harness(store: RegistryStore, harness_id: str, alias: str) -> None:
    queries.upsert_entity(store, "eval_harnesses", {
        "id": harness_id,
        "display_name": harness_id,
        "description": None,
        "repo_url": None,
        "metadata": "{}",
        "review_status": "reviewed",
    })
    queries.add_alias(store, {
        "raw_value": alias,
        "entity_type": "harness",
        "canonical_id": harness_id,
        "source_config": None,
        "source_field": None,
        "status": "confirmed",
        "strategy": "exact",
        "confidence": 1.0,
        "notes": None,
    })


def _alias_rows(store: RegistryStore, raw_value: str) -> list[dict]:
    queries.flush_pending(store)
    df = store.table("aliases")
    return df[df["raw_value"] == raw_value].to_dict("records")


def test_exact_mode_writes_nothing_and_stays_no_match():
    store = _fresh_store()
    _seed_harness(store, "helm", "helm")
    svc = ResolutionService(store)
    out = svc.resolve("helm 1.2.3", "harness", None, None, mode="exact")
    assert out["canonical_id"] is None
    assert _alias_rows(store, "helm 1.2.3") == []
    assert store.table("resolution_log").empty


def test_resolve_mode_strips_and_records_the_marker():
    store = _fresh_store()
    _seed_harness(store, "helm", "helm")
    svc = ResolutionService(store)
    out = svc.resolve("helm 1.2.3", "harness", None, None)
    assert out["canonical_id"] == "helm"
    assert out["strategy"] == "normalized"
    assert out["confidence"] == 0.95
    assert out["resolution_detail"] == {
        "harness_version_stripped": "1.2.3",
        "bare_name": "helm",
        "bare_tier": "exact",
    }
    rows = _alias_rows(store, "helm 1.2.3")
    assert len(rows) == 1
    assert rows[0]["status"] == "auto"
    assert rows[0]["strategy"] == "normalized"
    assert rows[0]["confidence"] == 0.95
    assert rows[0]["notes"].startswith("harness version stripped: 1.2.3")

    # After the write, exact mode serves the PERSISTED provenance (this is
    # pre-existing behaviour for every normalized hit, pinned on purpose).
    again = svc.resolve("helm 1.2.3", "harness", None, None, mode="exact")
    assert (again["canonical_id"], again["strategy"], again["confidence"]) == (
        "helm", "normalized", 0.95,
    )


def test_rerun_keeps_strip_marker():
    store = _fresh_store()
    _seed_harness(store, "helm", "helm")
    svc = ResolutionService(store)
    svc.resolve("helm 1.2.3", "harness", None, None)
    queries.flush_pending(store)
    svc.resolve("helm 1.2.3", "harness", None, None, rerun=True)
    rows = _alias_rows(store, "helm 1.2.3")
    assert len(rows) == 1
    assert rows[0]["notes"].startswith("harness version stripped: 1.2.3")


def test_existing_uncertain_draft_masks_retry():
    """C only reaches strings with no non-rejected alias. A versioned string
    the service has already seen carries an `uncertain` alias to a draft, and
    neither a later seed nor `rerun=True` repoints it — the clean-up is
    manual (reject the alias, delete the draft)."""
    store = _fresh_store()
    svc = ResolutionService(store)
    first = svc.resolve("alpaca_eval 2.0", "harness", None, None)
    queries.flush_pending(store)
    draft_id = first["canonical_id"]
    assert first["created_new"] is True
    rows = _alias_rows(store, "alpaca_eval 2.0")
    assert rows[0]["status"] == "uncertain"

    _seed_harness(store, "alpaca-eval", "alpaca_eval")
    svc.invalidate_resolver()
    svc._resolve_cache.clear()
    second = svc.resolve("alpaca_eval 2.0", "harness", None, None)
    assert second["canonical_id"] == draft_id != "alpaca-eval"
    assert "harness_version_stripped" not in (second.get("resolution_detail") or {})
