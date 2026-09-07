"""Tests for resolution_service: entity lifecycle, orphan cleanup, rerun behavior."""
import json
import pytest

from eval_card_registry.store.hf_store import RegistryStore
from eval_card_registry.store import schemas
from eval_card_registry.services.resolution_service import ResolutionService


def _fresh_store() -> RegistryStore:
    store = RegistryStore()
    from eval_card_registry.store import schemas as s
    store._tables = {name: s.empty(name) for name in [
        "canonical_orgs",
        "canonical_models", "canonical_benchmarks", "canonical_metrics",
        "eval_harnesses", "aliases", "resolution_log", "eval_results", "sync_runs",
    ]}
    store._loaded = True
    return store


def _seed_benchmark(store: RegistryStore, id: str, display_name: str):
    from eval_card_registry.store import queries
    import json
    queries.upsert_entity(store, "canonical_benchmarks", {
        "id": id,
        "display_name": display_name,
        "description": None,
        "dataset_repo": None,
        "parent_benchmark_id": None,
        "tags": "[]",
        "metadata": "{}",
        "review_status": "reviewed",
    })
    # Add alias for exact lookup
    queries.add_alias(store, {
        "raw_value": display_name,
        "entity_type": "benchmark",
        "canonical_id": id,
        "source_config": None,
        "source_field": None,
        "status": "confirmed",
        "strategy": "exact",
        "confidence": 1.0,
        "notes": None,
    })


class TestHubStatsIndexPriority:
    """The live hub-stats alias index must register canonical IDS FIRST (ids beat
    aliases on a normalized-form collision), matching the generator's order.
    Building aliases-first would make an identical baseModels edge resolve to a
    DIFFERENT parent live vs at generate time — a wrong-id bug."""

    def test_canonical_id_wins_over_a_colliding_alias(self):
        import pandas as pd

        store = _fresh_store()
        # Model A's id == the normalized form that Model B also lists as an alias.
        store._tables["canonical_models"] = pd.DataFrame(
            [{"id": "deepseek-ai/DeepSeek-V3.1"}, {"id": "deepseek-ai/deepseek-v3.1-collapsed"}]
        )
        store._tables["aliases"] = pd.DataFrame(
            [{
                "entity_type": "model",
                "raw_value": "deepseek-ai/DeepSeek-V3.1",   # B claims A's id as an alias
                "canonical_id": "deepseek-ai/deepseek-v3.1-collapsed",
            }]
        )
        svc = ResolutionService(store)
        a2c, _ = svc._build_hub_stats_indices()
        from eval_card_registry.services.hub_stats import normalize as _hsnorm

        # ids-first: the contested form resolves to the ID owner (A), not the
        # alias owner (B) — a baseModels edge to DeepSeek-V3.1 points at the real
        # repo, identical to what the generator produces.
        assert a2c[_hsnorm("deepseek-ai/DeepSeek-V3.1")] == "deepseek-ai/DeepSeek-V3.1"

    @pytest.mark.parametrize(
        ("hf_org", "canonical_org"),
        [("CohereLabs", "cohere"), ("ibm-granite", "ibm")],
    )
    def test_live_hub_stats_index_uses_complete_curated_org_map(
        self, hf_org, canonical_org
    ):
        """Runtime enrichment and the bulk cron must share one org map."""
        import pandas as pd

        store = _fresh_store()
        store._tables["canonical_orgs"] = pd.DataFrame(
            [
                {"id": "cohere", "hf_org": "CohereForAI"},
                {"id": "ibm", "hf_org": "ibm"},
            ]
        )
        svc = ResolutionService(store)

        _, org_map = svc._build_hub_stats_indices()

        assert org_map[hf_org.lower()] == canonical_org


class TestResolutionService:
    def test_resolves_known_entity(self):
        store = _fresh_store()
        _seed_benchmark(store, "ifeval", "IFEval")
        svc = ResolutionService(store)
        result = svc.resolve("IFEval", "benchmark", None, None)
        assert result["canonical_id"] == "ifeval"
        assert result["created_new"] is False

    def test_auto_creates_draft_on_no_match(self):
        store = _fresh_store()
        svc = ResolutionService(store)
        result = svc.resolve("Some Unknown Benchmark", "benchmark", "test_config", None)
        assert result["canonical_id"] is not None
        assert result["created_new"] is True
        assert result["review_status"] == "draft"

    def test_idempotent_on_second_call(self):
        from eval_card_registry.store import queries
        store = _fresh_store()
        svc = ResolutionService(store)
        r1 = svc.resolve("Novel Benchmark X", "benchmark", "cfg", None)
        r2 = svc.resolve("Novel Benchmark X", "benchmark", "cfg", None)
        assert r1["canonical_id"] == r2["canonical_id"]
        # Flush pending writes and verify only one entity was created
        queries.flush_pending(store)
        df = store.table("canonical_benchmarks")
        assert len(df) == 1

    def test_rerun_bypasses_cache(self):
        store = _fresh_store()
        _seed_benchmark(store, "real-benchmark-y", "Novel Benchmark Y")
        svc = ResolutionService(store)

        # First call populates alias
        r1 = svc.resolve("Novel Benchmark Y", "benchmark", "cfg", None)
        assert r1["canonical_id"] == "real-benchmark-y"

        # Second call without rerun uses alias cache
        r2 = svc.resolve("Novel Benchmark Y", "benchmark", "cfg", None, rerun=False)
        assert r2["canonical_id"] == "real-benchmark-y"
        assert r2["created_new"] is False

        # rerun=True re-evaluates; result is the same since seed data hasn't changed
        svc.invalidate_resolver()
        r3 = svc.resolve("Novel Benchmark Y", "benchmark", "cfg", None, rerun=True)
        assert r3["canonical_id"] == "real-benchmark-y"

    def test_subset_alias_resolves_to_parent(self):
        """A subset aliased to a parent entity resolves to the parent, not a new entity."""
        store = _fresh_store()
        _seed_benchmark(store, "parent-bench", "Parent Bench")
        from eval_card_registry.store import queries
        queries.add_alias(store, {
            "raw_value": "Parent Bench Subset X",
            "entity_type": "benchmark",
            "canonical_id": "parent-bench",
            "source_config": None,
            "source_field": None,
            "status": "confirmed",
            "strategy": "exact",
            "confidence": 1.0,
            "notes": None,
        })
        svc = ResolutionService(store)
        result = svc.resolve("Parent Bench Subset X", "benchmark", "some_config", None)
        assert result["canonical_id"] == "parent-bench"
        assert result["created_new"] is False

    def test_empty_raw_value_returns_none(self):
        store = _fresh_store()
        svc = ResolutionService(store)
        result = svc.resolve("", "benchmark", "cfg", None)
        assert result["canonical_id"] is None
        assert result["strategy"] == "no_match"

    def test_resolve_preserves_resolved_leaf_id_for_version_chain(self):
        """When a model raw value matches a leaf canonical that has a
        version-axis chain to a group pointer, `svc.resolve` returns the
        LEAF as canonical_id with the group root in
        `model_group_id`. `resolved_leaf_id == canonical_id`.
        Regression: the previous implementation re-enriched via
        `build_result(canonical_id=root, ...)` which clobbered the leaf —
        guard that the leaf still flows through unmodified."""
        from eval_card_registry.store import queries
        store = _fresh_store()
        # Seed org so the model FK resolves.
        queries.upsert_entity(store, "canonical_orgs", {
            "id": "allenai", "display_name": "Allen AI", "parent_org_id": None,
            "website": None, "hf_org": "allenai", "kind": "lab",
            "tags": "[]", "metadata": "{}", "review_status": "reviewed",
        })
        # Family pointer with release_date.
        queries.upsert_entity(store, "canonical_models", {
            "id": "allenai/olmo-3-32b", "display_name": "OLMo-3 32B",
            "developer": None, "org_id": "allenai", "family": "olmo-3-32b",
            "architecture": None, "params_billions": 32.0,
            "parents": "[]", "model_group_id": None,
            "lineage_origin_model_org_id": "allenai", "open_weights": True,
            "release_date": "2025-11-25", "tags": "[]",
            "metadata": "{}", "review_status": "reviewed",
        })
        # Snapshot canonical with version-axis parent → family.
        # model_group_id is set explicitly here to match what
        # derive_model_lineage_fields would produce after seed.
        queries.upsert_entity(store, "canonical_models", {
            "id": "allenai/olmo-3-1125-32b", "display_name": "OLMo-3 32B (1125)",
            "developer": None, "org_id": "allenai", "family": "olmo-3-32b",
            "architecture": None, "params_billions": 32.0,
            "parents": json.dumps([{
                "id": "allenai/olmo-3-32b",
                "relationship": "variant",
                "axis": "version",
            }]),
            "model_group_id": "allenai/olmo-3-32b",
            "lineage_origin_model_org_id": "allenai", "open_weights": True,
            "release_date": "2025-11-25", "tags": "[]",
            "metadata": "{}", "review_status": "reviewed",
        })
        # Alias on the snapshot.
        queries.add_alias(store, {
            "raw_value": "allenai/Olmo-3-1125-32B",
            "entity_type": "model",
            "canonical_id": "allenai/olmo-3-1125-32b",
            "source_config": None, "source_field": None,
            "status": "confirmed", "strategy": "seed",
            "confidence": 1.0, "notes": None,
        })

        svc = ResolutionService(store)
        # Existing-alias path: get_alias hits the seeded alias above,
        # then the fix re-runs the strategy chain to recover the leaf
        # (the alias table only stores root-collapsed canonical_id).
        r1 = svc.resolve("allenai/Olmo-3-1125-32B", "model", None, None)
        # canonical_id is the LEAF snapshot; the group root
        # moves to model_group_id. resolved_leaf_id == canonical_id.
        assert r1["canonical_id"] == "allenai/olmo-3-1125-32b"
        assert r1["resolved_leaf_id"] == "allenai/olmo-3-1125-32b"
        assert r1["model_group_id"] == "allenai/olmo-3-32b"

        # _resolve_cache short-circuits the second call to the same
        # (raw, entity_type, source_config). To exercise the existing-
        # alias path again under the same fixture we'd need to clear
        # the cache; the live verification already covers cache-hit
        # idempotence so we don't re-test it here.

    def test_resolve_preserves_leaf_via_fresh_resolve_path(self):
        """The fresh-resolve path: no `aliases` row exists for the
        exact-cased raw value, so the strategy chain runs from scratch
        and matches via the normalized index (case-insensitive). The
        fix returns the resolver's `result` directly — the previous
        implementation called `build_result(canonical_id=root, ...)`
        which clobbered `resolved_leaf_id` to equal the root."""
        from eval_card_registry.store import queries
        store = _fresh_store()
        queries.upsert_entity(store, "canonical_orgs", {
            "id": "allenai", "display_name": "Allen AI", "parent_org_id": None,
            "website": None, "hf_org": "allenai", "kind": "lab",
            "tags": "[]", "metadata": "{}", "review_status": "reviewed",
        })
        queries.upsert_entity(store, "canonical_models", {
            "id": "allenai/olmo-3-32b", "display_name": "OLMo-3 32B",
            "developer": None, "org_id": "allenai", "family": "olmo-3-32b",
            "architecture": None, "params_billions": 32.0,
            "parents": "[]", "model_group_id": None,
            "lineage_origin_model_org_id": "allenai", "open_weights": True,
            "release_date": "2025-11-25", "tags": "[]",
            "metadata": "{}", "review_status": "reviewed",
        })
        queries.upsert_entity(store, "canonical_models", {
            "id": "allenai/olmo-3-1125-32b", "display_name": "OLMo-3 32B (1125)",
            "developer": None, "org_id": "allenai", "family": "olmo-3-32b",
            "architecture": None, "params_billions": 32.0,
            "parents": json.dumps([{
                "id": "allenai/olmo-3-32b",
                "relationship": "variant",
                "axis": "version",
            }]),
            "model_group_id": "allenai/olmo-3-32b",
            "lineage_origin_model_org_id": "allenai", "open_weights": True,
            "release_date": "2025-11-25", "tags": "[]",
            "metadata": "{}", "review_status": "reviewed",
        })
        # Seed only the LOWERCASE alias on the snapshot. The mixed-case
        # raw below has NO matching entry in `aliases` (case-sensitive
        # lookup), so get_alias misses → fresh-resolve path runs.
        queries.add_alias(store, {
            "raw_value": "allenai/olmo-3-1125-32b",
            "entity_type": "model",
            "canonical_id": "allenai/olmo-3-1125-32b",
            "source_config": None, "source_field": None,
            "status": "confirmed", "strategy": "seed",
            "confidence": 1.0, "notes": None,
        })

        svc = ResolutionService(store)
        # Mixed-case input — no exact alias match → falls to the strategy
        # chain → normalized match hits the snapshot canonical → leaf
        # is preserved through the fix's `enriched = result` branch.
        r = svc.resolve("allenai/Olmo-3-1125-32B", "model", None, None)
        # Leaf snapshot as canonical_id, group root in
        # model_group_id; resolved_leaf_id == canonical_id.
        assert r["canonical_id"] == "allenai/olmo-3-1125-32b"
        assert r["resolved_leaf_id"] == "allenai/olmo-3-1125-32b"
        assert r["model_group_id"] == "allenai/olmo-3-32b"
        assert r["strategy"] == "normalized"
        assert r["created_new"] is False


def _seed_org(store, org_id="meta"):
    from eval_card_registry.store import queries
    queries.upsert_entity(store, "canonical_orgs", {
        "id": org_id, "display_name": org_id, "parent_org_id": None,
        "website": None, "hf_org": org_id, "kind": "lab",
        "tags": "[]", "metadata": "{}", "review_status": "reviewed",
    })


def _seed_model(store, mid, org_id, *, aliases=None):
    from eval_card_registry.store import queries
    queries.upsert_entity(store, "canonical_models", {
        "id": mid, "display_name": mid,
        "developer": None, "org_id": org_id, "family": None,
        "architecture": None, "params_billions": None,
        "parents": "[]", "model_group_id": None,
        "lineage_origin_model_org_id": org_id, "open_weights": True,
        "tags": "[]", "metadata": "{}", "review_status": "reviewed",
        "resolution_source": "curated", "resolution_granularity": "variant",
    })
    for a in [mid] + list(aliases or []):
        queries.add_alias(store, {
            "raw_value": a, "entity_type": "model", "canonical_id": mid,
            "source_config": None, "source_field": None, "status": "confirmed",
            "strategy": "seed", "confidence": 1.0, "notes": None,
        })


class TestTier3Inference:
    def test_inferred_source_and_granularity_on_auto_create(self):
        """A no_match model draft (not HF-confirmed) is flagged
        resolution_source=inferred, resolution_granularity=variant."""
        store = _fresh_store()
        _seed_org(store, "someorg")
        svc = ResolutionService(store)
        r = svc.resolve("someorg/Totally-Vanity-Name-3B", "model", None, None)
        assert r["created_new"] is True
        assert r["resolution_source"] == "inferred"
        assert r["resolution_granularity"] == "variant"
        assert r["review_status"] == "draft"

    def test_org_less_no_slash_flagged_org_unknown(self):
        """A bare free-text label (no org prefix) mints with org_id=None
        and tags:[org-unknown] — never an auto-guessed org."""
        store = _fresh_store()
        svc = ResolutionService(store)
        r = svc.resolve("Cohere May 2024", "model", None, None)
        assert r["created_new"] is True
        assert r["resolution_source"] == "inferred"
        from eval_card_registry.services.resolution_service import _table_with_pending
        mid = r["canonical_id"]
        row = _table_with_pending(store, "canonical_models")
        row = row[row["id"] == mid].iloc[0]
        assert row["org_id"] is None or (isinstance(row["org_id"], float))
        assert "org-unknown" in json.loads(row["tags"])

    def test_unknown_placeholder_org_is_org_less(self):
        """`unknown/foo` has a slash but the org part is a placeholder →
        treated as org-less (org_id None + org-unknown tag), not org 'unknown'."""
        store = _fresh_store()
        svc = ResolutionService(store)
        r = svc.resolve("unknown/iSWE_Agent", "model", None, None)
        from eval_card_registry.services.resolution_service import _table_with_pending
        mid = r["canonical_id"]
        row = _table_with_pending(store, "canonical_models")
        row = row[row["id"] == mid].iloc[0]
        assert row["org_id"] is None or isinstance(row["org_id"], float)
        assert "org-unknown" in json.loads(row["tags"])

    def test_alias_confirmed_base_edge_emitted(self):
        """A community finetune with a recognizable base that alias-confirms
        gets a single finetune edge to the confirmed base canonical."""
        store = _fresh_store()
        _seed_org(store, "meta")
        _seed_model(store, "meta/llama-3.1-8b", "meta",
                    aliases=["meta-llama/Llama-3.1-8B", "llama-3.1-8b"])
        svc = ResolutionService(store)
        r = svc.resolve("3rd-Degree-Burn/Llama-3.1-8B-Squareroot", "model", None, None)
        assert r["created_new"] is True
        parents = r["parents"] or []
        assert any(
            p["id"] == "meta/llama-3.1-8b" and p["relationship"] == "finetune"
            for p in parents
        ), parents

    def test_no_invented_edge_for_opaque_name(self):
        """An opaque vanity name with no alias-confirmable base gets NO
        parent edge — never an invented edge."""
        store = _fresh_store()
        _seed_org(store, "auraindustries")
        svc = ResolutionService(store)
        r = svc.resolve("AuraIndustries/Aura-MoE-2x4B", "model", None, None)
        assert r["created_new"] is True
        assert (r["parents"] or []) == []

    def test_no_self_edge_when_base_is_same_identity(self):
        """If the only alias-confirmable 'base' is the same model identity
        (org-less twin), no edge is emitted (no self-edge)."""
        store = _fresh_store()
        _seed_model(store, "yi-lightning", None)  # bare org-less canonical
        svc = ResolutionService(store)
        r = svc.resolve("01-ai/yi-lightning", "model", None, None)
        assert r["created_new"] is True
        # base candidate `yi-lightning` matches but is the same name → rejected
        assert (r["parents"] or []) == []
