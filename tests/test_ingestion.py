"""Tests for ingestion pipeline: sub-category detection, per-record processing, and eval_results table."""
import pytest

from eval_card_registry.services.ingestion import (
    _detect_sub_categories, process_record,
)
from eval_card_registry.store.hf_store import RegistryStore
from eval_card_registry.services.resolution_service import ResolutionService
from eval_card_registry.store import schemas as s
from eval_card_registry.store import queries


def _fresh_store() -> RegistryStore:
    store = RegistryStore()
    store._tables = {name: s.empty(name) for name in [
        "canonical_models", "canonical_benchmarks", "canonical_metrics",
        "canonical_orgs", "canonical_families", "canonical_composites",
        "eval_harnesses", "aliases", "resolution_log", "eval_results", "sync_runs",
    ]}
    store._loaded = True
    return store


def _add_entity_with_aliases(store, table, entity_type, entity, extra_aliases=None):
    """Helper: insert entity + aliases (id, display_name, and extras)."""
    queries.upsert_entity(store, table, entity)
    alias_strings = {entity["id"], entity.get("display_name", "")}
    if extra_aliases:
        alias_strings |= set(extra_aliases)
    for raw in alias_strings:
        if not raw:
            continue
        try:
            queries.add_alias(store, {
                "raw_value": raw, "entity_type": entity_type,
                "canonical_id": entity["id"], "source_config": None,
                "source_field": "seed", "status": "confirmed",
                "strategy": "seed", "confidence": 1.0, "notes": None,
            })
        except ValueError:
            pass


def _seeded_store() -> RegistryStore:
    """Store pre-populated with seed entities AND aliases, like a real seed --local run."""
    store = _fresh_store()

    # Benchmarks — mirrors seed/benchmarks.yaml
    benchmarks = [
        {"id": "ifeval", "display_name": "IFEval"},
        {"id": "bbh", "display_name": "BBH"},
        {"id": "math", "display_name": "MATH", "_aliases": ["MATH Level 5"]},
        {"id": "mmlu-pro", "display_name": "MMLU-Pro", "_aliases": ["MMLU-PRO", "mmlu_pro"]},
        {"id": "gpqa", "display_name": "GPQA"},
        {"id": "musr", "display_name": "MuSR", "_aliases": ["MUSR"]},
        {"id": "mmlu", "display_name": "MMLU", "_aliases": ["MMLU All Subjects"]},
        {"id": "gsm8k", "display_name": "GSM8K"},
        {"id": "narrativeqa", "display_name": "NarrativeQA"},
        {"id": "naturalquestions", "display_name": "NaturalQuestions",
         "_aliases": ["NaturalQuestions (closed-book)"]},
        {"id": "openbookqa", "display_name": "OpenbookQA"},
        {"id": "legalbench", "display_name": "LegalBench"},
        {"id": "medqa", "display_name": "MedQA"},
        {"id": "swe-bench", "display_name": "SWE-bench", "_aliases": ["swe-bench"]},
        {"id": "rewardbench", "display_name": "RewardBench"},
        {"id": "rewardbench-chat", "display_name": "RewardBench Chat", "_aliases": ["Chat"]},
        {"id": "rewardbench-chat-hard", "display_name": "RewardBench Chat Hard",
         "_aliases": ["Chat Hard"]},
        {"id": "rewardbench-safety", "display_name": "RewardBench Safety", "_aliases": ["Safety"]},
        {"id": "rewardbench-reasoning", "display_name": "RewardBench Reasoning",
         "_aliases": ["Reasoning"]},
        {"id": "bfcl", "display_name": "BFCL"},
        {"id": "wildbench", "display_name": "WildBench"},
    ]
    for b in benchmarks:
        extra = b.pop("_aliases", [])
        entity = {**b, "review_status": "reviewed", "description": None,
                  "dataset_repo": None, "parent_benchmark_id": None,
                  "tags": "[]", "metadata": "{}"}
        _add_entity_with_aliases(store, "canonical_benchmarks", "benchmark", entity, extra)

    # Metrics — mirrors seed/metrics.yaml
    metrics = [
        {"id": "accuracy", "display_name": "Accuracy", "_aliases": ["Acc"]},
        {"id": "exact-match", "display_name": "Exact Match", "_aliases": ["EM"]},
        {"id": "f1", "display_name": "F1 Score", "_aliases": ["F1"]},
        {"id": "pass-at-1", "display_name": "Pass@1"},
        {"id": "mean-win-rate", "display_name": "Mean Win Rate", "_aliases": ["Mean win rate"]},
        {"id": "score", "display_name": "Score", "_aliases": ["score"]},
        {"id": "bleu-4", "display_name": "BLEU-4"},
        {"id": "cot-correct", "display_name": "COT Correct",
         "_aliases": ["COT correct", "chain_of_thought_correctness"]},
        {"id": "win-rate", "display_name": "Win Rate", "_aliases": ["Win rate (%)"]},
        {"id": "mean-score", "display_name": "Mean Score", "_aliases": ["Mean score"]},
    ]
    for m in metrics:
        extra = m.pop("_aliases", [])
        entity = {**m, "review_status": "reviewed", "score_type": "continuous",
                  "lower_is_better": False, "min_score": 0.0, "max_score": 1.0, "metadata": "{}"}
        _add_entity_with_aliases(store, "canonical_metrics", "metric", entity, extra)

    # Harnesses — mirrors seed/harnesses.yaml
    harnesses = [
        {"id": "lm-evaluation-harness", "display_name": "LM Evaluation Harness"},
        {"id": "helm", "display_name": "HELM"},
        {"id": "rewardbench", "display_name": "RewardBench", "_aliases": ["rewardbench"]},
        {"id": "wordle-arena", "display_name": "Wordle Arena", "_aliases": ["wordle_arena"]},
        {"id": "bfcl-harness", "display_name": "BFCL", "_aliases": ["BFCL"]},
    ]
    for h in harnesses:
        extra = h.pop("_aliases", [])
        entity = {**h, "review_status": "reviewed", "version": None,
                  "fork_url": None, "metadata": "{}"}
        _add_entity_with_aliases(store, "eval_harnesses", "harness", entity, extra)

    return store


class TestSubCategoryDetection:
    def test_standalone_benchmarks(self):
        evaluation_results = [
            {"evaluation_name": "IFEval", "source_data": {"dataset_name": "IFEval"}},
            {"evaluation_name": "BBH", "source_data": {"dataset_name": "BBH"}},
        ]
        mapping = _detect_sub_categories(evaluation_results)
        assert mapping["IFEval"] is None
        assert mapping["BBH"] is None

    def test_sub_category_detection(self):
        evaluation_results = [
            {"evaluation_name": "Chat", "source_data": {"dataset_name": "RewardBench"}},
            {"evaluation_name": "Chat Hard", "source_data": {"dataset_name": "RewardBench"}},
            {"evaluation_name": "Safety", "source_data": {"dataset_name": "RewardBench"}},
        ]
        mapping = _detect_sub_categories(evaluation_results)
        assert mapping["Chat"] == "RewardBench"
        assert mapping["Chat Hard"] == "RewardBench"
        assert mapping["Safety"] == "RewardBench"

    def test_missing_dataset_name(self):
        evaluation_results = [
            {"evaluation_name": "SomeBenchmark", "source_data": {}},
        ]
        mapping = _detect_sub_categories(evaluation_results)
        assert mapping["SomeBenchmark"] is None


class TestProcessRecord:
    def _make_record(self):
        return {
            "model_info": {"id": "meta-llama/Llama-3.1-8B"},
            "eval_library": {"name": "lm-evaluation-harness", "version": "0.4.2"},
            "evaluation_results": [
                {
                    "evaluation_name": "IFEval",
                    "source_data": {"dataset_name": "IFEval"},
                    "metric_config": {"evaluation_description": "Accuracy on IFEval"},
                    "score": 0.42,
                }
            ],
            "source_metadata": {"evaluation_id": "test/eval-001"},
        }

    def test_process_record_returns_flat_result_rows(self):
        store = _fresh_store()
        svc = ResolutionService(store)
        from eval_card_registry.store import queries
        run_id = queries.start_sync_run(store, "test_config", False)

        record = self._make_record()
        rows = process_record(record, "test_config", svc, run_id, rerun=False)

        assert rows is not None
        assert len(rows) == 1
        row = rows[0]
        assert row["evaluation_id"] == "test/eval-001"
        assert row["result_index"] == 0
        assert row["model_id"] is not None
        assert row["harness_id"] is not None
        assert row["benchmark_id"] is not None
        assert row["benchmark_card_id"] is None

    def test_process_record_missing_model_returns_none(self):
        store = _fresh_store()
        svc = ResolutionService(store)
        from eval_card_registry.store import queries
        run_id = queries.start_sync_run(store, "test_config", False)

        record = {"model_info": {}, "evaluation_results": []}
        row = process_record(record, "test_config", svc, run_id, rerun=False)
        assert row is None

    def test_multiple_eval_results_produce_multiple_rows(self):
        """One EEE record with 3 benchmarks → 3 flat rows."""
        store = _fresh_store()
        svc = ResolutionService(store)
        from eval_card_registry.store import queries
        run_id = queries.start_sync_run(store, "test_config", False)

        record = {
            "model_info": {"id": "meta-llama/Llama-3.1-8B"},
            "eval_library": {"name": "lm-evaluation-harness", "version": "0.4.2"},
            "evaluation_results": [
                {"evaluation_name": "IFEval", "source_data": {"dataset_name": "IFEval"},
                 "metric_config": {"evaluation_description": "Accuracy"}, "score": 0.42},
                {"evaluation_name": "BBH", "source_data": {"dataset_name": "BBH"},
                 "metric_config": {"evaluation_description": "Exact Match"}, "score": 0.55},
                {"evaluation_name": "MMLU", "source_data": {"dataset_name": "MMLU"},
                 "metric_config": {"evaluation_description": "Accuracy"}, "score": 0.73},
            ],
            "source_metadata": {"evaluation_id": "test/eval-multi"},
        }
        rows = process_record(record, "test_config", svc, run_id, rerun=False)

        assert len(rows) == 3
        # All rows share the same model and evaluation_id
        assert all(r["evaluation_id"] == "test/eval-multi" for r in rows)
        assert all(r["model_id"] == rows[0]["model_id"] for r in rows)
        # Each row has a distinct result_index
        assert [r["result_index"] for r in rows] == [0, 1, 2]
        # Each row has a distinct benchmark_id
        benchmark_ids = [r["benchmark_id"] for r in rows]
        assert len(set(benchmark_ids)) == 3

    def test_score_of_zero_is_preserved(self):
        """A score of 0 or 0.0 must not be treated as missing."""
        store = _fresh_store()
        svc = ResolutionService(store)
        from eval_card_registry.store import queries
        run_id = queries.start_sync_run(store, "test_config", False)

        record = {
            "model_info": {"id": "some-model/v1"},
            "eval_library": {"name": "harness"},
            "evaluation_results": [
                {"evaluation_name": "ZeroBench", "source_data": {"dataset_name": "ZeroBench"},
                 "metric_config": {}, "score": 0},
                {"evaluation_name": "ZeroBench2", "source_data": {"dataset_name": "ZeroBench2"},
                 "metric_config": {}, "score_details": {"score": 0.0}},
            ],
            "source_metadata": {"evaluation_id": "test/zero-score"},
        }
        rows = process_record(record, "test_config", svc, run_id, rerun=False)
        assert rows[0]["score"] == 0
        assert rows[1]["score"] == 0.0

    def test_skipped_eval_result_preserves_indices(self):
        """If an eval result has no evaluation_name, it's skipped but indices stay stable."""
        store = _fresh_store()
        svc = ResolutionService(store)
        from eval_card_registry.store import queries
        run_id = queries.start_sync_run(store, "test_config", False)

        record = {
            "model_info": {"id": "some-model/v1"},
            "eval_library": {"name": "harness"},
            "evaluation_results": [
                {"evaluation_name": "First", "source_data": {"dataset_name": "First"},
                 "metric_config": {}, "score": 0.1},
                {"evaluation_name": "", "source_data": {},  # skipped — no name
                 "metric_config": {}, "score": 0.2},
                {"evaluation_name": "Third", "source_data": {"dataset_name": "Third"},
                 "metric_config": {}, "score": 0.3},
            ],
            "source_metadata": {"evaluation_id": "test/skip"},
        }
        rows = process_record(record, "test_config", svc, run_id, rerun=False)
        assert len(rows) == 2
        # Index 1 was skipped, so we get 0 and 2 (source array positions)
        assert rows[0]["result_index"] == 0
        assert rows[1]["result_index"] == 2

    def test_empty_eval_results_returns_empty_list(self):
        """Record with model but no evaluation_results → empty list (not None)."""
        store = _fresh_store()
        svc = ResolutionService(store)
        from eval_card_registry.store import queries
        run_id = queries.start_sync_run(store, "test_config", False)

        record = {
            "model_info": {"id": "some-model/v1"},
            "eval_library": {"name": "harness"},
            "evaluation_results": [],
            "source_metadata": {"evaluation_id": "test/empty"},
        }
        rows = process_record(record, "test_config", svc, run_id, rerun=False)
        assert rows == []

    def test_deterministic_eval_id_fallback(self):
        """When evaluation_id is missing, fallback is deterministic (not based on object id)."""
        store1 = _fresh_store()
        svc1 = ResolutionService(store1)
        run_id1 = queries.start_sync_run(store1, "test_config", False)

        record = {
            "model_info": {"id": "meta-llama/Llama-3.1-8B"},
            "eval_library": {"name": "harness"},
            "evaluation_results": [
                {"evaluation_name": "IFEval", "source_data": {"dataset_name": "IFEval"},
                 "metric_config": {}, "score": 0.5},
            ],
            # No evaluation_id anywhere
        }
        rows1 = process_record(record, "test_config", svc1, run_id1, rerun=False)

        # Re-process the same record with a fresh store (simulating second run)
        store2 = _fresh_store()
        svc2 = ResolutionService(store2)
        run_id2 = queries.start_sync_run(store2, "test_config", False)
        rows2 = process_record(record, "test_config", svc2, run_id2, rerun=False)
        assert rows1[0]["evaluation_id"] == rows2[0]["evaluation_id"]
        assert rows1[0]["evaluation_id"].startswith("test_config/auto-")


class TestEvalResultsUpsert:
    def test_upsert_inserts_then_updates(self):
        """First call inserts, second call with same key updates in pending buffer."""
        store = _fresh_store()
        row = {
            "evaluation_id": "eval-001",
            "result_index": 0,
            "source_config": "cfg",
            "model_id": "old-model",
            "harness_id": None,
            "benchmark_id": "old-bench",
            "parent_benchmark_id": None,
            "metric_id": "old-metric",
            "benchmark_card_id": None,
            "score": 0.5,
            "score_details": None,
        }
        result1 = queries.upsert_eval_result(store, row)
        row_id = result1["id"]

        # Simulate rerun — benchmark resolved differently
        row2 = {**row, "benchmark_id": "corrected-bench", "metric_id": "corrected-metric"}
        result2 = queries.upsert_eval_result(store, row2)
        assert result2["id"] == row_id  # same deterministic ID
        assert result2["benchmark_id"] == "corrected-bench"
        assert result2["metric_id"] == "corrected-metric"
        # created_at should be preserved, updated_at should change
        assert result2["created_at"] == result1["created_at"]
        assert result2["updated_at"] >= result1["updated_at"]

        # After flush, table should have exactly 1 row
        queries.flush_pending(store)
        assert len(store.table("eval_results")) == 1

    def test_different_result_index_creates_separate_row(self):
        """Same evaluation_id but different result_index → different rows."""
        store = _fresh_store()
        base = {
            "evaluation_id": "eval-001",
            "source_config": "cfg",
            "model_id": "model-a",
            "harness_id": None,
            "benchmark_id": "bench",
            "parent_benchmark_id": None,
            "metric_id": "metric",
            "benchmark_card_id": None,
            "score": 0.5,
            "score_details": None,
        }
        queries.upsert_eval_result(store, {**base, "result_index": 0})
        queries.upsert_eval_result(store, {**base, "result_index": 1})
        queries.flush_pending(store)
        assert len(store.table("eval_results")) == 2




class TestResolutionCorrectness:
    """Verify that resolution produces the RIGHT canonical IDs, not just non-null ones."""

    def _make_eee_record(self, model_id, benchmarks_and_metrics):
        """Build an EEE-like record from (benchmark_name, metric_desc, score) tuples."""
        return {
            "evaluation_id": f"test/{model_id}",
            "model_info": {"id": model_id},
            "eval_library": {"name": "lm-evaluation-harness", "version": "0.4.2"},
            "evaluation_results": [
                {
                    "evaluation_name": bench,
                    "source_data": {"dataset_name": bench},
                    "metric_config": {"evaluation_description": metric_desc},
                    "score_details": {"score": score},
                }
                for bench, metric_desc, score in benchmarks_and_metrics
            ],
            "source_metadata": {},
        }

    def test_benchmarks_resolve_to_seeded_entities(self):
        """IFEval, BBH, GPQA, MMLU-PRO all resolve to correct canonical IDs."""
        store = _seeded_store()
        svc = ResolutionService(store)
        run_id = queries.start_sync_run(store, "test", False)

        record = self._make_eee_record("meta-llama/Llama-3.1-8B", [
            ("IFEval", "Accuracy on IFEval", 0.42),
            ("BBH", "Accuracy on BBH", 0.55),
            ("GPQA", "Accuracy on GPQA", 0.30),
            ("MMLU-PRO", "Accuracy on MMLU-PRO", 0.73),
        ])
        rows = process_record(record, "test", svc, run_id, rerun=False)

        by_bench = {r["benchmark_id"]: r for r in rows}
        assert "ifeval" in by_bench, f"Expected 'ifeval', got {list(by_bench.keys())}"
        assert "bbh" in by_bench
        assert "gpqa" in by_bench
        assert "mmlu-pro" in by_bench

    def test_metrics_resolve_after_stripping_qualifier(self):
        """'Accuracy on IFEval' → 'Accuracy' → canonical 'accuracy'."""
        store = _seeded_store()
        svc = ResolutionService(store)
        run_id = queries.start_sync_run(store, "test", False)

        record = self._make_eee_record("meta-llama/Llama-3.1-8B", [
            ("IFEval", "Accuracy on IFEval", 0.42),
            ("BBH", "Exact Match on BBH", 0.30),
        ])
        rows = process_record(record, "test", svc, run_id, rerun=False)

        ifeval_row = [r for r in rows if r["benchmark_id"] == "ifeval"][0]
        bbh_row = [r for r in rows if r["benchmark_id"] == "bbh"][0]
        assert ifeval_row["metric_id"] == "accuracy"
        assert bbh_row["metric_id"] == "exact-match"

    def test_harness_resolves_to_seeded_entity(self):
        """lm-evaluation-harness resolves to the seeded canonical ID."""
        store = _seeded_store()
        svc = ResolutionService(store)
        run_id = queries.start_sync_run(store, "test", False)

        record = self._make_eee_record("meta-llama/Llama-3.1-8B", [
            ("IFEval", "Accuracy on IFEval", 0.42),
        ])
        rows = process_record(record, "test", svc, run_id, rerun=False)
        assert rows[0]["harness_id"] == "lm-evaluation-harness"

    def test_model_uses_hf_id_directly(self):
        """Model IDs are HF paths — should become the canonical ID (slugified)."""
        store = _seeded_store()
        svc = ResolutionService(store)
        run_id = queries.start_sync_run(store, "test", False)

        record = self._make_eee_record("meta-llama/Llama-3.1-8B", [
            ("IFEval", "Accuracy on IFEval", 0.42),
        ])
        rows = process_record(record, "test", svc, run_id, rerun=False)
        # HF model IDs should be preserved (slash is kept by _slugify)
        assert "meta-llama" in rows[0]["model_id"].lower()

    def test_unknown_benchmark_creates_draft(self):
        """A benchmark not in seed data should auto-create a draft, not crash."""
        store = _seeded_store()
        svc = ResolutionService(store)
        run_id = queries.start_sync_run(store, "test", False)

        record = self._make_eee_record("meta-llama/Llama-3.1-8B", [
            ("SomeNewBenchmark2025", "Accuracy on SomeNewBenchmark2025", 0.99),
        ])
        rows = process_record(record, "test", svc, run_id, rerun=False)
        assert rows[0]["benchmark_id"] is not None
        # Should be a draft
        entity = queries.get_entity(store, "canonical_benchmarks", rows[0]["benchmark_id"])
        assert entity is not None
        assert entity["review_status"] == "draft"

    def test_em_alias_resolves_to_exact_match(self):
        """EM (common abbreviation) should resolve to exact-match via seed alias."""
        store = _seeded_store()
        svc = ResolutionService(store)
        run_id = queries.start_sync_run(store, "test", False)

        record = self._make_eee_record("meta-llama/Llama-3.1-8B", [
            ("IFEval", "EM on IFEval", 0.42),
        ])
        rows = process_record(record, "test", svc, run_id, rerun=False)
        assert rows[0]["metric_id"] == "exact-match"

    def test_helm_benchmarks_resolve(self):
        """HELM config benchmarks: MMLU, NarrativeQA, GSM8K should resolve."""
        store = _seeded_store()
        svc = ResolutionService(store)
        run_id = queries.start_sync_run(store, "test", False)

        record = self._make_eee_record("meta-llama/Llama-3.1-8B", [
            ("MMLU", "Accuracy on MMLU", 0.85),
            ("NarrativeQA", "F1 on NarrativeQA", 0.60),
            ("GSM8K", "EM on GSM8K", 0.70),
        ])
        rows = process_record(record, "test", svc, run_id, rerun=False)
        by_bench = {r["benchmark_id"]: r for r in rows}
        assert "mmlu" in by_bench
        assert "narrativeqa" in by_bench
        assert "gsm8k" in by_bench
        # Check metrics too
        assert by_bench["mmlu"]["metric_id"] == "accuracy"
        assert by_bench["narrativeqa"]["metric_id"] == "f1"
        assert by_bench["gsm8k"]["metric_id"] == "exact-match"

    def test_rewardbench_subcategories_resolve(self):
        """RewardBench sub-benchmarks: Chat, Chat Hard, Safety, Reasoning."""
        store = _seeded_store()
        svc = ResolutionService(store)
        run_id = queries.start_sync_run(store, "test", False)

        record = {
            "evaluation_id": "test/rewardbench",
            "model_info": {"id": "meta-llama/Llama-3.1-8B"},
            "eval_library": {"name": "rewardbench"},
            "evaluation_results": [
                {"evaluation_name": "Chat", "source_data": {"dataset_name": "RewardBench"},
                 "metric_config": {"evaluation_description": "Overall RewardBench Score"}, "score": 0.9},
                {"evaluation_name": "Chat Hard", "source_data": {"dataset_name": "RewardBench"},
                 "metric_config": {"evaluation_description": "Overall RewardBench Score"}, "score": 0.7},
                {"evaluation_name": "Safety", "source_data": {"dataset_name": "RewardBench"},
                 "metric_config": {"evaluation_description": "Overall RewardBench Score"}, "score": 0.95},
                {"evaluation_name": "Reasoning", "source_data": {"dataset_name": "RewardBench"},
                 "metric_config": {"evaluation_description": "Overall RewardBench Score"}, "score": 0.8},
            ],
            "source_metadata": {},
        }
        rows = process_record(record, "test", svc, run_id, rerun=False)
        by_bench = {r["benchmark_id"]: r for r in rows}
        assert "rewardbench-chat" in by_bench
        assert "rewardbench-chat-hard" in by_bench
        assert "rewardbench-safety" in by_bench
        assert "rewardbench-reasoning" in by_bench
        # All should have RewardBench as parent
        for row in rows:
            assert row["parent_benchmark_id"] == "rewardbench"

    def test_musr_uppercase_alias_resolves(self):
        """MUSR (uppercase from EEE data) should resolve to musr via alias."""
        store = _seeded_store()
        svc = ResolutionService(store)
        run_id = queries.start_sync_run(store, "test", False)

        record = self._make_eee_record("meta-llama/Llama-3.1-8B", [
            ("MUSR", "Accuracy on MUSR", 0.45),
        ])
        rows = process_record(record, "test", svc, run_id, rerun=False)
        assert rows[0]["benchmark_id"] == "musr"

    def test_f1_metric_resolves(self):
        """F1 metric (HELM uses bare 'F1') should resolve to f1."""
        store = _seeded_store()
        svc = ResolutionService(store)
        run_id = queries.start_sync_run(store, "test", False)

        record = self._make_eee_record("meta-llama/Llama-3.1-8B", [
            ("NarrativeQA", "F1", 0.60),
        ])
        rows = process_record(record, "test", svc, run_id, rerun=False)
        assert rows[0]["metric_id"] == "f1"

    def test_harness_helm_resolves(self):
        """HELM harness should resolve to seeded entity."""
        store = _seeded_store()
        svc = ResolutionService(store)
        run_id = queries.start_sync_run(store, "test", False)

        record = {
            "evaluation_id": "test/helm",
            "model_info": {"id": "meta-llama/Llama-3.1-8B"},
            "eval_library": {"name": "helm"},
            "evaluation_results": [
                {"evaluation_name": "MMLU", "source_data": {"dataset_name": "MMLU"},
                 "metric_config": {"evaluation_description": "Accuracy"}, "score": 0.85},
            ],
            "source_metadata": {},
        }
        rows = process_record(record, "test", svc, run_id, rerun=False)
        assert rows[0]["harness_id"] == "helm"

    def test_multiple_models_produce_distinct_results(self):
        """Two different models on the same benchmark produce rows with different model_ids."""
        store = _seeded_store()
        svc = ResolutionService(store)
        run_id = queries.start_sync_run(store, "test", False)

        for model_id in ["meta-llama/Llama-3.1-8B", "google/gemma-2-9b"]:
            record = self._make_eee_record(model_id, [
                ("IFEval", "Accuracy on IFEval", 0.42),
            ])
            process_record(record, "test", svc, run_id, rerun=False)

        # Flush pending entity writes so they appear in tables
        queries.flush_pending(store)

        # Check model entities were created
        models = store.table("canonical_models")
        model_ids = set(models["id"].tolist())
        assert any("llama" in mid.lower() for mid in model_ids)
        assert any("gemma" in mid.lower() for mid in model_ids)
