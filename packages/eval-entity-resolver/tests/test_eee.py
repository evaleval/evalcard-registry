"""Tests for EEE-specific preprocessing: metric extraction and benchmark name cleaning."""
from eval_entity_resolver.eee import (
    clean_eval_name,
    extract_metric,
    prepare_eval_name_segments,
)


class TestExtractMetric:
    # --- "X on Y" pattern ---

    def test_strips_on_suffix(self):
        assert extract_metric("Accuracy on IFEval") == "Accuracy"

    def test_strips_on_suffix_multiword(self):
        assert extract_metric("Exact Match on MATH Level 5") == "Exact Match"

    def test_strips_em_abbreviation(self):
        assert extract_metric("EM on GSM8K") == "EM"

    # --- Short metric names pass through unchanged ---

    def test_preserves_bare_metric(self):
        assert extract_metric("Accuracy") == "Accuracy"

    def test_preserves_empty(self):
        assert extract_metric("") == ""

    def test_preserves_short_name(self):
        assert extract_metric("F1") == "F1"

    def test_preserves_two_word_name(self):
        assert extract_metric("Win Rate") == "Win Rate"

    # --- Dot notation ---

    def test_dot_notation_extracts_accuracy(self):
        assert extract_metric("bfcl.live.live_accuracy") == "Accuracy"

    def test_dot_notation_extracts_ast_accuracy(self):
        assert extract_metric("bfcl.non_live.simple_ast_accuracy") == "AST Accuracy"

    def test_ast_accuracy_distinct_from_accuracy(self):
        """AST accuracy (function call AST matching) is a different metric from plain accuracy."""
        assert extract_metric("Non-live simple AST accuracy") == "AST Accuracy"
        assert extract_metric("Live accuracy") == "Accuracy"

    def test_class_averaged_f1_distinct_from_f1(self):
        """Macro and micro F1 average over classes differently, and neither is F1.

        The generic ``\\bf1\\b`` pattern matches inside both, so without their own
        patterns the registry's macro-f1 and micro-f1 entries are unreachable from
        an EEE metric name.
        """
        assert extract_metric("Macro F1") == "Macro F1"
        assert extract_metric("Micro F1") == "Micro F1"
        assert extract_metric("Tokenized F1") == "F1"

    def test_dot_notation_extracts_win_rate(self):
        assert extract_metric("fibble1_arena.win_rate") == "Win Rate"

    def test_dot_notation_extracts_rank(self):
        assert extract_metric("bfcl.overall.rank") == "rank"

    def test_dot_notation_extracts_cost(self):
        assert extract_metric("bfcl.overall.total_cost_usd") == "cost"

    def test_dot_notation_extracts_latency(self):
        assert extract_metric("bfcl.overall.latency_mean_s") == "mean-latency"

    def test_dot_notation_extracts_stddev(self):
        assert extract_metric("bfcl.format_sensitivity.stddev") == "stddev"

    # --- Verbose descriptions → keyword extraction ---

    def test_description_extracts_keyword(self):
        assert extract_metric("Chat accuracy - includes easy chat subsets") == "Accuracy"

    def test_description_extracts_multiword_keyword(self):
        assert extract_metric("Corporate lawyer world mean score.") == "Mean Score"

    def test_no_keyword_description_falls_back_to_score(self):
        assert extract_metric("Global MMLU Lite - Arabic") == "score"

    def test_normalized_accuracy_not_swallowed_by_generic_accuracy(self):
        assert extract_metric("Normalized accuracy") == "Normalized Accuracy"
        assert extract_metric("Normalised accuracy on HellaSwag") == "Normalized Accuracy"

    # --- pass@N family ---

    def test_pass_at_1_extracted(self):
        assert extract_metric("pass@1 (filter: create_test)") == "Pass@1"

    def test_pass_at_10_not_truncated_to_pass_at_1(self):
        assert extract_metric("pass@10 (filter: create_test)") == "Pass@10"

    def test_uncovered_pass_at_n_does_not_fall_back_to_pass_at_1(self):
        assert extract_metric("pass@100 (filter: create_test)") != "Pass@1"
        assert extract_metric("pass@16 (filter: create_test)") != "Pass@1"

    def test_underscore_pass_at_1_extracted(self):
        assert extract_metric("pass_at_1 (filter: extract_code)") == "Pass@1"

    def test_underscore_pass_at_10_extracted(self):
        assert extract_metric("pass_at_10 (filter: extract_code)") == "Pass@10"

    def test_first_keyword_wins_by_position(self):
        # "score" appears before "accuracy" in this description
        assert extract_metric("Factuality score - measures factual accuracy") == "score"


class TestCleanEvalName:
    # --- Dot notation (last segment is metric, rest is benchmark) ---

    def test_dot_drops_last_segment(self):
        assert clean_eval_name("bfcl.overall.rank") == "bfcl overall"

    def test_dot_two_segment_benchmark(self):
        assert clean_eval_name("bfcl.overall.overall_accuracy") == "bfcl overall"

    def test_dot_live_category(self):
        assert clean_eval_name("bfcl.live.live_accuracy") == "bfcl live"

    def test_dot_non_live_category(self):
        assert clean_eval_name("bfcl.non_live.simple_ast_accuracy") == "bfcl non live"

    # --- Underscore metric suffix (fibble/wordle) ---

    def test_underscore_strips_win_rate(self):
        assert clean_eval_name("fibble1_arena_win_rate") == "fibble1 arena"

    def test_underscore_strips_avg_attempts(self):
        assert clean_eval_name("wordle_arena_avg_attempts") == "wordle arena"

    def test_underscore_strips_elo(self):
        assert clean_eval_name("overall_elo") == "overall"

    # --- Trailing metric words (ACE/APEX) ---

    def test_trailing_score(self):
        assert clean_eval_name("Gaming Score") == "Gaming"

    def test_trailing_pass_at_1(self):
        assert clean_eval_name("Investment Banking Pass@1") == "Investment Banking"

    def test_trailing_mean_score(self):
        assert clean_eval_name("Corporate Lawyer Mean Score") == "Corporate Lawyer"

    # --- Clean names pass through ---

    def test_clean_name_unchanged(self):
        assert clean_eval_name("IFEval") == "IFEval"

    def test_multi_word_clean_unchanged(self):
        assert clean_eval_name("Chat Hard") == "Chat Hard"

    def test_empty(self):
        assert clean_eval_name("") == ""


class TestMidSentenceOnTruncation:
    """The "X on Y" pattern must not truncate prose sentences — keywords
    disclosed after a mid-sentence " on " were previously never seen."""

    def test_prose_with_mid_sentence_on_keeps_late_keyword(self):
        desc = (
            "GDPval-AA evaluates AI agents on economically valuable "
            "professional knowledge-work tasks and reports performance "
            "as an Elo score."
        )
        assert extract_metric(desc) == "Elo Rating"

    def test_short_metric_on_benchmark_still_truncates(self):
        assert extract_metric("Accuracy on IFEval") == "Accuracy"
        assert (
            extract_metric("Exact match accuracy on mmlu_clinical_knowledge_af (5-shot)")
            == "Exact Match"
        )


class TestPrepareEvalNameSegments:
    """Registry-free preparation of a dotted evaluation_name."""

    def test_collapses_identical_adjacent_segments(self):
        assert prepare_eval_name_segments("bbq.bbq.overall") == ["bbq"]

    def test_drops_terminal_aggregate_marker(self):
        assert prepare_eval_name_segments("MMLU.MMLU-Pro.overall") == ["MMLU", "MMLU-Pro"]

    def test_keeps_distinct_segments(self):
        assert prepare_eval_name_segments("vals_ai.mmlu_pro.biology") == [
            "vals_ai", "mmlu_pro", "biology",
        ]

    def test_drops_numeric_version_fragment(self):
        assert prepare_eval_name_segments("terminal-bench-2.0") == ["terminal-bench-2"]

    def test_bare_and_spaced_names_are_not_split(self):
        assert prepare_eval_name_segments("gsm8k") is None
        assert prepare_eval_name_segments("MMLU-Pro (Biology)") is None
        assert prepare_eval_name_segments(None) is None
