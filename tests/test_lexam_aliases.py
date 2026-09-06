"""LEXam leaderboard slugs resolve to the intended canonicals.

Every raw form here is one every_eval_ever's `lexam` adapter emits (dotted
`evaluation_name`s, metric names, the harness name, and the 36 leaderboard
model labels that this PR bridges). A regression fragments the LEXam results
across two ids. Skips if fixtures aren't built (run
`eval-card-registry seed --local` first).
"""
import json
from pathlib import Path

import pandas as pd
import pytest
from eval_entity_resolver import Resolver

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"

# Adapter `evaluation_name` (and the leaderboard's own labels) -> benchmark id.
BENCHMARKS = {
    "lexam": "lexam",
    "LEXam": "lexam",
    "lexam.open_question": "lexam-open-question",
    "lexam.mcq_4_choices": "lexam-mcq-4-choices",
    "Judge Scores on Open Questions": "lexam-open-question",
    "Accuracy on Multiple-Choice Questions": "lexam-mcq-4-choices",
}

# Adapter `metric_name` / `metric_id` -> canonical metric id.
METRICS = {
    "Multiple-Choice Accuracy": "accuracy",
    "Accuracy on Multiple-Choice Questions": "accuracy",
    "Open Question Judge Score": "lexam-open-question-judge-score",
    "Judge Scores on Open Questions": "lexam-open-question-judge-score",
    "lexam-open-question-judge-score": "lexam-open-question-judge-score",
}

HARNESSES = {
    "lighteval": "lighteval",
    "Lighteval": "lighteval",
    "hf-lighteval": "lighteval",
}

# Leaderboard model labels the adapter emits that needed a seed bridge.
MODELS = {
    "DeepSeek-V3.2-chat": "deepseek-ai/DeepSeek-V3.2",
    "DeepSeek-V3.2-reasoner": "deepseek-ai/DeepSeek-V3.2",
    "DeepSeek-V3.2-Exp": "deepseek/deepseek-v3.2-exp",
    "Llama-3.1-8B-it": "meta-llama/Llama-3.1-8B-Instruct",
    "Llama-3.3-70B-it": "meta-llama/Llama-3.3-70B-Instruct",
    "Llama-3.1-405B-it": "meta/llama-3-1-405b-instruct",
    "Qwen-2.5-7B-it": "Qwen/Qwen2.5-7B-Instruct",
    "EuroLLM-9B-it": "utter-project/EuroLLM-9B-Instruct",
    "EuroLLM-9B-Instruct": "utter-project/EuroLLM-9B-Instruct",
    "utter-project/EuroLLM-9B-Instruct": "utter-project/EuroLLM-9B-Instruct",
    # the base must stay distinct from the instruct checkpoint
    "utter-project/EuroLLM-9B": "utter-project/EuroLLM-9B",
}

FAMILY_MEMBERS = {"lexam", "lexam-open-question", "lexam-mcq-4-choices"}

pytestmark = pytest.mark.skipif(
    not (_FIXTURES / "aliases.parquet").exists(),
    reason="fixtures not built; run `eval-card-registry seed --local`",
)


@pytest.fixture(scope="module")
def resolver():
    return Resolver.from_parquet(str(_FIXTURES))


@pytest.fixture(scope="module")
def benchmarks_df():
    return pd.read_parquet(_FIXTURES / "canonical_benchmarks.parquet")


@pytest.fixture(scope="module")
def families_df():
    return pd.read_parquet(_FIXTURES / "canonical_families.parquet")


@pytest.mark.parametrize("raw,expected", BENCHMARKS.items())
def test_benchmark_resolves(resolver, raw, expected):
    got = resolver.resolve(raw, entity_type="benchmark").canonical_id
    assert got == expected, f"{raw} -> {got} (expected {expected})"


@pytest.mark.parametrize("raw,expected", METRICS.items())
def test_metric_resolves(resolver, raw, expected):
    got = resolver.resolve(raw, entity_type="metric").canonical_id
    assert got == expected, f"{raw} -> {got} (expected {expected})"


@pytest.mark.parametrize("raw,expected", HARNESSES.items())
def test_harness_resolves(resolver, raw, expected):
    got = resolver.resolve(raw, entity_type="harness").canonical_id
    assert got == expected, f"{raw} -> {got} (expected {expected})"


@pytest.mark.parametrize("raw,expected", MODELS.items())
def test_model_label_resolves(resolver, raw, expected):
    got = resolver.resolve(raw, entity_type="model").canonical_id
    assert got == expected, f"{raw} -> {got} (expected {expected})"


def test_lexam_is_a_family(families_df):
    row = families_df[families_df["id"] == "lexam"]
    assert len(row) == 1, "lexam missing from canonical_families"
    assert set(json.loads(row.iloc[0]["benchmark_ids"])) == FAMILY_MEMBERS


@pytest.mark.parametrize("member", sorted(FAMILY_MEMBERS))
def test_sub_tasks_are_standalone(benchmarks_df, member):
    """The two sub-tasks are separately reported datasets (own HF config, own
    n, own metric on its own scale), so neither may carry a parent edge: a
    parented benchmark is a slice to the producer — no merged page of its own,
    and its rows drop out of merged_evals_view once the parent has top-level
    data."""
    row = benchmarks_df[benchmarks_df["id"] == member]
    assert len(row) == 1, f"{member} missing from canonical_benchmarks"
    parent = row.iloc[0]["parent_benchmark_id"]
    assert parent is None or pd.isna(parent), f"{member} has parent_benchmark_id={parent!r}"


def test_bare_hf_config_names_are_not_global_aliases(resolver):
    """`open_question` / `mcq_4_choices` are generic HF config names that mean
    different things in different sources; the adapter emits only the dotted
    forms, so neither may claim a global benchmark alias."""
    for raw in ("open_question", "mcq_4_choices"):
        assert resolver.resolve(raw, entity_type="benchmark").canonical_id is None, raw


def test_judge_score_scale_is_zero_to_one_hundred():
    """The adapter publishes the judge score on the leaderboard's 0-100 scale
    (percent_divisor=1.0); accuracy stays a 0-1 proportion. Folding the two
    scales together would silently rescale one of them."""
    metrics = pd.read_parquet(_FIXTURES / "canonical_metrics.parquet")
    judge = metrics[metrics["id"] == "lexam-open-question-judge-score"].iloc[0]
    assert (judge["min_score"], judge["max_score"]) == (0.0, 100.0)
    acc = metrics[metrics["id"] == "accuracy"].iloc[0]
    assert (acc["min_score"], acc["max_score"]) == (0.0, 1.0)
