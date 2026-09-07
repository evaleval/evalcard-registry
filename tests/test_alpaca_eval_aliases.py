"""AlpacaEval leaderboard slugs resolve to the intended canonicals.

Every raw string below is one the every_eval_ever AlpacaEval converter
(`every_eval_ever/converters/alpaca_eval/adapter.py`) hands to the registry.
A miss there is not a resolver error — the converter falls back to a local
`alpaca_eval.*` id, which fragments AlpacaEval results across two ids in the
merged view. Skips if fixtures aren't built (run `eval-card-registry seed
--local` first).
"""
from pathlib import Path

import pandas as pd
import pytest
from eval_entity_resolver import Resolver

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"

# The converter's `benchmark_query` per leaderboard, plus the seeded aliases.
BENCHMARKS = {
    "AlpacaEval 1.0": "alpacaeval-1-0",
    "AlpacaEval 2.0": "alpacaeval-2-0",
    "AlpacaEval v1": "alpacaeval-1-0",
    "AlpacaEval 1": "alpacaeval-1-0",
    "alpaca_eval_v1": "alpacaeval-1-0",
    "alpaca_eval_v2": "alpacaeval-2-0",
}

# The converter looks metrics up by the RAW leaderboard-CSV column name
# (`registry.metric(spec.column)`), not by its own `metric_name`. Both
# leaderboard CSVs carry all four columns, so both versions need all four.
METRICS = {
    "win_rate": "win-rate",
    "length_controlled_winrate": "length-controlled-win-rate",
    "discrete_win_rate": "discrete-win-rate",
    "avg_length": "average-response-length",
    # the converter's own emitted metric_name spelling, and the label the
    # leaderboard page prints for the column
    "length_controlled_win_rate": "length-controlled-win-rate",
    "LC Win Rate": "length-controlled-win-rate",
}

pytestmark = pytest.mark.skipif(
    not (_FIXTURES / "aliases.parquet").exists(),
    reason="fixtures not built; run `eval-card-registry seed --local`",
)


@pytest.fixture(scope="module")
def resolver():
    return Resolver.from_parquet(str(_FIXTURES))


@pytest.fixture(scope="module")
def metrics_df():
    return pd.read_parquet(_FIXTURES / "canonical_metrics.parquet")


@pytest.mark.parametrize("slug,expected", BENCHMARKS.items())
def test_benchmark_resolves(resolver, slug, expected):
    res = resolver.resolve(slug, entity_type="benchmark")
    assert res.canonical_id == expected, f"{slug} -> {res.canonical_id} (expected {expected})"


@pytest.mark.parametrize("slug,expected", METRICS.items())
def test_metric_resolves(resolver, slug, expected):
    res = resolver.resolve(slug, entity_type="metric")
    assert res.canonical_id == expected, f"{slug} -> {res.canonical_id} (expected {expected})"


def test_harness_resolves(resolver):
    """`registry.harness('alpaca_eval')` — the converter's literal lookup key,
    reached from the `alpaca-eval` id by the normalized matcher."""
    assert resolver.resolve("alpaca_eval", entity_type="harness").canonical_id == "alpaca-eval"


def test_bare_alpacaeval_is_not_a_benchmark_alias(resolver):
    """Bare `AlpacaEval` must stay unresolved: it names the project, and
    binding it to either leaderboard would silently mix two score scales
    (different reference model and judge, not comparable)."""
    for raw in ("AlpacaEval", "alpaca_eval"):
        got = resolver.resolve(raw, entity_type="benchmark").canonical_id
        assert got is None, f"{raw} -> {got}"


@pytest.mark.parametrize(
    "metric_id,lower_is_better,min_score,max_score",
    [
        # All three win-rate columns are `mean * 100` in upstream
        # (src/alpaca_eval/metrics/{helpers,glm_winrate}.py); the
        # length-controlled one stays inside [0,100] because the GLM
        # prediction passes through a logistic before scaling.
        ("win-rate", False, 0.0, 100.0),
        ("length-controlled-win-rate", False, 0.0, 100.0),
        ("discrete-win-rate", False, 0.0, 100.0),
        # `avg_length` is int(model_outputs["output"].str.len().mean()) —
        # unbounded characters (`.inf` in the seed: unbounded by definition,
        # not "not stated"), and not a quality score in either direction.
        ("average-response-length", None, 0.0, float("inf")),
    ],
)
def test_metric_bounds(metrics_df, metric_id, lower_is_better, min_score, max_score):
    row = metrics_df[metrics_df["id"] == metric_id]
    assert len(row) == 1, f"{metric_id}: expected exactly one row, got {len(row)}"
    row = row.iloc[0]
    assert row["lower_is_better"] is lower_is_better or (
        lower_is_better is None and pd.isna(row["lower_is_better"])
    ), f"{metric_id}: lower_is_better={row['lower_is_better']!r}"
    assert row["min_score"] == min_score
    if max_score is None:
        assert pd.isna(row["max_score"]), f"{metric_id}: max_score={row['max_score']!r}"
    else:
        assert row["max_score"] == max_score, f"{metric_id}: max_score={row['max_score']!r}"
