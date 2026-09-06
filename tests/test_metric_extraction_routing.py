"""The producer reaches a metric through `extract_metric` when a record's
structured metric_id misses (COALESCE(evaluation_description, metric_name,
evaluation_name) -> extract_metric_udf -> resolve). A display name or alias
whose extraction resolves to a DIFFERENT canonical is a silent mis-merge on
that path even though `Resolver.resolve(raw)` is right. This pins every
surface form the harness-metric batch added; the pre-existing tail of such
mis-routes (macro-accuracy, median-win-rate, total-cost, ...) and the
suffix-order spellings the extractor still cannot see ("Win Rate (LC)",
"Exact match (quasi)") are tracked separately and not asserted here.
"""
from pathlib import Path

import pytest
import yaml
from eval_entity_resolver import Resolver
from eval_entity_resolver.eee import extract_metric

_ROOT = Path(__file__).resolve().parent.parent
_FIXTURES = _ROOT / "fixtures"

PINNED = (
    "length-controlled-win-rate", "discrete-win-rate", "quasi-exact-match",
    "prefix-exact-match", "quasi-prefix-exact-match", "ifeval-strict-accuracy",
    "brier-score", "math-equivalent-chain-of-thought", "micro-f1", "macro-f1",
    "normalized-accuracy", "lexam-open-question-judge-score",
)

pytestmark = pytest.mark.skipif(
    not (_FIXTURES / "aliases.parquet").exists(),
    reason="fixtures not built; run `eval-card-registry seed --local`",
)


def _surface_forms():
    for entry in yaml.safe_load((_ROOT / "seed" / "metrics.yaml").read_text()):
        if entry["id"] in PINNED:
            for raw in [entry.get("display_name"), *(entry.get("aliases") or [])]:
                if raw:
                    yield entry["id"], raw


@pytest.fixture(scope="module")
def resolver():
    return Resolver.from_parquet(str(_FIXTURES))


@pytest.mark.parametrize("canonical,raw", sorted(_surface_forms()))
def test_extraction_lands_on_its_own_canonical(resolver, canonical, raw):
    extracted = extract_metric(raw)
    got = resolver.resolve(extracted, entity_type="metric").canonical_id if extracted else None
    assert got == canonical, f"{raw!r} -> extract {extracted!r} -> {got!r}, expected {canonical!r}"
