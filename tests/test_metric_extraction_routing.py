"""Every metric surface form survives the producer's metric-resolution path.

The producer (eval_cards_backend_pipeline stage C) reaches a metric, when a
record's structured metric_id misses, by trying the raw metric_name as an
alias first, skipping the registry's catch-all ids (`metadata.catch_all`),
and only then running `extract_metric` on the description-like text. A
display name or alias that lands on a DIFFERENT canonical under that order
is a silent mis-merge even though `Resolver.resolve(raw)` is right. This
asserts the whole seed, not a sample; the extraction-only leg is pinned for
the compound names whose tail spells a generic metric, so the keyword
extractor cannot regress to swallowing them.
"""
import json
from pathlib import Path

import pytest
import yaml
from eval_entity_resolver import Resolver
from eval_entity_resolver.eee import extract_metric

_ROOT = Path(__file__).resolve().parent.parent
_FIXTURES = _ROOT / "fixtures"

# Metrics whose surface forms contain a generic keyword and whose extraction
# alone (no direct alias hit) must still land on them.
EXTRACTION_PINNED = (
    "length-controlled-win-rate", "discrete-win-rate", "quasi-exact-match",
    "prefix-exact-match", "quasi-prefix-exact-match", "ifeval-strict-accuracy",
    "brier-score", "math-equivalent-chain-of-thought", "micro-f1", "macro-f1",
    "normalized-accuracy", "lexam-open-question-judge-score",
)

pytestmark = pytest.mark.skipif(
    not (_FIXTURES / "aliases.parquet").exists(),
    reason="fixtures not built; run `eval-card-registry seed --local`",
)


def _entries():
    return yaml.safe_load((_ROOT / "seed" / "metrics.yaml").read_text())


def _catch_all_ids():
    return {e["id"] for e in _entries()
            if json.loads(e.get("metadata") or "{}").get("catch_all")}


def _surface_forms(only=None):
    for entry in _entries():
        if only is None or entry["id"] in only:
            for raw in [entry.get("display_name"), *(entry.get("aliases") or [])]:
                if raw:
                    yield entry["id"], raw


@pytest.fixture(scope="module")
def resolver():
    return Resolver.from_parquet(str(_FIXTURES))


@pytest.fixture(scope="module")
def catch_all():
    return _catch_all_ids()


def _producer_order(resolver, catch_all, raw):
    direct = resolver.resolve(raw, entity_type="metric").canonical_id
    if direct and direct not in catch_all:
        return direct
    extracted = extract_metric(raw)
    return resolver.resolve(extracted, entity_type="metric").canonical_id if extracted else None


@pytest.mark.parametrize("canonical,raw", sorted(_surface_forms()))
def test_producer_order_lands_on_its_own_canonical(resolver, catch_all, canonical, raw):
    got = _producer_order(resolver, catch_all, raw)
    assert got == canonical, f"{raw!r} -> {got!r} on the producer path, expected {canonical!r}"


@pytest.mark.parametrize("canonical,raw", sorted(_surface_forms(set(EXTRACTION_PINNED))))
def test_extraction_alone_lands_on_its_own_canonical(resolver, canonical, raw):
    extracted = extract_metric(raw)
    got = resolver.resolve(extracted, entity_type="metric").canonical_id if extracted else None
    assert got == canonical, f"{raw!r} -> extract {extracted!r} -> {got!r}, expected {canonical!r}"
