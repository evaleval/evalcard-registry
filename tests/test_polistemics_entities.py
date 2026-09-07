"""Polistemics canonicals (benchmark / harness / metric) resolve exactly.

Guards the entities minted for the Polistemics EEE_datastore submission
(arXiv:2607.25953): the raw ids its records carry must resolve to the seeded
canonicals, and the generic surface form "Rubric Score" must NOT resolve to
the namespaced metric (display_name is benchmark-scoped precisely to avoid
claiming other sources' raw rubric-score fields). Skips if fixtures aren't
built (run `eval-card-registry seed --local` first).
"""
from pathlib import Path

import pytest
from eval_entity_resolver import Resolver

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FIXTURES = _REPO_ROOT / "fixtures"

# Raw string carried by the EEE records -> (entity_type, expected canonical).
ENTITY_CANONICALS = {
    "polistemics": [("benchmark", "polistemics"), ("harness", "polistemics")],
    "polistemics.adherence": [("metric", "polistemics.adherence")],
}

pytestmark = pytest.mark.skipif(
    not (_FIXTURES / "aliases.parquet").exists(),
    reason="fixtures not built; run `eval-card-registry seed --local`",
)


@pytest.fixture(scope="module")
def resolver():
    return Resolver.from_parquet(str(_FIXTURES))


@pytest.mark.parametrize(
    "raw,entity_type,expected",
    [(raw, et, cid) for raw, pairs in ENTITY_CANONICALS.items() for et, cid in pairs],
)
def test_polistemics_entity_resolves(resolver, raw, entity_type, expected):
    res = resolver.resolve(raw, entity_type=entity_type)
    assert res.canonical_id == expected, (
        f"{raw!r} ({entity_type}) -> {res.canonical_id!r} (expected {expected!r})"
    )


def test_generic_rubric_score_does_not_hijack(resolver):
    # Bare "Rubric Score" / "rubric_score" must stay unclaimed: other sources'
    # raw rubric-score fields are NOT polistemics.adherence.
    for raw in ("Rubric Score", "rubric_score"):
        res = resolver.resolve(raw, entity_type="metric")
        assert res.canonical_id != "polistemics.adherence", (
            f"generic form {raw!r} resolved to polistemics.adherence — "
            "display_name must stay benchmark-scoped"
        )
