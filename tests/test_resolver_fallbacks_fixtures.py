"""The three additive fallbacks against the REAL seed (built fixtures).

A — a datastore folder that is itself a registered benchmark is the fallback
identity for the dotted names under it, once nothing else in the name resolves.
B — the whole namespaced `metric_id` may itself be a registered alias.
C — a versioned harness string resolves via its bare name.

Skips if fixtures aren't built (run `eval-card-registry seed --local` first).
"""
from pathlib import Path

import pytest
from eval_entity_resolver import Resolver

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"

CATCH_ALL = frozenset({"score", "mean-score", "overall"})

pytestmark = pytest.mark.skipif(
    not (_FIXTURES / "aliases.parquet").exists(),
    reason="fixtures not built; run `eval-card-registry seed --local`",
)


@pytest.fixture(scope="module")
def resolver():
    return Resolver.from_parquet(str(_FIXTURES))


# --- C: harness version strip ---------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("lm-evaluation-harness 0.4.0", "lm-evaluation-harness"),
    ("inspect_ai inspect_ai:0.3.75", "inspect-ai"),
    ("inspect_ai 0.3.142", "inspect-ai"),
    # AlpacaEval 1 and 2 are different judging protocols but the registry has
    # one canonical; `harness_raw` keeps the version and an explicit
    # per-version alias would win before the strip.
    ("alpaca_eval 2.0", "alpaca-eval"),
    ("lm_eval 0.4.12", "lm-evaluation-harness"),
    ("helm 1.2.3", "helm"),
    # no digits / dotless / comma multi-library tails stay unmatched
    ("helm unknown", None),
    ("BFCL v4", None),
    ("kaggle kernel 4", None),
])
def test_harness_version_strip(resolver, raw, expected):
    assert resolver.resolve(raw, "harness").canonical_id == expected


def test_harness_strip_is_resolve_mode_only(resolver):
    assert resolver.resolve("helm 1.2.3", "harness", mode="exact").canonical_id is None


# --- A: benchmark folder fallback -----------------------------------------

def test_cocoabench_names_report_the_folder(resolver):
    for name in ("cocoabench.overall.accuracy_percent",
                 "cocoabench.overall.avg_time_seconds"):
        match = resolver.resolve_structured_benchmark(name, "cocoabench")
        assert (match.canonical_id, match.benchmark_raw, match.subset) == (
            "cocoabench", "cocoabench", None,
        ), name


@pytest.mark.parametrize("name,canonical,raw", [
    ("lexam.open_question", "lexam-open-question", "open question"),
    ("lexam.mcq_4_choices", "lexam-mcq-4-choices", "mcq 4 choices"),
])
def test_lexam_sub_tasks_hit_the_first_pass(resolver, name, canonical, raw):
    """The scoped slice aliases make these FIRST-pass hits: the folder
    fallback never runs for lexam, and `benchmark_raw` is the slice spelling
    (not the folder-kept one), which is what lets `slice_promotion` promote
    the bucket instead of overriding the sub-task back to the parent."""
    match = resolver.resolve_structured_benchmark(name, "lexam")
    assert (match.canonical_id, match.benchmark_raw, match.subset) == (canonical, raw, None)


def test_registered_child_still_wins(resolver):
    match = resolver.resolve_structured_benchmark("llm_stats.mcp-atlas", "llm-stats")
    assert match.canonical_id == "mcp-atlas"


def test_unregistered_child_becomes_a_slice_of_the_aggregator(resolver):
    match = resolver.resolve_structured_benchmark("llm_stats.deepswe-1.1", "llm-stats")
    assert (match.canonical_id, match.benchmark_raw, match.subset) == (
        "llm-stats", "llm stats deepswe-1", "deepswe-1",
    )


def test_deeper_hit_still_outranks_the_folder(resolver):
    match = resolver.resolve_structured_benchmark("vals_ai.mmlu_pro.biology", "vals-ai")
    assert (match.canonical_id, match.subset) == ("mmlu-pro", "biology")


def test_unregistered_child_under_a_registered_folder(resolver):
    match = resolver.resolve_structured_benchmark("vals_ai.programbench.strict", "vals-ai")
    assert match.canonical_id == "vals-ai"
    assert match.subset == "programbench strict"


@pytest.mark.parametrize("name,folder", [
    ("benchpress.aa-briefcase-elo", "benchpress"),
    ("paperswithcode.image_restoration.cbsd68_color_gaussian_denoising_sigma_15",
     "paperswithcode"),
    ("paperswithcode.depth_estimation.nyuv2_relative", "paperswithcode"),
])
def test_unregistered_aggregator_folders_stay_unresolved(resolver, name, folder):
    """SEED-REVIEW PIN. Neither aggregator is a registered benchmark, so the
    folder fallback finds nothing. Seeding `benchpress` or `paperswithcode` as
    a benchmark would make it the fallback identity for 238 / 64 distinct
    dotted names (one slice each) — flip this expectation only deliberately,
    and re-run the census when you do.

    The paperswithcode names are chosen so that this test fails for exactly
    that reason and no other: `paperswithcode.<category>.<child>` stops at the
    category segment, so the pin turns on the category resolving, not the
    child (`paperswithcode.agents.browsecomp` is unresolved even though
    `browsecomp` IS a registered benchmark; `math` and `reasoning` are
    categories that already resolve). `image_restoration` /
    `depth_estimation` and their children are computer-vision task
    configurations, not plausible registrations in an LLM-eval registry, and
    two pins are kept so one incidental seed cannot silently retire the
    check."""
    assert resolver.resolve_structured_benchmark(name, folder) is None


# --- B: whole metric id ----------------------------------------------------

@pytest.mark.parametrize("raw_id,source_config,expected", [
    ("vals_ai.mgsm.mgsm_de.accuracy", "vals-ai", "accuracy"),
    ("llm_stats.gdpval-aa.score", "llm-stats", None),
    ("cocoabench.overall.avg_time_seconds", "cocoabench", "mean-task-time"),
])
def test_structured_metric_ids(resolver, raw_id, source_config, expected):
    assert resolver.resolve_structured_metric_id(
        raw_id, source_config, catch_all_ids=CATCH_ALL
    ) == expected
