"""Every metric surface form survives the producer's metric-resolution path.

The producer (eval_cards_backend_pipeline stage C) resolves a record's
metric, when the structured metric_id misses, by letting the record's own
metric_name decide when it resolves to a non-catch-all metric (catch-alls
are `metadata.catch_all` in the seed) and the description adds nothing
more specific (`metric_name_wins`), else by running `extract_metric` over
the description-like text. A display name or alias that lands on a
DIFFERENT canonical under that order is a silent mis-merge even though
`Resolver.resolve(raw)` is right. This models the description-absent case
a bare surface form is: no structured metric_id, no description, no
source_config; for that case the producer rule reduces to "direct
non-catch-all hit, else extraction". The extraction-only leg is gated by an
inventory of the forms it is known to mis-route, kept exact in both
directions, so a new compound name the keyword table does not know fails
instead of joining the tail silently, and a fix has to shrink the list.
"""
import json
from pathlib import Path

import pytest
import yaml
from eval_entity_resolver import Resolver
from eval_entity_resolver.eee import extract_metric

_ROOT = Path(__file__).resolve().parent.parent
_FIXTURES = _ROOT / "fixtures"

# Every metric surface form the keyword extractor alone still routes to a
# different canonical or to nothing, on the day this inventory was taken. The
# producer never relies on extraction alone for these (the direct alias leg
# takes them), but a NEW entry joining this set is a new hazard: the test
# below fails on any mis-route that is not listed here, and shrinking the
# list is the way to record a fix.
KNOWN_EXTRACTION_MISROUTES = frozenset({
    ("acc-at-0-25", "Acc@0.25"),
    ("action-space-generalization-mse", "Action Space Generalization MSE"),
    ("action-top-1-accuracy", "Action Top-1 Accuracy"),
    ("air-bench-chat-gpt-4-score", "AIR-Bench Chat GPT-4 Score"),
    ("arena-hard-score", "Arena-Hard Score"),
    ("average-rank", "Average Rank"),
    ("avg-attempts", "Average guesses used per game"),
    ("bd-rate-weighted-psnr-yuv420", "BD-Rate (Weighted PSNR YUV420)"),
    ("bootstrap-score", "Bootstrap Score"),
    ("bootstrap-score", "Bootstrap score"),
    ("codegolf.score", "codegolf.score"),
    ("compilation-success-rate-csr", "Compilation Success Rate (CSR)"),
    ("cvebench.mean", "cvebench.mean"),
    ("cyse2-vulnerability-exploit.mean", "cyse2-vulnerability-exploit.mean"),
    ("detection-auroc-only-logical", "Detection AUROC (only logical)"),
    ("detection-auroc-only-structural", "Detection AUROC (only structural)"),
    ("detection-f1-max", "Detection F1-max"),
    ("dpg-bench-score", "DPG-Bench Score"),
    ("dreamgenbench-average-score", "DreamGenBench Average Score"),
    ("drivelm-p1-3-gpt-score-chatgpt-3-5", "DriveLM P1-3 GPT Score (ChatGPT-3.5)"),
    ("driving-score", "Driving Score"),
    ("element-accuracy", "Element Accuracy"),
    ("f1-at-0-25", "F1@0.25"),
    ("f1-at-0-5", "F1@0.5"),
    ("final-acc", "Final Acc"),
    ("final-acc", "Final Accuracy"),
    ("final-score-fs", "Final Score (FS)"),
    ("framework-accuracy-fa", "Framework Accuracy (FA)"),
    ("generated-scene-consistency-mse", "Generated Scene Consistency MSE"),
    ("geneval-score", "GenEval Score"),
    ("healthbench-score", "HealthBench Score"),
    ("hpsv3", "HPSv3 Score"),
    ("human-normalized-score", "Mean Human Normalized Score"),
    ("imgedit-score", "ImgEdit Score"),
    ("inst-level-loose-accuracy", "Instruction-Level Loose Accuracy"),
    ("inst-level-strict-accuracy", "Instruction-Level Strict Accuracy"),
    ("judge-score-1-10", "Score (1-10)"),
    ("l2-bench.mean", "l2-bench.mean"),
    ("laion-aesthetic-score", "LAION Aesthetic Score"),
    ("lenient-accuracy", "Lenient Accuracy"),
    ("long-context-memory-mse", "Long Context Memory MSE"),
    ("longaudiobench-overall-judge-score", "LongAudioBench Overall Judge Score"),
    ("macro-accuracy", "Macro Accuracy"),
    ("median-human-normalized-score", "Median Human Normalized Score"),
    ("median-win-rate", "Median Win Rate"),
    ("mega-bench-macro-score", "Macro score"),
    ("mmau-pro-open-ended-judge-score", "MMAU-Pro Open-Ended Judge Score"),
    ("mmau-pro-overall-weighted-performance", "MMAU-Pro Overall Weighted Performance"),
    ("mmteb-score", "MMTEB Score"),
    ("mteb-score", "MTEB Score"),
    ("non-lin-score", "Non-Lin Score"),
    ("normalized-score", "Normalized Score"),
    ("noun-top-1-accuracy", "Noun Top-1 Accuracy"),
    ("ocrbench-v2-chinese-score", "Chinese Score"),
    ("ocrbench-v2-english-score", "English Score"),
    ("omnicontext-score", "OmniContext score"),
    ("oneig-overall-score", "OneIG Overall Score"),
    ("operation-f1", "Operation F1"),
    ("paired-accuracy", "Paired Accuracy"),
    ("pie-bench-edit-region-clip-similarity", "PIE-Bench Edit Region CLIP Similarity"),
    ("pie-bench-whole-image-clip-similarity", "PIE-Bench Whole Image CLIP Similarity"),
    ("pointmap-accuracy", "Pointmap Accuracy"),
    ("polistemics.adherence", "Polistemics Rubric Score"),
    ("prompt-level-loose-accuracy", "Prompt-Level Loose Accuracy"),
    ("prompt-level-strict-accuracy", "Prompt-Level Strict Accuracy"),
    ("r-at-1-mean-0-3-and-0-5", "R@1 Mean(0.3 and 0.5)"),
    ("r1-at-0-3", "R1@0.3"),
    ("r1-at-0-5", "R1@0.5"),
    ("r1-at-0-7", "R1@0.7"),
    ("radgraph-f1", "RadGraph F1"),
    ("restoration-score-rs", "Restoration Score (RS)"),
    ("risebench-overall-accuracy", "RISEBench Overall Accuracy"),
    ("scicode.main", "SciCode Main-Problem Solve Rate"),
    ("score-0-10", "Score (0-10)"),
    ("segmentation-au-spro-until-fpr-5-pct", "Segmentation AU-sPRO (until FPR 5%)"),
    ("segmentation-f1-max", "Segmentation F1-max"),
    ("segmentation-f1-private", "Segmentation F1 (Private)"),
    ("segmentation-f1-private-mixed", "Segmentation F1 (Private Mixed)"),
    ("segmentation-f1-public", "Segmentation F1 (Public)"),
    ("stft-dist", "STFT-Dist."),
    ("success-suboptimal-rate", "Success + Suboptimal Rate"),
    ("swebench-verified-mini-mariushobbhahn.mean", "swebench-verified-mini-mariushobbhahn.mean"),
    ("top-5-accuracy", "Top-5 Accuracy"),
    ("total-cost", "Total Cost"),
    ("verb-top-1-accuracy", "Verb Top-1 Accuracy"),
    ("vqascore", "VQA Score"),
    ("waics", "WAICS (Average, excl. Incomplete)"),
    ("wb-score", "WildBench Score"),
    ("wise-score", "WISE Score"),
    ("word-accuracy", "Word Accuracy"),
    ("worldmodelbench-total-score", "WorldModelBench Total Score"),
})

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


def test_extraction_alone_introduces_no_new_misroute(resolver):
    """A surface form the extractor mis-routes must be in the reviewed
    inventory; a fresh one means a new compound name the keyword table does
    not know (add a compound pattern before the generic one, as for
    length-controlled win rate), not a new exemption."""
    fresh = sorted(
        (canonical, raw) for canonical, raw in _surface_forms()
        if (canonical, raw) not in KNOWN_EXTRACTION_MISROUTES
        and (resolver.resolve(extract_metric(raw), entity_type="metric").canonical_id
             if extract_metric(raw) else None) != canonical
    )
    assert fresh == [], f"new extraction-only mis-routes: {fresh}"
    actual = {
        (canonical, raw) for canonical, raw in _surface_forms()
        if (resolver.resolve(extract_metric(raw), entity_type="metric").canonical_id
            if extract_metric(raw) else None) != canonical
    }
    stale = sorted(KNOWN_EXTRACTION_MISROUTES - actual)
    assert stale == [], f"inventory entries the extractor now routes correctly; remove them: {stale}"


def test_extraction_inventory_names_real_entries():
    ids = {e["id"] for e in _entries()}
    stale = sorted(c for c, _ in KNOWN_EXTRACTION_MISROUTES if c not in ids)
    assert stale == [], f"inventory names metrics that no longer exist: {stale}"
