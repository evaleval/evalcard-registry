"""
EEE-specific preprocessing for entity resolution.

Raw strings from the EEE datastore often encode multiple entity types in a
single field (e.g. ``evaluation_name`` contains both benchmark and metric).
These helpers extract clean, resolvable strings before passing them to the
resolver.

Usage::

    from eval_entity_resolver.eee import extract_metric, clean_eval_name

    metric_raw = extract_metric("Accuracy on IFEval")       # → "Accuracy"
    bench_raw  = clean_eval_name("bfcl.live.live_accuracy")  # → "bfcl live"
"""
from __future__ import annotations

import re


# ------------------------------------------------------------------
# Metric extraction
# ------------------------------------------------------------------

def extract_metric(metric_desc: str) -> str:
    """Extract a reusable metric name from an EEE evaluation description.

    EEE configs rarely provide a structured metric_id.  Instead the metric
    lives inside ``evaluation_description`` in one of several formats:

    * **"X on Y"** — ``"Accuracy on IFEval"`` → ``"Accuracy"``
    * **Dot notation** — ``"bfcl.live.live_accuracy"`` → ``"accuracy"``
    * **Verbose description** — ``"Chat accuracy - includes easy subsets"``
      → ``"accuracy"`` (keyword extraction)
    * **No keyword** — ``"Global MMLU Lite - Arabic"`` → ``"score"``
      (generic fallback)

    The returned string is passed to the resolver, which maps it to a
    canonical metric entity via alias lookup / normalized match.
    """
    text = metric_desc.strip()
    if not text:
        return text

    from_dot = False

    # 1. Dot notation: "bfcl.live.live_accuracy" → last segment → "live accuracy"
    if "." in text and " " not in text:
        text = text.rsplit(".", 1)[1].replace("_", " ").strip()
        from_dot = True

    # 2. "X on Y" pattern: "Accuracy on IFEval" → "Accuracy". Only when the
    #    prefix is a short metric-like phrase — a prose sentence with a
    #    mid-sentence " on " ("GDPval-AA evaluates AI agents on …") must NOT
    #    be truncated there, or keywords disclosed later in the sentence
    #    ("… reports performance as an Elo score.") are never seen.
    if not from_dot:
        # A compound metric phrase that itself contains " on " ("Judge Scores
        # on Open Questions") would be truncated to its generic head; match
        # those on the untruncated text first.
        for pattern, canonical in _COMPOUND_ON_PHRASES:
            if re.search(pattern, text.lower()):
                return canonical
        m = re.match(r"^(.+?)\s+on\s+\S+", text, re.IGNORECASE)
        if m and len(m.group(1).split()) <= 3:
            text = m.group(1).strip()

    # 3. Try keyword extraction on any multi-word text or dot-notation segment.
    #    Single bare words ("Accuracy", "F1", "EM") pass straight to the resolver.
    word_count = len(text.split())
    needs_extraction = from_dot or word_count > 1

    if needs_extraction:
        canonical = _keyword_extract(text)
        if canonical:
            return canonical
        # No keyword found — verbose descriptions (4+ words) → generic fallback.
        # Short phrases (2-3 words) pass through so the resolver can still
        # match them via alias (e.g. "Equivalent (CoT)" →
        # math-equivalent-chain-of-thought).
        if not from_dot and word_count > 3:
            return "score"

    return text


# Ordered from most-specific to most-generic.  When multiple patterns
# match, the earliest *position* in the input text wins (see
# _keyword_extract).
# Metric phrases whose own wording contains " on ", checked before the
# "X on Y" rule can split them (see extract_metric step 2).
_COMPOUND_ON_PHRASES: list[tuple[str, str]] = [
    (r"judge[\s_-]*scores?[\s_-]*on[\s_-]*open[\s_-]*questions?", "Open Question Judge Score"),
]

_METRIC_KEYWORDS: list[tuple[str, str]] = [
    # Multi-word / compound patterns
    # pass@N: the trailing (?!\d) is load-bearing. Without it `pass@1`
    # matches inside `pass@10` and a pass@10 score is stored as pass@1.
    (r"pass@10(?!\d)",                   "Pass@10"),
    (r"pass@8(?!\d)",                    "Pass@8"),
    (r"pass@1(?!\d)",                    "Pass@1"),
    (r"pass[\s_-]*at[\s_-]*10(?!\d)",    "Pass@10"),
    (r"pass[\s_-]*at[\s_-]*8(?!\d)",     "Pass@8"),
    (r"pass[\s_-]*at[\s_-]*1(?!\d)",     "Pass@1"),
    (r"mean[\s_-]*win[\s_-]*rate",       "Mean Win Rate"),
    # AlpacaEval's two win-rate variants are different estimators from the
    # plain win rate; they must not be swallowed by the generic pattern.
    (r"(?:(?<![a-z])lc|length[\s_-]*controlled)[\s_-]*win[\s_-]*rate", "Length-Controlled Win Rate"),
    (r"discrete[\s_-]*win[\s_-]*rate",   "Discrete Win Rate"),
    (r"win[\s_-]*rate",                  "Win Rate"),
    (r"mean[\s_-]*response[\s_-]*time",  "Mean Response Time"),
    (r"mean[\s_-]*score",                "Mean Score"),
    # HELM's near-miss exact-match variants are separately computed stats.
    (r"prefix[\s_-]*quasi[\s_-]*exact[\s_-]*match", "Prefix Quasi-Exact Match"),
    (r"quasi[\s_-]*prefix[\s_-]*exact[\s_-]*match", "Prefix Quasi-Exact Match"),
    (r"quasi[\s_-]*exact[\s_-]*match",   "Quasi-Exact Match"),
    (r"prefix[\s_-]*exact[\s_-]*match",  "Prefix Exact Match"),
    (r"exact[\s_-]*match",               "Exact Match"),
    (r"equivalent[\s_-]*\(?chain[\s_-]*of[\s_-]*thought", "Equivalent (CoT)"),
    (r"\bbrier\b",                       "Brier Score"),
    # A benchmark-specific judge score is its own canonical, not the generic score.
    (r"open[\s_-]*questions?[\s_-]*judge[\s_-]*scores?", "Open Question Judge Score"),
    (r"bleu[\s_-]*4",                    "BLEU-4"),
    (r"cot[\s_-]*correct",              "COT correct"),
    (r"wb[\s_-]*score",                  "WB Score"),
    (r"avg[\s_-]*attempts",              "Average Attempts"),
    (r"latency[\s_-]*mean",              "mean-latency"),
    (r"latency.*(?:p95|95th)",            "p95-latency"),
    (r"latency.*(?:std|standard)",        "latency-stddev"),
    (r"max[\s_-]*delta",                 "max-delta"),
    (r"benchmark\s+evaluation",          "score"),
    (r"outperform",                      "rank"),
    # Compound accuracy types (before generic accuracy)
    # Patterns sourced from metric_names in evaleval/card_backend eval-list.
    (r"ast[\s_-]*accuracy",              "AST Accuracy"),
    (r"ifeval[\s_-]*strict",             "IFEval Strict Acc"),
    (r"normali[sz]ed[\s_-]*accuracy",    "Normalized Accuracy"),
    (r"overall[\s_-]*accuracy",          "Accuracy"),
    (r"(?:ir)?relevance[\s_-]*detection[\s_-]*accuracy", "Accuracy"),
    (r"no[\s_-]*snippet[\s_-]*accuracy", "Accuracy"),
    (r"long[\s_-]*context[\s_-]*accuracy", "Accuracy"),
    (r"kv[\s_-]*accuracy",               "Accuracy"),
    (r"vector[\s_-]*accuracy",           "Accuracy"),
    (r"recursive[\s_-]*summarization[\s_-]*accuracy", "Accuracy"),
    (r"total[\s_-]*cost",                "cost"),
    (r"cost[\s_-]*per[\s_-]*task",       "cost-per-task"),
    # Class-averaged F1 (before generic F1, which would swallow both)
    (r"macro[\s_-]*f1",                  "Macro F1"),
    (r"micro[\s_-]*f1",                  "Micro F1"),
    # Single-word patterns (generic, checked last by position)
    (r"\baccuracy\b",                    "Accuracy"),
    (r"\bacc\b",                         "Accuracy"),
    (r"\bscores?\b",                     "score"),
    (r"\bf1\b",                          "F1"),
    (r"\bem\b",                          "Exact Match"),
    (r"\belo\b",                         "Elo Rating"),
    (r"\branks?\b",                      "rank"),
    (r"\bcosts?\b",                      "cost"),
    (r"\bharmlessness\b",                "harmlessness"),
    (r"\bstddev\b",                      "stddev"),
]


def _keyword_extract(text: str) -> str | None:
    """Return the canonical metric name for the first keyword found in *text*."""
    lower = text.lower()
    best: str | None = None
    best_pos = len(lower) + 1
    for pattern, canonical in _METRIC_KEYWORDS:
        m = re.search(pattern, lower)
        if m and m.start() < best_pos:
            best_pos = m.start()
            best = canonical
    return best


# ------------------------------------------------------------------
# Benchmark-name cleaning
# ------------------------------------------------------------------

# Trailing metric patterns for space-separated names (e.g.
# "Gaming Score" → "Gaming").  Checked with ``re.search`` against
# the lowered name; the first match wins.
_TRAILING_METRIC_RE: list[str] = [
    r"mean\s+win\s+rate$",
    r"mean\s+response\s+time$",
    r"mean\s+score$",
    r"win\s+rate$",
    r"avg\s+attempts$",
    r"avg\s+latency\s+ms$",
    r"cost\s+per\s+\d+\s+calls\s+usd$",
    r"cost\s+per\s+task$",
    r"pass@\d+$",
    r"\b(?:score|accuracy|acc|elo|rank|f1|em)$",
]


# Segments that never carry benchmark identity in a dotted
# ``evaluation_name``: the aggregate marker sources use for the
# whole-benchmark rollup, and bare version fragments left behind by
# splitting a dotted version (``terminal-bench-2.0`` → ``2``, ``0``).
_AGGREGATE_SEGMENTS = {"overall"}


def prepare_eval_name_segments(eval_name: str | None) -> list[str] | None:
    """Split a dotted ``evaluation_name`` into candidate identity segments.

    Applies only the rewrites that need no registry knowledge: collapse
    identical adjacent segments (``bbq.bbq.overall`` → ``bbq.overall``),
    drop aggregate markers and numeric fragments. Returns None for names
    ``clean_eval_name`` would not dot-split, so the structured path and the
    legacy path see exactly the same set of names.
    """
    if not eval_name or not isinstance(eval_name, str):
        return None
    name = eval_name.strip()
    if "." not in name or " " in name:
        return None

    segments: list[str] = []
    for part in name.split("."):
        part = part.strip()
        if not part or part.isdigit():
            continue
        if part.lower() in _AGGREGATE_SEGMENTS:
            continue
        if segments and segments[-1].lower() == part.lower():
            continue
        segments.append(part)
    return segments or None


def clean_eval_name(eval_name: str) -> str:
    """Strip embedded metric information from an ``evaluation_name``.

    EEE configs often encode both benchmark *and* metric in a single
    ``evaluation_name`` string.  This function extracts the benchmark
    portion so that the metric lives only in ``metric_id``.

    Patterns handled:

    * **Dot notation** — ``"bfcl.live.live_accuracy"`` → ``"bfcl live"``
      (last segment is the metric, everything before is the benchmark)
    * **Underscore suffix** — ``"fibble1_arena_win_rate"`` → ``"fibble1 arena"``
    * **Trailing words** — ``"Gaming Score"`` → ``"Gaming"``
    """
    name = eval_name.strip()
    if not name:
        return name

    # --- 1. Dot notation: split on last dot ------------------------------
    # The last segment is the metric; everything before is the benchmark.
    # e.g. "bfcl.live.live_simple_ast_accuracy" → "bfcl live"
    if "." in name and " " not in name:
        parts = name.rsplit(".", 1)[0].split(".")
        return " ".join(p.replace("_", " ") for p in parts)

    # --- 2. Underscore/space names: strip trailing metric keywords -------
    # Normalise underscores to spaces so "fibble1_arena_win_rate" and
    # "Gaming Score" use the same codepath.
    has_underscores = "_" in name and " " not in name
    normalized = name.replace("_", " ") if has_underscores else name

    lower = normalized.lower()
    for pattern in _TRAILING_METRIC_RE:
        m = re.search(pattern, lower)
        if m:
            prefix = normalized[: m.start()].strip()
            if prefix:
                return prefix
            break  # matched but prefix is empty — fall through

    return name
