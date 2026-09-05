"""Day 5/6 — the frozen engine, the alert seam, and the knowledge base.

Same principle as tests/test_leakage.py: every failure covered here produces a
plausible number rather than an exception. A bootstrap that resamples the wrong axis,
an alert table that drops one side of each transaction, a KB chunk longer than the
embedder's window — none of them raise, and all of them quietly change what the
published results mean.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import alerts as alerts_mod  # noqa: E402
import config  # noqa: E402
import evaluate  # noqa: E402
import kb_index  # noqa: E402
import metrics  # noqa: E402
import splits  # noqa: E402

needs_data = pytest.mark.skipif(
    not config.TRANS_PARQUET.exists(), reason="run `make data` first")


@pytest.fixture(scope="module")
def frame():
    return splits.load_transactions()


# ---------------------------------------------------------------------------
# The bootstrap
# ---------------------------------------------------------------------------
def test_weighted_average_precision_matches_sklearn():
    """The fast bootstrap path must agree with the metric we actually report.

    `_weighted_average_precision` exists only because sklearn's implementation is too
    slow to call 2,000 times. If it disagreed even slightly, the published confidence
    interval would describe a different quantity than the point estimate it brackets,
    and the two numbers would still look perfectly consistent side by side.
    """
    rng = np.random.default_rng(0)
    for n, rate in ((50_000, 0.002), (200_000, 0.001)):
        y = (rng.random(n) < rate).astype("int8")
        scores = rng.random(n) + 0.3 * y
        order = np.argsort(-scores, kind="stable")
        fast = evaluate._weighted_average_precision(
            y[order].astype("float64"), np.ones(n))
        assert fast == pytest.approx(metrics.pr_auc(y, scores), abs=1e-9)


def test_bootstrap_is_stratified_and_reproducible():
    """Stratified resampling holds the base rate fixed, and the seed must bind.

    PR-AUC moves with the base rate. An unstratified bootstrap lets the positive count
    wander per draw, so the interval would partly measure class-balance jitter rather
    than model uncertainty — wider than the truth, and wide for the wrong reason.
    """
    rng = np.random.default_rng(1)
    n = 20_000
    y = (rng.random(n) < 0.01).astype("int8")
    scores = rng.random(n) + 0.4 * y

    result = evaluate.bootstrap_pr_auc(y, scores, n_resamples=50, ci=0.95, seed=7)
    assert result["ci_low"] < result["point_estimate"] < result["ci_high"]
    assert result["stratified"] is True

    again = evaluate.bootstrap_pr_auc(y, scores, n_resamples=50, ci=0.95, seed=7)
    assert (result["ci_low"], result["ci_high"]) == (again["ci_low"], again["ci_high"])


def test_cost_strip_is_monotone_in_the_cost_ratio():
    """A higher price on a miss must never buy FEWER alerts.

    If it did, the sweep would be reporting an optimiser bug as a business insight.
    """
    rng = np.random.default_rng(2)
    n = 50_000
    y = (rng.random(n) < 0.005).astype("int8")
    scores = rng.random(n) + 0.5 * y

    ratios = (10, 50, 100, 500)
    strip = evaluate.cost_sensitive_strip(y, scores, ratios)
    counts = [strip[str(r)]["alerts"] for r in ratios]
    recalls = [strip[str(r)]["recall"] for r in ratios]
    assert counts == sorted(counts), f"alert count not monotone in cost ratio: {counts}"
    assert recalls == sorted(recalls)


# ---------------------------------------------------------------------------
# The alert seam
# ---------------------------------------------------------------------------
@needs_data
def test_alerts_count_both_sides_of_every_transaction(frame):
    """Each transaction must contribute to its sender AND its receiver.

    Scoring only the sender would systematically miss collection points — the
    funnel-account signature this project exists to find. The check is arithmetic:
    the alert table's transaction total must be exactly twice the split's row count.
    """
    part = frame[frame["split"] == "val"]
    table = alerts_mod.account_alerts(frame, np.linspace(0, 1, len(part)), "val")
    assert int(table["n_transactions"].sum()) == 2 * len(part), (
        "alert table lost one side of some transactions")


@needs_data
def test_alert_score_is_the_max_over_the_accounts_transactions(frame):
    """Spot-check the aggregation rule against a direct recomputation."""
    part = frame[frame["split"] == "val"]
    scores = np.random.default_rng(3).random(len(part))
    table = alerts_mod.account_alerts(frame, scores, "val")
    series = pd.Series(scores, index=part.index)

    for node in table.head(5).index:
        mask = ((part["from_id"] == node) | (part["to_id"] == node)).to_numpy()
        assert table.loc[node, "alert_score"] == pytest.approx(series[mask].max())


@needs_data
def test_productive_alerts_use_only_the_scored_split(frame):
    """An account that laundered in TRAINING but was clean in val is not a val hit.

    Counting it would inflate precision with cases that were not there to find at
    scoring time.
    """
    part = frame[frame["split"] == "val"]
    table = alerts_mod.account_alerts(frame, np.zeros(len(part)), "val")
    productive = set(table.index[table["is_productive"] == 1])
    laundering = part[part["is_laundering"] == 1]
    assert productive == set(laundering["from_id"]) | set(laundering["to_id"])


@needs_data
def test_alert_evidence_stays_inside_the_split(frame):
    """The agent's citable evidence must exclude training-window activity.

    Handing the agent training rows would let it justify a test alert with evidence
    the engine never scored.
    """
    part = frame[frame["split"] == "val"]
    evidence = alerts_mod.alert_evidence(frame, part["from_id"].iloc[0], "val")
    assert (evidence["split"] == "val").all()
    assert evidence["txn_id"].str.match(r"^T\d{7}$").all()


@needs_data
def test_account_alerts_rejects_mismatched_scores(frame):
    """Passing scores computed on a different frame must fail loudly, not align silently."""
    with pytest.raises(ValueError, match="different frame"):
        alerts_mod.account_alerts(frame, np.zeros(10), "val")


# ---------------------------------------------------------------------------
# Pattern ground truth
# ---------------------------------------------------------------------------
@needs_data
def test_pattern_join_does_not_duplicate_transactions(frame):
    """The pattern key must match at most one transaction each.

    A duplicating join would double-count caught patterns and inflate pattern-level
    recall silently, because the numbers would still be in range.
    """
    membership = evaluate.load_pattern_membership(frame)
    assert not membership.index.duplicated().any()
    assert membership.index.isin(frame.index).all()


@needs_data
def test_pattern_transactions_are_all_labelled_laundering(frame):
    """Every row the patterns file claims must carry is_laundering == 1.

    Otherwise the two ground truths this project reports against disagree, and one of
    them is wrong.
    """
    membership = evaluate.load_pattern_membership(frame)
    assert (frame.loc[membership.index, "is_laundering"] == 1).all()


# ---------------------------------------------------------------------------
# The knowledge base
# ---------------------------------------------------------------------------
def test_every_corpus_document_declares_a_registered_source():
    """No chunk enters the index without provenance; load_corpus raises otherwise."""
    texts, metadatas, ids = kb_index.load_corpus()
    assert len(texts) == len(metadatas) == len(ids)
    assert len(set(ids)) == len(ids), "duplicate chunk ids"
    assert all(m["source_id"] for m in metadatas)


def test_no_chunk_exceeds_the_embedder_window():
    """Chunks past the window are silently truncated: the stored text is complete, the
    vector represents only the opening, and retrieval never matches the tail.

    Measured with the embedder's own tokenizer and chroma's real 256-token limit — not
    the 128 that ships inside tokenizer.json, which chroma explicitly overrides.
    """
    texts, _, ids = kb_index.load_corpus()
    counts = kb_index.token_counts(texts)
    if not counts:
        pytest.skip("embedder not downloaded; run `make kb` first")
    over = [(i, n) for i, n in zip(ids, counts) if n > kb_index.EMBEDDER_MAX_TOKENS]
    assert not over, f"chunks over {kb_index.EMBEDDER_MAX_TOKENS} tokens: {over}"


def test_every_chunk_carries_its_document_title():
    """A bare red-flag list embeds almost identically to any other bare red-flag list."""
    texts, metadatas, _ = kb_index.load_corpus()
    for text, meta in zip(texts, metadatas):
        assert text.startswith(meta["title"]), f"chunk missing title: {text[:60]}"


def test_all_eight_typologies_are_covered_by_the_corpus():
    """The agent classifies into 8 typologies + NONE; each needs retrievable material."""
    _, metadatas, _ = kb_index.load_corpus()
    tagged: set[str] = set()
    for meta in metadatas:
        tagged.update(t for t in meta["typologies"].split("|") if t)
    expected = set(config.TYPOLOGIES) - {"NONE"}
    assert expected <= tagged, f"no corpus material for: {sorted(expected - tagged)}"
