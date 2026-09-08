"""Day 5/6 — the frozen engine, the alert seam, and the knowledge base.

Same principle as tests/test_leakage.py: every failure covered here produces a
plausible number rather than an exception. A bootstrap that resamples the wrong axis,
an alert table that drops one side of each transaction, a KB chunk longer than the
embedder's window — none of them raise, and all of them quietly change what the
published results mean.
"""

from __future__ import annotations

import json
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


# ---------------------------------------------------------------------------
# The graph cache key (Day 9) — a cache that omits an input answers the wrong question
# ---------------------------------------------------------------------------
import features_graph as fg  # noqa: E402


def test_graph_cache_key_covers_the_currency_table():
    """Graph edges are weighted by USD volume and PageRank follows those weights, so the
    FX table is an input to every graph feature. Before this, the key was `train_end` and
    `seed` only — rebuilding under corrected rates would have returned the old
    money-weighted graph and the FX experiment would have measured nothing."""
    before = fg.fx_digest()
    original = dict(config.FX_TO_USD)
    try:
        config.FX_TO_USD = {**original, "Euro": original["Euro"] * 1.05}
        assert fg.fx_digest() != before
    finally:
        config.FX_TO_USD = original
    assert fg.fx_digest() == before


def test_graph_cache_key_covers_the_training_rows():
    """The single-bank experiment builds arm C from one institution's visible subset —
    same train_end, same seed, same FX, different rows. Without the training frame in the
    key it would have been served the full inter-bank graph from cache and concluded that
    partial visibility costs nothing."""
    a = pd.DataFrame(index=pd.RangeIndex(1000))
    b = pd.DataFrame(index=pd.RangeIndex(500))
    assert fg.train_digest(a) != fg.train_digest(b)
    assert fg.train_digest(a) == fg.train_digest(pd.DataFrame(index=pd.RangeIndex(1000)))


def test_graph_cache_paths_are_distinct_per_input():
    a = pd.DataFrame(index=pd.RangeIndex(1000))
    b = pd.DataFrame(index=pd.RangeIndex(999))
    assert fg._cache_paths(a)[0] != fg._cache_paths(b)[0]
    assert fg._cache_paths(a)[1] != fg._cache_paths(b)[1]


# ---------------------------------------------------------------------------
# FX provenance
# ---------------------------------------------------------------------------
def test_every_currency_has_a_sourced_rate():
    """The sourced table must cover exactly the currencies the operating table does, or
    the comparison silently skips one."""
    assert set(config.FX_TO_USD) == set(config.FX_TO_USD_SOURCED)
    assert all(v > 0 for v in config.FX_TO_USD_SOURCED.values())
    assert config.FX_TO_USD_SOURCED["US Dollar"] == 1.0


def test_fx_sources_are_registered_with_a_date():
    """Same discipline as kb/sources.json: a number without a retrievable source is the
    thing this project keeps refusing to ship."""
    assert config.FX_SOURCE_DATE == "2022-09-01"
    for key in ("ecb", "cbr", "sama_peg", "bitcoin"):
        assert config.FX_SOURCES[key].startswith("https://")


def test_partial_graph_uses_only_the_requested_fraction_of_training_edges():
    """The visibility experiment degrades ONLY the graph. Account and typology features
    stay on the full training window, because an institution does hold its own customers'
    histories — degrading everything would confound network visibility with simply having
    less data."""
    import inspect

    import single_bank
    src = inspect.getsource(single_bank.build_arm_with_partial_graph)
    # the graph frame is sampled; the account/typology tables are not
    assert "train.sample(frac=fraction" in src
    assert "features_account.build_account_table(train)" in src
    assert "features_typology.build_typology_table(train)" in src


def test_visibility_curve_spans_floor_to_ceiling():
    import single_bank
    assert single_bank.FRACTIONS[0] == 1.00, "the ceiling must be measured, not assumed"
    assert min(single_bank.FRACTIONS) < 0.25
    assert sorted(single_bank.FRACTIONS, reverse=True) == list(single_bank.FRACTIONS)


def test_the_abandoned_single_bank_attempt_is_recorded():
    """Bank 070 turned out to be a clearing entity — 15 accounts, 452,751 transactions,
    no internal transfers — so the per-institution experiment was abandoned. The evidence
    for that decision travels in the results rather than living only in a commit message."""
    import single_bank
    d = single_bank.DATASET_STRUCTURE
    assert d["n_banks"] > 30_000
    assert d["median_accounts_per_bank"] <= 5
    assert d["bank_070"]["internal_transfer_share"] == 0.0
    assert d["bank_070"]["transactions_per_account"] > 10_000
    assert "clearing" in d["bank_070"]["verdict"]


# ---------------------------------------------------------------------------
# README provenance (Day 10) — no number in the README is typed by hand
# ---------------------------------------------------------------------------
readme_built = pytest.mark.skipif(
    not (config.RESULTS / "readme_provenance.json").exists(),
    reason="README not generated yet; run `make readme`")


@readme_built
def test_every_readme_number_still_resolves_to_a_results_path():
    """The generator records the dotted path behind each value it renders. If a results
    file is regenerated with a different shape, this fails instead of the README quietly
    keeping a number whose source no longer exists."""
    import readme_data
    prov = json.loads((config.RESULTS / "readme_provenance.json").read_text())
    fetch = readme_data.Fetch()
    assert prov["paths"], "no values were recorded — the generator is not using Fetch"
    for path in prov["paths"]:
        fetch(path)          # raises readme_data.Missing if it no longer resolves


@readme_built
def test_readme_is_reproducible_from_results():
    """Regenerating must be a no-op. A hand-edit to README.md is therefore detectable,
    which is the whole point of generating it."""
    import subprocess
    readme = config.ROOT / "README.md"
    before = readme.read_text()
    subprocess.run([sys.executable, str(config.ROOT / "src" / "make_readme.py")],
                   check=True, capture_output=True)
    assert readme.read_text() == before, (
        "README.md differs from what make_readme.py generates — it was hand-edited, "
        "or a results file changed without the README being regenerated")


@readme_built
def test_the_readme_reports_the_negative_result():
    """The agent's disposition does not work, and the README says so. This is a guard
    against a future edit quietly promoting the parts that do work."""
    text = (config.ROOT / "README.md").read_text().lower()
    assert "simpson" in text
    assert "does not" in text or "no signal" in text
    assert "limitations" in text


def test_paired_bootstrap_keeps_the_pairing():
    """Two arms scored on identical rows must be compared paired, not by overlapping
    marginal CIs. The tell that the pairing is real: two IDENTICAL score vectors differ
    by exactly zero on every resample, so the interval has zero width. An unpaired
    comparison of the same two vectors would produce a wide interval around zero and
    call a real difference 'not significant'."""
    import numpy as np

    import single_bank
    rng = np.random.default_rng(0)
    n = 50_000
    y = (rng.random(n) < 0.002).astype(int)
    noise = rng.random(n)
    informative = np.clip(y * 0.5 + rng.random(n) * 0.5, 0, 1)

    out = single_bank.paired_deltas(
        {"graph_none": noise, "identical": noise.copy(), "better": informative},
        y, n_resamples=200)

    assert out["identical"]["delta_vs_no_graph"] == 0.0
    assert out["identical"]["ci95"] == [0.0, 0.0]
    assert not out["identical"]["excludes_zero"]
    assert out["better"]["delta_vs_no_graph"] > 0
    assert out["better"]["excludes_zero"]


def test_paired_bootstrap_is_stratified():
    """Same reason bootstrap_pr_auc is: at a 1-in-937 base rate an unstratified draw
    varies its positive count by several percent, and PR-AUC moves with the base rate."""
    import inspect

    import single_bank
    src = inspect.getsource(single_bank.paired_deltas)
    assert "np.flatnonzero(y == 1)" in src and "np.flatnonzero(y == 0)" in src
    assert "counts[pos]" in src and "counts[neg]" in src


# ---------------------------------------------------------------------------
# Seed replicates for the visibility curve
# ---------------------------------------------------------------------------
def test_the_seed_changes_which_edges_are_visible():
    """A replicate must actually resample the graph. If the seed did not reach
    `train.sample`, every 'replicate' would rebuild the identical graph, the spread would
    be exactly zero, and the experiment would report perfect reproducibility as evidence
    of a robust effect."""
    import inspect

    import single_bank
    src = inspect.getsource(single_bank.build_arm_with_partial_graph)
    assert "random_state=seed" in src
    sig = inspect.signature(single_bank.build_arm_with_partial_graph)
    assert "seed" in sig.parameters


@needs_data
def test_two_seeds_produce_different_graph_cache_keys():
    """The digest has to separate them, or the second seed is served the first's graph —
    the same class of bug that has bitten this project five times."""
    import features_graph as fg
    frame = splits.load_transactions(columns=["timestamp", "from_id", "to_id"])
    train = splits.train_frame(frame)
    a = train.sample(frac=0.25, random_state=42)
    b = train.sample(frac=0.25, random_state=7)
    assert fg.train_digest(a) != fg.train_digest(b)
    assert fg._cache_paths(a)[0] != fg._cache_paths(b)[0]


def test_replicates_do_not_rerun_the_seed_independent_arms():
    """Arm B has no graph and the 100% arm does no sampling, so neither depends on the
    seed. Re-running them would cost ~18 minutes per replicate and produce identical
    numbers; the floor is read back from the persisted scores instead."""
    import inspect

    import single_bank
    src = inspect.getsource(single_bank.replicate)
    assert "SCORES_OUT" in src and "graph_none" in src
    assert "build_features.build_arm(\"B\"" not in src


def test_aggregator_reports_a_count_not_a_p_value():
    """Three seeds supports a mean, a range and a count of replicates separating from the
    floor. It does not support a significance test, and running one would dress the same
    three numbers up as an inference they cannot carry."""
    import inspect

    import single_bank
    src = inspect.getsource(single_bank.aggregate)
    assert "n_separated_from_floor" in src
    assert "pr_auc_range" in src
    for forbidden in ("ttest", "t_test", "scipy.stats.ttest"):
        assert forbidden not in src


@readme_built
def test_the_readme_states_an_interval_beside_the_ablation_lift():
    """The graph lift is at the edge of significance. The README must not present
    0.1815 -> 0.1938 as a settled result anywhere, including its opening paragraph."""
    text = (config.ROOT / "README.md").read_text()
    assert "edge of significance" in text or "edge of conventional significance" in text
    assert "+0.0123" in text and "[-0.0025, +0.0268]" in text.replace("−", "-")


def test_replicate_merges_rather_than_overwrites():
    """A pre-flight runs one cheap fraction first; overwriting the seed file afterwards
    would discard ~14 minutes of work for no reason."""
    import inspect

    import single_bank
    src = inspect.getsource(single_bank.replicate)
    assert "json.loads(out.read_text()) if out.exists()" in src
    assert '**report.get("arms", {})' in src


@readme_built
def test_the_readme_does_not_claim_a_partial_graph_is_worse_than_none():
    """That claim was made, published in a draft, and then overturned by its own replicate:
    at 25% visibility one edge draw gave PR-AUC 0.1421 and another 0.1904. It must not
    creep back in on a later edit."""
    # The README is wrapped at 92 columns, so any phrase check has to be
    # whitespace-insensitive or it silently passes whenever a line break lands mid-claim.
    text = " ".join((config.ROOT / "README.md").read_text().lower().split())
    for retracted in ("worse than no graph",
                      "do not degrade gracefully",
                      "partial graph is worse"):
        assert retracted not in text, f"retracted claim reappeared: {retracted!r}"
    # what replaced it must still be there
    assert "which edges you see swamps how many" in text


@readme_built
def test_the_visibility_experiment_is_reported_as_inconclusive():
    """It belongs in Limitations as a fact about the dataset, not as a result about the
    model. If it ever grows back into a section, this fails."""
    raw = (config.ROOT / "README.md").read_text()
    flat = " ".join(raw.split())
    assert "cannot quantify what it is worth" in flat
    # position check has to run on the raw text; compare section offsets, not the flattened
    # copy, so "is it inside Limitations" stays a real question about document order
    limitations = raw.index("## Limitations")
    finding = raw.index("Full inter-bank visibility is a synthetic-data luxury")
    assert finding > limitations, "the visibility finding moved out of Limitations"
