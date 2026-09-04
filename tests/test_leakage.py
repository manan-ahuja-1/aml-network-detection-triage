"""Tests that make the leak-free claim (A1) checkable instead of asserted.

Every failure mode covered here produces a BETTER-looking number, not an error. That
is what makes them worth writing: nothing else in the pipeline would complain.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import build_features  # noqa: E402
import config  # noqa: E402
import features_account  # noqa: E402
import features_typology  # noqa: E402
import splits  # noqa: E402

pytestmark = pytest.mark.skipif(
    not config.TRANS_PARQUET.exists(),
    reason="run `make data` first",
)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return splits.load_transactions()


@pytest.fixture(scope="module")
def account_table(frame) -> pd.DataFrame:
    return features_account.build_account_table(splits.train_frame(frame))


def test_account_table_uses_only_training_accounts(frame, account_table):
    """The account table must contain no account that appears solely after train_end.

    If it did, features would encode behaviour from the future — the exact failure a
    time-based split does NOT prevent on its own.
    """
    train = splits.train_frame(frame)
    train_accounts = set(pd.concat([train["from_id"], train["to_id"]]).unique())
    extra = set(account_table.index) - train_accounts
    assert not extra, f"{len(extra)} accounts in the table never appear in training"


def test_account_aggregates_match_training_window_only(frame, account_table):
    """Recompute one account's counts from training rows and demand an exact match.

    A direct check that the aggregate is a function of the training window alone: if
    the table had been built over the full dataset, these numbers would be larger.
    """
    train = splits.train_frame(frame)
    sample = account_table.sort_values("out_n", ascending=False).head(20).index

    for node in sample:
        expected = int((train["from_id"] == node).sum())
        actual = int(account_table.loc[node, "out_n"])
        assert actual == expected, f"{node}: table says {actual}, training rows say {expected}"


def test_unseen_accounts_are_nan_not_zero(frame, account_table):
    """Cold-start accounts must be NaN, and the explicit flag must be set.

    Zero would assert "this account has no counterparties", which is a real and
    different statement from "we have never seen this account". Filling with zero
    fabricates a signal indistinguishable from a genuine one.
    """
    features = features_account.build(frame, account_table)
    unseen = features["from_is_unseen"] == 1

    if unseen.sum() == 0:
        pytest.skip("no cold-start accounts in this split configuration")

    assert features.loc[unseen, "from_out_n"].isna().all(), "unseen accounts were filled"
    assert features.loc[unseen, "from_degree_total"].isna().all()
    # And the flag must not fire for accounts we DO know.
    known = features["from_is_unseen"] == 0
    assert features.loc[known, "from_out_n"].notna().any()


def test_unseen_flag_matches_training_membership(frame, account_table):
    """The flag must mean exactly what it says."""
    features = features_account.build(frame, account_table)
    in_table = frame["from_id"].isin(account_table.index).to_numpy()
    flagged_unseen = (features["from_is_unseen"] == 1).to_numpy()
    assert np.array_equal(flagged_unseen, ~in_table)


def test_no_training_row_is_cold_start(frame, account_table):
    """Every training account is in the table by construction; 0% is the only valid rate."""
    features = features_account.build(frame, account_table)
    train_mask = (frame["split"] == "train").to_numpy()
    assert features.loc[train_mask, "from_is_unseen"].sum() == 0


@pytest.mark.parametrize("arm", ["A", "B", "C"])
def test_label_never_reaches_the_feature_matrix(frame, arm):
    """The mistake that yields a 0.99 PR-AUC and a wasted week."""
    X, y, split, _ = build_features.build_arm(arm, frame)
    for column in X.columns:
        assert "laundering" not in column.lower(), f"label-like feature: {column}"
    assert build_features.LABEL not in X.columns


@pytest.mark.parametrize("arm", ["A", "B", "C"])
def test_features_align_with_labels_and_splits(frame, arm):
    """A join that drops or duplicates rows is the most common silent failure here."""
    X, y, split, _ = build_features.build_arm(arm, frame)
    assert len(X) == len(frame) == len(y) == len(split)
    assert X.index.equals(frame.index)
    boundaries = splits.load_boundaries()
    for name in ("train", "val", "test"):
        assert int((split == name).sum()) == boundaries[f"{name}_rows"]


def test_entity_features_actually_join(frame):
    """Regression test for the Day 2 bug that produced 100% NaN entity features.

    HI-Small_Trans.csv zero-pads bank IDs ("010") and HI-Small_accounts.csv does not
    ("10"), so an un-normalised composite key matched NOTHING across the two files —
    silently, because a failed join produces NaN rather than an error. `node_id`
    strips leading zeros to fix it; this test ensures it stays fixed.
    """
    entity_table = features_typology.build_entity_table()
    matched = frame["from_id"].isin(entity_table.index).mean()
    assert matched > 0.99, f"only {matched*100:.2f}% of accounts joined to entity data"


# ---------------------------------------------------------------------------
# Day 3 — graph-specific leakage and arm-boundary tests
# ---------------------------------------------------------------------------

import features_graph  # noqa: E402


@pytest.fixture(scope="module")
def train_graph(frame):
    return features_graph.build_graph(splits.train_frame(frame))


def test_graph_contains_no_post_train_edge(frame, train_graph):
    """The graph must be a function of the training window alone.

    This is the failure a time-based split does NOT prevent: build one graph over the
    whole dataset and a training row's PageRank encodes edges that had not happened
    yet, so the model appears to know a mule's future counterparties. The split still
    looks correct and the metrics still look plausible.
    """
    train = splits.train_frame(frame)
    expected = set(
        map(tuple,
            train.loc[train["from_id"] != train["to_id"], ["from_id", "to_id"]]
            .drop_duplicates().to_numpy())
    )
    actual = set(train_graph.edges())
    assert actual == expected, (
        f"{len(actual - expected)} edges in the graph are absent from the training "
        f"window; {len(expected - actual)} training pairs are missing from the graph"
    )


def test_graph_has_no_self_loops(train_graph):
    """Self-loops inflate centrality without carrying relational information.

    18% of training rows are self-transfers (mostly Reinvestment). Left in, they would
    dominate PageRank. The behaviour is already captured by `is_self_transaction` in
    arm A.
    """
    assert nx_selfloop_count(train_graph) == 0


def nx_selfloop_count(graph) -> int:
    return sum(1 for u, v in graph.edges() if u == v)


def test_graph_nodes_are_a_subset_of_training_accounts(frame, train_graph):
    train = splits.train_frame(frame)
    train_accounts = set(pd.concat([train["from_id"], train["to_id"]]).unique())
    assert set(train_graph.nodes) <= train_accounts


def test_in_graph_flag_separates_the_third_account_state(frame):
    """Self-loop-only accounts must be distinguishable from never-seen accounts.

    Three states exist, not two:
      1. in training and in the graph        -> full features
      2. in training, self-loops only        -> arm B aggregates, NaN topology
      3. never in training                   -> NaN everywhere

    States 2 and 3 both produce NaN topology. Without `is_in_graph` the model cannot
    tell them apart, and they mean genuinely different things — one is a real account
    that never transacted with anyone, the other is an account we have no history for.
    """
    boundaries = splits.load_boundaries()
    train = splits.train_frame(frame)
    graph_table = features_graph.build_graph_table(train, boundaries["train_end"])
    graph_features = features_graph.build(frame, graph_table)

    account_table = features_account.build_account_table(train)
    account_features = features_account.build(frame, account_table)

    seen = account_features["from_is_unseen"] == 0
    in_graph = graph_features["from_is_in_graph"] == 1

    # State 2 must be non-empty — 20.6% of accounts were measured to be self-loop-only.
    state_2 = seen & ~in_graph
    assert state_2.sum() > 0, "expected self-loop-only accounts to exist"

    # State 2 has account features but no topology.
    assert account_features.loc[state_2, "from_out_n"].notna().any()
    assert graph_features.loc[state_2, "from_pagerank"].isna().all()

    # Nothing may be in the graph without having been seen in training.
    assert not (in_graph & ~seen).any()


def test_arm_C_is_a_strict_superset_of_arm_B(frame):
    """Nested arms are what make the ablation interpretable.

    If C were not a superset of B, a difference in PR-AUC could come from features C
    dropped rather than features it added, and the "graph lift" claim would not hold.
    """
    X_b, _, _, _ = build_features.build_arm("B", frame)
    X_c, _, _, _ = build_features.build_arm("C", frame)
    missing = set(X_b.columns) - set(X_c.columns)
    assert not missing, f"arm C dropped {len(missing)} arm B columns: {sorted(missing)[:5]}"
    assert len(X_c.columns) > len(X_b.columns)


def test_arm_C_adds_no_degree_equivalent_feature(frame):
    """Guards the arm B/C boundary that the whole graph claim rests on.

    Degree lives in arm B because a groupby produces it. If an arm C feature were
    near-perfectly correlated with degree, the "multi-hop lift" would silently be a
    degree lift, and the headline claim would be wrong in a way no other test catches.

    MEASURED PER ACCOUNT, NOT PER TRANSACTION. Correlating at the row level weights
    every account by how often it transacts, which inflates the correlation toward
    high-activity accounts: reverse-PageRank measured that way reads 1.0000 against
    degree, versus 0.993 per account. The account is the unit these features are
    defined on, so it is the unit the check must use.

    This test is what caught reverse-PageRank collapsing onto out-degree on a graph
    with average degree 2.80 — see features_graph.FEATURES.
    """
    import networkx as nx  # noqa: F401  (imported via features_graph)

    boundaries = splits.load_boundaries()
    train = splits.train_frame(frame)

    graph_table = features_graph.build_graph_table(train, boundaries["train_end"])
    graph = features_graph.build_graph(train)

    degree = pd.DataFrame({
        "in_deg": pd.Series(dict(graph.in_degree())),
        "out_deg": pd.Series(dict(graph.out_degree())),
    })
    degree["total_deg"] = degree["in_deg"] + degree["out_deg"]

    joined = graph_table.join(degree, how="inner")

    for column in graph_table.columns:
        if joined[column].nunique() <= 1:
            continue
        for degree_col in ("in_deg", "out_deg", "total_deg"):
            corr = joined[column].corr(joined[degree_col])
            if pd.isna(corr):
                continue
            assert abs(corr) < 0.99, (
                f"{column} correlates {corr:.4f} with {degree_col} at the account "
                "level — arm C would be reintroducing the quantity the arm boundary "
                "exists to exclude"
            )
