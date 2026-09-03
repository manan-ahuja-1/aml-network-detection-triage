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


@pytest.mark.parametrize("arm", ["A", "B"])
def test_label_never_reaches_the_feature_matrix(frame, arm):
    """The mistake that yields a 0.99 PR-AUC and a wasted week."""
    X, y, split, _ = build_features.build_arm(arm, frame)
    for column in X.columns:
        assert "laundering" not in column.lower(), f"label-like feature: {column}"
    assert build_features.LABEL not in X.columns


@pytest.mark.parametrize("arm", ["A", "B"])
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
