"""Assertions that protect the split's correctness (verification section of the plan).

These are the tests that would catch the two failure modes which produce
plausible-looking but invalid results: temporal overlap between splits, and the
laundering-heavy tail leaking back into evaluation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config  # noqa: E402

pytestmark = pytest.mark.skipif(
    not config.SPLIT_BOUNDARIES_JSON.exists(),
    reason="run `python src/make_splits.py` first",
)


@pytest.fixture(scope="module")
def boundaries() -> dict:
    return json.loads(config.SPLIT_BOUNDARIES_JSON.read_text())


def test_splits_are_strictly_ordered_in_time(boundaries):
    """train < val < test, with no overlap. The core anti-leakage property."""
    start = pd.Timestamp(boundaries["data_start"])
    train_end = pd.Timestamp(boundaries["train_end"])
    val_end = pd.Timestamp(boundaries["val_end"])
    end = pd.Timestamp(boundaries["data_end"])
    assert start < train_end < val_end <= end


def test_tail_is_excluded(boundaries):
    """The Sept 11-18 tail is ~59% laundering and must not reach any split.

    Including it would inflate PR-AUC by an artifact of the data generator rather
    than by anything the model learned.
    """
    assert pd.Timestamp(boundaries["data_end"]) < pd.Timestamp(config.DATA_CUTOFF)


def test_every_split_contains_positives(boundaries):
    """A split with no positives makes PR-AUC undefined."""
    for name in ("train", "val", "test"):
        assert boundaries[f"{name}_positives"] > 0, f"{name} has no laundering rows"


def test_class_imbalance_is_severe_enough_to_justify_pr_auc(boundaries):
    """Documents WHY we report PR-AUC rather than ROC-AUC, as an executable claim."""
    for name in ("train", "val", "test"):
        assert boundaries[f"{name}_rate"] < 0.01, f"{name} is not severely imbalanced"
