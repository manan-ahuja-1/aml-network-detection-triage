"""Load the frozen temporal split and attach it to the transaction table.

Every module that needs train/val/test goes through here rather than re-deriving
boundaries from the data. That is the whole point: `make_splits.py` computed the
boundaries once and wrote them to JSON, and if any later stage recomputed them from
possibly-different data, a model trained under one split would be evaluated under
another. Nothing would error; the metrics would simply be wrong.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


def load_boundaries() -> dict:
    """Read the frozen split boundaries, failing loudly if Day 1 has not run."""
    if not config.SPLIT_BOUNDARIES_JSON.exists():
        raise FileNotFoundError(
            f"{config.SPLIT_BOUNDARIES_JSON} not found. Run `make data` first "
            "(src/make_data.py then src/make_splits.py)."
        )
    return json.loads(config.SPLIT_BOUNDARIES_JSON.read_text())


def load_transactions(columns: list[str] | None = None) -> pd.DataFrame:
    """Transactions, truncated at the cutoff and tagged with a `split` column.

    Truncation is applied here rather than left to callers because forgetting it
    reintroduces the Sept 11-18 tail, which is 59% laundering and would inflate every
    metric downstream. Making it the default behaviour of the loader means no caller
    can forget.
    """
    boundaries = load_boundaries()
    df = pd.read_parquet(config.TRANS_PARQUET, columns=columns)
    df = df[df["timestamp"] < pd.Timestamp(config.DATA_CUTOFF)].copy()

    train_end = pd.Timestamp(boundaries["train_end"])
    val_end = pd.Timestamp(boundaries["val_end"])

    # pd.cut would be tidier but returns NaN outside its bins; explicit assignment
    # guarantees every row lands in exactly one split.
    split = pd.Series("test", index=df.index, dtype="object")
    split[df["timestamp"] < train_end] = "train"
    split[(df["timestamp"] >= train_end) & (df["timestamp"] < val_end)] = "val"
    df["split"] = split.astype("category")

    # The loader asserts its own contract. If these ever disagree with the frozen
    # JSON, the parquet has changed underneath us and every downstream number is
    # suspect — better to stop here than to report metrics from a different dataset.
    for name in ("train", "val", "test"):
        expected = boundaries[f"{name}_rows"]
        actual = int((df["split"] == name).sum())
        if actual != expected:
            raise ValueError(
                f"{name} split has {actual:,} rows but split_boundaries.json says "
                f"{expected:,}. The parquet no longer matches the frozen split; "
                "re-run `make data`."
            )

    return df


def train_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rows the model may learn from, and the ONLY rows features may be built from."""
    return df[df["split"] == "train"]


if __name__ == "__main__":
    frame = load_transactions()
    print(f"{len(frame):,} transactions after truncation at {config.DATA_CUTOFF}")
    summary = frame.groupby("split", observed=True).agg(
        rows=("is_laundering", "size"),
        positives=("is_laundering", "sum"),
    )
    summary["rate_%"] = (summary["positives"] / summary["rows"] * 100).round(4)
    print(summary.to_string())
