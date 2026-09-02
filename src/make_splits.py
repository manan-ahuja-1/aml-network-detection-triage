"""Compute and FREEZE the temporal train/val/test boundaries (A3).

Run once; the boundaries are written to data/processed/split_boundaries.json and
every later stage reads them from there rather than recomputing.

WHY FREEZE THEM
---------------
If the boundaries are recomputed at each stage, any change to the input data
silently shifts them, and a model trained under one split gets evaluated under
another. Nothing errors; the metrics are simply wrong. Writing them once and
reading them thereafter makes the split an artifact of the project rather than a
side effect of whatever data happened to be present.

WHY THE SPLIT IS TEMPORAL
-------------------------
A random split lets the model learn from transactions that occur AFTER the ones it
scores. More subtly, it lets graph features encode edges that had not yet happened
at prediction time — so the model appears to know a mule's future counterparties.
Train on the earliest window, validate on the middle, test on the latest.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


def main() -> int:
    trans = pd.read_parquet(config.TRANS_PARQUET, columns=["timestamp", "is_laundering"])

    before = len(trans)
    trans = trans[trans["timestamp"] < pd.Timestamp(config.DATA_CUTOFF)]
    print(f"Truncated at {config.DATA_CUTOFF}: {before:,} -> {len(trans):,} rows "
          f"({(before-len(trans)):,} removed)")

    # Quantiles of the timestamp distribution, not calendar dates: daily volume here
    # varies by 5x, so calendar thirds would produce wildly uneven row counts.
    t_train_end = trans["timestamp"].quantile(config.TRAIN_FRACTION)
    t_val_end = trans["timestamp"].quantile(config.TRAIN_FRACTION + config.VAL_FRACTION)

    boundaries = {
        "data_cutoff": config.DATA_CUTOFF,
        "train_end": t_train_end.isoformat(),
        "val_end": t_val_end.isoformat(),
        "data_start": trans["timestamp"].min().isoformat(),
        "data_end": trans["timestamp"].max().isoformat(),
        "random_seed": config.RANDOM_SEED,
    }

    train = trans[trans["timestamp"] < t_train_end]
    val = trans[(trans["timestamp"] >= t_train_end) & (trans["timestamp"] < t_val_end)]
    test = trans[trans["timestamp"] >= t_val_end]

    # These asserts encode the correctness property itself. If a future change to the
    # data or fractions breaks temporal ordering, this fails here rather than
    # producing quietly invalid metrics three stages later.
    assert train["timestamp"].max() < val["timestamp"].min(), "train overlaps val"
    assert val["timestamp"].max() < test["timestamp"].min(), "val overlaps test"
    assert len(train) + len(val) + len(test) == len(trans), "split loses rows"

    for name, part in (("train", train), ("val", val), ("test", test)):
        boundaries[f"{name}_rows"] = int(len(part))
        boundaries[f"{name}_positives"] = int(part["is_laundering"].sum())
        boundaries[f"{name}_rate"] = float(part["is_laundering"].mean())
        print(f"  {name:6s} {len(part):>10,} rows  {int(part['is_laundering'].sum()):>6,} positives  "
              f"{part['is_laundering'].mean()*100:.4f}%")

    config.SPLIT_BOUNDARIES_JSON.write_text(json.dumps(boundaries, indent=2))
    print(f"\nWrote {config.SPLIT_BOUNDARIES_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
