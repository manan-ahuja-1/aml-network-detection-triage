"""Arm B, part 1 — per-account behavioural aggregates.

THE LEAK-FREE RULE (A1), established here and reused for graph features on Day 3
-------------------------------------------------------------------------------
Aggregates are computed from the TRAINING WINDOW ONLY. Validation and test rows then
*look up* their accounts' features from that table. An account absent from training
gets NaN plus an explicit `is_unseen` flag.

NaN, never zero. Zero would say "this account has no counterparties", which is a real
and different statement from "we have never seen this account". LightGBM handles NaN
natively by learning a default direction at each split, so the distinction survives
into the model.

WHY DEGREE LIVES IN THIS FILE AND NOT IN THE GRAPH ARM
------------------------------------------------------
`n_counterparties_in/out` is in/out degree. It is produced here by a groupby with no
graph library involved, so any competent baseline would include it. Putting it in arm
B means arm C has to earn its lift from genuine MULTI-HOP structure — PageRank,
betweenness, communities, cycles — which a groupby cannot produce. That is the harder
test for the graph thesis and the one that survives scrutiny.

KNOWN ASYMMETRY, documented rather than hidden
----------------------------------------------
A training row's aggregates include that row itself; a validation row's do not. This
is standard practice and mirrors production, where features come from the last
retraining window. It does make training features mildly optimistic relative to
validation. Recorded in docs/NOTES.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import features_txn  # noqa: E402

# Aggregates are stored as float32: LightGBM casts to float32 internally anyway, and
# at 5M rows x ~40 columns the halving is over a gigabyte of RAM.
_F32 = "float32"


def _side_aggregates(train: pd.DataFrame, actor: str, counterparty: str,
                     bank_col: str, prefix: str) -> pd.DataFrame:
    """Aggregate one direction of flow for every account.

    Called twice: once grouping by sender (outgoing flow) and once by receiver
    (incoming). `actor` is the account being described; `counterparty` is the other
    side, which is what makes degree and bank-diversity computable.
    """
    grouped = train.groupby(actor, observed=True)
    agg = grouped.agg(
        n=("paid_usd", "size"),
        total_usd=("paid_usd", "sum"),
        mean_usd=("paid_usd", "mean"),
        # std is NaN for single-transaction accounts. That is correct and meaningful:
        # "no variability observed" differs from "variability of zero".
        std_usd=("paid_usd", "std"),
        max_usd=("paid_usd", "max"),
        n_counterparties=(counterparty, "nunique"),
        n_banks=(bank_col, "nunique"),
        n_currencies=("payment_currency", "nunique"),
        n_formats=("payment_format", "nunique"),
        first_seen=("timestamp", "min"),
        last_seen=("timestamp", "max"),
    )

    # Velocity: transactions per active day. An account doing 200 transfers in a day
    # behaves differently from one doing 200 over a month, and raw counts cannot tell
    # them apart.
    active_days = (agg["last_seen"] - agg["first_seen"]).dt.total_seconds() / 86400.0
    agg["active_days"] = active_days
    agg["txn_per_day"] = agg["n"] / active_days.clip(lower=1.0 / 24.0)

    # Concentration: how much of this account's activity goes to its single busiest
    # counterparty. A pass-through mule funnels to few destinations.
    agg["counterparties_per_txn"] = agg["n_counterparties"] / agg["n"]

    agg = agg.drop(columns=["first_seen", "last_seen"])
    agg.columns = [f"{prefix}_{c}" for c in agg.columns]
    return agg.astype(_F32)


def build_account_table(train: pd.DataFrame) -> pd.DataFrame:
    """Per-account feature table, derived from the training window ONLY.

    The caller is responsible for passing a training-only frame; `build()` below
    enforces it rather than trusting convention.
    """
    train = features_txn.add_usd_amounts(train)

    out = _side_aggregates(train, "from_id", "to_id", "to_bank", "out")
    inn = _side_aggregates(train, "to_id", "from_id", "from_bank", "in")

    acct = out.join(inn, how="outer")
    acct.index.name = "node_id"

    # Accounts that only ever sent (or only received) have no rows on the other side.
    # Those counts are genuinely zero, not unknown, so filling them is correct here —
    # unlike the unseen-account case below, where NaN is the honest answer.
    for col in acct.columns:
        if col.endswith(("_n", "_n_counterparties", "_n_banks", "_n_currencies", "_n_formats")):
            acct[col] = acct[col].fillna(0)

    # Degree, the quantity that decides the arm B/C boundary.
    acct["degree_total"] = acct["in_n_counterparties"] + acct["out_n_counterparties"]

    # Flow balance: total in versus total out. The building block for the
    # pass-through ratio in features_typology.
    total_in = acct["in_total_usd"].fillna(0)
    total_out = acct["out_total_usd"].fillna(0)
    acct["net_flow_usd"] = (total_in - total_out).astype(_F32)
    acct["gross_flow_usd"] = (total_in + total_out).astype(_F32)

    return acct.astype(_F32)


def build(df: pd.DataFrame, account_table: pd.DataFrame) -> pd.DataFrame:
    """Attach account features to every transaction, for BOTH counterparties.

    A transaction is suspicious partly because of who sent it and partly because of
    who received it, so each row carries both sides' history under `from_`/`to_`
    prefixes.
    """
    out = pd.DataFrame(index=df.index)

    for side, id_col in (("from", "from_id"), ("to", "to_id")):
        # reindex() aligns by key and inserts NaN for accounts absent from the table.
        # That NaN IS the cold-start signal; it is deliberately not filled.
        joined = account_table.reindex(df[id_col].to_numpy())
        joined.index = df.index

        # The explicit flag matters because NaN alone is ambiguous to a reader of the
        # feature importance chart — this makes "never seen before" a first-class,
        # learnable feature rather than an inference from missingness.
        out[f"{side}_is_unseen"] = joined.isna().all(axis=1).astype("int8")

        for col in account_table.columns:
            out[f"{side}_{col}"] = joined[col]

    return out


def feature_names(account_table: pd.DataFrame) -> list[str]:
    """Column order produced by build(), so callers need not hardcode it."""
    names: list[str] = []
    for side in ("from", "to"):
        names.append(f"{side}_is_unseen")
        names.extend(f"{side}_{c}" for c in account_table.columns)
    return names


if __name__ == "__main__":
    import splits

    frame = splits.load_transactions()
    train = splits.train_frame(frame)
    print(f"building account table from {len(train):,} TRAINING rows only")

    table = build_account_table(train)
    print(f"account table: {table.shape[0]:,} accounts x {table.shape[1]} features")

    features = build(frame, table)
    print(f"joined: {features.shape[0]:,} rows x {features.shape[1]} features")

    for name in ("train", "val", "test"):
        mask = frame["split"] == name
        unseen = features.loc[mask, "from_is_unseen"].mean() * 100
        print(f"  {name:6s} sender unseen: {unseen:.2f}%")
