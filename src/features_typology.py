"""Arm B, part 2 — features motivated by named laundering behaviours, plus entity data.

WHAT WAS DROPPED, AND WHY IT MATTERS
------------------------------------
The build plan called for structuring features (amounts clustering just under the
$10,000 US CTR reporting threshold) and round-number-amount flags. Both were tested
against the data before being written, and both were dropped:

  structuring   laundering enriched 2.79% in $9-10k vs 2.96% in $10-11k. Real
                structuring clusters BELOW a threshold and falls off sharply above it.
                Here the enrichment is the same either side, which means it is not a
                threshold effect at all -- it is just laundering amounts skewing larger
                (median $5,104 vs $872). The generator does not model reporting
                thresholds.

  round numbers only 0.013% of amounts are multiples of 100, and 0.000% of laundering
                is a multiple of 1000. Amounts are continuous with cents.

Shipping them would have been domain-authentic theatre: features that signal knowledge
of AML typologies while detecting nothing. Testing each behaviour for existence before
building a feature for it is the point.

WHAT SURVIVED, WITH HONEST EXPECTATIONS
---------------------------------------
The pass-through ratio -- the canonical mule signature -- was also tested and is WEAK
here: median 0.251 for clean accounts vs 0.261 for accounts involved in laundering.
It is kept because it is cheap and genuinely motivated, but it is not expected to
carry the arm, and Day 5's feature importance will say so either way.

ENTITY FEATURES
---------------
From HI-Small_accounts.csv, found on Day 1. This is STATIC REFERENCE DATA -- the
account-to-customer mapping a bank holds at all times -- not something derived from
future transactions, so joining it wholesale is not leakage. It matters because real
AML investigates a customer who may hold many accounts across many banks: 38.2% of
entities here hold more than one account and 38.1% span more than one bank.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import features_txn  # noqa: E402

_F32 = "float32"

# Payment formats grouped by AML stage rather than left as anonymous categories.
# Cash and crypto are classic PLACEMENT-stage indicators; ACH and Wire are the rails
# used for LAYERING. Grouping them this way is what lets the feature importance chart
# be read in domain terms on Day 5.
PLACEMENT_FORMATS = ["Cash", "Bitcoin"]
LAYERING_FORMATS = ["ACH", "Wire"]


def build_typology_table(train: pd.DataFrame) -> pd.DataFrame:
    """Per-account typology features, from the TRAINING WINDOW ONLY."""
    train = features_txn.add_usd_amounts(train)

    # ---- pass-through / mule signature -----------------------------------------
    total_in = train.groupby("to_id", observed=True)["paid_usd"].sum()
    total_out = train.groupby("from_id", observed=True)["paid_usd"].sum()
    flows = pd.DataFrame({"in_usd": total_in, "out_usd": total_out}).fillna(0.0)

    lo = np.minimum(flows["in_usd"], flows["out_usd"])
    hi = np.maximum(flows["in_usd"], flows["out_usd"])
    # 1.0 means everything received was passed on; 0.0 means funds only moved one way.
    flows["flow_through_ratio"] = np.where(hi > 0, lo / hi, 0.0)
    # What fraction of inbound value was retained rather than forwarded. A mule
    # retains almost nothing.
    flows["retention_ratio"] = np.where(
        flows["in_usd"] > 0,
        (flows["in_usd"] - flows["out_usd"]).clip(lower=0) / flows["in_usd"],
        np.nan,
    )
    table = flows[["flow_through_ratio", "retention_ratio"]].copy()
    table.index.name = "node_id"

    # ---- time to forward --------------------------------------------------------
    # For every incoming transaction, how long until this account next sends money?
    # merge_asof with direction="forward" pairs each inbound event with the next
    # outbound event for the SAME account -- an O(n log n) operation over the whole
    # table rather than a per-account Python loop.
    ins = (train[["to_id", "timestamp"]]
           .rename(columns={"to_id": "node_id"})
           .sort_values("timestamp"))
    outs = (train[["from_id", "timestamp"]]
            .rename(columns={"from_id": "node_id", "timestamp": "out_time"})
            .sort_values("out_time"))

    paired = pd.merge_asof(
        ins, outs,
        left_on="timestamp", right_on="out_time",
        by="node_id", direction="forward", allow_exact_matches=False,
    )
    gap_hours = (paired["out_time"] - paired["timestamp"]).dt.total_seconds() / 3600.0
    paired["gap_hours"] = gap_hours
    ttf = paired.groupby("node_id", observed=True)["gap_hours"].median()
    table["median_hours_to_forward"] = ttf

    # ---- dormancy and burstiness ------------------------------------------------
    # Every event this account took part in, either direction, in time order.
    events = pd.concat([
        train[["from_id", "timestamp"]].rename(columns={"from_id": "node_id"}),
        train[["to_id", "timestamp"]].rename(columns={"to_id": "node_id"}),
    ]).sort_values(["node_id", "timestamp"])

    gaps = events.groupby("node_id", observed=True)["timestamp"].diff()
    events["gap_h"] = gaps.dt.total_seconds() / 3600.0
    gap_stats = events.groupby("node_id", observed=True)["gap_h"].agg(["median", "std", "mean"])

    table["median_gap_hours"] = gap_stats["median"]
    # Coefficient of variation of inter-transaction gaps. High = bursty: long quiet
    # spells punctuated by flurries, which is how a dormant mule account activates.
    table["gap_burstiness"] = gap_stats["std"] / gap_stats["mean"].replace(0, np.nan)

    # ---- payment-format mix (placement vs layering) -----------------------------
    fmt = train.groupby(["from_id", "payment_format"], observed=True).size().unstack(fill_value=0)
    fmt_share = fmt.div(fmt.sum(axis=1).replace(0, np.nan), axis=0)

    present = [c for c in PLACEMENT_FORMATS if c in fmt_share.columns]
    table["placement_format_share"] = fmt_share[present].sum(axis=1) if present else 0.0
    present = [c for c in LAYERING_FORMATS if c in fmt_share.columns]
    table["layering_format_share"] = fmt_share[present].sum(axis=1) if present else 0.0

    return table.astype(_F32)


def build_entity_table() -> pd.DataFrame:
    """Account -> entity metadata. Static reference data, not derived from transactions."""
    accounts = pd.read_parquet(config.DATA_PROCESSED / "accounts.parquet")

    per_entity = accounts.groupby("entity_id").agg(
        entity_n_accounts=("node_id", "size"),
        entity_n_banks=("bank_id", "nunique"),
    )
    merged = accounts.merge(per_entity, left_on="entity_id", right_index=True, how="left")

    table = merged.set_index("node_id")[["entity_type", "entity_n_accounts", "entity_n_banks"]]
    table["entity_n_accounts"] = table["entity_n_accounts"].astype(_F32)
    table["entity_n_banks"] = table["entity_n_banks"].astype(_F32)
    # Index is unique because node_id is the composite (bank_id, account_number) key
    # established on Day 1 -- account number alone would collide for 8 accounts.
    return table[~table.index.duplicated(keep="first")]


def build(df: pd.DataFrame, typology_table: pd.DataFrame,
          entity_table: pd.DataFrame) -> pd.DataFrame:
    """Attach typology and entity features for both counterparties."""
    out = pd.DataFrame(index=df.index)

    for side, id_col in (("from", "from_id"), ("to", "to_id")):
        keys = df[id_col].to_numpy()

        typ = typology_table.reindex(keys)
        typ.index = df.index
        for col in typology_table.columns:
            out[f"{side}_{col}"] = typ[col]

        ent = entity_table.reindex(keys)
        ent.index = df.index
        # entity_type is categorical; reindex preserves the dtype and unmatched
        # accounts become NaN, which LightGBM treats as its own category.
        out[f"{side}_entity_type"] = ent["entity_type"].astype("category")
        out[f"{side}_entity_n_accounts"] = ent["entity_n_accounts"]
        out[f"{side}_entity_n_banks"] = ent["entity_n_banks"]

    return out


def feature_names(typology_table: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Returns (all feature names, categorical feature names)."""
    names: list[str] = []
    categorical: list[str] = []
    for side in ("from", "to"):
        names.extend(f"{side}_{c}" for c in typology_table.columns)
        names.append(f"{side}_entity_type")
        categorical.append(f"{side}_entity_type")
        names.extend([f"{side}_entity_n_accounts", f"{side}_entity_n_banks"])
    return names, categorical


if __name__ == "__main__":
    import splits

    frame = splits.load_transactions()
    train = splits.train_frame(frame)

    typ = build_typology_table(train)
    print(f"typology table: {typ.shape[0]:,} accounts x {typ.shape[1]} features")
    print(typ.describe().round(3).to_string())

    ent = build_entity_table()
    print(f"\nentity table: {ent.shape[0]:,} accounts")

    feats = build(frame, typ, ent)
    print(f"\njoined: {feats.shape[0]:,} rows x {feats.shape[1]} features")
    print(f"null rate on from_entity_type: {feats['from_entity_type'].isna().mean()*100:.2f}%")
