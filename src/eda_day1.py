"""Day 1 EDA: the measurements that decide the rest of the build.

Answers, with numbers rather than assumptions:
  1. Time span, and whether a 3-way temporal split is viable
  2. Class imbalance (justifies PR-AUC over ROC-AUC, and scale_pos_weight)
  3. Currency set -> the FX table config.py needs
  4. Payment format set -> the placement-signal feature group (B2)
  5. COLD-START RATE: what fraction of test activity involves accounts the training
     graph never saw. This is the single number that decides whether arms C/D can work.
  6. Whether Patterns.txt joins cleanly back to the transactions table
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

pd.set_option("display.width", 200)

trans = pd.read_parquet(config.TRANS_PARQUET)
patterns = pd.read_parquet(config.PATTERNS_PARQUET)

print("=" * 78)
print("1. TIME SPAN")
print("=" * 78)
tmin, tmax = trans["timestamp"].min(), trans["timestamp"].max()
span_days = (tmax - tmin).total_seconds() / 86400
print(f"  first transaction : {tmin}")
print(f"  last transaction  : {tmax}")
print(f"  span              : {span_days:.2f} days")
per_day = trans.groupby(trans["timestamp"].dt.date).size()
print(f"\n  transactions per day (n={len(per_day)} days):")
print(per_day.to_string())

print()
print("=" * 78)
print("2. CLASS IMBALANCE")
print("=" * 78)
n_pos = int(trans["is_laundering"].sum())
n = len(trans)
print(f"  laundering transactions : {n_pos:,}")
print(f"  total transactions      : {n:,}")
print(f"  positive rate           : {n_pos / n:.6f}  ({n_pos / n * 100:.4f}%)")
print(f"  imbalance ratio         : 1 in {n / n_pos:,.0f}")
print(f"  scale_pos_weight would be ~{(n - n_pos) / n_pos:,.1f}")

print()
print("=" * 78)
print("3. CURRENCIES (for the FX table)")
print("=" * 78)
cur = pd.concat([trans["receiving_currency"], trans["payment_currency"]]).value_counts()
print(cur.to_string())
xcur = (trans["receiving_currency"] != trans["payment_currency"]).mean()
print(f"\n  cross-currency transactions: {xcur * 100:.3f}%")
amt_mismatch = (trans["amount_received"] != trans["amount_paid"]).mean()
print(f"  amount_received != amount_paid: {amt_mismatch * 100:.3f}%")

print()
print("=" * 78)
print("4. PAYMENT FORMATS (placement-signal grouping, B2)")
print("=" * 78)
fmt = trans["payment_format"].value_counts()
laundering_by_fmt = trans.groupby("payment_format", observed=True)["is_laundering"].mean()
fmt_table = pd.DataFrame({"count": fmt, "laundering_rate": laundering_by_fmt}).sort_values(
    "count", ascending=False
)
fmt_table["laundering_rate"] = (fmt_table["laundering_rate"] * 100).round(4).astype(str) + "%"
print(fmt_table.to_string())

print()
print("=" * 78)
print("5. TEMPORAL SPLIT + COLD-START RATE  <-- decides arms C/D")
print("=" * 78)
# Split on TIME, at the 60% / 80% quantiles of the timestamp distribution. Using
# quantiles rather than calendar dates keeps the row counts near the intended
# proportions even if activity is uneven across days.
t60 = trans["timestamp"].quantile(0.60)
t80 = trans["timestamp"].quantile(0.80)
print(f"  train : {tmin}  ->  {t60}")
print(f"  val   : {t60}  ->  {t80}")
print(f"  test  : {t80}  ->  {tmax}")

train = trans[trans["timestamp"] < t60]
val = trans[(trans["timestamp"] >= t60) & (trans["timestamp"] < t80)]
test = trans[trans["timestamp"] >= t80]
for name, part in (("train", train), ("val", val), ("test", test)):
    rate = part["is_laundering"].mean()
    print(f"    {name:6s} {len(part):>10,} rows  ({len(part)/n*100:5.2f}%)   "
          f"laundering rate {rate*100:.4f}%  ({int(part['is_laundering'].sum()):,} positives)")

# The training GRAPH may only contain nodes seen in the training window (A1).
train_nodes = pd.Index(pd.concat([train["from_id"], train["to_id"]]).unique())
print(f"\n  distinct accounts in training window: {len(train_nodes):,}")

for name, part in (("val", val), ("test", test)):
    from_seen = part["from_id"].isin(train_nodes)
    to_seen = part["to_id"].isin(train_nodes)
    part_nodes = pd.Index(pd.concat([part["from_id"], part["to_id"]]).unique())
    unseen_nodes = (~part_nodes.isin(train_nodes)).sum()
    print(f"\n  {name.upper()}")
    print(f"    distinct accounts                     : {len(part_nodes):,}")
    print(f"    accounts unseen in train (cold-start) : {unseen_nodes:,} "
          f"({unseen_nodes/len(part_nodes)*100:.1f}%)")
    print(f"    rows where SENDER is cold-start       : {(~from_seen).mean()*100:.1f}%")
    print(f"    rows where RECEIVER is cold-start     : {(~to_seen).mean()*100:.1f}%")
    print(f"    rows where BOTH sides are known       : {(from_seen & to_seen).mean()*100:.1f}%")
    lp = part[part["is_laundering"] == 1]
    if len(lp):
        both_known = (lp["from_id"].isin(train_nodes) & lp["to_id"].isin(train_nodes)).mean()
        print(f"    LAUNDERING rows with both sides known : {both_known*100:.1f}%")

print()
print("=" * 78)
print("6. PATTERNS -> TRANSACTIONS JOIN")
print("=" * 78)
key = ["timestamp", "from_id", "to_id", "amount_paid"]
t_keyed = trans[trans["is_laundering"] == 1][key].drop_duplicates()
p_keyed = patterns[key].drop_duplicates()
merged = p_keyed.merge(t_keyed, on=key, how="inner")
print(f"  pattern transactions (unique keys) : {len(p_keyed):,}")
print(f"  matched in transactions table      : {len(merged):,} "
      f"({len(merged)/len(p_keyed)*100:.2f}%)")
print(f"\n  pattern type distribution:")
print(patterns.groupby("pattern_type", observed=True)
      .agg(transactions=("pattern_id", "size"), patterns=("pattern_id", "nunique"))
      .sort_values("patterns", ascending=False).to_string())
