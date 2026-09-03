"""Arm A — features computable from a single transaction row in isolation.

No aggregation, no history, no graph. This arm exists to show what a naive baseline
looks like; the real control is arm B.

A note on transforms: amounts here span 0 to ~2.8e10 USD and are wildly skewed, which
would normally argue for a log transform. It is NOT applied, because gradient-boosted
trees split on thresholds and are invariant to any monotonic transform — log(x) and x
produce identical splits. Adding both would only split feature importance between two
copies of the same signal and make the Day 5 importance chart harder to read.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

# Column names produced here, so build_features and the tests can agree on the arm A
# feature set without hardcoding it in three places.
FEATURES = [
    "paid_usd",
    "recv_usd",
    "amount_spread_usd",
    "is_cross_currency",
    "hour",
    "day_of_week",
    "payment_format",
    "payment_currency",
    "receiving_currency",
    "is_same_bank",
    "is_self_transaction",
]

CATEGORICAL = ["payment_format", "payment_currency", "receiving_currency"]


def add_usd_amounts(df: pd.DataFrame) -> pd.DataFrame:
    """Convert both amounts to USD.

    Necessary because currency varies BETWEEN rows: without this, an amount feature
    treats 500 Yen and 500 US Dollars as the same number. `require_fx_rates` raises
    if the table was never populated, rather than letting `.map()` produce NaN that
    LightGBM would silently accept.
    """
    fx = config.require_fx_rates()

    # .map on a categorical returns a categorical; cast to float so arithmetic works.
    paid_rate = df["payment_currency"].map(fx).astype("float64")
    recv_rate = df["receiving_currency"].map(fx).astype("float64")

    missing = paid_rate.isna().sum() + recv_rate.isna().sum()
    if missing:
        unknown = set(df.loc[paid_rate.isna(), "payment_currency"].unique()) | set(
            df.loc[recv_rate.isna(), "receiving_currency"].unique()
        )
        raise ValueError(f"currencies missing from config.FX_TO_USD: {sorted(unknown)}")

    df = df.copy()
    df["paid_usd"] = df["amount_paid"] * paid_rate
    df["recv_usd"] = df["amount_received"] * recv_rate
    return df


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Return the arm A feature matrix, row-aligned with the input."""
    df = add_usd_amounts(df)
    out = pd.DataFrame(index=df.index)

    out["paid_usd"] = df["paid_usd"]
    out["recv_usd"] = df["recv_usd"]

    # Non-zero only on cross-currency transfers. Stands in for the implied FX spread,
    # which is a place value can be hidden.
    out["amount_spread_usd"] = df["paid_usd"] - df["recv_usd"]
    out["is_cross_currency"] = (
        df["payment_currency"].astype(str) != df["receiving_currency"].astype(str)
    ).astype("int8")

    # Timing. Laundering automation tends not to respect business hours.
    out["hour"] = df["timestamp"].dt.hour.astype("int8")
    out["day_of_week"] = df["timestamp"].dt.dayofweek.astype("int8")

    out["payment_format"] = df["payment_format"]
    out["payment_currency"] = df["payment_currency"]
    out["receiving_currency"] = df["receiving_currency"]

    # Measured on Day 1: cross-bank transactions launder at 0.1012% vs 0.0120%
    # same-bank -- an 8x difference. Moving funds between institutions to break the
    # audit trail is textbook layering, and it shows up plainly here.
    out["is_same_bank"] = (
        df["from_bank"].astype(str) == df["to_bank"].astype(str)
    ).astype("int8")

    # 11.64% of rows are self-transfers (mostly Reinvestment), containing 8 laundering
    # transactions out of 590,819. A strong negative signal, and it matters again on
    # Day 3: self-loops distort degree and PageRank if not handled deliberately.
    out["is_self_transaction"] = (df["from_id"] == df["to_id"]).astype("int8")

    return out[FEATURES]


if __name__ == "__main__":
    import splits

    frame = splits.load_transactions()
    features = build(frame)
    print(f"arm A: {features.shape[0]:,} rows x {features.shape[1]} features")
    print(features.dtypes.to_string())
    print("\nnulls per column:")
    print(features.isna().sum().to_string())
