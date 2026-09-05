"""The seam between the engine and the agent: transaction scores -> account alerts.

WHY THIS FILE EXISTS SEPARATELY
-------------------------------
The model scores TRANSACTIONS, because `Is Laundering` labels transactions. An
investigator opens a case on an ACCOUNT. Those are different units, and the
conversion between them is a modelling decision with consequences — it decides what
"precision" even means downstream.

Putting it in its own module means the Day 5 evaluation and the Day 6+ agent consume
the SAME alert definition. Two implementations of "top 200 alerts" that disagree would
make the engine's reported precision and the agent's measured triage quality
incomparable, and nothing would error.

THE AGGREGATION RULE (config.ALERT_AGGREGATION = "max")
-------------------------------------------------------
An account's alert score is the HIGHEST score among the transactions it participates
in. Not the mean: laundering is a minority of even a mule account's activity — a
professional mule's whole purpose is to look ordinary in aggregate — so averaging
dilutes precisely the signal we are looking for. Max is also what a real monitoring
system does: one transaction breaching a threshold raises the alert.

BOTH SIDES OF A TRANSACTION ARE IMPLICATED
------------------------------------------
A high-scoring transaction A -> B contributes its score to A and to B. This is
deliberate. In a layering chain the pass-through account is often the one worth
investigating, and it is the RECEIVER on the transaction that exposed it. Attributing
the score only to the sender would systematically miss collection points, which is the
mule signature the whole project is built to find.

WHAT COUNTS AS A PRODUCTIVE ALERT
---------------------------------
An account is a true positive if at least one of its transactions IN THE SCORED SPLIT
is labelled laundering. "In the scored split" matters: an account that laundered in
training and behaved in test is not a productive test alert, because at test time
there is nothing there to find.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


def transaction_ids(frame: pd.DataFrame) -> pd.Series:
    """Stable, human-legible transaction identifiers.

    The raw dataset ships no transaction ID. The frame's index is the row position in
    the parquet, which `splits.load_transactions` preserves through truncation, so it
    is stable across runs for a fixed parquet. Formatting it as `T0001234` rather than
    a bare integer matters for the agent: `cited_transaction_ids` is validated by set
    membership, and a distinctive prefix makes a hallucinated citation obvious on
    sight rather than only under assertion.
    """
    return pd.Series([f"T{i:07d}" for i in frame.index], index=frame.index, dtype="string")


def account_alerts(frame: pd.DataFrame, scores: np.ndarray, split: str) -> pd.DataFrame:
    """Aggregate transaction scores into per-account alerts for one split.

    Returns one row per account active in `split`, sorted by descending alert score.
    """
    if config.ALERT_AGGREGATION != "max":
        raise NotImplementedError(
            f"config.ALERT_AGGREGATION is {config.ALERT_AGGREGATION!r}; only 'max' is "
            "implemented. Changing it changes what every downstream precision number "
            "means, so it fails loudly rather than silently falling back."
        )

    part = frame[frame["split"] == split]
    if len(part) != len(scores):
        raise ValueError(
            f"{len(scores):,} scores for {len(part):,} {split} transactions — the "
            "scores were computed on a different frame than the one passed here."
        )

    # Stack the two sides into one long table: each transaction appears twice, once
    # under its sender and once under its receiver.
    long = pd.concat([
        pd.DataFrame({"node_id": part["from_id"].to_numpy(), "score": scores,
                      "y": part["is_laundering"].to_numpy(), "side": "from"}),
        pd.DataFrame({"node_id": part["to_id"].to_numpy(), "score": scores,
                      "y": part["is_laundering"].to_numpy(), "side": "to"}),
    ], ignore_index=True)

    alerts = long.groupby("node_id", observed=True).agg(
        alert_score=("score", "max"),
        n_transactions=("score", "size"),
        n_laundering=("y", "sum"),
    )
    # A productive alert is one where there was in fact something to find, in this
    # split. Compliance calls this a "productive alert"; the negation is not "wrong",
    # it is a non-productive alert that still consumed analyst time.
    alerts["is_productive"] = (alerts["n_laundering"] > 0).astype("int8")

    return alerts.sort_values("alert_score", ascending=False)


def top_alerts(alerts: pd.DataFrame, n: int | None = None) -> pd.DataFrame:
    """The N highest-scoring accounts — the queue an L1 analyst would actually work."""
    n = config.ALERT_SET_SIZE if n is None else n
    return alerts.head(n)


def alert_evidence(frame: pd.DataFrame, node_id: str, split: str,
                   scores: pd.Series | None = None) -> pd.DataFrame:
    """Every transaction the alerted account participated in, within the scored split.

    This is the agent's evidence window and the ONLY set of transaction IDs a citation
    may legitimately reference (A9). Restricting it to the split is not a technicality:
    handing the agent training-window activity would let it justify a test-split alert
    with evidence the engine never saw.
    """
    part = frame[frame["split"] == split]
    mask = (part["from_id"] == node_id) | (part["to_id"] == node_id)
    evidence = part[mask].copy()
    evidence.insert(0, "txn_id", transaction_ids(evidence))
    if scores is not None:
        evidence["model_score"] = scores.reindex(evidence.index).to_numpy()
    evidence["direction"] = np.where(evidence["from_id"] == node_id, "OUT", "IN")
    return evidence.sort_values("timestamp")


def precision_at_alert_depth(alerts: pd.DataFrame, depths: tuple[int, ...]) -> dict:
    """Account-level precision@k — the number an operations lead actually budgets against.

    Reported alongside the transaction-level precision@k in metrics.py because they
    answer different questions: transaction-level asks "of the alerts we raise, how
    many are real", account-level asks "of the CASES we open, how many are worth
    opening". Compliance capacity is measured in cases.
    """
    total_productive = int(alerts["is_productive"].sum())
    out = {}
    for k in depths:
        k = min(k, len(alerts))
        head = alerts.head(k)
        hits = int(head["is_productive"].sum())
        out[str(k)] = {
            "k": k,
            "productive": hits,
            "precision": hits / k if k else 0.0,
            "recall": hits / total_productive if total_productive else 0.0,
        }
    return out


if __name__ == "__main__":
    import splits

    frame = splits.load_transactions()
    ids = transaction_ids(frame)
    print(f"{len(frame):,} transactions, ids {ids.iloc[0]} .. {ids.iloc[-1]}")
    for split in ("train", "val", "test"):
        part = frame[frame["split"] == split]
        accounts = pd.unique(np.concatenate([part["from_id"].to_numpy(),
                                             part["to_id"].to_numpy()]))
        print(f"  {split:6s} {len(part):>9,} txns  {len(accounts):>9,} distinct accounts")
