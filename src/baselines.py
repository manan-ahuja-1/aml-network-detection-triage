"""Arm R — the rules baseline.

WHY A RULES BASELINE EXISTS AT ALL
----------------------------------
Two reasons, one domain and one statistical.

Domain: real AML stacks begin with a rules engine. A bank does not replace working
rules with a model unless the model beats them, so "we beat the rules" is the
question a fintech interviewer actually asks. Reporting PR-AUC without a rules floor
answers a question nobody posed.

Statistical: Day 1 EDA found that the simulator injects laundering almost exclusively
as ACH — 11.8% of transactions carry 84.7% of laundering, and Wire and Reinvestment
contain zero laundering across 652,911 rows. So "flag every ACH" is a very strong
trivial rule. Every model arm has to clear it, and quantifying it is how the artifact
gets measured rather than silently exploited.

This arm has no model, no training, and no parameters.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import metrics  # noqa: E402
import splits  # noqa: E402


def ach_rule_scores(frame: pd.DataFrame) -> np.ndarray:
    """Score 1.0 for ACH, 0.0 otherwise.

    A binary score rather than a probability. PR-AUC on a binary score is degenerate
    — there is only one operating point — so the honest reading of arm R is its
    precision and recall at that single point, not its curve area.
    """
    return (frame["payment_format"].astype(str) == "ACH").astype("float64").to_numpy()


def evaluate(frame: pd.DataFrame, split: str = "val") -> dict:
    part = frame[frame["split"] == split]
    y = part["is_laundering"].to_numpy()
    scores = ach_rule_scores(part)

    flagged = scores == 1.0
    n_flagged = int(flagged.sum())
    hits = int(y[flagged].sum())
    positives = int(y.sum())

    return {
        "arm": "R",
        "description": config.ARMS["R"],
        "split": split,
        "n": int(len(part)),
        "positives": positives,
        "alerts_raised": n_flagged,
        "alerts_share_of_all": n_flagged / len(part),
        "true_positives": hits,
        "precision": hits / n_flagged if n_flagged else 0.0,
        "recall": hits / positives if positives else 0.0,
        # Included for comparability with the model arms, with the caveat above:
        # on a binary score this is not a curve, and it should not be read as one.
        "pr_auc_degenerate": metrics.pr_auc(y, scores),
    }


if __name__ == "__main__":
    frame = splits.load_transactions()
    result = evaluate(frame, "val")
    print("Arm R — flag every ACH (validation split)\n")
    print(f"  transactions        {result['n']:,}")
    print(f"  laundering          {result['positives']:,}")
    print(f"  alerts raised       {result['alerts_raised']:,} "
          f"({result['alerts_share_of_all']*100:.2f}% of all traffic)")
    print(f"  true positives      {result['true_positives']:,}")
    print(f"  precision           {result['precision']*100:.4f}%")
    print(f"  recall              {result['recall']*100:.2f}%")
    print(f"\n  For every real case caught, an analyst reviews "
          f"{result['alerts_raised']/max(result['true_positives'],1):.0f} alerts.")
