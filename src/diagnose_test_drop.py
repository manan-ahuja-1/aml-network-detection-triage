"""Why is test PR-AUC roughly half of validation? (Day 5 follow-up)

THE OBSERVATION
---------------
The frozen engine scores 0.1938 on validation and 0.0949 on test — a 51% relative
drop. Validation was consulted dozens of times across Days 2-4 (min_data_in_leaf,
scale_pos_weight, reverse_pagerank, the arm choice), so some optimism was expected.
A halving is more than selection effect can plausibly explain, so it gets diagnosed
rather than shrugged at.

THE HYPOTHESIS
--------------
Every account-derived feature — arm B aggregates, arm C topology — is computed on the
TRAINING WINDOW ONLY. That is the leak-free rule, and it is correct. But it has a
consequence that grows with time: the further a scoring window sits from the training
window, the more of its accounts the model has never seen, and for those accounts every
one of the 92 account-derived features is NaN.

Validation covers 2022-09-06 13:34 to 2022-09-08 16:09. Test covers 09-08 16:09 to
09-10 23:59 — strictly further away. And test contains 356,263 distinct accounts against
validation's 236,109 on essentially the same transaction count, which is the shape of a
cold-start problem rather than a modelling one.

WHAT THIS SCRIPT MEASURES
-------------------------
The test drop decomposed into (a) how much is cold-start coverage and (b) how much is
genuine degradation on accounts the model does know. It does this by scoring the SAME
frozen engine on comparable subsets of each split, so nothing new is fitted and the
test split is not consulted for any decision — only described.

This is a diagnostic, not a fix. Reporting the drop and its mechanism is the honest
outcome; quietly re-tuning until test looks better would turn test into a second
validation set.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_features  # noqa: E402
import config  # noqa: E402
import metrics  # noqa: E402
import splits  # noqa: E402


def coverage_profile(X: pd.DataFrame, mask: np.ndarray) -> dict:
    """Cold-start rates for one split."""
    part = X[mask]
    return {
        "n": int(len(part)),
        "sender_unseen_pct": float(part["from_is_unseen"].mean() * 100),
        "receiver_unseen_pct": float(part["to_is_unseen"].mean() * 100),
        "either_unseen_pct": float(
            ((part["from_is_unseen"] == 1) | (part["to_is_unseen"] == 1)).mean() * 100),
        "both_in_graph_pct": float(
            ((part["from_is_in_graph"] == 1) & (part["to_is_in_graph"] == 1)).mean() * 100),
        "pagerank_missing_pct": float(part["from_pagerank"].isna().mean() * 100),
    }


def main() -> int:
    booster = lgb.Booster(model_file=str(config.ENGINE_MODEL))
    frame = splits.load_transactions()
    X, y, split_series, _ = build_features.build_arm(config.ENGINE_ARM, frame)

    report: dict = {"coverage": {}, "pr_auc_by_coverage": {}, "base_rates": {}}

    print("=" * 72)
    print("WHY TEST PR-AUC IS HALF OF VALIDATION")
    print("=" * 72)

    print("\nCOLD-START COVERAGE (features are fitted on the training window only)")
    print(f"  {'split':6} {'rows':>10} {'sender':>9} {'receiver':>9} {'either':>9} "
          f"{'both in graph':>14}")
    for split in ("train", "val", "test"):
        mask = (split_series == split).to_numpy()
        profile = coverage_profile(X, mask)
        report["coverage"][split] = profile
        print(f"  {split:6} {profile['n']:>10,} "
              f"{profile['sender_unseen_pct']:>8.2f}% {profile['receiver_unseen_pct']:>8.2f}% "
              f"{profile['either_unseen_pct']:>8.2f}% {profile['both_in_graph_pct']:>13.2f}%")

    print("\nPR-AUC DECOMPOSED BY COVERAGE")
    print("  Scored with the SAME frozen engine on comparable subsets, so any remaining")
    print("  gap is degradation the model suffers on accounts it HAS seen.")
    print(f"\n  {'split':6} {'subset':22} {'rows':>10} {'pos':>6} {'base rate':>10} {'PR-AUC':>8}")
    for split in ("val", "test"):
        mask = (split_series == split).to_numpy()
        part_X, part_y = X[mask], y[mask].to_numpy()
        scores = booster.predict(part_X, num_iteration=booster.best_iteration)

        known = ((part_X["from_is_unseen"] == 0) & (part_X["to_is_unseen"] == 0)).to_numpy()
        in_graph = ((part_X["from_is_in_graph"] == 1)
                    & (part_X["to_is_in_graph"] == 1)).to_numpy()

        report["pr_auc_by_coverage"][split] = {}
        for label, subset in (("all rows", np.ones(len(part_y), dtype=bool)),
                              ("both accounts seen", known),
                              ("both in graph", in_graph)):
            n_pos = int(part_y[subset].sum())
            if n_pos < 20:
                print(f"  {split:6} {label:22} {int(subset.sum()):>10,} {n_pos:>6}  "
                      f"{'—':>9} {'too few positives':>8}")
                continue
            value = metrics.pr_auc(part_y[subset], scores[subset])
            rate = float(part_y[subset].mean())
            report["pr_auc_by_coverage"][split][label] = {
                "n": int(subset.sum()), "positives": n_pos,
                "base_rate": rate, "pr_auc": value,
            }
            print(f"  {split:6} {label:22} {int(subset.sum()):>10,} {n_pos:>6} "
                  f"{rate:>10.5f} {value:>8.4f}")

    # Positives specifically: are the laundering transactions in test more often
    # cold-start than those in validation? If so, the model literally cannot see them.
    print("\nCOLD START AMONG THE POSITIVES (the rows we are trying to catch)")
    for split in ("val", "test"):
        mask = (split_series == split).to_numpy()
        part_X, part_y = X[mask], y[mask].to_numpy()
        pos = part_y == 1
        unseen = ((part_X["from_is_unseen"] == 1)
                  | (part_X["to_is_unseen"] == 1)).to_numpy()
        share = float(unseen[pos].mean() * 100)
        report["base_rates"][split] = {
            "positives": int(pos.sum()),
            "positives_with_an_unseen_account_pct": share,
        }
        print(f"  {split:6} {int(pos.sum()):>5} positives, "
              f"{share:5.2f}% involve an account the model never saw in training")

    # ------------------------------------------------------------------
    # Time decay: is the gap a CLIFF at the val/test boundary, or a SLOPE?
    # ------------------------------------------------------------------
    # Coverage explained none of the drop, so the two candidates left are (a) temporal
    # degradation — every account feature describes training-window behaviour, and that
    # description goes stale — and (b) validation optimism, because early stopping chose
    # iteration 2064 precisely to maximise validation AP.
    #
    # These predict different shapes. Optimism is a STEP: validation is inflated, test is
    # honest, and nothing varies within either. Temporal decay is a SLOPE: performance
    # falls steadily with distance from the training window, and the val/test boundary is
    # just where we happened to cut. Slicing the post-training period into equal windows
    # and scoring each one separates them.
    print("\nTIME DECAY: PR-AUC BY WINDOW SINCE TRAINING ENDED")
    post = split_series.isin(["val", "test"]).to_numpy()
    times = frame.loc[post, "timestamp"]
    X_post, y_post = X[post], y[post].to_numpy()
    scores_post = booster.predict(X_post, num_iteration=booster.best_iteration)

    edges = pd.date_range(times.min(), times.max(), periods=7)
    report["time_decay"] = []
    print(f"  {'window':32} {'split':10} {'rows':>9} {'pos':>5} {'base':>9} {'PR-AUC':>8}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        window = ((times >= lo) & (times < hi)).to_numpy()
        n_pos = int(y_post[window].sum())
        if n_pos < 30:
            continue
        value = metrics.pr_auc(y_post[window], scores_post[window])
        which = split_series[post][window].value_counts()
        label = "val" if which.get("val", 0) > which.get("test", 0) else "test"
        mix = f"{label} ({which.max() / which.sum() * 100:.0f}%)"
        report["time_decay"].append({
            "start": str(lo), "end": str(hi), "mostly": label,
            "n": int(window.sum()), "positives": n_pos,
            "base_rate": float(y_post[window].mean()), "pr_auc": value,
        })
        print(f"  {str(lo)[5:16]} -> {str(hi)[5:16]}  {mix:10} {int(window.sum()):>9,} "
              f"{n_pos:>5} {y_post[window].mean():>9.5f} {value:>8.4f}")

    out = config.ENGINE_JSON
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing["test_drop_diagnosis"] = report
    out.write_text(json.dumps(existing, indent=2))
    print(f"\nmerged into {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
