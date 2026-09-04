"""Train one LightGBM model per arm and evaluate on VALIDATION.

The test split is deliberately untouched here. It is scored exactly once, on Day 5,
after the engine is frozen. A test set consulted while iterating is a training set
with extra steps.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_features  # noqa: E402
import config  # noqa: E402
import metrics  # noqa: E402
import splits  # noqa: E402

PARAMS = {
    "objective": "binary",
    # LightGBM's own name for average precision. Early stopping therefore optimises
    # the metric we actually report, rather than log-loss or AUC.
    "metric": "average_precision",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    # DELIBERATELY LARGE. With only ~2,300 positives in 3M training rows, a small leaf
    # can memorise a handful of laundering transactions. That produces a spurious spike
    # in validation average-precision on some early round, early stopping fires on it,
    # and training halts after 1-5 iterations with a bad model.
    #
    # Measured on arm B: min_data_in_leaf=20 stops at iteration 5 with PR-AUC 0.127;
    # =100 stops at iteration 1 with 0.091; =300 trains for 746 iterations and reaches
    # 0.170. The constraint is what makes training stable, not merely regularised.
    "min_data_in_leaf": 300,
    "verbose": -1,
    "seed": config.RANDOM_SEED,
    "num_threads": 8,
    #
    # NOTE ON CLASS IMBALANCE — scale_pos_weight is deliberately NOT set.
    #
    # The build plan called for it, and it is the standard advice for imbalanced
    # classification. It is wrong here, and measurably so. On arm A:
    #
    #     scale_pos_weight = 1325 (the true ratio)  -> stops at iter 1, PR-AUC 0.0119
    #     is_unbalance = True (equivalent)          -> stops at iter 1, PR-AUC 0.0119
    #     scale_pos_weight = 36 (sqrt of ratio)     -> stops at iter 1, PR-AUC 0.0464
    #     no reweighting at all                     -> 62 iterations,  PR-AUC 0.0495
    #
    # The reason: reweighting exists to fix CALIBRATION — to stop the loss ignoring a
    # rare class when you need trustworthy probabilities. PR-AUC does not use
    # probabilities, only their ORDER. Multiplying positive gradients by 1,325 distorts
    # every split so severely that the model wrecks its own ranking while chasing
    # calibration we never consume. We rank, so we do not reweight.
}

NUM_BOOST_ROUND = 2500
EARLY_STOPPING = 150


def train_arm(arm: str, frame: pd.DataFrame) -> dict:
    print(f"\n{'=' * 70}\nARM {arm}: {config.ARMS[arm]}\n{'=' * 70}")

    X, y, split, categorical = build_features.build_arm(arm, frame)

    train_mask = (split == "train").to_numpy()
    val_mask = (split == "val").to_numpy()

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    n_pos = int(y_train.sum())
    params = dict(PARAMS)  # no reweighting — see the note in PARAMS

    print(f"  train {len(X_train):,} rows ({n_pos:,} positive, 1 in {len(X_train)/n_pos:,.0f})")
    print(f"  val   {len(X_val):,} rows ({int(y_val.sum()):,} positive)")
    print(f"  {X.shape[1]} features, {len(categorical)} categorical")

    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical,
                         free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, categorical_feature=categorical,
                       reference=dtrain, free_raw_data=False)

    # monotonic(), not time(): on macOS the monotonic clock pauses while the
    # machine sleeps, so a laptop lid closed mid-run does not turn a 20-minute
    # training into a reported 21 hours. This bit us once — arm C's first run
    # recorded 77,896s of wall clock across an overnight suspend.
    started = time.monotonic()
    booster = lgb.train(
        params, dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dval], valid_names=["val"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )
    elapsed = time.monotonic() - started

    scores = booster.predict(X_val, num_iteration=booster.best_iteration)
    summary = metrics.summarise(y_val.to_numpy(), scores, config.PRECISION_AT_K)

    importance = pd.Series(
        booster.feature_importance(importance_type="gain"), index=booster.feature_name()
    ).sort_values(ascending=False)

    print(f"\n  best iteration {booster.best_iteration} ({elapsed:.0f}s)")
    print(f"  PR-AUC   {summary['pr_auc']:.4f}")
    print(f"  ROC-AUC  {summary['roc_auc']:.4f}   <- high and misleading; see METHODOLOGY §1")
    for k in config.PRECISION_AT_K:
        pk = summary["precision_at_k"][str(k)]
        print(f"  P@{k:<4} {pk['precision']*100:6.2f}%  ({pk['hits']}/{pk['k']} hits, "
              f"recall {pk['recall']*100:.1f}%)")
    a80 = summary["alerts_at_80_recall"]
    if a80["achievable"]:
        print(f"  alerts to review for 80% recall: {a80['alerts_to_review']:,}")

    print("\n  top 15 features by gain:")
    for name, gain in importance.head(15).items():
        print(f"    {gain:14,.0f}  {name}")

    return {
        "arm": arm,
        "description": config.ARMS[arm],
        "n_features": int(X.shape[1]),
        "best_iteration": int(booster.best_iteration),
        "train_seconds": round(elapsed, 1),

        "val": summary,
        "top_features": {k: float(v) for k, v in importance.head(25).items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", default="A,B,C", help="comma-separated arms to train")
    args = parser.parse_args()

    frame = splits.load_transactions()

    import baselines
    rules = baselines.evaluate(frame, "val")
    print(f"\n{'=' * 70}\nARM R: {rules['description']}\n{'=' * 70}")
    print(f"  precision {rules['precision']*100:.4f}%   recall {rules['recall']*100:.2f}%   "
          f"alerts {rules['alerts_raised']:,}")

    out = config.RESULTS / "arms.json"
    # Merge rather than overwrite: retraining one arm should not discard the others,
    # which each cost several minutes. Arms are compared under identical
    # hyperparameters, so previously-trained arms stay valid as long as PARAMS is
    # unchanged — the stored params below make that checkable.
    results = json.loads(out.read_text()) if out.exists() else {"arms": {}}
    results["arm_R"] = rules
    results.setdefault("arms", {})
    results["params"] = {k: v for k, v in PARAMS.items() if not isinstance(v, dict)}
    results["num_boost_round"] = NUM_BOOST_ROUND
    results["early_stopping"] = EARLY_STOPPING

    for arm in args.arms.split(","):
        results["arms"][arm.strip()] = train_arm(arm.strip(), frame)
    results["arms"] = {k: results["arms"][k] for k in sorted(results["arms"])}
    out.write_text(json.dumps(results, indent=2))

    print(f"\n{'=' * 70}\nSUMMARY (validation)\n{'=' * 70}")
    print(f"  {'arm':4} {'PR-AUC':>9}  {'P@100':>8}  {'recall@100':>11}  features")
    print(f"  {'R':4} {'—':>9}  {rules['precision']*100:7.2f}%  "
          f"{rules['recall']*100:10.1f}%  (rule, {rules['alerts_raised']:,} alerts)")
    for arm, r in results["arms"].items():
        p100 = r["val"]["precision_at_k"]["100"]
        print(f"  {arm:4} {r['val']['pr_auc']:9.4f}  {p100['precision']*100:7.2f}%  "
              f"{p100['recall']*100:10.1f}%  {r['n_features']}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
