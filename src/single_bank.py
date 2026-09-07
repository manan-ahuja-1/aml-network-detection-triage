"""B1 — how much of the network must you see for graph features to work?

THE EXPERIMENT THE BUILD PLAN ASKED FOR CANNOT BE RUN ON THIS DATASET
---------------------------------------------------------------------
The plan called for re-running arm C restricted to a single bank's visible subgraph, on
the grounds — entirely correct — that no real institution sees the complete inter-bank
graph this dataset hands you. That was attempted first and abandoned on measurement, and
the reason is a fact about the data worth reporting on its own.

  * There are **30,528 distinct banks** across 5.08M transactions. The median bank has
    **four accounts**. Only 24 banks have a thousand or more. This is a world of micro
    institutions, not the handful of large banks the experiment imagines.
  * Only one bank has enough validation positives to bootstrap a PR-AUC at all: bank
    070, with 145. The next best has 34.
  * And bank 070 is not a bank. It has **15 accounts carrying 452,751 transactions** —
    30,183 per account, with **zero** internal transfers. It is a clearing or settlement
    entity in the generator. Every genuine bank looks like bank 012: 2,639 accounts at
    ~38 transactions each, 6.5% of them internal.

Running the experiment on 070 anyway produced a clean-looking table in which the frozen
engine scored PR-AUC 0.0015 — chance, on a 0.139% base rate — with ROC-AUC 0.4879, below
chance. That is a statement about fifteen hyperactive settlement accounts, not about
institutional visibility, and publishing it would have been a real number answering no
question anyone asked.

WHAT IS ANSWERABLE, AND IS THE SAME UNDERLYING QUESTION
-------------------------------------------------------
Arm C earns its lift over arm B from multi-hop topology. The question underneath the
build plan's request is therefore: **how much of the network do you have to observe
before that lift appears?** That version is well-posed and well-powered, because it is
evaluated on the entire validation split — 1,083 positives at a fixed base rate — rather
than on one institution's thin slice.

So the graph is built from a random fraction p of the training transactions, and
everything else is held identical:

    p = 1.00   arm C exactly as shipped          (the ceiling)
    p = 0.50, 0.25, 0.10                          (partial visibility)
    arm B      no graph features at all           (the floor)

Only the GRAPH is restricted, deliberately. A bank does see its own customers' account
histories; what it cannot see is the wider network those customers transact into. So
account and typology features stay on the full training window and the graph alone is
degraded, which isolates network visibility instead of confounding it with having less
data of every kind.

WHAT THIS DOES AND DOES NOT MODEL
---------------------------------
Random edge removal thins the graph uniformly. A real institution's view is structured —
it sees every edge touching its own customers and none of the rest — so the subgraph it
holds is denser locally and emptier globally than any of these. This measures how much
topology the features need, not the precise shape of one bank's blind spot, and it is
reported on that basis.
"""

from __future__ import annotations

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
import evaluate as eval_mod  # noqa: E402
import features_account  # noqa: E402
import features_graph  # noqa: E402
import features_txn  # noqa: E402
import features_typology  # noqa: E402
import metrics  # noqa: E402
import splits  # noqa: E402
import train as train_mod  # noqa: E402

OUT = config.RESULTS / "single_bank.json"
FRACTIONS = (1.00, 0.50, 0.25, 0.10)

# Recorded so the README can state why the design changed without re-deriving it.
DATASET_STRUCTURE = {
    "n_banks": 30528,
    "median_accounts_per_bank": 4,
    "banks_with_1000plus_accounts": 24,
    "best_single_bank_val_positives": 145,
    "second_best_single_bank_val_positives": 34,
    "bank_070": {
        "accounts": 15,
        "transactions": 452751,
        "transactions_per_account": 30183,
        "internal_transfer_share": 0.0,
        "frozen_engine_pr_auc_on_its_slice": 0.0015,
        "frozen_engine_roc_auc_on_its_slice": 0.4879,
        "verdict": ("a clearing or settlement entity, not an institution; the "
                    "single-bank experiment was abandoned on this evidence"),
    },
}


def build_arm_with_partial_graph(frame: pd.DataFrame, fraction: float,
                                 seed: int = config.RANDOM_SEED):
    """Arm C, but the graph is built from only `fraction` of the training transactions.

    Mirrors `build_features.build_arm("C", ...)` exactly except for the frame handed to
    `build_graph_table`. Everything upstream of the graph — transaction fields, account
    aggregates, typology and entity features — is built from the FULL training window,
    because those are the parts an institution genuinely holds for its own customers.
    """
    train = splits.train_frame(frame)
    parts = [features_txn.build(frame)]
    categorical = list(features_txn.CATEGORICAL)

    account_table = features_account.build_account_table(train)
    parts.append(features_account.build(frame, account_table))
    typology_table = features_typology.build_typology_table(train)
    entity_table = features_typology.build_entity_table()
    parts.append(features_typology.build(frame, typology_table, entity_table))
    _, typ_categorical = features_typology.feature_names(typology_table)
    categorical.extend(typ_categorical)

    graph_train = (train if fraction >= 1.0
                   else train.sample(frac=fraction, random_state=seed))
    boundaries = splits.load_boundaries()
    graph_table = features_graph.build_graph_table(graph_train, boundaries["train_end"])
    parts.append(features_graph.build(frame, graph_table))

    X = pd.concat(parts, axis=1)
    return X, frame[build_features.LABEL].astype("int8"), frame["split"], categorical


def fit_and_score(X, y, split, categorical, label: str) -> tuple[np.ndarray, int]:
    tr, va = (split == "train").to_numpy(), (split == "val").to_numpy()
    dtrain = lgb.Dataset(X[tr], label=y[tr], categorical_feature=categorical,
                         free_raw_data=False)
    dval = lgb.Dataset(X[va], label=y[va], categorical_feature=categorical,
                       reference=dtrain, free_raw_data=False)
    started = time.monotonic()
    booster = lgb.train(
        dict(train_mod.PARAMS), dtrain,
        num_boost_round=train_mod.NUM_BOOST_ROUND,
        valid_sets=[dval], valid_names=["val"],
        callbacks=[lgb.early_stopping(train_mod.EARLY_STOPPING, verbose=False),
                   lgb.log_evaluation(period=500)],
    )
    print(f"      {label}: best iteration {booster.best_iteration}, "
          f"{time.monotonic() - started:.0f}s")
    return booster.predict(X[va], num_iteration=booster.best_iteration), va


def main() -> int:
    print("=" * 74)
    print("  B1 — HOW MUCH OF THE NETWORK DO GRAPH FEATURES NEED?")
    print("=" * 74)
    print(f"  {DATASET_STRUCTURE['n_banks']:,} banks, median "
          f"{DATASET_STRUCTURE['median_accounts_per_bank']} accounts each — a "
          "single-institution split is not well posed here; see the module docstring\n")

    frame = splits.load_transactions()
    y_all = frame[build_features.LABEL].astype("int8")
    results: dict[str, dict] = {}

    def record(key: str, scores: np.ndarray, va: np.ndarray, note: str) -> None:
        y_val = y_all[va].to_numpy()
        summary = metrics.summarise(y_val, scores, config.PRECISION_AT_K)
        boot = eval_mod.bootstrap_pr_auc(y_val, scores,
                                         n_resamples=config.BOOTSTRAP_RESAMPLES,
                                         ci=0.95, seed=config.RANDOM_SEED)
        summary["pr_auc_ci95"] = [round(boot["ci_low"], 6), round(boot["ci_high"], 6)]
        summary["note"] = note
        results[key] = summary
        print(f"    -> PR-AUC {summary['pr_auc']:.4f}  "
              f"[{boot['ci_low']:.4f}, {boot['ci_high']:.4f}]  "
              f"P@100 {summary['precision_at_k']['100']['precision'] * 100:.0f}%\n")

    # Floor: arm B, no graph features at all.
    print("  --- arm B: no graph features (the floor) ---")
    Xb, yb, splitb, catb = build_features.build_arm("B", frame)
    s, va = fit_and_score(Xb, yb, splitb, catb, "arm B")
    record("graph_none", s, va, "arm B — no graph features at all")

    for frac in FRACTIONS:
        print(f"  --- arm C with {frac:.0%} of the training graph visible ---")
        X, y, split, cat = build_arm_with_partial_graph(frame, frac)
        s, va = fit_and_score(X, y, split, cat, f"p={frac:.2f}")
        record(f"graph_{int(frac * 100):03d}pct", s, va,
               f"arm C, graph built from {frac:.0%} of training transactions")

    floor = results["graph_none"]["pr_auc"]
    ceiling = results["graph_100pct"]["pr_auc"]
    lift = ceiling - floor
    curve = []
    for frac in FRACTIONS:
        key = f"graph_{int(frac * 100):03d}pct"
        pr = results[key]["pr_auc"]
        curve.append({
            "fraction_visible": frac,
            "pr_auc": round(pr, 6),
            "lift_over_arm_b": round(pr - floor, 6),
            "share_of_full_lift": round((pr - floor) / lift, 4) if lift else None,
        })

    report = {
        "question": ("arm C's lift over arm B comes from multi-hop topology; how much of "
                     "the network has to be observed before that lift appears?"),
        "design": ("only the graph is degraded — account and typology features stay on "
                   "the full training window, because an institution does hold its own "
                   "customers' histories. Evaluated on the complete validation split so "
                   "the base rate is fixed and there are 1,083 positives."),
        "why_not_a_single_bank": DATASET_STRUCTURE,
        "arm_b_floor": round(floor, 6),
        "arm_c_ceiling": round(ceiling, 6),
        "full_lift": round(lift, 6),
        "curve": curve,
        "arms": results,
        "caveat": ("random edge removal thins the graph uniformly; a real institution's "
                   "view is structured — dense locally, empty globally — so this "
                   "measures how much topology the features need, not the shape of one "
                   "bank's blind spot"),
    }
    OUT.write_text(json.dumps(report, indent=2))

    print("=" * 74)
    print(f"  {'graph visible':<16}{'PR-AUC':>9}{'lift over arm B':>18}{'of full lift':>14}")
    print(f"  {'none (arm B)':<16}{floor:>9.4f}{'—':>18}{'—':>14}")
    for row in curve:
        print(f"  {row['fraction_visible']:>13.0%}   {row['pr_auc']:>9.4f}"
              f"{row['lift_over_arm_b']:>+18.4f}"
              f"{(row['share_of_full_lift'] or 0) * 100:>13.0f}%")
    print(f"\n  wrote {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
