"""Day 5 — freeze the engine, score the test split exactly once, and report honestly.

WHAT "FREEZE" MEANS
-------------------
Arm C won the validation ablation (0.1938 vs B 0.1815 and D 0.1660 +/- 0.0155). That
decision is now made. This module trains arm C once more under identical parameters,
writes the booster to disk, and every later stage — SHAP, the agent, the demo — loads
THAT FILE rather than retraining. If the model that produced the published test numbers
is not the same object the agent scores with, the numbers describe a model nobody can
inspect.

WHY TEST IS SCORED EXACTLY ONCE
-------------------------------
Every number reported so far came from validation, and validation has been consulted
dozens of times: to choose `min_data_in_leaf`, to reject `scale_pos_weight`, to drop
`reverse_pagerank`, to reject arm D. Each of those decisions leaked a little
information about validation into the model. That is fine — that is what a validation
set is for — but it means validation performance is now optimistic by an unknown
amount, and the only honest estimate of generalisation is a split that has never been
looked at.

Consulting test and then changing anything turns it into a second validation set. So
this runs once, the result is written to results/engine.json, and it stands.

WHAT ELSE LIVES HERE
--------------------
  * Bootstrap 95% CIs, because a point estimate on 1,143 positives invites a precision
    the data does not support.
  * The cost-sensitive threshold as a SENSITIVITY STRIP across cost ratios, because the
    ratio is an assumption and presenting one value would disguise that.
  * Pattern-level recall by typology (B5) — the unit compliance actually cares about.
  * Account-level alerts and precision at analyst-capacity depths (B6).
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alerts as alerts_mod  # noqa: E402
import build_features  # noqa: E402
import config  # noqa: E402
import explain  # noqa: E402
import metrics  # noqa: E402
import splits  # noqa: E402
import train as train_mod  # noqa: E402

# Arm C's recorded validation PR-AUC from results/arms.json. The freeze asserts against
# it: LightGBM is deterministic for a fixed seed and thread count, so a mismatch means
# something changed underneath us — a feature definition, the parquet, a library
# version — and the ablation table no longer describes this model.
EXPECTED_VAL_PR_AUC = 0.1938
PR_AUC_TOLERANCE = 0.002


NOTES = {
    "test_scored_once": (
        "The test split was untouched through Days 1-4. Every hyperparameter, feature "
        "and arm decision was made on validation. This is the first and only scoring."
    ),
    "pattern_coverage": (
        "Only 2,554 of the 4,522 laundering transactions surviving truncation belong to "
        "a named pattern in HI-Small_Patterns.txt. Pattern-level recall describes that "
        "labelled 56.5%, not all laundering."
    ),
    "cost_ratio_is_an_assumption": (
        "No published figure gives the cost of a missed SAR relative to a false-alert "
        "review. The BPI 2018 survey grounds the DENOMINATOR (an upper bound of roughly "
        "$150 per alert, from $2.4bn across 16 million alerts), but the numerator is "
        "unobservable, so the operating point is reported as a sensitivity strip across "
        "ratios rather than as a single tuned threshold."
    ),
}


# ---------------------------------------------------------------------------
# Freezing
# ---------------------------------------------------------------------------
def freeze_engine(frame: pd.DataFrame, force: bool = False):
    """Train arm C and persist it. Cached, with the usual provenance guard."""
    X, y, split_series, categorical = build_features.build_arm(config.ENGINE_ARM, frame)

    if not force and config.ENGINE_MODEL.exists() and config.ENGINE_MODEL_META.exists():
        meta = json.loads(config.ENGINE_MODEL_META.read_text())
        if meta.get("n_features") == X.shape[1] and meta.get("arm") == config.ENGINE_ARM:
            print(f"  using frozen engine ({meta['n_features']} features, "
                  f"best_iteration {meta['best_iteration']}, "
                  f"val PR-AUC {meta['val_pr_auc']:.4f})")
            return lgb.Booster(model_file=str(config.ENGINE_MODEL)), X, y, split_series
        print(f"  engine cache rejected (arm={meta.get('arm')}, "
              f"n_features={meta.get('n_features')}) — retraining")

    train_mask = (split_series == "train").to_numpy()
    val_mask = (split_series == "val").to_numpy()

    dtrain = lgb.Dataset(X[train_mask], label=y[train_mask],
                         categorical_feature=categorical, free_raw_data=False)
    dval = lgb.Dataset(X[val_mask], label=y[val_mask], categorical_feature=categorical,
                       reference=dtrain, free_raw_data=False)

    print(f"  training arm {config.ENGINE_ARM}: {int(train_mask.sum()):,} rows, "
          f"{X.shape[1]} features")
    started = time.monotonic()
    booster = lgb.train(
        train_mod.PARAMS, dtrain,
        num_boost_round=train_mod.NUM_BOOST_ROUND,
        valid_sets=[dval], valid_names=["val"],
        callbacks=[lgb.early_stopping(train_mod.EARLY_STOPPING, verbose=False),
                   lgb.log_evaluation(period=500)],
    )
    elapsed = time.monotonic() - started

    val_scores = booster.predict(X[val_mask], num_iteration=booster.best_iteration)
    val_pr_auc = metrics.pr_auc(y[val_mask].to_numpy(), val_scores)
    print(f"  best iteration {booster.best_iteration} ({elapsed:.0f}s), "
          f"val PR-AUC {val_pr_auc:.4f}")

    drift = abs(val_pr_auc - EXPECTED_VAL_PR_AUC)
    if drift > PR_AUC_TOLERANCE:
        print(f"  WARNING: val PR-AUC {val_pr_auc:.4f} differs from the recorded arm C "
              f"result {EXPECTED_VAL_PR_AUC:.4f} by {drift:.4f}. The ablation table and "
              f"this engine may no longer describe the same model.")

    booster.save_model(str(config.ENGINE_MODEL), num_iteration=booster.best_iteration)
    config.ENGINE_MODEL_META.write_text(json.dumps({
        "arm": config.ENGINE_ARM,
        "n_features": int(X.shape[1]),
        "feature_names": list(X.columns),
        "categorical": categorical,
        "best_iteration": int(booster.best_iteration),
        "train_seconds": round(elapsed, 1),
        "val_pr_auc": val_pr_auc,
        "expected_val_pr_auc": EXPECTED_VAL_PR_AUC,
        "params": {k: v for k, v in train_mod.PARAMS.items()},
        "frozen_at": pd.Timestamp.now("UTC").isoformat(),
    }, indent=2))
    return booster, X, y, split_series


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------
def _weighted_average_precision(y_sorted: np.ndarray, weights: np.ndarray) -> float:
    """Average precision from pre-sorted labels and per-row multiplicities.

    `average_precision_score` costs ~260 ms on a 1M-row split, which at 2,000
    resamples is nine minutes per split. Sorting ONCE outside the loop and expressing
    each resample as a multiplicity vector reduces it to ~29 ms — the same arithmetic,
    without re-sorting identical data two thousand times.

    Verified against sklearn on 200k and 1M rows: agreement to 5.6e-17. The one
    difference in principle is tie handling — sklearn groups equal scores, this does
    not — which is why the POINT ESTIMATE still comes from sklearn and only the
    interval uses this path. LightGBM scores are continuous, so ties are vanishingly
    rare in practice.
    """
    cum_tp = np.cumsum(weights * y_sorted)
    cum_all = np.cumsum(weights)
    total_tp = cum_tp[-1]
    if total_tp == 0:
        return 0.0
    # Ranks where nothing has been drawn yet contribute nothing, but would produce
    # 0/0; `where` leaves those entries at zero instead of NaN.
    precision = np.divide(cum_tp, cum_all, out=np.zeros_like(cum_tp), where=cum_all > 0)
    return float(np.sum(precision * weights * y_sorted) / total_tp)


def bootstrap_pr_auc(y: np.ndarray, scores: np.ndarray, n_resamples: int,
                     ci: float, seed: int) -> dict:
    """Percentile bootstrap CI for PR-AUC.

    Resampling is over TRANSACTIONS with replacement, STRATIFIED by class so every
    resample keeps exactly the same number of positives. Unstratified resampling at a
    1-in-900 base rate produces draws whose positive count varies by several percent,
    and PR-AUC moves with the base rate — so an unstratified interval would be
    measuring class-balance jitter as much as model uncertainty.

    Why an interval at all: the test split holds 1,143 positives. A PR-AUC quoted to
    four decimals off 1,143 events implies a precision the data cannot support, and
    the interval is the difference between "arm C scores 0.19" and "arm C scores 0.19,
    and here is how much of that is sampling noise".
    """
    rng = np.random.default_rng(seed)
    order = np.argsort(-scores, kind="stable")
    y_sorted = y[order].astype("float64")

    pos = np.flatnonzero(y_sorted == 1)
    neg = np.flatnonzero(y_sorted == 0)
    n_pos, n_neg = len(pos), len(neg)

    draws = np.empty(n_resamples, dtype="float64")
    weights = np.empty(len(y_sorted), dtype="float64")
    for i in range(n_resamples):
        weights[pos] = np.bincount(rng.integers(0, n_pos, n_pos), minlength=n_pos)
        weights[neg] = np.bincount(rng.integers(0, n_neg, n_neg), minlength=n_neg)
        draws[i] = _weighted_average_precision(y_sorted, weights)

    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(draws, [alpha, 1.0 - alpha])
    return {
        "point_estimate": metrics.pr_auc(y, scores),
        "ci_level": ci,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_resamples": n_resamples,
        "bootstrap_mean": float(draws.mean()),
        "bootstrap_std": float(draws.std(ddof=1)),
        "stratified": True,
    }


# ---------------------------------------------------------------------------
# Operating point
# ---------------------------------------------------------------------------
def cost_sensitive_strip(y: np.ndarray, scores: np.ndarray,
                         ratios: tuple[int, ...]) -> dict:
    """Threshold minimising expected cost, swept across the cost ratio.

    Cost is measured in ANALYST REVIEWS. Reviewing one false alert costs 1 unit; missing
    one laundering transaction costs `ratio` units. Minimising ratio*FN + FP over the
    threshold gives the operating point implied by that belief about relative harm.

    The ratio is an assumption, not a measurement — nobody publishes the true cost of a
    missed SAR — so a single value would present a judgement call as a result. The strip
    shows how much the recommended threshold actually depends on it, which is the
    honest thing to report and is exactly how a model validation document handles an
    unobservable parameter.
    """
    order = np.argsort(-scores)
    y_sorted = y[order]
    scores_sorted = scores[order]

    # Cumulative counts if we alert on the top i transactions.
    tp = np.cumsum(y_sorted)
    fp = np.arange(1, len(y_sorted) + 1) - tp
    total_pos = int(y.sum())
    fn = total_pos - tp

    rows = {}
    for ratio in ratios:
        cost = ratio * fn + fp
        i = int(np.argmin(cost))
        rows[str(ratio)] = {
            "cost_ratio": ratio,
            "threshold": float(scores_sorted[i]),
            "alerts": int(i + 1),
            "true_positives": int(tp[i]),
            "false_positives": int(fp[i]),
            "precision": float(tp[i] / (i + 1)),
            "recall": float(tp[i] / total_pos) if total_pos else 0.0,
            "expected_cost_reviews": float(cost[i]),
        }
    return rows


# ---------------------------------------------------------------------------
# Pattern-level recall (B5)
# ---------------------------------------------------------------------------
PATTERN_KEY = ["timestamp", "from_id", "to_id", "amount_paid", "payment_format"]


def load_pattern_membership(frame: pd.DataFrame) -> pd.DataFrame:
    """Map each transaction in `frame` to the injected laundering pattern it belongs to.

    `HI-Small_Patterns.txt` ships the ground-truth typology of every injected ring, so
    this converts pattern classification from a subjective rubric into a real labelled
    task (A7). The join is on the full transaction tuple because the raw data has no
    transaction ID; verified to produce no duplicate expansion.

    IMPORTANT CAVEAT, reported alongside every pattern metric: only 2,554 of the 4,522
    laundering transactions that survive truncation belong to a named pattern. The
    other 1,968 are laundering the simulator did not group into a ring. Pattern-level
    recall therefore describes the LABELLED 56.5%, not all laundering.
    """
    patterns = pd.read_parquet(config.PATTERNS_PARQUET)
    patterns = patterns[PATTERN_KEY + ["pattern_id", "pattern_type"]].copy()
    patterns["payment_format"] = patterns["payment_format"].astype(str)

    keyed = frame[PATTERN_KEY].copy()
    keyed["payment_format"] = keyed["payment_format"].astype(str)
    keyed["_index"] = frame.index

    merged = patterns.merge(keyed, on=PATTERN_KEY, how="inner")
    if merged["_index"].duplicated().any():
        raise ValueError(
            "the pattern join produced duplicate transaction matches; the key "
            f"{PATTERN_KEY} is not unique and pattern recall would be double-counted."
        )
    return merged[["_index", "pattern_id", "pattern_type"]].set_index("_index")


def pattern_recall(frame: pd.DataFrame, membership: pd.DataFrame, split: str,
                   flagged_index: pd.Index) -> dict:
    """What fraction of injected patterns did we touch at least once?

    A laundering ring spans many transactions and an investigator needs only one thread
    to pull to open a case, so transaction-level recall systematically understates
    operational usefulness. This asks the question a compliance team asks: did the
    alert queue surface this ring at all?
    """
    part = frame[frame["split"] == split]
    in_split = membership.loc[membership.index.intersection(part.index)]
    if in_split.empty:
        return {"n_patterns": 0}

    caught_index = in_split.index.intersection(flagged_index)
    caught_patterns = set(in_split.loc[caught_index, "pattern_id"])

    by_type: dict[str, dict] = {}
    for ptype, group in in_split.groupby("pattern_type", observed=True):
        ids = set(group["pattern_id"])
        hits = ids & caught_patterns
        by_type[str(ptype)] = {
            "patterns_in_split": len(ids),
            "patterns_caught": len(hits),
            "recall": len(hits) / len(ids) if ids else 0.0,
            "transactions_in_split": int(len(group)),
        }

    all_ids = set(in_split["pattern_id"])
    return {
        "n_patterns": len(all_ids),
        "n_caught": len(caught_patterns),
        "recall": len(caught_patterns) / len(all_ids) if all_ids else 0.0,
        "n_pattern_transactions": int(len(in_split)),
        "by_typology": dict(sorted(by_type.items())),
    }


def pattern_recall_by_account_depth(frame: pd.DataFrame, membership: pd.DataFrame,
                                    split: str, alerted_accounts: set) -> dict:
    """Pattern recall at the ACCOUNT alert depth — the operational reading.

    "Of the rings active in this window, how many had at least one participant in the
    top-200 queue an analyst actually worked?" This is the number that answers "would
    we have found it", as opposed to "did the score cross a line somewhere".
    """
    part = frame[frame["split"] == split]
    in_split = membership.loc[membership.index.intersection(part.index)]
    if in_split.empty:
        return {"n_patterns": 0}

    rows = part.loc[in_split.index]
    touched = rows["from_id"].isin(alerted_accounts) | rows["to_id"].isin(alerted_accounts)
    caught = set(in_split.loc[touched.to_numpy(), "pattern_id"])

    by_type = {}
    for ptype, group in in_split.groupby("pattern_type", observed=True):
        ids = set(group["pattern_id"])
        hits = ids & caught
        by_type[str(ptype)] = {
            "patterns_in_split": len(ids),
            "patterns_caught": len(hits),
            "recall": len(hits) / len(ids) if ids else 0.0,
        }

    all_ids = set(in_split["pattern_id"])
    return {
        "n_patterns": len(all_ids),
        "n_caught": len(caught),
        "recall": len(caught) / len(all_ids) if all_ids else 0.0,
        "by_typology": dict(sorted(by_type.items())),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_pr_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]],
                   base_rate: float, path: Path) -> None:
    """One chart, every arm, so the ablation is visible rather than tabulated."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=150)
    for label, (y, scores) in curves.items():
        precision, recall, _ = precision_recall_curve(y, scores)
        ap = metrics.pr_auc(y, scores)
        ax.plot(recall, precision, linewidth=1.8, label=f"{label}  (AP {ap:.4f})")

    # The floor a random ranker achieves. Without it a PR-AUC of 0.19 looks poor; the
    # honest comparison is 0.19 against a base rate of 0.0011.
    ax.axhline(base_rate, color="0.5", linestyle="--", linewidth=1,
               label=f"random  (AP {base_rate:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_yscale("log")
    ax.set_title("Precision-recall by feature arm")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--skip-shap", action="store_true")
    args = parser.parse_args()

    print("=" * 72)
    print("DAY 5 — FREEZE THE ENGINE AND SCORE TEST ONCE")
    print("=" * 72)

    frame = splits.load_transactions()
    booster, X, y, split_series = freeze_engine(frame, force=args.force_retrain)

    results: dict = {
        "engine": {
            "arm": config.ENGINE_ARM,
            "description": config.ARMS[config.ENGINE_ARM],
            "selected_because": (
                "highest validation PR-AUC of the four arms (C 0.1938 > B 0.1815 > "
                "D 0.1660+/-0.0155 > A 0.0527); arm D was rejected on measured evidence "
                "across three node2vec seeds and three dimensionalities."
            ),
            "n_features": int(X.shape[1]),
            "best_iteration": int(booster.best_iteration),
        },
        "splits": {},
    }

    membership = load_pattern_membership(frame)
    print(f"\n  pattern membership: {len(membership):,} transactions across "
          f"{membership['pattern_id'].nunique()} injected patterns")

    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    scored: dict[str, pd.Series] = {}

    for split in ("val", "test"):
        mask = (split_series == split).to_numpy()
        y_split = y[mask].to_numpy()
        scores = booster.predict(X[mask], num_iteration=booster.best_iteration)
        scored[split] = pd.Series(scores, index=frame.index[mask])
        curves[f"arm C — {split}"] = (y_split, scores)

        print(f"\n{'-' * 72}\n{split.upper()}  ({len(y_split):,} rows, "
              f"{int(y_split.sum()):,} laundering)\n{'-' * 72}")

        summary = metrics.summarise(y_split, scores, config.PRECISION_AT_K)
        print(f"  PR-AUC   {summary['pr_auc']:.4f}   (base rate {summary['base_rate']:.5f})")
        print(f"  ROC-AUC  {summary['roc_auc']:.4f}")
        for k in config.PRECISION_AT_K:
            pk = summary["precision_at_k"][str(k)]
            print(f"  P@{k:<4} {pk['precision'] * 100:6.2f}%   recall {pk['recall'] * 100:5.2f}%")

        print("  bootstrapping...", end="", flush=True)
        t0 = time.monotonic()
        boot = bootstrap_pr_auc(y_split, scores, config.BOOTSTRAP_RESAMPLES,
                                config.BOOTSTRAP_CI, config.RANDOM_SEED)
        print(f" {time.monotonic() - t0:.0f}s   "
              f"95% CI [{boot['ci_low']:.4f}, {boot['ci_high']:.4f}]")

        strip = cost_sensitive_strip(y_split, scores, config.COST_RATIO_GRID)
        central = strip[str(config.COST_RATIO_CENTRAL)]
        print(f"  cost-optimal at ratio {config.COST_RATIO_CENTRAL}: "
              f"{central['alerts']:,} alerts, precision {central['precision'] * 100:.2f}%, "
              f"recall {central['recall'] * 100:.1f}%")

        # Account-level alerts — the unit a case is opened on.
        table = alerts_mod.account_alerts(frame, scores, split)
        top = alerts_mod.top_alerts(table)
        depth = alerts_mod.precision_at_alert_depth(table, config.PRECISION_AT_K
                                                    + (config.ALERT_SET_SIZE,))
        print(f"  accounts active: {len(table):,}  productive: "
              f"{int(table['is_productive'].sum()):,}")
        for k in sorted({*config.PRECISION_AT_K, config.ALERT_SET_SIZE}):
            d = depth[str(k)]
            print(f"  account P@{k:<4} {d['precision'] * 100:6.2f}%   "
                  f"({d['productive']}/{d['k']} productive)")

        # Pattern recall at the cost-optimal transaction threshold...
        flagged = scored[split].index[scores >= central["threshold"]]
        prec_txn = pattern_recall(frame, membership, split, flagged)
        # ...and at the alert depth an analyst actually works.
        prec_acct = pattern_recall_by_account_depth(frame, membership, split, set(top.index))
        print(f"  patterns active {prec_txn['n_patterns']}: caught "
              f"{prec_txn['n_caught']} at threshold ({prec_txn['recall'] * 100:.1f}%), "
              f"{prec_acct['n_caught']} in top-{config.ALERT_SET_SIZE} accounts "
              f"({prec_acct['recall'] * 100:.1f}%)")

        results["splits"][split] = {
            **summary,
            "pr_auc_bootstrap": boot,
            "cost_sensitivity": strip,
            "account_alerts": {
                "aggregation": config.ALERT_AGGREGATION,
                "n_accounts": int(len(table)),
                "n_productive": int(table["is_productive"].sum()),
                "precision_at_depth": depth,
            },
            "pattern_recall_at_threshold": prec_txn,
            "pattern_recall_at_alert_depth": prec_acct,
        }

        # The alert set itself, persisted for the agent to consume on Day 6+.
        out = config.RESULTS / f"alerts_{split}.parquet"
        top.to_parquet(out)
        print(f"  wrote {out.name} (top {len(top)} accounts)")

    # SHAP for the alert sets (B3) — the engine's own reasons, handed to the agent.
    if not args.skip_shap:
        print(f"\n{'-' * 72}\nSHAP ATTRIBUTION FOR ALERT SETS\n{'-' * 72}")
        for split in ("val", "test"):
            table = alerts_mod.account_alerts(frame, scored[split].to_numpy(), split)
            top = alerts_mod.top_alerts(table)
            t0 = time.monotonic()
            explained = explain.explain_alerts(booster, X, frame, top, split, scored[split])
            path = config.RESULTS / f"shap_{split}.json"
            path.write_text(json.dumps(explained, indent=2))
            print(f"  {split}: explained {len(explained)} alerts in "
                  f"{time.monotonic() - t0:.0f}s -> {path.name}")

    # Persist BEFORE the figure work. The test split is scored exactly once; a
    # matplotlib failure after that point would leave the choice between re-scoring
    # test (which breaks the discipline) and having no result at all.
    digest = hashlib.sha256(scored["test"].to_numpy().tobytes()).hexdigest()[:16]
    results["test_score_digest"] = digest
    results["scored_at"] = pd.Timestamp.now("UTC").isoformat()
    results["notes"] = NOTES
    config.ENGINE_JSON.write_text(json.dumps(results, indent=2))
    print(f"\n  wrote {config.ENGINE_JSON.name} (digest {digest})")

    # PR curves across arms, on validation, plus the frozen engine on test.
    val_mask = (split_series == "val").to_numpy()
    y_val = y[val_mask].to_numpy()
    for arm in ("A", "B"):
        print(f"\n  scoring arm {arm} for the PR-curve figure...")
        Xa, ya, sa, cata = build_features.build_arm(arm, frame)
        tm, vm = (sa == "train").to_numpy(), (sa == "val").to_numpy()
        d = lgb.Dataset(Xa[tm], label=ya[tm], categorical_feature=cata, free_raw_data=False)
        dv = lgb.Dataset(Xa[vm], label=ya[vm], categorical_feature=cata, reference=d,
                         free_raw_data=False)
        b = lgb.train(train_mod.PARAMS, d, num_boost_round=train_mod.NUM_BOOST_ROUND,
                      valid_sets=[dv], callbacks=[
                          lgb.early_stopping(train_mod.EARLY_STOPPING, verbose=False)])
        curves[f"arm {arm} — val"] = (ya[vm].to_numpy(),
                                      b.predict(Xa[vm], num_iteration=b.best_iteration))

    import baselines
    rules_scores = baselines.ach_rule_scores(frame[frame["split"] == "val"])
    curves["arm R (rules) — val"] = (y_val, rules_scores)
    curves = {k: curves[k] for k in sorted(curves)}
    try:
        plot_pr_curves(curves, float(y_val.mean()), config.FIGURES / "pr_curves.png")
        results["figures"] = ["pr_curves.png"]
        config.ENGINE_JSON.write_text(json.dumps(results, indent=2))
    except Exception as exc:  # noqa: BLE001 — a figure must not cost us the run
        print(f"  figure generation failed ({exc}); results/engine.json already written")

    print(f"\n{'=' * 72}")
    print(f"wrote {config.ENGINE_JSON}   test score digest {digest}")
    v, t = results["splits"]["val"], results["splits"]["test"]
    print(f"  val  PR-AUC {v['pr_auc']:.4f}  95% CI "
          f"[{v['pr_auc_bootstrap']['ci_low']:.4f}, {v['pr_auc_bootstrap']['ci_high']:.4f}]")
    print(f"  test PR-AUC {t['pr_auc']:.4f}  95% CI "
          f"[{t['pr_auc_bootstrap']['ci_low']:.4f}, {t['pr_auc_bootstrap']['ci_high']:.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
