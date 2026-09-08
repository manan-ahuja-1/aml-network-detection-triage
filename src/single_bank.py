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
import evaluate as eval_mod  # noqa: E402
import features_account  # noqa: E402
import features_graph  # noqa: E402
import features_txn  # noqa: E402
import features_typology  # noqa: E402
import metrics  # noqa: E402
import splits  # noqa: E402
import train as train_mod  # noqa: E402

OUT = config.RESULTS / "single_bank.json"
SCORES_OUT = config.RESULTS / "single_bank_scores.parquet"
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


def fit_and_score(X, y, split, categorical, label: str):
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
    # best_iteration is reported alongside PR-AUC because it turned out to be a second,
    # independent read on the same phenomenon: where early stopping peaks says how fast
    # the model starts overfitting the features it was given.
    return (booster.predict(X[va], num_iteration=booster.best_iteration), va,
            int(booster.best_iteration))


def graph_size_of(frame: pd.DataFrame, fraction: float,
                  seed: int = config.RANDOM_SEED) -> dict:
    """Nodes and edges in the sampled graph, and the edges-per-node ratio.

    Recorded because it is the mechanism behind the recovery at 10% visibility: once the
    graph has fewer edges than nodes it is mostly isolated fragments, PageRank and
    betweenness go near-degenerate, and the model stops being misled by them.
    """
    train = splits.train_frame(frame)
    sub = train if fraction >= 1.0 else train.sample(frac=fraction, random_state=seed)
    edges = sub.loc[sub["from_id"] != sub["to_id"], ["from_id", "to_id"]]
    n_edges = int(edges.drop_duplicates().shape[0])
    n_nodes = int(pd.concat([edges["from_id"], edges["to_id"]]).nunique())
    return {"nodes": n_nodes, "edges": n_edges,
            "edges_per_node": round(n_edges / n_nodes, 3) if n_nodes else None}


def paired_deltas(scores: dict[str, np.ndarray], y: np.ndarray,
                  baseline: str = "graph_none",
                  n_resamples: int = config.BOOTSTRAP_RESAMPLES) -> dict:
    """Bootstrap the DIFFERENCE in PR-AUC between each arm and the no-graph floor.

    Every arm is scored on identical rows, so comparing two marginal 95% CIs throws away
    the pairing and is badly conservative — the intervals here overlap heavily while the
    differences may not straddle zero at all. Drawing ONE resample and evaluating every
    arm on it keeps the correlation and estimates the difference directly.

    Reuses `evaluate._weighted_average_precision`, which exists because
    `average_precision_score` costs ~260ms on a 1M-row split and this loop would
    otherwise take hours. The trick it relies on — sort once, express a resample as a
    multiplicity vector — extends to the paired case as long as the SAME per-row counts
    are permuted into each arm's own sort order, which is what `orders` below is for.

    Stratified by class for the reason `bootstrap_pr_auc` documents: at a 1-in-937 base
    rate an unstratified draw varies its positive count by several percent, and PR-AUC
    moves with the base rate, so the interval would measure class-balance jitter.
    """
    keys = list(scores)
    orders = {k: np.argsort(-scores[k], kind="stable") for k in keys}
    y_sorted = {k: y[orders[k]].astype("float64") for k in keys}

    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    n_pos, n_neg = len(pos), len(neg)
    rng = np.random.default_rng(config.RANDOM_SEED)

    counts = np.empty(len(y), dtype="float64")
    draws = {k: np.empty(n_resamples) for k in keys}
    for i in range(n_resamples):
        counts[pos] = np.bincount(rng.integers(0, n_pos, n_pos), minlength=n_pos)
        counts[neg] = np.bincount(rng.integers(0, n_neg, n_neg), minlength=n_neg)
        for k in keys:
            draws[k][i] = eval_mod._weighted_average_precision(
                y_sorted[k], counts[orders[k]])

    out: dict[str, dict] = {}
    for k in keys:
        if k == baseline:
            continue
        d = draws[k] - draws[baseline]
        lo, hi = np.percentile(d, [2.5, 97.5])
        point = metrics.pr_auc(y, scores[k]) - metrics.pr_auc(y, scores[baseline])
        out[k] = {
            "delta_vs_no_graph": round(float(point), 6),
            "ci95": [round(float(lo), 6), round(float(hi), 6)],
            "excludes_zero": bool(lo > 0 or hi < 0),
            "share_of_draws_worse_than_no_graph": round(float((d < 0).mean()), 4),
        }
    return out


def replicate(seed: int, fractions: tuple[float, ...]) -> int:
    """Re-run the partial-visibility arms under a different edge subsample.

    WHAT VARYING THE SEED ACTUALLY VARIES
    -------------------------------------
    `train.sample(frac=..., random_state=seed)` — that is, WHICH EDGES ARE VISIBLE.
    LightGBM is deterministic given its data (confirmed: seed 42 reproduced identical best
    iterations and PR-AUCs across three full runs), so all spread across seeds is
    attributable to the visibility draw and not to training noise. That is the whole point:
    a single subsample cannot distinguish "25% visibility is bad" from "this particular
    draw of edges was unlucky".

    TWO ARMS ARE DELIBERATELY NOT RE-RUN
    ------------------------------------
    Arm B has no graph and the 100% arm does no sampling, so neither depends on the seed.
    Arm B's validation scores are read back from `single_bank_scores.parquet` instead of
    being retrained, which is ~471 seconds saved per seed and is only possible because the
    main run persists its score vectors.
    """
    if not SCORES_OUT.exists():
        print(f"missing {SCORES_OUT.name} — run `python src/single_bank.py` first; the "
              "replicates are compared against its arm B floor")
        return 1

    baseline = pd.read_parquet(SCORES_OUT)
    y_val = baseline["y"].to_numpy()
    print("=" * 74)
    print(f"  B1 REPLICATE — seed {seed}, fractions {fractions}")
    print("=" * 74)
    print(f"  arm B floor read from {SCORES_OUT.name} "
          f"(PR-AUC {metrics.pr_auc(y_val, baseline['graph_none'].to_numpy()):.4f}); "
          "it does not depend on the seed\n")

    frame = splits.load_transactions()
    y_all = frame[build_features.LABEL].astype("int8")
    results: dict[str, dict] = {}
    graph_sizes: dict[str, dict] = {}
    scores = {"graph_none": baseline["graph_none"].to_numpy()}

    for frac in fractions:
        key = f"graph_{int(frac * 100):03d}pct"
        print(f"  --- {frac:.0%} of the training graph visible (seed {seed}) ---")
        X, y, split, cat = build_arm_with_partial_graph(frame, frac, seed=seed)
        graph_sizes[key] = graph_size_of(frame, frac, seed=seed)
        s, va, it = fit_and_score(X, y, split, cat, f"p={frac:.2f} seed={seed}")
        scores[key] = s
        summary = metrics.summarise(y_all[va].to_numpy(), s, config.PRECISION_AT_K)
        boot = eval_mod.bootstrap_pr_auc(y_all[va].to_numpy(), s,
                                         n_resamples=config.BOOTSTRAP_RESAMPLES,
                                         ci=0.95, seed=config.RANDOM_SEED)
        summary["pr_auc_ci95"] = [round(boot["ci_low"], 6), round(boot["ci_high"], 6)]
        summary["best_iteration"] = it
        results[key] = summary
        print(f"    -> PR-AUC {summary['pr_auc']:.4f}  "
              f"[{boot['ci_low']:.4f}, {boot['ci_high']:.4f}]  "
              f"P@100 {summary['precision_at_k']['100']['precision'] * 100:.0f}%\n")

    out = config.RESULTS / f"single_bank_seed{seed}.json"
    # MERGE rather than overwrite. A pre-flight runs one cheap fraction to check the shape
    # before committing to the rest; overwriting would throw that fraction away and it
    # costs ~14 minutes to recompute. Existing entries for the fractions just run are
    # replaced, everything else is kept.
    report = json.loads(out.read_text()) if out.exists() else {"seed": seed}
    report["arms"] = {**report.get("arms", {}), **results}
    report["graph_sizes"] = {**report.get("graph_sizes", {}), **graph_sizes}
    report["paired_vs_no_graph"] = {**report.get("paired_vs_no_graph", {}),
                                    **paired_deltas(scores, y_val)}
    report["fractions"] = sorted({*report.get("fractions", []), *fractions}, reverse=True)
    out.write_text(json.dumps(report, indent=2))
    print("  paired against the no-graph floor:")
    for key, d in report["paired_vs_no_graph"].items():
        verdict = ("worse" if d["ci95"][1] < 0 else
                   "better" if d["ci95"][0] > 0 else "not separated from zero")
        print(f"    {key:<16} {d['delta_vs_no_graph']:+.4f}  "
              f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  {verdict}")
    print(f"\n  wrote {out.name}")
    return 0


def aggregate() -> int:
    """Merge the seed replicates into single_bank.json.

    WHAT IS AND IS NOT CLAIMED FROM THREE SEEDS
    -------------------------------------------
    Three subsamples per fraction supports a mean, a range, and a count of how many
    replicates separate from the no-graph floor. It does not support a t-test, and running
    one on n=3 would dress up the same three numbers as an inference they cannot carry.
    So the summary reports exactly those three things and the per-seed values behind them.

    The count is the load-bearing part. A shape that appears in one subsample and not the
    other two is that subsample; a shape that appears in all three is the visibility level.
    """
    main_report = json.loads(OUT.read_text())
    replicate_files = sorted(config.RESULTS.glob("single_bank_seed*.json"))
    if not replicate_files:
        print("no single_bank_seed*.json files found — nothing to aggregate")
        return 1

    runs = [json.loads(f.read_text()) for f in replicate_files]
    by_fraction: dict[str, dict] = {}
    partial = [f"graph_{int(f * 100):03d}pct" for f in FRACTIONS if f < 1.0]

    for key in partial:
        entries = []
        # seed 42 is the main run, already in `arms` / `paired_vs_no_graph`
        if key in main_report["arms"]:
            entries.append({
                "seed": config.RANDOM_SEED,
                "pr_auc": main_report["arms"][key]["pr_auc"],
                "best_iteration": main_report["arms"][key].get("best_iteration"),
                "paired": main_report.get("paired_vs_no_graph", {}).get(key),
            })
        for run in runs:
            if key in run.get("arms", {}):
                entries.append({
                    "seed": run["seed"],
                    "pr_auc": run["arms"][key]["pr_auc"],
                    "best_iteration": run["arms"][key].get("best_iteration"),
                    "paired": run.get("paired_vs_no_graph", {}).get(key),
                })
        if not entries:
            continue
        prs = [e["pr_auc"] for e in entries]
        deltas = [e["paired"]["delta_vs_no_graph"] for e in entries if e["paired"]]
        separated = [e for e in entries
                     if e["paired"] and e["paired"]["excludes_zero"]]
        by_fraction[key] = {
            "n_seeds": len(entries),
            "seeds": [e["seed"] for e in entries],
            "pr_auc_mean": round(sum(prs) / len(prs), 6),
            "pr_auc_min": round(min(prs), 6),
            "pr_auc_max": round(max(prs), 6),
            "pr_auc_range": round(max(prs) - min(prs), 6),
            "delta_mean": round(sum(deltas) / len(deltas), 6) if deltas else None,
            "n_separated_from_floor": len(separated),
            "all_below_floor": all(d < 0 for d in deltas) if deltas else None,
            "best_iterations": [e["best_iteration"] for e in entries],
            "per_seed": entries,
        }

    main_report["seed_replicates"] = {
        "note": ("each seed is a different draw of WHICH EDGES ARE VISIBLE; LightGBM is "
                 "deterministic given its data, so all spread here is the visibility "
                 "draw. Arm B and the 100% arm are not replicated because neither "
                 "depends on the sampling seed."),
        "claim_limits": ("three seeds supports a mean, a range and a count of replicates "
                         "separating from the floor — not a significance test on n=3"),
        "by_fraction": by_fraction,
    }
    OUT.write_text(json.dumps(main_report, indent=2))

    print(f"  merged {len(replicate_files)} replicate file(s)\n")
    print(f"  {'fraction':<16}{'seeds':>6}{'mean':>9}{'range':>9}"
          f"{'mean delta':>12}{'separated':>11}")
    for key, d in by_fraction.items():
        print(f"  {key:<16}{d['n_seeds']:>6}{d['pr_auc_mean']:>9.4f}"
              f"{d['pr_auc_range']:>9.4f}{d['delta_mean']:>+12.4f}"
              f"{d['n_separated_from_floor']:>7} of {d['n_seeds']}")
    print(f"\n  wrote {OUT.name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", action="store_true",
                        help="merge results/single_bank_seed*.json into single_bank.json")
    parser.add_argument("--seed", type=int, default=None,
                        help="run the partial-visibility arms under a different edge "
                             "subsample; writes results/single_bank_seed{N}.json")
    parser.add_argument("--fractions", type=str, default=None,
                        help="comma-separated subset, e.g. 0.25 — for a cheap pre-flight "
                             "before committing to the full set")
    args = parser.parse_args()
    if args.aggregate:
        return aggregate()
    fractions = (tuple(float(x) for x in args.fractions.split(","))
                 if args.fractions else tuple(f for f in FRACTIONS if f < 1.0))
    if args.seed is not None:
        return replicate(args.seed, fractions)

    print("=" * 74)
    print("  B1 — HOW MUCH OF THE NETWORK DO GRAPH FEATURES NEED?")
    print("=" * 74)
    print(f"  {DATASET_STRUCTURE['n_banks']:,} banks, median "
          f"{DATASET_STRUCTURE['median_accounts_per_bank']} accounts each — a "
          "single-institution split is not well posed here; see the module docstring\n")

    frame = splits.load_transactions()
    y_all = frame[build_features.LABEL].astype("int8")
    results: dict[str, dict] = {}
    graph_sizes: dict[str, dict] = {}

    # Per-row validation scores for every arm, kept so the arms can be compared
    # PAIRED. Each arm's marginal 95% CI is wide — they overlap heavily — but the arms
    # are scored on identical rows, so the interesting quantity is the distribution of
    # the DIFFERENCE, which a paired bootstrap estimates far more tightly than two
    # independent intervals suggest. Persisting the vectors means that test can be run
    # later without repeating hours of graph construction.
    score_columns: dict[str, np.ndarray] = {}

    def record(key: str, scores: np.ndarray, va: np.ndarray, note: str,
               best_iteration: int) -> None:
        score_columns[key] = scores
        y_val = y_all[va].to_numpy()
        summary = metrics.summarise(y_val, scores, config.PRECISION_AT_K)
        boot = eval_mod.bootstrap_pr_auc(y_val, scores,
                                         n_resamples=config.BOOTSTRAP_RESAMPLES,
                                         ci=0.95, seed=config.RANDOM_SEED)
        summary["pr_auc_ci95"] = [round(boot["ci_low"], 6), round(boot["ci_high"], 6)]
        summary["note"] = note
        summary["best_iteration"] = best_iteration
        results[key] = summary
        print(f"    -> PR-AUC {summary['pr_auc']:.4f}  "
              f"[{boot['ci_low']:.4f}, {boot['ci_high']:.4f}]  "
              f"P@100 {summary['precision_at_k']['100']['precision'] * 100:.0f}%\n")

    # Floor: arm B, no graph features at all.
    print("  --- arm B: no graph features (the floor) ---")
    Xb, yb, splitb, catb = build_features.build_arm("B", frame)
    s, va, it = fit_and_score(Xb, yb, splitb, catb, "arm B")
    record("graph_none", s, va, "arm B — no graph features at all", it)

    for frac in FRACTIONS:
        print(f"  --- arm C with {frac:.0%} of the training graph visible ---")
        X, y, split, cat = build_arm_with_partial_graph(frame, frac)
        graph_sizes[f"graph_{int(frac * 100):03d}pct"] = graph_size_of(frame, frac)
        s, va, it = fit_and_score(X, y, split, cat, f"p={frac:.2f}")
        record(f"graph_{int(frac * 100):03d}pct", s, va,
               f"arm C, graph built from {frac:.0%} of training transactions", it)

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
        "graph_sizes": graph_sizes,
        "caveat": ("random edge removal thins the graph uniformly; a real institution's "
                   "view is structured — dense locally, empty globally — so this "
                   "measures how much topology the features need, not the shape of one "
                   "bank's blind spot"),
    }
    report["paired_vs_no_graph"] = paired_deltas(
        score_columns, y_all[(frame["split"] == "val").to_numpy()].to_numpy())
    report["paired_note"] = (
        "each arm is scored on identical rows, so the marginal CIs above overlap far "
        "more than the paired differences do; the paired test is the one to read")
    OUT.write_text(json.dumps(report, indent=2))
    val_index = frame.index[(frame["split"] == "val").to_numpy()]
    pd.DataFrame(score_columns, index=val_index).assign(
        y=y_all[(frame["split"] == "val").to_numpy()].to_numpy()
    ).to_parquet(SCORES_OUT)

    print("=" * 74)
    print(f"  {'graph visible':<16}{'PR-AUC':>9}{'lift over arm B':>18}{'of full lift':>14}")
    print(f"  {'none (arm B)':<16}{floor:>9.4f}{'—':>18}{'—':>14}")
    for row in curve:
        print(f"  {row['fraction_visible']:>13.0%}   {row['pr_auc']:>9.4f}"
              f"{row['lift_over_arm_b']:>+18.4f}"
              f"{(row['share_of_full_lift'] or 0) * 100:>13.0f}%")
    print("\n  paired against the no-graph floor (same rows, difference bootstrapped):")
    for key, d in report["paired_vs_no_graph"].items():
        verdict = ("worse" if d["ci95"][1] < 0 else
                   "better" if d["ci95"][0] > 0 else "not separated from zero")
        print(f"    {key:<16} {d['delta_vs_no_graph']:+.4f}  "
              f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  {verdict}")
    print(f"\n  wrote {OUT.name} and {SCORES_OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
