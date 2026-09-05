"""Regenerate the PR-curve figure, caching each arm's scores so it is cheap to redraw.

WHY THIS IS SEPARATE FROM evaluate.py
-------------------------------------
Drawing the ablation chart needs arm A and arm B scores, and those arms are not the
engine — they exist only for comparison. Retraining them costs about thirteen minutes,
which is a long time to pay every time a colour or an axis label changes. So the scores
are cached to parquet on first run and the figure is redrawn from the cache after that.

WHY ARM R IS A POINT, NOT A CURVE
---------------------------------
The rules baseline scores 1.0 for ACH and 0.0 otherwise. It has exactly ONE operating
point. Handing that to `precision_recall_curve` produces a straight line from (0, 1) to
(1, base_rate), because the function interpolates between the single real threshold and
the degenerate endpoints — and on a log axis that line visually dominates arms A and C
across most of the recall range.

It is an artifact of plotting a binary score as a curve, and it says the opposite of
what is true: the rule raises 122,876 alerts at 0.75% precision, which is the queue the
model arms exist to shrink. So arm R is drawn as the single marker it actually is.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baselines  # noqa: E402
import build_features  # noqa: E402
import config  # noqa: E402
import metrics  # noqa: E402
import splits  # noqa: E402
import train as train_mod  # noqa: E402

SCORE_CACHE = config.RESULTS / "curve_scores.parquet"


def arm_val_scores(arm: str, frame: pd.DataFrame) -> np.ndarray:
    """Train one comparison arm and return its validation scores."""
    X, y, split_series, categorical = build_features.build_arm(arm, frame)
    tm = (split_series == "train").to_numpy()
    vm = (split_series == "val").to_numpy()

    dtrain = lgb.Dataset(X[tm], label=y[tm], categorical_feature=categorical,
                         free_raw_data=False)
    dval = lgb.Dataset(X[vm], label=y[vm], categorical_feature=categorical,
                       reference=dtrain, free_raw_data=False)
    started = time.monotonic()
    booster = lgb.train(train_mod.PARAMS, dtrain,
                        num_boost_round=train_mod.NUM_BOOST_ROUND,
                        valid_sets=[dval],
                        callbacks=[lgb.early_stopping(train_mod.EARLY_STOPPING,
                                                      verbose=False)])
    print(f"  arm {arm}: iteration {booster.best_iteration}, "
          f"{time.monotonic() - started:.0f}s")
    return booster.predict(X[vm], num_iteration=booster.best_iteration)


def collect_scores(frame: pd.DataFrame, rebuild: bool = False) -> pd.DataFrame:
    """Validation scores for arms A, B, C plus the frozen engine on test."""
    if SCORE_CACHE.exists() and not rebuild:
        print(f"  using cached scores ({SCORE_CACHE.name})")
        return pd.read_parquet(SCORE_CACHE)

    X, y, split_series, _ = build_features.build_arm(config.ENGINE_ARM, frame)
    vm = (split_series == "val").to_numpy()
    tm = (split_series == "test").to_numpy()

    booster = lgb.Booster(model_file=str(config.ENGINE_MODEL))
    engine_val = booster.predict(X[vm], num_iteration=booster.best_iteration)
    engine_test = booster.predict(X[tm], num_iteration=booster.best_iteration)

    # val and test have different row counts, so the frame is padded with NaN rather
    # than truncated — losing test rows to make the columns line up would silently
    # change the test curve.
    n = max(int(vm.sum()), int(tm.sum()))

    def pad(a):
        out = np.full(n, np.nan)
        out[:len(a)] = a
        return out

    data = {
        "y_val": pad(y[vm].to_numpy()),
        "y_test": pad(y[tm].to_numpy()),
        "C_val": pad(engine_val),
        "C_test": pad(engine_test),
    }
    for arm in ("A", "B"):
        data[f"{arm}_val"] = pad(arm_val_scores(arm, frame))

    table = pd.DataFrame(data)
    table.to_parquet(SCORE_CACHE)
    print(f"  wrote {SCORE_CACHE.name}")
    return table


def plot(table: pd.DataFrame, frame: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    y_val = table["y_val"].dropna().to_numpy()
    y_test = table["y_test"].dropna().to_numpy()

    series = [
        ("arm A — transaction only", table["A_val"], y_val, "#8c8c8c", "-"),
        ("arm B — + account/typology", table["B_val"], y_val, "#1f6fb4", "-"),
        ("arm C — + graph (engine), val", table["C_val"], y_val, "#c0392b", "-"),
        ("arm C — + graph (engine), TEST", table["C_test"], y_test, "#c0392b", "--"),
    ]

    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=150)
    for label, scores, y, colour, style in series:
        scores = scores.dropna().to_numpy()
        precision, recall, _ = precision_recall_curve(y, scores)
        ax.plot(recall, precision, linewidth=1.9, color=colour, linestyle=style,
                label=f"{label}  (AP {metrics.pr_auc(y, scores):.4f})")

    # Arm R: ONE point, because a binary score has one operating point. See module
    # docstring — drawing it as a curve inverts what it says.
    rules = baselines.evaluate(frame, "val")
    ax.plot(rules["recall"], rules["precision"], marker="D", markersize=8,
            color="#e67e22", linestyle="none",
            label=(f"arm R — flag every ACH  ({rules['alerts_raised']:,} alerts, "
                   f"{rules['precision'] * 100:.2f}% precision)"))

    base = float(y_val.mean())
    ax.axhline(base, color="0.55", linestyle=":", linewidth=1.1,
               label=f"random  (AP {base:.4f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision  (log scale)")
    ax.set_yscale("log")
    ax.set_xlim(-0.02, 1.02)
    ax.set_title("Precision–recall by feature arm\n"
                 "validation unless marked TEST; test scored once, after freezing",
                 fontsize=11)
    ax.legend(fontsize=8, loc="lower left", framealpha=0.93,
              facecolor="white", edgecolor="0.85")
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


def plot_decay(path: Path) -> None:
    """PR-AUC against days since the training window closed.

    The single most consequential chart in the project. Test PR-AUC (0.0949) is half of
    validation (0.1938), and the obvious explanation — validation optimism from having
    been consulted all week, including by early stopping — predicts a STEP at the
    val/test boundary. What the data shows is a SLOPE that starts inside validation and
    continues straight through the boundary, which means the boundary is incidental and
    the mechanism is feature staleness: every account and graph feature describes
    training-window behaviour, and that description ages.
    """
    import json

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads(config.ENGINE_JSON.read_text()).get("test_drop_diagnosis", {})
    windows = report.get("time_decay")
    if not windows:
        print("  no time_decay section in engine.json — run src/diagnose_test_drop.py")
        return

    train_end = pd.Timestamp(splits.load_boundaries()["train_end"])
    days, values, rates, labels = [], [], [], []
    for w in windows:
        midpoint = pd.Timestamp(w["start"]) + (pd.Timestamp(w["end"])
                                               - pd.Timestamp(w["start"])) / 2
        days.append((midpoint - train_end).total_seconds() / 86400)
        values.append(w["pr_auc"])
        rates.append(w["base_rate"])
        labels.append(w["mostly"])

    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150)

    # ONE continuous line through every window, then coloured markers on top.
    # Drawing validation and test as separate series breaks the line exactly at the
    # boundary and makes the chart look like the step it exists to rule out — the
    # windows are evenly spaced, so there is no gap in the data to represent.
    ax.plot(days, values, "-", color="0.45", linewidth=2, zorder=1)
    for split, colour in (("val", "#c0392b"), ("test", "#1f6fb4")):
        xs = [d for d, l in zip(days, labels) if l == split]
        ys = [v for v, l in zip(values, labels) if l == split]
        ax.plot(xs, ys, "o", color=colour, markersize=9, zorder=2,
                label=f"{'validation' if split == 'val' else 'test'} windows")

    # Mark where the split boundary falls, so a reader can see it is not where the
    # decline happens.
    boundary = (pd.Timestamp(splits.load_boundaries()["val_end"])
                - train_end).total_seconds() / 86400
    ax.axvline(boundary, color="0.65", linestyle="--", linewidth=1.1, zorder=0)
    ax.text(boundary + 0.05, 0.012, "val / test\nboundary", fontsize=7.5, color="0.45")

    ax.axvline(0, color="0.3", linewidth=1.2)
    ax.text(0.06, 0.012, "training\nwindow ends", fontsize=7.5, color="0.35")

    # The final window's uptick is a base-rate artifact, not a recovery: PR-AUC rises
    # with prevalence, and that window's base rate is 2.7x the others. Saying so on the
    # chart is cheaper than having a reader misread it.
    worst = min(range(len(rates)), key=lambda i: -rates[i])
    ax.annotate(f"base rate {rates[worst]:.5f}\n(2.7x the others)",
                xy=(days[worst], values[worst]), xytext=(days[worst] - 0.95,
                                                         values[worst] + 0.045),
                fontsize=7.5, color="0.35",
                arrowprops=dict(arrowstyle="->", color="0.55", linewidth=0.8))

    ax.set_xlabel("Days since the training window closed (window midpoint)")
    ax.set_ylabel("PR-AUC")
    ax.set_ylim(0, max(values) * 1.25)
    ax.set_title("The engine has a half-life\n"
                 "PR-AUC decays with feature staleness, continuously across the "
                 "val/test boundary", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true",
                        help="retrain arms A and B instead of using the score cache")
    args = parser.parse_args()

    frame = splits.load_transactions()
    table = collect_scores(frame, rebuild=args.rebuild)
    plot(table, frame, config.FIGURES / "pr_curves.png")
    plot_decay(config.FIGURES / "temporal_decay.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
