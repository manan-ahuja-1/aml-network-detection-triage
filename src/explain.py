"""TreeSHAP attribution for alerted transactions (B3).

WHY THE AGENT NEEDS THIS
------------------------
Without it, the two halves of this project are disconnected: the engine decides an
account is suspicious, then the agent is handed the account and independently
re-derives a reason. The model's actual evidence is thrown away, and the case note
explains the LLM's guess rather than why *this model* fired.

Passing SHAP attributions into the agent's context makes the narrative grounded in the
model's real reasoning. It is also the correct regulatory posture: AML models sit under
model risk management expectations (SR 11-7 in the US), and an examiner asking "why did
this alert fire?" is entitled to an answer that comes from the model rather than from a
plausible-sounding reconstruction.

WHY NOT THE `shap` PACKAGE
--------------------------
LightGBM computes exact TreeSHAP natively via `predict(pred_contrib=True)`. It returns
an (n_rows, n_features + 1) array whose last column is the base value, and the row sums
to the model's raw margin. That is the same algorithm `shap.TreeExplainer` calls into,
without the dependency, the wrapper, or the version drift. The `shap` package stays in
requirements for its plotting utilities, not for the computation.

A NOTE ON UNITS
---------------
Contributions are in RAW MARGIN (log-odds) space, not probability. They sum with the
base value to `predict(raw_score=True)`, and are additive there — which is exactly why
they are attributions at all. Converting each one to "probability points" would break
that additivity, because the sigmoid is not linear. The agent is told they are relative
weights, and the sign is what carries meaning.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


def contributions(booster, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Exact TreeSHAP values for every row of X.

    Returns (contribs, base_value) where contribs has one column per feature, aligned
    with X.columns, and base_value is the model's expected raw margin.
    """
    raw = booster.predict(X, num_iteration=booster.best_iteration, pred_contrib=True)
    raw = np.asarray(raw)
    if raw.shape[1] != X.shape[1] + 1:
        raise ValueError(
            f"pred_contrib returned {raw.shape[1]} columns for {X.shape[1]} features; "
            "expected one extra column for the base value."
        )
    return raw[:, :-1], float(raw[0, -1])


def top_reasons(contrib_row: np.ndarray, feature_names: list[str],
                values: pd.Series, k: int = 8) -> list[dict]:
    """The k features that moved THIS prediction furthest, in either direction.

    Ranked by absolute contribution, not by positive contribution only. A strong
    negative reason is information the analyst needs: "this looks like a mule except
    that the account has 340 days of consistent history" is a disposition-changing
    fact, and suppressing it would give the agent a one-sided case file.
    """
    order = np.argsort(-np.abs(contrib_row))[:k]
    out = []
    for i in order:
        value = values.iloc[i]
        # Categorical features arrive as pandas Categorical; str() keeps the label
        # rather than serialising an opaque integer code into the agent's prompt.
        if pd.isna(value):
            rendered = None
        elif isinstance(value, (np.integer, np.floating, int, float)):
            rendered = float(value)
        else:
            rendered = str(value)
        out.append({
            "feature": feature_names[i],
            "value": rendered,
            "contribution": float(contrib_row[i]),
            "direction": "increases risk" if contrib_row[i] > 0 else "decreases risk",
        })
    return out


def explain_alerts(booster, X: pd.DataFrame, frame: pd.DataFrame,
                   alerts: pd.DataFrame, split: str, scores: pd.Series,
                   k: int = 8) -> dict[str, dict]:
    """Explain each alerted account through its single highest-scoring transaction.

    An account's alert score IS the max over its transactions (config.ALERT_AGGREGATION),
    so exactly one transaction is responsible for the alert firing. Explaining that
    transaction explains the alert. Averaging SHAP across an account's activity would
    describe a composite transaction that never happened.
    """
    part = frame[frame["split"] == split]
    part_scores = scores.reindex(part.index)

    # For each alerted account, locate the transaction that produced its score. Doing
    # this with two vectorised idxmax lookups rather than a per-account scan keeps it
    # to seconds on a 1M-row split.
    node_ids = set(alerts.index)
    relevant = part[part["from_id"].isin(node_ids) | part["to_id"].isin(node_ids)]

    driver: dict[str, int] = {}
    best: dict[str, float] = {}
    for id_col in ("from_id", "to_id"):
        sub = relevant[relevant[id_col].isin(node_ids)]
        idx = part_scores.reindex(sub.index).groupby(sub[id_col].to_numpy(),
                                                     observed=True).idxmax()
        for node, row_index in idx.items():
            score = float(part_scores.loc[row_index])
            if node not in best or score > best[node]:
                best[node] = score
                driver[node] = int(row_index)

    rows = sorted(set(driver.values()))
    X_driver = X.loc[rows]
    contribs, base = contributions(booster, X_driver)
    names = list(X.columns)
    row_position = {r: i for i, r in enumerate(rows)}

    out: dict[str, dict] = {}
    for node, row_index in driver.items():
        i = row_position[row_index]
        out[node] = {
            "driving_txn_index": row_index,
            "driving_txn_id": f"T{row_index:07d}",
            "score": best[node],
            "base_value": base,
            "reasons": top_reasons(contribs[i], names, X_driver.iloc[i], k),
        }
    return out


if __name__ == "__main__":
    import lightgbm as lgb

    import alerts as alerts_mod
    import build_features
    import splits

    if not config.ENGINE_MODEL.exists():
        raise SystemExit(f"{config.ENGINE_MODEL} not found — run `make eval` first.")

    booster = lgb.Booster(model_file=str(config.ENGINE_MODEL))
    frame = splits.load_transactions()
    X, y, split_series, _ = build_features.build_arm(config.ENGINE_ARM, frame)

    mask = (split_series == "val").to_numpy()
    scores = pd.Series(booster.predict(X[mask], num_iteration=booster.best_iteration),
                       index=frame.index[mask])
    table = alerts_mod.account_alerts(frame, scores.to_numpy(), "val")
    top = alerts_mod.top_alerts(table, 5)

    explained = explain_alerts(booster, X, frame, top, "val", scores)
    for node, info in list(explained.items())[:3]:
        print(f"\n{node}  score {info['score']:.4f}  via {info['driving_txn_id']}")
        for r in info["reasons"][:5]:
            print(f"    {r['contribution']:+8.3f}  {r['feature']:38s} = {r['value']}")
