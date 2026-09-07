"""Export everything the demo needs, so the app never touches `data/` or `models/`.

WHY A BUNDLE EXISTS AT ALL
--------------------------
Case dossiers are built from the 475MB raw transaction file and the 15MB booster, both
gitignored and neither present on a deployment host. The app therefore cannot rebuild
them, and an app that quietly renders less than the agent actually saw would misrepresent
the system it exists to demonstrate.

So the dossiers are exported once, here, and committed. The app reads them and re-renders
the prompt with the same `triage.render_case` the agent used — the prompt is NOT exported,
because rendering it live is what proves the displayed prompt is the real one rather than
a copy that has since drifted.

THE SUBGRAPH IS KEPT OUT OF THE DOSSIER, DELIBERATELY
-----------------------------------------------------
The C6 walkthrough needs graph edges to draw a ring. Evidence rows carry a `counterparty`
but not both endpoints, so the edge list has to come from somewhere. It is attached
BESIDE the dossier rather than inside it: the dossier is a record of exactly what the
model was shown, and adding fields the agent never saw to that record — even harmless
ones — would make the demo's central claim unverifiable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alerts as alerts_mod  # noqa: E402
import build_features  # noqa: E402
import config  # noqa: E402
import splits  # noqa: E402
from agent import cases as cases_mod  # noqa: E402
from agent import dossier as dossier_mod  # noqa: E402

OUT = config.RESULTS / "demo_bundle.json"
SPLIT = "test"


def jsonable(obj):
    """numpy scalars are not JSON-serialisable, and dossiers are full of them.

    Dossiers have never been written to disk before — `run_cases.py` persists the
    agent's *result*, not its input — so nothing has previously forced the numpy int64
    and float64 that pandas leaves in `case_topology` and `member_summaries` through
    `json.dumps`. Without this the export dies on the first case.
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    raise TypeError(f"{type(obj).__name__} is not JSON serialisable: {obj!r}")


def score_split(frame: pd.DataFrame, split: str) -> pd.Series:
    booster = lgb.Booster(model_file=str(config.ENGINE_MODEL))
    X, _, split_series, _ = build_features.build_arm(config.ENGINE_ARM, frame)
    mask = (split_series == split).to_numpy()
    return pd.Series(booster.predict(X[mask], num_iteration=booster.best_iteration),
                     index=frame.index[mask])


def subgraph(case, evidence: pd.DataFrame, limit: int = 300) -> dict:
    """Edge list for drawing the case, capped so one hub cannot produce a hairball."""
    rows = evidence.nlargest(min(limit, len(evidence)), "paid_usd")
    members = set(case.members)
    edges, nodes = [], {}
    for _, r in rows.iterrows():
        a, b = str(r["from_id"]), str(r["to_id"])
        if a == b:
            continue                       # self-transfers are 11.6% of this data
        for node in (a, b):
            nodes.setdefault(node, {"id": node, "is_member": node in members,
                                    "bank": node.split(":")[0]})
        edges.append({"source": a, "target": b,
                      "usd": round(float(r["paid_usd"]), 2),
                      "timestamp": str(r["timestamp"]),
                      "format": str(r["payment_format"])})
    return {"nodes": list(nodes.values()), "edges": edges,
            "edges_shown": len(edges), "edges_total": int(len(evidence))}


def main() -> int:
    frame = splits.load_transactions()
    print(f"scoring {SPLIT}...")
    scores = score_split(frame, SPLIT)
    table = pd.read_parquet(config.RESULTS / f"alerts_{SPLIT}.parquet")
    shap = dossier_mod.load_shap(SPLIT)

    queue = cases_mod.build_cases(frame, table, SPLIT, scores)
    print(f"{len(queue)} cases; building dossiers")

    cases = {}
    for case in queue:
        d = dossier_mod.build_case(frame, case, table, shap, retrieve=True)
        cases[case.case_id] = {
            "dossier": d,
            "subgraph": subgraph(case, case.evidence),
            "members": case.members,
            "n_members": case.n_members,
            "is_productive": int(case.is_productive),
            "max_alert_score": round(case.max_alert_score, 6),
        }

    bundle = {
        "split": SPLIT,
        "engine_arm": config.ENGINE_ARM,
        "model": config.ANTHROPIC_MODEL,
        "n_cases": len(cases),
        "note": ("dossiers are exactly what the agent was shown; the prompt is rendered "
                 "live by the app from these, and `subgraph` is visualisation data the "
                 "agent never saw"),
        "cases": cases,
    }
    OUT.write_text(json.dumps(bundle, indent=1, default=jsonable))
    size_mb = OUT.stat().st_size / 1e6
    print(f"wrote {OUT.name}  ({size_mb:.2f} MB, {len(cases)} cases)")
    if size_mb > 8:
        print("  WARNING: large for a committed artifact; consider capping evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
