"""Run the triage agent over an alert set.

DEVELOPMENT HAPPENS ON VALIDATION
---------------------------------
`--split val` by default, and that is not incidental. The test alert set exists for one
scored evaluation on Day 8. Iterating the prompt against test alerts would tune the
agent on the same data used to report its performance — the same mistake as scoring the
test split while choosing a feature arm, one layer up the stack.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import lightgbm as lgb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import build_features  # noqa: E402
import config  # noqa: E402
import splits  # noqa: E402
from agent import dossier as dossier_mod  # noqa: E402
from agent import triage  # noqa: E402


def score_split(frame: pd.DataFrame, split: str) -> pd.Series:
    booster = lgb.Booster(model_file=str(config.ENGINE_MODEL))
    X, _, split_series, _ = build_features.build_arm(config.ENGINE_ARM, frame)
    mask = (split_series == split).to_numpy()
    return pd.Series(booster.predict(X[mask], num_iteration=booster.best_iteration),
                     index=frame.index[mask])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--limit", type=int, default=None,
                        help="run only the first N alerts (development)")
    parser.add_argument("--no-rag", action="store_true",
                        help="build dossiers without retrieval — the Day 8 RAG ablation")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--workers", type=int, default=8,
                        help="concurrent API calls; 200 alerts is 72 min sequentially")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.split == "test":
        print("NOTE: running on the TEST alert set. This is the Day 8 scored evaluation, "
              "not a development loop.\n")

    triage.load_env()

    frame = splits.load_transactions()
    scores = score_split(frame, args.split)
    top = pd.read_parquet(config.RESULTS / f"alerts_{args.split}.parquet")
    shap = dossier_mod.load_shap(args.split)

    nodes = list(top.index)[: args.limit] if args.limit else list(top.index)
    print(f"triaging {len(nodes)} alerts from the {args.split} set "
          f"({int(top.loc[nodes, 'is_productive'].sum())} productive, "
          f"{int((1 - top.loc[nodes, 'is_productive']).sum())} not)")
    print(f"model {config.ANTHROPIC_MODEL}, structured outputs, "
          f"retrieval {'OFF' if args.no_rag else 'ON'}\n")

    # Dossiers are built up front, single-threaded: they touch pandas frames and the
    # Chroma index, neither of which is worth making concurrent, and building them all
    # first means a failure in assembly surfaces before any money is spent on the API.
    ranks = {node: i + 1 for i, node in enumerate(top.index)}
    print("  building dossiers...", end="", flush=True)
    dossiers = [
        dossier_mod.build(frame, node, args.split, scores, top.loc[node],
                          ranks[node], shap.get(node), retrieve=not args.no_rag)
        for node in nodes
    ]
    print(f" {len(dossiers)} built\n")

    lock = Lock()
    done = [0]

    def run_one(item):
        node, dossier = item
        try:
            record = triage.triage_alert(dossier, use_cache=not args.no_cache)
        except Exception as exc:  # noqa: BLE001
            # One bad alert must not destroy the batch. A single max_tokens overflow
            # took out a 200-alert run and discarded 190 completed, already-billed
            # calls, because the exception propagated straight out of pool.map. The
            # failure is recorded and reported instead; a partial run with a named
            # gap is far more useful than no run.
            with lock:
                done[0] += 1
                print(f"  [{done[0]:>3}/{len(nodes)}] {node:<20} FAILED: "
                      f"{type(exc).__name__}: {exc}")
            return {"account": node, "split": args.split, "failed": True,
                    "error": f"{type(exc).__name__}: {exc}",
                    "is_productive": int(top.loc[node, "is_productive"])}
        record["is_productive"] = int(top.loc[node, "is_productive"])
        with lock:
            done[0] += 1
            r, v = record["result"], record["validation"]
            truth = "TP" if record["is_productive"] else "FP"
            flag = " HALLUCINATED" if v["hallucinated_citation"] else ""
            agree = (r["disposition"] == "escalate") == bool(record["is_productive"])
            print(f"  [{done[0]:>3}/{len(nodes)}] {node:<20} truth={truth}  "
                  f"{r['disposition']:<9} {r['pattern_classification']:<15} "
                  f"{r['confidence']:<7} cites={v['n_cited']:<3} "
                  f"{'OK ' if agree else 'MISS'}{flag}"
                  f"{'  (cached)' if record['cached'] else ''}")
        return record

    started = time.monotonic()
    # Threads, not processes: these calls are entirely network-bound, and the SDK
    # tolerates concurrent use from separate threads.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(run_one, zip(nodes, dossiers)))

    elapsed = time.monotonic() - started
    failures = [r for r in records if r.get("failed")]
    records = [r for r in records if not r.get("failed")]
    billed = [r for r in records if not r["cached"]]
    total_cost = sum(r["usage"]["cost_usd"] for r in billed)

    out = Path(args.out) if args.out else (
        config.RESULTS / f"triage_{args.split}{'_norag' if args.no_rag else ''}.json")
    out.write_text(json.dumps(records, indent=2))

    print(f"\n{'=' * 70}")
    print(f"  {len(records)} alerts in {elapsed:.0f}s "
          f"({len(billed)} billed, {len(records) - len(billed)} cached)")
    if billed:
        print(f"  cost ${total_cost:.4f} total, ${total_cost / len(billed):.4f} per alert")
        print(f"  latency median "
              f"{sorted(r['usage']['latency_seconds'] for r in billed)[len(billed) // 2]:.1f}s")
    if failures:
        print(f"\n  {len(failures)} ALERTS FAILED and are excluded from the results:")
        for f in failures:
            print(f"    {f['account']}: {f['error']}")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
