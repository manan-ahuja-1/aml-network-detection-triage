"""Run the triage agent over a queue of CASES, under a hard spend ceiling.

DEVELOPMENT HAPPENS ON VALIDATION
---------------------------------
`--split val` by default. The test case queue is scored once, on Day 8. Iterating a
prompt against test cases would tune the agent on the data used to report it — the same
error as scoring the test split while choosing a feature arm, one layer up.

THE DEV SUBSET IS STRATIFIED, AND THAT MATTERS
----------------------------------------------
`--dev N` samples N cases holding both productive and non-productive ones, rather than
taking the top of the queue. The account-level top-25 was 92% productive: with two false
positives in it, no calibration of a close/escalate decision could be read at all, which
is why the first two attempts produced 21.7% true-positive loss and then zero closes
with nobody able to tell which was closer to right. Case-level val is 53% productive
overall, and a stratified subset preserves that.

SPENDING IS PRICED BEFORE IT HAPPENS
------------------------------------
The first case is run alone, its real cost measured, and the batch projected from it.
If the projection would breach the ceiling the run stops there, having spent one call.
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
from agent import budget  # noqa: E402
from agent import cases as cases_mod  # noqa: E402
from agent import dossier as dossier_mod  # noqa: E402
from agent import triage  # noqa: E402


def score_split(frame: pd.DataFrame, split: str) -> pd.Series:
    booster = lgb.Booster(model_file=str(config.ENGINE_MODEL))
    X, _, split_series, _ = build_features.build_arm(config.ENGINE_ARM, frame)
    mask = (split_series == split).to_numpy()
    return pd.Series(booster.predict(X[mask], num_iteration=booster.best_iteration),
                     index=frame.index[mask])


def stratified(cases: list, n: int, seed: int = config.RANDOM_SEED) -> list:
    """Sample n cases keeping the productive / non-productive balance."""
    import random
    rng = random.Random(seed)
    productive = [c for c in cases if c.is_productive]
    other = [c for c in cases if not c.is_productive]
    half = n // 2
    picked = (rng.sample(productive, min(half, len(productive)))
              + rng.sample(other, min(n - half, len(other))))
    return sorted(picked, key=lambda c: -c.max_alert_score)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--dev", type=int, default=None,
                        help="stratified subset of N cases, for prompt iteration")
    parser.add_argument("--no-rag", action="store_true", help="the Day 8 RAG ablation")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model", default=None)
    parser.add_argument("--effort", default=None,
                        choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--confirm-spend", type=float, default=None,
                        help="authorise a run projected to exceed the budget ceiling")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.split == "test":
        print("NOTE: the TEST case queue is the Day 8 scored evaluation, not a "
              "development loop.\n")

    triage.load_env()
    model = args.model or config.ANTHROPIC_MODEL
    effort = args.effort or config.AGENT_EFFORT

    frame = splits.load_transactions()
    scores = score_split(frame, args.split)
    table = pd.read_parquet(config.RESULTS / f"alerts_{args.split}.parquet")
    shap = dossier_mod.load_shap(args.split)

    queue = cases_mod.build_cases(frame, table, args.split, scores)
    info = cases_mod.summarise(queue)
    if args.dev:
        queue = stratified(queue, args.dev)

    n_prod = sum(1 for c in queue if c.is_productive)
    print(f"{info['n_accounts']} alerted accounts -> {info['n_cases']} cases; "
          f"triaging {len(queue)}")
    print(f"  {n_prod} productive / {len(queue) - n_prod} not "
          f"({n_prod / len(queue) * 100:.0f}% productive)")
    print(f"  model {model}, effort {effort}, "
          f"retrieval {'OFF' if args.no_rag else 'ON'}")
    print(f"  {budget.status_line()}\n")

    print("  building dossiers...", end="", flush=True)
    dossiers = [dossier_mod.build_case(frame, c, table, shap, retrieve=not args.no_rag)
                for c in queue]
    print(f" {len(dossiers)} built")

    # Price the batch from a real call before committing to it.
    print("  pricing one case...", end="", flush=True)
    first = triage.triage_case(dossiers[0], use_cache=not args.no_cache,
                               model=model, effort=effort)
    per_call = first["usage"]["cost_usd"] if not first["cached"] else 0.0
    print(f" ${per_call:.5f}"
          f"{' (cached, cannot price from it)' if first['cached'] else ''}")

    if per_call > 0:
        report = budget.preflight(f"run_cases --split {args.split}", per_call,
                                  len(queue) - 1, args.confirm_spend)
        print(f"  projected ${report['projected_usd']:.2f} for the remaining "
              f"{report['n_calls']} cases; ${report['remaining_usd']:.2f} left\n")

    lock, done = Lock(), [1]
    records = [first]

    def run_one(item):
        case, dossier = item
        try:
            record = triage.triage_case(dossier, use_cache=not args.no_cache,
                                        model=model, effort=effort)
        except Exception as exc:  # noqa: BLE001 — one case must not kill the batch
            with lock:
                done[0] += 1
                print(f"  [{done[0]:>3}/{len(queue)}] {case.case_id} FAILED: "
                      f"{type(exc).__name__}: {exc}")
            return {"case_id": case.case_id, "failed": True,
                    "error": f"{type(exc).__name__}: {exc}",
                    "is_productive": int(case.is_productive)}
        record["is_productive"] = int(case.is_productive)
        record["n_members"] = case.n_members
        with lock:
            done[0] += 1
            r, v = record["result"], record["validation"]
            truth = "TP" if record["is_productive"] else "FP"
            agree = (r["disposition"] == "escalate") == bool(record["is_productive"])
            print(f"  [{done[0]:>3}/{len(queue)}] {case.case_id} "
                  f"{case.n_members:>3}acct truth={truth}  "
                  f"{r['disposition']:<9} {r['pattern_classification']:<15} "
                  f"{r['confidence']:<7} cites={v['n_cited']:<3} "
                  f"{'OK ' if agree else 'MISS'}"
                  f"{' HALLUCINATED' if v['hallucinated_citation'] else ''}"
                  f"{'  (cached)' if record['cached'] else ''}")
        return record

    records[0]["is_productive"] = int(queue[0].is_productive)
    records[0]["n_members"] = queue[0].n_members
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records += list(pool.map(run_one, zip(queue[1:], dossiers[1:])))
    elapsed = time.monotonic() - started

    failures = [r for r in records if r.get("failed")]
    records = [r for r in records if not r.get("failed")]
    billed = [r for r in records if not r["cached"]]

    if billed:
        budget.record(
            script="run_cases", model=model,
            calls=len(billed),
            input_tokens=sum(r["usage"]["input_tokens"] for r in billed),
            output_tokens=sum(r["usage"]["output_tokens"] for r in billed),
            cost_usd=sum(r["usage"]["cost_usd"] for r in billed),
            note=f"{args.split}{' no-rag' if args.no_rag else ''} "
                 f"effort={effort} n={len(billed)}")

    suffix = f"{'_norag' if args.no_rag else ''}{'_dev' if args.dev else ''}"
    out = Path(args.out) if args.out else (
        config.RESULTS / f"triage_cases_{args.split}{suffix}.json")
    out.write_text(json.dumps(records, indent=2))

    print(f"\n{'=' * 70}")
    print(f"  {len(records)} cases in {elapsed:.0f}s "
          f"({len(billed)} billed, {len(records) - len(billed)} cached)")
    if billed:
        cost = sum(r["usage"]["cost_usd"] for r in billed)
        print(f"  ${cost:.4f} this run, ${cost / len(billed):.5f} per case")
    if failures:
        print(f"  {len(failures)} FAILED and excluded:")
        for f in failures:
            print(f"    {f['case_id']}: {f['error']}")
    print(f"  {budget.status_line()}")
    print(f"  wrote {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
