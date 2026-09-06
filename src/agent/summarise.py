"""Summarise a triage run.

The full agent evaluation is Day 8 — RAG ablation, typology macro-F1, confusion matrix
against Patterns.txt. This is the subset needed to know whether the agent works at all,
and the framing it establishes is the one Day 8 keeps.

THE CONTROL MATTERS MORE THAN THE SCORE (A8)
--------------------------------------------
"The agent is 84% accurate" means nothing without knowing what escalate-everything
scores. On a queue that is 68% productive, an agent that escalates every alert is 68%
accurate and has done no work. So the headline is not accuracy — it is:

    how many false positives were closed, and how many true positives did that cost?

A triage layer that closes nothing is free to build and worth nothing. A triage layer
that closes true positives is worse than nothing.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def summarise(records: list[dict]) -> dict:
    n = len(records)
    productive = [r for r in records if r["is_productive"]]
    non_productive = [r for r in records if not r["is_productive"]]

    escalated = [r for r in records if r["result"]["disposition"] == "escalate"]
    closed = [r for r in records if r["result"]["disposition"] == "close"]

    tp_closed = [r for r in closed if r["is_productive"]]
    fp_closed = [r for r in closed if not r["is_productive"]]

    billed = [r for r in records if not r.get("cached")]
    costs = [r["usage"]["cost_usd"] for r in records]
    latencies = [r["usage"]["latency_seconds"] for r in records]

    hallucinated = [r for r in records if r["validation"]["hallucinated_citation"]]
    cited_nothing = [r for r in records if r["validation"]["cited_nothing"]]

    return {
        "n_alerts": n,
        "queue_mix": {
            "productive": len(productive),
            "non_productive": len(non_productive),
            "productive_share": round(len(productive) / n, 4) if n else 0.0,
        },
        # The business case, stated as the trade it actually is.
        "triage_effect": {
            "escalated": len(escalated),
            "closed": len(closed),
            "false_positives_closed": len(fp_closed),
            "true_positives_closed": len(tp_closed),
            "fp_removal_rate": round(len(fp_closed) / len(non_productive), 4)
            if non_productive else None,
            "tp_loss_rate": round(len(tp_closed) / len(productive), 4)
            if productive else None,
            "review_volume_reduction": round(len(closed) / n, 4) if n else 0.0,
        },
        # The control: escalate everything. Any agent metric must be read against this.
        "baseline_escalate_everything": {
            "escalated": n,
            "precision": round(len(productive) / n, 4) if n else 0.0,
            "recall": 1.0,
            "false_positives_reviewed": len(non_productive),
        },
        "agent_as_classifier": {
            "precision": round(
                sum(1 for r in escalated if r["is_productive"]) / len(escalated), 4)
            if escalated else None,
            "recall": round(
                sum(1 for r in escalated if r["is_productive"]) / len(productive), 4)
            if productive else None,
        },
        "hallucination": {
            "rate": round(len(hallucinated) / n, 4) if n else 0.0,
            "n_with_invalid_citation": len(hallucinated),
            "n_citing_nothing": len(cited_nothing),
            "invalid_source_citations": sum(
                len(r["validation"]["invalid_sources"]) for r in records),
        },
        "classification": dict(Counter(
            r["result"]["pattern_classification"] for r in records).most_common()),
        "confidence": dict(Counter(r["result"]["confidence"] for r in records)),
        "cost": {
            "model": records[0]["model"] if records else None,
            "total_usd": round(sum(costs), 4),
            "per_alert_usd": round(sum(costs) / n, 5) if n else 0.0,
            "projected_full_queue_usd": round(
                sum(costs) / n * config.ALERT_SET_SIZE, 2) if n else 0.0,
            "median_latency_seconds": round(sorted(latencies)[len(latencies) // 2], 1)
            if latencies else None,
            "billed_calls": len(billed),
        },
    }


def render(summary: dict, unit: str = "alert") -> str:
    t, b, a = (summary["triage_effect"], summary["baseline_escalate_everything"],
               summary["agent_as_classifier"])
    q, h, c = summary["queue_mix"], summary["hallucination"], summary["cost"]
    plural = unit + "s"

    lines = [
        "=" * 70,
        f"TRIAGE SUMMARY — {summary['n_alerts']} {plural}",
        "=" * 70,
        f"  queue mix: {q['productive']} productive / {q['non_productive']} not "
        f"({q['productive_share'] * 100:.0f}% productive)",
        "",
        "  THE CONTROL — escalate everything",
        f"    {b['escalated']} {plural} reviewed, precision {b['precision'] * 100:.1f}%, "
        f"recall 100%, {b['false_positives_reviewed']} wasted reviews",
        "",
        "  THE AGENT",
        f"    escalated {t['escalated']}, closed {t['closed']} "
        f"({t['review_volume_reduction'] * 100:.0f}% less to review)",
        f"    false positives closed: {t['false_positives_closed']}"
        + (f" of {q['non_productive']} ({t['fp_removal_rate'] * 100:.0f}%)"
           if t["fp_removal_rate"] is not None else ""),
        f"    true positives closed:  {t['true_positives_closed']}"
        + (f" of {q['productive']} ({t['tp_loss_rate'] * 100:.1f}% LOST)"
           if t["tp_loss_rate"] is not None else ""),
    ]
    if a["precision"] is not None:
        lift = (a["precision"] - b["precision"]) * 100
        lines.append(f"    escalation precision {a['precision'] * 100:.1f}% "
                     f"vs {b['precision'] * 100:.1f}% control  ({lift:+.1f} pts), "
                     f"recall {a['recall'] * 100:.1f}%")

    lines += [
        "",
        "  GROUNDING (programmatic, A9)",
        f"    hallucinated citations: {h['n_with_invalid_citation']}/{summary['n_alerts']}"
        f"  ({h['rate'] * 100:.1f}%)",
        f"    {plural} citing nothing:  {h['n_citing_nothing']}",
        f"    invalid source ids:     {h['invalid_source_citations']}",
        "",
        "  CLASSIFICATION",
    ]
    for label, count in summary["classification"].items():
        lines.append(f"    {label:<16} {count}")
    lines += [
        "",
        f"  COST — {c['model']}",
        f"    ${c['per_alert_usd']:.5f} per {unit}, ${c['total_usd']:.4f} for this run",
        f"    median latency {c['median_latency_seconds']}s",
    ]
    return "\n".join(lines)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?",
                        default=str(config.RESULTS / "triage_val.json"))
    args = parser.parse_args()

    records = json.loads(Path(args.path).read_text())
    unit = "case" if records and records[0].get("case_id") else "alert"
    summary = summarise(records)
    print(render(summary, unit))

    out = Path(args.path).with_name(Path(args.path).stem + "_summary.json")
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n  wrote {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
