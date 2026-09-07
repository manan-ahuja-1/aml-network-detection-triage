"""Day 8: score the triage agent once, on the test case queue.

WHAT THIS FILE IS FOR
---------------------
`summarise.py` answers "did the agent do anything" during development. This answers the
four questions the README has to answer, on data the prompt never saw:

  1. Does triage beat the control?  FP removed against TP lost, versus escalate-everything.
  2. Can it name the typology?      Against Patterns.txt, at the ring level, as a real
                                    labelled task rather than a rubric.
  3. Is it grounded?                Fabricated identifiers, in the citation list AND in
                                    the prose of the case note.
  4. Does retrieval earn its place? The same queue with the knowledge base switched off.

SCORED ONCE MEANS SCORED ONCE
-----------------------------
Both test configurations — retrieval on and retrieval off — were fixed before either ran
and are reported whatever they say. An ablation is two pre-registered runs, not an
iteration loop: the thing that would invalidate the number is looking at test, changing
the prompt, and running again, and that has not happened.

WHY TYPOLOGY IS SCORED ON A SUBSET, AND SAID SO EVERY TIME
----------------------------------------------------------
Three kinds of case reach the scorer and only two of them can be marked:

  * a case whose members touch a NAMED pattern in the scored window — truth is that
    pattern's type, and this is the only bucket typology accuracy is computed on;
  * a productive case whose laundering belongs to no named ring — 43.5% of laundering
    transactions are like this. Truth is unknown, NOT "NONE". Scoring these against NONE
    would manufacture either credit or blame out of a labelling gap, so they are counted
    and excluded;
  * a clean case — truth is NONE, and predicting NONE here is a real correct answer.

A case can span several rings, so two readings are reported: ANY-match (the label is one
of the types present) and DOMINANT-match (the label is the type with the most laundering
transactions in the case). Any-match is the operational reading — an investigator handed
"this is a fan-out" is pointed the right way even if the case also contains a stack.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
import evaluate as engine_eval  # noqa: E402
import splits  # noqa: E402
from agent import summarise as summarise_mod  # noqa: E402


def case_truth(records: list[dict], frame: pd.DataFrame,
               membership: pd.DataFrame, split: str) -> dict[str, dict]:
    """Ground-truth typology for each case, from Patterns.txt.

    A case's evidence is every transaction in the scored window touching any member
    account, which is exactly the set `alerts.alert_evidence` draws on — so the truth
    label covers what the agent could in principle have seen, not the ring's full extent
    across the whole dataset.
    """
    part = frame[frame["split"] == split]
    labelled = membership.loc[membership.index.intersection(part.index)]
    tagged = part.loc[labelled.index, ["from_id", "to_id"]].join(labelled)

    # account -> {pattern_type: n transactions}
    by_account: dict[str, Counter] = defaultdict(Counter)
    for side in ("from_id", "to_id"):
        for account, group in tagged.groupby(side, observed=True):
            by_account[account].update(group["pattern_type"].astype(str))

    launder = part[part["is_laundering"] == 1]
    laundering_accounts = set(launder["from_id"]) | set(launder["to_id"])

    truth: dict[str, dict] = {}
    for rec in records:
        members = rec.get("members") or []
        counts: Counter = Counter()
        for m in members:
            counts.update(by_account.get(m, {}))
        touches_laundering = bool(set(members) & laundering_accounts)

        if counts:
            bucket = "labelled"
        elif touches_laundering or rec["is_productive"]:
            bucket = "unlabelled_laundering"
        else:
            bucket = "clean"
        truth[rec["unit_id"]] = {
            "bucket": bucket,
            "types": sorted(counts),
            "dominant": counts.most_common(1)[0][0] if counts else None,
            "pattern_transactions": int(sum(counts.values())),
        }
    return truth


def typology_scores(records: list[dict], truth: dict[str, dict]) -> dict:
    """Accuracy on the labelled cases, plus the confusion matrix that explains it."""
    labelled = [r for r in records if truth[r["unit_id"]]["bucket"] == "labelled"]
    clean = [r for r in records if truth[r["unit_id"]]["bucket"] == "clean"]
    unlabelled = [r for r in records
                  if truth[r["unit_id"]]["bucket"] == "unlabelled_laundering"]

    any_hit, dom_hit = [], []
    confusion: Counter = Counter()
    for r in labelled:
        pred = r["result"]["pattern_classification"]
        t = truth[r["unit_id"]]
        any_hit.append(pred in t["types"])
        dom_hit.append(pred == t["dominant"])
        confusion[(t["dominant"], pred)] += 1

    per_type: dict[str, dict] = {}
    for r in labelled:
        t = truth[r["unit_id"]]["dominant"]
        d = per_type.setdefault(t, {"n": 0, "any": 0, "dominant": 0})
        d["n"] += 1
        d["any"] += int(r["result"]["pattern_classification"]
                        in truth[r["unit_id"]]["types"])
        d["dominant"] += int(r["result"]["pattern_classification"] == t)
    for d in per_type.values():
        d["any_accuracy"] = round(d["any"] / d["n"], 4)
        d["dominant_accuracy"] = round(d["dominant"] / d["n"], 4)

    n_lab = len(labelled)

    # TYPOLOGY NEEDS A CONTROL TOO, AND IT NEARLY SHIPPED WITHOUT ONE
    # ---------------------------------------------------------------
    # The disposition was given a control from the start; typology accuracy was reported
    # bare, and bare it is misleading in two separate ways.
    #
    # ANY-MATCH is not a fixed-difficulty task. A large case touches many injected rings
    # — one 11-account case has all eight typologies in its truth set — so "the label is
    # one of the types present" is nearly free wherever the case is big. The chance rate
    # is therefore per-case, mean(len(types)/8), not 1/8.
    #
    # DOMINANT-MATCH has a fixed chance rate of 1/8, but the class distribution is very
    # skewed: most labelled cases are GATHER-SCATTER. Always guessing the majority class
    # is the baseline any classifier has to beat, and it is a stronger one than chance.
    dominants = [truth[r["unit_id"]]["dominant"] for r in labelled]
    majority_label = Counter(dominants).most_common(1)[0][0] if dominants else None
    majority_acc = (dominants.count(majority_label) / n_lab) if n_lab else None
    chance_any = (sum(len(truth[r["unit_id"]]["types"]) / len(config.TYPOLOGIES[:-1])
                      for r in labelled) / n_lab) if n_lab else None

    return {
        "baselines": {
            "any_match_expected_by_chance": (round(chance_any, 4) if chance_any
                                             else None),
            "dominant_match_chance": round(1 / len(config.TYPOLOGIES[:-1]), 4),
            "majority_class": majority_label,
            "majority_class_dominant_accuracy": (round(majority_acc, 4) if majority_acc
                                                 else None),
            "reading": (
                "any-match must be read against a per-case chance rate, because a large "
                "case containing many rings makes it nearly free; dominant-match must be "
                "read against always predicting the majority class, which is the real "
                "control"),
        },
        "n_labelled_cases": n_lab,
        "n_clean_cases": len(clean),
        "n_unlabelled_laundering_cases": len(unlabelled),
        "coverage_note": (
            f"typology accuracy is computed on the {n_lab} cases whose members touch a "
            f"named ring in the scored window; {len(unlabelled)} productive cases have "
            "laundering the simulator did not group into a ring and are excluded rather "
            "than scored against NONE"),
        "any_match_accuracy": round(sum(any_hit) / n_lab, 4) if n_lab else None,
        "dominant_match_accuracy": round(sum(dom_hit) / n_lab, 4) if n_lab else None,
        "said_none_on_a_labelled_case": sum(
            1 for r in labelled if r["result"]["pattern_classification"] == "NONE"),
        "clean_cases_called_none": sum(
            1 for r in clean if r["result"]["pattern_classification"] == "NONE"),
        "by_true_type": dict(sorted(per_type.items())),
        "confusion": {f"{t or 'NONE'} -> {p}": c
                      for (t, p), c in sorted(confusion.items(),
                                              key=lambda kv: -kv[1])},
    }


def discrimination(records: list[dict]) -> dict:
    """Does the disposition carry information, or does it only look like it does?

    THIS IS THE TEST THAT CHANGED THE PROJECT'S CLAIM.
    A pooled escalation precision above the queue base rate is not evidence of skill.
    Precision is a function of WHICH cases you close as much as of how well you judge
    them: close more heavily in a bucket where the base rate is low and pooled precision
    rises on its own. Validation showed exactly that — one field ordering scored +6.7
    points over the control while its escalation precision inside every case-size bucket
    sat on the base rate to within a point.

    So the question is asked directly: is the escalated set enriched in productive cases
    relative to the queue it was drawn from? Fisher's exact test, one-sided, on the
    2x2 of disposition against ground truth. Reported per size bucket as well, because
    pooling across buckets with different base rates is the thing that manufactured the
    illusion in the first place.
    """
    from scipy.stats import fisher_exact

    def table(part: list[dict]) -> dict | None:
        esc = [r for r in part if r["result"]["disposition"] == "escalate"]
        clo = [r for r in part if r["result"]["disposition"] == "close"]
        if not esc or not clo:
            return None
        a = sum(r["is_productive"] for r in esc)
        c = sum(r["is_productive"] for r in clo)
        odds, p = fisher_exact([[a, len(esc) - a], [c, len(clo) - c]],
                               alternative="greater")
        return {
            "n": len(part),
            "escalated": len(esc),
            "escalation_precision": round(a / len(esc), 4),
            "queue_base_rate": round((a + c) / len(part), 4),
            "lift_pts": round((a / len(esc) - (a + c) / len(part)) * 100, 1),
            "odds_ratio": None if odds in (float("inf"),) else round(float(odds), 3),
            "fisher_p_one_sided": round(float(p), 4),
        }

    out = {"overall": table(records)}
    for name, lo, hi in (("1-2", 1, 2), ("3-5", 3, 5), ("6+", 6, 10_000)):
        t = table([r for r in records if lo <= r["n_members"] <= hi])
        if t:
            out[name] = t
    overall = out["overall"] or {}
    out["reading"] = (
        "the escalated set is not significantly enriched in productive cases "
        f"(p = {overall.get('fisher_p_one_sided')}); the disposition is reported as "
        "measured-not-useful rather than as a headline"
        if (overall.get("fisher_p_one_sided") or 1) > 0.05 else
        "the escalated set is significantly enriched in productive cases")
    return out


def by_case_size(records: list[dict]) -> dict:
    """Where the errors are. On validation every TP loss was in a case of 1-2 accounts;
    this is where that either replicates on test or does not."""
    buckets = {"1-2": (1, 2), "3-5": (3, 5), "6+": (6, 10_000)}
    out = {}
    for name, (lo, hi) in buckets.items():
        part = [r for r in records if lo <= r["n_members"] <= hi]
        prod = [r for r in part if r["is_productive"]]
        closed_tp = [r for r in prod if r["result"]["disposition"] == "close"]
        nonprod = [r for r in part if not r["is_productive"]]
        closed_fp = [r for r in nonprod if r["result"]["disposition"] == "close"]
        out[name] = {
            "n_cases": len(part),
            "n_productive": len(prod),
            "tp_lost": len(closed_tp),
            "tp_loss_rate": round(len(closed_tp) / len(prod), 4) if prod else None,
            "fp_removed": len(closed_fp),
            "fp_removal_rate": round(len(closed_fp) / len(nonprod), 4)
            if nonprod else None,
        }
    return out


def ablation_significance(on_recs: list[dict], off_recs: list[dict],
                          on_sum: dict, off_sum: dict) -> dict:
    """Which ablation deltas are real and which are 53 cases of noise.

    Written because the raw deltas invite exactly the wrong reading. Typology looks like
    the headline — it nearly doubles — but it rests on twelve labelled cases and does not
    survive a test. The indicator count looks duller and is overwhelming. Reporting the
    deltas without this table would put the emphasis on the one number that cannot hold
    it.
    """
    from scipy.stats import fisher_exact, mannwhitneyu

    n_lab = on_sum["typology"]["n_labelled_cases"]
    a = round((on_sum["typology"]["any_match_accuracy"] or 0) * n_lab)
    b = round((off_sum["typology"]["any_match_accuracy"] or 0)
              * off_sum["typology"]["n_labelled_cases"])
    _, p_typ = fisher_exact([[a, n_lab - a], [b, n_lab - b]], alternative="greater")

    def flags(recs):
        return {r["unit_id"]: len(r["result"].get("red_flag_indicators") or [])
                for r in recs}
    fa, fb = flags(on_recs), flags(off_recs)
    keys = sorted(set(fa) & set(fb))
    _, p_flags = mannwhitneyu([fa[k] for k in keys], [fb[k] for k in keys],
                              alternative="greater")

    return {
        "typology_any_match": {
            "on": f"{a}/{n_lab}", "off": f"{b}/{n_lab}",
            "fisher_p_one_sided": round(float(p_typ), 4),
            "reading": ("suggestive but NOT significant at this sample size — reported "
                        "as a direction, not a result"),
        },
        "red_flag_indicators_per_case": {
            "on": on_sum["red_flags"]["mean_per_case"],
            "off": off_sum["red_flags"]["mean_per_case"],
            "cases_naming_none_on": on_sum["red_flags"]["cases_naming_none"],
            "cases_naming_none_off": off_sum["red_flags"]["cases_naming_none"],
            "mann_whitney_p_one_sided": float(f"{p_flags:.3g}"),
            "reading": ("the ablation's real result: retrieval is what makes the note "
                        "cite published indicators rather than assert suspicion"),
        },
    }


def ablation(on: dict, off: dict) -> dict:
    """Retrieval on minus retrieval off, on the metrics retrieval could plausibly move."""
    def delta(path: list[str]) -> float | None:
        a, b = on, off
        for k in path:
            a, b = (a or {}).get(k), (b or {}).get(k)
        return round(a - b, 4) if isinstance(a, (int, float)) and isinstance(
            b, (int, float)) else None

    return {
        "escalation_precision": delta(["agent_as_classifier", "precision"]),
        "recall": delta(["agent_as_classifier", "recall"]),
        "fp_removal_rate": delta(["triage_effect", "fp_removal_rate"]),
        "tp_loss_rate": delta(["triage_effect", "tp_loss_rate"]),
        "typology_any_match": delta(["typology", "any_match_accuracy"]),
        "cost_per_case_usd": delta(["cost", "per_alert_usd"]),
        "red_flags_named": delta(["red_flags", "mean_per_case"]),
    }


def red_flags(records: list[dict]) -> dict:
    counts = [len(r["result"].get("red_flag_indicators") or []) for r in records]
    named = Counter(f for r in records
                    for f in (r["result"].get("red_flag_indicators") or []))
    sources = Counter(s for r in records
                      for s in (r["result"].get("sources_cited") or []))
    return {
        "mean_per_case": round(sum(counts) / len(counts), 3) if counts else 0.0,
        "cases_naming_none": sum(1 for c in counts if c == 0),
        "distinct_indicators": len(named),
        "most_named": dict(named.most_common(8)),
        "sources_relied_on": dict(sources.most_common()),
    }


def score(path: Path, frame: pd.DataFrame, membership: pd.DataFrame,
          split: str) -> dict:
    records = json.loads(path.read_text())
    summary = summarise_mod.summarise(records)
    truth = case_truth(records, frame, membership, split)
    summary["typology"] = typology_scores(records, truth)
    # Per-case truth travels in the artifact so the README and the demo can name a
    # specific case's ground-truth ring without reloading 5M transactions and redoing
    # the pattern join. It is small (one row per case) and it is the only place the
    # mapping from case to injected ring is written down.
    summary["case_truth"] = truth
    summary["by_case_size"] = by_case_size(records)
    summary["discrimination"] = discrimination(records)
    summary["red_flags"] = red_flags(records)
    summary["source_file"] = path.name
    return summary


def render(report: dict) -> str:
    t = report["test_retrieval_on"]
    lines = [
        "=" * 74,
        f"  AGENT EVALUATION — {t['n_alerts']} cases, {report['split']} split, "
        "scored once",
        "=" * 74,
        summarise_mod.render(t, unit="case"),
        "",
        "  TYPOLOGY vs Patterns.txt (ring level)",
        f"    {t['typology']['coverage_note']}",
        f"    any-match      {(t['typology']['any_match_accuracy'] or 0) * 100:.1f}%"
        f"   vs {(t['typology']['baselines']['any_match_expected_by_chance'] or 0) * 100:.1f}%"
        " expected by chance (per-case: big cases contain many rings)",
        f"    dominant-match {(t['typology']['dominant_match_accuracy'] or 0) * 100:.1f}%"
        f"   vs {(t['typology']['baselines']['majority_class_dominant_accuracy'] or 0) * 100:.1f}%"
        f" for always predicting {t['typology']['baselines']['majority_class']}",
        f"    clean cases correctly called NONE: "
        f"{t['typology']['clean_cases_called_none']}/{t['typology']['n_clean_cases']}",
        "",
        "  DOES THE DISPOSITION CARRY SIGNAL? (Fisher exact, one-sided)",
    ]
    for name, d in t["discrimination"].items():
        if name == "reading" or not d:
            continue
        lines.append(
            f"    {name:<8} n={d['n']:>3}  escalation precision "
            f"{d['escalation_precision'] * 100:5.1f}%  vs base "
            f"{d['queue_base_rate'] * 100:5.1f}%  ({d['lift_pts']:+.1f} pts)  "
            f"p={d['fisher_p_one_sided']}")
    lines += [
        f"    -> {t['discrimination']['reading']}",
        "",
        "  WHERE THE ERRORS ARE (case size)",
    ]
    for name, d in t["by_case_size"].items():
        if not d["n_cases"]:
            continue
        lines.append(
            f"    {name:<5} {d['n_cases']:>3} cases  "
            f"TP lost {d['tp_lost']}/{d['n_productive']}"
            + (f" ({d['tp_loss_rate'] * 100:.0f}%)" if d["tp_loss_rate"] is not None
               else "")
            + f"   FP removed {d['fp_removed']}"
            + (f" ({d['fp_removal_rate'] * 100:.0f}%)"
               if d["fp_removal_rate"] is not None else ""))

    if report.get("ablation"):
        off, a = report["test_retrieval_off"], report["ablation"]
        lines += [
            "",
            "  RAG ABLATION — retrieval on minus retrieval off",
            f"    escalation precision  {t['agent_as_classifier']['precision'] * 100:.1f}%"
            f"  vs {off['agent_as_classifier']['precision'] * 100:.1f}%"
            f"   ({a['escalation_precision'] * 100:+.1f} pts)",
            f"    fp removal            "
            f"{(t['triage_effect']['fp_removal_rate'] or 0) * 100:.1f}%"
            f"  vs {(off['triage_effect']['fp_removal_rate'] or 0) * 100:.1f}%",
            f"    tp loss               "
            f"{(t['triage_effect']['tp_loss_rate'] or 0) * 100:.1f}%"
            f"  vs {(off['triage_effect']['tp_loss_rate'] or 0) * 100:.1f}%",
            f"    typology any-match    "
            f"{(t['typology']['any_match_accuracy'] or 0) * 100:.1f}%"
            f"  vs {(off['typology']['any_match_accuracy'] or 0) * 100:.1f}%",
            f"    red flags named/case  {t['red_flags']['mean_per_case']}"
            f"  vs {off['red_flags']['mean_per_case']}",
            f"    cost per case         ${t['cost']['per_alert_usd']:.5f}"
            f"  vs ${off['cost']['per_alert_usd']:.5f}",
        ]
        sig = report.get("ablation_significance", {})
        if sig:
            ty, rf = sig["typology_any_match"], sig["red_flag_indicators_per_case"]
            lines += [
                "    which of those survive a test:",
                f"      typology {ty['on']} vs {ty['off']}  p={ty['fisher_p_one_sided']}"
                "  -> NOT significant, twelve labelled cases",
                f"      indicators/case {rf['on']} vs {rf['off']}  "
                f"p={rf['mann_whitney_p_one_sided']}  -> the real effect; notes naming "
                f"no indicator at all: {rf['cases_naming_none_on']} vs "
                f"{rf['cases_naming_none_off']} of 53",
            ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    on_path = config.RESULTS / f"triage_cases_{args.split}.json"
    off_path = config.RESULTS / f"triage_cases_{args.split}_norag.json"
    if not on_path.exists():
        print(f"missing {on_path.name} — run `python src/agent/run_cases.py "
              f"--split {args.split}` first")
        return 1

    frame = splits.load_transactions()
    membership = engine_eval.load_pattern_membership(frame)

    report = {
        "split": args.split,
        "model": config.ANTHROPIC_MODEL,
        "effort": config.AGENT_EFFORT,
        "test_retrieval_on": score(on_path, frame, membership, args.split),
    }
    if off_path.exists():
        report["test_retrieval_off"] = score(off_path, frame, membership, args.split)
        report["ablation"] = ablation(report["test_retrieval_on"],
                                      report["test_retrieval_off"])
        report["ablation_significance"] = ablation_significance(
            json.loads(on_path.read_text()), json.loads(off_path.read_text()),
            report["test_retrieval_on"], report["test_retrieval_off"])
    else:
        print(f"note: {off_path.name} not present, skipping the RAG ablation\n")

    print(render(report))
    # `agent.json` is the reported artifact and belongs to the test split; a val run of
    # the same scorer is a development read and must not overwrite it.
    out = config.RESULTS / ("agent.json" if args.split == "test"
                            else f"agent_{args.split}.json")
    out.write_text(json.dumps(report, indent=2))
    print(f"\n  wrote {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
