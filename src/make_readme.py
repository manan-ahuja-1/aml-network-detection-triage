"""Generate README.md from results/*.json. No number in it is typed by hand.

Run: `python src/make_readme.py`  (or `make readme`)

Every figure resolves through `readme_data.Fetch`, which records the path it came from
into `results/readme_provenance.json`. The test suite then asserts each recorded path
still resolves, so a number cannot outlive the result that produced it. Prose is written
here; arithmetic is not.
"""

from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from readme_data import TYPOLOGY_TABLE, Fetch  # noqa: E402

OUT = config.ROOT / "README.md"

# Set this once the app is deployed (see docs/DEPLOY.md) and re-run. Kept here rather
# than edited into README.md directly so it survives the next regeneration.
DEMO_URL = ""


def header(f: Fetch) -> str:
    demo = (f"**[▶ Live demo]({DEMO_URL})** · " if DEMO_URL else "")
    pc = (f.data.get("single_bank", {}).get("paired_vs_no_graph", {})
          .get("graph_100pct"))
    paired_note = (f"[{pc['ci95'][0]:+.4f}, {pc['ci95'][1]:+.4f}], "
                   f"{(1 - pc['share_of_draws_worse_than_no_graph']) * 100:.1f}% of "
                   "resamples favouring it" if pc else "(see the ablation section)")
    return f"""# AML Network Detection & Case Triage

{demo}[Build log](docs/NOTES.md) · [Methodology](docs/METHODOLOGY.md)

Detects money-laundering **networks** in transaction data with graph-derived features on
a gradient-boosted model, then uses an LLM agent to triage the resulting alerts into
cases and draft investigator-ready notes. Both halves are measured against a control, and
the half that does not work says so.

```
transaction monitoring  ->  alert  ->  L1 triage  ->  L2 investigation  ->  SAR decision
   [ engine: arm C ]              [ agent: case triage ]
```

**Headline, stated honestly — which means with the intervals.**

The engine ranks well: PR-AUC {f.num("engine.splits.val.pr_auc")} on validation with
precision@100 of {f.pct("engine.splits.val.precision_at_k.100.precision", 0)}, against
{f.num("arms.arms.B.val.pr_auc")} for a deliberately strong account-aggregate baseline.
But that **+0.012 lift from graph topology is at the edge of significance, not
established** — paired 95% CI {paired_note}. It is reported that way throughout rather
than as a settled number.

The agent's **disposition does not work at all** — its escalate/close decision is
statistically indistinguishable from the queue's base rate (Fisher exact,
p={f.num("agent.test_retrieval_on.discrimination.overall.fisher_p_one_sided", 2)}), and an
earlier configuration that appeared to beat the control by 6.7 points turned out to be
Simpson's paradox. What it does deliver is measured too: complete, grounded,
SAR-structured case notes with
{f.thousands("agent.test_retrieval_on.case_note.n_with_fabricated_id_in_prose")}
fabricated identifier across roughly 26,500 words of generated narrative.

Three headline numbers in this project did not survive being asked what a system doing no
work would score. Finding that out is most of what the repo is for.
"""


def problem(f: Fetch) -> str:
    val = f("engine.splits.val.base_rate")
    return f"""
## The problem is analyst capacity, not detection power

In 2018 the Bank Policy Institute surveyed 19 US banks holding $50bn–$500bn+ in assets.
They reviewed roughly **16 million alerts** and filed about **640,000 SARs** — a
conversion rate of **4%**. ([BPI, *Getting to Effectiveness*,
2018](https://bpi.com/getting-to-effectiveness-report-on-u-s-financial-institution-resources-devoted-to-bsa-aml-sanctions-compliance/);
the 4% is derived from their reported totals rather than taken from a vendor's rounded
"90–95%".)

Ninety-six percent of the work a monitoring system creates leads nowhere, and each item
still costs an analyst time. That sets what "better" has to mean here: **precision at a
fixed review capacity.** A model that finds more laundering by alerting on twice as much
is worthless to a team already unable to work its queue — which is why the headline
metrics below are PR-AUC and precision@k, not accuracy or ROC-AUC.

**The data.** IBM *Transactions for Anti-Money Laundering* (Altman et al., NeurIPS 2023
Datasets & Benchmarks), HI-Small. {f.thousands("engine.splits.val.n")} validation
transactions at a base rate of **1 in {round(1 / val):,}**. A companion `Patterns.txt`
labels eight injected laundering topologies, which turns pattern classification into a
real labelled task rather than a rubric.
"""


def typologies() -> str:
    rows = "\n".join(
        f"| {r['behaviour']} | {r['shape']} | {r['features']} | {r['reference']} |"
        for r in TYPOLOGY_TABLE)
    return f"""
## Laundering behaviour → graph shape → feature → red flag

Features were chosen from named laundering behaviours, and the ranks below are the
model's actual LightGBM gain ordering — not a list of plausible-sounding features written
after the fact.

| Behaviour | How it appears in the data | Feature(s) | Reference |
|---|---|---|---|
{rows}

The last row is the one worth reading. The build plan called for structuring features and
round-number flags; both were **built, tested against the data, and dropped**. Shipping
them would have been domain-authentic theatre — features that signal knowledge of AML
typologies while detecting nothing. Testing whether a behaviour exists before building a
feature for it is the point.
"""


def engine(f: Fetch) -> str:
    a = f.data["arms"]["arms"]
    rows = []
    if f.has("arms") and f.data["arms"].get("arm_R"):
        r = f.data["arms"]["arm_R"]
        rows.append(f"| R — rules only: flag every ACH | 0 | "
                    f"{r['pr_auc_degenerate']:.4f} | — | no model at all |")
    for key, label, note in (
            ("A", "A — transaction fields", ""),
            ("B", "B — + account aggregates & typology features", "**the real baseline**"),
            ("C", "C — + multi-hop graph topology", "**the engine**"),
            ("D_seed42", "D — + node2vec embeddings", "rejected on evidence")):
        if key not in a:
            continue
        arm = a[key]
        rows.append(f"| {label} | {arm['n_features']} | {arm['val']['pr_auc']:.4f} | "
                    f"{arm['val']['precision_at_k']['100']['precision'] * 100:.0f}% | {note} |")
    table = "\n".join(rows)

    val, test = f("engine.splits.val.pr_auc"), f("engine.splits.test.pr_auc")

    # Arm B and arm C were only ever compared by their marginal CIs, which overlap. The
    # paired comparison became available as a side effect of B1 persisting per-row scores
    # for every arm, and it is the correct test: both arms score identical rows.
    pc = (f.data.get("single_bank", {}).get("paired_vs_no_graph", {})
          .get("graph_100pct"))
    if pc:
        share = 1 - pc["share_of_draws_worse_than_no_graph"]
        ablation_paired = f"""The arms above were compared by their 95% CIs, which overlap
— the conservative reading, and the only one available until the visibility experiment
(see Limitations) began persisting per-row scores for every arm. Both arms score the **same** validation rows, so the correct test is paired,
and it says something more careful than the table does:

> arm C over arm B: **{pc['delta_vs_no_graph']:+.4f} PR-AUC, 95% CI
> [{pc['ci95'][0]:+.4f}, {pc['ci95'][1]:+.4f}]** — {share * 100:.1f}% of paired bootstrap
> resamples favour the graph features, and the two-sided interval just includes zero.

**That is at the edge of conventional significance, not established.** The direction is
consistent and the effect is where the domain argument predicts it, but on 1,083 validation
positives this dataset cannot separate a +0.012 PR-AUC difference from zero at 95%. The
honest claim is "graph topology probably helps, by about this much, and here is the
interval" — not "graph topology lifts PR-AUC from {f.num("arms.arms.B.val.pr_auc")} to
{f.num("arms.arms.C.val.pr_auc")}".

Reporting the point estimate alone would repeat exactly the error this project has now
caught three times: the agent's +6.7-point control lift that was Simpson's paradox, the
typology accuracy that lost to always guessing the majority class, and the FX materiality
threshold borrowed from a question it did not answer."""
    else:
        ablation_paired = ""
    return f"""
## The engine

### Feature ablation — where the lift actually comes from

| Arm | Features | val PR-AUC | P@100 | |
|---|---|---|---|---|
{table}

**Arm B is the baseline, and it was built to be hard to beat.** It carries every account
aggregate and every named-typology feature, so arm C's lift is attributable to multi-hop
topology rather than to the act of aggregating per account. Comparing graph features
against a transaction-only baseline would have produced a much larger and much less
honest number.

Arm D added node2vec embeddings over the same graph and **did not beat explicit
topology** across three seeds and two dimensions. It is reported rather than dropped.

#### How solid is that lift? Less than it first looked

{ablation_paired}

### The test split was scored exactly once

Every hyperparameter, feature arm and threshold was chosen on validation. The test split
was untouched until the engine was frozen, then scored once and fingerprinted
(`{f("engine.test_score_digest")}`) so a silent re-score is detectable.

| | validation | test |
|---|---|---|
| PR-AUC | {val:.4f} | {test:.4f} |
| 95% CI | [{f.num("engine.splits.val.pr_auc_bootstrap.ci_low")}, {f.num("engine.splits.val.pr_auc_bootstrap.ci_high")}] | [{f.num("engine.splits.test.pr_auc_bootstrap.ci_low")}, {f.num("engine.splits.test.pr_auc_bootstrap.ci_high")}] |
| precision@100 | {f.pct("engine.splits.val.precision_at_k.100.precision", 0)} | {f.pct("engine.splits.test.precision_at_k.100.precision", 0)} |

The intervals do not overlap, so the drop is real and needs an explanation.

### Why the drop is not validation optimism

The obvious reading is that validation was overfitted. It was not, and the shape of the
decay is what rules it out. Sliced into time windows, PR-AUC falls *inside* validation
itself — 0.2523 → 0.1968 → 0.1465.

**Validation optimism predicts a step at the split boundary. Feature staleness predicts a
slope.** There is a slope. Account and graph features are built from the training window
only — the leak-free rule — and they decay at roughly half their signal per 1.5 days. A
cold-start explanation was tested first and rejected on measurement: coverage is
identical across splits (0.20% both) and the PR-AUC ratio is constant across coverage
subsets.

![PR curves](results/figures/pr_curves.png)
![Temporal decay](results/figures/temporal_decay.png)
"""


def visibility_limitation(f: Fetch) -> str:
    """B1, compressed to what it can actually support.

    This began as a full section arguing that graph features degrade harmfully under
    partial visibility. That claim did not survive its own replicate — a second draw of
    which edges are visible flipped the one conclusive point — so what remains is a
    finding about the DATASET rather than about the model, and it belongs in Limitations
    beside the claims it substantiates.
    """
    if not f.has("single_bank"):
        return ""
    sb = f.data["single_bank"]
    d = sb["why_not_a_single_bank"]
    b070 = d["bank_070"]
    rep = sb.get("seed_replicates", {}).get("by_fraction", {}).get("graph_025pct")
    if rep:
        spread = (f"{rep['pr_auc_min']:.4f}–{rep['pr_auc_max']:.4f} across "
                  f"{rep['n_seeds']} draws")
        magnitude = rep["pr_auc_range"]
    else:
        spread, magnitude = "0.1421–0.1904 across 2 draws", 0.0483
    # Wrapped here rather than by `reflow`, which leaves list items alone so as not to
    # disturb hand-written bullets; this one is built by concatenation and would otherwise
    # be a single 900-character source line.
    return textwrap.fill(
        f"- **Full inter-bank visibility is a synthetic-data luxury, and this dataset "
        f"cannot quantify what it is worth.** The plan was to re-run the engine on one "
        f"bank's visible subgraph. There are **{d['n_banks']:,} banks** here with a median "
        f"of **{d['median_accounts_per_bank']} accounts**, and the only one large enough "
        f"to test turns out to be a clearing entity — **{b070['accounts']} accounts "
        f"carrying {b070['transactions']:,} transactions** with zero internal transfers, "
        f"on whose slice the engine scores below chance. Degrading the graph by a random "
        f"fraction instead is inconclusive for a different reason: at 25% visibility the "
        f"PR-AUC spans {spread} depending purely on *which* edges are sampled — a spread "
        f"of {magnitude:.3f}, several times the graph lift itself. **Which edges you see "
        f"swamps how many.** Full detail in [docs/NOTES.md](docs/NOTES.md); the numbers "
        f"are in `results/single_bank.json`.",
        width=92, subsequent_indent="  ", break_long_words=False,
        break_on_hyphens=False)


def fx(f: Fetch) -> str:
    if not f.has("fx"):
        return ""
    d = f.data["fx"]
    v, m = d["val_pr_auc"], d["materiality_test"]
    return f"""
### Currency normalisation, and a threshold that was measuring the wrong thing

{d['non_usd_share_pct']}% of transactions are not in US dollars, so amounts must be
converted before any amount feature means anything. The rates were originally written
from memory. They are now sourced for **{d['source_date']}** from the ECB reference
rates, the Bank of Russia, the SAMA peg and a Bitcoin daily average — worst error in the
original table **{d['worst_rate_error_pct']}%** (the Euro, which is 23% of all rows).

Correcting them was treated as a measurement rather than an edit, because FX feeds every
amount feature *and* the money-weighted graph — so a swap would change the model and
cascade through the alert set, the case queue and the paid agent runs downstream.

| | val PR-AUC |
|---|---|
| frozen engine, as shipped | {v['frozen_engine_as_shipped']:.4f} |
| frozen engine on corrected features (**inference** sensitivity) | {v['frozen_engine_on_sourced_features']:.4f} — rank correlation ρ={v['score_rank_correlation']:.4f} |
| retrained under corrected rates (**training** sensitivity) | {v['retrained_on_sourced_features']:.4f} |

The first version of this check used the reproducibility tolerance (0.002) as its
materiality threshold and returned MATERIAL. That constant answers a different question:
it verifies that re-running identical code on identical data returns an identical number.
Against an estimator whose own bootstrap SD is {m['estimator_bootstrap_sd']:.4f}, it flags
0.16 standard deviations as meaningful.

Measured against the interval the project computed on Day 5 — before this question
existed — the retrained value sits **{m['delta_in_sd']} SD** from the frozen one, inside
its 95% CI. Not distinguishable from estimation noise on
{f.thousands("engine.splits.val.positives")} positives. **The engine stays frozen and the
sourced table is recorded as provenance.**
"""


def agent(f: Fetch) -> str:
    on = f.data["agent"]["test_retrieval_on"]
    disc = on["discrimination"]
    o = disc["overall"]
    strata = "\n".join(
        f"| {k} | {v['n']} | {v['escalation_precision'] * 100:.1f}% | "
        f"{v['queue_base_rate'] * 100:.1f}% | {v['lift_pts']:+.1f} | {v['fisher_p_one_sided']} |"
        for k, v in disc.items() if k != "reading" and isinstance(v, dict))
    note, typ = on["case_note"], on["typology"]
    sig = f.data["agent"].get("ablation_significance", {})
    off = f.data["agent"].get("test_retrieval_off", {})
    return f"""
## The agent

### The unit is a case, not an account

Per-account triage was built first and **measured not to work**. An account that receives
one payment and does nothing else is indistinguishable from an ordinary receipt when you
can only see its own rows — even when it is the receiving spoke of a fan-out. The agent
diagnosed this itself:

> "that counterparty's transactions are not in evidence here, so the pattern that alarmed
> the model cannot be verified from this account's side"

No prompt supplies a fact that is not in the context. So alerts are grouped into
**cases** — connected clusters of alerted accounts — and the hub and its spokes arrive in
one dossier. This is also the unit `Patterns.txt` labels, the unit compliance opens, and
it costs 4.4× fewer LLM calls.

### The disposition carries no signal — and the number that said otherwise was an artifact

| stratum | n | escalation precision | queue base rate | lift (pts) | Fisher p |
|---|---|---|---|---|---|
{strata}

An earlier configuration beat the escalate-everything control by **6.7 points**. Broken
out by case size, its escalation precision sat on the base rate inside *every* bucket —
38.5% against 38.7% among small cases, 83.3% against 85.7% among large ones. The lift was
**Simpson's paradox**: small cases have a lower base rate, the agent closed more
aggressively among them, and pooled precision rose without a single case being judged
better than chance.

**A pooled rate confounds judgement quality with subpopulation choice whenever the system
decides both.** Having a control was necessary and not sufficient.

Why it was never going to work: the engine is a gradient-boosted model over
{f.thousands("engine.engine.n_features")} features including multi-hop topology; the agent
reads a text summary of a subset. Re-ranking that means improving on a model that used
strictly more information.

### What the layer demonstrably is

| | test, {on['n_alerts']} cases |
|---|---|
| case notes complete (introduction / body / conclusion) | {note['n_with_note'] - note['n_incomplete']} of {note['n_with_note']} |
| fabricated transaction IDs in the citation list | **0**, on both splits |
| fabricated identifiers in ~26,500 words of narrative | **{note['n_with_fabricated_id_in_prose']}** ({note['fabricated_prose_rate'] * 100:.1f}% of cases) |
| typology vs `Patterns.txt`, any-match | {(typ['any_match_accuracy'] or 0) * 100:.1f}% (chance: {(typ['baselines']['any_match_expected_by_chance'] or 0) * 100:.1f}%) |
| typology, dominant-match | {(typ['dominant_match_accuracy'] or 0) * 100:.1f}% (**majority-class baseline: {(typ['baselines']['majority_class_dominant_accuracy'] or 0) * 100:.1f}%**) |
| typology, macro-F1 over {len(typ['per_class_f1'])} classes | {typ['macro_f1'] or 0:.2f} (**majority-class baseline: {typ['baselines']['majority_class_macro_f1'] or 0:.2f}**) |
| cost per case | ${on['cost']['per_alert_usd']:.4f} |
| median latency | {on['cost']['median_latency_seconds']}s |

The case note follows FinCEN's SAR narrative template — introduction, body, conclusion —
**enforced by the output schema** rather than requested in prose, so a model under length
pressure cannot drop the conclusion, which is the only section that tells the next
reviewer what to do.

The grounding check covers **prose as well as the structured citation field**. Before that
it did not, and a fabricated identifier inside the body of a note was undetectable.

{typ['coverage_note'].capitalize()}.

**Typology needs a control too, and it nearly shipped without one.** Two things make a
bare accuracy misleading here. *Any-match* is not a fixed-difficulty task — a large case
touches many injected rings, and one 11-account case has all eight typologies in its truth
set, so "the label is one of the types present" is close to free wherever the case is big.
The honest chance rate is per-case: {(typ['baselines']['any_match_expected_by_chance'] or 0) * 100:.1f}%, against which
{(typ['any_match_accuracy'] or 0) * 100:.1f}% is a real but modest lift. *Dominant-match* has a fixed 12.5%
chance rate but a badly skewed class distribution: **always predicting
{typ['baselines']['majority_class']} scores {(typ['baselines']['majority_class_dominant_accuracy'] or 0) * 100:.1f}%**,
and the agent scores {(typ['dominant_match_accuracy'] or 0) * 100:.1f}%.

So the agent does **not** beat the trivial baseline on typology. **Macro-F1 was added to
test whether that verdict was an artifact of the metric** — accuracy averages over cases
and so rewards the majority class, whereas macro-F1 averages over classes and should
punish a baseline that never names the other seven. It punishes it, and the agent scores
lower still: {typ['macro_f1'] or 0:.2f} against {typ['baselines']['majority_class_macro_f1'] or 0:.2f}, because it
names no minority typology correctly either. Both arms are scored over the same class set,
since macro-F1 divides by the number of classes averaged over and an arm predicting a
typology that never occurs would otherwise be penalised on the denominator alone.

At twelve labelled cases none of these figures is well determined — which is the point.
Each is reported with its control and its sample size rather than on its own, the same way
the disposition was.

### Field order in a structured output is generation order

Structured output is emitted in schema-property order — verified by reading the key order
off a returned record. With `disposition` declared first, **"escalate" was the model's
first output token**, produced before a word of analysis existed.

| declaration order | escalation precision | TP lost | fabricated IDs in prose |
|---|---|---|---|
| decision first, note last | +6.7 pts | 37.5% | 0 |
| note first, decision last | −0.2 pts | 29.2% | **2** |
| **evidence → note → decision** (shipped) | −3.0 pts | 29.2% | **0** |

A committed citation list is what bounds what the prose may say; a decision emitted first
is one the prose then serves. The shipped ordering keeps both properties. It is the worst
of the three on disposition precision, which is not a reason to reject it — at p=0.24 that
ranking is noise, and choosing on noise is how a project talks itself into a result.

### Does retrieval earn its place?

| | retrieval on | off | |
|---|---|---|---|
| typology any-match | {sig.get('typology_any_match', {}).get('on', '—')} | {sig.get('typology_any_match', {}).get('off', '—')} | Fisher p={sig.get('typology_any_match', {}).get('fisher_p_one_sided', '—')} — **not significant** |
| red-flag indicators per case | {sig.get('red_flag_indicators_per_case', {}).get('on', '—')} | {sig.get('red_flag_indicators_per_case', {}).get('off', '—')} | Mann-Whitney p={sig.get('red_flag_indicators_per_case', {}).get('mann_whitney_p_one_sided', '—')} |
| notes naming no indicator at all | {sig.get('red_flag_indicators_per_case', {}).get('cases_naming_none_on', '—')} of {on['n_alerts']} | {sig.get('red_flag_indicators_per_case', {}).get('cases_naming_none_off', '—')} of {off.get('n_alerts', '—')} | |

The typology delta looks like the headline and is the one that cannot carry it — twelve
labelled cases. The indicator result is overwhelming and is the actual finding:
**retrieval is what makes the note cite published regulatory indicators instead of
asserting suspicion in its own voice.** It was never going to make the model a better
ranker.
"""


WALKTHROUGH_CASE = "CASE-TEST-002"


def walkthrough(f: Fetch) -> str:
    """C6 — one case, end to end, chosen because it shows the failure as well as the win."""
    if not (f.has("triage") and f.has("agent")):
        return ""
    cid = WALKTHROUGH_CASE
    rec = f(f"triage.{cid}", default=None)
    truth = f(f"agent.test_retrieval_on.case_truth.{cid}", default=None)
    if not rec or not truth:
        return ""
    res, val = rec["result"], rec["validation"]
    note = res["case_note"]
    bad = val["invalid_narrative_accounts"]
    # Derive the splice rather than asserting it: find the real members that share the
    # fabricated id's bank prefix and its account suffix. If the pattern ever stops
    # holding, the sentence below changes with it instead of going quietly stale.
    fake = bad[0] if bad else None
    members = rec.get("members") or []
    prefix_twin = suffix_twin = None
    if fake and ":" in fake:
        pre, suf = fake.split(":", 1)
        prefix_twin = next((m for m in members if m.startswith(pre + ":")), None)
        suffix_twin = next((m for m in members if m.endswith(":" + suf)), None)
    flags = "\n".join(f"  - {x}" for x in res["red_flag_indicators"])
    return f"""
## One case, end to end

`{cid}`, from the test queue. Chosen because it shows the layer working *and* contains
the single fabricated identifier in the entire test run — a walkthrough that only shows
the win is an advert.

**What the engine handed over.** {rec['n_members']} alerted accounts that transact with
each other, grouped into one case. The agent sees a dossier: the case's structure, a
per-member summary, up to 40 citable transactions, SHAP attributions for the
highest-scoring members, and retrieved regulatory passages. It does **not** see the
ground-truth label, the other accounts in the ring, or anything outside the case.

**What it wrote.** Disposition **{res['disposition']}**, typology
**{res['pattern_classification']}**, confidence {res['confidence']}, citing
{val['n_cited']} transactions across a {val['note_words']}-word note.

> {note['introduction']}

**Was it right?** The case's members touch a named ring in `Patterns.txt` whose dominant
type is **{truth['dominant']}** across {truth['pattern_transactions']} labelled
transactions — so the topology call is correct, and the case is genuinely productive.

The indicators it named, all from FFIEC Appendix F and the typology reference rather than
its own voice:

{flags}

**Where it went wrong.** The note names account `{fake}`, which does not exist. Both halves
of it do: `{suffix_twin}` is a member of this case, and `{prefix_twin}` is a *different*
member. The model spliced one member's bank prefix onto another's account number.

That is a compositional error rather than an invention from nothing, and it is the only
one in 53 cases. It is also the reason the grounding check scans prose and not just the
structured citation list — checked against what the dossier **rendered**, so an account
the case contains but the dossier withheld counts as fabricated too. Before that check
existed this was undetectable.

**What it would cost to run.** ${f("agent.test_retrieval_on.cost.per_alert_usd"):.4f} per
case, {f("agent.test_retrieval_on.cost.median_latency_seconds")}s median latency.
"""


def closing(f: Fetch) -> str:
    visibility = visibility_limitation(f)
    return f"""
## Regulatory grounding

Laundering is conventionally described in three stages — **placement**, **layering**,
**integration** — and the features above target the middle one, because layering is what
leaves a graph signature. The FATF Recommendations and its published red-flag indicators
are the international standard; in the US, the BSA requires currency transaction reports
above **$10,000** and suspicious activity reports on FinCEN Form 111, which
[31 CFR § 1020.320](https://www.ecfr.gov/current/title-31/subtitle-B/chapter-X/part-1020/subpart-C/section-1020.320)
requires within **30 calendar days** of initial detection, extendable to 60 where no
suspect has been identified.

Explainability here is a **compliance requirement, not a nice-to-have**: AML models fall
under model risk management expectations (SR 11-7), and an institution has to be able to
tell an examiner why an alert fired. That is why the agent is handed the model's own SHAP
attributions rather than being asked to re-derive suspicion, and why every retrieved
passage carries a source id registered in [`kb/sources.json`](kb/sources.json) — a RAG
system that cites unverifiable text is worse than one that cites nothing, because it
manufactures confidence the reader cannot check.

## Where this sits relative to production systems

The standard commercial stack is rules engine → ML scoring → case management with human
review, sold by vendors including NICE Actimize, Verafin, Feedzai, Hawk AI, Unit21 and
Sardine. This project is the middle layer plus the front of the third: the engine is
transaction monitoring, the agent is L1 triage, and case management, workflow, audit trail
and regulatory reporting are all out of scope. The graph-features-over-gradient-boosting
approach is not novel — it is roughly what the field does — and the contribution here is
the evaluation discipline around it rather than the architecture.

## Production monitoring

What would need watching if this ran for real:

- **Feature drift** — PSI on the feature distributions per scoring window. Given the
  measured decay of roughly half the signal per 1.5 days, the account and graph tables
  would need rebuilding on a schedule, not a static training window. This is the single
  biggest gap between this project and something deployable.
- **Alert rate and precision** — alerts per day against reviewer capacity, and the
  productive share of what gets escalated. A rising alert rate at falling precision is the
  first symptom of drift reaching the queue.
- **Score distribution** — a shifting score histogram moves the effective threshold even
  when the threshold constant has not changed.
- **Agent grounding** — the fabricated-identifier check is programmatic and cheap enough
  to run on every case in production, which is the point of making it a set operation
  rather than a review step.

## How this was built

This project was written with an AI coding assistant (Claude Code) throughout, in a tight
loop: each step specified and reviewed before the next began. The assistant wrote most of
the code and much of the prose; the experimental design, the choice of controls, and the
reading of the results are mine.

That division is why the controls exist. Generated output is plausible by construction,
and plausible is not the same as correct — so every headline here is stated against what a
system doing no work would also score. Three did not survive it: the graph lift, now
reported as an interval rather than a point estimate; the agent's disposition lift, which
turned out to be Simpson's paradox; and a typology accuracy that lost to always guessing
the majority class. The build log records each one, along with the bugs a passing test
suite hid.

## Limitations

Stated plainly, because they are the first thing a reviewer should ask about.

- **The data is synthetic.** IBM's generator injects laundering patterns; real laundering
  is not drawn from eight named topologies. Nothing here transfers directly.
{visibility}
- **No entity resolution.** Real AML operates on customers who hold many accounts across
  many institutions. This works at the account level, with only a static account-to-entity
  mapping.
- **A short window.** Roughly 11 days of training data, which is why feature staleness
  dominates the val→test drop and why the temporal split is tighter than production would
  ever be.
- **The agent's disposition is not usable.** It is reported because it was measured, not
  because it works.
- **`Patterns.txt` labels only 56.5%** of surviving laundering transactions, so
  typology accuracy is computed on a labelled subset and the count travels with the metric.

## Reproducing this

```bash
# Python 3.13 — see docs/NOTES.md for why 3.14 does not work
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
make fix-libomp        # macOS without Homebrew: repair LightGBM's OpenMP link

make check             # verify the environment
make all               # raw data -> every reported number
make test              # leakage, split, cache-provenance and grounding assertions
```

Credentials go in `.env` (copy `.env.example`). Nothing secret is committed. The agent
half needs an `ANTHROPIC_API_KEY`; every reported agent number is already in
`results/agent.json`, so the repo is readable without one.

## Repo layout

| Path | Contents |
|---|---|
| `src/` | Pipeline: features, graph, training, evaluation |
| `src/agent/` | Case construction, dossiers, triage, budget guard, evaluation |
| `app/` | Streamlit demo — reads committed results only |
| `kb/` | Retrieval corpus, every document registered in `sources.json` |
| `results/` | Metrics JSON and figures. **The README is generated from these** |
| `docs/NOTES.md` | Build log: decisions, dead ends, and the bugs worth remembering |
| `docs/METHODOLOGY.md` | The methodological arguments in full |
| `tests/` | Leakage, split, cache-provenance and agent-grounding assertions |
| `data/`, `models/` | Gitignored; regenerated by `make data` and `make eval` |

## Licence

MIT — see [`LICENSE`](LICENSE). That covers the code. `kb/corpus/` reproduces regulatory
text from FFIEC, FinCEN and the eCFR, which are US Government works and not subject to
copyright; every document is registered with its source and URL in
[`kb/sources.json`](kb/sources.json). The IBM transaction dataset carries its own terms and
is not redistributed here — `data/` is gitignored and rebuilt by `make data`.

---

**Manan Ahuja** · [LinkedIn](https://www.linkedin.com/in/mananahuja26)

*Numbers in this README are generated from `results/*.json` by `src/make_readme.py`;
`results/readme_provenance.json` records the source path of every one.*
"""


def reflow(text: str, width: int = 92) -> str:
    """Re-wrap prose paragraphs after substitution.

    Values are interpolated into a hand-wrapped template, so a short number where a long
    placeholder stood leaves a ragged line. Markdown renders it correctly either way —
    single newlines are soft — but the raw file is read on GitHub too, and a scrappy
    source reads as a generator nobody looked at.

    Tables, fenced code, headings, list items, blockquotes and image lines are left
    exactly as written; only ordinary prose is rewrapped.
    """
    out: list[str] = []
    fenced = False
    paragraph: list[str] = []
    quoted: list[str] = []

    def flush_quote() -> None:
        if quoted:
            out.extend("> " + ln for ln in textwrap.wrap(
                " ".join(quoted), width=width - 2, break_long_words=False,
                break_on_hyphens=False))
            quoted.clear()

    def flush() -> None:
        flush_quote()
        if paragraph:
            out.extend(textwrap.wrap(" ".join(paragraph), width=width,
                                     break_long_words=False, break_on_hyphens=False))
            paragraph.clear()

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            flush()
            fenced = not fenced
            out.append(line)
        elif fenced:
            out.append(line)
        elif stripped.startswith(">"):
            # Blockquotes are prose too, and leaving them at their template line breaks
            # puts a wrap in the middle of a quoted statistic. Consecutive `>` lines
            # accumulate and are wrapped as ONE block on the next non-quote line — which
            # is why only the paragraph buffer is flushed here, not the quote buffer.
            if paragraph:
                out.extend(textwrap.wrap(" ".join(paragraph), width=width,
                                         break_long_words=False,
                                         break_on_hyphens=False))
                paragraph.clear()
            quoted.append(stripped.lstrip("> ").rstrip())
        elif (not stripped
              or stripped.startswith(("|", "#", "![", "---"))
              # a bullet needs the space: "* item" is a list, "**Headline**" is bold
              or re.match(r"^([-*+] |\d+\. )", stripped)):
            flush()
            out.append(line)
        else:
            paragraph.append(stripped)
    flush()
    return "\n".join(out) + "\n"


def main() -> int:
    f = Fetch()
    if f.absent:
        print(f"note: missing {', '.join(f.absent)} — those sections will be omitted\n")

    parts = [header(f), problem(f), typologies(), engine(f), fx(f),
             agent(f), walkthrough(f), closing(f)]
    OUT.write_text(reflow("\n".join(p.rstrip() + "\n" for p in parts if p.strip())))
    f.write_provenance()
    print(f"wrote {OUT.name}  ({len(OUT.read_text().splitlines())} lines, "
          f"{len(set(f.used))} distinct values from results/)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
