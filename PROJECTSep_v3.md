# AML Laundering-Network Detection & Triage — Build Plan v3

## Context

`PROJECTSep.md` is a 2-week solo build plan for a CV/portfolio project: a LightGBM
laundering detector enriched with graph features, an LLM agent that triages flagged
alerts and drafts investigator case notes, and an eval harness.

The original plan is directionally excellent. The thesis (laundering is relational, so
graph features should beat transaction-only features), PR-AUC over ROC-AUC, and the
insistence on evals are all correct, and are what separate this from a typical
portfolio project.

Two rounds of revision are folded in here:

- **Pass 1 (§A)** fixes methodological defects that a technically sharp interviewer
  would find in about five minutes — chiefly a leakage hole the plan only half-closes,
  and a strawman baseline that would make the headline number indefensible.
- **Pass 2 (§B, §C)** addresses a different gap: v1 reads as a graph-ML project that
  happens to use bank data. The highest-value features and metrics available here are
  ones you would only reach for if you knew the compliance workflow — so they do double
  duty, improving the model *and* making domain fluency visible. §C is the explicit
  strategy for displaying finance-domain knowledge alongside the technical work.

Scope decision (confirmed): **everything ships inside the 2 weeks.** The schedule in §D
is built to make that real — chiefly by moving all reading-and-writing work off the
critical path into pre-work, and by flagging the two items to cut first if a day slips.

---

# §A — Pass 1: methodological corrections

## A1. The leakage guardrail is only half-correct — highest-risk defect

v1 says "use a time-based train/test split." Necessary, **not sufficient**. If you build
one NetworkX graph over the whole dataset and compute degree / centrality / Louvain /
node2vec on it, every training row carries features derived from *future* edges. The
split looks correct while the features leak anyway. This is precisely what a reviewer
probes for, and "I did a time split" is not an answer.

**Fix — graph fitted on the training window only:**
- Build `G_train` from transactions in the train window only.
- Compute all topology features and node2vec embeddings on `G_train`.
- For val/test rows, **look up** each account's features from `G_train` by account ID.
- Unseen accounts get `NaN` plus an explicit `is_unseen_account` flag. LightGBM handles
  `NaN` natively — do not impute zeros, that fabricates a signal.

This creates a genuine cold-start problem, which is *good*: it is real, it is what
production looks like, and it is a strong interview answer. Report the cold-start rate
on test.

## A2. The baseline is a strawman, which makes the headline lift dishonest

Transaction-only vs transaction-plus-graph will show a huge lift for the wrong reason:
most of it comes from *any* per-account aggregation, not from graph topology. An
interviewer will say "your baseline had no account features at all — of course it lost."

**Fix — four ablation arms; headline lift is measured against arm B:**

| Arm | Features |
|---|---|
| **A** | Raw transaction attributes only (amount, currency, format, hour, day-of-week) |
| **B** | A + account behavioural aggregates: counts, amount stats, unique counterparties, currency diversity, velocity over 1d/7d, **plus the typology features in B2** — no graph topology |
| **C** | B + graph topology: in/out degree, PageRank, reverse-PageRank, approx. betweenness, Louvain community id + size, fan-in/fan-out counts, in-cycle flag |
| **D** | C + node2vec embeddings |

An honest small lift over a real baseline beats a large one over a strawman — and "I
built the strong baseline specifically so the graph lift couldn't be an artifact" is the
most senior thing you can say about this project.

## A3. No validation set — thresholds and prompts get tuned on test

**Fix:** three-way *temporal* split, train / val / test in time order. Val carries early
stopping, threshold selection, and all prompt iteration. **Test is scored once, at the
end.** Add a bootstrap 95% CI on test PR-AUC (1000 resamples).

## A4. Multi-currency amounts are not normalised

Schema: `Timestamp, From Bank, Account, To Bank, Account.1, Amount Received, Receiving
Currency, Amount Paid, Payment Currency, Payment Format, Is Laundering`. Paid and
Received are in **different currencies** — raw amounts compare 500 JPY with 500 GBP.

**Fix:** normalise both to USD via a static FX dict (state that it is static). Then
derive `is_cross_currency` and `amount_paid_usd − amount_received_usd` (implied spread),
both of which are signal in their own right.

## A5. Two compute traps that will each eat a day

- **Betweenness centrality is O(V·E)** and will not finish. Use
  `nx.betweenness_centrality(G, k=500, seed=42)`, or drop it for **PageRank and
  reverse-PageRank** — cheap, and a better fit for money flow anyway (reverse-PageRank ≈
  how strongly flow concentrates *into* an account, which is the mule signature).
- **The `node2vec` pip package is pure Python and slow.** Use `pecanpy` or
  `fastnode2vec` (compiled, gensim-backed). dim=64, walk length 20, 10 walks/node.

## A6. Label granularity is never resolved

`Is Laundering` is a **per-transaction** label; v1 alternates between transaction and
account scoring, leaving the engine→agent handoff undefined.

**Fix, decided Day 1:** the model scores **transactions**. An **account-level alert** is a
defined aggregation over that account's transactions in the window — `max(score)` and
`count(score > threshold)`. Write the rule down; it is the seam between the two halves.

## A7. The dataset ships ground-truth typology labels and v1 doesn't use them

The Kaggle download includes **`HI-Small_Patterns.txt`** alongside `HI-Small_Trans.csv`,
containing the laundering transactions grouped into the eight typologies the generator
injected: **Fan-out, Fan-in, Gather-Scatter, Scatter-Gather, Simple Cycle, Random,
Bipartite, Stack**.

Biggest free upgrade in the plan. It converts the agent's pattern classification from a
subjective rubric into a **real 8-class task with ground truth** — accuracy, macro-F1,
confusion matrix. Parse it on Day 1 and confirm the format before building around it.
It also unlocks pattern-level recall (§B5).

## A8. The agent eval has no control

"Triage accuracy = 84%" is meaningless without knowing what escalate-everything scores.
And the business case for a triage agent is *killing false positives*, which the v1 eval
barely tests.

**Fix:**
- **Alert set** = model's top-N accounts by score, naturally containing TPs and FPs.
  Report the mix.
- **Baselines:** (a) escalate everything the model flags — the no-agent control;
  (b) agent with RAG disabled — does the knowledge base earn its place?
- **Headline agent metric:** false positives removed at **zero true-positive loss**, or
  the FP/TP trade curve if not zero.

## A9. Hallucination checking should be programmatic

**Fix:** require `cited_transaction_ids: list[str]` in the structured output, then assert
`set(cited) ⊆ set(ids in the pulled subgraph)`. Hallucination rate = share of alerts with
≥1 invalid citation. Automated and honest.

## A10. "Est. analyst-time reduction" is a made-up number

The one row in v1's metrics table that cannot be computed; it will read as padding next
to seven real metrics. **Replace with alert-volume reduction at fixed recall**, plus
precision@capacity (§B6). Add **cost and latency per alert** from logged token usage.

---

# §B — Pass 2: domain-driven additions

These improve the model *and* are the ones that signal you understand AML rather than
just graphs. Each is cheap.

## B1. Nothing in v1 uses `From Bank` / `To Bank`

Moving funds across institutions to break the audit trail is core layering behaviour,
and the bank columns are sitting unused.

- **Features:** distinct counterparty banks, cross-bank transaction ratio, count of
  institutions the account's funds traverse.
- **The realism experiment (high value):** a real bank only sees its own customers'
  transactions — never the full inter-bank graph this dataset hands you. Re-run arm C
  restricted to a **single bank's visible subgraph** and report the performance drop.
  Almost no portfolio project demonstrates awareness that the full-network view is a
  synthetic-data luxury. This is one of the strongest domain signals available, and it
  costs about half a day.

## B2. Engineer the textbook typologies explicitly

v1 names structuring, smurfing and layering in prose but never builds a feature for any
of them. These are cheap, interpretable, and each one maps to a named red flag:

- **Structuring:** count and share of transactions falling just under a reporting
  threshold (the US CTR threshold is $10,000 under the BSA — verify before citing).
  Features: `count_in_9k_to_10k_band`, `share_just_under_threshold`, round-number-amount
  flag.
- **Pass-through / mule signature:** money in ≈ money out within a short window with
  near-zero retained balance. `flow_through_ratio = min(in_usd, out_usd) / max(in_usd,
  out_usd)` over 1d/7d, and `median_time_to_forward` (how fast received funds move on).
  This is *the* canonical mule fingerprint and it is missing from v1 entirely.
- **Dormancy burst:** account age at transaction time, days since previous activity,
  burstiness. Mule accounts sit quiet then activate.
- **Payment format as placement signal:** the format column carries values like Cash,
  Cheque, ACH, Wire, Credit Card, Bitcoin (confirm the exact set on Day 1). Cash and
  crypto are classic placement-stage indicators — treat them as a named feature group,
  not anonymous one-hots.

Put these in **arm B**, not arm C. They are not graph features, and folding them into
the baseline is what makes the graph lift honest.

## B3. Feed the model's reasons to the agent — the two halves are currently disconnected

As written, the agent receives "here is a flagged account" and independently re-derives
why it is suspicious. The model's actual evidence is thrown away.

**Fix:** compute **SHAP values** per alert and pass the top contributing features into
the agent's context. The case note then explains why *this model* fired, grounded in the
model's real attribution rather than the LLM's guess.

This is also the correct regulatory posture: AML models fall under model risk management
expectations (SR 11-7 in the US), and being able to justify to an examiner why an alert
fired is not optional. "Explainability is a compliance requirement, not a nice-to-have"
is a sentence that lands in a fintech interview.

## B4. Consider time-respecting structure, not just static structure

v1 builds one static graph. A static cycle is not the same thing as money actually
travelling A→B→C→A with increasing timestamps. IBM's own graph feature preprocessor for
this dataset computes **temporal** cycles and scatter-gather patterns specifically.

At minimum add a `participates_in_time_respecting_path` flag on short cycles. Full
temporal motif enumeration is expensive — **cut this first if a day slips.**

## B5. Evaluate at the unit compliance actually cares about

A laundering ring spans many transactions; catching 1 of 20 is enough to open a case.
Transaction-level recall therefore understates operational usefulness.

**Add pattern-level recall** using `Patterns.txt`: what fraction of injected laundering
patterns did you catch *at least one* transaction from, at your operating threshold?
Report it broken down by typology — "we catch fan-in reliably and stacks poorly" is a far
more interesting finding than a single scalar, and it is exactly how a real model
validation report reads.

## B6. Set the operating point by analyst capacity, not just an invented cost ratio

Real teams have fixed review capacity — "we can work 100 alerts a day." Tuning is done
against that constraint.

- Report **precision@k** for realistic k (top 50 / 100 / 500 alerts).
- Keep the cost-sensitive threshold too, but **ground the cost ratio** in public figures
  (published per-alert review cost estimates; public AML enforcement penalties) rather
  than inventing 50×, and show a **sensitivity strip** across the ratio since it is an
  assumption. Verify any figure you cite — do not take a number from me or from an LLM
  into a CV without checking the source.

## B7. Two short README sections that cost nothing and signal seniority

- **Production monitoring:** how you would detect drift (PSI on feature distributions,
  alert-rate and precision monitoring, periodic retraining triggers). Half a page.
- **Limitations:** the data is synthetic; the full inter-bank graph is unrealistic (see
  B1); there is no customer-vs-account entity resolution, whereas real AML operates on
  customers holding many accounts; results do not transfer directly to production. State
  this plainly. Volunteering it reads as maturity and pre-empts the first question you
  will be asked.

---

# §C — Making the finance-domain knowledge visible

Domain knowledge that is only implicit in your feature code is invisible to a recruiter
and nearly invisible to an interviewer. These are the mechanisms that make it legible.
Most are writing, not compute.

## C1. The typology → feature → red flag table (highest-leverage single artifact)

One table in the README, roughly a dozen rows:

| Laundering behaviour | How it appears in the graph/data | Feature(s) | Red-flag reference |
|---|---|---|---|
| Smurfing / structuring | Many small inbound transfers just under threshold | `fan_in_count`, `count_in_9k_to_10k_band` | FATF / FinCEN structuring indicators |
| Mule pass-through | Funds in and out within hours, nothing retained | `flow_through_ratio`, `median_time_to_forward` | Mule account indicators |
| Layering across institutions | Chains traversing many banks | `distinct_counterparty_banks`, `cross_bank_ratio` | Layering stage, FATF |
| Circular flows | Time-respecting cycle returning to origin | `in_cycle`, temporal-cycle flag | Cycle typology |
| Placement | Cash/crypto-heavy inbound activity | payment-format group features | Placement stage |

This single table does more domain signalling than several paragraphs of prose, because
it proves each modelling choice was driven by a laundering behaviour rather than by
feature-engineering habit. Build it as you build the features, not at the end.

## C2. Make the narrative output look like a real SAR

FinCEN publishes guidance on SAR narrative structure — the narrative should cover who,
what, when, where and why, and describe the activity in a way an investigator can act
on. Structure the agent's output template to mirror that. Anyone from the industry
recognises the shape instantly, and it costs you a prompt template.

Also frame the pipeline in the real workflow vocabulary: **transaction monitoring →
alert → L1 triage → L2 investigation → SAR filing decision**. Your engine is the
monitoring layer; your agent is L1 triage. Say exactly that.

## C3. Use the industry's vocabulary consistently

Alert (not "flag"), disposition, escalate/close, productive vs non-productive alert,
typology, red flag indicator, false positive rate, case, investigator. Getting this
register right throughout the README and demo is a quiet but continuous signal.

## C4. Open the README with the business case, not the architecture

Transaction monitoring systems are widely reported to run false-positive rates in the
90%+ range — that number *is* the entire justification for this project, and it belongs
in the first two lines. **Find and cite a real source you have actually read** (an
industry survey, a regulator publication, a vendor study), and link it. One cited
statistic beats three uncited ones.

Then state the problem as a compliance team would: alert volume exceeds review capacity,
so the constraint is precision at capacity, not raw detection power.

## C5. Regulatory grounding — short, accurate, unshowy

A compact section referencing: the three stages (placement, layering, integration); FATF
and its 40 Recommendations plus published red-flag indicators; BSA/CTR reporting
thresholds and SAR filing (FinCEN Form 111, with its filing deadline); and SR 11-7 model
risk management as the reason explainability matters. Four or five accurate sentences
beat a page of vague regulatory name-dropping — **and verify every specific before you
publish it**, because a wrong threshold or deadline in a CV project is worse than
omitting it.

## C6. One worked typology walkthrough with a figure

Pick a single detected ring. Show the subgraph visualisation, then walk through it in
prose: how the money moved, why the shape is a scatter-gather, which features fired,
what the agent concluded. One good figure plus two paragraphs of domain explanation is
the most memorable thing in the README and demonstrates technical and domain skill in
the same breath.

## C7. Situate the project in the real vendor landscape

One or two sentences on where this sits relative to production stacks (NICE Actimize,
Verafin, Feedzai, Hawk AI, Unit21, Sardine) and the standard architecture: rules engine →
ML scoring → case management with human review. Naming the landscape accurately shows
you have looked at the industry, not just the dataset.

## C8. Domain-first interview story

Reorder v1's story so domain leads and ML supports:

1. **The real pain** — transaction monitoring drowns compliance teams in false positives;
   the binding constraint is analyst capacity, not detection power.
2. **The domain insight** — laundering is relational and structural. Smurfing, layering
   and pass-through mule behaviour are *shapes*, so I engineered features for the named
   typologies and for the network topology they produce.
3. **The result, honestly framed** — graph features lifted PR-AUC from B to D over a
   strong account-aggregate baseline; I built that baseline specifically so the lift
   couldn't be an artifact of aggregation.
4. **The rigor** — leak-free graph construction on a train-window-only graph, PR-AUC over
   ROC because of a 1-in-N base rate, thresholds set by analyst capacity, and a
   single-institution experiment showing what the model loses when it can only see one
   bank's side.
5. **The second half** — detection is only half the job; every alert still needs a human.
   The agent does L1 triage, cites the model's SHAP evidence, classifies the typology
   against ground-truth pattern labels, and drafts a SAR-structured narrative.
6. **The honesty** — synthetic data, no entity resolution, full inter-bank visibility a
   real bank wouldn't have. Here is what would need to change in production.

---

# §D — Schedule (2 weeks, everything in)

Everything fits because all reading-and-writing work moves off the critical path. Days
2–5 and 9 are long days; that is the cost of the decision to keep full scope.

**Pre-work (evenings, before Day 1 — no code, no data needed).** Curate the AML knowledge
base for RAG: FATF red flags, mule indicators, structuring/smurfing/layering definitions,
and the eight AMLworld typology definitions (~12 pages). Draft the README skeleton and
the §C1 typology table with the feature column left blank. Read enough on SAR narrative
structure to write the §C2 template. **This single move frees a full day in week 2.**

**Day 1 — Setup, data, three go/no-go checks.** Venv, install, download HI-Small. CSV →
parquet with `category` dtypes. EDA on schema and class imbalance. Then:
1. **Time span** — a temporal split needs enough history. If the span is too short for a
   meaningful train window, fall back to HI-Medium or split on timeline proportion.
   Decide now, not on Day 5.
2. **Parse `HI-Small_Patterns.txt`**, confirm typology labels join back to transactions.
3. **Node identity** — are account IDs unique globally or only within a bank? If the
   latter, node ID = `(bank, account)`.

Also: fix the FX table, confirm the payment-format value set, write down the
transaction→account alert aggregation rule, freeze the three-way temporal split to disk.

**Day 2 — Arms A and B.** Transaction features, then account behavioural aggregates **plus
the B2 typology features** (structuring bands, pass-through ratio, dormancy burst,
payment-format groups), all computed train-window-only. LightGBM with `scale_pos_weight`,
early stopping on val. **Arm B is your real baseline.**

**Day 3 — Graph + topology (arm C).** Build `G_train` from the training window only.
Degree, PageRank, reverse-PageRank, approx. betweenness (`k=500`), Louvain community id +
size, fan-in/fan-out counts, in-cycle flag, plus the B1 cross-bank features. Join to
val/test by account lookup; `NaN` + `is_unseen_account` for cold-start.

**Day 4 — node2vec (arm D) + ablation table.** `pecanpy`/`fastnode2vec` on `G_train`,
dim 64. Retrain. Produce the four-arm ablation table — **the headline result.** Add the
B4 time-respecting-cycle flag if the day allows.

**Day 5 — Eval harness, operating point, SHAP.** PR curves for all four arms on one
chart. Bootstrap 95% CIs. Threshold selection on **val**: precision@k for k = 50/100/500,
plus the grounded cost-sensitive threshold with a sensitivity strip. Confusion matrix,
feature importance, alert volume at fixed recall, **pattern-level recall by typology
(B5)**, and the **single-bank realism experiment (B1)**. Compute SHAP for the alert set
(B3). **Freeze the engine, score test once, write `results/engine.json`.**

**Day 6 — RAG index + triage agent core.** Index the pre-built knowledge base in
ChromaDB (fast, because curation is already done). Agent takes a top-N alert account,
pulls its subgraph and its **SHAP top features**, retrieves typologies, and emits
structured output: `risk_rationale`, `pattern_classification` (8 typologies + `none`),
`confidence`, `disposition`, `cited_transaction_ids`. `temperature=0`.

**Day 7 — SAR-structured narrative + one bounded agentic step.** Narrative following the
§C2 template, citing specific transactions and the model's attribution. One genuine
agentic decision: the agent chooses whether to pull additional context (e.g. a flagged
counterparty's subgraph) before finalising, with a hard iteration cap.

**Day 8 — Agent evals.** Alert set = top-N with TP/FP mix reported. Measure: triage
precision/recall **vs escalate-everything**; typology accuracy + macro-F1 + confusion
matrix **against `Patterns.txt`**; programmatic hallucination rate; RAG-ablation delta;
cost and latency per alert. Write `results/agent.json`.

**Day 9 — Integration + demo + deploy.** Wire engine → agent end to end. Streamlit demo.
**Deploy it to Streamlit Community Cloud** so the CV carries a clickable link — most
reviewers will never clone a repo. Record a 90-second walkthrough.

**Day 10 — Writeup.** README generated from `results/*.json` so no number is hand-typed.
Complete the §C layer: typology table, business-case opening with a cited source,
regulatory grounding, worked typology walkthrough with figure, vendor-landscape framing,
monitoring and limitations sections. Clean commit history.

**Cut first if a day slips:** (1) B4 temporal motifs, (2) the agentic step on Day 7 —
reduce to single-shot triage. **Never cut:** Day 8 agent evals, or the §C layer on
Day 10. The evals and the framing are the entire differentiator.

---

# §E — Metrics to report

| Metric | Why it matters |
|---|---|
| Class imbalance ratio + train/val/test time spans | Shows you understand the problem shape |
| **PR-AUC across arms A→D, with 95% CI** | **Headline. Lift measured vs arm B, not arm A.** |
| Cold-start rate on test | Proves you understood the leakage/generalisation trade |
| **Precision@k (k = 50/100/500)** | Tuning against analyst capacity — how banks actually do it |
| Cost-sensitive threshold + sensitivity to the cost ratio | You own your assumptions rather than hiding them |
| **Pattern-level recall, by typology** | Evaluates at the unit compliance cares about |
| **Single-bank vs full-graph performance drop** | You know full visibility is a synthetic-data luxury |
| Top features by importance + SHAP | Visual proof of the thesis, and the explainability requirement |
| **Alert volume at fixed recall (A→D→post-agent)** | Real, computable business value |
| Agent triage precision/recall **vs escalate-everything** | Evaluated against a control |
| **Typology accuracy + macro-F1 vs `Patterns.txt`** | Ground-truth agent eval — rare and very strong |
| Hallucination rate (programmatic citation check) | Production rigor on the LLM half |
| RAG-ablation delta | Retrieval earns its place, or you report that it didn't |
| Cost + latency per alert | Unit economics |

---

# §F — Repo and reproducibility

```
data/          # gitignored: raw CSV + parquet cache
src/           # features_txn.py, features_account.py, features_typology.py,
               # features_graph.py, embeddings.py, train.py, evaluate.py,
               # explain.py (SHAP), agent/, kb/
results/       # engine.json, agent.json, figures/   ← README reads from here
notebooks/     # 01_eda.ipynb only; all real logic lives in src/
app.py         # Streamlit demo
Makefile       # make data | features | train | eval | agent-eval | all
requirements.txt  # pinned
README.md
```

Seed everything (`numpy`, `lightgbm`, node2vec walks, `k`-sample betweenness). `make all`
reproduces every number in the README from raw data.

---

# §G — CV framing

**Bullet (fill in real numbers, never fabricate):**

> Built an end-to-end AML transaction-monitoring and triage system on 5M+ synthetic
> transactions: engineered leak-free graph features (node2vec, PageRank, Louvain) on a
> train-window-only graph alongside typology-driven features (structuring bands,
> pass-through ratio, cross-institution layering), lifting PR-AUC from **[B]** to **[D]**
> over a strong account-aggregate baseline and cutting alert volume **[X]%** at fixed 80%
> recall; then built an LLM triage agent that cites SHAP evidence, classifies laundering
> typology at **[Y]%** accuracy against ground-truth pattern labels, and drafts
> SAR-structured case narratives — evaluated for decision accuracy against an
> escalate-everything control, hallucination rate, and cost per alert.

**Short version:**

> AML graph-ML + LLM triage system: leak-free graph and typology features lifted PR-AUC
> **[B]→[D]** over a strong baseline; LLM agent classifies typology at **[Y]%** vs ground
> truth and auto-dismisses **[X]%** of false positives with zero missed cases.

**Two things that matter more than the wording:** the **deployed Streamlit link** on the
CV, and the **Limitations section** in the README.

---

# §H — The four things that make or break it

1. **Fit graph features on the training window only.** Not just a time split — the graph
   itself must not see the future. Everything else is downstream of this.
2. **Build the strong baseline (arm B), typology features included.** An honest 0.08
   PR-AUC lift over a real baseline is worth more than a fake 0.30 over a strawman.
3. **Ship the eval harness with controls on both halves.** A metric without a control is
   not a result.
4. **Ship the §C domain layer.** It is the difference between "graph ML on a fraud
   dataset" and "someone who understands financial crime compliance." It is also the
   cheapest work in the plan — do not let it be the thing that gets dropped on Day 10.

---

# §I — Verification

- `make all` runs clean from raw data on a fresh venv and regenerates every figure and
  both `results/*.json`.
- **Leakage check (write it as a test):** assert no account feature on a val/test row was
  computed from a transaction with timestamp ≥ train-window end.
- **Split check:** assert `max(train.ts) < min(val.ts)` and `max(val.ts) < min(test.ts)`.
- **Label-shuffle sanity check:** shuffle labels, retrain arm D — PR-AUC must collapse to
  the base rate. If it doesn't, there is a leak.
- **Agent check:** the citation validator runs over every alert in the eval set and
  reports a hallucination rate; hand-read three narratives end to end.
- **Fact check before publishing:** every regulatory specific in the README (thresholds,
  form numbers, deadlines, the false-positive statistic) verified against a source you
  have actually opened.
- Test-set metrics computed exactly once, in `evaluate.py`, after the engine is frozen.
