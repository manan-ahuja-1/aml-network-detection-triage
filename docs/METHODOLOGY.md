# Methodology

Why this project is built the way it is. `docs/NOTES.md` records *what happened* in
sequence; this document records *why*, for a reader evaluating the repo.

The short version: in a problem with a 1-in-1,123 base rate, almost every way of
getting an impressive number is a way of fooling yourself. Most of the engineering
here exists to make that harder.

---

## 1. The evaluation stance

**Reported metric is PR-AUC, not ROC-AUC.** ROC-AUC plots true-positive rate against
false-positive rate. With 1,122 negatives for every positive, a model can raise
thousands of false alarms and barely move the false-positive *rate*, so ROC-AUC stays
flatteringly high while the alert queue is useless. Precision-recall asks the question
a compliance team actually asks: of the alerts we raised, how many were real?

**Operating points are set by analyst capacity, not by a default.** A threshold of 0.5
is an arbitrary inheritance from balanced classification. Real teams can review a
fixed number of alerts per day, so we report precision@k for k ∈ {50, 100, 500}
(`config.PRECISION_AT_K`) alongside a cost-sensitive threshold whose assumed cost
ratio is stated and sensitivity-tested rather than asserted.

**Class reweighting is deliberately not used, against standard advice.** The obvious
move at a 1-in-1,123 base rate is `scale_pos_weight` or `is_unbalance`. Measured on
arm A, it is actively harmful:

| setting | best iteration | PR-AUC |
|---|---|---|
| `scale_pos_weight=1325` (true ratio) | 1 | 0.0119 |
| `is_unbalance=True` | 1 | 0.0119 |
| `scale_pos_weight=36` (sqrt of ratio) | 1 | 0.0464 |
| **none** | **62** | **0.0495** |

Reweighting exists to fix *calibration* — to stop the loss ignoring a rare class when
you need trustworthy probabilities. PR-AUC consumes only the *order* of scores, never
their values. Multiplying positive gradients by 1,325 distorts every split so severely
that the model destroys its own ranking chasing calibration the metric never reads.

`min_data_in_leaf=300` is load-bearing for the same reason. With 2,296 positives in 3M
rows, a small leaf memorises a handful of them, validation average-precision spikes
spuriously, and early stopping fires on the spike: arm B halts at iteration 5 with
`min_data_in_leaf=20`, and runs 1,961 iterations at 300.

**The test set is scored once.** Thresholds, hyperparameters and prompts are all
tuned on validation. A test set consulted repeatedly is a training set with extra
steps.

---

## 2. Leakage: the threat model

This is the part most portfolio projects get wrong, so it is worth being explicit
about the two distinct failures.

**The obvious one — temporal leakage.** A random split trains on transactions that
occur *after* the ones it scores. The split here is temporal: train on the earliest
window, validate on the middle, test on the latest (`src/make_splits.py`). Boundaries
are computed once and persisted to `data/processed/split_boundaries.json`, so no
later stage can recompute and silently shift them.

**The subtle one — graph leakage.** This is the failure a time-based split does *not*
fix. If the transaction graph is built over the whole dataset and centrality, Louvain
communities and node2vec embeddings are computed on it, then every training row
carries features derived from edges that had not happened yet at prediction time. The
split looks correct; the features leak anyway. The model appears to know a mule
account's future counterparties.

The rule this project follows: **the graph is fitted on the training window only.**
Topology features and embeddings are computed on `G_train`; validation and test rows
*look up* their accounts' features from that graph. Accounts never seen in training
get `NaN` plus an explicit `is_unseen_account` flag — never zero, which would
fabricate a "this account has no connections" signal that is indistinguishable from a
real one.

That design creates a genuine cold-start problem, which is correct: it is what
production looks like. Measured on this data it is mild — 98.9% of accounts appear in
the training window and 99.8% of val/test rows have both counterparties known — but
the mechanism is in place regardless of how favourable the numbers happened to be.

`tests/test_splits.py` asserts the ordering properties directly, so a future change
that breaks them fails loudly instead of producing plausible-looking nonsense.

---

## 3. The ablation: why the baseline is arm B

The project's thesis is that laundering is relational and graph features should beat
transaction-level features. The obvious way to demonstrate that is to compare
transaction-only against transaction-plus-graph — and it is misleading, because most
of the apparent lift comes from *any* per-account aggregation, not from graph topology
specifically.

Five arms (`config.ARMS`):

| Arm | Contents |
|---|---|
| **R** | Rules baseline: flag every ACH. No model. |
| **A** | Raw transaction attributes only |
| **B** | A + account aggregates + typology features — **no graph topology** |
| **C** | B + graph topology (degree, PageRank, reverse-PageRank, betweenness, Louvain) |
| **D** | C + node2vec embeddings |

**The headline lift is C/D measured against B, not against A.** Arm B is the real
control: it already contains every non-graph account-level signal, so whatever C and D
add is attributable to topology. An honest small lift over a strong baseline is worth
more than a large one over a strawman, and building the strong baseline deliberately
is the point.

**Arm D was tested and rejected on evidence.** node2vec embeddings were built over the
same training-window graph and swept across three dimensionalities with three random
seeds each. No configuration beat arm C:

| dim | columns added | mean PR-AUC | std across seeds |
|---|---|---|---|
| 16 | 32 | 0.1598 | 0.0028 |
| 32 | 64 | 0.1660 | 0.0155 |
| 64 | 128 | 0.1349 | 0.0367 |

Arm C scores 0.1938. The best single arm D run anywhere in the sweep was 0.1834.

The graph averages 2.80 degree across 28,326 disconnected components, so most accounts
appear in too few random walks for Word2Vec to learn from — median embedding norm was
0.0803 against a max of 15.98, meaning most vectors barely moved from initialisation.
Learned embeddings need a denser graph than this to beat explicit topology.

Note also that variance scales with dimensionality (std 0.0028 → 0.0155 → 0.0367). Arm
C's entire lift over arm B was +0.0123, which is smaller than the seed-to-seed spread
of arm D at dim=64 (0.0656). **A lift is only meaningful relative to the noise of the
procedure that produced it** — which is why every arm D figure here is a mean over
three seeds rather than a single run.

**Arm R exists because of a finding, not a convention.** EDA showed that "flag every
ACH" alone achieves 84.7% recall at 0.64% precision. Real AML stacks begin with a
rules engine and the ML layer has to justify replacing it, so a rules baseline is both
domain-authentic and the honest floor every model arm must clear.

---

## 4. Data decisions

### 4.1 Truncation at 2022-09-11

The raw file spans 2022-09-01 to 2022-09-18, but the two halves are not the same
data. Legitimate transaction generation stops after 09-10 while injected laundering
patterns continue to 09-18:

| Period | Rows | Laundering | Rate |
|---|---|---|---|
| Sept 1–10 | 5,077,237 | 4,522 | 0.089% |
| Sept 11–18 | 1,108 | 655 | **59.1%** |

A test split covering that tail would be scored on a region where laundering is the
majority class. PR-AUC would be inflated by an artifact of the simulator rather than
by anything the model learned, and it is the first thing a reviewer plotting label
rate over time would notice.

Truncating costs 0.02% of rows and 12.7% of positives, and reduces the train→test
base-rate shift from 2.3× to 1.5×. Enforced by
`tests/test_splits.py::test_tail_is_excluded`.

### 4.2 Node identity is a composite key

Account numbers are **not unique**. Eight account numbers exist at two different banks
owned by *different entities* — `80A7FD400` is a Partnership at Australia Bank #44 and
a Corporation at Australia Bank #47. Keyed on account number alone, those collapse
into a single graph node fusing two unrelated transaction histories.

`(Bank ID, Account Number)` is provably unique: 518,581 composite keys for 518,581
rows. Node identity is `bank_id:account_number`, constructed once in
`make_data.py::node_id` so no downstream module has to remember.

### 4.3 Two typing traps

Both would have failed silently rather than loudly.

**Duplicate column names.** The CSV header names two columns `Account` (sender and
receiver). pandas silently mangles the second to `Account.1`. Columns are renamed
*positionally* instead of trusting that behaviour.

**Leading zeros in bank IDs.** Bank IDs look like `010` and `021174`. Parsed as
integers they become `10` and `21174`, and every join against the accounts table
returns nothing — with no error. All identifier columns are read as `string`.

**Bank IDs are formatted differently in the two source files.** `HI-Small_Trans.csv`
zero-pads them (`010`, `03208`); `HI-Small_accounts.csv` does not (`10`, `3208`). A
composite key built without normalising matched *nothing* across the two files, so
every entity feature arrived as NaN — present in the model, contributing nothing, and
silent about it. Account numbers matched 100%, which is what made it easy to miss.
`make_data.py::node_id` strips leading zeros; verified that no two bank IDs collide
when normalised and that the canonical key stays unique across all 518,581 accounts.

This was the second silent-NaN join failure in the project. Both were caught only
because the module printed a null rate rather than trusting the join. That diagnostic
habit is now standard here.

### 4.4 Currency normalisation

Amounts appear in 15 currencies, and 1.42% of transactions are cross-currency. Without
normalisation an amount feature compares 500 Yen with 500 US Dollars as equal, so both
sides are converted to USD before any amount feature is computed.

**The rates in `config.FX_TO_USD` are approximate, static, mid-2022 values — not a
historical series and not independently sourced.** This is a stated simplification:
the data is synthetic and has no real FX series behind it, and the features that
depend on it (order-of-magnitude of an amount, structuring bands) tolerate a few
percent of error. It is recorded here rather than left for a reader to discover.

---

## 5. A known artifact: ACH

The simulator injects laundering almost exclusively as ACH.

| | Share of rows | Share of laundering |
|---|---|---|
| ACH | 11.8% | **84.7%** |
| Wire + Reinvestment | 12.9% | **0 of 652,911 rows** |

3,208 of the 3,209 labelled pattern transactions are ACH.

Real laundering is not 85% ACH, and a model that leans on `payment_format` will
achieve strong metrics for a reason that does not transfer. This is handled three
ways rather than ignored: arm R quantifies how far the artifact alone gets you; a
variant without `payment_format` measures how much performance survives without it;
and it is stated prominently in the limitations.

---

## 6. Reproducibility

- **One seed** (`config.RANDOM_SEED`) passed explicitly to every stochastic component
  — LightGBM bagging, sampled betweenness, node2vec walks, bootstrap resampling.
- **Pinned lock file.** `requirements.in` records intent with reasons;
  `requirements.txt` pins all 159 packages including transitive dependencies.
- **Python 3.13 is a hard requirement**, asserted by `make check`. `gensim` publishes
  no cp314 wheel and node2vec trains through gensim's Word2Vec, so 3.14 silently
  removes the ability to produce arm D.
- **`make all` regenerates every reported number from raw data.** Pipeline stubs exit
  non-zero rather than succeeding silently, so the target cannot report success
  having produced nothing.
- **Every README number will be read from `results/*.json`**, not typed by hand.

---

## 7. Limitations

Stated here because they are the first questions a reviewer should ask.

- **The data is synthetic.** Results do not transfer to production. The contribution
  is the method and the evaluation rigour.
- **The ACH artifact** described in §5.
- **Full inter-bank visibility is unrealistic.** A real institution sees only its own
  customers' side of a transaction, never the complete graph this dataset provides.
  Planned mitigation: re-run arm C restricted to a single bank's visible subgraph and
  report the performance drop.
- **FX rates are approximate** (§4.4).
- **Residual base-rate drift** remains after truncation: train 0.0754%, val 0.1067%,
  test 0.1125%. Laundering patterns ramp up over the first days of the simulation, so
  the earliest window is genuinely quieter.

---

## 7. Scoring the test split exactly once

Every number reported through Days 1–4 came from validation, and validation was consulted
repeatedly: to set `min_data_in_leaf`, to reject `scale_pos_weight`, to drop
`reverse_pagerank`, to choose among four feature arms, and — continuously — by early
stopping, which selected iteration 2064 precisely because it maximised validation average
precision. Each of those consultations leaked a little information about validation into
the model. That is what a validation set is for, but it means validation performance is
optimistic by an unknown amount.

So the test split was scored once, after the engine was frozen, and the result stands.
Consulting test and then changing anything would convert it into a second validation set.
Two mechanisms enforce this rather than merely declaring it: the booster is persisted to
`models/engine_armC.txt` and reloaded by every later stage, and a SHA-256 digest of the
test scores is recorded in `results/engine.json`, so a future run producing different
scores is detectable.

Arm C's retraining reproduced its Day 3 validation result exactly (iteration 2064,
PR-AUC 0.1938), confirming that the ablation table and the frozen engine describe the same
model.

## 8. Confidence intervals, and why they are stratified

The test split holds 1,143 positives. A PR-AUC quoted to four decimals off 1,143 events
implies a precision the data cannot support, so every headline figure carries a percentile
bootstrap interval over 2,000 resamples.

Resampling is **stratified by class**, holding the number of positives fixed. PR-AUC moves
with the base rate; at a 1-in-900 rate an unstratified resample's positive count varies by
several percent, and the resulting interval would be measuring class-balance jitter as much
as model uncertainty — wider than the truth, and wide for the wrong reason.

The intervals earned their place immediately: validation [0.1700, 0.2190] and test
[0.0788, 0.1137] **do not overlap**, which converts "test looks worse" from an impression
into a finding that has to be explained.

`average_precision_score` costs ~260 ms on a 1M-row split, making 2,000 resamples a
nine-minute operation per split. Sorting once outside the loop and expressing each
resample as a per-row multiplicity vector reduces it to ~29 ms, and the two agree to
5.6e-17 (asserted in `tests/test_engine.py`). The point estimate still comes from sklearn;
only the interval uses the fast path, because the fast path assumes no tied scores.

## 9. Diagnosing the test drop: shape, not just size

Test PR-AUC (0.0949) is half of validation (0.1938). The project's value here is not the
number but the diagnosis, and the diagnosis worked by asking which explanations predict
which *shapes*.

**Cold start was the obvious hypothesis and it was wrong.** Features are fitted on the
training window only, so accounts absent from training get NaN everywhere; test sits
further from training and holds 356,263 distinct accounts against validation's 236,109.
Measured, though, coverage is identical (0.20% of rows with an unseen account in both
splits), and the test/validation ratio stays at 0.49–0.51 across every coverage subset. A
constant ratio means coverage explains none of the gap. The extra test accounts had been
seen in training; they were simply idle during validation.

**Validation optimism and temporal decay predict different shapes.** Optimism is a step:
validation uniformly inflated, test honest, no variation inside either. Decay is a slope.
Slicing the post-training period into six equal windows shows PR-AUC falling monotonically
from 0.2523 to 0.1465 *within validation*, before test begins — already most of the way to
test's 0.0949. It is a slope. The val/test boundary is incidental; the mechanism is feature
staleness.

This reframes the headline. Test PR-AUC is not "the model is worse than we thought"; it is
the model at two to four days of feature staleness, and PR-AUC roughly halves per ~1.5 days
of distance from the training window. It also converts the generic "we would monitor for
drift" section into a specific, measured requirement: recompute account and graph features
on a rolling window, and retrain on a cadence set by that half-life.

It does **not** change the engine. Arm C was chosen on validation, which is correct, and
re-choosing after seeing test would be the error this whole section exists to avoid.

## 10. Evaluating at the unit compliance cares about

Transaction-level metrics understate operational usefulness, because a laundering ring
spans many transactions and an investigator needs one thread to pull. Reporting at three
units makes that visible:

| unit | val | test |
|---|---|---|
| transaction PR-AUC | 0.1938 | 0.0949 |
| account precision@100 | 88% | 50% |
| pattern recall @ threshold | 92.3% | 83.6% |

Transaction-level performance halves between splits; **pattern-level recall falls only from
92.3% to 83.6%**. The operational degradation is far milder than the headline metric
implies, and that gap is only observable because `HI-Small_Patterns.txt` was parsed on
Day 1 into real ground truth.

Two caveats travel with every pattern metric. Only 2,554 of the 4,522 surviving laundering
transactions belong to a named pattern, so pattern recall describes the labelled 56.5%.
And the weakest typology, CYCLE at 68.0%, is weak for a known reason: `in_cycle` is
computed on the *static* training-window graph, so a cycle formed during the test window
is invisible to it — the measured cost of deferring time-respecting motif detection.

## 11. Grounding half of an assumption, and saying which half

The cost-optimal threshold depends on the ratio of a missed case to a false-alert review.
Only one side of that ratio can be sourced.

The **denominator** can: BPI's 2018 *Getting to Effectiveness*, a survey of 19 US banks,
reports $2.4bn of BSA/AML spend against 16 million alerts reviewed — roughly $150 per
alert, and an upper bound at that, since the $2.4bn also covers KYC, CTR filing, systems
and model validation. The same survey yields the false-positive figure from primary data:
640,000 SARs from 16 million alerts is a 4% conversion rate, so 96% of alerts produced no
SAR.

The **numerator** cannot. No published figure gives the expected cost of one missed
laundering transaction; penalties are levied for programme failures rather than per
undetected transaction, and the counterfactual harm is unobservable.

So the operating point is reported as a sensitivity strip across seven cost ratios rather
than a single tuned threshold. Presenting one number would disguise a judgement call as a
measurement.

## 12. Retrieval grounded in sources that were actually read

The triage agent's knowledge base is built from primary regulatory text — FFIEC
Appendices F, G and L; FinCEN's SAR Narrative Guidance, FIN-2014-A005 and FIN-2020-A003;
31 CFR § 1020.320 — with every document registered in `kb/sources.json` against a URL and
retrieval date. `load_corpus()` raises on an unregistered `source_id`, so the constraint is
enforced rather than intended. Where the corpus contains our own synthesis — the mapping
from the dataset's eight injected shapes to regulatory typologies and to the specific model
features that fire — it is labelled as ours.

A RAG system that cites unverifiable text is worse than one with no citations, because it
manufactures confidence the reader cannot check.

The one non-obvious engineering constraint is the embedder's 256-token window. Documents
run 400–600 tokens, so indexing them whole would store complete text while embedding only
each document's opening, and retrieval would silently never match the remainder. Nothing
errors. Chunking is measured against the real limit, which required disabling the
tokenizer's own 128-token truncation first — with it left on, every long chunk reports
exactly 128 tokens and an over-length chunk is invisible to the check meant to catch it.

## 13. Why a control is not enough: disaggregating before believing a lift

The agent's escalation precision was compared against escalate-everything from the start,
because "84% accurate" on a 68%-productive queue means nothing. That control is necessary
and it turned out not to be sufficient.

One configuration beat the control by 6.7 points on validation. Broken out by case size,
its escalation precision sat on the base rate inside *every* bucket — 38.5% against 38.7%
among cases of one or two accounts, 83.3% against 85.7% among larger ones. The pooled lift
was Simpson's paradox: small cases have a much lower base rate, the agent closed far more
aggressively among them, and pooled precision rose without a single case being judged
better than chance.

The general principle, which applies to any triage or routing layer: **when a system
chooses both how to judge and which subpopulation to act on, a pooled rate confounds the
two.** A lift over a control is evidence only once it survives disaggregation by whatever
the system is selecting on, plus a significance test on the 2×2 it is actually claiming.

Both are now computed and reported unconditionally in `results/agent.json` —
`discrimination.overall` and the per-size-bucket breakdown, with a one-sided Fisher exact
p-value. On the test split the answer is p=0.51, and the project reports the disposition as
measured-and-not-useful rather than burying it.

## 14. What an ablation is allowed to claim at n=53

The retrieval ablation moved two numbers. Typology any-match rose from 33.3% to 58.3% —
which looks like the result, and rests on twelve labelled cases at p=0.21. Red-flag
indicators named per case rose from 1.06 to 2.74, with the share of notes naming no
indicator at all falling from 35 of 53 to 3 of 53, at p=4.3e-08.

Reporting only the deltas would have put the emphasis on the one number that cannot hold
it. So `ablation_significance` is computed alongside `ablation` and carries the reading
with the number, rather than leaving the reader to discover which of the two is real.

The typology figure is reported as a direction. The indicator figure is reported as the
finding, and it is the honest claim for a RAG layer in this position: retrieval did not
make the model a better ranker and was never going to, but it is what makes the output
cite published regulatory text instead of asserting suspicion in its own voice.

## 15. Scoring a classification task against a partially labelled ground truth

`Patterns.txt` names the typology of every injected ring, which makes case-level typology
a real labelled task rather than a rubric. But only 56.5% of surviving laundering
transactions belong to a named ring. Ten of the 22 productive test cases contain laundering
the simulator never grouped.

Those cases have an *unknown* truth label, not NONE. Scoring them against NONE would
manufacture credit whenever the agent said NONE and blame whenever it named a shape, out of
a gap in the labelling rather than anything the agent did. They are excluded from the
accuracy and counted separately, and the count travels with the metric everywhere it is
printed.

A case can also span several rings, so two readings are reported: any-match (the label is
one of the types present) and dominant-match (the type with the most laundering
transactions). Any-match is the operational reading — an investigator told "this is a
fan-out" is pointed the right way even if the case also contains a stack.

## 16. Borrowing a threshold from a different question

The FX diagnostic needed a rule for when a change to the currency table counts as
changing the engine. The first version used `evaluate.PR_AUC_TOLERANCE` (0.002), on the
reasonable-sounding grounds that `freeze_engine` already uses that constant to decide
whether the engine has changed. It returned MATERIAL — rebuild the engine, re-score test.

That constant answers a different question. 0.002 is a **reproducibility** tolerance: it
checks that re-running identical code on identical data returns an identical number, and
in that setting anything above floating-point noise is a real defect. It carries no
information about whether two models trained on slightly different data are meaningfully
different.

The estimator's own bootstrap standard deviation on validation is 0.0127 and its 95% CI
is [0.1700, 0.2190] — both computed on Day 5, before this question existed. Against that
sampling distribution, a 0.002 threshold declares anything above 0.16 SD material.

Measured against the interval instead, the retrained value sits 0.74 SD from the frozen
one, inside the CI, and is not distinguishable from estimation noise on 1,083 positives.

Two general points, both of which cost nothing to apply and would have cost a great deal
to miss here — the rebuild would have cascaded through the alert set, the case queue and
the paid agent runs downstream of it:

1. **A tolerance is defined by the question it was calibrated for.** Reusing one because
   it lives nearby and has the right units is how a determinism check becomes a
   significance test.
2. **Compare a difference against the sampling distribution of the thing being
   differenced.** The bootstrap already sitting in `engine.json` was the correct
   yardstick and was four days old when it was needed.

The verdict logic is split into `judge()` so a corrected criterion can be re-applied to a
saved run rather than re-derived — which mattered, because the graph rebuild it sits on
top of takes 8,109 seconds in the Louvain pass alone.

## 17. Two questions a robustness check has to keep apart

Correcting an input can affect a shipped model in two unrelated ways, and collapsing them
produces a misleading answer either way:

- **Inference sensitivity.** Score the existing frozen model on features rebuilt under
  the corrected input. This asks how wrong the predictions already made are. Here: val
  PR-AUC 0.1938 → 0.1984, with a Spearman rank correlation of 0.9953 between the two
  score vectors. The shipped artifact ranks essentially identically.
- **Training sensitivity.** Retrain from scratch under the corrected input. This asks
  whether the *published number* is right. Here: 0.1938 → 0.1844.

They point in opposite directions, and only the second bears on whether anything must be
re-run. Reporting one and calling it "the FX result" would have been true and useless.

## 18. Every accuracy needs the number a system doing no work would score

The build plan demanded a control for the agent's disposition (A8), and it got one from
the first run: escalate-everything. Typology accuracy was added later and reported bare —
58.3% any-match, 50.0% dominant-match — until per-case ground truth was written into
`results/agent.json` and made two problems visible at once.

**Any-match is not a fixed-difficulty task.** A case's truth label is the set of injected
rings its member accounts touch, and a large case touches many: one eleven-account case
has all eight typologies in its set, so any prediction is "correct" by construction. The
chance rate is therefore per-case — mean(|types| / 8) = 34.4% across the twelve labelled
test cases — not a uniform 1/8. Against that, 58.3% is a real but modest lift rather than
the strong result it reads as.

**Dominant-match has a fixed 12.5% chance rate and a badly skewed class distribution.**
Eight of twelve labelled cases are GATHER-SCATTER, so always predicting the majority class
scores 66.7% against the agent's 50.0%. The agent does not beat the trivial baseline.

Both are now computed in `typology_scores.baselines` and printed beside every accuracy.

The general rule is the one this project had already learned the expensive way one section
earlier, applied to a metric that was not covered by it: **a rate is uninterpretable
without the rate a system doing no work would achieve, and "no work" has to be defined per
metric.** For a triage disposition that is escalate-everything. For a skewed multi-class
label it is the majority class, not uniform chance. For a set-membership metric it is the
expected set size, which varies per item. Choosing the wrong "no work" baseline is the
same error as omitting one.

## 19. The ablation's headline lift, tested properly

Arm B (0.1815) and arm C (0.1938) were compared for eight days by their marginal 95% CIs,
which overlap: [0.1588, 0.2063] against [0.1700, 0.2190]. Overlapping marginal intervals
are the *conservative* test and were the only one available, because nothing until B1 kept
per-row validation scores for more than one arm at a time.

B1 needed a paired bootstrap for its own reasons, and persisting the score vectors made the
arm B / arm C comparison free. Both arms score the same 1,015,300 validation rows, so the
correct question is the distribution of the difference:

> **+0.0123 PR-AUC, 95% CI [−0.0025, +0.0268].** 95.3% of paired resamples favour arm C;
> the two-sided interval just includes zero.

The pairing tightens the estimate a great deal relative to comparing the marginals — and it
still does not clear the bar. **The graph lift is at the edge of conventional significance,
not established.** On 1,083 validation positives this dataset cannot separate a +0.012
PR-AUC difference from zero at 95%.

What follows from that, and what does not:

- The README reports the interval rather than the point estimate. "Graph topology lifts
  PR-AUC from 0.1815 to 0.1938" is replaced everywhere by the delta and its CI.
- **Arm D's rejection is unaffected.** It was rejected for failing to beat arm C across
  three seeds and two dimensions, which does not depend on how arm C compares to arm B.
- The engine is not retrained or re-selected. Arm C remains the best validation arm, and
  selecting on a point estimate is the right thing to do when a choice must be made; the
  correction is to how the result is *reported*, not to which model was chosen.
- The design decision this vindicates is building arm B to be hard to beat. A weaker
  baseline would have produced a large, comfortable, and much less honest lift.

The general point: **an interval is not decoration on a point estimate, it is the claim.**
A difference worth reporting to four decimals off 1,083 events needs the interval printed
beside it, and if the interval spans zero that has to be said in the same sentence rather
than left in an appendix.

## 20. Choosing an experimental subject on one criterion selects an outlier

B1 called for re-running the engine on a single institution's visible subgraph. The subject
was picked on one criterion — the bank with the most validation positives, because anything
smaller could not be bootstrapped — and that produced bank 070: 15 accounts carrying 452,751
transactions, no internal transfers at all, a clearing entity rather than a bank. The engine
scored below chance on its slice, and the resulting table was clean, real and meaningless.

The criterion that made it the only viable subject is the same criterion that made it
unrepresentative. In a dataset of 30,528 banks with a median of 4 accounts, the only entity
with enough events to measure is by construction the one that is not like the others.
**Sample-size sufficiency and representativeness pull against each other, and satisfying the
first without checking the second is how a study ends up measuring its own selection rule.**

Two minutes describing the candidate — accounts, transactions per account, internal share —
before running anything would have caught it. That description now lives in
`single_bank.DATASET_STRUCTURE`, so the evidence for abandoning the design travels with the
results rather than only in a commit message.

The generalisation is a pre-flight, not a rule about banks: **before committing to a long
run, describe the thing you selected and check it looks like the population you mean to
generalise to.** It costs minutes against hours, and the failure it catches is silent —
nothing errors, and the output is a table you would otherwise have believed.

## 21. Two measurements from one training run are one observation

The visibility experiment produced a shape — PR-AUC worst in the middle of the range — and
I argued it was corroborated by an independent measurement: early stopping peaked at 1,961
rounds with no graph and 2,064 with the full one, but collapsed to 381 and 196 at 50% and
25% visibility. Two quantities, same story, apparently converging evidence.

They are not independent. Best-iteration and final PR-AUC come from **the same training
run on the same sampled graph**. If that particular draw of edges produced misleading
features, both numbers move together by construction. Reading them as two witnesses is
double-counting a single observation, and it made a one-draw result feel like a replicated
one.

The replicate settled it. At 25% visibility a second edge draw gave PR-AUC 0.1904 against
0.1421, and best-iteration 2,066 against 196 — both numbers flipped together, exactly as
the objection predicts.

**The general rule: independence is a property of the sampling, not of the metric.** Two
statistics computed from one fitted model are one draw from the process being studied, no
matter how different they look or how mechanistically linked the story connecting them is.
Independent evidence requires an independent draw — here, a different subsample.

## 22. One draw per condition cannot separate a condition effect from a draw effect

The curve varied graph visibility across four levels with one subsample at each, then
attributed the differences to visibility. That design cannot do it: each point confounds
"this visibility level" with "this particular set of edges", and nothing in the data
separates them.

How badly it confounds them is now measured. Holding visibility fixed at 25% and varying
only the seed moves PR-AUC across roughly 0.048 — and at 50% the three draws span 0.1210,
0.1654 and 0.1691, a comparable spread. **The within-condition variance is several times
the arm C over arm B lift the project reports as its headline.** No between-condition
comparison at n=1 per condition can survive that.

The fix is not a better test on the same data; it is more draws per condition. Three is
enough to establish that the variance dominates and therefore that the original question
is unanswerable at this scale — which is what the README now says — and nowhere near
enough to estimate the visibility effect itself.

The cheap version of this check, which would have saved several hours: **before believing
a difference between conditions, vary the nuisance factor with the condition held fixed.**
One extra run at one condition is enough to find out whether the noise floor is above the
effect you are about to write up.
