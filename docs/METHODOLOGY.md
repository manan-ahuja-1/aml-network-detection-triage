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
