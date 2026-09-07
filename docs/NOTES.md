# Build log

Running record of decisions, measured numbers, and dead ends. Written as we go,
because the README (Day 10), the limitations section, and the interview story all
need this detail and it is not reconstructable from memory afterwards.

---

## Phase 0 — Environment and scaffold

**Goal:** a verified environment and repo skeleton, so no setup problem can surface
later disguised as a modelling problem.

### Machine

Apple M4, 10 cores, 16 GB RAM, ~99 GB free. Comfortable for HI-Small.

### Finding 1 — Python 3.14 would have broken the project on Day 4

The machine's default `python3` is **3.14.6**. `gensim` publishes CPython wheels only
up to **cp313** — there is no cp314 build. Both node2vec implementations
(`pecanpy`, `fastnode2vec`) train their embeddings through gensim's Word2Vec.

So on 3.14, **arm D — the headline result — cannot run at all**, and we would have
discovered this on Day 4 after three days of work.

**Decision:** venv pinned to **Python 3.13.12**, which was already installed at
`/Library/Frameworks/Python.framework/Versions/3.13/`. The venv must be created with
that interpreter *by absolute path*; a bare `python3 -m venv` silently reintroduces
the problem.

Verified compatible on 3.13 by inspecting actual PyPI wheel tags rather than assuming:
lightgbm (`py3-none-macosx`), chromadb (`cp39-abi3`), shap (`cp312-abi3`, and abi3
wheels are forward-compatible), numba (cp313 present).

### Finding 2 — LightGBM could not load: missing OpenMP runtime

`import lightgbm` failed with
`OSError: Library not loaded: @rpath/libomp.dylib`.

LightGBM's macOS wheel is built with OpenMP for multithreading, but Apple's clang
ships no OpenMP runtime, and the only search paths baked into the wheel are
Homebrew's (`/opt/homebrew/opt/libomp/lib`) and MacPorts' (`/opt/local/lib/libomp`).
Neither package manager is installed here.

Tried and rejected: importing `sklearn` first, hoping its vendored libomp would
already be in the process image. It does not work — dyld resolves `@rpath` against
paths baked into the binary at load time, not against what happens to be loaded.

**Fix:** `scripts/fix_macos_libomp.py` copies scikit-learn's vendored `libomp.dylib`
next to `lib_lightgbm.dylib` and adds `@loader_path` to LightGBM's rpath list via
`install_name_tool`. Copying (rather than pointing at sklearn's directory) keeps
LightGBM self-contained, so a future scikit-learn upgrade cannot break training.

Idempotent and re-runnable via `make fix-libomp`. It patches a file inside
site-packages, so it must be re-run after any `pip install --force-reinstall
lightgbm` or venv rebuild.

Verified afterwards: trains multithreaded, and `predict(pred_contrib=True)` returns
`(n_rows, n_features + 1)` — the native TreeSHAP path that B3 depends on.

### Finding 3 — pecanpy is unusable on NumPy 2

`import pecanpy` failed with
`AttributeError: module 'numpy' has no attribute 'bool8'`.

Traced to `nptyping 2.0.1` (`nptyping/typing_.py:66`), a transitive dependency
pecanpy uses only for type annotations. `np.bool8` was removed in NumPy 2.0. Since
pandas 3.0 requires NumPy 2.x, downgrading NumPy is not an option.

**Decision:** switched to **`fastnode2vec`**, which has no `nptyping` dependency and
works on NumPy 2 unmodified. Same algorithm (p/q-biased random walks + gensim
Word2Vec). Avoids a second vendored patch in a repo other people will clone.

`pecanpy`, `nptyping` and `numba-progress` uninstalled so they cannot leak into the
frozen lock file and break a fresh install.

API note for Day 4: the parameter is `window`, not `context`. Signature is
`Node2Vec(graph, dim, walk_length, window, p=1.0, q=1.0, workers=1, seed=None)`.

### Finding 4 — Kaggle's auth flow has changed

The `kaggle` 2.2.4 CLI prints its own instructions on import, which corrected our
plan. **There is no longer a `kaggle.json` download**, and the widely-documented
`KAGGLE_USERNAME` + `KAGGLE_KEY` pair is out of date. Current options:

1. `kaggle auth login` — OAuth browser flow, no token to manage. Simplest.
2. `KAGGLE_API_TOKEN` environment variable (single token from the settings UI).
3. Token saved to `~/.kaggle/access_token`.

`.env.example` documents options 1 and 2.

### Decisions frozen in `src/config.py`

- `RANDOM_SEED = 42`, passed explicitly to every stochastic component.
- Temporal split fractions 60/20/20; **boundaries set on Day 1** from the measured
  time span, then persisted to `split_boundaries.json` so they never drift.
- `FX_TO_USD` left deliberately empty; `require_fx_rates()` raises a named error if
  anything tries to compute an amount feature before Day 1 populates it. A missing
  rate would otherwise produce NaN, which LightGBM silently accepts.
- Alert rule (A6): model scores **transactions**; an account's alert score is the
  **max** over its transactions. Max not mean, because laundering is a minority of
  even a mule account's activity and averaging dilutes the signal.
- node2vec `q = 2.0` (> 1), biasing walks to stay local so embeddings encode
  **structural role** rather than community membership — a mule account is defined by
  its shape of connectivity.
- `BETWEENNESS_K = 500`: exact betweenness is O(V·E) and will not finish at this
  scale; networkx's k-pivot sampled approximation is used instead.

### Environment status

`make check` passes: Python 3.13.12, 19 dependencies importing, networkx Louvain
available (so `python-louvain` is not needed), LightGBM `pred_contrib` confirmed.
`requirements.txt` locked at 159 packages via `pip freeze`.

**Open / next:** Kaggle auth, then Day 1 — download, parquet conversion, and the
three go/no-go checks (time span, `Patterns.txt` format, account-ID uniqueness).

---

## Day 1 — Data, and three go/no-go checks

`HI-Small_Trans.csv` (454 MB, **5,078,345** transactions), `HI-Small_Patterns.txt`,
and `HI-Small_accounts.csv`. Downloaded selectively: the full dataset is ~40 GB
(the Large variants are 17 GB each), the three HI-Small files are ~486 MB.

Kaggle auth: `kaggle auth login` OAuth flow worked; credentials cached to
`~/.kaggle/credentials.json` (mode 600). No secret in the repo.

### Check 1 — time span: PASSED, but with a trap

Raw span is 2022-09-01 to 2022-09-18 (17.7 days). **The data is not uniform across
it.** Legitimate transaction generation stops after 2022-09-10, while injected
laundering patterns continue to 09-18:

| period | rows | laundering | rate |
|---|---|---|---|
| Sept 1–10 | 5,077,237 | 4,522 | 0.089% |
| **Sept 11–18** | **1,108** | **655** | **59.1%** |

A test split covering that tail would be scored on a region where the majority class
is laundering. PR-AUC would have been inflated by a pure artifact of the simulator,
and it is exactly what a reviewer plotting label rate over time would find first.

**Decision: truncate at 2022-09-11 (exclusive).** Costs 0.02% of rows and 12.7% of
positives. Reduces the train→test base-rate shift from 2.3x to 1.5x.
`tests/test_splits.py::test_tail_is_excluded` enforces it.

### Check 2 — `Patterns.txt` format: PASSED, fully

370 BEGIN/END blocks, perfectly balanced, 3,209 transaction rows, all exactly 11
fields, zero unexpected lines. **Joins back to the transactions table at 100.00%**
on `(timestamp, from_id, to_id, amount_paid)`.

All eight typologies present and near-balanced — CYCLE 54, GATHER-SCATTER 51,
BIPARTITE 49, FAN-OUT 48, SCATTER-GATHER 44, STACK 43, RANDOM 41, FAN-IN 40.
That balance is what makes macro-F1 over 8 classes meaningful. **A7 and B5 are fully
unlocked.**

### Check 3 — account identity: FAILED as stated, fixed

Account Number alone is **not unique**. Eight account numbers exist at two different
banks owned by *different entities* (e.g. `80A7FD400` is a Partnership at Australia
Bank #44 and a Corporation at Australia Bank #47). Keyed on account number alone,
those become one graph node fusing two unrelated histories.

`(Bank ID, Account Number)` is provably unique: 518,581 composite keys for 518,581
rows. **Node identity is `bank_id:account_number`**, built once in
`make_data.py::node_id`.

### Cold-start: NOT a problem (earlier concern was wrong)

The 18-day span made cold-start look like a serious risk for arms C/D. Measured, it
is not: 512,713 of 518,581 accounts (98.9%) appear in the training window.

- val: 99.8% of rows have both counterparties known; 97.2% for laundering rows
- test: 99.8% both known; 96.1% for laundering rows

Arms C and D are safe.

### Two data traps neutralised in `make_data.py`

1. **Duplicate column names.** The CSV header names two columns `Account`. pandas
   silently mangles the second to `Account.1`. Columns are renamed *positionally*.
2. **Leading zeros in bank IDs** (`010`, `021174`). Read as int they become 10 and
   21174 and every join against the accounts table silently returns nothing. All
   identifier columns are read as `string`.

### `HI-Small_accounts.csv` — not in the original plan

518,581 accounts → **166,207 entities**, with an entity type in the name.

- **38.2% of entities hold more than one account**
- **38.1% of entities hold accounts at more than one bank** (max: 1,185 banks)
- Types (by account): Partnership 189,683 · Corporation 172,351 · Sole Proprietorship
  149,048 · Country 6,692 · Individual 740 · Direct 67

This partially **removes a limitation §B7 assumed was unavoidable** ("no
customer-vs-account entity resolution"). Real AML investigates a customer holding
many accounts, not an isolated account. Opens up entity-level features and an
entity-level graph. To be exploited on Day 2/3.

### ACH is a simulator artifact — must be measured, not silently exploited

| | share of rows | share of laundering |
|---|---|---|
| ACH | 11.8% | **84.7%** |
| Wire + Reinvestment | 12.9% | **0 of 652,911 rows** |

3,208 of 3,209 labelled pattern transactions are ACH. The naive rule *"flag every
ACH"* alone gets **84.7% recall at 0.64% precision**.

Real laundering is not 85% ACH; this will not transfer. Consequences:
1. Added **arm R** (rules baseline) to `config.ARMS` — real AML stacks begin with a
   rules engine, so this is both domain-authentic and the honest floor every model
   arm must clear.
2. Goes in Limitations prominently.
3. Plan a variant without `payment_format` to quantify how much performance depends
   on the artifact.

### Frozen numbers

- **5,077,237** transactions after truncation; **4,522** laundering; **1 in 1,123**
- Splits (persisted to `data/processed/split_boundaries.json`):
  | split | rows | positives | rate |
  |---|---|---|---|
  | train | 3,046,186 | 2,296 | 0.0754% |
  | val | 1,015,300 | 1,083 | 0.1067% |
  | test | 1,015,751 | 1,143 | 0.1125% |
- 15 currencies; cross-currency is 1.42% of rows. `FX_TO_USD` populated with
  **approximate, static, mid-2022 rates** — stated as such in config and README.
- Parquet: 454 MB CSV → 147 MB; loads in ~1s vs ~13s, 0.58 GB in memory.

**Next (Day 2):** arms R/A/B — transaction features, then account aggregates plus the
B2 typology features (structuring bands, pass-through ratio, dormancy burst,
payment-format groups).

---

## Day 2 — Arms R, A, B

### Headline (validation split; test untouched)

| arm | features | PR-AUC | P@100 | ROC-AUC | best iter |
|---|---|---|---|---|---|
| **R** rules: flag every ACH | 0 | — | 0.75% | — | — |
| **A** transaction only | 11 | 0.0527 | 20.0% | 0.9144 | 139 |
| **B** + account/typology/entity | 87 | **0.1815** | **88.0%** | 0.9366 | 1,961 |

**Arm B lifts PR-AUC 3.44x over arm A** (0.0527 → 0.1815) and takes precision@100
from 20% to 88%. Arm R catches 85.4% of laundering but at 0.75% precision — 133 alerts
reviewed per real case.

### The ROC-AUC demonstration (better than any argument)

Arm A ROC-AUC **0.9144**, arm B **0.9366** — nearly identical, both "excellent".
Their PR-AUCs differ by **3.44x**. ROC-AUC would have reported these two models as
near-equivalent when one is dramatically more useful. Keep both columns in the README:
it demonstrates the metric argument instead of asserting it.

### scale_pos_weight was actively harmful — measured, not assumed

The first run produced arm B *worse* than arm A, stopping after 1 boosting round.
Cause was class reweighting. On arm A:

| setting | best iter | PR-AUC |
|---|---|---|
| `scale_pos_weight=1325` (true ratio) | 1 | 0.0119 |
| `is_unbalance=True` | 1 | 0.0119 |
| `scale_pos_weight=36` (sqrt) | 1 | 0.0464 |
| **none** | **62** | **0.0495** |

The build plan called for class weights; standard advice, wrong here. Reweighting fixes
**calibration** — it stops the loss ignoring a rare class when you need trustworthy
probabilities. PR-AUC uses only the **order** of scores. Multiplying positive gradients
by 1,325 distorts every split so badly the model wrecks its own ranking chasing
calibration we never consume.

`min_data_in_leaf` mattered as much. With 2,296 positives in 3M rows a small leaf
memorises a few laundering rows, validation AP spikes spuriously, and early stopping
fires on the spike. Arm B: `20` → stops at iter 5 (0.127); `100` → iter 1 (0.091);
`300` → 1,961 iterations (0.182).

### Bug: the entity join matched nothing (100% NaN, silently)

`HI-Small_Trans.csv` zero-pads bank IDs (`010`, `03208`); `HI-Small_accounts.csv` does
not (`10`, `3208`). The composite keys had **zero overlap** between files, so every
entity feature arrived as NaN — in the model, contributing nothing, and silent about
it. Account numbers matched 100%, which is what made it easy to miss.

Fixed in `make_data.py::node_id` by stripping leading zeros. Verified: no two bank IDs
collide when normalised, the canonical composite stays unique across all 518,581
accounts, and 100.00% of transaction accounts then join.
Regression test: `tests/test_leakage.py::test_entity_features_actually_join`.

**Second instance of the same class of bug this project has hit: a failed join produces
NaN, not an error.** Both were caught only because the module printed a null rate.

### What the model actually uses (arm B, top gain)

1. `from_out_n_currencies` — currency diversity of the sender
2. `is_cross_currency`
3. `from_out_txn_per_day` — velocity
4. `from_out_counterparties_per_txn`
5. `from_gap_burstiness` — dormant-then-active
6. `to_median_hours_to_forward` — **the pass-through timing feature**
7. `from_layering_format_share`

Two things worth noting. **Time-to-forward ranks high on both sides** even though the
static `flow_through_ratio` tested weak on Day 1 — the *speed* of forwarding carries the
mule signal, not the in/out balance. The typology reasoning was right; the first
formulation of it was not.

And **`payment_format` falls from #5 in arm A to outside the top 6 in arm B**: given
behavioural features, the model leans less on the ACH artifact. Worth quantifying
properly on Day 5 with a no-`payment_format` variant.

`amount_spread_usd` has gain **0** — never used. It is non-zero only on the 1.42% of
cross-currency rows, and `is_cross_currency` captures that better. Kept (a never-split
feature costs nothing) but noted.

### Honest nuance: the arms cross over on the PR curve

Alerts needed for **80% recall**: arm A **55,816**, arm B **88,626**. Arm B needs *more*.

Not a bug — the curves cross. Arm B is far better at the **top** of the ranking (88% of
its first 100 alerts are real vs 20%), while arm A is more precise out in the
high-recall tail. Since compliance teams work the top of a capacity-limited queue, arm B
is the operationally better model, and this is a concrete argument for reporting
**precision@k** rather than a single fixed-recall figure. Reported rather than hidden.

### Verification

10 leakage tests pass: no future data in account features, cold-start rows are NaN not
zero (train cold-start exactly 0%), hand-recomputed aggregates match the table, the
label never reaches the feature matrix, row counts match the frozen split, and the
entity join covers >99%.

**Next (Day 3):** arm C — multi-hop graph topology only (PageRank, reverse-PageRank,
sampled betweenness, Louvain, cycles). Degree-like counts already live in arm B by
design, so arm C must earn its lift from structure a groupby cannot produce.

---

## Day 3 — Arm C: multi-hop graph topology

### Headline (validation; test still untouched)

| arm | features | PR-AUC | P@50 | P@100 | P@500 |
|---|---|---|---|---|---|
| **R** rules: flag every ACH | 0 | — | — | 0.75% | — |
| **A** transaction only | 11 | 0.0527 | 30% | 20% | 14.4% |
| **B** + account/typology/entity | 87 | 0.1815 | 88% | 88% | 35.6% |
| **C** + multi-hop graph | 103 | **0.1938** | **94%** | 88% | **39.6%** |

**Arm C lifts PR-AUC 6.8% over arm B** (0.1815 → 0.1938). Modest, and reported as
modest. The thesis "laundering is relational" is *supported but not dramatically* at
the transaction level, once the baseline already contains per-account aggregation.

Where it helps most is the top of the queue — P@50 88% → 94%, P@500 35.6% → 39.6% —
which is the regime a capacity-limited compliance team actually works in.

### The graph is far smaller than the transaction table suggests

2.5M non-self transactions collapse to **406,998 nodes / 569,215 unique directed
edges**. Average degree 2.80, density 3.4e-06, 28,326 weakly connected components
(largest holds 84% of nodes). Only **1.90% of accounts sit on a cycle**.

Runtimes: PageRank 7.7s · betweenness (k=500) **300s** · core/SCC/WCC 11s · Louvain
46s (33,300 communities). Cached to `graph_features.parquet` with the `train_end`
boundary stored alongside, so a stale cache is rejected rather than silently reused.

### The finding: PageRank degenerates toward degree on a sparse graph

The arm-boundary test flagged `to_pagerank` — arm C's #2 feature by gain. Two separate
things came out of investigating it.

**1. The test itself was measuring wrong.** It correlated features at the TRANSACTION
level, which weights each account by how often it transacts and inflates correlations
toward high-activity accounts. Reverse-PageRank read **1.0000** against degree that
way versus **0.993** per account. The account is the unit these features are defined
on, so it is the unit the check must use. Fixed.

**2. Reverse-PageRank really is degree here.** Account-level correlations:

| feature | vs degree | verdict |
|---|---|---|
| `reverse_pagerank` | **0.993** (out-degree) | essentially degree — **dropped** |
| `pagerank` | 0.822 (in-degree) | correlated but distinct — kept |
| `core_number` | 0.069 | genuinely independent |
| `betweenness` | 0.010 | genuinely independent |

With average degree 2.80, 31.8% of nodes having no in-edges, and 28,326 disconnected
components, the random surfer barely propagates and reverse-PageRank collapses onto an
out-degree count. Degree already lives in arm B, so keeping it would have let arm C
claim a "multi-hop" lift for something a groupby produces.

PageRank is kept on evidence: r=0.822 is high but it takes 212,648 distinct values and
varies *within* each degree bucket, so it carries information degree does not. It still
ranks #5 by gain in the final model.

**Dropping reverse-PageRank IMPROVED arm C: 0.1888 → 0.1938.** The stricter, more
defensible arm is also the better one — a redundant near-duplicate of an existing
feature was costing the model rather than helping it.

### Standalone graph signal (train accounts, before modelling)

| feature | top-decile dirty rate | vs base |
|---|---|---|
| `in_cycle` | 4.77% on-cycle vs 0.70% off | **6.8x** |
| `pagerank` | 1.97% | 2.53x |
| `betweenness` | 1.60% | 2.06x |
| `core_number` | 1.60% | 2.05x |
| `louvain_community_size` | 1.02% | 1.31x |

`in_cycle` is by far the strongest structural signal and maps directly to the CYCLE
typology — but it fires on only 1.9% of accounts, which bounds its contribution. This
predicted the modest overall lift before training confirmed it.

### The third account state, now modelled

Day 2 assumed two account states; there are three. **105,715 accounts (20.6%) appear in
training but have no graph position** — they only ever transacted with themselves
(18% of training rows are self-transfers, mostly Reinvestment). They have arm B
aggregates but NaN topology, which is genuinely different from a never-seen account.
`is_in_graph` makes the distinction learnable.

Self-loops are excluded from the graph: they inflate centrality while carrying no
relational information, and `is_self_transaction` already captures the behaviour in
arm A.

### Bug: training time reported as 21.6 hours

Arm C's first run recorded `train_seconds: 77896`. The machine suspended mid-run and
`time.time()` counted the sleep. Switched to `time.monotonic()`, which pauses during
system sleep on macOS, and nulled the bogus value in `results/arms.json` rather than
leave a number that could not be defended.

### Verification

22 tests pass. New this day: the graph contains exactly the training-window edge set
(exact equality, not a subset check), no self-loops, graph nodes are a subset of
training accounts, `is_in_graph` separates state 2 from state 3, arm C is a strict
column-wise superset of arm B, and no arm C feature correlates >0.99 with degree at the
account level.

**Next (Day 4):** arm D — node2vec embeddings on `G_train`. Given how sparse this graph
is and how far PageRank degenerated toward degree, the honest expectation for embeddings
is another modest lift, not a step change.

---

## Day 4 — Arm D: node2vec embeddings. A negative result.

### Final ablation (validation; test still untouched)

| arm | features | PR-AUC | note |
|---|---|---|---|
| R rules: flag every ACH | 0 | — | 0.75% precision, 85.4% recall |
| A transaction only | 11 | 0.0527 | |
| B + account/typology/entity | 87 | 0.1815 | lift over A **+0.1288** |
| **C + multi-hop graph** | 103 | **0.1938** | lift over B **+0.0123** — **BEST** |
| D + node2vec (dim=32, best) | 167 | 0.1660 ± 0.0155 | **−0.0278 vs C** |

**node2vec embeddings do not improve on explicit graph features here, at any
dimensionality tested.** Arm C is the final engine.

### The dimensionality sweep, and what it actually shows

Arm D at the configured dim=64 scored **0.1349 ± 0.0367** — far below arm C. Rather
than conclude "embeddings are useless", the plan called for a dim sweep to distinguish
that from "64 dimensions was too much capacity for 2,296 positives". Three seeds each:

| dim | columns added | mean PR-AUC | std | range | best single run |
|---|---|---|---|---|---|
| 16 | 32 | 0.1598 | **0.0028** | 0.0055 | 0.1629 |
| 32 | 64 | **0.1660** | 0.0155 | 0.0294 | **0.1834** |
| 64 | 128 | 0.1349 | **0.0367** | 0.0656 | 0.1581 |

Two things fall out.

**Over-parameterisation was real.** dim=64 is the *worst* configuration, not the best.
With only 2,296 training positives, 128 additional dense columns give the model enough
capacity to fit noise — visible in the best-iteration counts, where dim=64 seed 42
stopped after **48** rounds while arm C ran 2,064.

**Variance scales with dimensionality**: std 0.0028 → 0.0155 → 0.0367 as dim goes
16 → 32 → 64. More embedding capacity means the result depends more on which random
walks happened to be drawn.

**But no dimensionality beats arm C.** The best single arm D run anywhere in the sweep
(dim=32, seed 42, 0.1834) still trails arm C's 0.1938. The conclusion is not "we
picked the wrong dim" — it is that learned embeddings add nothing over explicit
topology on this graph.

### Why this was predictable, and was predicted

Three measurements before training pointed the same way:

- Embeddings **alone** predicted account-level laundering at **2.37x** base rate — no
  better than PageRank (2.53x) already in arm C.
- Median embedding vector norm was **0.0803** against a max of 15.98: most accounts
  barely moved from random initialisation.
- The graph averages **2.80 degree** with 28,326 disconnected components, so most
  accounts appear in very few walks and Word2Vec has almost nothing to learn from them.

node2vec needs a graph dense enough for random walks to reveal structure. This one is
not.

### Why running three seeds mattered

Arm C's entire lift over arm B was **+0.0123**. The embedding seed spread at dim=64 was
**0.0656 — 5.3x that lift**. A single arm D run would have reported 0.0925, 0.1539 or
0.1581 depending purely on which seed was drawn, and any of those would have looked
like a real number.

This is the strongest methodological point of the project so far: **a lift is only
meaningful relative to the noise of the procedure that produced it.**

### Bug: arm D silently trained on the wrong feature set

The first three arm D runs used **155 features instead of 231**. Extending the graph
block to `{"C", "D"}` without also extending the arm B block meant arm D was
arm A + graph + embeddings, missing all 87 account and typology features.

It trained without error and reported plausible numbers (0.169 / 0.184 / 0.166). Had
the feature count not been checked against expectation, the reported conclusion would
have been "embeddings degrade the model by 0.021" — from a comparison of arm C against
a crippled arm D.

`tests/test_leakage.py::test_arm_D_is_a_strict_superset_of_arm_C` was written
specifically to catch this, and would have. It was written and then not run — only
`--collect-only` was used. Lesson: collection is not execution.

### Reproducibility: node2vec requires workers=1

gensim's Word2Vec is non-deterministic with multiple worker threads regardless of seed,
because workers consume training examples in nondeterministic order. Measured: two runs
with an identical seed differ by up to **0.055** per dimension at `workers=4`, and are
**bit-identical** at `workers=1`. Arm D runs single-threaded; the cost is only ~50s per
embedding.

The suppressed `gensim ... our_dot_float` exception seen during training was checked
and is cosmetic: no NaN, no Inf, no all-zero rows, no collapsed dimensions.

### Engineering notes

- Embedding cache keyed by **both** dim and seed, with the `train_end` provenance guard,
  so sweep runs cannot clobber one another.
- `N2V_DIM` is env-overridable so the sweep needed no code edits.
- `embeddings.build()` uses a single `pd.concat` rather than 128 individual column
  inserts, which pandas warns about and which measurably slowed the join.

**Next (Day 5):** freeze arm C as the engine, score the **test** split exactly once,
bootstrap CIs, cost-sensitive threshold, pattern-level recall by typology, the
single-bank realism experiment, and SHAP for the agent handoff.

---

# Day 5 — Freeze the engine, score test once

## The engine

Arm C, frozen and persisted to `models/engine_armC.txt`. Retraining reproduced the Day 3
result exactly — **best iteration 2064, validation PR-AUC 0.1938** — which confirms
LightGBM is deterministic here for a fixed seed and thread count, and means the ablation
table and this engine describe the same model. `evaluate.py` asserts that agreement and
warns loudly on drift.

Every later stage loads that file rather than retraining. If the model producing the
published test numbers is not the object the agent scores with, the numbers describe
something nobody can inspect.

## The headline: test is half of validation

| | validation | test |
|---|---|---|
| PR-AUC | **0.1938** | **0.0949** |
| 95% CI (2,000 stratified bootstrap resamples) | [0.1700, 0.2190] | [0.0788, 0.1137] |
| ROC-AUC | 0.8904 | 0.8721 |
| account P@50 | 88% | 68% |
| account P@100 | 88% | 50% |
| account P@200 | 68% | 37.5% |
| pattern recall @ threshold | 92.3% (155/168) | 83.6% (153/183) |
| pattern recall @ top-200 accounts | 42.3% | 29.0% |

**The intervals do not overlap.** That is what the bootstrap bought: without it, "0.19 on
val, 0.09 on test" is a number that might be noise on 1,143 positives. With it, the drop
is established and has to be explained rather than mentioned.

Note also how much smaller the drop is at the unit compliance actually cares about.
Transaction-level PR-AUC halves; **pattern-level recall falls only from 92.3% to 83.6%**.
A ring spans many transactions and an investigator needs one thread to pull, so the
operational degradation is far milder than the headline metric implies. This is the whole
argument for §B5, and it only shows up because `Patterns.txt` was parsed on Day 1.

## Why test is half of validation — the hypothesis that was wrong

The obvious explanation was **cold start**: features are fitted on the training window
only, test sits further from it, so more test accounts should be unseen. Test also has
356,263 distinct accounts against validation's 236,109 on the same transaction count,
which looked like confirmation.

It is not the explanation. Measured:

| split | either account unseen | both in graph |
|---|---|---|
| val | 0.20% | 98.87% |
| test | 0.20% | 98.76% |

Coverage is **identical**. The extra test accounts were seen in training; they were simply
inactive during validation. And restricting to well-covered rows moves both splits by
about the same proportion, leaving the ratio untouched:

| subset | val | test | test/val |
|---|---|---|---|
| all rows | 0.1938 | 0.0949 | 0.49 |
| both accounts seen | 0.1983 | 0.0984 | 0.50 |
| both in graph | 0.2135 | 0.1079 | 0.51 |

A constant ratio across every coverage subset means coverage explains none of it.

## Why test is half of validation — what it actually is

Two candidates remained, and they predict different **shapes**.

*Validation optimism* predicts a **step**. Validation was consulted all week — for
`min_data_in_leaf`, for rejecting `scale_pos_weight`, for dropping `reverse_pagerank`, for
the arm choice — and early stopping picked iteration 2064 precisely to maximise validation
average precision. That inflates validation uniformly and leaves test honest, with no
variation inside either.

*Temporal feature decay* predicts a **slope**. Every account and graph feature describes
training-window behaviour, and that description ages.

Slicing the whole post-training period into six equal windows separates them:

| window | mostly | base rate | PR-AUC |
|---|---|---|---|
| 09-06 13:34 → 09-07 07:18 | val | 0.00099 | **0.2523** |
| 09-07 07:18 → 09-08 01:02 | val | 0.00108 | 0.1968 |
| 09-08 01:02 → 09-08 18:46 | val | 0.00120 | 0.1465 |
| 09-08 18:46 → 09-09 12:30 | test | 0.00079 | 0.0816 |
| 09-09 12:30 → 09-10 06:14 | test | 0.00093 | 0.0856 |
| 09-10 06:14 → 09-10 23:59 | test | 0.00254 | 0.1004 |

**It is a slope.** Performance falls monotonically from 0.2523 to 0.1465 *within
validation itself*, before the test split begins — already most of the way down to test's
0.0949. The val/test boundary is not where the decline happens; it is just where we cut.

So the mechanism is **feature staleness, not validation optimism**. Roughly, PR-AUC halves
for every ~1.5 days of distance from the training window.

The final window's uptick to 0.1004 is a base-rate artifact, not a recovery: its base rate
is 0.00254, about 2.7x the others, and PR-AUC rises with prevalence.

### What this changes

- **The test number is not "the model is worse than we thought."** It is *the model at two
  to four days of feature staleness*. Reported without the decay curve it would invite
  exactly the wrong conclusion.
- **The production implication is concrete**: this engine needs frequent retraining, and a
  real deployment would recompute account and graph features on a rolling window rather
  than a frozen one. That is a monitoring requirement with a measured number attached,
  which is far stronger than the generic "we would monitor for drift" paragraph §B7 asked
  for.
- **It does not change the engine.** Arm C was selected on validation, which is correct
  methodology. Re-choosing after seeing test would turn test into a second validation set.
  The number stands.

## Operating point

The cost ratio (missed case ÷ false-alert review) is an assumption, not a measurement, so
it is reported as a sensitivity strip rather than a single tuned threshold. Test:

| cost ratio | alerts | precision | recall |
|---|---|---|---|
| 10 | 838 | 18.14% | 13.3% |
| 25 | 4,455 | 7.38% | 28.8% |
| 50 | 9,436 | 4.97% | 41.0% |
| **100** | **14,632** | **3.70%** | **47.4%** |
| 250 | 31,244 | 2.06% | 56.3% |
| 500 | 83,004 | 0.97% | 70.6% |
| 1000 | 159,397 | 0.56% | 78.7% |

The **denominator** is sourced. BPI's 2018 *Getting to Effectiveness* survey of 19 US banks
reports $2.4bn of BSA/AML spend across 16 million alerts — about **$150 per alert**, and
that is an upper bound, because the $2.4bn also covers KYC, CTR filing, systems and model
validation rather than alert review alone. The **numerator cannot be sourced**: nobody
publishes the expected cost of one missed laundering transaction, penalties are levied for
programme failures rather than per undetected transaction, and the counterfactual harm is
unobservable. Sweeping the ratio is the honest response to half a grounded quantity.

For scale, arm R (flag every ACH) raises **122,876** alerts at 0.75% precision for 85.4%
recall. At ratio 100 the engine reviews **8.4x fewer** alerts.

## Pattern recall by typology (test, at the cost-optimal threshold)

| typology | caught | recall |
|---|---|---|
| BIPARTITE | 16/17 | 94.1% |
| FAN-IN | 17/19 | 89.5% |
| SCATTER-GATHER | 20/23 | 87.0% |
| FAN-OUT | 17/20 | 85.0% |
| GATHER-SCATTER | 33/39 | 84.6% |
| STACK | 19/23 | 82.6% |
| RANDOM | 14/17 | 82.4% |
| **CYCLE** | **17/25** | **68.0%** |

CYCLE is the clear weak spot, and the reason is known rather than mysterious: `in_cycle`
and `scc_size` are computed on the **static training-window graph**, so a cycle that forms
during the test window is invisible to them. This is precisely the limitation §B4 flagged
when time-respecting motifs were scoped and deferred — the deferral has a measured cost,
which is a better outcome than an unexamined one.

RANDOM at 82.4% is worth stating plainly: the engine is not detecting shape alone. If it
were, the shapeless control class would collapse. It does not, which means transaction and
behavioural attributes carry real weight — consistent with the arm B → C lift having been
only +0.0123 in the first place.

## Caveat carried with every pattern metric

Only **2,554 of the 4,522** laundering transactions surviving truncation belong to a named
pattern in `Patterns.txt`. The other 1,968 are laundering the generator did not group into
a ring. Pattern-level recall describes the labelled 56.5%, not all laundering, and
`results/engine.json` records that alongside the numbers.

## Engineering notes

- **The bootstrap is 9x faster and provably identical.** `average_precision_score` costs
  ~260 ms on a 1M-row split; 2,000 resamples per split is nine minutes. Sorting once
  outside the loop and expressing each resample as a multiplicity vector brings it to
  ~29 ms — agreement with sklearn to **5.6e-17**, verified in `test_engine.py`. The point
  estimate still comes from sklearn; only the interval uses the fast path.
- Resampling is **stratified by class**. PR-AUC moves with the base rate, so an
  unstratified interval at a 1-in-900 rate would partly measure class-balance jitter.
- `results/engine.json` is written **before** the figure work, and figure failures are
  caught. The test split is scored once; losing the result to a matplotlib error would
  force a choice between re-scoring test and having no result.
- A **SHA-256 digest of the test scores** is recorded (`6b577bcf067ef11f`). A future run
  producing a different digest means the engine changed after freezing.
- SHAP via LightGBM's native `pred_contrib=True` — exact TreeSHAP, no `shap` dependency in
  the path. 200 alerts explained in ~1 s per split, because only each alert's single
  score-driving transaction is explained rather than the whole split.

## Bug caught: the rules baseline was drawn as a curve

The first PR-curve figure plotted arm R as a **line from (0, 1) to (1, base_rate)**,
because `precision_recall_curve` interpolates between a binary score's one real threshold
and the degenerate endpoints. On a log axis that line appeared to *dominate arms A and C*
across most of the recall range — the exact opposite of what the rules baseline says.
`baselines.py` had warned about this in a docstring since Day 2 and the figure code
ignored it. Arm R is now drawn as the single marker it is.

**Next (Day 6):** the triage agent — the knowledge base and its index are built (below);
the agent core needs the Anthropic key.

---

# Day 6 (part 1) — Knowledge base and retrieval index

The agent needs to say *why* an account looks like laundering, in language a compliance
function recognises. It could invent that language and it would sound plausible, which is
the problem. Retrieval grounds each judgement in text a supervisor actually published.

## What was built

`kb/corpus/` — **17 documents, 67 chunks**, every one traceable to an entry in
`kb/sources.json` with a URL and a retrieval date. `load_corpus()` **raises** if a document
declares a `source_id` that is not registered, so provenance is enforced rather than
intended.

Sources retrieved and read, not paraphrased from memory:

| source | what it gives |
|---|---|
| FFIEC BSA/AML Manual, Appendix F | funds-transfer and shell-company red flags — the list US examiners work from |
| FFIEC Appendix G | structuring: 31 USC 5324, 31 CFR 1010.100(xx), the $10,000 CTR and $3,000 recordkeeping thresholds |
| FFIEC Appendix L | SAR quality: who/what/when/where/why **and how** ("modus operandi") |
| FinCEN SAR Narrative Guidance (Nov 2003) | the introduction / body / conclusion structure the case-note template mirrors |
| FinCEN FIN-2014-A005 | the **funnel account** definition and red flags — the regulator's name for our FAN-IN shape |
| FinCEN FIN-2020-A003 | the unwitting / witting / complicit money-mule taxonomy |
| 31 CFR § 1020.320 (eCFR) | SAR filing deadline: **30 calendar days**, extendable to 60 if no suspect identified |
| BPI, *Getting to Effectiveness* (2018) | alert economics — see below |
| FATF Virtual Assets Red Flag Indicators (2020) | relevant because Bitcoin is one of the fifteen currencies |

The eight typology documents map each dataset shape to its real-world name, the matching
regulatory red flags, and the specific model features that fire — the §C1 artifact. That
mapping is **our synthesis and is labelled as such**; it is not presented as regulatory
text.

## The false-positive statistic, traced to its source

Every vendor page repeating "90–95% of AML alerts are false positives" cites, directly or
at one remove, the **Bank Policy Institute's 2018 survey of 19 US banks**. Reading the
primary source gives better numbers than the slogan:

- 16 million alerts reviewed in 2017; **640,000+ SARs filed** — a **4% alert-to-SAR
  conversion rate**, so 96% of alerts produced no SAR. Slightly *worse* than the figure
  usually quoted, and computed from BPI's own figures rather than repeated.
- A median **4% of those SARs** drew law-enforcement follow-up, putting roughly **0.16% of
  all alerts** on a path to law-enforcement interest.
- 18% of alerts related to structuring.

## The chunking bug that would not have errored

ChromaDB's default embedder is ONNX all-MiniLM-L6-v2 with a **256-token window**. The
corpus documents run 1,800–2,400 characters (400–600 tokens). Embedding whole files would
have stored the complete text while embedding only the opening — retrieval would silently
never match anything in the second half of any document. Nothing raises.

Documents are therefore split at markdown structure boundaries with the **document title
prepended to every chunk** (a bare list of funds-transfer red flags embeds almost
identically to a bare list of shell-company red flags without it). Measured: median 114
tokens, max 194, **62 tokens of headroom**, nothing truncated.

Getting that measurement right needed one more step. `tokenizer.json` ships a truncation
setting of **128**, so measuring with the tokenizer as loaded reported *every* long chunk
as exactly 128 tokens — over-length chunks would have been invisible to the very check
meant to find them. ChromaDB overrides it to 256 at runtime (its own source comments on
the discrepancy). `kb_index.token_counts()` disables truncation before measuring, and
`test_engine.py` asserts against the real 256.

## Retrieval, spot-checked

| query | top hit |
|---|---|
| "account receives many small deposits then immediately wires them out" | FinCEN funnel accounts (0.552) |
| "how long do I have to file a SAR after detecting suspicious activity" | Regulatory framework / 31 CFR 1020.320 (0.690) |
| "money moving in a circle back to where it started" | Typology: CYCLE (0.544) |

Cosine distance, not the L2 default — these are normalised sentence embeddings and L2
would let chunk length influence ranking.

## The embedder runs locally

all-MiniLM-L6-v2 executes through onnxruntime on this machine: no API key, no per-query
cost, reproducible offline. The Anthropic key is needed for the agent, not for retrieval.

Getting the model onto the machine took three attempts and the diagnosis is worth keeping.
ChromaDB's own S3 download stalled; so did HuggingFace's 90 MB fp32 export — **at byte
76,214,272 every single time**, across two hosts and three HTTP clients, including on a
freshly-opened ranged request. A fixed offset across independent connections is an
environment transfer cap, not a flaky link, so the fix was a smaller file: the 23 MB
quantized ONNX build, which is what ChromaDB ships anyway. It downloaded in one attempt.
Placing the six files ChromaDB checks for into its cache directory makes it skip its own
download entirely.

**Next:** the triage agent core. It needs `ANTHROPIC_API_KEY` in `.env`, which is currently
empty.

---

# Day 6 (part 2) — The triage agent

## What was built

`src/agent/` — a three-module L1 triage layer:

- **`dossier.py`** assembles what the agent sees: alert header, account profile over the
  scored window, capped evidence table, the engine's own SHAP attributions, and retrieved
  regulatory passages. It also defines `citable_ids()`, which is the whitelist the
  hallucination check runs against.
- **`triage.py`** makes the call and validates the result. Structured outputs via
  `output_config.format`, so the schema is enforced by the API rather than requested in
  prose.
- **`run_triage.py`** runs a queue, 8-way concurrent.
- **`summarise.py`** reports the metrics, always against the escalate-everything control.

## The API had moved on

`messages.create` no longer accepts `temperature`. The original design said
"temperature=0 for reproducibility"; that is not a claim this project can make any more,
and the code and README say so rather than quietly dropping the parameter. What actually
makes a re-run reproduce is the on-disk response cache, keyed by the exact prompt.

`output_config.format` replaced the tool-use trick for structured output, which is a
straight improvement: the schema is the contract rather than a tool the model chooses to
call.

## The headline: the agent oscillated between two useless calibrations

**v1 — the original prompt.** On the top 25 alerts:

| | |
|---|---|
| closed | 6 of 25 (24% of the queue) |
| false positives closed | 1 of 2 |
| **true positives closed** | **5 of 23 — 21.7% LOST** |

Losing a fifth of real cases is not a triage layer a compliance function would accept.

**v2 — recalibrated for error asymmetry.** The prompt was rewritten to state that the two
errors are not symmetric, that closing requires a positive legitimate explanation rather
than an absence of visible suspicion, and that inability to decide is the definition of an
escalation. On 72 alerts:

| | |
|---|---|
| closed | **0** |
| true positives closed | 0 — 0.0% lost |
| false positives closed | 0 of 9 |
| escalation precision | 87.5% vs 87.5% control — **+0.0 points** |

Zero true-positive loss, and zero value. This is exactly the escalate-everything baseline
that §A8 exists to measure against.

## Why prompt tuning cannot fix this

The v1 rationales say plainly what the problem is. Four of the five true positives it
closed were single-inbound-transaction accounts, and the agent's reasoning on them was:

> "the model's top-weighted drivers (9 distinct currencies and 12 distinct banks in
> outgoing activity) describe the counterparty/sender's broader profile, not anything
> observable in this account's transaction history"

> "that counterparty's transactions are not in evidence here, so the pattern that alarmed
> the model cannot be verified from this account's side"

Those accounts are **the receiving spokes of a fan-out**. A spoke receives one payment and
does nothing else. From that account's own rows it is indistinguishable from an ordinary
receipt — the laundering is visible only in the *sender's* shape, which the dossier does
not contain.

So the agent was not being careless in v1. It was reasoning correctly about a context that
cannot answer the question, and when forced to decide anyway it can only be reckless (v1)
or useless (v2). **The information is missing, not the judgement.**

This is a better finding than a tuned number would have been, and it was only visible
because the evaluation reports the control alongside the agent. "Triage accuracy 87.5%"
would have looked respectable and meant nothing — it is exactly the control's score.

**The fix is Day 7's agentic step** (pull a flagged counterparty's subgraph), which the
plan scoped before this was measured and which now has a measurement motivating it.

## What does work

- **Zero hallucinated citations across every alert run** (0/25, 0/72). Citations are
  validated by set membership against the IDs actually shown, and an ID that exists but
  was not in the dossier still counts as fabricated, because the model could not have read
  it. One alert cited nothing.
- **Zero invalid source ids.** When the agent names FFIEC Appendix F, that passage was in
  its context.
- The SHAP cross-check works and is genuinely useful. The agent repeatedly noticed
  mismatches between the model's stated drivers and the visible evidence — "the dominant
  SHAP driver is 9 distinct currencies used by the sender, far more than the 3 currencies
  visible in this window's evidence" — which is precisely the behaviour B3 was after.
- Labelling self-transfers mattered. The rank-1 alert's rationale correctly described
  "a same-account self-transfer that converts a Yuan balance into a different currency
  before the funds leave", which would have read as five external fan-out spokes without
  the label. 11.6% of this dataset is self-transfers.

## Bugs found

**The response cache ignored the system prompt.** The key hashed the model and the
rendered user prompt only. Recalibrating the agent's closing criteria and re-running 25
alerts returned all 25 from cache, 0 billed, byte-identical — indistinguishable from a
prompt change that had no effect. Third time this project has been bitten by a cache key
missing part of its own provenance (entity join, Day 2; graph features, Day 3). The key
now covers the system prompt, the schema and `max_tokens`.

**One failed alert destroyed a 200-alert run.** A single `max_tokens` overflow propagated
out of `pool.map` and discarded 190 already-billed completions. Per-alert failures are now
caught, recorded and reported; the batch continues.

**Structured output truncation is silent.** Exceeding `max_tokens` mid-object returns an
empty string, not an error. The diagnostic now reports `stop_reason` and the token count
so the cause is named rather than guessed at. Raised 2000 -> 4000 -> 8000.

## Cost and latency

$0.052 per alert, median 26.9 s, 8-way concurrent. A full 200-alert queue is about
**$10.40** and roughly 15 minutes.

## Blocked

The run stopped at 72 of 200 alerts: **the Anthropic account is out of credit.** The
remaining 128 alerts, the RAG ablation (`--no-rag`), and Day 7 all need it topped up.

**Next (Day 7):** the agentic context-pull step, which the v1/v2 result above now
motivates directly, plus the SAR-structured narrative (§C2).

---

# Day 7 — The pivot: triage cases, not accounts

## Why the unit changed

Day 6 measured that per-account triage cannot work, and the agent explained why itself:

> "that counterparty's transactions are not in evidence here, so the pattern that
> alarmed the model cannot be verified from this account's side"

Four of the five true positives it wrongly closed were **receiving spokes of a fan-out** —
one inbound payment and nothing else. In isolation a spoke is indistinguishable from an
ordinary receipt; the laundering lives in the *sender's* shape. Asked to judge a spoke
alone the agent can only be reckless or useless, and prompt engineering cannot supply a
fact that is not in the context.

So alerts are now grouped into **cases** — connected clusters of alerted accounts — and
the agent triages a case. This is not a cost compromise, it is the correct unit:
`Patterns.txt` labels rings, investigators open cases on networks, and naming a topology
from one spoke was never a fair task. That it also costs 4.4x fewer LLM calls is a
convenience, not the argument.

## How cases are formed

Two alerted accounts join the same case if they transact directly, or if they share a
non-alerted counterparty that is shared by at most 25 alerted accounts.

That cap is load-bearing. Including counterparty-to-counterparty edges collapsed the
validation queue into a **single component of 15,027 accounts** — one "case" containing
everything, which is no grouping at all. With the cap:

| | accounts | cases | max | median | productive |
|---|---|---|---|---|---|
| val | 200 | **45** | 62 | 2 | 53% |
| test | 200 | **53** | 72 | 2 | 42% |

The 53% / 42% balance matters as much as the count. The account queue was 68% productive
and its top-25 slice 92%, with **two** false positives in it — which is why the Day 6
v1/v2 calibration was unreadable. A near-balanced queue can actually be calibrated.

## Results, and an operating-point frontier

Four configurations, each a full 45-case validation run:

| config | escalation precision vs control | FP removed | TP lost |
|---|---|---|---|
| A no counterparty context | **+13.3 pts** | 71% | 50.0% |
| B every external counterparty listed | +3.2 pts | 52% | 45.8% |
| C hub flag only (reach >= 10) | +10.3 pts | 62% | 41.7% |
| D hub flag + small-case caution | +4.7 pts | 38% | **25.0%** |

Control (escalate everything) is 53.3% precision, 100% recall, 21 wasted reviews.

The first thing to say is that **all four beat the control**, which nothing at
account level ever did — v2 scored +0.0 points, exactly the control. The pivot moved the
agent from "identical to doing nothing" to a real trade.

The second is that these are not four attempts at one answer; they are four points on the
triage layer's own precision/recall frontier, and which one is correct depends on the same
cost ratio the engine's threshold depends on. **D is the default here on domain grounds**:
in AML a missed network costs more than a wasted review, so 25% TP loss at 38% FP removal
beats 50% TP loss at 71%.

The third is that **none of these is production-ready.** 25% true-positive loss would not
pass a model validation review. The honest description is a working architecture with a
measured operating-point frontier, not a shippable triage layer.

## Where the remaining loss lives, and why

Broken down by case size, config C:

| case size | TP loss | FP removal |
|---|---|---|
| 1-2 accounts | 67% | 79% |
| 3-5 accounts | 57% | 0% |
| **6+ accounts** | **0%** | — |

**Cases of six or more accounts were dispositioned correctly every time.** Every failure
is in small cases, and the mechanism is structural rather than a reasoning failure: a case
is built from accounts the model *alerted on*, so when only two members of a ring cleared
the threshold, the case contains two accounts and one transfer and the ring is invisible.

That points at the engine, not the agent. A deeper alert queue would produce more complete
cases, at the cost of reviewing more of them — which is the Day 5 cost-ratio argument
appearing one layer up.

## The finding worth keeping: same data, different signal-to-noise

Config B added every external counterparty's window activity, on the reasoning that more
context is better. It made the agent **worse** — precision over control fell from +13.3
to +3.2 points. A section present in every case, populated with unremarkable numbers,
reads as background and nudged the agent toward escalating generally rather than
discriminating.

Filtered to a sparse flag (config C), the identical data separates cleanly:

    external counterparty reach >= 10 fires on 42% of productive val cases
    and 0% of non-productive ones

Same information, opposite effect. The threshold is tuned on validation over 24
productive and 21 non-productive cases, so the interval on that 0% is wide, and it is
recorded in `config.CASE_HUB_REACH` rather than buried in a prompt.

## Cost

The pivot and the model change together took the cost of a full pass from **$6.92 to
$0.33**:

| | unit | calls | per unit | full pass |
|---|---|---|---|---|
| Day 6 per-account, Sonnet 5 | account | 200 | $0.0346 | $6.92 |
| Day 7 per-case, Haiku 4.5 | case | 45 | **$0.0074** | **$0.33** |

Case prompts are *smaller* than account prompts (~3,400 tokens against ~5,844) because
the evidence cap applies once per case rather than once per account.

Three things had to be fixed to get an honest number:

- **The price constants were wrong.** `$3/$15` is Sonnet 4.5; the calls went to Sonnet 5
  at `$2/$10`. Every cost reported on Day 6 was ~33% too high. Prices now live in a table
  keyed by model id, and an unknown model raises rather than reporting a guess.
- **82% of output tokens were reasoning, not text.** The emitted JSON is ~412 tokens
  against 2,291 billed. `output_config.effort` defaults to `high` and was never set.
- **Haiku 4.5 rejects `effort` outright** with a 400. The fallback is learned once per
  run rather than retried on every call — without that a 45-case run made 90 requests,
  half of them errors.

## Zero hallucinated citations, still

0 of 45 cases produced an invalid transaction citation, 0 cited a source that was not
retrieved, and 0 cited nothing — holding across every configuration and both units.
Citations are validated by set membership against exactly the rows rendered into the
prompt, so an ID that exists but was not shown still counts as fabricated.

## Budget

A hard $10 ceiling is now enforced rather than remembered: every billed call is appended
to `results/spend_ledger.json`, and a batch prices one real call and refuses to start if
the projection would breach the remaining budget. Exceeding it requires typing
`--confirm-spend <amount>`.

This exists because a 200-alert run died at alert 72 with credits exhausted, after an
earlier attempt had billed 51 completions that were then discarded when one bad alert
killed the batch. Nothing had priced the work before starting it.

Four full validation runs and a dev subset: **$1.34 of $10.00**.

## Bugs fixed

- **Direction was per-account in a per-case dossier.** `alert_evidence` writes IN/OUT
  relative to whichever account it was called for, and a case pools rows across members,
  so shared rows carried an arbitrary member's perspective. An account that only ever
  received money was reported as having sent it. Direction is now derived from the
  account being described, and case evidence is labelled IN / OUT / INTERNAL / SELF
  relative to the case boundary.
- `evidence_table` crashed when a case was built without model scores; it now ranks by
  USD value and renders the score as absent rather than raising.

---

# Day 8 — The SAR narrative, and the measurement that decided what this layer is for

## What was built

- `case_note` added to the output schema as a structured object — `introduction`,
  `body`, `conclusion` — following the FinCEN SAR Narrative Guidance Package (Nov 2003),
  which is already in the knowledge base as `fincen-sar-narrative-2003`. The template is
  enforced by the API rather than requested in prose, so a model under length pressure
  cannot quietly drop the conclusion, which is the only section that tells the next
  reviewer what to do.
- The hallucination check extended to the narrative. `cited_transaction_ids` was a clean
  set operation because the API types it; prose is not, and an invented id in the body of
  a case note misleads an investigator exactly as much. Transaction ids and account ids
  are now scanned out of every free-text field and checked against `citable_accounts()` —
  which is *what the dossier rendered*, not what the case contains, because a 62-account
  case shows only twelve of its members.
- `src/agent/evaluate_agent.py` — the Day 8 scorer. Writes `results/agent.json`.
- Test case queue scored once, both configurations declared before either ran.

## The headline: the disposition carries no signal, and the number that said otherwise was an artifact

The Day 7 frontier reported escalation precision above the escalate-everything control.
That number does not survive being asked whether it means anything.

Fisher's exact test on the 2×2 of disposition against ground truth, one-sided, asking
whether the escalated set is enriched in productive cases relative to the queue it was
drawn from:

| split | escalation precision | queue base rate | p |
|---|---|---|---|
| **test, n=53** | 42.5% | 41.5% | **0.51** |
| test, cases of 1-2 accounts (n=43) | 29.7% | 30.2% | 0.75 |
| val, n=45, best of three orderings | 60.0% | 53.3% | **0.24** |
| val, same run, cases of 1-2 (n=31) | 38.5% | 38.7% | — |
| val, same run, cases of 3+ (n=14) | 83.3% | 85.7% | — |

Read the last two rows together. That configuration scored **+6.7 points over the
control**, and inside *every* size bucket its escalation precision sits on the base rate
to within a point and a half. The lift is Simpson's paradox and nothing else: small cases
have a lower base rate (38.7%) than large ones (85.7%), the agent closed far more
aggressively among the small ones, and pooled precision rose without a single case being
judged better than chance.

**Pooled precision above a control is not evidence of skill when the agent also chooses
which subpopulation to act on.** This is the single most useful thing measured in the
project, and it was found only because the size breakdown from Day 7 was already there to
disaggregate against.

The test split confirms it out of sample: +1.0 points, p=0.51.

## Why it was never going to work, stated properly

The engine is a gradient-boosted model over ~100 features including multi-hop graph
topology. The agent receives a rendered text summary of a subset of that. Asking it to
re-rank the engine's own output is asking a model with strictly less information to
improve on one with more, using the same evidence. There is no prompt for that.

Day 7 diagnosed the missing fact as the fan-out shape and fixed it by changing the unit
to the case. That was correct and it did help — 6+ account cases now lose zero true
positives on both splits — but it fixed the *shape* problem, not the *information*
problem. The remaining ranking signal that would separate a productive small case from an
unproductive one is in the engine's features, not in the dossier.

## Field order in a structured output is not cosmetic

Three orderings of the same schema, same prompt, same 45 validation cases:

| declaration order | escalation precision | TP lost | fabricated ids in prose |
|---|---|---|---|
| 1. decision first, note last | +6.7 pts | 37.5% | 0 |
| 2. note first, decision last | −0.2 pts | 29.2% | **2 (4.4%)** |
| 3. evidence → note → decision | −3.0 pts | 29.2% | 0 |

Structured output is generated in schema-property order — verified by reading the key
order off a returned record. So with `disposition` declared first, **"escalate" was the
model's first output token**, produced before a word of analysis existed, and every field
after it was written to justify a call already made.

Moving the note to the front fixed the reasoning order and broke the grounding: with no
citation list committed before the prose, two accounts appeared in narratives that were
never in a dossier. A committed citation list is what bounds what the prose can say.

Ordering 3 keeps both properties — pull the evidence, write from it, then decide — and it
is what ships. Note that it is the *worst* of the three on disposition precision and that
is not a reason to reject it: given p=0.24, the ranking of those three numbers is noise,
and choosing on noise is how a project talks itself into a result. It was chosen on
grounding, which is the property that measures.

## What the layer is actually for

Strip out the disposition and what remains is measured, real, and worth having:

**A grounded case note.** 53 test cases, median 501 words, median 8 transaction ids
written into prose — roughly 26,500 words of generated narrative, of which **one**
fabricated identifier (an account, 1.9% of cases). Zero fabricated transaction ids in the
structured citation list, on either split. All three sections present in 53 of 53.

**Typology naming against ground truth.** Scored against `Patterns.txt` at ring level:
58.3% any-match, 50.0% dominant-match — on the 12 test cases whose members touch a named
ring. Ten productive test cases have laundering the simulator never grouped into a ring;
their truth is *unknown*, not NONE, and scoring them against NONE would manufacture
credit or blame out of a labelling gap, so they are excluded and counted.

**A retrieval layer that demonstrably changes the note.** See below.

## The RAG ablation, and which half of it survives a test

Both test configurations were fixed before either ran.

| | retrieval on | off | |
|---|---|---|---|
| escalation precision | 42.5% | 50.0% | both ≈ base rate, p=0.51 / p=0.08 |
| typology any-match | 58.3% (7/12) | 33.3% (4/12) | Fisher p=**0.21** — not significant |
| red-flag indicators per case | **2.74** | 1.06 | Mann-Whitney p=**4.3e-08** |
| notes naming no indicator at all | 3 of 53 | **35 of 53** | |
| notes citing a source | 50 of 53 | 7 of 53 | |
| cost per case | $0.0123 | $0.0106 | |

The typology delta is the one that looks like the headline and it is the one that cannot
carry it — twelve labelled cases, p=0.21. Reported as a direction.

The indicator result is overwhelming and is the actual finding: **retrieval is what makes
the note cite published regulatory indicators instead of asserting suspicion in its own
voice.** Two thirds of no-retrieval notes name no indicator at all; with retrieval that
falls to 6%, and the sources actually leaned on are FFIEC Appendix F (50 cases), the
AMLSIM typology reference (49) and FinCEN FIN-2014-A005 on funnel accounts (15).

That is the correct claim for a RAG layer in this position. It was never going to make
the model a better ranker; it makes the output defensible to an examiner.

## Where the errors still are — the Day 7 finding replicates

| case size | test: TP lost | val: TP lost |
|---|---|---|
| 1-2 accounts | 2/13 (15%) | 5/12 (42%) |
| 3-5 accounts | 0/7 (0%) | 2/7 (29%) |
| 6+ accounts | **0/2 (0%)** | **0/5 (0%)** |

Zero true-positive loss in large cases, on both splits, in every configuration tried. The
structural reading holds: a case is assembled from accounts the model alerted on, so when
only two members of a ring cross the threshold, the ring arrives invisible rather than
absent. It is an alert-depth property, not an agent property.

The typology confusion matrix says the same thing from the other side. On validation the
dominant error was **FAN-OUT → NONE, 5 of 6** — a fan-out reaches the agent as one or two
alerted spokes, the hub is not itself alerted so it is not a case member, and "no shape"
is the correct description of what was handed over. GATHER-SCATTER (57%) and
SCATTER-GATHER (60%) score far better because they arrive as multi-account cases where
the shape is visible.

## Cost

$0.0123 per case with retrieval, $0.0106 without. The test evaluation — 53 cases scored
twice — cost $1.21 in total. Day 8 spend $2.40 including three validation runs; **$4.29
of the $10 ceiling used, $5.71 remaining.**

## Bugs fixed

- `citable_accounts()` raised on evidence rows without a `counterparty` key. Now reads
  every field defensively, because the same function serves case and account dossiers.
- A bank-qualified account id contains its own suffix, so a single fabricated
  `999:80DEADBEE` was counted twice — once by the qualified pattern and once by the bare
  one — silently inflating the fabrication rate. The bare scan now runs on prose with the
  qualified matches removed.
- One validation case died on a 500 from the API and was excluded; re-running filled it
  from cache for one billed call.
- `evaluate_agent.py --split val` wrote `results/agent.json`, the reported test artifact.
  It now writes `agent_val.json` for anything that is not the test split.

## The one fabrication, looked at

Worth reading rather than reporting as a rate. In `CASE-TEST-002` — eleven members — the
note names `148016:811C597B0`. That account does not exist in the dossier, but both of its
halves do: `119:811C597B0` is a member, and `148016:811FCA7B0` is a *different* member.
The model spliced one member's bank prefix onto another's account number.

That is a compositional error, not an invention from nothing, and it has a cheap
mitigation available if it recurs: the bank prefix carries no information the note needs,
so members could be rendered with a case-local index and the full id kept in a lookup.
Left as measured for now — one occurrence in 53 cases is not enough to design against, and
the check that would catch a recurrence is running.

Separately, `CASE-TEST-053` returned an **empty** `cited_transaction_ids` while writing two
valid transaction ids into its prose. Under the shipped ordering the citation list is
emitted first, so the model committed to citing nothing and then cited anyway. The prose
ids were both real, and they were real because the check validates against the *dossier*
rather than against the model's own list — which is the reason to define the citable set
from what was rendered rather than from what the model claims to have used.

---

# Day 9 — Sourcing the last unsourced number, and a threshold that was measuring the wrong thing

## The FX table

`config.FX_TO_USD` held approximate mid-2022 rates written from memory on Day 1. It was
the only number in the project without a source, and `docs/LEARNING_NOTES.md` already
listed "your FX rates aren't sourced" as one of the three genuinely hard interview
questions, with the note *fix it before this goes on a CV*.

Rates are now sourced for a single fixed date, **2022-09-01**, the dataset's first day:

| source | covers |
|---|---|
| ECB euro foreign exchange reference rates (daily, 14:15 CET) | EUR, GBP, CHF, CAD, AUD, ILS, BRL, CNY, MXN, INR, JPY |
| Bank of Russia official rate, USD/RUB 60.2386 | RUB — the ECB suspended its rouble reference rate in March 2022 |
| SAMA peg, 3.75 SAR per USD since June 1986 (via IMF DSBB) | SAR — a peg, so no date sensitivity |
| blockchain.com daily average, $20,047.68 | BTC — labelled separately; not a currency and not from a central bank |

## Correcting it had to be a measurement, not an edit

FX feeds every amount-derived feature **and** the money-weighted graph, so swapping the
table changes the model — which would invalidate the arm ablation, the frozen booster,
and a test split scored exactly once. Worse, the whole agent half is built on the alert
set that engine produces, so a rebuild cascades into re-running Days 6–8 including their
paid API calls.

Spending that to correct a number that might not matter is a bad trade made silently. So
`src/diagnose_fx.py` asks two separate questions before anything is promoted.

**62.7% of transactions are not in US dollars**, so this is real exposure, not rounding.
The guessed table turned out to be within 5% everywhere — worst case the **Euro at
4.96%**, and the Euro is 23% of all rows, which makes it far and away the dominant
exposure.

| question | result |
|---|---|
| INFERENCE — frozen model, features rebuilt under sourced rates | val PR-AUC 0.1938 → **0.1984** (+0.0047), Spearman ρ of the scores **0.9953** |
| TRAINING — arm C retrained from scratch under sourced rates | val PR-AUC 0.1938 → **0.1844** (−0.0094) |

The shipped artifact is robust: its ranking is essentially unchanged.

## The threshold I got wrong, and what caught it

The first version of the diagnostic used `evaluate.PR_AUC_TOLERANCE` (0.002) as the
materiality threshold, reasoning that `freeze_engine` already uses it to decide whether
the engine has changed. It fired: **MATERIAL, rebuild the engine, declare a second test
scoring.**

That constant is a **reproducibility** tolerance. It exists to check that re-running
identical code on identical data returns an identical number — a determinism test, where
anything above floating-point noise is a genuine defect. It says nothing about whether
two models trained on slightly different data differ meaningfully.

The estimator's own bootstrap SD on validation is **0.0127**, with a 95% CI of
**[0.1700, 0.2190]** — both computed on Day 5, long before this question existed. Against
that, a 0.002 threshold calls anything above **0.16 standard deviations** material, which
would flag almost any retrain of anything.

Measured properly, the retrained value **0.1844 sits 0.74 SD from 0.1938, inside the
frozen engine's 95% CI.** Not distinguishable from estimation noise on 1,083 positives.

> **IMMATERIAL.** The engine stays frozen, the sourced table is recorded as provenance,
> and this measurement is the justification.

The lesson is not about FX. **A threshold borrowed from a different question will answer
that different question.** 0.002 was a perfectly good number for "did this re-run
reproduce?" and a nonsense one for "are these two models different?" — and the only
reason the error surfaced is that the verdict was surprising enough to check, with the
CI already sitting in `engine.json` from four days earlier.

`judge()` is split out from the run so a corrected criterion can be re-applied to a saved
result without repeating the work — which mattered here, because the Louvain pass alone
takes **8,109 seconds**.

## Bug fixed: the graph cache key, again

Graph edges are weighted by USD volume and PageRank follows those weights, so the FX
table is an input to every graph feature. The cache key was `train_end` and `seed` only.
Rebuilding under corrected rates would have silently returned the old money-weighted
graph and the entire FX experiment would have measured nothing while looking like it
measured something.

**Fourth time this project has been bitten by a cache key that omitted part of its own
provenance** — the entity join on Day 2, graph features on Day 3, the agent's system
prompt on Day 6, and now the currency table those same graph features are weighted by.

Then immediately a fifth, caught before it could run: the single-bank experiment builds
arm C from one institution's visible subset — same `train_end`, same seed, same FX,
*different rows*. Without the training frame in the key it would have been handed the
full inter-bank graph from cache and concluded that partial visibility costs nothing. A
clean, publishable, entirely wrong number.

Both are now in the key, and the cache path carries the digests so two experiments cannot
overwrite each other's tables.

## Typology accuracy nearly shipped without a control

Day 8 reported typology as **58.3% any-match, 50.0% dominant-match** against
`Patterns.txt`. Both numbers are correct and neither should have been reported alone.

Storing per-case ground truth in `results/agent.json` — so the README could name one
case's ring without reloading 5M transactions — made the problem visible immediately. The
truth entry for `CASE-TEST-002`, an eleven-account case, lists **all eight typologies**.

**Any-match is not a fixed-difficulty task.** A large case touches many injected rings, so
"the predicted label is one of the types present" gets easier the bigger the case is. Two
of the twelve labelled test cases have chance rates of 1.00 and 0.88 on their own. The
honest baseline is per-case, mean(|types| / 8) = **34.4%**, against which 58.3% is a real
but much smaller lift than it looked.

**Dominant-match has a fixed 12.5% chance rate but a badly skewed class distribution.**
Eight of the twelve labelled cases are GATHER-SCATTER, so:

| | dominant-match |
|---|---|
| always predict GATHER-SCATTER | **66.7%** |
| the agent | **50.0%** |

**The agent does not beat the trivial baseline on typology.** At n=12 neither figure is
well determined, and that is exactly why both now travel with their controls.

The uncomfortable part is that this is the *same* error as the disposition: a metric
reported without asking what a system doing no work would score. The disposition got a
control from day one because the build plan demanded one (A8). Typology did not, so it
was reported bare — and by a project whose central finding that week was that a pooled
number without a control had fooled it once already.

`typology_scores` now emits `baselines` unconditionally, and the renderer prints the
control beside every accuracy.

**Next:** B1, then the README generated from `results/*.json` and the demo.
