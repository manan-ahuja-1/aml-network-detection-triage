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
