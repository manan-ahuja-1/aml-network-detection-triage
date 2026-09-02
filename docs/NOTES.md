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
