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
