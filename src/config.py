"""Single source of truth for paths, constants and frozen decisions.

Every module imports from here. Nothing in this project defines a path, a seed, a
split boundary or a threshold locally.

WHY THIS FILE EXISTS
--------------------
Three requirements in the build plan are "decide once, apply identically everywhere"
problems, and each fails silently rather than loudly if a module drifts:

  A1  the leak-free boundary — graph features must be computed only from the training
      window. If one feature module uses a different cutoff than another, results are
      quietly inflated and nothing errors.
  A3  the temporal split — if the boundaries move between training and evaluation,
      every reported metric is invalid, and the numbers still look plausible.
  A6  the alert aggregation rule — the model scores transactions, but triage happens
      per account. Two different aggregations in two modules means the engine and the
      agent are not talking about the same alerts.

Centralising them here does not make them correct; it makes them *consistent*, which
is the part that is otherwise impossible to verify by reading the code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Resolved from this file's location, never from the current working directory, so
# a notebook in notebooks/ and a script run from the repo root agree on where data is.
ROOT: Final[Path] = Path(__file__).resolve().parent.parent

DATA_RAW: Final[Path] = ROOT / "data" / "raw"
DATA_PROCESSED: Final[Path] = ROOT / "data" / "processed"
RESULTS: Final[Path] = ROOT / "results"
FIGURES: Final[Path] = RESULTS / "figures"
DOCS: Final[Path] = ROOT / "docs"

# Raw dataset files. We deliberately download only these two of the dataset's twelve
# files — the others are the Medium/Large variants we are not using and are many GB.
TRANS_CSV: Final[Path] = DATA_RAW / "HI-Small_Trans.csv"
PATTERNS_TXT: Final[Path] = DATA_RAW / "HI-Small_Patterns.txt"

# Parquet cache. CSV parsing of ~5M rows takes tens of seconds every single run;
# parquet reloads in about a second and preserves dtypes (notably the categoricals),
# which CSV cannot. Written once on Day 1, read by everything downstream.
TRANS_PARQUET: Final[Path] = DATA_PROCESSED / "transactions.parquet"
PATTERNS_PARQUET: Final[Path] = DATA_PROCESSED / "patterns.parquet"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
# One seed, passed explicitly to every stochastic component: LightGBM's bagging,
# the k-sample approximation in betweenness centrality, node2vec's random walks,
# and the bootstrap resampling for confidence intervals. Set in one place so that
# "reproducible" is a property of the repo rather than a claim in the README.
RANDOM_SEED: Final[int] = 42

# ---------------------------------------------------------------------------
# Kaggle dataset identifier
# ---------------------------------------------------------------------------
KAGGLE_DATASET: Final[str] = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"

# ---------------------------------------------------------------------------
# Temporal split — SET ON DAY 1 from the measured time span
# ---------------------------------------------------------------------------
# Deliberately left unset. The correct boundaries depend on the dataset's actual time
# span, which is one of Day 1's three go/no-go checks. Hardcoding a guess now and
# discovering later that the data spans days rather than months is exactly the kind of
# late, expensive failure this file is meant to prevent.
#
# The split is TEMPORAL, never random: train on the earliest window, validate on the
# middle, test on the latest. A random split would let the model learn from
# transactions that occur after the ones it is scoring, and — more subtly — would let
# graph features encode edges that had not happened yet at prediction time.
#
# Fractions of the time range, applied on Day 1 once the span is known.
TRAIN_FRACTION: Final[float] = 0.60
VAL_FRACTION: Final[float] = 0.20
TEST_FRACTION: Final[float] = 0.20

# Populated on Day 1 by src/make_splits.py and persisted, so every later run uses
# byte-identical boundaries rather than recomputing them from possibly-changed data.
SPLIT_BOUNDARIES_JSON: Final[Path] = DATA_PROCESSED / "split_boundaries.json"

# ---------------------------------------------------------------------------
# Currency normalisation — POPULATED ON DAY 1
# ---------------------------------------------------------------------------
# The dataset records Amount Paid and Amount Received in potentially DIFFERENT
# currencies. Feeding raw amounts to the model compares 500 JPY with 500 GBP as
# equal, so both are converted to USD before any amount feature is computed.
#
# These are STATIC rates, not historical ones. That is a stated simplification: the
# dataset is synthetic and has no real FX time series behind it. The README says so
# explicitly rather than implying a precision we do not have.
#
# Filled on Day 1 once we have seen the actual set of currencies in the data.
FX_TO_USD: dict[str, float] = {}

# ---------------------------------------------------------------------------
# Alert construction (A6) — the seam between the engine and the agent
# ---------------------------------------------------------------------------
# The model scores TRANSACTIONS, because that is the unit `Is Laundering` labels.
# Triage happens per ACCOUNT, because that is what an investigator opens a case on.
# This constant defines the one conversion between them.
#
# "max" — an account's alert score is the highest score among its transactions in the
# window. Chosen over mean because laundering is a minority of even a mule account's
# activity; averaging dilutes exactly the signal we are looking for.
ALERT_AGGREGATION: Final[str] = "max"

# How many top-scoring accounts form the agent's alert set (A8). Deliberately includes
# both true and false positives so the agent's ability to DISMISS is measurable —
# that, not detection, is the business case for a triage layer.
ALERT_SET_SIZE: Final[int] = 200

# Analyst-capacity operating points (B6). Real teams tune against how many alerts they
# can review per day, so precision@k at realistic k is reported alongside PR-AUC.
PRECISION_AT_K: Final[tuple[int, ...]] = (50, 100, 500)

# ---------------------------------------------------------------------------
# Feature arms (A2) — the ablation that makes the headline lift honest
# ---------------------------------------------------------------------------
# The graph lift is reported against arm B, not arm A. Arm A exists only to show what
# a naive baseline looks like; arm B is the real control, because it already contains
# every non-graph account aggregate. Without it, most of the apparent "graph lift" is
# just the effect of aggregating per account at all.
ARMS: Final[dict[str, str]] = {
    "A": "transaction attributes only",
    "B": "A + account aggregates + typology features (no graph topology)",
    "C": "B + graph topology features",
    "D": "C + node2vec embeddings",
}

# ---------------------------------------------------------------------------
# node2vec (Day 4)
# ---------------------------------------------------------------------------
N2V_DIM: Final[int] = 64
N2V_WALK_LENGTH: Final[int] = 20
N2V_WINDOW: Final[int] = 5
N2V_EPOCHS: Final[int] = 5
# p and q bias the random walk. q > 1 keeps walks close to their origin, which encodes
# STRUCTURAL ROLE (is this node a hub, a pass-through, a leaf?). q < 1 explores outward
# and encodes community membership instead. Laundering detection wants structural role
# — a mule account is defined by its shape of connectivity — so q starts above 1.
N2V_P: Final[float] = 1.0
N2V_Q: Final[float] = 2.0

# ---------------------------------------------------------------------------
# Graph features (Day 3)
# ---------------------------------------------------------------------------
# Exact betweenness centrality is O(V*E) and will not finish on a graph this size.
# networkx supports a sampled approximation using k pivot nodes; k=500 is the
# accuracy/runtime trade-off, and the seed makes it reproducible.
BETWEENNESS_K: Final[int] = 500


# ---------------------------------------------------------------------------
# Directory bootstrap
# ---------------------------------------------------------------------------
# data/ is gitignored, so on a fresh clone these directories do not exist. Creating
# them at import time — rather than expecting the user to mkdir by hand — means a
# clone-and-run works immediately, and no pipeline stage dies halfway through on a
# missing output directory. exist_ok makes it a no-op on every subsequent run.
for _directory in (DATA_RAW, DATA_PROCESSED, RESULTS, FIGURES, DOCS):
    _directory.mkdir(parents=True, exist_ok=True)


def require_fx_rates() -> dict[str, float]:
    """Return the FX table, failing loudly if Day 1 has not populated it.

    A missing conversion rate would otherwise silently produce NaN amounts, which
    LightGBM accepts without complaint — turning a setup error into a quietly wrong
    model. Better to stop here with a message that names the fix.
    """
    if not FX_TO_USD:
        raise RuntimeError(
            "FX_TO_USD is empty. Populate it in src/config.py from the currency set "
            "found during Day 1 EDA before computing any amount feature."
        )
    return FX_TO_USD


if __name__ == "__main__":
    # `python src/config.py` prints the resolved configuration — a fast way to confirm
    # paths resolve correctly from wherever you are running.
    print(f"ROOT            {ROOT}")
    print(f"DATA_RAW        {DATA_RAW}")
    print(f"DATA_PROCESSED  {DATA_PROCESSED}")
    print(f"RESULTS         {RESULTS}")
    print(f"RANDOM_SEED     {RANDOM_SEED}")
    print(f"SPLIT           train={TRAIN_FRACTION} val={VAL_FRACTION} test={TEST_FRACTION}")
    print(f"ALERT RULE      {ALERT_AGGREGATION}, top {ALERT_SET_SIZE} accounts")
    print(f"FX TABLE        {'populated' if FX_TO_USD else 'EMPTY — set on Day 1'}")
