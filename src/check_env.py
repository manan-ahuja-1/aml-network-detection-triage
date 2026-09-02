"""Environment smoke test — run via `make check`.

Why this file exists: `pip install` succeeding only proves that files were
downloaded. Packages with compiled extensions (numba, lightgbm, chromadb, shap)
can install cleanly and then fail at import time on an interpreter they were not
built for. This script forces every dependency to actually load, so a version
problem surfaces here in five seconds rather than on Day 4 after three days of work.

Exit code 0 means the environment is fit to build on. Non-zero means stop and fix.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import sys
from pathlib import Path

# Repo root is one level up from src/. Resolved from __file__ rather than the
# working directory so the check gives the same answer wherever it is invoked from.
ROOT = Path(__file__).resolve().parent.parent

# The Python version is not a preference here, it is a hard constraint.
# gensim ships no cp314 wheel, and pecanpy trains node2vec embeddings through
# gensim's Word2Vec — so on 3.14 the headline result (arm D) cannot be produced.
REQUIRED_PY = (3, 13)

# (import name, human label). Import name differs from the pip name often enough
# that hardcoding the import name is the only reliable check — e.g. pip installs
# "scikit-learn" but you import "sklearn".
PACKAGES = [
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("pyarrow", "pyarrow"),
    ("sklearn", "scikit-learn"),
    ("lightgbm", "lightgbm"),
    ("shap", "shap"),
    ("networkx", "networkx"),
    ("fastnode2vec", "fastnode2vec"),
    ("gensim", "gensim"),
    ("numba", "numba"),
    ("anthropic", "anthropic"),
    ("chromadb", "chromadb"),
    ("streamlit", "streamlit"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("dotenv", "python-dotenv"),
    ("kaggle", "kaggle"),
    ("tqdm", "tqdm"),
    ("pytest", "pytest"),
]

# Directories the pipeline writes into. Created in Phase 0; checked here because a
# missing output directory is a confusing mid-run failure rather than an obvious one.
REQUIRED_DIRS = [
    "data/raw",
    "data/processed",
    "src",
    "results/figures",
    "tests",
    "docs",
]


def _fail(msg: str) -> None:
    print(f"  FAIL  {msg}")


def _ok(msg: str) -> None:
    print(f"  ok    {msg}")


def check_python() -> bool:
    actual = sys.version_info[:2]
    if actual != REQUIRED_PY:
        _fail(
            f"Python {actual[0]}.{actual[1]} — this project requires "
            f"{REQUIRED_PY[0]}.{REQUIRED_PY[1]}. Rebuild the venv with the 3.13 "
            f"interpreter (see requirements.in for why)."
        )
        return False
    _ok(f"Python {sys.version.split()[0]}")
    return True


def check_imports() -> bool:
    all_ok = True
    for module_name, pip_name in PACKAGES:
        try:
            # Some packages print on import — `kaggle` dumps a full authentication
            # guide when no credentials are set. That is not an error, but it buries
            # real failures, so import chatter is swallowed and only our own
            # pass/fail lines are shown.
            with io.StringIO() as sink, contextlib.redirect_stdout(sink):
                mod = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 — we want to report any failure kind
            # Deliberately broad: a bad native extension can raise almost anything,
            # and reporting the real exception type is what makes it debuggable.
            _fail(f"{pip_name}: {type(exc).__name__}: {exc}")
            all_ok = False
            continue
        version = getattr(mod, "__version__", "(no __version__ attribute)")
        _ok(f"{pip_name} {version}")
    return all_ok


def check_capabilities() -> bool:
    """Verify the specific APIs this project depends on actually exist.

    These are not generic import checks — each one stands in for a design decision
    made in the build plan, so if the API is missing we need to know now.
    """
    all_ok = True

    # We chose NOT to add python-louvain as a dependency, because networkx >= 3.0
    # ships Louvain community detection itself. Confirm that is true here rather
    # than discovering it on Day 3.
    try:
        import networkx as nx

        assert callable(nx.community.louvain_communities)
        _ok("networkx.community.louvain_communities available (no python-louvain needed)")
    except Exception as exc:  # noqa: BLE001
        _fail(f"networkx Louvain unavailable: {exc} — add python-louvain to requirements.in")
        all_ok = False

    # B3 in the build plan feeds per-alert SHAP attributions to the triage agent.
    # LightGBM computes exact TreeSHAP natively via pred_contrib, which means that
    # requirement does not hard-depend on the `shap` package. Confirm the flag exists.
    try:
        import inspect

        import lightgbm as lgb

        sig = inspect.signature(lgb.Booster.predict)
        assert "pred_contrib" in sig.parameters
        _ok("lightgbm Booster.predict supports pred_contrib (native TreeSHAP)")
    except Exception as exc:  # noqa: BLE001
        _fail(f"lightgbm pred_contrib check failed: {exc}")
        all_ok = False

    return all_ok


def check_dirs() -> bool:
    all_ok = True
    for rel in REQUIRED_DIRS:
        path = ROOT / rel
        if path.is_dir():
            _ok(f"{rel}/")
        else:
            _fail(f"missing directory: {rel}/")
            all_ok = False
    return all_ok


def main() -> int:
    print(f"\nEnvironment check — repo root: {ROOT}\n")

    print("Interpreter")
    py_ok = check_python()

    print("\nDependencies")
    imports_ok = check_imports()

    print("\nRequired capabilities")
    caps_ok = check_capabilities()

    print("\nDirectories")
    dirs_ok = check_dirs()

    passed = all([py_ok, imports_ok, caps_ok, dirs_ok])
    print("\n" + ("PASS — environment is ready." if passed else "FAILED — see above."))
    # Exit code is what lets `make check` gate the rest of the pipeline.
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
