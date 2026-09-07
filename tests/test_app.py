"""The demo must render, and must render off committed artifacts alone.

WHY THIS IS A TEST AND NOT A MANUAL CHECK
-----------------------------------------
Two failure modes here are silent, which is exactly why they need asserting.

A Streamlit table is converted through Arrow before it is sent to the browser. A column
mixing types — SHAP feature values holding "ACH" beside 12.4 — makes that conversion
fail, and the failure is *logged*, not raised: the app renders fine with one table simply
absent. Both such bugs in this app were found by executing the script rather than by
looking at it.

And the deployed app has no `data/` and no `models/`, so any import that reaches for them
works locally and fails only in production. `test_app_needs_no_pipeline_imports` pins the
lazy-import arrangement the slim `app/requirements.txt` depends on.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import config  # noqa: E402

streamlit = pytest.importorskip("streamlit", reason="streamlit not installed")

needs_results = pytest.mark.skipif(
    not (config.RESULTS / "engine.json").exists(),
    reason="results not generated yet")


@needs_results
def test_the_demo_renders_without_exceptions():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"),
                            default_timeout=300).run()
    assert not app.exception, [e.value for e in app.exception]
    assert len(app.tabs) == 5
    assert app.metric, "no metrics rendered — the results files may be empty"


@needs_results
def test_every_table_survives_arrow_conversion():
    """A mixed-type column does not raise; the table just vanishes. So the check is that
    the tables are actually there, not merely that nothing threw."""
    import pyarrow as pa
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"),
                            default_timeout=300).run()
    assert len(app.dataframe) >= 4, "fewer tables than expected — one may have been lost"
    for element in app.dataframe:
        pa.Table.from_pandas(element.value)     # raises exactly as Streamlit's would


def test_app_needs_no_pipeline_imports():
    """The deploy manifest pins only streamlit, pandas, numpy and pyarrow. That is only
    safe while the modules the app touches keep their heavy imports inside functions —
    if one moves to module scope this fails here rather than at deploy time."""
    code = textwrap.dedent("""
        import sys
        sys.path.insert(0, "src")
        from agent import triage, dossier            # noqa: F401
        heavy = [m for m in ("lightgbm", "chromadb", "anthropic", "shap", "sklearn",
                             "networkx", "torch") if m in sys.modules]
        print(",".join(heavy))
    """)
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                            capture_output=True, text=True, check=True)
    pulled = result.stdout.strip()
    assert not pulled, f"the app's import chain now pulls in: {pulled}"


def test_deploy_manifest_matches_the_pinned_versions():
    """A demo running a different pandas from the pipeline that wrote its parquet is a
    debugging session nobody needs."""
    app_reqs = (ROOT / "app" / "requirements.txt").read_text().splitlines()
    project = (ROOT / "requirements.txt").read_text()
    pins = [line.strip() for line in app_reqs
            if line.strip() and not line.startswith("#")]
    assert pins, "deploy manifest is empty"
    for pin in pins:
        assert pin in project, f"{pin} disagrees with requirements.txt"
