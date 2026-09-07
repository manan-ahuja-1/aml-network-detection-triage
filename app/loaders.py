"""Read-only access to the committed artifacts the demo runs on.

The whole app depends on one property: it must not need `data/` or `models/`, which are
gitignored and absent on any deployment host. Every path below points at something the
repo actually carries, and each loader returns None rather than raising when its file is
missing — so a partially built repo renders a degraded page instead of a stack trace.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

RESULTS = ROOT / "results"


@st.cache_data(show_spinner=False)
def load_json(name: str):
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_parquet(name: str):
    path = RESULTS / name
    return pd.read_parquet(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def figure(name: str) -> Path | None:
    path = RESULTS / "figures" / name
    return path if path.exists() else None


@st.cache_resource(show_spinner=False)
def renderer():
    """`triage.render_case`, imported lazily.

    The prompt shown in the demo is rendered live from the exported dossier rather than
    exported alongside it. That is deliberate: a stored prompt can drift from the code
    that produced it, and then the demo is showing a screenshot of something that no
    longer happens. Rendering it here means what you read is what the agent reads.
    """
    from agent import dossier as dossier_mod
    from agent import triage
    return triage, dossier_mod
