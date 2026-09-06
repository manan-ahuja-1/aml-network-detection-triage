"""Day 6 — the triage agent's evidence boundary and output contract.

The theme of these tests is the same one running through the rest of the suite: the
failures worth catching are the ones that produce a plausible result. An evidence cap
that silently drops the citation whitelist, a hallucination check that validates against
the wrong set, a self-transfer counted as a fan-out spoke — none of these raise, and all
of them change what the agent's reported numbers mean.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import config  # noqa: E402
from agent import dossier as dossier_mod  # noqa: E402
from agent import triage  # noqa: E402


def make_evidence(n: int, node: str = "1:A", self_transfers: int = 0) -> pd.DataFrame:
    """A minimal evidence frame shaped like alerts.alert_evidence() output."""
    rows = []
    for i in range(n):
        is_self = i < self_transfers
        rows.append({
            "txn_id": f"T{i:07d}",
            "timestamp": pd.Timestamp("2022-09-07") + pd.Timedelta(hours=i),
            "from_id": node,
            "to_id": node if is_self else f"2:B{i}",
            "direction": "OUT",
            "paid_usd": 1000.0 + i,
            "recv_usd": 1000.0 + i,
            "payment_currency": "US Dollar",
            "receiving_currency": "US Dollar",
            "payment_format": "ACH",
            "model_score": i / max(n - 1, 1),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The evidence boundary
# ---------------------------------------------------------------------------
def test_evidence_is_capped_and_the_remainder_is_declared():
    """Over the cap, the dossier must show a slice AND say that it did.

    Presenting 40 of 39,155 transactions without saying so would let the agent describe
    a busy corporate account as a small one, in good faith.
    """
    evidence = make_evidence(100)
    rows, omitted = dossier_mod.evidence_table(evidence, limit=40)
    assert len(rows) == 40
    assert omitted["count"] == 60
    assert "MUST NOT be cited" in omitted["note"]


def test_uncapped_evidence_declares_no_remainder():
    rows, omitted = dossier_mod.evidence_table(make_evidence(10), limit=40)
    assert len(rows) == 10
    assert omitted == {}


def test_the_cap_keeps_the_highest_scoring_transactions():
    """If the cap dropped the score-driving rows, the agent would be asked to explain an
    alert using the evidence least responsible for it."""
    rows, _ = dossier_mod.evidence_table(make_evidence(100), limit=10)
    kept = {r["txn_id"] for r in rows}
    expected = {f"T{i:07d}" for i in range(90, 100)}
    assert kept == expected


def test_citable_ids_are_exactly_the_shown_rows():
    """The hallucination check's whitelist must track the cap, not the full evidence.

    If citable_ids() were built from everything that exists, an agent citing a real but
    unshown transaction would pass — and the measured hallucination rate would be
    understating by exactly the rows the cap removed.
    """
    rows, _ = dossier_mod.evidence_table(make_evidence(100), limit=25)
    dossier = {"evidence": rows}
    assert dossier_mod.citable_ids(dossier) == {r["txn_id"] for r in rows}
    assert len(dossier_mod.citable_ids(dossier)) == 25


def test_self_transfers_are_labelled_and_excluded_from_counterparty_counts():
    """A transfer to yourself is not a fan-out spoke.

    11.6% of this dataset is self-transfers. Counted as external legs they would inflate
    apparent dispersal and push the agent toward FAN-OUT on accounts that did nothing of
    the kind.
    """
    evidence = make_evidence(10, self_transfers=4)
    profile = dossier_mod.account_profile(evidence, "1:A")
    assert profile["self_transfers"] == 4
    assert profile["outgoing"] == 6
    assert profile["distinct_counterparties_out"] == 6

    rows, _ = dossier_mod.evidence_table(evidence, limit=40)
    self_rows = [r for r in rows if r["direction"] == "SELF"]
    assert len(self_rows) == 4
    assert all("self transfer" in r["counterparty"] for r in self_rows)


# ---------------------------------------------------------------------------
# Validation (A9)
# ---------------------------------------------------------------------------
def _dossier_with(ids: list[str]) -> dict:
    return {"evidence": [{"txn_id": i} for i in ids],
            "knowledge_base": [{"source": "ffiec-appendix-f"}]}


def test_validate_flags_a_fabricated_citation():
    dossier = _dossier_with(["T0000001", "T0000002"])
    result = {"cited_transaction_ids": ["T0000001", "T9999999"],
              "sources_cited": [], "pattern_classification": "FAN-OUT",
              "disposition": "escalate"}
    v = triage.validate(result, dossier)
    assert v["hallucinated_citation"] is True
    assert v["invalid_citations"] == ["T9999999"]


def test_validate_passes_clean_output():
    dossier = _dossier_with(["T0000001", "T0000002"])
    result = {"cited_transaction_ids": ["T0000001"],
              "sources_cited": ["ffiec-appendix-f"],
              "pattern_classification": "FAN-IN", "disposition": "close"}
    v = triage.validate(result, dossier)
    assert v["hallucinated_citation"] is False
    assert v["invalid_sources"] == []
    assert v["typology_valid"] and v["disposition_valid"]


def test_validate_flags_a_source_that_was_not_retrieved():
    """Citing a real regulator the agent was not shown is the same class of error."""
    dossier = _dossier_with(["T0000001"])
    result = {"cited_transaction_ids": ["T0000001"],
              "sources_cited": ["fatf-va-red-flags-2020"],
              "pattern_classification": "NONE", "disposition": "close"}
    assert triage.validate(result, dossier)["invalid_sources"] == ["fatf-va-red-flags-2020"]


# ---------------------------------------------------------------------------
# The output contract
# ---------------------------------------------------------------------------
def test_schema_enum_matches_the_configured_typologies():
    """The agent's label set and the scoring label set must be the same object.

    They are compared against Patterns.txt on Day 8. A label the schema permits but the
    scorer does not know would show up as a wrong answer rather than as a bug.
    """
    schema = triage.TRIAGE_SCHEMA["properties"]["pattern_classification"]["enum"]
    assert schema == list(config.TYPOLOGIES)
    assert "NONE" in schema


def test_schema_requires_every_field_the_evaluation_reads():
    required = set(triage.TRIAGE_SCHEMA["required"])
    assert {"disposition", "pattern_classification", "confidence", "risk_rationale",
            "cited_transaction_ids", "red_flag_indicators", "sources_cited"} <= required


def test_agent_can_close_as_well_as_escalate():
    """The business case for triage is removing false positives, so `close` must exist.

    A schema that only permitted `escalate` would make the headline agent metric
    unmeasurable while still producing output that looked fine.
    """
    assert set(triage.TRIAGE_SCHEMA["properties"]["disposition"]["enum"]) == {
        "escalate", "close"}


# ---------------------------------------------------------------------------
# Retrieval query construction
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("in_n,out_n,expected", [
    (0, 6, "fan out"),
    (6, 0, "fan in"),
    (5, 5, "gather scatter"),
    (1, 1, "pass-through"),
])
def test_shape_query_describes_the_right_topology(in_n, out_n, expected):
    """Retrieval is phrasing-sensitive: the first version of this query appended SHAP
    feature names and stopped returning FAN-OUT for a pure fan-out account entirely."""
    profile = {"distinct_counterparties_in": in_n, "distinct_counterparties_out": out_n}
    assert expected in dossier_mod.shape_query(profile)


def test_queries_contain_no_raw_feature_names():
    """Schema vocabulary and regulatory prose do not share an embedding neighbourhood.

    Measured: mixing them dropped FAN-OUT from rank 1 (similarity 0.634) to absent from
    the top 5 for a pure fan-out account.
    """
    profile = {"distinct_counterparties_in": 0, "distinct_counterparties_out": 5,
               "flow_through": 0.9, "window_hours": 30.0, "currencies": ["a", "b", "c"],
               "distinct_banks": 4, "self_transfers": 0, "payment_formats": ["ACH"]}
    combined = dossier_mod.shape_query(profile) + dossier_mod.behaviour_query(profile)
    for token in ("from_out_", "to_in_", "pagerank", "n_currencies", "_ratio"):
        assert token not in combined


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------
def test_every_citable_id_appears_in_the_rendered_prompt():
    """The whitelist and the prompt must not drift apart.

    An ID in citable_ids() that never reaches the prompt would be a citation the agent
    is permitted to make but has no way to know about; the reverse would let a
    fabrication validate.
    """
    rows, omitted = dossier_mod.evidence_table(make_evidence(12), limit=40)
    dossier = {
        "alert": {"account": "1:A", "queue_rank": 1, "model_score": 0.99,
                  "scored_window": "val"},
        "account_profile": dossier_mod.account_profile(make_evidence(12), "1:A"),
        "evidence": rows, "evidence_omitted": omitted,
        "model_reasons": [], "knowledge_base": [],
    }
    prompt = triage.render(dossier)
    for txn_id in dossier_mod.citable_ids(dossier):
        assert txn_id in prompt


def test_system_prompt_forbids_claiming_a_sar_decision():
    """L1 triage does not decide filings. Saying so is a domain-correctness requirement,
    not a style preference."""
    assert "SAR" in triage.SYSTEM_PROMPT
    assert "do NOT decide" in triage.SYSTEM_PROMPT or "You do NOT" in triage.SYSTEM_PROMPT
