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
import splits  # noqa: E402
from agent import dossier as dossier_mod  # noqa: E402
from agent import triage  # noqa: E402

needs_data = pytest.mark.skipif(
    not config.TRANS_PARQUET.exists(), reason="run `make data` first")


@pytest.fixture(scope="module")
def frame():
    return splits.load_transactions()


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


# ---------------------------------------------------------------------------
# Cases — the unit the agent actually triages
# ---------------------------------------------------------------------------
from agent import budget  # noqa: E402
from agent import cases as cases_mod  # noqa: E402


@pytest.fixture(scope="module")
def val_cases(frame):
    table = pd.read_parquet(config.RESULTS / "alerts_val.parquet")
    return cases_mod.build_cases(frame, table, "val"), table


@needs_data
def test_cases_partition_the_alert_queue(val_cases):
    """Every alerted account in exactly one case — none lost, none duplicated.

    An account that falls out of every case is never reviewed, and the queue silently
    shrinks while every reported rate still looks fine because it is computed over the
    cases that do exist.
    """
    cases, table = val_cases
    members = [m for c in cases for m in c.members]
    assert len(members) == len(set(members)), "an account appears in two cases"
    assert set(members) == set(table.index), "an alerted account is in no case"


@needs_data
def test_no_case_becomes_a_hairball(val_cases):
    """The bridge cap is load-bearing, not decorative.

    Allowing counterparty-to-counterparty edges collapsed the val queue into a single
    component of 15,027 accounts — one 'case' containing everything, which is no
    grouping at all and would not fit in a prompt either.
    """
    cases, _ = val_cases
    biggest = max(c.n_members for c in cases)
    assert biggest <= config.CASE_MAX_ACCOUNTS, (
        f"largest case holds {biggest} accounts, over the {config.CASE_MAX_ACCOUNTS} cap")


@needs_data
def test_grouping_actually_reduces_the_queue(val_cases):
    """The pivot has to buy something: fewer, richer units of review."""
    cases, table = val_cases
    assert len(cases) < len(table) / 2, (
        f"{len(cases)} cases from {len(table)} accounts is barely a reduction")


@needs_data
def test_case_is_productive_if_any_member_is(val_cases):
    cases, table = val_cases
    for case in cases[:20]:
        expected = bool(table.loc[case.members, "is_productive"].any())
        assert case.is_productive is expected


@needs_data
def test_case_evidence_deduplicates_internal_transactions(val_cases, frame):
    """A transaction between two members must appear once, not once per member.

    Counted twice it inflates both the apparent volume and the citation whitelist, and
    the second copy is a transaction ID the agent can cite for a payment that happened
    only once.
    """
    cases, _ = val_cases
    multi = next(c for c in cases if c.n_members > 2 and c.evidence is not None)
    assert not multi.evidence.index.duplicated().any()


@needs_data
def test_case_directions_are_relative_to_the_case(val_cases, frame):
    """IN/OUT must describe the case boundary, not one member's point of view.

    `alert_evidence` writes `direction` for whichever account it was called with, and a
    case pools rows across members — so a shared row arrives carrying an arbitrary
    member's perspective. Unfixed, this reported an account that only received money as
    having sent it.
    """
    from agent import dossier as d
    cases, _ = val_cases
    case = next(c for c in cases if c.n_members > 2)
    labelled = d.label_case_directions(case.evidence, set(case.members))
    members = set(case.members)
    for _, row in labelled.head(50).iterrows():
        if row["from_id"] == row["to_id"]:
            assert row["direction"] == "SELF"
        elif row["from_id"] in members and row["to_id"] in members:
            assert row["direction"] == "INTERNAL"
        elif row["to_id"] in members:
            assert row["direction"] == "IN"
        else:
            assert row["direction"] == "OUT"


@needs_data
def test_account_profile_direction_ignores_the_stored_column(val_cases, frame):
    """account_profile must derive direction from the account it describes.

    Reading the shared `direction` column is exactly the bug above, one layer down.
    """
    from agent import dossier as d
    cases, _ = val_cases
    case = next(c for c in cases if c.n_members > 2)
    node = case.members[0]
    evidence = case.evidence.copy()
    evidence["direction"] = "OUT"          # deliberately wrong for every row
    profile = d.account_profile(evidence, node)
    external = evidence[evidence["from_id"] != evidence["to_id"]]
    assert profile["outgoing"] == int((external["from_id"] == node).sum())
    assert profile["incoming"] == int((external["to_id"] == node).sum())


@needs_data
def test_case_citable_ids_match_the_rendered_prompt(val_cases, frame):
    """The whitelist and the prompt must not drift apart."""
    from agent import dossier as d
    cases, table = val_cases
    shap = d.load_shap("val")
    case = next(c for c in cases if c.n_members > 2)
    dossier = d.build_case(frame, case, table, shap, retrieve=False)
    prompt = triage.render_case(dossier)
    for txn_id in d.citable_ids(dossier):
        assert txn_id in prompt


# ---------------------------------------------------------------------------
# The budget guard
# ---------------------------------------------------------------------------
def test_preflight_refuses_a_run_that_would_breach_the_ceiling(monkeypatch, tmp_path):
    """The ceiling must stop a batch BEFORE it spends, not partway through.

    A 200-alert run once died at alert 72 with credits exhausted, having billed for work
    that was then partly discarded. Nothing had priced the batch first.
    """
    monkeypatch.setattr(config, "SPEND_LEDGER", tmp_path / "ledger.json")
    monkeypatch.setattr(config, "BUDGET_CEILING_USD", 1.00)
    with pytest.raises(budget.BudgetExceeded, match="refusing to start"):
        budget.preflight("test", per_call_usd=0.05, n_calls=100)


def test_preflight_allows_a_run_within_budget(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "SPEND_LEDGER", tmp_path / "ledger.json")
    monkeypatch.setattr(config, "BUDGET_CEILING_USD", 10.00)
    assert budget.preflight("test", 0.01, 45)["approved"] is True


def test_preflight_requires_the_override_to_name_the_amount(monkeypatch, tmp_path):
    """Going over budget has to be a typed, specific act — not a boolean flag."""
    monkeypatch.setattr(config, "SPEND_LEDGER", tmp_path / "ledger.json")
    monkeypatch.setattr(config, "BUDGET_CEILING_USD", 1.00)
    with pytest.raises(budget.BudgetExceeded):
        budget.preflight("test", 0.05, 100, confirm_spend=1.00)   # too small
    assert budget.preflight("test", 0.05, 100, confirm_spend=99.0)["approved"]


def test_ledger_counts_billed_calls_only(monkeypatch, tmp_path):
    """Cached calls cost nothing. Recording them would inflate the running total and
    make the guard refuse work that is in fact free."""
    monkeypatch.setattr(config, "SPEND_LEDGER", tmp_path / "ledger.json")
    budget.record("t", "m", calls=10, input_tokens=100, output_tokens=50, cost_usd=0.25)
    budget.record("t", "m", calls=0, input_tokens=0, output_tokens=0, cost_usd=0.0)
    assert budget.total_spent() == 0.25
    assert len(budget.load_ledger()) == 1


def test_pricing_is_known_for_the_configured_model():
    """An unknown model must fail loudly rather than report a guessed cost.

    Two constants holding Sonnet 4.5's rates while calling Sonnet 5 made every cost
    figure ~33% too high for a full day.
    """
    price_in, price_out = config.model_pricing(config.ANTHROPIC_MODEL)
    assert price_in > 0 and price_out > price_in
    with pytest.raises(KeyError, match="no published price"):
        config.model_pricing("claude-imaginary-9")
