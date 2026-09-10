"""AML laundering-network detection and triage — the walkthrough.

WHAT THIS DEMO IS FOR
---------------------
Showing the system honestly, which includes the part that does not work. The triage
agent's disposition carries no measurable signal (Fisher p=0.51 on held-out data) and
that sits on the evaluation tab at the same size as everything else, because a demo that
hides its negative result is an advert.

Runs entirely off committed artifacts — no raw data, no booster, no API key, no network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from loaders import figure, load_json, load_parquet, renderer  # noqa: E402

st.set_page_config(page_title="AML Network Detection & Case Triage", page_icon="🕸️",
                   layout="wide", initial_sidebar_state="collapsed")

ENGINE = load_json("engine.json")
ARMS = load_json("arms.json")
AGENT = load_json("agent.json")
BUNDLE = load_json("demo_bundle.json")
SINGLE_BANK = load_json("single_bank.json")
FX = load_json("fx_diagnosis.json")
TRIAGE = load_json("triage_cases_test.json")

st.markdown("""
<style>
  .stTabs [data-baseweb="tab-list"] { gap: 2px; }
  .stTabs [data-baseweb="tab"] { padding: 10px 18px; }
  code { font-size: 0.86em; }
  .finding { border-left: 3px solid #b0413e; padding: 0.6em 1em; background: #b0413e11;
             border-radius: 0 4px 4px 0; margin: 0.5em 0; }
  .good    { border-left: 3px solid #2f7a4f; padding: 0.6em 1em; background: #2f7a4f11;
             border-radius: 0 4px 4px 0; margin: 0.5em 0; }
</style>
""", unsafe_allow_html=True)


def finding(text: str) -> None:
    st.markdown(f'<div class="finding">{text}</div>', unsafe_allow_html=True)


def good(text: str) -> None:
    st.markdown(f'<div class="good">{text}</div>', unsafe_allow_html=True)


def safe_df(rows) -> pd.DataFrame:
    """Build a DataFrame Arrow can actually serialise.

    Several of these tables mix types inside one column — SHAP feature values hold "ACH"
    beside 12.4, the ablation table holds "7/12" beside 2.736. Streamlit converts through
    Arrow, which rejects a mixed object column, and the failure surfaces as a logged
    traceback with no exception at the app level: the table simply does not appear. So
    any column that is not cleanly numeric is rendered as text.
    """
    df = pd.DataFrame(rows)
    for col in df.columns:
        if df[col].dtype == "object":
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.isna().any():
                df[col] = df[col].astype(str)
            else:
                df[col] = coerced
    return df


def missing(what: str) -> None:
    st.info(f"`{what}` is not in this checkout — run the pipeline to generate it.")


st.title("AML Network Detection & Case Triage")
st.caption("Transaction monitoring → **alert** → **L1 triage** → L2 investigation → "
           "SAR decision.  The engine is the monitoring layer; the agent is L1 triage.")

tab_problem, tab_engine, tab_queue, tab_case, tab_eval = st.tabs([
    "The problem", "The engine", "Alert queue", "Case triage", "What works, what doesn't",
])


# ---------------------------------------------------------------------------
with tab_problem:
    st.header("The constraint is analyst capacity, not detection power")
    c1, c2, c3 = st.columns(3)
    c1.metric("Alerts reviewed, 19 US banks (2018)", "16 million")
    c2.metric("SARs actually filed", "640,000")
    c3.metric("Alerts that convert", "4%", delta="-96%", delta_color="inverse")
    st.caption("Bank Policy Institute, *Getting to Effectiveness* (2018), a survey of 19 "
               "US banks. The 4% is derived from their reported totals rather than taken "
               "from a vendor's rounded '90–95%'.")

    st.markdown("""
Ninety-six percent of the work a monitoring system creates leads nowhere, and every
item still costs an analyst an hour. That is the entire justification for this project,
and it sets what "better" means: **precision at a fixed review capacity**, not raw
detection power. A model that finds more laundering by alerting on twice as much is not
useful to a team that can already only work half its queue.

Two halves, evaluated against a control each:

- **The engine** ranks transactions. Measured against a deliberately strong
  account-aggregate baseline, so that the graph features have to earn their lift rather
  than inherit it from aggregation.
- **The agent** reads an alert and writes the case note an investigator opens first.
  Measured against escalate-everything, which is what "no triage layer" scores.
""")
    if ENGINE:
        v = ENGINE["splits"]["val"]
        st.markdown(f"""
**The data.** IBM *Transactions for Anti-Money Laundering* (Altman et al., NeurIPS 2023),
HI-Small. {v['n']:,} validation transactions at a base rate of
**1 in {round(1 / v['base_rate']):,}** — which is why every headline number here is PR-AUC
and not ROC-AUC. `Patterns.txt` ships ground-truth typology labels for eight injected
laundering shapes, so pattern classification is scored as a real labelled task.
""")


# ---------------------------------------------------------------------------
with tab_engine:
    st.header("Feature ablation — where the lift comes from")
    if not (ARMS and ENGINE):
        missing("arms.json / engine.json")
    else:
        rows = []
        if ARMS.get("arm_R"):
            r = ARMS["arm_R"]
            rows.append({"arm": "R — rules (flag every ACH)", "features": 0,
                         "val PR-AUC": r["pr_auc_degenerate"],
                         "P@100": None, "note": "no model at all"})
        for key, label in (("A", "A — transaction fields only"),
                           ("B", "B — + account aggregates & typology features"),
                           ("C", "C — + multi-hop graph topology"),
                           ("D_seed42", "D — + node2vec embeddings")):
            a = ARMS["arms"].get(key)
            if not a:
                continue
            rows.append({"arm": label, "features": a["n_features"],
                         "val PR-AUC": a["val"]["pr_auc"],
                         "P@100": a["val"]["precision_at_k"]["100"]["precision"],
                         "note": "the engine" if key == "C" else ""})
        df = pd.DataFrame(rows)
        st.dataframe(
            df.style.format({"val PR-AUC": "{:.4f}", "P@100": "{:.1%}"}, na_rep="—"),
            width='stretch', hide_index=True)
        good("**Arm B is the real baseline.** It holds every account aggregate and every "
             "named-typology feature, so arm C's lift is attributable to multi-hop "
             "topology and not to the act of aggregating per account. Arm D added "
             "node2vec and was <b>rejected on evidence</b> — it did not beat explicit "
             "topology across three seeds and two dimensions.")

        st.subheader("Validation → test, and why the drop is not what it looks like")
        val, test = ENGINE["splits"]["val"], ENGINE["splits"]["test"]
        c1, c2, c3 = st.columns(3)
        c1.metric("val PR-AUC", f"{val['pr_auc']:.4f}")
        c2.metric("test PR-AUC", f"{test['pr_auc']:.4f}",
                  delta=f"{test['pr_auc'] - val['pr_auc']:+.4f}")
        c3.metric("test P@100", f"{test['precision_at_k']['100']['precision']:.1%}")
        finding("The obvious reading is validation optimism. It is not. The test split "
                "was scored <b>exactly once</b>, after every choice was frozen — so "
                "there was no opportunity to overfit it. Slicing validation itself into "
                "time windows shows PR-AUC falling <i>inside</i> validation "
                "(0.2523 → 0.1968 → 0.1465): a <b>slope</b>, not a step at the split "
                "boundary. Validation optimism predicts a step. Feature staleness "
                "predicts a slope, and that is what is there — account and graph "
                "features are built from the training window only, and they decay at "
                "roughly half their signal per 1.5 days.")
        fig = figure("temporal_decay.png")
        if fig:
            st.image(str(fig), width='stretch')
        fig = figure("pr_curves.png")
        if fig:
            st.image(str(fig), width='stretch')

    if SINGLE_BANK:
        d = SINGLE_BANK["why_not_a_single_bank"]
        b070 = d["bank_070"]
        rep = (SINGLE_BANK.get("seed_replicates", {})
               .get("by_fraction", {}).get("graph_025pct"))
        spread = (f"{rep['pr_auc_min']:.4f}–{rep['pr_auc_max']:.4f} across "
                  f"{rep['n_seeds']} draws" if rep else "0.1421–0.1904 across 2 draws")
        with st.expander("What a real bank would see — and why this data cannot say"):
            st.markdown(f"""
This dataset hands you the complete inter-bank graph; no institution has that. The plan
was to re-run the engine on one bank's visible subgraph, and **this data cannot support
that experiment.**

There are **{d['n_banks']:,} banks** with a median of **{d['median_accounts_per_bank']}
accounts**, and the only one large enough to test is a clearing entity —
**{b070['accounts']} accounts carrying {b070['transactions']:,} transactions** with zero
internal transfers, on whose slice the engine scores *below chance*.

Degrading the graph by a random fraction instead is inconclusive for a different reason:
at 25% visibility PR-AUC spans **{spread}** depending only on *which* edges are sampled —
several times the graph lift itself. **Which edges you see swamps how many.** The full
account is in `docs/NOTES.md`; the numbers are in `results/single_bank.json`.
""")

    if FX:
        with st.expander("Currency normalisation — the one unsourced number, resolved"):
            st.markdown(f"""
{FX['non_usd_share_pct']}% of transactions are not in US dollars, so amounts have to be
converted before any amount feature means anything. The rates were originally written
from memory. They have now been sourced for {FX['source_date']} from the ECB reference
rates, the Bank of Russia, the SAMA peg and a Bitcoin daily average — worst single error
in the original table: **{FX['worst_rate_error_pct']}%**.

Correcting them was treated as a measurement rather than an edit, because FX feeds every
amount feature *and* the money-weighted graph, so changing it changes the model and would
invalidate a test split that has been scored exactly once.

> {FX['verdict']}
""")
            st.json(FX["val_pr_auc"])


# ---------------------------------------------------------------------------
with tab_queue:
    st.header("The alert queue")
    alerts = load_parquet("alerts_test.parquet")
    shap = load_json("shap_test.json")
    if alerts is None:
        missing("alerts_test.parquet")
    else:
        st.markdown("""
The model scores **transactions**; an investigator works **accounts**. An account's alert
score is the maximum score over its transactions, and every transaction contributes to
both the sender and the receiver — a laundering account is one that participates, whether
it sends or receives.
""")
        show = alerts.head(60).copy()
        show.insert(0, "account", show.index)
        cols = [c for c in ("account", "alert_score", "is_productive", "n_transactions")
                if c in show.columns]
        st.dataframe(show[cols], width='stretch', hide_index=True, height=280)

        if shap:
            picked = st.selectbox("Why did the model fire on this account?",
                                  list(alerts.index[:60]))
            entry = shap.get(picked)
            if entry:
                _, dossier_mod = renderer()
                reasons = dossier_mod.model_reasons(entry)
                st.caption(f"Exact TreeSHAP attributions for the single transaction that "
                           f"drove this account's score (`{entry['driving_txn_id']}`), "
                           f"in log-odds. Score {entry['score']:.4f}.")
                st.dataframe(safe_df(reasons), width='stretch',
                             hide_index=True)
                st.caption("These are attributions, not evidence: they say what moved "
                           "the score, while the transactions say what happened. The "
                           "agent is given both and told to say so when they disagree.")


# ---------------------------------------------------------------------------
with tab_case:
    st.header("Case triage — what the agent is actually handed")
    if not (BUNDLE and TRIAGE):
        missing("demo_bundle.json / triage_cases_test.json")
    else:
        st.markdown("""
Per-**account** triage was built first and measured not to work: an account that receives
one payment and does nothing else is indistinguishable from an ordinary receipt when you
can only see its own rows — even when it is the receiving spoke of a fan-out. The agent
said so itself, and no prompt supplies a fact that is not in the context.

So alerts are grouped into **cases** — connected clusters of alerted accounts — and the
hub and its spokes arrive in one dossier.
""")
        by_id = {r["unit_id"]: r for r in TRIAGE}
        options = sorted(BUNDLE["cases"], key=lambda c: -BUNDLE["cases"][c]["n_members"])
        labels = {c: f"{c} — {BUNDLE['cases'][c]['n_members']} accounts"
                     f"{'  (productive)' if BUNDLE['cases'][c]['is_productive'] else ''}"
                  for c in options}
        case_id = st.selectbox("Case", options, format_func=lambda c: labels[c])
        entry = BUNDLE["cases"][case_id]
        record = by_id.get(case_id)
        triage_mod, dossier_mod = renderer()

        left, right = st.columns([1, 1])
        with left:
            st.subheader("The dossier")
            st.caption("Exactly what the model was shown — nothing added for display.")
            st.code(triage_mod.render_case(entry["dossier"]), language=None)
        with right:
            if not record:
                st.warning("no triage record for this case")
            else:
                res, val = record["result"], record["validation"]
                st.subheader("The case note")
                st.caption("Structured to FinCEN's SAR narrative template — "
                           "introduction, body, conclusion — enforced by the output "
                           "schema rather than requested in prose.")
                note = res.get("case_note") or {}
                for section in ("introduction", "body", "conclusion"):
                    st.markdown(f"**{section.title()}**")
                    st.write(note.get(section, "_(absent)_"))

                m1, m2, m3 = st.columns(3)
                m1.metric("disposition", res["disposition"])
                m2.metric("typology called", res["pattern_classification"])
                m3.metric("ground truth",
                          "productive" if entry["is_productive"] else "non-productive")

                st.subheader("Grounding check, run live")
                recheck = triage_mod.validate(res, entry["dossier"])
                citable = dossier_mod.citable_ids(entry["dossier"])
                if recheck["hallucinated_anywhere"]:
                    finding(
                        "Fabricated identifier found: "
                        f"<code>{', '.join(recheck['invalid_citations'] + recheck['invalid_narrative_txn_ids'] + recheck['invalid_narrative_accounts'])}</code>")
                else:
                    good(f"All {recheck['n_cited']} cited transaction IDs and every "
                         f"identifier in {recheck['note_words']} words of narrative "
                         f"appear in the dossier. Checked against the "
                         f"{len(citable)} IDs actually rendered above — not against "
                         "the transactions that exist.")
                if res.get("red_flag_indicators"):
                    st.markdown("**Red-flag indicators named** — "
                                + ", ".join(res["red_flag_indicators"]))
                if res.get("sources_cited"):
                    st.caption("Sources relied on: " + ", ".join(res["sources_cited"]))

        sg = entry["subgraph"]
        with st.expander(f"Case subgraph — {len(sg['nodes'])} accounts, "
                         f"{sg['edges_shown']} of {sg['edges_total']} transactions"):
            st.caption("Visualisation data, held outside the dossier: the agent never "
                       "saw this drawing, only the tables above.")
            st.dataframe(safe_df(sg["edges"][:200]),
                         width='stretch', hide_index=True)


# ---------------------------------------------------------------------------
with tab_eval:
    st.header("What works, and what does not")
    if not AGENT:
        missing("agent.json")
    else:
        on = AGENT["test_retrieval_on"]
        disc = on["discrimination"]["overall"]

        st.subheader("The disposition carries no signal — and the number that said otherwise was an artifact")
        c1, c2, c3 = st.columns(3)
        c1.metric("escalation precision", f"{disc['escalation_precision']:.1%}")
        c2.metric("queue base rate", f"{disc['queue_base_rate']:.1%}")
        c3.metric("Fisher exact, one-sided", f"p = {disc['fisher_p_one_sided']}")
        finding("""
An earlier configuration beat the escalate-everything control by <b>6.7 points</b>.
Broken out by case size, its escalation precision sat on the base rate inside
<i>every</i> bucket — 38.5% against 38.7% among small cases, 83.3% against 85.7% among
large ones. The lift was Simpson's paradox: small cases have a lower base rate, the agent
closed more aggressively among them, and pooled precision rose without a single case
being judged better than chance.<br><br>
<b>A pooled rate confounds judgement quality with subpopulation choice whenever the
system decides both.</b> The control was necessary and not sufficient.
""")
        st.dataframe(pd.DataFrame([
            {"stratum": k, "n": v["n"], "escalation precision": v["escalation_precision"],
             "base rate": v["queue_base_rate"], "lift (pts)": v["lift_pts"],
             "p": v["fisher_p_one_sided"]}
            for k, v in on["discrimination"].items()
            if k != "reading" and isinstance(v, dict)
        ]).style.format({"escalation precision": "{:.1%}", "base rate": "{:.1%}"}),
            width='stretch', hide_index=True)
        st.caption("Why it was never going to work: the engine is a gradient-boosted "
                   "model over ~100 features including multi-hop topology; the agent "
                   "reads a text summary of a subset. Re-ranking that means improving on "
                   "a model that used strictly more information.")

        st.subheader("What the layer demonstrably is")
        note, typ = on["case_note"], on["typology"]
        c1, c2, c3 = st.columns(3)
        c1.metric("case notes complete",
                  f"{note['n_with_note'] - note['n_incomplete']}/{note['n_with_note']}")
        c2.metric("fabricated IDs in prose",
                  f"{note['n_with_fabricated_id_in_prose']}",
                  delta=f"{note['fabricated_prose_rate']:.1%} of cases",
                  delta_color="off")
        c3.metric("typology dominant-match",
                  f"{(typ['dominant_match_accuracy'] or 0):.1%}",
                  delta=f"{((typ['dominant_match_accuracy'] or 0) - (typ['baselines']['majority_class_dominant_accuracy'] or 0)) * 100:+.1f} pts vs majority class",
                  delta_color="inverse")
        good(f"Zero fabricated transaction IDs in the structured citation list on either "
             f"split. Across roughly {note['median_words'] * note['n_with_note']:,} words "
             f"of generated narrative, <b>{note['n_with_fabricated_id_in_prose']}</b> "
             f"fabricated identifier. The grounding check covers prose, not just the "
             f"structured field — before that, a fabrication in the body of a note was "
             f"undetectable.")
        st.caption(typ["coverage_note"])
        finding(
            "<b>Typology needs a control too, and it nearly shipped without one.</b> "
            "Any-match is not fixed-difficulty — a large case touches many injected "
            "rings, and one 11-account case has all eight typologies in its truth set, "
            f"so the per-case chance rate is {(typ['baselines']['any_match_expected_by_chance'] or 0) * 100:.1f}%, not 12.5%. "
            f"And always predicting {typ['baselines']['majority_class']} scores "
            f"{(typ['baselines']['majority_class_dominant_accuracy'] or 0) * 100:.1f}% "
            f"dominant-match against the agent's {(typ['dominant_match_accuracy'] or 0) * 100:.1f}%. "
            "The agent does not beat the trivial baseline here. Twelve labelled cases, "
            "so neither number is well determined.")

        st.subheader("Does retrieval earn its place?")
        sig = AGENT.get("ablation_significance")
        if sig:
            off = AGENT["test_retrieval_off"]
            # Every cell is stringified: these columns mix "7/12" with 2.736, and a
            # mixed-type object column cannot be converted to Arrow, which silently
            # loses the table rather than raising anywhere the reader would see.
            rf = sig["red_flag_indicators_per_case"]
            ty = sig["typology_any_match"]
            st.dataframe(pd.DataFrame([
                {"metric": "typology any-match", "retrieval on": str(ty["on"]),
                 "off": str(ty["off"]), "p": str(ty["fisher_p_one_sided"]),
                 "verdict": "not significant — 12 labelled cases"},
                {"metric": "red-flag indicators / case", "retrieval on": str(rf["on"]),
                 "off": str(rf["off"]), "p": str(rf["mann_whitney_p_one_sided"]),
                 "verdict": "the real effect"},
                {"metric": "notes naming no indicator at all",
                 "retrieval on": f"{rf['cases_naming_none_on']} of {on['n_alerts']}",
                 "off": f"{rf['cases_naming_none_off']} of {off['n_alerts']}",
                 "p": "", "verdict": ""},
            ]).astype(str), width='stretch', hide_index=True)
            st.markdown("""
The typology delta is the one that looks like the headline and it is the one that cannot
carry it. The indicator result is overwhelming and is the actual finding: **retrieval is
what makes the note cite published regulatory indicators instead of asserting suspicion
in its own voice.** It was never going to make the model a better ranker.
""")

        st.subheader("Unit economics")
        c = on["cost"]
        c1, c2, c3 = st.columns(3)
        c1.metric("cost per case", f"${c['per_alert_usd']:.4f}")
        c2.metric("median latency", f"{c['median_latency_seconds']}s")
        c3.metric("model", c["model"].replace("claude-", ""))
