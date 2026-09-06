"""Assemble everything the triage agent sees for one alert.

WHY THIS IS A SEPARATE MODULE FROM THE AGENT ITSELF
---------------------------------------------------
The dossier defines the agent's entire epistemic position: what it knows, and — more
importantly — what it is allowed to claim. `cited_transaction_ids` is validated against
the transaction IDs this module put in front of the model, so the boundary between
"evidence" and "everything else" has to be one explicit object rather than a side effect
of prompt construction.

Keeping it separate also makes the RAG ablation on Day 8 a one-line change: build the
same dossier with `retrieve=False` and measure what the knowledge base was worth.

WHAT GOES IN
------------
  1. the alert itself — score, rank in the queue
  2. an account profile aggregated over the scored window
  3. the evidence transactions, capped (see below)
  4. the ENGINE'S OWN REASONS — SHAP attributions for the transaction that drove the
     alert. Without these the agent re-derives suspicion independently and the case note
     explains the LLM's guess rather than why this model fired (B3).
  5. retrieved regulatory passages, keyed on a description built from 2 and 4

THE EVIDENCE CAP, AND WHY IT CHANGES WHAT COUNTS AS A HALLUCINATION
-------------------------------------------------------------------
Evidence volume per alert is wildly skewed: median 7 transactions, p90 24, max 39,155.
Sending all of it would be expensive and would bury the signal, so the dossier shows the
highest-scoring EVIDENCE_LIMIT transactions plus aggregate statistics over the rest —
which is roughly what a case management system shows an analyst anyway.

The consequence is that citations are validated against what was SHOWN, not against
everything that exists. An ID the agent produces that is real but was never in its
context is still a fabrication: the model did not have it, so it cannot have read it off
the evidence. The stricter check is the honest one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import alerts as alerts_mod  # noqa: E402
import config  # noqa: E402
import kb_index  # noqa: E402

# Chosen from the measured distribution: covers every evidence row for 187 of the 200
# val alerts outright, and gives the 13 busy accounts a representative top slice.
EVIDENCE_LIMIT = 40

# Plain-English renderings for the feature names SHAP returns. The agent is a language
# model reading a case file, not a data scientist reading a schema: "the sender moved
# money in 3 different currencies" is evidence, `from_out_n_currencies = 3.0` is a
# variable binding. This also feeds the retrieval query, where the vocabulary has to
# match regulatory prose for the embedding to find anything.
GLOSSARY: dict[str, str] = {
    "n": "number of transactions",
    "total_usd": "total USD moved",
    "mean_usd": "average transaction size (USD)",
    "std_usd": "variability of transaction size",
    "max_usd": "largest single transaction (USD)",
    "n_counterparties": "number of distinct counterparties",
    "n_banks": "number of distinct banks dealt with",
    "n_currencies": "number of distinct currencies used",
    "n_formats": "number of distinct payment formats used",
    "active_days": "days active in the window",
    "txn_per_day": "transactions per active day",
    "counterparties_per_txn": "distinct counterparties per transaction",
    "flow_through_ratio": "share of money received that was passed straight on",
    "retention_ratio": "share of money received that was kept",
    "median_hours_to_forward": "median hours before funds were forwarded",
    "gap_burstiness": "burstiness of transaction timing",
    "median_gap_hours": "median hours between transactions",
    "placement_format_share": "share of activity in cash-like formats (placement stage)",
    "layering_format_share": "share of activity in transfer formats (layering stage)",
    "pagerank": "money-weighted inbound importance in the network",
    "betweenness": "how often the account sits on paths between others",
    "core_number": "embeddedness in a densely connected region",
    "in_cycle": "sits on a cycle where money can return to its origin",
    "scc_size": "size of the cycle-capable component it belongs to",
    "wcc_size": "size of its connected component",
    "louvain_community_size": "size of its network community",
    "degree_total": "total distinct counterparties",
    "is_in_graph": "transacted with someone other than itself",
    "is_unseen": "never seen during the training window",
    "entity_n_banks": "number of banks this entity holds accounts at",
    "entity_n_accounts": "number of accounts this entity holds",
    "paid_usd": "amount sent (USD)",
    "recv_usd": "amount received (USD)",
    "amount_spread_usd": "difference between sent and received amounts",
    "is_cross_currency": "sent and received in different currencies",
    "hour": "hour of day",
    "day_of_week": "day of week",
    "payment_format": "payment format",
    "payment_currency": "currency sent",
    "receiving_currency": "currency received",
    "is_same_bank": "both parties at the same bank",
    "is_self_transaction": "transfer to itself",
}


def describe_feature(name: str) -> str:
    """Turn `from_out_n_currencies` into 'the sender, outbound: number of currencies'."""
    side = ""
    rest = name
    for prefix, label in (("from_", "the sender"), ("to_", "the receiver")):
        if name.startswith(prefix):
            side, rest = label, name[len(prefix):]
            break

    direction = ""
    for prefix, label in (("in_", "incoming"), ("out_", "outgoing")):
        if rest.startswith(prefix):
            direction, rest = label, rest[len(prefix):]
            break

    base = GLOSSARY.get(rest, rest.replace("_", " "))
    parts = [p for p in (side, direction, base) if p]
    return ", ".join(parts) if side else base


def account_profile(evidence: pd.DataFrame, node_id: str) -> dict:
    """Aggregate the account's behaviour over the scored window."""
    external = evidence[evidence["from_id"] != evidence["to_id"]]
    n_self = int(len(evidence) - len(external))
    out = external[external["direction"] == "OUT"]
    inn = external[external["direction"] == "IN"]

    sent = float(out["paid_usd"].sum()) if "paid_usd" in out else 0.0
    received = float(inn["recv_usd"].sum()) if "recv_usd" in inn else 0.0

    return {
        "account": node_id,
        "bank": node_id.split(":")[0],
        "transactions": int(len(evidence)),
        "outgoing": int(len(out)),
        "incoming": int(len(inn)),
        "self_transfers": n_self,
        "usd_sent": round(sent, 2),
        "usd_received": round(received, 2),
        "flow_through": round(sent / received, 3) if received > 0 else None,
        "distinct_counterparties_out": int(out["to_id"].nunique()) if len(out) else 0,
        "distinct_counterparties_in": int(inn["from_id"].nunique()) if len(inn) else 0,
        "distinct_banks": int(pd.concat([
            out["to_id"].str.split(":").str[0] if len(out) else pd.Series(dtype="object"),
            inn["from_id"].str.split(":").str[0] if len(inn) else pd.Series(dtype="object"),
        ]).nunique()),
        "payment_formats": sorted(evidence["payment_format"].astype(str).unique().tolist()),
        "currencies": sorted(set(evidence["payment_currency"].astype(str))
                             | set(evidence["receiving_currency"].astype(str))),
        "first_seen": str(evidence["timestamp"].min()),
        "last_seen": str(evidence["timestamp"].max()),
        "window_hours": round(
            (evidence["timestamp"].max() - evidence["timestamp"].min()).total_seconds()
            / 3600, 1),
    }


def evidence_table(evidence: pd.DataFrame, limit: int = EVIDENCE_LIMIT) -> tuple[list[dict], dict]:
    """The citable transactions, plus a summary of anything the cap excluded."""
    shown = evidence
    omitted: dict = {}
    if len(evidence) > limit:
        shown = evidence.nlargest(limit, "model_score").sort_values("timestamp")
        rest = evidence.drop(shown.index)
        omitted = {
            "count": int(len(rest)),
            "note": (f"{len(rest):,} further transactions are NOT shown and MUST NOT be "
                     "cited. Aggregate figures for them are given here only so the "
                     "shown sample is not mistaken for the whole account."),
            "usd_total": round(float(rest["paid_usd"].sum()), 2),
            "max_model_score": round(float(rest["model_score"].max()), 4),
            "distinct_counterparties": int(pd.concat([rest["from_id"], rest["to_id"]])
                                           .nunique()),
        }

    rows = []
    for _, r in shown.iterrows():
        counterparty = r["to_id"] if r["direction"] == "OUT" else r["from_id"]
        # 11.6% of this dataset is self-transfers. Presented without a label they read
        # as ordinary outbound legs and would inflate an apparent fan-out; a spoke that
        # goes back to the same account is not a spoke.
        is_self = counterparty == r["from_id"] == r["to_id"]
        rows.append({
            "txn_id": r["txn_id"],
            "timestamp": str(r["timestamp"]),
            "direction": "SELF" if is_self else r["direction"],
            "counterparty": "(same account — self transfer)" if is_self else counterparty,
            "counterparty_bank": counterparty.split(":")[0],
            "amount_usd": round(float(r["paid_usd"]), 2),
            "currency_sent": str(r["payment_currency"]),
            "currency_received": str(r["receiving_currency"]),
            "payment_format": str(r["payment_format"]),
            "model_score": round(float(r["model_score"]), 4),
        })
    return rows, omitted


def model_reasons(shap_entry: dict | None) -> list[dict]:
    """SHAP attributions, rendered readably."""
    if not shap_entry:
        return []
    out = []
    for r in shap_entry["reasons"]:
        out.append({
            "factor": describe_feature(r["feature"]),
            "raw_feature": r["feature"],
            "value": r["value"],
            "effect": r["direction"],
            "weight": round(abs(r["contribution"]), 3),
        })
    return out


def shape_query(profile: dict) -> str:
    """A query describing the account's TOPOLOGY, in typology vocabulary.

    Retrieval is sensitive to phrasing in a way that is easy to underestimate. The first
    version of this function appended the SHAP feature names to a behavioural
    description, on the reasoning that more signal is better. Measured on a pure fan-out
    account, that query did not return the FAN-OUT document at all (top hits:
    GATHER-SCATTER, then the stages document); dropping the feature names moved FAN-OUT
    to rank 2, and phrasing the query purely in shape vocabulary moved it to rank 1 with
    similarity 0.634 against the polluted query's 0.443.

    Schema vocabulary and regulatory prose do not occupy the same region of embedding
    space, so mixing them dilutes the query toward neither. Hence two narrow queries
    rather than one wide one.
    """
    out_n = profile["distinct_counterparties_out"]
    in_n = profile["distinct_counterparties_in"]

    if in_n == 0 and out_n > 2:
        return ("single source distributing money to many destination accounts, "
                "fan out, dispersal to multiple recipients")
    if out_n == 0 and in_n > 2:
        return ("many source accounts paying into one collection account, fan in, "
                "funnel account receiving multiple deposits")
    if in_n > 2 and out_n > 2:
        return ("funds collected from many sources into one account then redistributed "
                "to many destinations, gather scatter, consolidation and dispersal")
    if in_n == 1 and out_n == 1:
        return ("pass-through intermediary account, one payer and one payee, layering "
                "chain, funds forwarded onward")
    if in_n <= 2 and out_n <= 2:
        return ("small number of counterparties, bilateral transfers between a few "
                "accounts, possible cycle or stack layer")
    return ("account with both incoming and outgoing transfers to several "
            "counterparties, intermediary in a laundering network")


def behaviour_query(profile: dict) -> str:
    """A query describing what the account DID, phrased the way red-flag lists are.

    Deliberately free of feature names and account identifiers — those match nothing in
    regulatory prose and only dilute the vector.
    """
    bits = []
    if profile["flow_through"] is not None and profile["flow_through"] > 0.8:
        bits.append("funds received are almost immediately transferred out again")
    if profile["window_hours"] < 48:
        bits.append("all activity compressed into less than two days")
    if len(profile["currencies"]) > 2:
        bits.append("transfers in several different currencies")
    if profile["distinct_banks"] > 2:
        bits.append("transfers spanning multiple financial institutions")
    if profile["self_transfers"]:
        bits.append("transfers between accounts held by the same party")
    formats = {f.lower() for f in profile["payment_formats"]}
    if formats & {"cash", "cheque"}:
        bits.append("cash and cheque deposits, placement of currency")
    if "ach" in formats or "wire" in formats:
        bits.append("wire and ACH funds transfers with no apparent business purpose")
    if not bits:
        bits.append("unusual funds transfer activity inconsistent with account history")
    return "; ".join(bits)


def retrieve_context(profile: dict, k: int | None = None) -> tuple[list[dict], dict]:
    """Retrieve on both queries and merge, keeping the best score for a repeated chunk.

    Two narrow queries rather than one wide one, because the knowledge base is doing two
    different jobs: naming the SHAPE (typology reference) and naming the RED FLAGS
    (regulatory language for the rationale). One query optimised for both is optimised
    for neither.
    """
    k = config.RAG_TOP_K if k is None else k
    queries = {"shape": shape_query(profile), "behaviour": behaviour_query(profile)}

    merged: dict[str, dict] = {}
    for label, query in queries.items():
        for hit in kb_index.retrieve(query, k):
            key = hit["title"] + hit["text"][:60]
            if key not in merged or hit["similarity"] > merged[key]["similarity"]:
                merged[key] = {**hit, "matched_on": label}
    hits = sorted(merged.values(), key=lambda h: -h["similarity"])
    return hits, queries


def build(frame: pd.DataFrame, node_id: str, split: str, scores: pd.Series,
          alert_row: pd.Series, rank: int, shap_entry: dict | None,
          retrieve: bool = True) -> dict:
    """The complete dossier for one alert."""
    evidence = alerts_mod.alert_evidence(frame, node_id, split, scores)
    profile = account_profile(evidence, node_id)
    rows, omitted = evidence_table(evidence)
    reasons = model_reasons(shap_entry)

    dossier = {
        "alert": {
            "account": node_id,
            "queue_rank": rank,
            "model_score": round(float(alert_row["alert_score"]), 4),
            "scored_window": split,
        },
        "account_profile": profile,
        "evidence": rows,
        "evidence_omitted": omitted,
        "model_reasons": reasons,
        "knowledge_base": [],
    }

    if retrieve:
        hits, queries = retrieve_context(profile)
        dossier["retrieval_queries"] = queries
        dossier["knowledge_base"] = [
            {"title": h["title"], "source": h["source_id"], "url": h["url"],
             "text": h["text"], "similarity": h["similarity"],
             "matched_on": h["matched_on"]}
            for h in hits
        ]
    return dossier


def citable_ids(dossier: dict) -> set[str]:
    """Exactly the transaction IDs the agent was shown — the hallucination check's basis."""
    return {row["txn_id"] for row in dossier["evidence"]}


def load_shap(split: str) -> dict:
    path = config.RESULTS / f"shap_{split}.json"
    return json.loads(path.read_text()) if path.exists() else {}


if __name__ == "__main__":
    import splits

    frame = splits.load_transactions()
    import lightgbm as lgb

    import build_features
    booster = lgb.Booster(model_file=str(config.ENGINE_MODEL))
    X, _, split_series, _ = build_features.build_arm(config.ENGINE_ARM, frame)
    mask = (split_series == "val").to_numpy()
    scores = pd.Series(booster.predict(X[mask], num_iteration=booster.best_iteration),
                       index=frame.index[mask])

    top = pd.read_parquet(config.RESULTS / "alerts_val.parquet")
    shap = load_shap("val")
    node = top.index[0]
    d = build(frame, node, "val", scores, top.loc[node], 1, shap.get(node))
    print(json.dumps(d, indent=2, default=str)[:4000])
