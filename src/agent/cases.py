"""Group the alert queue into CASES — the unit an investigator actually opens.

WHY THE UNIT CHANGED
--------------------
Per-account triage was built first, run, and measured not to work. Two calibrations,
both useless: one closed 21.7% of true positives, the other closed nothing at all and
scored exactly the escalate-everything control. The agent explained why itself —

    "that counterparty's transactions are not in evidence here, so the pattern that
     alarmed the model cannot be verified from this account's side"

Four of the five true positives it wrongly closed were **receiving spokes of a
fan-out**: one inbound payment and nothing else. From that account's own rows a spoke is
indistinguishable from an ordinary receipt. The laundering lives in the SENDER's shape.
Asked to judge a spoke alone, the agent can only be reckless or useless, and no amount
of prompt engineering supplies a fact that is not in the context.

Grouping fixes it structurally: the hub and its spokes land in one dossier, so a spoke
is described in context instead of judged blind.

It is also the right unit on its own terms. `Patterns.txt` labels rings, not accounts,
so case-level typology classification can be scored honestly where account-level could
not. Compliance opens a case on a network. And it costs 4.4x fewer LLM calls, which is
what brought the project inside its budget.

HOW CASES ARE FORMED
--------------------
Two alerted accounts join the same case if either:

  1. they transact directly with each other in the scored window, or
  2. they share a non-alerted counterparty, and that counterparty is shared by at most
     CASE_BRIDGE_MAX_SHARED alerted accounts.

The cap in rule 2 is doing real work. Without it — including every
counterparty-to-counterparty edge — the val queue collapsed into a single component of
**15,027 accounts**, because a handful of popular counterparties chain everything
together. With it: 45 cases, median 2 accounts, maximum 62, and all 200 alerted
accounts covered.

Non-alerted counterparties are NOT case members. They appear in the evidence, because
that is where the fan-out shape becomes visible, but they are not themselves under
review and the case is not accountable for them.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import alerts as alerts_mod  # noqa: E402
import config  # noqa: E402


@dataclass
class Case:
    """One cluster of alerted accounts, reviewed together."""
    case_id: str
    members: list[str]
    max_alert_score: float
    mean_alert_score: float
    n_productive_members: int
    split: str

    # Filled by build_cases; kept off the constructor so the dataclass stays printable.
    evidence: pd.DataFrame = field(default=None, repr=False)

    @property
    def is_productive(self) -> bool:
        """A case is productive if ANY member account laundered in the scored window.

        The disjunction is the operationally correct rule: an investigator who opens
        this case and finds laundering anywhere in it has spent their time well, which
        is exactly what "productive alert" means in compliance usage.
        """
        return self.n_productive_members > 0

    @property
    def n_members(self) -> int:
        return len(self.members)


def _link_graph(val: pd.DataFrame, alerted: set[str]) -> nx.Graph:
    """Build the account-to-account graph the cases are components of."""
    graph = nx.Graph()
    graph.add_nodes_from(alerted)

    # Rule 1 — direct transactions between two alerted accounts.
    direct = val[val["from_id"].isin(alerted) & val["to_id"].isin(alerted)]
    graph.add_edges_from(zip(direct["from_id"], direct["to_id"]))

    # Rule 2 — a shared non-alerted counterparty, capped.
    touching = val[val["from_id"].isin(alerted) | val["to_id"].isin(alerted)]
    shared: dict[str, set[str]] = {}
    for sender, receiver in zip(touching["from_id"], touching["to_id"]):
        member, other = ((sender, receiver) if sender in alerted
                         else (receiver, sender))
        if other not in alerted:
            shared.setdefault(other, set()).add(member)

    for counterparty, group in shared.items():
        if 1 < len(group) <= config.CASE_BRIDGE_MAX_SHARED:
            # Star, not clique: n-1 edges instead of n(n-1)/2, same component.
            ordered = sorted(group)
            graph.add_edges_from((ordered[0], other) for other in ordered[1:])

    return graph


def build_cases(frame: pd.DataFrame, alert_table: pd.DataFrame, split: str,
                scores: pd.Series | None = None) -> list[Case]:
    """Cluster an alert table into cases, highest-scoring case first.

    Ordering is deterministic — by max alert score, then case id — so a re-run produces
    the same case ids and the response cache stays valid.
    """
    alerted = set(alert_table.index)
    part = frame[frame["split"] == split]
    part = part[part["from_id"] != part["to_id"]]

    graph = _link_graph(part, alerted)
    components = [sorted(c) for c in nx.connected_components(graph)]

    cases: list[Case] = []
    for members in components:
        rows = alert_table.loc[members]
        cases.append(Case(
            case_id="",  # assigned after sorting, so ids follow queue order
            members=members,
            max_alert_score=float(rows["alert_score"].max()),
            mean_alert_score=float(rows["alert_score"].mean()),
            n_productive_members=int(rows["is_productive"].sum()),
            split=split,
        ))

    cases.sort(key=lambda c: (-c.max_alert_score, c.members[0]))
    for i, case in enumerate(cases, 1):
        case.case_id = f"CASE-{split.upper()}-{i:03d}"
        case.evidence = case_evidence(frame, case, scores)

    _assert_partition(cases, alerted)
    return cases


def case_evidence(frame: pd.DataFrame, case: Case,
                  scores: pd.Series | None = None) -> pd.DataFrame:
    """Every transaction any member took part in, deduplicated.

    A transaction between two members would otherwise appear twice — once from each
    side — and inflate both the apparent volume and the citation whitelist.
    """
    parts = [alerts_mod.alert_evidence(frame, member, case.split, scores)
             for member in case.members]
    evidence = pd.concat(parts)
    evidence = evidence[~evidence.index.duplicated(keep="first")]
    return evidence.sort_values("timestamp")


def _assert_partition(cases: list[Case], alerted: set[str]) -> None:
    """Every alerted account in exactly one case, or the queue silently shrank."""
    seen: list[str] = []
    for case in cases:
        seen.extend(case.members)
    if len(seen) != len(set(seen)):
        raise ValueError("an alerted account appears in more than one case")
    if set(seen) != alerted:
        missing = alerted - set(seen)
        raise ValueError(
            f"{len(missing)} alerted accounts are in no case, so they would never be "
            f"reviewed: {sorted(missing)[:5]}"
        )


def summarise(cases: list[Case]) -> dict:
    sizes = sorted((c.n_members for c in cases), reverse=True)
    productive = [c for c in cases if c.is_productive]
    return {
        "n_cases": len(cases),
        "n_accounts": sum(sizes),
        "accounts_per_case": {
            "max": sizes[0], "median": sizes[len(sizes) // 2], "min": sizes[-1],
            "over_cap": sum(1 for s in sizes if s > config.CASE_MAX_ACCOUNTS),
        },
        "productive_cases": len(productive),
        "non_productive_cases": len(cases) - len(productive),
        "productive_share": round(len(productive) / len(cases), 4) if cases else 0.0,
        "bridge_cap": config.CASE_BRIDGE_MAX_SHARED,
    }


if __name__ == "__main__":
    import splits

    frame = splits.load_transactions()
    for split in ("val", "test"):
        path = config.RESULTS / f"alerts_{split}.parquet"
        if not path.exists():
            continue
        table = pd.read_parquet(path)
        cases = build_cases(frame, table, split)
        info = summarise(cases)
        print(f"\n{split}: {info['n_accounts']} alerted accounts -> "
              f"{info['n_cases']} cases")
        print(f"  accounts/case: max {info['accounts_per_case']['max']}, "
              f"median {info['accounts_per_case']['median']}, "
              f"over cap {info['accounts_per_case']['over_cap']}")
        print(f"  productive {info['productive_cases']} / "
              f"non-productive {info['non_productive_cases']} "
              f"({info['productive_share'] * 100:.0f}% productive)")
        for case in cases[:5]:
            print(f"    {case.case_id}  {case.n_members:>3} accts  "
                  f"score {case.max_alert_score:.4f}  "
                  f"{'PRODUCTIVE' if case.is_productive else 'not productive'}  "
                  f"{len(case.evidence):>5} txns")
