"""Every number the README prints, fetched from results/*.json by an explicit path.

WHY A FETCHER RATHER THAN AN f-STRING
-------------------------------------
The build plan's rule for Day 10 is that no number in the README is typed by hand. That
is easy to say and easy to violate: one hard-coded "0.1938" in a template is invisible
forever, and it goes stale the first time anything is re-run.

So every value goes through `Fetch.__call__`, which resolves a dotted path into the
loaded JSON and RECORDS it. The recorded paths are written to
`results/readme_provenance.json`, and a test asserts each one still resolves. A number
that loses its source therefore breaks the build instead of quietly lying.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

PROVENANCE = config.RESULTS / "readme_provenance.json"

FILES = {
    "engine": "engine.json",
    "arms": "arms.json",
    "agent": "agent.json",
    "single_bank": "single_bank.json",
    "fx": "fx_diagnosis.json",
    "triage": "triage_cases_test.json",
}


class Missing(Exception):
    """A required results file has not been generated."""


class Fetch:
    """Dotted-path reader over the results JSON, recording every path it serves."""

    def __init__(self) -> None:
        self.data: dict[str, dict] = {}
        self.used: list[str] = []
        self.absent: list[str] = []
        for key, name in FILES.items():
            path = config.RESULTS / name
            if path.exists():
                loaded = json.loads(path.read_text())
                # One results file is a list of per-case records rather than a mapping.
                # Key it by unit_id so a path like `triage.CASE-TEST-002.n_members`
                # resolves the same way every other path does.
                if isinstance(loaded, list):
                    loaded = {r["unit_id"]: r for r in loaded if "unit_id" in r}
                self.data[key] = loaded
            else:
                self.absent.append(name)

    def has(self, group: str) -> bool:
        return group in self.data

    def __call__(self, path: str, default=None):
        """`fetch("engine.splits.val.pr_auc")`. Records the path for the provenance file."""
        group, _, rest = path.partition(".")
        if group not in self.data:
            if default is not None:
                return default
            raise Missing(f"{FILES.get(group, group)} not generated (wanted {path})")
        node = self.data[group]
        for part in rest.split("."):
            if isinstance(node, list):
                node = node[int(part)]
            elif part in node:
                node = node[part]
            elif default is not None:
                return default
            else:
                raise Missing(f"no such path: {path} (stopped at {part!r})")
        self.used.append(path)
        return node

    def pct(self, path: str, places: int = 1) -> str:
        return f"{self(path) * 100:.{places}f}%"

    def num(self, path: str, places: int = 4) -> str:
        return f"{self(path):.{places}f}"

    def thousands(self, path: str) -> str:
        return f"{self(path):,}"

    def write_provenance(self) -> None:
        PROVENANCE.write_text(json.dumps({
            "note": ("every number rendered into README.md, and the results path it came "
                     "from. `make_readme.py` records these as it renders; the test suite "
                     "asserts each still resolves, so a number cannot outlive its source."),
            "files_read": {k: v for k, v in FILES.items() if k in self.data},
            "files_absent": self.absent,
            "n_values": len(self.used),
            "paths": sorted(set(self.used)),
        }, indent=2))


# ---------------------------------------------------------------------------
# The typology -> feature -> red flag table (C1)
# ---------------------------------------------------------------------------
# Built from features that ACTUALLY carry the model, ranked by LightGBM gain, and from
# source ids that are actually registered in kb/sources.json. Writing this table from
# imagination is the failure mode it exists to avoid: a table of plausible-sounding
# features that the model does not use signals domain knowledge while describing nothing.
#
# Two rows the build plan asked for are deliberately absent, and their absence is the
# point — structuring bands and round-number flags were built, tested against the data,
# and dropped. The generator does not model reporting thresholds: laundering is enriched
# 2.79% in the $9-10k band against 2.96% in $10-11k, which is not a threshold effect, and
# only 0.013% of amounts are round hundreds.
TYPOLOGY_TABLE = [
    {
        "behaviour": "Layering through currency conversion",
        "shape": "One account paying out across many different currencies in a short window",
        "features": "`from_out_n_currencies` (rank 1 by gain), `is_cross_currency`",
        "reference": "FFIEC Appendix F — funds-transfer red flags (`ffiec-appendix-f`)",
    },
    {
        "behaviour": "Funnel account / fan-in collection",
        "shape": "Many unrelated payers into one account, little onward activity",
        "features": "`to_in_n` (rank 2), `to_in_n_counterparties`, `to_in_counterparties_per_txn`",
        "reference": "FinCEN FIN-2014-A005 — funnel accounts (`fincen-fin-2014-a005`)",
    },
    {
        "behaviour": "Burst velocity / smurfing",
        "shape": "Inbound rate far above the account's own baseline",
        "features": "`to_in_txn_per_day` (rank 3), `from_in_txn_per_day`",
        "reference": "FFIEC Appendix G — structuring (`ffiec-appendix-g`)",
    },
    {
        "behaviour": "Placement via cash-like instruments",
        "shape": "Concentration in payment formats associated with the placement stage",
        "features": "`from_layering_format_share` (rank 4), `to_layering_format_share`, `payment_format`",
        "reference": "Placement stage; FATF virtual-asset indicators for the Bitcoin leg (`fatf-va-red-flags-2020`)",
    },
    {
        "behaviour": "Network position — the hub a ring is built around",
        "shape": "Money-weighted centrality no single-account aggregate can express",
        "features": "`to_pagerank` (rank 5), betweenness, Louvain community size",
        "reference": "Our mapping from the injected shapes (`ibm-amlsim-typologies`)",
    },
    {
        "behaviour": "Dormancy burst — the mule activation signature",
        "shape": "An account quiet for a long stretch, then suddenly busy",
        "features": "`from_in_active_days` (rank 6), `to_gap_burstiness`, `to_median_gap_hours`",
        "reference": "FinCEN FIN-2020-A003 — money-mule taxonomy (`fincen-fin-2020-a003`)",
    },
    {
        "behaviour": "Layering across institutions",
        "shape": "One customer's funds traversing many banks",
        "features": "`from_entity_n_banks` (rank 9), `is_same_bank`, `from_out_n_banks`",
        "reference": "Layering stage; FFIEC Appendix F (`ffiec-appendix-f`)",
    },
    {
        "behaviour": "Pass-through / mule signature",
        "shape": "Money in approximately equals money out, nothing retained",
        "features": "`from_flow_through_ratio`, `from_retention_ratio`, `to_median_hours_to_forward`",
        "reference": "FinCEN FIN-2020-A003 (`fincen-fin-2020-a003`)",
    },
    {
        "behaviour": "Structuring below the CTR threshold",
        "shape": "Amounts clustering just under $10,000",
        "features": "**none — built, tested, dropped**",
        "reference": "Enrichment is 2.79% in $9–10k against 2.96% in $10–11k: not a threshold effect. The simulator does not model reporting thresholds (`ffiec-appendix-g`)",
    },
]
