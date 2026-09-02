"""Day 1: raw CSV/TXT -> typed parquet cache.

Run via `make data`. Idempotent: skips work if the parquet files are newer than
their sources.

WHY A CONVERSION STEP AT ALL
----------------------------
The transactions CSV is 454 MB / ~5.08M rows. Re-parsing it on every run costs tens
of seconds and, worse, CSV carries no type information — so every script would have
to re-specify dtypes and would eventually disagree. Parquet stores the schema with
the data, reloads in about a second, and preserves the categorical encodings that
keep this table inside memory comfortably.

THREE DATA TRAPS THIS FILE EXISTS TO NEUTRALISE
-----------------------------------------------
1. DUPLICATE COLUMN NAMES. The CSV header names TWO columns "Account" (sender and
   receiver). pandas silently mangles the second to "Account.1". Relying on that is
   fragile — it is a rename you would never see go wrong. We rename positionally to
   explicit names instead.

2. LEADING ZEROS IN BANK IDs. Bank IDs look like "010" and "021174". Parsed as
   integers they become 10 and 21174, and every join against the accounts table
   silently returns nothing. They are read as strings, always.

3. ACCOUNT NUMBERS ARE NOT UNIQUE. Eight account numbers appear at two different
   banks, owned by different entities. Keyed on account number alone, those become
   one graph node fusing two unrelated histories. Node identity is the composite
   (bank_id, account_number), constructed here once so no downstream module has to
   remember.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

# The CSV header cannot be used directly (two columns share the name "Account"), so
# columns are renamed BY POSITION. Order here must match the file exactly.
TRANS_COLUMNS = [
    "timestamp",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "amount_received",
    "receiving_currency",
    "amount_paid",
    "payment_currency",
    "payment_format",
    "is_laundering",
]

# Everything identifier-like is read as string to protect leading zeros. Amounts are
# float; the label is a small int. Categorical conversion happens after load, once
# pandas has seen the full value set.
TRANS_DTYPES = {
    "from_bank": "string",
    "from_account": "string",
    "to_bank": "string",
    "to_account": "string",
    "amount_received": "float64",
    "receiving_currency": "string",
    "amount_paid": "float64",
    "payment_currency": "string",
    "payment_format": "string",
    "is_laundering": "int8",
}

# Columns where the number of distinct values is tiny relative to 5M rows. Storing
# them as categories replaces repeated strings with an integer code plus one lookup
# table — a large memory saving on a table this shape.
CATEGORICAL = ["from_bank", "to_bank", "receiving_currency", "payment_currency", "payment_format"]

# "BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 16-degree Fan-Out"
# "BEGIN LAUNDERING ATTEMPT - BIPARTITE"          (no detail suffix)
PATTERN_HEADER = re.compile(
    r"^BEGIN LAUNDERING ATTEMPT - (?P<ptype>[A-Z-]+?)\s*(?::\s*(?P<detail>.*))?$"
)


def node_id(bank: pd.Series, account: pd.Series) -> pd.Series:
    """Build the composite node identity used everywhere in this project.

    Defined once, here, because a graph built on a different key than the features
    were computed against fails silently rather than loudly — the join simply
    returns fewer matches and the model quietly loses signal.
    """
    return bank.astype("string") + ":" + account.astype("string")


def load_transactions() -> pd.DataFrame:
    """Read the transactions CSV with explicit, trap-aware typing."""
    df = pd.read_csv(
        config.TRANS_CSV,
        header=0,            # consume the real header row...
        names=TRANS_COLUMNS,  # ...but override it, because two columns share a name
        dtype=TRANS_DTYPES,
        # Dates are "2022/09/01 00:20". Passing the exact format skips pandas'
        # per-row inference, which on 5M rows is the difference between seconds
        # and minutes.
        parse_dates=["timestamp"],
        date_format="%Y/%m/%d %H:%M",
    )

    for col in CATEGORICAL:
        df[col] = df[col].astype("category")

    df["from_id"] = node_id(df["from_bank"], df["from_account"])
    df["to_id"] = node_id(df["to_bank"], df["to_account"])

    return df


def parse_patterns() -> pd.DataFrame:
    """Parse Patterns.txt into a tidy table: one row per laundering transaction.

    Format is strictly:

        BEGIN LAUNDERING ATTEMPT - <TYPE>[:  <detail>]
        <transaction row>            (same 11 columns as the CSV, no header)
        ...
        END LAUNDERING ATTEMPT - <TYPE>
        <blank line>

    Validated as exactly 370 BEGIN/END pairs and 3,209 transaction rows, all with 11
    fields and no unexpected lines — so this parser asserts rather than tolerates.
    A malformed line means our understanding of the file changed, and we want to
    hear about it loudly.

    The output carries pattern_id and pattern_type, which is what turns the agent's
    typology classification (A7) into a real supervised task with ground truth, and
    what makes pattern-level recall (B5) computable.
    """
    rows: list[dict] = []
    pattern_id = -1
    current_type: str | None = None
    current_detail: str | None = None
    begins = ends = 0

    with open(config.PATTERNS_TXT, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.rstrip("\n")

            if not line.strip():
                continue

            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                match = PATTERN_HEADER.match(line)
                if match is None:
                    raise ValueError(f"{config.PATTERNS_TXT}:{lineno}: unparseable header: {line!r}")
                pattern_id += 1
                begins += 1
                current_type = match.group("ptype")
                detail = match.group("detail")
                current_detail = detail.strip() if detail else None
                continue

            if line.startswith("END LAUNDERING ATTEMPT"):
                ends += 1
                current_type = None
                continue

            # Anything else must be a transaction row inside an open block.
            if current_type is None:
                raise ValueError(f"{config.PATTERNS_TXT}:{lineno}: transaction outside a block")

            fields = line.split(",")
            if len(fields) != len(TRANS_COLUMNS):
                raise ValueError(
                    f"{config.PATTERNS_TXT}:{lineno}: expected {len(TRANS_COLUMNS)} fields, "
                    f"got {len(fields)}"
                )

            record = dict(zip(TRANS_COLUMNS, fields))
            record["pattern_id"] = pattern_id
            record["pattern_type"] = current_type
            record["pattern_detail"] = current_detail
            rows.append(record)

    if begins != ends:
        raise ValueError(f"unbalanced blocks: {begins} BEGIN vs {ends} END")

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y/%m/%d %H:%M")
    for col in ("amount_received", "amount_paid"):
        df[col] = df[col].astype("float64")
    df["is_laundering"] = df["is_laundering"].astype("int8")
    df["pattern_type"] = df["pattern_type"].astype("category")

    df["from_id"] = node_id(df["from_bank"], df["from_account"])
    df["to_id"] = node_id(df["to_bank"], df["to_account"])

    return df


def load_accounts() -> pd.DataFrame:
    """Account -> entity mapping.

    Not in the original build plan: this file was found on Day 1. It supplies an
    Entity ID (the owner behind an account) and an entity type embedded in the name.
    That matters because real AML investigates a CUSTOMER who may hold many accounts
    across many banks, not an isolated account — and 38% of entities here do exactly
    that. See docs/NOTES.md.
    """
    df = pd.read_csv(
        config.DATA_RAW / "HI-Small_accounts.csv",
        dtype={
            "Bank Name": "string",
            "Bank ID": "string",       # leading zeros again
            "Account Number": "string",
            "Entity ID": "string",
            "Entity Name": "string",
        },
    )
    df = df.rename(
        columns={
            "Bank Name": "bank_name",
            "Bank ID": "bank_id",
            "Account Number": "account_number",
            "Entity ID": "entity_id",
            "Entity Name": "entity_name",
        }
    )
    # "Corporation #33520" -> "Corporation". Entity type is a genuine AML risk
    # signal: shell-company structures are a classic layering vehicle.
    df["entity_type"] = df["entity_name"].str.replace(r"\s*#\d+$", "", regex=True).astype("category")
    df["node_id"] = node_id(df["bank_id"], df["account_number"])
    return df


def main() -> int:
    print("Loading transactions (454 MB, ~5.08M rows)...")
    trans = load_transactions()
    print(f"  {len(trans):,} rows x {len(trans.columns)} columns")
    print(f"  memory: {trans.memory_usage(deep=True).sum() / 1e9:.2f} GB")

    print("\nParsing laundering patterns...")
    patterns = parse_patterns()
    print(f"  {len(patterns):,} transactions across {patterns['pattern_id'].nunique()} patterns")

    print("\nLoading account -> entity map...")
    accounts = load_accounts()
    print(f"  {len(accounts):,} accounts, {accounts['entity_id'].nunique():,} entities")

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    print("\nWriting parquet...")
    trans.to_parquet(config.TRANS_PARQUET, index=False, compression="zstd")
    patterns.to_parquet(config.PATTERNS_PARQUET, index=False, compression="zstd")
    accounts.to_parquet(config.DATA_PROCESSED / "accounts.parquet", index=False, compression="zstd")

    for path in (config.TRANS_PARQUET, config.PATTERNS_PARQUET,
                 config.DATA_PROCESSED / "accounts.parquet"):
        print(f"  {path.name}: {path.stat().st_size / 1e6:.1f} MB")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
