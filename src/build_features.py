"""Assemble the feature matrix for a given arm.

Arms are cumulative (see config.ARMS):

    R  rules baseline, no model      -> src/baselines.py
    A  transaction attributes only
    B  A + account aggregates + typology + entity features
    C  B + multi-hop graph topology  (Day 3)
    D  C + node2vec embeddings       (Day 4)

Deliberately NOT cached to disk. Rebuilding takes well under a minute, and a cache is
a staleness failure mode: a feature file built under an older definition of `node_id`
would keep loading silently after the definition changed. That exact class of bug —
a join that silently produces NaN rather than failing — already cost us the entity
features once on Day 2.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import embeddings  # noqa: E402
import features_account  # noqa: E402
import features_graph  # noqa: E402
import features_txn  # noqa: E402
import features_typology  # noqa: E402
import splits  # noqa: E402

LABEL = "is_laundering"


def build_arm(arm: str, frame: pd.DataFrame | None = None, n2v_seed: int | None = None):
    """Return (X, y, split_series) for the requested arm.

    All account-derived features are fitted on the TRAINING WINDOW ONLY and looked up
    for val/test rows — the leak-free rule (A1). The training frame is selected here
    rather than trusted from the caller so the rule cannot be bypassed by accident.
    """
    if arm not in {"A", "B", "C", "D"}:
        raise ValueError(f"unknown arm {arm!r}")

    if frame is None:
        frame = splits.load_transactions()

    train = splits.train_frame(frame)

    parts = [features_txn.build(frame)]
    categorical = list(features_txn.CATEGORICAL)

    # Arms are strictly NESTED: C contains all of B, which contains all of A. That is
    # what makes the ablation interpretable — any lift from C over B is attributable
    # to the features C adds and to nothing else.
    # NOTE: every arm from B onward includes these. Forgetting to add "D" here once
    # produced an arm D with 155 features instead of 231 — missing all 87 account and
    # typology features — which made the C vs D comparison meaningless while still
    # training and reporting a plausible-looking number.
    if arm in {"B", "C", "D"}:
        account_table = features_account.build_account_table(train)
        parts.append(features_account.build(frame, account_table))

        typology_table = features_typology.build_typology_table(train)
        entity_table = features_typology.build_entity_table()
        parts.append(features_typology.build(frame, typology_table, entity_table))

        _, typ_categorical = features_typology.feature_names(typology_table)
        categorical.extend(typ_categorical)

    if arm in {"C", "D"}:
        # Multi-hop topology only. Degree-like counts already live in arm B, so this
        # arm has to earn its lift from structure a groupby cannot produce.
        boundaries = splits.load_boundaries()
        graph_table = features_graph.build_graph_table(train, boundaries["train_end"])
        parts.append(features_graph.build(frame, graph_table))

    if arm == "D":
        # node2vec over the SAME graph features_graph built, so the C->D comparison
        # isolates the embeddings rather than confounding them with a different graph.
        seed = config.RANDOM_SEED if n2v_seed is None else n2v_seed
        embedding_table = embeddings.build_embedding_table(
            train, boundaries["train_end"], seed
        )
        parts.append(embeddings.build(frame, embedding_table))

    X = pd.concat(parts, axis=1)

    # A duplicate column name would make LightGBM's feature importance ambiguous and
    # silently shadow one of the two.
    duplicates = X.columns[X.columns.duplicated()].tolist()
    if duplicates:
        raise ValueError(f"duplicate feature columns: {duplicates}")

    # The label must never reach the feature matrix. This is the single mistake that
    # produces a 0.99 PR-AUC and a wasted week, so it is checked rather than assumed.
    leaked = [c for c in X.columns if LABEL in c or "laundering" in c.lower()]
    if leaked:
        raise ValueError(f"label leaked into features: {leaked}")

    y = frame[LABEL].astype("int8")
    return X, y, frame["split"], categorical


if __name__ == "__main__":
    frame = splits.load_transactions()
    for arm in ("A", "B"):
        X, y, split, cat = build_arm(arm, frame)
        mem = X.memory_usage(deep=True).sum() / 1e9
        print(f"arm {arm}: {X.shape[0]:,} rows x {X.shape[1]} features  ({mem:.2f} GB)")
        print(f"  categorical: {len(cat)}  |  null-bearing columns: {int((X.isna().sum() > 0).sum())}")
