"""Arm D — node2vec embeddings over the training-window graph.

WHAT NODE2VEC DOES
------------------
It takes random walks over the graph and treats each walk as a "sentence" of account
IDs, then applies Word2Vec. Accounts appearing in similar walk contexts get similar
vectors. The result is a 64-number summary of each account's STRUCTURAL POSITION that
nobody hand-engineered — the hope being that it captures shapes the explicit features
in arms B and C missed.

`p` and `q` bias the walk. q > 1 keeps walks near their origin, encoding structural
role (is this a hub, a pass-through, a leaf?); q < 1 wanders outward, encoding
community membership. A mule account is defined by its shape of connectivity, so
config sets q = 2.0 to target structural role.

REPRODUCIBILITY: workers=1 IS NOT NEGOTIABLE
--------------------------------------------
gensim's Word2Vec is non-deterministic with multiple worker threads, regardless of
seed, because workers consume training examples in nondeterministic order. Measured on
this graph, two runs with an identical seed differ by up to 0.055 per dimension at
workers=4, and are bit-identical at workers=1.

The repo claims `make all` reproduces every number in the README. Running the headline
arm on a non-reproducible feature set would make that claim false, so this trades
runtime for correctness.

THE SAME LEAK-FREE RULE
-----------------------
Embeddings are trained on `features_graph.build_graph(train)` — the training-window
graph, self-loops removed, weighted by USD. Validation and test rows look up their
accounts' vectors. Accounts absent from the graph get NaN, and `is_in_graph` (added in
arm C) already distinguishes the three account states, so no new flag is needed here.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import features_graph  # noqa: E402


def _cache_paths(seed: int) -> tuple[Path, Path]:
    """Seed-specific cache, so the three-seed variance runs do not overwrite each other."""
    return (
        config.DATA_PROCESSED / f"embeddings_d{config.N2V_DIM}_seed{seed}.parquet",
        config.DATA_PROCESSED / f"embeddings_d{config.N2V_DIM}_seed{seed}_meta.json",
    )


def compute_embeddings(train: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Train node2vec on the training-window graph. Returns one row per account."""
    from fastnode2vec import Graph, Node2Vec

    graph = features_graph.build_graph(train)
    print(f"  graph: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges")

    # Raw USD weights, matching features_graph exactly. Arms C and D must describe the
    # SAME graph, or the comparison confounds embeddings with graph construction.
    # Measured: raw USD gave the strongest standalone signal (2.37x) versus unweighted
    # (2.05x) and log1p (1.97x).
    edges = [(u, v, float(d["usd"])) for u, v, d in graph.edges(data=True)]

    started = time.monotonic()
    n2v = Node2Vec(
        Graph(edges, directed=True, weighted=True, verbose=False),
        dim=config.N2V_DIM,
        walk_length=config.N2V_WALK_LENGTH,
        window=config.N2V_WINDOW,
        p=config.N2V_P,
        q=config.N2V_Q,
        workers=1,  # see module docstring — determinism, not performance
        seed=seed,
    )
    n2v.train(epochs=config.N2V_EPOCHS, verbose=False)
    print(f"  node2vec (dim={config.N2V_DIM}, seed={seed}): "
          f"{time.monotonic() - started:.0f}s")

    keys = list(n2v.wv.key_to_index)
    matrix = np.vstack([n2v.wv[k] for k in keys]).astype("float32")

    coverage = len(keys) / graph.number_of_nodes()
    print(f"  vocabulary: {len(keys):,} accounts ({coverage * 100:.1f}% of graph nodes)")

    table = pd.DataFrame(
        matrix,
        index=pd.Index(keys, name="node_id"),
        columns=[f"n2v_{i:02d}" for i in range(config.N2V_DIM)],
    )
    return table


def build_embedding_table(train: pd.DataFrame, train_end: str, seed: int,
                          use_cache: bool = True) -> pd.DataFrame:
    """Cached embeddings, with the provenance guard used for graph features.

    The cache records the split boundary AND the seed it was built from. A mismatch
    rebuilds rather than silently serving vectors trained on a different graph — the
    failure mode that produced 100% NaN entity features on Day 2.
    """
    cache, meta_path = _cache_paths(seed)

    if use_cache and cache.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if (meta.get("train_end") == train_end
                and meta.get("seed") == seed
                and meta.get("dim") == config.N2V_DIM):
            print(f"  using cached embeddings (seed={seed}, {meta['n_accounts']:,} accounts)")
            return pd.read_parquet(cache)
        print(f"  embedding cache rejected (built for train_end={meta.get('train_end')}, "
              f"seed={meta.get('seed')}, dim={meta.get('dim')}) — rebuilding")

    table = compute_embeddings(train, seed)

    table.to_parquet(cache)
    meta_path.write_text(json.dumps({
        "train_end": train_end,
        "seed": seed,
        "dim": config.N2V_DIM,
        "walk_length": config.N2V_WALK_LENGTH,
        "window": config.N2V_WINDOW,
        "p": config.N2V_P,
        "q": config.N2V_Q,
        "epochs": config.N2V_EPOCHS,
        "workers": 1,
        "n_accounts": int(len(table)),
    }, indent=2))
    return table


def build(df: pd.DataFrame, embedding_table: pd.DataFrame) -> pd.DataFrame:
    """Attach embedding vectors for both counterparties.

    No `is_*` flag is added: vocabulary coverage is 100% of graph nodes, so an account
    has a vector exactly when it is in the graph, which `from_is_in_graph` /
    `to_is_in_graph` from arm C already reports.
    """
    # Built with a single concat rather than 128 individual column assignments.
    # Inserting columns one at a time into a 5M-row frame re-fragments the block
    # manager on every insert, which pandas warns about and which measurably slows
    # this join.
    frames = []
    for side, id_col in (("from", "from_id"), ("to", "to_id")):
        joined = embedding_table.reindex(df[id_col].to_numpy())
        joined.index = df.index
        joined.columns = [f"{side}_{c}" for c in embedding_table.columns]
        frames.append(joined)

    return pd.concat(frames, axis=1)


def feature_names(embedding_table: pd.DataFrame) -> list[str]:
    return [f"{side}_{c}" for side in ("from", "to") for c in embedding_table.columns]


if __name__ == "__main__":
    import argparse

    import splits

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    frame = splits.load_transactions()
    train = splits.train_frame(frame)
    boundaries = splits.load_boundaries()

    table = build_embedding_table(train, boundaries["train_end"], args.seed)
    print(f"\nembedding table: {table.shape[0]:,} accounts x {table.shape[1]} dims")

    features = build(frame, table)
    print(f"joined: {features.shape[0]:,} rows x {features.shape[1]} features")
    missing = features["from_n2v_00"].isna().mean() * 100
    print(f"rows with no sender embedding: {missing:.2f}% "
          "(accounts absent from the graph — self-loop-only or unseen)")
