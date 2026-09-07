"""Arm C — multi-hop graph topology.

WHAT MAY AND MAY NOT LIVE HERE
------------------------------
Degree — `n_counterparties_in/out` — is in arm B, because a groupby produces it with
no graph library involved. Arm C therefore contains ONLY quantities a groupby cannot
produce: measures that depend on paths through the graph rather than on an account's
immediate neighbours.

That makes this the harder test of the project's thesis. If arm C lifts PR-AUC over
arm B, the lift is attributable to network structure and nothing else. If it barely
lifts, that is a real finding and gets reported as one.

THE LEAK-FREE RULE, APPLIED TO THE GRAPH ITSELF
-----------------------------------------------
`G_train` is built from training-window transactions only. Validation and test rows
look up their accounts' positions in that graph. This is the failure a time-based
split does NOT prevent: build one graph over all the data and a training row's
PageRank encodes edges that had not yet happened, so the model appears to know a
mule's future counterparties.

THREE ACCOUNT STATES, NOT TWO
-----------------------------
  1. in training AND in the graph      -> full features           (79.4%)
  2. in training, SELF-LOOPS ONLY      -> arm B aggregates, but
                                          no graph position       (20.6%)
  3. not in training at all            -> NaN everywhere          (~0.1% of val/test)

State 2 exists because 18% of training rows are self-transfers (mostly Reinvestment),
and 105,715 accounts did nothing else. Without the `is_in_graph` flag, those accounts
are indistinguishable from never-seen accounts in the feature matrix, and the model
cannot learn that the two mean different things.

Self-loops are dropped from the graph deliberately: they inflate centrality while
carrying no relational information, and `is_self_transaction` already captures the
behaviour in arm A.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import features_txn  # noqa: E402

_F32 = "float32"

GRAPH_CACHE = config.DATA_PROCESSED / "graph_features.parquet"
GRAPH_CACHE_META = config.DATA_PROCESSED / "graph_features_meta.json"


def fx_digest() -> str:
    """Fingerprint of the FX table the graph would be built with.

    Edges are weighted by USD volume and PageRank follows those weights, so the FX
    table is an INPUT to every graph feature — but the cache key was `train_end` and
    `seed` only. Rebuilding under corrected rates would have silently returned the old
    money-weighted graph, and the FX experiment would have measured nothing while
    looking like it measured something.

    Fourth time this project has been bitten by a cache key that omitted part of its own
    provenance: the entity join on Day 2, graph features on Day 3, the agent's system
    prompt on Day 6, and now the currency table those same graph features are weighted
    by. The rule has not changed — hash everything the output depends on.
    """
    material = json.dumps(config.require_fx_rates(), sort_keys=True)
    return hashlib.sha256(material.encode()).hexdigest()[:12]


def train_digest(train: pd.DataFrame) -> str:
    """Fingerprint of the exact training rows the graph would be built from.

    The graph IS a function of this frame, and the cache key did not mention it. The
    single-bank experiment (B1) builds arm C from one institution's visible subset —
    same `train_end`, same seed, same FX, different rows — so it would have been handed
    the full inter-bank graph from cache and reported that partial visibility costs
    nothing. The experiment would have produced a clean, publishable, entirely wrong
    number.

    Same lesson as `fx_digest`, found the same day: a cache key has to name every input,
    and "the data" is an input.
    """
    idx = pd.util.hash_pandas_object(train.index, index=False).values
    return hashlib.sha256(idx.tobytes()).hexdigest()[:12]


def _cache_paths(train: pd.DataFrame) -> tuple[Path, Path]:
    """One cached table per (FX table, training frame), so neither the FX diagnostic nor
    the single-bank experiment can be served another run's graph."""
    d = f"{fx_digest()}_{train_digest(train)}"
    return (GRAPH_CACHE.with_name(f"graph_features_{d}.parquet"),
            GRAPH_CACHE_META.with_name(f"graph_features_{d}_meta.json"))

# reverse_pagerank is COMPUTED (below) but deliberately EXCLUDED from arm C.
# Measured at the account level it correlates 0.993 with out-degree, which already
# lives in arm B. On a graph this sparse -- average degree 2.80, 28,326 disconnected
# components, 31.8% of nodes with no in-edges -- the random surfer barely propagates
# and reverse-PageRank collapses onto a degree count. Including it would let arm C
# claim a "multi-hop" lift for a quantity a groupby produces.
#
# pagerank IS kept: it correlates 0.822 with in-degree, which is high but not
# degenerate -- it takes 212,648 distinct values and varies within each degree bucket,
# so it carries information degree does not.
FEATURES = [
    "pagerank",
    "betweenness",
    "core_number",
    "in_cycle",
    "scc_size",
    "wcc_size",
    "louvain_community_size",
]


def build_graph(train: pd.DataFrame) -> nx.DiGraph:
    """Directed money-flow graph over the training window.

    Edges are weighted by USD VOLUME rather than transaction count: what propagates
    through a laundering network is money, so that is what centrality should follow.
    """
    train = features_txn.add_usd_amounts(train)

    # Self-loops carry no relational information and distort every centrality measure.
    edges = train.loc[train["from_id"] != train["to_id"], ["from_id", "to_id", "paid_usd"]]

    # Collapse parallel transactions between the same pair into one weighted edge.
    # 2.5M transactions become ~569k unique directed pairs.
    pairs = (edges.groupby(["from_id", "to_id"], observed=True)["paid_usd"]
             .sum().reset_index(name="usd"))

    return nx.from_pandas_edgelist(
        pairs, "from_id", "to_id", edge_attr="usd", create_using=nx.DiGraph
    )


def compute_graph_features(G: nx.DiGraph) -> pd.DataFrame:
    """Per-account topology table. Roughly 8 minutes, dominated by betweenness."""
    nodes = list(G.nodes)
    table = pd.DataFrame(index=pd.Index(nodes, name="node_id"))

    t0 = time.time()
    # How much money-weighted importance flows TO this account through the network.
    pr = nx.pagerank(G, alpha=0.85, weight="usd")
    table["pagerank"] = pd.Series(pr)
    # PageRank on the reversed graph: how strongly flow CONCENTRATES INTO this
    # account when you follow money backwards. This is the mule signature — a
    # collection point that many paths terminate at.
    rpr = nx.pagerank(G.reverse(copy=True), alpha=0.85, weight="usd")
    table["reverse_pagerank"] = pd.Series(rpr)
    print(f"    pagerank + reverse: {time.time() - t0:.1f}s")

    t0 = time.time()
    # How often this account sits on shortest paths between other accounts — the
    # layering-intermediary measure. Exact betweenness is O(V*E) and will not finish
    # at this scale; networkx approximates it from k sampled pivot nodes.
    bt = nx.betweenness_centrality(G, k=config.BETWEENNESS_K, seed=config.RANDOM_SEED)
    table["betweenness"] = pd.Series(bt)
    print(f"    betweenness (k={config.BETWEENNESS_K}): {time.time() - t0:.1f}s")

    t0 = time.time()
    # Embeddedness in a densely connected region: repeatedly peel nodes with degree
    # < k. A high core number means the account sits inside a mutually-connected
    # cluster, not on the periphery.
    core = nx.core_number(nx.DiGraph(G))
    table["core_number"] = pd.Series(core)

    # A non-trivial strongly connected component means money can return to where it
    # started — the cycle typology, directly. Only ~1.9% of nodes qualify, so this is
    # a sparse, high-signal flag rather than noise.
    scc_size = {}
    for component in nx.strongly_connected_components(G):
        size = len(component)
        for node in component:
            scc_size[node] = size
    table["scc_size"] = pd.Series(scc_size)
    table["in_cycle"] = (table["scc_size"] > 1).astype("int8")

    wcc_size = {}
    for component in nx.weakly_connected_components(G):
        size = len(component)
        for node in component:
            wcc_size[node] = size
    table["wcc_size"] = pd.Series(wcc_size)
    print(f"    core + scc + wcc: {time.time() - t0:.1f}s")

    t0 = time.time()
    # Communities of densely interconnected accounts. A laundering ring IS a
    # community. NOTE: only the SIZE is kept, never the community ID — the ID is an
    # arbitrary integer label with no meaning a tree could transfer from train to
    # validation, and its high cardinality would invite memorisation.
    communities = nx.community.louvain_communities(
        G.to_undirected(), seed=config.RANDOM_SEED, weight="usd"
    )
    community_size = {}
    for community in communities:
        size = len(community)
        for node in community:
            community_size[node] = size
    table["louvain_community_size"] = pd.Series(community_size)
    print(f"    louvain: {time.time() - t0:.1f}s ({len(communities):,} communities)")

    # Keep reverse_pagerank in the cached table for analysis, but arm C consumes
    # only FEATURES. Subsetting happens in build_graph_table.
    return table[FEATURES + ["reverse_pagerank"]].astype(_F32)


def build_graph_table(train: pd.DataFrame, train_end: str,
                      use_cache: bool = True) -> pd.DataFrame:
    """Graph feature table, cached because it costs ~8 minutes to compute.

    Day 2 deliberately declined to cache features, because a stale cache is a silent
    failure mode — exactly the class of bug that produced 100% NaN entity features.
    Graph features are the exception on cost grounds, so the cache carries the
    `train_end` boundary it was built from and is REJECTED on mismatch. Speed without
    reintroducing the staleness bug.
    """
    cache, cache_meta = _cache_paths(train)
    if use_cache and cache.exists() and cache_meta.exists():
        meta = json.loads(cache_meta.read_text())
        if (meta.get("train_end") == train_end
                and meta.get("seed") == config.RANDOM_SEED
                and meta.get("fx_digest") == fx_digest()
                and meta.get("train_digest") == train_digest(train)):
            print(f"  using cached graph features ({meta['n_nodes']:,} nodes)")
            # Subset to FEATURES so an excluded column in an older cache cannot
            # silently re-enter the model.
            return pd.read_parquet(cache)[FEATURES]
        print(f"  cache rejected: built for train_end={meta.get('train_end')} "
              f"fx={meta.get('fx_digest')}, need {train_end} fx={fx_digest()} "
              "— rebuilding")

    print("  building graph from TRAINING window only")
    G = build_graph(train)
    print(f"  graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    table = compute_graph_features(G)

    table.to_parquet(cache)
    table = table[FEATURES]
    cache_meta.write_text(json.dumps({
        "train_end": train_end,
        "seed": config.RANDOM_SEED,
        "fx_digest": fx_digest(),
        "train_digest": train_digest(train),
        "train_rows": int(len(train)),
        "betweenness_k": config.BETWEENNESS_K,
        "n_nodes": int(G.number_of_nodes()),
        "n_edges": int(G.number_of_edges()),
    }, indent=2))
    return table


def build(df: pd.DataFrame, graph_table: pd.DataFrame) -> pd.DataFrame:
    """Attach graph features for both counterparties, with the state-2 flag."""
    out = pd.DataFrame(index=df.index)

    for side, id_col in (("from", "from_id"), ("to", "to_id")):
        joined = graph_table.reindex(df[id_col].to_numpy())
        joined.index = df.index

        # Distinguishes "seen in training but never transacted with anyone else"
        # (state 2) from "never seen at all" (state 3, flagged by from_is_unseen in
        # features_account). Both produce NaN topology; they are not the same thing.
        out[f"{side}_is_in_graph"] = joined.notna().any(axis=1).astype("int8")

        for col in graph_table.columns:
            out[f"{side}_{col}"] = joined[col]

    return out


def feature_names(graph_table: pd.DataFrame) -> list[str]:
    names: list[str] = []
    for side in ("from", "to"):
        names.append(f"{side}_is_in_graph")
        names.extend(f"{side}_{c}" for c in graph_table.columns)
    return names


if __name__ == "__main__":
    import splits

    frame = splits.load_transactions()
    train = splits.train_frame(frame)
    boundaries = splits.load_boundaries()

    table = build_graph_table(train, boundaries["train_end"])
    print(f"\ngraph table: {table.shape[0]:,} accounts x {table.shape[1]} features")
    print(table.describe().round(6).to_string())

    print(f"\naccounts on a cycle: {int(table['in_cycle'].sum()):,} "
          f"({table['in_cycle'].mean() * 100:.2f}%)")
    print(f"betweenness exactly zero: {(table['betweenness'] == 0).mean() * 100:.2f}%")

    feats = build(frame, table)
    print(f"\njoined: {feats.shape[0]:,} rows x {feats.shape[1]} features")
    for name in ("train", "val", "test"):
        mask = frame["split"] == name
        print(f"  {name:6s} sender in graph: {feats.loc[mask, 'from_is_in_graph'].mean() * 100:.2f}%")
