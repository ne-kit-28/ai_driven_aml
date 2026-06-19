"""Select a whole fraud chain/cluster to block, not just one node.

Given the traced ego-subgraph around an account, return the set of accounts
that form its laundering chain: the connected component reachable from the seed,
restricted to genuinely suspicious nodes (toxicity above threshold) and with
legit cash-out **hubs excluded** so we don't block a shared exchange/merchant
(guilt-by-association guard, per the model card). The seed is always included.

Pure and dependency-light (pandas + networkx, both already UI deps) so it can be
unit-tested without Streamlit, Kafka, or the lake.
"""
from __future__ import annotations

import networkx as nx
import pandas as pd


def _tox(attrs, node):
    if attrs is None or node not in attrs.index:
        return 0.0
    v = attrs.loc[node, "toxicity"] if "toxicity" in attrs.columns else 0.0
    return float(v) if pd.notna(v) else 0.0


def _role(attrs, node):
    if attrs is None or node not in attrs.index or "fraud_role" not in attrs.columns:
        return ""
    v = attrs.loc[node, "fraud_role"]
    return str(v) if pd.notna(v) else ""


def select_chain(edges, attrs, seed, tox_threshold=0.5, exclude_hubs=True):
    """Return the set of account ids forming the fraud chain around ``seed``.

    ``edges`` has ``source_account``/``target_account`` columns; ``attrs`` is
    indexed by account id with ``toxicity`` and ``fraud_role`` columns.
    """
    g = nx.Graph()
    g.add_node(seed)
    if edges is not None and len(edges):
        for s, d in zip(edges["source_account"], edges["target_account"]):
            g.add_edge(s, d)

    chain = {seed}
    for node in nx.node_connected_component(g, seed):
        if node == seed:
            continue
        if exclude_hubs and _role(attrs, node) == "hub":
            continue
        if _tox(attrs, node) >= tox_threshold:
            chain.add(node)
    return chain
