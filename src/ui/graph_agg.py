"""Aggregate parallel transactions between the same pair into one weighted edge.

The rendered graph is a directed graph, so many transactions A→B otherwise
collapse into a single line that shows only the *last* transaction's risk and
amount — a dense fan-in/smurf relationship then looks like one thin edge. We
group by (source, target) and keep the transaction count, total amount, and
peak risk, so the edge width/tooltip reflect the real intensity.

Pure pandas (no pyvis/matplotlib) so it is unit-testable.
"""
from __future__ import annotations

import pandas as pd

COLS = ["source_account", "target_account", "amount", "risk_score", "n_tx"]


def aggregate_edges(edges):
    if edges is None or not len(edges):
        return pd.DataFrame(columns=COLS)
    e = edges.copy()
    e["risk_score"] = (pd.to_numeric(e["risk_score"], errors="coerce").fillna(0.0)
                       if "risk_score" in e.columns else 0.0)
    out = (e.groupby(["source_account", "target_account"])
            .agg(amount=("amount", "sum"), risk_score=("risk_score", "max"),
                 n_tx=("amount", "size"))
            .reset_index())
    return out[COLS]
