"""
Feature builders for the graph service.

NOTE: node/edge features are *owned by Stage 2* (the SQL feature pipeline).
This module is the reference implementation of that contract — the graph
service consumes `node_features` / `features_matrix` from Iceberg. Here we
recompute them from the raw parquet so the model can be trained/validated
standalone. When Stage 2 lands, swap the source, keep the column order.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# Order matters: this IS the contract for accounts_state.node_features
NODE_FEATURE_NAMES = [
    "out_degree", "in_degree",
    "log_out_amount_sum", "log_in_amount_sum",
    "out_amount_mean", "in_amount_mean",
    "distinct_out_cp", "distinct_in_cp",
    "account_age_days", "structuring_ratio", "log_net_flow_abs",
    "in_structuring_ratio",   # share of INCOMING in the structuring band — flags smurf collectors
]
EDGE_FEATURE_NAMES = ["log_amount", "in_structuring_band", "amount_zscore"]

STRUCT_LO, STRUCT_HI = 9000.0, 9500.0   # "just under reporting threshold" band


def build_node_features(accounts: pd.DataFrame, tx: pd.DataFrame):
    """Return (feature_matrix [N, F], account_id -> row index)."""
    ids = accounts["account_id"].tolist()
    idx = {a: i for i, a in enumerate(ids)}

    out = tx.groupby("source_account")
    inc = tx.groupby("target_account")
    out_cnt, in_cnt = out.size(), inc.size()
    out_sum, in_sum = out["amount"].sum(), inc["amount"].sum()
    out_mean, in_mean = out["amount"].mean(), inc["amount"].mean()
    out_cp = out["target_account"].nunique()
    in_cp = inc["source_account"].nunique()
    band = tx.assign(b=tx["amount"].between(STRUCT_LO, STRUCT_HI))
    struct = band.groupby("source_account")["b"].mean()
    struct_in = band.groupby("target_account")["b"].mean()   # incoming structuring (smurf collectors)

    a = accounts.set_index("account_id")
    df = pd.DataFrame(index=ids)
    df["out_degree"] = out_cnt.reindex(ids).fillna(0)
    df["in_degree"] = in_cnt.reindex(ids).fillna(0)
    df["log_out_amount_sum"] = np.log1p(out_sum.reindex(ids).fillna(0))
    df["log_in_amount_sum"] = np.log1p(in_sum.reindex(ids).fillna(0))
    df["out_amount_mean"] = out_mean.reindex(ids).fillna(0)
    df["in_amount_mean"] = in_mean.reindex(ids).fillna(0)
    df["distinct_out_cp"] = out_cp.reindex(ids).fillna(0)
    df["distinct_in_cp"] = in_cp.reindex(ids).fillna(0)
    df["account_age_days"] = a["opened_days_ago"].reindex(ids).fillna(0)
    df["structuring_ratio"] = struct.reindex(ids).fillna(0)
    net = in_sum.reindex(ids).fillna(0) - out_sum.reindex(ids).fillna(0)
    df["log_net_flow_abs"] = np.sign(net) * np.log1p(net.abs())
    df["in_structuring_ratio"] = struct_in.reindex(ids).fillna(0)

    return df[NODE_FEATURE_NAMES].to_numpy(dtype=np.float32), idx


def build_edge_features(tx: pd.DataFrame, logamt_mean=None, logamt_std=None) -> np.ndarray:
    # Pass fixed (training) stats at inference; falls back to batch stats for offline full-data builds.
    amt = tx["amount"].to_numpy(dtype=np.float64)
    log_amt = np.log1p(amt)
    band = tx["amount"].between(STRUCT_LO, STRUCT_HI).to_numpy(dtype=np.float32)
    mu = log_amt.mean() if logamt_mean is None else logamt_mean
    sd = log_amt.std() if logamt_std is None else logamt_std
    z = (log_amt - mu) / (sd + 1e-9)
    return np.stack([log_amt, band, z], axis=1).astype(np.float32)
