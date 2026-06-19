"""
Stage 2 — feature pipeline (ELT state machine PENDING -> FEATURES_READY).

Reads raw PENDING transactions + the accounts dimension, runs features.sql
(DuckDB; identical SQL runs in Spark on the stand), and writes:
  accounts_state.node_features   (11 node features, contract = src/ml/features.py)
  transactions.features_matrix   (3 edge features) + ml_status='FEATURES_READY'

Edge z-score uses FIXED training stats from the model meta (not per-batch).

  python stage2.py --io parquet --data data --out data
  python stage2.py --io iceberg
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

SQL = Path(__file__).with_name("features.sql").read_text()
NODE_SQL = SQL.split("-- ---------- EDGE")[0]
NODE_SQL = NODE_SQL[NODE_SQL.index("WITH"):].rstrip().rstrip(";")
EDGE_SQL = SQL[SQL.rindex("SELECT tx_id"):].rstrip().rstrip(";")
NODE_COLS = ["out_degree", "in_degree", "log_out_amount_sum", "log_in_amount_sum",
             "out_amount_mean", "in_amount_mean", "distinct_out_cp", "distinct_in_cp",
             "account_age_days", "structuring_ratio", "log_net_flow_abs"]
EDGE_COLS = ["log_amount", "in_structuring_band", "amount_zscore"]


def compute(tx: pd.DataFrame, acc: pd.DataFrame, logamt_mean: float, logamt_std: float):
    con = duckdb.connect(); con.register("tx", tx); con.register("acc", acc)
    nf = con.execute(NODE_SQL.replace("{TX}", "tx").replace("{ACC}", "acc")).df()
    ef = con.execute(EDGE_SQL.replace("{TX}", "tx")
                     .replace("{LOGAMT_MEAN}", repr(logamt_mean))
                     .replace("{LOGAMT_STD}", repr(logamt_std))).df()
    nf["node_features"] = nf[NODE_COLS].to_numpy(dtype="float64").tolist()
    ef["features_matrix"] = ef[EDGE_COLS].to_numpy(dtype="float64").tolist()
    return nf[["account_id", "node_features"]], ef[["tx_id", "features_matrix"]]


# ---------------- IO ----------------
class ParquetIO:
    def __init__(self, data, out):
        self.tx = pd.read_parquet(f"{data}/transactions.parquet")
        self.acc = pd.read_parquet(f"{data}/accounts.parquet")
        self.out = Path(out)

    def read(self):
        return self.tx, self.acc

    def write(self, node_feat, edge_feat):
        tx = self.tx.merge(edge_feat, on="tx_id", how="left").assign(ml_status="FEATURES_READY")
        acc = self.acc.merge(node_feat, on="account_id", how="left")
        tx.to_parquet(self.out / "transactions.parquet", index=False)
        acc.to_parquet(self.out / "accounts_state_features.parquet", index=False)
        print(f"wrote {len(tx)} tx (FEATURES_READY) + {len(acc)} accounts with node_features")


class IcebergIO:
    def __init__(self):
        from pyiceberg.catalog import load_catalog
        self.cat = load_catalog("default")
        self.tx_t = self.cat.load_table("banking.transactions")
        self.acc_t = self.cat.load_table("banking.accounts_state")

    def read(self):
        from pyiceberg.expressions import EqualTo
        tx = self.tx_t.scan(row_filter=EqualTo("ml_status", "PENDING")).to_pandas()
        return tx, self.acc_t.scan().to_pandas()

    def write(self, node_feat, edge_feat):
        import pyarrow as pa
        tx = self.tx_t.scan().to_pandas().drop(columns=["features_matrix"], errors="ignore") \
                 .merge(edge_feat, on="tx_id", how="left")
        tx["ml_status"] = "FEATURES_READY"
        acc = self.acc_t.scan().to_pandas().drop(columns=["node_features"], errors="ignore") \
                  .merge(node_feat, on="account_id", how="left")
        self.tx_t.overwrite(pa.Table.from_pandas(tx, preserve_index=False))
        self.acc_t.overwrite(pa.Table.from_pandas(acc, preserve_index=False))
        print(f"overwrote transactions ({len(tx)}) + accounts_state ({len(acc)}) with features")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--io", choices=["parquet", "iceberg"], default="parquet")
    ap.add_argument("--data", default="data"); ap.add_argument("--out", default="data")
    ap.add_argument("--meta", default="src/ml/artifacts/tgnlite_meta.json")
    args = ap.parse_args()
    meta = json.load(open(args.meta))
    io = ParquetIO(args.data, args.out) if args.io == "parquet" else IcebergIO()
    tx, acc = io.read()
    node_feat, edge_feat = compute(tx, acc, meta["edge_logamt_mean"], meta["edge_logamt_std"])
    io.write(node_feat, edge_feat)
    print("Stage 2 done.")


if __name__ == "__main__":
    main()
