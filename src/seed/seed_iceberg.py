"""
Seed service — stands in for Stage 2 until it's ready.

On startup it (idempotently) creates the Iceberg tables in MinIO and loads
synthetic data in the FEATURES_READY contract, so the scoring service, Trino
and the dashboard can run end-to-end on the lake:

  banking.transactions       — edges, ml_status='FEATURES_READY', features_matrix
  banking.accounts_state     — nodes, node_features (Stage-2 vector), empty h_v
  banking.scored_transactions— empty output table the scoring service appends to

Catalog/S3 come from PYICEBERG_CATALOG__DEFAULT__* env (same as the scoring svc).
  python seed_iceberg.py          # SEED_SCALE env controls volume
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import pyarrow as pa
from pyiceberg.catalog import load_catalog

from generate_graph import build

NS = "banking"


def replace_table(cat, name, tbl: pa.Table):
    ident = f"{NS}.{name}"
    if cat.table_exists(ident):
        cat.drop_table(ident)
    cat.create_table(ident, schema=tbl.schema).append(tbl)
    print(f"  {ident}: {tbl.num_rows} rows")


def ensure_empty(cat, name, schema: pa.Schema):
    ident = f"{NS}.{name}"
    if not cat.table_exists(ident):
        cat.create_table(ident, schema=schema)
        print(f"  {ident}: created (empty)")
    else:
        print(f"  {ident}: exists")


def main():
    scale = float(os.environ.get("SEED_SCALE", "1.0"))
    seed = int(os.environ.get("SEED", "42"))
    cat = load_catalog("default")
    cat.create_namespace_if_not_exists(NS)

    accounts, tx = build(seed, scale)

    # raw edges, PENDING — features_matrix is a placeholder, filled by Stage 2
    tx_tbl = pa.Table.from_pandas(pd.DataFrame({
        "tx_id": tx.tx_id, "source_account": tx.source_account, "target_account": tx.target_account,
        "amount": tx.amount.astype("float64"), "ts": tx.ts.astype("int64"),
        "typology_id": tx.typology_id, "is_fraud": tx.is_fraud.astype("int64"),
        "features_matrix": [[0.0, 0.0, 0.0] for _ in range(len(tx))],
        "risk_score": np.full(len(tx), np.nan),
        "ml_status": "PENDING",
    }), preserve_index=False)

    # accounts dimension — node_features is a placeholder, filled by Stage 2
    acc_tbl = pa.Table.from_pandas(pd.DataFrame({
        "account_id": accounts.account_id, "opened_days_ago": accounts.opened_days_ago.astype("int64"),
        "is_fraud": accounts.is_fraud.astype("int64"), "fraud_role": accounts.fraud_role,
        "typology_id": accounts.typology_id,
        "node_features": [[0.0] * 11 for _ in range(len(accounts))],
        "node_embedding": [[0.0] for _ in range(len(accounts))],
        "toxicity": np.full(len(accounts), np.nan),
        "emb_version": np.zeros(len(accounts), "int64"), "updated_ts": np.zeros(len(accounts), "int64"),
    }), preserve_index=False)

    print("seeding Iceberg tables:")
    replace_table(cat, "transactions", tx_tbl)
    replace_table(cat, "accounts_state", acc_tbl)
    ensure_empty(cat, "scored_transactions", pa.schema([
        ("tx_id", pa.string()), ("source_account", pa.string()), ("target_account", pa.string()),
        ("amount", pa.float64()), ("ts", pa.int64()),
        ("risk_score", pa.float64()), ("ml_status", pa.string())]))
    print("seed done.")


if __name__ == "__main__":
    main()
