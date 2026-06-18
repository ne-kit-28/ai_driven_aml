"""
Scoring microservice — the second deployable service.

Streams FEATURES_READY transactions in chronological micro-batches, scores them
with the trained TGN-lite (same delayed-message logic as score_export, so scores
match), advances per-node memory h_v, and writes results back:
  * per-transaction risk_score  (-> scored_transactions / SCORED)
  * per-account  toxicity + h_v (-> accounts_state snapshot)

IO backends (swappable, model core is IO-agnostic):
  --io parquet : local replay over data/*.parquet           (offline test)
  --io iceberg : incremental read + append via pyiceberg     (the stand)

  python infer_stream.py --io parquet --data data --out data
  python infer_stream.py --io iceberg --loop --interval 60
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from features import build_node_features, build_edge_features
from stream_model import TGNLite

TXCOLS = ["tx_id", "source_account", "target_account", "amount", "ts"]


# ----------------------------- IO backends -----------------------------
class ParquetIO:
    """Local replay: read transactions.parquet in ts order, write scored parquet."""
    def __init__(self, data, out, batch_size):
        self.tx = pd.read_parquet(f"{data}/transactions.parquet").sort_values("ts").reset_index(drop=True)
        self.accounts = pd.read_parquet(f"{data}/accounts.parquet")
        self.out = Path(out); self.bs = batch_size
        self.ts_offset = float(self.tx.ts.min())     # match training's time normalization
        self._scored = []

    def features(self):
        node_x, idx = build_node_features(self.accounts, self.tx)
        return node_x, idx, self.accounts

    def batches(self, loop):
        for i in range(0, len(self.tx), self.bs):
            yield self.tx.iloc[i:i + self.bs][TXCOLS].copy()
        # local data is finite; nothing more to stream

    def write_tx(self, df):
        self._scored.append(df)
        pd.concat(self._scored).to_parquet(self.out / "scored_transactions.parquet", index=False)

    def snapshot_accounts(self, accounts, toxicity, emb):
        a = accounts.assign(toxicity=toxicity, node_embedding=list(emb))
        a.to_parquet(self.out / "scored_accounts.parquet", index=False)


class IcebergIO:
    """Stand: incremental read of FEATURES_READY, append SCORED back. Runs on the lake."""
    def __init__(self, batch_size, bookmark="/state/bookmark.txt"):
        from pyiceberg.catalog import load_catalog
        self.cat = load_catalog("default")        # configured via env (PYICEBERG_CATALOG__*)
        self.tx_t = self.cat.load_table("banking.transactions")
        self.scored_t = self.cat.load_table("banking.scored_transactions")
        self.acc_t = self.cat.load_table("banking.accounts_state")
        self.bs = batch_size
        self.ts_offset = float(self.tx_t.scan().to_pandas().ts.min())  # global time origin
        self.bm = Path(bookmark); self.bm.parent.mkdir(parents=True, exist_ok=True)

    def features(self):
        acc = self.acc_t.scan().to_pandas()        # node_features come from Stage 2 (accounts_state)
        # accounts_state already carries the Stage-2 feature vector; expand to matrix
        node_x = np.stack(acc["node_features"].to_numpy())
        idx = {a: i for i, a in enumerate(acc["account_id"])}
        return node_x.astype(np.float32), idx, acc

    def batches(self, loop):
        from pyiceberg.expressions import EqualTo
        while True:
            df = self.tx_t.scan(row_filter=EqualTo("ml_status", "FEATURES_READY")).to_pandas()
            last = float(self.bm.read_text()) if self.bm.exists() else -1
            df = df[df.ts > last].sort_values("ts")
            for i in range(0, len(df), self.bs):
                yield df.iloc[i:i + self.bs][TXCOLS].copy()
            if not loop:
                break
            time.sleep(self._interval)

    def write_tx(self, df):
        import pyarrow as pa
        self.scored_t.append(pa.Table.from_pandas(df, preserve_index=False))
        self.bm.write_text(str(float(df.ts.max())))

    def snapshot_accounts(self, accounts, toxicity, emb):
        import pyarrow as pa
        snap = accounts.assign(toxicity=toxicity, node_embedding=list(emb),
                               updated_ts=int(time.time()))
        # full-node snapshot -> overwrite keeps one latest row per account (MVP; prod: merge/upsert)
        self.acc_t.overwrite(pa.Table.from_pandas(snap, preserve_index=False))


# ----------------------------- scoring loop -----------------------------
def run(io, artifacts, loop, snapshot_every):
    meta = json.load(open(f"{artifacts}/tgnlite_meta.json"))
    node_x, idx, accounts = io.features()
    mean, std = np.array(meta["node_mean"], np.float32), np.array(meta["node_std"], np.float32)
    x = torch.tensor((node_x - mean) / std)

    model = TGNLite(len(idx), node_x.shape[1], len(meta["edge_feature_names"]), mem=meta["mem"])
    model.load_state_dict(torch.load(f"{artifacts}/tgnlite.pt")); model.eval()

    e_mean, e_std = meta.get("edge_logamt_mean"), meta.get("edge_logamt_std")
    mem = model.memory; prev = None; n_batches = 0
    for batch in io.batches(loop):
        if batch.empty:
            continue
        src = torch.tensor(batch.source_account.map(idx).to_numpy(), dtype=torch.long)
        dst = torch.tensor(batch.target_account.map(idx).to_numpy(), dtype=torch.long)
        ef = torch.tensor(build_edge_features(batch, e_mean, e_std))
        ts = torch.tensor(batch.ts.to_numpy(), dtype=torch.float32)
        ts_n = (ts - io.ts_offset) / 86400.0
        with torch.no_grad():
            mem_in = mem if prev is None else model.updated_memory(mem, *prev, x)
            risk = torch.sigmoid(model.score_edges(mem_in, src, dst, ef, x)).numpy()
            mem = mem_in.detach(); model.memory = mem
            prev = (src, dst, ts_n, ef)
        out = batch.assign(risk_score=risk, ml_status="SCORED")
        io.write_tx(out)
        n_batches += 1
        if n_batches % snapshot_every == 0:
            with torch.no_grad():
                tox = torch.sigmoid(model.score_nodes(model.memory, x)).numpy()
            io.snapshot_accounts(accounts, tox, model.memory.numpy())
        print(f"batch {n_batches}: scored {len(out)} tx | mean risk {risk.mean():.3f}", flush=True)

    with torch.no_grad():
        tox = torch.sigmoid(model.score_nodes(model.memory, x)).numpy()
    io.snapshot_accounts(accounts, tox, model.memory.numpy())
    print(f"done: {n_batches} batches", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--io", choices=["parquet", "iceberg"], default="parquet")
    ap.add_argument("--data", default="data"); ap.add_argument("--out", default="data")
    ap.add_argument("--artifacts", default="src/ml/artifacts")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--loop", action="store_true"); ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--snapshot-every", type=int, default=10)
    args = ap.parse_args()
    if args.io == "parquet":
        io = ParquetIO(args.data, args.out, args.batch_size)
    else:
        io = IcebergIO(args.batch_size); io._interval = args.interval
    run(io, args.artifacts, args.loop, args.snapshot_every)


if __name__ == "__main__":
    main()
