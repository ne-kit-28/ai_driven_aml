"""
Scoring microservice — adapted for a LIVE, growing graph.

Each cycle it (1) refreshes the account set + node features from the lake,
(2) grows the per-node memory h_v for new accounts (stable append-only index),
(3) scores the new FEATURES_READY batch, (4) appends SCORED + snapshots toxicity.

Model params (msg/gru/heads) are loaded once; the memory buffer is managed
externally so it can grow as new accounts appear.

  python infer_stream.py --io parquet --data data --out data        # one cycle (test)
  python infer_stream.py --io iceberg --loop --interval 60           # live
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
    def __init__(self, data, out, batch_size):
        self.tx = pd.read_parquet(f"{data}/transactions.parquet").sort_values("ts").reset_index(drop=True)
        self.accounts = pd.read_parquet(f"{data}/accounts.parquet")
        self.out = Path(out); self.bs = batch_size
        self.ts_offset = float(self.tx.ts.min())
        self._served = False; self._scored = []

    def account_features(self):
        node_x, idx = build_node_features(self.accounts, self.tx)
        ids = [a for a, _ in sorted(idx.items(), key=lambda kv: kv[1])]
        return ids, node_x, self.accounts

    def poll(self):
        if self._served:
            return pd.DataFrame(columns=TXCOLS)
        self._served = True
        return self.tx[TXCOLS].copy()

    def write_tx(self, df):
        self._scored.append(df)
        pd.concat(self._scored).to_parquet(self.out / "scored_transactions.parquet", index=False)

    def snapshot_accounts(self, accounts, ids, tox, emb):
        t = pd.Series(tox, index=ids)
        a = accounts.assign(toxicity=accounts.account_id.map(t))
        a.to_parquet(self.out / "scored_accounts.parquet", index=False)


class IcebergIO:
    def __init__(self, batch_size, bookmark="/state/bookmark.txt"):
        from pyiceberg.catalog import load_catalog
        self.cat = load_catalog("default")
        self.tx_t = self.cat.load_table("banking.transactions")
        self.scored_t = self.cat.load_table("banking.scored_transactions")
        self.acc_t = self.cat.load_table("banking.accounts_state")
        self.bs = batch_size
        self.ts_offset = 0.0
        self.bm = Path(bookmark); self.bm.parent.mkdir(parents=True, exist_ok=True)

    def account_features(self):
        acc = self.acc_t.scan().to_pandas()
        node_x = np.stack(acc["node_features"].to_numpy()).astype(np.float32)
        return acc["account_id"].tolist(), node_x, acc

    def poll(self):
        from pyiceberg.expressions import EqualTo
        df = self.tx_t.scan(row_filter=EqualTo("ml_status", "FEATURES_READY")).to_pandas()
        last = float(self.bm.read_text()) if self.bm.exists() else -1
        df = df[df.ts > last].sort_values("ts")
        print(f"[reader] FEATURES_READY new (ts>{last:.0f}): {len(df)} rows", flush=True)
        return df[TXCOLS].copy() if len(df) else pd.DataFrame(columns=TXCOLS)

    def write_tx(self, df):
        import pyarrow as pa
        self.scored_t.append(pa.Table.from_pandas(df, preserve_index=False))
        if len(df):
            self.bm.write_text(str(float(df.ts.max())))

    def snapshot_accounts(self, accounts, ids, tox, emb):
        import pyarrow as pa
        from pyiceberg.io.pyarrow import schema_to_pyarrow
        t = dict(zip(ids, tox)); e = {a: emb[i].tolist() for i, a in enumerate(ids)}
        snap = accounts.copy()
        snap["toxicity"] = snap["account_id"].map(t).astype("float64")
        snap["node_embedding"] = snap["account_id"].map(e)
        snap["updated_ts"] = int(time.time())
        # write strictly to the table schema: avoids pyarrow inferring 'null' for all-null columns
        arrow = schema_to_pyarrow(self.acc_t.schema())
        snap = snap[[f.name for f in arrow]]
        self.acc_t.overwrite(pa.Table.from_pandas(snap, schema=arrow, preserve_index=False))


# ----------------------------- live state -----------------------------
class MemoryState:
    """Persistent append-only account index + growable memory/last_ts."""
    def __init__(self, mem_dim):
        self.mem_dim = mem_dim; self.idx = {}
        self.mem = torch.zeros(0, mem_dim); self.last_ts = torch.zeros(0)

    def sync(self, ids, node_x_std):
        new = [a for a in ids if a not in self.idx]
        for a in new:
            self.idx[a] = len(self.idx)
        if new:
            self.mem = torch.cat([self.mem, torch.zeros(len(new), self.mem_dim)], 0)
            self.last_ts = torch.cat([self.last_ts, torch.zeros(len(new))], 0)
        N = len(self.idx)
        x = torch.zeros(N, node_x_std.shape[1])
        for p, a in enumerate(ids):
            x[self.idx[a]] = torch.tensor(node_x_std[p])
        return x, N

    def ensure(self, accts, node_dim):
        new = [a for a in accts if a not in self.idx]
        for a in new:
            self.idx[a] = len(self.idx)
        if new:
            self.mem = torch.cat([self.mem, torch.zeros(len(new), self.mem_dim)], 0)
            self.last_ts = torch.cat([self.last_ts, torch.zeros(len(new))], 0)


def load_model(artifacts, node_dim, edge_dim, mem):
    sd = torch.load(f"{artifacts}/tgnlite.pt", weights_only=True)
    for k in ["memory", "last_ts", "seen"]:
        sd.pop(k, None)
    model = TGNLite(1, node_dim, edge_dim, mem=mem)
    model.load_state_dict(sd, strict=False); model.eval()
    return model


def run(io, artifacts, loop, interval, snapshot_every):
    meta = json.load(open(f"{artifacts}/tgnlite_meta.json"))
    mean, std = np.array(meta["node_mean"], np.float32), np.array(meta["node_std"], np.float32)
    e_mean, e_std = meta.get("edge_logamt_mean"), meta.get("edge_logamt_std")
    nt, et = meta.get("node_temp", 1.0), meta.get("edge_temp", 1.0)
    state = MemoryState(meta["mem"]); model = None; prev = None; cyc = 0

    while True:
        ids, node_x, accounts = io.account_features()
        if model is None:
            model = load_model(artifacts, node_x.shape[1], len(meta["edge_feature_names"]), meta["mem"])
        x, N = state.sync(ids, (node_x - mean) / std)

        batch = io.poll()
        if len(batch):
            batch = batch.sort_values("ts").reset_index(drop=True)
            state.ensure(set(batch.source_account) | set(batch.target_account), node_x.shape[1])
            x, N = state.sync(ids, (node_x - mean) / std)   # covers any just-added ids (zeros)
            model.last_ts, model.n_nodes = state.last_ts, N
            risks = []
            for i in range(0, len(batch), io.bs):     # chronological sub-batches: memory evolves
                ch = batch.iloc[i:i + io.bs]
                src = torch.tensor(ch.source_account.map(state.idx).to_numpy(), dtype=torch.long)
                dst = torch.tensor(ch.target_account.map(state.idx).to_numpy(), dtype=torch.long)
                ef = torch.tensor(build_edge_features(ch, e_mean, e_std))
                ts = (torch.tensor(ch.ts.to_numpy(), dtype=torch.float32) - io.ts_offset) / 86400.0
                with torch.no_grad():
                    mem_in = state.mem if prev is None else model.updated_memory(state.mem, *prev, x)
                    risks.append(torch.sigmoid(model.score_edges(mem_in, src, dst, ef, x) / et).numpy())
                    state.mem = mem_in.detach(); model.last_ts = state.last_ts; model.n_nodes = N
                    prev = (src, dst, ts, ef)
            io.write_tx(batch.assign(risk_score=np.concatenate(risks), ml_status="SCORED"))
            cyc += 1
            if cyc % snapshot_every == 0:
                _snapshot(io, model, state, x, accounts, ids, nt)
            print(f"cycle {cyc}: scored {len(batch)} | accounts {N} | mean risk "
                  f"{float(np.concatenate(risks).mean()):.3f}", flush=True)
        if not loop:
            break
        time.sleep(interval)

    _snapshot(io, model, state, x, accounts, ids, nt)
    print("done.", flush=True)


def _snapshot(io, model, state, x, accounts, ids, nt):
    model.memory, model.n_nodes = state.mem, len(state.idx)
    with torch.no_grad():
        tox_all = torch.sigmoid(model.score_nodes(state.mem, x) / nt).numpy()
    tox = np.array([tox_all[state.idx[a]] for a in ids])
    emb = np.stack([state.mem[state.idx[a]].numpy() for a in ids])
    io.snapshot_accounts(accounts, ids, tox, emb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--io", choices=["parquet", "iceberg"], default="parquet")
    ap.add_argument("--data", default="data"); ap.add_argument("--out", default="data")
    ap.add_argument("--artifacts", default="src/ml/artifacts")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--loop", action="store_true"); ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--snapshot-every", type=int, default=1)
    args = ap.parse_args()
    io = ParquetIO(args.data, args.out, args.batch_size) if args.io == "parquet" \
        else IcebergIO(args.batch_size)
    run(io, args.artifacts, args.loop, args.interval, args.snapshot_every)


if __name__ == "__main__":
    main()
