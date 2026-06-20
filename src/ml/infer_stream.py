"""
Scoring microservice — stateless windowed scoring.

Each cycle it (1) reads the rolling window of recent transactions (the same span
the ETL features over) and the current node features, (2) replays that window
chronologically from a ZEROED memory — exactly like the offline scorer — and
(3) writes risk for the newly-scored transactions (exactly-once) plus a fresh
toxicity snapshot. Resetting memory each cycle keeps serving identical to
training (a single chronological pass), so live scores match the offline AUC
instead of drifting as memory accumulates.

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
from exactly_once import updated_seen
from stream_model import TGNLite

TXCOLS = ["tx_id", "source_account", "target_account", "amount", "ts"]
SCORED_COLS = TXCOLS + ["risk_score", "ml_status"]


# ----------------------------- IO backends -----------------------------
class ParquetIO:
    def __init__(self, data, out, batch_size):
        self.tx = pd.read_parquet(f"{data}/transactions.parquet").sort_values("ts").reset_index(drop=True)
        self.accounts = pd.read_parquet(f"{data}/accounts.parquet")
        self.out = Path(out); self.bs = batch_size
        self._written = set()

    def account_features(self):
        node_x, idx = build_node_features(self.accounts, self.tx)
        ids = [a for a, _ in sorted(idx.items(), key=lambda kv: kv[1])]
        return ids, node_x, self.accounts

    def window_edges(self, window_days=30):
        if not len(self.tx):
            return self.tx
        maxts = self.tx.ts.max()
        w = self.tx[self.tx.ts >= maxts - window_days * 86400]
        return w.sort_values("ts").reset_index(drop=True)[TXCOLS].copy()

    def unscored(self, edges):
        return edges[~edges.tx_id.isin(self._written)] if len(edges) else edges

    def write_tx(self, df):
        self._written |= set(df.tx_id)
        df[SCORED_COLS].to_parquet(self.out / "scored_transactions.parquet", index=False)

    def snapshot_accounts(self, accounts, ids, tox, emb):
        t = pd.Series(tox, index=ids)
        accounts.assign(toxicity=accounts.account_id.map(t)).to_parquet(
            self.out / "scored_accounts.parquet", index=False)


class IcebergIO:
    def __init__(self, batch_size):
        from pyiceberg.catalog import load_catalog
        self.cat = load_catalog("default")
        self.scored_t = self.cat.load_table("banking.scored_transactions")
        self.scores_t = self._scores_table()
        self.bs = batch_size
        try:
            self.seen = set(self.scored_t.scan(selected_fields=("tx_id",)).to_pandas()["tx_id"])
        except Exception:
            self.seen = set()

    def _scores_table(self):
        import pyarrow as pa
        try:
            return self.cat.load_table("banking.account_scores")
        except Exception:
            schema = pa.schema([("account_id", pa.string()), ("toxicity", pa.float64()),
                                ("node_embedding", pa.list_(pa.float64())), ("updated_ts", pa.int64())])
            self.cat.create_namespace_if_not_exists("banking")
            return self.cat.create_table("banking.account_scores", schema=schema)

    def account_features(self):
        acc = self.cat.load_table("banking.accounts_state").scan().to_pandas()
        if not len(acc):
            return [], np.empty((0, 0), np.float32), acc
        node_x = np.stack(acc["node_features"].to_numpy()).astype(np.float32)
        return acc["account_id"].tolist(), node_x, acc

    def window_edges(self, window_days=30):
        from pyiceberg.expressions import In
        tx_t = self.cat.load_table("banking.transactions")            # refresh: see new ETL commits
        df = tx_t.scan(row_filter=In("ml_status", ("FEATURES_READY", "SCORED"))).to_pandas()
        if not len(df):
            return df
        blk = self._blocked()                                          # recovery: drop blocked accounts'
        if blk:                                                        # edges from the replay so their
            df = df[~df.source_account.isin(blk) & ~df.target_account.isin(blk)]  # victims heal next cycle
        maxts = df.ts.max()
        df = df[df.ts >= maxts - window_days * 86400].sort_values("ts").reset_index(drop=True)
        print(f"[reader] window edges: {len(df)} | new: {(~df.tx_id.isin(self.seen)).sum()} | blocked: {len(blk)}", flush=True)
        return df[TXCOLS].copy()

    def _blocked(self):
        try:
            return set(self.cat.load_table("banking.blocklist")
                       .scan(selected_fields=("account_id",)).to_pandas()["account_id"])
        except Exception:
            return set()

    def unscored(self, edges):
        return edges[~edges.tx_id.isin(self.seen)] if len(edges) else edges

    def write_tx(self, df):
        import pyarrow as pa
        self.scored_t.append(pa.Table.from_pandas(df[SCORED_COLS], preserve_index=False))
        self.seen = updated_seen(self.seen, df)

    def snapshot_accounts(self, accounts, ids, tox, emb):
        import pyarrow as pa
        from pyiceberg.io.pyarrow import schema_to_pyarrow
        rows = pd.DataFrame({"account_id": list(ids),
                             "toxicity": np.asarray(tox, dtype="float64"),
                             "node_embedding": [emb[i].tolist() for i in range(len(ids))],
                             "updated_ts": int(time.time())})
        self.scores_t = self.cat.load_table("banking.account_scores")
        arrow = schema_to_pyarrow(self.scores_t.schema())
        self.scores_t.overwrite(pa.Table.from_pandas(rows[[f.name for f in arrow]], schema=arrow,
                                                      preserve_index=False))


def load_model(artifacts, node_dim, edge_dim, mem):
    sd = torch.load(f"{artifacts}/tgnlite.pt", weights_only=True)
    for k in ["memory", "last_ts", "seen"]:
        sd.pop(k, None)
    model = TGNLite(1, node_dim, edge_dim, mem=mem)
    model.load_state_dict(sd, strict=False); model.eval()
    return model


def score_window(model, memdim, ids, node_x_std, edges, e_mean, e_std, nt, et, bs=512):
    """Replay the window from zeroed memory (offline-equivalent). Returns (risks, tox, emb)."""
    idx = {a: i for i, a in enumerate(ids)}
    for a in (set(edges.source_account) | set(edges.target_account)):
        if a not in idx:                       # edge endpoint not yet in accounts_state -> zero features
            idx[a] = len(idx)
    n = len(idx); f = node_x_std.shape[1]
    x = torch.zeros(n, f)
    x[:len(ids)] = torch.tensor(node_x_std)
    src = torch.tensor(edges.source_account.map(idx).to_numpy(), dtype=torch.long)
    dst = torch.tensor(edges.target_account.map(idx).to_numpy(), dtype=torch.long)
    ef = torch.tensor(build_edge_features(edges, e_mean, e_std))
    ts = torch.tensor(edges.ts.to_numpy(), dtype=torch.float32); ts = (ts - ts.min()) / 86400.0

    model.memory = torch.zeros(n, memdim); model.last_ts = torch.zeros(n); model.n_nodes = n
    prev = None; mem = model.memory; probs = torch.zeros(len(edges))
    with torch.no_grad():
        for i in range(0, len(edges), bs):
            b = slice(i, i + bs)
            mem_in = mem if prev is None else model.updated_memory(mem, *prev, x)
            probs[b] = torch.sigmoid(model.score_edges(mem_in, src[b], dst[b], ef[b], x) / et)
            mem = mem_in.detach(); model.memory = mem
            prev = (src[b], dst[b], ts[b], ef[b])
        tox_all = torch.sigmoid(model.score_nodes(model.memory, x) / nt).numpy()
    tox = np.array([tox_all[idx[a]] for a in ids])
    emb = np.stack([model.memory[idx[a]].numpy() for a in ids])
    return probs.numpy(), tox, emb


def run(io, artifacts, loop, interval, snapshot_every, window_days=30, **_):
    meta = json.load(open(f"{artifacts}/tgnlite_meta.json"))
    mean, std = np.array(meta["node_mean"], np.float32), np.array(meta["node_std"], np.float32)
    e_mean, e_std = meta.get("edge_logamt_mean"), meta.get("edge_logamt_std")
    nt, et = meta.get("node_temp", 1.0), meta.get("edge_temp", 1.0)
    model = None; cyc = 0

    while True:
        ids, node_x, accounts = io.account_features()
        edges = io.window_edges(window_days) if ids else []
        if not ids or not len(edges):
            print("[scoring] lake not ready (no accounts/edges yet); waiting…", flush=True)
            if not loop:
                break
            time.sleep(interval); continue
        if model is None:
            model = load_model(artifacts, node_x.shape[1], len(meta["edge_feature_names"]), meta["mem"])

        risks, tox, emb = score_window(model, meta["mem"], ids, (node_x - mean) / std,
                                       edges, e_mean, e_std, nt, et, bs=io.bs)
        new = io.unscored(edges)
        if len(new):
            risk_by_tx = dict(zip(edges.tx_id, risks))
            io.write_tx(new.assign(risk_score=new.tx_id.map(risk_by_tx), ml_status="SCORED"))
        cyc += 1
        if cyc % snapshot_every == 0:
            io.snapshot_accounts(accounts, ids, tox, emb)
        print(f"cycle {cyc}: window {len(edges)} | new scored {len(new)} | accounts {len(ids)} | "
              f"mean risk {float(risks.mean()):.3f} | mean toxicity {float(np.mean(tox)):.3f}", flush=True)
        if not loop:
            break
        time.sleep(interval)
    print("done.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--io", choices=["parquet", "iceberg"], default="parquet")
    ap.add_argument("--data", default="data"); ap.add_argument("--out", default="data")
    ap.add_argument("--artifacts", default="src/ml/artifacts")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--loop", action="store_true"); ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--snapshot-every", type=int, default=1)
    ap.add_argument("--window-days", type=int, default=30,
                    help="rolling window replayed each cycle (match the ETL feature window)")
    args = ap.parse_args()
    io = ParquetIO(args.data, args.out, args.batch_size) if args.io == "parquet" \
        else IcebergIO(args.batch_size)
    run(io, args.artifacts, args.loop, args.interval, args.snapshot_every, args.window_days)


if __name__ == "__main__":
    main()
