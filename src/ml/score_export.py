"""
Run the trained TGN-lite over the data and export scores:
  scored_transactions.parquet : + risk_score (p per transaction)
  scored_accounts.parquet     : + toxicity   (dropper probability per account)

On the stand these become the SCORED columns in Iceberg; locally they back the
visualization. Uses the same delayed-message replay as serving.

  python src/ml/score_export.py --data data --artifacts src/ml/artifacts --out data
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch

from features import build_node_features, build_edge_features
from stream_model import TGNLite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--artifacts", default="src/ml/artifacts")
    ap.add_argument("--out", default="data"); ap.add_argument("--bs", type=int, default=128)
    args = ap.parse_args()

    meta = json.load(open(f"{args.artifacts}/tgnlite_meta.json"))
    accounts = pd.read_parquet(f"{args.data}/accounts.parquet")
    tx = pd.read_parquet(f"{args.data}/transactions.parquet").sort_values("ts").reset_index(drop=True)

    node_x, idx = build_node_features(accounts, tx)
    mean, std = np.array(meta["node_mean"], np.float32), np.array(meta["node_std"], np.float32)
    x = torch.tensor((node_x - mean) / std)
    src = torch.tensor(tx["source_account"].map(idx).to_numpy(), dtype=torch.long)
    dst = torch.tensor(tx["target_account"].map(idx).to_numpy(), dtype=torch.long)
    ef = torch.tensor(build_edge_features(tx, meta.get("edge_logamt_mean"), meta.get("edge_logamt_std")))
    ts = torch.tensor(tx["ts"].to_numpy(), dtype=torch.float32); ts = (ts - ts.min()) / 86400.0

    model = TGNLite(len(accounts), node_x.shape[1], ef.shape[1], mem=meta["mem"])
    model.load_state_dict(torch.load(f"{args.artifacts}/tgnlite.pt", weights_only=True)); model.eval()
    model.memory.zero_(); model.last_ts.zero_()   # serving semantic: memory starts empty, evolves over the stream

    nt, et = meta.get("node_temp", 1.0), meta.get("edge_temp", 1.0)   # temperature scaling
    probs = torch.zeros(len(tx)); prev = None; mem = model.memory
    with torch.no_grad():
        for i in range(0, len(tx), args.bs):
            b = slice(i, i + args.bs)
            mem_in = mem if prev is None else model.updated_memory(mem, *prev, x)
            probs[b] = torch.sigmoid(model.score_edges(mem_in, src[b], dst[b], ef[b], x) / et)
            mem = mem_in.detach(); model.memory = mem
            prev = (src[b], dst[b], ts[b], ef[b])
        toxicity = torch.sigmoid(model.score_nodes(model.memory, x) / nt).numpy()

    tx = tx.assign(risk_score=probs.numpy())
    accounts = accounts.assign(toxicity=toxicity)
    out = Path(args.out)
    tx.to_parquet(out / "scored_transactions.parquet", index=False)
    accounts.to_parquet(out / "scored_accounts.parquet", index=False)
    print(f"scored {len(tx)} tx | mean risk {tx.risk_score.mean():.3f} | "
          f"accounts {len(accounts)} | mean toxicity {accounts.toxicity.mean():.3f}")
    print("top toxic accounts:")
    print(accounts.nlargest(5, "toxicity")[["account_id", "fraud_role", "toxicity"]].to_string(index=False))


if __name__ == "__main__":
    main()
