"""
Train TGN-lite by chronological replay with the delayed-message scheme so the
recurrent memory module actually learns. Multitask: edge fraud + node dropper.

  python src/ml/train_temporal.py --data data --out src/ml/artifacts
  python src/ml/train_temporal.py --ablate-node-features   # structure/temporal only
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

from features import build_node_features, build_edge_features, NODE_FEATURE_NAMES, EDGE_FEATURE_NAMES
from stream_model import TGNLite


def replay(model, x, src, dst, ef, ts, y_edge, ynode_t, order, bs, train,
           opt=None, edge_loss=None, node_loss=None):
    """Delayed-message pass: predict batch i from memory updated by batch i-1."""
    model.train(train)
    probs = torch.zeros(len(order))
    prev = None
    mem = model.memory                       # detached carrier across batches
    for i in range(0, len(order), bs):
        b = order[i:i + bs]
        mem_in = mem if prev is None else model.updated_memory(mem, *prev, x)
        logit = model.score_edges(mem_in, src[b], dst[b], ef[b], x)
        if train:
            loss = edge_loss(logit, y_edge[b])
            nl = model.score_nodes(mem_in, x)            # node toxicity head
            loss = loss + 0.5 * node_loss(nl, ynode_t)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            probs[i:i + bs] = torch.sigmoid(logit).detach()
        mem = mem_in.detach()
        model.memory = mem
        prev = (src[b], dst[b], ts[b], ef[b])
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--out", default="src/ml/artifacts")
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--mem", type=int, default=64); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ablate-node-features", action="store_true",
                    help="zero node features -> only temporal memory carries signal")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    accounts = pd.read_parquet(f"{args.data}/accounts.parquet")
    tx = pd.read_parquet(f"{args.data}/transactions.parquet").sort_values("ts").reset_index(drop=True)

    node_x, idx = build_node_features(accounts, tx)
    mean, std = node_x.mean(0), node_x.std(0) + 1e-6
    x = torch.tensor((node_x - mean) / std)
    if args.ablate_node_features:
        x = torch.zeros_like(x)              # ABLATION: model sees no aggregate features
    src = torch.tensor(tx["source_account"].map(idx).to_numpy(), dtype=torch.long)
    dst = torch.tensor(tx["target_account"].map(idx).to_numpy(), dtype=torch.long)
    _la = np.log1p(tx["amount"].to_numpy()); e_mean, e_std = float(_la.mean()), float(_la.std())
    ef = torch.tensor(build_edge_features(tx, e_mean, e_std))
    ts = torch.tensor(tx["ts"].to_numpy(), dtype=torch.float32); ts = (ts - ts.min()) / 86400.0
    y = torch.tensor(tx["is_fraud"].to_numpy(), dtype=torch.float32)
    ynode = torch.tensor(accounts["is_fraud"].to_numpy(), dtype=torch.float32)   # dropper labels

    n = len(tx); split = int(0.7 * n)
    order = torch.arange(n); tr, te = order[:split], order[split:]

    model = TGNLite(len(accounts), node_x.shape[1], ef.shape[1], mem=args.mem)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=5e-4)   # more reg -> less overconfident
    # cap class weights so logits don't blow up to ±50 (saturated 0/1 scores)
    ew = torch.tensor([min(float((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)), 10.0)])
    nw = torch.tensor([min(float((ynode == 0).sum() / max((ynode == 1).sum(), 1)), 3.0)])
    edge_loss = nn.BCEWithLogitsLoss(pos_weight=ew)
    node_loss = nn.BCEWithLogitsLoss(pos_weight=nw)

    for ep in range(1, args.epochs + 1):
        model.reset_state()
        replay(model, x, src, dst, ef, ts, y, ynode, tr, args.bs, True, opt, edge_loss, node_loss)
        if ep % 5 == 0 or ep == args.epochs:
            model.reset_state()
            p = replay(model, x, src, dst, ef, ts, y, ynode, order, args.bs, False)
            pt, yt = p[te].numpy(), y[te].numpy()
            print(f"epoch {ep:2d} | test edge PR-AUC {average_precision_score(yt, pt):.3f} "
                  f"| ROC-AUC {roc_auc_score(yt, pt):.3f}")

    # final edge metrics + per-typology recall + node(dropper) metrics
    model.reset_state()
    p = replay(model, x, src, dst, ef, ts, y, ynode, order, args.bs, False)
    pt, yt = p[te].numpy(), y[te].numpy()
    k = int(yt.sum()); flagged = set(te[np.argsort(-pt)[:k]].tolist())
    te_df = tx.iloc[te.numpy()].copy(); te_df["flag"] = [i in flagged for i in te.tolist()]
    typ = te_df[te_df.is_fraud == 1].copy(); typ["t"] = typ["typology_id"].str.split("_").str[0]
    with torch.no_grad():
        node_p = torch.sigmoid(model.score_nodes(model.memory, x)).numpy()
    nauc = roc_auc_score(ynode.numpy(), node_p)

    tag = " [ABLATION: no node features]" if args.ablate_node_features else ""
    print(f"\n=== TEST{tag} ===")
    print(f"edge: ROC-AUC {roc_auc_score(yt, pt):.3f} | PR-AUC {average_precision_score(yt, pt):.3f} "
          f"| precision@{k} {yt[np.argsort(-pt)[:k]].mean():.3f} | base {yt.mean():.4f}")
    print("edge recall@k by typology:", typ.groupby('t')['flag'].mean().round(3).to_dict(),
          "| counts:", typ['t'].value_counts().to_dict())
    print(f"node(dropper) ROC-AUC: {nauc:.3f}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if not args.ablate_node_features:
        torch.save(model.state_dict(), out / "tgnlite.pt")
        json.dump({"node_feature_names": NODE_FEATURE_NAMES, "edge_feature_names": EDGE_FEATURE_NAMES,
                   "node_mean": mean.tolist(), "node_std": std.tolist(), "mem": args.mem,
                   "edge_logamt_mean": e_mean, "edge_logamt_std": e_std,
                   "node_temp": 4.0, "edge_temp": 1.0},   # temperature-scale outputs (sharper separation)
                  open(out / "tgnlite_meta.json", "w"), indent=2)
        print(f"saved -> {out}/tgnlite.pt")


if __name__ == "__main__":
    main()
