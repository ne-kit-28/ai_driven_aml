"""
Train the graph fraud model on the synthetic parquet and report quality
per typology. Saves a model artifact for the inference service.

  python src/ml/train.py --data data --out src/ml/artifacts
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import to_undirected

from features import build_node_features, build_edge_features, NODE_FEATURE_NAMES, EDGE_FEATURE_NAMES
from model import GraphFraudModel


def standardize(x, mean, std):
    return (x - mean) / std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="src/ml/artifacts")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--emb", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    accounts = pd.read_parquet(f"{args.data}/accounts.parquet")
    tx = pd.read_parquet(f"{args.data}/transactions.parquet").sort_values("ts").reset_index(drop=True)

    # --- graph tensors ---
    node_x, idx = build_node_features(accounts, tx)
    mean, std = node_x.mean(0), node_x.std(0) + 1e-6
    x = torch.tensor(standardize(node_x, mean, std))
    src = torch.tensor(tx["source_account"].map(idx).to_numpy(), dtype=torch.long)
    dst = torch.tensor(tx["target_account"].map(idx).to_numpy(), dtype=torch.long)
    edge_feats = torch.tensor(build_edge_features(tx))
    y = torch.tensor(tx["is_fraud"].to_numpy(), dtype=torch.float32)
    mp_edge_index = to_undirected(torch.stack([src, dst]))   # message-passing graph

    # --- temporal split (train on earlier txns, test on later) ---
    n = len(tx); tr_end, va_end = int(0.7 * n), int(0.85 * n)
    perm = torch.arange(n)  # already time-sorted
    tr, va, te = perm[:tr_end], perm[tr_end:va_end], perm[va_end:]

    model = GraphFraudModel(node_x.shape[1], edge_feats.shape[1], emb=args.emb)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-5)
    pos_w = torch.tensor([(y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    def evaluate(split):
        model.eval()
        with torch.no_grad():
            logit, _ = model(x, mp_edge_index, src[split], dst[split], edge_feats[split])
            p = torch.sigmoid(logit).numpy()
        yt = y[split].numpy()
        return p, yt

    best_ap = 0.0
    for ep in range(1, args.epochs + 1):
        model.train(); opt.zero_grad()
        logit, _ = model(x, mp_edge_index, src[tr], dst[tr], edge_feats[tr])
        loss = loss_fn(logit, y[tr])
        loss.backward(); opt.step()
        if ep % 20 == 0 or ep == args.epochs:
            p, yt = evaluate(va)
            ap = average_precision_score(yt, p); auc = roc_auc_score(yt, p)
            best_ap = max(best_ap, ap)
            print(f"epoch {ep:3d} | loss {loss.item():.4f} | val PR-AUC {ap:.3f} | val ROC-AUC {auc:.3f}")

    # --- test metrics + recall per typology ---
    p, yt = evaluate(te)
    auc = roc_auc_score(yt, p); ap = average_precision_score(yt, p)
    k = int(yt.sum())                                   # alert budget = #true frauds
    topk = np.argsort(-p)[:k]
    prec_at_k = yt[topk].mean() if k else float("nan")
    flagged = set(np.argsort(-p)[:k].tolist())

    te_df = tx.iloc[te.numpy()].copy(); te_df["p"] = p
    te_df["flagged"] = [i in flagged for i in range(len(te_df))]
    typ = te_df[te_df.is_fraud == 1].copy()
    typ["typ"] = typ["typology_id"].str.split("_").str[0]
    recall_by_typ = typ.groupby("typ")["flagged"].mean().round(3).to_dict()

    print("\n=== TEST ===")
    print(f"ROC-AUC {auc:.3f} | PR-AUC {ap:.3f} | precision@{k} {prec_at_k:.3f}")
    print(f"recall@{k} by typology: {recall_by_typ}")
    print(f"(base fraud rate in test: {yt.mean():.4f})")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "graphsage.pt")
    json.dump({
        "node_feature_names": NODE_FEATURE_NAMES, "edge_feature_names": EDGE_FEATURE_NAMES,
        "node_mean": mean.tolist(), "node_std": std.tolist(),
        "emb": args.emb, "metrics": {"roc_auc": auc, "pr_auc": ap, "precision_at_k": float(prec_at_k)},
    }, open(out / "model_meta.json", "w"), indent=2)
    print(f"\nsaved -> {out}/graphsage.pt + model_meta.json")


if __name__ == "__main__":
    main()
