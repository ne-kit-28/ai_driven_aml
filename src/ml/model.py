"""
Graph fraud model: GraphSAGE encoder (node features -> h_v) + edge-level
scorer (h_src, h_dst, edge_features -> p). Matches ADD: risk on transactions,
derived from the embeddings of the two accounts it connects.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_dim: int, hid: int = 64, out: int = 64, dropout: float = 0.3):
        super().__init__()
        self.c1 = SAGEConv(in_dim, hid)
        self.c2 = SAGEConv(hid, out)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = F.relu(self.c1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.c2(h, edge_index)   # h_v


class EdgeScorer(nn.Module):
    def __init__(self, emb: int, edge_dim: int, hid: int = 64):
        super().__init__()
        # [h_src, h_dst, h_src*h_dst, edge_feats]
        self.mlp = nn.Sequential(
            nn.Linear(emb * 3 + edge_dim, hid), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(hid, 1),
        )

    def forward(self, h, src, dst, edge_feats):
        hs, hd = h[src], h[dst]
        z = torch.cat([hs, hd, hs * hd, edge_feats], dim=1)
        return self.mlp(z).squeeze(-1)   # logit per transaction


class GraphFraudModel(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, emb: int = 64):
        super().__init__()
        self.encoder = GraphSAGEEncoder(node_dim, hid=emb, out=emb)
        self.scorer = EdgeScorer(emb, edge_dim)

    def embed(self, x, mp_edge_index):
        return self.encoder(x, mp_edge_index)

    def forward(self, x, mp_edge_index, src, dst, edge_feats):
        h = self.embed(x, mp_edge_index)
        return self.scorer(h, src, dst, edge_feats), h
