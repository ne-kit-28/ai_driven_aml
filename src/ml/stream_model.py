"""
TGN-lite: stateful streaming graph model (correctly trainable).

Per-node memory h_v evolves via a GRU fed by messages from incoming
transactions. Training uses the standard TGN "delayed message" scheme: a batch
is predicted from memory updated by the PREVIOUS batch's events, so the
recurrent params (msg, gru) receive gradient AND there is no future leakage
(an edge never participates in building the memory that scores it).

Heads:
  * edge head  -> p(transaction is fraud)        [drives ml_status SCORED]
  * node head  -> p(account is a dropper/mule)    [drives the UI toxicity colour]

Persistent state (the graph): memory [N, mem], last_ts [N]  (-> accounts_state).
"""
from __future__ import annotations
import torch
import torch.nn as nn


class TGNLite(nn.Module):
    def __init__(self, n_nodes: int, node_dim: int, edge_dim: int, mem: int = 64):
        super().__init__()
        self.n_nodes, self.mem_dim = n_nodes, mem
        self.msg = nn.Sequential(nn.Linear(mem + node_dim + edge_dim + 1, mem), nn.ReLU())
        self.gru = nn.GRUCell(mem, mem)
        self.edge_head = nn.Sequential(
            nn.Linear(mem * 2 + node_dim * 2 + edge_dim, mem), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(mem, 1))
        self.node_head = nn.Sequential(
            nn.Linear(mem + node_dim, mem), nn.ReLU(), nn.Linear(mem, 1))
        self.register_buffer("memory", torch.zeros(n_nodes, mem))
        self.register_buffer("last_ts", torch.zeros(n_nodes))

    def reset_state(self):
        self.memory.zero_(); self.last_ts.zero_()

    def updated_memory(self, prev_mem, src, dst, ts, ef, x):
        """Differentiable: advance receivers' memory from a batch of events."""
        dt = ((ts - self.last_ts[dst]).clamp(min=0) / 30.0).unsqueeze(1)
        m = self.msg(torch.cat([prev_mem[src], x[src], ef, dt], dim=1))
        agg = torch.zeros_like(prev_mem); cnt = torch.zeros(self.n_nodes, 1, device=m.device)
        agg.index_add_(0, dst, m); cnt.index_add_(0, dst, torch.ones_like(dt))
        touched = dst.unique()
        new = prev_mem.clone()
        new[touched] = self.gru(agg[touched] / cnt[touched].clamp(min=1), prev_mem[touched])
        self.last_ts[dst] = ts.detach()
        return new

    def score_edges(self, mem, src, dst, ef, x):
        z = torch.cat([mem[src], mem[dst], x[src], x[dst], ef], dim=1)
        return self.edge_head(z).squeeze(-1)

    def score_nodes(self, mem, x):
        return self.node_head(torch.cat([mem, x], dim=1)).squeeze(-1)   # dropper logit per account
