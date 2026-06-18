"""
Synthetic AML transaction-graph generator with ground-truth labels.

v2 — HARDENED so the benchmark is non-trivial (the easy version scored 1.0
because fraud was separable by a single feature). Hardening levers:
  * legit "hubs" (merchants/exchange/payroll) with high degree -> degree is
    no longer a giveaway for fraud collectors.
  * only a fraction of fraud accounts are fresh -> account age stops separating.
  * fraud amounts overlap legit; some legit accounts transact near the
    structuring band -> structuring_ratio is no longer fraud-exclusive.
  * contamination: mules also send legit-looking txns; legit accounts
    occasionally pay fraud accounts -> the graph is not cleanly separable.
The remaining signal is *structural/temporal* (who pays whom, over time) —
which is exactly what the graph model must learn.

Outputs (edge/node split): accounts.parquet, transactions.parquet.
Typologies: T1 transit_ring, T2 smurfing, T3 fan_in_cashout, T4 layering_chain.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass, field
from pathlib import Path
import random

import numpy as np
import pandas as pd

STRUCT_LO, STRUCT_HI = 9000.0, 9500.0


@dataclass
class World:
    rng: random.Random
    nprng: np.random.Generator
    accounts: dict = field(default_factory=dict)
    edges: list = field(default_factory=list)
    hubs: list = field(default_factory=list)
    legit_ids: list = field(default_factory=list)
    _aid: int = 0
    _tid: int = 0
    base_ts: int = 1_700_000_000

    def new_account(self, fraud=False, role="legit", typology_id=None, fresh=False):
        self._aid += 1
        aid = f"ACC{self._aid:07d}"
        # HARDENED: fraud accounts are only *sometimes* fresh (40%); else aged like legit.
        if fresh and self.rng.random() < 0.4:
            opened = self.rng.randint(0, 9)
        else:
            opened = self.rng.randint(20, 2000)
        self.accounts[aid] = dict(account_id=aid, opened_days_ago=opened,
                                  is_fraud=int(fraud), fraud_role=role, typology_id=typology_id)
        return aid

    def tx(self, src, dst, amount, day, typology_id=None, fraud=False):
        self._tid += 1
        ts = self.base_ts + day * 86400 + self.rng.randint(0, 86399)
        self.edges.append(dict(tx_id=f"TX{self._tid:09d}", source_account=src, target_account=dst,
                               amount=round(float(max(amount, 1.0)), 2), ts=ts,
                               typology_id=typology_id, is_fraud=int(fraud), ml_status="PENDING"))

    def contaminate(self, fraud_acc, day):
        """A legit account interacts with a fraud account (label stays legit edge)."""
        if self.legit_ids and self.rng.random() < 0.5:
            other = self.rng.choice(self.legit_ids)
            amt = float(np.clip(self.nprng.lognormal(6.2, 1.0), 5, 40_000))
            if self.rng.random() < 0.5:
                self.tx(other, fraud_acc, amt, day)   # legit pays the mule
            else:
                self.tx(fraud_acc, other, amt, day)   # mule pays out legitimately (noise)


def _legit_amount(w: World) -> float:
    """Mixture so legit amounts OVERLAP fraud (kills the amount giveaway)."""
    r = w.rng.random()
    if r < 0.06:
        return w.rng.uniform(STRUCT_LO, STRUCT_HI)          # legit near the threshold band
    if r < 0.76:
        return float(np.clip(w.nprng.lognormal(6.0, 1.2), 5, 60_000))     # everyday small
    if r < 0.93:
        return float(np.clip(w.nprng.lognormal(8.6, 0.8), 1000, 80_000))  # business mid
    return float(np.clip(w.nprng.lognormal(10.2, 0.9), 1e4, 3e5))         # large, overlaps fraud


def inject_background(w: World, n_accounts, n_hubs, n_tx, horizon):
    w.legit_ids = [w.new_account() for _ in range(n_accounts)]
    w.hubs = [w.new_account(role="hub") for _ in range(n_hubs)]
    allnodes = w.legit_ids + w.hubs
    # Preferential attachment via a multiset: nodes with more edges get picked more,
    # producing a heavy-tailed degree distribution -> many active legit accounts whose
    # degree/counterparty counts overlap fraud mules. (degree no longer separates.)
    bag = list(allnodes) + [h for h in w.hubs for _ in range(40)]   # hubs pre-seeded as active
    for _ in range(n_tx):
        if w.rng.random() < 0.75:
            s = w.rng.choice(bag); d = w.rng.choice(bag)
            if s == d:
                d = w.rng.choice(allnodes)
        else:
            s, d = w.rng.sample(allnodes, 2)
        w.tx(s, d, _legit_amount(w), w.rng.randint(0, horizon))
        bag.append(s); bag.append(d)


def inject_legit_payroll(w, n):
    """Legit fan-OUT (employer -> employees): mimics smurfing structure, but legit."""
    emp = w.rng.choice(w.legit_ids)
    for _ in range(n):
        w.tx(emp, w.rng.choice(w.legit_ids), float(np.clip(w.nprng.lognormal(8.4, 0.5), 1500, 20_000)),
             w.rng.randint(0, 30))


def inject_legit_merchant(w, n):
    """Legit fan-IN (customers -> merchant): mimics a collector, but legit."""
    mer = w.rng.choice(w.legit_ids)
    for _ in range(n):
        amt = w.nprng.lognormal(7.0, 0.9)
        if w.rng.random() < 0.1:
            amt = w.rng.uniform(STRUCT_LO, STRUCT_HI)
        w.tx(w.rng.choice(w.legit_ids), mer, float(np.clip(amt, 20, 40_000)), w.rng.randint(0, 30))


def inject_transit_ring(w, tid, ring_size, day, amount):
    ring = [w.new_account(fraud=True, role="mule", typology_id=tid, fresh=True) for _ in range(ring_size)]
    for i in range(ring_size):
        amt = amount * w.rng.uniform(0.6, 1.1)            # HARDENED: noisier skim, overlaps legit
        w.tx(ring[i], ring[(i + 1) % ring_size], amt, day + i, typology_id=tid, fraud=True)
        w.contaminate(ring[i], day + i)
    return ring


def inject_smurfing(w, tid, n_smurfs, day, total):
    src = w.new_account(fraud=True, role="originator", typology_id=tid, fresh=True)
    agg = w.new_account(fraud=True, role="aggregator", typology_id=tid, fresh=True)
    smurfs = [w.new_account(fraud=True, role="smurf", typology_id=tid, fresh=True) for _ in range(n_smurfs)]
    for s in smurfs:
        # HARDENED: only ~half are tightly structured; rest are plausibly normal amounts
        if w.rng.random() < 0.5:
            amt = w.rng.uniform(STRUCT_LO, STRUCT_HI)
        else:
            amt = float(np.clip(w.nprng.lognormal(8.6, 0.6), 2000, 30_000))
        w.tx(src, s, amt, day, typology_id=tid, fraud=True)
        w.tx(s, agg, amt * w.rng.uniform(0.9, 0.99), day + w.rng.randint(0, 3), typology_id=tid, fraud=True)
        w.contaminate(s, day)
    return [src, agg, *smurfs]


def inject_fan_in_cashout(w, tid, n_droppers, day, amount):
    collector = w.new_account(fraud=True, role="collector", typology_id=tid, fresh=True)
    exchange = w.rng.choice(w.hubs) if w.hubs else w.new_account(role="hub")  # cash-out via a legit-looking hub
    droppers = [w.new_account(fraud=True, role="dropper", typology_id=tid, fresh=True) for _ in range(n_droppers)]
    tot = 0.0
    for dr in droppers:
        amt = amount * w.rng.uniform(0.5, 1.8)
        w.tx(dr, collector, amt, day + w.rng.randint(0, 4), typology_id=tid, fraud=True)
        tot += amt
        w.contaminate(dr, day)
    w.tx(collector, exchange, tot * w.rng.uniform(0.9, 0.98), day + 5, typology_id=tid, fraud=True)
    return [collector, *droppers]


def inject_layering_chain(w, tid, length, day, amount):
    chain = [w.new_account(fraud=True, role="layer_mule", typology_id=tid, fresh=True) for _ in range(length)]
    for i in range(length - 1):
        amt = amount * (0.99 ** i) * w.rng.uniform(0.8, 1.2)   # HARDENED: per-hop noise
        w.tx(chain[i], chain[i + 1], amt, day + i, typology_id=tid, fraud=True)
        w.contaminate(chain[i], day + i)
    return chain


def build(seed: int, scale: float):
    rng = random.Random(seed); nprng = np.random.default_rng(seed)
    w = World(rng=rng, nprng=nprng); horizon = 30
    inject_background(w, int(2000 * scale), int(30 * scale), int(22000 * scale), horizon)
    # legit look-alikes for fraud structure (defeat the degree/fan-in/fan-out giveaway)
    for _ in range(int(60 * scale)):
        inject_legit_payroll(w, rng.randint(8, 20))
    for _ in range(int(60 * scale)):
        inject_legit_merchant(w, rng.randint(6, 18))
    t = 0
    for _ in range(int(8 * scale)):
        t += 1; inject_transit_ring(w, f"T1_ring_{t}", rng.randint(4, 8), rng.randint(0, horizon), rng.uniform(5e4, 2e5))
    for _ in range(int(6 * scale)):
        t += 1; inject_smurfing(w, f"T2_smurf_{t}", rng.randint(8, 20), rng.randint(0, horizon), rng.uniform(8e4, 3e5))
    for _ in range(int(6 * scale)):
        t += 1; inject_fan_in_cashout(w, f"T3_fanin_{t}", rng.randint(5, 15), rng.randint(0, horizon), rng.uniform(3e3, 1.5e4))
    for _ in range(int(3 * scale)):
        t += 1; inject_layering_chain(w, f"T4_chain_{t}", rng.randint(20, 100), rng.randint(0, horizon), rng.uniform(5e4, 2e5))
    accounts = pd.DataFrame(w.accounts.values())
    txs = pd.DataFrame(w.edges).sort_values("ts").reset_index(drop=True)
    return accounts, txs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    accounts, txs = build(args.seed, args.scale)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    accounts.to_parquet(out / "accounts.parquet", index=False)
    txs.to_parquet(out / "transactions.parquet", index=False)
    nf = int(txs.is_fraud.sum())
    print(f"accounts     : {len(accounts):>7}  (fraud nodes {int(accounts.is_fraud.sum())}, hubs {len(accounts[accounts.fraud_role=='hub'])})")
    print(f"transactions : {len(txs):>7}  (fraud edges {nf}, {100*nf/len(txs):.2f}%)")
    print(f"typologies   : {sorted(accounts.typology_id.dropna().str.split('_').str[0].unique())}")
    print(f"written to   : {out.resolve()}")


if __name__ == "__main__":
    main()
