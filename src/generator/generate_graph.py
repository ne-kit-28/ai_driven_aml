"""
Synthetic AML transaction-graph generator with ground-truth labels.

Produces two Parquet tables (edge/node split — see ADD section "node vs edge"):
  - accounts.parquet     : nodes  (account_id, opened_days_ago, is_fraud, fraud_role, typology_id)
  - transactions.parquet : edges  (tx_id, src, dst, amount, ts, typology_id, is_fraud, ml_status)

Injected typologies (each with a unique typology_id and ground-truth labels):
  T1 transit_ring     : money cycles back through a closed loop of mules
  T2 smurfing         : one source -> many small structured transfers -> one aggregator
  T3 fan_in_cashout   : many droppers -> single collector -> crypto exchange cash-out
  T4 layering_chain   : long linear chain (the "100 droppers" stress case)

Determinism: seeded via a CLI arg (no Math.random / wall-clock dependency).
"""
from __future__ import annotations
import argparse
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class World:
    rng: random.Random
    nprng: np.random.Generator
    accounts: dict = field(default_factory=dict)   # id -> node attrs
    edges: list = field(default_factory=list)       # edge dicts
    _aid: int = 0
    _tid: int = 0
    base_ts: int = 1_700_000_000                     # fixed epoch seconds (deterministic)

    def new_account(self, fraud=False, role="legit", typology_id=None, fresh=False):
        self._aid += 1
        aid = f"ACC{self._aid:07d}"
        opened = self.rng.randint(0, 5) if fresh else self.rng.randint(30, 2000)
        self.accounts[aid] = dict(
            account_id=aid, opened_days_ago=opened,
            is_fraud=int(fraud), fraud_role=role, typology_id=typology_id,
        )
        return aid

    def tx(self, src, dst, amount, day, typology_id=None, fraud=False):
        self._tid += 1
        ts = self.base_ts + day * 86400 + self.rng.randint(0, 86399)
        self.edges.append(dict(
            tx_id=f"TX{self._tid:09d}", source_account=src, target_account=dst,
            amount=round(float(amount), 2), ts=ts,
            typology_id=typology_id, is_fraud=int(fraud), ml_status="PENDING",
        ))


def inject_background(w: World, n_accounts: int, n_tx: int, horizon_days: int):
    ids = [w.new_account() for _ in range(n_accounts)]
    for _ in range(n_tx):
        s, d = w.rng.sample(ids, 2)
        amt = float(np.clip(w.nprng.lognormal(6.0, 1.1), 5, 50_000))
        w.tx(s, d, amt, w.rng.randint(0, horizon_days))
    return ids


def inject_transit_ring(w: World, tid, ring_size, day, amount):
    ring = [w.new_account(fraud=True, role="mule", typology_id=tid) for _ in range(ring_size)]
    for i in range(ring_size):
        s, d = ring[i], ring[(i + 1) % ring_size]
        amt = amount * w.rng.uniform(0.9, 0.99)  # small skim each hop
        w.tx(s, d, amt, day + i, typology_id=tid, fraud=True)
    return ring


def inject_smurfing(w: World, tid, n_smurfs, day, total):
    src = w.new_account(fraud=True, role="originator", typology_id=tid, fresh=True)
    agg = w.new_account(fraud=True, role="aggregator", typology_id=tid, fresh=True)
    smurfs = [w.new_account(fraud=True, role="smurf", typology_id=tid, fresh=True) for _ in range(n_smurfs)]
    for s in smurfs:
        part = total / n_smurfs * w.rng.uniform(0.85, 1.0)
        structured = min(part, 9_500 * w.rng.uniform(0.9, 0.99))  # just under reporting threshold
        w.tx(src, s, structured, day, typology_id=tid, fraud=True)
        w.tx(s, agg, structured * 0.99, day + w.rng.randint(0, 2), typology_id=tid, fraud=True)
    return [src, agg, *smurfs]


def inject_fan_in_cashout(w: World, tid, n_droppers, day, amount):
    collector = w.new_account(fraud=True, role="collector", typology_id=tid, fresh=True)
    exchange = w.new_account(fraud=True, role="crypto_exchange", typology_id=tid)
    droppers = [w.new_account(fraud=True, role="dropper", typology_id=tid, fresh=True) for _ in range(n_droppers)]
    for dr in droppers:
        w.tx(dr, collector, amount * w.rng.uniform(0.8, 1.2), day + w.rng.randint(0, 3),
             typology_id=tid, fraud=True)
    total = sum(e["amount"] for e in w.edges if e["target_account"] == collector)
    w.tx(collector, exchange, total * 0.98, day + 4, typology_id=tid, fraud=True)
    return [collector, exchange, *droppers]


def inject_layering_chain(w: World, tid, length, day, amount):
    chain = [w.new_account(fraud=True, role="layer_mule", typology_id=tid, fresh=True) for _ in range(length)]
    for i in range(length - 1):
        amt = amount * (0.995 ** i)
        w.tx(chain[i], chain[i + 1], amt, day + i, typology_id=tid, fraud=True)
    return chain


def build(seed: int, scale: float):
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)
    w = World(rng=rng, nprng=nprng)
    horizon = 30

    inject_background(w, int(2000 * scale), int(20000 * scale), horizon)

    t = 0
    for _ in range(int(8 * scale)):
        t += 1; inject_transit_ring(w, f"T1_ring_{t}", rng.randint(4, 8), rng.randint(0, horizon), rng.uniform(5e4, 2e5))
    for _ in range(int(6 * scale)):
        t += 1; inject_smurfing(w, f"T2_smurf_{t}", rng.randint(8, 20), rng.randint(0, horizon), rng.uniform(8e4, 3e5))
    for _ in range(int(6 * scale)):
        t += 1; inject_fan_in_cashout(w, f"T3_fanin_{t}", rng.randint(5, 15), rng.randint(0, horizon), rng.uniform(3e3, 1.5e4))
    # The stress case: long layering chains ("100 droppers")
    for _ in range(int(3 * scale)):
        t += 1; inject_layering_chain(w, f"T4_chain_{t}", rng.randint(20, 100), rng.randint(0, horizon), rng.uniform(5e4, 2e5))

    accounts = pd.DataFrame(w.accounts.values())
    txs = pd.DataFrame(w.edges).sort_values("ts").reset_index(drop=True)
    return accounts, txs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", type=float, default=1.0, help="multiplier on volume")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    accounts, txs = build(args.seed, args.scale)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    accounts.to_parquet(out / "accounts.parquet", index=False)
    txs.to_parquet(out / "transactions.parquet", index=False)

    n_fraud_tx = int(txs.is_fraud.sum())
    print(f"accounts      : {len(accounts):>8}  (fraud nodes: {int(accounts.is_fraud.sum())})")
    print(f"transactions  : {len(txs):>8}  (fraud edges: {n_fraud_tx}, {100*n_fraud_tx/len(txs):.2f}%)")
    print(f"typologies     : {sorted(accounts.typology_id.dropna().str.split('_').str[0].unique())}")
    print(f"written to     : {out.resolve()}")


if __name__ == "__main__":
    main()
