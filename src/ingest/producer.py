"""
Continuous synthetic transaction producer -> Kafka topic `tx_raw`.

v2 — PERSISTENT fraud + blocklist-aware (for the live "block -> health improves" demo):
  * legit trickle (some via hubs);
  * persistent fraud CASES that keep emitting over time and contaminate a fixed
    legit "victim" (so blocking them later visibly cleans the victim up);
  * each case logs `[INJECT] case=... nodes=[...]` (ground truth = typology_id carries case id);
  * consumes Kafka `blocklist`: a blocked account is dropped AND, if it's a fraud
    actor, REPLACED by a fresh account so the scheme adapts (`[REPLACE] ...`).

  python producer.py --dry-run                       # print ticks + simulate a block
  python producer.py --bootstrap kafka:9092 --rate 5
"""
from __future__ import annotations
import argparse, json, random, threading, time

from blocklist import edge_blocked

STRUCT_LO, STRUCT_HI = 9000.0, 9500.0


class Stream:
    def __init__(self, seed=7, n_legit=2500, n_hubs=30, n_victims=40,
                 max_cases=8, case_activity=0.5):
        # mirror the offline generator's scale so the live stream matches training:
        # a large legit majority, a handful of hubs, and only a few active fraud cases.
        self.rng = random.Random(seed)
        self._aid = 0
        self.tid = 0
        self._ages = {}
        self.legit = [self._new_acc() for _ in range(n_legit)]
        self.hubs = [self._new_acc() for _ in range(n_hubs)]
        self.victims = self.rng.sample(self.legit, n_victims)   # legit targets of persistent fraud
        self.demo = False                                       # --demo-contamination (visible recovery)
        self.demo_victims = self.rng.sample(self.legit, 3)      # legit accounts fed structured deposits
        self._demo_rings = {}                                   # victim -> its own small fraud ring
        self.cases = []                 # active persistent fraud cases (bounded)
        self.max_cases = max_cases      # retire the oldest beyond this -> fraud stays a minority
        self.case_activity = case_activity   # prob a case emits on a given tick (burstier, less volume)
        self.blocked = set()
        self.lock = threading.Lock()
        self._case_n = 0
        self.fraud_ids = set()      # ground-truth fraud accounts (case participants, NOT victims)
        # simulated clock: spread transactions over time (like the 30-day offline snapshot)
        # so the ETL rolling window is meaningful and temporal deltas match training.
        self.base_ts = 1_700_000_000
        self.sim_ts = self.base_ts
        # 2 simulated hours per tick keeps per-account counts in a 30-day window at the
        # model's trained scale (out_degree ~10), so live toxicity matches the offline AUC.
        self.sim_step = 7200

    def _new_acc(self, fraud=False, fresh=False):
        self._aid += 1
        a = f"ACC{self._aid:07d}"
        self._ages[a] = self.rng.randint(0, 9) if fresh else self.rng.randint(20, 2000)
        if fraud:
            self.fraud_ids.add(a)      # case participant — the ground-truth fraud label
        return a

    def _msg(self, s, d, amount, case_id=None, fraud=False):
        self.tid += 1
        return {"tx_id": f"TX{int(time.time()*1000)}{self.tid:06d}",
                "source_account": s, "target_account": d,
                "amount": round(float(max(amount, 1.0)), 2),
                "ts": int(self.sim_ts + self.rng.randint(0, max(1, self.sim_step))),
                "typology_id": case_id, "is_fraud": int(fraud),
                "src_opened": self._ages.get(s, 100), "dst_opened": self._ages.get(d, 100)}

    def legit_tx(self):
        if self.rng.random() < 0.25:
            s, d = self.rng.choice(self.legit), self.rng.choice(self.hubs)
            if self.rng.random() < 0.5:
                s, d = d, s
        else:
            s, d = self.rng.sample(self.legit, 2)
        amt = self.rng.lognormvariate(6.2, 1.1)
        if self.rng.random() < 0.05:
            amt = self.rng.uniform(STRUCT_LO, STRUCT_HI)
        return [] if edge_blocked(s, d, self.blocked) else [self._msg(s, d, amt)]

    def open_case(self):
        self._case_n += 1
        kind = self.rng.choice(["fanin", "chain", "ring", "smurf"])
        cid = f"{kind}_{self._case_n}"
        victim = self.rng.choice(self.victims)
        if kind == "fanin":
            accts = [self._new_acc(True, True)] + [self._new_acc(True, True) for _ in range(self.rng.randint(5, 10))]
        elif kind == "chain":
            accts = [self._new_acc(True, True) for _ in range(self.rng.randint(6, 12))]
        elif kind == "ring":
            accts = [self._new_acc(True, True) for _ in range(self.rng.randint(4, 7))]
        else:  # smurf: [src, agg, smurfs...]
            accts = [self._new_acc(True, True), self._new_acc(True, True)] + \
                    [self._new_acc(True, True) for _ in range(self.rng.randint(6, 12))]
        self.cases.append({"id": cid, "kind": kind, "accounts": accts, "victim": victim})
        print(f"[INJECT] case={cid} typology={kind} victim={victim} nodes={accts}", flush=True)
        while len(self.cases) > self.max_cases:      # retire oldest -> bounded fraud population
            old = self.cases.pop(0)
            print(f"[RETIRE] case={old['id']} (stops emitting)", flush=True)

    def _emit_case(self, c):
        out, k, a = [], c["kind"], c["accounts"]

        def tx(s, d, amt):
            if not edge_blocked(s, d, self.blocked):
                out.append(self._msg(s, d, amt, c["id"], fraud=True))

        if k == "fanin":
            coll = a[0]
            for dr in self.rng.sample(a[1:], min(3, len(a) - 1)):
                tx(dr, coll, self.rng.uniform(3e3, 1.5e4))
            if self.rng.random() < 0.4:
                tx(coll, c["victim"], self.rng.uniform(2e4, 8e4))    # cash-out into a legit victim
        elif k == "chain":
            i = self.rng.randrange(len(a) - 1)
            tx(a[i], a[i + 1], self.rng.uniform(5e4, 2e5) * (0.99 ** i))
            if i == len(a) - 2 and self.rng.random() < 0.4:
                tx(a[-1], c["victim"], self.rng.uniform(2e4, 8e4))
        elif k == "ring":
            i = self.rng.randrange(len(a))
            tx(a[i], a[(i + 1) % len(a)], self.rng.uniform(5e4, 1.5e5))
        else:  # smurf
            src, agg = a[0], a[1]
            for sm in self.rng.sample(a[2:], min(3, len(a) - 2)):
                amt = self.rng.uniform(STRUCT_LO, STRUCT_HI)
                tx(src, sm, amt); tx(sm, agg, amt * 0.98)
            if self.rng.random() < 0.3:
                tx(agg, c["victim"], self.rng.uniform(2e4, 8e4))
        return out

    def _demo_tick(self):
        """Visible recovery: each demo victim has its OWN small fraud ring feeding it
        structured deposits. Investigate the victim and 'Block fraud around it' -> the ring
        is removed and only that victim heals (no other victim is touched)."""
        if not self._demo_rings:
            self._demo_rings = {v: [self._new_acc(fraud=True, fresh=True) for _ in range(4)]
                                for v in self.demo_victims}
        out = []
        for dv, ring in self._demo_rings.items():
            for r in ring:
                if not edge_blocked(r, dv, self.blocked):
                    out.append(self._msg(r, dv, self.rng.uniform(STRUCT_LO, STRUCT_HI),
                                         case_id="demo_ring", fraud=True))
        return out

    def tick(self, n_legit=80):
        out = []
        self.sim_ts += self.sim_step             # advance simulated time each tick
        for _ in range(n_legit):                 # legit volume dominates (realistic base rate)
            out += self.legit_tx()
        for c in self.cases:                     # only some cases fire each tick -> burstier, less volume
            if self.rng.random() < self.case_activity:
                out += self._emit_case(c)
        if self.demo:
            out += self._demo_tick()
        return out

    def block(self, acc):
        with self.lock:
            if acc in self.blocked:
                return
            self.blocked.add(acc)
            for c in self.cases:
                if acc in c["accounts"]:
                    new = self._new_acc(True, True)
                    c["accounts"] = [new if x == acc else x for x in c["accounts"]]
                    print(f"[REPLACE] case={c['id']} blocked={acc} new={new}", flush=True)


def consume_blocklist(stream, bootstrap, topic):
    from kafka import KafkaConsumer
    c = KafkaConsumer(topic, bootstrap_servers=bootstrap, auto_offset_reset="latest",
                      value_deserializer=lambda b: json.loads(b.decode()))
    for m in c:
        acc = (m.value or {}).get("account_id")
        if acc:
            print(f"[BLOCK] received {acc}", flush=True); stream.block(acc)


def dump_parquet(s, out, days, n_legit, case_every_days=2.0):
    """Offline: run the SAME stream and write a labelled training set, so the model
    trains on the serving distribution. Produces transactions/accounts.parquet."""
    import os
    import pandas as pd
    K = max(1, int(case_every_days * 86400 / s.sim_step))
    ticks = max(1, int(days * 86400 / s.sim_step))
    edges = []
    for t in range(ticks):
        if t % K == 0:
            s.open_case()
        edges += s.tick(n_legit=n_legit)
    tx = pd.DataFrame(edges); tx["ml_status"] = "PENDING"
    # ground-truth labels: an account is fraud only if it is a CASE PARTICIPANT (s.fraud_ids).
    # legit "victims" receive a fraud cash-out but stay legit — labelling them fraud would teach
    # the model that receiving a large transfer = fraud and break the recovery story.
    fe = tx[tx.is_fraud == 1]
    inc = pd.concat([fe[["source_account", "typology_id"]].rename(columns={"source_account": "account_id"}),
                     fe[["target_account", "typology_id"]].rename(columns={"target_account": "account_id"})])
    typ_map = inc.groupby("account_id").typology_id.first().to_dict()
    ids = sorted(set(tx.source_account) | set(tx.target_account))
    acc = pd.DataFrame({"account_id": ids})
    acc["opened_days_ago"] = acc.account_id.map(lambda a: s._ages.get(a, 100))
    acc["is_fraud"] = acc.account_id.map(lambda a: int(a in s.fraud_ids))
    acc["typology_id"] = acc.account_id.map(lambda a: typ_map.get(a) if a in s.fraud_ids else None)
    acc["fraud_role"] = acc.typology_id.map(lambda v: v.split("_")[0] if isinstance(v, str) else "legit")
    os.makedirs(out, exist_ok=True)
    tx.to_parquet(f"{out}/transactions.parquet", index=False)
    acc.to_parquet(f"{out}/accounts.parquet", index=False)
    print(f"[dump] {len(tx)} tx, {len(acc)} accounts ({acc.is_fraud.mean()*100:.1f}% fraud), "
          f"~{days} sim-days -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", default="kafka:9092")
    ap.add_argument("--topic", default="tx_raw")
    ap.add_argument("--blocklist-topic", default="blocklist")
    ap.add_argument("--rate", type=float, default=80.0, help="legit transactions per tick (the majority)")
    ap.add_argument("--case-every", type=float, default=90.0, help="seconds between new fraud cases")
    ap.add_argument("--max-cases", type=int, default=8, help="max simultaneously active fraud cases")
    ap.add_argument("--sim-step", type=int, default=7200,
                    help="simulated seconds per tick (spreads tx over time for the ETL window)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dump-parquet", help="offline: write a labelled training set to this dir and exit")
    ap.add_argument("--dump-days", type=int, default=45, help="sim-days to generate for the dump")
    ap.add_argument("--demo-contamination", action="store_true",
                    help="feed a few legit accounts structured deposits for a visible block->recovery demo")
    args = ap.parse_args()
    s = Stream(max_cases=args.max_cases)
    s.sim_step = args.sim_step
    s.demo = args.demo_contamination

    if args.dump_parquet:
        dump_parquet(s, args.dump_parquet, args.dump_days, n_legit=int(max(args.rate, 1)))
        return

    if args.dry_run:
        s.open_case(); s.open_case()
        for _ in range(3):
            for m in s.tick()[:4]:
                print(json.dumps(m, ensure_ascii=False))
        target = s.cases[0]["accounts"][1]
        print(f"\n-- simulate blocking {target} --")
        s.block(target)
        print("blocked:", s.blocked)
        return

    from kafka import KafkaProducer
    from kafka.errors import NoBrokersAvailable
    prod = None
    for attempt in range(60):
        try:
            prod = KafkaProducer(bootstrap_servers=args.bootstrap,
                                 value_serializer=lambda v: json.dumps(v).encode(),
                                 api_version_auto_timeout_ms=10000, retries=5)
            break
        except NoBrokersAvailable:
            print(f"[producer] kafka not ready, retry {attempt+1}/60…", flush=True); time.sleep(5)
    if prod is None:
        raise SystemExit("kafka unavailable")
    threading.Thread(target=consume_blocklist, args=(s, args.bootstrap, args.blocklist_topic),
                     daemon=True).start()
    print(f"producing to {args.topic} (rate {args.rate}/s), blocklist <- {args.blocklist_topic}", flush=True)
    last_case = 0.0; sent = 0
    while True:
        if time.time() - last_case >= args.case_every:
            s.open_case(); last_case = time.time()
        for m in s.tick(n_legit=int(max(args.rate, 1))):
            prod.send(args.topic, m); sent += 1
        if sent % 400 < 90:
            print(f"[producer] sent ~{sent}, active cases {len(s.cases)}/{s.max_cases}", flush=True)
        time.sleep(1.0)


if __name__ == "__main__":
    main()
