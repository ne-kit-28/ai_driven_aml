"""
Continuous synthetic transaction producer -> Kafka topic `tx_raw`.

Emits a trickle of mostly-legit transactions and periodically injects a fraud
burst (fan-in, ring, chain, smurfing). Each message carries the endpoints'
account age so the ETL can build the accounts dimension without a separate feed.

  python producer.py --dry-run -n 8         # print sample messages, no Kafka
  python producer.py --bootstrap kafka:9092 --rate 5   # ~5 tx/sec to Kafka
"""
from __future__ import annotations
import argparse, json, random, time

STRUCT_LO, STRUCT_HI = 9000.0, 9500.0


class Stream:
    def __init__(self, seed=7, n_accounts=400):
        self.rng = random.Random(seed)
        self.t = 0
        self.tid = 0
        self.accounts = {f"ACC{i:07d}": self.rng.randint(20, 2000) for i in range(n_accounts)}
        self.ids = list(self.accounts)

    def _new_fraud_acc(self, fresh=True):
        i = len(self.accounts)
        a = f"ACC{i:07d}"
        self.accounts[a] = self.rng.randint(0, 9) if fresh else self.rng.randint(20, 2000)
        self.ids.append(a)
        return a

    def _msg(self, s, d, amount, fraud=False, typ=None):
        self.tid += 1
        return {"tx_id": f"TX{int(time.time()*1000)}{self.tid:06d}",
                "source_account": s, "target_account": d,
                "amount": round(float(max(amount, 1.0)), 2), "ts": int(time.time()),
                "typology_id": typ, "is_fraud": int(fraud),
                "src_opened": self.accounts[s], "dst_opened": self.accounts[d]}

    def legit(self):
        s, d = self.rng.sample(self.ids, 2)
        amt = self.rng.lognormvariate(6.2, 1.1)
        if self.rng.random() < 0.05:
            amt = self.rng.uniform(STRUCT_LO, STRUCT_HI)
        return [self._msg(s, d, amt)]

    def fraud_burst(self):
        self.t += 1
        kind = self.rng.choice(["fanin", "chain", "smurf"])
        out = []
        if kind == "fanin":
            tid = f"T3_fanin_{self.t}"
            collector = self._new_fraud_acc()
            for _ in range(self.rng.randint(5, 12)):
                dr = self._new_fraud_acc()
                out.append(self._msg(dr, collector, self.rng.uniform(3e3, 1.5e4), True, tid))
            out.append(self._msg(collector, self.rng.choice(self.ids), sum(m["amount"] for m in out) * 0.97, True, tid))
        elif kind == "chain":
            tid = f"T4_chain_{self.t}"
            chain = [self._new_fraud_acc() for _ in range(self.rng.randint(6, 15))]
            amt = self.rng.uniform(5e4, 2e5)
            for i in range(len(chain) - 1):
                out.append(self._msg(chain[i], chain[i + 1], amt * (0.99 ** i) * self.rng.uniform(.8, 1.2), True, tid))
        else:
            tid = f"T2_smurf_{self.t}"
            src, agg = self._new_fraud_acc(), self._new_fraud_acc()
            for _ in range(self.rng.randint(8, 16)):
                sm = self._new_fraud_acc()
                a = self.rng.uniform(STRUCT_LO, STRUCT_HI)
                out += [self._msg(src, sm, a, True, tid), self._msg(sm, agg, a * 0.98, True, tid)]
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", default="kafka:9092")
    ap.add_argument("--topic", default="tx_raw")
    ap.add_argument("--rate", type=float, default=5.0, help="legit tx per second")
    ap.add_argument("--fraud-every", type=float, default=60.0, help="seconds between fraud bursts")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("-n", type=int, default=10, help="dry-run: how many messages to print")
    args = ap.parse_args()
    s = Stream()

    if args.dry_run:
        msgs = []
        while len(msgs) < args.n:
            msgs += s.legit()
        msgs += s.fraud_burst()
        for m in msgs[:args.n] + msgs[-3:]:
            print(json.dumps(m, ensure_ascii=False))
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
            print(f"[producer] kafka not ready, retry {attempt+1}/60…", flush=True)
            time.sleep(5)
    if prod is None:
        raise SystemExit("kafka unavailable after retries")
    print(f"producing to {args.bootstrap}/{args.topic} (rate {args.rate}/s)", flush=True)
    last_fraud = time.time(); sent = 0
    while True:
        batch = s.legit()
        if time.time() - last_fraud >= args.fraud_every:
            batch += s.fraud_burst(); last_fraud = time.time()
            print(f"[producer] fraud burst (+{len(batch)} msgs)", flush=True)
        for m in batch:
            prod.send(args.topic, m); sent += 1
        if sent % 200 < len(batch):
            print(f"[producer] sent ~{sent}", flush=True)
        time.sleep(1.0 / max(args.rate, 0.1))


if __name__ == "__main__":
    main()
