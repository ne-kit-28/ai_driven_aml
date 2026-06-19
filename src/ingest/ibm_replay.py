"""
IBM AML (AMLWorld) -> Kafka `tx_raw` replay (side test source; the synthetic
producer.py is untouched). Reads the Kaggle CSV + the *_Patterns.txt file,
maps to our message schema, and replays in timestamp order.

Place the downloaded files under data/raw/ (mounted at /raw in the container):
  HI-Small_Trans.csv, HI-Small_Patterns.txt   (or LI-/medium/large variants)

  python ibm_replay.py --csv /raw/HI-Small_Trans.csv --patterns /raw/HI-Small_Patterns.txt \
                       --bootstrap kafka:9092 --rate 20
  python ibm_replay.py --csv ... --mode parquet --out data   # dump for retraining
  python ibm_replay.py --csv ... --mode dry-run -n 5
"""
from __future__ import annotations
import argparse, json, re, time
import pandas as pd

# IBM AML transactions header (two "Account" cols -> pandas makes "Account" / "Account.1")
COLS = ["Timestamp", "From Bank", "Account", "To Bank", "Account.1",
        "Amount Received", "Receiving Currency", "Amount Paid", "Payment Currency",
        "Payment Format", "Is Laundering"]


def _native(o):
    return o.item() if hasattr(o, "item") else str(o)   # numpy int64/float64 -> python


def acct(bank, num):
    return f"{bank}_{num}"


def parse_patterns(path):
    """Map (ts, from_acct, to_acct, amount_paid) -> laundering typology, from *_Patterns.txt."""
    typ = {}
    if not path:
        return typ
    cur = None
    rx = re.compile(r"LAUNDERING ATTEMPT.*?-\s*(.+)$", re.I)
    try:
        for line in open(path, encoding="utf-8", errors="ignore"):
            line = line.strip()
            if line.upper().startswith("BEGIN LAUNDERING"):
                m = rx.search(line); cur = (m.group(1).strip().lower().replace(" ", "_") if m else "laundering")
            elif line.upper().startswith("END LAUNDERING"):
                cur = None
            elif cur and "," in line:
                p = [x.strip() for x in line.split(",")]
                if len(p) >= 8:
                    nb = lambda x: str(int(x)) if x.isdigit() else x   # match pandas int-parse of bank
                    key = (p[0], acct(nb(p[1]), p[2]), acct(nb(p[3]), p[4]), round(float(p[7]), 2))
                    typ[key] = cur
    except FileNotFoundError:
        print(f"[ibm] patterns file not found: {path} (typology_id will be generic)", flush=True)
    print(f"[ibm] parsed {len(typ)} pattern-tagged transactions", flush=True)
    return typ


def load(csv, patterns, limit):
    df = pd.read_csv(csv, nrows=limit)
    # normalise column names to known order if header differs
    if "Account.1" not in df.columns and len(df.columns) >= 11:
        df.columns = COLS[:len(df.columns)]
    df["ts"] = (pd.to_datetime(df["Timestamp"]).astype("int64") // 10**9)
    df["source_account"] = df["From Bank"].astype(str) + "_" + df["Account"].astype(str)
    df["target_account"] = df["To Bank"].astype(str) + "_" + df["Account.1"].astype(str)
    df["amount"] = df["Amount Paid"].astype(float)
    df["is_fraud"] = df["Is Laundering"].astype(int)
    df = df.sort_values("ts").reset_index(drop=True)
    typ = parse_patterns(patterns)
    df["typology_id"] = [
        typ.get((str(r.Timestamp), r.source_account, r.target_account, round(r.amount, 2)),
                "laundering" if r.is_fraud else None)
        for r in df.itertuples()]
    df["tx_id"] = ["IBM" + str(i) for i in range(len(df))]
    # account age not in IBM AML -> constant proxy (retrain learns it's uninformative)
    df["src_opened"] = 365; df["dst_opened"] = 365
    return df[["tx_id", "source_account", "target_account", "amount", "ts",
               "typology_id", "is_fraud", "src_opened", "dst_opened"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--patterns", default=None)
    ap.add_argument("--bootstrap", default="kafka:9092")
    ap.add_argument("--topic", default="tx_raw")
    ap.add_argument("--rate", type=float, default=20.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--mode", choices=["kafka", "parquet", "dry-run"], default="kafka")
    ap.add_argument("--out", default="data"); ap.add_argument("-n", type=int, default=5)
    args = ap.parse_args()

    df = load(args.csv, args.patterns, args.limit)
    print(f"[ibm] loaded {len(df)} tx | fraud {int(df.is_fraud.sum())} "
          f"({100*df.is_fraud.mean():.2f}%) | typologies "
          f"{sorted(df.typology_id.dropna().unique())[:8]}", flush=True)

    if args.mode == "dry-run":
        for r in df.head(args.n).to_dict("records"):
            print(json.dumps(r, ensure_ascii=False, default=_native))
        return
    if args.mode == "parquet":
        from pathlib import Path
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out / "transactions.parquet", index=False)
        ids = pd.unique(df[["source_account", "target_account"]].values.ravel())
        acc = pd.DataFrame({"account_id": ids, "opened_days_ago": 365,
                            "is_fraud": 0, "fraud_role": None, "typology_id": None})
        # mark accounts that ever appear in a laundering edge as fraud (ground truth)
        fr = set(df[df.is_fraud == 1].source_account) | set(df[df.is_fraud == 1].target_account)
        acc["is_fraud"] = acc.account_id.isin(fr).astype(int)
        acc.to_parquet(out / "accounts.parquet", index=False)
        print(f"[ibm] wrote {out}/transactions.parquet + accounts.parquet ({len(acc)} accounts)", flush=True)
        return

    from kafka import KafkaProducer
    from kafka.errors import NoBrokersAvailable
    prod = None
    for attempt in range(60):
        try:
            prod = KafkaProducer(bootstrap_servers=args.bootstrap, api_version_auto_timeout_ms=10000,
                                 retries=5, value_serializer=lambda v: json.dumps(v, default=_native).encode()); break
        except NoBrokersAvailable:
            print(f"[ibm] kafka not ready, retry {attempt+1}/60…", flush=True); time.sleep(5)
    if prod is None:
        raise SystemExit("kafka unavailable")
    print(f"[ibm] replaying {len(df)} tx -> {args.topic} at {args.rate}/s", flush=True)
    for i, r in enumerate(df.to_dict("records")):
        prod.send(args.topic, r)
        if i % 500 == 0:
            print(f"[ibm] sent {i}/{len(df)}", flush=True)
        time.sleep(1.0 / max(args.rate, 0.1))
    prod.flush()
    print("[ibm] replay done", flush=True)


if __name__ == "__main__":
    main()
