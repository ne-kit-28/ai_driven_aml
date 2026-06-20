# AML Graph Platform — Handoff / Context Prompt

> Use this as the seed prompt for a fresh chat. It captures the project, repo,
> architecture, what's done, what's left, how to run, and the working conventions.

## 0. What this is
Real-time **anti-money-laundering** platform on a graph of bank transactions.
Pipeline: transactions stream in → features are computed → a **streaming graph
neural net (TGN-lite)** scores each transaction's risk and each account's
**toxicity** (probability it's a dropper/mule) → a **Streamlit dashboard** shows
suspicious accounts, lets an officer **block** them, and an **LLM** writes a
SAR-style explanation. Blocking feeds back: blocked accounts are excluded, so
contaminated legit accounts "recover".

## 1. Repository & credentials
- Repo (private): `https://github.com/ne-kit-28/ai_driven_aml`  (branch `main`)
- GitHub user: `ne-kit-28`
- Token (PAT): **NOT stored here.** Set it locally as an env var, never commit it: `export GH_TOKEN=...` and use `https://ne-kit-28:$GH_TOKEN@github.com/...`.
- Commit author: `ne-kit-28 <ne-kit-28@users.noreply.github.com>`

Push pattern (token used inline, never stored in the remote; commit **title only**, no body):
```bash
git clone https://ne-kit-28:<TOKEN>@github.com/ne-kit-28/ai_driven_aml.git
# edit...
git -c user.name=ne-kit-28 -c user.email=ne-kit-28@users.noreply.github.com commit -am "feat: short title"
git push https://ne-kit-28:<TOKEN>@github.com/ne-kit-28/ai_driven_aml.git HEAD:main
```

## 2. Architecture (live MVP)
```mermaid
flowchart LR
    PROD["producer: legit + persistent fraud"] -->|tx_raw| K[("Kafka")]
    UI["dashboard Streamlit"] -->|blocklist| K
    K -->|Spark streaming| ICE[("Iceberg on MinIO")]
    K -->|blocklist| ICE
    NB["notebook ETL"] --> ICE
    ICE --> NB
    SC["scoring service TGN-lite"] --> ICE
    ICE --> SC
    ICE --> TR["Trino"]
    TR --> UI
    UI -->|GraphRAG evidence| LLM["external LLM"]
    PROD -.->|consumes blocklist, replaces blocked node| PROD

    classDef store fill:#e8f5e9,stroke:#2e7d32;
    classDef svc fill:#e3f2fd,stroke:#1565c0;
    class ICE,K store;
    class PROD,NB,SC,UI,TR,LLM svc;
```

State machine on `banking.transactions.ml_status`: `PENDING → FEATURES_READY → SCORED` (+ `BLOCKED` for blocked-account edges).

## 3. Services (docker-compose, in `infra/`)
| Service | Profile | Port | Role |
|---|---|---|---|
| minio (+minio-init) | core | 9000/9001 | S3 object store (admin/password123) |
| postgres | core | — | Hive Metastore backend |
| hive-metastore | core | 9083 | Iceberg catalog (apache/hive:4.0.0 — NOT 4.0.1) |
| jars-init | core | — | downloads hadoop-aws/jdbc into metastore classpath |
| trino | core | 8080 | SQL over Iceberg (dashboard reads via it) |
| pyspark-notebook | core | 8888/4040 | Spark 3.5.8 + Iceberg 1.11 (token `aml`); runs the ETL notebook |
| dashboard | core | 8501 | Streamlit UI (DASH_SOURCE=trino) |
| spark-master/worker | cluster | 8081/8082 | optional standalone cluster |
| kafka | streaming | 9092 | broker (KRaft); advertised `kafka:9092` |
| producer | streaming | — | continuous synthetic transactions + fraud |
| seed | seed | — | one-shot offline ingest (alt to live; **don't mix with live notebook**) |
| stage2 | stage2 | — | one-shot offline feature pipeline (alt to notebook) |
| scoring | scoring | — | live GNN scoring loop |

Two paths exist — **don't mix them**:
- **Live**: producer → Kafka → notebook (ingest+ETL) → scoring → dashboard.
- **Offline/batch**: seed → stage2 → scoring (parquet/iceberg), no Kafka.

## 4. Iceberg tables (namespace `banking`)
- `transactions` — edges; `ml_status`, `features_matrix`, `risk_score`, `is_fraud`, `typology_id`, `src_opened`, `dst_opened` (ground-truth from synthetic).
- `accounts_state` — nodes; `node_features` (Stage-2), `is_fraud`, `opened_days_ago`. Written **only by ETL**.
- `account_scores` — `toxicity`, `node_embedding`. Written **only by scoring** (decoupled to avoid writer race).
- `scored_transactions` — `risk_score` per tx (scoring append).
- `blocklist` — blocked `account_id` (from `blocklist` Kafka topic).

## 5. Model (TGN-lite)
- Streaming graph net: per-node memory h_v updated by a GRU from incoming-edge messages; edge head → risk p; node head → toxicity. Trained by chronological replay (delayed-message scheme).
- **Feature contract** (`src/ml/features.py`): 12 node features (incl. `in_structuring_ratio` — flags smurf collectors) + 3 edge features; the ETL SQL (`src/features/features.sql`) and the live notebook ETL reproduce them exactly. Edge z-score uses **fixed** stats from `tgnlite_meta.json`.
- **Calibration**: outputs temperature-scaled (`node_temp=4`) for sharp fraud/legit separation.
- Serving semantics: **stateless windowed** — each cycle replays a rolling 30-day window from zeroed memory (matches training), scoring **exactly-once** (anti-join on scored tx_ids). Blocked accounts' edges are excluded from the window, so victims recover next cycle.
- Artifact committed: `src/ml/artifacts/tgnlite.pt` + `tgnlite_meta.json`, trained on the producer distribution (`producer.py --dump-parquet`). Retrain via `src/ml/train_temporal.py`.

## 6. Done
- Lakehouse infra (MinIO+Iceberg+Hive+Trino+Spark), all pinned, verified interop.
- Synthetic generator (hardened: hubs, amount/age overlap, contamination, 4 typologies) + ground truth.
- Stage 2 feature pipeline: SQL (`features.sql`), offline `stage2.py`, and live notebook ETL (`streaming_etl.ipynb`).
- TGN-lite model: train/score, calibrated toxicity, exactly-once stateless windowed live scoring.
- Dashboard: **Investigate** (ego-graph any depth, AI SAR + in/out flow analysis, block), **Suspicious nodes** (live list + graph of most toxic nodes + block), **Monitor** (live suspicious tx), **Verification** (recall/health vs ground truth).
- Live MVP: Kafka producer (persistent fraud + `[INJECT]`/`[REPLACE]` logs, blocklist-aware), Spark streaming ingest + ETL, scoring loop, Trino-backed UI.
- Blocklist feedback loop: dashboard → Kafka `blocklist` → producer replaces blocked node + ETL excludes blocked edges (legit "recovery").
- LLM GraphRAG explanation (OpenAI-compatible, key in `.env`), prompt distinguishes hub/mule and avoids guilt-by-association.

## 7. TODO / known gaps
- **Chain/cluster block** (block a whole fraud chain at once) — built: Investigate → "Block whole chain" (`src/ui/chain_select.py`, legit hubs excluded).
- **Verification "before/after health"** is current-snapshot only (no time series of legit toxicity around a block).
- **Recovery after block** — done: scoring is stateless windowed and excludes blocked accounts' edges from the window, so a contaminated legit account recovers the next cycle (no lingering memory). Use `--demo-contamination` for a visible demo.
- Spark/Kafka/Iceberg live path validated by replaying the streaming scorer offline (live node ROC-AUC ≈ 1.0); still recommend a full on-stand soak test.
- Prod scale: scoring is single-process (vertical); for huge volume → Flink keyed-state. Account-state snapshots use overwrite (fine for MVP).
- Versions to re-verify against registries before prod: `pyiceberg==0.8.1`, `duckdb==1.1.3`, `trino==0.330.0`, `kafka-python==2.0.2`.
- `.env` (LLM key) is gitignored; only `.env.example` is committed.

## 8. Run (full reset → live)
```bash
cd infra
docker compose down -v                                    # wipe everything
docker compose --profile core --profile streaming up -d --build
docker compose logs -f hive-metastore                      # wait "Starting Hive Metastore Server"
# Jupyter http://localhost:8888 (token aml): open src/etl/streaming_etl.ipynb and run the cells
#   1 -> 1b(reset) -> 2(DDL) -> 3(tx stream) -> 3b(blocklist stream) -> 4(ETL loop)
docker compose --profile scoring up -d --build scoring
# UI http://localhost:8501  (Investigate / Suspicious nodes / Monitor / Verification)
# LLM: cp .env.example .env, set LLM_API_KEY/LLM_BASE_URL/LLM_MODEL, rebuild dashboard
```
Offline-only demo (no stand internals): dashboard works on shipped `data/scored_*.parquet` with `DASH_SOURCE=parquet`.

## 9. File map
- `infra/docker-compose.yml`, `infra/{trino,hive,spark,pyspark}/…` — stack
- `src/ingest/producer.py` — Kafka producer (persistent fraud, blocklist-aware)
- `src/etl/streaming_etl.ipynb` — notebook (Spark streaming ingest + ETL)
- `src/features/{features.sql,stage2.py}` — offline feature pipeline
- `src/ml/{features.py,stream_model.py,train_temporal.py,score_export.py,infer_stream.py}` — model + scoring
- `src/ml/artifacts/tgnlite.pt(+meta)` — trained model
- `src/ui/{app.py,graph_query.py,llm_explain.py,Dockerfile,requirements.txt}` — dashboard
- `docs/ADD_v2.md` — architecture design doc

## 10. Working conventions (IMPORTANT for the agent)
- **Commit messages: title only**, no long body (`feat: …` / `fix: …`).
- After changes: name which files changed + push.
- **Supply chain**: pin every dependency to a current stable version ≥1 week old; verify against the registry, never from memory.
- Dev sandbox has **no Docker** → can't run the stack; verify Python logic locally (parquet/DuckDB), flag Spark/Kafka/Iceberg code as on-stand-only.
- Rebuild rule: edits in `src/ui/*`→`build dashboard`; `src/ml/*`→`build scoring`; `src/features/*`→`build stage2`; `src/ingest/*`→`build producer`; compose/env-only→just `up -d`.
- Schema changed in notebook → run reset cell (1b) + DDL, or `down -v`.
- Two pipelines (live vs offline) must not be mixed (different `transactions` schema/state).
