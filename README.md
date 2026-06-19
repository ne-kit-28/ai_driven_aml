# AML Graph Platform

Real-time anti-money-laundering on a graph of bank transactions. Transfers
stream in as edges between accounts, a streaming graph neural network scores
each transfer and each account, and an analyst console surfaces the riskiest
accounts, blocks them, and drafts a SAR-style explanation. Blocking feeds back
into the graph, so accounts wrongly dragged up by a bad neighbor recover over
the next few cycles.

## Features

- Streaming graph scoring with **TGN-lite**: per-account memory updated by a GRU
  from incoming transfers; one head scores transaction risk, another scores
  account toxicity (dropper/mule probability).
- Calibrated outputs (temperature scaling) so risk propagates instead of
  pinning at 1.0.
- Exactly-once serving: scored transactions are anti-joined on their ids;
  account memory grows as new accounts appear.
- Human-in-the-loop console: ego-graph exploration, in/out flow analysis, one
  click block, live monitoring, and recall against ground truth.
- Blocklist feedback loop that lets contaminated legitimate accounts recover;
  block a single account or a whole fraud chain at once (legit hubs excluded).
- LLM explanations over the local subgraph (GraphRAG) that separate hubs from
  mules and avoid guilt-by-association.

## Architecture

```mermaid
flowchart LR
  P["Producer (synthetic generator)"] -->|tx_raw| K["Kafka"]
  K -->|Spark streaming| ETL["Notebook ETL (5 min)"]
  ETL -->|features| ICE[("Iceberg on MinIO")]
  ICE --> SC["Scoring (TGN-lite)"]
  SC -->|risk + toxicity| ICE
  ICE --> TR["Trino (SQL)"]
  TR --> DASH["Streamlit dashboard"]
  DASH -->|GraphRAG| LLM["External LLM (SAR)"]
  DASH -->|blocklist| K
  K -.->|blocked nodes| P
```

The data layer is a lakehouse: MinIO for storage, Iceberg tables, a Hive
Metastore catalog, Trino for SQL, and Spark for the ETL. Transactions move
through `PENDING → FEATURES_READY → SCORED`, with `BLOCKED` reserved for
feedback edges. Two paths exist and must not be mixed:

- **Live** — producer → Kafka → 5-minute notebook ETL → scoring loop → dashboard.
- **Offline** — seed → feature pipeline → scoring over Parquet/Iceberg, no Kafka.

## Quickstart

```bash
cd infra
docker compose down -v
docker compose --profile core --profile streaming up -d --build
# run src/etl/streaming_etl.py in the notebook (http://localhost:8888)
docker compose --profile scoring up -d --build scoring
# dashboard: http://localhost:8501
```

For an offline demo without the stack, point the dashboard at the shipped
`data/scored_*.parquet` with `DASH_SOURCE=parquet`.

## Layout

| Path | Contents |
|------|----------|
| `infra/` | docker-compose, Trino, Hive, Spark, PySpark |
| `src/ingest/` | Kafka producer (synthetic generator) |
| `src/etl/`, `src/features/` | streaming and batch feature pipelines |
| `src/ml/` | model, training, scoring, artifacts |
| `src/ui/` | Streamlit app, graph queries, LLM explainer |
| `docs/` | architecture notes |

Models train on a synthetic generator covering four laundering typologies, with
ground-truth labels for evaluation.

## Status

The offline (Parquet/DuckDB) path is verified. The Spark/Kafka/Iceberg live
path is built but still needs on-stand testing. Open items: time-series
verification, gradual memory decay after a block, and horizontal scale-out for
high throughput.
