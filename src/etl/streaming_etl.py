# Live ETL notebook (paste cells into Jupyter at http://localhost:8888).
# Producer -> Kafka(tx_raw) -> [this notebook: land to Iceberg PENDING + 5-min feature ETL]
#            -> scoring service -> Trino -> dashboard.
# NOTE: written for the stand (Spark+Kafka+Iceberg); not runnable in the dev sandbox.

# %% [cell 1] Spark session (Iceberg catalog from spark-defaults + Kafka connector)
from pyspark.sql import SparkSession
spark = (SparkSession.builder.appName("aml-live-etl")
         .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.8")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

# %% [cell 1b] RESET — run ONCE if tables exist with an OLD schema (e.g. after a seed run).
# Drops the banking tables so the DDL below recreates them with the live schema
# (transactions here carries src_opened/dst_opened). WIPES existing data.
for _t in ["transactions", "accounts_state", "scored_transactions", "account_scores", "blocklist"]:
    spark.sql(f"DROP TABLE IF EXISTS iceberg.banking.{_t}")
print("dropped banking tables — run the DDL cell next")

# %% [cell 2] DDL — namespace + tables (idempotent)
spark.sql("CREATE NAMESPACE IF NOT EXISTS iceberg.banking")
spark.sql("""CREATE TABLE IF NOT EXISTS iceberg.banking.transactions (
    tx_id STRING, source_account STRING, target_account STRING, amount DOUBLE, ts BIGINT,
    typology_id STRING, is_fraud INT, src_opened INT, dst_opened INT,
    features_matrix ARRAY<DOUBLE>, risk_score DOUBLE, ml_status STRING)
  USING iceberg PARTITIONED BY (ml_status)
  TBLPROPERTIES ('format-version'='2',
    'write.delete.mode'='merge-on-read', 'write.update.mode'='merge-on-read',
    'write.merge.mode'='merge-on-read',
    'write.merge.isolation-level'='snapshot', 'write.update.isolation-level'='snapshot')""")
spark.sql("""CREATE TABLE IF NOT EXISTS iceberg.banking.accounts_state (
    account_id STRING, opened_days_ago INT, is_fraud INT, fraud_role STRING, typology_id STRING,
    node_features ARRAY<DOUBLE>, node_embedding ARRAY<DOUBLE>, toxicity DOUBLE,
    emb_version INT, updated_ts BIGINT)
  USING iceberg TBLPROPERTIES ('format-version'='2',
    'write.delete.mode'='merge-on-read', 'write.update.mode'='merge-on-read',
    'write.merge.mode'='merge-on-read', 'write.merge.isolation-level'='snapshot')""")
spark.sql("""CREATE TABLE IF NOT EXISTS iceberg.banking.scored_transactions (
    tx_id STRING, source_account STRING, target_account STRING, amount DOUBLE, ts BIGINT,
    risk_score DOUBLE, ml_status STRING) USING iceberg""")
# scoring-owned: toxicity/embedding (decoupled from accounts_state -> no writer race with ETL)
spark.sql("""CREATE TABLE IF NOT EXISTS iceberg.banking.account_scores (
    account_id STRING, toxicity DOUBLE, node_embedding ARRAY<DOUBLE>, updated_ts BIGINT) USING iceberg""")
# blocklist (landed from the `blocklist` Kafka topic) — ETL excludes these accounts' edges
spark.sql("""CREATE TABLE IF NOT EXISTS iceberg.banking.blocklist (
    account_id STRING, reason STRING, ts BIGINT) USING iceberg""")

# %% [cell 3] Kafka -> Iceberg (PENDING): continuous landing, Spark manages offsets
from pyspark.sql.functions import from_json, col, lit, array
SCHEMA = ("tx_id string, source_account string, target_account string, amount double, ts long, "
          "typology_id string, is_fraud int, src_opened int, dst_opened int")
raw = (spark.readStream.format("kafka")
       .option("kafka.bootstrap.servers", "kafka:9092")
       .option("subscribe", "tx_raw").option("startingOffsets", "latest").load())
parsed = raw.select(from_json(col("value").cast("string"), SCHEMA).alias("d")).select("d.*")

def land(batch_df, _):
    (batch_df.withColumn("features_matrix", array().cast("array<double>"))
             .withColumn("risk_score", lit(None).cast("double"))
             .withColumn("ml_status", lit("PENDING"))
             .select("tx_id", "source_account", "target_account", "amount", "ts", "typology_id",
                     "is_fraud", "src_opened", "dst_opened", "features_matrix", "risk_score", "ml_status")
             .writeTo("iceberg.banking.transactions").append())

ingest = (parsed.writeStream.foreachBatch(land)
          .option("checkpointLocation", "/home/jovyan/work/chk/tx_raw")
          .trigger(processingTime="30 seconds").start())
print("Kafka->Iceberg streaming started:", ingest.id)

# %% [cell 3b] Kafka `blocklist` -> Iceberg banking.blocklist (append)
bl_raw = (spark.readStream.format("kafka").option("kafka.bootstrap.servers", "kafka:9092")
          .option("subscribe", "blocklist").option("startingOffsets", "earliest").load())
bl_parsed = bl_raw.select(from_json(col("value").cast("string"),
                          "account_id string, reason string, ts long").alias("d")).select("d.*")
bl_q = (bl_parsed.writeStream.foreachBatch(lambda b, _: b.writeTo("iceberg.banking.blocklist").append())
        .option("checkpointLocation", "/home/jovyan/work/chk/blocklist")
        .trigger(processingTime="20 seconds").start())
print("blocklist streaming started:", bl_q.id)

# %% [cell 4] 5-minute feature ETL loop (mini-Airflow): PENDING -> FEATURES_READY
import json, time
meta = json.load(open("/home/jovyan/work/src/ml/artifacts/tgnlite_meta.json"))
EM, ES = meta["edge_logamt_mean"], meta["edge_logamt_std"]
TX, ACC, BL = ("iceberg.banking.transactions", "iceberg.banking.accounts_state",
               "iceberg.banking.blocklist")
NOTBLK = f"source_account NOT IN (SELECT account_id FROM {BL}) AND target_account NOT IN (SELECT account_id FROM {BL})"

NODE_SRC = f"""
WITH o AS (SELECT source_account acc, count(*) od, sum(amount) os, avg(amount) om,
             count(distinct target_account) doc,
             avg(CASE WHEN amount BETWEEN 9000 AND 9500 THEN 1.0 ELSE 0.0 END) sr
           FROM {TX} WHERE {NOTBLK} GROUP BY source_account),
     i AS (SELECT target_account acc, count(*) idg, sum(amount) isum, avg(amount) im,
             count(distinct source_account) dic FROM {TX} WHERE {NOTBLK} GROUP BY target_account)
SELECT a.account_id, array(
   double(coalesce(o.od,0)), double(coalesce(i.idg,0)),
   ln(1+coalesce(o.os,0)), ln(1+coalesce(i.isum,0)),
   double(coalesce(o.om,0)), double(coalesce(i.im,0)),
   double(coalesce(o.doc,0)), double(coalesce(i.dic,0)),
   double(a.opened_days_ago), double(coalesce(o.sr,0)),
   sign(coalesce(i.isum,0)-coalesce(o.os,0))*ln(1+abs(coalesce(i.isum,0)-coalesce(o.os,0)))
 ) node_features
FROM {ACC} a LEFT JOIN o ON a.account_id=o.acc LEFT JOIN i ON a.account_id=i.acc"""

while True:
    # 1) upsert accounts dimension (id, age, is_fraud) from the landed transactions
    spark.sql(f"""MERGE INTO {ACC} t USING (
        SELECT account_id, max(opened) opened_days_ago, max(is_fraud) is_fraud FROM (
          SELECT source_account account_id, src_opened opened, is_fraud FROM {TX}
          UNION ALL SELECT target_account, dst_opened, is_fraud FROM {TX}) GROUP BY account_id) s
      ON t.account_id=s.account_id
      WHEN MATCHED THEN UPDATE SET t.opened_days_ago=s.opened_days_ago, t.is_fraud=s.is_fraud
      WHEN NOT MATCHED THEN INSERT (account_id,opened_days_ago,is_fraud,fraud_role,typology_id,
          node_features,node_embedding,toxicity,emb_version,updated_ts)
        VALUES (s.account_id,s.opened_days_ago,s.is_fraud,NULL,NULL,
                array(double(0),double(0),double(0),double(0),double(0),double(0),
                      double(0),double(0),double(0),double(0),double(0)),
                array(double(0)),NULL,0,0)""")
    # 2) node features -> accounts_state.node_features
    spark.sql(f"MERGE INTO {ACC} t USING ({NODE_SRC}) s ON t.account_id=s.account_id "
              f"WHEN MATCHED THEN UPDATE SET t.node_features=s.node_features")
    # 2b) blocked accounts' PENDING edges -> BLOCKED (excluded from scoring)
    spark.sql(f"""MERGE INTO {TX} t USING (
        SELECT tx_id FROM {TX} WHERE ml_status='PENDING' AND NOT ({NOTBLK})) s
      ON t.tx_id=s.tx_id WHEN MATCHED THEN UPDATE SET t.ml_status='BLOCKED'""")
    # 3) edge features for PENDING -> FEATURES_READY
    n = spark.sql(f"SELECT count(*) c FROM {TX} WHERE ml_status='PENDING'").first()["c"]
    spark.sql(f"""MERGE INTO {TX} t USING (
        SELECT tx_id, array(ln(1+amount),
            CASE WHEN amount BETWEEN 9000 AND 9500 THEN 1.0 ELSE 0.0 END,
            (ln(1+amount)-{EM})/({ES}+1e-9)) fm
        FROM {TX} WHERE ml_status='PENDING') s ON t.tx_id=s.tx_id
      WHEN MATCHED THEN UPDATE SET t.features_matrix=s.fm, t.ml_status='FEATURES_READY'""")
    print(f"[etl] featurized {n} PENDING -> FEATURES_READY; sleeping 5 min", flush=True)
    time.sleep(300)
