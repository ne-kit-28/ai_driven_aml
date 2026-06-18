"""
Load synthetic parquet into the Iceberg lakehouse. Runs INSIDE the
pyspark-notebook container (Hive catalog + S3 wired via spark-defaults).

  docker compose --profile core exec pyspark-notebook \
      spark-sql -f /home/jovyan/work/src/lakehouse/ddl.sql
  docker compose --profile core exec pyspark-notebook \
      spark-submit /home/jovyan/work/src/lakehouse/load_synthetic.py
"""
import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

DATA = os.environ.get("DATA_DIR", "/home/jovyan/work/data")

spark = SparkSession.builder.appName("load-synthetic").getOrCreate()

# Edges: start every transaction in PENDING for the state machine.
tx = (spark.read.parquet(f"{DATA}/transactions.parquet")
      .withColumn("features_matrix", F.array().cast("array<double>"))
      .withColumn("risk_score", F.lit(None).cast("double"))
      .withColumn("ml_status", F.lit("PENDING")))
(tx.select("tx_id", "source_account", "target_account", "amount", "ts",
           "typology_id", "is_fraud", "features_matrix", "risk_score", "ml_status")
   .writeTo("iceberg.banking.transactions").overwritePartitions())

# Nodes: one row per account, embedding initialised empty.
acc = (spark.read.parquet(f"{DATA}/accounts.parquet")
       .withColumn("node_embedding", F.array().cast("array<double>"))
       .withColumn("emb_version", F.lit(0))
       .withColumn("risk_score", F.lit(None).cast("double"))
       .withColumn("updated_ts", F.lit(0).cast("bigint")))
(acc.select("account_id", "opened_days_ago", "is_fraud", "fraud_role",
            "typology_id", "node_embedding", "emb_version", "risk_score", "updated_ts")
    .writeTo("iceberg.banking.accounts_state").createOrReplace())

print("transactions:", spark.table("iceberg.banking.transactions").count())
print("accounts:", spark.table("iceberg.banking.accounts_state").count())
spark.stop()
