# Iceberg lifecycle demo: create -> insert -> delete (MOR) -> maintenance.
# Open in Jupyter (cells split on '# %%') or paste cell-by-cell.

# %% Spark session (spark-defaults.conf already wires the Iceberg/Hive/S3 catalog)
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("iceberg-demo").getOrCreate()

# %% Create table (format-version 2 + merge-on-read enables row-level DELETE)
spark.sql("CREATE NAMESPACE IF NOT EXISTS iceberg.demo")
spark.sql("""
CREATE OR REPLACE TABLE iceberg.demo.accounts (
    id      BIGINT,
    name    STRING,
    amount  DOUBLE,
    country STRING
) USING iceberg
TBLPROPERTIES ('format-version'='2', 'write.delete.mode'='merge-on-read')
""")

# %% Insert rows (two commits -> two snapshots)
spark.sql("""
INSERT INTO iceberg.demo.accounts VALUES
 (1,'Alice',100.0,'KZ'), (2,'Bob',250.5,'AE'),
 (3,'Carol',75.0,'KZ'),  (4,'Dan',999.9,'GE'), (5,'Eve',12.0,'AE')
""")
spark.createDataFrame(
    [(6, 'Frank', 300.0, 'KZ'), (7, 'Grace', 45.0, 'GE')],
    "id bigint, name string, amount double, country string"
).writeTo("iceberg.demo.accounts").append()
spark.table("iceberg.demo.accounts").orderBy("id").show()

# %% Delete rows (MOR writes delete files; data files untouched)
spark.sql("DELETE FROM iceberg.demo.accounts WHERE country = 'GE'")
spark.table("iceberg.demo.accounts").orderBy("id").show()

# %% Inspect metadata (snapshots + files) — motivates maintenance
spark.sql("SELECT snapshot_id, committed_at, operation FROM iceberg.demo.accounts.snapshots").show(truncate=False)
spark.sql("SELECT content, file_path, record_count FROM iceberg.demo.accounts.files").show(truncate=False)

# %% Maintenance
spark.sql("CALL iceberg.system.rewrite_data_files(table => 'demo.accounts')").show()
spark.sql("CALL iceberg.system.rewrite_manifests(table => 'demo.accounts')").show()
spark.sql("""
CALL iceberg.system.expire_snapshots(
  table       => 'demo.accounts',
  older_than  => TIMESTAMP '2999-01-01 00:00:00',
  retain_last => 1)
""").show()
spark.sql("CALL iceberg.system.remove_orphan_files(table => 'demo.accounts')").show()

# %% Verify
print("rows:", spark.table("iceberg.demo.accounts").count())
spark.sql("SELECT snapshot_id, operation FROM iceberg.demo.accounts.snapshots").show(truncate=False)
