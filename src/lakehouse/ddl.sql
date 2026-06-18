-- AML lakehouse schema (ADD v2: node/edge split). Run via spark-sql.
CREATE NAMESPACE IF NOT EXISTS iceberg.banking;

-- Edges: transactions. ml_status drives the ELT state machine.
CREATE TABLE IF NOT EXISTS iceberg.banking.transactions (
    tx_id            STRING,
    source_account   STRING,
    target_account   STRING,
    amount           DOUBLE,
    ts               BIGINT,
    typology_id      STRING,        -- ground-truth (synthetic only)
    is_fraud         INT,           -- ground-truth (synthetic only)
    features_matrix  ARRAY<DOUBLE>, -- edge/window features
    risk_score       DOUBLE,
    ml_status        STRING         -- PENDING -> FEATURES_READY -> SCORED
) USING iceberg
PARTITIONED BY (ml_status)
TBLPROPERTIES ('format-version'='2', 'write.delete.mode'='merge-on-read');

-- Nodes: per-account state. Holds the memory vector h_v (one row per account).
CREATE TABLE IF NOT EXISTS iceberg.banking.accounts_state (
    account_id       STRING,
    opened_days_ago  INT,
    is_fraud         INT,
    fraud_role       STRING,
    typology_id      STRING,
    node_embedding   ARRAY<DOUBLE>, -- h_v
    emb_version      INT,
    risk_score       DOUBLE,
    updated_ts       BIGINT
) USING iceberg
TBLPROPERTIES ('format-version'='2', 'write.delete.mode'='merge-on-read');
