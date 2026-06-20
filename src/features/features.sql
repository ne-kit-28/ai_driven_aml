-- Stage 2 feature pipeline (pure SQL; runs on Spark SQL / Comet on the stand,
-- DuckDB locally). Reproduces the model contract in features.py.
-- Placeholders: {TX} = transactions source, {ACC} = accounts source.

-- ---------- NODE features -> accounts_state.node_features ----------
-- order MUST match NODE_FEATURE_NAMES in src/ml/features.py
WITH out_agg AS (
  SELECT source_account AS acc, count(*) AS out_degree, sum(amount) AS out_sum,
         avg(amount) AS out_mean, count(DISTINCT target_account) AS distinct_out_cp,
         avg(CASE WHEN amount BETWEEN 9000 AND 9500 THEN 1.0 ELSE 0.0 END) AS structuring_ratio
  FROM {TX} GROUP BY source_account),
in_agg AS (
  SELECT target_account AS acc, count(*) AS in_degree, sum(amount) AS in_sum,
         avg(amount) AS in_mean, count(DISTINCT source_account) AS distinct_in_cp,
         avg(CASE WHEN amount BETWEEN 9000 AND 9500 THEN 1.0 ELSE 0.0 END) AS in_structuring_ratio
  FROM {TX} GROUP BY target_account)
SELECT a.account_id,
       coalesce(o.out_degree, 0)                              AS out_degree,
       coalesce(i.in_degree, 0)                               AS in_degree,
       ln(1 + coalesce(o.out_sum, 0))                         AS log_out_amount_sum,
       ln(1 + coalesce(i.in_sum, 0))                          AS log_in_amount_sum,
       coalesce(o.out_mean, 0)                                AS out_amount_mean,
       coalesce(i.in_mean, 0)                                 AS in_amount_mean,
       coalesce(o.distinct_out_cp, 0)                         AS distinct_out_cp,
       coalesce(i.distinct_in_cp, 0)                          AS distinct_in_cp,
       a.opened_days_ago                                      AS account_age_days,
       coalesce(o.structuring_ratio, 0)                       AS structuring_ratio,
       sign(coalesce(i.in_sum,0) - coalesce(o.out_sum,0))
         * ln(1 + abs(coalesce(i.in_sum,0) - coalesce(o.out_sum,0))) AS log_net_flow_abs,
       coalesce(i.in_structuring_ratio, 0)                    AS in_structuring_ratio
FROM {ACC} a
LEFT JOIN out_agg o ON a.account_id = o.acc
LEFT JOIN in_agg  i ON a.account_id = i.acc;

-- ---------- EDGE features -> transactions.features_matrix ----------
-- z-score uses FIXED training stats (model_meta.edge_logamt_mean/std), not per-batch.
-- order MUST match EDGE_FEATURE_NAMES: [log_amount, in_structuring_band, amount_zscore]
SELECT tx_id,
       ln(1 + amount)                                                       AS log_amount,
       CASE WHEN amount BETWEEN 9000 AND 9500 THEN 1.0 ELSE 0.0 END         AS in_structuring_band,
       (ln(1 + amount) - {LOGAMT_MEAN}) / ({LOGAMT_STD} + 1e-9)             AS amount_zscore
FROM {TX};
