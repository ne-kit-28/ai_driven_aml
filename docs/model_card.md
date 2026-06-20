# Model Card — TGN-lite

A streaming temporal graph network that scores anti-money-laundering risk on a
graph of bank transactions. This card follows the spirit of
[Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993).

## Model details

- **Type:** streaming temporal graph neural network (TGN-lite). Each account
  carries a memory vector `h_v` (dimension 64) updated by a GRU from incoming
  transfer messages under a delayed-message training scheme.
- **Outputs:** an edge head produces a per-transaction risk probability; a node
  head produces account *toxicity* — the probability the account behaves as a
  dropper or mule.
- **Calibration:** outputs are temperature-scaled. The shipped artifact uses
  `node_temp = 4.0` and `edge_temp = 1.0` for a sharp fraud/legit separation.
- **Artifacts:** `src/ml/artifacts/tgnlite.pt` with `tgnlite_meta.json`
  (feature names, normalization statistics, temperatures, memory size).
- **Serving:** stateless windowed — each cycle replays a rolling 30-day window
  from zeroed memory (matching training), writing risk exactly-once (anti-join
  on scored ids). Blocked accounts' edges are excluded, so victims recover.

## Feature contract

Twelve node features and three edge features, frozen and verified against the
artifact by `tests/test_feature_contract.py`.

- **Node:** `out_degree`, `in_degree`, `log_out_amount_sum`, `log_in_amount_sum`,
  `out_amount_mean`, `in_amount_mean`, `distinct_out_cp`, `distinct_in_cp`,
  `account_age_days`, `structuring_ratio`, `log_net_flow_abs`,
  `in_structuring_ratio` (share of incoming in the structuring band — flags
  smurf collectors).
- **Edge:** `log_amount`, `in_structuring_band` (amount in the
  `[9000, 9500]` just-under-threshold band), `amount_zscore` (computed with
  fixed training statistics, not batch statistics, at inference time).

The SQL pipeline (`src/features/features.sql`) and the reference Python
implementation (`src/ml/features.py`) must produce identical columns in the
same order.

## Intended use

- **In scope:** prioritizing accounts and transactions for human investigation;
  generating evidence for an analyst who makes the final blocking decision.
- **Out of scope:** automated account blocking without human review; legal or
  regulatory determinations; use on populations or transaction types unlike the
  training distribution.

The system is **decision support**. A human officer reviews the ego-graph and
the LLM explanation before any block, and blocking feeds back into the graph so
that contaminated legitimate accounts recover.

## Training data

- **Synthetic generator** — the live producer dumped to parquet
  (`producer.py --dump-parquet`), i.e. the **same distribution served in
  production**: a large legit majority, bounded fraud cases, and four laundering
  typologies (transit-ring, smurfing, fan-in cash-out, layering) with
  ground-truth labels (case participants are fraud; victims stay legit). Train
  via `src/ml/train_temporal.py`. (`generate_graph.py` is an alternative static
  generator.)

## Evaluation

Held-out producer-distribution set, scored through the live streaming scorer
(`infer_stream`, the real serving path), ~8% fraud accounts:

| Metric | Value |
|--------|-------|
| Node (account) ROC-AUC | 1.000 |
| Recall @ 0.5 | 100% |
| False positives @ 0.5 | 0% |
| Fraud vs legit mean toxicity | 0.85 / 0.15 |

Node recall@0.5 by typology: smurfing 1.00, fan-in cash-out 1.00, transit-ring
1.00, layering 0.96.

These numbers describe the **synthetic benchmark only** and are near-perfect
because the generator is separable for a temporal model. Treat them as a
regression baseline, not as a predictor of real-world performance. Before
production use, re-evaluate on a held-out, chronologically split set of the
target institution's data and add calibration
(reliability curve, ECE) and alert-load metrics.

## Limitations and ethical considerations

- **False positives** cause real harm (frozen funds, account friction). Operate
  at a threshold tuned with the investigations team, not at a default.
- **Guilt-by-association:** graph propagation can taint accounts merely close to
  bad actors. The LLM explainer is prompted to separate hubs from mules; node
  toxicity is calibrated to spread gradually, not absolutely.
- **Feedback effects:** the blocklist loop changes the graph the model sees.
  Memory decays over several cycles, so toxicity drops gradually rather than
  instantly after a block.
- **Distribution shift:** trained and evaluated on synthetic data only;
  performance on a specific institution's traffic must be re-validated before
  relying on it.
- **Fairness:** features include account age and flow patterns. Audit for
  disparate impact on legitimate customer segments before deployment.

## Maintenance

Retraining, calibration, and threshold selection should be re-run on current
data on a regular cadence and whenever the feature contract changes.
