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
  `node_temp = 8.0` and `edge_temp = 1.0`, so toxicity propagates across the
  graph instead of pinning at 1.0.
- **Artifacts:** `src/ml/artifacts/tgnlite.pt` with `tgnlite_meta.json`
  (feature names, normalization statistics, temperatures, memory size).
- **Serving:** exactly-once — scored transactions are anti-joined on their ids;
  account memory grows as new accounts appear.

## Feature contract

Eleven node features and three edge features, frozen and verified against the
artifact by `tests/test_feature_contract.py`.

- **Node:** `out_degree`, `in_degree`, `log_out_amount_sum`, `log_in_amount_sum`,
  `out_amount_mean`, `in_amount_mean`, `distinct_out_cp`, `distinct_in_cp`,
  `account_age_days`, `structuring_ratio`, `log_net_flow_abs`.
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

- **Synthetic generator** (`src/generator/`): hub/mule topologies with amount
  and account-age overlap, contamination, and four laundering typologies, with
  ground-truth labels.
- **Real data:** IBM AML (AMLWorld), Kaggle
  `ealtman2019/ibm-transactions-for-anti-money-laundering-aml`
  (CDLA-Sharing-1.0). Retrain via `src/ml/train_temporal.py`.

## Evaluation

> TODO: populate from a tracked evaluation run before any production use.

Report on a held-out, chronologically split set, by typology:

- Ranking quality: AUPRC, AUROC, precision@k / recall@k at the operating
  threshold.
- Calibration: reliability curve and expected calibration error.
- Alert load: alerts per day and analyst review burden at the chosen threshold.

## Limitations and ethical considerations

- **False positives** cause real harm (frozen funds, account friction). Operate
  at a threshold tuned with the investigations team, not at a default.
- **Guilt-by-association:** graph propagation can taint accounts merely close to
  bad actors. The LLM explainer is prompted to separate hubs from mules; node
  toxicity is calibrated to spread gradually, not absolutely.
- **Feedback effects:** the blocklist loop changes the graph the model sees.
  Memory decays over several cycles, so toxicity drops gradually rather than
  instantly after a block.
- **Distribution shift:** trained on synthetic and AMLWorld data; performance on
  a specific institution's traffic must be re-validated before relying on it.
- **Fairness:** features include account age and flow patterns. Audit for
  disparate impact on legitimate customer segments before deployment.

## Maintenance

Retraining, calibration, and threshold selection should be re-run on current
data on a regular cadence and whenever the feature contract changes.
