# Architecture Design Document v2
## AI-Driven Graph AML Platform — Revised

> v2 changelog vs. исходного ADD: (1) GNN переименован в Temporal GNN с честной семантикой памяти; (2) разнесены таблицы узлов и рёбер; (3) добавлен XAI-слой (faithful-объяснение драйверов скора); (4) приватный LLM + de-identification для prod; (5) audit-trail переведён на hash-chain + Merkle-якорь; (6) введены SLA-уровни L0/L1/L2; (7) добавлены entity resolution, дисбаланс классов, feedback loop и model registry.

## 1. Что сохранено из v1
- Decoupled compute/storage, Iceberg как единый источник правды и Feature Store.
- ELT-конечный автомат `PENDING → FEATURES_READY → SCORED` (Zero-UDF, защита от OOM и JVM↔Python overhead).
- Двухконтурный AI: GNN — «быстрый фильтр», LLM — «интерпретатор/аудитор».
- Trino для извлечения k=2 эго-графа без дорогой Graph DB.

## 2. Ключевые исправления

### 2.1 Temporal GNN, а не «stateful GraphSAGE»
Память узла обновляется time-respecting и directed:
$$h_v^{(t)} = \sigma\!\left(W_1 h_v^{(t-1)} + W_2 \cdot \text{AGG}\big(\{h_u^{(t-1)} \mid u \in N_{\text{in}}(v),\ t_u < t\}\big)\right)$$
Ключевое: агрегируем только входящих соседей с более ранним временем — поздняя транзакция не «заражает» прошлое. Это TGN/EvolveGCN-семантика. Claim $$O(1)$$ на инференс верен при контроле staleness соседних эмбеддингов.

### 2.2 Узлы и рёбра — раздельно
- `accounts_state` (узлы): `account_id`, `node_embedding`, `emb_version`, `updated_ts`, `risk_score`.
- `transactions` (рёбра): перевод + `risk_score` ребра + `ml_status`. Без копий эмбеддинга в каждой строке.

### 2.3 XAI-слой (faithful explainability)
GNNExplainer / integrated gradients выдаёт подмножество рёбер-драйверов скора. LLM получает **только** эти факты + структуру, не выдумывает нарратив. Требование регуляторной защищаемости (model risk management, SR 11-7).

### 2.4 Приватный LLM + de-identification
Prod: self-hosted open-weight модель или приватный no-retention endpoint; счета псевдонимизируются до промпта (data residency EMEA/ЦА). MVP на синтетике — внешний API допустим.

### 2.5 Hash-chain Audit Trail
$$H_i = \text{SHA-256}(\text{record}_i \,\Vert\, H_{i-1})$$
Цепочка ловит и удаление/усечение строк (per-record подпись — нет). Периодическое якорение Merkle-корня. Ключи в KMS/HSM. В хеш входят `model_version` и `feature_version`.

### 2.6 SLA-уровни
- **L0** (<1s): инлайн правила + lookup закэшированного $$h_v$$ → жёсткий блок.
- **L1** (~15 мин): батч TGN → алерты для расследования.
- **L2** (раз в сутки): глобальные мотивы на Spark GraphFrames.

### 2.7 Governance
Entity resolution (счёт→сущность) до построения графа; focal loss / precision@k под дисбаланс <0.1%; бюджет алертов на следователя; петля обратной связи TP/FP → дообучение; model registry с champion/challenger и drift-мониторингом.

## 3. Стек (дополнения к v1)
| Слой | v1 | v2 добавления |
|---|---|---|
| ML | GraphSAGE | TGN/EvolveGCN + модуль памяти, NeighborLoader из Iceberg |
| XAI | — | GNNExplainer / IG |
| RAG | edge-list → LLM | + vector store прошлых SAR/типологий, structured output с цитированием tx_id |
| Audit | per-record RSA | hash-chain + Merkle-якорь, KMS |
| LLM | внешний API | приватный/self-hosted + de-id |
| Realtime | — | L0 инлайн-контур |

## 4. Дорожная карта (этапы)
| Этап | Содержание | Статус |
|---|---|---|
| 0 Решения | scope, LLM-хостинг, метрики, SLA | принято (демо-умолчания) |
| 1 Инфра + данные | docker-compose + генератор синтетики с ground-truth | **генератор готов** |
| 2 Data/Feature | Spark SQL оконные фичи, конечный автомат, split node/edge | следующий |
| 3 GNN | baseline GraphSAGE → TGN-память, NeighborLoader | |
| 4 XAI+RAG+Audit | drivers → приватный LLM SAR, hash-chain | |
| 5 UI | Streamlit + pyvis, валидация лога, авто-SAR, симуляция времени | |
| 6 Hardening | drift, feedback, registry, Iceberg compaction | roadmap |

## 5. Метрики успеха
- Качество: precision@k, recall по типологиям T1–T4, SAR-conversion rate.
- Производительность: инференс $$O(1)$$ на узел, p95 латентность L0 < 1s.
- Защищаемость: 100% решений в hash-chain, faithful-объяснение к каждому алерту.
