# AML Lakehouse — local infrastructure

Lakehouse для MVP на **Hive Metastore** (каталог) + **Jupyter pyspark-notebook**. Поднимается на вашей машине (Docker Engine + compose v2).

## Требования
- ~10 GB RAM, 4 vCPU, 12 GB диска.
- Интернет на этапе `build` (образ pyspark собирается, Iceberg-джары запекаются внутрь;
  init метастора тянет s3a-джары). В рантайме интернет не нужен.

## Сервисы
| Сервис | Порт (host) | Назначение |
|---|---|---|
| MinIO API / Console | 9000 / 9001 | S3-хранилище (admin / password123) |
| Postgres | — | бэкенд Hive Metastore |
| Hive Metastore | 9083 | каталог таблиц (thrift) |
| Trino | 8080 | SQL + извлечение эго-графа |
| Jupyter (pyspark) | 8888 / 4040 | Spark 3.5.8 + Iceberg 1.11.0 (собственный образ, токен: `aml`) |
| Kafka (профиль streaming) | 9092 | приём потока |

## Запуск

```bash
cd infra

# 1. Собрать образ pyspark и поднять core (MinIO, Postgres, Hive Metastore, Trino, Jupyter)
docker compose --profile core build pyspark-notebook
docker compose --profile core up -d
#    Дождитесь, пока hive-metastore проинициализирует схему (первый запуск ~30-60с):
docker compose --profile core logs -f hive-metastore   # ждём "Starting Hive Metastore Server"

# 2. Схема Iceberg (namespace + таблицы узлов/рёбер)
docker compose --profile core exec pyspark-notebook \
    spark-sql -f /home/jovyan/work/src/lakehouse/ddl.sql

# 3. Синтетика (на хосте -> ../data)
python ../src/generator/generate_graph.py --seed 42 --scale 1.0 --out ../data

# 4. Загрузка в Iceberg
docker compose --profile core exec pyspark-notebook \
    spark-submit /home/jovyan/work/src/lakehouse/load_synthetic.py

# 5. Проверка через Trino
docker compose --profile core exec trino \
    trino --execute "SELECT ml_status, count(*) FROM iceberg.banking.transactions GROUP BY 1"
```

Jupyter Lab: http://localhost:8888/?token=aml — ноутбуки в `work/src`.
Опционально стриминг: `docker compose --profile core --profile streaming up -d`.

## Перезапуск метастора
Схема инициализируется при первом старте. При повторном `up` без сброса данных добавьте
`IS_RESUME: "true"` в `environment` сервиса `hive-metastore` (иначе schematool попытается
инициализировать уже существующую схему). Полный сброс: `down -v`.

## Остановка
```bash
docker compose --profile core --profile streaming down       # сохранить данные
docker compose --profile core --profile streaming down -v    # стереть тома
```

## Примечания supply-chain
- Spark `3.5.8` + Iceberg `1.11.0` (Spark runtime + aws-bundle, Scala 2.12), JDK 17.
  Готовых образов `pyspark-notebook` с 3.5.x больше нет (quay держит только свежий,
  сейчас Spark 4.x) — поэтому свой `Dockerfile` (`infra/pyspark/`), джары запечены внутрь.
- Версия джар запекается через `ARG` в `infra/pyspark/Dockerfile` (`SPARK_VERSION`, `ICEBERG_VERSION`).
- Hive `4.0.0` (Hadoop 3.3.6) -> на classpath метастора добавлены `hadoop-aws:3.3.6`
  и `aws-java-sdk-bundle:1.12.367` (init-контейнер), JDBC `postgresql:42.7.4`.
  ВАЖНО: именно `4.0.0`, не `4.0.1` — в 4.0.1 удалили легаси thrift-методы
  (`get_table_objects_by_name`), которые всё ещё зовёт Hive-клиент внутри iceberg-spark-runtime.
- MinIO: проект прекратил публикацию community-образов (репозиторий заархивирован, апр 2026) —
  закреплён последний официальный тег. Для production: Bitnami/Chainguard/сборка из исходников.
- Все теги закреплены явно; `latest` не используется.
