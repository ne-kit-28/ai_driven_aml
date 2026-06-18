# Seed/test service: fills MinIO/Iceberg with synthetic FEATURES_READY data.
# Stands in for Stage 2. Build context = ../src  (needs generator + features).
FROM python:3.11-slim-bookworm
WORKDIR /app
COPY seed/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY generator/generate_graph.py ml/features.py seed/seed_iceberg.py ./
ENV PYTHONUNBUFFERED=1
CMD ["python", "seed_iceberg.py"]
