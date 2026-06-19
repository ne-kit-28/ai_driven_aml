# Continuous transaction producer -> Kafka. Build context = ../src.
FROM python:3.11-slim-bookworm
WORKDIR /app
COPY ingest/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ingest/producer.py ./
ENV PYTHONUNBUFFERED=1
CMD ["python", "producer.py", "--bootstrap", "kafka:9092", "--rate", "5", "--fraud-every", "60"]
