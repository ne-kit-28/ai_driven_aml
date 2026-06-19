# Stage 2 feature pipeline (PENDING -> FEATURES_READY). Build context = ../src.
FROM python:3.11-slim-bookworm
WORKDIR /app
COPY features/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY features/stage2.py features/features.sql ./
ENV PYTHONUNBUFFERED=1
CMD ["python", "stage2.py", "--io", "iceberg", "--meta", "/artifacts/tgnlite_meta.json"]
