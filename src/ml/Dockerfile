# Graph scoring service (TGN-lite inference loop). Build context = this dir.
FROM python:3.11-slim-bookworm
WORKDIR /app

# CPU torch (matches what the model was trained with)
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY features.py exactly_once.py stream_model.py infer_stream.py ./

ENV PYTHONUNBUFFERED=1
# model artifact is mounted at /artifacts; bookmark/state at /state
CMD ["python", "infer_stream.py", "--io", "iceberg", "--loop", "--interval", "4", "--artifacts", "/artifacts"]
