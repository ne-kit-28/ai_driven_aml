# AML compliance dashboard (Streamlit microservice). Build context = this dir.
FROM python:3.11-slim-bookworm

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY graph_query.py graph_agg.py chain_select.py llm_explain.py app.py ./

EXPOSE 8501
ENV DASH_SOURCE=parquet DATA_DIR=/data
HEALTHCHECK --interval=15s --timeout=4s --retries=10 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
