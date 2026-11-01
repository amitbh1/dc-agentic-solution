FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY requirements.txt requirements-prod.txt ./
RUN pip install -r requirements-prod.txt
COPY dc_agent ./dc_agent
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
# One worker per container: rate limits, plan cache and breakers are per process.
# Scale with replicas, and move the rate limiter to Redis before running more than one.
CMD ["uvicorn", "dc_agent.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
