FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY server/ /app/server/
COPY models.py client.py /app/

ENV PYTHONPATH=/app
ENV EMAIL_TRIAGE_TASK=easy_triage

EXPOSE 7860

HEALTHCHECK --interval=5s --timeout=5s --start-period=15s \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
