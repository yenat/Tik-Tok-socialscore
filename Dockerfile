FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8000
ENV MODEL_PATH=/app/tiktok_scoring_model.pkl
ENV SCALING_PARAMS_PATH=/app/scaling_params.pkl
ENV PYTHONUNBUFFERED=1

EXPOSE $PORT

HEALTHCHECK --interval=30s --timeout=5s \
    CMD curl -f http://localhost:$PORT/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "60"]