FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (minimal, just what's needed)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies (pinned versions for stability)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy only necessary files (improves layer caching)
COPY main.py .
COPY tiktok_scoring_model.pkl .
COPY scaling_params.pkl .

# Environment variables
ENV PORT=8000
ENV MODEL_PATH=/app/tiktok_scoring_model.pkl
ENV SCALING_PARAMS_PATH=/app/scaling_params.pkl

# Expose and healthcheck
EXPOSE $PORT
HEALTHCHECK --interval=30s --timeout=5s \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Run with auto-reload in dev, but better for production:
# Use `--workers 2` for production (with Gunicorn)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "60"]