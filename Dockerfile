# Use official Python image with slim-buster
FROM python:3.9-slim-buster

# Set working directory (all files will be copied here)
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    libgomp1 && \  # Required for scikit-learn
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy ALL files (including models) to /app
COPY . .

# Environment variables (use direct paths since everything is in /app)
ENV PORT=8000
ENV MODEL_PATH=/app/tiktok_scoring_model.pkl  # Direct path to model
ENV SCALING_PARAMS_PATH=/app/scaling_params.pkl
ENV PYTHONUNBUFFERED=1

# Set permissions
RUN chmod -R 755 /app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Expose and run
EXPOSE $PORT
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
