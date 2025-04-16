FROM python:3.10.12-slim

WORKDIR /app

# 1. Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev curl && \
    rm -rf /var/lib/apt/lists/*

# 2. Install Python dependencies with VERSION PINNING
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 3. Copy ALL necessary files
COPY main.py ./
COPY tiktok_scoring_model.pkl ./
COPY scaling_params.pkl ./

# 4. Environment variables
ENV MODEL_PATH=/app/tiktok_scoring_model.pkl
ENV SCALING_PARAMS_PATH=/app/scaling_params.pkl

# 5. Run the API
CMD ["uvicorn", "main:app", "--port", "8000"]