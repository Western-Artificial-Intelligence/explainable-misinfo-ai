# TruthLens API - production image
FROM python:3.11-slim

WORKDIR /app

# System deps (minimal; add tesseract/ffmpeg if enabling image/audio routes)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Prefer CPU-only PyTorch for smaller image (~500MB vs ~2.5GB) on CPU-only hosts
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch --index-url ${TORCH_INDEX}

# Copy and install Python deps (torch already installed, so requirements will skip/reuse)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API code and baseline model (used when TRUTHLENS_CHECKPOINT_PATH is set)
COPY api/ ./api/
COPY baseline_model/ ./baseline_model/
COPY .env.example .env.example

# Expose port (Railway/Render/Fly set PORT at runtime)
ENV PORT=8000
EXPOSE 8000

# Run from /app so `api` module resolves; use PORT from env
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
