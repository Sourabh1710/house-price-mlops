# Stage 1: Builder

FROM python:3.11-slim AS builder

# Install system dependencies needed to compile Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install dependencies FIRST (before copying source code).
COPY requirements-serve.txt .
RUN pip install --upgrade pip && \
    pip install --user --no-cache-dir -r requirements-serve.txt


# Stage 2: Runtime

FROM python:3.11-slim AS runtime

# Create a non-root user - running as root in containers is a security risk
RUN useradd --create-home appuser
WORKDIR /home/appuser/app
USER appuser

# Copy installed Python packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application source
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser models/ ./models/

# Ensure user-installed packages are on PATH
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONPATH=/home/appuser/app
ENV MODEL_PATH=/home/appuser/app/models/best_model.pkl

# FastAPI runs on port 8000 by default
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:' + os.environ.get('PORT', '8000') + '/health')"

CMD uvicorn src.app:app --host 0.0.0.0 --port ${PORT:-8000}
