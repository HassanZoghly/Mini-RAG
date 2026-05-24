FROM ghcr.io/astral-sh/uv:0.6.14-python3.10-bookworm

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libavif-dev \
    pkg-config \
    libjpeg-dev \
    gcc \
    unzip \
    zip \
    python3-dev \
    libxml2-dev \
    libxslt1-dev \
    libffi-dev \
    curl \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY src/requirements.txt .

# Install Python dependencies
RUN uv pip install -r requirements.txt --system

# Copy application source
COPY src/ .

# Create alembic directory structure
RUN mkdir -p /app/models/db_schemes/minirag/

# Copy alembic config
COPY docker/minirag/alembic.ini /app/models/db_schemes/minirag/alembic.ini

# Copy scripts
COPY scripts /app/scripts

# Make scripts executable
RUN chmod +x /app/scripts/*.sh

# Optional: create logs directory
RUN mkdir -p /app/logs

# Set startup script
ENTRYPOINT ["/app/scripts/startup.sh"]

# Run FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
