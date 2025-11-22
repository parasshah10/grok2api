# Build Stage
FROM python:3.11-slim AS builder

WORKDIR /build

# Install dependencies to independent directory
COPY requirements.txt .
RUN pip install --no-cache-dir --only-binary=:all: --prefix=/install -r requirements.txt && \
    find /install -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /install -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /install -type d -name "test" -exec rm -rf {} + 2>/dev/null || true && \
    find /install -type d -name "*.dist-info" -exec sh -c 'rm -f "$1"/RECORD "$1"/INSTALLER' _ {} \; && \
    find /install -type f -name "*.pyc" -delete && \
    find /install -type f -name "*.pyo" -delete && \
    find /install -name "*.so" -exec strip --strip-unneeded {} \; 2>/dev/null || true

# Runtime Stage - Use minimal image
FROM python:3.11-slim

WORKDIR /app

# Clean up redundant files in base image
RUN rm -rf /usr/share/doc/* \
    /usr/share/man/* \
    /usr/share/locale/* \
    /var/cache/apt/* \
    /var/lib/apt/lists/* \
    /tmp/* \
    /var/tmp/*

# Copy installed packages from build stage
COPY --from=builder /install /usr/local

# Create necessary directories and files
RUN mkdir -p /app/logs /app/data/temp/image /app/data/temp/video && \
    echo '{"ssoNormal": {}, "ssoSuper": {}}' > /app/data/token.json

# Copy application code
COPY app/ ./app/
COPY main.py .
COPY data/setting.toml ./data/

# Remove Python bytecode and cache
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
