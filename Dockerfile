# Dockerfile

# 1. Base Image
FROM python:3.10-slim-bookworm AS base

# 2. Environment Variables
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    DEBIAN_FRONTEND=noninteractive

# 3. System Dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-spa \
    tesseract-ocr-fra \
    tesseract-ocr-deu \
    # Añade más idiomas aquí si los necesitas
    libgl1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 4. Work Directory
WORKDIR /app

# 5. Python Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy Application Code
COPY . .

# 7. Create Non-root User
RUN groupadd -r appuser && useradd --no-log-init -r -g appuser appuser && \
    chown -R appuser:appuser /app
USER appuser

# 8. Expose Port
EXPOSE 5018

# 9. Run Command (Gunicorn with 1 worker and long timeout)
# Workers=1 es CRUCIAL para el almacenamiento de progreso en memoria.
# Timeout aumentado a 1800s (30 minutos) para PDFs muy largos. Ajusta si es necesario.
CMD ["gunicorn", "--bind", "0.0.0.0:5018", "--workers", "1", "--timeout", "1800", "--log-level", "info", "app:app"]