FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    tesseract-ocr \
  && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code
COPY . /app/

# Set Python path
ENV PYTHONPATH=/app

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Production command (no --reload)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]