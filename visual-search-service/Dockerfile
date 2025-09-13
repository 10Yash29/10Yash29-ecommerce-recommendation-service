FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for image processing and PostgreSQL
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    libjpeg-dev \
    libpng-dev \
    libwebp-dev \
    && rm -rf /var/lib/apt/lists/*

# Create cache directory with proper permissions
RUN mkdir -p /.cache && chmod 777 /.cache

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create a non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app /.cache
USER appuser

# Expose port
EXPOSE 7860

# Start the application
CMD ["python", "app.py"]
