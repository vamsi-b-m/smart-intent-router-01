FROM python:3.11-slim

# Prevent Python from creating .pyc files
# and make logs appear immediately.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Create a non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Install Python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model
COPY src/ ./src/
COPY models/ ./models

# Run as non-root
USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
