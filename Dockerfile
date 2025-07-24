
FROM python:3.11-slim

# Install system dependencies for python-magic
RUN apt-get update && apt-get install -y \
    libmagic1 \
    libmagic-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY fastapi_backend.py .

EXPOSE 8000

CMD ["python", "fastapi_backend.py"]
