
# Simple Python backend container
FROM python:3.11-slim

# Install system dependencies for python-magic
RUN apt-get update && apt-get install -y \
    libmagic1 \
    libmagic-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY fastapi_backend.py .

# Frontend will be built and copied by render.yaml

EXPOSE 8000

CMD ["python", "fastapi_backend.py"]
