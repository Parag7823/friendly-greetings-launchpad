
# Multi-stage build for frontend and backend
FROM node:18-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy package files
COPY package*.json ./
COPY src/ ./src/
COPY public/ ./public/
COPY index.html ./
COPY tailwind.config.ts ./
COPY tsconfig*.json ./
COPY vite.config.ts ./
COPY postcss.config.js ./
COPY components.json ./

# Install dependencies and build
RUN npm ci
RUN npm run build

# Backend stage
FROM python:3.11-slim

# Install system dependencies for python-magic and Node.js
RUN apt-get update && apt-get install -y \
    libmagic1 \
    libmagic-dev \
    gcc \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY fastapi_backend.py .

# Copy built frontend from frontend stage
COPY --from=frontend-builder /app/frontend/dist ./dist

EXPOSE 8000

CMD ["python", "fastapi_backend.py"]
