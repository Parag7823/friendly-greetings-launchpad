
# Multi-stage build for frontend and backend
FROM node:18-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy package files first for better caching
COPY package*.json ./

# Install dependencies
RUN npm ci --production=false

# Copy source files
COPY src/ ./src/
COPY public/ ./public/
COPY index.html ./
COPY tailwind.config.ts ./
COPY tsconfig*.json ./
COPY vite.config.ts ./
COPY postcss.config.js ./
COPY components.json ./
COPY eslint.config.js ./

# Build frontend
RUN npm run build

# Verify build output
RUN ls -la dist/
RUN ls -la dist/assets/

# Backend stage
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

# Copy built frontend from frontend stage
COPY --from=frontend-builder /app/frontend/dist ./dist

EXPOSE 8000

CMD ["python", "fastapi_backend.py"]
