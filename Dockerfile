
# Multi-stage build for frontend and backend
FROM node:20-alpine AS frontend-builder

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
COPY eslint.config.js ./

# Install dependencies and build
RUN npm ci --production=false
RUN npm run build

# Backend stage - Use Python 3.9 for maximum pandas compatibility (most stable)
FROM python:3.9.18-slim

# Force cache invalidation - updated 2025-10-28 to include BGE embeddings
ARG CACHEBUST=20251028-v3
# Install system dependencies for python-magic, Tesseract (OCR), Java (Tabula), and basic functionality
# Added gfortran and build-essential for scipy compilation
# Added dependencies for PyTorch/sentence-transformers (BGE embeddings)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    tesseract-ocr \
    default-jre \
    gcc \
    g++ \
    gfortran \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libglib2.0-0 \
    libgomp1 \
    pkg-config \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python requirements and install
COPY backend-requirements.txt .

# Verify Python version and upgrade pip
RUN python --version && echo "Python version check passed" && pip install --upgrade pip wheel setuptools

# Install pandas first with specific strategy to avoid compilation issues
RUN echo "Installing pandas with Python $(python --version)" && \
    pip install --no-cache-dir --only-binary=all pandas==2.0.3 numpy==1.24.4 && \
    echo "Pandas installation completed successfully"

# Install remaining dependencies
RUN pip install --no-cache-dir -r backend-requirements.txt

# Copy all necessary Python files and modules
COPY fastapi_backend.py .
COPY universal_field_detector.py .
COPY universal_document_classifier_optimized.py .
COPY universal_platform_detector_optimized.py .
COPY universal_extractors_optimized.py .
COPY entity_resolver_optimized.py .
COPY enhanced_relationship_detector.py .
COPY neo4j_relationship_detector.py .
COPY semantic_relationship_extractor.py .
COPY field_mapping_learner.py .
COPY embedding_service.py .
COPY causal_inference_engine.py .
COPY temporal_pattern_learner.py .
COPY intelligent_chat_orchestrator.py .
COPY database_optimization_utils.py .
COPY production_duplicate_detection_service.py .
COPY transaction_manager.py .
COPY streaming_processor.py .
COPY error_recovery_system.py .
COPY ai_cache_system.py .
COPY batch_optimizer.py .
COPY observability_system.py .
COPY security_system.py .
COPY nango_client.py .
COPY arq_worker.py .
COPY provenance_tracker.py .
COPY debug_logger.py .
COPY start.sh .
 
# Copy built frontend from frontend stage
COPY --from=frontend-builder /app/frontend/dist ./dist

# Make start.sh executable
RUN chmod +x start.sh

EXPOSE 8000

CMD ["bash", "start.sh"]
