
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
# NUCLEAR CACHE BUST: Changed base image tag to force complete rebuild
FROM python:3.9-slim

# Force cache invalidation - updated 2025-11-07 for security & integration fixes
ARG CACHEBUST=20251107-v22-SECURITY-INTEGRATION-FIXES
ENV DEPLOYMENT_VERSION="2025-11-07-17:10-SECURITY-INTEGRATION-FIXES"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
RUN echo "🚨 ABSOLUTE NUCLEAR CACHE BUST: $CACHEBUST - Forcing complete rebuild"
RUN echo "🚨 DEPLOYMENT VERSION: $DEPLOYMENT_VERSION"
RUN echo "🚨 TIMESTAMP: $(date +%s)"
RUN echo "🚨 PYTHON BYTECODE DISABLED: PYTHONDONTWRITEBYTECODE=1"
# Install system dependencies for python-magic and basic functionality
# REMOVED: Tesseract (replaced by easyocr), Java (replaced by pdfminer.six)
# Added gfortran and build-essential for scipy compilation
# Added dependencies for PyTorch/sentence-transformers (BGE embeddings)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
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

# Install PyTorch CPU-only wheels first to avoid CUDA bloat (reduces image size by ~3GB)
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.1.0+cpu \
    torchvision==0.16.0+cpu

# Install pandas with specific strategy to avoid compilation issues
RUN echo "Installing pandas with Python $(python --version)" && \
    pip install --no-cache-dir --only-binary=all pandas==2.0.3 numpy==1.24.4 && \
    echo "Pandas installation completed successfully"

# Install remaining dependencies (torch/torchvision already installed above)
RUN pip install --no-cache-dir -r backend-requirements.txt

# CRITICAL: Force cache invalidation for Python file copies
ARG CACHEBUST=20251107-v22-SECURITY-INTEGRATION-FIXES
RUN echo "Copying Python files with cache bust: $CACHEBUST"

# Delete any existing __pycache__ directories to force fresh imports
RUN find /app -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Copy all necessary Python files and modules
COPY fastapi_backend_v2.py .
COPY universal_field_detector.py .
COPY universal_document_classifier_optimized.py .
COPY universal_platform_detector_optimized.py .
COPY universal_extractors_optimized.py .
COPY entity_resolver_optimized.py .
COPY enhanced_relationship_detector.py .
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
COPY centralized_cache.py .
COPY observability_system.py .
COPY security_system.py .
COPY nango_client.py .
COPY arq_worker.py .
COPY worker_entry.py .
COPY provenance_tracker.py .
COPY debug_logger.py .
COPY inference_service.py .
COPY persistent_lsh_service.py .
COPY streaming_source.py .
COPY start.sh .
 
# Copy built frontend from frontend stage
COPY --from=frontend-builder /app/frontend/dist ./dist

# Make start.sh executable
RUN chmod +x start.sh

# VERIFICATION: Show IntelligentChatOrchestrator __init__ signature to confirm correct version
RUN echo "🔍 VERIFYING IntelligentChatOrchestrator signature:" && \
    grep -A 5 "def __init__" intelligent_chat_orchestrator.py | head -6 || echo "File not found"

EXPOSE 8000

CMD ["bash", "start.sh"]
