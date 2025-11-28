
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

# OPTIMIZATION: Configure pip to use faster PyPI mirror (Aliyun) for faster package downloads
RUN mkdir -p ~/.pip && echo "[global]\nindex-url = https://mirrors.aliyun.com/pypi/simple/\ntrusted-host = mirrors.aliyun.com" > ~/.pip/pip.conf

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

# CRITICAL FIX: Force cache invalidation for Python file copies
ARG CACHEBUST=20251107-v22-SECURITY-INTEGRATION-FIXES
RUN echo "Copying Python files with cache bust: $CACHEBUST"

# Delete any existing __pycache__ directories to force fresh imports
RUN find /app -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Copy startup validator (runs before main app to catch errors early)
COPY startup_validator.py .

# Copy all necessary Python files and modules from subdirectories
COPY core_infrastructure/fastapi_backend_v2.py .
COPY core_infrastructure/supabase_client.py .
COPY data_ingestion_normalization/universal_field_detector.py .
COPY data_ingestion_normalization/universal_document_classifier_optimized.py .
COPY data_ingestion_normalization/universal_platform_detector_optimized.py .
COPY data_ingestion_normalization/universal_extractors_optimized.py .
COPY data_ingestion_normalization/entity_resolver_optimized.py .
COPY aident_cfo_brain/enhanced_relationship_detector.py .
COPY aident_cfo_brain/semantic_relationship_extractor.py .
COPY aident_cfo_brain/finley_graph_engine.py .
COPY aident_cfo_brain/aident_memory_manager.py .
COPY data_ingestion_normalization/field_mapping_learner.py .
COPY data_ingestion_normalization/embedding_service.py .
COPY aident_cfo_brain/causal_inference_engine.py .
COPY aident_cfo_brain/temporal_pattern_learner.py .
COPY aident_cfo_brain/intelligent_chat_orchestrator.py .
COPY core_infrastructure/database_optimization_utils.py .
COPY duplicate_detection_fraud/production_duplicate_detection_service.py .
COPY core_infrastructure/transaction_manager.py .
COPY data_ingestion_normalization/streaming_processor.py .
COPY core_infrastructure/error_recovery_system.py .
COPY core_infrastructure/centralized_cache.py .
COPY core_infrastructure/security_system.py .
COPY data_ingestion_normalization/nango_client.py .
COPY background_jobs/arq_worker.py .
COPY background_jobs/worker_entry.py .
COPY core_infrastructure/provenance_tracker.py .
COPY duplicate_detection_fraud/inference_service.py .
COPY duplicate_detection_fraud/persistent_lsh_service.py .
COPY data_ingestion_normalization/streaming_source.py .
COPY data_ingestion_normalization/shared_learning_system.py .
COPY core_infrastructure/utils/helpers.py ./core_infrastructure/utils/
COPY start.sh .

# CRITICAL FIX: Copy all configuration files (YAML configs for platform mappings, exchange rates, etc.)
COPY config/ ./config/

# Copy built frontend from frontend stage
COPY --from=frontend-builder /app/frontend/dist ./dist

# Make start.sh executable
RUN chmod +x start.sh

# VERIFICATION: Show IntelligentChatOrchestrator __init__ signature to confirm correct version
RUN echo "🔍 VERIFYING IntelligentChatOrchestrator signature:" && \
    grep -A 5 "def __init__" intelligent_chat_orchestrator.py | head -6 || echo "File not found"

EXPOSE 8000

CMD ["bash", "start.sh"]
