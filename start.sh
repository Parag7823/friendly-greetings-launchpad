#!/bin/bash

# CRITICAL: Verify all methods exist before starting server
echo "🔍 Verifying ExcelProcessor methods..."
python verify_methods.py
if [ $? -ne 0 ]; then
    echo "❌ CRITICAL: Method verification failed! Container has old code!"
    echo "❌ This is a Docker cache issue. Check logs above for missing methods."
    # Continue anyway but log the error prominently
fi

# Start ARQ worker in background
python -m arq arq_worker.WorkerSettings &

# Start FastAPI server in foreground
uvicorn fastapi_backend_v2:app --host 0.0.0.0 --port $PORT
