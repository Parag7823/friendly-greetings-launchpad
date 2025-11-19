#!/bin/bash

# Run startup validator to catch errors early
echo "🔍 Running startup validator..."
python startup_validator.py
if [ $? -ne 0 ]; then
    echo "❌ Startup validation failed! Check logs above for details."
    exit 1
fi

echo "✅ Startup validation passed! Starting application..."

# Start ARQ worker in background
python -m arq arq_worker.WorkerSettings &

# Start FastAPI server in foreground with Socket.IO support
uvicorn fastapi_backend_v2:socketio_app --host 0.0.0.0 --port $PORT
