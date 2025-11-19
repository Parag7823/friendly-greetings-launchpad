#!/bin/bash

# Enable verbose logging to see what's happening
set -x

# Run startup validator to catch errors early
echo "🔍 Running startup validator..."
python startup_validator.py
if [ $? -ne 0 ]; then
    echo "❌ Startup validation failed! Check logs above for details."
    exit 1
fi

echo "✅ Startup validation passed! Starting application..."

# Start ARQ worker in background
echo "🔧 Starting ARQ worker in background..."
python -m arq arq_worker.WorkerSettings &
ARQ_PID=$!
echo "✅ ARQ worker started with PID: $ARQ_PID"

# Start FastAPI server in foreground with Socket.IO support
echo "🚀 Starting Uvicorn server..."
echo "   Module: fastapi_backend_v2:socketio_app"
echo "   Host: 0.0.0.0"
echo "   Port: $PORT"

# Add timeout and detailed logging
uvicorn fastapi_backend_v2:socketio_app \
    --host 0.0.0.0 \
    --port $PORT \
    --log-level debug \
    --timeout-keep-alive 30 \
    2>&1 | tee /tmp/uvicorn.log

# If uvicorn exits, show the error
EXIT_CODE=$?
echo "❌ Uvicorn exited with code: $EXIT_CODE"
echo "📋 Last 50 lines of uvicorn log:"
tail -n 50 /tmp/uvicorn.log
exit $EXIT_CODE
