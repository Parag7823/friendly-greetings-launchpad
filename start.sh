#!/bin/bash

# Start ARQ worker in background
python -m arq arq_worker.WorkerSettings &

# Start FastAPI server in foreground
uvicorn fastapi_backend_v2:app --host 0.0.0.0 --port $PORT
