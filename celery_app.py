import os
from celery import Celery

# Celery configuration
REDIS_URL = os.environ.get("REDIS_URL") or os.environ.get("CELERY_BROKER_URL") or "redis://localhost:6379/0"

celery_app = Celery(
    "finley_ingestion",
    broker=REDIS_URL,
    backend=os.environ.get("CELERY_RESULT_BACKEND", REDIS_URL),
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    worker_prefetch_multiplier=int(os.environ.get("CELERY_PREFETCH", "2")),
    task_acks_late=True,
    task_time_limit=int(os.environ.get("CELERY_TASK_TIME_LIMIT", "900")),  # 15m
    task_soft_time_limit=int(os.environ.get("CELERY_TASK_SOFT_TIME_LIMIT", "840")),  # 14m
)

__all__ = ["celery_app"]
