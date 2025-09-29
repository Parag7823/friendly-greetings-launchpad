import os
import uuid
import asyncio
from typing import Dict, Any

from celery_app import celery_app


def _use_celery() -> bool:
    return (os.environ.get("USE_CELERY", "").lower() in ("1", "true", "yes"))


@celery_app.task(name="connectors.gmail.sync", bind=True, max_retries=3, default_retry_delay=30)
def task_gmail_sync(self, req: Dict[str, Any]):
    """Run Gmail sync via Nango inside Celery worker.

    req expects keys: { user_id, connection_id, integration_id, mode, lookback_days, max_results }
    """
    try:
        # Lazy import to avoid circular dependencies at module import time
        from fastapi_backend import NangoClient, NANGO_BASE_URL, ConnectorSyncRequest, _gmail_sync_run
    except Exception as e:
        raise self.retry(exc=e, countdown=30)

    try:
        nango = NangoClient(base_url=NANGO_BASE_URL)
        payload = ConnectorSyncRequest(**req)
        result = asyncio.run(_gmail_sync_run(nango, payload))
        return result
    except Exception as e:
        # Retry transient errors
        raise self.retry(exc=e, countdown=min(300, 30 * (self.request.retries + 1)))


@celery_app.task(name="processing.pdf.run", bind=True, max_retries=2, default_retry_delay=20)
def task_pdf_processing(self, user_id: str, filename: str, storage_path: str, job_id: str = None):
    """Process a PDF stored in Supabase Storage and persist results."""
    try:
        from fastapi_backend import start_pdf_processing_job, supabase
    except Exception as e:
        raise self.retry(exc=e, countdown=20)

    if not job_id:
        job_id = str(uuid.uuid4())

    # Insert/ensure ingestion_jobs entry
    try:
        supabase.table('ingestion_jobs').insert({
            'id': job_id,
            'user_id': user_id,
            'file_name': filename,
            'status': 'queued',
            'storage_path': storage_path
        }).execute()
    except Exception:
        pass

    try:
        asyncio.run(start_pdf_processing_job(user_id, job_id, storage_path, filename))
        return {"status": "completed", "job_id": job_id}
    except Exception as e:
        raise self.retry(exc=e, countdown=60)


@celery_app.task(name="processing.spreadsheet.run", bind=True, max_retries=2, default_retry_delay=20)
def task_spreadsheet_processing(self, user_id: str, filename: str, storage_path: str, job_id: str = None):
    """Process a spreadsheet stored in Supabase Storage and persist results."""
    try:
        from fastapi_backend import start_processing_job, supabase
    except Exception as e:
        raise self.retry(exc=e, countdown=20)

    if not job_id:
        job_id = str(uuid.uuid4())

    try:
        supabase.table('ingestion_jobs').insert({
            'id': job_id,
            'user_id': user_id,
            'file_name': filename,
            'status': 'queued',
            'storage_path': storage_path
        }).execute()
    except Exception:
        pass

    try:
        asyncio.run(start_processing_job(user_id, job_id, storage_path, filename))
        return {"status": "completed", "job_id": job_id}
    except Exception as e:
        raise self.retry(exc=e, countdown=60)
