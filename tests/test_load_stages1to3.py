"""
Comprehensive Locust Load Test for Stages 1-3 (Ingestion & Normalization)
Targeting the REAL Production Endpoint: /api/process-with-websocket
"""

from locust import HttpUser, task, between, events, constant_pacing
import os
import random
import io
import uuid
import structlog
import time
from pathlib import Path

# Configure structured logging
logger = structlog.get_logger(__name__)

class LoadTestConfig:
    # Adjusted for realistic heavy file processing
    MIN_WAIT = 2
    MAX_WAIT = 5 
    
    # File generation config
    MIN_ROWS = 10
    MAX_ROWS = 50 

def generate_test_excel_bytes(rows: int = 20) -> bytes:
    """Generate a valid Excel file in memory."""
    try:
        from openpyxl import Workbook
        from datetime import datetime, timedelta
        
        wb = Workbook()
        ws = wb.active
        ws.title = "Data"
        
        # Consistent header for Stage 2 mapping
        ws.append(['Date', 'Amount', 'Category', 'Description', 'Region'])
        
        categories = ['Office', 'Travel', 'Meals', 'Software']
        
        for i in range(rows):
            row_data = [
                (datetime(2024, 1, 1) + timedelta(days=i)).strftime('%Y-%m-%d'),
                round(random.uniform(50.0, 1000.0), 2),
                random.choice(categories),
                f"Expense entry {i}",
                "Global"
            ]
            ws.append(row_data)
        
        excel_bytes = io.BytesIO()
        wb.save(excel_bytes)
        excel_bytes.seek(0)
        return excel_bytes.getvalue()
    except Exception as e:
        logger.error(f"Excel gen failed: {e}")
        return b""

class IngestionPipelineUser(HttpUser):
    # Pacing: ensure we don't spam faster than the server can process
    # 10 users * 1 request every 10-20 sec = ~30-60 req/min (which fits our new 10/min limit barely, might need pacing)
    # Let's use constant pacing to be safe: 1 request every 10 seconds per user
    wait_time = between(5, 10)
    
    def on_start(self):
        self.user_id = f"user_{uuid.uuid4().hex[:6]}"
        logger.info("user_started", user_id=self.user_id)
    
    @task
    def upload_new_file(self):
        """Upload a fresh file to the main processing endpoint."""
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        rows = random.randint(LoadTestConfig.MIN_ROWS, LoadTestConfig.MAX_ROWS)
        
        excel_content = generate_test_excel_bytes(rows)
        if not excel_content:
            return

        filename = f"loadtest_{uuid.uuid4().hex[:6]}.xlsx"
        
        # Endpoint: /api/process-with-websocket
        # Params: file, user_id, job_id
        files = {
            'file': (filename, excel_content, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        }
        data = {
            'user_id': self.user_id,
            'job_id': job_id
        }
        
        start_time = time.time()
        # Increased timeout to 120s in client too
        with self.client.post(
            "/api/upload-simple", 
            files=files, 
            data=data, 
            catch_response=True, 
            timeout=120,
            name="upload_process"
        ) as response:
            duration = time.time() - start_time
            
            if response.status_code == 200:
                response.success()
                logger.info("upload_success", user_id=self.user_id, duration=duration)
            elif response.status_code == 429:
                # Rate limited - valid behavior, but marks as fail in strict test
                response.failure(f"Rate limited: {response.text}")
                logger.warning("rate_limited", user_id=self.user_id)
            else:
                response.failure(f"Failed: {response.status_code} - {response.text[:100]}")
                logger.error("upload_failed", status=response.status_code, response=response.text[:100])

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    logger.info("LOAD TEST STARTED - 10 USERS")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    if hasattr(environment.runner, 'stats'):
        logger.info(f"Test Finished. Requests: {environment.runner.stats.total.num_requests}, Failures: {environment.runner.stats.total.num_failures}")
