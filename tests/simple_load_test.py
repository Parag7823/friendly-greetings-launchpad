"""
Simple Locust Load Test for Ingestion Pipeline
Tests file upload endpoint against Railway deployment
"""

from locust import HttpUser, task, between, events
import random
import io
import uuid
from openpyxl import Workbook
import structlog

logger = structlog.get_logger(__name__)

def generate_excel_bytes(rows=100):
    """Generate simple Excel file"""
    wb = Workbook()
    ws = wb.active
    ws.append(['Date', 'Amount', 'Vendor'])
    for i in range(rows):
        ws.append([f'2024-01-{i%28+1:02d}', round(random.uniform(10, 1000), 2), f'Vendor_{i%5}'])
    excel_bytes = io.BytesIO()
    wb.save(excel_bytes)
    excel_bytes.seek(0)
    return excel_bytes.getvalue()

class IngestionUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        self.user_id = f"load_test_{uuid.uuid4().hex[:8]}"
    
    @task(10)
    def upload_file(self):
        """Upload new file"""
        excel_bytes = generate_excel_bytes(rows=random.randint(50, 150))
        filename = f"test_{uuid.uuid4().hex[:8]}.xlsx"
        
        files = {'file': (filename, excel_bytes, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
        data = {'user_id': self.user_id}
        
        with self.client.post("/api/v1/upload", files=files, data=data, catch_response=True, name="upload") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(1)
    def health_check(self):
        """Health check"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    logger.info("=" * 60)
    logger.info("🚀 LOAD TEST STARTED")
    logger.info(f"Target: {environment.host}")
    logger.info("=" * 60)

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    logger.info("=" * 60)
    logger.info("🏁 LOAD TEST COMPLETED")
    if hasattr(environment.runner, 'stats'):
        stats = environment.runner.stats
        logger.info(f"Total Requests: {stats.total.num_requests}")
        logger.info(f"Total Failures: {stats.total.num_failures}")
        logger.info(f"Avg Response Time: {stats.total.avg_response_time:.0f}ms")
        logger.info(f"Requests/sec: {stats.total.total_rps:.2f}")
        if stats.total.num_requests > 0:
            error_rate = (stats.total.num_failures / stats.total.num_requests) * 100
            logger.info(f"Error Rate: {error_rate:.2f}%")
    logger.info("=" * 60)
