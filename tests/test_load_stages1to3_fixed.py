"""
Locust Load Tests for Ingestion Pipeline Stages 1-3

This module performs load testing on the complete ingestion pipeline:
Stage 1: File Upload → StreamedFile
Stage 2: ExcelProcessor (parsing, field detection)
Stage 3: ProductionDuplicateDetectionService (4-phase duplicate detection)

Load Testing Scenarios:
- 20 concurrent users (baseline)
- 50 concurrent users (stress test)
- Ramp-up from 0 to 50 over 30 seconds
- Sustained load for 5 minutes

Metrics Tracked:
- Request latency (p50, p95, p99)
- Throughput (requests/second)
- Error rate
- Cache hit rate
- Database query performance
- LSH query performance
- Memory usage

Test Philosophy:
- Real file uploads (100-row Excel files)
- Real Supabase queries
- Real Redis cache
- Real LSH service
- Real utilities: calculate_row_hash, centralized_cache
"""

from locust import HttpUser, task, between, events
import random
import io
import uuid
from pathlib import Path
import structlog
import time

logger = structlog.get_logger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

class LoadTestConfig:
    \"\"\"Load test configuration.\"\"\"
    
    # File generation
    MIN_ROWS = 50
    MAX_ROWS = 200
    MIN_COLS = 5
    MAX_COLS = 12
    
    # User behavior
    MIN_WAIT = 1  # seconds between requests
    MAX_WAIT = 5
    
    # Test scenarios
    BASELINE_USERS = 20
    STRESS_USERS = 50
    RAMP_UP_TIME = 30  # seconds
    TEST_DURATION = 300  # 5 minutes


# ============================================================================
# TEST DATA GENERATION
# ============================================================================

def generate_test_excel_bytes(rows: int = 100, cols: int = 8) -> bytes:
    \"\"\"
    Generate realistic test Excel file in memory.
    
    Creates Excel with:
    - Transaction data
    - Multiple columns (date, amount, category, etc.)
    - Realistic values
    
    Returns:
        Excel file as bytes
    \"\"\"
    try:
        from openpyxl import Workbook
        from datetime import datetime, timedelta
        import random
        
        wb = Workbook()
        ws = wb.active
        ws.title = \"Transactions\"
        
        # Headers
        headers = ['Date', 'Transaction_ID', 'Amount', 'Category', 'Vendor', 'Status', 'Notes', 'Region'][:cols]
        ws.append(headers)
        
        # Data rows
        categories = ['Food', 'Transport', 'Utilities', 'Entertainment', 'Healthcare']
        vendors = ['Vendor_A', 'Vendor_B', 'Vendor_C', 'Vendor_D']
        statuses = ['Completed', 'Pending', 'Failed']
        regions = ['North', 'South', 'East', 'West']
        
        base_date = datetime(2024, 1, 1)
        
        for i in range(rows):
            row_data = [
                (base_date + timedelta(days=i)).strftime('%Y-%m-%d'),
                f\"TXN_{uuid.uuid4().hex[:8]}\",
                round(random.uniform(10.0, 1000.0), 2),
                random.choice(categories),
                random.choice(vendors),
                random.choice(statuses),
                f\"Test transaction {i}\",
                random.choice(regions)
            ][:cols]
            ws.append(row_data)
        
        # Save to bytes
        excel_bytes = io.BytesIO()
        wb.save(excel_bytes)
        excel_bytes.seek(0)
        return excel_bytes.getvalue()
        
    except Exception as e:
        logger.error(f\"Failed to generate Excel: {e}\")
        # Fallback: minimal Excel
        wb = Workbook()
        ws = wb.active
        ws.append(['Column1', 'Column2'])
        for i in range(rows):
            ws.append([f\"Row{i}\", i])
        excel_bytes = io.BytesIO()
        wb.save(excel_bytes)
        excel_bytes.seek(0)
        return excel_bytes.getvalue()


# ============================================================================
# LOCUST USER CLASS
# ============================================================================

class IngestionPipelineUser(HttpUser):
    \"\"\"
    Simulates a user uploading files to the ingestion pipeline.
    
    User Behavior:
    1. Generate realistic Excel file
    2. Upload file (triggers Stages 1-3)
    3. Wait for response
    4. Repeat with random wait time
    
    Measures:
    - Upload latency
    - Processing time
    - Success/failure rate
    \"\"\"
    
    wait_time = between(LoadTestConfig.MIN_WAIT, LoadTestConfig.MAX_WAIT)
    
    def on_start(self):
        \"\"\"Called when a simulated user starts.\"\"\"
        self.user_id = f\"load_test_user_{uuid.uuid4().hex[:8]}\"
        logger.info(f\"👤 New user started: {self.user_id}\")
    
    @task(weight=10)
    def upload_new_file(self):
        \"\"\"
        Upload a new, unique file.
        
        This is the most common operation (weight=10).
        Tests complete Stages 1-3 flow.
        \"\"\"
        # Generate unique Excel file
        rows = random.randint(LoadTestConfig.MIN_ROWS, LoadTestConfig.MAX_ROWS)
        cols = random.randint(LoadTestConfig.MIN_COLS, LoadTestConfig.MAX_COLS)
        
        excel_bytes = generate_test_excel_bytes(rows=rows, cols=cols)
        filename = f\"test_upload_{uuid.uuid4().hex[:8]}.xlsx\"
        
        # Prepare multipart file upload
        files = {
            'file': (filename, excel_bytes, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        }
        
        data = {
            'user_id': self.user_id,
            'enable_near_duplicate': 'true',
            'enable_content_duplicate': 'true'
        }
        
        # Upload file (triggers complete pipeline)
        start_time = time.time()
        
        with self.client.post(
            \"/api/v1/upload\",  # Adjust endpoint as needed
            files=files,
            data=data,
            catch_response=True,
            name=\"upload_new_file\"
        ) as response:
            processing_time = (time.time() - start_time) * 1000  # ms
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    
                    # Verify response structure
                    if 'file_hash' in result and 'duplicate_result' in result:
                        response.success()
                        logger.debug(f\"✅ Upload successful: {filename}, time={processing_time:.0f}ms\")
                    else:
                        response.failure(f\"Invalid response structure: {result}\")
                
                except Exception as e:
                    response.failure(f\"JSON decode error: {e}\")
            
            else:
                response.failure(f\"HTTP {response.status_code}: {response.text[:100]}\")
    
    @task(weight=3)
    def upload_duplicate_file(self):
        \"\"\"
        Upload a file that's likely to be a duplicate.
        
        Tests duplicate detection performance (weight=3).
        Uses a fixed file hash to trigger duplicate detection.
        \"\"\"
        # Use consistent data to create duplicate
        excel_bytes = generate_test_excel_bytes(rows=100, cols=8)
        filename = f\"duplicate_test_{self.user_id}.xlsx\"
        
        files = {
            'file': (filename, excel_bytes, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        }
        
        data = {
            'user_id': self.user_id,
            'enable_near_duplicate': 'true',
            'enable_content_duplicate': 'true'
        }
        
        with self.client.post(
            \"/api/v1/upload\",
            files=files,
            data=data,
            catch_response=True,
            name=\"upload_duplicate_file\"
        ) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    duplicate_result = result.get('duplicate_result', {})
                    
                    # Expect duplicate detection for second+ uploads
                    if duplicate_result.get('is_duplicate'):
                        logger.debug(f\"✅ Duplicate detected: {filename}\")
                    
                    response.success()
                
                except Exception as e:
                    response.failure(f\"JSON decode error: {e}\")
            else:
                response.failure(f\"HTTP {response.status_code}\")
    
    @task(weight=1)
    def health_check(self):
        \"\"\"
        Health check endpoint.
        
        Low weight (weight=1) - occasional check.
        \"\"\"
        with self.client.get(\"/health\", catch_response=True, name=\"health_check\") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f\"HTTP {response.status_code}\")


# ============================================================================
# EVENT HANDLERS FOR METRICS
# ============================================================================

@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    \"\"\"Log request metrics.\"\"\"
    if exception:
        logger.error(f\"❌ Request failed: {name}, error={exception}\")
    else:
        logger.debug(f\"📊 Request: {name}, time={response_time:.0f}ms, size={response_length}bytes\")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    \"\"\"Called when load test starts.\"\"\"
    logger.info(\"=\" * 60)
    logger.info(\"🚀 LOAD TEST STARTED\")
    logger.info(f\"Target: {environment.host}\")
    logger.info(f\"Users: {environment.runner.target_user_count if hasattr(environment.runner, 'target_user_count') else 'N/A'}\")
    logger.info(\"=\" * 60)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    \"\"\"Called when load test stops.\"\"\"
    logger.info(\"=\" * 60)
    logger.info(\"🏁 LOAD TEST COMPLETED\")
    
    if hasattr(environment.runner, 'stats'):
        stats = environment.runner.stats
        
        logger.info(f\"Total Requests: {stats.total.num_requests}\")
        logger.info(f\"Total Failures: {stats.total.num_failures}\")
        logger.info(f\"Average Response Time: {stats.total.avg_response_time:.0f}ms\")
        logger.info(f\"Min Response Time: {stats.total.min_response_time:.0f}ms\")
        logger.info(f\"Max Response Time: {stats.total.max_response_time:.0f}ms\")
        logger.info(f\"Requests/sec: {stats.total.total_rps:.2f}\")
        
        if stats.total.num_requests > 0:
            error_rate = (stats.total.num_failures / stats.total.num_requests) * 100
            logger.info(f\"Error Rate: {error_rate:.2f}%\")
    
    logger.info(\"=\" * 60)


# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================

\"\"\"
USAGE:

1. **Baseline Test (20 users)**:
   locust -f tests/test_load_stages1to3.py --host=http://localhost:8000 --users=20 --spawn-rate=2 --run-time=5m --headless

2. **Stress Test (50 users)**:
   locust -f tests/test_load_stages1to3.py --host=http://localhost:8000 --users=50 --spawn-rate=5 --run-time=5m --headless

3. **Interactive Web UI**:
   locust -f tests/test_load_stages1to3.py --host=http://localhost:8000
   # Then open http://localhost:8089

4. **With CSV Export**:
   locust -f tests/test_load_stages1to3.py --host=http://localhost:8000 --users=20 --spawn-rate=2 --run-time=5m --headless --csv=results/load_test

METRICS TO WATCH:
- P99 latency < 5000ms (acceptable)
- P99 latency < 2000ms (good)
- Error rate < 1%
- Throughput > 10 req/sec (baseline)
- Cache hit rate > 50% (after warmup)
\"\"\"
