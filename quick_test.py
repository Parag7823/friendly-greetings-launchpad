"""Quick test to verify file upload works with extended timeout."""
import requests
import uuid
import io
import time
from openpyxl import Workbook

# Generate small Excel file (just 3 rows to be fast)
wb = Workbook()
ws = wb.active
ws.append(['Date', 'Amount'])
ws.append(['2024-01-01', 100.00])
ws.append(['2024-01-02', 200.00])
excel_bytes = io.BytesIO()
wb.save(excel_bytes)
excel_bytes.seek(0)

url = "http://localhost:8000/api/process-with-websocket"
files = {'file': ('test_small.xlsx', excel_bytes.getvalue(), 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
data = {
    'user_id': 'quick_test_user',
    'job_id': f"job_{uuid.uuid4().hex[:8]}"
}

print(f"Testing upload to {url}...")
print("Using 120 second timeout...")
start = time.time()
try:
    response = requests.post(url, files=files, data=data, timeout=120)
    elapsed = time.time() - start
    print(f"Status: {response.status_code}")
    print(f"Time: {elapsed:.1f}s")
    if response.status_code == 200:
        print("SUCCESS! File processed successfully")
        print(f"Response: {response.text[:500]}")
    else:
        print(f"FAILED: {response.text[:500]}")
except requests.exceptions.Timeout:
    print(f"TIMEOUT after {time.time() - start:.1f}s")
except Exception as e:
    print(f"Error: {e}")
