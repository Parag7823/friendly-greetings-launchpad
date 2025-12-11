"""Test the SIMPLE endpoint on Railway"""
import requests
import uuid
import io
from openpyxl import Workbook

# Create test Excel
wb = Workbook()
ws = wb.active
ws.append(['Date', 'Amount', 'Category'])
ws.append(['2024-01-01', 100.00, 'Food'])
ws.append(['2024-01-02', 50.00, 'Transport'])
excel_bytes = io.BytesIO()
wb.save(excel_bytes)
excel_bytes.seek(0)

# Test SIMPLE endpoint
url = "https://friendly-greetings-launchpad-production.up.railway.app/api/upload-simple"
files = {'file': ('test.xlsx', excel_bytes.getvalue(), 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
data = {'user_id': 'debug_user', 'job_id': f"job_debug_{uuid.uuid4().hex[:8]}"}

print(f"Testing: {url}")
try:
    r = requests.post(url, files=files, data=data, timeout=60)
    print(f"\nStatus: {r.status_code}")
    print(f"Response: {r.text}")
except Exception as e:
    print(f"Error: {e}")
