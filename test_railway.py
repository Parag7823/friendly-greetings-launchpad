"""Test single request against Railway to see detailed error"""
import requests
import uuid
import io
from openpyxl import Workbook

# Create test Excel
wb = Workbook()
ws = wb.active
ws.append(['Date', 'Amount'])
ws.append(['2024-01-01', 100.00])
excel_bytes = io.BytesIO()
wb.save(excel_bytes)
excel_bytes.seek(0)

url = "https://friendly-greetings-launchpad-production.up.railway.app/api/process-with-websocket"
files = {'file': ('test.xlsx', excel_bytes.getvalue(), 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
data = {'user_id': 'test_user', 'job_id': f"job_{uuid.uuid4().hex[:8]}"}

print(f"Testing Railway: {url}")
try:
    r = requests.post(url, files=files, data=data, timeout=120)
    print(f"Status: {r.status_code}")
    print(f"Response: {r.text}")
except Exception as e:
    print(f"Error: {e}")
