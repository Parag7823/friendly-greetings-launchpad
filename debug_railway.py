"""Single test to get detailed error from Railway"""
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
data = {'user_id': 'debug_user', 'job_id': f"job_debug_{uuid.uuid4().hex[:8]}"}

print(f"Sending request to Railway...")
try:
    r = requests.post(url, files=files, data=data, timeout=60)
    print(f"\nStatus Code: {r.status_code}")
    print(f"\nFull Response Text:\n{r.text}")
    
    # Try to parse JSON
    try:
        import json
        print(f"\nParsed JSON:\n{json.dumps(r.json(), indent=2)}")
    except:
        pass
        
except Exception as e:
    print(f"Request failed: {e}")
