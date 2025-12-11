"""Check Railway deployment status and get detailed logs"""
import requests

# Check what commit Railway is running
url = "https://friendly-greetings-launchpad-production.up.railway.app/health"
try:
    r = requests.get(url, timeout=10)
    print(f"Health Check Status: {r.status_code}")
    print(f"Response: {r.text[:500]}")
except Exception as e:
    print(f"Health check failed: {e}")

# Try to get version info if available
try:
    version_url = "https://friendly-greetings-launchpad-production.up.railway.app/version"
    r = requests.get(version_url, timeout=10)
    print(f"\nVersion endpoint: {r.status_code}")
    print(r.text[:500])
except:
    print("\nNo version endpoint")
