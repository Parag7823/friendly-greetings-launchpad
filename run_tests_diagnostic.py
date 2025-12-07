#!/usr/bin/env python3
"""
Quick diagnostic script to run tests and show summary
"""
import subprocess
import sys
import json
from pathlib import Path

def run_tests():
    """Run pytest with timeout and capture results"""
    cmd = [
        sys.executable, '-m', 'pytest',
        'tests/test_ingestion_phases_1_to_5.py',
        '-v',
        '--tb=no',
        '--timeout=300',  # 5 minute timeout per test
        '-ra',  # Show summary of all test outcomes
        '--json-report',
        '--json-report-file=test_report.json'
    ]
    
    print("🚀 Running tests with 5-minute timeout per test...")
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent,
            capture_output=False,
            text=True,
            timeout=600  # 10 minute overall timeout
        )
        return result.returncode
    except subprocess.TimeoutExpired:
        print("\n⏱️ Tests timed out after 10 minutes")
        return 1
    except Exception as e:
        print(f"\n❌ Error running tests: {e}")
        return 1

if __name__ == '__main__':
    exit_code = run_tests()
    sys.exit(exit_code)
