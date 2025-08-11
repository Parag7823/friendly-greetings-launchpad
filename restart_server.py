#!/usr/bin/env python3
"""
Simple script to restart the FastAPI server and check for syntax errors
"""

import subprocess
import sys
import os
import time

def check_syntax():
    """Check if the FastAPI backend file has syntax errors"""
    print("🔍 Checking for syntax errors...")
    try:
        with open('fastapi_backend.py', 'r', encoding='utf-8') as f:
            compile(f.read(), 'fastapi_backend.py', 'exec')
        print("✅ No syntax errors found!")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error found: {e}")
        return False
    except Exception as e:
        print(f"❌ Error checking syntax: {e}")
        return False

def restart_server():
    """Restart the FastAPI server"""
    print("🔄 Restarting FastAPI server...")
    
    # Kill any existing processes
    try:
        subprocess.run(['pkill', '-f', 'fastapi_backend.py'], capture_output=True)
        time.sleep(2)
    except:
        pass
    
    # Start the server
    try:
        process = subprocess.Popen([
            sys.executable, 'fastapi_backend.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Check if process is still running
        if process.poll() is None:
            print("✅ Server started successfully!")
            print("🌐 Server should be running on http://localhost:8000")
            return True
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Server failed to start:")
            print(f"STDOUT: {stdout.decode()}")
            print(f"STDERR: {stderr.decode()}")
            return False
            
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False

def main():
    print("🚀 Finley AI Server Restart Tool")
    print("=" * 40)
    
    # Check syntax first
    if not check_syntax():
        print("\n❌ Please fix syntax errors before restarting the server.")
        return
    
    # Restart server
    if restart_server():
        print("\n✅ Server restart completed successfully!")
        print("\n📋 Next steps:")
        print("1. Test the endpoints in Postman")
        print("2. Check if 404 errors are resolved")
        print("3. Verify relationship detection is working")
    else:
        print("\n❌ Server restart failed. Check the error messages above.")

if __name__ == "__main__":
    main()
