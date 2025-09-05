"""
Behave environment configuration for Finley AI BDD tests
"""
import os
import sys
import asyncio
from pathlib import Path

# Set test environment variables BEFORE importing the app
os.environ['TESTING'] = 'true'
os.environ['SUPABASE_URL'] = os.getenv('SUPABASE_URL', 'https://test.supabase.co')
os.environ['SUPABASE_SERVICE_KEY'] = os.getenv('SUPABASE_SERVICE_KEY', 'test-key')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'test-key')

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import FastAPI app
from fastapi_backend import app
from fastapi.testclient import TestClient

def before_all(context):
    """Set up test environment before all tests"""
    print("🚀 Setting up Finley AI BDD test environment...")
    
    # Create test client
    context.client = TestClient(app)
    context.base_url = "http://testserver"
    
    print("✅ Test environment ready!")

def before_scenario(context, scenario):
    """Set up before each scenario"""
    context.scenario_data = {}
    context.response = None
    context.error = None
    print(f"\n📋 Running scenario: {scenario.name}")

def after_scenario(context, scenario):
    """Clean up after each scenario"""
    if hasattr(context, 'scenario_data'):
        context.scenario_data.clear()
    print(f"✅ Completed scenario: {scenario.name}")

def after_all(context):
    """Clean up after all tests"""
    print("\n🎉 All BDD tests completed!")
    if hasattr(context, 'client'):
        context.client.close()
