"""Pytest configuration and fixtures for integration tests"""

import os
import sys
from pathlib import Path

# Load environment variables from .env.test BEFORE any imports
import dotenv
env_path = Path(__file__).parent.parent / '.env.test'
dotenv.load_dotenv(env_path)

# Verify critical environment variables are loaded
required_vars = ['SUPABASE_URL', 'SUPABASE_KEY', 'SUPABASE_ANON_KEY', 'GROQ_API_KEY']
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    print(f"⚠️ WARNING: Missing environment variables: {missing_vars}")
    print(f"   Load from: {env_path}")
else:
    print(f"✅ Environment variables loaded from {env_path}")
