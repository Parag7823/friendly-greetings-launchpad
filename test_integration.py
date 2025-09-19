import pytest
from fastapi.testclient import TestClient
from fastapi_backend import app

client = TestClient(app)

def test_end_to_end_pipeline():
    """Test the full data processing pipeline from file upload to entity resolution."""
    with open("test_data.csv", "rb") as f:
        response = client.post(
            "/api/process-excel-universal",
            files={"file_content": ("test_data.csv", f, "text/csv")},
            data={"filename": "test_data.csv", "user_id": "test-user"},
        )
    assert response.status_code == 200

    # Verify the data in the database
    from supabase import create_client
    import os

    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    supabase = create_client(supabase_url, supabase_key)

    # Verify raw_events table
    response = supabase.table("raw_events").select("*").eq("user_id", "test-user").execute()
    assert len(response.data) == 6

    # Verify normalized_entities table
    response = supabase.table("normalized_entities").select("*").eq("user_id", "test-user").execute()
    assert len(response.data) == 3  # Whole Foods, Best Buy, Shell

def test_rls_policies():
    """Test the Row Level Security policies."""
    from supabase import create_client
    import os

    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_ANON_KEY")
    supabase = create_client(supabase_url, supabase_key)

    # Create a test user and sign in
    test_email = f"test-user-{os.urandom(4).hex()}@test.com"
    test_password = "password"
    user = supabase.auth.sign_up({"email": test_email, "password": test_password})
    supabase.auth.sign_in_with_password({"email": test_email, "password": test_password})

    # Attempt to access data as the test user (should succeed)
    response = supabase.table("raw_events").select("*").eq("user_id", user.user.id).execute()
    assert response.data is not None

    # Attempt to access data as an anonymous user (should fail)
    supabase.auth.sign_out()
    response = supabase.table("raw_events").select("*").execute()
    assert len(response.data) == 0
