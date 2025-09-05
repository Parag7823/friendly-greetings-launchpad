"""
Unit tests for Finley AI API endpoints
"""
import pytest
from fastapi.testclient import TestClient
from fastapi_backend import app

client = TestClient(app)

class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_check(self):
        """Test health check returns 200"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "status" in data
        assert data["status"] == "healthy"

class TestChatEndpoints:
    """Test chat-related endpoints"""
    
    def test_chat_endpoint(self):
        """Test chat endpoint with valid data"""
        chat_data = {
            "message": "Hello, how can you help me?",
            "user_id": "test_user_123",
            "chat_id": "test_chat_123"
        }
        response = client.post("/chat", json=chat_data)
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "chat_id" in data
    
    def test_chat_history_endpoint(self):
        """Test chat history endpoint"""
        response = client.get("/chat-history/test_user_123")
        assert response.status_code == 200
        data = response.json()
        assert "chats" in data
    
    def test_chat_rename_endpoint(self):
        """Test chat rename endpoint"""
        rename_data = {
            "chat_id": "test_chat_123",
            "new_title": "New Title",
            "user_id": "test_user_123"
        }
        response = client.put("/chat/rename", json=rename_data)
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Chat renamed successfully" in data["message"]
    
    def test_chat_delete_endpoint(self):
        """Test chat delete endpoint"""
        delete_data = {
            "chat_id": "test_chat_123",
            "user_id": "test_user_123"
        }
        response = client.request("DELETE", "/chat/delete", json=delete_data)
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Chat deleted successfully" in data["message"]

class TestFileProcessing:
    """Test file processing endpoints"""
    
    def test_process_file_endpoint_exists(self):
        """Test that process file endpoint exists"""
        # This would test with actual file upload in integration tests
        response = client.post("/upload-and-process")
        # Should return 422 for missing file, not 404
        assert response.status_code == 422
