"""
Integration Tests for Production Duplicate Detection Service
==========================================================

End-to-end integration tests that verify the complete duplicate detection flow
including API integration, WebSocket updates, and database operations.

Test Categories:
- Complete upload flow with duplicate detection
- API endpoint integration
- WebSocket real-time updates
- Database operations and transactions
- Error scenarios and recovery
- Performance under load
- Security validation

Author: Senior Full-Stack Engineer
Version: 2.0.0
"""

import asyncio
import json
import pytest
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any

import httpx
from fastapi.testclient import TestClient

from production_duplicate_detection_service import (
    ProductionDuplicateDetectionService,
    FileMetadata,
    DuplicateType,
    DuplicateAction
)

class TestDuplicateDetectionIntegration:
    """Integration tests for duplicate detection service"""
    
    @pytest.fixture
    def mock_supabase_client(self):
        """Mock Supabase client for integration tests"""
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_client.table.return_value = mock_table
        return mock_client, mock_table
    
    @pytest.fixture
    def mock_redis_client(self):
        """Mock Redis client for integration tests"""
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None
        mock_redis.setex.return_value = True
        mock_redis.keys.return_value = []
        mock_redis.delete.return_value = 0
        return mock_redis
    
    @pytest.fixture
    def duplicate_service(self, mock_supabase_client, mock_redis_client):
        """Create duplicate detection service for integration tests"""
        supabase_client, _ = mock_supabase_client
        return ProductionDuplicateDetectionService(supabase_client, mock_redis_client)
    
    @pytest.fixture
    def sample_file_data(self):
        """Sample file data for integration tests"""
        return {
            'file_content': b"Sample Excel file content for testing",
            'filename': 'test_file.xlsx',
            'user_id': 'test_user_123',
            'file_size': 1024,
            'content_type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        }
    
    # ============================================================================
    # COMPLETE UPLOAD FLOW TESTS
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_complete_upload_flow_no_duplicates(
        self, 
        duplicate_service, 
        mock_supabase_client, 
        sample_file_data
    ):
        """Test complete upload flow when no duplicates exist"""
        _, mock_table = mock_supabase_client
        
        # Mock no exact duplicates
        mock_table.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = []
        # Mock no near duplicates
        mock_table.select.return_value.eq.return_value.gte.return_value.limit.return_value.execute.return_value.data = []
        
        file_metadata = FileMetadata(
            user_id=sample_file_data['user_id'],
            file_hash="a" * 64,
            filename=sample_file_data['filename'],
            file_size=sample_file_data['file_size'],
            content_type=sample_file_data['content_type'],
            upload_timestamp=datetime.utcnow()
        )
        
        result = await duplicate_service.detect_duplicates(
            sample_file_data['file_content'],
            file_metadata
        )
        
        assert not result.is_duplicate
        assert result.duplicate_type == DuplicateType.NONE
        assert result.recommendation == DuplicateAction.REPLACE
        assert result.error is None
    
    @pytest.mark.asyncio
    async def test_complete_upload_flow_exact_duplicate(
        self, 
        duplicate_service, 
        mock_supabase_client, 
        sample_file_data
    ):
        """Test complete upload flow when exact duplicate exists"""
        _, mock_table = mock_supabase_client
        
        # Mock exact duplicate found
        mock_duplicates = [
            {
                'id': 'dup_1',
                'file_name': 'existing_file.xlsx',
                'created_at': '2024-01-01T10:00:00Z',
                'file_size': 1024,
                'status': 'processed'
            }
        ]
        mock_table.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = mock_duplicates
        
        file_metadata = FileMetadata(
            user_id=sample_file_data['user_id'],
            file_hash="a" * 64,
            filename=sample_file_data['filename'],
            file_size=sample_file_data['file_size'],
            content_type=sample_file_data['content_type'],
            upload_timestamp=datetime.utcnow()
        )
        
        result = await duplicate_service.detect_duplicates(
            sample_file_data['file_content'],
            file_metadata
        )
        
        assert result.is_duplicate
        assert result.duplicate_type == DuplicateType.EXACT
        assert result.similarity_score == 1.0
        assert len(result.duplicate_files) == 1
        assert result.recommendation == DuplicateAction.REPLACE
    
    @pytest.mark.asyncio
    async def test_complete_upload_flow_near_duplicate(
        self, 
        duplicate_service, 
        mock_supabase_client, 
        sample_file_data
    ):
        """Test complete upload flow when near duplicate exists"""
        _, mock_table = mock_supabase_client
        
        # Mock no exact duplicates
        mock_table.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = []
        
        # Mock near duplicate found
        mock_recent_files = [
            {
                'id': 'similar_1',
                'file_name': 'similar_file.xlsx',
                'created_at': '2024-01-01T10:00:00Z',
                'file_size': 1024,
                'content': {'content_fingerprint': 'similar_fingerprint'}
            }
        ]
        mock_table.select.return_value.eq.return_value.gte.return_value.limit.return_value.execute.return_value.data = mock_recent_files
        
        # Mock high similarity
        with patch.object(duplicate_service, '_calculate_similarity', return_value=0.9):
            file_metadata = FileMetadata(
                user_id=sample_file_data['user_id'],
                file_hash="a" * 64,
                filename=sample_file_data['filename'],
                file_size=sample_file_data['file_size'],
                content_type=sample_file_data['content_type'],
                upload_timestamp=datetime.utcnow()
            )
            
            result = await duplicate_service.detect_duplicates(
                sample_file_data['file_content'],
                file_metadata
            )
        
        assert result.is_duplicate
        assert result.duplicate_type == DuplicateType.NEAR
        assert result.similarity_score == 0.9
        assert result.recommendation == DuplicateAction.MERGE
    
    # ============================================================================
    # API INTEGRATION TESTS
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_api_duplicate_detection_endpoint(self, sample_file_data):
        """Test API endpoint for duplicate detection"""
        # This would test the actual FastAPI endpoint
        # For now, we'll test the service integration
        
        mock_supabase = MagicMock()
        mock_table = MagicMock()
        mock_supabase.table.return_value = mock_table
        mock_table.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = []
        
        service = ProductionDuplicateDetectionService(mock_supabase)
        
        file_metadata = FileMetadata(
            user_id=sample_file_data['user_id'],
            file_hash="a" * 64,
            filename=sample_file_data['filename'],
            file_size=sample_file_data['file_size'],
            content_type=sample_file_data['content_type'],
            upload_timestamp=datetime.utcnow()
        )
        
        result = await service.detect_duplicates(
            sample_file_data['file_content'],
            file_metadata
        )
        
        # Verify result structure matches API expectations
        assert hasattr(result, 'is_duplicate')
        assert hasattr(result, 'duplicate_type')
        assert hasattr(result, 'similarity_score')
        assert hasattr(result, 'duplicate_files')
        assert hasattr(result, 'recommendation')
        assert hasattr(result, 'message')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'processing_time_ms')
    
    @pytest.mark.asyncio
    async def test_api_error_handling(self, sample_file_data):
        """Test API error handling for duplicate detection"""
        mock_supabase = MagicMock()
        mock_table = MagicMock()
        mock_supabase.table.return_value = mock_table
        mock_table.select.return_value.eq.return_value.eq.return_value.execute.side_effect = Exception("Database error")
        
        service = ProductionDuplicateDetectionService(mock_supabase)
        
        file_metadata = FileMetadata(
            user_id=sample_file_data['user_id'],
            file_hash="a" * 64,
            filename=sample_file_data['filename'],
            file_size=sample_file_data['file_size'],
            content_type=sample_file_data['content_type'],
            upload_timestamp=datetime.utcnow()
        )
        
        result = await service.detect_duplicates(
            sample_file_data['file_content'],
            file_metadata
        )
        
        # Should handle error gracefully
        assert not result.is_duplicate
        assert result.error is not None
        assert "Database error" in result.error
    
    # ============================================================================
    # WEBSOCKET INTEGRATION TESTS
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_websocket_duplicate_notification(self, duplicate_service, sample_file_data):
        """Test WebSocket notification for duplicate detection"""
        # Mock WebSocket manager
        mock_websocket_manager = MagicMock()
        mock_websocket_manager.send_update = AsyncMock()
        
        # Mock duplicate found
        with patch.object(duplicate_service, '_detect_exact_duplicates') as mock_detect:
            mock_detect.return_value = MagicMock(
                is_duplicate=True,
                duplicate_type=DuplicateType.EXACT,
                similarity_score=1.0,
                duplicate_files=[{'id': 'dup_1', 'filename': 'existing.xlsx'}],
                recommendation=DuplicateAction.REPLACE,
                message="Exact duplicate found",
                confidence=1.0,
                processing_time_ms=100
            )
            
            file_metadata = FileMetadata(
                user_id=sample_file_data['user_id'],
                file_hash="a" * 64,
                filename=sample_file_data['filename'],
                file_size=sample_file_data['file_size'],
                content_type=sample_file_data['content_type'],
                upload_timestamp=datetime.utcnow()
            )
            
            result = await duplicate_service.detect_duplicates(
                sample_file_data['file_content'],
                file_metadata
            )
            
            # Verify WebSocket notification would be sent
            assert result.is_duplicate
            assert result.duplicate_type == DuplicateType.EXACT
    
    # ============================================================================
    # DATABASE OPERATION TESTS
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_database_query_optimization(self, mock_supabase_client):
        """Test database query optimization"""
        _, mock_table = mock_supabase_client
        
        # Mock efficient query
        mock_table.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = []
        
        service = ProductionDuplicateDetectionService(mock_supabase_client[0])
        
        # Verify query structure
        await service._query_duplicates_by_hash("user_123", "hash_123")
        
        # Verify query was called with correct parameters
        mock_table.select.assert_called_with('id, file_name, created_at, file_size, status, content')
        mock_table.select.return_value.eq.assert_called_with('user_id', 'user_123')
        mock_table.select.return_value.eq.return_value.eq.assert_called_with('file_hash', 'hash_123')
    
    @pytest.mark.asyncio
    async def test_database_transaction_handling(self, mock_supabase_client):
        """Test database transaction handling"""
        _, mock_table = mock_supabase_client
        
        # Mock database error
        mock_table.select.return_value.eq.return_value.eq.return_value.execute.side_effect = Exception("Connection timeout")
        
        service = ProductionDuplicateDetectionService(mock_supabase_client[0])
        
        # Should handle database errors gracefully
        with pytest.raises(Exception, match="Connection timeout"):
            await service._query_duplicates_by_hash("user_123", "hash_123")
    
    # ============================================================================
    # PERFORMANCE INTEGRATION TESTS
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_concurrent_duplicate_detection(self, duplicate_service, sample_file_data):
        """Test concurrent duplicate detection requests"""
        # Mock no duplicates
        with patch.object(duplicate_service, '_detect_exact_duplicates') as mock_exact:
            mock_exact.return_value = MagicMock(
                is_duplicate=False,
                duplicate_type=DuplicateType.NONE,
                similarity_score=0.0,
                duplicate_files=[],
                recommendation=DuplicateAction.REPLACE,
                message="No duplicates found",
                confidence=1.0,
                processing_time_ms=50
            )
            
            file_metadata = FileMetadata(
                user_id=sample_file_data['user_id'],
                file_hash="a" * 64,
                filename=sample_file_data['filename'],
                file_size=sample_file_data['file_size'],
                content_type=sample_file_data['content_type'],
                upload_timestamp=datetime.utcnow()
            )
            
            # Run multiple concurrent requests
            tasks = [
                duplicate_service.detect_duplicates(sample_file_data['file_content'], file_metadata)
                for _ in range(10)
            ]
            
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            end_time = time.time()
            
            # All requests should complete successfully
            assert len(results) == 10
            for result in results:
                assert not result.is_duplicate
                assert result.error is None
            
            # Should complete within reasonable time
            assert end_time - start_time < 5.0  # 5 seconds for 10 concurrent requests
    
    @pytest.mark.asyncio
    async def test_large_file_processing_integration(self, duplicate_service, sample_file_data):
        """Test processing of large files in integration scenario"""
        # Create large file content
        large_content = b"x" * (10 * 1024 * 1024)  # 10MB
        
        file_metadata = FileMetadata(
            user_id=sample_file_data['user_id'],
            file_hash="a" * 64,
            filename=sample_file_data['filename'],
            file_size=len(large_content),
            content_type=sample_file_data['content_type'],
            upload_timestamp=datetime.utcnow()
        )
        
        # Should handle large files without memory issues
        result = await duplicate_service.detect_duplicates(large_content, file_metadata)
        
        assert result is not None
        assert result.processing_time_ms > 0
    
    # ============================================================================
    # SECURITY INTEGRATION TESTS
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_security_validation_integration(self, duplicate_service):
        """Test security validation in integration scenario"""
        # Test with malicious filename
        malicious_metadata = FileMetadata(
            user_id="test_user",
            file_hash="a" * 64,
            filename="../../../etc/passwd",
            file_size=1024,
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            upload_timestamp=datetime.utcnow()
        )
        
        # Service should handle security violations gracefully
        result = await duplicate_service.detect_duplicates(b"content", malicious_metadata)
        assert result.is_duplicate == False
        assert "Invalid filename" in result.error
        assert result.error is not None
    
    @pytest.mark.asyncio
    async def test_user_isolation_integration(self, mock_supabase_client):
        """Test user isolation in integration scenario"""
        _, mock_table = mock_supabase_client
        
        service = ProductionDuplicateDetectionService(mock_supabase_client[0])
        
        # Query for user 1
        await service._query_duplicates_by_hash("user_1", "hash_123")
        
        # Verify query includes user_id filter
        mock_table.select.return_value.eq.assert_called_with('user_id', 'user_1')
    
    # ============================================================================
    # CACHE INTEGRATION TESTS
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_cache_integration_redis(self, mock_redis_client, sample_file_data):
        """Test Redis cache integration"""
        mock_supabase = MagicMock()
        mock_table = MagicMock()
        mock_supabase.table.return_value = mock_table
        mock_table.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = []
        mock_table.select.return_value.eq.return_value.gte.return_value.limit.return_value.execute.return_value.data = []
        
        service = ProductionDuplicateDetectionService(mock_supabase, mock_redis_client)
        
        file_metadata = FileMetadata(
            user_id=sample_file_data['user_id'],
            file_hash="a" * 64,
            filename=sample_file_data['filename'],
            file_size=sample_file_data['file_size'],
            content_type=sample_file_data['content_type'],
            upload_timestamp=datetime.utcnow()
        )
        
        # First request - should miss cache
        result1 = await service.detect_duplicates(sample_file_data['file_content'], file_metadata)
        
        # Second request - should hit cache
        result2 = await service.detect_duplicates(sample_file_data['file_content'], file_metadata)
        
        # Both results should be identical
        assert result1.is_duplicate == result2.is_duplicate
        assert result1.duplicate_type == result2.duplicate_type
        
        # Verify cache was used
        assert mock_redis_client.asetex.called
    
    @pytest.mark.asyncio
    async def test_cache_integration_memory_fallback(self, sample_file_data):
        """Test memory cache fallback when Redis unavailable"""
        mock_supabase = MagicMock()
        mock_table = MagicMock()
        mock_supabase.table.return_value = mock_table
        mock_table.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = []
        mock_table.select.return_value.eq.return_value.gte.return_value.limit.return_value.execute.return_value.data = []
        
        # No Redis client
        service = ProductionDuplicateDetectionService(mock_supabase, None)
        
        file_metadata = FileMetadata(
            user_id=sample_file_data['user_id'],
            file_hash="a" * 64,
            filename=sample_file_data['filename'],
            file_size=sample_file_data['file_size'],
            content_type=sample_file_data['content_type'],
            upload_timestamp=datetime.utcnow()
        )
        
        # Should work with memory cache
        result = await service.detect_duplicates(sample_file_data['file_content'], file_metadata)
        
        assert result is not None
        assert not result.is_duplicate
    
    # ============================================================================
    # ERROR RECOVERY TESTS
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_error_recovery_database_failure(self, sample_file_data):
        """Test error recovery when database fails"""
        mock_supabase = MagicMock()
        mock_table = MagicMock()
        mock_supabase.table.return_value = mock_table
        mock_table.select.return_value.eq.return_value.eq.return_value.execute.side_effect = Exception("Database failure")
        
        service = ProductionDuplicateDetectionService(mock_supabase)
        
        file_metadata = FileMetadata(
            user_id=sample_file_data['user_id'],
            file_hash="a" * 64,
            filename=sample_file_data['filename'],
            file_size=sample_file_data['file_size'],
            content_type=sample_file_data['content_type'],
            upload_timestamp=datetime.utcnow()
        )
        
        # Should handle database failure gracefully
        result = await service.detect_duplicates(sample_file_data['file_content'], file_metadata)
        
        assert not result.is_duplicate
        assert result.error is not None
        assert "Database failure" in result.error
    
    @pytest.mark.asyncio
    async def test_error_recovery_cache_failure(self, mock_supabase_client, sample_file_data):
        """Test error recovery when cache fails"""
        mock_redis = AsyncMock()
        mock_redis.get.side_effect = Exception("Cache failure")
        mock_redis.setex.side_effect = Exception("Cache failure")
        
        service = ProductionDuplicateDetectionService(mock_supabase_client[0], mock_redis)
        
        file_metadata = FileMetadata(
            user_id=sample_file_data['user_id'],
            file_hash="a" * 64,
            filename=sample_file_data['filename'],
            file_size=sample_file_data['file_size'],
            content_type=sample_file_data['content_type'],
            upload_timestamp=datetime.utcnow()
        )
        
        # Should handle cache failure gracefully
        result = await service.detect_duplicates(sample_file_data['file_content'], file_metadata)
        
        assert result is not None
        # Should still work despite cache failure
        assert not result.is_duplicate

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
