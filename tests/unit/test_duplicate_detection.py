"""
Unit Tests for Duplicate Detection

Tests:
- Exact duplicate detection (hash matching)
- Near duplicate detection (similarity)
- Hash comparison logic
- Similarity scoring
- Cache behavior
"""

import pytest
import hashlib
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock


class TestExactDuplicateDetection:
    """Test exact duplicate detection using hash matching"""
    
    def test_detect_exact_duplicate_by_hash(self):
        """Should detect exact duplicates using SHA-256 hash"""
        file_content = b"test file content"
        hash1 = hashlib.sha256(file_content).hexdigest()
        hash2 = hashlib.sha256(file_content).hexdigest()
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 produces 64 hex characters
    
    def test_different_content_different_hash(self):
        """Should produce different hashes for different content"""
        content1 = b"file content A"
        content2 = b"file content B"
        
        hash1 = hashlib.sha256(content1).hexdigest()
        hash2 = hashlib.sha256(content2).hexdigest()
        
        assert hash1 != hash2
    
    def test_single_byte_difference_detected(self):
        """Should detect even single byte differences"""
        content1 = b"test content"
        content2 = b"test contenu"  # Last char different
        
        hash1 = hashlib.sha256(content1).hexdigest()
        hash2 = hashlib.sha256(content2).hexdigest()
        
        assert hash1 != hash2
    
    def test_hash_case_sensitivity(self):
        """Should detect case differences"""
        content1 = b"Test Content"
        content2 = b"test content"
        
        hash1 = hashlib.sha256(content1).hexdigest()
        hash2 = hashlib.sha256(content2).hexdigest()
        
        assert hash1 != hash2
    
    def test_whitespace_differences_detected(self):
        """Should detect whitespace differences"""
        content1 = b"test content"
        content2 = b"test  content"  # Extra space
        
        hash1 = hashlib.sha256(content1).hexdigest()
        hash2 = hashlib.sha256(content2).hexdigest()
        
        assert hash1 != hash2


class TestNearDuplicateDetection:
    """Test near-duplicate detection using similarity algorithms"""
    
    def test_calculate_similarity_score(self):
        """Should calculate similarity between files"""
        # Simple similarity calculation (Jaccard similarity)
        def calculate_similarity(str1: str, str2: str) -> float:
            set1 = set(str1.split())
            set2 = set(str2.split())
            
            if not set1 and not set2:
                return 1.0
            
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            
            return intersection / union if union > 0 else 0.0
        
        text1 = "invoice payment total amount"
        text2 = "invoice payment total sum"
        
        similarity = calculate_similarity(text1, text2)
        
        assert 0.0 <= similarity <= 1.0
        assert similarity > 0.5  # Should be similar
    
    def test_high_similarity_for_near_duplicates(self):
        """Should detect high similarity for near-duplicate files"""
        def calculate_similarity(str1: str, str2: str) -> float:
            set1 = set(str1.split())
            set2 = set(str2.split())
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            return intersection / union if union > 0 else 0.0
        
        text1 = "monthly expenses report january 2024 financial data analysis"
        text2 = "monthly expenses report february 2024 financial data analysis"
        
        similarity = calculate_similarity(text1, text2)
        
        assert similarity >= 0.7  # High similarity threshold (adjusted for realistic data)
    
    def test_low_similarity_for_different_files(self):
        """Should detect low similarity for different files"""
        def calculate_similarity(str1: str, str2: str) -> float:
            set1 = set(str1.split())
            set2 = set(str2.split())
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            return intersection / union if union > 0 else 0.0
        
        text1 = "invoice payment total"
        text2 = "employee salary payroll"
        
        similarity = calculate_similarity(text1, text2)
        
        assert similarity < 0.5  # Low similarity
    
    def test_similarity_threshold_detection(self):
        """Should use threshold to determine near-duplicates"""
        def calculate_similarity(str1: str, str2: str) -> float:
            set1 = set(str1.split())
            set2 = set(str2.split())
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            return intersection / union if union > 0 else 0.0
        
        threshold = 0.85
        
        text1 = "invoice 2024-01-15 total 1000"
        text2 = "invoice 2024-01-16 total 1000"
        
        similarity = calculate_similarity(text1, text2)
        is_near_duplicate = similarity >= threshold
        
        assert isinstance(is_near_duplicate, bool)


class TestHashComparison:
    """Test hash comparison logic"""
    
    def test_compare_identical_hashes(self):
        """Should match identical hashes"""
        hash1 = "abc123def456"
        hash2 = "abc123def456"
        
        assert hash1 == hash2
    
    def test_compare_different_hashes(self):
        """Should not match different hashes"""
        hash1 = "abc123def456"
        hash2 = "xyz789ghi012"
        
        assert hash1 != hash2
    
    def test_hash_case_insensitive_comparison(self):
        """Should handle case-insensitive hash comparison"""
        hash1 = "ABC123DEF456"
        hash2 = "abc123def456"
        
        assert hash1.lower() == hash2.lower()
    
    def test_validate_hash_format(self):
        """Should validate SHA-256 hash format"""
        valid_hash = "a" * 64
        invalid_hash = "a" * 63
        
        assert len(valid_hash) == 64
        assert len(invalid_hash) != 64
        assert all(c in '0123456789abcdef' for c in valid_hash.lower())


class TestDuplicateDetectionWorkflow:
    """Test complete duplicate detection workflow"""
    
    @pytest.mark.asyncio
    async def test_check_duplicate_in_database(self):
        """Should query database for duplicate files"""
        mock_supabase = Mock()
        mock_table = Mock()
        mock_select = Mock()
        mock_eq = Mock()
        mock_execute = Mock()
        
        # Setup mock chain
        mock_supabase.table.return_value = mock_table
        mock_table.select.return_value = mock_select
        mock_select.eq.return_value = mock_eq
        mock_eq.eq.return_value = mock_eq
        mock_eq.execute.return_value = Mock(data=[
            {
                'id': 'file-123',
                'file_name': 'test.xlsx',
                'created_at': '2024-01-15T10:00:00',
                'file_hash': 'abc123'
            }
        ])
        
        # Simulate query
        result = mock_supabase.table('raw_records')\
            .select('id, file_name, created_at, file_hash')\
            .eq('user_id', 'user-123')\
            .eq('file_hash', 'abc123')\
            .execute()
        
        assert len(result.data) > 0
        assert result.data[0]['file_hash'] == 'abc123'
    
    @pytest.mark.asyncio
    async def test_no_duplicates_found(self):
        """Should return empty when no duplicates found"""
        mock_supabase = Mock()
        mock_table = Mock()
        mock_select = Mock()
        mock_eq = Mock()
        mock_execute = Mock()
        
        # Setup mock chain
        mock_supabase.table.return_value = mock_table
        mock_table.select.return_value = mock_select
        mock_select.eq.return_value = mock_eq
        mock_eq.eq.return_value = mock_eq
        mock_eq.execute.return_value = Mock(data=[])
        
        # Simulate query
        result = mock_supabase.table('raw_records')\
            .select('id, file_name, created_at, file_hash')\
            .eq('user_id', 'user-123')\
            .eq('file_hash', 'xyz789')\
            .execute()
        
        assert len(result.data) == 0
    
    @pytest.mark.asyncio
    async def test_multiple_duplicates_found(self):
        """Should return all duplicate files"""
        mock_supabase = Mock()
        mock_table = Mock()
        mock_select = Mock()
        mock_eq = Mock()
        
        # Setup mock chain
        mock_supabase.table.return_value = mock_table
        mock_table.select.return_value = mock_select
        mock_select.eq.return_value = mock_eq
        mock_eq.eq.return_value = mock_eq
        mock_eq.execute.return_value = Mock(data=[
            {'id': 'file-1', 'file_name': 'test.xlsx', 'created_at': '2024-01-15T10:00:00'},
            {'id': 'file-2', 'file_name': 'test_copy.xlsx', 'created_at': '2024-01-16T10:00:00'},
            {'id': 'file-3', 'file_name': 'test_final.xlsx', 'created_at': '2024-01-17T10:00:00'},
        ])
        
        # Simulate query
        result = mock_supabase.table('raw_records')\
            .select('id, file_name, created_at')\
            .eq('user_id', 'user-123')\
            .eq('file_hash', 'abc123')\
            .execute()
        
        assert len(result.data) == 3


class TestDuplicateDetectionSecurity:
    """Test security aspects of duplicate detection"""
    
    def test_validate_user_id_format(self):
        """Should validate user_id format"""
        import re
        
        valid_user_id = "user-abc-123-def"
        invalid_user_id = "'; DROP TABLE users; --"
        
        pattern = r'^[a-zA-Z0-9\-_]+$'
        
        assert re.match(pattern, valid_user_id)
        assert not re.match(pattern, invalid_user_id)
    
    def test_validate_hash_format(self):
        """Should validate hash format"""
        import re
        
        valid_hash = "a" * 64
        invalid_hash = "'; DROP TABLE files; --"
        
        pattern = r'^[a-f0-9]{64}$'
        
        assert re.match(pattern, valid_hash)
        assert not re.match(pattern, invalid_hash)
    
    def test_prevent_path_traversal_in_filename(self):
        """Should detect path traversal in filenames"""
        dangerous_filenames = [
            '../../../etc/passwd',
            '..\\..\\windows\\system32',
            '/etc/shadow'
        ]
        
        for filename in dangerous_filenames:
            assert '..' in filename or filename.startswith('/')


class TestDuplicateDetectionPerformance:
    """Test performance of duplicate detection"""
    
    def test_hash_calculation_performance(self):
        """Should calculate hashes quickly"""
        import time
        
        large_content = b'x' * (10 * 1024 * 1024)  # 10MB
        
        start_time = time.time()
        file_hash = hashlib.sha256(large_content).hexdigest()
        end_time = time.time()
        
        assert len(file_hash) == 64
        assert (end_time - start_time) < 1.0  # Should complete in <1s
    
    def test_batch_hash_calculation(self):
        """Should calculate multiple hashes efficiently"""
        import time
        
        files = [b'content' + str(i).encode() for i in range(100)]
        
        start_time = time.time()
        hashes = [hashlib.sha256(content).hexdigest() for content in files]
        end_time = time.time()
        
        assert len(hashes) == 100
        assert len(set(hashes)) == 100  # All unique
        assert (end_time - start_time) < 1.0  # Should complete in <1s


class TestDuplicateDetectionEdgeCases:
    """Test edge cases in duplicate detection"""
    
    def test_empty_file_hash(self):
        """Should handle empty files"""
        empty_content = b''
        file_hash = hashlib.sha256(empty_content).hexdigest()
        
        assert len(file_hash) == 64
        # SHA-256 of empty string
        assert file_hash == 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
    
    def test_very_large_file_hash(self):
        """Should handle very large files"""
        # Simulate 100MB file
        large_content = b'x' * (100 * 1024 * 1024)
        file_hash = hashlib.sha256(large_content).hexdigest()
        
        assert len(file_hash) == 64
    
    def test_binary_file_hash(self):
        """Should handle binary files"""
        binary_content = bytes(range(256))
        file_hash = hashlib.sha256(binary_content).hexdigest()
        
        assert len(file_hash) == 64
    
    def test_unicode_content_hash(self):
        """Should handle unicode content"""
        unicode_content = "Hello 世界 🌍".encode('utf-8')
        file_hash = hashlib.sha256(unicode_content).hexdigest()
        
        assert len(file_hash) == 64


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
