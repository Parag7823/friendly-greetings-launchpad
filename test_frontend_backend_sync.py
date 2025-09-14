"""
Frontend/Backend Synchronization Tests
=====================================

This module tests the synchronization between frontend and backend components:
1. WebSocket real-time updates
2. Progress tracking and display
3. Error handling and user feedback
4. Data consistency between UI and backend
5. User experience under load
6. Responsive design and performance
7. State management and data flow

Author: Senior Full-Stack Engineer
Version: 1.0.0
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
from typing import Dict, List, Any

# Mock frontend components
class MockWebSocketConnection:
    """Mock WebSocket connection for testing"""
    
    def __init__(self):
        self.messages = []
        self.closed = False
        self.send_json = AsyncMock()
    
    async def send_json(self, message):
        """Mock send_json method"""
        if not self.closed:
            self.messages.append(message)
    
    async def close(self):
        """Mock close method"""
        self.closed = True

class MockProgressCallback:
    """Mock progress callback for testing"""
    
    def __init__(self):
        self.calls = []
    
    async def __call__(self, step, message, progress):
        """Mock progress callback"""
        self.calls.append({
            'step': step,
            'message': message,
            'progress': progress,
            'timestamp': datetime.utcnow()
        })

class MockFileUploadComponent:
    """Mock file upload component for testing"""
    
    def __init__(self):
        self.uploaded_files = []
        self.upload_errors = []
        self.upload_progress = {}
    
    async def upload_file(self, file_content, filename):
        """Mock file upload"""
        try:
            # Simulate upload process
            await asyncio.sleep(0.1)  # Simulate network delay
            
            file_info = {
                'filename': filename,
                'size': len(file_content),
                'upload_time': datetime.utcnow(),
                'status': 'uploaded'
            }
            
            self.uploaded_files.append(file_info)
            return file_info
            
        except Exception as e:
            error_info = {
                'filename': filename,
                'error': str(e),
                'timestamp': datetime.utcnow()
            }
            self.upload_errors.append(error_info)
            raise

class MockDuplicateDetectionModal:
    """Mock duplicate detection modal for testing"""
    
    def __init__(self):
        self.duplicate_results = []
        self.user_actions = []
    
    async def show_duplicate_results(self, duplicate_result):
        """Mock showing duplicate results"""
        self.duplicate_results.append(duplicate_result)
        
        # Simulate user action
        if duplicate_result.get('is_duplicate', False):
            action = 'replace'  # Default action
        else:
            action = 'continue'
        
        self.user_actions.append({
            'action': action,
            'timestamp': datetime.utcnow()
        })
        
        return action

class MockVendorStandardizationDisplay:
    """Mock vendor standardization display for testing"""
    
    def __init__(self):
        self.standardization_results = []
        self.user_corrections = []
    
    async def display_standardization_results(self, results):
        """Mock displaying standardization results"""
        self.standardization_results.append(results)
        
        # Simulate user corrections
        corrections = []
        for result in results:
            if result.get('confidence', 1.0) < 0.8:
                correction = {
                    'original': result['vendor_raw'],
                    'corrected': result['vendor_standard'],
                    'user_input': result['vendor_standard'] + '_corrected'
                }
                corrections.append(correction)
        
        self.user_corrections.append(corrections)
        return corrections

class MockPlatformIDDisplay:
    """Mock platform ID display for testing"""
    
    def __init__(self):
        self.platform_results = []
        self.extracted_ids = []
    
    async def display_platform_results(self, results):
        """Mock displaying platform results"""
        self.platform_results.append(results)
        
        # Extract platform IDs
        for result in results:
            if result.get('extracted_ids'):
                self.extracted_ids.extend(result['extracted_ids'].values())
        
        return self.extracted_ids

class TestFrontendBackendSynchronization:
    """Test frontend/backend synchronization"""
    
    @pytest.fixture
    def mock_frontend_components(self):
        """Create mock frontend components"""
        return {
            'websocket': MockWebSocketConnection(),
            'progress_callback': MockProgressCallback(),
            'file_upload': MockFileUploadComponent(),
            'duplicate_modal': MockDuplicateDetectionModal(),
            'vendor_display': MockVendorStandardizationDisplay(),
            'platform_display': MockPlatformIDDisplay()
        }
    
    @pytest.fixture
    def mock_backend_components(self):
        """Create mock backend components"""
        mock_supabase = MagicMock()
        mock_redis = AsyncMock()
        mock_openai = Mock()
        
        # Mock duplicate detection service
        duplicate_service = MagicMock()
        duplicate_service.detect_duplicates = AsyncMock()
        
        # Mock file processor
        file_processor = MagicMock()
        file_processor.process_file_enhanced = AsyncMock()
        
        # Mock vendor standardizer
        vendor_standardizer = MagicMock()
        vendor_standardizer.standardize_vendor = AsyncMock()
        
        # Mock platform extractor
        platform_extractor = MagicMock()
        platform_extractor.extract_platform_ids = MagicMock()
        
        return {
            'duplicate_service': duplicate_service,
            'file_processor': file_processor,
            'vendor_standardizer': vendor_standardizer,
            'platform_extractor': platform_extractor
        }
    
    @pytest.fixture
    def sample_file_content(self):
        """Sample file content for testing"""
        return b"vendor,amount,date\nAmazon,100.50,2024-01-15\nMicrosoft,250.75,2024-01-16"
    
    # ============================================================================
    # WEBSOCKET REAL-TIME UPDATES TESTS
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_websocket_progress_updates(self, mock_frontend_components, mock_backend_components, sample_file_content):
        """Test WebSocket real-time progress updates"""
        frontend = mock_frontend_components
        backend = mock_backend_components
        
        # Mock backend processing with progress updates
        async def mock_process_with_progress(file_content, filename, progress_callback):
            """Mock file processing with progress updates"""
            await progress_callback("security", "Validating file security...", 10)
            await asyncio.sleep(0.1)
            
            await progress_callback("detecting", "Detecting file format...", 25)
            await asyncio.sleep(0.1)
            
            await progress_callback("processing", "Processing file content...", 50)
            await asyncio.sleep(0.1)
            
            await progress_callback("duplicate_check", "Checking for duplicates...", 75)
            await asyncio.sleep(0.1)
            
            await progress_callback("complete", "Processing complete!", 100)
            
            return {'Sheet1': 'mock_data'}
        
        backend['file_processor'].process_file_enhanced.side_effect = mock_process_with_progress
        
        # Process file
        await backend['file_processor'].process_file_enhanced(
            sample_file_content,
            "test_file.csv",
            frontend['progress_callback']
        )
        
        # Verify progress callbacks were called
        assert len(frontend['progress_callback'].calls) == 5
        assert frontend['progress_callback'].calls[0]['step'] == "security"
        assert frontend['progress_callback'].calls[0]['progress'] == 10
        assert frontend['progress_callback'].calls[-1]['step'] == "complete"
        assert frontend['progress_callback'].calls[-1]['progress'] == 100
    
    @pytest.mark.asyncio
    async def test_websocket_error_updates(self, mock_frontend_components, mock_backend_components):
        """Test WebSocket error updates"""
        frontend = mock_frontend_components
        backend = mock_backend_components
        
        # Mock backend error
        backend['file_processor'].process_file_enhanced.side_effect = Exception("Processing failed")
        
        try:
            await backend['file_processor'].process_file_enhanced(
                b"invalid content",
                "test_file.txt",
                frontend['progress_callback']
            )
        except Exception as e:
            # Verify error was caught and could be sent via WebSocket
            assert str(e) == "Processing failed"
    
    # ============================================================================
    # FILE UPLOAD SYNCHRONIZATION TESTS
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_file_upload_synchronization(self, mock_frontend_components, sample_file_content):
        """Test file upload synchronization between frontend and backend"""
        frontend = mock_frontend_components
        
        # Upload file
        file_info = await frontend['file_upload'].upload_file(sample_file_content, "test_file.csv")
        
        # Verify upload was successful
        assert file_info['filename'] == "test_file.csv"
        assert file_info['size'] == len(sample_file_content)
        assert file_info['status'] == 'uploaded'
        
        # Verify file was added to uploaded files list
        assert len(frontend['file_upload'].uploaded_files) == 1
        assert frontend['file_upload'].uploaded_files[0]['filename'] == "test_file.csv"
    
    @pytest.mark.asyncio
    async def test_file_upload_error_handling(self, mock_frontend_components):
        """Test file upload error handling"""
        frontend = mock_frontend_components
        
        # Mock upload error
        frontend['file_upload'].upload_file.side_effect = Exception("Upload failed")
        
        try:
            await frontend['file_upload'].upload_file(b"content", "test_file.csv")
        except Exception as e:
            # Verify error was handled
            assert str(e) == "Upload failed"
            assert len(frontend['file_upload'].upload_errors) == 1
            assert frontend['file_upload'].upload_errors[0]['filename'] == "test_file.csv"
    
    # ============================================================================
    # DUPLICATE DETECTION UI SYNCHRONIZATION TESTS
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_duplicate_detection_ui_sync(self, mock_frontend_components, mock_backend_components):
        """Test duplicate detection UI synchronization"""
        frontend = mock_frontend_components
        backend = mock_backend_components
        
        # Mock duplicate detection result
        duplicate_result = {
            'is_duplicate': True,
            'duplicate_type': 'exact',
            'similarity_score': 1.0,
            'duplicate_files': [
                {'filename': 'existing_file.xlsx', 'created_at': '2024-01-01T00:00:00Z'}
            ],
            'recommendation': 'replace'
        }
        
        backend['duplicate_service'].detect_duplicates.return_value = duplicate_result
        
        # Detect duplicates
        result = await backend['duplicate_service'].detect_duplicates(b"content", "metadata")
        
        # Show results in UI
        user_action = await frontend['duplicate_modal'].show_duplicate_results(result)
        
        # Verify UI synchronization
        assert len(frontend['duplicate_modal'].duplicate_results) == 1
        assert frontend['duplicate_modal'].duplicate_results[0]['is_duplicate'] == True
        assert user_action == 'replace'
        assert len(frontend['duplicate_modal'].user_actions) == 1
    
    @pytest.mark.asyncio
    async def test_duplicate_detection_no_duplicates_ui(self, mock_frontend_components, mock_backend_components):
        """Test duplicate detection UI when no duplicates found"""
        frontend = mock_frontend_components
        backend = mock_backend_components
        
        # Mock no duplicates result
        duplicate_result = {
            'is_duplicate': False,
            'duplicate_type': 'none',
            'similarity_score': 0.0,
            'duplicate_files': [],
            'recommendation': 'continue'
        }
        
        backend['duplicate_service'].detect_duplicates.return_value = duplicate_result
        
        # Detect duplicates
        result = await backend['duplicate_service'].detect_duplicates(b"content", "metadata")
        
        # Show results in UI
        user_action = await frontend['duplicate_modal'].show_duplicate_results(result)
        
        # Verify UI synchronization
        assert len(frontend['duplicate_modal'].duplicate_results) == 1
        assert frontend['duplicate_modal'].duplicate_results[0]['is_duplicate'] == False
        assert user_action == 'continue'
    
    # ============================================================================
    # VENDOR STANDARDIZATION UI SYNCHRONIZATION TESTS
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_vendor_standardization_ui_sync(self, mock_frontend_components, mock_backend_components):
        """Test vendor standardization UI synchronization"""
        frontend = mock_frontend_components
        backend = mock_backend_components
        
        # Mock vendor standardization results
        vendor_results = [
            {
                'vendor_raw': 'Amazon.com Inc',
                'vendor_standard': 'Amazon',
                'confidence': 0.95,
                'cleaning_method': 'ai_powered'
            },
            {
                'vendor_raw': 'Microsoft Corporation',
                'vendor_standard': 'Microsoft',
                'confidence': 0.90,
                'cleaning_method': 'rule_based'
            }
        ]
        
        backend['vendor_standardizer'].standardize_vendor.return_value = vendor_results[0]
        
        # Standardize vendors
        standardized_vendors = []
        for vendor_data in vendor_results:
            result = await backend['vendor_standardizer'].standardize_vendor(vendor_data['vendor_raw'])
            standardized_vendors.append(result)
        
        # Display results in UI
        user_corrections = await frontend['vendor_display'].display_standardization_results(standardized_vendors)
        
        # Verify UI synchronization
        assert len(frontend['vendor_display'].standardization_results) == 1
        assert len(frontend['vendor_display'].standardization_results[0]) == 2
        assert frontend['vendor_display'].standardization_results[0][0]['vendor_standard'] == 'Amazon'
        assert frontend['vendor_display'].standardization_results[0][1]['vendor_standard'] == 'Microsoft'
    
    @pytest.mark.asyncio
    async def test_vendor_standardization_low_confidence_ui(self, mock_frontend_components, mock_backend_components):
        """Test vendor standardization UI with low confidence results"""
        frontend = mock_frontend_components
        backend = mock_backend_components
        
        # Mock low confidence results
        low_confidence_results = [
            {
                'vendor_raw': 'Unclear Vendor Name',
                'vendor_standard': 'Unclear Vendor',
                'confidence': 0.6,
                'cleaning_method': 'ai_powered'
            }
        ]
        
        # Display results in UI
        user_corrections = await frontend['vendor_display'].display_standardization_results(low_confidence_results)
        
        # Verify UI shows corrections for low confidence
        assert len(frontend['vendor_display'].user_corrections) == 1
        assert len(frontend['vendor_display'].user_corrections[0]) == 1
        assert 'corrected' in frontend['vendor_display'].user_corrections[0][0]['user_input']
    
    # ============================================================================
    # PLATFORM ID EXTRACTION UI SYNCHRONIZATION TESTS
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_platform_id_extraction_ui_sync(self, mock_frontend_components, mock_backend_components):
        """Test platform ID extraction UI synchronization"""
        frontend = mock_frontend_components
        backend = mock_backend_components
        
        # Mock platform ID extraction results
        platform_results = [
            {
                'platform': 'razorpay',
                'extracted_ids': {
                    'payment_id': 'pay_12345678901234',
                    'order_id': 'order_98765432109876'
                },
                'total_ids_found': 2
            },
            {
                'platform': 'stripe',
                'extracted_ids': {
                    'charge_id': 'ch_123456789012345678901234',
                    'customer_id': 'cus_12345678901234'
                },
                'total_ids_found': 2
            }
        ]
        
        backend['platform_extractor'].extract_platform_ids.return_value = platform_results[0]
        
        # Extract platform IDs
        extracted_results = []
        for platform_data in platform_results:
            result = backend['platform_extractor'].extract_platform_ids(
                {'test': 'data'}, 'test_platform', ['test']
            )
            extracted_results.append(result)
        
        # Display results in UI
        extracted_ids = await frontend['platform_display'].display_platform_results(extracted_results)
        
        # Verify UI synchronization
        assert len(frontend['platform_display'].platform_results) == 1
        assert len(frontend['platform_display'].platform_results[0]) == 1
        assert frontend['platform_display'].platform_results[0][0]['platform'] == 'razorpay'
    
    # ============================================================================
    # END-TO-END UI WORKFLOW TESTS
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_complete_ui_workflow(self, mock_frontend_components, mock_backend_components, sample_file_content):
        """Test complete UI workflow from upload to completion"""
        frontend = mock_frontend_components
        backend = mock_backend_components
        
        # Step 1: Upload file
        file_info = await frontend['file_upload'].upload_file(sample_file_content, "test_file.csv")
        assert file_info['status'] == 'uploaded'
        
        # Step 2: Process file with progress updates
        async def mock_process_with_progress(file_content, filename, progress_callback):
            await progress_callback("processing", "Processing file...", 50)
            await progress_callback("complete", "Processing complete!", 100)
            return {'Sheet1': 'mock_data'}
        
        backend['file_processor'].process_file_enhanced.side_effect = mock_process_with_progress
        
        await backend['file_processor'].process_file_enhanced(
            sample_file_content,
            "test_file.csv",
            frontend['progress_callback']
        )
        
        # Step 3: Check for duplicates
        duplicate_result = {
            'is_duplicate': False,
            'duplicate_type': 'none',
            'similarity_score': 0.0,
            'duplicate_files': [],
            'recommendation': 'continue'
        }
        
        backend['duplicate_service'].detect_duplicates.return_value = duplicate_result
        
        result = await backend['duplicate_service'].detect_duplicates(sample_file_content, "metadata")
        user_action = await frontend['duplicate_modal'].show_duplicate_results(result)
        assert user_action == 'continue'
        
        # Step 4: Standardize vendors
        vendor_results = [
            {
                'vendor_raw': 'Amazon.com Inc',
                'vendor_standard': 'Amazon',
                'confidence': 0.95,
                'cleaning_method': 'ai_powered'
            }
        ]
        
        backend['vendor_standardizer'].standardize_vendor.return_value = vendor_results[0]
        
        standardized_vendors = []
        for vendor_data in vendor_results:
            result = await backend['vendor_standardizer'].standardize_vendor(vendor_data['vendor_raw'])
            standardized_vendors.append(result)
        
        user_corrections = await frontend['vendor_display'].display_standardization_results(standardized_vendors)
        
        # Step 5: Extract platform IDs
        platform_results = [
            {
                'platform': 'razorpay',
                'extracted_ids': {'payment_id': 'pay_12345678901234'},
                'total_ids_found': 1
            }
        ]
        
        backend['platform_extractor'].extract_platform_ids.return_value = platform_results[0]
        
        extracted_results = []
        for platform_data in platform_results:
            result = backend['platform_extractor'].extract_platform_ids(
                {'test': 'data'}, 'test_platform', ['test']
            )
            extracted_results.append(result)
        
        extracted_ids = await frontend['platform_display'].display_platform_results(extracted_results)
        
        # Verify complete workflow
        assert len(frontend['file_upload'].uploaded_files) == 1
        assert len(frontend['progress_callback'].calls) == 2
        assert len(frontend['duplicate_modal'].duplicate_results) == 1
        assert len(frontend['vendor_display'].standardization_results) == 1
        assert len(frontend['platform_display'].platform_results) == 1
    
    # ============================================================================
    # PERFORMANCE AND RESPONSIVENESS TESTS
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_ui_responsiveness_under_load(self, mock_frontend_components, mock_backend_components):
        """Test UI responsiveness under load"""
        frontend = mock_frontend_components
        backend = mock_backend_components
        
        # Simulate multiple concurrent operations
        async def simulate_operation(operation_id):
            """Simulate a single operation"""
            start_time = time.time()
            
            # Upload file
            file_info = await frontend['file_upload'].upload_file(
                f"content_{operation_id}".encode(),
                f"file_{operation_id}.csv"
            )
            
            # Process file
            backend['file_processor'].process_file_enhanced.return_value = {'Sheet1': 'data'}
            await backend['file_processor'].process_file_enhanced(
                f"content_{operation_id}".encode(),
                f"file_{operation_id}.csv",
                frontend['progress_callback']
            )
            
            # Check duplicates
            backend['duplicate_service'].detect_duplicates.return_value = {
                'is_duplicate': False,
                'duplicate_type': 'none',
                'similarity_score': 0.0,
                'duplicate_files': [],
                'recommendation': 'continue'
            }
            
            result = await backend['duplicate_service'].detect_duplicates(
                f"content_{operation_id}".encode(),
                "metadata"
            )
            
            await frontend['duplicate_modal'].show_duplicate_results(result)
            
            end_time = time.time()
            return end_time - start_time
        
        # Run multiple operations concurrently
        operations = [simulate_operation(i) for i in range(10)]
        results = await asyncio.gather(*operations)
        
        # Verify all operations completed
        assert len(results) == 10
        
        # Verify performance (each operation should complete within reasonable time)
        max_time = max(results)
        assert max_time < 5.0  # Should complete within 5 seconds
        
        # Verify UI state consistency
        assert len(frontend['file_upload'].uploaded_files) == 10
        assert len(frontend['duplicate_modal'].duplicate_results) == 10
    
    @pytest.mark.asyncio
    async def test_ui_error_recovery(self, mock_frontend_components, mock_backend_components):
        """Test UI error recovery"""
        frontend = mock_frontend_components
        backend = mock_backend_components
        
        # Simulate backend error
        backend['file_processor'].process_file_enhanced.side_effect = Exception("Processing failed")
        
        try:
            await backend['file_processor'].process_file_enhanced(
                b"content",
                "test_file.csv",
                frontend['progress_callback']
            )
        except Exception as e:
            # Verify error was caught and could be displayed in UI
            assert str(e) == "Processing failed"
            
            # Simulate error display in UI
            error_message = f"Error: {str(e)}"
            assert "Processing failed" in error_message
    
    # ============================================================================
    # DATA CONSISTENCY TESTS
    # ============================================================================
    
    @pytest.mark.asyncio
    async def test_data_consistency_between_ui_and_backend(self, mock_frontend_components, mock_backend_components):
        """Test data consistency between UI and backend"""
        frontend = mock_frontend_components
        backend = mock_backend_components
        
        # Test data
        test_data = {
            'vendor': 'Amazon.com Inc',
            'amount': 100.50,
            'platform_id': 'AMZ-12345'
        }
        
        # Backend processing
        backend['vendor_standardizer'].standardize_vendor.return_value = {
            'vendor_raw': test_data['vendor'],
            'vendor_standard': 'Amazon',
            'confidence': 0.95,
            'cleaning_method': 'ai_powered'
        }
        
        backend['platform_extractor'].extract_platform_ids.return_value = {
            'platform': 'razorpay',
            'extracted_ids': {'payment_id': test_data['platform_id']},
            'total_ids_found': 1
        }
        
        # Process in backend
        vendor_result = await backend['vendor_standardizer'].standardize_vendor(test_data['vendor'])
        platform_result = backend['platform_extractor'].extract_platform_ids(
            test_data, 'razorpay', ['vendor', 'amount', 'platform_id']
        )
        
        # Display in UI
        await frontend['vendor_display'].display_standardization_results([vendor_result])
        await frontend['platform_display'].display_platform_results([platform_result])
        
        # Verify data consistency
        assert frontend['vendor_display'].standardization_results[0][0]['vendor_raw'] == test_data['vendor']
        assert frontend['vendor_display'].standardization_results[0][0]['vendor_standard'] == 'Amazon'
        assert frontend['platform_display'].platform_results[0][0]['platform'] == 'razorpay'
        assert frontend['platform_display'].extracted_ids[0] == test_data['platform_id']


class TestWebSocketRealTimeUpdates:
    """Test WebSocket real-time updates"""
    
    @pytest.fixture
    def websocket_manager(self):
        """Create WebSocket manager for testing"""
        from duplicate_detection_api_integration import WebSocketManager
        return WebSocketManager()
    
    @pytest.mark.asyncio
    async def test_websocket_connection_management(self, websocket_manager):
        """Test WebSocket connection management"""
        mock_websocket1 = AsyncMock()
        mock_websocket2 = AsyncMock()
        
        # Connect multiple WebSockets for same job
        await websocket_manager.connect(mock_websocket1, "job_123")
        await websocket_manager.connect(mock_websocket2, "job_123")
        
        # Send update
        await websocket_manager.send_update("job_123", {"message": "test"})
        
        # Verify both WebSockets received the update
        assert mock_websocket1.send_json.call_count == 1
        assert mock_websocket2.send_json.call_count == 1
        
        # Disconnect
        websocket_manager.disconnect("job_123")
        assert "job_123" not in websocket_manager.active_connections
    
    @pytest.mark.asyncio
    async def test_websocket_error_handling(self, websocket_manager):
        """Test WebSocket error handling"""
        mock_websocket = AsyncMock()
        mock_websocket.send_json.side_effect = Exception("Connection closed")
        
        # Connect and send update
        await websocket_manager.connect(mock_websocket, "job_456")
        await websocket_manager.send_update("job_456", {"message": "test"})
        
        # Verify connection was removed after error
        assert "job_456" not in websocket_manager.active_connections
    
    @pytest.mark.asyncio
    async def test_websocket_concurrent_updates(self, websocket_manager):
        """Test concurrent WebSocket updates"""
        mock_websocket = AsyncMock()
        await websocket_manager.connect(mock_websocket, "job_789")
        
        # Send multiple updates concurrently
        updates = [
            {"step": "processing", "progress": 25},
            {"step": "duplicate_check", "progress": 50},
            {"step": "standardization", "progress": 75},
            {"step": "complete", "progress": 100}
        ]
        
        # Send updates concurrently
        tasks = [websocket_manager.send_update("job_789", update) for update in updates]
        await asyncio.gather(*tasks)
        
        # Verify all updates were sent
        assert mock_websocket.send_json.call_count == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

