"""
Test Utilities and Infrastructure for Enterprise-Grade Testing
============================================================

This module provides comprehensive testing utilities, mocks, and infrastructure
for enterprise-grade testing of mission-critical financial systems.

Features:
- Test data generators for various file types and scenarios
- Mock clients for external services (Supabase, OpenAI, etc.)
- Performance monitoring and metrics collection
- Accuracy validation for ML models
- Security testing utilities
- Memory and resource monitoring
- Concurrent testing support

Author: Principal Engineer - Quality, Testing & Resilience
Version: 1.0.0 - Enterprise Grade
"""

import asyncio
import json
import tempfile
import os
import time
import uuid
import hashlib
import random
import string
import io
import zipfile
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from unittest.mock import Mock, AsyncMock, MagicMock
import pandas as pd
import numpy as np
from pathlib import Path
import psutil
import gc
import sqlite3
from concurrent.futures import ThreadPoolExecutor
try:
    import aiofiles
except ImportError:
    aiofiles = None

try:
    import httpx
except ImportError:
    httpx = None


class TestDataGenerator:
    """Comprehensive test data generator for all scenarios"""
    
    def __init__(self):
        self.sample_data = {
            'financial_data': self._generate_financial_data(),
            'vendor_data': self._generate_vendor_data(),
            'platform_data': self._generate_platform_data(),
            'document_data': self._generate_document_data(),
            'entity_data': self._generate_entity_data()
        }
    
    def _generate_financial_data(self) -> List[Dict[str, Any]]:
        """Generate realistic financial data for testing"""
        data = []
        vendors = ['Apple Inc', 'Microsoft Corp', 'Google LLC', 'Amazon Inc', 'Tesla Inc']
        categories = ['Office Supplies', 'Software', 'Hardware', 'Services', 'Travel']
        
        for i in range(1000):
            data.append({
                'transaction_id': f'TXN_{i:06d}',
                'date': (datetime.now() - timedelta(days=random.randint(0, 365))).isoformat(),
                'vendor': random.choice(vendors),
                'amount': round(random.uniform(10.0, 10000.0), 2),
                'category': random.choice(categories),
                'description': f'Payment to {random.choice(vendors)} for {random.choice(categories)}',
                'currency': 'USD',
                'status': random.choice(['paid', 'pending', 'cancelled']),
                'invoice_number': f'INV_{i:06d}',
                'payment_method': random.choice(['credit_card', 'bank_transfer', 'check', 'cash'])
            })
        return data
    
    def _generate_vendor_data(self) -> List[Dict[str, Any]]:
        """Generate vendor data for testing"""
        vendors = []
        base_names = [
            'Microsoft Corporation', 'Apple Inc', 'Google LLC', 'Amazon.com Inc',
            'Tesla Inc', 'Meta Platforms Inc', 'Netflix Inc', 'Adobe Inc',
            'Salesforce Inc', 'Oracle Corporation', 'IBM Corporation', 'Intel Corporation'
        ]
        
        for name in base_names:
            vendors.append({
                'name': name,
                'aliases': [name.upper(), name.lower(), name.replace(' Inc', '').replace(' LLC', '').replace(' Corporation', '')],
                'domain': name.lower().replace(' ', '').replace('inc', '').replace('llc', '').replace('corporation', '') + '.com',
                'category': random.choice(['Technology', 'Software', 'Hardware', 'Services', 'Retail']),
                'country': random.choice(['USA', 'Canada', 'UK', 'Germany', 'Japan']),
                'tax_id': ''.join(random.choices(string.digits, k=9)),
                'address': f'{random.randint(100, 9999)} {random.choice(["Main St", "Oak Ave", "Pine Rd"])}, {random.choice(["New York", "Los Angeles", "Chicago", "Houston"])}, {random.choice(["NY", "CA", "IL", "TX"])} {random.randint(10000, 99999)}'
            })
        return vendors
    
    def _generate_platform_data(self) -> List[Dict[str, Any]]:
        """Generate platform-specific data for testing"""
        platforms = [
            {
                'name': 'QuickBooks',
                'file_patterns': ['.qbb', '.qbx', '.qbw'],
                'headers': ['Date', 'Account', 'Description', 'Amount', 'Balance'],
                'metadata': {'version': '2024', 'company': 'Test Company'}
            },
            {
                'name': 'Xero',
                'file_patterns': ['.xlsx', '.csv'],
                'headers': ['Date', 'Contact', 'Description', 'Amount', 'Account'],
                'metadata': {'tenant_id': 'test-tenant', 'org_id': 'test-org'}
            },
            {
                'name': 'Sage',
                'file_patterns': ['.sage', '.csv'],
                'headers': ['Date', 'Reference', 'Description', 'Debit', 'Credit'],
                'metadata': {'version': '50', 'company_code': '001'}
            }
        ]
        return platforms
    
    def _generate_document_data(self) -> List[Dict[str, Any]]:
        """Generate document data for testing"""
        documents = []
        types = ['invoice', 'receipt', 'contract', 'statement', 'report', 'payroll']
        
        for doc_type in types:
            documents.append({
                'type': doc_type,
                'content': f'Sample {doc_type} content with financial data and vendor information.',
                'metadata': {
                    'pages': random.randint(1, 50),
                    'format': random.choice(['pdf', 'docx', 'xlsx', 'csv']),
                    'size_bytes': random.randint(1000, 10000000),
                    'created_date': datetime.now().isoformat(),
                    'language': 'en'
                },
                'fields': {
                    'amount': random.uniform(10.0, 10000.0),
                    'date': datetime.now().isoformat(),
                    'vendor': random.choice(['Apple Inc', 'Microsoft Corp', 'Google LLC']),
                    'description': f'{doc_type.title()} document'
                }
            })
        return documents
    
    def _generate_entity_data(self) -> List[Dict[str, Any]]:
        """Generate entity data for testing"""
        entities = []
        entity_types = ['vendor', 'customer', 'employee', 'product', 'account']
        
        for entity_type in entity_types:
            for i in range(100):
                entities.append({
                    'id': f'{entity_type.upper()}_{i:06d}',
                    'type': entity_type,
                    'name': f'{entity_type.title()} {i}',
                    'aliases': [f'{entity_type} {i}', f'{entity_type.upper()} {i}', f'{entity_type.lower()}{i}'],
                    'attributes': {
                        'email': f'{entity_type}{i}@example.com',
                        'phone': f'+1-555-{random.randint(100, 999)}-{random.randint(1000, 9999)}',
                        'address': f'{random.randint(100, 9999)} {random.choice(["Main St", "Oak Ave"])}',
                        'status': random.choice(['active', 'inactive', 'pending'])
                    },
                    'created_date': datetime.now().isoformat(),
                    'updated_date': datetime.now().isoformat()
                })
        return entities
    
    def generate_excel_file(self, rows: int = 1000, sheets: int = 1) -> bytes:
        """Generate Excel file with test data"""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            with pd.ExcelWriter(tmp.name, engine='openpyxl') as writer:
                for sheet_num in range(sheets):
                    df = pd.DataFrame(self.sample_data['financial_data'][:rows])
                    sheet_name = f'Sheet{sheet_num + 1}' if sheets > 1 else 'Sheet1'
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            with open(tmp.name, 'rb') as f:
                content = f.read()
            
            os.unlink(tmp.name)
            return content
    
    def generate_csv_file(self, rows: int = 1000) -> bytes:
        """Generate CSV file with test data"""
        df = pd.DataFrame(self.sample_data['financial_data'][:rows])
        return df.to_csv(index=False).encode('utf-8')
    
    def generate_pdf_content(self) -> bytes:
        """Generate PDF-like content for testing"""
        # This would normally generate actual PDF content
        # For testing, we'll generate text content that mimics PDF structure
        content = f"""
        INVOICE
        
        Invoice Number: INV_{random.randint(100000, 999999)}
        Date: {datetime.now().strftime('%Y-%m-%d')}
        Due Date: {(datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')}
        
        Vendor: {random.choice(self.sample_data['vendor_data'])['name']}
        Amount: ${random.uniform(100.0, 10000.0):.2f}
        
        Description: Services rendered for the month of {datetime.now().strftime('%B %Y')}
        
        Payment Terms: Net 30
        """
        return content.encode('utf-8')
    
    def generate_large_file(self, size_mb: int = 100) -> bytes:
        """Generate large file for performance testing"""
        size_bytes = size_mb * 1024 * 1024
        chunk_size = 1024
        content = b''
        
        while len(content) < size_bytes:
            chunk = os.urandom(min(chunk_size, size_bytes - len(content)))
            content += chunk
        
        return content
    
    def generate_corrupted_file(self, file_type: str = 'xlsx') -> bytes:
        """Generate corrupted file for error testing"""
        if file_type == 'xlsx':
            # Generate valid Excel content and corrupt it
            valid_content = self.generate_excel_file(100)
            # Corrupt by truncating or modifying bytes
            return valid_content[:-100] + b'CORRUPTED_DATA'
        elif file_type == 'csv':
            valid_content = self.generate_csv_file(100)
            return valid_content[:-50] + b'CORRUPTED_DATA'
        else:
            return b'INVALID_FILE_CONTENT'


class MockSupabaseClient:
    """Mock Supabase client for testing"""
    
    def __init__(self):
        self.tables = {}
        self.queries = []
        self.inserts = []
        self.updates = []
        self.deletes = []
    
    def table(self, table_name: str):
        """Mock table method"""
        if table_name not in self.tables:
            self.tables[table_name] = MockTable(table_name)
        return self.tables[table_name]
    
    def auth(self):
        """Mock auth method"""
        return MockAuth()
    
    def storage(self):
        """Mock storage method"""
        return MockStorage()


class MockTable:
    """Mock table for Supabase testing"""
    
    def __init__(self, name: str):
        self.name = name
        self.data = []
        self.queries = []
    
    def select(self, *args):
        """Mock select method"""
        query = MockQuery(self, 'select', args)
        self.queries.append(query)
        return query
    
    def insert(self, data):
        """Mock insert method"""
        if isinstance(data, list):
            self.data.extend(data)
        else:
            self.data.append(data)
        return MockResponse([data] if not isinstance(data, list) else data)
    
    def update(self, data):
        """Mock update method"""
        return MockQuery(self, 'update', data)
    
    def delete(self):
        """Mock delete method"""
        return MockQuery(self, 'delete')


class MockQuery:
    """Mock query for Supabase testing"""
    
    def __init__(self, table, operation, args=None):
        self.table = table
        self.operation = operation
        self.args = args
        self.filters = []
    
    def eq(self, column, value):
        """Mock eq filter"""
        self.filters.append(('eq', column, value))
        return self
    
    def neq(self, column, value):
        """Mock neq filter"""
        self.filters.append(('neq', column, value))
        return self
    
    def gt(self, column, value):
        """Mock gt filter"""
        self.filters.append(('gt', column, value))
        return self
    
    def gte(self, column, value):
        """Mock gte filter"""
        self.filters.append(('gte', column, value))
        return self
    
    def lt(self, column, value):
        """Mock lt filter"""
        self.filters.append(('lt', column, value))
        return self
    
    def lte(self, column, value):
        """Mock lte filter"""
        self.filters.append(('lte', column, value))
        return self
    
    def like(self, column, value):
        """Mock like filter"""
        self.filters.append(('like', column, value))
        return self
    
    def ilike(self, column, value):
        """Mock ilike filter"""
        self.filters.append(('ilike', column, value))
        return self
    
    def in_(self, column, values):
        """Mock in filter"""
        self.filters.append(('in', column, values))
        return self
    
    def order(self, column, desc=False):
        """Mock order method"""
        self.order_by = (column, desc)
        return self
    
    def limit(self, count):
        """Mock limit method"""
        self.limit_count = count
        return self
    
    def execute(self):
        """Mock execute method"""
        # Apply filters to table data
        filtered_data = self.table.data.copy()
        
        for filter_type, column, value in self.filters:
            if filter_type == 'eq':
                filtered_data = [row for row in filtered_data if row.get(column) == value]
            elif filter_type == 'neq':
                filtered_data = [row for row in filtered_data if row.get(column) != value]
            elif filter_type == 'gt':
                filtered_data = [row for row in filtered_data if row.get(column) > value]
            elif filter_type == 'gte':
                filtered_data = [row for row in filtered_data if row.get(column) >= value]
            elif filter_type == 'lt':
                filtered_data = [row for row in filtered_data if row.get(column) < value]
            elif filter_type == 'lte':
                filtered_data = [row for row in filtered_data if row.get(column) <= value]
            elif filter_type == 'like':
                filtered_data = [row for row in filtered_data if str(value) in str(row.get(column, ''))]
            elif filter_type == 'ilike':
                filtered_data = [row for row in filtered_data if str(value).lower() in str(row.get(column, '')).lower()]
            elif filter_type == 'in':
                filtered_data = [row for row in filtered_data if row.get(column) in value]
        
        # Apply ordering
        if hasattr(self, 'order_by'):
            column, desc = self.order_by
            filtered_data.sort(key=lambda x: x.get(column, ''), reverse=desc)
        
        # Apply limit
        if hasattr(self, 'limit_count'):
            filtered_data = filtered_data[:self.limit_count]
        
        return MockResponse(filtered_data)


class MockResponse:
    """Mock response for Supabase testing"""
    
    def __init__(self, data):
        self.data = data
        self.count = len(data)
        self.error = None


class MockAuth:
    """Mock auth for Supabase testing"""
    
    def get_user(self):
        """Mock get_user method"""
        return MockUser()
    
    def sign_in_with_password(self, email, password):
        """Mock sign_in_with_password method"""
        return MockAuthResponse()


class MockUser:
    """Mock user for Supabase testing"""
    
    def __init__(self):
        self.id = str(uuid.uuid4())
        self.email = 'test@example.com'
        self.user_metadata = {}


class MockAuthResponse:
    """Mock auth response for Supabase testing"""
    
    def __init__(self):
        self.user = MockUser()
        self.session = MockSession()


class MockSession:
    """Mock session for Supabase testing"""
    
    def __init__(self):
        self.access_token = 'mock_access_token'
        self.refresh_token = 'mock_refresh_token'


class MockStorage:
    """Mock storage for Supabase testing"""
    
    def from_(self, bucket):
        """Mock from_ method"""
        return MockBucket(bucket)


class MockBucket:
    """Mock bucket for Supabase testing"""
    
    def __init__(self, name):
        self.name = name
        self.files = {}
    
    def upload(self, path, file_content):
        """Mock upload method"""
        self.files[path] = file_content
        return MockUploadResponse()
    
    def download(self, path):
        """Mock download method"""
        if path in self.files:
            return MockDownloadResponse(self.files[path])
        return MockDownloadResponse(None)
    
    def remove(self, paths):
        """Mock remove method"""
        if isinstance(paths, list):
            for path in paths:
                self.files.pop(path, None)
        else:
            self.files.pop(paths, None)
        return MockRemoveResponse()


class MockUploadResponse:
    """Mock upload response for Supabase testing"""
    
    def __init__(self):
        self.error = None


class MockDownloadResponse:
    """Mock download response for Supabase testing"""
    
    def __init__(self, content):
        self.content = content
        self.error = None


class MockRemoveResponse:
    """Mock remove response for Supabase testing"""
    
    def __init__(self):
        self.error = None


class MockOpenAIClient:
    """Mock OpenAI client for testing"""
    
    def __init__(self):
        self.chat = MockChat()
        self.embeddings = MockEmbeddings()
        self.completions = MockCompletions()
    
    class Chat:
        def completions(self):
            return MockChatCompletions()
    
    class Embeddings:
        def create(self, input_text, model="text-embedding-ada-002"):
            # Generate mock embeddings
            embedding_dim = 1536  # OpenAI ada-002 dimension
            embeddings = []
            for text in input_text if isinstance(input_text, list) else [input_text]:
                # Generate deterministic but varied embeddings
                np.random.seed(hash(text) % 2**32)
                embedding = np.random.randn(embedding_dim).tolist()
                embeddings.append(embedding)
            
            return MockEmbeddingResponse(embeddings)
    
    class Completions:
        def create(self, **kwargs):
            return MockCompletionResponse()


class MockChatCompletions:
    """Mock chat completions for OpenAI testing"""
    
    def create(self, **kwargs):
        messages = kwargs.get('messages', [])
        model = kwargs.get('model', 'gpt-3.5-turbo')
        
        # Generate mock response based on the last message
        last_message = messages[-1] if messages else {'content': 'Hello'}
        content = last_message.get('content', 'Hello')
        
        # Generate contextual response based on content
        if 'classify' in content.lower():
            response = 'invoice'
        elif 'detect' in content.lower():
            response = 'financial_document'
        elif 'extract' in content.lower():
            response = '{"amount": 1250.00, "date": "2024-01-15", "vendor": "Test Vendor"}'
        else:
            response = 'This is a mock response from the AI model.'
        
        return MockCompletionResponse(response)


class MockEmbeddingResponse:
    """Mock embedding response for OpenAI testing"""
    
    def __init__(self, embeddings):
        self.data = [MockEmbedding(emb) for emb in embeddings]


class MockEmbedding:
    """Mock embedding for OpenAI testing"""
    
    def __init__(self, embedding):
        self.embedding = embedding


class MockCompletionResponse:
    """Mock completion response for OpenAI testing"""
    
    def __init__(self, content="Mock AI response"):
        self.choices = [MockChoice(content)]


class MockChoice:
    """Mock choice for OpenAI testing"""
    
    def __init__(self, content):
        self.message = MockMessage(content)


class MockMessage:
    """Mock message for OpenAI testing"""
    
    def __init__(self, content):
        self.content = content


class PerformanceMonitor:
    """Performance monitoring and metrics collection"""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
        self.memory_snapshots = []
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration"""
        if operation not in self.start_times:
            return 0.0
        
        duration = time.time() - self.start_times[operation]
        if operation not in self.metrics:
            self.metrics[operation] = []
        self.metrics[operation].append(duration)
        del self.start_times[operation]
        return duration
    
    def record_memory_usage(self, operation: str):
        """Record current memory usage"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        self.memory_snapshots.append({
            'operation': operation,
            'memory_mb': memory_mb,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        summary = {}
        
        for operation, times in self.metrics.items():
            if times:
                summary[operation] = {
                    'count': len(times),
                    'avg_duration': sum(times) / len(times),
                    'min_duration': min(times),
                    'max_duration': max(times),
                    'total_duration': sum(times)
                }
        
        # Memory summary
        if self.memory_snapshots:
            memory_values = [snap['memory_mb'] for snap in self.memory_snapshots]
            summary['memory'] = {
                'avg_memory_mb': sum(memory_values) / len(memory_values),
                'min_memory_mb': min(memory_values),
                'max_memory_mb': max(memory_values),
                'peak_memory_mb': max(memory_values)
            }
        
        return summary
    
    def check_performance_thresholds(self, thresholds: Dict[str, float]) -> List[str]:
        """Check if performance meets thresholds"""
        violations = []
        summary = self.get_performance_summary()
        
        for operation, threshold in thresholds.items():
            if operation in summary:
                avg_duration = summary[operation]['avg_duration']
                if avg_duration > threshold:
                    violations.append(f"{operation}: {avg_duration:.2f}s > {threshold}s")
        
        return violations


class AccuracyValidator:
    """Accuracy validation for ML models and algorithms"""
    
    def __init__(self):
        self.predictions = []
        self.actuals = []
        self.metrics = {}
    
    def add_prediction(self, predicted: Any, actual: Any, confidence: float = 1.0):
        """Add a prediction for accuracy calculation"""
        self.predictions.append({
            'predicted': predicted,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        })
        self.actuals.append(actual)
    
    def calculate_accuracy(self) -> float:
        """Calculate overall accuracy"""
        if not self.predictions or not self.actuals:
            return 0.0
        
        correct = 0
        for pred, actual in zip(self.predictions, self.actuals):
            if pred['predicted'] == actual:
                correct += 1
        
        return correct / len(self.predictions)
    
    def calculate_precision_recall(self, positive_class: str = None) -> Dict[str, float]:
        """Calculate precision and recall"""
        if not self.predictions or not self.actuals:
            return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
        
        if positive_class is None:
            # Use most common class as positive
            from collections import Counter
            positive_class = Counter(self.actuals).most_common(1)[0][0]
        
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for pred, actual in zip(self.predictions, self.actuals):
            if pred['predicted'] == positive_class and actual == positive_class:
                true_positives += 1
            elif pred['predicted'] == positive_class and actual != positive_class:
                false_positives += 1
            elif pred['predicted'] != positive_class and actual == positive_class:
                false_negatives += 1
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'positive_class': positive_class
        }
    
    def get_confidence_distribution(self) -> Dict[str, Any]:
        """Get confidence score distribution"""
        if not self.predictions:
            return {}
        
        confidences = [pred['confidence'] for pred in self.predictions]
        
        return {
            'avg_confidence': sum(confidences) / len(confidences),
            'min_confidence': min(confidences),
            'max_confidence': max(confidences),
            'confidence_std': np.std(confidences) if len(confidences) > 1 else 0.0
        }
    
    def validate_accuracy_threshold(self, threshold: float) -> bool:
        """Validate if accuracy meets threshold"""
        accuracy = self.calculate_accuracy()
        return accuracy >= threshold


class SecurityTester:
    """Security testing utilities"""
    
    def __init__(self):
        self.vulnerabilities = []
        self.security_tests = []
    
    def test_sql_injection(self, endpoint: str, payloads: List[str]) -> List[str]:
        """Test for SQL injection vulnerabilities"""
        vulnerabilities = []
        
        for payload in payloads:
            # Mock test - in real implementation, would make actual HTTP requests
            if any(keyword in payload.lower() for keyword in ['union', 'select', 'drop', 'delete', 'insert']):
                vulnerabilities.append({
                    'type': 'sql_injection',
                    'endpoint': endpoint,
                    'payload': payload,
                    'severity': 'high'
                })
        
        return vulnerabilities
    
    def test_xss(self, endpoint: str, payloads: List[str]) -> List[str]:
        """Test for XSS vulnerabilities"""
        vulnerabilities = []
        
        for payload in payloads:
            if any(tag in payload.lower() for tag in ['<script>', '<img>', '<iframe>', 'javascript:']):
                vulnerabilities.append({
                    'type': 'xss',
                    'endpoint': endpoint,
                    'payload': payload,
                    'severity': 'medium'
                })
        
        return vulnerabilities
    
    def test_file_upload_security(self, file_types: List[str]) -> List[str]:
        """Test file upload security"""
        vulnerabilities = []
        
        dangerous_extensions = ['.exe', '.bat', '.sh', '.php', '.jsp', '.asp']
        
        for file_type in file_types:
            if any(file_type.endswith(ext) for ext in dangerous_extensions):
                vulnerabilities.append({
                    'type': 'dangerous_file_upload',
                    'file_type': file_type,
                    'severity': 'high'
                })
        
        return vulnerabilities
    
    def test_authentication_bypass(self, endpoints: List[str]) -> List[str]:
        """Test for authentication bypass vulnerabilities"""
        vulnerabilities = []
        
        for endpoint in endpoints:
            # Mock test - in real implementation, would test without auth tokens
            if endpoint.startswith('/api/') and 'public' not in endpoint:
                vulnerabilities.append({
                    'type': 'auth_bypass',
                    'endpoint': endpoint,
                    'severity': 'critical'
                })
        
        return vulnerabilities
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        return {
            'total_vulnerabilities': len(self.vulnerabilities),
            'by_severity': {
                'critical': len([v for v in self.vulnerabilities if v.get('severity') == 'critical']),
                'high': len([v for v in self.vulnerabilities if v.get('severity') == 'high']),
                'medium': len([v for v in self.vulnerabilities if v.get('severity') == 'medium']),
                'low': len([v for v in self.vulnerabilities if v.get('severity') == 'low'])
            },
            'vulnerabilities': self.vulnerabilities,
            'recommendations': self._generate_security_recommendations()
        }
    
    def _generate_security_recommendations(self) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        
        critical_count = len([v for v in self.vulnerabilities if v.get('severity') == 'critical'])
        high_count = len([v for v in self.vulnerabilities if v.get('severity') == 'high'])
        
        if critical_count > 0:
            recommendations.append("CRITICAL: Fix all critical vulnerabilities immediately")
        
        if high_count > 0:
            recommendations.append("HIGH: Address high-severity vulnerabilities within 24 hours")
        
        if any(v.get('type') == 'sql_injection' for v in self.vulnerabilities):
            recommendations.append("Implement parameterized queries and input validation")
        
        if any(v.get('type') == 'xss' for v in self.vulnerabilities):
            recommendations.append("Implement proper output encoding and CSP headers")
        
        if any(v.get('type') == 'dangerous_file_upload' for v in self.vulnerabilities):
            recommendations.append("Implement strict file type validation and scanning")
        
        return recommendations


class ConcurrentTestRunner:
    """Concurrent test execution utilities"""
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.results = []
    
    async def run_concurrent_tests(self, test_functions: List[callable], *args, **kwargs):
        """Run multiple test functions concurrently"""
        tasks = []
        
        for test_func in test_functions:
            if asyncio.iscoroutinefunction(test_func):
                task = asyncio.create_task(test_func(*args, **kwargs))
            else:
                task = asyncio.create_task(self._run_sync_test(test_func, *args, **kwargs))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            self.results.append({
                'test_function': test_functions[i].__name__,
                'result': result,
                'success': not isinstance(result, Exception),
                'timestamp': datetime.now().isoformat()
            })
        
        return self.results
    
    async def _run_sync_test(self, test_func: callable, *args, **kwargs):
        """Run synchronous test function in async context"""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, test_func, *args, **kwargs)
    
    def get_success_rate(self) -> float:
        """Get success rate of concurrent tests"""
        if not self.results:
            return 0.0
        
        successful = len([r for r in self.results if r['success']])
        return successful / len(self.results)
    
    def get_failed_tests(self) -> List[Dict[str, Any]]:
        """Get list of failed tests"""
        return [r for r in self.results if not r['success']]


class DatabaseTestHelper:
    """Database testing utilities"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or ':memory:'
        self.connection = None
        self.setup_test_database()
    
    def setup_test_database(self):
        """Setup test database with required tables"""
        self.connection = sqlite3.connect(self.db_path)
        cursor = self.connection.cursor()
        
        # Create test tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS raw_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                content TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processed_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                raw_event_id INTEGER,
                component_type TEXT NOT NULL,
                result_data TEXT,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (raw_event_id) REFERENCES raw_events (id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entity_resolution (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                resolved_name TEXT,
                aliases TEXT,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.connection.commit()
    
    def insert_test_data(self, table: str, data: Dict[str, Any]) -> int:
        """Insert test data into specified table"""
        cursor = self.connection.cursor()
        
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data.keys()])
        values = list(data.values())
        
        cursor.execute(f'INSERT INTO {table} ({columns}) VALUES ({placeholders})', values)
        self.connection.commit()
        
        return cursor.lastrowid
    
    def query_test_data(self, table: str, conditions: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Query test data from specified table"""
        cursor = self.connection.cursor()
        
        query = f'SELECT * FROM {table}'
        params = []
        
        if conditions:
            where_clauses = []
            for column, value in conditions.items():
                where_clauses.append(f'{column} = ?')
                params.append(value)
            query += ' WHERE ' + ' AND '.join(where_clauses)
        
        cursor.execute(query, params)
        
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        
        return [dict(zip(columns, row)) for row in rows]
    
    def cleanup_test_data(self):
        """Clean up test data"""
        cursor = self.connection.cursor()
        cursor.execute('DELETE FROM processed_events')
        cursor.execute('DELETE FROM raw_events')
        cursor.execute('DELETE FROM entity_resolution')
        self.connection.commit()
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()


# Export all utilities
__all__ = [
    'TestDataGenerator',
    'MockSupabaseClient',
    'MockOpenAIClient',
    'PerformanceMonitor',
    'AccuracyValidator',
    'SecurityTester',
    'ConcurrentTestRunner',
    'DatabaseTestHelper'
]
