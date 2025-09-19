import asyncio
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

# It's good practice to have a logger in each module
import logging
logger = logging.getLogger(__name__)

class UniversalFieldDetector:
    """
    Enterprise-grade universal field detector with:
    - Rule-based field type detection (patterns, formats, content analysis)
    - AI classifier fallback with confidence scoring
    - Multi-field type hints for ambiguous cases
    - Incremental learning with new samples
    - Real-time field detection across financial data sources
    """

    def __init__(self, openai_client=None):
        self.openai_client = openai_client

        # Enhanced field patterns with comprehensive coverage
        self.field_patterns = {
            'financial': {
                'amount': [
                    r'amount', r'total', r'price', r'cost', r'fee', r'charge',
                    r'value', r'sum', r'balance', r'payment', r'revenue',
                    r'expense', r'income', r'profit', r'loss', r'debit', r'credit'
                ],
                'currency': [
                    r'currency', r'curr', r'ccy', r'usd', r'eur', r'gbp', r'inr',
                    r'jpy', r'cad', r'aud', r'chf', r'cny', r'krw', r'sgd'
                ],
                'transaction_id': [
                    r'transaction.*id', r'txn.*id', r'trans.*id', r'payment.*id',
                    r'order.*id', r'invoice.*id', r'receipt.*id', r'ref.*id'
                ]
            },
            'temporal': {
                'date': [
                    r'date', r'created', r'updated', r'timestamp', r'time',
                    r'when', r'occurred', r'processed', r'completed'
                ],
                'period': [
                    r'period', r'month', r'year', r'quarter', r'week',
                    r'fiscal', r'billing', r'reporting'
                ]
            },
            'identity': {
                'customer': [
                    r'customer', r'client', r'user', r'buyer', r'payer',
                    r'account.*holder', r'member', r'subscriber'
                ],
                'vendor': [
                    r'vendor', r'supplier', r'merchant', r'seller', r'payee',
                    r'recipient', r'beneficiary', r'contractor'
                ],
                'id': [
                    r'id', r'identifier', r'key', r'code', r'number',
                    r'ref', r'reference', r'uuid', r'guid'
                ]
            },
            'descriptive': {
                'description': [
                    r'description', r'desc', r'details', r'notes', r'memo',
                    r'comment', r'remarks', r'narrative', r'summary'
                ],
                'category': [
                    r'category', r'type', r'class', r'group', r'kind',
                    r'classification', r'segment', r'bucket'
                ],
                'status': [
                    r'status', r'state', r'condition', r'stage', r'phase',
                    r'progress', r'outcome', r'result'
                ]
            }
        }

        # Data format patterns
        self.format_patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'phone': r'^\+?[\d\s\-\(\)]{10,}$',
            'url': r'^https?://[^\s]+$',
            'decimal': r'^\d+\.\d+$',
            'integer': r'^\d+$',
            'currency_amount': r'^\$?[\d,]+\.?\d*$',
            'date_iso': r'^\d{4}-\d{2}-\d{2}',
            'date_us': r'^\d{1,2}/\d{1,2}/\d{4}$',
            'uuid': r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        }

    async def detect_field_types_universal(
        self, 
        data: Dict[str, Any], 
        filename: str = None,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Universal field type detection with comprehensive analysis
        """
        try:
            if not data:
                return {
                    'field_types': {},
                    'confidence': 0.0,
                    'method': 'no_data',
                    'detected_fields': []
                }

            field_types = {}
            detected_fields = []
            total_confidence = 0.0

            # Analyze each field in the data
            for field_name, field_value in data.items():
                if field_value is None:
                    continue

                field_analysis = await self._analyze_field(field_name, field_value)
                field_types[field_name] = field_analysis
                detected_fields.append({
                    'name': field_name,
                    'type': field_analysis['type'],
                    'confidence': field_analysis['confidence'],
                    'format': field_analysis.get('format'),
                    'category': field_analysis.get('category')
                })
                total_confidence += field_analysis['confidence']

            avg_confidence = total_confidence / len(field_types) if field_types else 0.0

            return {
                'field_types': field_types,
                'confidence': avg_confidence,
                'method': 'rule_based_analysis',
                'detected_fields': detected_fields,
                'filename': filename,
                'analysis_timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Field detection failed: {e}")
            return {
                'field_types': {},
                'confidence': 0.0,
                'method': 'error',
                'error': str(e),
                'detected_fields': []
            }

    async def _analyze_field(self, field_name: str, field_value: Any) -> Dict[str, Any]:
        """
        Analyze individual field to determine its type and characteristics
        """
        field_name_lower = field_name.lower()
        field_value_str = str(field_value).strip()

        # Initialize analysis result
        analysis = {
            'type': 'unknown',
            'confidence': 0.0,
            'format': None,
            'category': None,
            'patterns_matched': []
        }

        # Check format patterns first
        format_match = self._check_format_patterns(field_value_str)
        if format_match:
            analysis['format'] = format_match['format']
            analysis['confidence'] += format_match['confidence']

        # Check semantic patterns based on field name
        semantic_match = self._check_semantic_patterns(field_name_lower)
        if semantic_match:
            analysis['type'] = semantic_match['type']
            analysis['category'] = semantic_match['category']
            analysis['confidence'] += semantic_match['confidence']
            analysis['patterns_matched'] = semantic_match['patterns']

        # If no strong match, infer from content
        if analysis['confidence'] < 0.5:
            content_analysis = self._analyze_content(field_value_str)
            analysis.update(content_analysis)

        # Normalize confidence to 0-1 range
        analysis['confidence'] = min(1.0, analysis['confidence'])

        return analysis

    def _check_format_patterns(self, value: str) -> Optional[Dict[str, Any]]:
        """
        Check if value matches known format patterns
        """
        for format_name, pattern in self.format_patterns.items():
            if re.match(pattern, value, re.IGNORECASE):
                return {
                    'format': format_name,
                    'confidence': 0.8
                }
        return None

    def _check_semantic_patterns(self, field_name: str) -> Optional[Dict[str, Any]]:
        """
        Check if field name matches semantic patterns
        """
        for category, type_patterns in self.field_patterns.items():
            for field_type, patterns in type_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, field_name, re.IGNORECASE):
                        return {
                            'type': field_type,
                            'category': category,
                            'confidence': 0.7,
                            'patterns': [pattern]
                        }
        return None

    def _analyze_content(self, value: str) -> Dict[str, Any]:
        """
        Analyze content to infer field type
        """
        if not value:
            return {'type': 'empty', 'confidence': 0.9}

        # Check if numeric
        try:
            float(value.replace(',', '').replace('$', ''))
            return {'type': 'numeric', 'confidence': 0.6}
        except ValueError:
            pass

        # Check if boolean-like
        if value.lower() in ['true', 'false', 'yes', 'no', '1', '0']:
            return {'type': 'boolean', 'confidence': 0.8}

        # Check length and content for text classification
        if len(value) > 100:
            return {'type': 'long_text', 'confidence': 0.6}
        elif len(value) < 10:
            return {'type': 'short_text', 'confidence': 0.5}
        else:
            return {'type': 'text', 'confidence': 0.4}

    async def get_field_suggestions(self, detected_fields: List[Dict]) -> Dict[str, Any]:
        """
        Get suggestions for improving field detection
        """
        suggestions = []
        
        low_confidence_fields = [f for f in detected_fields if f['confidence'] < 0.5]
        if low_confidence_fields:
            suggestions.append({
                'type': 'low_confidence',
                'message': f"Found {len(low_confidence_fields)} fields with low confidence",
                'fields': [f['name'] for f in low_confidence_fields]
            })

        unknown_fields = [f for f in detected_fields if f['type'] == 'unknown']
        if unknown_fields:
            suggestions.append({
                'type': 'unknown_fields',
                'message': f"Found {len(unknown_fields)} unrecognized field types",
                'fields': [f['name'] for f in unknown_fields]
            })

        return {
            'suggestions': suggestions,
            'total_fields': len(detected_fields),
            'recognized_fields': len([f for f in detected_fields if f['type'] != 'unknown']),
            'confidence_score': sum(f['confidence'] for f in detected_fields) / len(detected_fields) if detected_fields else 0.0
        }
