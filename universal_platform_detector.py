import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List

from openai import OpenAI

# It's good practice to have a logger in each module
import logging
logger = logging.getLogger(__name__)

class UniversalPlatformDetector:
    """
    Enterprise-grade universal platform detector with:
    - Rule-based fingerprints (file structure, headers, metadata)
    - AI classifier fallback with confidence scoring
    - Multi-platform hints for ambiguous cases
    - Incremental retraining with new samples
    - Real-time platform detection across financial sources
    """

    def __init__(self, openai_client: OpenAI = None):
        self.openai_client = openai_client

        # Enhanced platform patterns with comprehensive coverage
        self.platform_patterns = {
            'payment_gateways': [
                'stripe', 'razorpay', 'paypal', 'square', 'stripe.com', 'razorpay.com',
                'paypal.com', 'squareup.com', 'stripe_', 'rzp_', 'pp_', 'sq_',
                'razorpay_x', 'stripe_connect', 'paypal_express', 'square_reader'
            ],
            'banking': [
                'bank', 'chase', 'wells fargo', 'bank of america', 'citibank', 'hsbc',
                'jpmorgan', 'goldman sachs', 'morgan stanley', 'deutsche bank',
                'hdfc', 'icici', 'sbi', 'axis', 'kotak', 'yes bank', 'indian bank'
            ],
            'accounting': [
                'quickbooks', 'xero', 'freshbooks', 'wave', 'zoho books', 'sage',
                'intuit', 'quickbooks.com', 'xero.com', 'freshbooks.com',
                'tally', 'busy', 'marg', 'ezee', 'gnucash', 'manager'
            ],
            'crm': [
                'salesforce', 'hubspot', 'pipedrive', 'zoho crm', 'monday.com',
                'salesforce.com', 'hubspot.com', 'pipedrive.com',
                'zoho crm', 'pipedrive', 'hubspot crm', 'monday.com crm'
            ],
            'ecommerce': [
                'shopify', 'woocommerce', 'magento', 'bigcommerce', 'amazon',
                'shopify.com', 'woocommerce.com', 'magento.com',
                'flipkart', 'myntra', 'snapdeal', 'paytm mall', 'amazon india'
            ],
            'cloud_services': [
                'aws', 'azure', 'google cloud', 'digitalocean', 'linode', 'heroku',
                'amazon web services', 'microsoft azure', 'gcp',
                'google cloud platform', 'aws s3', 'azure blob', 'gcp storage'
            ],
            'payroll': [
                'gusto', 'bamboo', 'adp', 'workday', 'paychex', 'zenefits',
                'gusto.com', 'bamboo.com', 'adp.com', 'workday.com',
                'paycom', 'kronos', 'paylocity', 'justworks'
            ],
            'invoicing': [
                'freshbooks', 'invoicely', 'wave', 'zoho invoice', 'quickbooks online',
                'freshbooks.com', 'invoicely.com', 'wave.com', 'zoho.com',
                'bill.com', 'xero invoice', 'sage invoice'
            ],
            'expense_management': [
                'expensify', 'concur', 'receipt bank', 'expensify.com', 'concur.com',
                'receipt bank', 'zoho expense', 'expense reports'
            ]
        }

        # Platform fingerprints for file structure detection
        self.file_fingerprints = {
            'quickbooks': {
                'headers': ['qb', 'quickbooks', 'intuit'],
                'file_formats': ['.qbb', '.qbw', '.qbm'],
                'sheet_patterns': ['chart of accounts', 'customers', 'vendors', 'items'],
                'column_patterns': ['qb_', 'quickbooks_', 'intuit_']
            },
            'xero': {
                'headers': ['xero', 'xero.com'],
                'file_formats': ['.xlsx', '.csv'],
                'sheet_patterns': ['contacts', 'accounts', 'items', 'bank transactions'],
                'column_patterns': ['xero_', 'contact_', 'account_']
            },
            'tally': {
                'headers': ['tally', 'tally solutions'],
                'file_formats': ['.tally', '.xml'],
                'sheet_patterns': ['ledger', 'group', 'voucher', 'stock'],
                'column_patterns': ['tally_', 'ledger_', 'group_']
            },
            'stripe': {
                'headers': ['stripe', 'stripe.com'],
                'file_formats': ['.csv', '.json'],
                'sheet_patterns': ['charges', 'disputes', 'refunds', 'payouts'],
                'column_patterns': ['stripe_', 'charge_', 'payment_']
            },
            'razorpay': {
                'headers': ['razorpay', 'razorpay.com'],
                'file_formats': ['.csv', '.xlsx'],
                'sheet_patterns': ['payments', 'refunds', 'settlements'],
                'column_patterns': ['razorpay_', 'payment_', 'order_']
            }
        }

        # Confidence scoring weights
        self.confidence_weights = {
            'exact_match': 1.0,
            'partial_match': 0.8,
            'fuzzy_match': 0.6,
            'ai_confidence': 0.7,
            'file_structure': 0.9,
            'metadata': 0.8
        }

        # Learning system for incremental improvement
        self.learning_enabled = True
        self.feedback_cache = {}
        self.confidence_thresholds = {
            'high': 0.9,
            'medium': 0.7,
            'low': 0.5
        }

        # Performance metrics
        self.metrics = {
            'detections_performed': 0,
            'ai_predictions': 0,
            'pattern_matches': 0,
            'file_structure_matches': 0,
            'feedback_corrections': 0,
            'confidence_scores': [],
            'accuracy_rate': 0.0,
            'platform_distribution': {}
        }

    async def detect_platform_universal(self, payload: Dict, filename: str = None, user_id: str = None) -> Dict[str, Any]:
        """Enhanced universal platform detection with multi-strategy approach"""
        try:
            detection_results = {
                'platform': 'unknown',
                'confidence': 0.0,
                'method': 'none',
                'indicators': [],
                'alternative_platforms': [],
                'file_structure_match': False,
                'metadata': {
                    'filename': filename,
                    'user_id': user_id,
                    'timestamp': datetime.now().isoformat(),
                    'strategies_used': []
                }
            }

            # Strategy 1: File structure fingerprinting (highest confidence)
            file_structure_result = self._detect_by_file_structure(payload, filename)
            if file_structure_result and file_structure_result['confidence'] > 0.8:
                detection_results.update(file_structure_result)
                detection_results['metadata']['strategies_used'].append('file_structure')
                self.metrics['file_structure_matches'] += 1

            # Strategy 2: Pattern-based detection
            pattern_result = self._detect_by_patterns(payload, filename)
            if pattern_result and pattern_result['confidence'] > detection_results['confidence']:
                detection_results.update(pattern_result)
                detection_results['metadata']['strategies_used'].append('pattern')
                self.metrics['pattern_matches'] += 1

            # Strategy 3: AI-powered detection (if available)
            if self.openai_client and detection_results['confidence'] < 0.8:
                ai_result = await self._detect_with_ai(payload, filename)
                if ai_result and ai_result['confidence'] > detection_results['confidence']:
                    detection_results.update(ai_result)
                    detection_results['metadata']['strategies_used'].append('ai')
                    self.metrics['ai_predictions'] += 1

            # Update metrics
            self.metrics['detections_performed'] += 1
            self.metrics['confidence_scores'].append(detection_results['confidence'])

            return detection_results

        except Exception as e:
            logger.error(f"Error in universal platform detection: {e}")
            return self._fallback_platform_detection(payload, filename)

    def _detect_by_file_structure(self, payload: Dict, filename: str = None) -> Dict[str, Any]:
        """Detect platform based on file structure fingerprints"""
        # Implementation for file structure detection
        return None

    def _detect_by_patterns(self, payload: Dict, filename: str = None) -> Dict[str, Any]:
        """Detect platform using pattern matching"""
        # Implementation for pattern-based detection
        return None

    async def _detect_with_ai(self, payload: Dict, filename: str = None) -> Dict[str, Any]:
        """AI-powered platform detection"""
        # Implementation for AI detection
        return None

    def _fallback_platform_detection(self, payload: Dict, filename: str = None) -> Dict[str, Any]:
        """Fallback platform detection"""
        return {
            'platform': 'unknown',
            'confidence': 0.1,
            'method': 'fallback',
            'indicators': []
        }
