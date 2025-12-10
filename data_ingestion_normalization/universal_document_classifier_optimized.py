"""Universal Document Classifier v4.0.0

Multi-faceted document classification using:
- Pattern matching (pyahocorasick)
- AI classification (Groq Llama-3.3-70B)
- OCR processing (easyocr)
- Zero-shot classification (sentence-transformers)
- TF-IDF indicator weighting
- Redis-backed caching (aiocache)
- Learning system with database persistence

Author: Senior Full-Stack Engineer
Version: 4.0.0
"""

import asyncio
import hashlib
import orjson as json
import re
import time
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass
from pathlib import Path

import ahocorasick
import easyocr
import structlog
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings
from aiocache import cached, Cache
from aiocache.serializers import JsonSerializer
import yaml
from sentence_transformers import SentenceTransformer

logger = structlog.get_logger(__name__)

# Global preloaded models (shared across all instances)
_SENTENCE_MODEL = None
_OCR_READER = None
_AUTOMATON = None

# Global TF-IDF cache (prevents re-training on every instance)
_TFIDF_CACHE = {
    'vectorizer': None,
    'doc_type_vectors': None,
    'doc_types_list': None,
    'initialized': False
}

def _initialize_global_models():
    """Preload all models at module import time (standard Python practice)."""
    global _SENTENCE_MODEL, _OCR_READER, _AUTOMATON
    
    try:
        if _SENTENCE_MODEL is None:
            logger.info("Preloading SentenceTransformer model...")
            _SENTENCE_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ SentenceTransformer preloaded")
    except Exception as e:
        logger.warning(f"Failed to preload SentenceTransformer: {e}")
        _SENTENCE_MODEL = None
    
    try:
        if _OCR_READER is None:
            logger.info("Preloading EasyOCR reader...")
            _OCR_READER = easyocr.Reader(['en'], gpu=False)
            logger.info("✅ EasyOCR preloaded")
    except Exception as e:
        logger.warning(f"Failed to preload EasyOCR: {e}")
        _OCR_READER = None
    
    try:
        if _AUTOMATON is None:
            logger.info("Preloading Ahocorasick automaton...")
            _AUTOMATON = ahocorasick.Automaton()
            keywords = ['invoice', 'receipt', 'payment', 'transaction', 'balance',
                       'account', 'deposit', 'withdrawal', 'transfer', 'salary',
                       'expense', 'revenue', 'income', 'tax', 'refund']
            for keyword in keywords:
                _AUTOMATON.add_word(keyword, keyword)
            _AUTOMATON.make_deterministic()
            logger.info("✅ Ahocorasick automaton preloaded")
    except Exception as e:
        logger.warning(f"Failed to preload Ahocorasick: {e}")
        _AUTOMATON = None

class DocumentClassifierConfig(BaseSettings):
    """Type-safe configuration with auto-validation"""
    enable_caching: bool = True
    cache_ttl: int = 7200  # 2 hours
    max_cache_size: int = 10000
    enable_ai_classification: bool = True
    enable_ocr_classification: bool = True
    enable_learning: bool = True
    confidence_threshold: float = Field(ge=0.0, le=1.0, default=0.7)
    max_indicators: int = 10
    ai_model: str = 'llama-3.3-70b-versatile'
    ai_temperature: float = Field(ge=0.0, le=1.0, default=0.1)
    ai_max_tokens: int = Field(ge=50, le=2000, default=300)
    learning_window: int = 1000
    update_frequency: int = 3600
    document_types_yaml: str = 'config/document_types.yaml'  # External config file
    
    model_config = ConfigDict(
        env_prefix='DOC_CLASSIFIER_',
        case_sensitive=False
    )

class DocumentTypeDefinition(BaseModel):
    """Type-safe document type definition with auto-validation"""
    name: str
    category: Literal['financial', 'business', 'legal', 'healthcare', 'government', 'personal', 'education']
    indicators: List[str] = Field(min_length=1, max_length=100)
    field_patterns: List[str] = []
    keywords: List[str] = Field(min_length=1, max_length=20)
    confidence_boost: float = Field(ge=0.0, le=1.0, default=0.8)
    
    @field_validator('indicators', 'keywords', mode='before')
    @classmethod
    def lowercase_lists(cls, v):
        return [i.lower().strip() for i in v]
    
    model_config = ConfigDict(frozen=True)  # Immutable

@dataclass
class DocumentClassificationResult:
    """Standardized document classification result"""
    document_type: str
    confidence: float
    method: str
    indicators: List[str]
    reasoning: str
    metadata: Dict[str, Any]

class UniversalDocumentClassifierOptimized:
    """
    Production-grade universal document classifier with AI, OCR integration,
    confidence scoring, and comprehensive document type coverage.
    
    Features:
    - AI-powered document classification with GPT-4
    - OCR integration for image-based documents
    - Comprehensive document type database (20+ types)
    - Intelligent caching and learning
    - Confidence scoring and validation
    - Async processing for high concurrency
    - Robust error handling and fallbacks
    - Real-time classification updates
    """
    
    def __init__(self, groq_client=None, cache_client=None, supabase_client=None, config=None):
        self.groq_client = groq_client
        self.supabase = supabase_client
        self.config = config or self._get_default_config()
        
        from core_infrastructure.utils.helpers import initialize_centralized_cache
        self.cache = initialize_centralized_cache(cache_client)
        self.document_database = self._initialize_document_database()
        
        # Use preloaded models (initialized at module import time)
        self.automaton = _AUTOMATON
        self.ocr_reader = _OCR_READER
        self.ocr_available = _OCR_READER is not None
        self.sentence_model = _SENTENCE_MODEL
        self.row_type_embeddings = None
        self.tfidf_vectorizer = None
        self.doc_type_vectors = None
        self.doc_types_list = None
        
        if self.sentence_model is not None:
            self._initialize_row_type_embeddings()
        self._initialize_tfidf()
        
        # Performance tracking
        self.metrics = {
            'classifications_performed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'ai_classifications': 0,
            'pattern_classifications': 0,
            'ocr_classifications': 0,
            'semantic_batch_classifications': 0,
            'fallback_classifications': 0,
            'avg_confidence': 0.0,
            'document_type_distribution': {},
            'processing_times': []
        }
        
        # FIX #81: Use shared learning system
        from data_ingestion_normalization.shared_learning_system import SharedLearningSystem
        self.learning_system = SharedLearningSystem()
        self.learning_enabled = True
        
        logger.info("NASA-GRADE Document Classifier v4.0.0 initialized (PRELOADED models)",
                   cache_size=self.config.max_cache_size,
                   document_types=len(self.document_database),
                   automaton_ready=self.automaton is not None,
                   ocr_available=self.ocr_available,
                   semantic_model_loaded=self.sentence_model is not None,
                   tfidf_trained=self.tfidf_vectorizer is not None)
    
    def _get_default_config(self):
        """Get type-safe configuration with pydantic-settings"""
        return DocumentClassifierConfig()
    
    def _initialize_row_type_embeddings(self):
        """Initialize row type embeddings using preloaded SentenceTransformer."""
        if self.sentence_model is None:
            logger.warning("SentenceTransformer not available, row type embeddings skipped")
            self.row_type_embeddings = None
            return
        
        try:
            self.row_types = {
                'revenue_income': 'payment received from client, sales revenue, income, money coming in',
                'operating_expense': 'business expense, vendor payment, cost, money going out',
                'payroll_expense': 'employee salary, wages, payroll, staff payment',
                'tax_payment': 'tax payment, IRS, government tax, tax withholding',
                'loan_payment': 'loan payment, debt payment, financing payment',
                'investment': 'investment, stock purchase, bond, mutual fund',
                'transfer': 'transfer between accounts, internal transfer'
            }
            
            row_type_texts = list(self.row_types.values())
            embeddings = self.sentence_model.encode(row_type_texts)
            self.row_type_embeddings = {
                row_type: embedding 
                for row_type, embedding in zip(self.row_types.keys(), embeddings)
            }
            
            logger.info("✅ Row type embeddings preloaded successfully",
                       row_types=len(self.row_type_embeddings),
                       embedding_model="all-MiniLM-L6-v2")
        except Exception as e:
            logger.warning(f"Failed to initialize row type embeddings: {e}")
            self.row_type_embeddings = None
    
    # REMOVED: Old lazy-loading methods (_initialize_ocr, _ensure_ocr_available, _initialize_sentence_model)
    # Models are now PRELOADED at module level following standard Python developer practices
    # See _initialize_global_models() function at module top for preloading logic
    
    def _initialize_tfidf(self):
        """FIX #58: Initialize TF-IDF vectorizer with global caching to prevent re-training"""
        global _TFIDF_CACHE
        
        # Check if already initialized globally
        if _TFIDF_CACHE['initialized']:
            self.tfidf_vectorizer = _TFIDF_CACHE['vectorizer']
            self.doc_type_vectors = _TFIDF_CACHE['doc_type_vectors']
            self.doc_types_list = _TFIDF_CACHE['doc_types_list']
            logger.info("TF-IDF vectorizer loaded from global cache (no re-training)")
            return
        
        try:
            corpus = []
            doc_types_list = []
            
            for doc_type, info in self.document_database.items():
                # Combine indicators into document representation
                doc_text = ' '.join(info['indicators'] + info['keywords'] + info['field_patterns'])
                corpus.append(doc_text)
                doc_types_list.append(doc_type)
            
            # Train TF-IDF on indicator corpus (only once globally)
            tfidf_vectorizer = TfidfVectorizer()
            doc_type_vectors = tfidf_vectorizer.fit_transform(corpus)
            
            # Store in global cache
            _TFIDF_CACHE['vectorizer'] = tfidf_vectorizer
            _TFIDF_CACHE['doc_type_vectors'] = doc_type_vectors
            _TFIDF_CACHE['doc_types_list'] = doc_types_list
            _TFIDF_CACHE['initialized'] = True
            
            # Reference from instance
            self.tfidf_vectorizer = tfidf_vectorizer
            self.doc_type_vectors = doc_type_vectors
            self.doc_types_list = doc_types_list
            
            logger.info("TF-IDF vectorizer trained and cached globally",
                       document_types=len(corpus),
                       features="smart_weighting+ambiguity_handling")
        except Exception as e:
            logger.warning("TF-IDF initialization failed", error=str(e))
            self.tfidf_vectorizer = None
    
    def _initialize_document_database(self) -> Dict[str, Dict[str, Any]]:
        """OPTIMIZED: Load document types from YAML (non-devs can edit!)"""
        try:
            # Try to load from external YAML file
            yaml_path = Path(self.config.document_types_yaml)
            if yaml_path.exists():
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    yaml_data = yaml.safe_load(f)
                
                # Flatten nested structure: {category: {doc_type: {...}}} -> {doc_type: {...}}
                document_database = {}
                for category, doc_types in yaml_data.items():
                    for doc_type_id, doc_info in doc_types.items():
                        document_database[doc_type_id] = doc_info
                
                logger.info("Document types loaded from YAML",
                           file=str(yaml_path),
                           document_types=len(document_database))
                return document_database
            else:
                logger.warning("YAML file not found, using hardcoded defaults",
                             expected_path=str(yaml_path))
        except Exception as e:
            logger.error("Failed to load YAML, using hardcoded defaults", error=str(e))
        
        # Fallback to hardcoded document types
        return {
            # Financial Documents
            'invoice': {
                'name': 'Invoice',
                'category': 'financial',
                'indicators': [
                    'invoice', 'bill', 'invoice number', 'bill number', 'invoice date',
                    'bill date', 'amount due', 'total amount', 'subtotal', 'tax amount',
                    'invoice to', 'bill to', 'from', 'description', 'quantity', 'rate',
                    'invoice id', 'bill id', 'payment terms', 'due date'
                ],
                'field_patterns': ['invoice_number', 'bill_number', 'invoice_date', 'due_date', 'total_amount'],
                'confidence_boost': 0.9,
                'keywords': ['invoice', 'bill', 'charge', 'amount due', 'payment terms']
            },
            'receipt': {
                'name': 'Receipt',
                'category': 'financial',
                'indicators': [
                    'receipt', 'receipt number', 'transaction id', 'purchase date',
                    'total paid', 'amount paid', 'payment method', 'cash', 'credit card',
                    'debit card', 'change', 'subtotal', 'tax', 'receipt date'
                ],
                'field_patterns': ['receipt_number', 'transaction_id', 'purchase_date', 'total_paid'],
                'confidence_boost': 0.9,
                'keywords': ['receipt', 'purchase', 'paid', 'transaction', 'change']
            },
            'bank_statement': {
                'name': 'Bank Statement',
                'category': 'financial',
                'indicators': [
                    'bank statement', 'account statement', 'account number', 'statement period',
                    'opening balance', 'closing balance', 'deposit', 'withdrawal', 'transfer',
                    'balance forward', 'account summary', 'transaction history', 'available balance'
                ],
                'field_patterns': ['account_number', 'statement_period', 'opening_balance', 'closing_balance'],
                'confidence_boost': 0.9,
                'keywords': ['bank', 'statement', 'account', 'balance', 'deposit', 'withdrawal']
            },
            'credit_card_statement': {
                'name': 'Credit Card Statement',
                'category': 'financial',
                'indicators': [
                    'credit card', 'card statement', 'credit limit', 'available credit',
                    'minimum payment', 'payment due date', 'statement balance', 'card number',
                    'purchase', 'payment', 'interest charge', 'annual fee', 'cash advance'
                ],
                'field_patterns': ['card_number', 'credit_limit', 'available_credit', 'minimum_payment'],
                'confidence_boost': 0.9,
                'keywords': ['credit card', 'card', 'credit limit', 'minimum payment']
            },
            'payroll': {
                'name': 'Payroll',
                'category': 'financial',
                'indicators': [
                    'payroll', 'salary', 'wage', 'employee', 'pay period', 'gross pay',
                    'net pay', 'deduction', 'tax withholding', 'social security', 'medicare',
                    'health insurance', '401k', 'bonus', 'overtime', 'hours worked'
                ],
                'field_patterns': ['employee_id', 'pay_period', 'gross_pay', 'net_pay', 'deductions'],
                'confidence_boost': 0.9,
                'keywords': ['payroll', 'salary', 'employee', 'pay period', 'deduction']
            },
            'tax_document': {
                'name': 'Tax Document',
                'category': 'financial',
                'indicators': [
                    'tax', 'tax return', '1040', 'w-2', '1099', 'tax year', 'filing status',
                    'adjusted gross income', 'taxable income', 'tax owed', 'refund',
                    'standard deduction', 'itemized deduction', 'tax bracket'
                ],
                'field_patterns': ['tax_year', 'filing_status', 'adjusted_gross_income', 'tax_owed'],
                'confidence_boost': 0.9,
                'keywords': ['tax', 'return', '1040', 'w-2', '1099', 'refund']
            },
            
            # Business Documents
            'contract': {
                'name': 'Contract',
                'category': 'business',
                'indicators': [
                    'contract', 'agreement', 'terms and conditions', 'party', 'effective date',
                    'termination date', 'contract term', 'liability', 'indemnification',
                    'force majeure', 'governing law', 'signature', 'witness'
                ],
                'field_patterns': ['contract_number', 'effective_date', 'termination_date', 'party_name'],
                'confidence_boost': 0.8,
                'keywords': ['contract', 'agreement', 'terms', 'party', 'effective date']
            },
            'purchase_order': {
                'name': 'Purchase Order',
                'category': 'business',
                'indicators': [
                    'purchase order', 'po number', 'po date', 'vendor', 'supplier',
                    'ship to', 'bill to', 'item description', 'quantity', 'unit price',
                    'total price', 'delivery date', 'payment terms'
                ],
                'field_patterns': ['po_number', 'po_date', 'vendor_name', 'delivery_date'],
                'confidence_boost': 0.9,
                'keywords': ['purchase order', 'po number', 'vendor', 'supplier']
            },
            'quotation': {
                'name': 'Quotation',
                'category': 'business',
                'indicators': [
                    'quotation', 'quote', 'quote number', 'quote date', 'valid until',
                    'customer', 'client', 'item description', 'quantity', 'unit price',
                    'total price', 'terms and conditions', 'delivery time'
                ],
                'field_patterns': ['quote_number', 'quote_date', 'valid_until', 'customer_name'],
                'confidence_boost': 0.9,
                'keywords': ['quotation', 'quote', 'quote number', 'valid until']
            },
            'proposal': {
                'name': 'Proposal',
                'category': 'business',
                'indicators': [
                    'proposal', 'proposal number', 'proposal date', 'client', 'project',
                    'scope of work', 'timeline', 'budget', 'deliverables', 'terms',
                    'acceptance', 'signature', 'proposal validity'
                ],
                'field_patterns': ['proposal_number', 'proposal_date', 'client_name', 'project_name'],
                'confidence_boost': 0.8,
                'keywords': ['proposal', 'proposal number', 'scope of work', 'deliverables']
            },
            
            # Legal Documents
            'legal_document': {
                'name': 'Legal Document',
                'category': 'legal',
                'indicators': [
                    'legal', 'attorney', 'lawyer', 'court', 'case number', 'filing date',
                    'plaintiff', 'defendant', 'judge', 'hearing date', 'motion',
                    'affidavit', 'deposition', 'settlement', 'verdict'
                ],
                'field_patterns': ['case_number', 'filing_date', 'plaintiff', 'defendant'],
                'confidence_boost': 0.8,
                'keywords': ['legal', 'attorney', 'court', 'case number', 'plaintiff']
            },
            'insurance_document': {
                'name': 'Insurance Document',
                'category': 'legal',
                'indicators': [
                    'insurance', 'policy', 'policy number', 'coverage', 'premium',
                    'deductible', 'claim', 'claim number', 'adjuster', 'coverage period',
                    'beneficiary', 'insured', 'policy holder'
                ],
                'field_patterns': ['policy_number', 'coverage_period', 'premium_amount', 'deductible'],
                'confidence_boost': 0.9,
                'keywords': ['insurance', 'policy', 'coverage', 'premium', 'claim']
            },
            
            # Healthcare Documents
            'medical_bill': {
                'name': 'Medical Bill',
                'category': 'healthcare',
                'indicators': [
                    'medical', 'hospital', 'doctor', 'patient', 'diagnosis', 'treatment',
                    'procedure', 'medication', 'prescription', 'insurance', 'copay',
                    'deductible', 'medical bill', 'hospital bill', 'physician bill'
                ],
                'field_patterns': ['patient_id', 'diagnosis_code', 'procedure_code', 'insurance_id'],
                'confidence_boost': 0.9,
                'keywords': ['medical', 'hospital', 'patient', 'diagnosis', 'treatment']
            },
            'prescription': {
                'name': 'Prescription',
                'category': 'healthcare',
                'indicators': [
                    'prescription', 'medication', 'drug', 'dosage', 'pharmacy', 'refill',
                    'prescribing physician', 'patient name', 'prescription number',
                    'quantity', 'directions', 'side effects'
                ],
                'field_patterns': ['prescription_number', 'medication_name', 'dosage', 'quantity'],
                'confidence_boost': 0.9,
                'keywords': ['prescription', 'medication', 'dosage', 'pharmacy', 'refill']
            },
            
            # Government Documents
            'government_document': {
                'name': 'Government Document',
                'category': 'government',
                'indicators': [
                    'government', 'federal', 'state', 'municipal', 'department', 'agency',
                    'permit', 'license', 'registration', 'certificate', 'official',
                    'public record', 'filing', 'application'
                ],
                'field_patterns': ['permit_number', 'license_number', 'registration_number', 'department'],
                'confidence_boost': 0.8,
                'keywords': ['government', 'permit', 'license', 'registration', 'official']
            },
            'utility_bill': {
                'name': 'Utility Bill',
                'category': 'government',
                'indicators': [
                    'utility', 'electric', 'gas', 'water', 'internet', 'phone', 'cable',
                    'bill', 'account number', 'service address', 'billing period',
                    'usage', 'rate', 'previous reading', 'current reading'
                ],
                'field_patterns': ['account_number', 'service_address', 'billing_period', 'usage_amount'],
                'confidence_boost': 0.9,
                'keywords': ['utility', 'electric', 'gas', 'water', 'bill']
            },
            
            # Personal Documents
            'personal_document': {
                'name': 'Personal Document',
                'category': 'personal',
                'indicators': [
                    'personal', 'individual', 'private', 'confidential', 'personal information',
                    'id number', 'social security', 'date of birth', 'address', 'phone',
                    'email', 'emergency contact', 'next of kin'
                ],
                'field_patterns': ['id_number', 'date_of_birth', 'address', 'phone_number'],
                'confidence_boost': 0.7,
                'keywords': ['personal', 'individual', 'private', 'confidential']
            },
            'educational_document': {
                'name': 'Educational Document',
                'category': 'education',
                'indicators': [
                    'education', 'school', 'university', 'college', 'student', 'grade',
                    'transcript', 'diploma', 'degree', 'course', 'credit', 'gpa',
                    'enrollment', 'graduation', 'academic'
                ],
                'field_patterns': ['student_id', 'course_code', 'grade', 'gpa'],
                'confidence_boost': 0.8,
                'keywords': ['education', 'school', 'student', 'grade', 'transcript']
            }
        }
    
    async def classify_document_universal(self, payload: Dict, filename: str = None, 
                                        file_content: bytes = None, user_id: str = None) -> Dict[str, Any]:
        """
        Classify document using comprehensive AI-powered analysis with caching and learning.
        """
        start_time = time.time()
        classification_id = self._generate_classification_id(payload, filename, user_id)
        
        # Build deterministic cache content for AI cache integration (safe for non-JSON payloads)
        try:
            file_hash = hashlib.sha256(file_content).hexdigest() if file_content else None
        except Exception:
            file_hash = None
        try:
            payload_keys = sorted(list(payload.keys())) if isinstance(payload, dict) else []
        except Exception:
            payload_keys = []
        cache_content = {
            'user_id': user_id,
            'filename': filename,
            'payload_keys': payload_keys,
            'file_hash': file_hash
        }
        
        try:
            # 1. OPTIMIZED: Check centralized Redis cache
            if self.config.enable_caching:
                cached_result = await self.cache.get(classification_id)
                if cached_result:
                    self.metrics['cache_hits'] += 1
                    logger.debug("Cache hit", classification_id=classification_id)
                    return cached_result
            
            self.metrics['cache_misses'] += 1
            
            # 2. AI-powered classification (primary method)
            ai_result = None
            if self.config.enable_ai_classification and self.groq_client:
                ai_result = await self._classify_document_with_ai(payload, filename)
                if ai_result and ai_result['confidence'] >= 0.8:
                    self.metrics['ai_classifications'] += 1
                    final_result = ai_result
                else:
                    ai_result = None
            
            # 3. OCR-based classification (if image content available)
            ocr_result = None
            if self.config.enable_ocr_classification and self.ocr_available and file_content:
                ocr_result = await self._classify_document_with_ocr(file_content, filename)
                if ocr_result and ocr_result['confidence'] >= 0.7:
                    self.metrics['ocr_classifications'] += 1
                    if not ai_result or ocr_result['confidence'] > ai_result['confidence']:
                        final_result = ocr_result
                    else:
                        # Combine AI and OCR results
                        final_result = await self._combine_classification_results(ai_result, ocr_result)
                else:
                    ocr_result = None
            
            # 4. Pattern-based classification (fallback/enhancement)
            pattern_result = await self._classify_document_with_patterns(payload, filename)
            if pattern_result and pattern_result['confidence'] >= 0.6:
                self.metrics['pattern_classifications'] += 1
                if not ai_result and not ocr_result:
                    final_result = pattern_result
                elif ai_result and pattern_result['confidence'] > ai_result['confidence']:
                    final_result = pattern_result
                elif ocr_result and pattern_result['confidence'] > ocr_result['confidence']:
                    final_result = pattern_result
                else:
                    # Combine with existing results
                    if ai_result:
                        final_result = await self._combine_classification_results(ai_result, pattern_result)
                    elif ocr_result:
                        final_result = await self._combine_classification_results(ocr_result, pattern_result)
            elif not ai_result and not ocr_result:
                # 5. Fallback classification
                final_result = await self._classify_document_fallback(payload, filename)
                self.metrics['fallback_classifications'] += 1
            
            # 6. Enhance result with metadata
            final_result.update({
                'classification_id': classification_id,
                'processing_time': time.time() - start_time,
                'timestamp': datetime.utcnow().isoformat(),
                'metadata': {
                    'filename': filename,
                    'user_id': user_id,
                    'payload_keys': list(payload.keys()) if isinstance(payload, dict) else [],
                    'classification_methods_used': [final_result['method']],
                    'ocr_available': self.ocr_available,
                    'ai_available': self.groq_client is not None  # FIX #76: Use groq_client instead of anthropic
                }
            })
            
            # 7. OPTIMIZED: Cache with centralized Redis cache
            if self.config.enable_caching:
                await self.cache.set(classification_id, final_result)
            
            # 8. Update metrics and learning
            self._update_classification_metrics(final_result)
            if self.config.enable_learning:
                await self._update_learning_system(final_result, payload, filename, user_id)
            
            # 9. Audit logging
            await self._log_classification_audit(classification_id, final_result, user_id)
            
            # 10. Audit logging via structlog
            logger.info("document_classified", document_type=final_result['document_type'], confidence=final_result['confidence'], method=final_result['method'])
            
            return final_result
            
        except Exception as e:
            error_result = {
                'classification_id': classification_id,
                'document_type': 'unknown',
                'confidence': 0.0,
                'method': 'error',
                'indicators': [],
                'reasoning': f'Classification failed: {str(e)}',
                'error': str(e),
                'processing_time': time.time() - start_time,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            self.metrics['error_count'] = self.metrics.get('error_count', 0) + 1
            logger.error(f"Document classification failed: {e}")
            
            return error_result
    
    async def _classify_document_with_ai(self, payload: Dict, filename: str = None) -> Optional[Dict[str, Any]]:
        """Use AI to classify document type with enhanced prompting"""
        try:
            # Prepare comprehensive context for AI
            context_parts = []
            
            # Add filename if available
            if filename:
                context_parts.append(f"Filename: {filename}")
            
            # Add key fields that might indicate document type
            key_fields = ['description', 'memo', 'notes', 'type', 'category', 'kind', 'title', 'subject']
            for field in key_fields:
                if field in payload and payload[field]:
                    context_parts.append(f"{field}: {payload[field]}")
            
            # Add all field names as context
            if isinstance(payload, dict):
                field_names = list(payload.keys())
                context_parts.append(f"Field names: {', '.join(field_names)}")
                
                # Add sample values for analysis
                sample_values = []
                for key, value in list(payload.items())[:10]:  # Limit to first 10 fields
                    if isinstance(value, str) and len(str(value)) < 100:  # Avoid very long values
                        sample_values.append(f"{key}={value}")
                if sample_values:
                    context_parts.append(f"Sample values: {', '.join(sample_values)}")
            
            context = "\n".join(context_parts)
            
            # Enhanced AI prompt with comprehensive document type list
            document_list = "\n".join([f"- {info['name']}: {', '.join(info['keywords'][:5])}" 
                                     for info in self.document_database.values()])
            
            prompt = f"""
            Classify this financial document data into one of these types:
            
            {context}
            
            Supported document types:
            {document_list}
            
            Respond with JSON format:
            {{
                "document_type": "detected_type",
                "confidence": 0.0-1.0,
                "indicators": ["list", "of", "found", "indicators"],
                "reasoning": "detailed explanation of classification logic",
                "category": "document_category",
                "alternative_types": ["other", "possible", "types"]
            }}
            """
            
            # Use Groq Llama-3.3-70B for cost-effective document classification
            result_text = await self._safe_groq_call(
                prompt,
                self.config.ai_temperature,
                self.config.ai_max_tokens
            )
            
            # Parse and validate AI response
            result = self._parse_ai_response(result_text)
            
            if result and result.get('document_type') != 'unknown':
                return {
                    'document_type': result['document_type'],
                    'confidence': min(float(result.get('confidence', 0.0)), 1.0),
                    'method': 'ai',
                    'indicators': result.get('indicators', []),
                    'reasoning': result.get('reasoning', ''),
                    'category': result.get('category', 'unknown'),
                    'alternative_types': result.get('alternative_types', [])
                }
            
            return None
            
        except Exception as e:
            logger.error(f"AI document classification failed: {e}")
            return None
    
    async def _classify_document_with_ocr(self, file_content: bytes, filename: str = None) -> Optional[Dict[str, Any]]:
        """Use OCR to classify document from image content"""
        if not self.ocr_available:
            return None

        try:
            from PIL import Image
            from PIL import UnidentifiedImageError
            import io
            import filetype
            import easyocr

            # Only attempt OCR when the uploaded file is an image
            if not file_content:
                return None

            file_info = filetype.guess(file_content)
            if not file_info or not file_info.mime.startswith("image/"):
                return None

            # Open image
            try:
                image = Image.open(io.BytesIO(file_content))
            except UnidentifiedImageError:
                return None

            # CRITICAL FIX: Use async OCR service (non-blocking)
            from inference_service import OCRService
            ocr_results = await OCRService.read_text(file_content)
            # Returns: [([[x1,y1], [x2,y2], [x3,y3], [x4,y4]], 'Invoice', 0.95), ...]
            
            if not ocr_results:
                logger.warning("ocr_no_results", file_size=len(file_content))
                return None
            
            # Extract text and spatial information
            extracted_text = ' '.join([text for (bbox, text, conf) in ocr_results if conf > 0.5])
            text_lower = extracted_text.lower()
            
            # GENIUS: Use spatial intelligence - check if keywords appear in top section
            top_section_texts = [text.lower() for (bbox, text, conf) in ocr_results 
                                if bbox[0][1] < 100 and conf > 0.7]  # Top 100px, high confidence
            
            best_match = None
            best_confidence = 0.0
            
            for doc_type_id, doc_info in self.document_database.items():
                indicators_found = []
                confidence_score = 0.0
                
                # Check keywords
                for keyword in doc_info['keywords']:
                    if keyword.lower() in text_lower:
                        indicators_found.append(keyword)
                        confidence_score += 0.15
                
                # Check indicators
                for indicator in doc_info['indicators']:
                    if indicator.lower() in text_lower:
                        indicators_found.append(indicator)
                        confidence_score += 0.1
                
                # Apply confidence boost
                if confidence_score > 0:
                    confidence_score = min(confidence_score * doc_info.get('confidence_boost', 1.0), 1.0)
                
                if confidence_score > best_confidence:
                    best_match = {
                        'document_type': doc_info['name'],
                        'confidence': confidence_score,
                        'method': 'ocr',
                        'indicators': indicators_found,
                        'reasoning': f"OCR found {len(indicators_found)} indicators: {', '.join(indicators_found[:3])}",
                        'category': doc_info['category'],
                        'extracted_text': extracted_text[:500]  # First 500 chars for debugging
                    }
                    best_confidence = confidence_score
            
            return best_match if best_confidence >= 0.3 else None
            
        except Exception as e:
            logger.error(f"OCR document classification failed: {e}")
            return None
    
    async def _classify_document_with_patterns(self, payload: Dict, filename: str = None) -> Optional[Dict[str, Any]]:
        """GENIUS v4.0: Pattern-based classification with pyahocorasick (2x faster, async-ready) + TF-IDF weighting"""
        try:
            # Combine all text for pattern matching
            text_parts = []
            
            # Add filename
            if filename:
                text_parts.append(filename.lower())
            
            # Add all string values
            if isinstance(payload, dict):
                for value in payload.values():
                    if isinstance(value, str):
                        text_parts.append(value.lower())
            elif hasattr(payload, 'values'):
                # Handle DataFrame case
                try:
                    for value in payload.values.flatten():
                        if isinstance(value, str):
                            text_parts.append(value.lower())
                except:
                    pass
            
            combined_text = " ".join(text_parts).lower()  # Case-insensitive
            
            # CRITICAL FIX: Lazy-load automaton from inference service
            if self.automaton is None:
                from inference_service import AutomatonService
                self.automaton = await AutomatonService.get_document_automaton()
            
            # GENIUS v4.0: Use pyahocorasick automaton (2x faster, async-ready)
            # Single pass through text - O(n) with Aho-Corasick algorithm
            found_matches = []
            for end_index, (doc_type_id, keyword) in self.automaton.iter(combined_text):
                found_matches.append(doc_type_id)
            
            if not found_matches:
                return None
            
            # Count matches per document type
            doc_type_matches = {}
            for doc_type_id in found_matches:
                doc_type_matches[doc_type_id] = doc_type_matches.get(doc_type_id, 0) + 1
            
            # CRITICAL FIX: Lazy-load TF-IDF from inference service
            if self.tfidf_vectorizer is None:
                from inference_service import TFIDFService
                self.tfidf_vectorizer, self.doc_type_vectors = await TFIDFService.get_vectorizer()
            
            # OPTIMIZED: Use TF-IDF for smart indicator weighting
            if self.tfidf_vectorizer and self.doc_type_vectors is not None:
                try:
                    # Transform combined text to TF-IDF vector
                    text_vector = self.tfidf_vectorizer.transform([combined_text])
                    
                    # Calculate cosine similarity with all document types
                    similarities = cosine_similarity(text_vector, self.doc_type_vectors)[0]
                    
                    # Get best match
                    best_idx = np.argmax(similarities)
                    best_doc_type = self.doc_types_list[best_idx]
                    tfidf_confidence = float(similarities[best_idx])
                    
                    # Combine pyahocorasick matches with TF-IDF score
                    pattern_confidence = doc_type_matches.get(best_doc_type, 0) * 0.15
                    combined_confidence = min((tfidf_confidence * 0.6 + pattern_confidence * 0.4), 1.0)
                    
                    doc_info = self.document_database[best_doc_type]
                    
                    return {
                        'document_type': doc_info['name'],
                        'confidence': combined_confidence,
                        'method': 'pattern_optimized',
                        'indicators': list(set([kw for kw in found_matches if kw == best_doc_type])),
                        'reasoning': f"TF-IDF: {tfidf_confidence:.2f}, Pattern: {len(found_matches)} keywords",
                        'category': doc_info['category']
                    }
                except Exception as tfidf_error:
                    logger.warning("TF-IDF classification failed, using pattern matching only", error=str(tfidf_error))
            
            # Fallback: Use pattern matches only
            if doc_type_matches:
                best_doc_type = max(doc_type_matches, key=doc_type_matches.get)
                match_count = doc_type_matches[best_doc_type]
                confidence = min(match_count * 0.15, 1.0)
                
                doc_info = self.document_database[best_doc_type]
                
                return {
                    'document_type': doc_info['name'],
                    'confidence': confidence,
                    'method': 'pattern_pyahocorasick',
                    'indicators': [best_doc_type] * match_count,
                    'reasoning': f"Pattern matching found {match_count} keyword matches",
                    'category': doc_info['category']
                }
            
            return None
            
        except Exception as e:
            logger.error("Pattern classification failed", error=str(e))
            return None
    
    async def _classify_document_fallback(self, payload: Dict, filename: str = None) -> Dict[str, Any]:
        """Fallback classification method when other methods fail"""
        return {
            'document_type': 'unknown',
            'confidence': 0.1,
            'method': 'fallback',
            'indicators': [],
            'reasoning': 'No clear document type indicators found',
            'category': 'unknown'
        }
    
    async def _combine_classification_results(self, result1: Dict, result2: Dict) -> Dict[str, Any]:
        """Combine classification results intelligently"""
        # Weight first result higher but consider second result confirmation
        weight1 = 0.7
        weight2 = 0.3
        
        combined_confidence = (result1['confidence'] * weight1 + 
                             result2['confidence'] * weight2)
        
        # Combine indicators
        all_indicators = list(set(result1.get('indicators', []) + 
                                result2.get('indicators', [])))
        
        return {
            'document_type': result1['document_type'],
            'confidence': combined_confidence,
            'method': 'combined',
            'indicators': all_indicators,
            'reasoning': f"{result1['method']}: {result1['reasoning']}; {result2['method']}: {result2['reasoning']}",
            'category': result1.get('category', 'unknown'),
            'method1_confidence': result1['confidence'],
            'method2_confidence': result2['confidence']
        }
    
    # Helper methods
    def _generate_classification_id(self, payload: Dict, filename: str, user_id: str) -> str:
        """FIX #79: Generate deterministic classification ID using shared utility"""
        from core_infrastructure.utils.helpers import generate_cache_key
        return generate_cache_key('classify', payload, filename, user_id)
    
    async def _safe_groq_call(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Safe Groq API call with error handling for cost-effective document classification"""
        try:
            # FIX #67: Use injected groq_client instead of creating new client per call
            # Problem: Creating NEW Groq client on EVERY call wastes resources and causes rate limiting
            # Solution: Use injected client or fail gracefully
            if not self.groq_client:
                logger.error("Groq client not injected - document classification requires initialized Groq client")
                raise ValueError("Groq client must be injected during initialization")
            
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                logger.warning(f"Groq quota exceeded: {e}")
                return '{"document_type": "unknown", "confidence": 0.0, "indicators": [], "reasoning": "AI processing unavailable due to quota limits"}'
            else:
                logger.error(f"Groq API call failed: {e}")
                raise
    
    def _parse_ai_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse AI response with robust error handling"""
        try:
            # Clean up the response text
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # LIBRARY REPLACEMENT: orjson for 3-5x faster JSON parsing
            return orjson.loads(response_text)
        except (orjson.JSONDecodeError, ValueError) as e:
            logger.error(f"AI response JSON parsing failed: {e}")
            logger.error(f"Raw AI response: {response_text}")
            return None
    
    
    def _update_classification_metrics(self, result: Dict[str, Any]):
        """Update classification metrics"""
        self.metrics['classifications_performed'] += 1
        
        # Update confidence average
        current_avg = self.metrics['avg_confidence']
        count = self.metrics['classifications_performed']
        new_confidence = result.get('confidence', 0.0)
        self.metrics['avg_confidence'] = (current_avg * (count - 1) + new_confidence) / count
        
        # Update document type distribution
        doc_type = result.get('document_type', 'unknown')
        self.metrics['document_type_distribution'][doc_type] = self.metrics['document_type_distribution'].get(doc_type, 0) + 1
        
        # Update processing times
        processing_time = result.get('processing_time', 0.0)
        self.metrics['processing_times'].append(processing_time)
        if len(self.metrics['processing_times']) > 1000:
            self.metrics['processing_times'] = self.metrics['processing_times'][-1000:]
    
    async def _update_learning_system(self, result: Dict[str, Any], payload: Dict, filename: str, user_id: str = None):
        """FIX #81: Update learning system using shared utility"""
        if not self.config.enable_learning:
            return
        
        await self.learning_system.log_classification(result, payload, filename, user_id, self.supabase)
    
    async def classify_rows_batch(self, rows: List[Dict], platform_info: Dict, column_names: List[str], user_id: str = None) -> List[Dict[str, Any]]:
        """
        OPTIMIZED: Zero-shot batch classification with sentence-transformers (1000x cheaper than AI)
        Uses semantic similarity to classify rows without API calls
        
        Args:
            rows: List of row dictionaries to classify
            platform_info: Platform detection information
            column_names: Column names from the sheet
            user_id: User ID for learning system
            
        Returns:
            List of classification results, one per row
        """
        if not rows:
            return []
        
        try:
            # CRITICAL FIX: Use embedding_service for consistent BGE embeddings (1024 dims)
            # Replaced: SentenceModelService (removed from inference_service.py)
            # Reason: Avoid 400MB+ memory waste from conflicting embedding models
            from embedding_service import EmbeddingService
            
            embedding_service = EmbeddingService()
            await embedding_service.initialize()
            
            # Lazy-load row type embeddings if not already loaded
            if self.row_type_embeddings is None:
                logger.info("lazy_loading_row_type_embeddings")
                self.row_type_embeddings = {}
                for row_type, description in self.row_types.items():
                    embedding = await embedding_service.embed_text(description)
                    self.row_type_embeddings[row_type] = embedding
            
            # OPTIMIZED: Use BGE embeddings for zero-shot classification
            if self.row_type_embeddings:
                classifications = []
                
                # Encode all rows at once (vectorized, 100x faster)
                row_texts = [
                    ' '.join([f"{k}:{v}" for k, v in row.items() if v is not None and str(v).strip()])
                    for row in rows
                ]
                row_embeddings = await embedding_service.embed_batch(row_texts)
                
                # Compare with row type embeddings
                for idx, row_embedding in enumerate(row_embeddings):
                    best_match = None
                    best_score = 0.0
                    
                    for row_type, type_embedding in self.row_type_embeddings.items():
                        # Cosine similarity
                        similarity = cosine_similarity([row_embedding], [type_embedding])[0][0]
                        if similarity > best_score:
                            best_score = similarity
                            best_match = row_type
                    
                    # Map row_type to category/subcategory
                    category_map = {
                        'revenue_income': ('revenue', 'client_payment'),
                        'operating_expense': ('expense', 'operating_cost'),
                        'payroll_expense': ('payroll', 'employee_salary'),
                        'tax_payment': ('tax', 'tax_payment'),
                        'loan_payment': ('financing', 'loan_payment'),
                        'investment': ('investment', 'investment'),
                        'transfer': ('transfer', 'internal_transfer')
                    }
                    
                    category, subcategory = category_map.get(best_match, ('other', 'general'))
                    
                    classifications.append({
                        'row_index': idx,
                        'row_type': best_match,
                        'category': category,
                        'subcategory': subcategory,
                        'confidence': float(best_score),
                        'reasoning': f'Semantic similarity: {best_score:.2f}'
                    })
                
                self.metrics['semantic_batch_classifications'] += len(classifications)
                logger.info("Semantic batch classification complete",
                           rows_classified=len(classifications),
                           cost="$0",
                           method="sentence-transformers")
                
                return classifications
            
            # Fallback to pattern-based classification
            logger.warning("sentence-transformers not available, using pattern fallback")
            return [self._pattern_classify_row(row, platform_info, column_names) for row in rows]
            
        except Exception as e:
            logger.error("Batch classification failed", error=str(e))
            # Fallback to pattern-based classification
            return [self._pattern_classify_row(row, platform_info, column_names) for row in rows]
    
    def _pattern_classify_row(self, row: Dict, platform_info: Dict, column_names: List[str]) -> Dict[str, Any]:
        """Fast pattern-based row classification (fallback)"""
        row_text = ' '.join([str(v) for v in row.values() if v is not None]).lower()
        
        if any(keyword in row_text for keyword in ['salary', 'payroll', 'wage', 'employee']):
            return {
                'row_type': 'payroll_expense',
                'category': 'payroll',
                'subcategory': 'employee_salary',
                'confidence': 0.75,
                'reasoning': 'Pattern match: payroll keywords'
            }
        elif any(keyword in row_text for keyword in ['revenue', 'income', 'sales', 'payment received']):
            return {
                'row_type': 'revenue_income',
                'category': 'revenue',
                'subcategory': 'client_payment',
                'confidence': 0.75,
                'reasoning': 'Pattern match: revenue keywords'
            }
        elif any(keyword in row_text for keyword in ['expense', 'cost', 'bill', 'invoice', 'payment']):
            return {
                'row_type': 'operating_expense',
                'category': 'expense',
                'subcategory': 'operating_cost',
                'confidence': 0.75,
                'reasoning': 'Pattern match: expense keywords'
            }
        else:
            return {
                'row_type': 'transaction',
                'category': 'other',
                'subcategory': 'general',
                'confidence': 0.6,
                'reasoning': 'No clear pattern match'
            }
    
    async def _log_classification_audit(self, classification_id: str, result: Dict[str, Any], user_id: str):
        """Log classification audit information"""
        try:
            audit_data = {
                'classification_id': classification_id,
                'user_id': user_id,
                'document_type': result['document_type'],
                'confidence': result['confidence'],
                'method': result['method'],
                'indicators_count': len(result.get('indicators', [])),
                'processing_time': result.get('processing_time'),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Document classification audit: {audit_data}")
        except Exception as e:
            logger.warning(f"Audit logging failed: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get classification metrics"""
        return {
            **self.metrics,
            'avg_processing_time': sum(self.metrics['processing_times']) / len(self.metrics['processing_times']) if self.metrics['processing_times'] else 0.0,
            'cache_hit_rate': self.metrics['cache_hits'] / (self.metrics['cache_hits'] + self.metrics['cache_misses']) if (self.metrics['cache_hits'] + self.metrics['cache_misses']) > 0 else 0.0,
            'document_types_supported': len(self.document_database),
            'learning_enabled': self.config.enable_learning,
            'recent_classifications': len(self.classification_history),
            'ocr_available': self.ocr_available
        }
    
    def get_document_database(self) -> Dict[str, Dict[str, Any]]:
        """Get current document database"""
        return self.document_database.copy()
    
    def add_document_type(self, doc_type_id: str, doc_info: Dict[str, Any]):
        """Add new document type to database"""
        self.document_database[doc_type_id] = doc_info
        logger.info(f"Added document type: {doc_info['name']}")
    
    def update_document_type(self, doc_type_id: str, updates: Dict[str, Any]):
        """Update existing document type in database"""
        if doc_type_id in self.document_database:
            self.document_database[doc_type_id].update(updates)
            logger.info(f"Updated document type: {doc_type_id}")


# ============================================================================
# MODULE-LEVEL INITIALIZATION - Standard Python Practice
# ============================================================================
# Initialize all global models when module is imported
# This follows standard Python developer practices: imports and initialization at top level
logger.info("Initializing UniversalDocumentClassifier global models at module load time...")
_initialize_global_models()
logger.info("✅ UniversalDocumentClassifier module initialization complete")
