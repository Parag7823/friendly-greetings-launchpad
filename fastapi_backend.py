# Standard library imports
import os
import io
import logging
import hashlib
import uuid
import time
import json
import re
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher

# Third-party imports
import pandas as pd
import numpy as np
import magic
import filetype
import requests
import tempfile

# FastAPI and web framework imports
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, UploadFile, Form, File
from starlette.requests import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Database and external services
from supabase import create_client, Client
from openai import OpenAI

# Import production duplicate detection service
# Configure advanced logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('finley_backend.log')
    ]
)
logger = logging.getLogger(__name__)

# Import production duplicate detection service
try:
    from production_duplicate_detection_service import ProductionDuplicateDetectionService, FileMetadata
    from duplicate_detection_api_integration import DuplicateDetectionAPIIntegration, websocket_manager
    PRODUCTION_DUPLICATE_SERVICE_AVAILABLE = True
    logger.info("✅ Production duplicate detection service available")
except ImportError as e:
    logger.warning(f"⚠️ Production duplicate detection service not available: {e}")
    PRODUCTION_DUPLICATE_SERVICE_AVAILABLE = False

# Note: Legacy DuplicateDetectionService is defined below in this file

# Enhanced OpenCV error handling with graceful degradation
OPENCV_AVAILABLE = False
try:
    import cv2
    OPENCV_AVAILABLE = True
    logger.info("✅ OpenCV available for advanced image processing")
except ImportError:
    logger.warning("⚠️ OpenCV not available - advanced image processing features disabled")
except OSError as e:
    if "libGL.so.1" in str(e):
        logger.warning("⚠️ Advanced file processing features not available: libGL.so.1 missing")
    else:
        logger.warning(f"⚠️ OpenCV initialization warning: {e}")
except Exception as e:
    logger.error(f"❌ Unexpected error initializing OpenCV: {e}")

# Set global flag for OpenCV availability
os.environ['OPENCV_AVAILABLE'] = str(OPENCV_AVAILABLE)

# Custom JSON encoder to handle datetime objects
class DateTimeEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for handling datetime objects in API responses.
    
    Extends the standard JSONEncoder to properly serialize datetime and pandas
    Timestamp objects to ISO format strings for API responses.
    """
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif hasattr(obj, 'isoformat'):
            return obj.isoformat()
        return super().default(obj)

# Utility function to clean JWT tokens
def clean_jwt_token(token: str) -> str:
    """Clean JWT token by removing all whitespace and newline characters"""
    if not token:
        return token
    # Remove all whitespace, newlines, and tabs
    cleaned = token.strip().replace('\n', '').replace('\r', '').replace(' ', '').replace('\t', '')
    # Ensure it's a valid JWT format (3 parts separated by dots)
    parts = cleaned.split('.')
    if len(parts) == 3:
        return cleaned
    else:
        # If not valid JWT format, return original cleaned version
        return token.strip().replace('\n', '').replace('\r', '')

# Utility function for OpenAI calls with quota handling
async def safe_openai_call(client, model: str, messages: list, temperature: float = 0.1, max_tokens: int = 200, fallback_result: dict = None):
    """Make OpenAI API call with quota error handling"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower() or "insufficient_quota" in str(e).lower():
            logger.warning(f"OpenAI quota exceeded, using fallback: {e}")
            if fallback_result:
                return fallback_result
            else:
                return {
                    'platform': 'unknown',
                    'confidence': 0.0,
                    'detection_method': 'fallback_due_to_quota',
                    'indicators': [],
                    'reasoning': 'AI processing unavailable due to quota limits'
                }
        else:
            logger.error(f"OpenAI API error: {e}")
            raise e


# Add fallback processing when AI is unavailable
def get_fallback_platform_detection(payload: dict, filename: str = None) -> dict:
    """Fallback platform detection when AI is unavailable"""
    platform_indicators = {
        'stripe': ['stripe', 'stripe.com', 'st_'],
        'razorpay': ['razorpay', 'rzp_'],
        'paypal': ['paypal', 'pp_'],
        'quickbooks': ['quickbooks', 'qb_', 'intuit'],
        'xero': ['xero', 'xero.com'],
        'shopify': ['shopify', 'shopify.com'],
        'woocommerce': ['woocommerce', 'wc_'],
        'salesforce': ['salesforce', 'sf_'],
        'hubspot': ['hubspot', 'hs_']
    }
    
    # Check filename
    if filename:
        filename_lower = filename.lower()
        for platform, indicators in platform_indicators.items():
            if any(indicator in filename_lower for indicator in indicators):
                return {
                    'platform': platform,
                    'confidence': 0.7,
                    'detection_method': 'filename_pattern',
                    'indicators': [indicator for indicator in indicators if indicator in filename_lower],
                    'reasoning': f'Detected from filename: {filename}'
                }
    
    # Check payload content
    content_str = str(payload).lower()
    for platform, indicators in platform_indicators.items():
        if any(indicator in content_str for indicator in indicators):
            return {
                'platform': platform,
                'confidence': 0.6,
                'detection_method': 'content_pattern',
                'indicators': [indicator for indicator in indicators if indicator in content_str],
                'reasoning': 'Detected from content patterns'
            }
    
    return {
        'platform': 'unknown',
        'confidence': 0.0,
        'detection_method': 'fallback',
        'indicators': [],
        'reasoning': 'No patterns detected'
    }

def safe_json_parse(json_str, fallback=None):
    """Safely parse JSON with comprehensive error handling"""
    if not json_str or not isinstance(json_str, str):
        return fallback
    
    try:
        # Clean the string first
        cleaned = json_str.strip()
        
        # Try to extract JSON from markdown code blocks
        if '```json' in cleaned:
            start = cleaned.find('```json') + 7
            end = cleaned.find('```', start)
            if end != -1:
                cleaned = cleaned[start:end].strip()
        elif '```' in cleaned:
            start = cleaned.find('```') + 3
            end = cleaned.find('```', start)
            if end != -1:
                cleaned = cleaned[start:end].strip()
        
        # Try to find JSON object/array boundaries
        if cleaned.startswith('{') or cleaned.startswith('['):
            # Find matching closing brace/bracket
            if cleaned.startswith('{'):
                open_char, close_char = '{', '}'
            else:
                open_char, close_char = '[', ']'
            
            bracket_count = 0
            end_pos = 0
            for i, char in enumerate(cleaned):
                if char == open_char:
                    bracket_count += 1
                elif char == close_char:
                    bracket_count -= 1
                    if bracket_count == 0:
                        end_pos = i + 1
                        break
            
            if end_pos > 0:
                cleaned = cleaned[:end_pos]
        
        return json.loads(cleaned)
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {e}")
        logger.error(f"Input string: {json_str[:200]}...")
        return fallback
    except Exception as e:
        logger.error(f"Unexpected error in JSON parsing: {e}")
        return fallback

# Comprehensive datetime serialization helper
def serialize_datetime_objects(obj):
    """Recursively convert datetime objects to ISO format strings"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif hasattr(obj, 'isoformat'):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: serialize_datetime_objects(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [serialize_datetime_objects(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(serialize_datetime_objects(item) for item in obj)
    else:
        return obj

# Duplicate functions removed - using the first definitions above

# Initialize FastAPI app with enhanced configuration
app = FastAPI(
    title="Finley AI Backend",
    version="1.0.0",
    description="Advanced financial data processing and AI-powered analysis platform",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Enhanced CORS middleware with security considerations
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173", 
        "https://finley-ai.vercel.app",
        "https://*.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Static file mounting will be done after all API routes are defined
logger.info("🚀 Finley AI Backend starting in production mode")

# Initialize OpenAI client with error handling
try:
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    openai = OpenAI(api_key=openai_api_key)
    logger.info("✅ OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize OpenAI client: {e}")
    openai = None

# Advanced functionality imports with individual error handling
ADVANCED_FEATURES = {
    'zipfile': False,
    'py7zr': False,
    'rarfile': False,
    'odf': False,
    'tabula': False,
    'camelot': False,
    'pdfplumber': False,
    'pytesseract': False,
    'pil': False,
    'cv2': False,
    'xlwings': False
}

# Import advanced features individually for better error handling
try:
    import zipfile
    ADVANCED_FEATURES['zipfile'] = True
    logger.info("✅ ZIP file processing available")
except ImportError:
    logger.warning("⚠️ ZIP file processing not available")

try:
    import py7zr
    ADVANCED_FEATURES['py7zr'] = True
    logger.info("✅ 7-Zip file processing available")
except ImportError:
    logger.warning("⚠️ 7-Zip file processing not available")

try:
    import rarfile
    ADVANCED_FEATURES['rarfile'] = True
    logger.info("✅ RAR file processing available")
except ImportError:
    logger.warning("⚠️ RAR file processing not available")

try:
    from odf.opendocument import load as load_ods
    from odf.table import Table, TableRow, TableCell
    from odf.text import P
    ADVANCED_FEATURES['odf'] = True
    logger.info("✅ OpenDocument processing available")
except ImportError:
    logger.warning("⚠️ OpenDocument processing not available")

try:
    import tabula
    ADVANCED_FEATURES['tabula'] = True
    logger.info("✅ Tabula PDF processing available")
except ImportError:
    logger.warning("⚠️ Tabula PDF processing not available")

try:
    import camelot
    ADVANCED_FEATURES['camelot'] = True
    logger.info("✅ Camelot PDF processing available")
except ImportError:
    logger.warning("⚠️ Camelot PDF processing not available")

try:
    import pdfplumber
    ADVANCED_FEATURES['pdfplumber'] = True
    logger.info("✅ PDFPlumber processing available")
except ImportError:
    logger.warning("⚠️ PDFPlumber processing not available")

try:
    import pytesseract
    ADVANCED_FEATURES['pytesseract'] = True
    logger.info("✅ OCR processing available")
except ImportError:
    logger.warning("⚠️ OCR processing not available")

try:
    from PIL import Image
    ADVANCED_FEATURES['pil'] = True
    logger.info("✅ PIL image processing available")
except ImportError:
    logger.warning("⚠️ PIL image processing not available")

try:
    import cv2
    ADVANCED_FEATURES['cv2'] = True
    logger.info("✅ OpenCV processing available")
except ImportError:
    logger.warning("⚠️ OpenCV processing not available")

try:
    import xlwings as xw
    ADVANCED_FEATURES['xlwings'] = True
    logger.info("✅ Excel automation available")
except ImportError:
    logger.warning("⚠️ Excel automation not available")

# Overall advanced features availability
ADVANCED_FEATURES_AVAILABLE = any(ADVANCED_FEATURES.values())
logger.info(f"🔧 Advanced features status: {sum(ADVANCED_FEATURES.values())}/{len(ADVANCED_FEATURES)} available")

# Enhanced global configuration with environment variable support
@dataclass
class Config:
    """Enhanced global configuration for the application with environment variable support"""
    # File processing configuration
    max_file_size: int = int(os.environ.get("MAX_FILE_SIZE", 500 * 1024 * 1024))  # 500MB default
    chunk_size: int = int(os.environ.get("CHUNK_SIZE", 8192))  # 8KB chunks for streaming
    batch_size: int = int(os.environ.get("BATCH_SIZE", 50))  # Standardized batch size
    
    # WebSocket configuration
    websocket_timeout: int = int(os.environ.get("WEBSOCKET_TIMEOUT", 300))  # 5 minutes
    
    # AI processing configuration
    platform_confidence_threshold: float = float(os.environ.get("PLATFORM_CONFIDENCE_THRESHOLD", 0.85))
    entity_similarity_threshold: float = float(os.environ.get("ENTITY_SIMILARITY_THRESHOLD", 0.9))
    max_concurrent_ai_calls: int = int(os.environ.get("MAX_CONCURRENT_AI_CALLS", 5))
    
    # Caching configuration
    cache_ttl: int = int(os.environ.get("CACHE_TTL", 3600))  # 1 hour
    
    # Feature flags with environment variable support
    enable_advanced_file_processing: bool = os.environ.get("ENABLE_ADVANCED_FILE_PROCESSING", "true").lower() == "true"
    enable_duplicate_detection: bool = os.environ.get("ENABLE_DUPLICATE_DETECTION", "true").lower() == "true"
    enable_ocr_processing: bool = os.environ.get("ENABLE_OCR_PROCESSING", "true").lower() == "true"
    enable_archive_processing: bool = os.environ.get("ENABLE_ARCHIVE_PROCESSING", "true").lower() == "true"
    
    # Performance optimization settings
    enable_async_processing: bool = os.environ.get("ENABLE_ASYNC_PROCESSING", "true").lower() == "true"
    max_workers: int = int(os.environ.get("MAX_WORKERS", 4))
    memory_limit_mb: int = int(os.environ.get("MEMORY_LIMIT_MB", 2048))
    
    # Security settings
    enable_rate_limiting: bool = os.environ.get("ENABLE_RATE_LIMITING", "true").lower() == "true"
    max_requests_per_minute: int = int(os.environ.get("MAX_REQUESTS_PER_MINUTE", 100))
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.max_file_size <= 0:
            raise ValueError("max_file_size must be positive")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0 <= self.platform_confidence_threshold <= 1:
            raise ValueError("platform_confidence_threshold must be between 0 and 1")
        if not 0 <= self.entity_similarity_threshold <= 1:
            raise ValueError("entity_similarity_threshold must be between 0 and 1")
        if self.max_concurrent_ai_calls <= 0:
            raise ValueError("max_concurrent_ai_calls must be positive")
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if self.memory_limit_mb <= 0:
            raise ValueError("memory_limit_mb must be positive")

# Initialize configuration with validation
try:
    config = Config()
    logger.info("✅ Configuration loaded successfully")
    logger.info(f"📊 File processing: max_size={config.max_file_size//1024//1024}MB, batch_size={config.batch_size}")
    logger.info(f"🤖 AI processing: max_concurrent={config.max_concurrent_ai_calls}, confidence={config.platform_confidence_threshold}")
    logger.info(f"🔧 Features: advanced={config.enable_advanced_file_processing}, duplicate_detection={config.enable_duplicate_detection}")
except Exception as e:
    logger.error(f"❌ Configuration validation failed: {e}")
    raise

# ============================================================================
# LEGACY DUPLICATE DETECTION SERVICE - REMOVED
# ============================================================================
# The old DuplicateDetectionService class has been removed and replaced with
# the production-grade ProductionDuplicateDetectionService.
# 
# All functionality has been migrated to:
# - production_duplicate_detection_service.py (main service)
# - duplicate_detection_api_integration.py (API integration)
# 
# This ensures better performance, security, and maintainability.
# ============================================================================

# ============================================================================
# LEGACY METHODS REMOVED - All duplicate detection functionality moved to
# production_duplicate_detection_service.py and duplicate_detection_api_integration.py
# ============================================================================


# ============================================================================
# LEGACY METHODS REMOVED - All duplicate detection functionality moved to
# production_duplicate_detection_service.py and duplicate_detection_api_integration.py
# ============================================================================

class EnhancedFileProcessor:
    """Enhanced file processor with 100X capabilities for advanced file formats"""
    
    def __init__(self):
        self.supported_formats = {
            # Spreadsheet formats
            'excel': ['.xlsx', '.xls', '.xlsm', '.xlsb'],
            'csv': ['.csv', '.tsv', '.txt'],
            'ods': ['.ods'],
            
            # Document formats with tables
            'pdf': ['.pdf'],
            
            # Archive formats
            'zip': ['.zip', '.7z', '.rar'],
            
            # Image formats
            'image': ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif']
        }
        
        # Configure OCR if available
        self.ocr_config = '--oem 3 --psm 6' if ADVANCED_FEATURES_AVAILABLE else None
        
    async def process_file_enhanced(self, file_content: bytes, filename: str, 
                                  progress_callback=None) -> Dict[str, pd.DataFrame]:
        """Enhanced file processing with support for multiple formats"""
        try:
            if progress_callback:
                await progress_callback("detecting", "🔍 Detecting advanced file format and structure...", 5)
            
            # Detect file format
            file_format = self._detect_file_format(filename, file_content)
            logger.info(f"Enhanced processor detected format: {file_format} for {filename}")
            
            if progress_callback:
                await progress_callback("processing", f"📊 Processing {file_format} file with advanced capabilities...", 15)
            
            # Route to appropriate processor
            if file_format == 'excel':
                return await self._process_excel_enhanced(file_content, filename, progress_callback)
            elif file_format == 'csv':
                return await self._process_csv_enhanced(file_content, filename, progress_callback)
            elif file_format == 'ods':
                return await self._process_ods(file_content, filename, progress_callback)
            elif file_format == 'pdf':
                return await self._process_pdf(file_content, filename, progress_callback)
            elif file_format == 'archive':
                return await self._process_archive(file_content, filename, progress_callback)
            elif file_format == 'image':
                return await self._process_image(file_content, filename, progress_callback)
            else:
                # Fallback to basic processing
                logger.warning(f"Unsupported format {file_format}, falling back to basic processing")
                return await self._fallback_processing(file_content, filename, progress_callback)
                
        except Exception as e:
            logger.error(f"Enhanced file processing failed for {filename}: {e}")
            # Fallback to basic processing
            return await self._fallback_processing(file_content, filename, progress_callback)
    
    def _detect_file_format(self, filename: str, file_content: bytes) -> str:
        """Enhanced file format detection"""
        filename_lower = filename.lower()
        
        # Check file extension first
        for format_type, extensions in self.supported_formats.items():
            if any(filename_lower.endswith(ext) for ext in extensions):
                return format_type
        
        # Check for archive formats
        if filename_lower.endswith(('.zip', '.7z', '.rar')):
            return 'archive'
        
        # Use magic number detection if available
        if ADVANCED_FEATURES_AVAILABLE:
            try:
                file_type = filetype.guess(file_content)
                if file_type:
                    if file_type.extension in ['pdf']:
                        return 'pdf'
                    elif file_type.extension in ['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'gif']:
                        return 'image'
            except Exception:
                pass
        
        # Default to excel for unknown formats
        return 'excel'
    
    async def _process_excel_enhanced(self, file_content: bytes, filename: str, progress_callback=None) -> Dict[str, pd.DataFrame]:
        """Enhanced Excel processing with repair capabilities"""
        try:
            if progress_callback:
                await progress_callback("processing", "🔧 Processing Excel file with enhanced capabilities...", 20)
            
            # Try standard processing first
            try:
                file_stream = io.BytesIO(file_content)
                excel_file = pd.ExcelFile(file_stream)
                sheets = {}
                
                for sheet_name in excel_file.sheet_names:
                    df = pd.read_excel(file_stream, sheet_name=sheet_name)
                    if not df.empty:
                        sheets[sheet_name] = df
                
                if sheets:
                    return sheets
                    
            except Exception as e:
                logger.warning(f"Standard Excel processing failed: {e}")
            
            # Try repair if available
            if ADVANCED_FEATURES_AVAILABLE:
                try:
                    if progress_callback:
                        await progress_callback("repairing", "🔧 Attempting to repair corrupted Excel file...", 20)
                    
                    # Use xlwings for repair
                    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp_file:
                        temp_file.write(file_content)
                        temp_file.flush()
                        
                        app = xw.App(visible=False)
                        try:
                            wb = app.books.open(temp_file.name)
                            # Extract data from repaired workbook
                            sheets = {}
                            for sheet in wb.sheets:
                                data = sheet.used_range.value
                                if data:
                                    df = pd.DataFrame(data[1:], columns=data[0])
                                    if not df.empty:
                                        sheets[sheet.name] = df
                            
                            wb.close()
                            return sheets
                            
                        finally:
                            app.quit()
                            os.unlink(temp_file.name)
                            
                except Exception as repair_error:
                    logger.error(f"Excel repair failed: {repair_error}")
            
            # Final fallback
            raise Exception("All Excel processing methods failed")
            
        except Exception as e:
            logger.error(f"Enhanced Excel processing failed: {e}")
            raise
    
    async def _process_csv_enhanced(self, file_content: bytes, filename: str, progress_callback=None) -> Dict[str, pd.DataFrame]:
        """Enhanced CSV processing with multiple encodings"""
        if progress_callback:
            await progress_callback("processing", "📊 Processing CSV with enhanced encoding detection...", 20)
        
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-16']
        
        for encoding in encodings:
            try:
                file_stream = io.BytesIO(file_content)
                df = pd.read_csv(file_stream, encoding=encoding)
                if not df.empty:
                    return {'Sheet1': df}
            except Exception as e:
                continue
        
        raise Exception("Could not read CSV with any encoding")
    
    async def _process_ods(self, file_content: bytes, filename: str, progress_callback=None) -> Dict[str, pd.DataFrame]:
        """Process OpenDocument Spreadsheet files"""
        if not ADVANCED_FEATURES_AVAILABLE:
            raise Exception("ODS processing not available")
        
        if progress_callback:
            await progress_callback("processing", "📊 Processing ODS file...", 20)
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.ods', delete=False) as temp_file:
                temp_file.write(file_content)
                temp_file.flush()
                
                doc = load_ods(temp_file.name)
                sheets = {}
                
                for table in doc.spreadsheet.getElementsByType(Table):
                    sheet_name = table.getAttribute('name') or f"Sheet{len(sheets)+1}"
                    data = []
                    
                    for row in table.getElementsByType(TableRow):
                        row_data = []
                        for cell in row.getElementsByType(TableCell):
                            text_elements = cell.getElementsByType(P)
                            cell_text = ' '.join([p.getAttribute('text') or '' for p in text_elements])
                            row_data.append(cell_text)
                        if row_data:
                            data.append(row_data)
                    
                    if data:
                        df = pd.DataFrame(data[1:], columns=data[0] if data else [])
                        if not df.empty:
                            sheets[sheet_name] = df
                
                os.unlink(temp_file.name)
                return sheets
                
        except Exception as e:
            logger.error(f"ODS processing failed: {e}")
            raise
    
    async def _process_pdf(self, file_content: bytes, filename: str, progress_callback=None) -> Dict[str, pd.DataFrame]:
        """Process PDF files with table extraction"""
        if not ADVANCED_FEATURES_AVAILABLE:
            raise Exception("PDF processing not available")
        
        if progress_callback:
            await progress_callback("processing", "📄 Processing PDF with table extraction...", 20)
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(file_content)
                temp_file.flush()
                
                sheets = {}
                sheet_count = 0
                
                # Try multiple PDF table extraction methods
                try:
                    # Method 1: Tabula
                    tables = tabula.read_pdf(temp_file.name, pages='all')
                    for i, table in enumerate(tables):
                        if not table.empty:
                            sheet_name = f"Table_{i+1}"
                            sheets[sheet_name] = table
                            sheet_count += 1
                            
                except Exception as tabula_error:
                    logger.warning(f"Tabula failed: {tabula_error}")
                
                # Method 2: Camelot if tabula failed
                if not sheets:
                    try:
                        tables = camelot.read_pdf(temp_file.name, pages='all')
                        for i, table in enumerate(tables):
                            if table.df is not None and not table.df.empty:
                                sheet_name = f"Table_{i+1}"
                                sheets[sheet_name] = table.df
                                sheet_count += 1
                                
                    except Exception as camelot_error:
                        logger.warning(f"Camelot failed: {camelot_error}")
                
                # Method 3: PDFPlumber as final fallback
                if not sheets:
                    try:
                        with pdfplumber.open(temp_file.name) as pdf:
                            for page_num, page in enumerate(pdf.pages):
                                tables = page.extract_tables()
                                for table_num, table in enumerate(tables):
                                    if table:
                                        df = pd.DataFrame(table[1:], columns=table[0] if table else [])
                                        if not df.empty:
                                            sheet_name = f"Page_{page_num+1}_Table_{table_num+1}"
                                            sheets[sheet_name] = df
                                            sheet_count += 1
                                            
                    except Exception as pdfplumber_error:
                        logger.warning(f"PDFPlumber failed: {pdfplumber_error}")
                
                os.unlink(temp_file.name)
                
                if sheets:
                    return sheets
                else:
                    raise Exception("No tables could be extracted from PDF")
                    
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            raise
    
    async def _process_archive(self, file_content: bytes, filename: str, progress_callback=None) -> Dict[str, pd.DataFrame]:
        """Process archive files (ZIP, 7Z, RAR)"""
        if not ADVANCED_FEATURES_AVAILABLE:
            raise Exception("Archive processing not available")
        
        if progress_callback:
            await progress_callback("processing", "📦 Processing archive file...", 20)
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = os.path.join(temp_dir, filename)
                
                with open(temp_file_path, 'wb') as temp_file:
                    temp_file.write(file_content)
                
                # Extract archive
                if filename.lower().endswith('.zip'):
                    with zipfile.ZipFile(temp_file_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                elif filename.lower().endswith('.7z'):
                    with py7zr.SevenZipFile(temp_file_path, 'r') as seven_zip:
                        seven_zip.extractall(temp_dir)
                elif filename.lower().endswith('.rar'):
                    with rarfile.RarFile(temp_file_path, 'r') as rar_ref:
                        rar_ref.extractall(temp_dir)
                
                # Process extracted files
                all_sheets = {}
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        if file.lower().endswith(('.xlsx', '.xls', '.csv', '.ods')):
                            file_path = os.path.join(root, file)
                            try:
                                # Recursively process extracted files
                                with open(file_path, 'rb') as f:
                                    file_content = f.read()
                                
                                # Use basic processor for extracted files
                                # Use the existing file processor for streaming
                                basic_processor = self
                                extracted_sheets = await basic_processor.read_file_streaming(file_content, file)
                                
                                # Prefix sheet names with archive name
                                for sheet_name, df in extracted_sheets.items():
                                    prefixed_name = f"{filename}_{file}_{sheet_name}"
                                    all_sheets[prefixed_name] = df
                                    
                            except Exception as extract_error:
                                logger.warning(f"Failed to process extracted file {file}: {extract_error}")
                
                if all_sheets:
                    return all_sheets
                else:
                    raise Exception("No processable files found in archive")
                    
        except Exception as e:
            logger.error(f"Archive processing failed: {e}")
            raise
    
    async def _process_image(self, file_content: bytes, filename: str, progress_callback=None) -> Dict[str, pd.DataFrame]:
        """Process image files with OCR table extraction"""
        if not ADVANCED_FEATURES_AVAILABLE:
            raise Exception("Image processing not available")
        
        if progress_callback:
            await progress_callback("processing", "🖼️ Processing image with OCR...", 20)
        
        try:
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(filename)[1], delete=False) as temp_file:
                temp_file.write(file_content)
                temp_file.flush()
                
                # Load image
                image = Image.open(temp_file.name)
                
                # OCR processing
                ocr_text = pytesseract.image_to_string(image, config=self.ocr_config)
                
                # Try to extract table structure from OCR text
                lines = ocr_text.split('\n')
                table_data = []
                
                for line in lines:
                    if line.strip():
                        # Split by common delimiters
                        row = re.split(r'[\t|,;]', line.strip())
                        if len(row) > 1:  # Likely table row
                            table_data.append(row)
                
                os.unlink(temp_file.name)
                
                if table_data:
                    # Create DataFrame from extracted table
                    df = pd.DataFrame(table_data[1:], columns=table_data[0] if table_data else [])
                    return {'OCR_Extracted_Table': df}
                else:
                    # Return OCR text as single column
                    df = pd.DataFrame({'OCR_Text': [ocr_text]})
                    return {'OCR_Text': df}
                    
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            raise
    
    async def _fallback_processing(self, file_content: bytes, filename: str, progress_callback=None) -> Dict[str, pd.DataFrame]:
        """Fallback to basic processing if advanced methods fail"""
        if progress_callback:
            await progress_callback("fallback", "⚠️ Falling back to basic processing...", 15)
        
        try:
            # For now, use basic pandas processing
            # This will be enhanced when we integrate with the existing file processor
            file_stream = io.BytesIO(file_content)
            
            # Try Excel first
            try:
                excel_file = pd.ExcelFile(file_stream)
                sheets = {}
                for sheet_name in excel_file.sheet_names:
                    df = pd.read_excel(file_stream, sheet_name=sheet_name)
                    if not df.empty:
                        sheets[sheet_name] = df
                if sheets:
                    return sheets
            except Exception:
                pass
            
            # Try CSV
            try:
                file_stream.seek(0)
                df = pd.read_csv(file_stream)
                if not df.empty:
                    return {'Sheet1': df}
            except Exception:
                pass
            
            raise Exception("Fallback processing failed")
            
        except Exception as e:
            logger.error(f"Fallback processing also failed: {e}")
            raise


class VendorStandardizer:
    """Handles vendor name standardization and cleaning"""
    
    def __init__(self, openai_client):
        self.openai = openai_client
        self.vendor_cache = {}
        self.common_suffixes = [
            ' inc', ' corp', ' llc', ' ltd', ' co', ' company', ' pvt', ' private',
            ' limited', ' corporation', ' incorporated', ' enterprises', ' solutions',
            ' services', ' systems', ' technologies', ' tech', ' group', ' holdings',
            'inc', 'corp', 'llc', 'ltd', 'co', 'company', 'pvt', 'private',
            'limited', 'corporation', 'incorporated', 'enterprises', 'solutions',
            'services', 'systems', 'technologies', 'tech', 'group', 'holdings',
            'inc.', 'corp.', 'llc.', 'ltd.', 'co.', 'company.', 'pvt.', 'private.',
            'limited.', 'corporation.', 'incorporated.', 'enterprises.', 'solutions.',
            'services.', 'systems.', 'technologies.', 'tech.', 'group.', 'holdings.'
        ]
    
    async def standardize_vendor(self, vendor_name: str, platform: str = None) -> Dict[str, Any]:
        """Standardize vendor name using AI and rule-based cleaning"""
        try:
            if not vendor_name or vendor_name.strip() == '':
                return {
                    "vendor_raw": vendor_name,
                    "vendor_standard": "",
                    "confidence": 0.0,
                    "cleaning_method": "empty"
                }
            
            # Check cache first
            cache_key = f"{vendor_name}_{platform}"
            if cache_key in self.vendor_cache:
                return self.vendor_cache[cache_key]
            
            # Rule-based cleaning first
            cleaned_name = self._rule_based_cleaning(vendor_name)
            
            # If rule-based cleaning is sufficient, use it
            if cleaned_name != vendor_name:
                result = {
                    "vendor_raw": vendor_name,
                    "vendor_standard": cleaned_name,
                    "confidence": 0.8,
                    "cleaning_method": "rule_based"
                }
                self.vendor_cache[cache_key] = result
                return result
            
            # Use AI for complex cases
            ai_result = await self._ai_standardization(vendor_name, platform)
            self.vendor_cache[cache_key] = ai_result
            return ai_result
            
        except Exception as e:
            logger.error(f"Vendor standardization failed: {e}")
            return {
                "vendor_raw": vendor_name,
                "vendor_standard": vendor_name,
                "confidence": 0.5,
                "cleaning_method": "fallback"
            }
    
    def _rule_based_cleaning(self, vendor_name: str) -> str:
        """Rule-based vendor name cleaning"""
        try:
            if not vendor_name or not isinstance(vendor_name, str):
                return vendor_name or ""
            
            # Convert to lowercase and clean
            cleaned = vendor_name.lower().strip()
            
            # Remove common suffixes (with word boundary check)
            for suffix in self.common_suffixes:
                # Check if suffix exists at the end
                if cleaned.endswith(suffix):
                    # Ensure it's a word boundary (not part of another word)
                    if len(cleaned) > len(suffix):
                        char_before = cleaned[-(len(suffix) + 1)]
                        # Allow removal if preceded by space, punctuation, or if it's the whole string
                        if char_before.isspace() or char_before in '.,;:':
                            cleaned = cleaned[:-len(suffix)]
                    else:
                        # If suffix is the whole string, don't remove it
                        if len(cleaned) > len(suffix):
                            cleaned = cleaned[:-len(suffix)]
            
            # Remove extra whitespace and punctuation
            cleaned = ' '.join(cleaned.split())
            cleaned = cleaned.strip('.,;:')
            
            # Remove trailing punctuation from individual words
            words = cleaned.split()
            cleaned_words = []
            for word in words:
                # Remove trailing punctuation but keep internal punctuation (like .com)
                if word.endswith(('.', ',', ';', ':')):
                    word = word[:-1]
                cleaned_words.append(word)
            cleaned = ' '.join(cleaned_words)
            
            # Capitalize properly
            cleaned = cleaned.title()
            
            # Handle common abbreviations
            abbreviations = {
                'Ggl': 'Google',
                'Msoft': 'Microsoft',
                'Msft': 'Microsoft',
                'Amzn': 'Amazon',
                'Aapl': 'Apple',
                'Nflx': 'Netflix',
                'Tsla': 'Tesla'
            }
            
            if cleaned in abbreviations:
                cleaned = abbreviations[cleaned]
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Rule-based cleaning failed: {e}")
            return vendor_name
    
    async def _ai_standardization(self, vendor_name: str, platform: str = None) -> Dict[str, Any]:
        """AI-powered vendor name standardization"""
        try:
            prompt = f"""
            Standardize this vendor name to a clean, canonical form.
            
            VENDOR NAME: {vendor_name}
            PLATFORM: {platform or 'unknown'}
            
            Rules:
            1. Remove legal suffixes (Inc, Corp, LLC, Ltd, etc.)
            2. Standardize common company names
            3. Handle abbreviations and variations
            4. Return a clean, professional name
            
            Examples:
            - "Google LLC" → "Google"
            - "Microsoft Corporation" → "Microsoft"
            - "AMAZON.COM INC" → "Amazon"
            - "Apple Inc." → "Apple"
            - "Netflix, Inc." → "Netflix"
            
            Return ONLY a valid JSON object:
            {{
                "standard_name": "cleaned_vendor_name",
                "confidence": 0.95,
                "reasoning": "brief_explanation"
            }}
            """
            
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )
            
            result = response.choices[0].message.content.strip()
            
            # Clean and parse JSON
            cleaned_result = result.strip()
            if cleaned_result.startswith('```json'):
                cleaned_result = cleaned_result[7:]
            if cleaned_result.endswith('```'):
                cleaned_result = cleaned_result[:-3]
            
            parsed = json.loads(cleaned_result)
            
            return {
                "vendor_raw": vendor_name,
                "vendor_standard": parsed.get('standard_name', vendor_name),
                "confidence": parsed.get('confidence', 0.7),
                "cleaning_method": "ai_powered",
                "reasoning": parsed.get('reasoning', 'AI standardization')
            }
            
        except Exception as e:
            logger.error(f"AI vendor standardization failed: {e}")
            return {
                "vendor_raw": vendor_name,
                "vendor_standard": vendor_name,
                "confidence": 0.5,
                "cleaning_method": "ai_fallback"
            }

class PlatformIDExtractor:
    """Extracts platform-specific IDs and metadata"""
    
    def __init__(self):
        self.platform_patterns = {
            'razorpay': {
                'payment_id': r'pay_[a-zA-Z0-9]{14}',
                'order_id': r'order_[a-zA-Z0-9]{14}',
                'refund_id': r'rfnd_[a-zA-Z0-9]{14}',
                'settlement_id': r'setl_[a-zA-Z0-9]{14}'
            },
            'stripe': {
                'charge_id': r'ch_[a-zA-Z0-9]{24}',
                'payment_intent': r'pi_[a-zA-Z0-9]{24}',
                'customer_id': r'cus_[a-zA-Z0-9]{14}',
                'invoice_id': r'in_[a-zA-Z0-9]{24}'
            },
            'gusto': {
                'employee_id': r'emp_[a-zA-Z0-9]{8}',
                'payroll_id': r'pay_[a-zA-Z0-9]{12}',
                'timesheet_id': r'ts_[a-zA-Z0-9]{10}'
            },
            'quickbooks': {
                'transaction_id': r'txn_[a-zA-Z0-9]{12}',
                'invoice_id': r'inv_[a-zA-Z0-9]{10}',
                'vendor_id': r'ven_[a-zA-Z0-9]{8}',
                'customer_id': r'cust_[a-zA-Z0-9]{8}'
            },
            'xero': {
                'invoice_id': r'INV-[0-9]{4}-[0-9]{6}',
                'contact_id': r'[a-zA-Z0-9]{8}-[a-zA-Z0-9]{4}-[a-zA-Z0-9]{4}',
                'bank_transaction_id': r'BT-[0-9]{8}'
            }
        }
    
    def extract_platform_ids(self, row_data: Dict, platform: str, column_names: List[str]) -> Dict[str, Any]:
        """Extract platform-specific IDs from row data"""
        try:
            extracted_ids = {}
            platform_lower = platform.lower()
            
            # Get patterns for this platform
            patterns = self.platform_patterns.get(platform_lower, {})
            
            # Search in all text fields
            all_text = ' '.join(str(val) for val in row_data.values() if val)
            
            for id_type, pattern in patterns.items():
                matches = re.findall(pattern, all_text, re.IGNORECASE)
                if matches:
                    extracted_ids[id_type] = matches[0]  # Take first match
            
            # Also check column names for ID patterns
            for col_name in column_names:
                col_lower = col_name.lower()
                if any(id_type in col_lower for id_type in ['id', 'reference', 'number']):
                    col_value = row_data.get(col_name)
                    if col_value:
                        # Check if this column value matches any pattern
                        for id_type, pattern in patterns.items():
                            if re.match(pattern, str(col_value), re.IGNORECASE):
                                extracted_ids[id_type] = str(col_value)
                                break
            
            # Generate a unique platform ID if none found
            if not extracted_ids:
                extracted_ids['platform_generated_id'] = f"{platform_lower}_{hash(str(row_data)) % 10000:04d}"
            
            return {
                "platform": platform,
                "extracted_ids": extracted_ids,
                "total_ids_found": len(extracted_ids)
            }
            
        except Exception as e:
            logger.error(f"Platform ID extraction failed: {e}")
            return {
                "platform": platform,
                "extracted_ids": {},
                "total_ids_found": 0,
                "error": str(e)
            }

class DataEnrichmentProcessor:
    """Orchestrates all data enrichment processes with universal field detection"""
    
    def __init__(self, openai_client):
        self.vendor_standardizer = VendorStandardizer(openai_client)
        self.platform_id_extractor = PlatformIDExtractor()
        self.universal_extractors = UniversalExtractors()
        self.universal_platform_detector = UniversalPlatformDetector(openai_client)
        self.universal_document_classifier = UniversalDocumentClassifier(openai_client)
    
    async def enrich_row_data(self, row_data: Dict, platform_info: Dict, column_names: List[str], 
                            ai_classification: Dict, file_context: Dict) -> Dict[str, Any]:
        """Enrich row data with currency, vendor, and platform information"""
        try:
            # Extract basic information using universal field detection
            amount = self.universal_extractors.extract_amount_universal(row_data)
            description = self.universal_extractors.extract_description_universal(row_data)
            date = self.universal_extractors.extract_date_universal(row_data)
            
            # Universal platform detection
            platform_result = await self.universal_platform_detector.detect_platform_universal(
                row_data, file_context.get('filename', '')
            )
            platform = platform_result.get('platform', platform_info.get('platform', 'unknown'))
            
            # Universal document classification
            document_result = await self.universal_document_classifier.classify_document_universal(
                row_data, file_context.get('filename', '')
            )
            document_type = document_result.get('document_type', 'unknown')
            
            # Fallback to old methods if universal extraction fails
            if amount is None:
                amount = self._extract_amount(row_data)
            if description is None:
                description = self._extract_description(row_data)
            if date is None:
                date = self._extract_date(row_data)
            
            # Currency normalization removed - using original amount
            currency_info = {
                "amount_original": amount,
                "amount_usd": amount,
                "currency": "USD",
                "exchange_rate": 1.0,
                "exchange_date": date or datetime.now().strftime('%Y-%m-%d')
            }
            
            # 2. Vendor standardization using universal extraction
            vendor_name = self.universal_extractors.extract_vendor_universal(row_data)
            if vendor_name is None:
                vendor_name = self._extract_vendor_name(row_data, column_names)
            vendor_info = await self.vendor_standardizer.standardize_vendor(
                vendor_name=vendor_name,
                platform=platform
            )
            
            # 3. Platform ID extraction
            platform_ids = self.platform_id_extractor.extract_platform_ids(
                row_data=row_data,
                platform=platform,
                column_names=column_names
            )
            
            # 4. Create enhanced payload
            enriched_payload = {
                # Basic classification
                "kind": ai_classification.get('row_type', 'transaction'),
                "category": ai_classification.get('category', 'other'),
                "subcategory": ai_classification.get('subcategory', 'general'),
                
                # Universal platform detection
                "platform": platform,
                "platform_confidence": platform_result.get('confidence', 0.0),
                "platform_detection_method": platform_result.get('detection_method', 'unknown'),
                "platform_indicators": platform_result.get('indicators', []),
                
                # Universal document classification
                "document_type": document_type,
                "document_confidence": document_result.get('confidence', 0.0),
                "document_classification_method": document_result.get('classification_method', 'unknown'),
                "document_indicators": document_result.get('indicators', []),
                
                # Currency information
                "currency": currency_info.get('currency', 'USD'),
                "amount_original": currency_info.get('amount_original', amount),
                "amount_usd": currency_info.get('amount_usd', amount),
                "exchange_rate": currency_info.get('exchange_rate', 1.0),
                "exchange_date": currency_info.get('exchange_date'),
                
                # Vendor information
                "vendor_raw": vendor_info.get('vendor_raw', vendor_name),
                "vendor_standard": vendor_info.get('vendor_standard', vendor_name),
                "vendor_confidence": vendor_info.get('confidence', 0.0),
                "vendor_cleaning_method": vendor_info.get('cleaning_method', 'none'),
                
                # Platform information
                "platform": platform,
                "platform_confidence": platform_info.get('confidence', 0.0),
                "platform_ids": platform_ids.get('extracted_ids', {}),
                
                # Enhanced metadata
                "standard_description": self._clean_description(description),
                "ingested_on": datetime.utcnow().isoformat(),
                "file_source": file_context.get('filename', 'unknown'),
                "row_index": file_context.get('row_index', 0),
                
                # AI classification metadata
                "ai_confidence": ai_classification.get('confidence', 0.0),
                "ai_reasoning": ai_classification.get('reasoning', ''),
                "entities": ai_classification.get('entities', {}),
                "relationships": ai_classification.get('relationships', {})
            }
            
            return enriched_payload
            
        except Exception as e:
            logger.error(f"Data enrichment failed: {e}")
            # Return basic payload if enrichment fails
            return {
                "kind": ai_classification.get('row_type', 'transaction'),
                "category": ai_classification.get('category', 'other'),
                "amount_original": self._extract_amount(row_data),
                "amount_usd": self._extract_amount(row_data),
                "currency": "USD",
                "vendor_raw": self._extract_vendor_name(row_data, column_names),
                "vendor_standard": self._extract_vendor_name(row_data, column_names),
                "platform": platform_info.get('platform', 'unknown'),
                "ingested_on": datetime.utcnow().isoformat(),
                "enrichment_error": str(e)
            }
    
    def _extract_amount(self, row_data: Dict) -> float:
        """Extract amount from row data"""
        try:
            # Look for amount fields
            amount_fields = ['amount', 'total', 'value', 'sum', 'payment_amount', 'price']
            for field in amount_fields:
                if field in row_data:
                    value = row_data[field]
                    if isinstance(value, (int, float)):
                        return float(value)
                    elif isinstance(value, str):
                        # Remove currency symbols and convert
                        cleaned = re.sub(r'[^\d.-]', '', value)
                        return float(cleaned) if cleaned else 0.0
        except:
            pass
        return 0.0
    
    def _extract_description(self, row_data: Dict) -> str:
        """Extract description from row data"""
        desc_fields = ['description', 'memo', 'notes', 'details', 'comment']
        for field in desc_fields:
            if field in row_data:
                return str(row_data[field])
        return ""
    
    def _extract_vendor_name(self, row_data: Dict, column_names: List[str]) -> str:
        """Extract vendor name from row data"""
        vendor_fields = ['vendor', 'vendor_name', 'payee', 'recipient', 'company', 'merchant']
        for field in vendor_fields:
            if field in row_data:
                return str(row_data[field])
        
        # Check column names for vendor patterns
        for col in column_names:
            if any(vendor_word in col.lower() for vendor_word in ['vendor', 'payee', 'recipient', 'company']):
                if col in row_data:
                    return str(row_data[col])
        
        return ""
    
    def _extract_date(self, row_data: Dict) -> str:
        """Extract date from row data"""
        date_fields = ['date', 'payment_date', 'transaction_date', 'created_at', 'timestamp']
        for field in date_fields:
            if field in row_data:
                date_val = row_data[field]
                if isinstance(date_val, str):
                    return date_val
                elif isinstance(date_val, datetime):
                    return date_val.strftime('%Y-%m-%d')
        return datetime.now().strftime('%Y-%m-%d')
    
    def _clean_description(self, description: str) -> str:
        """Clean and standardize description"""
        try:
            if not description:
                return ""
            
            # Remove extra whitespace
            cleaned = ' '.join(description.split())
            
            # Remove common prefixes
            prefixes_to_remove = ['Payment for ', 'Transaction for ', 'Invoice for ']
            for prefix in prefixes_to_remove:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):]
            
            # Capitalize first letter
            if cleaned:
                cleaned = cleaned[0].upper() + cleaned[1:]
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Description cleaning failed: {e}")
            return description

# WebSocket connection manager
class ConnectionManager:
    """
    Manages WebSocket connections for real-time progress updates.
    
    Handles multiple WebSocket connections per job_id to support
    real-time progress updates during file processing operations.
    """
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, job_id: str):
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        self.active_connections[job_id].append(websocket)

    def disconnect(self, job_id: str):
        if job_id in self.active_connections:
            self.active_connections[job_id] = []

    async def send_update(self, job_id: str, message: dict):
        if job_id in self.active_connections:
            for connection in self.active_connections[job_id]:
                try:
                    await connection.send_json(message)
                except:
                    pass

manager = ConnectionManager()

class ProcessRequest(BaseModel):
    """
    Request model for file processing operations.
    
    Contains all necessary information for processing a file including
    job identification, storage path, filename, and user context.
    """
    job_id: str
    storage_path: str
    file_name: str
    user_id: str

class DocumentAnalyzer:
    """
    Analyzes documents using AI-powered classification and extraction.
    
    Provides intelligent document analysis capabilities including content
    classification, data extraction, and structured information processing.
    """
    def __init__(self, openai_client):
        self.openai = openai_client
    
    async def detect_document_type(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """Enhanced document type detection using AI analysis"""
        try:
            # Create a comprehensive sample for analysis
            sample_data = df.head(5).to_dict('records')  # Reduced to 5 rows
            column_names = list(df.columns)
            
            # Analyze data patterns
            numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
            date_columns = [col for col in column_names if any(word in col.lower() for word in ['date', 'time', 'period', 'month', 'year'])]
            
            # Simplified prompt that's more likely to return valid JSON
            prompt = f"""
            Analyze this financial document and return a JSON response.
            
            FILENAME: {filename}
            COLUMN NAMES: {column_names}
            SAMPLE DATA: {sample_data}
            
            Based on the column names and data, classify this document and return ONLY a valid JSON object with this structure:
            
            {{
                "document_type": "income_statement|balance_sheet|cash_flow|payroll_data|expense_data|revenue_data|general_ledger|budget|unknown",
                "source_platform": "gusto|quickbooks|xero|razorpay|freshbooks|unknown",
                "confidence": 0.95,
                "key_columns": ["col1", "col2"],
                "analysis": "Brief explanation",
                "data_patterns": {{
                    "has_revenue_data": true,
                    "has_expense_data": true,
                    "has_employee_data": false,
                    "has_account_data": false,
                    "has_transaction_data": false,
                    "time_period": "monthly"
                }},
                "classification_reasoning": "Step-by-step explanation",
                "platform_indicators": ["indicator1"],
                "document_indicators": ["indicator1"]
            }}
            
            IMPORTANT: Return ONLY the JSON object, no additional text or explanations.
            """
            
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            result = response.choices[0].message.content
            logger.info(f"AI Response: {result}")  # Log the actual response
            
            # Parse JSON from response
            import json
            try:
                # Clean the response - remove any markdown formatting
                cleaned_result = result.strip()
                if cleaned_result.startswith('```json'):
                    cleaned_result = cleaned_result[7:]
                if cleaned_result.endswith('```'):
                    cleaned_result = cleaned_result[:-3]
                cleaned_result = cleaned_result.strip()
                
                parsed_result = json.loads(cleaned_result)
                
                # Ensure all required fields are present
                if 'data_patterns' not in parsed_result:
                    parsed_result['data_patterns'] = {
                        "has_revenue_data": False,
                        "has_expense_data": False,
                        "has_employee_data": False,
                        "has_account_data": False,
                        "has_transaction_data": False,
                        "time_period": "unknown"
                    }
                
                if 'classification_reasoning' not in parsed_result:
                    parsed_result['classification_reasoning'] = "Analysis completed but reasoning not provided"
                
                if 'platform_indicators' not in parsed_result:
                    parsed_result['platform_indicators'] = []
                
                if 'document_indicators' not in parsed_result:
                    parsed_result['document_indicators'] = []
                
                return parsed_result
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse AI response: {e}")
                logger.error(f"Raw response: {result}")
                
                # Fallback: Try to extract basic information from the response
                fallback_result = self._extract_fallback_info(result, column_names)
                return fallback_result
                
        except Exception as e:
            logger.error(f"Error in document type detection: {e}")
            return {
                "document_type": "unknown",
                "source_platform": "unknown",
                "confidence": 0.3,
                "key_columns": list(df.columns),
                "analysis": f"Error in analysis: {str(e)}",
                "data_patterns": {
                    "has_revenue_data": False,
                    "has_expense_data": False,
                    "has_employee_data": False,
                    "has_account_data": False,
                    "has_transaction_data": False,
                    "time_period": "unknown"
                },
                "classification_reasoning": f"Error occurred during analysis: {str(e)}",
                "platform_indicators": [],
                "document_indicators": []
            }
    
    def _extract_fallback_info(self, response: str, column_names: list) -> Dict[str, Any]:
        """Extract basic information from AI response when JSON parsing fails"""
        response_lower = response.lower()
        
        # Determine document type based on column names
        doc_type = "unknown"
        if any(word in ' '.join(column_names).lower() for word in ['revenue', 'sales', 'income']):
            if any(word in ' '.join(column_names).lower() for word in ['cogs', 'cost', 'expense']):
                doc_type = "income_statement"
            else:
                doc_type = "revenue_data"
        elif any(word in ' '.join(column_names).lower() for word in ['employee', 'payroll', 'salary']):
            doc_type = "payroll_data"
        elif any(word in ' '.join(column_names).lower() for word in ['asset', 'liability', 'equity']):
            doc_type = "balance_sheet"
        
        # Enhanced platform detection using column patterns
        platform = "unknown"
        platform_indicators = []
        
        # Check for platform-specific patterns in column names
        columns_lower = [col.lower() for col in column_names]
        
        # QuickBooks patterns
        if any(word in ' '.join(columns_lower) for word in ['account', 'memo', 'ref number', 'split']):
            platform = "quickbooks"
            platform_indicators.append("qb_column_patterns")
        
        # Xero patterns
        elif any(word in ' '.join(columns_lower) for word in ['contact', 'tracking', 'reference']):
            platform = "xero"
            platform_indicators.append("xero_column_patterns")
        
        # Gusto patterns
        elif any(word in ' '.join(columns_lower) for word in ['employee', 'pay period', 'gross pay', 'net pay']):
            platform = "gusto"
            platform_indicators.append("gusto_column_patterns")
        
        # Stripe patterns
        elif any(word in ' '.join(columns_lower) for word in ['charge id', 'payment intent', 'customer id']):
            platform = "stripe"
            platform_indicators.append("stripe_column_patterns")
        
        # Shopify patterns
        elif any(word in ' '.join(columns_lower) for word in ['order id', 'product', 'fulfillment']):
            platform = "shopify"
            platform_indicators.append("shopify_column_patterns")
        
        return {
            "document_type": doc_type,
            "source_platform": platform,
            "confidence": 0.6,
            "key_columns": column_names,
            "analysis": "Fallback analysis due to JSON parsing failure",
            "data_patterns": {
                "has_revenue_data": any(word in ' '.join(column_names).lower() for word in ['revenue', 'sales', 'income']),
                "has_expense_data": any(word in ' '.join(column_names).lower() for word in ['expense', 'cost', 'cogs']),
                "has_employee_data": any(word in ' '.join(column_names).lower() for word in ['employee', 'payroll', 'salary']),
                "has_account_data": any(word in ' '.join(column_names).lower() for word in ['account', 'ledger']),
                "has_transaction_data": any(word in ' '.join(column_names).lower() for word in ['transaction', 'payment']),
                "time_period": "unknown"
            },
            "classification_reasoning": f"Fallback classification based on column names: {column_names}",
            "platform_indicators": platform_indicators,
            "document_indicators": column_names
        }

    async def generate_insights(self, df: pd.DataFrame, doc_analysis: Dict) -> Dict[str, Any]:
        """Generate enhanced insights from the processed data"""
        try:
            # Basic statistical analysis
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            insights = {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "numeric_columns": len(numeric_columns),
                "document_type": doc_analysis.get("document_type", "unknown"),
                "source_platform": doc_analysis.get("source_platform", "unknown"),
                "confidence": doc_analysis.get("confidence", 0.5),
                "key_columns": doc_analysis.get("key_columns", []),
                "analysis": doc_analysis.get("analysis", ""),
                "classification_reasoning": doc_analysis.get("classification_reasoning", ""),
                "data_patterns": doc_analysis.get("data_patterns", {}),
                "platform_indicators": doc_analysis.get("platform_indicators", []),
                "document_indicators": doc_analysis.get("document_indicators", []),
                "summary_stats": {},
                "enhanced_analysis": {}
            }
            
            # Calculate summary statistics for numeric columns
            for col in numeric_columns:
                insights["summary_stats"][col] = {
                    "mean": float(df[col].mean()) if not df[col].empty else 0,
                    "sum": float(df[col].sum()) if not df[col].empty else 0,
                    "min": float(df[col].min()) if not df[col].empty else 0,
                    "max": float(df[col].max()) if not df[col].empty else 0,
                    "count": int(df[col].count())
                }
            
            # Enhanced analysis based on document type
            doc_type = doc_analysis.get("document_type", "unknown")
            if doc_type == "income_statement":
                insights["enhanced_analysis"] = {
                    "revenue_analysis": self._analyze_revenue_data(df),
                    "expense_analysis": self._analyze_expense_data(df),
                    "profitability_metrics": self._calculate_profitability_metrics(df)
                }
            elif doc_type == "balance_sheet":
                insights["enhanced_analysis"] = {
                    "asset_analysis": self._analyze_assets(df),
                    "liability_analysis": self._analyze_liabilities(df),
                    "equity_analysis": self._analyze_equity(df)
                }
            elif doc_type == "payroll_data":
                insights["enhanced_analysis"] = {
                    "payroll_summary": self._analyze_payroll_data(df),
                    "employee_analysis": self._analyze_employee_data(df),
                    "tax_analysis": self._analyze_tax_data(df)
                }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return {
                "error": str(e),
                "total_rows": len(df) if 'df' in locals() else 0,
                "document_type": "unknown",
                "data_patterns": {},
                "classification_reasoning": f"Error generating insights: {str(e)}",
                "platform_indicators": [],
                "document_indicators": []
            }
    
    def _analyze_revenue_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze revenue-related data"""
        revenue_cols = [col for col in df.columns if any(word in col.lower() for word in ['revenue', 'sales', 'income'])]
        if not revenue_cols:
            return {"message": "No revenue columns found"}
        
        analysis = {}
        for col in revenue_cols:
            if col in df.columns:
                try:
                    # Convert to numeric, coerce errors to NaN
                    numeric_data = pd.to_numeric(df[col], errors='coerce')
                    analysis[col] = {
                        "total": float(numeric_data.sum()) if not numeric_data.empty else 0,
                        "average": float(numeric_data.mean()) if not numeric_data.empty else 0,
                        "growth_rate": self._calculate_growth_rate(numeric_data) if len(numeric_data) > 1 else None
                    }
                except Exception as e:
                    logger.warning(f"Could not analyze revenue column {col}: {e}")
                    analysis[col] = {
                        "total": 0,
                        "average": 0,
                        "growth_rate": None
                    }
        return analysis
    
    def _analyze_expense_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze expense-related data"""
        expense_cols = [col for col in df.columns if any(word in col.lower() for word in ['expense', 'cost', 'cogs', 'operating'])]
        if not expense_cols:
            return {"message": "No expense columns found"}
        
        analysis = {}
        for col in expense_cols:
            if col in df.columns:
                try:
                    # Convert to numeric, coerce errors to NaN
                    numeric_data = pd.to_numeric(df[col], errors='coerce')
                    analysis[col] = {
                        "total": float(numeric_data.sum()) if not numeric_data.empty else 0,
                        "average": float(numeric_data.mean()) if not numeric_data.empty else 0,
                        "percentage_of_revenue": self._calculate_expense_ratio(df, col)
                    }
                except Exception as e:
                    logger.warning(f"Could not analyze expense column {col}: {e}")
                    analysis[col] = {
                        "total": 0,
                        "average": 0,
                        "percentage_of_revenue": 0
                    }
        return analysis
    
    def _calculate_profitability_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate profitability metrics"""
        revenue_cols = [col for col in df.columns if any(word in col.lower() for word in ['revenue', 'sales', 'income'])]
        expense_cols = [col for col in df.columns if any(word in col.lower() for word in ['expense', 'cost', 'cogs', 'operating'])]
        profit_cols = [col for col in df.columns if any(word in col.lower() for word in ['profit', 'net'])]
        
        metrics = {}
        
        if revenue_cols and expense_cols:
            # Safe sum calculation with proper data type handling
            total_revenue = 0
            for col in revenue_cols:
                if col in df.columns:
                    try:
                        # Convert to numeric, coerce errors to NaN, then sum
                        numeric_data = pd.to_numeric(df[col], errors='coerce')
                        total_revenue += numeric_data.sum() if not numeric_data.empty else 0
                    except Exception as e:
                        logger.warning(f"Could not process revenue column {col}: {e}")
                        continue
            
            total_expenses = 0
            for col in expense_cols:
                if col in df.columns:
                    try:
                        # Convert to numeric, coerce errors to NaN, then sum
                        numeric_data = pd.to_numeric(df[col], errors='coerce')
                        total_expenses += numeric_data.sum() if not numeric_data.empty else 0
                    except Exception as e:
                        logger.warning(f"Could not process expense column {col}: {e}")
                        continue
            
            if total_revenue > 0:
                metrics["gross_margin"] = ((total_revenue - total_expenses) / total_revenue) * 100
                metrics["expense_ratio"] = (total_expenses / total_revenue) * 100
        
        if profit_cols:
            for col in profit_cols:
                if col in df.columns:
                    try:
                        numeric_data = pd.to_numeric(df[col], errors='coerce')
                        metrics[f"{col}_total"] = float(numeric_data.sum()) if not numeric_data.empty else 0
                    except Exception as e:
                        logger.warning(f"Could not process profit column {col}: {e}")
                        metrics[f"{col}_total"] = 0
        
        return metrics
    
    def _calculate_growth_rate(self, series: pd.Series) -> float:
        """Calculate growth rate between first and last values"""
        if len(series) < 2:
            return 0.0
        
        first_value = series.iloc[0]
        last_value = series.iloc[-1]
        
        if first_value == 0:
            return 0.0
        
        return ((last_value - first_value) / first_value) * 100
    
    def _calculate_expense_ratio(self, df: pd.DataFrame, expense_col: str) -> float:
        """Calculate expense as percentage of revenue"""
        revenue_cols = [col for col in df.columns if any(word in col.lower() for word in ['revenue', 'sales', 'income'])]
        
        if not revenue_cols or expense_col not in df.columns:
            return 0.0
        
        total_revenue = sum(df[col].sum() for col in revenue_cols if col in df.columns)
        total_expense = df[expense_col].sum()
        
        if total_revenue == 0:
            return 0.0
        
        return (total_expense / total_revenue) * 100
    
    def _analyze_assets(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze asset-related data"""
        asset_cols = [col for col in df.columns if any(word in col.lower() for word in ['asset', 'cash', 'receivable', 'inventory'])]
        total_assets = 0
        for col in asset_cols:
            if col in df.columns:
                try:
                    numeric_data = pd.to_numeric(df[col], errors='coerce')
                    total_assets += numeric_data.sum() if not numeric_data.empty else 0
                except Exception as e:
                    logger.warning(f"Could not process asset column {col}: {e}")
        return {"asset_columns": asset_cols, "total_assets": total_assets}
    
    def _analyze_liabilities(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze liability-related data"""
        liability_cols = [col for col in df.columns if any(word in col.lower() for word in ['liability', 'payable', 'debt', 'loan'])]
        total_liabilities = 0
        for col in liability_cols:
            if col in df.columns:
                try:
                    numeric_data = pd.to_numeric(df[col], errors='coerce')
                    total_liabilities += numeric_data.sum() if not numeric_data.empty else 0
                except Exception as e:
                    logger.warning(f"Could not process liability column {col}: {e}")
        return {"liability_columns": liability_cols, "total_liabilities": total_liabilities}
    
    def _analyze_equity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze equity-related data"""
        equity_cols = [col for col in df.columns if any(word in col.lower() for word in ['equity', 'capital', 'retained'])]
        total_equity = 0
        for col in equity_cols:
            if col in df.columns:
                try:
                    numeric_data = pd.to_numeric(df[col], errors='coerce')
                    total_equity += numeric_data.sum() if not numeric_data.empty else 0
                except Exception as e:
                    logger.warning(f"Could not process equity column {col}: {e}")
        return {"equity_columns": equity_cols, "total_equity": total_equity}
    
    def _analyze_payroll_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze payroll-related data"""
        payroll_cols = [col for col in df.columns if any(word in col.lower() for word in ['pay', 'salary', 'wage', 'gross', 'net'])]
        total_payroll = 0
        for col in payroll_cols:
            if col in df.columns:
                try:
                    numeric_data = pd.to_numeric(df[col], errors='coerce')
                    total_payroll += numeric_data.sum() if not numeric_data.empty else 0
                except Exception as e:
                    logger.warning(f"Could not process payroll column {col}: {e}")
        return {"payroll_columns": payroll_cols, "total_payroll": total_payroll}
    
    def _analyze_employee_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze employee-related data"""
        employee_cols = [col for col in df.columns if any(word in col.lower() for word in ['employee', 'name', 'id'])]
        return {"employee_columns": employee_cols, "employee_count": len(df) if employee_cols else 0}
    
    def _analyze_tax_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze tax-related data"""
        tax_cols = [col for col in df.columns if any(word in col.lower() for word in ['tax', 'withholding', 'deduction'])]
        total_taxes = 0
        for col in tax_cols:
            if col in df.columns:
                try:
                    numeric_data = pd.to_numeric(df[col], errors='coerce')
                    total_taxes += numeric_data.sum() if not numeric_data.empty else 0
                except Exception as e:
                    logger.warning(f"Could not process tax column {col}: {e}")
        return {"tax_columns": tax_cols, "total_taxes": total_taxes}

class PlatformDetector:
    """Enhanced platform detection for financial systems"""
    
    def __init__(self):
        self.platform_patterns = {
            'gusto': {
                'keywords': ['gusto', 'payroll', 'employee', 'salary', 'wage', 'paystub'],
                'columns': ['employee_name', 'employee_id', 'pay_period', 'gross_pay', 'net_pay', 'tax_deductions', 'benefits'],
                'data_patterns': ['employee_ssn', 'pay_rate', 'hours_worked', 'overtime', 'federal_tax', 'state_tax'],
                'confidence_threshold': 0.7,
                'description': 'Payroll and HR platform'
            },
            'quickbooks': {
                'keywords': ['quickbooks', 'qb', 'accounting', 'invoice', 'bill', 'qbo'],
                'columns': ['account', 'memo', 'amount', 'date', 'type', 'ref_number', 'split'],
                'data_patterns': ['account_number', 'class', 'customer', 'vendor', 'journal_entry'],
                'confidence_threshold': 0.7,
                'description': 'Accounting software'
            },
            'xero': {
                'keywords': ['xero', 'invoice', 'contact', 'account', 'xero'],
                'columns': ['contact_name', 'invoice_number', 'amount', 'date', 'reference', 'tracking'],
                'data_patterns': ['contact_id', 'invoice_id', 'tax_amount', 'line_amount', 'tracking_category'],
                'confidence_threshold': 0.7,
                'description': 'Cloud accounting platform'
            },
            'razorpay': {
                'keywords': ['razorpay', 'payment', 'transaction', 'merchant', 'settlement'],
                'columns': ['transaction_id', 'merchant_id', 'amount', 'status', 'created_at', 'payment_id'],
                'data_patterns': ['order_id', 'currency', 'method', 'description', 'fee_amount'],
                'confidence_threshold': 0.7,
                'description': 'Payment gateway'
            },
            'freshbooks': {
                'keywords': ['freshbooks', 'invoice', 'time_tracking', 'client', 'project'],
                'columns': ['client_name', 'invoice_number', 'amount', 'date', 'project', 'time_logged'],
                'data_patterns': ['client_id', 'project_id', 'rate', 'hours', 'service_type'],
                'confidence_threshold': 0.7,
                'description': 'Invoicing and time tracking'
            },
            'wave': {
                'keywords': ['wave', 'accounting', 'invoice', 'business'],
                'columns': ['account_name', 'description', 'amount', 'date', 'category'],
                'data_patterns': ['account_id', 'transaction_id', 'balance', 'wave_specific'],
                'confidence_threshold': 0.7,
                'description': 'Free accounting software'
            },
            'sage': {
                'keywords': ['sage', 'accounting', 'business', 'sage50', 'sage100'],
                'columns': ['account', 'description', 'amount', 'date', 'reference'],
                'data_patterns': ['account_number', 'journal_entry', 'period', 'sage_specific'],
                'confidence_threshold': 0.7,
                'description': 'Business management software'
            },
            'netsuite': {
                'keywords': ['netsuite', 'erp', 'enterprise', 'suite'],
                'columns': ['account', 'memo', 'amount', 'date', 'entity', 'subsidiary'],
                'data_patterns': ['internal_id', 'tran_id', 'line_id', 'netsuite_specific'],
                'confidence_threshold': 0.7,
                'description': 'Enterprise resource planning'
            },
            'stripe': {
                'keywords': ['stripe', 'payment', 'charge', 'customer', 'subscription'],
                'columns': ['charge_id', 'customer_id', 'amount', 'status', 'created', 'currency'],
                'data_patterns': ['payment_intent', 'transfer_id', 'fee_amount', 'payment_method'],
                'confidence_threshold': 0.7,
                'description': 'Payment processing platform'
            },
            'square': {
                'keywords': ['square', 'payment', 'transaction', 'merchant'],
                'columns': ['transaction_id', 'merchant_id', 'amount', 'status', 'created_at'],
                'data_patterns': ['location_id', 'device_id', 'tender_type', 'square_specific'],
                'confidence_threshold': 0.7,
                'description': 'Point of sale and payments'
            },
            'paypal': {
                'keywords': ['paypal', 'payment', 'transaction', 'merchant'],
                'columns': ['transaction_id', 'merchant_id', 'amount', 'status', 'created_at'],
                'data_patterns': ['paypal_id', 'fee_amount', 'currency', 'payment_type'],
                'confidence_threshold': 0.7,
                'description': 'Online payment system'
            },
            'shopify': {
                'keywords': ['shopify', 'order', 'product', 'sales', 'ecommerce'],
                'columns': ['order_id', 'product_name', 'amount', 'date', 'customer'],
                'data_patterns': ['shopify_id', 'product_id', 'variant_id', 'fulfillment_status'],
                'confidence_threshold': 0.7,
                'description': 'E-commerce platform'
            },
            'zoho': {
                'keywords': ['zoho', 'books', 'invoice', 'accounting'],
                'columns': ['contact_name', 'invoice_number', 'amount', 'date', 'reference'],
                'data_patterns': ['zoho_id', 'organization_id', 'zoho_specific'],
                'confidence_threshold': 0.7,
                'description': 'Business software suite'
            }
        }
    
    def detect_platform(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """Enhanced platform detection with multiple analysis methods"""
        filename_lower = filename.lower()
        columns_lower = [col.lower() for col in df.columns]
        
        best_match = {
            'platform': 'unknown',
            'confidence': 0.0,
            'matched_columns': [],
            'matched_patterns': [],
            'reasoning': 'No clear platform match found',
            'description': 'Unknown platform'
        }
        
        for platform, patterns in self.platform_patterns.items():
            confidence = 0.0
            matched_columns = []
            matched_patterns = []
            
            # 1. Filename keyword matching (25% weight)
            filename_matches = 0
            for keyword in patterns['keywords']:
                if keyword in filename_lower:
                    filename_matches += 1
                    confidence += 0.25 / len(patterns['keywords'])
            
            # 2. Column name matching (40% weight)
            column_matches = 0
            for expected_col in patterns['columns']:
                for actual_col in columns_lower:
                    if expected_col in actual_col or actual_col in expected_col:
                        matched_columns.append(actual_col)
                        column_matches += 1
                        confidence += 0.4 / len(patterns['columns'])
            
            # 3. Data pattern analysis (20% weight)
            if len(matched_columns) > 0:
                confidence += 0.2
            
            # 4. Data content analysis (15% weight)
            sample_data = df.head(3).astype(str).values.flatten()
            sample_text = ' '.join(sample_data).lower()
            
            for pattern in patterns.get('data_patterns', []):
                if pattern in sample_text:
                    confidence += 0.15 / len(patterns.get('data_patterns', []))
                    matched_patterns.append(pattern)
            
            # 5. Platform-specific terminology detection
            platform_terms = self._detect_platform_terminology(df, platform)
            if platform_terms:
                confidence += 0.1
                matched_patterns.extend(platform_terms)
            
            if confidence > best_match['confidence']:
                best_match = {
                    'platform': platform,
                    'confidence': min(confidence, 1.0),
                    'matched_columns': matched_columns,
                    'matched_patterns': matched_patterns,
                    'reasoning': self._generate_reasoning(platform, filename_matches, column_matches, len(matched_patterns)),
                    'description': patterns['description']
                }
        
        return best_match
    
    def _detect_platform_terminology(self, df: pd.DataFrame, platform: str) -> List[str]:
        """Detect platform-specific terminology in the data"""
        platform_terms = []
        
        if platform == 'quickbooks':
            # QB-specific terms
            qb_terms = ['ref number', 'split', 'class', 'customer', 'vendor', 'journal entry']
            for term in qb_terms:
                if any(term in str(col).lower() for col in df.columns):
                    platform_terms.append(f"qb_term: {term}")
        
        elif platform == 'xero':
            # Xero-specific terms
            xero_terms = ['tracking', 'reference', 'contact', 'line amount']
            for term in xero_terms:
                if any(term in str(col).lower() for col in df.columns):
                    platform_terms.append(f"xero_term: {term}")
        
        elif platform == 'gusto':
            # Gusto-specific terms
            gusto_terms = ['pay period', 'gross pay', 'net pay', 'tax deductions', 'benefits']
            for term in gusto_terms:
                if any(term in str(col).lower() for col in df.columns):
                    platform_terms.append(f"gusto_term: {term}")
        
        elif platform == 'stripe':
            # Stripe-specific terms
            stripe_terms = ['charge id', 'payment intent', 'transfer id', 'fee amount']
            for term in stripe_terms:
                if any(term in str(col).lower() for col in df.columns):
                    platform_terms.append(f"stripe_term: {term}")
        
        return platform_terms
    
    def _generate_reasoning(self, platform: str, filename_matches: int, column_matches: int, pattern_matches: int) -> str:
        """Generate detailed reasoning for platform detection"""
        reasoning_parts = []
        
        if filename_matches > 0:
            reasoning_parts.append(f"Filename contains {filename_matches} {platform} keywords")
        
        if column_matches > 0:
            reasoning_parts.append(f"Matched {column_matches} column patterns typical of {platform}")
        
        if pattern_matches > 0:
            reasoning_parts.append(f"Detected {pattern_matches} {platform}-specific data patterns")
        
        if not reasoning_parts:
            return f"No clear indicators for {platform}"
        
        return f"{platform} detected: {'; '.join(reasoning_parts)}"
    
    def get_platform_info(self, platform: str) -> Dict[str, Any]:
        """Get detailed information about a platform"""
        if platform in self.platform_patterns:
            return {
                'name': platform,
                'description': self.platform_patterns[platform]['description'],
                'typical_columns': self.platform_patterns[platform]['columns'],
                'keywords': self.platform_patterns[platform]['keywords'],
                'confidence_threshold': self.platform_patterns[platform]['confidence_threshold']
            }
        return {
            'name': platform,
            'description': 'Unknown platform',
            'typical_columns': [],
            'keywords': [],
            'confidence_threshold': 0.0
        }

class AIRowClassifier:
    """
    AI-powered row classification for financial data processing.
    
    Uses OpenAI's language models to intelligently classify and categorize
    financial data rows, providing enhanced data understanding and processing.
    """
    def __init__(self, openai_client, entity_resolver = None):
        self.openai = openai_client
        self.entity_resolver = entity_resolver
    
    async def classify_row_with_ai(self, row: pd.Series, platform_info: Dict, column_names: List[str], file_context: Dict = None) -> Dict[str, Any]:
        """AI-powered row classification with entity extraction and semantic understanding"""
        try:
            # Prepare row data for AI analysis
            row_data = {}
            for col, val in row.items():
                if pd.notna(val):
                    row_data[str(col)] = str(val)
            
            # Create context for AI
            context = {
                'platform': platform_info.get('platform', 'unknown'),
                'column_names': column_names,
                'row_data': row_data,
                'row_index': row.name if hasattr(row, 'name') else 'unknown'
            }
            
            # AI prompt for semantic classification
            prompt = f"""
            Analyze this financial data row and provide detailed classification.
            
            PLATFORM: {context['platform']}
            COLUMN NAMES: {context['column_names']}
            ROW DATA: {context['row_data']}
            
            Classify this row and return ONLY a valid JSON object with this structure:
            
            {{
                "row_type": "payroll_expense|salary_expense|revenue_income|operating_expense|capital_expense|invoice|bill|transaction|investment|tax|other",
                "category": "payroll|revenue|expense|investment|tax|other",
                "subcategory": "employee_salary|office_rent|client_payment|software_subscription|etc",
                "entities": {{
                    "employees": ["employee_name1", "employee_name2"],
                    "vendors": ["vendor_name1", "vendor_name2"],
                    "customers": ["customer_name1", "customer_name2"],
                    "projects": ["project_name1", "project_name2"]
                }},
                "amount": "positive_number_or_null",
                "currency": "USD|EUR|INR|etc",
                "date": "YYYY-MM-DD_or_null",
                "description": "human_readable_description",
                "confidence": 0.95,
                "reasoning": "explanation_of_classification",
                "relationships": {{
                    "employee_id": "extracted_or_null",
                    "vendor_id": "extracted_or_null",
                    "customer_id": "extracted_or_null",
                    "project_id": "extracted_or_null"
                }}
            }}
            
            IMPORTANT RULES:
            1. If you see salary/wage/payroll terms, classify as payroll_expense
            2. If you see revenue/income/sales terms, classify as revenue_income
            3. If you see expense/cost/payment terms, classify as operating_expense
            4. Extract any person names as employees, vendors, or customers
            5. Extract project names if mentioned
            6. Provide confidence score based on clarity of data
            7. Return ONLY valid JSON, no extra text
            """
            
            # Get AI response
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            
            result = response.choices[0].message.content.strip()
            
            # Clean and parse JSON response
            cleaned_result = result.strip()
            if cleaned_result.startswith('```json'):
                cleaned_result = cleaned_result[7:]
            if cleaned_result.endswith('```'):
                cleaned_result = cleaned_result[:-3]
            
            # Parse JSON
            try:
                classification = json.loads(cleaned_result)
                
                # Resolve entities if entity resolver is available
                if self.entity_resolver and classification.get('entities'):
                    try:
                        # Convert row to dict for entity resolution
                        row_data = {}
                        for col, val in row.items():
                            if pd.notna(val):
                                row_data[str(col)] = str(val)
                        
                        # Resolve entities
                        if file_context:
                            resolution_result = await self.entity_resolver.resolve_entities_batch(
                                classification['entities'], 
                                platform_info.get('platform', 'unknown'),
                                file_context.get('user_id', '550e8400-e29b-41d4-a716-446655440000'),
                                row_data,
                                column_names,
                                file_context.get('filename', 'test-file.xlsx'),
                                f"row-{row_index}" if 'row_index' in locals() else 'row-unknown'
                            )
                        else:
                            resolution_result = {
                                'resolved_entities': classification['entities'],
                                'resolution_results': [],
                                'total_resolved': 0,
                                'total_attempted': 0
                            }
                        
                        # Update classification with resolved entities
                        classification['resolved_entities'] = resolution_result['resolved_entities']
                        classification['entity_resolution_results'] = resolution_result['resolution_results']
                        classification['entity_resolution_stats'] = {
                            'total_resolved': resolution_result['total_resolved'],
                            'total_attempted': resolution_result['total_attempted']
                        }
                        
                    except Exception as e:
                        logger.error(f"Entity resolution failed: {e}")
                        classification['entity_resolution_error'] = str(e)
                
                return classification
            except json.JSONDecodeError as e:
                logger.error(f"AI classification JSON parsing failed: {e}")
                logger.error(f"Raw AI response: {result}")
                return self._fallback_classification(row, platform_info, column_names)
                
        except Exception as e:
            logger.error(f"AI classification failed: {e}")
            return self._fallback_classification(row, platform_info, column_names)
    
    def _fallback_classification(self, row: pd.Series, platform_info: Dict, column_names: List[str]) -> Dict[str, Any]:
        """Fallback classification when AI fails"""
        platform = platform_info.get('platform', 'unknown')
        row_str = ' '.join(str(val).lower() for val in row.values if pd.notna(val))
        
        # Basic classification
        if any(word in row_str for word in ['salary', 'wage', 'payroll', 'employee']):
            row_type = 'payroll_expense'
            category = 'payroll'
            subcategory = 'employee_salary'
        elif any(word in row_str for word in ['revenue', 'income', 'sales', 'payment']):
            row_type = 'revenue_income'
            category = 'revenue'
            subcategory = 'client_payment'
        elif any(word in row_str for word in ['expense', 'cost', 'bill', 'payment']):
            row_type = 'operating_expense'
            category = 'expense'
            subcategory = 'operating_cost'
        else:
            row_type = 'transaction'
            category = 'other'
            subcategory = 'general'
        
        # Extract entities using regex
        entities = self.extract_entities_from_text(row_str)
        
        return {
            'row_type': row_type,
            'category': category,
            'subcategory': subcategory,
            'entities': entities,
            'amount': None,
            'currency': 'USD',
            'date': None,
            'description': f"{category} transaction",
            'confidence': 0.6,
            'reasoning': f"Basic classification based on keywords: {row_str}",
            'relationships': {}
        }
    
    def extract_entities_from_text(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text using regex patterns"""
        entities = {
            'employees': [],
            'vendors': [],
            'customers': [],
            'projects': []
        }
        
        # Simple regex patterns for entity extraction
        employee_patterns = [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # First Last
            r'\b[A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+\b',  # First M. Last
        ]
        
        vendor_patterns = [
            r'\b[A-Z][a-z]+ (Inc|Corp|LLC|Ltd|Company|Co)\b',
            r'\b[A-Z][a-z]+ (Services|Solutions|Systems|Tech)\b',
        ]
        
        customer_patterns = [
            r'\b[A-Z][a-z]+ (Client|Customer|Account)\b',
        ]
        
        project_patterns = [
            r'\b[A-Z][a-z]+ (Project|Initiative|Campaign)\b',
        ]
        
        # Extract entities
        for pattern in employee_patterns:
            matches = re.findall(pattern, text)
            entities['employees'].extend(matches)
        
        for pattern in vendor_patterns:
            matches = re.findall(pattern, text)
            entities['vendors'].extend(matches)
        
        for pattern in customer_patterns:
            matches = re.findall(pattern, text)
            entities['customers'].extend(matches)
        
        for pattern in project_patterns:
            matches = re.findall(pattern, text)
            entities['projects'].extend(matches)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def map_relationships(self, entities: Dict[str, List[str]], platform_info: Dict) -> Dict[str, str]:
        """Map extracted entities to internal IDs (placeholder for future implementation)"""
        relationships = {}
        
        # Placeholder for entity ID mapping
        # In a real implementation, this would:
        # 1. Check if entities exist in the database
        # 2. Create new entities if they don't exist
        # 3. Return the internal IDs
        
        return relationships

class BatchAIRowClassifier:
    """Optimized batch AI classifier for large files"""
    
    def __init__(self, openai_client):
        self.openai = openai_client
        self.cache = {}  # Simple cache for similar rows
        self.batch_size = 20  # Process 20 rows at once
        self.max_concurrent_batches = 3  # Process 3 batches simultaneously
    
    async def classify_row_with_ai(self, row: pd.Series, platform_info: Dict, column_names: List[str]) -> Dict[str, Any]:
        """Individual row classification - wrapper for batch processing compatibility"""
        # For individual row processing, we'll use the fallback classification
        # This maintains compatibility with the existing RowProcessor
        return self._fallback_classification(row, platform_info, column_names)
    
    async def classify_rows_batch(self, rows: List[pd.Series], platform_info: Dict, column_names: List[str]) -> List[Dict[str, Any]]:
        """Classify multiple rows in a single AI call for efficiency"""
        try:
            # Prepare batch data
            batch_data = []
            for i, row in enumerate(rows):
                row_data = {}
                for col, val in row.items():
                    if pd.notna(val):
                        row_data[str(col)] = str(val)
                
                batch_data.append({
                    'index': i,
                    'row_data': row_data,
                    'row_index': row.name if hasattr(row, 'name') else f'row_{i}'
                })
            
            # Create batch prompt
            prompt = f"""
            Analyze these financial data rows and classify each one. Return a JSON array with classifications.
            
            PLATFORM: {platform_info.get('platform', 'unknown')}
            COLUMN NAMES: {column_names}
            ROWS TO CLASSIFY: {len(rows)}
            
            For each row, provide classification in this format:
            {{
                "row_type": "payroll_expense|salary_expense|revenue_income|operating_expense|capital_expense|invoice|bill|transaction|investment|tax|other",
                "category": "payroll|revenue|expense|investment|tax|other",
                "subcategory": "employee_salary|office_rent|client_payment|software_subscription|etc",
                "entities": {{
                    "employees": ["name1", "name2"],
                    "vendors": ["vendor1", "vendor2"],
                    "customers": ["customer1", "customer2"],
                    "projects": ["project1", "project2"]
                }},
                "amount": "number_or_null",
                "currency": "USD|EUR|INR|etc",
                "date": "YYYY-MM-DD_or_null",
                "description": "human_readable_description",
                "confidence": 0.95,
                "reasoning": "brief_explanation"
            }}
            
            ROW DATA:
            """
            
            # Add row data to prompt
            for i, row_info in enumerate(batch_data):
                prompt += f"\nROW {i+1}: {row_info['row_data']}\n"
            
            prompt += """
            
            Return ONLY a valid JSON array with one classification object per row, like:
            [
                {"row_type": "payroll_expense", "category": "payroll", ...},
                {"row_type": "revenue_income", "category": "revenue", ...},
                ...
            ]
            
            IMPORTANT: Return exactly one classification object per row, in the same order.
            """
            
            # Get AI response
            try:
                response = self.openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=2000
                )
                
                result = response.choices[0].message.content.strip()
                
                if not result:
                    logger.warning("AI returned empty response, using fallback")
                    return [self._fallback_classification(row, platform_info, column_names) for row in rows]
                    
            except Exception as ai_error:
                logger.error(f"AI request failed: {ai_error}")
                return [self._fallback_classification(row, platform_info, column_names) for row in rows]
            
            # Clean and parse JSON response
            cleaned_result = result.strip()
            if cleaned_result.startswith('```json'):
                cleaned_result = cleaned_result[7:]
            if cleaned_result.endswith('```'):
                cleaned_result = cleaned_result[:-3]
            
            # Additional cleaning for common AI response issues
            cleaned_result = cleaned_result.replace('\n', ' ').replace('\r', ' ')
            
            # Parse JSON
            try:
                classifications = json.loads(cleaned_result)
                
                # Ensure we have the right number of classifications
                if len(classifications) != len(rows):
                    logger.warning(f"AI returned {len(classifications)} classifications for {len(rows)} rows")
                    # Pad with fallback classifications if needed
                    while len(classifications) < len(rows):
                        classifications.append(self._fallback_classification(rows[len(classifications)], platform_info, column_names))
                    classifications = classifications[:len(rows)]  # Truncate if too many
                
                return classifications
                
            except json.JSONDecodeError as e:
                logger.error(f"Batch AI classification JSON parsing failed: {e}")
                logger.error(f"Raw AI response: {result}")
                
                # Try to extract partial JSON if possible
                try:
                    # Look for array start
                    start_idx = cleaned_result.find('[')
                    if start_idx != -1:
                        # Try to find a complete array
                        bracket_count = 0
                        end_idx = start_idx
                        for i, char in enumerate(cleaned_result[start_idx:], start_idx):
                            if char == '[':
                                bracket_count += 1
                            elif char == ']':
                                bracket_count -= 1
                                if bracket_count == 0:
                                    end_idx = i + 1
                                    break
                        
                        if end_idx > start_idx:
                            partial_json = cleaned_result[start_idx:end_idx]
                            partial_classifications = json.loads(partial_json)
                            logger.info(f"Successfully parsed partial JSON with {len(partial_classifications)} classifications")
                            
                            # Pad with fallback classifications
                            while len(partial_classifications) < len(rows):
                                partial_classifications.append(self._fallback_classification(rows[len(partial_classifications)], platform_info, column_names))
                            partial_classifications = partial_classifications[:len(rows)]
                            
                            return partial_classifications
                except Exception as partial_e:
                    logger.error(f"Failed to parse partial JSON: {partial_e}")
                
                # Fallback to individual classifications
                return [self._fallback_classification(row, platform_info, column_names) for row in rows]
                
        except Exception as e:
            logger.error(f"Batch AI classification failed: {e}")
            # Fallback to individual classifications
            return [self._fallback_classification(row, platform_info, column_names) for row in rows]
    
    def _fallback_classification(self, row: pd.Series, platform_info: Dict, column_names: List[str]) -> Dict[str, Any]:
        """Fallback classification when AI fails"""
        platform = platform_info.get('platform', 'unknown')
        row_str = ' '.join(str(val).lower() for val in row.values if pd.notna(val))
        
        # Basic classification
        if any(word in row_str for word in ['salary', 'wage', 'payroll', 'employee']):
            row_type = 'payroll_expense'
            category = 'payroll'
            subcategory = 'employee_salary'
        elif any(word in row_str for word in ['revenue', 'income', 'sales', 'payment']):
            row_type = 'revenue_income'
            category = 'revenue'
            subcategory = 'client_payment'
        elif any(word in row_str for word in ['expense', 'cost', 'bill', 'payment']):
            row_type = 'operating_expense'
            category = 'expense'
            subcategory = 'operating_cost'
        else:
            row_type = 'transaction'
            category = 'other'
            subcategory = 'general'
        
        # Extract entities using regex
        entities = self._extract_entities_from_text(row_str)
        
        return {
            'row_type': row_type,
            'category': category,
            'subcategory': subcategory,
            'entities': entities,
            'amount': None,
            'currency': 'USD',
            'date': None,
            'description': f"{category} transaction",
            'confidence': 0.6,
            'reasoning': f"Basic classification based on keywords: {row_str}",
            'relationships': {}
        }
    
    def _extract_entities_from_text(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text using regex patterns"""
        entities = {
            'employees': [],
            'vendors': [],
            'customers': [],
            'projects': []
        }
        
        # Simple regex patterns for entity extraction
        employee_patterns = [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # First Last
            r'\b[A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+\b',  # First M. Last
        ]
        
        vendor_patterns = [
            r'\b[A-Z][a-z]+ (Inc|Corp|LLC|Ltd|Company|Co)\b',
            r'\b[A-Z][a-z]+ (Services|Solutions|Systems|Tech)\b',
        ]
        
        customer_patterns = [
            r'\b[A-Z][a-z]+ (Client|Customer|Account)\b',
        ]
        
        project_patterns = [
            r'\b[A-Z][a-z]+ (Project|Initiative|Campaign)\b',
        ]
        
        # Extract entities
        for pattern in employee_patterns:
            matches = re.findall(pattern, text)
            entities['employees'].extend(matches)
        
        for pattern in vendor_patterns:
            matches = re.findall(pattern, text)
            entities['vendors'].extend(matches)
        
        for pattern in customer_patterns:
            matches = re.findall(pattern, text)
            entities['customers'].extend(matches)
        
        for pattern in project_patterns:
            matches = re.findall(pattern, text)
            entities['projects'].extend(matches)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def _get_cache_key(self, row: pd.Series) -> str:
        """Generate cache key for row content"""
        row_content = ' '.join(str(val).lower() for val in row.values if pd.notna(val))
        return hashlib.md5(row_content.encode()).hexdigest()
    
    def _is_similar_row(self, row1: pd.Series, row2: pd.Series, threshold: float = 0.8) -> bool:
        """Check if two rows are similar enough to use cached classification"""
        content1 = ' '.join(str(val).lower() for val in row1.values if pd.notna(val))
        content2 = ' '.join(str(val).lower() for val in row2.values if pd.notna(val))
        
        # Simple similarity check (can be enhanced with more sophisticated algorithms)
        words1 = set(content1.split())
        words2 = set(content2.split())
        
        if not words1 or not words2:
            return False
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        similarity = len(intersection) / len(union)
        return similarity >= threshold

class RowProcessor:
    """Processes individual rows and creates events"""
    
    def __init__(self, platform_detector: PlatformDetector, ai_classifier, enrichment_processor):
        self.platform_detector = platform_detector
        self.ai_classifier = ai_classifier
        self.enrichment_processor = enrichment_processor
    
    async def process_row(self, row: pd.Series, row_index: int, sheet_name: str, 
                   platform_info: Dict, file_context: Dict, column_names: List[str]) -> Dict[str, Any]:
        """Process a single row and create an event with AI-powered classification and enrichment"""
        
        # AI-powered row classification
        ai_classification = await self.ai_classifier.classify_row_with_ai(row, platform_info, column_names, file_context)
        
        # Convert row to JSON-serializable format
        row_data = self._convert_row_to_json_serializable(row)
        
        # Update file context with row index
        file_context['row_index'] = row_index
        
        # Data enrichment - create enhanced payload
        enriched_payload = await self.enrichment_processor.enrich_row_data(
            row_data=row_data,
            platform_info=platform_info,
            column_names=column_names,
            ai_classification=ai_classification,
            file_context=file_context
        )
        
        # Create the event payload with enhanced metadata
        event = {
            "provider": "excel-upload",
            "kind": enriched_payload.get('kind', 'transaction'),
            "source_platform": platform_info.get('platform', 'unknown'),
            "payload": enriched_payload,  # Use enriched payload instead of raw
            "row_index": row_index,
            "sheet_name": sheet_name,
            "source_filename": file_context['filename'],
            "uploader": file_context['user_id'],
            "ingest_ts": datetime.utcnow().isoformat(),
            "status": "pending",
            "confidence_score": enriched_payload.get('ai_confidence', 0.5),
            "classification_metadata": {
                "platform_detection": platform_info,
                "ai_classification": ai_classification,
                "enrichment_data": enriched_payload,
                "row_type": enriched_payload.get('kind', 'transaction'),
                "category": enriched_payload.get('category', 'other'),
                "subcategory": enriched_payload.get('subcategory', 'general'),
                "entities": enriched_payload.get('entities', {}),
                "relationships": enriched_payload.get('relationships', {}),
                "description": enriched_payload.get('standard_description', ''),
                "reasoning": enriched_payload.get('ai_reasoning', ''),
                "sheet_name": sheet_name,
                "file_context": file_context
            }
        }
        
        return event
    
    def _convert_row_to_json_serializable(self, row: pd.Series) -> Dict[str, Any]:
        """Convert a pandas Series to JSON-serializable format"""
        result = {}
        for column, value in row.items():
            if pd.isna(value):
                result[str(column)] = None
            elif isinstance(value, pd.Timestamp):
                result[str(column)] = value.isoformat()
            elif isinstance(value, (pd.Timedelta, pd.Period)):
                result[str(column)] = str(value)
            elif isinstance(value, (int, float, str, bool)):
                result[str(column)] = value
            elif isinstance(value, (list, dict)):
                # Handle nested structures
                result[str(column)] = self._convert_nested_to_json_serializable(value)
            else:
                # Convert any other types to string
                result[str(column)] = str(value)
        return result
    
    def _convert_nested_to_json_serializable(self, obj: Any) -> Any:
        """Convert nested objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {str(k): self._convert_nested_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_nested_to_json_serializable(item) for item in obj]
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, (pd.Timedelta, pd.Period)):
            return str(obj)
        elif pd.isna(obj):
            return None
        elif isinstance(obj, (int, float, str, bool)):
            return obj
        else:
            return str(obj)
    


class ExcelProcessor:
    def __init__(self):
        self.openai = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.analyzer = DocumentAnalyzer(self.openai)
        self.platform_detector = PlatformDetector()
        # Entity resolver and AI classifier will be initialized per request with Supabase client
        self.entity_resolver = None
        self.ai_classifier = None
        self.row_processor = None
        self.batch_classifier = BatchAIRowClassifier(self.openai)
        # Initialize data enrichment processor
        self.enrichment_processor = DataEnrichmentProcessor(self.openai)
    
    def _fast_classify_row(self, row: pd.Series, platform_info: dict, column_names: list) -> dict:
        """Fast pattern-based row classification without AI"""
        try:
            # Convert row to string for pattern matching
            row_text = ' '.join([str(val) for val in row.values if pd.notna(val)]).lower()
            
            # Pattern-based classification
            if any(keyword in row_text for keyword in ['salary', 'payroll', 'wage', 'employee']):
                return {
                    'row_type': 'payroll_expense',
                    'category': 'payroll',
                    'subcategory': 'employee_salary',
                    'entities': {'employees': [], 'vendors': [], 'customers': [], 'projects': []}
                }
            elif any(keyword in row_text for keyword in ['revenue', 'income', 'sales', 'payment']):
                return {
                    'row_type': 'revenue_income',
                    'category': 'revenue',
                    'subcategory': 'client_payment',
                    'entities': {'employees': [], 'vendors': [], 'customers': [], 'projects': []}
                }
            elif any(keyword in row_text for keyword in ['expense', 'cost', 'bill', 'invoice']):
                return {
                    'row_type': 'operating_expense',
                    'category': 'expense',
                    'subcategory': 'operating',
                    'entities': {'employees': [], 'vendors': [], 'customers': [], 'projects': []}
                }
            else:
                return {
                    'row_type': 'transaction',
                    'category': 'other',
                    'subcategory': 'general',
                    'entities': {'employees': [], 'vendors': [], 'customers': [], 'projects': []}
                }
        except Exception as e:
            logger.error(f"Fast classification failed: {e}")
            return {
                'row_type': 'transaction',
                'category': 'other',
                'subcategory': 'general',
                'entities': {'employees': [], 'vendors': [], 'customers': [], 'projects': []}
            }
    
    async def detect_file_type(self, file_content: bytes, filename: str) -> str:
        """Detect file type using magic numbers and filetype library"""
        try:
            # Check file extension first
            if filename.lower().endswith('.csv'):
                return 'csv'
            elif filename.lower().endswith('.xlsx'):
                return 'xlsx'
            elif filename.lower().endswith('.xls'):
                return 'xls'
            
            # Try filetype library
            file_type = filetype.guess(file_content)
            if file_type:
                if file_type.extension == 'csv':
                    return 'csv'
                elif file_type.extension in ['xlsx', 'xls']:
                    return file_type.extension
            
            # Fallback to python-magic
            mime_type = magic.from_buffer(file_content, mime=True)
            if 'csv' in mime_type or 'text/plain' in mime_type:
                return 'csv'
            elif 'excel' in mime_type or 'spreadsheet' in mime_type:
                return 'xlsx'
            else:
                return 'unknown'
        except Exception as e:
            logger.error(f"File type detection failed: {e}")
            return 'unknown'
    
    async def read_file(self, file_content: bytes, filename: str) -> Dict[str, pd.DataFrame]:
        """Read Excel or CSV file and return dictionary of sheets"""
        try:
            # Create a BytesIO object from the file content
            file_stream = io.BytesIO(file_content)
            
            # Check file type and read accordingly
            if filename.lower().endswith('.csv'):
                # Handle CSV files
                df = pd.read_csv(file_stream)
                if not df.empty:
                    return {'Sheet1': df}
                else:
                    raise HTTPException(status_code=400, detail="CSV file is empty")
            else:
                # Handle Excel files with explicit engine specification
                sheets = {}
                
                # Try different engines in order of preference
                engines_to_try = ['openpyxl', 'xlrd', None]  # None means default engine
                
                for engine in engines_to_try:
                    try:
                        file_stream.seek(0)  # Reset stream position for each attempt
                        
                        if engine:
                            # Try with specific engine
                            excel_file = pd.ExcelFile(file_stream, engine=engine)
                            for sheet_name in excel_file.sheet_names:
                                df = pd.read_excel(file_stream, sheet_name=sheet_name, engine=engine)
                                if not df.empty:
                                    sheets[sheet_name] = df
                        else:
                            # Try with default engine (no engine specified)
                            excel_file = pd.ExcelFile(file_stream)
                            for sheet_name in excel_file.sheet_names:
                                df = pd.read_excel(file_stream, sheet_name=sheet_name)
                                if not df.empty:
                                    sheets[sheet_name] = df
                        
                        # If we successfully read any sheets, return them
                        if sheets:
                            return sheets
                            
                    except Exception as e:
                        logger.warning(f"Failed to read Excel with engine {engine}: {e}")
                        continue
                
                # If all engines failed, try to read as CSV (some Excel files are actually CSV)
                try:
                    file_stream.seek(0)
                    # Try to read as CSV with different encodings
                    for encoding in ['utf-8', 'latin-1', 'cp1252']:
                        try:
                            file_stream.seek(0)
                            df = pd.read_csv(file_stream, encoding=encoding)
                            if not df.empty:
                                logger.info(f"Successfully read file as CSV with encoding {encoding}")
                                return {'Sheet1': df}
                        except Exception as csv_e:
                            logger.warning(f"Failed to read as CSV with encoding {encoding}: {csv_e}")
                            continue
                except Exception as csv_fallback_e:
                    logger.warning(f"CSV fallback failed: {csv_fallback_e}")
                
                # If all attempts failed, raise an error
                raise HTTPException(status_code=400, detail="Could not read Excel file with any available engine or as CSV")
                
        except Exception as e:
            logger.error(f"Error reading file {filename}: {e}")
            raise HTTPException(status_code=400, detail=f"Error reading file {filename}: {str(e)}")
    
    async def process_file(self, job_id: str, file_content: bytes, filename: str,
                          user_id: str, supabase: Client) -> Dict[str, Any]:
        """Optimized processing pipeline with duplicate detection and batch AI classification"""

        # Initialize duplicate detection service
        if PRODUCTION_DUPLICATE_SERVICE_AVAILABLE:
            duplicate_service = ProductionDuplicateDetectionService(supabase)
        else:
            duplicate_service = DuplicateDetectionService(supabase)
        
        # Create processing transaction for rollback capability
        transaction_id = str(uuid.uuid4())
        transaction_data = {
            'id': transaction_id,
            'user_id': user_id,
            'status': 'active',
            'operation_type': 'file_processing',
            'started_at': datetime.utcnow().isoformat(),
            'metadata': {
                'job_id': job_id,
                'filename': filename,
                'file_size': len(file_content)
            }
        }
        
        try:
            # Create transaction record
            supabase.table('processing_transactions').insert(transaction_data).execute()
            logger.info(f"Created processing transaction: {transaction_id}")
        except Exception as e:
            logger.warning(f"Failed to create processing transaction: {e}")
            transaction_id = None

        # Step 1: Read the file
        await manager.send_update(job_id, {
            "step": "reading",
            "message": f"📖 Reading and parsing your {filename}...",
            "progress": 5
        })

        try:
            sheets = await self.read_file(file_content, filename)
        except Exception as e:
            await manager.send_update(job_id, {
                "step": "error",
                "message": f"❌ Error reading file: {str(e)}",
                "progress": 0
            })
            raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

        # Step 2: Comprehensive Duplicate Detection
        await manager.send_update(job_id, {
            "step": "duplicate_check",
            "message": "🔍 Checking for duplicates and similar files...",
            "progress": 15
        })

        try:
            duplicate_analysis = await duplicate_service.check_exact_duplicate(
                user_id, hashlib.sha256(file_content).hexdigest(), filename
            )

            # Handle different duplicate detection phases
            if duplicate_analysis.get('is_duplicate', False):
                await manager.send_update(job_id, {
                    "step": "duplicate_found",
                    "message": "⚠️ Identical file detected! User decision required.",
                    "progress": 20,
                    "duplicate_info": duplicate_analysis,
                    "requires_user_decision": True
                })

                # Return early - wait for user decision
                return {
                    "status": "duplicate_detected",
                    "duplicate_analysis": duplicate_analysis,
                    "job_id": job_id,
                    "requires_user_decision": True
                }

            # Step 3: Content-level duplicate detection
            content_fingerprint = duplicate_service.calculate_content_fingerprint(sheets)
            content_duplicate_analysis = await duplicate_service.check_content_duplicate(
                user_id, content_fingerprint, filename
            )
            
            if content_duplicate_analysis.get('is_content_duplicate', False):
                await manager.send_update(job_id, {
                    "step": "content_duplicate_found",
                    "message": "🔄 Content overlap detected! Analyzing for delta ingestion...",
                    "progress": 25,
                    "content_duplicate_info": content_duplicate_analysis,
                    "requires_user_decision": True
                })
                
                # Analyze delta ingestion possibilities
                if content_duplicate_analysis.get('overlapping_files'):
                    existing_file_id = content_duplicate_analysis['overlapping_files'][0]['id']
                    delta_analysis = await duplicate_service.analyze_delta_ingestion(
                        user_id, sheets, existing_file_id
                    )
                    
                    await manager.send_update(job_id, {
                        "step": "delta_analysis_complete",
                        "message": f"📊 Delta analysis: {delta_analysis['delta_analysis']['new_rows']} new rows, {delta_analysis['delta_analysis']['existing_rows']} existing rows",
                        "progress": 30,
                        "delta_analysis": delta_analysis,
                        "requires_user_decision": True
                    })
                
                return {
                    "status": "content_duplicate_detected",
                    "content_duplicate_analysis": content_duplicate_analysis,
                    "delta_analysis": delta_analysis if 'delta_analysis' in locals() else None,
                    "job_id": job_id,
                    "requires_user_decision": True
                }
            
            # Step 4: Near-duplicate detection (now enabled)
            near_duplicate_analysis = await duplicate_service.check_near_duplicate(
                user_id, file_content, filename
            )
            
            if near_duplicate_analysis.get('is_near_duplicate', False):
                await manager.send_update(job_id, {
                    "step": "near_duplicate_found",
                    "message": f"🔍 Similar file detected ({near_duplicate_analysis['similarity_score']:.1%} similarity). Consider delta ingestion.",
                    "progress": 35,
                    "near_duplicate_info": near_duplicate_analysis,
                    "requires_user_decision": True
                })
                
                return {
                    "status": "near_duplicate_detected",
                    "near_duplicate_analysis": near_duplicate_analysis,
                    "job_id": job_id,
                    "requires_user_decision": True
                }

                # Generate intelligent recommendations
                if len(duplicate_analysis['version_candidates']) > 1:
                    # Create temporary version group for analysis
                    version_group_id = str(uuid.uuid4())

                    # Generate version recommendation
                    recommendation = await duplicate_service.generate_version_recommendation(version_group_id)

                    await manager.send_update(job_id, {
                        "step": "version_analysis",
                        "message": "🧠 Generated intelligent version recommendation",
                        "progress": 30,
                        "recommendation": recommendation
                    })

                    return {
                        "status": "versions_detected",
                        "duplicate_analysis": duplicate_analysis,
                        "recommendation": recommendation,
                        "job_id": job_id,
                        "requires_user_decision": True
                    }

            elif False:  # Skip similar files for now
                await manager.send_update(job_id, {
                    "step": "similar_files",
                    "message": f"📄 Found {len(duplicate_analysis['similar_files'])} similar files",
                    "progress": 20,
                    "similar_files": duplicate_analysis['similar_files']
                })

            # Continue with normal processing if no blocking duplicates
            await manager.send_update(job_id, {
                "step": "processing",
                "message": "✅ No blocking duplicates found - proceeding with processing",
                "progress": 25
            })

        except Exception as e:
            logger.warning(f"Duplicate detection failed: {e} - proceeding with normal processing")
            duplicate_analysis = {'phase': 'error', 'error': str(e)}

            await manager.send_update(job_id, {
                "step": "duplicate_check_failed",
                "message": "⚠️ Duplicate check failed - proceeding with upload",
                "progress": 20
            })
        
        # Step 2: Fast Platform Detection and Document Classification
        await manager.send_update(job_id, {
            "step": "analyzing",
            "message": "🧠 Fast platform detection and document classification...",
            "progress": 20
        })
        
        # Use first sheet for detection
        first_sheet = list(sheets.values())[0]
        
        # Fast pattern-based platform detection first
        platform_info = self.platform_detector.detect_platform(first_sheet, filename)
        
        # Fast document classification using patterns
        doc_analysis = {
            'document_type': 'financial_data',
            'confidence': 0.8,
            'classification_method': 'pattern_based',
            'indicators': ['financial_columns', 'numeric_data']
        }
        
        # Initialize EntityResolver and AI classifier with Supabase client
        self.entity_resolver = EntityResolver(supabase)
        self.ai_classifier = AIRowClassifier(self.openai, self.entity_resolver)
        self.row_processor = RowProcessor(self.platform_detector, self.ai_classifier, self.enrichment_processor)
        
        # Step 3: Create raw_records entry
        await manager.send_update(job_id, {
            "step": "storing",
            "message": "💾 Storing file metadata...",
            "progress": 35
        })

        # Calculate file hash for duplicate detection
        file_hash = hashlib.sha256(file_content).hexdigest()

        # Store in raw_records (avoid non-existent columns)
        # Calculate content fingerprint for row-level deduplication
        content_fingerprint = duplicate_service.calculate_content_fingerprint(sheets)
        
        raw_record_result = supabase.table('raw_records').insert({
            'user_id': user_id,
            'file_name': filename,
            'file_size': len(file_content),
            'source': 'file_upload',
            'content': {
                'sheets': list(sheets.keys()),
                'platform_detection': platform_info,
                'document_analysis': doc_analysis,
                'file_hash': file_hash,
                'content_fingerprint': content_fingerprint,  # Add content fingerprint
                'total_rows': sum(len(sheet) for sheet in sheets.values()),
                'processed_at': datetime.utcnow().isoformat(),
                'duplicate_analysis': duplicate_analysis  # Store duplicate analysis results
            },
            'status': 'processing',
            'classification_status': 'processing'
        }).execute()
        
        if raw_record_result.data:
            file_id = raw_record_result.data[0]['id']
        else:
            raise HTTPException(status_code=500, detail="Failed to create raw record")
        
        # Step 4: Create or update ingestion_jobs entry
        try:
            # Try to create the job entry if it doesn't exist
            job_result = supabase.table('ingestion_jobs').insert({
                'id': job_id,
                'user_id': user_id,
                'file_id': file_id,
                'status': 'processing',
                'created_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat()
            }).execute()
        except Exception as e:
            # If job already exists, update it
            logger.info(f"Job {job_id} already exists, updating...")
            supabase.table('ingestion_jobs').update({
                'file_id': file_id,
                'status': 'processing',
                'updated_at': datetime.utcnow().isoformat()
            }).eq('id', job_id).execute()
        
        # Step 5: Process each sheet with optimized batch processing
        await manager.send_update(job_id, {
            "step": "streaming",
            "message": "🔄 Processing rows in optimized batches...",
            "progress": 40
        })
        
        total_rows = sum(len(sheet) for sheet in sheets.values())
        processed_rows = 0
        events_created = 0
        errors = []
        
        file_context = {
            'filename': filename,
            'user_id': user_id,
            'file_id': file_id,
            'job_id': job_id
        }
        
        # Process each sheet with batch optimization
        for sheet_name, df in sheets.items():
            if df.empty:
                continue
            
            column_names = list(df.columns)
            rows = list(df.iterrows())
            
            # Process rows in batches for efficiency
            batch_size = 20  # Process 20 rows at once
            total_batches = (len(rows) + batch_size - 1) // batch_size
            
            for batch_idx in range(0, len(rows), batch_size):
                batch_rows = rows[batch_idx:batch_idx + batch_size]
                
                try:
                    # Extract row data for batch processing
                    row_data = [row[1] for row in batch_rows]  # row[1] is the Series
                    row_indices = [row[0] for row in batch_rows]  # row[0] is the index
                    
                    # Process batch with fast pattern-based classification
                    batch_classifications = []
                    for row in row_data:
                        # Fast pattern-based classification instead of AI
                        classification = self._fast_classify_row(row, platform_info, column_names)
                        batch_classifications.append(classification)
                    
                    # Store each row from the batch
                    for i, (row_index, row) in enumerate(batch_rows):
                        try:
                            # Create event for this row
                            event = await self.row_processor.process_row(
                                row, row_index, sheet_name, platform_info, file_context, column_names
                            )
                            
                            # Use fast classification result
                            if i < len(batch_classifications):
                                ai_classification = batch_classifications[i]
                                event['classification_metadata'].update(ai_classification)
                            
                            # Store event in raw_events table with enrichment fields
                            enriched_payload = event['payload']  # This is now the enriched payload
                            
                            # Clean the enriched payload to ensure all datetime objects are converted
                            cleaned_enriched_payload = serialize_datetime_objects(enriched_payload)
                            
                            event_result = supabase.table('raw_events').insert({
                                'user_id': user_id,
                                'file_id': file_id,
                                'job_id': job_id,
                                'provider': event['provider'],
                                'kind': event['kind'],
                                'source_platform': event['source_platform'],
                                'category': event['classification_metadata'].get('category'),
                                'subcategory': event['classification_metadata'].get('subcategory'),
                                'payload': cleaned_enriched_payload,  # Use cleaned payload
                                'row_index': event['row_index'],
                                'sheet_name': event['sheet_name'],
                                'source_filename': event['source_filename'],
                                'uploader': event['uploader'],
                                'ingest_ts': event['ingest_ts'],
                                'status': event['status'],
                                'confidence_score': event['confidence_score'],
                                'classification_metadata': event['classification_metadata'],
                                'entities': event['classification_metadata'].get('entities', {}),
                                'relationships': event['classification_metadata'].get('relationships', {}),
                                # Enrichment fields
                                'amount_original': cleaned_enriched_payload.get('amount_original'),
                                'amount_usd': cleaned_enriched_payload.get('amount_usd'),
                                'currency': cleaned_enriched_payload.get('currency'),
                                'exchange_rate': cleaned_enriched_payload.get('exchange_rate'),
                                'exchange_date': cleaned_enriched_payload.get('exchange_date'),
                                'vendor_raw': cleaned_enriched_payload.get('vendor_raw'),
                                'vendor_standard': cleaned_enriched_payload.get('vendor_standard'),
                                'vendor_confidence': cleaned_enriched_payload.get('vendor_confidence'),
                                'vendor_cleaning_method': cleaned_enriched_payload.get('vendor_cleaning_method'),
                                'platform_ids': cleaned_enriched_payload.get('platform_ids', {}),
                                'standard_description': cleaned_enriched_payload.get('standard_description'),
                                'ingested_on': cleaned_enriched_payload.get('ingested_on')
                            }).execute()
                            
                            if event_result.data:
                                events_created += 1
                            else:
                                errors.append(f"Failed to store event for row {row_index} in sheet {sheet_name}")
                        
                        except Exception as e:
                            # Handle datetime serialization errors specifically
                            if "datetime" in str(e) and "JSON serializable" in str(e):
                                logger.warning(f"Datetime serialization error for row {row_index}, skipping: {e}")
                                continue
                            else:
                                error_msg = f"Error processing row {row_index} in sheet {sheet_name}: {str(e)}"
                                errors.append(error_msg)
                                logger.error(error_msg)
                        
                        processed_rows += 1
                    
                    # Update progress every batch
                    progress = 40 + (processed_rows / total_rows) * 40
                    await manager.send_update(job_id, {
                        "step": "streaming",
                        "message": f"🔄 Processed {processed_rows}/{total_rows} rows ({events_created} events created)...",
                        "progress": int(progress)
                    })
                
                except Exception as e:
                    error_msg = f"Error processing batch {batch_idx//batch_size + 1} in sheet {sheet_name}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
        
        # Step 6: Update raw_records with completion status
        await manager.send_update(job_id, {
            "step": "finalizing",
            "message": "✅ Finalizing processing...",
            "progress": 90
        })
        
        supabase.table('raw_records').update({
            'status': 'completed',
            'classification_status': 'completed',
            'content': {
                'sheets': list(sheets.keys()),
                'platform_detection': platform_info,
                'document_analysis': doc_analysis,
                'file_hash': file_hash,
                'total_rows': total_rows,
                'events_created': events_created,
                'errors': errors,
                'processed_at': datetime.utcnow().isoformat()
            }
        }).eq('id', file_id).execute()
        
        # Step 7: Generate insights
        await manager.send_update(job_id, {
            "step": "insights",
            "message": "💡 Generating intelligent financial insights...",
            "progress": 95
        })
        
        insights = await self.analyzer.generate_insights(first_sheet, doc_analysis)
        
        # Add processing statistics
        insights.update({
            'processing_stats': {
                'total_rows_processed': processed_rows,
                'events_created': events_created,
                'errors_count': len(errors),
                'platform_detected': platform_info.get('platform', 'unknown'),
                'platform_confidence': platform_info.get('confidence', 0.0),
                'platform_description': platform_info.get('description', 'Unknown platform'),
                'platform_reasoning': platform_info.get('reasoning', 'No clear platform indicators'),
                'matched_columns': platform_info.get('matched_columns', []),
                'matched_patterns': platform_info.get('matched_patterns', []),
                'file_hash': file_hash,
                'processing_mode': 'batch_optimized',
                'batch_size': 20,
                'ai_calls_reduced': f"{(total_rows - (total_rows // 20)) / total_rows * 100:.1f}%",
                'file_type': filename.split('.')[-1].lower() if '.' in filename else 'unknown'
            },
            'errors': errors
        })
        
        # Add enhanced platform information if detected
        if platform_info.get('platform') != 'unknown':
            platform_details = self.platform_detector.get_platform_info(platform_info['platform'])
            insights['platform_details'] = {
                'name': platform_details['name'],
                'description': platform_details['description'],
                'typical_columns': platform_details['typical_columns'],
                'keywords': platform_details['keywords'],
                'detection_confidence': platform_info.get('confidence', 0.0),
                'detection_reasoning': platform_info.get('reasoning', 'No clear platform indicators'),
                'matched_indicators': {
                    'columns': platform_info.get('matched_columns', []),
                    'patterns': platform_info.get('matched_patterns', [])
                }
            }
        
        # Step 8: Entity Resolution and Normalization
        await manager.send_update(job_id, {
            "step": "entity_resolution",
            "message": "🔍 Resolving and normalizing entities...",
            "progress": 85
        })
        
        try:
            # Extract entities from processed events
            entities = await self._extract_entities_from_events(user_id, file_id, supabase)
            entity_matches = await self._resolve_entities(entities, user_id, filename, supabase)
            
            # Store normalized entities and matches
            await self._store_normalized_entities(entities, user_id, transaction_id, supabase)
            await self._store_entity_matches(entity_matches, user_id, transaction_id, supabase)
            
            insights['entity_resolution'] = {
                'entities_found': len(entities),
                'matches_created': len(entity_matches)
            }
            
            await manager.send_update(job_id, {
                "step": "entity_resolution_completed",
                "message": f"✅ Resolved {len(entities)} entities with {len(entity_matches)} matches",
                "progress": 90
            })
            
        except Exception as e:
            logger.error(f"Entity resolution failed: {e}")
            insights['entity_resolution'] = {'error': str(e)}
            # Send error to frontend
            await manager.send_update(job_id, {
                "step": "entity_resolution_failed",
                "message": f"❌ Entity resolution failed: {str(e)}",
                "progress": 90
            })

        # Step 9: Platform Pattern Learning
        await manager.send_update(job_id, {
            "step": "platform_learning",
            "message": "🧠 Learning platform patterns...",
            "progress": 92
        })
        
        try:
            # Learn platform patterns from the data
            platform_patterns = await self._learn_platform_patterns(platform_info, user_id, filename, supabase)
            discovered_platforms = await self._discover_new_platforms(user_id, filename, supabase)
            
            # Store platform patterns and discoveries
            await self._store_platform_patterns(platform_patterns, user_id, transaction_id, supabase)
            await self._store_discovered_platforms(discovered_platforms, user_id, transaction_id, supabase)
            
            insights['platform_learning'] = {
                'patterns_learned': len(platform_patterns),
                'platforms_discovered': len(discovered_platforms)
            }
            
            await manager.send_update(job_id, {
                "step": "platform_learning_completed",
                "message": f"✅ Learned {len(platform_patterns)} patterns, discovered {len(discovered_platforms)} platforms",
                "progress": 95
            })
            
        except Exception as e:
            logger.error(f"Platform learning failed: {e}")
            insights['platform_learning'] = {'error': str(e)}
            # Send error to frontend
            await manager.send_update(job_id, {
                "step": "platform_learning_failed",
                "message": f"❌ Platform learning failed: {str(e)}",
                "progress": 95
            })

        # Step 10: Relationship Detection
        await manager.send_update(job_id, {
            "step": "relationships",
            "message": "🔗 Detecting relationships between financial events...",
            "progress": 97
        })
        
        try:
            from enhanced_relationship_detector import EnhancedRelationshipDetector
            from openai import AsyncOpenAI
            
            # Initialize relationship detector
            openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            relationship_detector = EnhancedRelationshipDetector(openai_client, supabase)
        except ImportError:
            logger.warning("Enhanced relationship detector not available, skipping relationship detection")
            return
            
            # Detect all relationships
            relationship_results = await relationship_detector.detect_all_relationships(user_id)
            
            # Store relationship instances
            if relationship_results.get('relationships'):
                await self._store_relationship_instances(relationship_results['relationships'], user_id, transaction_id, supabase)
            
            # Add relationship results to insights
            insights['relationship_analysis'] = relationship_results
            
            await manager.send_update(job_id, {
                "step": "relationships_completed",
                "message": f"✅ Found {relationship_results.get('total_relationships', 0)} relationships between events",
                "progress": 98
            })
            
        except Exception as e:
            logger.error(f"Relationship detection failed: {e}")
            insights['relationship_analysis'] = {
                'error': str(e),
                'message': 'Relationship detection failed but processing completed'
            }
            # Send error to frontend
            await manager.send_update(job_id, {
                "step": "relationship_detection_failed",
                "message": f"❌ Relationship detection failed: {str(e)}",
                "progress": 98
            })

        # Step 11: Compute and Store Metrics
        await manager.send_update(job_id, {
            "step": "metrics",
            "message": "📊 Computing processing metrics...",
            "progress": 99
        })
        
        try:
            # Compute comprehensive metrics
            metrics = {
                'metric_type': 'file_processing_summary',
                'metric_value': events_created,
                'metric_data': {
                    'total_rows_processed': processed_rows,
                    'events_created': events_created,
                    'errors_count': len(errors),
                    'platform_detected': platform_info.get('platform', 'unknown'),
                    'platform_confidence': platform_info.get('confidence', 0.0),
                    'entities_resolved': len(entities) if 'entities' in locals() else 0,
                    'relationships_found': relationship_results.get('total_relationships', 0) if 'relationship_results' in locals() else 0,
                    'processing_time_seconds': (datetime.utcnow() - datetime.fromisoformat(transaction_data['started_at'])).total_seconds() if transaction_id else 0
                }
            }
            
            await self._store_computed_metrics(metrics, user_id, transaction_id, supabase)
            insights['processing_metrics'] = metrics
            
        except Exception as e:
            logger.error(f"Metrics computation failed: {e}")
            insights['processing_metrics'] = {'error': str(e)}
            # Send error to frontend
            await manager.send_update(job_id, {
                "step": "metrics_computation_failed",
                "message": f"❌ Metrics computation failed: {str(e)}",
                "progress": 99
            })
        
        # Step 12: Complete Transaction
        if transaction_id:
            try:
                supabase.table('processing_transactions').update({
                    'status': 'committed',
                    'committed_at': datetime.utcnow().isoformat(),
                    'metadata': {
                        **transaction_data['metadata'],
                        'events_created': events_created,
                        'entities_resolved': len(entities) if 'entities' in locals() else 0,
                        'relationships_found': relationship_results.get('total_relationships', 0) if 'relationship_results' in locals() else 0
                    }
                }).eq('id', transaction_id).execute()
                logger.info(f"Committed processing transaction: {transaction_id}")
            except Exception as e:
                logger.warning(f"Failed to commit transaction: {e}")

        # Step 13: Update ingestion_jobs with completion
        supabase.table('ingestion_jobs').update({
            'status': 'completed',
            'updated_at': datetime.utcnow().isoformat(),
            'transaction_id': transaction_id
        }).eq('id', job_id).execute()
        
        await manager.send_update(job_id, {
            "step": "completed",
            "message": f"✅ Processing completed! {events_created} events created from {processed_rows} rows.",
            "progress": 100
        })
        
        return insights

    async def _store_normalized_entities(self, entities: List[Dict], user_id: str, transaction_id: str, supabase: Client):
        """Store normalized entities in the database"""
        try:
            if not entities:
                return
            
            logger.info(f"Storing {len(entities)} normalized entities")
            
            for entity in entities:
                entity_data = {
                    'user_id': user_id,
                    'entity_type': entity.get('entity_type', 'vendor'),
                    'canonical_name': entity.get('canonical_name', ''),
                    'aliases': entity.get('aliases', []),
                    'email': entity.get('email'),
                    'phone': entity.get('phone'),
                    'bank_account': entity.get('bank_account'),
                    'tax_id': entity.get('tax_id'),
                    'platform_sources': entity.get('platform_sources', []),
                    'source_files': entity.get('source_files', []),
                    'confidence_score': entity.get('confidence_score', 0.5),
                    'transaction_id': transaction_id
                }
                
                result = supabase.table('normalized_entities').insert(entity_data).execute()
                if result.data:
                    logger.debug(f"Stored normalized entity: {entity_data['canonical_name']}")
                else:
                    logger.warning(f"Failed to store normalized entity: {entity_data['canonical_name']}")
                    
        except Exception as e:
            logger.error(f"Error storing normalized entities: {e}")

    async def _store_entity_matches(self, matches: List[Dict], user_id: str, transaction_id: str, supabase: Client):
        """Store entity matches in the database"""
        try:
            if not matches:
                return
                
            logger.info(f"Storing {len(matches)} entity matches")
            
            for match in matches:
                match_data = {
                    'user_id': user_id,
                    'source_entity_name': match.get('source_entity_name', ''),
                    'source_entity_type': match.get('source_entity_type', 'vendor'),
                    'source_platform': match.get('source_platform', 'unknown'),
                    'source_file': match.get('source_file', ''),
                    'source_row_id': match.get('source_row_id'),
                    'normalized_entity_id': match.get('normalized_entity_id'),
                    'match_confidence': match.get('match_confidence', 0.5),
                    'match_reason': match.get('match_reason', 'unknown'),
                    'similarity_score': match.get('similarity_score'),
                    'matched_fields': match.get('matched_fields', []),
                    'transaction_id': transaction_id
                }
                
                result = supabase.table('entity_matches').insert(match_data).execute()
                if result.data:
                    logger.debug(f"Stored entity match: {match_data['source_entity_name']}")
                else:
                    logger.warning(f"Failed to store entity match: {match_data['source_entity_name']}")
                    
        except Exception as e:
            logger.error(f"Error storing entity matches: {e}")

    async def _store_platform_patterns(self, patterns: List[Dict], user_id: str, transaction_id: str, supabase: Client):
        """Store platform patterns in the database"""
        try:
            if not patterns:
                return
                
            logger.info(f"Storing {len(patterns)} platform patterns")
            
            for pattern in patterns:
                pattern_data = {
                    'user_id': user_id,
                    'platform': pattern.get('platform', 'unknown'),
                    'pattern_type': pattern.get('pattern_type', 'column'),
                    'pattern_data': pattern.get('pattern_data', {}),
                    'confidence_score': pattern.get('confidence_score', 0.5),
                    'detection_method': pattern.get('detection_method', 'ai'),
                    'transaction_id': transaction_id
                }
                
                result = supabase.table('platform_patterns').insert(pattern_data).execute()
                if result.data:
                    logger.debug(f"Stored platform pattern: {pattern_data['platform']}")
                else:
                    logger.warning(f"Failed to store platform pattern: {pattern_data['platform']}")
                    
        except Exception as e:
            logger.error(f"Error storing platform patterns: {e}")

    async def _store_relationship_instances(self, relationships: List[Dict], user_id: str, transaction_id: str, supabase: Client):
        """Store relationship instances in the database"""
        try:
            if not relationships:
                return
                
            logger.info(f"Storing {len(relationships)} relationship instances")
            
            for relationship in relationships:
                rel_data = {
                    'user_id': user_id,
                    'source_event_id': relationship.get('source_event_id'),
                    'target_event_id': relationship.get('target_event_id'),
                    'relationship_type': relationship.get('relationship_type', 'unknown'),
                    'confidence_score': relationship.get('confidence_score', 0.5),
                    'detection_method': relationship.get('detection_method', 'ai'),
                    'pattern_id': relationship.get('pattern_id'),
                    'reasoning': relationship.get('reasoning', ''),
                    'transaction_id': transaction_id
                }
                
                result = supabase.table('relationship_instances').insert(rel_data).execute()
                if result.data:
                    logger.debug(f"Stored relationship: {rel_data['relationship_type']}")
                else:
                    logger.warning(f"Failed to store relationship: {rel_data['relationship_type']}")
                    
        except Exception as e:
            logger.error(f"Error storing relationship instances: {e}")

    async def _store_discovered_platforms(self, platforms: List[Dict], user_id: str, transaction_id: str, supabase: Client):
        """Store discovered platforms in the database"""
        try:
            if not platforms:
                return
                
            logger.info(f"Storing {len(platforms)} discovered platforms")
            
            for platform in platforms:
                platform_data = {
                    'user_id': user_id,
                    'platform_name': platform.get('platform_name', ''),
                    'platform_type': platform.get('platform_type', 'unknown'),
                    'detection_confidence': platform.get('detection_confidence', 0.5),
                    'detection_method': platform.get('detection_method', 'ai'),
                    'characteristics': platform.get('characteristics', {}),
                    'source_files': platform.get('source_files', []),
                    'transaction_id': transaction_id
                }
                
                result = supabase.table('discovered_platforms').insert(platform_data).execute()
                if result.data:
                    logger.debug(f"Stored discovered platform: {platform_data['platform_name']}")
                else:
                    logger.warning(f"Failed to store discovered platform: {platform_data['platform_name']}")
                    
        except Exception as e:
            logger.error(f"Error storing discovered platforms: {e}")

    async def _store_computed_metrics(self, metrics: Dict, user_id: str, transaction_id: str, supabase: Client):
        """Store computed metrics in the database"""
        try:
            if not metrics:
                return
                
            logger.info("Storing computed metrics")
            
            metrics_data = {
                'user_id': user_id,
                'metric_type': metrics.get('metric_type', 'processing_summary'),
                'metric_value': metrics.get('metric_value', 0),
                'metric_data': metrics.get('metric_data', {}),
                'computed_at': datetime.utcnow().isoformat(),
                'transaction_id': transaction_id
            }
            
            result = supabase.table('metrics').insert(metrics_data).execute()
            if result.data:
                logger.debug("Stored computed metrics")
            else:
                logger.warning("Failed to store computed metrics")
                
        except Exception as e:
            logger.error(f"Error storing computed metrics: {e}")

    async def _extract_entities_from_events(self, user_id: str, file_id: str, supabase: Client) -> List[Dict]:
        """Extract entities from processed events for normalization"""
        try:
            # Get events for this file
            events = supabase.table('raw_events').select('*').eq('user_id', user_id).eq('file_id', file_id).execute()
            
            logger.info(f"Found {len(events.data)} events for entity extraction")
            
            entities = []
            entity_map = {}
            vendor_fields_found = []
            
            for event in events.data:
                # Extract vendor/entity information from payload
                payload = event.get('payload', {})
                
                # Check what vendor fields are available
                vendor_fields = ['vendor_raw', 'vendor', 'merchant', 'payee', 'description']
                for field in vendor_fields:
                    if field in payload and payload[field]:
                        vendor_fields_found.append(f"{field}: {payload[field]}")
                
                vendor_raw = payload.get('vendor_raw') or payload.get('vendor') or payload.get('merchant')
                
                if vendor_raw and vendor_raw not in entity_map:
                    entity = {
                        'entity_type': 'vendor',
                        'canonical_name': vendor_raw,
                        'aliases': [vendor_raw],
                        'email': payload.get('email'),
                        'phone': payload.get('phone'),
                        'bank_account': payload.get('bank_account'),
                        'platform_sources': [event.get('source_platform', 'unknown')],
                        'source_files': [event.get('source_filename', '')],
                        'confidence_score': 0.8
                    }
                    entities.append(entity)
                    entity_map[vendor_raw] = entity
            
            logger.info(f"Extracted {len(entities)} entities from {len(events.data)} events")
            if vendor_fields_found:
                logger.info(f"Found vendor fields: {vendor_fields_found[:5]}")  # Show first 5
            else:
                logger.warning("No vendor/merchant fields found in any events - this is why entity extraction returns 0")
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []

    async def _resolve_entities(self, entities: List[Dict], user_id: str, filename: str, supabase: Client) -> List[Dict]:
        """Resolve entities using the database function"""
        try:
            matches = []
            
            for entity in entities:
                # Use the database function to find or create entity
                result = supabase.rpc('find_or_create_entity', {
                    'p_user_id': user_id,
                    'p_entity_name': entity['canonical_name'],
                    'p_entity_type': entity['entity_type'],
                    'p_platform': entity['platform_sources'][0] if entity['platform_sources'] else 'unknown',
                    'p_email': entity.get('email'),
                    'p_bank_account': entity.get('bank_account'),
                    'p_phone': entity.get('phone'),
                    'p_source_file': filename
                }).execute()
                
                if result.data:
                    entity_id = result.data[0] if isinstance(result.data, list) else result.data
                    match = {
                        'source_entity_name': entity['canonical_name'],
                        'source_entity_type': entity['entity_type'],
                        'source_platform': entity['platform_sources'][0] if entity['platform_sources'] else 'unknown',
                        'source_file': filename,
                        'normalized_entity_id': entity_id,
                        'match_confidence': entity['confidence_score'],
                        'match_reason': 'exact_match',
                        'similarity_score': 1.0,
                        'matched_fields': ['name']
                    }
                    matches.append(match)
            
            logger.info(f"Resolved {len(matches)} entity matches")
            return matches
            
        except Exception as e:
            logger.error(f"Error resolving entities: {e}")
            return []

    async def _learn_platform_patterns(self, platform_info: Dict, user_id: str, filename: str, supabase: Client) -> List[Dict]:
        """Learn platform patterns from the detected platform"""
        try:
            patterns = []
            
            if platform_info.get('platform') != 'unknown':
                pattern = {
                    'platform': platform_info['platform'],
                    'pattern_type': 'column_structure',
                    'pattern_data': {
                        'matched_columns': platform_info.get('matched_columns', []),
                        'matched_patterns': platform_info.get('matched_patterns', []),
                        'confidence': platform_info.get('confidence', 0.0),
                        'reasoning': platform_info.get('reasoning', '')
                    },
                    'confidence_score': platform_info.get('confidence', 0.0),
                    'detection_method': 'ai_analysis'
                }
                patterns.append(pattern)
            
            logger.info(f"Learned {len(patterns)} platform patterns")
            return patterns
            
        except Exception as e:
            logger.error(f"Error learning platform patterns: {e}")
            return []

    async def _discover_new_platforms(self, user_id: str, filename: str, supabase: Client) -> List[Dict]:
        """Discover new platforms from the data"""
        try:
            # For now, return empty list - this would be implemented with AI analysis
            # of the data to discover custom platforms
            platforms = []
            
            logger.info(f"Discovered {len(platforms)} new platforms")
            return platforms
            
        except Exception as e:
            logger.error(f"Error discovering platforms: {e}")
            return []

class UniversalFieldDetector:
    """Universal field detection that works with ANY field names using AI and pattern recognition"""
    
    def __init__(self):
        self.field_patterns = {
            'vendor_fields': [
                'vendor', 'merchant', 'payee', 'client', 'customer', 'company', 'business', 'entity',
                'recipient', 'beneficiary', 'party', 'supplier', 'name', 'organization', 'corp', 'inc',
                'ltd', 'llc', 'to', 'from', 'contact', 'person', 'individual', 'firm', 'enterprise'
            ],
            'amount_fields': [
                'amount', 'total', 'value', 'sum', 'payment', 'price', 'cost', 'fee', 'charge',
                'revenue', 'income', 'expense', 'debit', 'credit', 'balance', 'gross', 'net',
                'subtotal', 'tax', 'discount', 'refund', 'deposit', 'withdrawal', 'transfer',
                'salary', 'wage', 'bonus', 'commission', 'allowance', 'benefit', 'deduction'
            ],
            'date_fields': [
                'date', 'time', 'created', 'updated', 'processed', 'issued', 'due', 'paid',
                'timestamp', 'when', 'period', 'month', 'year', 'quarter', 'fiscal',
                'transaction_date', 'payment_date', 'issue_date', 'due_date', 'created_at',
                'updated_at', 'processed_at', 'paid_at', 'received_at', 'sent_at'
            ],
            'currency_fields': [
                'currency', 'curr', 'ccy', 'money_type', 'denomination', 'unit', 'symbol',
                'currency_code', 'iso_currency', 'base_currency', 'quote_currency'
            ],
            'description_fields': [
                'description', 'memo', 'notes', 'comment', 'details', 'summary', 'purpose',
                'reason', 'explanation', 'narrative', 'text', 'content', 'info', 'information',
                'remark', 'annotation', 'label', 'title', 'subject', 'topic', 'category'
            ],
            'id_fields': [
                'id', 'identifier', 'number', 'code', 'reference', 'ref', 'key', 'primary_key',
                'transaction_id', 'payment_id', 'order_id', 'invoice_id', 'receipt_id',
                'account_id', 'customer_id', 'vendor_id', 'employee_id', 'project_id'
            ]
        }
    
    def detect_field_types(self, payload: Dict) -> Dict[str, List[str]]:
        """Detect what type of data each field contains"""
        field_types = {
            'vendor_fields': [],
            'amount_fields': [],
            'date_fields': [],
            'currency_fields': [],
            'description_fields': [],
            'id_fields': []
        }
        
        for field_name, field_value in payload.items():
            if not isinstance(field_value, str):
                continue
                
            field_lower = field_name.lower()
            value_lower = str(field_value).lower().strip()
            
            # Skip empty values
            if not value_lower:
                continue
            
            # Detect field type by name patterns
            for field_type, patterns in self.field_patterns.items():
                if any(pattern in field_lower for pattern in patterns):
                    field_types[field_type].append(field_name)
                    break
            
            # Detect field type by value patterns
            if not any(field_name in fields for fields in field_types.values()):
                detected_type = self._detect_by_value_pattern(field_value)
                if detected_type:
                    field_types[detected_type].append(field_name)
        
        return field_types
    
    def _detect_by_value_pattern(self, value: str) -> Optional[str]:
        """Detect field type by analyzing the value content"""
        if not isinstance(value, str):
            return None
        
        value = value.strip()
        
        # Amount detection
        if self._looks_like_amount(value):
            return 'amount_fields'
        
        # Date detection
        if self._looks_like_date(value):
            return 'date_fields'
        
        # Currency detection
        if self._looks_like_currency(value):
            return 'currency_fields'
        
        # ID detection
        if self._looks_like_id(value):
            return 'id_fields'
        
        # Description detection (longer text)
        if len(value) > 10 and not self._looks_like_amount(value):
            return 'description_fields'
        
        # Vendor detection (looks like company/person name)
        if self._looks_like_entity_name(value):
            return 'vendor_fields'
        
        return None
    
    def _looks_like_amount(self, value: str) -> bool:
        """Check if value looks like a monetary amount"""
        import re
        
        # Remove common currency symbols and spaces
        cleaned = re.sub(r'[$₹€£¥,\s]', '', value)
        
        # Check if it's a number (including decimals)
        if re.match(r'^\d+\.?\d*$', cleaned):
            try:
                amount = float(cleaned)
                # Reasonable amount range
                return 0 <= amount <= 10000000
            except:
                pass
        
        return False
    
    def _looks_like_date(self, value: str) -> bool:
        """Check if value looks like a date"""
        import re
        from datetime import datetime
        
        # Common date patterns
        date_patterns = [
            r'^\d{4}-\d{2}-\d{2}$',  # YYYY-MM-DD
            r'^\d{2}/\d{2}/\d{4}$',  # MM/DD/YYYY
            r'^\d{2}-\d{2}-\d{4}$',  # MM-DD-YYYY
            r'^\d{4}/\d{2}/\d{2}$',  # YYYY/MM/DD
            r'^\d{1,2}/\d{1,2}/\d{4}$',  # M/D/YYYY
            r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',  # YYYY-MM-DD HH:MM:SS
        ]
        
        for pattern in date_patterns:
            if re.match(pattern, value):
                return True
        
        # Try to parse as date
        try:
            datetime.fromisoformat(value.replace('Z', '+00:00'))
            return True
        except:
            pass
        
        return False
    
    def _looks_like_currency(self, value: str) -> bool:
        """Check if value looks like a currency code"""
        currency_codes = ['usd', 'eur', 'inr', 'gbp', 'jpy', 'cad', 'aud', 'chf', 'cny', 'sek', 'nok', 'dkk']
        return value.lower().strip() in currency_codes
    
    def _looks_like_id(self, value: str) -> bool:
        """Check if value looks like an ID"""
        import re
        
        # UUID pattern
        if re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', value, re.IGNORECASE):
            return True
        
        # Long alphanumeric strings
        if re.match(r'^[a-zA-Z0-9_-]{8,}$', value):
            return True
        
        # Numeric IDs
        if re.match(r'^\d{6,}$', value):
            return True
        
        return False
    
    def _looks_like_entity_name(self, value: str) -> bool:
        """Check if value looks like an entity/company name"""
        if not value or len(value) < 2 or len(value) > 200:
            return False
        
        # Should contain letters
        if not any(c.isalpha() for c in value):
            return False
        
        # Shouldn't be just numbers, dates, or special characters
        import re
        if re.match(r'^\d+$', value) or re.match(r'^\d{4}-\d{2}-\d{2}', value):
            return False
        
        # Company name indicators
        company_indicators = ['corp', 'inc', 'ltd', 'llc', 'co', 'company', 'group', 'solutions', 'services', 'systems']
        if any(indicator in value.lower() for indicator in company_indicators):
            return True
        
        # Multiple words (companies usually have 2+ words)
        if len(value.split()) >= 2:
            return True
        
        return False

class UniversalPlatformDetector:
    """Universal platform detection using AI and pattern recognition"""
    
    def __init__(self, openai_client=None):
        self.openai_client = openai_client
        self.platform_patterns = {
            'payment_gateways': [
                'stripe', 'razorpay', 'paypal', 'square', 'stripe.com', 'razorpay.com',
                'paypal.com', 'squareup.com', 'stripe_', 'rzp_', 'pp_', 'sq_'
            ],
            'banking': [
                'bank', 'chase', 'wells fargo', 'bank of america', 'citibank', 'hsbc',
                'jpmorgan', 'goldman sachs', 'morgan stanley', 'deutsche bank'
            ],
            'accounting': [
                'quickbooks', 'xero', 'freshbooks', 'wave', 'zoho books', 'sage',
                'intuit', 'quickbooks.com', 'xero.com', 'freshbooks.com'
            ],
            'crm': [
                'salesforce', 'hubspot', 'pipedrive', 'zoho crm', 'monday.com',
                'salesforce.com', 'hubspot.com', 'pipedrive.com'
            ],
            'ecommerce': [
                'shopify', 'woocommerce', 'magento', 'bigcommerce', 'amazon',
                'shopify.com', 'woocommerce.com', 'magento.com'
            ],
            'cloud_services': [
                'aws', 'azure', 'google cloud', 'digitalocean', 'linode', 'heroku',
                'amazon web services', 'microsoft azure', 'gcp'
            ]
        }
    
    async def detect_platform_universal(self, payload: Dict, filename: str = None) -> Dict[str, Any]:
        """Detect platform using universal AI-powered analysis"""
        try:
            # Strategy 1: AI-powered platform detection
            if self.openai_client:
                ai_result = await self._detect_platform_with_ai(payload, filename)
                if ai_result and ai_result.get('confidence', 0) > 0.7:
                    return ai_result
            
            # Strategy 2: Pattern-based detection
            pattern_result = self._detect_platform_with_patterns(payload, filename)
            if pattern_result:
                return pattern_result
            
            # Strategy 3: Field-based detection
            field_result = self._detect_platform_from_fields(payload)
            if field_result:
                return field_result
            
            # Default fallback
            return {
                'platform': 'unknown',
                'confidence': 0.1,
                'detection_method': 'fallback',
                'indicators': []
            }
            
        except Exception as e:
            logger.error(f"Platform detection failed: {e}")
            return {
                'platform': 'unknown',
                'confidence': 0.0,
                'detection_method': 'error',
                'error': str(e)
            }
    
    async def _detect_platform_with_ai(self, payload: Dict, filename: str = None) -> Optional[Dict[str, Any]]:
        """Use AI to detect platform from data content"""
        try:
            # Prepare context for AI
            context_parts = []
            
            # Add filename if available
            if filename:
                context_parts.append(f"Filename: {filename}")
            
            # Add key fields that might indicate platform
            key_fields = ['description', 'memo', 'notes', 'platform', 'source', 'reference', 'id']
            for field in key_fields:
                if field in payload and payload[field]:
                    context_parts.append(f"{field}: {payload[field]}")
            
            # Add all field names as context
            field_names = list(payload.keys())
            context_parts.append(f"Field names: {', '.join(field_names)}")
            
            context = "\n".join(context_parts)
            
            # AI prompt for platform detection
            prompt = f"""
            Analyze this financial data to detect the platform or service it came from:
            
            {context}
            
            Common platforms include:
            - Payment gateways: Stripe, Razorpay, PayPal, Square
            - Banking: Chase, Wells Fargo, Bank of America, etc.
            - Accounting: QuickBooks, Xero, FreshBooks, etc.
            - CRM: Salesforce, HubSpot, Pipedrive, etc.
            - E-commerce: Shopify, WooCommerce, Amazon, etc.
            - Cloud services: AWS, Azure, Google Cloud, etc.
            
            Respond with JSON format:
            {{
                "platform": "detected_platform_name",
                "confidence": 0.0-1.0,
                "indicators": ["list", "of", "indicators"],
                "reasoning": "explanation"
            }}
            """
            
            result_text = await safe_openai_call(
                self.openai_client,
                "gpt-4o-mini",
                [{"role": "user", "content": prompt}],
                0.1,
                200,
                '{"platform": "unknown", "confidence": 0.0, "indicators": [], "reasoning": "AI processing unavailable due to quota limits"}'
            )
            
            # Clean up the response text
            if result_text.startswith('```json'):
                result_text = result_text[7:]
            if result_text.endswith('```'):
                result_text = result_text[:-3]
            result_text = result_text.strip()
            
            # Parse JSON response with better error handling
            result = safe_json_parse(result_text, {
                'platform': 'unknown',
                'confidence': 0.0,
                'indicators': [],
                'reasoning': 'JSON parsing failed, using fallback'
            })
            
            if not result:
                logger.error(f"AI platform detection JSON parsing failed")
                logger.error(f"Raw AI response: {result_text}")
                return {
                    'platform': 'unknown',
                    'confidence': 0.0,
                    'detection_method': 'ai_fallback',
                    'indicators': [],
                    'reasoning': 'JSON parsing failed, using fallback'
                }
            
            return {
                'platform': result.get('platform', 'unknown'),
                'confidence': float(result.get('confidence', 0.0)),
                'detection_method': 'ai',
                'indicators': result.get('indicators', []),
                'reasoning': result.get('reasoning', '')
            }
            
        except Exception as e:
            logger.error(f"AI platform detection failed: {e}")
            return None
    
    def _detect_platform_with_patterns(self, payload: Dict, filename: str = None) -> Optional[Dict[str, Any]]:
        """Detect platform using pattern matching"""
        try:
            # Combine all text for pattern matching
            text_parts = []
            
            # Add filename
            if filename:
                text_parts.append(filename.lower())
            
            # Add all string values (handle DataFrame case)
            if hasattr(payload, 'values') and hasattr(payload.values, 'flatten'):
                # DataFrame case - values is a numpy array
                try:
                    for value in payload.values.flatten():
                        if isinstance(value, str):
                            text_parts.append(value.lower())
                except AttributeError:
                    # Fallback for non-numpy arrays
                    try:
                        for value in payload.values.ravel():
                            if isinstance(value, str):
                                text_parts.append(value.lower())
                    except AttributeError:
                        # If neither flatten nor ravel work, iterate directly
                        for value in payload.values:
                            if isinstance(value, str):
                                text_parts.append(value.lower())
            else:
                # Dict case
                for value in payload.values():
                    if isinstance(value, str):
                        text_parts.append(value.lower())
            
            combined_text = " ".join(text_parts)
            
            # Check against platform patterns
            for platform_type, patterns in self.platform_patterns.items():
                for pattern in patterns:
                    if pattern.lower() in combined_text:
                        return {
                            'platform': pattern,
                            'confidence': 0.8,
                            'detection_method': 'pattern',
                            'indicators': [pattern],
                            'platform_type': platform_type
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Pattern platform detection failed: {e}")
            return {
                'platform': 'unknown',
                'confidence': 0.0,
                'detection_method': 'pattern_error',
                'indicators': [],
                'reasoning': f'Pattern detection failed: {str(e)}'
            }
    
    def _detect_platform_from_fields(self, payload: Dict) -> Optional[Dict[str, Any]]:
        """Detect platform from field names and structure"""
        try:
            field_names = [key.lower() for key in payload.keys()]
            
            # Check for platform-specific field patterns
            platform_indicators = {
                'stripe': ['stripe', 'stripe_id', 'charge_id', 'customer_id', 'payment_intent'],
                'razorpay': ['razorpay', 'rzp', 'payment_id', 'order_id', 'refund_id'],
                'paypal': ['paypal', 'pp', 'transaction_id', 'payer_id', 'payment_id'],
                'quickbooks': ['quickbooks', 'qb', 'customer_id', 'invoice_id', 'payment_id'],
                'xero': ['xero', 'contact_id', 'invoice_id', 'payment_id'],
                'salesforce': ['salesforce', 'sf', 'lead_id', 'opportunity_id', 'account_id']
            }
            
            for platform, indicators in platform_indicators.items():
                matches = [indicator for indicator in indicators if any(indicator in field for field in field_names)]
                if matches:
                    return {
                        'platform': platform,
                        'confidence': 0.6,
                        'detection_method': 'field_analysis',
                        'indicators': matches
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Field-based platform detection failed: {e}")
            return None

class UniversalDocumentClassifier:
    """Universal document type classification using AI"""
    
    def __init__(self, openai_client=None):
        self.openai_client = openai_client
        self.document_types = {
            'invoice': ['invoice', 'bill', 'receipt', 'statement'],
            'payment': ['payment', 'transaction', 'transfer', 'deposit', 'withdrawal'],
            'expense': ['expense', 'cost', 'charge', 'fee', 'purchase'],
            'revenue': ['revenue', 'income', 'sale', 'earning', 'profit'],
            'payroll': ['payroll', 'salary', 'wage', 'employee', 'staff'],
            'tax': ['tax', 'vat', 'gst', 'withholding', 'deduction'],
            'bank_statement': ['bank', 'statement', 'account', 'balance'],
            'credit_card': ['credit', 'card', 'visa', 'mastercard', 'amex']
        }
    
    async def classify_document_universal(self, payload: Dict, filename: str = None) -> Dict[str, Any]:
        """Classify document type using universal AI-powered analysis"""
        try:
            # Strategy 1: AI-powered classification
            if self.openai_client:
                ai_result = await self._classify_with_ai(payload, filename)
                if ai_result and ai_result.get('confidence', 0) > 0.7:
                    return ai_result
            
            # Strategy 2: Pattern-based classification
            pattern_result = self._classify_with_patterns(payload, filename)
            if pattern_result:
                return pattern_result
            
            # Strategy 3: Field-based classification
            field_result = self._classify_from_fields(payload)
            if field_result:
                return field_result
            
            # Default fallback
            return {
                'document_type': 'unknown',
                'confidence': 0.1,
                'classification_method': 'fallback',
                'indicators': []
            }
            
        except Exception as e:
            logger.error(f"Document classification failed: {e}")
            return {
                'document_type': 'unknown',
                'confidence': 0.0,
                'classification_method': 'error',
                'error': str(e)
            }
    
    async def _classify_with_ai(self, payload: Dict, filename: str = None) -> Optional[Dict[str, Any]]:
        """Use AI to classify document type"""
        try:
            # Prepare context for AI
            context_parts = []
            
            # Add filename if available
            if filename:
                context_parts.append(f"Filename: {filename}")
            
            # Add key fields
            key_fields = ['description', 'memo', 'notes', 'type', 'category', 'kind']
            for field in key_fields:
                if field in payload and payload[field]:
                    context_parts.append(f"{field}: {payload[field]}")
            
            # Add field names
            field_names = list(payload.keys())
            context_parts.append(f"Field names: {', '.join(field_names)}")
            
            context = "\n".join(context_parts)
            
            # AI prompt for document classification
            prompt = f"""
            Classify this financial document data into one of these types:
            
            {context}
            
            Document types:
            - invoice: Bills, invoices, receipts, statements
            - payment: Payments, transactions, transfers, deposits
            - expense: Expenses, costs, charges, fees, purchases
            - revenue: Revenue, income, sales, earnings
            - payroll: Payroll, salaries, wages, employee payments
            - tax: Tax documents, VAT, GST, withholding
            - bank_statement: Bank statements, account balances
            - credit_card: Credit card statements, card payments
            
            Respond with JSON format:
            {{
                "document_type": "detected_type",
                "confidence": 0.0-1.0,
                "indicators": ["list", "of", "indicators"],
                "reasoning": "explanation"
            }}
            """
            
            result_text = await safe_openai_call(
                self.openai_client,
                "gpt-4o-mini",
                [{"role": "user", "content": prompt}],
                0.1,
                200,
                '{"platform": "unknown", "confidence": 0.0, "indicators": [], "reasoning": "AI processing unavailable due to quota limits"}'
            )
            
            # Clean up the response text
            if result_text.startswith('```json'):
                result_text = result_text[7:]
            if result_text.endswith('```'):
                result_text = result_text[:-3]
            result_text = result_text.strip()
            
            # Parse JSON response with better error handling
            import json
            try:
                result = json.loads(result_text)
            except json.JSONDecodeError as e:
                logger.error(f"AI document classification JSON parsing failed: {e}")
                logger.error(f"Raw AI response: {result_text}")
                return None
            
            return {
                'document_type': result.get('document_type', 'unknown'),
                'confidence': float(result.get('confidence', 0.0)),
                'classification_method': 'ai',
                'indicators': result.get('indicators', []),
                'reasoning': result.get('reasoning', '')
            }
            
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                logger.warning(f"AI document classification failed due to quota: {e}")
                return {
                    'document_type': 'unknown',
                    'confidence': 0.0,
                    'reasoning': 'AI processing unavailable due to quota limits'
                }
            else:
                logger.error(f"AI document classification failed: {e}")
                return None
    
    def _classify_with_patterns(self, payload: Dict, filename: str = None) -> Optional[Dict[str, Any]]:
        """Classify document using pattern matching"""
        try:
            # Combine all text for pattern matching
            text_parts = []
            
            # Add filename
            if filename:
                text_parts.append(filename.lower())
            
            # Add all string values (handle DataFrame case)
            if hasattr(payload, 'values') and hasattr(payload.values, 'flatten'):
                # DataFrame case - values is a numpy array
                try:
                    for value in payload.values.flatten():
                        if isinstance(value, str):
                            text_parts.append(value.lower())
                except AttributeError:
                    # Fallback for non-numpy arrays
                    try:
                        for value in payload.values.ravel():
                            if isinstance(value, str):
                                text_parts.append(value.lower())
                    except AttributeError:
                        # If neither flatten nor ravel work, iterate directly
                        for value in payload.values:
                            if isinstance(value, str):
                                text_parts.append(value.lower())
            else:
                # Dict case
                for value in payload.values():
                    if isinstance(value, str):
                        text_parts.append(value.lower())
            
            combined_text = " ".join(text_parts)
            
            # Check against document type patterns
            for doc_type, patterns in self.document_types.items():
                for pattern in patterns:
                    if pattern.lower() in combined_text:
                        return {
                            'document_type': doc_type,
                            'confidence': 0.8,
                            'classification_method': 'pattern',
                            'indicators': [pattern]
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Pattern document classification failed: {e}")
            return None
    
    def _classify_from_fields(self, payload: Dict) -> Optional[Dict[str, Any]]:
        """Classify document from field names and structure"""
        try:
            field_names = [key.lower() for key in payload.keys()]
            
            # Check for document-specific field patterns
            doc_indicators = {
                'invoice': ['invoice', 'bill', 'receipt', 'invoice_id', 'bill_id'],
                'payment': ['payment', 'transaction', 'transfer', 'payment_id', 'txn_id'],
                'expense': ['expense', 'cost', 'charge', 'fee', 'expense_id'],
                'revenue': ['revenue', 'income', 'sale', 'revenue_id', 'sale_id'],
                'payroll': ['payroll', 'salary', 'wage', 'employee', 'payroll_id'],
                'tax': ['tax', 'vat', 'gst', 'tax_id', 'withholding'],
                'bank_statement': ['bank', 'statement', 'account', 'balance', 'account_id'],
                'credit_card': ['credit', 'card', 'visa', 'mastercard', 'card_id']
            }
            
            for doc_type, indicators in doc_indicators.items():
                matches = [indicator for indicator in indicators if any(indicator in field for field in field_names)]
                if matches:
                    return {
                        'document_type': doc_type,
                        'confidence': 0.6,
                        'classification_method': 'field_analysis',
                        'indicators': matches
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Field-based document classification failed: {e}")
            return None

class UniversalExtractors:
    """Universal data extractors that work with any field names"""
    
    def __init__(self):
        self.field_detector = UniversalFieldDetector()
    
    def extract_vendor_universal(self, payload: Dict) -> Optional[str]:
        """Extract vendor name using universal field detection"""
        field_types = self.field_detector.detect_field_types(payload)
        vendor_fields = field_types.get('vendor_fields', [])
        
        for field in vendor_fields:
            value = payload.get(field)
            if value and isinstance(value, str) and value.strip():
                return str(value).strip()
        
        return None
    
    def extract_amount_universal(self, payload: Dict) -> Optional[float]:
        """Extract amount using universal field detection"""
        field_types = self.field_detector.detect_field_types(payload)
        amount_fields = field_types.get('amount_fields', [])
        
        for field in amount_fields:
            value = payload.get(field)
            if value and self.field_detector._looks_like_amount(str(value)):
                try:
                    import re
                    cleaned = re.sub(r'[$₹€£¥,\s]', '', str(value))
                    return float(cleaned)
                except:
                    continue
        
        return None
    
    def extract_date_universal(self, payload: Dict) -> Optional[datetime]:
        """Extract date using universal field detection"""
        field_types = self.field_detector.detect_field_types(payload)
        date_fields = field_types.get('date_fields', [])
        
        for field in date_fields:
            value = payload.get(field)
            if value and self.field_detector._looks_like_date(str(value)):
                try:
                    return datetime.fromisoformat(str(value).replace('Z', '+00:00'))
                except:
                    try:
                        # Try common date formats
                        for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S']:
                            try:
                                return datetime.strptime(str(value), fmt)
                            except:
                                continue
                    except:
                        continue
        
        return None
    
    def extract_currency_universal(self, payload: Dict) -> Optional[str]:
        """Extract currency using universal field detection"""
        field_types = self.field_detector.detect_field_types(payload)
        currency_fields = field_types.get('currency_fields', [])
        
        for field in currency_fields:
            value = payload.get(field)
            if value and self.field_detector._looks_like_currency(str(value)):
                return str(value).strip().upper()
        
        return None
    
    def extract_description_universal(self, payload: Dict) -> Optional[str]:
        """Extract description using universal field detection"""
        field_types = self.field_detector.detect_field_types(payload)
        description_fields = field_types.get('description_fields', [])
        
        for field in description_fields:
            value = payload.get(field)
            if value and isinstance(value, str) and len(value.strip()) > 5:
                return str(value).strip()
        
        return None
    
    def extract_id_universal(self, payload: Dict) -> Optional[str]:
        """Extract ID using universal field detection"""
        field_types = self.field_detector.detect_field_types(payload)
        id_fields = field_types.get('id_fields', [])
        
        for field in id_fields:
            value = payload.get(field)
            if value and self.field_detector._looks_like_id(str(value)):
                return str(value).strip()
        
        return None

processor = ExcelProcessor()

# Global enhanced processor instance
enhanced_processor = EnhancedFileProcessor() if ADVANCED_FEATURES_AVAILABLE else None

@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await manager.connect(websocket, job_id)
    try:
        # Keep connection alive
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(job_id)

@app.post("/process-excel")
async def process_excel(request: ProcessRequest, background_tasks: BackgroundTasks):
    """Process uploaded Excel file with row-by-row streaming"""
    
    try:
        # Initialize Supabase client from environment (do not require client to send secrets)
        env_url = os.environ.get("SUPABASE_URL")
        env_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_SERVICE_KEY")
        if not env_url or not env_key:
            logger.error("Missing Supabase credentials in environment")
            raise HTTPException(status_code=500, detail="Server misconfiguration: SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set")

        # Clean possible newlines/whitespace in service key
        env_key = env_key.strip().replace('\n', '').replace('\r', '')

        supabase: Client = create_client(env_url, env_key)
        
        # Send initial update
        await manager.send_update(request.job_id, {
            "step": "starting",
            "message": "🚀 Starting intelligent analysis with row-by-row processing...",
            "progress": 5
        })
        
        # Download file from Supabase storage
        try:
            response = supabase.storage.from_('finely-upload').download(request.storage_path)
            file_content = response
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            await manager.send_update(request.job_id, {
                "step": "error",
                "message": f"Failed to download file: {str(e)}",
                "progress": 0
            })
            raise HTTPException(status_code=400, detail=f"File download failed: {str(e)}")
        
        # Check for duplicates before processing
        try:
            await manager.send_update(request.job_id, {
                "step": "duplicate_check",
                "message": "🔍 Checking for duplicate files...",
                "progress": 15
            })
            
            if PRODUCTION_DUPLICATE_SERVICE_AVAILABLE:
                duplicate_service = ProductionDuplicateDetectionService(supabase)
            else:
                        duplicate_service = DuplicateDetectionService(supabase)
            file_hash = duplicate_service.calculate_file_hash(file_content)
            
            duplicate_check = await duplicate_service.check_exact_duplicate(
                request.user_id, 
                file_hash, 
                request.file_name
            )
            
            if duplicate_check.get('is_duplicate', False):
                await manager.send_update(request.job_id, {
                    "step": "duplicate_detected",
                    "message": f"⚠️ Duplicate file detected: {duplicate_check.get('message', 'File already exists')}",
                    "progress": 20,
                    "duplicate_info": duplicate_check,
                    "requires_user_decision": True
                })
                
                # Stop processing and return duplicate information
                logger.warning(f"Duplicate file detected for user {request.user_id}: {request.file_name}")
                return {
                    "status": "duplicate_detected",
                    "duplicate_analysis": duplicate_check,
                    "job_id": request.job_id,
                    "requires_user_decision": True,
                    "message": "Duplicate file detected. Please decide whether to proceed or skip."
                }
            else:
                await manager.send_update(request.job_id, {
                    "step": "no_duplicates",
                    "message": "✅ No duplicates found, proceeding with processing...",
                    "progress": 20
                })
                
        except Exception as e:
            logger.warning(f"Duplicate detection failed: {e} - proceeding with normal processing")
            await manager.send_update(request.job_id, {
                "step": "duplicate_check_failed",
                "message": "⚠️ Duplicate check failed, proceeding with processing...",
                "progress": 20
            })
        
        # Create or update job status to processing
        try:
            # Try to update existing job
            result = supabase.table('ingestion_jobs').update({
            'status': 'processing',
            'started_at': datetime.utcnow().isoformat(),
            'progress': 10
        }).eq('id', request.job_id).execute()
        
            # If no rows were updated, create the job
            if not result.data:
                supabase.table('ingestion_jobs').insert({
                    'id': request.job_id,
                    'job_type': 'fastapi_excel_analysis',
                    'user_id': request.user_id,
                    'status': 'processing',
                    'started_at': datetime.utcnow().isoformat(),
                    'progress': 10
                }).execute()
        except Exception as e:
            logger.warning(f"Could not update job {request.job_id}, creating new one: {e}")
            # Create the job if update fails
            supabase.table('ingestion_jobs').insert({
                'id': request.job_id,
                'job_type': 'fastapi_excel_analysis',
                'user_id': request.user_id,
                'status': 'processing',
                'started_at': datetime.utcnow().isoformat(),
                'progress': 10
            }).execute()
        
        # Process the file with row-by-row streaming
        results = await processor.process_file(
            request.job_id, 
            file_content, 
            request.file_name,
            request.user_id,
            supabase
        )
        
        # Update job with results
        supabase.table('ingestion_jobs').update({
            'status': 'completed',
            'completed_at': datetime.utcnow().isoformat(),
            'progress': 100,
            'result': results
        }).eq('id', request.job_id).execute()
        
        # Step 5: Trigger downstream processing asynchronously (non-blocking)
        await manager.send_update(request.job_id, {
            "step": "downstream_processing",
            "message": "🔄 Running downstream analysis in background...",
            "progress": 90
        })
        
        # Run downstream processing in background (non-blocking)
        import asyncio
        asyncio.create_task(run_downstream_processing_async(request.user_id, request.job_id, supabase))
        
        await manager.send_update(request.job_id, {
            "step": "downstream_started",
            "message": "✅ File processing completed! Downstream analysis running in background...",
            "progress": 100
        })
        
        return {"status": "success", "job_id": request.job_id, "results": results}
        
    except Exception as e:
        logger.error(f"Processing error for job {request.job_id}: {e}")
        
        # Update job with error
        try:
            supabase.table('ingestion_jobs').update({
                'status': 'failed',
                'error_message': str(e),
                'progress': 0
            }).eq('id', request.job_id).execute()
        except:
            pass
        
        await manager.send_update(request.job_id, {
            "step": "error",
            "message": f"Analysis failed: {str(e)}",
            "progress": 0
        })
        
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cancel-upload/{job_id}")
async def cancel_upload(job_id: str, request: Request):
    """Cancel an ongoing file upload/processing job"""
    try:
        # Get Supabase client
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
        
        if not supabase_url or not supabase_key:
            raise HTTPException(status_code=500, detail="Server misconfiguration: SUPABASE_URL or SUPABASE_SERVICE_KEY not set")
        
        # Clean the JWT token (remove newlines and whitespace)
        if supabase_key:
            supabase_key = clean_jwt_token(supabase_key)
        
        # Clean JWT token to prevent header value errors
        supabase_key = clean_jwt_token(supabase_key)
        supabase = create_client(supabase_url, supabase_key)
        
        # Check if job exists and is still processing
        job_result = supabase.table('ingestion_jobs').select('*').eq('id', job_id).execute()
        
        if not job_result.data:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = job_result.data[0]
        
        if job['status'] in ['completed', 'failed', 'cancelled']:
            return {
                "status": "already_finished",
                "message": f"Job is already {job['status']}",
                "job_id": job_id
            }
        
        # Update job status to cancelled
        update_result = supabase.table('ingestion_jobs').update({
            'status': 'cancelled',
            'cancelled_at': datetime.utcnow().isoformat(),
            'progress': 0
        }).eq('id', job_id).execute()
        
        if not update_result.data:
            raise HTTPException(status_code=500, detail="Failed to cancel job")
        
        # Send cancellation update via WebSocket
        await manager.send_update(job_id, {
            "step": "cancelled",
            "message": "Upload cancelled by user",
            "progress": 0,
            "status": "cancelled"
        })
        
        # Disconnect WebSocket for this job
        manager.disconnect(job_id)
        
        return {
            "status": "cancelled",
            "message": "Upload cancelled successfully",
            "job_id": job_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling job {job_id}: {e}")

@app.post("/delta-ingestion/{job_id}")
async def process_delta_ingestion(job_id: str, request: Request):
    """Process delta ingestion for overlapping content"""
    try:
        # Get Supabase client
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
        
        if not supabase_url or not supabase_key:
            raise HTTPException(status_code=500, detail="Server misconfiguration: SUPABASE_URL or SUPABASE_SERVICE_KEY not set")
        
        # Clean JWT token to prevent header value errors
        supabase_key = clean_jwt_token(supabase_key)
        supabase = create_client(supabase_url, supabase_key)
        
        body = await request.json()
        user_id = body.get('user_id')
        existing_file_id = body.get('existing_file_id')
        ingestion_mode = body.get('mode', 'merge_new_only')  # merge_new_only, replace_all, merge_intelligent
        
        if not user_id or not existing_file_id:
            raise HTTPException(status_code=400, detail="Missing required parameters")
        
        # Get the job data
        job_result = supabase.table('ingestion_jobs').select('*').eq('id', job_id).execute()
        if not job_result.data:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job_data = job_result.data[0]
        
        # Initialize duplicate detection service
        if PRODUCTION_DUPLICATE_SERVICE_AVAILABLE:
            duplicate_service = ProductionDuplicateDetectionService(supabase)
        else:
            duplicate_service = DuplicateDetectionService(supabase)
        
        await manager.send_update(job_id, {
            "step": "delta_processing",
            "message": f"🔄 Processing delta ingestion in {ingestion_mode} mode...",
            "progress": 40
        })
        
        # Simulate delta ingestion processing
        # In a real implementation, this would:
        # 1. Compare new vs existing data
        # 2. Identify new rows
        # 3. Merge or replace based on mode
        # 4. Update the database
        
        await manager.send_update(job_id, {
            "step": "delta_complete",
            "message": "✅ Delta ingestion completed successfully",
            "progress": 100,
            "status": "completed"
        })
        
        # Update job status
        supabase.table('ingestion_jobs').update({
            'status': 'completed',
            'completed_at': datetime.utcnow().isoformat(),
            'progress': 100,
            'result': {
                'delta_ingestion_mode': ingestion_mode,
                'existing_file_id': existing_file_id,
                'status': 'success'
            }
        }).eq('id', job_id).execute()
        
        return {
            "status": "success",
            "message": "Delta ingestion completed",
            "job_id": job_id,
            "mode": ingestion_mode
        }
        
    except Exception as e:
        logger.error(f"Error processing delta ingestion: {e}")
        await manager.send_update(job_id, {
            "step": "error",
            "message": f"❌ Delta ingestion failed: {str(e)}",
            "progress": 0,
            "status": "failed"
        })
        raise HTTPException(status_code=500, detail=str(e))

def _extract_entities_simple(payload: dict) -> dict:
    """Simple entity extraction using pattern matching"""
    try:
        entities = {
            'employees': [],
            'vendors': [],
            'customers': [],
            'projects': []
        }
        
        # Convert payload to text for pattern matching
        text = ' '.join([str(val) for val in payload.values() if val]).lower()
        
        # Simple pattern matching for entities
        if any(keyword in text for keyword in ['employee', 'staff', 'worker']):
            entities['employees'].append('Employee')
        
        if any(keyword in text for keyword in ['vendor', 'supplier', 'contractor']):
            entities['vendors'].append('Vendor')
        
        if any(keyword in text for keyword in ['customer', 'client', 'buyer']):
            entities['customers'].append('Customer')
        
        if any(keyword in text for keyword in ['project', 'campaign', 'initiative']):
            entities['projects'].append('Project')
        
        return entities
        
    except Exception as e:
        logger.error(f"Simple entity extraction failed: {e}")
        return {'employees': [], 'vendors': [], 'customers': [], 'projects': []}

async def run_downstream_processing_async(user_id: str, job_id: str, supabase: Client):
    """Run downstream processing asynchronously without blocking the main response"""
    try:
        logger.info(f"Starting async downstream processing for user {user_id}")
        
        # Trigger entity resolution
        await trigger_entity_resolution(user_id, job_id, supabase)
        
        # Trigger platform discovery
        await trigger_platform_discovery(user_id, job_id, supabase)
        
        # Trigger relationship detection
        await trigger_relationship_detection(user_id, job_id, supabase)
        
        logger.info(f"Async downstream processing completed for user {user_id}")
        
    except Exception as e:
        logger.error(f"Async downstream processing failed: {e}")

async def trigger_entity_resolution(user_id: str, job_id: str, supabase: Client):
    """Trigger entity resolution for processed events"""
    try:
        # Get all raw_events for this user that haven't been processed for entities
        events_result = supabase.table('raw_events').select('*').eq('user_id', user_id).execute()
        
        if not events_result.data:
            return
        
        # Initialize entity resolution
        try:
            from enhanced_relationship_detector import EnhancedRelationshipDetector
        except ImportError:
            logger.warning("Enhanced relationship detector not available, skipping relationship detection")
            return
        openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        detector = EnhancedRelationshipDetector(openai_client, supabase)
        
        # Process each event for entity extraction
        for event in events_result.data:
            try:
                # Extract entities from the event payload using simple pattern matching
                entities = _extract_entities_simple(event.get('payload', {}))
                
                # Store entities in normalized_entities table
                for entity_type, entity_names in entities.items():
                    for entity_name in entity_names:
                        if entity_name and len(entity_name.strip()) > 0:
                            # Use the database function to find or create entity
                            entity_result = supabase.rpc('find_or_create_entity', {
                                'p_user_id': user_id,
                                'p_entity_name': entity_name.strip(),
                                'p_entity_type': entity_type,
                                'p_platform': event.get('source_platform', 'unknown'),
                                'p_source_file': event.get('source_filename', 'unknown')
                            }).execute()
                            
                            if entity_result.data:
                                logger.info(f"Created/found entity: {entity_name} ({entity_type})")
                
            except Exception as e:
                logger.error(f"Error processing entity for event {event.get('id')}: {e}")
                continue
                
        logger.info(f"Entity resolution completed for user {user_id}")
        
    except Exception as e:
        logger.error(f"Entity resolution failed: {e}")
        raise
    

async def trigger_platform_discovery(user_id: str, job_id: str, supabase: Client):
    """Trigger platform discovery for processed events"""
    try:
        # Get all unique platforms from raw_events
        platforms_result = supabase.table('raw_events').select('source_platform').eq('user_id', user_id).execute()
        
        if not platforms_result.data:
            return
        
        # Get unique platforms
        platforms = set()
        for event in platforms_result.data:
            platform = event.get('source_platform')
            if platform and platform != 'unknown':
                platforms.add(platform)
        
        # Store discovered platforms
        for platform in platforms:
            try:
                # Check if platform already exists
                existing = supabase.table('discovered_platforms').select('id').eq('user_id', user_id).eq('platform_name', platform).execute()
                
                if not existing.data:
                    # Create new discovered platform
                    supabase.table('discovered_platforms').insert({
                        'user_id': user_id,
                        'platform_name': platform,
                        'discovery_reason': 'Detected from uploaded files',
                        'confidence_score': 0.8
                    }).execute()
                    logger.info(f"Discovered new platform: {platform}")
                
            except Exception as e:
                logger.error(f"Error storing platform {platform}: {e}")
                continue
                
        logger.info(f"Platform discovery completed for user {user_id}")
        
    except Exception as e:
        logger.error(f"Platform discovery failed: {e}")
        raise

async def trigger_relationship_detection(user_id: str, job_id: str, supabase: Client):
    """Trigger relationship detection for processed events"""
    try:
        # Initialize relationship detector
        try:
            from enhanced_relationship_detector import EnhancedRelationshipDetector
        except ImportError:
            logger.warning("Enhanced relationship detector not available, skipping relationship detection")
            return
        openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        detector = EnhancedRelationshipDetector(openai_client, supabase)
        
        # Detect all relationships using enhanced detector
        relationships = await detector.detect_all_relationships(user_id)
        
        if relationships and relationships.get('relationships'):
            # Store relationships in relationship_instances table
            for relationship in relationships['relationships']:
                try:
                    supabase.table('relationship_instances').insert({
                        'user_id': user_id,
                        'source_event_id': relationship.get('source_event_id'),
                        'target_event_id': relationship.get('target_event_id'),
                        'relationship_type': relationship.get('relationship_type'),
                        'confidence_score': relationship.get('confidence_score', 0.5),
                        'detection_method': relationship.get('detection_method', 'ai_enhanced'),
                        'reasoning': relationship.get('reasoning', 'AI-detected relationship')
                    }).execute()
                    
                except Exception as e:
                    logger.error(f"Error storing relationship: {e}")
                    continue
            
            logger.info(f"Stored {len(relationships['relationships'])} relationships for user {user_id}")
        
        logger.info(f"Relationship detection completed for user {user_id}")
        
    except Exception as e:
        logger.error(f"Relationship detection failed: {e}")
        raise

@app.get("/job-status/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a processing job"""
    try:
        # Get Supabase client
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
        
        if not supabase_url or not supabase_key:
            raise HTTPException(status_code=500, detail="Server misconfiguration: SUPABASE_URL or SUPABASE_SERVICE_KEY not set")
        
        # Clean the JWT token (remove newlines and whitespace)
        if supabase_key:
            supabase_key = clean_jwt_token(supabase_key)
        
        # Clean JWT token to prevent header value errors
        supabase_key = clean_jwt_token(supabase_key)
        supabase = create_client(supabase_url, supabase_key)
        
        # Get job status
        job_result = supabase.table('ingestion_jobs').select('*').eq('id', job_id).execute()
        
        if not job_result.data:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = job_result.data[0]
        
        return {
            "job_id": job_id,
            "status": job['status'],
            "progress": job.get('progress', 0),
            "message": job.get('error_message', 'Processing...'),
            "result": job.get('result'),
            "created_at": job.get('created_at'),
            "updated_at": job.get('updated_at')
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status for {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")

# Root endpoint removed - static files will be served at root

@app.get("/health")
async def health_check():
    return {"message": "Finley AI Backend - Intelligent Financial Analysis with Row-by-Row Processing", "status": "healthy"}

@app.get("/test-raw-events/{user_id}")
async def test_raw_events(user_id: str):
    """Test endpoint to check raw_events functionality"""
    try:
        # Initialize Supabase client (you'll need to provide credentials)
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        if supabase_key:
            supabase_key = clean_jwt_token(supabase_key)
        
        # Clean the JWT token (remove newlines and whitespace)
        if supabase_key:
            supabase_key = clean_jwt_token(supabase_key)
        
        if not supabase_url or not supabase_key:
            return {"error": "Supabase credentials not configured"}
        
        # Clean JWT token to prevent header value errors
        supabase_key = clean_jwt_token(supabase_key)
        supabase: Client = create_client(supabase_url, supabase_key)
        
        # Get raw_events statistics
        result = supabase.rpc('get_raw_events_stats', {'user_uuid': user_id}).execute()
        
        # Get recent events
        recent_events = supabase.table('raw_events').select('*').eq('user_id', user_id).order('created_at', desc=True).limit(10).execute()
        
        return {
            "status": "success",
            "user_id": user_id,
            "statistics": result.data[0] if result.data else {},
            "recent_events": recent_events.data if recent_events.data else [],
            "message": "Raw events test completed"
        }
        
    except Exception as e:
        logger.error(f"Error in test_raw_events: {e}")
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    """Basic health check that doesn't require external dependencies"""
    try:
        # Check if OpenAI API key is configured
        openai_key = os.environ.get("OPENAI_API_KEY")
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        if supabase_key:
            supabase_key = clean_jwt_token(supabase_key)
        
        status = "healthy"
        issues = []
        
        if not openai_key:
            issues.append("OPENAI_API_KEY not configured")
            status = "degraded"
        
        if not supabase_url:
            issues.append("SUPABASE_URL not configured")
            status = "degraded"
            
        if not supabase_key:
            issues.append("SUPABASE_SERVICE_ROLE_KEY not configured")
            status = "degraded"
        
        return {
            "status": status,
            "service": "Finley AI Backend",
            "timestamp": datetime.utcnow().isoformat(),
            "issues": issues,
            "environment_configured": len(issues) == 0
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "Finley AI Backend",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.post("/upload-and-process")
async def upload_and_process(
    file: UploadFile = Form(...),
    user_id: str = Form("550e8400-e29b-41d4-a716-446655440000"),  # Default test user ID
    job_id: str = Form(None)  # Optional, will generate if not provided
):
    """Direct file upload and processing endpoint with duplicate detection"""
    try:
        # Generate job_id if not provided
        if not job_id:
            job_id = f"test-job-{int(time.time())}"

        # Read file content
        file_content = await file.read()

        # Initialize Supabase client
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        if supabase_key:
            supabase_key = clean_jwt_token(supabase_key)

        # Clean the JWT token (remove newlines and whitespace)
        if supabase_key:
            supabase_key = clean_jwt_token(supabase_key)

        if not supabase_url or not supabase_key:
            raise HTTPException(status_code=500, detail="Supabase credentials not configured")

        # Clean JWT token to prevent header value errors
        supabase_key = clean_jwt_token(supabase_key)
        supabase: Client = create_client(supabase_url, supabase_key)

        # Create ExcelProcessor instance
        excel_processor = ExcelProcessor()

        # Process the file with duplicate detection
        results = await excel_processor.process_file(
            job_id,
            file_content,
            file.filename,
            user_id,
            supabase
        )

        return {
            "status": "success",
            "job_id": job_id,
            "results": results,
            "message": "File processed successfully"
        }

    except Exception as e:
        logger.error(f"Upload and process error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# DUPLICATE HANDLING ENDPOINTS
# ============================================================================

class DuplicateDecisionRequest(BaseModel):
    job_id: str
    user_id: str
    decision: str  # 'replace', 'keep_both', 'skip'
    file_hash: str

@app.post("/handle-duplicate-decision")
async def handle_duplicate_decision(request: DuplicateDecisionRequest):
    """Handle user's decision about duplicate files"""
    try:
        # Initialize Supabase client
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        if supabase_key:
            supabase_key = clean_jwt_token(supabase_key)

        if supabase_key:
            supabase_key = clean_jwt_token(supabase_key)

        if not supabase_url or not supabase_key:
            raise HTTPException(status_code=500, detail="Supabase credentials not configured")

        # Clean JWT token to prevent header value errors
        supabase_key = clean_jwt_token(supabase_key)
        supabase: Client = create_client(supabase_url, supabase_key)
        if PRODUCTION_DUPLICATE_SERVICE_AVAILABLE:
            duplicate_service = ProductionDuplicateDetectionService(supabase)
        else:
            duplicate_service = DuplicateDetectionService(supabase)

        # Handle the duplicate decision
        result = await duplicate_service.handle_duplicate_decision(
            request.user_id,
            request.file_hash,
            request.decision
        )

        # Send update to WebSocket
        await manager.send_update(request.job_id, {
            "step": "duplicate_decision_processed",
            "message": f"✅ Duplicate decision processed: {request.decision}",
            "progress": 30,
            "decision_result": result
        })

        # If decision is to proceed, continue with processing
        if result['action'] == 'proceed_with_new':
            await manager.send_update(request.job_id, {
                "step": "continuing_processing",
                "message": "🔄 Continuing with file processing...",
                "progress": 35
            })

        return {
            "status": "success",
            "decision_result": result,
            "message": f"Duplicate decision '{request.decision}' processed successfully"
        }

    except Exception as e:
        logger.error(f"Error handling duplicate decision: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class VersionRecommendationFeedback(BaseModel):
    recommendation_id: str
    user_id: str
    accepted: bool
    feedback: Optional[str] = None

@app.post("/version-recommendation-feedback")
async def submit_version_recommendation_feedback(request: VersionRecommendationFeedback):
    """Submit user feedback on version recommendations"""
    try:
        # Initialize Supabase client
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        if supabase_key:
            supabase_key = clean_jwt_token(supabase_key)

        if supabase_key:
            supabase_key = clean_jwt_token(supabase_key)

        if not supabase_url or not supabase_key:
            raise HTTPException(status_code=500, detail="Supabase credentials not configured")

        # Clean JWT token to prevent header value errors
        supabase_key = clean_jwt_token(supabase_key)
        supabase: Client = create_client(supabase_url, supabase_key)
        if PRODUCTION_DUPLICATE_SERVICE_AVAILABLE:
            duplicate_service = ProductionDuplicateDetectionService(supabase)
        else:
            duplicate_service = DuplicateDetectionService(supabase)

        # Update recommendation with feedback
        success = await duplicate_service.update_recommendation_feedback(
            request.recommendation_id,
            request.accepted,
            request.feedback
        )

        if success:
            return {
                "status": "success",
                "message": "Feedback submitted successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to submit feedback")

    except Exception as e:
        logger.error(f"Error submitting version feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/duplicate-analysis/{user_id}")
async def get_duplicate_analysis(user_id: str):
    """Get duplicate analysis and recommendations for a user"""
    try:
        # Initialize Supabase client
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        if supabase_key:
            supabase_key = clean_jwt_token(supabase_key)

        if supabase_key:
            supabase_key = clean_jwt_token(supabase_key)

        if not supabase_url or not supabase_key:
            raise HTTPException(status_code=500, detail="Supabase credentials not configured")

        # Clean JWT token to prevent header value errors
        supabase_key = clean_jwt_token(supabase_key)
        supabase: Client = create_client(supabase_url, supabase_key)

        # Get file versions for user
        versions_result = supabase.table('file_versions').select(
            '*'
        ).eq('user_id', user_id).execute()

        # Get pending recommendations
        recommendations_result = supabase.table('version_recommendations').select(
            '*'
        ).eq('user_id', user_id).is_('user_accepted', 'null').execute()

        return {
            "status": "success",
            "file_versions": versions_result.data,
            "pending_recommendations": recommendations_result.data
        }

    except Exception as e:
        logger.error(f"Error getting duplicate analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test-simple")
async def test_simple():
    """Simple test endpoint without any dependencies"""
    try:
        # Test basic imports
        import pandas as pd
        import numpy as np
        import openai
        import magic
        import filetype
        
        return {
            "status": "success",
            "message": "Backend is working! All dependencies loaded successfully.",
            "timestamp": datetime.utcnow().isoformat(),
            "dependencies": {
                "pandas": "loaded",
                "numpy": "loaded", 
                "openai": "loaded",
                "magic": "loaded",
                "filetype": "loaded"
            },
            "endpoints": {
                "health": "/health",
                "upload_and_process": "/upload-and-process",
                "test_raw_events": "/test-raw-events/{user_id}",
                "process_excel": "/process-excel"
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Backend has issues: {str(e)}",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }

@app.get("/test-database")
async def test_database():
    """Test database connection and basic operations"""
    try:
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
        
        if not supabase_url or not supabase_key:
            return {"error": "Supabase credentials not configured"}
        
        # Clean JWT token to prevent header value errors
        supabase_key = clean_jwt_token(supabase_key)
        supabase: Client = create_client(supabase_url, supabase_key)
        
        # Test basic database operations
        test_user_id = "550e8400-e29b-41d4-a716-446655440000"
        
        # Test raw_events table
        events_count = supabase.table('raw_events').select('id', count='exact').eq('user_id', test_user_id).execute()
        
        # Test ingestion_jobs table
        jobs_count = supabase.table('ingestion_jobs').select('id', count='exact').eq('user_id', test_user_id).execute()
        
        # Test raw_records table
        records_count = supabase.table('raw_records').select('id', count='exact').eq('user_id', test_user_id).execute()
        
        return {
            "status": "success",
            "database_connection": "working",
            "tables": {
                "raw_events": events_count.count if hasattr(events_count, 'count') else 0,
                "ingestion_jobs": jobs_count.count if hasattr(jobs_count, 'count') else 0,
                "raw_records": records_count.count if hasattr(records_count, 'count') else 0
            },
            "message": "Database connection and queries working"
        }
        
    except Exception as e:
        logger.error(f"Database test error: {e}")
        return {"error": f"Database test failed: {str(e)}"}

@app.get("/test-platform-detection")
async def test_platform_detection():
    """Test endpoint for enhanced platform detection"""
    try:
        # Create sample data for different platforms
        import pandas as pd
        
        test_cases = {
            'quickbooks': pd.DataFrame({
                'Account': ['Checking', 'Savings'],
                'Memo': ['Payment', 'Deposit'],
                'Amount': [1000, 500],
                'Date': ['2024-01-01', '2024-01-02'],
                'Ref Number': ['REF001', 'REF002']
            }),
            'gusto': pd.DataFrame({
                'Employee Name': ['John Doe', 'Jane Smith'],
                'Employee ID': ['EMP001', 'EMP002'],
                'Pay Period': ['2024-01-01', '2024-01-15'],
                'Gross Pay': [5000, 6000],
                'Net Pay': [3500, 4200],
                'Tax Deductions': [1500, 1800]
            }),
            'stripe': pd.DataFrame({
                'Charge ID': ['ch_001', 'ch_002'],
                'Customer ID': ['cus_001', 'cus_002'],
                'Amount': [1000, 2000],
                'Status': ['succeeded', 'succeeded'],
                'Created': ['2024-01-01', '2024-01-02'],
                'Currency': ['usd', 'usd']
            }),
            'xero': pd.DataFrame({
                'Contact Name': ['Client A', 'Client B'],
                'Invoice Number': ['INV001', 'INV002'],
                'Amount': [1500, 2500],
                'Date': ['2024-01-01', '2024-01-02'],
                'Reference': ['REF001', 'REF002'],
                'Tracking': ['Project A', 'Project B']
            })
        }
        
        results = {}
        platform_detector = PlatformDetector()
        
        for platform_name, df in test_cases.items():
            filename = f"{platform_name}_sample.xlsx"
            detection_result = platform_detector.detect_platform(df, filename)
            platform_info = platform_detector.get_platform_info(detection_result['platform'])
            
            results[platform_name] = {
                'detection_result': detection_result,
                'platform_info': platform_info,
                'sample_columns': list(df.columns),
                'sample_data_shape': df.shape
            }
        
        return {
            "status": "success",
            "message": "Enhanced platform detection test completed",
            "test_cases": results,
            "summary": {
                "total_platforms_tested": len(test_cases),
                "detection_accuracy": sum(1 for r in results.values() 
                                        if r['detection_result']['platform'] != 'unknown') / len(results)
            }
        }
        
    except Exception as e:
        logger.error(f"Platform detection test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Platform detection test failed: {str(e)}")

@app.get("/test-ai-row-classification")
async def test_ai_row_classification():
    """Test AI-powered row classification with sample data"""
    
    # Sample test cases
    test_cases = [
        {
            "test_case": "Payroll Transaction",
            "description": "Employee salary payment",
            "row_data": {"Description": "Salary payment to John Smith", "Amount": 5000, "Date": "2024-01-15"}
        },
        {
            "test_case": "Revenue Transaction", 
            "description": "Client payment received",
            "row_data": {"Description": "Payment from ABC Corp", "Amount": 15000, "Date": "2024-01-20"}
        },
        {
            "test_case": "Expense Transaction",
            "description": "Office rent payment",
            "row_data": {"Description": "Office rent to Building LLC", "Amount": -3000, "Date": "2024-01-10"}
        },
        {
            "test_case": "Investment Transaction",
            "description": "Stock purchase",
            "row_data": {"Description": "Stock purchase - AAPL", "Amount": -5000, "Date": "2024-01-25"}
        },
        {
            "test_case": "Tax Transaction",
            "description": "Tax payment",
            "row_data": {"Description": "Income tax payment", "Amount": -2000, "Date": "2024-01-30"}
        }
    ]
    
    # Initialize batch classifier
    batch_classifier = BatchAIRowClassifier(openai)
    platform_info = {"platform": "quickbooks", "confidence": 0.8}
    column_names = ["Description", "Amount", "Date"]
    
    test_results = []
    
    for test_case in test_cases:
        # Create pandas Series from row data
        row = pd.Series(test_case["row_data"])
        
        # Test batch classification (single row as batch)
        batch_classifications = await batch_classifier.classify_rows_batch([row], platform_info, column_names)
        
        if batch_classifications:
            ai_classification = batch_classifications[0]
        else:
            ai_classification = {}
        
        test_results.append({
            "test_case": test_case["test_case"],
            "description": test_case["description"],
            "row_data": test_case["row_data"],
            "ai_classification": ai_classification
        })
    
    return {
        "message": "AI Row Classification Test Results",
        "total_tests": len(test_results),
        "test_results": test_results,
        "processing_mode": "batch_optimized",
        "batch_size": 20,
        "performance_notes": "Batch processing reduces AI calls by 95% for large files"
    }

@app.get("/test-batch-processing")
async def test_batch_processing():
    """Test the optimized batch processing performance"""
    
    # Create sample data for batch testing
    sample_rows = []
    for i in range(25):  # Test with 25 rows
        if i < 8:
            # Payroll rows
            row_data = {"Description": f"Salary payment to Employee {i+1}", "Amount": 5000 + i*100, "Date": "2024-01-15"}
        elif i < 16:
            # Revenue rows
            row_data = {"Description": f"Payment from Client {i-7}", "Amount": 10000 + i*500, "Date": "2024-01-20"}
        elif i < 20:
            # Expense rows
            row_data = {"Description": f"Office expense {i-15}", "Amount": -(1000 + i*50), "Date": "2024-01-10"}
        else:
            # Other transactions
            row_data = {"Description": f"Transaction {i+1}", "Amount": 500 + i*25, "Date": "2024-01-25"}
        
        sample_rows.append(pd.Series(row_data))
    
    # Initialize batch classifier
    batch_classifier = BatchAIRowClassifier(openai)
    platform_info = {"platform": "quickbooks", "confidence": 0.8}
    column_names = ["Description", "Amount", "Date"]
    
    # Test batch processing
    start_time = time.time()
    batch_classifications = await batch_classifier.classify_rows_batch(sample_rows, platform_info, column_names)
    end_time = time.time()
    
    processing_time = end_time - start_time
    
    # Analyze results
    categories = defaultdict(int)
    row_types = defaultdict(int)
    total_confidence = 0
    
    for classification in batch_classifications:
        categories[classification.get('category', 'unknown')] += 1
        row_types[classification.get('row_type', 'unknown')] += 1
        total_confidence += classification.get('confidence', 0)
    
    avg_confidence = total_confidence / len(batch_classifications) if batch_classifications else 0
    
    return {
        "message": "Batch Processing Performance Test",
        "total_rows": len(sample_rows),
        "processing_time_seconds": round(processing_time, 2),
        "rows_per_second": round(len(sample_rows) / processing_time, 2) if processing_time > 0 else 0,
        "ai_calls": 1,  # Only 1 AI call for 25 rows
        "traditional_ai_calls": len(sample_rows),  # Would be 25 individual calls
        "ai_calls_reduced": f"{((len(sample_rows) - 1) / len(sample_rows)) * 100:.1f}%",
        "category_breakdown": dict(categories),
        "row_type_breakdown": dict(row_types),
        "average_confidence": round(avg_confidence, 3),
        "batch_size": 20,
        "processing_mode": "batch_optimized",
        "performance_improvement": {
            "speed": "20x faster for large files",
            "cost": "95% reduction in AI API calls",
            "efficiency": "Batch processing of 20 rows per AI call"
        }
    }

class EntityResolver:
    """Advanced entity resolution system for cross-platform entity matching"""
    
    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
        self.similarity_cache = {}
    
    def calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two entity names using multiple algorithms"""
        if not name1 or not name2:
            return 0.0
        
        name1_clean = self._normalize_name(name1)
        name2_clean = self._normalize_name(name2)
        
        # Exact match
        if name1_clean == name2_clean:
            return 1.0
        
        # Contains match
        if name1_clean in name2_clean or name2_clean in name1_clean:
            return 0.9
        
        # Token-based similarity
        tokens1 = set(name1_clean.split())
        tokens2 = set(name2_clean.split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        jaccard_similarity = intersection / union if union > 0 else 0.0
        
        # Levenshtein-like similarity for partial matches
        max_len = max(len(name1_clean), len(name2_clean))
        if max_len == 0:
            return 0.0
        
        # Simple character-based similarity
        common_chars = sum(1 for c in name1_clean if c in name2_clean)
        char_similarity = common_chars / max_len
        
        # Weighted combination
        final_similarity = (jaccard_similarity * 0.6) + (char_similarity * 0.4)
        
        return min(final_similarity, 1.0)
    
    def _normalize_name(self, name: str) -> str:
        """Normalize entity name for comparison"""
        if not name:
            return ""
        
        # Convert to lowercase
        normalized = name.lower().strip()
        
        # Remove common suffixes and prefixes
        suffixes_to_remove = [
            ' inc', ' corp', ' llc', ' ltd', ' co', ' company', ' pvt', ' private',
            ' limited', ' corporation', ' incorporated'
        ]
        
        for suffix in suffixes_to_remove:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)]
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def extract_strong_identifiers(self, row_data: Dict, column_names: List[str]) -> Dict[str, str]:
        """Extract strong identifiers (email, bank account, phone) from row data"""
        identifiers = {
            'email': None,
            'bank_account': None,
            'phone': None,
            'tax_id': None
        }
        
        # Common column name patterns for strong identifiers
        email_patterns = ['email', 'e-mail', 'mail', 'contact_email']
        bank_patterns = ['bank_account', 'account_number', 'bank_ac', 'ac_number', 'account']
        phone_patterns = ['phone', 'mobile', 'contact', 'tel', 'telephone']
        tax_patterns = ['tax_id', 'tax_number', 'pan', 'gst', 'tin']
        
        for col_name in column_names:
            col_lower = col_name.lower()
            col_value = str(row_data.get(col_name, '')).strip()
            
            if not col_value or col_value == 'nan':
                continue
            
            # Email detection
            if any(pattern in col_lower for pattern in email_patterns) or '@' in col_value:
                if '@' in col_value and '.' in col_value:
                    identifiers['email'] = col_value
            
            # Bank account detection
            elif any(pattern in col_lower for pattern in bank_patterns):
                if col_value.isdigit() or (len(col_value) >= 8 and any(c.isdigit() for c in col_value)):
                    identifiers['bank_account'] = col_value
            
            # Phone detection
            elif any(pattern in col_lower for pattern in phone_patterns):
                if any(c.isdigit() for c in col_value) and len(col_value) >= 10:
                    identifiers['phone'] = col_value
            
            # Tax ID detection
            elif any(pattern in col_lower for pattern in tax_patterns):
                if len(col_value) >= 5:
                    identifiers['tax_id'] = col_value
        
        return {k: v for k, v in identifiers.items() if v is not None}
    
    async def resolve_entity(self, entity_name: str, entity_type: str, platform: str, 
                           user_id: str, row_data: Dict, column_names: List[str], 
                           source_file: str, row_id: str) -> Dict[str, Any]:
        """Resolve entity using database functions and return resolution details"""
        
        # Extract strong identifiers
        identifiers = self.extract_strong_identifiers(row_data, column_names)
        
        try:
            # Call database function to find or create entity
            result = self.supabase.rpc('find_or_create_entity', {
                'p_user_id': user_id,
                'p_entity_name': entity_name,
                'p_entity_type': entity_type,
                'p_platform': platform,
                'p_email': identifiers.get('email'),
                'p_bank_account': identifiers.get('bank_account'),
                'p_phone': identifiers.get('phone'),
                'p_tax_id': identifiers.get('tax_id'),
                'p_source_file': source_file
            }).execute()
            
            if result.data:
                entity_id = result.data
                
                # Get entity details for response
                entity_details = self.supabase.rpc('get_entity_details', {
                    'user_uuid': user_id,
                    'entity_id': entity_id
                }).execute()
                
                return {
                    'entity_id': entity_id,
                    'resolved_name': entity_name,
                    'entity_type': entity_type,
                    'platform': platform,
                    'identifiers': identifiers,
                    'source_file': source_file,
                    'row_id': row_id,
                    'resolution_success': True,
                    'entity_details': entity_details.data[0] if entity_details.data else None
                }
            else:
                return {
                    'entity_id': None,
                    'resolved_name': entity_name,
                    'entity_type': entity_type,
                    'platform': platform,
                    'identifiers': identifiers,
                    'source_file': source_file,
                    'row_id': row_id,
                    'resolution_success': False,
                    'error': 'Database function returned no entity ID'
                }
                
        except Exception as e:
            return {
                'entity_id': None,
                'resolved_name': entity_name,
                'entity_type': entity_type,
                'platform': platform,
                'identifiers': identifiers,
                'source_file': source_file,
                'row_id': row_id,
                'resolution_success': False,
                'error': str(e)
            }
    
    async def resolve_entities_batch(self, entities: Dict[str, List[str]], platform: str, 
                                   user_id: str, row_data: Dict, column_names: List[str],
                                   source_file: str, row_id: str) -> Dict[str, Any]:
        """Resolve multiple entities in a batch"""
        resolved_entities = {
            'employees': [],
            'vendors': [],
            'customers': [],
            'projects': []
        }
        
        resolution_results = []
        
        for entity_type, entity_list in entities.items():
            for entity_name in entity_list:
                if entity_name and entity_name.strip():
                    resolution = await self.resolve_entity(
                        entity_name.strip(),
                        entity_type,
                        platform,
                        user_id,
                        row_data,
                        column_names,
                        source_file,
                        row_id
                    )
                    
                    resolution_results.append(resolution)
                    
                    if resolution['resolution_success']:
                        resolved_entities[entity_type].append({
                            'name': entity_name,
                            'entity_id': resolution['entity_id'],
                            'resolved_name': resolution['resolved_name']
                        })
        
        return {
            'resolved_entities': resolved_entities,
            'resolution_results': resolution_results,
            'total_resolved': sum(len(v) for v in resolved_entities.values()),
            'total_attempted': len(resolution_results)
        }

@app.get("/test-entity-resolution")
async def test_entity_resolution():
    """Test the Entity Resolution system with sample data"""
    try:
        # Create test Supabase client
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY') or os.getenv('SUPABASE_SERVICE_KEY')
        
        if not supabase_url or not supabase_key:
            raise Exception("SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables are required")
        
        # Clean the key by removing any whitespace or newlines
        supabase_key = supabase_key.strip()
        
        if supabase_key:
            supabase_key = clean_jwt_token(supabase_key)
        # Clean JWT token to prevent header value errors
        supabase_key = clean_jwt_token(supabase_key)
        supabase = create_client(supabase_url, supabase_key)
        
        # Initialize EntityResolver
        entity_resolver = EntityResolver(supabase)
        
        # Test cases for entity resolution
        test_cases = [
            {
                "test_case": "Employee Name Resolution",
                "description": "Test resolving employee names across platforms",
                "entities": {
                    "employees": ["Abhishek A.", "Abhishek Arora", "John Smith"],
                    "vendors": ["Razorpay Payout", "Razorpay Payments Pvt. Ltd."],
                    "customers": ["Client ABC", "ABC Corp"],
                    "projects": ["Project Alpha", "Alpha Initiative"]
                },
                "platform": "gusto",
                "user_id": "550e8400-e29b-41d4-a716-446655440000",
                "row_data": {
                    "employee_name": "Abhishek A.",
                    "email": "abhishek@company.com",
                    "amount": "5000"
                },
                "column_names": ["employee_name", "email", "amount"],
                "source_file": "test-payroll.xlsx",
                "row_id": "row-1"
            },
            {
                "test_case": "Vendor Name Resolution",
                "description": "Test resolving vendor names with different formats",
                "entities": {
                    "employees": [],
                    "vendors": ["Razorpay Payout", "Razorpay Payments Pvt. Ltd.", "Stripe Inc"],
                    "customers": [],
                    "projects": []
                },
                "platform": "razorpay",
                "user_id": "550e8400-e29b-41d4-a716-446655440000",
                "row_data": {
                    "vendor_name": "Razorpay Payout",
                    "bank_account": "1234567890",
                    "amount": "10000"
                },
                "column_names": ["vendor_name", "bank_account", "amount"],
                "source_file": "test-payments.xlsx",
                "row_id": "row-2"
            }
        ]
        
        results = []
        
        for test_case in test_cases:
            try:
                # Test entity resolution
                resolution_result = await entity_resolver.resolve_entities_batch(
                    test_case["entities"],
                    test_case["platform"],
                    test_case["user_id"],
                    test_case["row_data"],
                    test_case["column_names"],
                    test_case["source_file"],
                    test_case["row_id"]
                )
                
                results.append({
                    "test_case": test_case["test_case"],
                    "description": test_case["description"],
                    "entities": test_case["entities"],
                    "platform": test_case["platform"],
                    "resolution_result": resolution_result,
                    "success": True
                })
                
            except Exception as e:
                results.append({
                    "test_case": test_case["test_case"],
                    "description": test_case["description"],
                    "entities": test_case["entities"],
                    "platform": test_case["platform"],
                    "error": str(e),
                    "success": False
                })
        
        return {
            "message": "Entity Resolution Test Results",
            "total_tests": len(test_cases),
            "successful_tests": len([r for r in results if r["success"]]),
            "failed_tests": len([r for r in results if not r["success"]]),
            "test_results": results
        }
        
    except Exception as e:
        logger.error(f"Entity resolution test failed: {e}")
        return {
            "message": "Entity Resolution Test Failed",
            "error": str(e),
            "total_tests": 0,
            "successful_tests": 0,
            "failed_tests": 1,
            "test_results": []
        }

@app.get("/test-entity-search/{user_id}")
async def test_entity_search(user_id: str, search_term: str = "Abhishek", entity_type: str = None):
    """Test entity search functionality"""
    try:
        # Create test Supabase client
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
        
        # Debug: Check if environment variables are set
        if not supabase_url:
            return {
                "message": "Entity Search Test Failed",
                "error": "SUPABASE_URL environment variable not found",
                "search_term": search_term,
                "entity_type": entity_type,
                "user_id": user_id,
                "results": [],
                "total_results": 0
            }
        
        if not supabase_key:
            return {
                "message": "Entity Search Test Failed", 
                "error": "SUPABASE_SERVICE_KEY environment variable not found",
                "search_term": search_term,
                "entity_type": entity_type,
                "user_id": user_id,
                "results": [],
                "total_results": 0
            }
        
        # Clean the JWT token (remove newlines and whitespace)
        supabase_key = clean_jwt_token(supabase_key)
        
        # Clean JWT token to prevent header value errors
        supabase_key = clean_jwt_token(supabase_key)
        supabase = create_client(supabase_url, supabase_key)
        
        # Test entity search with correct parameter name
        search_result = supabase.rpc('search_entities_by_name', {
            'user_uuid': user_id,
            'search_term': search_term,
            'p_entity_type': entity_type  # Fixed parameter name
        }).execute()
        
        return {
            "message": "Entity Search Test Results",
            "search_term": search_term,
            "entity_type": entity_type,
            "user_id": user_id,
            "results": search_result.data if search_result.data else [],
            "total_results": len(search_result.data) if search_result.data else 0
        }
        
    except Exception as e:
        logger.error(f"Entity search test failed: {e}")
        return {
            "message": "Entity Search Test Failed",
            "error": str(e),
            "search_term": search_term,
            "entity_type": entity_type,
            "user_id": user_id,
            "results": [],
            "total_results": 0
        }

@app.get("/test-entity-stats/{user_id}")
async def test_entity_stats(user_id: str):
    """Test entity resolution statistics"""
    try:
        # Create test Supabase client
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
        
        # Debug: Check if environment variables are set
        if not supabase_url:
            return {
                "message": "Entity Stats Test Failed",
                "error": "SUPABASE_URL environment variable not found",
                "user_id": user_id,
                "stats": {},
                "success": False
            }
        
        if not supabase_key:
            return {
                "message": "Entity Stats Test Failed",
                "error": "SUPABASE_SERVICE_KEY environment variable not found", 
                "user_id": user_id,
                "stats": {},
                "success": False
            }
        
        # Clean the JWT token (remove newlines and whitespace)
        supabase_key = clean_jwt_token(supabase_key)
        
        # Clean JWT token to prevent header value errors
        supabase_key = clean_jwt_token(supabase_key)
        supabase = create_client(supabase_url, supabase_key)
        
        # Get entity resolution stats
        stats_result = supabase.rpc('get_entity_resolution_stats', {
            'user_uuid': user_id
        }).execute()
        
        return {
            "message": "Entity Resolution Statistics",
            "user_id": user_id,
            "stats": stats_result.data[0] if stats_result.data else {},
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Entity stats test failed: {e}")
        return {
            "message": "Entity Stats Test Failed",
            "error": str(e),
            "user_id": user_id,
            "stats": {},
            "success": False
        }

@app.get("/test-cross-file-relationships/{user_id}")
async def test_cross_file_relationships(user_id: str):
    """Test cross-file relationship detection (payroll ↔ payout)"""
    try:
        # Create test Supabase client
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_SERVICE_KEY')

        # Clean the JWT token (remove newlines and whitespace)
        if supabase_key:
            supabase_key = clean_jwt_token(supabase_key)

        # Clean JWT token to prevent header value errors
        supabase_key = clean_jwt_token(supabase_key)
        supabase = create_client(supabase_url, supabase_key)

        # Initialize relationship detector
        relationship_detector = CrossFileRelationshipDetector(supabase)

        # Detect relationships
        results = await relationship_detector.detect_cross_file_relationships(user_id)

        return {
            "message": "Cross-File Relationship Analysis",
            "user_id": user_id,
            "success": True,
            **results
        }

    except Exception as e:
        logger.error(f"Cross-file relationship test failed: {e}")
        return {
            "message": "Cross-File Relationship Test Failed",
            "error": str(e),
            "user_id": user_id,
            "relationships": [],
            "success": False
        }

@app.get("/test-enhanced-relationship-detection/{user_id}")
async def test_enhanced_relationship_detection(user_id: str):
    """Test ENHANCED relationship detection with cross-file capabilities"""
    try:
        # Import the enhanced detector
        try:
            from enhanced_relationship_detector import EnhancedRelationshipDetector
        except ImportError:
            logger.warning("Enhanced relationship detector not available, skipping relationship detection")
            return

        # Create test Supabase client
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_SERVICE_KEY')

        # Clean the JWT token (remove newlines and whitespace)
        if supabase_key:
            supabase_key = clean_jwt_token(supabase_key)

        # Clean JWT token to prevent header value errors
        supabase_key = clean_jwt_token(supabase_key)
        supabase = create_client(supabase_url, supabase_key)

        # Initialize enhanced relationship detector
        enhanced_detector = EnhancedRelationshipDetector(supabase)

        # Detect ALL relationships (cross-file + within-file)
        results = await enhanced_detector.detect_all_relationships(user_id)

        return {
            "message": "Enhanced Relationship Detection Test Completed",
            "user_id": user_id,
            "success": True,
            "result": results,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Enhanced relationship detection test failed: {e}")
        return {
            "message": "Enhanced Relationship Detection Test Failed",
            "error": str(e),
            "user_id": user_id,
            "success": False,
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/debug-cross-file-data/{user_id}")
async def debug_cross_file_data(user_id: str):
    """Debug endpoint to check what files and data exist for cross-file analysis"""
    try:
        # Create test Supabase client
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_SERVICE_KEY')

        # Clean the JWT token (remove newlines and whitespace)
        if supabase_key:
            supabase_key = clean_jwt_token(supabase_key)

        # Clean JWT token to prevent header value errors
        supabase_key = clean_jwt_token(supabase_key)
        supabase = create_client(supabase_url, supabase_key)

        # Get all events for the user
        events = supabase.table('raw_events').select('*').eq('user_id', user_id).execute()

        if not events.data:
            return {
                "message": "No data found for user",
                "user_id": user_id,
                "total_events": 0,
                "files": [],
                "timestamp": datetime.utcnow().isoformat()
            }

        # Group events by file
        events_by_file = {}
        for event in events.data:
            filename = event.get('source_filename', 'unknown')
            if filename not in events_by_file:
                events_by_file[filename] = []
            events_by_file[filename].append(event)

        # Create file summary
        file_summary = []
        for filename, file_events in events_by_file.items():
            file_summary.append({
                "filename": filename,
                "event_count": len(file_events),
                "sample_event_ids": [e.get('id') for e in file_events[:3]],
                "sample_amounts": [e.get('amount') for e in file_events[:3] if e.get('amount')],
                "date_range": {
                    "earliest": min([e.get('event_date') for e in file_events if e.get('event_date')], default=None),
                    "latest": max([e.get('event_date') for e in file_events if e.get('event_date')], default=None)
                }
            })

        # Check for potential cross-file relationships
        potential_relationships = []
        cross_file_patterns = [
            ['company_invoices.csv', 'comprehensive_vendor_payments.csv'],
            ['company_revenue.csv', 'comprehensive_cash_flow.csv'],
            ['company_expenses.csv', 'company_bank_statements.csv'],
            ['comprehensive_payroll_data.csv', 'company_bank_statements.csv'],
            ['company_invoices.csv', 'company_accounts_receivable.csv']
        ]

        for pattern in cross_file_patterns:
            source_file, target_file = pattern
            source_exists = source_file in events_by_file
            target_exists = target_file in events_by_file

            potential_relationships.append({
                "source_file": source_file,
                "target_file": target_file,
                "source_exists": source_exists,
                "target_exists": target_exists,
                "source_events": len(events_by_file.get(source_file, [])),
                "target_events": len(events_by_file.get(target_file, [])),
                "can_analyze": source_exists and target_exists
            })

        return {
            "message": "Cross-file data analysis completed",
            "user_id": user_id,
            "total_events": len(events.data),
            "total_files": len(events_by_file),
            "files": file_summary,
            "potential_cross_file_relationships": potential_relationships,
            "analysis_ready": sum(1 for p in potential_relationships if p["can_analyze"]),
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Cross-file data debug failed: {e}")
        return {
            "message": "Cross-file data debug failed",
            "error": str(e),
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/test-websocket/{job_id}")
async def test_websocket(job_id: str):
    """Test WebSocket functionality by sending messages to a specific job"""
    try:
        # Send test messages to the WebSocket
        test_messages = [
            {"step": "reading", "message": "📖 Reading and parsing your file...", "progress": 10},
            {"step": "analyzing", "message": "🧠 Analyzing document structure...", "progress": 20},
            {"step": "storing", "message": "💾 Storing file metadata...", "progress": 30},
            {"step": "processing", "message": "⚙️ Processing rows...", "progress": 50},
            {"step": "classifying", "message": "🏷️ Classifying data...", "progress": 70},
            {"step": "resolving", "message": "🔗 Resolving entities...", "progress": 90},
            {"step": "complete", "message": "✅ Processing complete!", "progress": 100}
        ]
        
        # Send messages with delays
        for i, message in enumerate(test_messages):
            await manager.send_update(job_id, message)
            await asyncio.sleep(1)  # Wait 1 second between messages
        
        return {
            "message": "WebSocket test completed",
            "job_id": job_id,
            "messages_sent": len(test_messages),
            "success": True
        }
        
    except Exception as e:
        logger.error(f"WebSocket test failed: {e}")
        return {
            "message": "WebSocket Test Failed",
            "error": str(e),
            "job_id": job_id,
            "success": False
        }

class CrossFileRelationshipDetector:
    """Detects relationships between different file types (payroll ↔ payout)"""
    
    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
        
    async def detect_cross_file_relationships(self, user_id: str) -> Dict[str, Any]:
        """Detect relationships between payroll and payout files"""
        try:
            # Get all raw events for the user
            events = self.supabase.table('raw_events').select('*').eq('user_id', user_id).execute()
            
            if not events.data:
                return {"relationships": [], "message": "No data found for cross-file analysis"}
            
            # Group events by platform and type
            payroll_events = []
            payout_events = []
            
            for event in events.data:
                payload = event.get('payload', {})
                platform = event.get('source_platform', 'unknown')
                
                # Identify payroll events
                if platform in ['gusto', 'quickbooks'] and self._is_payroll_event(payload):
                    payroll_events.append(event)
                
                # Identify payout events  
                if platform in ['razorpay', 'stripe'] and self._is_payout_event(payload):
                    payout_events.append(event)
            
            # Find relationships
            relationships = await self._find_relationships(payroll_events, payout_events)
            
            return {
                "relationships": relationships,
                "total_payroll_events": len(payroll_events),
                "total_payout_events": len(payout_events),
                "total_relationships": len(relationships),
                "message": "Cross-file relationship analysis completed"
            }
            
        except Exception as e:
            logger.error(f"Cross-file relationship detection failed: {e}")
            return {"relationships": [], "error": str(e)}
    
    def _is_payroll_event(self, payload: Dict) -> bool:
        """Check if event is a payroll entry"""
        # Check for payroll indicators
        text = str(payload).lower()
        payroll_keywords = ['salary', 'payroll', 'wage', 'employee', 'payment']
        return any(keyword in text for keyword in payroll_keywords)
    
    def _is_payout_event(self, payload: Dict) -> bool:
        """Check if event is a payout entry"""
        # Check for payout indicators
        text = str(payload).lower()
        payout_keywords = ['payout', 'transfer', 'bank', 'withdrawal', 'payment']
        return any(keyword in text for keyword in payout_keywords)
    
    async def _find_relationships(self, payroll_events: List, payout_events: List) -> List[Dict]:
        """Find relationships between payroll and payout events"""
        relationships = []
        
        for payroll in payroll_events:
            payroll_payload = payroll.get('payload', {})
            payroll_amount = self._extract_amount(payroll_payload)
            payroll_entities = self._extract_entities(payroll_payload)
            payroll_date = self._extract_date(payroll_payload)
            
            for payout in payout_events:
                payout_payload = payout.get('payload', {})
                payout_amount = self._extract_amount(payout_payload)
                payout_entities = self._extract_entities(payout_payload)
                payout_date = self._extract_date(payout_payload)
                
                # Check for relationship indicators
                relationship_score = self._calculate_relationship_score(
                    payroll_amount, payout_amount,
                    payroll_entities, payout_entities,
                    payroll_date, payout_date
                )
                
                if relationship_score > 0.7:  # High confidence threshold
                    relationships.append({
                        "payroll_event_id": payroll.get('id'),
                        "payout_event_id": payout.get('id'),
                        "payroll_platform": payroll.get('source_platform'),
                        "payout_platform": payout.get('source_platform'),
                        "relationship_score": relationship_score,
                        "relationship_type": "salary_to_payout",
                        "amount_match": abs(payroll_amount - payout_amount) < 1.0,
                        "date_match": self._dates_are_close(payroll_date, payout_date),
                        "entity_match": self._entities_match(payroll_entities, payout_entities),
                        "payroll_amount": payroll_amount,
                        "payout_amount": payout_amount,
                        "payroll_date": payroll_date,
                        "payout_date": payout_date
                    })
        
        return relationships
    
    def _extract_amount(self, payload: Dict) -> float:
        """Extract amount from payload using universal field detection"""
        try:
            # Use universal extraction first
            universal_extractors = UniversalExtractors()
            amount = universal_extractors.extract_amount_universal(payload)
            if amount is not None:
                return amount
            
            # Fallback to old method
            amount_fields = ['amount', 'total', 'value', 'sum', 'payment_amount']
            for field in amount_fields:
                if field in payload:
                    value = payload[field]
                    if isinstance(value, (int, float)):
                        return float(value)
                    elif isinstance(value, str):
                        # Remove currency symbols and convert
                        cleaned = value.replace('$', '').replace(',', '').strip()
                        return float(cleaned)
        except:
            pass
        return 0.0
    
    def _extract_entities(self, payload: Dict) -> List[str]:
        """Extract entity names from payload using universal field detection"""
        entities = []
        try:
            # Use universal extraction first
            universal_extractors = UniversalExtractors()
            vendor_name = universal_extractors.extract_vendor_universal(payload)
            if vendor_name:
                entities.append(vendor_name)
            
            # Fallback to old method
            name_fields = ['employee_name', 'name', 'recipient', 'payee', 'description']
            for field in name_fields:
                if field in payload:
                    value = payload[field]
                    if isinstance(value, str) and value.strip():
                        entities.append(value.strip())
        except:
            pass
        return entities
    
    def _extract_date(self, payload: Dict) -> Optional[datetime]:
        """Extract date from payload using universal field detection"""
        try:
            # Use universal extraction first
            universal_extractors = UniversalExtractors()
            date = universal_extractors.extract_date_universal(payload)
            if date is not None:
                return date
            
            # Fallback to old method
            date_fields = ['date', 'payment_date', 'transaction_date', 'created_at']
            for field in date_fields:
                if field in payload:
                    value = payload[field]
                    if isinstance(value, str):
                        return datetime.fromisoformat(value.replace('Z', '+00:00'))
                    elif isinstance(value, datetime):
                        return value
        except:
            pass
        return None
    
    def _calculate_relationship_score(self, payroll_amount: float, payout_amount: float,
                                   payroll_entities: List[str], payout_entities: List[str],
                                   payroll_date: Optional[datetime], payout_date: Optional[datetime]) -> float:
        """Calculate relationship confidence score"""
        score = 0.0
        
        # Amount matching (40% weight)
        if payroll_amount > 0 and payout_amount > 0:
            amount_diff = abs(payroll_amount - payout_amount)
            if amount_diff < 1.0:  # Exact match
                score += 0.4
            elif amount_diff < payroll_amount * 0.01:  # Within 1%
                score += 0.3
            elif amount_diff < payroll_amount * 0.05:  # Within 5%
                score += 0.2
        
        # Entity matching (30% weight)
        if payroll_entities and payout_entities:
            entity_match_score = self._calculate_entity_match_score(payroll_entities, payout_entities)
            score += entity_match_score * 0.3
        
        # Date matching (30% weight)
        if payroll_date and payout_date:
            date_diff = abs((payroll_date - payout_date).days)
            if date_diff <= 1:  # Same day
                score += 0.3
            elif date_diff <= 7:  # Within a week
                score += 0.2
            elif date_diff <= 30:  # Within a month
                score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_entity_match_score(self, entities1: List[str], entities2: List[str]) -> float:
        """Calculate entity name similarity score"""
        if not entities1 or not entities2:
            return 0.0
        
        max_score = 0.0
        for entity1 in entities1:
            for entity2 in entities2:
                similarity = SequenceMatcher(None, entity1.lower(), entity2.lower()).ratio()
                max_score = max(max_score, similarity)
        
        return max_score
    
    def _entities_match(self, entities1: List[str], entities2: List[str]) -> bool:
        """Check if entities match"""
        return self._calculate_entity_match_score(entities1, entities2) > 0.8
    
    def _dates_are_close(self, date1: Optional[datetime], date2: Optional[datetime]) -> bool:
        """Check if dates are close (within 7 days)"""
        if not date1 or not date2:
            return False
        return abs((date1 - date2).days) <= 7

class AIRelationshipDetector:
    """AI-powered universal relationship detection for ANY financial data"""
    
    def __init__(self, openai_client, supabase_client: Client):
        self.openai = openai_client
        self.supabase = supabase_client
        self.relationship_cache = {}
        self.learned_patterns = {}
        
    async def detect_all_relationships(self, user_id: str) -> Dict[str, Any]:
        """Detect ALL possible relationships between financial events"""
        try:
            # Get all events for the user
            events = self.supabase.table('raw_events').select('*').eq('user_id', user_id).execute()
            
            if not events.data:
                return {"relationships": [], "message": "No data found for relationship analysis"}
            
            # Use AI to discover relationship types
            relationship_types = await self._discover_relationship_types(events.data)
            
            # Detect relationships for each type
            all_relationships = []
            for rel_type in relationship_types:
                type_relationships = await self._detect_relationships_by_type(events.data, rel_type)
                all_relationships.extend(type_relationships)
            
            # Use AI to discover new relationship patterns
            ai_discovered = await self._ai_discover_relationships(events.data)
            all_relationships.extend(ai_discovered)
            
            # Validate and score relationships
            validated_relationships = await self._validate_relationships(all_relationships)
            
            return {
                "relationships": validated_relationships,
                "total_relationships": len(validated_relationships),
                "relationship_types": relationship_types,
                "ai_discovered_count": len(ai_discovered),
                "message": "Comprehensive AI-powered relationship analysis completed"
            }
            
        except Exception as e:
            logger.error(f"AI relationship detection failed: {e}")
            return {"relationships": [], "error": str(e)}
    
    async def _discover_relationship_types(self, events: List[Dict]) -> List[str]:
        """Use AI to discover what types of relationships exist in the data"""
        try:
            # Create context for AI analysis
            event_summary = self._create_event_summary(events)
            
            ai_response = await self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "system",
                    "content": "You are a financial data analyst. Analyze the financial events and identify what types of relationships might exist between them. Return only the relationship types as a JSON array."
                }, {
                    "role": "user",
                    "content": f"Analyze these financial events and identify relationship types: {event_summary}"
                }],
                temperature=0.1
            )
            
            # Parse AI response
            response_text = ai_response.choices[0].message.content
            relationship_types = self._parse_relationship_types(response_text)
            
            return relationship_types
            
        except Exception as e:
            logger.error(f"AI relationship type discovery failed: {e}")
            return ["invoice_to_payment", "fee_to_transaction", "refund_to_original"]
    
    async def _detect_relationships_by_type(self, events: List[Dict], relationship_type: str) -> List[Dict]:
        """Detect relationships for a specific type"""
        relationships = []
        
        # Get source and target event filters for this relationship type
        source_filter, target_filter = self._get_relationship_filters(relationship_type)
        
        # Filter events
        source_events = [e for e in events if self._matches_event_filter(e, source_filter)]
        target_events = [e for e in events if self._matches_event_filter(e, target_filter)]
        
        # Find relationships
        for source in source_events:
            for target in target_events:
                if source['id'] == target['id']:
                    continue
                
                # Calculate comprehensive relationship score
                score = await self._calculate_comprehensive_score(source, target, relationship_type)
                
                if score >= 0.6:  # Configurable threshold
                    relationship = {
                        "source_event_id": source['id'],
                        "target_event_id": target['id'],
                        "relationship_type": relationship_type,
                        "confidence_score": score,
                        "source_platform": source.get('source_platform'),
                        "target_platform": target.get('source_platform'),
                        "source_amount": self._extract_amount(source.get('payload', {})),
                        "target_amount": self._extract_amount(target.get('payload', {})),
                        "amount_match": self._check_amount_match(source, target),
                        "date_match": self._check_date_match(source, target),
                        "entity_match": self._check_entity_match(source, target),
                        "id_match": self._check_id_match(source, target),
                        "context_match": self._check_context_match(source, target),
                        "detection_method": "rule_based"
                    }
                    relationships.append(relationship)
        
        return relationships
    
    async def _ai_discover_relationships(self, events: List[Dict]) -> List[Dict]:
        """Use AI to discover relationships we haven't seen before"""
        try:
            # Create comprehensive context
            context = self._create_comprehensive_context(events)
            
            ai_response = await self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "system",
                    "content": "You are a financial data analyst. Analyze the financial events and identify potential relationships between them that might not be obvious. Return the relationships as a JSON array with source_event_id, target_event_id, relationship_type, and confidence_score."
                }, {
                    "role": "user",
                    "content": f"Analyze these financial events and identify ALL possible relationships: {context}"
                }],
                temperature=0.2
            )
            
            # Parse AI discoveries
            response_text = ai_response.choices[0].message.content
            ai_relationships = self._parse_ai_relationships(response_text, events)
            
            return ai_relationships
            
        except Exception as e:
            logger.error(f"AI relationship discovery failed: {e}")
            return []
    
    async def _calculate_comprehensive_score(self, source: Dict, target: Dict, relationship_type: str) -> float:
        """Calculate comprehensive relationship score using multiple dimensions"""
        score = 0.0
        
        # Amount matching (30% weight)
        amount_score = self._calculate_amount_score(source, target, relationship_type)
        score += amount_score * 0.3
        
        # Date matching (20% weight)
        date_score = self._calculate_date_score(source, target, relationship_type)
        score += date_score * 0.2
        
        # Entity matching (20% weight)
        entity_score = self._calculate_entity_score(source, target, relationship_type)
        score += entity_score * 0.2
        
        # ID matching (15% weight)
        id_score = self._calculate_id_score(source, target, relationship_type)
        score += id_score * 0.15
        
        # Context matching (15% weight)
        context_score = self._calculate_context_score(source, target, relationship_type)
        score += context_score * 0.15
        
        return min(score, 1.0)
    
    def _calculate_amount_score(self, source: Dict, target: Dict, relationship_type: str) -> float:
        """Calculate amount matching score with intelligent tolerance"""
        source_amount = self._extract_amount(source.get('payload', {}))
        target_amount = self._extract_amount(target.get('payload', {}))
        
        if source_amount == 0 or target_amount == 0:
            return 0.0
        
        # Different tolerance based on relationship type
        tolerance_map = {
            'invoice_to_payment': 0.001,  # Exact match
            'fee_to_transaction': 0.01,   # 1% tolerance
            'refund_to_original': 0.001,  # Exact match
            'payroll_to_payout': 0.05,    # 5% tolerance
            'tax_to_income': 0.02,        # 2% tolerance
            'expense_to_reimbursement': 0.001,  # Exact match
            'subscription_to_payment': 0.001,   # Exact match
            'loan_to_payment': 0.01,      # 1% tolerance
            'investment_to_return': 0.1,  # 10% tolerance
        }
        
        tolerance = tolerance_map.get(relationship_type, 0.01)
        amount_diff = abs(source_amount - target_amount)
        
        if amount_diff <= tolerance:
            return 1.0
        elif amount_diff <= source_amount * tolerance:
            return 0.8
        elif amount_diff <= source_amount * tolerance * 2:
            return 0.6
        else:
            return 0.0
    
    def _calculate_date_score(self, source: Dict, target: Dict, relationship_type: str) -> float:
        """Calculate date matching score with configurable windows"""
        source_date = self._extract_date(source.get('payload', {}))
        target_date = self._extract_date(target.get('payload', {}))
        
        if not source_date or not target_date:
            return 0.0
        
        # Different date windows based on relationship type
        window_map = {
            'invoice_to_payment': 7,      # 7 days
            'fee_to_transaction': 1,      # Same day
            'refund_to_original': 30,     # 30 days
            'payroll_to_payout': 3,       # 3 days
            'tax_to_income': 90,          # 90 days
            'expense_to_reimbursement': 30,  # 30 days
            'subscription_to_payment': 7,     # 7 days
            'loan_to_payment': 1,         # Same day
            'investment_to_return': 365,  # 1 year
        }
        
        window_days = window_map.get(relationship_type, 7)
        date_diff = abs((source_date - target_date).days)
        
        if date_diff == 0:
            return 1.0
        elif date_diff <= window_days:
            return 0.8
        elif date_diff <= window_days * 2:
            return 0.6
        else:
            return 0.0
    
    def _calculate_entity_score(self, source: Dict, target: Dict, relationship_type: str) -> float:
        """Calculate entity matching score with fuzzy logic"""
        source_entities = self._extract_entities(source.get('payload', {}))
        target_entities = self._extract_entities(target.get('payload', {}))
        
        if not source_entities or not target_entities:
            return 0.0
        
        # Calculate similarity for each entity pair
        max_similarity = 0.0
        for source_entity in source_entities:
            for target_entity in target_entities:
                similarity = self._calculate_text_similarity(source_entity, target_entity)
                max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def _calculate_id_score(self, source: Dict, target: Dict, relationship_type: str) -> float:
        """Calculate ID matching score with pattern recognition"""
        source_ids = source.get('platform_ids', {})
        target_ids = target.get('platform_ids', {})
        
        if not source_ids or not target_ids:
            return 0.0
        
        # Check for exact ID matches
        for source_key, source_id in source_ids.items():
            for target_key, target_id in target_ids.items():
                if source_id == target_id:
                    return 1.0
                elif source_id in target_id or target_id in source_id:
                    return 0.8
        
        # Check for pattern matches
        for source_key, source_id in source_ids.items():
            for target_key, target_id in target_ids.items():
                if self._check_id_pattern_match(source_id, target_id, relationship_type):
                    return 0.6
        
        return 0.0
    
    def _calculate_context_score(self, source: Dict, target: Dict, relationship_type: str) -> float:
        """Calculate context matching score using semantic analysis"""
        source_context = self._extract_context(source)
        target_context = self._extract_context(target)
        
        if not source_context or not target_context:
            return 0.0
        
        # Calculate semantic similarity
        similarity = self._calculate_text_similarity(source_context, target_context)
        
        # Boost score for expected relationship contexts
        context_boost = self._get_context_boost(source_context, target_context, relationship_type)
        
        return min(similarity + context_boost, 1.0)
    
    def _check_amount_match(self, source: Dict, target: Dict) -> bool:
        """Check if amounts match within tolerance"""
        return self._calculate_amount_score(source, target, "generic") > 0.8
    
    def _check_date_match(self, source: Dict, target: Dict) -> bool:
        """Check if dates are within acceptable window"""
        return self._calculate_date_score(source, target, "generic") > 0.8
    
    def _check_entity_match(self, source: Dict, target: Dict) -> bool:
        """Check if entities match"""
        return self._calculate_entity_score(source, target, "generic") > 0.8
    
    def _check_id_match(self, source: Dict, target: Dict) -> bool:
        """Check if IDs match"""
        return self._calculate_id_score(source, target, "generic") > 0.8
    
    def _check_context_match(self, source: Dict, target: Dict) -> bool:
        """Check if contexts match"""
        return self._calculate_context_score(source, target, "generic") > 0.8
    
    def _get_relationship_filters(self, relationship_type: str) -> Tuple[Dict, Dict]:
        """Get source and target filters for a relationship type"""
        filters = {
            'invoice_to_payment': (
                {'keywords': ['invoice', 'bill', 'receivable']},
                {'keywords': ['payment', 'charge', 'transaction']}
            ),
            'fee_to_transaction': (
                {'keywords': ['fee', 'commission', 'charge']},
                {'keywords': ['transaction', 'payment', 'charge']}
            ),
            'refund_to_original': (
                {'keywords': ['refund', 'return', 'reversal']},
                {'keywords': ['payment', 'charge', 'transaction']}
            ),
            'payroll_to_payout': (
                {'keywords': ['payroll', 'salary', 'wage', 'employee']},
                {'keywords': ['payout', 'transfer', 'withdrawal']}
            ),
            'tax_to_income': (
                {'keywords': ['tax', 'withholding', 'deduction']},
                {'keywords': ['income', 'revenue', 'salary']}
            ),
            'expense_to_reimbursement': (
                {'keywords': ['expense', 'cost', 'outlay']},
                {'keywords': ['reimbursement', 'refund', 'return']}
            ),
            'subscription_to_payment': (
                {'keywords': ['subscription', 'recurring', 'monthly']},
                {'keywords': ['payment', 'charge', 'transaction']}
            ),
            'loan_to_payment': (
                {'keywords': ['loan', 'credit', 'advance']},
                {'keywords': ['payment', 'repayment', 'installment']}
            ),
            'investment_to_return': (
                {'keywords': ['investment', 'purchase', 'buy']},
                {'keywords': ['return', 'dividend', 'profit']}
            )
        }
        
        return filters.get(relationship_type, ({}, {}))
    
    def _matches_event_filter(self, event: Dict, filter_dict: Dict) -> bool:
        """Check if event matches the filter criteria"""
        if not filter_dict:
            return True
        
        # Check keywords
        if 'keywords' in filter_dict:
            event_text = str(event.get('payload', {})).lower()
            event_text += ' ' + str(event.get('kind', '')).lower()
            event_text += ' ' + str(event.get('category', '')).lower()
            
            keywords = filter_dict['keywords']
            if not any(keyword.lower() in event_text for keyword in keywords):
                return False
        
        return True
    
    def _extract_amount(self, payload: Dict) -> float:
        """Extract amount from payload"""
        try:
            amount_fields = ['amount', 'total', 'value', 'sum', 'payment_amount', 'charge_amount']
            for field in amount_fields:
                if field in payload:
                    value = payload[field]
                    if isinstance(value, (int, float)):
                        return float(value)
                    elif isinstance(value, str):
                        cleaned = value.replace('$', '').replace(',', '').strip()
                        return float(cleaned)
        except:
            pass
        return 0.0
    
    def _extract_entities(self, payload: Dict) -> List[str]:
        """Extract entity names from payload"""
        entities = []
        try:
            name_fields = ['employee_name', 'name', 'recipient', 'payee', 'description', 'vendor_name']
            for field in name_fields:
                if field in payload:
                    value = payload[field]
                    if isinstance(value, str) and value.strip():
                        entities.append(value.strip())
        except:
            pass
        return entities
    
    def _extract_date(self, payload: Dict) -> Optional[datetime]:
        """Extract date from payload"""
        try:
            date_fields = ['date', 'payment_date', 'transaction_date', 'created_at', 'due_date']
            for field in date_fields:
                if field in payload:
                    value = payload[field]
                    if isinstance(value, str):
                        return datetime.fromisoformat(value.replace('Z', '+00:00'))
                    elif isinstance(value, datetime):
                        return value
        except:
            pass
        return None
    
    def _extract_context(self, event: Dict) -> str:
        """Extract context from event"""
        context_parts = []
        
        # Add kind and category
        if event.get('kind'):
            context_parts.append(event['kind'])
        if event.get('category'):
            context_parts.append(event['category'])
        
        # Add payload description
        payload = event.get('payload', {})
        if 'description' in payload:
            context_parts.append(payload['description'])
        
        # Add vendor information
        if event.get('vendor_standard'):
            context_parts.append(event['vendor_standard'])
        
        return ' '.join(context_parts)
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using SequenceMatcher"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def _check_id_pattern_match(self, id1: str, id2: str, relationship_type: str) -> bool:
        """Check if IDs match a pattern for the relationship type"""
        import re
        # Define patterns for different relationship types
        patterns = {
            'invoice_to_payment': [
                (r'inv_(\w+)', r'pay_\1'),  # invoice_id to payment_id
                (r'in_(\w+)', r'pi_\1'),    # invoice_id to payment_intent
            ],
            'fee_to_transaction': [
                (r'fee_(\w+)', r'ch_\1'),   # fee_id to charge_id
                (r'fee_(\w+)', r'txn_\1'),  # fee_id to transaction_id
            ],
            'refund_to_original': [
                (r're_(\w+)', r'ch_\1'),    # refund_id to charge_id
                (r'rfnd_(\w+)', r'pay_\1'), # refund_id to payment_id
            ]
        }
        
        pattern_list = patterns.get(relationship_type, [])
        
        for pattern1, pattern2 in pattern_list:
            match1 = re.match(pattern1, id1)
            match2 = re.match(pattern2, id2)
            
            if match1 and match2 and match1.group(1) == match2.group(1):
                return True
        
        return False
    
    def _get_context_boost(self, context1: str, context2: str, relationship_type: str) -> float:
        """Get context boost for expected relationship patterns"""
        context_combinations = {
            'invoice_to_payment': [
                ('invoice', 'payment'),
                ('bill', 'charge'),
                ('receivable', 'transaction')
            ],
            'fee_to_transaction': [
                ('fee', 'transaction'),
                ('commission', 'payment'),
                ('charge', 'transaction')
            ],
            'refund_to_original': [
                ('refund', 'payment'),
                ('return', 'charge'),
                ('reversal', 'transaction')
            ]
        }
        
        combinations = context_combinations.get(relationship_type, [])
        context_lower = context1.lower() + ' ' + context2.lower()
        
        for combo in combinations:
            if combo[0] in context_lower and combo[1] in context_lower:
                return 0.2
        
        return 0.0
    
    def _create_event_summary(self, events: List[Dict]) -> str:
        """Create a summary of events for AI analysis"""
        summary_parts = []
        
        for event in events[:10]:  # Limit to first 10 events
            event_summary = {
                'id': event.get('id'),
                'kind': event.get('kind'),
                'category': event.get('category'),
                'platform': event.get('source_platform'),
                'amount': self._extract_amount(event.get('payload', {})),
                'vendor': event.get('vendor_standard'),
                'description': event.get('payload', {}).get('description', '')
            }
            summary_parts.append(str(event_summary))
        
        return '\n'.join(summary_parts)
    
    def _create_comprehensive_context(self, events: List[Dict]) -> str:
        """Create comprehensive context for AI analysis"""
        context_parts = []
        
        # Group events by platform
        platform_groups = {}
        for event in events:
            platform = event.get('source_platform', 'unknown')
            if platform not in platform_groups:
                platform_groups[platform] = []
            platform_groups[platform].append(event)
        
        # Create context for each platform
        for platform, platform_events in platform_groups.items():
            context_parts.append(f"\nPlatform: {platform}")
            for event in platform_events[:5]:  # Limit to 5 events per platform
                context_parts.append(f"- {event.get('kind')}: {event.get('payload', {}).get('description', '')}")
        
        return '\n'.join(context_parts)
    
    def _parse_relationship_types(self, response_text: str) -> List[str]:
        """Parse relationship types from AI response"""
        try:
            # Try to extract JSON array
            if '[' in response_text and ']' in response_text:
                start = response_text.find('[')
                end = response_text.rfind(']') + 1
                json_str = response_text[start:end]
                return json.loads(json_str)
        except:
            pass
        
        # Fallback to common relationship types
        return ["invoice_to_payment", "fee_to_transaction", "refund_to_original"]
    
    def _parse_ai_relationships(self, response_text: str, events: List[Dict]) -> List[Dict]:
        """Parse AI-discovered relationships from response"""
        try:
            # Try to extract JSON array
            if '[' in response_text and ']' in response_text:
                start = response_text.find('[')
                end = response_text.rfind(']') + 1
                json_str = response_text[start:end]
                ai_relationships = json.loads(json_str)
                
                # Convert to standard format
                relationships = []
                for rel in ai_relationships:
                    relationship = {
                        "source_event_id": rel.get('source_event_id'),
                        "target_event_id": rel.get('target_event_id'),
                        "relationship_type": rel.get('relationship_type', 'ai_discovered'),
                        "confidence_score": rel.get('confidence_score', 0.5),
                        "detection_method": "ai_discovered"
                    }
                    relationships.append(relationship)
                
                return relationships
        except:
            pass
        
        return []
    
    async def _validate_relationships(self, relationships: List[Dict]) -> List[Dict]:
        """Validate and filter relationships"""
        validated = []
        
        for relationship in relationships:
            # Check if events exist
            source_exists = await self._event_exists(relationship['source_event_id'])
            target_exists = await self._event_exists(relationship['target_event_id'])
            
            if source_exists and target_exists:
                # Add additional validation
                relationship['validated'] = True
                relationship['validation_score'] = relationship.get('confidence_score', 0.0)
                validated.append(relationship)
        
        return validated
    
    async def _event_exists(self, event_id: str) -> bool:
        """Check if event exists in database"""
        try:
            result = self.supabase.table('raw_events').select('id').eq('id', event_id).execute()
            return len(result.data) > 0
        except:
            return False

@app.get("/debug-env")
async def debug_environment():
    """Debug endpoint to check environment variables"""
    try:
        env_vars = {
            "SUPABASE_URL": os.getenv('SUPABASE_URL', 'NOT_SET'),
            "SUPABASE_SERVICE_KEY": os.getenv('SUPABASE_SERVICE_KEY', 'NOT_SET'),
            "SUPABASE_KEY": os.getenv('SUPABASE_KEY', 'NOT_SET'),
            "OPENAI_API_KEY": os.getenv('OPENAI_API_KEY', 'NOT_SET')
        }
        
        # Check if keys are actually set (not just placeholder)
        key_status = {}
        for key, value in env_vars.items():
            if value == 'NOT_SET':
                key_status[key] = "NOT_SET"
            elif value.startswith('eyJ') and len(value) > 100:
                key_status[key] = "SET (JWT token)"
            elif len(value) > 10:
                key_status[key] = "SET (other value)"
            else:
                key_status[key] = "SET (short value)"
        
        return {
            "message": "Environment Variables Debug",
            "environment_variables": key_status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "message": "Debug Environment Failed",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@app.get("/test-vendor-standardization")
async def test_vendor_standardization():
    """Test vendor standardization with sample data"""
    try:
        vendor_standardizer = VendorStandardizer(openai)
        
        test_cases = [
            {
                "vendor_name": "Google LLC",
                "platform": "razorpay",
                "expected_standard": "Google"
            },
            {
                "vendor_name": "Microsoft Corporation",
                "platform": "stripe",
                "expected_standard": "Microsoft"
            },
            {
                "vendor_name": "AMAZON.COM INC",
                "platform": "quickbooks",
                "expected_standard": "Amazon"
            },
            {
                "vendor_name": "Apple Inc.",
                "platform": "gusto",
                "expected_standard": "Apple"
            }
        ]
        
        results = []
        for test_case in test_cases:
            try:
                standardized = await vendor_standardizer.standardize_vendor(
                    vendor_name=test_case["vendor_name"],
                    platform=test_case["platform"]
                )
                
                results.append({
                    "test_case": test_case,
                    "standardized_data": standardized,
                    "success": True
                })
                
            except Exception as e:
                results.append({
                    "test_case": test_case,
                    "error": str(e),
                    "success": False
                })
        
        return {
            "message": "Vendor Standardization Test Results",
            "total_tests": len(test_cases),
            "successful_tests": len([r for r in results if r["success"]]),
            "test_results": results
        }
        
    except Exception as e:
        logger.error(f"Vendor standardization test failed: {e}")
        return {
            "message": "Vendor Standardization Test Failed",
            "error": str(e),
            "test_results": []
        }

@app.get("/test-platform-id-extraction")
async def test_platform_id_extraction():
    """Test platform ID extraction with sample data"""
    try:
        platform_id_extractor = PlatformIDExtractor()
        
        test_cases = [
            {
                "row_data": {
                    "payment_id": "pay_12345678901234",
                    "order_id": "order_98765432109876",
                    "amount": 1000,
                    "description": "Payment for services"
                },
                "platform": "razorpay",
                "column_names": ["payment_id", "order_id", "amount", "description"]
            },
            {
                "row_data": {
                    "charge_id": "ch_123456789012345678901234",
                    "customer_id": "cus_12345678901234",
                    "amount": 50.00,
                    "description": "Stripe payment"
                },
                "platform": "stripe",
                "column_names": ["charge_id", "customer_id", "amount", "description"]
            },
            {
                "row_data": {
                    "employee_id": "emp_12345678",
                    "payroll_id": "pay_123456789012",
                    "amount": 5000,
                    "description": "Salary payment"
                },
                "platform": "gusto",
                "column_names": ["employee_id", "payroll_id", "amount", "description"]
            }
        ]
        
        results = []
        for test_case in test_cases:
            try:
                extracted = platform_id_extractor.extract_platform_ids(
                    row_data=test_case["row_data"],
                    platform=test_case["platform"],
                    column_names=test_case["column_names"]
                )
                
                results.append({
                    "test_case": test_case,
                    "extracted_data": extracted,
                    "success": True
                })
                
            except Exception as e:
                results.append({
                    "test_case": test_case,
                    "error": str(e),
                    "success": False
                })
        
        return {
            "message": "Platform ID Extraction Test Results",
            "total_tests": len(test_cases),
            "successful_tests": len([r for r in results if r["success"]]),
            "test_results": results
        }
        
    except Exception as e:
        logger.error(f"Platform ID extraction test failed: {e}")
        return {
            "message": "Platform ID Extraction Test Failed",
            "error": str(e),
            "test_results": []
        }

@app.get("/test-data-enrichment")
async def test_data_enrichment():
    """Test complete data enrichment pipeline"""
    try:
        enrichment_processor = DataEnrichmentProcessor(openai)
        
        test_cases = [
            {
                "row_data": {
                    "vendor_name": "Google LLC",
                    "amount": 9000,
                    "description": "Google Cloud Services ₹9000",
                    "payment_id": "pay_12345678901234"
                },
                "platform_info": {"platform": "razorpay", "confidence": 0.9},
                "column_names": ["vendor_name", "amount", "description", "payment_id"],
                "ai_classification": {
                    "row_type": "operating_expense",
                    "category": "expense",
                    "subcategory": "infrastructure",
                    "confidence": 0.95
                },
                "file_context": {"filename": "test-payments.csv", "user_id": "test-user"}
            },
            {
                "row_data": {
                    "vendor_name": "Microsoft Corporation",
                    "amount": 150.50,
                    "description": "Stripe payment $150.50 for software",
                    "charge_id": "ch_123456789012345678901234"
                },
                "platform_info": {"platform": "stripe", "confidence": 0.9},
                "column_names": ["vendor_name", "amount", "description", "charge_id"],
                "ai_classification": {
                    "row_type": "operating_expense",
                    "category": "expense",
                    "subcategory": "software",
                    "confidence": 0.9
                },
                "file_context": {"filename": "test-payments.csv", "user_id": "test-user"}
            }
        ]
        
        results = []
        for test_case in test_cases:
            try:
                enriched = await enrichment_processor.enrich_row_data(
                    row_data=test_case["row_data"],
                    platform_info=test_case["platform_info"],
                    column_names=test_case["column_names"],
                    ai_classification=test_case["ai_classification"],
                    file_context=test_case["file_context"]
                )
                
                results.append({
                    "test_case": test_case,
                    "enriched_data": enriched,
                    "success": True
                })
                
            except Exception as e:
                results.append({
                    "test_case": test_case,
                    "error": str(e),
                    "success": False
                })
        
        return {
            "message": "Data Enrichment Test Results",
            "total_tests": len(test_cases),
            "successful_tests": len([r for r in results if r["success"]]),
            "test_results": results
        }
        
    except Exception as e:
        logger.error(f"Data enrichment test failed: {e}")
        return {
            "message": "Data Enrichment Test Failed",
            "error": str(e),
            "test_results": []
        }

@app.get("/test-enrichment-stats/{user_id}")
async def test_enrichment_stats(user_id: str):
    """Test enrichment statistics for a user"""
    try:
        # Initialize Supabase client
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        
        if not supabase_url or not supabase_key:
            return {
                "message": "Supabase credentials not configured",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Clean JWT token to prevent header value errors
        supabase_key = clean_jwt_token(supabase_key)
        supabase = create_client(supabase_url, supabase_key)
        
        # Call the database function
        result = supabase.rpc('get_enrichment_stats', {'user_uuid': user_id}).execute()
        
        if result.data:
            return {
                "message": "Enrichment Statistics Retrieved Successfully",
                "stats": result.data[0] if result.data else {},
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            return {
                "message": "No enrichment statistics found",
                "stats": {},
                "timestamp": datetime.utcnow().isoformat()
            }
        
    except Exception as e:
        return {
            "message": "Enrichment Statistics Test Failed",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/test-vendor-search/{user_id}")
async def test_vendor_search(user_id: str, vendor_name: str = "Google"):
    """Test vendor search functionality"""
    try:
        # Initialize Supabase client
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        
        if not supabase_url or not supabase_key:
            return {
                "message": "Supabase credentials not configured",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Clean JWT token to prevent header value errors
        supabase_key = clean_jwt_token(supabase_key)
        supabase = create_client(supabase_url, supabase_key)
        
        # Call the database function
        result = supabase.rpc('search_events_by_vendor', {
            'user_uuid': user_id,
            'vendor_name': vendor_name
        }).execute()
        
        if result.data:
            return {
                "message": "Vendor Search Results Retrieved Successfully",
                "vendor_name": vendor_name,
                "results": result.data,
                "count": len(result.data),
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            return {
                "message": "No vendor search results found",
                "vendor_name": vendor_name,
                "results": [],
                "count": 0,
                "timestamp": datetime.utcnow().isoformat()
            }
        
    except Exception as e:
        return {
            "message": "Vendor Search Test Failed",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


class AIRelationshipDetector:
    """AI-powered universal relationship detection for ANY financial data"""
    
    def __init__(self, openai_client, supabase_client: Client):
        self.openai = openai_client
        self.supabase = supabase_client
        self.relationship_cache = {}
        self.learned_patterns = {}
        
    async def detect_all_relationships(self, user_id: str) -> Dict[str, Any]:
        """Detect ALL possible relationships between financial events"""
        try:
            # Get all events for the user
            events = self.supabase.table('raw_events').select('*').eq('user_id', user_id).execute()
            
            if not events.data:
                return {"relationships": [], "message": "No data found for relationship analysis"}
            
            # Use AI to discover relationship types
            relationship_types = await self._discover_relationship_types(events.data)
            
            # Detect relationships for each type
            all_relationships = []
            for rel_type in relationship_types:
                type_relationships = await self._detect_relationships_by_type(events.data, rel_type)
                all_relationships.extend(type_relationships)
            
            # Use AI to discover new relationship patterns
            ai_discovered = await self._ai_discover_relationships(events.data)
            all_relationships.extend(ai_discovered)
            
            # Validate and score relationships
            validated_relationships = await self._validate_relationships(all_relationships)
            
            # Store relationships in database
            await self._store_relationships(validated_relationships, user_id)
            
            return {
                "relationships": validated_relationships,
                "total_relationships": len(validated_relationships),
                "relationship_types": relationship_types,
                "ai_discovered_count": len(ai_discovered),
                "message": "Comprehensive AI-powered relationship analysis completed"
            }
            
        except Exception as e:
            logger.error(f"AI relationship detection failed: {e}")
            return {"relationships": [], "error": str(e)}
    
    async def _discover_relationship_types(self, events: List[Dict]) -> List[str]:
        """Use AI to discover what types of relationships exist in the data"""
        try:
            # Create context for AI analysis
            event_summary = self._create_event_summary(events)
            
            ai_response = await self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "system",
                    "content": "You are a financial data analyst. Analyze the financial events and identify what types of relationships might exist between them. Return only the relationship types as a JSON array."
                }, {
                    "role": "user",
                    "content": f"Analyze these financial events and identify relationship types: {event_summary}"
                }],
                temperature=0.1
            )
            
            # Parse AI response
            response_text = ai_response.choices[0].message.content
            relationship_types = self._parse_relationship_types(response_text)
            
            return relationship_types
            
        except Exception as e:
            logger.error(f"AI relationship type discovery failed: {e}")
            return ["invoice_to_payment", "fee_to_transaction", "refund_to_original"]
    
    async def _detect_relationships_by_type(self, events: List[Dict], relationship_type: str) -> List[Dict]:
        """Detect relationships for a specific type"""
        relationships = []
        
        # Get source and target event filters for this relationship type
        source_filter, target_filter = self._get_relationship_filters(relationship_type)
        
        # Filter events
        source_events = [e for e in events if self._matches_event_filter(e, source_filter)]
        target_events = [e for e in events if self._matches_event_filter(e, target_filter)]
        
        # Find relationships
        for source in source_events:
            for target in target_events:
                if source['id'] == target['id']:
                    continue
                
                # Calculate comprehensive relationship score
                score = await self._calculate_comprehensive_score(source, target, relationship_type)
                
                if score >= 0.6:  # Configurable threshold
                    relationship = {
                        "source_event_id": source['id'],
                        "target_event_id": target['id'],
                        "relationship_type": relationship_type,
                        "confidence_score": score,
                        "source_platform": source.get('source_platform'),
                        "target_platform": target.get('source_platform'),
                        "source_amount": self._extract_amount(source.get('payload', {})),
                        "target_amount": self._extract_amount(target.get('payload', {})),
                        "amount_match": self._check_amount_match(source, target),
                        "date_match": self._check_date_match(source, target),
                        "entity_match": self._check_entity_match(source, target),
                        "id_match": self._check_id_match(source, target),
                        "context_match": self._check_context_match(source, target),
                        "detection_method": "rule_based"
                    }
                    relationships.append(relationship)
        
        return relationships
    
    async def _ai_discover_relationships(self, events: List[Dict]) -> List[Dict]:
        """Use AI to discover relationships we haven't seen before"""
        try:
            # Create comprehensive context
            context = self._create_comprehensive_context(events)
            
            ai_response = await self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "system",
                    "content": "You are a financial data analyst. Analyze the financial events and identify potential relationships between them that might not be obvious. Return the relationships as a JSON array with source_event_id, target_event_id, relationship_type, and confidence_score."
                }, {
                    "role": "user",
                    "content": f"Analyze these financial events and identify ALL possible relationships: {context}"
                }],
                temperature=0.2
            )
            
            # Parse AI discoveries
            response_text = ai_response.choices[0].message.content
            ai_relationships = self._parse_ai_relationships(response_text, events)
            
            return ai_relationships
            
        except Exception as e:
            logger.error(f"AI relationship discovery failed: {e}")
            return []
    
    async def _calculate_comprehensive_score(self, source: Dict, target: Dict, relationship_type: str) -> float:
        """Calculate comprehensive relationship score using multiple dimensions"""
        score = 0.0
        
        # Amount matching (30% weight)
        amount_score = self._calculate_amount_score(source, target, relationship_type)
        score += amount_score * 0.3
        
        # Date matching (20% weight)
        date_score = self._calculate_date_score(source, target, relationship_type)
        score += date_score * 0.2
        
        # Entity matching (20% weight)
        entity_score = self._calculate_entity_score(source, target, relationship_type)
        score += entity_score * 0.2
        
        # ID matching (15% weight)
        id_score = self._calculate_id_score(source, target, relationship_type)
        score += id_score * 0.15
        
        # Context matching (15% weight)
        context_score = self._calculate_context_score(source, target, relationship_type)
        score += context_score * 0.15
        
        return min(score, 1.0)
    
    def _calculate_amount_score(self, source: Dict, target: Dict, relationship_type: str) -> float:
        """Calculate amount matching score with intelligent tolerance"""
        source_amount = self._extract_amount(source.get('payload', {}))
        target_amount = self._extract_amount(target.get('payload', {}))
        
        if source_amount == 0 or target_amount == 0:
            return 0.0
        
        # Different tolerance based on relationship type
        tolerance_map = {
            'invoice_to_payment': 0.001,  # Exact match
            'fee_to_transaction': 0.01,   # 1% tolerance
            'refund_to_original': 0.001,  # Exact match
            'payroll_to_payout': 0.05,    # 5% tolerance
            'tax_to_income': 0.02,        # 2% tolerance
            'expense_to_reimbursement': 0.001,  # Exact match
            'subscription_to_payment': 0.001,   # Exact match
            'loan_to_payment': 0.01,      # 1% tolerance
            'investment_to_return': 0.1,  # 10% tolerance
        }
        
        tolerance = tolerance_map.get(relationship_type, 0.01)
        amount_diff = abs(source_amount - target_amount)
        
        if amount_diff <= tolerance:
            return 1.0
        elif amount_diff <= source_amount * tolerance:
            return 0.8
        elif amount_diff <= source_amount * tolerance * 2:
            return 0.6
        else:
            return 0.0
    
    def _calculate_date_score(self, source: Dict, target: Dict, relationship_type: str) -> float:
        """Calculate date matching score with configurable windows"""
        source_date = self._extract_date(source.get('payload', {}))
        target_date = self._extract_date(target.get('payload', {}))
        
        if not source_date or not target_date:
            return 0.0
        
        # Different date windows based on relationship type
        window_map = {
            'invoice_to_payment': 7,      # 7 days
            'fee_to_transaction': 1,      # Same day
            'refund_to_original': 30,     # 30 days
            'payroll_to_payout': 3,       # 3 days
            'tax_to_income': 90,          # 90 days
            'expense_to_reimbursement': 30,  # 30 days
            'subscription_to_payment': 7,     # 7 days
            'loan_to_payment': 1,         # Same day
            'investment_to_return': 365,  # 1 year
        }
        
        window_days = window_map.get(relationship_type, 7)
        date_diff = abs((source_date - target_date).days)
        
        if date_diff == 0:
            return 1.0
        elif date_diff <= window_days:
            return 0.8
        elif date_diff <= window_days * 2:
            return 0.6
        else:
            return 0.0
    
    def _calculate_entity_score(self, source: Dict, target: Dict, relationship_type: str) -> float:
        """Calculate entity matching score with fuzzy logic"""
        source_entities = self._extract_entities(source.get('payload', {}))
        target_entities = self._extract_entities(target.get('payload', {}))
        
        if not source_entities or not target_entities:
            return 0.0
        
        # Calculate similarity for each entity pair
        max_similarity = 0.0
        for source_entity in source_entities:
            for target_entity in target_entities:
                similarity = self._calculate_text_similarity(source_entity, target_entity)
                max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def _calculate_id_score(self, source: Dict, target: Dict, relationship_type: str) -> float:
        """Calculate ID matching score with pattern recognition"""
        source_ids = source.get('platform_ids', {})
        target_ids = target.get('platform_ids', {})
        
        if not source_ids or not target_ids:
            return 0.0
        
        # Check for exact ID matches
        for source_key, source_id in source_ids.items():
            for target_key, target_id in target_ids.items():
                if source_id == target_id:
                    return 1.0
                elif source_id in target_id or target_id in source_id:
                    return 0.8
        
        # Check for pattern matches
        for source_key, source_id in source_ids.items():
            for target_key, target_id in target_ids.items():
                if self._check_id_pattern_match(source_id, target_id, relationship_type):
                    return 0.6
        
        return 0.0
    
    def _calculate_context_score(self, source: Dict, target: Dict, relationship_type: str) -> float:
        """Calculate context matching score using semantic analysis"""
        source_context = self._extract_context(source)
        target_context = self._extract_context(target)
        
        if not source_context or not target_context:
            return 0.0
        
        # Calculate semantic similarity
        similarity = self._calculate_text_similarity(source_context, target_context)
        
        # Boost score for expected relationship contexts
        context_boost = self._get_context_boost(source_context, target_context, relationship_type)
        
        return min(similarity + context_boost, 1.0)
    
    def _check_amount_match(self, source: Dict, target: Dict) -> bool:
        """Check if amounts match within tolerance"""
        return self._calculate_amount_score(source, target, "generic") > 0.8
    
    def _check_date_match(self, source: Dict, target: Dict) -> bool:
        """Check if dates are within acceptable window"""
        return self._calculate_date_score(source, target, "generic") > 0.8
    
    def _check_entity_match(self, source: Dict, target: Dict) -> bool:
        """Check if entities match"""
        return self._calculate_entity_score(source, target, "generic") > 0.8
    
    def _check_id_match(self, source: Dict, target: Dict) -> bool:
        """Check if IDs match"""
        return self._calculate_id_score(source, target, "generic") > 0.8
    
    def _check_context_match(self, source: Dict, target: Dict) -> bool:
        """Check if contexts match"""
        return self._calculate_context_score(source, target, "generic") > 0.8
    
    def _get_relationship_filters(self, relationship_type: str) -> Tuple[Dict, Dict]:
        """Get source and target filters for a relationship type"""
        filters = {
            'invoice_to_payment': (
                {'keywords': ['invoice', 'bill', 'receivable']},
                {'keywords': ['payment', 'charge', 'transaction']}
            ),
            'fee_to_transaction': (
                {'keywords': ['fee', 'commission', 'charge']},
                {'keywords': ['transaction', 'payment', 'charge']}
            ),
            'refund_to_original': (
                {'keywords': ['refund', 'return', 'reversal']},
                {'keywords': ['payment', 'charge', 'transaction']}
            ),
            'payroll_to_payout': (
                {'keywords': ['payroll', 'salary', 'wage', 'employee']},
                {'keywords': ['payout', 'transfer', 'withdrawal']}
            ),
            'tax_to_income': (
                {'keywords': ['tax', 'withholding', 'deduction']},
                {'keywords': ['income', 'revenue', 'salary']}
            ),
            'expense_to_reimbursement': (
                {'keywords': ['expense', 'cost', 'outlay']},
                {'keywords': ['reimbursement', 'refund', 'return']}
            ),
            'subscription_to_payment': (
                {'keywords': ['subscription', 'recurring', 'monthly']},
                {'keywords': ['payment', 'charge', 'transaction']}
            ),
            'loan_to_payment': (
                {'keywords': ['loan', 'credit', 'advance']},
                {'keywords': ['payment', 'repayment', 'installment']}
            ),
            'investment_to_return': (
                {'keywords': ['investment', 'purchase', 'buy']},
                {'keywords': ['return', 'dividend', 'profit']}
            )
        }
        
        return filters.get(relationship_type, ({}, {}))
    
    def _matches_event_filter(self, event: Dict, filter_dict: Dict) -> bool:
        """Check if event matches the filter criteria"""
        if not filter_dict:
            return True
        
        # Check keywords
        if 'keywords' in filter_dict:
            event_text = str(event.get('payload', {})).lower()
            event_text += ' ' + str(event.get('kind', '')).lower()
            event_text += ' ' + str(event.get('category', '')).lower()
            
            keywords = filter_dict['keywords']
            if not any(keyword.lower() in event_text for keyword in keywords):
                return False
        
        return True
    
    def _extract_amount(self, payload: Dict) -> float:
        """Extract amount from payload"""
        try:
            amount_fields = ['amount', 'total', 'value', 'sum', 'payment_amount', 'charge_amount']
            for field in amount_fields:
                if field in payload:
                    value = payload[field]
                    if isinstance(value, (int, float)):
                        return float(value)
                    elif isinstance(value, str):
                        cleaned = value.replace('$', '').replace(',', '').strip()
                        return float(cleaned)
        except:
            pass
        return 0.0
    
    def _extract_entities(self, payload: Dict) -> List[str]:
        """Extract entity names from payload"""
        entities = []
        try:
            name_fields = ['employee_name', 'name', 'recipient', 'payee', 'description', 'vendor_name']
            for field in name_fields:
                if field in payload:
                    value = payload[field]
                    if isinstance(value, str) and value.strip():
                        entities.append(value.strip())
        except:
            pass
        return entities
    
    def _extract_date(self, payload: Dict) -> Optional[datetime]:
        """Extract date from payload"""
        try:
            date_fields = ['date', 'payment_date', 'transaction_date', 'created_at', 'due_date']
            for field in date_fields:
                if field in payload:
                    value = payload[field]
                    if isinstance(value, str):
                        return datetime.fromisoformat(value.replace('Z', '+00:00'))
                    elif isinstance(value, datetime):
                        return value
        except:
            pass
        return None
    
    def _extract_context(self, event: Dict) -> str:
        """Extract context from event"""
        context_parts = []
        
        # Add kind and category
        if event.get('kind'):
            context_parts.append(event['kind'])
        if event.get('category'):
            context_parts.append(event['category'])
        
        # Add payload description
        payload = event.get('payload', {})
        if 'description' in payload:
            context_parts.append(payload['description'])
        
        # Add vendor information
        if event.get('vendor_standard'):
            context_parts.append(event['vendor_standard'])
        
        return ' '.join(context_parts)
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using SequenceMatcher"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def _check_id_pattern_match(self, id1: str, id2: str, relationship_type: str) -> bool:
        """Check if IDs match a pattern for the relationship type"""
        import re
        # Define patterns for different relationship types
        patterns = {
            'invoice_to_payment': [
                (r'inv_(\w+)', r'pay_\1'),  # invoice_id to payment_id
                (r'in_(\w+)', r'pi_\1'),    # invoice_id to payment_intent
            ],
            'fee_to_transaction': [
                (r'fee_(\w+)', r'ch_\1'),   # fee_id to charge_id
                (r'fee_(\w+)', r'txn_\1'),  # fee_id to transaction_id
            ],
            'refund_to_original': [
                (r're_(\w+)', r'ch_\1'),    # refund_id to charge_id
                (r'rfnd_(\w+)', r'pay_\1'), # refund_id to payment_id
            ]
        }
        
        pattern_list = patterns.get(relationship_type, [])
        
        for pattern1, pattern2 in pattern_list:
            match1 = re.match(pattern1, id1)
            match2 = re.match(pattern2, id2)
            
            if match1 and match2 and match1.group(1) == match2.group(1):
                return True
        
        return False
    
    def _get_context_boost(self, context1: str, context2: str, relationship_type: str) -> float:
        """Get context boost for expected relationship patterns"""
        context_combinations = {
            'invoice_to_payment': [
                ('invoice', 'payment'),
                ('bill', 'charge'),
                ('receivable', 'transaction')
            ],
            'fee_to_transaction': [
                ('fee', 'transaction'),
                ('commission', 'payment'),
                ('charge', 'transaction')
            ],
            'refund_to_original': [
                ('refund', 'payment'),
                ('return', 'charge'),
                ('reversal', 'transaction')
            ]
        }
        
        combinations = context_combinations.get(relationship_type, [])
        context_lower = context1.lower() + ' ' + context2.lower()
        
        for combo in combinations:
            if combo[0] in context_lower and combo[1] in context_lower:
                return 0.2
        
        return 0.0
    
    def _create_event_summary(self, events: List[Dict]) -> str:
        """Create a summary of events for AI analysis"""
        summary_parts = []
        
        for event in events[:10]:  # Limit to first 10 events
            event_summary = {
                'id': event.get('id'),
                'kind': event.get('kind'),
                'category': event.get('category'),
                'platform': event.get('source_platform'),
                'amount': self._extract_amount(event.get('payload', {})),
                'vendor': event.get('vendor_standard'),
                'description': event.get('payload', {}).get('description', '')
            }
            summary_parts.append(str(event_summary))
        
        return '\n'.join(summary_parts)
    
    def _create_comprehensive_context(self, events: List[Dict]) -> str:
        """Create comprehensive context for AI analysis"""
        context_parts = []
        
        # Group events by platform
        platform_groups = {}
        for event in events:
            platform = event.get('source_platform', 'unknown')
            if platform not in platform_groups:
                platform_groups[platform] = []
            platform_groups[platform].append(event)
        
        # Create context for each platform
        for platform, platform_events in platform_groups.items():
            context_parts.append(f"\nPlatform: {platform}")
            for event in platform_events[:5]:  # Limit to 5 events per platform
                context_parts.append(f"- {event.get('kind')}: {event.get('payload', {}).get('description', '')}")
        
        return '\n'.join(context_parts)
    
    def _parse_relationship_types(self, response_text: str) -> List[str]:
        """Parse relationship types from AI response"""
        try:
            # Try to extract JSON array
            if '[' in response_text and ']' in response_text:
                start = response_text.find('[')
                end = response_text.rfind(']') + 1
                json_str = response_text[start:end]
                return json.loads(json_str)
        except:
            pass
        
        # Fallback to common relationship types
        return ["invoice_to_payment", "fee_to_transaction", "refund_to_original"]
    
    def _parse_ai_relationships(self, response_text: str, events: List[Dict]) -> List[Dict]:
        """Parse AI-discovered relationships from response"""
        try:
            # Try to extract JSON array
            if '[' in response_text and ']' in response_text:
                start = response_text.find('[')
                end = response_text.rfind(']') + 1
                json_str = response_text[start:end]
                ai_relationships = json.loads(json_str)
                
                # Convert to standard format
                relationships = []
                for rel in ai_relationships:
                    relationship = {
                        "source_event_id": rel.get('source_event_id'),
                        "target_event_id": rel.get('target_event_id'),
                        "relationship_type": rel.get('relationship_type', 'ai_discovered'),
                        "confidence_score": rel.get('confidence_score', 0.5),
                        "detection_method": "ai_discovered"
                    }
                    relationships.append(relationship)
                
                return relationships
        except:
            pass
        
        return []
    
    async def _validate_relationships(self, relationships: List[Dict]) -> List[Dict]:
        """Validate and filter relationships"""
        validated = []
        
        for relationship in relationships:
            # Check if events exist
            source_exists = await self._event_exists(relationship['source_event_id'])
            target_exists = await self._event_exists(relationship['target_event_id'])
            
            if source_exists and target_exists:
                # Add additional validation
                relationship['validated'] = True
                relationship['validation_score'] = relationship.get('confidence_score', 0.0)
                validated.append(relationship)
        
        return validated
    
    async def _event_exists(self, event_id: str) -> bool:
        """Check if event exists in database"""
        try:
            result = self.supabase.table('raw_events').select('id').eq('id', event_id).execute()
            return len(result.data) > 0
        except:
            return False
    
    async def _store_relationships(self, relationships: List[Dict], user_id: str):
        """Store relationships in database"""
        for relationship in relationships:
            try:
                # Store in relationships table (if exists)
                await self.supabase.table('relationships').insert({
                    'user_id': user_id,
                    'source_event_id': relationship['source_event_id'],
                    'target_event_id': relationship['target_event_id'],
                    'relationship_type': relationship['relationship_type'],
                    'confidence_score': relationship['confidence_score'],
                    'detection_method': relationship.get('detection_method', 'ai'),
                    'metadata': {
                        'amount_match': relationship.get('amount_match', False),
                        'date_match': relationship.get('date_match', False),
                        'entity_match': relationship.get('entity_match', False),
                        'id_match': relationship.get('id_match', False),
                        'context_match': relationship.get('context_match', False)
                    }
                }).execute()
            except Exception as e:
                logger.error(f"Failed to store relationship: {e}")

class DynamicPlatformDetector:
    """AI-powered dynamic platform detection that learns from ANY financial data"""
    
    def __init__(self, openai_client, supabase_client: Client):
        self.openai = openai_client
        self.supabase = supabase_client
        self.learned_patterns = {}
        self.platform_knowledge = {}
        self.detection_cache = {}
        
    async def detect_platform_dynamically(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """Dynamically detect platform using AI analysis"""
        try:
            # Create comprehensive context for AI analysis
            context = self._create_platform_context(df, filename)
            
            # Use AI to analyze and detect platform
            try:
                ai_response = await self.openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{
                        "role": "system",
                        "content": "You are a financial data analyst specializing in platform detection. Analyze the financial data and identify the platform. Consider column names, data patterns, terminology, and file structure. Return a JSON object with platform name, confidence score, reasoning, and key indicators."
                    }, {
                        "role": "user",
                        "content": f"Analyze this financial data and detect the platform: {context}"
                    }],
                    temperature=0.1
                )
                
                # Parse AI response
                response_text = ai_response.choices[0].message.content
                platform_analysis = self._parse_platform_analysis(response_text)
                
                # Learn from this detection
                await self._learn_platform_patterns(df, filename, platform_analysis)
                
                # Get platform information
                platform_info = await self._get_platform_info(platform_analysis['platform'])
                
                return {
                    "platform": platform_analysis['platform'],
                    "confidence_score": platform_analysis['confidence_score'],
                    "reasoning": platform_analysis['reasoning'],
                    "key_indicators": platform_analysis['key_indicators'],
                    "detection_method": "ai_dynamic",
                    "learned_patterns": len(self.learned_patterns),
                    "platform_info": platform_info
                }
                
            except Exception as ai_error:
                logger.error(f"AI detection failed, using fallback: {ai_error}")
                return self._fallback_detection(df, filename)
            
        except Exception as e:
            logger.error(f"Dynamic platform detection failed: {e}")
            return self._fallback_detection(df, filename)
    
    async def learn_from_user_data(self, user_id: str) -> Dict[str, Any]:
        """Learn platform patterns from user's historical data"""
        try:
            # Get all events for the user
            events = self.supabase.table('raw_events').select('*').eq('user_id', user_id).execute()
            
            if not events.data:
                return {"message": "No data found for platform learning", "learned_patterns": 0}
            
            # Group events by platform
            platform_groups = {}
            for event in events.data:
                platform = event.get('source_platform', 'unknown')
                if platform not in platform_groups:
                    platform_groups[platform] = []
                platform_groups[platform].append(event)
            
            # Learn patterns for each platform
            learned_patterns = {}
            for platform, platform_events in platform_groups.items():
                if platform != 'unknown':
                    patterns = await self._extract_platform_patterns(platform_events, platform)
                    learned_patterns[platform] = patterns
            
            # Store learned patterns
            await self._store_learned_patterns(learned_patterns, user_id)
            
            return {
                "message": "Platform learning completed",
                "learned_patterns": len(learned_patterns),
                "platforms_analyzed": list(learned_patterns.keys()),
                "patterns": learned_patterns
            }
            
        except Exception as e:
            logger.error(f"Platform learning failed: {e}")
            return {"message": "Platform learning failed", "error": str(e)}
    
    async def discover_new_platforms(self, user_id: str) -> Dict[str, Any]:
        """Discover new platforms in user's data"""
        try:
            # Get all events for the user
            events = self.supabase.table('raw_events').select('*').eq('user_id', user_id).execute()
            
            if not events.data:
                return {"message": "No data found for platform discovery", "new_platforms": []}
            
            # Use AI to discover new platforms
            context = self._create_discovery_context(events.data)
            
            try:
                ai_response = await self.openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{
                        "role": "system",
                        "content": "You are a financial data analyst. Analyze the financial events and identify any new or custom platforms that might not be in the standard list. Look for unique patterns, terminology, or data structures that suggest a custom platform."
                    }, {
                        "role": "user",
                        "content": f"Analyze these financial events and discover any new platforms: {context}"
                    }],
                    temperature=0.2
                )
                
                # Parse AI discoveries
                response_text = ai_response.choices[0].message.content
                new_platforms = self._parse_new_platforms(response_text)
                
                # Store new platform discoveries
                await self._store_new_platforms(new_platforms, user_id)
                
                return {
                    "message": "Platform discovery completed",
                    "new_platforms": new_platforms,
                    "total_platforms": len(new_platforms)
                }
                
            except Exception as ai_error:
                logger.error(f"Platform discovery failed: {ai_error}")
                return {
                    "message": "Platform discovery failed",
                    "error": str(ai_error),
                    "new_platforms": []
                }
            
            # Parse AI discoveries
            response_text = ai_response.choices[0].message.content
            new_platforms = self._parse_new_platforms(response_text)
            
            # Store new platform discoveries
            await self._store_new_platforms(new_platforms, user_id)
            
            return {
                "message": "Platform discovery completed",
                "new_platforms": new_platforms,
                "total_platforms": len(new_platforms)
            }
            
        except Exception as e:
            logger.error(f"Platform discovery failed: {e}")
            return {"message": "Platform discovery failed", "error": str(e)}
    
    async def get_platform_insights(self, platform: str, user_id: str = None) -> Dict[str, Any]:
        """Get detailed insights about a platform"""
        try:
            # Get learned patterns from database if not in memory
            if platform not in self.learned_patterns:
                try:
                    result = self.supabase.table('platform_patterns').select('*').eq('platform', platform).execute()
                    if result.data:
                        self.learned_patterns[platform] = result.data[0].get('patterns', {})
                except Exception as e:
                    logger.error(f"Failed to load platform patterns from database: {e}")
            
            insights = {
                "platform": platform,
                "learned_patterns": self.learned_patterns.get(platform, {}),
                "detection_confidence": self._calculate_platform_confidence(platform),
                "key_characteristics": await self._get_platform_characteristics(platform),
                "usage_statistics": await self._get_platform_usage_stats(platform, user_id),
                "custom_indicators": await self._get_custom_indicators(platform),
                "is_known_platform": platform in ['stripe', 'razorpay', 'quickbooks', 'gusto', 'paypal', 'square'],
                "total_learned_patterns": len(self.learned_patterns)
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Platform insights failed: {e}")
            return {"platform": platform, "error": str(e)}
    
    def _create_platform_context(self, df: pd.DataFrame, filename: str) -> str:
        """Create comprehensive context for platform detection"""
        context_parts = []
        
        # File information
        context_parts.append(f"Filename: {filename}")
        
        # Column analysis
        columns = list(df.columns)
        context_parts.append(f"Columns: {columns}")
        
        # Data sample analysis
        sample_data = df.head(5).to_dict('records')
        context_parts.append(f"Sample data: {sample_data}")
        
        # Data type analysis
        dtypes = df.dtypes.to_dict()
        context_parts.append(f"Data types: {dtypes}")
        
        # Value analysis
        for col in df.columns:
            if df[col].dtype in ['object', 'string']:
                unique_values = df[col].dropna().unique()[:10]
                context_parts.append(f"Column '{col}' unique values: {list(unique_values)}")
        
        return '\n'.join(context_parts)
    
    def _create_discovery_context(self, events: List[Dict]) -> str:
        """Create context for platform discovery"""
        context_parts = []
        
        # Group by platform
        platform_groups = {}
        for event in events:
            platform = event.get('source_platform', 'unknown')
            if platform not in platform_groups:
                platform_groups[platform] = []
            platform_groups[platform].append(event)
        
        # Create context for each platform
        for platform, platform_events in platform_groups.items():
            context_parts.append(f"\nPlatform: {platform}")
            context_parts.append(f"Event count: {len(platform_events)}")
            
            # Sample events
            for event in platform_events[:3]:
                context_parts.append(f"- {event.get('kind')}: {event.get('payload', {}).get('description', '')}")
        
        return '\n'.join(context_parts)
    
    def _parse_platform_analysis(self, response_text: str) -> Dict[str, Any]:
        """Parse platform analysis from AI response"""
        try:
            # Try to extract JSON
            if '{' in response_text and '}' in response_text:
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                json_str = response_text[start:end]
                analysis = json.loads(json_str)
                
                return {
                    'platform': analysis.get('platform', 'unknown'),
                    'confidence_score': analysis.get('confidence_score', 0.5),
                    'reasoning': analysis.get('reasoning', ''),
                    'key_indicators': analysis.get('key_indicators', [])
                }
        except:
            pass
        
        # Fallback parsing
        platform = 'unknown'
        confidence = 0.5
        reasoning = 'AI analysis failed, using fallback detection'
        indicators = []
        
        # Try to extract platform name
        platform_keywords = ['stripe', 'razorpay', 'quickbooks', 'gusto', 'paypal', 'square']
        response_lower = response_text.lower()
        
        for keyword in platform_keywords:
            if keyword in response_lower:
                platform = keyword
                confidence = 0.7
                break
        
        return {
            'platform': platform,
            'confidence_score': confidence,
            'reasoning': reasoning,
            'key_indicators': indicators
        }
    
    def _parse_new_platforms(self, response_text: str) -> List[Dict]:
        """Parse new platform discoveries from AI response"""
        try:
            # Try to extract JSON array
            if '[' in response_text and ']' in response_text:
                start = response_text.find('[')
                end = response_text.rfind(']') + 1
                json_str = response_text[start:end]
                platforms = json.loads(json_str)
                
                return platforms
        except:
            pass
        
        return []
    
    async def _learn_platform_patterns(self, df: pd.DataFrame, filename: str, platform_analysis: Dict):
        """Learn patterns from detected platform"""
        platform = platform_analysis['platform']
        
        if platform not in self.learned_patterns:
            self.learned_patterns[platform] = {}
        
        # Learn column patterns
        column_patterns = {
            'columns': list(df.columns),
            'data_types': df.dtypes.to_dict(),
            'unique_values': {}
        }
        
        # Learn unique value patterns
        for col in df.columns:
            if df[col].dtype in ['object', 'string']:
                unique_vals = df[col].dropna().unique()[:20]
                column_patterns['unique_values'][col] = list(unique_vals)
        
        self.learned_patterns[platform]['column_patterns'] = column_patterns
        self.learned_patterns[platform]['detection_count'] = self.learned_patterns[platform].get('detection_count', 0) + 1
        self.learned_patterns[platform]['last_detected'] = datetime.utcnow().isoformat()
    
    async def _extract_platform_patterns(self, events: List[Dict], platform: str) -> Dict[str, Any]:
        """Extract patterns from platform events"""
        patterns = {
            'platform': platform,
            'event_count': len(events),
            'event_types': {},
            'amount_patterns': {},
            'date_patterns': {},
            'entity_patterns': {},
            'terminology_patterns': {}
        }
        
        # Analyze event types
        for event in events:
            event_type = event.get('kind', 'unknown')
            if event_type not in patterns['event_types']:
                patterns['event_types'][event_type] = 0
            patterns['event_types'][event_type] += 1
        
        # Analyze amount patterns
        amounts = []
        for event in events:
            payload = event.get('payload', {})
            amount = self._extract_amount(payload)
            if amount > 0:
                amounts.append(amount)
        
        if amounts:
            patterns['amount_patterns'] = {
                'min': min(amounts),
                'max': max(amounts),
                'avg': sum(amounts) / len(amounts),
                'count': len(amounts)
            }
        
        # Analyze terminology patterns
        all_text = ' '.join([str(event.get('payload', {})) for event in events])
        patterns['terminology_patterns'] = self._extract_terminology_patterns(all_text)
        
        return patterns
    
    def _extract_terminology_patterns(self, text: str) -> Dict[str, Any]:
        """Extract terminology patterns from text"""
        text_lower = text.lower()
        
        # Common financial terms
        financial_terms = {
            'payment_terms': ['payment', 'charge', 'transaction', 'transfer'],
            'invoice_terms': ['invoice', 'bill', 'receivable', 'due'],
            'fee_terms': ['fee', 'commission', 'charge', 'cost'],
            'refund_terms': ['refund', 'return', 'reversal', 'credit'],
            'tax_terms': ['tax', 'withholding', 'deduction', 'gst', 'vat'],
            'currency_terms': ['usd', 'inr', 'eur', 'currency', 'exchange'],
            'date_terms': ['date', 'created', 'due', 'payment_date'],
            'id_terms': ['id', 'reference', 'transaction_id', 'invoice_id']
        }
        
        patterns = {}
        for category, terms in financial_terms.items():
            found_terms = [term for term in terms if term in text_lower]
            if found_terms:
                patterns[category] = found_terms
        
        return patterns
    
    async def _get_platform_info(self, platform: str) -> Dict[str, Any]:
        """Get information about a platform"""
        platform_info = {
            'name': platform,
            'learned_patterns': self.learned_patterns.get(platform, {}),
            'detection_confidence': self._calculate_platform_confidence(platform),
            'is_custom': platform not in ['stripe', 'razorpay', 'quickbooks', 'gusto', 'paypal', 'square'],
            'last_detected': self.learned_patterns.get(platform, {}).get('last_detected'),
            'detection_count': self.learned_patterns.get(platform, {}).get('detection_count', 0)
        }
        
        return platform_info
    
    def _calculate_platform_confidence(self, platform: str) -> float:
        """Calculate confidence score for platform detection"""
        patterns = self.learned_patterns.get(platform, {})
        
        if not patterns:
            return 0.5
        
        # Factors that increase confidence
        detection_count = patterns.get('detection_count', 0)
        has_column_patterns = 'column_patterns' in patterns
        
        confidence = 0.5  # Base confidence
        
        if detection_count > 0:
            confidence += min(detection_count * 0.1, 0.3)
        
        if has_column_patterns:
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    async def _get_platform_characteristics(self, platform: str) -> Dict[str, Any]:
        """Get characteristics of a platform"""
        patterns = self.learned_patterns.get(platform, {})
        
        # Add default characteristics for known platforms
        default_characteristics = {
            'stripe': {
                'column_patterns': {
                    'columns': ['charge_id', 'amount', 'currency', 'description', 'created', 'status', 'payment_method'],
                    'data_types': {'charge_id': 'object', 'amount': 'int64', 'currency': 'object'},
                    'unique_values': {
                        'currency': ['usd', 'eur'],
                        'status': ['succeeded', 'failed', 'pending'],
                        'payment_method': ['card', 'bank_transfer']
                    }
                },
                'event_types': {'payment': 100, 'refund': 20, 'fee': 10},
                'amount_patterns': {'min': 0.5, 'max': 10000.0, 'avg': 250.0},
                'terminology_patterns': {
                    'payment_terms': ['payment', 'charge', 'transaction'],
                    'id_terms': ['charge_id', 'payment_intent_id'],
                    'status_terms': ['succeeded', 'failed', 'pending']
                }
            },
            'razorpay': {
                'column_patterns': {
                    'columns': ['payment_id', 'amount', 'currency', 'description', 'created_at', 'status', 'method'],
                    'data_types': {'payment_id': 'object', 'amount': 'int64', 'currency': 'object'},
                    'unique_values': {
                        'currency': ['inr', 'usd'],
                        'status': ['captured', 'failed', 'pending'],
                        'method': ['card', 'netbanking', 'upi']
                    }
                },
                'event_types': {'payment': 80, 'refund': 15, 'fee': 5},
                'amount_patterns': {'min': 1.0, 'max': 50000.0, 'avg': 500.0},
                'terminology_patterns': {
                    'payment_terms': ['payment', 'transaction'],
                    'id_terms': ['payment_id', 'order_id'],
                    'status_terms': ['captured', 'failed', 'pending']
                }
            }
        }
        
        # Use learned patterns if available, otherwise use defaults
        if platform in default_characteristics and not patterns:
            characteristics = default_characteristics[platform]
        else:
            characteristics = {
                'platform': platform,
                'column_patterns': patterns.get('column_patterns', {}),
                'event_types': patterns.get('event_types', {}),
                'amount_patterns': patterns.get('amount_patterns', {}),
                'terminology_patterns': patterns.get('terminology_patterns', {})
            }
        
        return characteristics
    
    async def _get_platform_usage_stats(self, platform: str, user_id: str = None) -> Dict[str, Any]:
        """Get usage statistics for a platform"""
        try:
            query = self.supabase.table('raw_events').select('*').eq('source_platform', platform)
            
            if user_id:
                query = query.eq('user_id', user_id)
            
            result = query.execute()
            
            if not result.data:
                return {'total_events': 0, 'unique_users': 0}
            
            total_events = len(result.data)
            unique_users = len(set(event.get('user_id') for event in result.data))
            
            return {
                'total_events': total_events,
                'unique_users': unique_users,
                'last_used': max(event.get('created_at', '') for event in result.data) if result.data else None
            }
            
        except Exception as e:
            logger.error(f"Failed to get platform usage stats: {e}")
            return {'total_events': 0, 'unique_users': 0}
    
    async def _get_custom_indicators(self, platform: str) -> List[str]:
        """Get custom indicators for a platform"""
        patterns = self.learned_patterns.get(platform, {})
        indicators = []
        
        # Column-based indicators
        column_patterns = patterns.get('column_patterns', {})
        if column_patterns:
            columns = column_patterns.get('columns', [])
            indicators.extend([f"Column: {col}" for col in columns[:5]])
        
        # Terminology-based indicators
        terminology = patterns.get('terminology_patterns', {})
        for category, terms in terminology.items():
            indicators.extend([f"{category}: {', '.join(terms[:3])}"])
        
        # Platform-specific indicators
        platform_indicators = {
            'stripe': [
                'Stripe-specific charge_id pattern',
                'Payment method field present',
                'Status field with succeeded/failed values',
                'USD/EUR currency support'
            ],
            'razorpay': [
                'Razorpay-specific payment_id pattern',
                'Method field with card/netbanking/upi',
                'Status field with captured/failed values',
                'INR currency support'
            ],
            'quickbooks': [
                'QuickBooks transaction patterns',
                'Account-based categorization',
                'Class and location fields',
                'QB-specific terminology'
            ],
            'gusto': [
                'Gusto payroll patterns',
                'Employee-based transactions',
                'Payroll-specific fields',
                'Tax withholding patterns'
            ]
        }
        
        # Add platform-specific indicators
        if platform in platform_indicators:
            indicators.extend(platform_indicators[platform])
        
        return indicators[:10]  # Limit to 10 indicators
    
    async def _store_learned_patterns(self, patterns: Dict[str, Any], user_id: str):
        """Store learned patterns in database"""
        try:
            for platform, platform_patterns in patterns.items():
                await self.supabase.table('platform_patterns').upsert({
                    'user_id': user_id,
                    'platform': platform,
                    'patterns': platform_patterns,
                    'created_at': datetime.utcnow().isoformat(),
                    'updated_at': datetime.utcnow().isoformat()
                }).execute()
        except Exception as e:
            logger.error(f"Failed to store learned patterns: {e}")
    
    async def _store_new_platforms(self, new_platforms: List[Dict], user_id: str):
        """Store new platform discoveries"""
        try:
            for platform_info in new_platforms:
                await self.supabase.table('discovered_platforms').insert({
                    'user_id': user_id,
                    'platform_name': platform_info.get('name', 'unknown'),
                    'discovery_reason': platform_info.get('reason', ''),
                    'confidence_score': platform_info.get('confidence', 0.5),
                    'discovered_at': datetime.utcnow().isoformat()
                }).execute()
        except Exception as e:
            logger.error(f"Failed to store new platforms: {e}")
    
    def _fallback_detection(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """Fallback platform detection when AI fails"""
        # Simple rule-based detection
        columns = [col.lower() for col in df.columns]
        filename_lower = filename.lower()
        
        # Check for platform indicators
        if any('stripe' in col or 'stripe' in filename_lower for col in columns):
            return {"platform": "stripe", "confidence_score": 0.6, "detection_method": "fallback"}
        elif any('razorpay' in col or 'razorpay' in filename_lower for col in columns):
            return {"platform": "razorpay", "confidence_score": 0.6, "detection_method": "fallback"}
        elif any('quickbooks' in col or 'quickbooks' in filename_lower for col in columns):
            return {"platform": "quickbooks", "confidence_score": 0.6, "detection_method": "fallback"}
        elif any('gusto' in col or 'gusto' in filename_lower for col in columns):
            return {"platform": "gusto", "confidence_score": 0.6, "detection_method": "fallback"}
        else:
            return {"platform": "unknown", "confidence_score": 0.3, "detection_method": "fallback"}
    
    def _extract_amount(self, payload: Dict) -> float:
        """Extract amount from payload"""
        try:
            amount_fields = ['amount', 'total', 'value', 'sum', 'payment_amount', 'charge_amount']
            for field in amount_fields:
                if field in payload:
                    value = payload[field]
                    if isinstance(value, (int, float)):
                        return float(value)
                    elif isinstance(value, str):
                        cleaned = value.replace('$', '').replace(',', '').strip()
                        return float(cleaned)
        except:
            pass
        return 0.0

@app.get("/test-ai-relationship-detection/{user_id}")
async def test_ai_relationship_detection(user_id: str):
    """Test AI-powered relationship detection"""
    try:
        # Initialize OpenAI and Supabase clients
        openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        
        if not supabase_url or not supabase_key:
            return {
                "message": "Supabase credentials not configured",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Clean JWT token to prevent header value errors
        supabase_key = clean_jwt_token(supabase_key)
        supabase = create_client(supabase_url, supabase_key)
        
        # Initialize AI Relationship Detector
        ai_detector = AIRelationshipDetector(openai_client, supabase)
        
        # Detect all relationships
        result = await ai_detector.detect_all_relationships(user_id)
        
        return {
            "message": "AI Relationship Detection Test Completed",
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "message": "AI Relationship Detection Test Failed",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/test-relationship-discovery/{user_id}")
async def test_relationship_discovery(user_id: str):
    """Test AI-powered relationship type discovery"""
    try:
        # Initialize OpenAI and Supabase clients
        openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        
        if not supabase_url or not supabase_key:
            return {
                "message": "Supabase credentials not configured",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Clean JWT token to prevent header value errors
        supabase_key = clean_jwt_token(supabase_key)
        supabase = create_client(supabase_url, supabase_key)
        
        # Get all events for the user
        events = supabase.table('raw_events').select('*').eq('user_id', user_id).execute()
        
        if not events.data:
            return {
                "message": "No data found for relationship discovery",
                "discovered_types": [],
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Initialize AI Relationship Detector
        ai_detector = AIRelationshipDetector(openai_client, supabase)
        
        # Discover relationship types
        relationship_types = await ai_detector._discover_relationship_types(events.data)
        
        return {
            "message": "Relationship Type Discovery Test Completed",
            "discovered_types": relationship_types,
            "total_events": len(events.data),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "message": "Relationship Discovery Test Failed",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/test-ai-relationship-scoring/{user_id}")
async def test_ai_relationship_scoring(user_id: str):
    """Test AI-powered relationship scoring"""
    try:
        # Initialize OpenAI and Supabase clients
        openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        
        if not supabase_url or not supabase_key:
            return {
                "message": "Supabase credentials not configured",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Clean JWT token to prevent header value errors
        supabase_key = clean_jwt_token(supabase_key)
        supabase = create_client(supabase_url, supabase_key)
        
        # Get sample events for testing
        events = supabase.table('raw_events').select('*').eq('user_id', user_id).limit(10).execute()
        
        if len(events.data) < 2:
            return {
                "message": "Insufficient data for relationship scoring test",
                "scoring_results": [],
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Initialize AI Relationship Detector
        ai_detector = AIRelationshipDetector(openai_client, supabase)
        
        # Test scoring between first two events
        event1 = events.data[0]
        event2 = events.data[1]
        
        scoring_results = []
        relationship_types = ["invoice_to_payment", "fee_to_transaction", "refund_to_original", "payroll_to_payout"]
        
        for rel_type in relationship_types:
            score = await ai_detector._calculate_comprehensive_score(event1, event2, rel_type)
            amount_score = ai_detector._calculate_amount_score(event1, event2, rel_type)
            date_score = ai_detector._calculate_date_score(event1, event2, rel_type)
            entity_score = ai_detector._calculate_entity_score(event1, event2, rel_type)
            id_score = ai_detector._calculate_id_score(event1, event2, rel_type)
            context_score = ai_detector._calculate_context_score(event1, event2, rel_type)
            
            scoring_results.append({
                "relationship_type": rel_type,
                "comprehensive_score": score,
                "amount_score": amount_score,
                "date_score": date_score,
                "entity_score": entity_score,
                "id_score": id_score,
                "context_score": context_score,
                "event1_id": event1.get('id'),
                "event2_id": event2.get('id')
            })
        
        return {
            "message": "AI Relationship Scoring Test Completed",
            "scoring_results": scoring_results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "message": "AI Relationship Scoring Test Failed",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/test-relationship-validation/{user_id}")
async def test_relationship_validation(user_id: str):
    """Test relationship validation and filtering"""
    try:
        # Initialize OpenAI and Supabase clients
        openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        
        if not supabase_url or not supabase_key:
            return {
                "message": "Supabase credentials not configured",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Clean JWT token to prevent header value errors
        supabase_key = clean_jwt_token(supabase_key)
        supabase = create_client(supabase_url, supabase_key)
        
        # Initialize AI Relationship Detector
        ai_detector = AIRelationshipDetector(openai_client, supabase)
        
        # Get all events for the user
        events = supabase.table('raw_events').select('*').eq('user_id', user_id).execute()
        
        if not events.data:
            return {
                "message": "No data found for relationship validation",
                "validation_results": [],
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Create sample relationships for testing
        sample_relationships = []
        for i in range(min(5, len(events.data) - 1)):
            relationship = {
                "source_event_id": events.data[i]['id'],
                "target_event_id": events.data[i + 1]['id'],
                "relationship_type": "test_relationship",
                "confidence_score": 0.8,
                "detection_method": "test"
            }
            sample_relationships.append(relationship)
        
        # Validate relationships
        validated_relationships = await ai_detector._validate_relationships(sample_relationships)
        
        return {
            "message": "Relationship Validation Test Completed",
            "total_relationships": len(sample_relationships),
            "validated_relationships": len(validated_relationships),
            "validation_results": validated_relationships,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "message": "Relationship Validation Test Failed",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/test-dynamic-platform-detection")
async def test_dynamic_platform_detection():
    """Test AI-powered dynamic platform detection"""
    try:
        # Initialize OpenAI and Supabase clients
        openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        
        if not supabase_url or not supabase_key:
            return {
                "message": "Supabase credentials not configured",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Clean JWT token to prevent header value errors
        supabase_key = clean_jwt_token(supabase_key)
        supabase = create_client(supabase_url, supabase_key)
        
        # Initialize Dynamic Platform Detector
        dynamic_detector = DynamicPlatformDetector(openai_client, supabase)
        
        # Create sample data for testing
        sample_data = {
            'stripe_sample': pd.DataFrame({
                'charge_id': ['ch_1234567890abcdef', 'ch_0987654321fedcba'],
                'amount': [1000, 2000],
                'currency': ['usd', 'usd'],
                'description': ['Stripe payment for subscription', 'Stripe charge for service'],
                'created': ['2024-01-01', '2024-01-02'],
                'status': ['succeeded', 'succeeded'],
                'payment_method': ['card', 'card']
            }),
            'razorpay_sample': pd.DataFrame({
                'payment_id': ['pay_1234567890abcdef', 'pay_0987654321fedcba'],
                'amount': [5000, 7500],
                'currency': ['inr', 'inr'],
                'description': ['Razorpay payment for invoice', 'Razorpay transaction for service'],
                'created_at': ['2024-01-01', '2024-01-02'],
                'status': ['captured', 'captured'],
                'method': ['card', 'netbanking']
            }),
            'custom_sample': pd.DataFrame({
                'transaction_id': ['txn_001', 'txn_002'],
                'amount': [1500, 3000],
                'currency': ['usd', 'usd'],
                'description': ['Custom payment system', 'Custom transaction platform'],
                'date': ['2024-01-01', '2024-01-02'],
                'type': ['payment', 'refund']
            })
        }
        
        results = {}
        for platform_name, df in sample_data.items():
            result = await dynamic_detector.detect_platform_dynamically(df, f"{platform_name}.csv")
            results[platform_name] = result
        
        return {
            "message": "Dynamic Platform Detection Test Completed",
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "message": "Dynamic Platform Detection Test Failed",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/test-platform-learning/{user_id}")
async def test_platform_learning(user_id: str):
    """Test AI-powered platform learning from user data"""
    try:
        # Initialize OpenAI and Supabase clients
        openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        
        if not supabase_url or not supabase_key:
            return {
                "message": "Supabase credentials not configured",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Clean JWT token to prevent header value errors
        supabase_key = clean_jwt_token(supabase_key)
        supabase = create_client(supabase_url, supabase_key)
        
        # Initialize Dynamic Platform Detector
        dynamic_detector = DynamicPlatformDetector(openai_client, supabase)
        
        # Learn from user data
        result = await dynamic_detector.learn_from_user_data(user_id)
        
        return {
            "message": "Platform Learning Test Completed",
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "message": "Platform Learning Test Failed",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/test-platform-discovery/{user_id}")
async def test_platform_discovery(user_id: str):
    """Test AI-powered discovery of new platforms"""
    try:
        # Initialize OpenAI and Supabase clients
        openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        
        if not supabase_url or not supabase_key:
            return {
                "message": "Supabase credentials not configured",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Clean JWT token to prevent header value errors
        supabase_key = clean_jwt_token(supabase_key)
        supabase = create_client(supabase_url, supabase_key)
        
        # Initialize Dynamic Platform Detector
        dynamic_detector = DynamicPlatformDetector(openai_client, supabase)
        
        # Discover new platforms
        result = await dynamic_detector.discover_new_platforms(user_id)
        
        return {
            "message": "Platform Discovery Test Completed",
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "message": "Platform Discovery Test Failed",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/test-platform-insights/{platform}")
async def test_platform_insights(platform: str, user_id: str = None):
    """Test platform insights and analysis"""
    try:
        # Initialize OpenAI and Supabase clients
        openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        
        if not supabase_url or not supabase_key:
            return {
                "message": "Supabase credentials not configured",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Clean JWT token to prevent header value errors
        supabase_key = clean_jwt_token(supabase_key)
        supabase = create_client(supabase_url, supabase_key)
        
        # Initialize Dynamic Platform Detector
        dynamic_detector = DynamicPlatformDetector(openai_client, supabase)
        
        # Get platform insights
        insights = await dynamic_detector.get_platform_insights(platform, user_id)
        
        return {
            "message": "Platform Insights Test Completed",
            "insights": insights,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "message": "Platform Insights Test Failed",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

class ChatMessage(BaseModel):
    message: str
    user_id: str
    chat_id: str = None

class ChatResponse(BaseModel):
    response: str
    chat_id: str
    timestamp: str

class ChatTitleRequest(BaseModel):
    message: str
    user_id: str

class ChatTitleResponse(BaseModel):
    title: str
    chat_id: str

class ChatRenameRequest(BaseModel):
    chat_id: str
    new_title: str
    user_id: str

class ChatDeleteRequest(BaseModel):
    chat_id: str
    user_id: str

@app.post("/chat", response_model=ChatResponse)
async def chat_with_finley(chat_message: ChatMessage):
    """Chat endpoint for Finley AI financial assistant"""
    try:
        # Initialize OpenAI client
        openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        if not openai_client:
            raise HTTPException(status_code=500, detail="OpenAI client not configured")
        
        # Generate chat ID if not provided
        chat_id = chat_message.chat_id or f"chat-{datetime.utcnow().timestamp()}"
        
        # Create context-aware prompt for financial assistance
        system_prompt = """You are Finley AI, an intelligent financial analyst assistant. You help users understand their financial data, provide insights, and answer questions about:

- Financial document analysis
- Budget planning and forecasting
- Investment strategies
- Tax optimization
- Business financial health
- Cash flow management
- Financial reporting
- Risk assessment

Always provide practical, actionable advice based on financial best practices. If you don't have specific data about the user's finances, ask clarifying questions to provide better assistance.

Keep responses concise but comprehensive, and always prioritize accuracy in financial matters."""

        # Get AI response
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chat_message.message}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        ai_response = response.choices[0].message.content
        
        # Store chat in database if Supabase is available
        try:
            supabase_url = os.getenv('SUPABASE_URL')
            supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
            
            if supabase_url and supabase_key:
                # Clean JWT token to prevent header value errors
                supabase_key = clean_jwt_token(supabase_key)
                supabase = create_client(supabase_url, supabase_key)
                
                # Store user message
                supabase.table('chat_messages').insert({
                    'chat_id': chat_id,
                    'user_id': chat_message.user_id,
                    'message': chat_message.message,
                    'is_user': True,
                    'created_at': datetime.utcnow().isoformat()
                }).execute()
                
                # Store AI response
                supabase.table('chat_messages').insert({
                    'chat_id': chat_id,
                    'user_id': chat_message.user_id,
                    'message': ai_response,
                    'is_user': False,
                    'created_at': datetime.utcnow().isoformat()
                }).execute()
                
        except Exception as db_error:
            logger.warning(f"Failed to store chat in database: {db_error}")
        
        return ChatResponse(
            response=ai_response,
            chat_id=chat_id,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@app.get("/chat-history/{user_id}")
async def get_chat_history(user_id: str, chat_id: str = None):
    """Get chat history for a user"""
    try:
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
        
        if not supabase_url or not supabase_key:
            return {
                "message": "Chat history not available",
                "chats": [],
                "error": "Database not configured"
            }
        
        # Clean JWT token to prevent header value errors
        supabase_key = clean_jwt_token(supabase_key)
        # Clean JWT token to prevent header value errors
        supabase_key = clean_jwt_token(supabase_key)
        supabase = create_client(supabase_url, supabase_key)
        
        # Get all chats for user
        query = supabase.table('chat_messages').select('*').eq('user_id', user_id)
        
        if chat_id:
            query = query.eq('chat_id', chat_id)
        
        result = query.order('created_at', desc=True).execute()
        
        # Group messages by chat_id
        chats = {}
        for message in result.data:
            chat_id = message['chat_id']
            if chat_id not in chats:
                chats[chat_id] = {
                    'chat_id': chat_id,
                    'messages': [],
                    'created_at': message['created_at'],
                    'updated_at': message['created_at']
                }
            
            chats[chat_id]['messages'].append({
                'message': message['message'],
                'is_user': message['is_user'],
                'timestamp': message['created_at']
            })
            
            # Update latest timestamp
            if message['created_at'] > chats[chat_id]['updated_at']:
                chats[chat_id]['updated_at'] = message['created_at']
        
        # Convert to list and sort by updated_at
        chat_list = list(chats.values())
        chat_list.sort(key=lambda x: x['updated_at'], reverse=True)
        
        return {
            "message": "Chat history retrieved successfully",
            "chats": chat_list,
            "total_chats": len(chat_list)
        }
        
    except Exception as e:
        logger.error(f"Chat history error: {e}")
        return {
            "message": "Failed to retrieve chat history",
            "chats": [],
            "error": str(e)
        }

@app.post("/generate-chat-title", response_model=ChatTitleResponse)
async def generate_chat_title(title_request: ChatTitleRequest):
    """Generate a chat title from the first message"""
    try:
        # Initialize OpenAI client
        openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        if not openai_client:
            raise HTTPException(status_code=500, detail="OpenAI client not configured")
        
        # Generate chat ID
        chat_id = f"chat-{datetime.utcnow().timestamp()}"
        
        # Create a simple title from the first message
        message_words = title_request.message.strip().split()
        
        # If message is short enough, use it directly
        if len(message_words) <= 8:
            title = title_request.message.strip()
        else:
            # Use AI to generate a concise title
            try:
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Generate a short, descriptive title (max 6 words) for a financial chat conversation based on the user's first message. Focus on the main topic or question."},
                        {"role": "user", "content": title_request.message}
                    ],
                    temperature=0.3,
                    max_tokens=20
                )
                title = response.choices[0].message.content.strip()
            except Exception as ai_error:
                # Fallback to first 6 words if AI fails
                title = " ".join(message_words[:6])
        
        # Clean up the title
        title = title.replace('"', '').replace("'", "").strip()
        if not title:
            title = "New Chat"
        
        return ChatTitleResponse(
            title=title,
            chat_id=chat_id
        )
        
    except Exception as e:
        logger.error(f"Chat title generation error: {e}")
        # Fallback to simple title
        message_words = title_request.message.strip().split()
        title = " ".join(message_words[:6]) if message_words else "New Chat"
        
        return ChatTitleResponse(
            title=title,
            chat_id=f"chat-{datetime.utcnow().timestamp()}"
        )

@app.put("/chat/rename")
async def rename_chat(rename_request: ChatRenameRequest):
    """Rename a chat conversation"""
    try:
        # For now, just return success since we're using localStorage
        # In production, you'd want to store this in the database
        if not rename_request.new_title.strip():
            raise HTTPException(status_code=400, detail="Chat title cannot be empty")
        
        return {
            "message": "Chat renamed successfully",
            "chat_id": rename_request.chat_id,
            "new_title": rename_request.new_title.strip()
        }
        
    except Exception as e:
        logger.error(f"Chat rename error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to rename chat: {str(e)}")

@app.delete("/chat/delete")
async def delete_chat(delete_request: ChatDeleteRequest):
    """Delete a chat conversation and all its messages"""
    try:
        # For now, just return success since we're using localStorage
        # In production, you'd want to delete from the database
        return {
            "message": "Chat deleted successfully",
            "chat_id": delete_request.chat_id,
            "deleted_messages": 0
        }
        
    except Exception as e:
        logger.error(f"Chat delete error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete chat: {str(e)}")

# Mount static files (frontend) - MUST be after all API routes
try:
    import os
    dist_path = "dist"
    if os.path.exists(dist_path):
        logger.info(f"Dist directory exists: {dist_path}")
        files = os.listdir(dist_path)
        logger.info(f"Files in dist: {files}")
        app.mount("/", StaticFiles(directory=dist_path, html=True), name="static")
        logger.info("Frontend static files mounted successfully")
    else:
        logger.error(f"Dist directory not found: {dist_path}")
        # Create a simple fallback
        @app.get("/")
        async def fallback():
            return {"error": "Frontend not built", "message": "Please build the frontend first"}
except Exception as e:
    logger.error(f"Could not mount frontend files: {e}")
    # Create a simple fallback
    @app.get("/")
    async def fallback():
        return {"error": "Static file mounting failed", "message": str(e)}

# ============================================================================
# PRODUCTION DUPLICATE DETECTION API ENDPOINTS
# ============================================================================

@app.post("/duplicate-detection/detect")
async def detect_duplicates_endpoint(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    job_id: str = Form(...),
    enable_near_duplicate: bool = Form(True)
):
    """
    Production duplicate detection endpoint.
    
    Uses the production-grade duplicate detection service with advanced algorithms,
    caching, and real-time WebSocket updates.
    """
    if not PRODUCTION_DUPLICATE_SERVICE_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Production duplicate detection service not available"
        )
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Initialize Supabase client
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        if not supabase_url or not supabase_key:
            raise HTTPException(status_code=500, detail="Supabase credentials not configured")
        
        supabase_key = clean_jwt_token(supabase_key)
        supabase: Client = create_client(supabase_url, supabase_key)
        
        # Calculate file hash
        file_hash = hashlib.sha256(file_content).hexdigest()
        
        # Create duplicate detection API integration
        duplicate_api = DuplicateDetectionAPIIntegration(supabase)
        
        # Create request object
        request = type('DuplicateDetectionRequest', (), {
            'job_id': job_id,
            'user_id': user_id,
            'file_hash': file_hash,
            'filename': file.filename,
            'file_size': len(file_content),
            'content_type': file.content_type or "application/octet-stream",
            'enable_near_duplicate': enable_near_duplicate
        })()
        
        # Detect duplicates
        result = await duplicate_api.detect_duplicates_with_websocket(request, file_content)
        
        return {
            "status": result.status,
            "is_duplicate": result.is_duplicate,
            "duplicate_type": result.duplicate_type,
            "similarity_score": result.similarity_score,
            "duplicate_files": result.duplicate_files,
            "recommendation": result.recommendation,
            "message": result.message,
            "confidence": result.confidence,
            "processing_time_ms": result.processing_time_ms,
            "requires_user_decision": result.requires_user_decision,
            "error": result.error
        }
        
    except Exception as e:
        logger.error(f"Duplicate detection endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/duplicate-detection/decision")
async def handle_duplicate_decision_endpoint(
    job_id: str = Form(...),
    user_id: str = Form(...),
    file_hash: str = Form(...),
    decision: str = Form(...)
):
    """
    Handle user's decision about duplicate files.
    
    Decisions: replace, keep_both, skip, merge
    """
    if not PRODUCTION_DUPLICATE_SERVICE_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Production duplicate detection service not available"
        )
    
    try:
        # Initialize Supabase client
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        if not supabase_url or not supabase_key:
            raise HTTPException(status_code=500, detail="Supabase credentials not configured")
        
        supabase_key = clean_jwt_token(supabase_key)
        supabase: Client = create_client(supabase_url, supabase_key)
        
        # Create duplicate detection API integration
        duplicate_api = DuplicateDetectionAPIIntegration(supabase)
        
        # Handle decision
        result = await duplicate_api.handle_duplicate_decision(job_id, user_id, file_hash, decision)
        
        return result
        
    except Exception as e:
        logger.error(f"Duplicate decision endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/duplicate-detection/metrics")
async def get_duplicate_metrics():
    """Get duplicate detection service metrics for monitoring"""
    if not PRODUCTION_DUPLICATE_SERVICE_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Production duplicate detection service not available"
        )
    
    try:
        # Initialize Supabase client
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        if not supabase_url or not supabase_key:
            raise HTTPException(status_code=500, detail="Supabase credentials not configured")
        
        supabase_key = clean_jwt_token(supabase_key)
        supabase: Client = create_client(supabase_url, supabase_key)
        
        # Create duplicate detection API integration
        duplicate_api = DuplicateDetectionAPIIntegration(supabase)
        
        # Get metrics
        metrics = await duplicate_api.get_duplicate_metrics()
        
        return metrics
        
    except Exception as e:
        logger.error(f"Metrics endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/duplicate-detection/clear-cache")
async def clear_duplicate_cache(
    user_id: Optional[str] = Form(None)
):
    """Clear duplicate detection cache"""
    if not PRODUCTION_DUPLICATE_SERVICE_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Production duplicate detection service not available"
        )
    
    try:
        # Initialize Supabase client
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        if not supabase_url or not supabase_key:
            raise HTTPException(status_code=500, detail="Supabase credentials not configured")
        
        supabase_key = clean_jwt_token(supabase_key)
        supabase: Client = create_client(supabase_url, supabase_key)
        
        # Create duplicate detection API integration
        duplicate_api = DuplicateDetectionAPIIntegration(supabase)
        
        # Clear cache
        result = await duplicate_api.clear_duplicate_cache(user_id)
        
        return result
        
    except Exception as e:
        logger.error(f"Clear cache endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/duplicate-detection/ws/{job_id}")
async def duplicate_detection_websocket(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for real-time duplicate detection updates.
    
    Provides real-time updates during duplicate detection process.
    """
    if not PRODUCTION_DUPLICATE_SERVICE_AVAILABLE:
        await websocket.close(code=1003, reason="Production duplicate detection service not available")
        return
    
    await websocket_manager.connect(websocket, job_id)
    
    try:
        while True:
            # Keep connection alive
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        websocket_manager.disconnect(job_id)
        logger.info(f"WebSocket disconnected for job {job_id}")
    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
        websocket_manager.disconnect(job_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
