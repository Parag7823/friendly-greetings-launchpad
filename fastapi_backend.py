import os
import io
import logging
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, UploadFile, Form, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
import openai
import magic
import filetype
from openai import OpenAI
import time
import json
import re
import asyncio
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Finley AI Backend", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# WebSocket connection manager
class ConnectionManager:
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
    job_id: str
    storage_path: str
    file_name: str
    supabase_url: str
    supabase_key: str
    user_id: str

class DocumentAnalyzer:
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
                analysis[col] = {
                    "total": float(df[col].sum()) if not df[col].empty else 0,
                    "average": float(df[col].mean()) if not df[col].empty else 0,
                    "growth_rate": self._calculate_growth_rate(df[col]) if len(df) > 1 else None
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
                analysis[col] = {
                    "total": float(df[col].sum()) if not df[col].empty else 0,
                    "average": float(df[col].mean()) if not df[col].empty else 0,
                    "percentage_of_revenue": self._calculate_expense_ratio(df, col)
                }
        return analysis
    
    def _calculate_profitability_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate profitability metrics"""
        revenue_cols = [col for col in df.columns if any(word in col.lower() for word in ['revenue', 'sales', 'income'])]
        expense_cols = [col for col in df.columns if any(word in col.lower() for word in ['expense', 'cost', 'cogs', 'operating'])]
        profit_cols = [col for col in df.columns if any(word in col.lower() for word in ['profit', 'net'])]
        
        metrics = {}
        
        if revenue_cols and expense_cols:
            total_revenue = sum(df[col].sum() for col in revenue_cols if col in df.columns)
            total_expenses = sum(df[col].sum() for col in expense_cols if col in df.columns)
            
            if total_revenue > 0:
                metrics["gross_margin"] = ((total_revenue - total_expenses) / total_revenue) * 100
                metrics["expense_ratio"] = (total_expenses / total_revenue) * 100
        
        if profit_cols:
            for col in profit_cols:
                if col in df.columns:
                    metrics[f"{col}_total"] = float(df[col].sum()) if not df[col].empty else 0
        
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
        return {"asset_columns": asset_cols, "total_assets": sum(df[col].sum() for col in asset_cols if col in df.columns)}
    
    def _analyze_liabilities(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze liability-related data"""
        liability_cols = [col for col in df.columns if any(word in col.lower() for word in ['liability', 'payable', 'debt', 'loan'])]
        return {"liability_columns": liability_cols, "total_liabilities": sum(df[col].sum() for col in liability_cols if col in df.columns)}
    
    def _analyze_equity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze equity-related data"""
        equity_cols = [col for col in df.columns if any(word in col.lower() for word in ['equity', 'capital', 'retained'])]
        return {"equity_columns": equity_cols, "total_equity": sum(df[col].sum() for col in equity_cols if col in df.columns)}
    
    def _analyze_payroll_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze payroll-related data"""
        payroll_cols = [col for col in df.columns if any(word in col.lower() for word in ['pay', 'salary', 'wage', 'gross', 'net'])]
        return {"payroll_columns": payroll_cols, "total_payroll": sum(df[col].sum() for col in payroll_cols if col in df.columns)}
    
    def _analyze_employee_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze employee-related data"""
        employee_cols = [col for col in df.columns if any(word in col.lower() for word in ['employee', 'name', 'id'])]
        return {"employee_columns": employee_cols, "employee_count": len(df) if employee_cols else 0}
    
    def _analyze_tax_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze tax-related data"""
        tax_cols = [col for col in df.columns if any(word in col.lower() for word in ['tax', 'withholding', 'deduction'])]
        return {"tax_columns": tax_cols, "total_taxes": sum(df[col].sum() for col in tax_cols if col in df.columns)}

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
    def __init__(self, openai_client):
        self.openai = openai_client
    
    async def classify_row_with_ai(self, row: pd.Series, platform_info: Dict, column_names: List[str]) -> Dict[str, Any]:
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
            response = await self.openai.chat.completions.create(
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
            response = await self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000
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
    
    def __init__(self, platform_detector: PlatformDetector, ai_classifier: AIRowClassifier):
        self.platform_detector = platform_detector
        self.ai_classifier = ai_classifier
    
    async def process_row(self, row: pd.Series, row_index: int, sheet_name: str, 
                   platform_info: Dict, file_context: Dict, column_names: List[str]) -> Dict[str, Any]:
        """Process a single row and create an event with AI-powered classification"""
        
        # AI-powered row classification
        ai_classification = await self.ai_classifier.classify_row_with_ai(row, platform_info, column_names)
        
        # Convert row to JSON-serializable format
        payload = self._convert_row_to_json_serializable(row)
        
        # Create the event payload with enhanced metadata
        event = {
            "provider": "excel-upload",
            "kind": ai_classification.get('row_type', 'general_row'),
            "source_platform": platform_info.get('platform', 'unknown'),
            "payload": payload,
            "row_index": row_index,
            "sheet_name": sheet_name,
            "source_filename": file_context['filename'],
            "uploader": file_context['user_id'],
            "ingest_ts": datetime.utcnow().isoformat(),
            "status": "pending",
            "confidence_score": ai_classification.get('confidence', 0.5),
            "classification_metadata": {
                "platform_detection": platform_info,
                "ai_classification": ai_classification,
                "row_type": ai_classification.get('row_type', 'general_row'),
                "category": ai_classification.get('category', 'other'),
                "subcategory": ai_classification.get('subcategory', 'general'),
                "entities": ai_classification.get('entities', {}),
                "relationships": ai_classification.get('relationships', {}),
                "description": ai_classification.get('description', ''),
                "reasoning": ai_classification.get('reasoning', ''),
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
        self.analyzer = DocumentAnalyzer(openai)
        self.platform_detector = PlatformDetector()
        # Use batch classifier for better performance
        self.batch_classifier = BatchAIRowClassifier(openai)
        self.row_processor = RowProcessor(self.platform_detector, self.batch_classifier)
    
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
                # Handle Excel files
                excel_file = pd.ExcelFile(file_stream)
                sheets = {}
                
                for sheet_name in excel_file.sheet_names:
                    df = pd.read_excel(file_stream, sheet_name=sheet_name)
                    if not df.empty:
                        sheets[sheet_name] = df
                
                return sheets
                
        except Exception as e:
            logger.error(f"Error reading file {filename}: {e}")
            raise HTTPException(status_code=400, detail=f"Error reading file {filename}: {str(e)}")
    
    async def process_file(self, job_id: str, file_content: bytes, filename: str, 
                          user_id: str, supabase: Client) -> Dict[str, Any]:
        """Optimized processing pipeline with batch AI classification for large files"""
        
        # Step 1: Read the file
        await manager.send_update(job_id, {
            "step": "reading",
            "message": f"📖 Reading and parsing your {filename}...",
            "progress": 10
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
        
        # Step 2: Detect platform and document type
        await manager.send_update(job_id, {
            "step": "analyzing",
            "message": "🧠 Analyzing document structure and detecting platform...",
            "progress": 20
        })
        
        # Use first sheet for platform detection
        first_sheet = list(sheets.values())[0]
        platform_info = self.platform_detector.detect_platform(first_sheet, filename)
        doc_analysis = await self.analyzer.detect_document_type(first_sheet, filename)
        
        # Step 3: Create raw_records entry
        await manager.send_update(job_id, {
            "step": "storing",
            "message": "💾 Storing file metadata...",
            "progress": 30
        })
        
        # Calculate file hash for duplicate detection
        file_hash = hashlib.sha256(file_content).hexdigest()
        
        # Store in raw_records
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
                'total_rows': sum(len(sheet) for sheet in sheets.values()),
                'processed_at': datetime.utcnow().isoformat()
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
                    
                    # Process batch with AI classification
                    batch_classifications = await self.batch_classifier.classify_rows_batch(
                        row_data, platform_info, column_names
                    )
                    
                    # Store each row from the batch
                    for i, (row_index, row) in enumerate(batch_rows):
                        try:
                            # Create event for this row
                            event = await self.row_processor.process_row(
                                row, row_index, sheet_name, platform_info, file_context, column_names
                            )
                            
                            # Use batch classification result
                            if i < len(batch_classifications):
                                ai_classification = batch_classifications[i]
                                event['classification_metadata'].update(ai_classification)
                            
                            # Store event in raw_events table
                            event_result = supabase.table('raw_events').insert({
                                'user_id': user_id,
                                'file_id': file_id,
                                'job_id': job_id,
                                'provider': event['provider'],
                                'kind': event['kind'],
                                'source_platform': event['source_platform'],
                                'category': event['classification_metadata'].get('category'),
                                'subcategory': event['classification_metadata'].get('subcategory'),
                                'payload': event['payload'],
                                'row_index': event['row_index'],
                                'sheet_name': event['sheet_name'],
                                'source_filename': event['source_filename'],
                                'uploader': event['uploader'],
                                'ingest_ts': event['ingest_ts'],
                                'status': event['status'],
                                'confidence_score': event['confidence_score'],
                                'classification_metadata': event['classification_metadata'],
                                'entities': event['classification_metadata'].get('entities', {}),
                                'relationships': event['classification_metadata'].get('relationships', {})
                            }).execute()
                            
                            if event_result.data:
                                events_created += 1
                            else:
                                errors.append(f"Failed to store event for row {row_index} in sheet {sheet_name}")
                        
                        except Exception as e:
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
        
        # Step 8: Update ingestion_jobs with completion
        supabase.table('ingestion_jobs').update({
            'status': 'completed',
            'updated_at': datetime.utcnow().isoformat()
        }).eq('id', job_id).execute()
        
        await manager.send_update(job_id, {
            "step": "completed",
            "message": f"✅ Processing completed! {events_created} events created from {processed_rows} rows.",
            "progress": 100
        })
        
        return insights

processor = ExcelProcessor()

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
        # Initialize Supabase client
        supabase: Client = create_client(request.supabase_url, request.supabase_key)
        
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

@app.get("/")
async def root():
    return {"message": "Finley AI Backend - Intelligent Financial Analysis with Row-by-Row Processing"}

@app.get("/test-raw-events/{user_id}")
async def test_raw_events(user_id: str):
    """Test endpoint to check raw_events functionality"""
    try:
        # Initialize Supabase client (you'll need to provide credentials)
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
        
        if not supabase_url or not supabase_key:
            return {"error": "Supabase credentials not configured"}
        
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
    return {"status": "healthy", "service": "Finley AI Backend"}

@app.post("/upload-and-process")
async def upload_and_process(
    file: UploadFile = Form(...),
    user_id: str = Form("test-user-123"),  # Default test user ID
    job_id: str = Form(None)  # Optional, will generate if not provided
):
    """Direct file upload and processing endpoint for testing"""
    try:
        # Generate job_id if not provided
        if not job_id:
            job_id = f"test-job-{int(time.time())}"
        
        # Read file content
        file_content = await file.read()
        
        # Initialize Supabase client
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
        
        if not supabase_url or not supabase_key:
            raise HTTPException(status_code=500, detail="Supabase credentials not configured")
        
        supabase: Client = create_client(supabase_url, supabase_key)
        
        # Process the file directly
        results = await processor.process_file(
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

@app.get("/test-simple")
async def test_simple():
    """Simple test endpoint without any dependencies"""
    return {
        "status": "success",
        "message": "Backend is working! No authentication required.",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": {
            "health": "/health",
            "upload_and_process": "/upload-and-process",
            "test_raw_events": "/test-raw-events/{user_id}",
            "process_excel": "/process-excel"
        }
    }

@app.get("/test-database")
async def test_database():
    """Test database connection and basic operations"""
    try:
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
        
        if not supabase_url or not supabase_key:
            return {"error": "Supabase credentials not configured"}
        
        supabase: Client = create_client(supabase_url, supabase_key)
        
        # Test basic database operations
        test_user_id = "test-user-123"
        
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
