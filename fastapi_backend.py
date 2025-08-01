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
            sample_data = df.head(10).to_dict('records')
            column_names = list(df.columns)
            
            # Analyze data patterns
            numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
            date_columns = [col for col in column_names if any(word in col.lower() for word in ['date', 'time', 'period', 'month', 'year'])]
            
            # Enhanced prompt with more context and detailed classification rules
            prompt = f"""
            Analyze this financial document and classify it accurately using the detailed rules below.
            
            FILENAME: {filename}
            COLUMN NAMES: {column_names}
            NUMERIC COLUMNS: {numeric_columns}
            DATE COLUMNS: {date_columns}
            SAMPLE DATA: {sample_data}
            
            CLASSIFICATION RULES:
            
            1. INCOME STATEMENT: Contains Revenue, Sales, Income, COGS, Cost of Goods, Operating Expenses, Net Profit, Net Income, Gross Profit, EBIT, EBITDA
            2. BALANCE SHEET: Contains Assets, Liabilities, Equity, Cash, Accounts Receivable, Accounts Payable, Inventory, Fixed Assets, Current Assets, Current Liabilities
            3. CASH FLOW: Contains Operating Cash Flow, Investing Cash Flow, Financing Cash Flow, Net Cash Flow, Cash from Operations, Cash from Investing, Cash from Financing
            4. PAYROLL DATA: Contains Employee, Salary, Wage, Pay Period, Gross Pay, Net Pay, Tax, Employee Name, Employee ID, Hours Worked, Overtime
            5. EXPENSE DATA: Contains Expense, Cost, Payment, Vendor, Category, Amount, Expense Type, Expense Category, Vendor Name
            6. REVENUE DATA: Contains Revenue, Sales, Income, Customer, Invoice, Amount, Sales Revenue, Service Revenue, Product Revenue
            7. GENERAL LEDGER: Contains Account, Debit, Credit, Balance, Transaction, Journal Entry, Account Number, Account Name
            8. BUDGET: Contains Budget, Planned, Actual, Variance, Period, Budget vs Actual, Forecast, Target
            9. BANK STATEMENT: Contains Transaction Date, Description, Debit, Credit, Balance, Account Number, Check Number
            10. CREDIT CARD STATEMENT: Contains Transaction Date, Merchant, Amount, Category, Statement Period, Credit Limit
            
            PLATFORM DETECTION RULES:
            
            1. GUSTO: Employee names, Pay periods, Gross/Net pay, Tax deductions, Employee SSN, Pay rate, Hours worked
            2. QUICKBOOKS: Account names, Memo fields, Transaction types, QB-specific terms, Ref number, Split transactions, Class tracking
            3. XERO: Contact names, Invoice numbers, Xero-specific formatting, Reference numbers, Tracking categories, Line amounts
            4. RAZORPAY: Transaction IDs, Merchant IDs, Payment status, Settlement data, Order IDs, Payment methods
            5. FRESHBOOKS: Invoice data, Client information, Time tracking, Project tracking, Client names, Invoice numbers
            6. WAVE: Wave-specific account names, Transaction types, Business account structure, Wave-specific terminology
            7. SAGE: Sage-specific terminology, Account structures, Journal entries, Sage 50/100 specific fields
            8. NETSUITE: NetSuite-specific fields, Enterprise account structures, Internal IDs, Transaction IDs
            9. STRIPE: Payment intents, Customer IDs, Charge IDs, Transfer IDs, Fee amounts, Payment methods
            10. SQUARE: Transaction IDs, Location IDs, Device IDs, Tender types, Square-specific payment data
            
            ANALYSIS REQUIREMENTS:
            
            1. Look at column names first - they are the strongest indicators
            2. Analyze the data patterns in the sample
            3. Consider the filename for additional context
            4. Check for platform-specific terminology or formatting
            5. Determine confidence based on how many indicators match
            
            Return a detailed JSON response with this exact structure:
            {{
                "document_type": "income_statement|balance_sheet|cash_flow|payroll_data|expense_data|revenue_data|general_ledger|budget|bank_statement|credit_card_statement|unknown",
                "source_platform": "gusto|quickbooks|xero|razorpay|freshbooks|wave|sage|netsuite|stripe|square|unknown",
                "confidence": 0.95,
                "key_columns": ["col1", "col2"],
                "analysis": "Detailed explanation of classification reasoning",
                "data_patterns": {{
                    "has_revenue_data": true/false,
                    "has_expense_data": true/false,
                    "has_employee_data": true/false,
                    "has_account_data": true/false,
                    "has_transaction_data": true/false,
                    "time_period": "monthly|quarterly|yearly|unknown"
                }},
                "classification_reasoning": "Step-by-step explanation of why this classification was chosen",
                "platform_indicators": ["indicator1", "indicator2"],
                "document_indicators": ["indicator1", "indicator2"]
            }}
            """
            
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            result = response.choices[0].message.content
            # Parse JSON from response
            import json
            try:
                parsed_result = json.loads(result)
                
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
                return {
                    "document_type": "unknown",
                    "source_platform": "unknown", 
                    "confidence": 0.5,
                    "key_columns": list(df.columns),
                    "analysis": "Could not determine document type - AI response parsing failed",
                    "data_patterns": {
                        "has_revenue_data": False,
                        "has_expense_data": False,
                        "has_employee_data": False,
                        "has_account_data": False,
                        "has_transaction_data": False,
                        "time_period": "unknown"
                    },
                    "classification_reasoning": "Failed to parse AI analysis response",
                    "platform_indicators": [],
                    "document_indicators": []
                }
                
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
    """Detects the source platform of uploaded files"""
    
    def __init__(self):
        self.platform_patterns = {
            'gusto': {
                'keywords': ['gusto', 'payroll', 'employee', 'salary', 'wage'],
                'columns': ['employee_name', 'employee_id', 'pay_period', 'gross_pay', 'net_pay'],
                'confidence_threshold': 0.7
            },
            'razorpay': {
                'keywords': ['razorpay', 'payment', 'transaction', 'merchant', 'settlement'],
                'columns': ['transaction_id', 'merchant_id', 'amount', 'status', 'created_at'],
                'confidence_threshold': 0.7
            },
            'quickbooks': {
                'keywords': ['quickbooks', 'qb', 'accounting', 'invoice', 'bill'],
                'columns': ['account', 'memo', 'amount', 'date', 'type'],
                'confidence_threshold': 0.7
            },
            'xero': {
                'keywords': ['xero', 'invoice', 'contact', 'account'],
                'columns': ['contact_name', 'invoice_number', 'amount', 'date'],
                'confidence_threshold': 0.7
            }
        }
    
    def detect_platform(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """Detect the source platform based on column names and data patterns"""
        filename_lower = filename.lower()
        columns_lower = [col.lower() for col in df.columns]
        
        best_match = {
            'platform': 'unknown',
            'confidence': 0.0,
            'matched_columns': [],
            'reasoning': 'No clear platform match found'
        }
        
        for platform, patterns in self.platform_patterns.items():
            confidence = 0.0
            matched_columns = []
            
            # Check filename keywords
            for keyword in patterns['keywords']:
                if keyword in filename_lower:
                    confidence += 0.3
            
            # Check column name matches
            for expected_col in patterns['columns']:
                for actual_col in columns_lower:
                    if expected_col in actual_col or actual_col in expected_col:
                        matched_columns.append(actual_col)
                        confidence += 0.2
            
            # Check data patterns (basic)
            if len(matched_columns) > 0:
                confidence += 0.1
            
            if confidence > best_match['confidence']:
                best_match = {
                    'platform': platform,
                    'confidence': min(confidence, 1.0),
                    'matched_columns': matched_columns,
                    'reasoning': f'Matched {len(matched_columns)} columns and filename patterns'
                }
        
        return best_match

class RowProcessor:
    """Processes individual rows and creates events"""
    
    def __init__(self, platform_detector: PlatformDetector):
        self.platform_detector = platform_detector
    
    def process_row(self, row: pd.Series, row_index: int, sheet_name: str, 
                   platform_info: Dict, file_context: Dict) -> Dict[str, Any]:
        """Process a single row and create an event"""
        
        # Determine row type based on content
        row_type = self._determine_row_type(row, platform_info)
        
        # Convert row to JSON-serializable format
        payload = self._convert_row_to_json_serializable(row)
        
        # Create the event payload
        event = {
            "provider": "excel-upload",
            "kind": row_type,
            "source_platform": platform_info.get('platform', 'unknown'),
            "payload": payload,
            "row_index": row_index,
            "sheet_name": sheet_name,
            "source_filename": file_context['filename'],
            "uploader": file_context['user_id'],
            "ingest_ts": datetime.utcnow().isoformat(),
            "status": "pending",
            "confidence_score": platform_info.get('confidence', 0.5),
            "classification_metadata": {
                "platform_detection": platform_info,
                "row_type": row_type,
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
    
    def _determine_row_type(self, row: pd.Series, platform_info: Dict) -> str:
        """Determine the type of row based on content and platform"""
        platform = platform_info.get('platform', 'unknown')
        
        # Platform-specific row type detection
        if platform == 'gusto':
            if any('employee' in str(col).lower() for col in row.index):
                return 'payroll_row'
            elif any('salary' in str(col).lower() for col in row.index):
                return 'salary_row'
        
        elif platform == 'razorpay':
            if any('transaction' in str(col).lower() for col in row.index):
                return 'transaction_row'
            elif any('payment' in str(col).lower() for col in row.index):
                return 'payment_row'
        
        elif platform == 'quickbooks':
            if any('invoice' in str(col).lower() for col in row.index):
                return 'invoice_row'
            elif any('bill' in str(col).lower() for col in row.index):
                return 'bill_row'
        
        # Generic detection based on content
        row_str = ' '.join(str(val).lower() for val in row.values if pd.notna(val))
        
        if any(word in row_str for word in ['salary', 'wage', 'payroll', 'employee']):
            return 'payroll_row'
        elif any(word in row_str for word in ['revenue', 'income', 'sales']):
            return 'revenue_row'
        elif any(word in row_str for word in ['expense', 'cost', 'payment']):
            return 'expense_row'
        elif any(word in row_str for word in ['invoice', 'bill']):
            return 'invoice_row'
        elif any(word in row_str for word in ['transaction', 'payment']):
            return 'transaction_row'
        else:
            return 'general_row'

class ExcelProcessor:
    def __init__(self):
        self.analyzer = DocumentAnalyzer(openai)
        self.platform_detector = PlatformDetector()
        self.row_processor = RowProcessor(self.platform_detector)
    
    async def detect_file_type(self, file_content: bytes, filename: str) -> str:
        """Detect file type using multiple methods"""
        try:
            # Try python-magic first
            file_type = magic.from_buffer(file_content, mime=True)
            return file_type
        except:
            # Fallback to filetype
            try:
                kind = filetype.guess(file_content)
                return kind.mime if kind else 'application/octet-stream'
            except:
                # Last resort: file extension
                if filename.endswith('.xlsx'):
                    return 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                elif filename.endswith('.xls'):
                    return 'application/vnd.ms-excel'
                elif filename.endswith('.csv'):
                    return 'text/csv'
                return 'unknown'
    
    async def read_excel_file(self, file_content: bytes, filename: str) -> Dict[str, pd.DataFrame]:
        """Read Excel file with multiple sheets and return as dictionary"""
        file_type = await self.detect_file_type(file_content, filename)
        
        try:
            if 'csv' in file_type or filename.endswith('.csv'):
                # CSV files - single sheet
                df = pd.read_csv(io.BytesIO(file_content))
                return {'Sheet1': df}
            else:
                # Excel files - multiple sheets
                excel_file = pd.ExcelFile(io.BytesIO(file_content))
                sheets = {}
                for sheet_name in excel_file.sheet_names:
                    sheets[sheet_name] = pd.read_excel(excel_file, sheet_name=sheet_name)
                return sheets
        except Exception as e:
            logger.error(f"Error reading file {filename}: {e}")
            raise HTTPException(status_code=400, detail=f"Could not read file: {str(e)}")
    
    async def process_file(self, job_id: str, file_content: bytes, filename: str, 
                          user_id: str, supabase: Client) -> Dict[str, Any]:
        """Main processing pipeline with row-by-row streaming"""
        
        # Step 1: Read the file
        await manager.send_update(job_id, {
            "step": "reading",
            "message": "📖 Reading and parsing your document...",
            "progress": 10
        })
        
        sheets = await self.read_excel_file(file_content, filename)
        
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
            'source': 'excel_upload',
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
        
        # Step 4: Process each sheet and stream individual rows
        await manager.send_update(job_id, {
            "step": "streaming",
            "message": "🔄 Processing individual rows and creating events...",
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
        
        for sheet_name, df in sheets.items():
            # Skip empty sheets
            if df.empty:
                continue
            
            # Process each row in the sheet
            for row_index, (index, row) in enumerate(df.iterrows()):
                try:
                    # Create event for this row
                    event = self.row_processor.process_row(
                        row, row_index, sheet_name, platform_info, file_context
                    )
                    
                    # Store event in raw_events table
                    event_result = supabase.table('raw_events').insert({
                        'user_id': user_id,
                        'file_id': file_id,
                        'job_id': job_id,
                        'provider': event['provider'],
                        'kind': event['kind'],
                        'source_platform': event['source_platform'],
                        'payload': event['payload'],
                        'row_index': event['row_index'],
                        'sheet_name': event['sheet_name'],
                        'source_filename': event['source_filename'],
                        'uploader': event['uploader'],
                        'ingest_ts': event['ingest_ts'],
                        'status': event['status'],
                        'confidence_score': event['confidence_score'],
                        'classification_metadata': event['classification_metadata']
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
                
                # Update progress every 10 rows
                if processed_rows % 10 == 0:
                    progress = 40 + (processed_rows / total_rows) * 40
                    await manager.send_update(job_id, {
                        "step": "streaming",
                        "message": f"🔄 Processed {processed_rows}/{total_rows} rows ({events_created} events created)...",
                        "progress": int(progress)
                    })
        
        # Step 5: Update raw_records with completion status
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
        
        # Step 6: Generate insights
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
                'file_hash': file_hash
            },
            'errors': errors
        })
        
        # Step 7: Complete
        await manager.send_update(job_id, {
            "step": "completed",
            "message": f"🎉 Analysis complete! Processed {processed_rows} rows, created {events_created} events",
            "progress": 100,
            "results": insights
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
