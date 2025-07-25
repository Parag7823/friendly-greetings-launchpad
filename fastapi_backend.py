from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import openpyxl
import xlrd
import pyxlsb
import magic
import filetype
import asyncio
import json
import os
import requests
import io
from typing import Dict, Any, Optional, List
from datetime import datetime
import openai
from supabase import create_client, Client
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Finley AI Backend", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI Configuration
openai.api_key = os.getenv("OPENAI_API_KEY")

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, job_id: str):
        await websocket.accept()
        self.active_connections[job_id] = websocket
        logger.info(f"WebSocket connected for job: {job_id}")

    def disconnect(self, job_id: str):
        if job_id in self.active_connections:
            del self.active_connections[job_id]
            logger.info(f"WebSocket disconnected for job: {job_id}")

    async def send_update(self, job_id: str, message: dict):
        if job_id in self.active_connections:
            try:
                await self.active_connections[job_id].send_text(json.dumps(message))
                logger.info(f"Sent update to job {job_id}: {message.get('message', '')}")
            except Exception as e:
                logger.error(f"Error sending message to {job_id}: {e}")
                self.disconnect(job_id)

manager = ConnectionManager()

# Pydantic models
class ProcessRequest(BaseModel):
    job_id: str
    storage_path: str
    file_name: str
    supabase_url: str
    supabase_key: str

class DocumentAnalyzer:
    def __init__(self, openai_client):
        self.openai_client = openai_client
    
    async def detect_document_type(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """Detect the type of financial document using AI"""
        
        # Get sample data for analysis
        sample_data = {
            "filename": filename,
            "columns": df.columns.tolist()[:10],  # First 10 columns
            "first_few_rows": df.head(3).to_dict('records'),
            "shape": df.shape
        }
        
        prompt = f"""
        Analyze this financial document and determine its type and structure:
        
        Filename: {sample_data['filename']}
        Columns: {sample_data['columns']}
        Sample Data: {sample_data['first_few_rows']}
        Shape: {sample_data['shape']} rows x columns
        
        Please identify:
        1. Document type (P&L, Balance Sheet, Cash Flow, Trial Balance, General Ledger, etc.)
        2. Key financial metrics present
        3. Time period (monthly, quarterly, annual)
        4. Currency if identifiable
        5. Any data quality issues
        
        Respond in JSON format with your analysis.
        """
        
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are Finley AI, an expert financial analyst. Analyze financial documents and provide structured insights."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            analysis = json.loads(response.choices[0].message.content)
            return analysis
        except Exception as e:
            logger.error(f"Error in document analysis: {e}")
            return {
                "document_type": "Unknown",
                "analysis_error": str(e),
                "basic_info": sample_data
            }

    async def generate_insights(self, df: pd.DataFrame, doc_analysis: Dict) -> Dict[str, Any]:
        """Generate intelligent insights about the financial data"""
        
        # Calculate basic statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        stats = {}
        
        if len(numeric_cols) > 0:
            stats = {
                "total_numeric_columns": len(numeric_cols),
                "total_rows": len(df),
                "summary_stats": df[numeric_cols].describe().to_dict()
            }
        
        insights_prompt = f"""
        Based on this financial document analysis:
        Document Type: {doc_analysis.get('document_type', 'Unknown')}
        Statistics: {stats}
        
        Provide 3-5 key insights that a financial analyst would find valuable.
        Focus on trends, anomalies, and actionable intelligence.
        
        Format as JSON with insights array.
        """
        
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are Finley AI, providing actionable financial insights."},
                    {"role": "user", "content": insights_prompt}
                ],
                temperature=0.3
            )
            
            insights = json.loads(response.choices[0].message.content)
            return {
                "ai_insights": insights,
                "statistics": stats,
                "document_analysis": doc_analysis
            }
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return {
                "statistics": stats,
                "document_analysis": doc_analysis,
                "insights_error": str(e)
            }

class ExcelProcessor:
    def __init__(self):
        self.analyzer = DocumentAnalyzer(openai)
    
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
    
    async def read_excel_file(self, file_content: bytes, filename: str) -> pd.DataFrame:
        """Read Excel file with multiple fallback methods"""
        file_type = await self.detect_file_type(file_content, filename)
        
        try:
            if 'spreadsheetml' in file_type or filename.endswith('.xlsx'):
                # Try openpyxl for .xlsx
                return pd.read_excel(io.BytesIO(file_content), engine='openpyxl')
            elif 'ms-excel' in file_type or filename.endswith('.xls'):
                # Try xlrd for .xls
                return pd.read_excel(io.BytesIO(file_content), engine='xlrd')
            elif filename.endswith('.xlsb'):
                # Try pyxlsb for .xlsb
                return pd.read_excel(io.BytesIO(file_content), engine='pyxlsb')
            elif 'csv' in file_type or filename.endswith('.csv'):
                # CSV files
                return pd.read_csv(io.BytesIO(file_content))
            else:
                # Try pandas default
                return pd.read_excel(io.BytesIO(file_content))
        except Exception as e:
            logger.error(f"Error reading file {filename}: {e}")
            # --- THIS IS THE CORRECTED LINE ---
            raise HTTPException(status_code=400, detail=f"Could not read file: {str(e)}")
    
    async def process_file(self, job_id: str, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Main processing pipeline"""
        
        # Step 1: Read the file
        await manager.send_update(job_id, {
            "step": "reading",
            "message": "📖 Reading and parsing your document...",
            "progress": 20
        })
        
        df = await self.read_excel_file(file_content, filename)
        
        # Step 2: Analyze document type
        await manager.send_update(job_id, {
            "step": "analyzing",
            "message": "🧠 Analyzing document structure with AI...",
            "progress": 40
        })
        
        doc_analysis = await self.analyzer.detect_document_type(df, filename)
        
        # Step 3: Generate insights
        await manager.send_update(job_id, {
            "step": "insights",
            "message": "💡 Generating intelligent financial insights...",
            "progress": 70
        })
        
        insights = await self.analyzer.generate_insights(df, doc_analysis)
        
        # Step 4: Complete
        await manager.send_update(job_id, {
            "step": "completed",
            "message": f"🎉 Analysis complete! Discovered {doc_analysis.get('document_type', 'financial document')}",
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
    """Process uploaded Excel file"""
    
    try:
        # Initialize Supabase client
        supabase: Client = create_client(request.supabase_url, request.supabase_key)
        
        # Send initial update
        await manager.send_update(request.job_id, {
            "step": "starting",
            "message": "🚀 Starting intelligent analysis...",
            "progress": 10
        })
        
        # Download file from Supabase storage
        try:
            response = supabase.storage.from_('finley-uploads').download(request.storage_path)
            file_content = response
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            await manager.send_update(request.job_id, {
                "step": "error",
                "message": f"Failed to download file: {str(e)}",
                "progress": 0
            })
            raise HTTPException(status_code=400, detail=f"File download failed: {str(e)}")
        
        # Update job status to processing
        supabase.table('ingestion_jobs').update({
            'status': 'processing',
            'started_at': datetime.utcnow().isoformat(),
            'progress': 10
        }).eq('id', request.job_id).execute()
        
        # Process the file
        results = await processor.process_file(request.job_id, file_content, request.file_name)
        
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

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Finley AI Backend"}

@app.get("/")
async def root():
    return {"message": "Finley AI Backend - Intelligent Financial Analysis"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
