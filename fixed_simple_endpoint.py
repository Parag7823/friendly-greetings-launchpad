"""
FIXED simple endpoint code - copy this into fastapi_backend_v2.py

The bug: polars.read_excel(sheet_name=None) might return a single DataFrame
         or dict depending on version. Need to handle both cases.
"""

FIXED_SIMPLE_ENDPOINT = '''
# ----- ULTRA SIMPLE ENDPOINT - NO REDIS, NO CACHE, NO AI -----
@app.post("/api/upload-simple")
async def upload_simple_endpoint(
    file: UploadFile = File(...),
    user_id: str = Form(default="test_user"),
    job_id: str = Form(default=""),
):
    """
    Ultra-simple file upload - NO dependencies.
    Just file upload + Excel parsing + return result.
    """
    import tempfile
    import os
    
    try:
        # Get filename
        filename = file.filename or "upload.xlsx"
        suffix = os.path.splitext(filename)[1] or ".xlsx"
        
        # Read file content directly (await for async)
        file_content = await file.read()
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(file_content)
            temp_path = temp.name
        
        try:
            # Calculate hash
            import xxhash
            file_hash = xxhash.xxh3_128(file_content).hexdigest()
            
            # Parse Excel with polars
            import polars as pl
            
            # Try reading with sheet_name=None (returns dict in some versions)
            try:
                result = pl.read_excel(temp_path, sheet_name=None, engine='calamine')
                
                # Handle both dict and single DataFrame cases
                if isinstance(result, dict):
                    sheets = result
                else:
                    # Single DataFrame - wrap in dict
                    sheets = {"Sheet1": result}
            except Exception as excel_error:
                # Fallback: read first sheet only
                df = pl.read_excel(temp_path, engine='calamine')
                sheets = {"Sheet1": df}
            
            sheets_info = {}
            total_rows = 0
            for sheet_name, df in sheets.items():
                sheets_info[sheet_name] = {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": list(df.columns)
                }
                total_rows += len(df)
            
            return {
                "status": "success",
                "job_id": job_id or str(uuid.uuid4()),
                "user_id": user_id,
                "filename": filename,
                "file_hash": file_hash,
                "file_size": len(file_content),
                "sheets": sheets_info,
                "total_rows": total_rows
            }
        finally:
            # Cleanup temp file
            try:
                os.unlink(temp_path)
            except:
                pass
                
    except Exception as e:
        import traceback
        logger.error(f"Simple upload error: {e}\\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc()[:500]
            }
        )
# ----- END SIMPLE ENDPOINT -----
'''

print("Copy the endpoint above and replace the existing /api/upload-simple in fastapi_backend_v2.py")
print("\nThe FIX: Handle both dict and DataFrame return types from polars.read_excel()")
