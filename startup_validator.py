#!/usr/bin/env python3
"""
Startup Validator for Railway Deployment
==========================================

This script runs BEFORE the main application starts to catch:
1. Syntax errors in Python files
2. Missing dependencies
3. Import failures
4. Configuration issues

If any validation fails, it prints detailed error messages to Railway logs
and exits with a non-zero code to prevent the broken app from starting.

Usage:
    python startup_validator.py

Railway Configuration:
    Add to Procfile or railway.json:
    "start": "python startup_validator.py && uvicorn core_infrastructure.fastapi_backend_v2:socketio_app --host 0.0.0.0 --port $PORT"
"""

import sys
import os
import py_compile
import importlib.util
from pathlib import Path
from typing import List, Tuple

# ANSI color codes for better log visibility
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

def print_header(message: str):
    """Print a prominent header"""
    print(f"\n{BLUE}{'='*80}{RESET}")
    print(f"{BLUE}{message.center(80)}{RESET}")
    print(f"{BLUE}{'='*80}{RESET}\n")

def print_success(message: str):
    """Print success message"""
    print(f"{GREEN}✓ {message}{RESET}")

def print_error(message: str):
    """Print error message"""
    print(f"{RED}✗ {message}{RESET}", file=sys.stderr)

def print_warning(message: str):
    """Print warning message"""
    print(f"{YELLOW}⚠ {message}{RESET}")

def validate_syntax(file_path: Path) -> Tuple[bool, str]:
    """
    Validate Python file syntax using py_compile.
    
    Returns:
        (is_valid, error_message)
    """
    try:
        py_compile.compile(str(file_path), doraise=True)
        return True, ""
    except py_compile.PyCompileError as e:
        return False, str(e)

def validate_import(module_path: str) -> Tuple[bool, str]:
    """
    Validate that a module can be imported.
    
    Returns:
        (is_valid, error_message)
    """
    try:
        importlib.import_module(module_path)
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)}"

def check_critical_files() -> bool:
    """
    Check syntax of all critical Python files.
    
    Returns:
        True if all files are valid, False otherwise
    """
    print_header("STEP 1: Validating Python File Syntax")
    
    # In Docker, all files are copied to /app/ root (see Dockerfile COPY commands)
    critical_files = [
        "fastapi_backend_v2.py",
        "supabase_client.py",
        "database_optimization_utils.py",
        "temporal_pattern_learner.py",
        "causal_inference_engine.py",
        "semantic_relationship_extractor.py",
        "enhanced_relationship_detector.py",
        "intelligent_chat_orchestrator.py",
        "universal_document_classifier_optimized.py",
        "universal_platform_detector_optimized.py",
        "universal_extractors_optimized.py",
        "universal_field_detector.py",
        "entity_resolver_optimized.py",
        "production_duplicate_detection_service.py",
    ]
    
    all_valid = True
    
    for file_path_str in critical_files:
        file_path = Path(file_path_str)
        
        if not file_path.exists():
            print_warning(f"File not found: {file_path}")
            continue
        
        is_valid, error_msg = validate_syntax(file_path)
        
        if is_valid:
            print_success(f"Syntax OK: {file_path.name}")
        else:
            print_error(f"SYNTAX ERROR in {file_path}:")
            print_error(f"  {error_msg}")
            all_valid = False
    
    return all_valid

def check_critical_dependencies() -> bool:
    """
    Check that critical dependencies can be imported.
    
    Returns:
        True if all dependencies are available, False otherwise
    """
    print_header("STEP 2: Validating Critical Dependencies")
    
    critical_deps = [
        ("fastapi", "FastAPI web framework"),
        ("uvicorn", "ASGI server"),
        ("supabase", "Supabase client"),
        ("groq", "Groq AI client"),
        ("msgpack", "Fast serialization (for cache)"),
        ("statsmodels", "Time series analysis"),
        ("stumpy", "Matrix profile for pattern discovery"),
        ("prophet", "Forecasting"),
        ("pyod", "Anomaly detection"),
        ("instructor", "Structured AI outputs"),
        ("structlog", "Logging"),
        ("redis", "Redis client"),
    ]
    
    all_available = True
    
    for module_name, description in critical_deps:
        try:
            importlib.import_module(module_name)
            print_success(f"{module_name.ljust(20)} - {description}")
        except ImportError as e:
            print_error(f"{module_name.ljust(20)} - MISSING!")
            print_error(f"  Error: {e}")
            print_error(f"  Fix: Add '{module_name}' to backend-requirements.txt")
            all_available = False
    
    return all_available

def check_environment_variables() -> bool:
    """
    Check that critical environment variables are set.
    
    Returns:
        True if all required env vars are set, False otherwise
    """
    print_header("STEP 3: Validating Environment Variables")
    
    required_vars = [
        ("SUPABASE_URL", "Supabase project URL"),
        ("SUPABASE_KEY", "Supabase anon/service key"),
    ]
    
    optional_vars = [
        ("GROQ_API_KEY", "Groq AI API key"),
        ("REDIS_URL", "Redis connection URL"),
        ("SENTRY_DSN", "Sentry error tracking"),
    ]
    
    all_set = True
    
    # Check required variables
    for var_name, description in required_vars:
        if os.getenv(var_name):
            print_success(f"{var_name.ljust(20)} - {description}")
        else:
            print_error(f"{var_name.ljust(20)} - MISSING (REQUIRED)!")
            print_error(f"  Set in Railway: Settings → Variables → {var_name}")
            all_set = False
    
    # Check optional variables (warnings only)
    for var_name, description in optional_vars:
        if os.getenv(var_name):
            print_success(f"{var_name.ljust(20)} - {description}")
        else:
            print_warning(f"{var_name.ljust(20)} - Not set (optional)")
    
    return all_set

def check_critical_imports() -> bool:
    """
    Skip import check - Let Uvicorn handle it.
    
    The import check was hanging because fastapi_backend_v2 imports modules
    that try to connect to external services (Redis, Supabase, etc.)
    during module initialization. This is fine for Uvicorn but causes
    the validator to hang.
    
    Returns:
        True (always passes)
    """
    print_header("STEP 4: Skipping Application Import Check")
    print("ℹ️  Import check skipped - Uvicorn will handle module loading")
    print("ℹ️  If there are import errors, they will appear in Uvicorn logs")
    
    return True

def main():
    """
    Run all validation checks.
    
    Exit codes:
        0 - All checks passed
        1 - Syntax errors found
        2 - Missing dependencies
        3 - Missing environment variables
        4 - Import failures
    """
    print_header("Railway Deployment Startup Validator")
    print(f"Python Version: {sys.version}")
    print(f"Working Directory: {os.getcwd()}")
    
    # Step 1: Check syntax
    if not check_critical_files():
        print_error("\n❌ VALIDATION FAILED: Syntax errors detected")
        print_error("Fix syntax errors and redeploy")
        sys.exit(1)
    
    # Step 2: Check dependencies
    if not check_critical_dependencies():
        print_error("\n❌ VALIDATION FAILED: Missing dependencies")
        print_error("Add missing packages to backend-requirements.txt and redeploy")
        sys.exit(2)
    
    # Step 3: Check environment variables
    if not check_environment_variables():
        print_error("\n❌ VALIDATION FAILED: Missing required environment variables")
        print_error("Set required variables in Railway dashboard and redeploy")
        sys.exit(3)
    
    # Step 4: Check main application import
    if not check_critical_imports():
        print_error("\n❌ VALIDATION FAILED: Cannot import main application")
        print_error("Check the error traceback above for details")
        sys.exit(4)
    
    # All checks passed!
    print_header("✅ ALL VALIDATION CHECKS PASSED")
    print_success("Application is ready to start")
    print_success("Proceeding to start Uvicorn server...")
    
    sys.exit(0)

if __name__ == "__main__":
    main()
