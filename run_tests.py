#!/usr/bin/env python3
"""
Test runner script for Finley AI BDD tests
Supports both pytest and behave commands
"""
import sys
import subprocess
import argparse
from pathlib import Path

def run_behave_tests(features=None, tags=None, verbose=False):
    """Run behave tests"""
    cmd = ["behave"]
    
    if features:
        cmd.extend(features)
    else:
        cmd.append("features/")
    
    if tags:
        cmd.extend(["--tags", tags])
    
    if verbose:
        cmd.append("--verbose")
    
    print(f"🚀 Running Behave tests: {' '.join(cmd)}")
    return subprocess.run(cmd)

def run_pytest_tests(test_path=None, markers=None, verbose=False):
    """Run pytest tests"""
    cmd = ["pytest"]
    
    if test_path:
        cmd.append(test_path)
    else:
        cmd.append("tests/")
        cmd.append("features/")
    
    if markers:
        cmd.extend(["-m", markers])
    
    if verbose:
        cmd.append("-v")
    
    cmd.extend(["--tb=short", "--strict-markers"])
    
    print(f"🚀 Running Pytest tests: {' '.join(cmd)}")
    return subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description="Finley AI Test Runner")
    parser.add_argument("--framework", choices=["behave", "pytest", "both"], 
                       default="both", help="Testing framework to use")
    parser.add_argument("--features", nargs="+", help="Specific feature files to test")
    parser.add_argument("--tags", help="Behave tags to filter tests")
    parser.add_argument("--markers", help="Pytest markers to filter tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--test-path", help="Specific test path for pytest")
    
    args = parser.parse_args()
    
    print("🧪 Finley AI Test Runner")
    print("=" * 50)
    
    exit_code = 0
    
    if args.framework in ["behave", "both"]:
        print("\n📋 Running Behave BDD Tests...")
        result = run_behave_tests(args.features, args.tags, args.verbose)
        if result.returncode != 0:
            exit_code = result.returncode
            print("❌ Behave tests failed")
        else:
            print("✅ Behave tests passed")
    
    if args.framework in ["pytest", "both"]:
        print("\n🔬 Running Pytest Tests...")
        result = run_pytest_tests(args.test_path, args.markers, args.verbose)
        if result.returncode != 0:
            exit_code = result.returncode
            print("❌ Pytest tests failed")
        else:
            print("✅ Pytest tests passed")
    
    if exit_code == 0:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Some tests failed!")
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
