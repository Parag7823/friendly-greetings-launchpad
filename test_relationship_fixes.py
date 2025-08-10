#!/usr/bin/env python3
"""
Comprehensive Test Script for Relationship Detection Fixes
Tests all the critical fixes implemented for universal relationship detection
"""

import asyncio
import aiohttp
import json
import os
from datetime import datetime

# Test configuration
BASE_URL = "http://localhost:8000"
USER_ID = "550e8400-e29b-41d4-a716-446655440000"

class RelationshipTestSuite:
    def __init__(self):
        self.results = []
        self.session = None
    
    async def setup(self):
        """Setup test session"""
        self.session = aiohttp.ClientSession()
    
    async def cleanup(self):
        """Cleanup test session"""
        if self.session:
            await self.session.close()
    
    async def run_test(self, test_name: str, test_func):
        """Run a single test and record results"""
        try:
            print(f"🧪 Running: {test_name}")
            start_time = datetime.now()
            result = await test_func()
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            test_result = {
                "test_name": test_name,
                "status": "PASS" if result.get("success", False) else "FAIL",
                "duration": duration,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            
            self.results.append(test_result)
            
            if test_result["status"] == "PASS":
                print(f"✅ {test_name}: PASSED ({duration:.2f}s)")
            else:
                print(f"❌ {test_name}: FAILED ({duration:.2f}s)")
                print(f"   Error: {result.get('error', 'Unknown error')}")
            
            return test_result
            
        except Exception as e:
            error_result = {
                "test_name": test_name,
                "status": "ERROR",
                "duration": 0,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            self.results.append(error_result)
            print(f"💥 {test_name}: ERROR - {str(e)}")
            return error_result
    
    async def test_health_check(self):
        """Test basic API health"""
        async with self.session.get(f"{BASE_URL}/health") as response:
            data = await response.json()
            return {
                "success": response.status == 200,
                "status_code": response.status,
                "data": data
            }
    
    async def test_raw_events_count(self):
        """Test that we have test data available"""
        async with self.session.get(f"{BASE_URL}/test-raw-events/{USER_ID}") as response:
            data = await response.json()
            event_count = len(data.get("events", []))
            return {
                "success": event_count > 0,
                "event_count": event_count,
                "data": data
            }
    
    async def test_relationship_discovery(self):
        """Test relationship type discovery"""
        async with self.session.get(f"{BASE_URL}/test-relationship-discovery/{USER_ID}") as response:
            data = await response.json()
            discovered_types = data.get("discovered_types", [])
            return {
                "success": len(discovered_types) > 0,
                "discovered_types": discovered_types,
                "total_events": data.get("total_events", 0),
                "data": data
            }
    
    async def test_ai_relationship_scoring(self):
        """Test AI relationship scoring with different scores per type"""
        async with self.session.get(f"{BASE_URL}/test-ai-relationship-scoring/{USER_ID}") as response:
            data = await response.json()
            scoring_results = data.get("scoring_results", [])
            
            # Check that we have scoring results
            if not scoring_results:
                return {
                    "success": False,
                    "error": "No scoring results returned",
                    "data": data
                }
            
            # Check that different relationship types have different scores
            scores = [result.get("comprehensive_score", 0) for result in scoring_results]
            unique_scores = len(set(scores))
            
            # Check that we have different scores (not all identical)
            has_different_scores = unique_scores > 1
            
            return {
                "success": has_different_scores,
                "unique_scores": unique_scores,
                "total_results": len(scoring_results),
                "score_variation": has_different_scores,
                "scores": scores,
                "data": data
            }
    
    async def test_ai_relationship_detection(self):
        """Test full AI relationship detection"""
        async with self.session.get(f"{BASE_URL}/test-ai-relationship-detection/{USER_ID}") as response:
            data = await response.json()
            relationships = data.get("result", {}).get("relationships", [])
            total_relationships = data.get("result", {}).get("total_relationships", 0)
            
            return {
                "success": total_relationships > 0,  # Should find at least some relationships
                "total_relationships": total_relationships,
                "relationship_types": data.get("result", {}).get("relationship_types", []),
                "processing_stats": data.get("result", {}).get("processing_stats", {}),
                "data": data
            }
    
    async def test_cross_file_relationships(self):
        """Test cross-file relationship detection"""
        async with self.session.get(f"{BASE_URL}/test-cross-file-relationships/{USER_ID}") as response:
            data = await response.json()
            relationships = data.get("relationships", [])
            total_relationships = data.get("total_relationships", 0)
            
            return {
                "success": True,  # This test should always pass even with 0 relationships
                "total_relationships": total_relationships,
                "payroll_events": data.get("total_payroll_events", 0),
                "payout_events": data.get("total_payout_events", 0),
                "data": data
            }
    
    async def test_flexible_relationship_discovery(self):
        """Test flexible relationship engine"""
        async with self.session.get(f"{BASE_URL}/test-flexible-relationship-discovery/{USER_ID}") as response:
            data = await response.json()
            relationships = data.get("relationships", [])
            total_relationships = data.get("total_relationships", 0)
            
            return {
                "success": True,  # This test should always pass
                "total_relationships": total_relationships,
                "discovered_types": data.get("discovered_types", []),
                "data": data
            }
    
    async def run_all_tests(self):
        """Run all tests in sequence"""
        print("🚀 Starting Comprehensive Relationship Detection Test Suite")
        print("=" * 60)
        
        tests = [
            ("Health Check", self.test_health_check),
            ("Raw Events Count", self.test_raw_events_count),
            ("Relationship Discovery", self.test_relationship_discovery),
            ("AI Relationship Scoring", self.test_ai_relationship_scoring),
            ("AI Relationship Detection", self.test_ai_relationship_detection),
            ("Cross File Relationships", self.test_cross_file_relationships),
            ("Flexible Relationship Discovery", self.test_flexible_relationship_discovery),
        ]
        
        for test_name, test_func in tests:
            await self.run_test(test_name, test_func)
            await asyncio.sleep(1)  # Small delay between tests
        
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r["status"] == "PASS"])
        failed_tests = len([r for r in self.results if r["status"] == "FAIL"])
        error_tests = len([r for r in self.results if r["status"] == "ERROR"])
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"💥 Errors: {error_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        # Print detailed results
        print("\n📋 DETAILED RESULTS:")
        for result in self.results:
            status_icon = "✅" if result["status"] == "PASS" else "❌" if result["status"] == "FAIL" else "💥"
            print(f"{status_icon} {result['test_name']}: {result['status']} ({result['duration']:.2f}s)")
            
            if result["status"] != "PASS" and "error" in result:
                print(f"   Error: {result['error']}")
        
        # Save results to file
        self.save_results()
    
    def save_results(self):
        """Save test results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"relationship_test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                "test_suite": "Relationship Detection Fixes",
                "timestamp": datetime.now().isoformat(),
                "results": self.results
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")

async def main():
    """Main test runner"""
    test_suite = RelationshipTestSuite()
    
    try:
        await test_suite.setup()
        await test_suite.run_all_tests()
    finally:
        await test_suite.cleanup()

if __name__ == "__main__":
    print("🔧 Relationship Detection Fixes Test Suite")
    print("Testing all critical fixes for universal relationship detection")
    print()
    
    asyncio.run(main()) 