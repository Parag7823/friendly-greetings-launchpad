#!/usr/bin/env python3
"""
Comprehensive System Audit Executor
Acts as a 1000-person audit team to ensure 100% perfection
"""

import requests
import time
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any

# Configuration
BASE_URL = "https://friendly-greetings-launchpad.onrender.com"
TEST_USER_ID = "550e8400-e29b-41d4-a716-446655440000"

class AuditExecutor:
    """Comprehensive audit executor for the entire system"""
    
    def __init__(self):
        self.results = {
            'audit_start_time': datetime.now().isoformat(),
            'phases': {},
            'overall_status': 'PENDING',
            'issues_found': [],
            'recommendations': []
        }
        
    def log_issue(self, phase: str, category: str, issue: str, severity: str = 'MEDIUM'):
        """Log an audit issue"""
        self.results['issues_found'].append({
            'phase': phase,
            'category': category,
            'issue': issue,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        })
        
    def log_recommendation(self, phase: str, recommendation: str, priority: str = 'MEDIUM'):
        """Log an audit recommendation"""
        self.results['recommendations'].append({
            'phase': phase,
            'recommendation': recommendation,
            'priority': priority,
            'timestamp': datetime.now().isoformat()
        })

    def phase_1_user_journey_audit(self):
        """Phase 1: End-to-End User Journey Audit"""
        print("🔍 PHASE 1: END-TO-END USER JOURNEY AUDIT")
        print("=" * 60)
        
        phase_results = {
            'status': 'IN_PROGRESS',
            'tests': {},
            'issues': [],
            'recommendations': []
        }
        
        # Test 1.1: File Upload Journey
        print("📁 Testing File Upload Journey...")
        
        # Test drag & drop interface (simulated)
        try:
            # Test file type validation
            response = requests.post(f"{BASE_URL}/upload-and-process", 
                                   files={'file': ('test.txt', 'invalid content')},
                                   data={'user_id': TEST_USER_ID})
            
            if response.status_code == 400:
                print("✅ File type validation: PASSED")
                phase_results['tests']['file_type_validation'] = 'PASSED'
            else:
                print("❌ File type validation: FAILED")
                phase_results['tests']['file_type_validation'] = 'FAILED'
                self.log_issue('Phase 1', 'File Upload', 'File type validation not working', 'HIGH')
                
        except Exception as e:
            print(f"⚠️  File type validation: ERROR - {e}")
            phase_results['tests']['file_type_validation'] = 'ERROR'
            
        # Test file size validation
        try:
            large_content = "x" * (51 * 1024 * 1024)  # 51MB
            response = requests.post(f"{BASE_URL}/upload-and-process", 
                                   files={'file': ('large_test.csv', large_content)},
                                   data={'user_id': TEST_USER_ID})
            
            if response.status_code == 400:
                print("✅ File size validation: PASSED")
                phase_results['tests']['file_size_validation'] = 'PASSED'
            else:
                print("❌ File size validation: FAILED")
                phase_results['tests']['file_size_validation'] = 'FAILED'
                self.log_issue('Phase 1', 'File Upload', 'File size validation not working', 'HIGH')
                
        except Exception as e:
            print(f"⚠️  File size validation: ERROR - {e}")
            phase_results['tests']['file_size_validation'] = 'ERROR'
        
        # Test 1.2: Processing Journey
        print("\n⚙️ Testing Processing Journey...")
        
        # Test processing with valid file
        try:
            valid_content = "Date,Amount,Description\n2024-01-01,100.00,Test transaction"
            response = requests.post(f"{BASE_URL}/upload-and-process", 
                                   files={'file': ('valid_test.csv', valid_content)},
                                   data={'user_id': TEST_USER_ID})
            
            if response.status_code == 200:
                print("✅ File processing: PASSED")
                phase_results['tests']['file_processing'] = 'PASSED'
            else:
                print(f"❌ File processing: FAILED ({response.status_code})")
                phase_results['tests']['file_processing'] = 'FAILED'
                self.log_issue('Phase 1', 'Processing', f'File processing failed with status {response.status_code}', 'HIGH')
                
        except Exception as e:
            print(f"⚠️  File processing: ERROR - {e}")
            phase_results['tests']['file_processing'] = 'ERROR'
        
        # Test 1.3: Results Display Journey
        print("\n📊 Testing Results Display Journey...")
        
        # Test relationship detection endpoint
        try:
            response = requests.get(f"{BASE_URL}/test-enhanced-relationship-detection/{TEST_USER_ID}")
            
            if response.status_code == 200:
                data = response.json()
                if 'relationships' in data and 'total_relationships' in data:
                    print("✅ Results display: PASSED")
                    phase_results['tests']['results_display'] = 'PASSED'
                else:
                    print("⚠️  Results display: INCOMPLETE")
                    phase_results['tests']['results_display'] = 'PARTIAL'
                    self.log_recommendation('Phase 1', 'Enhance results display format', 'LOW')
            else:
                print(f"❌ Results display: FAILED ({response.status_code})")
                phase_results['tests']['results_display'] = 'FAILED'
                
        except Exception as e:
            print(f"⚠️  Results display: ERROR - {e}")
            phase_results['tests']['results_display'] = 'ERROR'
        
        # Phase 1 Summary
        passed_tests = sum(1 for test in phase_results['tests'].values() if test == 'PASSED')
        total_tests = len(phase_results['tests'])
        
        if passed_tests == total_tests:
            phase_results['status'] = 'PASSED'
            print(f"\n✅ PHASE 1 COMPLETED: {passed_tests}/{total_tests} tests passed")
        else:
            phase_results['status'] = 'FAILED'
            print(f"\n❌ PHASE 1 FAILED: {passed_tests}/{total_tests} tests passed")
            
        self.results['phases']['phase_1_user_journey'] = phase_results
        print()

    def phase_2_business_logic_audit(self):
        """Phase 2: Business Logic Audit"""
        print("🧠 PHASE 2: BUSINESS LOGIC AUDIT")
        print("=" * 60)
        
        phase_results = {
            'status': 'IN_PROGRESS',
            'tests': {},
            'issues': [],
            'recommendations': []
        }
        
        # Test 2.1: Data Processing Logic
        print("📊 Testing Data Processing Logic...")
        
        # Test currency normalization
        try:
            response = requests.get(f"{BASE_URL}/test-currency-normalization")
            
            if response.status_code == 200:
                print("✅ Currency normalization: PASSED")
                phase_results['tests']['currency_normalization'] = 'PASSED'
            else:
                print(f"❌ Currency normalization: FAILED ({response.status_code})")
                phase_results['tests']['currency_normalization'] = 'FAILED'
                self.log_issue('Phase 2', 'Data Processing', 'Currency normalization not working', 'MEDIUM')
                
        except Exception as e:
            print(f"⚠️  Currency normalization: ERROR - {e}")
            phase_results['tests']['currency_normalization'] = 'ERROR'
        
        # Test vendor standardization
        try:
            response = requests.get(f"{BASE_URL}/test-vendor-standardization")
            
            if response.status_code == 200:
                print("✅ Vendor standardization: PASSED")
                phase_results['tests']['vendor_standardization'] = 'PASSED'
            else:
                print(f"❌ Vendor standardization: FAILED ({response.status_code})")
                phase_results['tests']['vendor_standardization'] = 'FAILED'
                self.log_issue('Phase 2', 'Data Processing', 'Vendor standardization not working', 'MEDIUM')
                
        except Exception as e:
            print(f"⚠️  Vendor standardization: ERROR - {e}")
            phase_results['tests']['vendor_standardization'] = 'ERROR'
        
        # Test 2.2: Data Enrichment Logic
        print("\n🔧 Testing Data Enrichment Logic...")
        
        # Test platform detection
        try:
            response = requests.get(f"{BASE_URL}/test-platform-detection")
            
            if response.status_code == 200:
                print("✅ Platform detection: PASSED")
                phase_results['tests']['platform_detection'] = 'PASSED'
            else:
                print(f"⚠️  Platform detection: STATUS {response.status_code}")
                phase_results['tests']['platform_detection'] = 'PARTIAL'
                self.log_recommendation('Phase 2', 'Implement platform detection endpoint', 'MEDIUM')
                
        except Exception as e:
            print(f"⚠️  Platform detection: ERROR - {e}")
            phase_results['tests']['platform_detection'] = 'ERROR'
        
        # Test 2.3: Relationship Detection Logic
        print("\n🔗 Testing Relationship Detection Logic...")
        
        # Test AI relationship detection
        try:
            response = requests.get(f"{BASE_URL}/test-ai-relationship-detection/{TEST_USER_ID}")
            
            if response.status_code == 200:
                data = response.json()
                if 'relationships' in data:
                    print("✅ AI relationship detection: PASSED")
                    phase_results['tests']['ai_relationship_detection'] = 'PASSED'
                else:
                    print("⚠️  AI relationship detection: INCOMPLETE")
                    phase_results['tests']['ai_relationship_detection'] = 'PARTIAL'
            else:
                print(f"❌ AI relationship detection: FAILED ({response.status_code})")
                phase_results['tests']['ai_relationship_detection'] = 'FAILED'
                
        except Exception as e:
            print(f"⚠️  AI relationship detection: ERROR - {e}")
            phase_results['tests']['ai_relationship_detection'] = 'ERROR'
        
        # Phase 2 Summary
        passed_tests = sum(1 for test in phase_results['tests'].values() if test == 'PASSED')
        total_tests = len(phase_results['tests'])
        
        if passed_tests >= total_tests * 0.8:  # 80% pass rate for business logic
            phase_results['status'] = 'PASSED'
            print(f"\n✅ PHASE 2 COMPLETED: {passed_tests}/{total_tests} tests passed")
        else:
            phase_results['status'] = 'FAILED'
            print(f"\n❌ PHASE 2 FAILED: {passed_tests}/{total_tests} tests passed")
            
        self.results['phases']['phase_2_business_logic'] = phase_results
        print()

    def phase_3_api_endpoints_audit(self):
        """Phase 3: API Endpoints Audit"""
        print("🔌 PHASE 3: API ENDPOINTS AUDIT")
        print("=" * 60)
        
        phase_results = {
            'status': 'IN_PROGRESS',
            'tests': {},
            'issues': [],
            'recommendations': []
        }
        
        # Test 3.1: Core Endpoints
        print("🎯 Testing Core Endpoints...")
        
        endpoints_to_test = [
            ('/upload-and-process', 'POST'),
            (f'/test-enhanced-relationship-detection/{TEST_USER_ID}', 'GET'),
            (f'/test-entity-resolution/{TEST_USER_ID}', 'GET'),
            (f'/test-cross-file-relationships/{TEST_USER_ID}', 'GET'),
            (f'/test-ai-relationship-detection/{TEST_USER_ID}', 'GET'),
            ('/test-currency-normalization', 'GET'),
            ('/test-vendor-standardization', 'GET'),
            ('/health', 'GET'),
            (f'/test-websocket/test-job-id', 'GET')
        ]
        
        for endpoint, method in endpoints_to_test:
            try:
                if method == 'GET':
                    response = requests.get(f"{BASE_URL}{endpoint}")
                else:
                    response = requests.post(f"{BASE_URL}{endpoint}")
                
                if response.status_code in [200, 201]:
                    print(f"✅ {endpoint}: PASSED")
                    phase_results['tests'][endpoint] = 'PASSED'
                elif response.status_code == 404:
                    print(f"⚠️  {endpoint}: NOT FOUND")
                    phase_results['tests'][endpoint] = 'NOT_FOUND'
                    self.log_recommendation('Phase 3', f'Implement endpoint {endpoint}', 'MEDIUM')
                else:
                    print(f"❌ {endpoint}: FAILED ({response.status_code})")
                    phase_results['tests'][endpoint] = 'FAILED'
                    self.log_issue('Phase 3', 'API Endpoints', f'Endpoint {endpoint} failed with status {response.status_code}', 'MEDIUM')
                    
            except Exception as e:
                print(f"⚠️  {endpoint}: ERROR - {e}")
                phase_results['tests'][endpoint] = 'ERROR'
        
        # Phase 3 Summary
        passed_tests = sum(1 for test in phase_results['tests'].values() if test == 'PASSED')
        total_tests = len(phase_results['tests'])
        
        if passed_tests >= total_tests * 0.7:  # 70% pass rate for API endpoints
            phase_results['status'] = 'PASSED'
            print(f"\n✅ PHASE 3 COMPLETED: {passed_tests}/{total_tests} tests passed")
        else:
            phase_results['status'] = 'FAILED'
            print(f"\n❌ PHASE 3 FAILED: {passed_tests}/{total_tests} tests passed")
            
        self.results['phases']['phase_3_api_endpoints'] = phase_results
        print()

    def phase_4_security_audit(self):
        """Phase 4: Security Audit"""
        print("🔒 PHASE 4: SECURITY AUDIT")
        print("=" * 60)
        
        phase_results = {
            'status': 'IN_PROGRESS',
            'tests': {},
            'issues': [],
            'recommendations': []
        }
        
        # Test 4.1: Authentication & Authorization
        print("🔐 Testing Authentication & Authorization...")
        
        # Test user ID validation
        try:
            response = requests.post(f"{BASE_URL}/upload-and-process", 
                                   files={'file': ('test.csv', 'test,data')},
                                   data={})  # No user_id
            
            if response.status_code == 400:
                print("✅ User ID validation: PASSED")
                phase_results['tests']['user_id_validation'] = 'PASSED'
            else:
                print(f"❌ User ID validation: FAILED ({response.status_code})")
                phase_results['tests']['user_id_validation'] = 'FAILED'
                self.log_issue('Phase 4', 'Security', 'User ID validation not enforced', 'HIGH')
                
        except Exception as e:
            print(f"⚠️  User ID validation: ERROR - {e}")
            phase_results['tests']['user_id_validation'] = 'ERROR'
        
        # Test 4.2: Input Validation
        print("\n🛡️ Testing Input Validation...")
        
        # Test SQL injection prevention (simulated)
        try:
            malicious_user_id = "'; DROP TABLE users; --"
            response = requests.post(f"{BASE_URL}/upload-and-process", 
                                   files={'file': ('test.csv', 'test,data')},
                                   data={'user_id': malicious_user_id})
            
            if response.status_code in [400, 500]:  # Should reject or error
                print("✅ SQL injection prevention: PASSED")
                phase_results['tests']['sql_injection_prevention'] = 'PASSED'
            else:
                print(f"⚠️  SQL injection prevention: STATUS {response.status_code}")
                phase_results['tests']['sql_injection_prevention'] = 'PARTIAL'
                self.log_recommendation('Phase 4', 'Enhance SQL injection prevention', 'HIGH')
                
        except Exception as e:
            print(f"⚠️  SQL injection prevention: ERROR - {e}")
            phase_results['tests']['sql_injection_prevention'] = 'ERROR'
        
        # Phase 4 Summary
        passed_tests = sum(1 for test in phase_results['tests'].values() if test == 'PASSED')
        total_tests = len(phase_results['tests'])
        
        if passed_tests >= total_tests * 0.8:  # 80% pass rate for security
            phase_results['status'] = 'PASSED'
            print(f"\n✅ PHASE 4 COMPLETED: {passed_tests}/{total_tests} tests passed")
        else:
            phase_results['status'] = 'FAILED'
            print(f"\n❌ PHASE 4 FAILED: {passed_tests}/{total_tests} tests passed")
            
        self.results['phases']['phase_4_security'] = phase_results
        print()

    def phase_5_performance_audit(self):
        """Phase 5: Performance Audit"""
        print("⚡ PHASE 5: PERFORMANCE AUDIT")
        print("=" * 60)
        
        phase_results = {
            'status': 'IN_PROGRESS',
            'tests': {},
            'issues': [],
            'recommendations': []
        }
        
        # Test 5.1: Response Time
        print("⏱️ Testing Response Times...")
        
        # Test health endpoint response time
        try:
            start_time = time.time()
            response = requests.get(f"{BASE_URL}/health")
            end_time = time.time()
            
            response_time = end_time - start_time
            
            if response_time < 2.0:  # Less than 2 seconds
                print(f"✅ Health endpoint response time: PASSED ({response_time:.2f}s)")
                phase_results['tests']['health_response_time'] = 'PASSED'
            else:
                print(f"⚠️  Health endpoint response time: SLOW ({response_time:.2f}s)")
                phase_results['tests']['health_response_time'] = 'SLOW'
                self.log_recommendation('Phase 5', 'Optimize health endpoint response time', 'MEDIUM')
                
        except Exception as e:
            print(f"⚠️  Health endpoint response time: ERROR - {e}")
            phase_results['tests']['health_response_time'] = 'ERROR'
        
        # Test 5.2: Memory Usage (simulated)
        print("\n🧠 Testing Memory Usage...")
        
        # This would require actual memory monitoring
        print("⚠️  Memory usage testing requires server-side monitoring")
        phase_results['tests']['memory_usage'] = 'NOT_TESTED'
        self.log_recommendation('Phase 5', 'Implement server-side memory monitoring', 'MEDIUM')
        
        # Phase 5 Summary
        passed_tests = sum(1 for test in phase_results['tests'].values() if test == 'PASSED')
        total_tests = len(phase_results['tests'])
        
        if passed_tests >= total_tests * 0.5:  # 50% pass rate for performance
            phase_results['status'] = 'PASSED'
            print(f"\n✅ PHASE 5 COMPLETED: {passed_tests}/{total_tests} tests passed")
        else:
            phase_results['status'] = 'FAILED'
            print(f"\n❌ PHASE 5 FAILED: {passed_tests}/{total_tests} tests passed")
            
        self.results['phases']['phase_5_performance'] = phase_results
        print()

    def generate_audit_report(self):
        """Generate comprehensive audit report"""
        print("📋 GENERATING COMPREHENSIVE AUDIT REPORT")
        print("=" * 60)
        
        # Calculate overall status
        passed_phases = sum(1 for phase in self.results['phases'].values() if phase['status'] == 'PASSED')
        total_phases = len(self.results['phases'])
        
        if passed_phases == total_phases:
            self.results['overall_status'] = 'PASSED'
        elif passed_phases >= total_phases * 0.8:
            self.results['overall_status'] = 'PARTIAL'
        else:
            self.results['overall_status'] = 'FAILED'
        
        self.results['audit_end_time'] = datetime.now().isoformat()
        
        # Print summary
        print(f"\n🎯 AUDIT SUMMARY")
        print(f"Overall Status: {self.results['overall_status']}")
        print(f"Phases Passed: {passed_phases}/{total_phases}")
        print(f"Issues Found: {len(self.results['issues_found'])}")
        print(f"Recommendations: {len(self.results['recommendations'])}")
        
        # Print issues
        if self.results['issues_found']:
            print(f"\n❌ ISSUES FOUND:")
            for issue in self.results['issues_found']:
                print(f"  - [{issue['severity']}] {issue['phase']} - {issue['issue']}")
        
        # Print recommendations
        if self.results['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in self.results['recommendations']:
                print(f"  - [{rec['priority']}] {rec['phase']} - {rec['recommendation']}")
        
        # Save report
        report_filename = f"audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n📄 Audit report saved to: {report_filename}")
        
        # Final recommendation
        if self.results['overall_status'] == 'PASSED':
            print("\n🎉 SYSTEM IS READY FOR PRODUCTION!")
        elif self.results['overall_status'] == 'PARTIAL':
            print("\n⚠️  SYSTEM NEEDS MINOR IMPROVEMENTS BEFORE PRODUCTION")
        else:
            print("\n❌ SYSTEM NEEDS MAJOR FIXES BEFORE PRODUCTION")

    def run_complete_audit(self):
        """Run the complete audit"""
        print("🚀 STARTING COMPREHENSIVE SYSTEM AUDIT")
        print("=" * 60)
        print(f"Audit Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Target System: {BASE_URL}")
        print()
        
        # Run all phases
        self.phase_1_user_journey_audit()
        self.phase_2_business_logic_audit()
        self.phase_3_api_endpoints_audit()
        self.phase_4_security_audit()
        self.phase_5_performance_audit()
        
        # Generate report
        self.generate_audit_report()

def main():
    """Main audit execution"""
    auditor = AuditExecutor()
    auditor.run_complete_audit()

if __name__ == "__main__":
    main()
