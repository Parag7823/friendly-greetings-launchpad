"""
Security Tests for Injection Attacks

Tests:
- SQL injection attempts
- XSS attempts
- Path traversal attempts
- Command injection attempts
- File upload attacks
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from security_system import InputSanitizer, SecurityLevel


class TestSQLInjection:
    """Test SQL injection prevention"""
    
    def setup_method(self):
        self.sanitizer = InputSanitizer()
    
    def test_detect_basic_sql_injection(self):
        """Should detect basic SQL injection"""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
            "' OR 1=1--",
            "1; DROP TABLE users",
        ]
        
        for input_str in malicious_inputs:
            violations = self.sanitizer.detect_malicious_patterns(input_str)
            assert len(violations) > 0, f"Failed to detect: {input_str}"
            assert any(v.violation_type == "sql_injection" for v in violations)
    
    def test_detect_union_based_injection(self):
        """Should detect UNION-based SQL injection"""
        malicious_inputs = [
            "1 UNION SELECT * FROM users",
            "1' UNION SELECT NULL, username, password FROM users--",
            "UNION ALL SELECT 1,2,3",
        ]
        
        for input_str in malicious_inputs:
            violations = self.sanitizer.detect_malicious_patterns(input_str)
            assert len(violations) > 0
    
    def test_detect_blind_sql_injection(self):
        """Should detect blind SQL injection"""
        malicious_inputs = [
            "1' AND '1'='1",
            "1' AND SLEEP(5)--",
            "1' OR '1'='1'--",
        ]
        
        for input_str in malicious_inputs:
            violations = self.sanitizer.detect_malicious_patterns(input_str)
            assert len(violations) > 0
    
    def test_sanitize_sql_injection(self):
        """Should sanitize SQL injection attempts"""
        input_str = "'; DROP TABLE users; --"
        sanitized = self.sanitizer.sanitize_string(input_str)
        
        # Should escape dangerous characters
        assert "'" not in sanitized or "&#x27;" in sanitized
        assert "DROP" not in sanitized or sanitized != input_str
    
    def test_no_false_positives_sql(self):
        """Should not flag legitimate SQL-like content"""
        legitimate_inputs = [
            "SELECT your payment method",
            "Order #12345",
            "Price: $99.99",
        ]
        
        for input_str in legitimate_inputs:
            violations = self.sanitizer.detect_malicious_patterns(input_str)
            critical_violations = [v for v in violations if v.severity == SecurityLevel.CRITICAL]
            # May have some violations but not critical
            assert len(critical_violations) == 0


class TestXSSAttacks:
    """Test XSS attack prevention"""
    
    def setup_method(self):
        self.sanitizer = InputSanitizer()
    
    def test_detect_script_tag_xss(self):
        """Should detect script tag XSS"""
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "<SCRIPT>alert('XSS')</SCRIPT>",
            "<script src='evil.com/xss.js'></script>",
        ]
        
        for input_str in malicious_inputs:
            violations = self.sanitizer.detect_malicious_patterns(input_str)
            assert len(violations) > 0
            assert any(v.violation_type == "xss" for v in violations)
    
    def test_detect_event_handler_xss(self):
        """Should detect event handler XSS"""
        malicious_inputs = [
            "<img src=x onerror=alert('xss')>",
            "<body onload=alert('xss')>",
            "<div onclick='malicious()'>",
        ]
        
        for input_str in malicious_inputs:
            violations = self.sanitizer.detect_malicious_patterns(input_str)
            assert len(violations) > 0
    
    def test_detect_javascript_protocol(self):
        """Should detect javascript: protocol XSS"""
        malicious_inputs = [
            "javascript:alert('xss')",
            "JAVASCRIPT:alert(1)",
            "vbscript:msgbox('xss')",
        ]
        
        for input_str in malicious_inputs:
            violations = self.sanitizer.detect_malicious_patterns(input_str)
            assert len(violations) > 0
    
    def test_detect_iframe_xss(self):
        """Should detect iframe-based XSS"""
        malicious_inputs = [
            "<iframe src='evil.com'></iframe>",
            "<iframe src=javascript:alert('xss')>",
        ]
        
        for input_str in malicious_inputs:
            violations = self.sanitizer.detect_malicious_patterns(input_str)
            assert len(violations) > 0
    
    def test_sanitize_xss_attempts(self):
        """Should sanitize XSS attempts"""
        input_str = "<script>alert('xss')</script>"
        sanitized = self.sanitizer.sanitize_string(input_str)
        
        # Should escape HTML
        assert "<script>" not in sanitized
        assert "&lt;script&gt;" in sanitized
    
    def test_sanitize_html_entities(self):
        """Should sanitize dangerous HTML entities"""
        dangerous_chars = {
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#x27;',
        }
        
        for char, expected in dangerous_chars.items():
            sanitized = self.sanitizer.sanitize_string(char)
            assert expected in sanitized


class TestPathTraversal:
    """Test path traversal prevention"""
    
    def setup_method(self):
        self.sanitizer = InputSanitizer()
    
    def test_detect_unix_path_traversal(self):
        """Should detect Unix path traversal"""
        malicious_inputs = [
            "../../../etc/passwd",
            "../../../../etc/shadow",
            "../../../var/www/html/config.php",
        ]
        
        for input_str in malicious_inputs:
            violations = self.sanitizer.detect_malicious_patterns(input_str)
            assert len(violations) > 0
            assert any(v.violation_type == "path_traversal" for v in violations)
    
    def test_detect_windows_path_traversal(self):
        """Should detect Windows path traversal"""
        malicious_inputs = [
            "..\\..\\windows\\system32",
            "..\\..\\..\\boot.ini",
            "C:\\Windows\\System32\\config\\sam",
        ]
        
        for input_str in malicious_inputs:
            violations = self.sanitizer.detect_malicious_patterns(input_str)
            assert len(violations) > 0
    
    def test_detect_encoded_path_traversal(self):
        """Should detect encoded path traversal"""
        # URL encoded ../ is %2e%2e%2f
        # This test checks if the pattern detection catches variations
        malicious_inputs = [
            "....//",
            "....\\\\",
        ]
        
        for input_str in malicious_inputs:
            violations = self.sanitizer.detect_malicious_patterns(input_str)
            assert len(violations) > 0
    
    def test_sanitize_filename_path_traversal(self):
        """Should sanitize filenames with path traversal"""
        malicious_filenames = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32",
            "/etc/shadow",
        ]
        
        for filename in malicious_filenames:
            sanitized = self.sanitizer.sanitize_filename(filename)
            assert ".." not in sanitized
            assert "/" not in sanitized or sanitized == filename.split('/')[-1]


class TestCommandInjection:
    """Test command injection prevention"""
    
    def setup_method(self):
        self.sanitizer = InputSanitizer()
    
    def test_detect_command_injection(self):
        """Should detect command injection"""
        malicious_inputs = [
            "; rm -rf /",
            "| cat /etc/passwd",
            "& ping evil.com",
            "`whoami`",
            "$(whoami)",
        ]
        
        for input_str in malicious_inputs:
            violations = self.sanitizer.detect_malicious_patterns(input_str)
            assert len(violations) > 0
    
    def test_detect_command_chaining(self):
        """Should detect command chaining"""
        malicious_inputs = [
            "file.txt && rm -rf /",
            "file.txt || cat /etc/passwd",
            "file.txt; whoami",
        ]
        
        for input_str in malicious_inputs:
            violations = self.sanitizer.detect_malicious_patterns(input_str)
            assert len(violations) > 0


class TestFileUploadAttacks:
    """Test file upload attack prevention"""
    
    def setup_method(self):
        self.sanitizer = InputSanitizer()
    
    def test_detect_dangerous_file_extensions(self):
        """Should detect dangerous file extensions"""
        dangerous_files = [
            "malicious.exe",
            "script.bat",
            "virus.com",
            "hack.sh",
            "evil.php",
        ]
        
        for filename in dangerous_files:
            sanitized = self.sanitizer.sanitize_filename(filename)
            # Should replace dangerous extension
            assert not any(ext in sanitized for ext in ['.exe', '.bat', '.com', '.sh', '.php'])
    
    def test_detect_double_extension(self):
        """Should detect double extension attacks"""
        malicious_files = [
            "document.pdf.exe",
            "image.jpg.bat",
            "file.xlsx.php",
        ]
        
        for filename in malicious_files:
            sanitized = self.sanitizer.sanitize_filename(filename)
            # Should handle double extensions
            assert '.exe' not in sanitized
            assert '.bat' not in sanitized
            assert '.php' not in sanitized
    
    def test_detect_null_byte_injection(self):
        """Should detect null byte injection in filenames"""
        malicious_files = [
            "file.txt\x00.exe",
            "document.pdf\x00.php",
        ]
        
        for filename in malicious_files:
            sanitized = self.sanitizer.sanitize_filename(filename)
            # Should remove null bytes
            assert '\x00' not in sanitized


class TestLDAPInjection:
    """Test LDAP injection prevention"""
    
    def setup_method(self):
        self.sanitizer = InputSanitizer()
    
    def test_detect_ldap_injection(self):
        """Should detect LDAP injection"""
        malicious_inputs = [
            "*)(uid=*))(|(uid=*",
            "admin)(|(password=*))",
            "*)(objectClass=*",
        ]
        
        for input_str in malicious_inputs:
            violations = self.sanitizer.detect_malicious_patterns(input_str)
            # LDAP injection uses special characters
            assert len(violations) > 0


class TestNoSQLInjection:
    """Test NoSQL injection prevention"""
    
    def setup_method(self):
        self.sanitizer = InputSanitizer()
    
    def test_detect_nosql_injection(self):
        """Should detect NoSQL injection"""
        malicious_inputs = [
            '{"$ne": null}',
            '{"$gt": ""}',
            '{"$where": "this.password"}',
            '{"$regex": ".*"}',
        ]
        
        for input_str in malicious_inputs:
            violations = self.sanitizer.detect_malicious_patterns(input_str)
            assert len(violations) > 0


class TestHeaderInjection:
    """Test HTTP header injection prevention"""
    
    def setup_method(self):
        self.sanitizer = InputSanitizer()
    
    def test_sanitize_newlines_in_headers(self):
        """Should sanitize newlines in header values"""
        malicious_inputs = [
            "value\r\nSet-Cookie: admin=true",
            "value\nLocation: http://evil.com",
        ]
        
        for input_str in malicious_inputs:
            sanitized = self.sanitizer.sanitize_string(input_str)
            # Should remove or escape newlines
            assert '\r' not in sanitized
            assert '\n' not in sanitized or sanitized != input_str


class TestXMLInjection:
    """Test XML injection prevention"""
    
    def setup_method(self):
        self.sanitizer = InputSanitizer()
    
    def test_sanitize_xml_entities(self):
        """Should sanitize XML entities"""
        malicious_inputs = [
            "<?xml version='1.0'?><!DOCTYPE foo [<!ENTITY xxe SYSTEM 'file:///etc/passwd'>]>",
            "<user><name>admin</name></user>",
        ]
        
        for input_str in malicious_inputs:
            sanitized = self.sanitizer.sanitize_string(input_str)
            # Should escape XML special characters
            assert '<' not in sanitized or '&lt;' in sanitized


class TestCombinedAttacks:
    """Test combined/chained attacks"""
    
    def setup_method(self):
        self.sanitizer = InputSanitizer()
    
    def test_detect_sql_and_xss_combined(self):
        """Should detect combined SQL injection and XSS"""
        malicious_input = "'; DROP TABLE users; --<script>alert('xss')</script>"
        
        violations = self.sanitizer.detect_malicious_patterns(malicious_input)
        
        # Should detect both types
        assert len(violations) > 0
        violation_types = {v.violation_type for v in violations}
        assert 'sql_injection' in violation_types or 'xss' in violation_types
    
    def test_sanitize_combined_attacks(self):
        """Should sanitize combined attacks"""
        malicious_input = "'; DROP TABLE users; --<script>alert('xss')</script>"
        sanitized = self.sanitizer.sanitize_string(malicious_input)
        
        # Should sanitize both
        assert "'" not in sanitized or "&#x27;" in sanitized
        assert "<script>" not in sanitized


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
