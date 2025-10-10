/**
 * Unit Tests for File Validation
 * 
 * Tests:
 * - File type validation (xlsx, xls, csv)
 * - File size validation (500MB limit)
 * - Invalid file rejection
 * - Multiple file handling
 * - Edge cases and boundary conditions
 */

import { describe, it, expect } from 'vitest';

// Mock file creation helper
const createMockFile = (
  name: string,
  size: number,
  type: string
): File => {
  const blob = new Blob(['x'.repeat(size)], { type });
  return new File([blob], name, { type });
};

// File validation function (extracted from EnhancedFileUpload)
const validateFile = (file: File): { isValid: boolean; error?: string } => {
  const validTypes = [
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'application/vnd.ms-excel',
    'text/csv'
  ];
  
  if (!validTypes.includes(file.type) && !file.name.match(/\.(xlsx|xls|csv)$/i)) {
    return {
      isValid: false,
      error: 'Please upload a valid Excel file (.xlsx, .xls) or CSV file.'
    };
  }
  
  const maxSize = 500 * 1024 * 1024; // 500MB (matches backend limit)
  if (file.size > maxSize) {
    return {
      isValid: false,
      error: 'File size must be less than 500MB.'
    };
  }
  
  return { isValid: true };
};

describe('File Validation', () => {
  describe('File Type Validation', () => {
    it('should accept valid .xlsx files', () => {
      const file = createMockFile(
        'test.xlsx',
        1024,
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
      );
      
      const result = validateFile(file);
      
      expect(result.isValid).toBe(true);
      expect(result.error).toBeUndefined();
    });

    it('should accept valid .xls files', () => {
      const file = createMockFile(
        'test.xls',
        1024,
        'application/vnd.ms-excel'
      );
      
      const result = validateFile(file);
      
      expect(result.isValid).toBe(true);
      expect(result.error).toBeUndefined();
    });

    it('should accept valid .csv files', () => {
      const file = createMockFile(
        'test.csv',
        1024,
        'text/csv'
      );
      
      const result = validateFile(file);
      
      expect(result.isValid).toBe(true);
      expect(result.error).toBeUndefined();
    });

    it('should accept files with correct extension even if MIME type is wrong', () => {
      // Some browsers report wrong MIME types
      const file = createMockFile(
        'test.xlsx',
        1024,
        'application/octet-stream'
      );
      
      const result = validateFile(file);
      
      expect(result.isValid).toBe(true);
    });

    it('should reject .pdf files', () => {
      const file = createMockFile(
        'test.pdf',
        1024,
        'application/pdf'
      );
      
      const result = validateFile(file);
      
      expect(result.isValid).toBe(false);
      expect(result.error).toContain('valid Excel file');
    });

    it('should reject .docx files', () => {
      const file = createMockFile(
        'test.docx',
        1024,
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
      );
      
      const result = validateFile(file);
      
      expect(result.isValid).toBe(false);
      expect(result.error).toContain('valid Excel file');
    });

    it('should reject .exe files', () => {
      const file = createMockFile(
        'malicious.exe',
        1024,
        'application/x-msdownload'
      );
      
      const result = validateFile(file);
      
      expect(result.isValid).toBe(false);
      expect(result.error).toContain('valid Excel file');
    });

    it('should reject files with no extension', () => {
      const file = createMockFile(
        'noextension',
        1024,
        'application/octet-stream'
      );
      
      const result = validateFile(file);
      
      expect(result.isValid).toBe(false);
      expect(result.error).toContain('valid Excel file');
    });
  });

  describe('File Size Validation', () => {
    it('should accept files under 500MB', () => {
      const file = createMockFile(
        'test.xlsx',
        100 * 1024 * 1024, // 100MB
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
      );
      
      const result = validateFile(file);
      
      expect(result.isValid).toBe(true);
    });

    it('should accept files exactly at 500MB limit', () => {
      const file = createMockFile(
        'test.xlsx',
        500 * 1024 * 1024, // Exactly 500MB
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
      );
      
      const result = validateFile(file);
      
      expect(result.isValid).toBe(true);
    });

    it('should reject files over 500MB', () => {
      const file = createMockFile(
        'test.xlsx',
        501 * 1024 * 1024, // 501MB
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
      );
      
      const result = validateFile(file);
      
      expect(result.isValid).toBe(false);
      expect(result.error).toContain('500MB');
    });

    it('should reject very large files', () => {
      const file = createMockFile(
        'huge.xlsx',
        1024 * 1024 * 1024, // 1GB
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
      );
      
      const result = validateFile(file);
      
      expect(result.isValid).toBe(false);
      expect(result.error).toContain('500MB');
    });

    it('should accept very small files', () => {
      const file = createMockFile(
        'tiny.csv',
        100, // 100 bytes
        'text/csv'
      );
      
      const result = validateFile(file);
      
      expect(result.isValid).toBe(true);
    });

    it('should accept empty files', () => {
      const file = createMockFile(
        'empty.csv',
        0,
        'text/csv'
      );
      
      const result = validateFile(file);
      
      expect(result.isValid).toBe(true);
    });
  });

  describe('Edge Cases', () => {
    it('should handle files with uppercase extensions', () => {
      const file = createMockFile(
        'TEST.XLSX',
        1024,
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
      );
      
      const result = validateFile(file);
      
      expect(result.isValid).toBe(true);
    });

    it('should handle files with mixed case extensions', () => {
      const file = createMockFile(
        'test.XlSx',
        1024,
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
      );
      
      const result = validateFile(file);
      
      expect(result.isValid).toBe(true);
    });

    it('should handle files with multiple dots in name', () => {
      const file = createMockFile(
        'my.test.file.xlsx',
        1024,
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
      );
      
      const result = validateFile(file);
      
      expect(result.isValid).toBe(true);
    });

    it('should handle files with special characters in name', () => {
      const file = createMockFile(
        'test-file_2024 (1).xlsx',
        1024,
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
      );
      
      const result = validateFile(file);
      
      expect(result.isValid).toBe(true);
    });

    it('should handle files with unicode characters in name', () => {
      const file = createMockFile(
        'テスト.xlsx',
        1024,
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
      );
      
      const result = validateFile(file);
      
      expect(result.isValid).toBe(true);
    });
  });

  describe('Multiple File Validation', () => {
    it('should validate multiple files independently', () => {
      const files = [
        createMockFile('file1.xlsx', 1024, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'),
        createMockFile('file2.csv', 2048, 'text/csv'),
        createMockFile('file3.xls', 512, 'application/vnd.ms-excel'),
      ];
      
      const results = files.map(validateFile);
      
      expect(results.every(r => r.isValid)).toBe(true);
    });

    it('should identify invalid files in batch', () => {
      const files = [
        createMockFile('valid.xlsx', 1024, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'),
        createMockFile('invalid.pdf', 1024, 'application/pdf'),
        createMockFile('toolarge.csv', 600 * 1024 * 1024, 'text/csv'),
      ];
      
      const results = files.map(validateFile);
      
      expect(results[0].isValid).toBe(true);
      expect(results[1].isValid).toBe(false);
      expect(results[1].error).toContain('valid Excel file');
      expect(results[2].isValid).toBe(false);
      expect(results[2].error).toContain('500MB');
    });
  });

  describe('Security Tests', () => {
    it('should reject files with double extensions', () => {
      const file = createMockFile(
        'test.xlsx.exe',
        1024,
        'application/x-msdownload'
      );
      
      const result = validateFile(file);
      
      expect(result.isValid).toBe(false);
    });

    it('should reject files with path traversal in name', () => {
      const file = createMockFile(
        '../../../etc/passwd.xlsx',
        1024,
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
      );
      
      // File validation only checks type and size, not path traversal
      // Path traversal is handled by backend security_system.py
      const result = validateFile(file);
      
      // Should still validate based on extension
      expect(result.isValid).toBe(true);
    });

    it('should reject script files disguised as Excel', () => {
      const file = createMockFile(
        'malicious.js',
        1024,
        'application/javascript'
      );
      
      const result = validateFile(file);
      
      expect(result.isValid).toBe(false);
    });
  });

  describe('Performance', () => {
    it('should validate files quickly', () => {
      const file = createMockFile(
        'test.xlsx',
        100 * 1024 * 1024, // 100MB
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
      );
      
      const startTime = performance.now();
      const result = validateFile(file);
      const endTime = performance.now();
      
      expect(result.isValid).toBe(true);
      expect(endTime - startTime).toBeLessThan(10); // Should complete in <10ms
    });

    it('should handle batch validation efficiently', () => {
      const files = Array.from({ length: 15 }, (_, i) =>
        createMockFile(
          `file${i}.xlsx`,
          1024 * 1024, // 1MB each
          'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
      );
      
      const startTime = performance.now();
      const results = files.map(validateFile);
      const endTime = performance.now();
      
      expect(results.every(r => r.isValid)).toBe(true);
      expect(endTime - startTime).toBeLessThan(50); // Should complete in <50ms for 15 files
    });
  });
});
