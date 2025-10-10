/**
 * Unit Tests for File Hash Calculation (SHA-256)
 */

import { describe, it, expect } from 'vitest';

// SHA-256 hash calculation (from FastAPIProcessor)
const calculateFileHash = async (fileBuffer: ArrayBuffer): Promise<string> => {
  const hashBuffer = await crypto.subtle.digest('SHA-256', fileBuffer);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
};

const createMockFile = (content: string, name: string = 'test.txt'): File => {
  const blob = new Blob([content], { type: 'text/plain' });
  return new File([blob], name);
};

describe('File Hash Calculation', () => {
  describe('Basic Hash Calculation', () => {
    it('should calculate SHA-256 hash correctly', async () => {
      const file = createMockFile('test content');
      const buffer = await file.arrayBuffer();
      
      const hash = await calculateFileHash(buffer);
      
      expect(hash).toBeDefined();
      expect(hash.length).toBe(64); // SHA-256 produces 64 hex characters
      expect(hash).toMatch(/^[a-f0-9]{64}$/); // Only lowercase hex
    });

    it('should produce consistent hashes for same content', async () => {
      const content = 'consistent test content';
      const file1 = createMockFile(content);
      const file2 = createMockFile(content);
      
      const buffer1 = await file1.arrayBuffer();
      const buffer2 = await file2.arrayBuffer();
      
      const hash1 = await calculateFileHash(buffer1);
      const hash2 = await calculateFileHash(buffer2);
      
      expect(hash1).toBe(hash2);
    });

    it('should produce different hashes for different content', async () => {
      const file1 = createMockFile('content A');
      const file2 = createMockFile('content B');
      
      const buffer1 = await file1.arrayBuffer();
      const buffer2 = await file2.arrayBuffer();
      
      const hash1 = await calculateFileHash(buffer1);
      const hash2 = await calculateFileHash(buffer2);
      
      expect(hash1).not.toBe(hash2);
    });

    it('should handle empty files', async () => {
      const file = createMockFile('');
      const buffer = await file.arrayBuffer();
      
      const hash = await calculateFileHash(buffer);
      
      expect(hash).toBeDefined();
      expect(hash.length).toBe(64);
      // SHA-256 of empty string
      expect(hash).toBe('e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855');
    });

    it('should handle large files', async () => {
      const largeContent = 'x'.repeat(10 * 1024 * 1024); // 10MB
      const file = createMockFile(largeContent);
      const buffer = await file.arrayBuffer();
      
      const startTime = performance.now();
      const hash = await calculateFileHash(buffer);
      const endTime = performance.now();
      
      expect(hash).toBeDefined();
      expect(hash.length).toBe(64);
      expect(endTime - startTime).toBeLessThan(5000); // Should complete in <5s
    });
  });

  describe('Hash Consistency', () => {
    it('should not change hash based on filename', async () => {
      const content = 'same content';
      const file1 = createMockFile(content, 'file1.txt');
      const file2 = createMockFile(content, 'file2.txt');
      
      const buffer1 = await file1.arrayBuffer();
      const buffer2 = await file2.arrayBuffer();
      
      const hash1 = await calculateFileHash(buffer1);
      const hash2 = await calculateFileHash(buffer2);
      
      expect(hash1).toBe(hash2);
    });

    it('should detect single byte difference', async () => {
      const file1 = createMockFile('test content');
      const file2 = createMockFile('test contenu'); // Last char different
      
      const buffer1 = await file1.arrayBuffer();
      const buffer2 = await file2.arrayBuffer();
      
      const hash1 = await calculateFileHash(buffer1);
      const hash2 = await calculateFileHash(buffer2);
      
      expect(hash1).not.toBe(hash2);
    });

    it('should handle unicode content', async () => {
      const file = createMockFile('Hello 世界 🌍');
      const buffer = await file.arrayBuffer();
      
      const hash = await calculateFileHash(buffer);
      
      expect(hash).toBeDefined();
      expect(hash.length).toBe(64);
    });

    it('should handle binary content', async () => {
      const binaryData = new Uint8Array([0, 1, 2, 255, 254, 253]);
      const buffer = binaryData.buffer;
      
      const hash = await calculateFileHash(buffer);
      
      expect(hash).toBeDefined();
      expect(hash.length).toBe(64);
    });
  });

  describe('Performance', () => {
    it('should hash small files quickly', async () => {
      const file = createMockFile('small content');
      const buffer = await file.arrayBuffer();
      
      const startTime = performance.now();
      await calculateFileHash(buffer);
      const endTime = performance.now();
      
      expect(endTime - startTime).toBeLessThan(100); // <100ms
    });

    it('should hash multiple files efficiently', async () => {
      const files = Array.from({ length: 10 }, (_, i) => 
        createMockFile(`content ${i}`)
      );
      
      const startTime = performance.now();
      
      const hashes = await Promise.all(
        files.map(async (file) => {
          const buffer = await file.arrayBuffer();
          return calculateFileHash(buffer);
        })
      );
      
      const endTime = performance.now();
      
      expect(hashes.length).toBe(10);
      expect(new Set(hashes).size).toBe(10); // All unique
      expect(endTime - startTime).toBeLessThan(1000); // <1s for 10 files
    });
  });

  describe('Edge Cases', () => {
    it('should handle files with null bytes', async () => {
      const content = 'test\x00content\x00with\x00nulls';
      const file = createMockFile(content);
      const buffer = await file.arrayBuffer();
      
      const hash = await calculateFileHash(buffer);
      
      expect(hash).toBeDefined();
      expect(hash.length).toBe(64);
    });

    it('should handle very long content', async () => {
      const longContent = 'a'.repeat(1024 * 1024); // 1MB
      const file = createMockFile(longContent);
      const buffer = await file.arrayBuffer();
      
      const hash = await calculateFileHash(buffer);
      
      expect(hash).toBeDefined();
      expect(hash.length).toBe(64);
    });

    it('should handle special characters', async () => {
      const specialContent = '!@#$%^&*()_+-=[]{}|;:,.<>?/~`';
      const file = createMockFile(specialContent);
      const buffer = await file.arrayBuffer();
      
      const hash = await calculateFileHash(buffer);
      
      expect(hash).toBeDefined();
      expect(hash.length).toBe(64);
    });

    it('should handle newlines and whitespace', async () => {
      const content = 'line1\nline2\r\nline3\ttab\t\tspaces   ';
      const file = createMockFile(content);
      const buffer = await file.arrayBuffer();
      
      const hash = await calculateFileHash(buffer);
      
      expect(hash).toBeDefined();
      expect(hash.length).toBe(64);
    });
  });

  describe('Known Hash Values', () => {
    it('should match known SHA-256 hash for "hello world"', async () => {
      const file = createMockFile('hello world');
      const buffer = await file.arrayBuffer();
      
      const hash = await calculateFileHash(buffer);
      
      // Known SHA-256 hash of "hello world"
      expect(hash).toBe('b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9');
    });

    it('should match known SHA-256 hash for "test"', async () => {
      const file = createMockFile('test');
      const buffer = await file.arrayBuffer();
      
      const hash = await calculateFileHash(buffer);
      
      // Known SHA-256 hash of "test"
      expect(hash).toBe('9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08');
    });
  });

  describe('Collision Resistance', () => {
    it('should produce different hashes for similar content', async () => {
      const variations = [
        'test',
        'Test',
        'test ',
        ' test',
        'test\n',
        'tset'
      ];
      
      const hashes = await Promise.all(
        variations.map(async (content) => {
          const file = createMockFile(content);
          const buffer = await file.arrayBuffer();
          return calculateFileHash(buffer);
        })
      );
      
      // All hashes should be unique
      expect(new Set(hashes).size).toBe(variations.length);
    });
  });
});
