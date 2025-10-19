"""
Unit Tests for Provenance Tracking
===================================

Purpose: Test row hash calculation and lineage path construction
What we're testing: Tamper detection, transformation tracking, audit trail
Why: Ensure every financial number can explain itself

Author: Finley AI Team
Date: 2025-10-19
"""

import pytest
import hashlib
import json
from datetime import datetime
from provenance_tracker import ProvenanceTracker, calculate_row_hash, create_lineage_path, append_lineage_step


class TestRowHashCalculation:
    """
    Test row hash calculation for tamper detection.
    
    Plain English: Every row gets a unique fingerprint (hash).
    If someone changes the data, the hash won't match - we detect tampering!
    """
    
    def setup_method(self):
        """Set up test fixtures"""
        self.tracker = ProvenanceTracker()
    
    def test_calculate_row_hash_basic(self):
        """
        Test basic row hash calculation.
        
        Plain English: Given a filename, row number, and data,
        calculate a unique hash that identifies this exact row.
        """
        # Given: Row data
        source_filename = "invoice_2025.xlsx"
        row_index = 42
        payload = {"vendor": "Acme Corp", "amount": 1500.00}
        
        # When: We calculate hash
        row_hash = self.tracker.calculate_row_hash(source_filename, row_index, payload)
        
        # Then: Hash should be valid SHA256 (64 hex characters)
        assert row_hash is not None
        assert len(row_hash) == 64
        assert all(c in '0123456789abcdef' for c in row_hash)
    
    def test_same_data_produces_same_hash(self):
        """
        Test that identical data produces identical hash.
        
        Plain English: If we calculate the hash twice for the same data,
        we should get the exact same result. This is called "deterministic".
        """
        # Given: Same data
        source_filename = "invoice_2025.xlsx"
        row_index = 42
        payload = {"vendor": "Acme Corp", "amount": 1500.00}
        
        # When: We calculate hash twice
        hash1 = self.tracker.calculate_row_hash(source_filename, row_index, payload)
        hash2 = self.tracker.calculate_row_hash(source_filename, row_index, payload)
        
        # Then: Hashes should be identical
        assert hash1 == hash2
    
    def test_different_data_produces_different_hash(self):
        """
        Test that different data produces different hash.
        
        Plain English: If we change even one character in the data,
        the hash should be completely different. This detects tampering!
        """
        # Given: Two different payloads
        payload1 = {"vendor": "Acme Corp", "amount": 1500.00}
        payload2 = {"vendor": "Acme Corp", "amount": 1500.01}  # Changed by $0.01!
        
        # When: We calculate hashes
        hash1 = self.tracker.calculate_row_hash("invoice.xlsx", 1, payload1)
        hash2 = self.tracker.calculate_row_hash("invoice.xlsx", 1, payload2)
        
        # Then: Hashes should be different
        assert hash1 != hash2
    
    def test_different_row_index_produces_different_hash(self):
        """Test that different row numbers produce different hashes"""
        # Given: Same payload, different row indices
        payload = {"vendor": "Acme Corp", "amount": 1500.00}
        
        # When: We calculate hashes for different rows
        hash1 = self.tracker.calculate_row_hash("invoice.xlsx", 1, payload)
        hash2 = self.tracker.calculate_row_hash("invoice.xlsx", 2, payload)
        
        # Then: Hashes should be different
        assert hash1 != hash2
    
    def test_different_filename_produces_different_hash(self):
        """Test that different filenames produce different hashes"""
        # Given: Same payload, different filenames
        payload = {"vendor": "Acme Corp", "amount": 1500.00}
        
        # When: We calculate hashes
        hash1 = self.tracker.calculate_row_hash("invoice1.xlsx", 1, payload)
        hash2 = self.tracker.calculate_row_hash("invoice2.xlsx", 1, payload)
        
        # Then: Hashes should be different
        assert hash1 != hash2
    
    def test_hash_handles_special_characters(self):
        """Test hash calculation with special characters"""
        # Given: Payload with special characters
        payload = {
            "vendor": "Café & Co. (Pty) Ltd.",
            "amount": 1500.00,
            "note": "Payment for 'services' — 50% discount!"
        }
        
        # When: We calculate hash
        row_hash = self.tracker.calculate_row_hash("invoice.xlsx", 1, payload)
        
        # Then: Should handle gracefully
        assert row_hash is not None
        assert len(row_hash) == 64
    
    def test_hash_handles_unicode(self):
        """Test hash calculation with Unicode characters"""
        # Given: Payload with Unicode
        payload = {
            "vendor": "北京公司",  # Chinese characters
            "amount": 1500.00,
            "note": "Payment for services 🎉"  # Emoji
        }
        
        # When: We calculate hash
        row_hash = self.tracker.calculate_row_hash("invoice.xlsx", 1, payload)
        
        # Then: Should handle gracefully
        assert row_hash is not None
        assert len(row_hash) == 64


class TestRowHashVerification:
    """
    Test row hash verification for tamper detection.
    
    Plain English: After storing a hash, we can verify if the data
    has been modified by recalculating and comparing hashes.
    """
    
    def setup_method(self):
        """Set up test fixtures"""
        self.tracker = ProvenanceTracker()
    
    def test_verify_row_hash_valid(self):
        """
        Test verification of valid (unmodified) data.
        
        Plain English: If data hasn't been changed, verification should pass.
        """
        # Given: Original data and its hash
        source_filename = "invoice.xlsx"
        row_index = 1
        payload = {"vendor": "Acme Corp", "amount": 1500.00}
        stored_hash = self.tracker.calculate_row_hash(source_filename, row_index, payload)
        
        # When: We verify with same data
        is_valid, message = self.tracker.verify_row_hash(
            stored_hash, source_filename, row_index, payload
        )
        
        # Then: Should be valid
        assert is_valid is True
        assert "verified" in message.lower()
    
    def test_verify_row_hash_detects_tampering(self):
        """
        Test detection of tampered data.
        
        Plain English: If someone changes the amount from $1500 to $1501,
        we should detect it immediately!
        """
        # Given: Original data and its hash
        original_payload = {"vendor": "Acme Corp", "amount": 1500.00}
        stored_hash = self.tracker.calculate_row_hash("invoice.xlsx", 1, original_payload)
        
        # When: We verify with modified data
        tampered_payload = {"vendor": "Acme Corp", "amount": 1501.00}  # Changed!
        is_valid, message = self.tracker.verify_row_hash(
            stored_hash, "invoice.xlsx", 1, tampered_payload
        )
        
        # Then: Should detect tampering
        assert is_valid is False
        assert "tamper" in message.lower() or "modified" in message.lower()


class TestLineagePathConstruction:
    """
    Test lineage path construction for transformation tracking.
    
    Plain English: Track every transformation a row goes through,
    like a GPS history for your data!
    """
    
    def setup_method(self):
        """Set up test fixtures"""
        self.tracker = ProvenanceTracker()
    
    def test_create_initial_lineage_path(self):
        """
        Test creation of initial lineage path.
        
        Plain English: When data first enters the system,
        create the first step in its journey.
        """
        # When: We create initial lineage
        lineage = self.tracker.create_lineage_path("file_upload")
        
        # Then: Should have one step
        assert len(lineage) == 1
        assert lineage[0]['step'] == "file_upload"
        assert lineage[0]['operation'] == "raw_extract"
        assert 'timestamp' in lineage[0]
    
    def test_append_lineage_step(self):
        """
        Test appending transformation steps.
        
        Plain English: As data goes through transformations
        (classification, enrichment, etc.), add each step to the path.
        """
        # Given: Initial lineage
        lineage = self.tracker.create_lineage_path("ingestion")
        
        # When: We add classification step
        lineage = self.tracker.append_lineage_step(
            lineage,
            step="classification",
            operation="ai_classify",
            metadata={"confidence": 0.95, "kind": "invoice"}
        )
        
        # Then: Should have two steps
        assert len(lineage) == 2
        assert lineage[1]['step'] == "classification"
        assert lineage[1]['operation'] == "ai_classify"
        assert lineage[1]['metadata']['confidence'] == 0.95
    
    def test_complete_lineage_chain(self):
        """
        Test building complete lineage chain.
        
        Plain English: Track the complete journey of data through
        all transformations: upload → classify → enrich → resolve entities.
        """
        # Given: Initial lineage
        lineage = self.tracker.create_lineage_path("file_upload")
        
        # When: We add multiple transformation steps
        lineage = self.tracker.append_lineage_step(
            lineage, "platform_detection", "ai_detect_platform",
            {"platform": "QuickBooks", "confidence": 0.95}
        )
        lineage = self.tracker.append_lineage_step(
            lineage, "classification", "ai_classify",
            {"kind": "invoice", "confidence": 0.92}
        )
        lineage = self.tracker.append_lineage_step(
            lineage, "enrichment", "currency_normalize",
            {"from": "INR", "to": "USD", "rate": 83.5}
        )
        lineage = self.tracker.append_lineage_step(
            lineage, "entity_resolution", "entity_match",
            {"entity_id": "uuid123", "confidence": 0.88}
        )
        
        # Then: Should have complete chain
        assert len(lineage) == 5
        assert lineage[0]['step'] == "file_upload"
        assert lineage[1]['step'] == "platform_detection"
        assert lineage[2]['step'] == "classification"
        assert lineage[3]['step'] == "enrichment"
        assert lineage[4]['step'] == "entity_resolution"
    
    def test_lineage_timestamps_are_sequential(self):
        """Test that timestamps in lineage are sequential"""
        # Given: Initial lineage
        lineage = self.tracker.create_lineage_path("ingestion")
        
        # When: We add steps
        import time
        time.sleep(0.01)  # Small delay
        lineage = self.tracker.append_lineage_step(
            lineage, "classification", "ai_classify", {}
        )
        
        # Then: Second timestamp should be after first
        ts1 = lineage[0]['timestamp']
        ts2 = lineage[1]['timestamp']
        assert ts1 <= ts2  # Should be sequential


class TestProvenanceHelpers:
    """
    Test helper methods for provenance tracking.
    
    Plain English: Test convenience methods that make it easy
    to add common transformation steps.
    """
    
    def setup_method(self):
        """Set up test fixtures"""
        self.tracker = ProvenanceTracker()
    
    def test_add_classification_step(self):
        """Test adding classification step"""
        # Given: Initial lineage
        lineage = self.tracker.create_lineage_path()
        
        # When: We add classification
        lineage = self.tracker.add_classification_step(
            lineage, kind="invoice", category="revenue", confidence=0.95
        )
        
        # Then: Should have classification step
        assert len(lineage) == 2
        assert lineage[1]['step'] == "classification"
        assert lineage[1]['metadata']['kind'] == "invoice"
        assert lineage[1]['metadata']['confidence'] == 0.95
    
    def test_add_enrichment_step(self):
        """Test adding enrichment step"""
        # Given: Initial lineage
        lineage = self.tracker.create_lineage_path()
        
        # When: We add enrichment
        lineage = self.tracker.add_enrichment_step(
            lineage, "currency_normalize", {"from": "INR", "to": "USD"}
        )
        
        # Then: Should have enrichment step
        assert len(lineage) == 2
        assert lineage[1]['step'] == "enrichment"
        assert lineage[1]['operation'] == "currency_normalize"
    
    def test_add_entity_resolution_step(self):
        """Test adding entity resolution step"""
        # Given: Initial lineage
        lineage = self.tracker.create_lineage_path()
        
        # When: We add entity resolution
        lineage = self.tracker.add_entity_resolution_step(
            lineage, entity_id="uuid123", entity_name="Acme Corp", confidence=0.88
        )
        
        # Then: Should have entity resolution step
        assert len(lineage) == 2
        assert lineage[1]['step'] == "entity_resolution"
        assert lineage[1]['metadata']['entity_id'] == "uuid123"


class TestCompleteProvenanceGeneration:
    """
    Test complete provenance generation.
    
    Plain English: Test the main function that generates all
    provenance data at once: hash + lineage + audit trail.
    """
    
    def setup_method(self):
        """Set up test fixtures"""
        self.tracker = ProvenanceTracker()
    
    def test_generate_complete_provenance(self):
        """
        Test generating complete provenance data.
        
        Plain English: When ingesting a row, generate everything needed:
        - Row hash for tamper detection
        - Initial lineage path
        - Audit trail (who created it)
        """
        # When: We generate complete provenance
        provenance = self.tracker.generate_complete_provenance(
            source_filename="invoice_2025.xlsx",
            row_index=42,
            original_payload={"vendor": "Acme Corp", "amount": 1500.00},
            created_by="user:123e4567-e89b-12d3-a456-426614174000"
        )
        
        # Then: Should have all components
        assert 'row_hash' in provenance
        assert 'lineage_path' in provenance
        assert 'created_by' in provenance
        
        # Validate row hash
        assert len(provenance['row_hash']) == 64
        
        # Validate lineage path
        assert len(provenance['lineage_path']) >= 1
        assert provenance['lineage_path'][0]['step'] == "ingestion"
        
        # Validate audit trail
        assert provenance['created_by'] == "user:123e4567-e89b-12d3-a456-426614174000"


class TestConvenienceFunctions:
    """
    Test convenience functions for easy import.
    
    Plain English: Test the simple functions that can be imported
    directly without creating a ProvenanceTracker instance.
    """
    
    def test_convenience_calculate_row_hash(self):
        """Test convenience function for row hash"""
        # When: We use convenience function
        row_hash = calculate_row_hash("invoice.xlsx", 1, {"amount": 100})
        
        # Then: Should work
        assert row_hash is not None
        assert len(row_hash) == 64
    
    def test_convenience_create_lineage_path(self):
        """Test convenience function for lineage creation"""
        # When: We use convenience function
        lineage = create_lineage_path("file_upload")
        
        # Then: Should work
        assert len(lineage) == 1
        assert lineage[0]['step'] == "file_upload"
    
    def test_convenience_append_lineage_step(self):
        """Test convenience function for appending step"""
        # Given: Initial lineage
        lineage = create_lineage_path()
        
        # When: We use convenience function
        lineage = append_lineage_step(
            lineage, "classification", "ai_classify", {"confidence": 0.95}
        )
        
        # Then: Should work
        assert len(lineage) == 2


# ============================================================================
# WHAT DID WE TEST? (Plain English Summary)
# ============================================================================

"""
SUMMARY OF PROVENANCE TESTS:

1. **Row Hash Calculation Tests**
   - Purpose: Test tamper detection via SHA256 hashing
   - Why: Every row needs a unique fingerprint to detect changes
   - Result: Ensures data integrity and audit compliance

2. **Row Hash Verification Tests**
   - Purpose: Test detection of data tampering
   - Why: If someone changes $1500 to $1501, we catch it!
   - Result: Validates tamper detection works correctly

3. **Lineage Path Construction Tests**
   - Purpose: Test transformation tracking
   - Why: Every number needs to explain its journey
   - Result: Enables "Ask Why" feature for CFOs

4. **Provenance Helper Tests**
   - Purpose: Test convenience methods for common steps
   - Why: Make it easy to add classification, enrichment, etc.
   - Result: Simplifies provenance tracking in code

5. **Complete Provenance Tests**
   - Purpose: Test end-to-end provenance generation
   - Why: Validate the main entry point works correctly
   - Result: Ensures complete provenance for every row

HOW TO RUN THESE TESTS:
```bash
# Run all provenance tests
pytest tests/unit/test_provenance_tracking.py -v

# Run specific test class
pytest tests/unit/test_provenance_tracking.py::TestRowHashCalculation -v

# Run with coverage
pytest tests/unit/test_provenance_tracking.py --cov=provenance_tracker --cov-report=html
```

WHAT WE VALIDATED:
✅ Row hashes are deterministic (same data = same hash)
✅ Row hashes detect tampering (changed data = different hash)
✅ Lineage paths track all transformations
✅ Timestamps are sequential
✅ Helper methods work correctly
✅ Complete provenance generation works end-to-end

These tests ensure every financial number can explain itself!
"""
