"""
Provenance Tracker - Complete Data Lineage and Tamper Detection
================================================================

This module provides comprehensive provenance tracking for Finley's financial data,
enabling "Google Maps for your financial data" - every number can explain itself.

Features:
- Row-level tamper detection via SHA256 hashing
- Full transformation chain tracking (lineage path)
- Audit trail with created_by/modified_by
- "Ask Why" explainability support

Author: Finley AI Team
Date: 2025-10-16
"""

import hashlib
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class LineageStep:
    """Represents a single step in the data transformation chain"""
    step: str
    operation: str
    timestamp: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSONB storage"""
        return {
            'step': self.step,
            'operation': self.operation,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }


class ProvenanceTracker:
    """
    Tracks complete provenance for financial data events.
    
    Provides:
    1. Row hash calculation for tamper detection
    2. Lineage path construction for transformation tracking
    3. Audit trail management
    """
    
    def __init__(self):
        self.logger = structlog.get_logger(__name__)
    
    # ========================================================================
    # ROW HASH CALCULATION (Tamper Detection)
    # ========================================================================
    
    def calculate_row_hash(
        self,
        source_filename: str,
        row_index: int,
        payload: Dict[str, Any]
    ) -> str:
        """
        Calculate SHA256 hash of row data for tamper detection.
        
        The hash is calculated from:
        - source_filename: Original file name
        - row_index: Row number in source file
        - payload: Original raw data (before any transformations)
        
        This creates a unique fingerprint that can detect any modifications
        to the original data.
        
        Args:
            source_filename: Name of source file
            row_index: Row index in source file
            payload: Original row data as dictionary
            
        Returns:
            SHA256 hash as hex string
            
        Example:
            >>> tracker = ProvenanceTracker()
            >>> hash_val = tracker.calculate_row_hash(
            ...     "invoice_2025.xlsx",
            ...     42,
            ...     {"vendor": "Acme Corp", "amount": 1500.00}
            ... )
            >>> print(hash_val)
            'a3f5b8c9d2e1f4a7b6c5d8e9f2a1b4c7d6e5f8a9b2c1d4e7f6a5b8c9d2e1f4a7'
        """
        try:
            # Create canonical representation
            hash_input = f"{source_filename}||{row_index}||{json.dumps(payload, sort_keys=True)}"
            
            # Calculate SHA256
            hash_obj = hashlib.sha256(hash_input.encode('utf-8'))
            row_hash = hash_obj.hexdigest()
            
            self.logger.debug(f"Calculated row hash for {source_filename}:{row_index} = {row_hash[:16]}...")
            return row_hash
            
        except Exception as e:
            self.logger.error(f"Failed to calculate row hash: {e}")
            # Return empty hash on error - better than crashing
            return ""
    
    def verify_row_hash(
        self,
        stored_hash: str,
        source_filename: str,
        row_index: int,
        payload: Dict[str, Any]
    ) -> tuple[bool, str]:
        """
        Verify row integrity by comparing stored hash with recalculated hash.
        
        Args:
            stored_hash: Hash stored in database
            source_filename: Name of source file
            row_index: Row index in source file
            payload: Current row data
            
        Returns:
            Tuple of (is_valid, message)
            
        Example:
            >>> is_valid, msg = tracker.verify_row_hash(
            ...     stored_hash="a3f5b8c9...",
            ...     source_filename="invoice_2025.xlsx",
            ...     row_index=42,
            ...     payload={"vendor": "Acme Corp", "amount": 1500.00}
            ... )
            >>> if not is_valid:
            ...     print(f"TAMPERING DETECTED: {msg}")
        """
        try:
            # Recalculate hash
            recalculated_hash = self.calculate_row_hash(source_filename, row_index, payload)
            
            # Compare
            if stored_hash == recalculated_hash:
                return True, "Row integrity verified - no tampering detected"
            else:
                return False, f"CRITICAL: Row data has been modified - tampering detected (stored: {stored_hash[:16]}..., calculated: {recalculated_hash[:16]}...)"
                
        except Exception as e:
            self.logger.error(f"Failed to verify row hash: {e}")
            return False, f"Hash verification failed: {str(e)}"
    
    # ========================================================================
    # LINEAGE PATH CONSTRUCTION (Transformation Tracking)
    # ========================================================================
    
    def create_lineage_path(self, initial_step: str = "ingestion") -> List[Dict[str, Any]]:
        """
        Create initial lineage path with ingestion step.
        
        Args:
            initial_step: Name of initial step (default: "ingestion")
            
        Returns:
            List containing initial lineage step
            
        Example:
            >>> lineage = tracker.create_lineage_path("file_upload")
            >>> print(lineage)
            [{'step': 'file_upload', 'operation': 'raw_extract', 'timestamp': '2025-10-16T10:00:00Z', 'metadata': {}}]
        """
        return [self._build_lineage_step(
            step=initial_step,
            operation="raw_extract",
            metadata={}
        )]
    
    def append_lineage_step(
        self,
        existing_path: List[Dict[str, Any]],
        step: str,
        operation: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Append a new transformation step to lineage path.
        
        Args:
            existing_path: Current lineage path
            step: Name of transformation step
            operation: Specific operation performed
            metadata: Additional context (confidence scores, parameters, etc.)
            
        Returns:
            Updated lineage path
            
        Example:
            >>> lineage = tracker.create_lineage_path()
            >>> lineage = tracker.append_lineage_step(
            ...     lineage,
            ...     step="classification",
            ...     operation="ai_classify",
            ...     metadata={"confidence": 0.95, "model": "gpt-4"}
            ... )
            >>> lineage = tracker.append_lineage_step(
            ...     lineage,
            ...     step="enrichment",
            ...     operation="currency_normalize",
            ...     metadata={"from": "INR", "to": "USD", "rate": 83.5}
            ... )
        """
        new_step = self._build_lineage_step(step, operation, metadata or {})
        return existing_path + [new_step]
    
    def _build_lineage_step(
        self,
        step: str,
        operation: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build a single lineage step with timestamp"""
        return {
            'step': step,
            'operation': operation,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'metadata': metadata
        }
    
    # ========================================================================
    # COMPLETE PROVENANCE GENERATION
    # ========================================================================
    
    def generate_complete_provenance(
        self,
        source_filename: str,
        row_index: int,
        original_payload: Dict[str, Any],
        created_by: str,
        initial_step: str = "ingestion"
    ) -> Dict[str, Any]:
        """
        Generate complete provenance data for a new event.
        
        This is the main entry point for creating provenance when ingesting data.
        
        Args:
            source_filename: Name of source file
            row_index: Row index in source file
            original_payload: Original raw data
            created_by: User ID or system agent (e.g., "user:uuid" or "system:excel_processor")
            initial_step: Name of initial step
            
        Returns:
            Dictionary with row_hash, lineage_path, and created_by
            
        Example:
            >>> provenance = tracker.generate_complete_provenance(
            ...     source_filename="invoice_2025.xlsx",
            ...     row_index=42,
            ...     original_payload={"vendor": "Acme Corp", "amount": 1500.00},
            ...     created_by="user:123e4567-e89b-12d3-a456-426614174000"
            ... )
            >>> print(provenance)
            {
                'row_hash': 'a3f5b8c9d2e1f4a7...',
                'lineage_path': [{'step': 'ingestion', ...}],
                'created_by': 'user:123e4567-e89b-12d3-a456-426614174000'
            }
        """
        return {
            'row_hash': self.calculate_row_hash(source_filename, row_index, original_payload),
            'lineage_path': self.create_lineage_path(initial_step),
            'created_by': created_by
        }
    
    # ========================================================================
    # LINEAGE PATH HELPERS
    # ========================================================================
    
    def add_classification_step(
        self,
        lineage_path: List[Dict[str, Any]],
        kind: str,
        category: str,
        confidence: float,
        model: str = "ai_classifier"
    ) -> List[Dict[str, Any]]:
        """Add AI classification step to lineage"""
        return self.append_lineage_step(
            lineage_path,
            step="classification",
            operation="ai_classify",
            metadata={
                'kind': kind,
                'category': category,
                'confidence': confidence,
                'model': model
            }
        )
    
    def add_enrichment_step(
        self,
        lineage_path: List[Dict[str, Any]],
        enrichment_type: str,
        details: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Add data enrichment step to lineage"""
        return self.append_lineage_step(
            lineage_path,
            step="enrichment",
            operation=enrichment_type,
            metadata=details
        )
    
    def add_entity_resolution_step(
        self,
        lineage_path: List[Dict[str, Any]],
        entity_id: str,
        entity_name: str,
        confidence: float,
        match_method: str = "fuzzy_match"
    ) -> List[Dict[str, Any]]:
        """Add entity resolution step to lineage"""
        return self.append_lineage_step(
            lineage_path,
            step="entity_resolution",
            operation="entity_match",
            metadata={
                'entity_id': entity_id,
                'entity_name': entity_name,
                'confidence': confidence,
                'match_method': match_method
            }
        )
    
    def add_validation_step(
        self,
        lineage_path: List[Dict[str, Any]],
        validation_type: str,
        passed: bool,
        details: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Add validation step to lineage"""
        return self.append_lineage_step(
            lineage_path,
            step="validation",
            operation=validation_type,
            metadata={
                'passed': passed,
                'details': details or {}
            }
        )
    
    # ========================================================================
    # AUDIT TRAIL HELPERS
    # ========================================================================
    
    def format_created_by(self, user_id: Optional[str] = None, system_agent: Optional[str] = None) -> str:
        """
        Format created_by field for audit trail.
        
        Args:
            user_id: User UUID (if user-initiated)
            system_agent: System agent name (if system-initiated)
            
        Returns:
            Formatted created_by string
            
        Example:
            >>> tracker.format_created_by(user_id="123e4567-e89b-12d3-a456-426614174000")
            'user:123e4567-e89b-12d3-a456-426614174000'
            >>> tracker.format_created_by(system_agent="excel_processor")
            'system:excel_processor'
        """
        if user_id:
            return f"user:{user_id}"
        elif system_agent:
            return f"system:{system_agent}"
        else:
            return "system:unknown"
    
    # ========================================================================
    # PROVENANCE QUERY HELPERS
    # ========================================================================
    
    def format_lineage_for_display(self, lineage_path: List[Dict[str, Any]]) -> str:
        """
        Format lineage path as human-readable string for display.
        
        Args:
            lineage_path: Lineage path from database
            
        Returns:
            Formatted string showing transformation chain
            
        Example:
            >>> lineage = [
            ...     {'step': 'ingestion', 'operation': 'raw_extract', 'timestamp': '2025-10-16T10:00:00Z'},
            ...     {'step': 'classification', 'operation': 'ai_classify', 'metadata': {'confidence': 0.95}},
            ...     {'step': 'enrichment', 'operation': 'currency_normalize', 'metadata': {'from': 'INR', 'to': 'USD'}}
            ... ]
            >>> print(tracker.format_lineage_for_display(lineage))
            1. ingestion → raw_extract (2025-10-16 10:00:00)
            2. classification → ai_classify (confidence: 0.95)
            3. enrichment → currency_normalize (INR → USD)
        """
        lines = []
        for i, step in enumerate(lineage_path, 1):
            timestamp = step.get('timestamp', '').split('T')[1].split('.')[0] if 'T' in step.get('timestamp', '') else ''
            metadata = step.get('metadata', {})
            
            # Format metadata
            meta_str = ""
            if metadata:
                if 'confidence' in metadata:
                    meta_str = f" (confidence: {metadata['confidence']:.2f})"
                elif 'from' in metadata and 'to' in metadata:
                    meta_str = f" ({metadata['from']} → {metadata['to']})"
                elif metadata:
                    meta_str = f" ({', '.join(f'{k}: {v}' for k, v in list(metadata.items())[:2])})"
            
            lines.append(f"{i}. {step['step']} → {step['operation']}{meta_str}")
        
        return '\n'.join(lines)
    
    def get_lineage_summary(self, lineage_path: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get summary statistics from lineage path.
        
        Returns:
            Dictionary with summary stats
        """
        return {
            'total_steps': len(lineage_path),
            'steps': [step['step'] for step in lineage_path],
            'operations': [step['operation'] for step in lineage_path],
            'duration_seconds': self._calculate_duration(lineage_path),
            'has_ai_classification': any(step['step'] == 'classification' for step in lineage_path),
            'has_enrichment': any(step['step'] == 'enrichment' for step in lineage_path),
            'has_entity_resolution': any(step['step'] == 'entity_resolution' for step in lineage_path)
        }
    
    def _calculate_duration(self, lineage_path: List[Dict[str, Any]]) -> Optional[float]:
        """Calculate total duration from first to last step"""
        if len(lineage_path) < 2:
            return None
        
        try:
            first_ts = datetime.fromisoformat(lineage_path[0]['timestamp'].replace('Z', '+00:00'))
            last_ts = datetime.fromisoformat(lineage_path[-1]['timestamp'].replace('Z', '+00:00'))
            return (last_ts - first_ts).total_seconds()
        except Exception:
            return None


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

# Create global instance for easy import
provenance_tracker = ProvenanceTracker()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def calculate_row_hash(source_filename: str, row_index: int, payload: Dict[str, Any]) -> str:
    """Convenience function for calculating row hash"""
    return provenance_tracker.calculate_row_hash(source_filename, row_index, payload)


def create_lineage_path(initial_step: str = "ingestion") -> List[Dict[str, Any]]:
    """Convenience function for creating lineage path"""
    return provenance_tracker.create_lineage_path(initial_step)


def append_lineage_step(
    existing_path: List[Dict[str, Any]],
    step: str,
    operation: str,
    metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Convenience function for appending lineage step"""
    return provenance_tracker.append_lineage_step(existing_path, step, operation, metadata)
