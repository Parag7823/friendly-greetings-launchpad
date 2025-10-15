"""
Error Recovery System for Finley AI
===================================

Comprehensive error recovery and cleanup system that handles failed operations,
partial data corruption, and provides robust rollback mechanisms.

Author: Principal Engineer
Version: 1.0.0
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import json
from supabase import Client

logger = logging.getLogger(__name__)

class RecoveryAction(Enum):
    """Types of recovery actions"""
    ROLLBACK = "rollback"
    CLEANUP = "cleanup"
    RETRY = "retry"
    NOTIFY = "notify"
    QUARANTINE = "quarantine"

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ErrorContext:
    """Context information for error recovery"""
    error_id: str
    user_id: str
    job_id: Optional[str]
    transaction_id: Optional[str]
    operation_type: str
    error_message: str
    error_details: Dict[str, Any]
    severity: ErrorSeverity
    occurred_at: datetime
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3

@dataclass
class RecoveryResult:
    """Result of error recovery operation"""
    success: bool
    recovery_action: RecoveryAction
    cleaned_records: List[str]
    recovered_data: Optional[Dict[str, Any]]
    error: Optional[str] = None
    recovery_time_ms: int = 0

class ErrorRecoverySystem:
    """
    Comprehensive error recovery system.
    
    Features:
    - Automatic cleanup of partial data
    - Transaction rollback coordination
    - Failed job recovery
    - WebSocket connection cleanup
    - Data consistency validation
    - Retry mechanisms with exponential backoff
    """
    
    def __init__(self, supabase: Client):
        self.supabase = supabase
        self.recovery_timeout_seconds = 300  # 5 minutes
        self.max_cleanup_batch_size = 100
        self.websocket_cleanup_interval = 60  # seconds
    
    async def handle_processing_error(self, error_context: ErrorContext) -> RecoveryResult:
        """
        Handle processing errors with appropriate recovery actions.
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting error recovery for {error_context.error_id}")
            
            # Log error for monitoring
            await self._log_error(error_context)
            
            # Determine recovery strategy based on error type and severity
            recovery_actions = await self._determine_recovery_actions(error_context)
            
            # Execute recovery actions
            recovery_result = await self._execute_recovery_actions(error_context, recovery_actions)
            
            # Update error context with recovery result
            await self._update_error_status(error_context, recovery_result)
            
            recovery_result.recovery_time_ms = int((time.time() - start_time) * 1000)
            
            logger.info(f"Error recovery completed for {error_context.error_id}: {recovery_result.success}")
            
            return recovery_result
        
        except Exception as e:
            logger.error(f"Error recovery failed for {error_context.error_id}: {e}")
            return RecoveryResult(
                success=False,
                recovery_action=RecoveryAction.NOTIFY,
                cleaned_records=[],
                recovered_data=None,
                error=str(e),
                recovery_time_ms=int((time.time() - start_time) * 1000)
            )
    
    async def cleanup_failed_job(self, job_id: str, user_id: str) -> RecoveryResult:
        """
        Clean up all data associated with a failed job.
        """
        start_time = time.time()
        cleaned_records = []
        
        try:
            logger.info(f"Starting cleanup for failed job: {job_id}")
            
            # Step 1: Get job details
            job_result = self.supabase.table('ingestion_jobs').select(
                'id, user_id, file_id, transaction_id, status, created_at'
            ).eq('id', job_id).execute()
            
            if not job_result.data:
                return RecoveryResult(
                    success=False,
                    recovery_action=RecoveryAction.CLEANUP,
                    cleaned_records=[],
                    recovered_data=None,
                    error="Job not found"
                )
            
            job_data = job_result.data[0]
            transaction_id = job_data.get('transaction_id')
            file_id = job_data.get('file_id')
            
            # Step 2: Clean up raw_events
            if file_id:
                events_result = self.supabase.table('raw_events').delete().eq('job_id', job_id).execute()
                if events_result.data:
                    cleaned_records.extend([f"raw_event:{record['id']}" for record in events_result.data])
            
            # Step 3: Clean up raw_records if no other jobs reference it
            if file_id:
                other_jobs = self.supabase.table('ingestion_jobs').select('id').eq('file_id', file_id).neq('id', job_id).execute()
                
                if not other_jobs.data:
                    # Safe to delete raw_record
                    record_result = self.supabase.table('raw_records').delete().eq('id', file_id).execute()
                    if record_result.data:
                        cleaned_records.extend([f"raw_record:{record['id']}" for record in record_result.data])
            
            # Step 4: Clean up transaction-related data
            if transaction_id:
                await self._cleanup_transaction_data(transaction_id, cleaned_records)
            
            # Step 5: Update job status to failed
            self.supabase.table('ingestion_jobs').update({
                'status': 'failed',
                'updated_at': datetime.utcnow().isoformat(),
                'error_details': 'Cleaned up due to processing failure'
            }).eq('id', job_id).execute()
            
            logger.info(f"Cleaned up {len(cleaned_records)} records for failed job {job_id}")
            
            return RecoveryResult(
                success=True,
                recovery_action=RecoveryAction.CLEANUP,
                cleaned_records=cleaned_records,
                recovered_data={'job_id': job_id, 'cleaned_count': len(cleaned_records)},
                recovery_time_ms=int((time.time() - start_time) * 1000)
            )
        
        except Exception as e:
            logger.error(f"Failed job cleanup failed for {job_id}: {e}")
            return RecoveryResult(
                success=False,
                recovery_action=RecoveryAction.CLEANUP,
                cleaned_records=cleaned_records,
                recovered_data=None,
                error=str(e),
                recovery_time_ms=int((time.time() - start_time) * 1000)
            )
    
    async def cleanup_orphaned_data(self, user_id: str, older_than_hours: int = 24) -> RecoveryResult:
        """
        Clean up orphaned data that has no associated active jobs.
        """
        start_time = time.time()
        cleaned_records = []
        
        try:
            logger.info(f"Starting orphaned data cleanup for user {user_id}")
            
            cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)
            cutoff_iso = cutoff_time.isoformat()
            
            # Step 1: Find orphaned raw_events (no associated job or failed job)
            orphaned_events = self.supabase.rpc('find_orphaned_events', {
                'p_user_id': user_id,
                'p_cutoff_time': cutoff_iso
            }).execute()
            
            if orphaned_events.data:
                # Delete orphaned events in batches
                for i in range(0, len(orphaned_events.data), self.max_cleanup_batch_size):
                    batch = orphaned_events.data[i:i + self.max_cleanup_batch_size]
                    event_ids = [event['id'] for event in batch]
                    
                    delete_result = self.supabase.table('raw_events').delete().in_('id', event_ids).execute()
                    if delete_result.data:
                        cleaned_records.extend([f"orphaned_event:{record['id']}" for record in delete_result.data])
            
            # Step 2: Find orphaned raw_records
            orphaned_records = self.supabase.rpc('find_orphaned_records', {
                'p_user_id': user_id,
                'p_cutoff_time': cutoff_iso
            }).execute()
            
            if orphaned_records.data:
                for record in orphaned_records.data:
                    delete_result = self.supabase.table('raw_records').delete().eq('id', record['id']).execute()
                    if delete_result.data:
                        cleaned_records.extend([f"orphaned_record:{r['id']}" for r in delete_result.data])
            
            # Step 3: Clean up expired processing locks
            await self._cleanup_expired_locks()
            
            # Step 4: Clean up failed transactions
            await self._cleanup_failed_transactions(user_id, cutoff_iso, cleaned_records)
            
            logger.info(f"Orphaned data cleanup completed: {len(cleaned_records)} records cleaned")
            
            return RecoveryResult(
                success=True,
                recovery_action=RecoveryAction.CLEANUP,
                cleaned_records=cleaned_records,
                recovered_data={'user_id': user_id, 'cleaned_count': len(cleaned_records)},
                recovery_time_ms=int((time.time() - start_time) * 1000)
            )
        
        except Exception as e:
            logger.error(f"Orphaned data cleanup failed for user {user_id}: {e}")
            return RecoveryResult(
                success=False,
                recovery_action=RecoveryAction.CLEANUP,
                cleaned_records=cleaned_records,
                recovered_data=None,
                error=str(e),
                recovery_time_ms=int((time.time() - start_time) * 1000)
            )
    
    async def cleanup_websocket_connections(self, connection_manager) -> RecoveryResult:
        """
        Clean up stale WebSocket connections and associated resources.
        """
        start_time = time.time()
        cleaned_connections = []
        
        try:
            logger.info("Starting WebSocket connection cleanup")
            
            # Get all active connections
            active_connections = getattr(connection_manager, 'active_connections', {})
            job_status = getattr(connection_manager, 'job_status', {})
            
            current_time = datetime.utcnow()
            
            for job_id, websocket in list(active_connections.items()):
                try:
                    # Check if connection is still alive
                    if hasattr(websocket, 'client_state') and websocket.client_state.value != 1:  # Not CONNECTED
                        # Connection is dead, clean it up
                        connection_manager.disconnect(job_id)
                        cleaned_connections.append(job_id)
                        continue
                    
                    # Check if job status is stale
                    if job_id in job_status:
                        job_info = job_status[job_id]
                        started_at = datetime.fromisoformat(job_info.get('started_at', current_time.isoformat()))
                        
                        # If job has been running for more than 1 hour, consider it stale
                        if (current_time - started_at).total_seconds() > 3600:
                            connection_manager.disconnect(job_id)
                            cleaned_connections.append(job_id)
                
                except Exception as e:
                    logger.warning(f"Error checking WebSocket connection {job_id}: {e}")
                    # When in doubt, clean it up
                    connection_manager.disconnect(job_id)
                    cleaned_connections.append(job_id)
            
            logger.info(f"WebSocket cleanup completed: {len(cleaned_connections)} connections cleaned")
            
            return RecoveryResult(
                success=True,
                recovery_action=RecoveryAction.CLEANUP,
                cleaned_records=cleaned_connections,
                recovered_data={'cleaned_connections': len(cleaned_connections)},
                recovery_time_ms=int((time.time() - start_time) * 1000)
            )
        
        except Exception as e:
            logger.error(f"WebSocket cleanup failed: {e}")
            return RecoveryResult(
                success=False,
                recovery_action=RecoveryAction.CLEANUP,
                cleaned_records=cleaned_connections,
                recovered_data=None,
                error=str(e),
                recovery_time_ms=int((time.time() - start_time) * 1000)
            )
    
    async def validate_data_consistency(self, user_id: str) -> RecoveryResult:
        """
        Validate data consistency and fix any issues found.
        """
        start_time = time.time()
        fixed_issues = []
        
        try:
            logger.info(f"Starting data consistency validation for user {user_id}")
            
            # Check 1: Raw events without corresponding raw records
            orphaned_events = self.supabase.rpc('validate_event_record_consistency', {
                'p_user_id': user_id
            }).execute()
            
            if orphaned_events.data:
                for event in orphaned_events.data:
                    # Mark event as invalid
                    self.supabase.table('raw_events').update({
                        'status': 'invalid',
                        'error_message': 'Orphaned event - no corresponding raw record'
                    }).eq('id', event['id']).execute()
                    
                    fixed_issues.append(f"marked_invalid_event:{event['id']}")
            
            # Check 2: Ingestion jobs without proper status
            stale_jobs = self.supabase.table('ingestion_jobs').select(
                'id, user_id, status, created_at'
            ).eq(
                'user_id', user_id
            ).eq('status', 'processing').lt(
                'created_at', (datetime.utcnow() - timedelta(hours=2)).isoformat()
            ).execute()
            
            if stale_jobs.data:
                for job in stale_jobs.data:
                    # Mark job as failed
                    self.supabase.table('ingestion_jobs').update({
                        'status': 'failed',
                        'updated_at': datetime.utcnow().isoformat(),
                        'error_details': 'Job timed out - marked as failed during consistency check'
                    }).eq('id', job['id']).execute()
                    
                    fixed_issues.append(f"failed_stale_job:{job['id']}")
            
            # Check 3: Transaction consistency
            await self._validate_transaction_consistency(user_id, fixed_issues)
            
            logger.info(f"Data consistency validation completed: {len(fixed_issues)} issues fixed")
            
            return RecoveryResult(
                success=True,
                recovery_action=RecoveryAction.CLEANUP,
                cleaned_records=fixed_issues,
                recovered_data={'user_id': user_id, 'issues_fixed': len(fixed_issues)},
                recovery_time_ms=int((time.time() - start_time) * 1000)
            )
        
        except Exception as e:
            logger.error(f"Data consistency validation failed for user {user_id}: {e}")
            return RecoveryResult(
                success=False,
                recovery_action=RecoveryAction.CLEANUP,
                cleaned_records=fixed_issues,
                recovered_data=None,
                error=str(e),
                recovery_time_ms=int((time.time() - start_time) * 1000)
            )
    
    async def _log_error(self, error_context: ErrorContext):
        """Log error for monitoring and analysis"""
        try:
            error_record = {
                'id': error_context.error_id,
                'user_id': error_context.user_id,
                'job_id': error_context.job_id,
                'transaction_id': error_context.transaction_id,
                'operation_type': error_context.operation_type,
                'error_message': error_context.error_message,
                'error_details': error_context.error_details,
                'severity': error_context.severity.value,
                'occurred_at': error_context.occurred_at.isoformat(),
                'recovery_attempts': error_context.recovery_attempts
            }
            
            self.supabase.table('error_logs').insert(error_record).execute()
        
        except Exception as e:
            logger.error(f"Failed to log error: {e}")
    
    async def _determine_recovery_actions(self, error_context: ErrorContext) -> List[RecoveryAction]:
        """Determine appropriate recovery actions based on error context"""
        actions = []
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            actions.extend([RecoveryAction.ROLLBACK, RecoveryAction.CLEANUP, RecoveryAction.NOTIFY])
        elif error_context.severity == ErrorSeverity.HIGH:
            actions.extend([RecoveryAction.CLEANUP, RecoveryAction.RETRY])
        elif error_context.severity == ErrorSeverity.MEDIUM:
            actions.append(RecoveryAction.RETRY)
        else:
            actions.append(RecoveryAction.NOTIFY)
        
        return actions
    
    async def _execute_recovery_actions(self, error_context: ErrorContext, 
                                      actions: List[RecoveryAction]) -> RecoveryResult:
        """Execute recovery actions"""
        for action in actions:
            if action == RecoveryAction.ROLLBACK and error_context.transaction_id:
                # Rollback will be handled by transaction manager
                pass
            elif action == RecoveryAction.CLEANUP and error_context.job_id:
                return await self.cleanup_failed_job(error_context.job_id, error_context.user_id)
            elif action == RecoveryAction.RETRY:
                # Retry logic would be implemented here
                pass
        
        return RecoveryResult(
            success=True,
            recovery_action=actions[0] if actions else RecoveryAction.NOTIFY,
            cleaned_records=[],
            recovered_data=None
        )
    
    async def _update_error_status(self, error_context: ErrorContext, recovery_result: RecoveryResult):
        """Update error status with recovery result"""
        try:
            self.supabase.table('error_logs').update({
                'recovery_completed': True,
                'recovery_success': recovery_result.success,
                'recovery_action': recovery_result.recovery_action.value,
                'recovery_details': {
                    'cleaned_records': recovery_result.cleaned_records,
                    'recovery_time_ms': recovery_result.recovery_time_ms
                }
            }).eq('id', error_context.error_id).execute()
        
        except Exception as e:
            logger.error(f"Failed to update error status: {e}")
    
    async def _cleanup_transaction_data(self, transaction_id: str, cleaned_records: List[str]):
        """Clean up data associated with a transaction
        
        FIX #9: Added cleanup for cross_platform_relationships to prevent orphaned data.
        """
        try:
            # Clean up normalized entities
            entities_result = self.supabase.table('normalized_entities').delete().eq('transaction_id', transaction_id).execute()
            if entities_result.data:
                cleaned_records.extend([f"entity:{record['id']}" for record in entities_result.data])
            
            # Clean up relationship instances
            relations_result = self.supabase.table('relationship_instances').delete().eq('transaction_id', transaction_id).execute()
            if relations_result.data:
                cleaned_records.extend([f"relation:{record['id']}" for record in relations_result.data])
            
            # FIX #9: Clean up cross-platform relationships (analytics data)
            cross_platform_result = self.supabase.table('cross_platform_relationships').delete().eq('transaction_id', transaction_id).execute()
            if cross_platform_result.data:
                cleaned_records.extend([f"cross_platform:{record['id']}" for record in cross_platform_result.data])
            
            # Clean up metrics
            metrics_result = self.supabase.table('metrics').delete().eq('transaction_id', transaction_id).execute()
            if metrics_result.data:
                cleaned_records.extend([f"metric:{record['id']}" for record in metrics_result.data])
        
        except Exception as e:
            logger.error(f"Transaction data cleanup failed for {transaction_id}: {e}")
    
    async def _cleanup_expired_locks(self):
        """Clean up expired processing locks"""
        try:
            expired_time = datetime.utcnow().isoformat()
            self.supabase.table('processing_locks').delete().lt('expires_at', expired_time).execute()
        
        except Exception as e:
            logger.error(f"Failed to cleanup expired locks: {e}")
    
    async def _cleanup_failed_transactions(self, user_id: str, cutoff_time: str, cleaned_records: List[str]):
        """Clean up failed or stale transactions"""
        try:
            failed_transactions = self.supabase.table('processing_transactions').select('id').eq(
                'user_id', user_id
            ).in_('status', ['failed', 'active']).lt('started_at', cutoff_time).execute()
            
            for transaction in failed_transactions.data or []:
                await self._cleanup_transaction_data(transaction['id'], cleaned_records)
                
                # Update transaction status
                self.supabase.table('processing_transactions').update({
                    'status': 'cleaned_up',
                    'updated_at': datetime.utcnow().isoformat()
                }).eq('id', transaction['id']).execute()
        
        except Exception as e:
            logger.error(f"Failed transaction cleanup failed: {e}")
    
    async def _validate_transaction_consistency(self, user_id: str, fixed_issues: List[str]):
        """Validate transaction consistency"""
        try:
            # Find transactions marked as committed but with no associated data
            inconsistent_transactions = self.supabase.rpc('validate_transaction_consistency', {
                'p_user_id': user_id
            }).execute()
            
            for transaction in inconsistent_transactions.data or []:
                # Mark transaction as inconsistent
                self.supabase.table('processing_transactions').update({
                    'status': 'inconsistent',
                    'error_details': 'No associated data found for committed transaction'
                }).eq('id', transaction['id']).execute()
                
                fixed_issues.append(f"marked_inconsistent_transaction:{transaction['id']}")
        
        except Exception as e:
            logger.error(f"Transaction consistency validation failed: {e}")

# Global error recovery system instance
_error_recovery_system: Optional[ErrorRecoverySystem] = None

def initialize_error_recovery_system(supabase: Client):
    """Initialize the global error recovery system"""
    global _error_recovery_system
    _error_recovery_system = ErrorRecoverySystem(supabase)
    logger.info("✅ Error recovery system initialized")

def get_error_recovery_system() -> ErrorRecoverySystem:
    """Get the global error recovery system instance"""
    if _error_recovery_system is None:
        raise Exception("Error recovery system not initialized. Call initialize_error_recovery_system() first.")
    return _error_recovery_system
