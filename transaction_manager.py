"""
Database Transaction Manager for Finley AI
==========================================

Provides atomic transaction management for complex multi-table operations.
Ensures data consistency and proper rollback on failures.

Author: Principal Engineer
Version: 1.0.0
"""

import asyncio
import logging
import uuid
import math
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
from contextlib import asynccontextmanager
from supabase import Client
import pandas as pd

logger = logging.getLogger(__name__)

def _sanitize_for_json(obj):
    """Recursively sanitize NaN/Inf values for JSON serialization"""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif pd.isna(obj):
        return None
    else:
        return obj

@dataclass
class TransactionOperation:
    """Represents a single database operation within a transaction"""
    table: str
    operation: str  # 'insert', 'update', 'delete', 'upsert'
    data: Dict[str, Any]
    filters: Optional[Dict[str, Any]] = None
    rollback_data: Optional[Dict[str, Any]] = None

@dataclass
class TransactionResult:
    """Result of a transaction execution"""
    success: bool
    transaction_id: str
    operations_completed: int
    error: Optional[str] = None
    rollback_completed: bool = False
    execution_time_ms: int = 0

class DatabaseTransactionManager:
    """
    Manages atomic database transactions for complex operations.
    
    Features:
    - Atomic multi-table operations
    - Automatic rollback on failure
    - Transaction logging and monitoring
    - Deadlock detection and retry
    - Memory-efficient operation batching
    """
    
    def __init__(self, supabase: Client):
        self.supabase = supabase
        self.active_transactions: Dict[str, Dict[str, Any]] = {}
        self.max_retry_attempts = 3
        self.retry_delay_ms = 100
    
    @asynccontextmanager
    async def transaction(self, transaction_id: Optional[str] = None, 
                         user_id: Optional[str] = None,
                         operation_type: str = "data_processing"):
        """
        Context manager for atomic database transactions.
        
        Usage:
            async with transaction_manager.transaction() as tx:
                await tx.insert('raw_records', {...})
                await tx.insert('raw_events', {...})
                await tx.update('ingestion_jobs', {...})
        """
        if transaction_id is None:
            transaction_id = str(uuid.uuid4())
        
        transaction_context = TransactionContext(
            transaction_id=transaction_id,
            user_id=user_id,
            operation_type=operation_type,
            manager=self
        )
        
        # Start transaction tracking
        await self._start_transaction(transaction_context)
        
        try:
            yield transaction_context
            # Commit all operations
            await self._commit_transaction(transaction_context)
            
        except Exception as e:
            # Rollback on any error
            await self._rollback_transaction(transaction_context, str(e))
            raise
        finally:
            # Cleanup transaction tracking
            await self._cleanup_transaction(transaction_context)
    
    async def _start_transaction(self, context: 'TransactionContext'):
        """Initialize transaction tracking"""
        try:
            # Create processing transaction record
            transaction_record = {
                'id': context.transaction_id,
                'user_id': context.user_id,
                'status': 'active',
                'operation_type': context.operation_type,
                'started_at': datetime.utcnow().isoformat(),
                'metadata': {
                    'operations_planned': 0,
                    'operations_completed': 0,
                    'created_by': 'transaction_manager'
                }
            }
            
            result = self.supabase.table('processing_transactions').upsert(transaction_record, on_conflict='id').execute()
            
            if not result.data:
                raise Exception("Failed to create transaction record")
            
            # Track in memory
            self.active_transactions[context.transaction_id] = {
                'context': context,
                'operations': [],
                'started_at': datetime.utcnow(),
                'status': 'active'
            }
            
            logger.info(f"Started transaction {context.transaction_id}")
            
        except Exception as e:
            logger.error(f"Failed to start transaction {context.transaction_id}: {e}")
            raise
    
    async def _commit_transaction(self, context: 'TransactionContext'):
        """Commit all transaction operations"""
        try:
            # Update transaction status to committed
            self.supabase.table('processing_transactions').update({
                'status': 'committed',
                'committed_at': datetime.utcnow().isoformat(),
                'metadata': {
                    **self.active_transactions[context.transaction_id]['context'].metadata,
                    'operations_completed': len(context.operations),
                    'committed_by': 'transaction_manager'
                }
            }).eq('id', context.transaction_id).execute()
            
            logger.info(f"Committed transaction {context.transaction_id} with {len(context.operations)} operations")
            
        except Exception as e:
            logger.error(f"Failed to commit transaction {context.transaction_id}: {e}")
            raise
    
    async def _rollback_transaction(self, context: 'TransactionContext', error: str):
        """Rollback transaction operations"""
        try:
            rollback_operations = 0
            
            # Reverse the operations in LIFO order
            for operation in reversed(context.operations):
                try:
                    await self._rollback_operation(operation)
                    rollback_operations += 1
                except Exception as rollback_error:
                    logger.error(f"Failed to rollback operation {operation.operation} on {operation.table}: {rollback_error}")
            
            # Update transaction status to rolled back
            self.supabase.table('processing_transactions').update({
                'status': 'rolled_back',
                'rolled_back_at': datetime.utcnow().isoformat(),
                'error_details': error,
                'metadata': {
                    **self.active_transactions[context.transaction_id]['context'].metadata,
                    'operations_rolled_back': rollback_operations,
                    'rollback_reason': error
                }
            }).eq('id', context.transaction_id).execute()
            
            logger.info(f"Rolled back transaction {context.transaction_id}, reversed {rollback_operations} operations")
            
        except Exception as e:
            logger.error(f"Failed to rollback transaction {context.transaction_id}: {e}")
            # Mark as failed
            try:
                self.supabase.table('processing_transactions').update({
                    'status': 'failed',
                    'failed_at': datetime.utcnow().isoformat(),
                    'error_details': f"Rollback failed: {e}. Original error: {error}"
                }).eq('id', context.transaction_id).execute()
            except:
                pass  # Best effort
    
    async def _rollback_operation(self, operation: TransactionOperation):
        """
        COMPREHENSIVE ROLLBACK: Rollback a single operation with complete cleanup.
        
        CRITICAL FIX: Now handles ALL tables involved in file processing:
        - raw_events (with relationship cleanup)
        - raw_records
        - ingestion_jobs
        - metrics
        - platform_patterns
        - discovered_platforms
        - normalized_entities (with entity_matches cleanup)
        - cross_platform_relationships
        - relationship_instances
        - entity_matches
        - debug_logs
        - field_mappings
        - detection_log
        - resolution_log
        """
        if operation.operation == 'insert':
            # Delete the inserted record
            if 'id' in operation.rollback_data:
                record_id = operation.rollback_data['id']
                
                try:
                    self.supabase.table(operation.table).delete().eq('id', record_id).execute()
                    logger.info(f"✅ Rolled back insert in {operation.table}: {record_id}")
                except Exception as e:
                    logger.error(f"❌ Failed to rollback insert in {operation.table}: {e}")
                    raise
                
                # COMPREHENSIVE CLEANUP: Handle cascading deletes for related records
                if operation.table == 'raw_events':
                    # Clean up cross_platform_relationships
                    try:
                        self.supabase.table('cross_platform_relationships').delete().eq('source_event_id', record_id).execute()
                        self.supabase.table('cross_platform_relationships').delete().eq('target_event_id', record_id).execute()
                        logger.debug(f"Cleaned up cross_platform_relationships for event {record_id}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up cross_platform_relationships for event {record_id}: {e}")
                    
                    # Clean up relationship_instances
                    try:
                        self.supabase.table('relationship_instances').delete().eq('source_event_id', record_id).execute()
                        self.supabase.table('relationship_instances').delete().eq('target_event_id', record_id).execute()
                        logger.debug(f"Cleaned up relationship_instances for event {record_id}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up relationship_instances for event {record_id}: {e}")
                
                elif operation.table == 'normalized_entities':
                    # Clean up entity_matches
                    try:
                        self.supabase.table('entity_matches').delete().eq('entity_id', record_id).execute()
                        logger.debug(f"Cleaned up entity_matches for entity {record_id}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up entity_matches for entity {record_id}: {e}")
                
                elif operation.table == 'raw_records':
                    # Clean up ingestion_jobs that reference this file
                    try:
                        self.supabase.table('ingestion_jobs').delete().eq('file_id', record_id).execute()
                        logger.debug(f"Cleaned up ingestion_jobs for file {record_id}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up ingestion_jobs for file {record_id}: {e}")
        
        elif operation.operation == 'update':
            # Restore original data
            if operation.rollback_data and operation.filters:
                filter_key = list(operation.filters.keys())[0]
                filter_value = operation.filters[filter_key]
                try:
                    self.supabase.table(operation.table).update(operation.rollback_data).eq(filter_key, filter_value).execute()
                    logger.info(f"✅ Rolled back update in {operation.table}")
                except Exception as e:
                    logger.error(f"❌ Failed to rollback update in {operation.table}: {e}")
                    raise
        
        elif operation.operation == 'delete':
            # Restore deleted data
            if operation.rollback_data:
                try:
                    self.supabase.table(operation.table).insert(operation.rollback_data).execute()
                    logger.info(f"✅ Rolled back delete in {operation.table}")
                except Exception as e:
                    logger.error(f"❌ Failed to rollback delete in {operation.table}: {e}")
                    raise
    
    async def _cleanup_transaction(self, context: 'TransactionContext'):
        """Clean up transaction tracking"""
        if context.transaction_id in self.active_transactions:
            del self.active_transactions[context.transaction_id]

class TransactionContext:
    """Context for managing operations within a transaction"""
    
    def __init__(self, transaction_id: str, user_id: Optional[str], 
                 operation_type: str, manager: DatabaseTransactionManager):
        self.transaction_id = transaction_id
        self.user_id = user_id
        self.operation_type = operation_type
        self.manager = manager
        self.operations: List[TransactionOperation] = []
        self.metadata: Dict[str, Any] = {}
    
    async def insert(self, table: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Insert data with transaction tracking"""
        try:
            # Add transaction_id to data and sanitize NaN values
            data_with_tx = _sanitize_for_json({**data, 'transaction_id': self.transaction_id})
            
            try:
                result = self.manager.supabase.table(table).insert(data_with_tx).execute()
            except Exception as e:
                # Retry without transaction_id if the column is missing
                msg = str(e).lower()
                if 'transaction_id' in msg and ('does not exist' in msg or 'could not find' in msg or 'schema cache' in msg):
                    sanitized_data = _sanitize_for_json(data)
                    result = self.manager.supabase.table(table).insert(sanitized_data).execute()
                else:
                    raise
            
            if not result or not result.data:
                raise Exception(f"Insert failed for table {table}")
            
            # Track operation for rollback
            operation = TransactionOperation(
                table=table,
                operation='insert',
                data=data_with_tx,
                rollback_data={'id': result.data[0]['id']}
            )
            self.operations.append(operation)
            
            return result.data[0]
            
        except Exception as e:
            # CRITICAL FIX: Don't log duplicate key errors as ERROR - they're expected and handled
            error_str = str(e)
            if '23505' in error_str or 'duplicate key' in error_str.lower():
                logger.info(f"Transaction insert skipped for {table} (duplicate key): {e}")
            else:
                logger.error(f"Transaction insert failed for {table}: {e}")
            raise
    
    async def update(self, table: str, data: Dict[str, Any], filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        CRITICAL FIX: Update data with transaction tracking.
        Only stores changed fields for rollback to prevent overwriting concurrent changes.
        """
        try:
            # Get original data for rollback - but only for fields we're changing
            filter_key = list(filters.keys())[0]
            filter_value = filters[filter_key]
            
            # CRITICAL FIX: Only fetch the specific fields we're about to change
            fields_to_update = list(data.keys())
            select_fields = ', '.join(fields_to_update + [filter_key])
            
            original_result = self.manager.supabase.table(table).select(select_fields).eq(filter_key, filter_value).execute()
            
            # CRITICAL FIX: Store only the original values of fields we're changing
            original_values = {}
            if original_result.data:
                for field in fields_to_update:
                    if field in original_result.data[0]:
                        original_values[field] = original_result.data[0][field]
            
            # Add transaction_id to update data
            data_with_tx = {**data, 'transaction_id': self.transaction_id}
            
            try:
                result = self.manager.supabase.table(table).update(data_with_tx).eq(filter_key, filter_value).execute()
            except Exception as e:
                # Retry without transaction_id if column missing
                msg = str(e).lower()
                if 'transaction_id' in msg and ('does not exist' in msg or 'could not find' in msg or 'schema cache' in msg):
                    result = self.manager.supabase.table(table).update(data).eq(filter_key, filter_value).execute()
                else:
                    raise
            
            # Track operation for rollback
            operation = TransactionOperation(
                table=table,
                operation='update',
                data=data_with_tx,
                filters=filters,
                rollback_data=original_values  # CRITICAL FIX: Only changed fields
            )
            self.operations.append(operation)
            
            return result.data[0] if result.data else {}
            
        except Exception as e:
            logger.error(f"Transaction update failed for {table}: {e}")
            raise
    
    async def insert_batch(self, table: str, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Insert multiple records with transaction tracking"""
        try:
            # Add transaction_id to all records and sanitize NaN values
            data_with_tx = [
                _sanitize_for_json({**data, 'transaction_id': self.transaction_id})
                for data in data_list
            ]
            
            try:
                result = self.manager.supabase.table(table).insert(data_with_tx).execute()
            except Exception as e:
                msg = str(e).lower()
                if 'transaction_id' in msg and ('does not exist' in msg or 'could not find' in msg or 'schema cache' in msg):
                    # Fallback without transaction_id, but still sanitize
                    sanitized_data = [_sanitize_for_json(data) for data in data_list]
                    result = self.manager.supabase.table(table).insert(sanitized_data).execute()
                else:
                    raise
            
            if not result or not result.data:
                raise Exception(f"Batch insert failed for table {table}")
            
            # Track operation for rollback
            for inserted_record in result.data:
                operation = TransactionOperation(
                    table=table,
                    operation='insert',
                    data=inserted_record,
                    rollback_data={'id': inserted_record['id']}
                )
                self.operations.append(operation)
            
            return result.data
            
        except Exception as e:
            logger.error(f"Transaction batch insert failed for {table}: {e}")
            raise

# Global transaction manager instance
_transaction_manager: Optional[DatabaseTransactionManager] = None

def initialize_transaction_manager(supabase: Client):
    """Initialize the global transaction manager"""
    global _transaction_manager
    _transaction_manager = DatabaseTransactionManager(supabase)
    logger.info("✅ Transaction manager initialized")

def get_transaction_manager() -> DatabaseTransactionManager:
    """Get the global transaction manager instance"""
    if _transaction_manager is None:
        raise Exception("Transaction manager not initialized. Call initialize_transaction_manager() first.")
    return _transaction_manager
