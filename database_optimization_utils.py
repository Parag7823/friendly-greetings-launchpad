"""
Database Optimization Utilities
This module provides optimized database query functions to replace inefficient queries
in the main codebase. These functions use proper indexing, pagination, and column selection.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, date
from supabase import create_client, Client
import asyncio
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    """Standardized query result container"""
    data: List[Dict[str, Any]]
    count: int
    has_more: bool
    next_offset: Optional[int] = None

class OptimizedDatabaseQueries:
    """
    Optimized database query class that replaces inefficient queries
    with properly indexed, paginated, and optimized versions.
    """
    
    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
        self.default_page_size = 100
        self.max_page_size = 1000
    
    # ============================================================================
    # RAW RECORDS & DUPLICATE MANAGEMENT QUERIES
    # ============================================================================

    async def get_duplicate_records(
        self,
        user_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Optimized query for retrieving duplicate records.
        Filters on the dedicated is_duplicate flag and selects necessary fields only.
        """
        try:
            result = (
                self.supabase
                .table('raw_records')
                .select('id, file_name, file_size, created_at, status, content')
                .eq('user_id', user_id)
                .eq('is_duplicate', True)
                .order('created_at', desc=True)
                .limit(limit)
                .execute()
            )

            return result.data or []

        except Exception as e:
            logger.error(f"Duplicate records query failed: {e}")
            return []

    async def get_pending_version_recommendations(
        self,
        user_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Optimized query for pending version recommendations awaiting user decision.
        """
        try:
            result = (
                self.supabase
                .table('version_recommendations')
                .select('id, user_id, file_id, version_group_id, recommendation_type, created_at, user_accepted, user_feedback')
                .eq('user_id', user_id)
                .is_('user_accepted', 'null')
                .order('created_at', desc=True)
                .limit(limit)
                .execute()
            )

            return result.data or []

        except Exception as e:
            logger.error(f"Pending version recommendations query failed: {e}")
            return []

    # ============================================================================
    # RAW EVENTS QUERIES - Most frequently used table
    # ============================================================================
    
    async def get_user_events_optimized(
        self, 
        user_id: str, 
        limit: int = 100, 
        offset: int = 0,
        kind: Optional[str] = None,
        source_platform: Optional[str] = None,
        status: Optional[str] = None,
        file_id: Optional[str] = None,
        job_id: Optional[str] = None
    ) -> QueryResult:
        """
        Optimized query for user events with proper pagination and filtering.
        Replaces inefficient SELECT * queries.
        """
        try:
            # Build query with only necessary columns
            query = self.supabase.table('raw_events').select(
                'id, kind, source_platform, payload, row_index, source_filename, '
                'status, confidence_score, created_at, processed_at'
            ).eq('user_id', user_id)
            
            # Apply filters
            if kind:
                query = query.eq('kind', kind)
            if source_platform:
                query = query.eq('source_platform', source_platform)
            if status:
                query = query.eq('status', status)
            if file_id:
                query = query.eq('file_id', file_id)
            if job_id:
                query = query.eq('job_id', job_id)
            
            # Apply pagination and ordering
            query = query.order('created_at', desc=True).range(offset, offset + limit - 1)
            
            # Execute query
            result = query.execute()
            
            # Get total count for pagination info
            count_query = self.supabase.table('raw_events').select('id', count='exact').eq('user_id', user_id)
            if kind:
                count_query = count_query.eq('kind', kind)
            if source_platform:
                count_query = count_query.eq('source_platform', source_platform)
            if status:
                count_query = count_query.eq('status', status)
            if file_id:
                count_query = count_query.eq('file_id', file_id)
            if job_id:
                count_query = count_query.eq('job_id', job_id)
            
            count_result = count_query.execute()
            total_count = count_result.count if hasattr(count_result, 'count') else len(result.data)
            
            return QueryResult(
                data=result.data or [],
                count=total_count,
                has_more=offset + limit < total_count,
                next_offset=offset + limit if offset + limit < total_count else None
            )
            
        except Exception as e:
            logger.error(f"Optimized user events query failed: {e}")
            return QueryResult(data=[], count=0, has_more=False)
    
    async def get_events_for_entity_extraction(self, user_id: str, file_id: str) -> List[Dict[str, Any]]:
        """
        Optimized query for entity extraction - only gets necessary fields.
        Replaces: SELECT * FROM raw_events WHERE user_id = ? AND file_id = ?
        """
        try:
            result = self.supabase.table('raw_events').select(
                'id, payload, kind, source_platform, row_index'
            ).eq('user_id', user_id).eq('file_id', file_id).execute()
            
            return result.data or []
            
        except Exception as e:
            logger.error(f"Entity extraction query failed: {e}")
            return []
    
    async def get_recent_events_optimized(
        self, 
        user_id: str, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Optimized query for recent events with proper column selection.
        Replaces: SELECT * FROM raw_events WHERE user_id = ? ORDER BY created_at DESC LIMIT 10
        """
        try:
            result = self.supabase.table('raw_events').select(
                'id, kind, source_platform, source_filename, status, confidence_score, created_at'
            ).eq('user_id', user_id).order('created_at', desc=True).limit(limit).execute()
            
            return result.data or []
            
        except Exception as e:
            logger.error(f"Recent events query failed: {e}")
            return []
    
    # ============================================================================
    # INGESTION JOBS QUERIES
    # ============================================================================
    
    async def get_job_status_optimized(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Optimized query for job status - only gets necessary fields.
        Replaces: SELECT * FROM ingestion_jobs WHERE id = ?
        """
        try:
            result = self.supabase.table('ingestion_jobs').select(
                'id, status, progress, error_message, created_at, completed_at'
            ).eq('id', job_id).execute()
            
            return result.data[0] if result.data else None
            
        except Exception as e:
            logger.error(f"Job status query failed: {e}")
            return None
    
    async def get_user_jobs_optimized(
        self, 
        user_id: str, 
        limit: int = 50, 
        offset: int = 0
    ) -> QueryResult:
        """
        Optimized query for user jobs with pagination.
        """
        try:
            query = self.supabase.table('ingestion_jobs').select(
                'id, job_type, status, progress, created_at, completed_at, error_message'
            ).eq('user_id', user_id).order('created_at', desc=True).range(offset, offset + limit - 1)
            
            result = query.execute()
            
            # Get total count
            count_result = self.supabase.table('ingestion_jobs').select(
                'id', count='exact'
            ).eq('user_id', user_id).execute()
            
            total_count = count_result.count if hasattr(count_result, 'count') else len(result.data)
            
            return QueryResult(
                data=result.data or [],
                count=total_count,
                has_more=offset + limit < total_count,
                next_offset=offset + limit if offset + limit < total_count else None
            )
            
        except Exception as e:
            logger.error(f"User jobs query failed: {e}")
            return QueryResult(data=[], count=0, has_more=False)
    
    # ============================================================================
    # METRICS QUERIES
    # ============================================================================
    
    async def get_user_metrics_optimized(
        self, 
        user_id: str, 
        metric_type: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Optimized query for user metrics with filtering.
        """
        try:
            query = self.supabase.table('metrics').select(
                'metric_type, category, amount, currency, date_recorded, confidence_score'
            ).eq('user_id', user_id)
            
            if metric_type:
                query = query.eq('metric_type', metric_type)
            if start_date:
                query = query.gte('date_recorded', start_date.isoformat())
            if end_date:
                query = query.lte('date_recorded', end_date.isoformat())
            
            query = query.order('date_recorded', desc=True).limit(limit)
            
            result = query.execute()
            return result.data or []
            
        except Exception as e:
            logger.error(f"User metrics query failed: {e}")
            return []
    
    # ============================================================================
    # ENTITY RESOLUTION QUERIES
    # ============================================================================
    
    async def get_entities_for_user_optimized(
        self, 
        user_id: str, 
        entity_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Optimized query for user entities.
        """
        try:
            query = self.supabase.table('normalized_entities').select(
                'id, entity_type, canonical_name, aliases, email, phone, '
                'platform_sources, confidence_score, last_seen_at'
            ).eq('user_id', user_id)
            
            if entity_type:
                query = query.eq('entity_type', entity_type)
            
            query = query.order('last_seen_at', desc=True)
            
            result = query.execute()
            return result.data or []
            
        except Exception as e:
            logger.error(f"User entities query failed: {e}")
            return []
    
    async def get_platforms_for_user_optimized(self, user_id: str) -> List[str]:
        """
        Optimized query to get unique platforms for a user.
        Replaces: SELECT DISTINCT source_platform FROM raw_events WHERE user_id = ?
        """
        try:
            result = self.supabase.table('raw_events').select(
                'source_platform'
            ).eq('user_id', user_id).not_.is_('source_platform', 'null').execute()
            
            # Extract unique platforms
            platforms = list(set([row['source_platform'] for row in result.data if row['source_platform']]))
            return platforms
            
        except Exception as e:
            logger.error(f"User platforms query failed: {e}")
            return []
    
    # ============================================================================
    # CHAT MESSAGES QUERIES
    # ============================================================================
    
    async def get_chat_history_optimized(
        self, 
        user_id: str, 
        chat_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Optimized query for chat history with pagination.
        """
        try:
            query = self.supabase.table('chat_messages').select(
                'id, chat_id, message, created_at'
            ).eq('user_id', user_id)
            
            if chat_id:
                query = query.eq('chat_id', chat_id)
            
            query = query.order('created_at', desc=True).limit(limit)
            
            result = query.execute()
            return result.data or []
            
        except Exception as e:
            logger.error(f"Chat history query failed: {e}")
            return []

    # ============================================================================
    # UNIVERSAL COMPONENT RESULTS QUERIES
    # ============================================================================
    
    async def get_component_results_optimized(
        self,
        user_id: str,
        component_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Optimized query for universal_component_results.
        Selects only necessary columns and supports filtering & ordering.
        """
        try:
            query = self.supabase.table('universal_component_results').select(
                'id, component_type, filename, result_data, metadata, created_at'
            ).eq('user_id', user_id)
            
            if component_type:
                query = query.eq('component_type', component_type)
            
            query = query.order('created_at', desc=True).limit(limit)
            result = query.execute()
            return result.data or []
            
        except Exception as e:
            logger.error(f"Component results query failed: {e}")
            return []

def create_optimized_db_client() -> OptimizedDatabaseQueries:
    """
    Factory function to create an optimized database client.
    """
    try:
        supabase_url = (
            os.getenv('SUPABASE_URL') or
            os.getenv('SUPABASE_PROJECT_URL') or
            os.getenv('DATABASE_URL')
        )
        supabase_key = (
            os.getenv('SUPABASE_SERVICE_ROLE_KEY') or
            os.getenv('SUPABASE_SERVICE_KEY') or
            os.getenv('SUPABASE_KEY') or
            os.getenv('SUPABASE_ANON_KEY')
        )
        
        if not supabase_url or not supabase_key:
            raise ValueError("Supabase credentials not configured")
        
        supabase_key = supabase_key.strip()
        supabase_client = create_client(supabase_url, supabase_key)
        return OptimizedDatabaseQueries(supabase_client)
        
    except Exception as e:
        logger.error(f"Failed to create optimized database client: {e}")
        raise

async def run_database_optimization_migration():
    """
    Run the database optimization migration.
    This should be called during deployment to ensure indexes are created.
    """
    try:
        # This would typically run the SQL migration file
        # For now, just log that it should be run
        logger.info("Database optimization migration should be run manually:")
        logger.info("Run: supabase/migrations/20250120000000-database-optimization.sql")
        
        return True
        
    except Exception as e:
        logger.error(f"Database optimization migration failed: {e}")
        return False


# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

class QueryPerformanceMonitor:
    """
    Monitor database query performance and provide optimization suggestions.
    """
    
    def __init__(self):
        self.query_times = {}
        self.slow_queries = []
    
    def log_query_time(self, query_name: str, execution_time: float):
        """Log query execution time for monitoring."""
        if query_name not in self.query_times:
            self.query_times[query_name] = []
        
        self.query_times[query_name].append(execution_time)
        
        # Track slow queries (> 1 second)
        if execution_time > 1.0:
            self.slow_queries.append({
                'query': query_name,
                'time': execution_time,
                'timestamp': datetime.utcnow()
            })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary with optimization suggestions."""
        summary = {}
        
        for query_name, times in self.query_times.items():
            if times:
                summary[query_name] = {
                    'avg_time': sum(times) / len(times),
                    'max_time': max(times),
                    'min_time': min(times),
                    'count': len(times)
                }
        
        summary['slow_queries'] = self.slow_queries[-10:]  # Last 10 slow queries
        
        return summary


# Global performance monitor instance
performance_monitor = QueryPerformanceMonitor()
