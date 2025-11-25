"""
FinleyGraph FastAPI Integration
Production-ready graph query endpoints
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from supabase import Client
import structlog

from finley_graph_engine import FinleyGraphEngine, PathResult, GraphStats
from core_infrastructure.supabase_client import get_supabase_client

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/graph", tags=["Knowledge Graph"])


# ============================================================================
# FIX #33: DEPENDENCY INJECTION FOR SUPABASE CLIENT
# ============================================================================

async def get_supabase_client_dependency() -> Client:
    """
    FastAPI dependency function for Supabase client injection.
    
    FIX #33: Provides proper dependency injection for all endpoints.
    Replaces broken Depends() with no argument.
    
    Returns:
        Pooled Supabase client instance
    
    Usage in endpoints:
        @router.post("/endpoint")
        async def my_endpoint(supabase: Client = Depends(get_supabase_client_dependency)):
            ...
    """
    try:
        client = get_supabase_client()
        if client is None:
            raise HTTPException(status_code=503, detail="Supabase client not available")
        return client
    except Exception as e:
        logger.error("supabase_dependency_injection_failed", error=str(e))
        raise HTTPException(status_code=503, detail="Database service unavailable")


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class GraphBuildRequest(BaseModel):
    user_id: str
    force_rebuild: bool = False


class GraphBuildResponse(BaseModel):
    status: str
    stats: GraphStats
    message: str


class PathQueryRequest(BaseModel):
    user_id: str
    source_entity_id: str
    target_entity_id: str
    max_length: int = Field(default=5, ge=1, le=10)


class PathQueryResponse(BaseModel):
    """
    Enhanced path query response with all 9 intelligence layers.
    
    Includes temporal patterns, seasonal insights, fraud risk, root causes, and predictions.
    """
    status: str
    path: Optional[PathResult]
    message: str
    # Intelligence insights
    temporal_insights: Optional[Dict[str, Any]] = None  # Recurring patterns, frequency
    seasonal_insights: Optional[Dict[str, Any]] = None  # Seasonal cycles, months
    fraud_risk: Optional[float] = None  # Duplicate/fraud confidence (0-1)
    root_causes: Optional[List[str]] = None  # Why this path exists
    predictions: Optional[Dict[str, Any]] = None  # Future connections


class ImportanceQueryRequest(BaseModel):
    user_id: str
    algorithm: str = Field(default="pagerank", regex="^(pagerank|betweenness|closeness)$")
    top_n: int = Field(default=20, ge=1, le=100)


class ImportanceQueryResponse(BaseModel):
    status: str
    top_entities: List[Dict[str, float]]
    algorithm: str


class CommunityQueryRequest(BaseModel):
    user_id: str


class CommunityQueryResponse(BaseModel):
    status: str
    communities: Dict[str, int]
    community_count: int


class IncrementalUpdateRequest(BaseModel):
    user_id: str
    since_minutes: int = Field(default=60, ge=1, le=1440)


class IncrementalUpdateResponse(BaseModel):
    status: str
    nodes_added: int
    edges_added: int
    message: str


# ============================================================================
# NEW INTELLIGENCE QUERY MODELS (Phase 2)
# ============================================================================

class TemporalPatternQueryRequest(BaseModel):
    """Query for recurring temporal patterns between entities"""
    user_id: str
    source_entity_id: str
    target_entity_id: str
    min_recurrence_score: float = Field(default=0.5, ge=0.0, le=1.0)


class TemporalPatternQueryResponse(BaseModel):
    """Response with temporal pattern insights"""
    status: str
    patterns: List[Dict[str, Any]]  # recurrence_frequency, recurrence_score, next_occurrence
    message: str


class SeasonalCycleQueryRequest(BaseModel):
    """Query for seasonal patterns"""
    user_id: str
    source_entity_id: str
    target_entity_id: str
    min_seasonal_strength: float = Field(default=0.5, ge=0.0, le=1.0)


class SeasonalCycleQueryResponse(BaseModel):
    """Response with seasonal cycle insights"""
    status: str
    seasonal_cycles: List[Dict[str, Any]]  # seasonal_months, seasonal_strength
    message: str


class FraudDetectionQueryRequest(BaseModel):
    """Query for duplicate/fraudulent transactions"""
    user_id: str
    source_entity_id: str
    target_entity_id: str
    min_fraud_confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class FraudDetectionQueryResponse(BaseModel):
    """Response with fraud detection insights"""
    status: str
    fraud_alerts: List[Dict[str, Any]]  # is_duplicate, duplicate_confidence, reason
    fraud_risk_score: float
    message: str


class RootCauseQueryRequest(BaseModel):
    """Query for root cause analysis"""
    user_id: str
    source_entity_id: str
    target_entity_id: str


class RootCauseQueryResponse(BaseModel):
    """Response with root cause analysis"""
    status: str
    root_causes: List[str]
    causal_chain: List[Dict[str, Any]]  # Path of causality
    message: str


class PredictionQueryRequest(BaseModel):
    """Query for predicted future relationships"""
    user_id: str
    source_entity_id: str
    target_entity_id: str
    min_prediction_confidence: float = Field(default=0.6, ge=0.0, le=1.0)


class PredictionQueryResponse(BaseModel):
    """Response with prediction insights"""
    status: str
    predictions: List[Dict[str, Any]]  # prediction_confidence, prediction_reason, next_entity
    message: str


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def get_graph_engine(
    user_id: str,
    supabase: Client,
    redis_url: Optional[str] = None
) -> FinleyGraphEngine:
    """
    Create and return a FinleyGraphEngine instance for the given user.
    
    FIX #25: Attempts to load from Redis cache first.
    If cache miss, returns empty engine ready for building.
    
    Args:
        user_id: User ID to load graph for
        supabase: Supabase client
        redis_url: Optional Redis URL for caching
        
    Returns:
        FinleyGraphEngine instance with graph loaded from cache if available
    """
    engine = FinleyGraphEngine(supabase, redis_url)
    
    # FIX #25: Attempt to load from cache
    if redis_url:
        cached_stats = await engine._load_from_cache(user_id)
        if cached_stats:
            logger.info("graph_loaded_from_cache", user_id=user_id, 
                       nodes=cached_stats.node_count, edges=cached_stats.edge_count)
            return engine
    
    # Cache miss - return empty engine for building
    return engine


# ============================================================================
# API ENDPOINTS
# ============================================================================
# FIX #1: Removed global _graph_engines dict and helper functions.
# Each request creates a new FinleyGraphEngine instance.
# Redis caching is handled internally by FinleyGraphEngine._load_from_cache()

@router.post("/build", response_model=GraphBuildResponse)
async def build_graph(
    request: GraphBuildRequest,
    supabase: Client = Depends(get_supabase_client_dependency),
    redis_url: Optional[str] = None
):
    """
    Build knowledge graph for user from Supabase tables.
    
    - Pulls entities from normalized_entities
    - Pulls relationships from relationship_instances
    - Enriches with causal/temporal/semantic metadata
    - Caches in Redis for fast reload
    """
    try:
        # FIX #1: Create new engine instance per request (no global state)
        engine = FinleyGraphEngine(supabase, redis_url)
        stats = await engine.build_graph(request.user_id, request.force_rebuild)
        
        return GraphBuildResponse(
            status="success",
            stats=stats,
            message=f"Graph built with {stats.node_count} nodes and {stats.edge_count} edges"
        )
    
    except Exception as e:
        logger.error("graph_build_failed", error=str(e), user_id=request.user_id)
        raise HTTPException(status_code=500, detail=f"Graph build failed: {str(e)}")


@router.post("/update", response_model=IncrementalUpdateResponse)
async def incremental_update(
    request: IncrementalUpdateRequest,
    supabase: Client = Depends(get_supabase_client_dependency),
    redis_url: Optional[str] = None
):
    """
    Incrementally update graph with new data since last N minutes.
    
    - Fetches only new/modified entities and relationships
    - Updates existing graph without full rebuild
    - Much faster than full rebuild for small updates
    """
    try:
        # FIX #1: Create new engine instance per request (no global state)
        engine = FinleyGraphEngine(supabase, redis_url)
        
        since = datetime.now() - timedelta(minutes=request.since_minutes)
        result = await engine.incremental_update(request.user_id, since)
        
        return IncrementalUpdateResponse(
            status="success",
            nodes_added=result['nodes_added'],
            edges_added=result['edges_added'],
            message=f"Added {result['nodes_added']} nodes and {result['edges_added']} edges"
        )
    
    except Exception as e:
        logger.error("incremental_update_failed", error=str(e), user_id=request.user_id)
        raise HTTPException(status_code=500, detail=f"Incremental update failed: {str(e)}")


@router.post("/query/path", response_model=PathQueryResponse)
async def query_path(
    request: PathQueryRequest,
    supabase: Client = Depends(get_supabase_client_dependency),
    redis_url: Optional[str] = None
):
    """
    Find shortest path between two entities.
    
    Use cases:
    - "How did Invoice A lead to Payment B?"
    - "What's the connection between Vendor X and Account Y?"
    - "Trace the flow from expense to revenue"
    """
    try:
        # FIX #26: Use consistent Pattern 1 initialization
        engine = FinleyGraphEngine(supabase, redis_url)
        
        # Attempt to load from cache first
        if redis_url:
            cached_stats = await engine._load_from_cache(request.user_id)
            if not cached_stats:
                raise HTTPException(status_code=400, detail="Graph not built. Call /build first.")
        else:
            if not engine.graph:
                raise HTTPException(status_code=400, detail="Graph not built. Call /build first.")
        
        path = engine.find_path(
            request.source_entity_id,
            request.target_entity_id,
            request.max_length
        )
        
        if not path:
            return PathQueryResponse(
                status="success",
                path=None,
                message="No path found between entities"
            )
        
        return PathQueryResponse(
            status="success",
            path=path,
            message=f"Found path with {path.path_length} steps"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("path_query_failed", error=str(e), user_id=request.user_id)
        raise HTTPException(status_code=500, detail=f"Path query failed: {str(e)}")


@router.post("/query/importance", response_model=ImportanceQueryResponse)
async def query_importance(
    request: ImportanceQueryRequest,
    supabase: Client = Depends(get_supabase_client_dependency),
    redis_url: Optional[str] = None
):
    """
    Calculate entity importance using graph algorithms.
    
    Algorithms:
    - pagerank: Weighted by causal strength (most influential entities)
    - betweenness: Entities that connect many others (key intermediaries)
    - closeness: Entities close to all others (central hubs)
    
    Use cases:
    - "Which vendors are most critical to our operations?"
    - "What accounts are central to cash flow?"
    - "Find key entities in the financial network"
    """
    try:
        # FIX #1: Create new engine instance per request (no global state)
        engine = FinleyGraphEngine(supabase, redis_url)
        
        if not engine.graph:
            raise HTTPException(status_code=400, detail="Graph not built. Call /build first.")
        
        scores = engine.get_entity_importance(request.algorithm)
        
        # Sort and get top N
        sorted_entities = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:request.top_n]
        
        # Get entity names
        top_entities = []
        for entity_id, score in sorted_entities:
            idx = engine.node_id_to_index.get(entity_id)
            if idx is not None:
                name = engine.graph.vs[idx]['canonical_name']
                top_entities.append({
                    'entity_id': entity_id,
                    'name': name,
                    'score': score
                })
        
        return ImportanceQueryResponse(
            status="success",
            top_entities=top_entities,
            algorithm=request.algorithm
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("importance_query_failed", error=str(e), user_id=request.user_id)
        raise HTTPException(status_code=500, detail=f"Importance query failed: {str(e)}")


@router.post("/query/communities", response_model=CommunityQueryResponse)
async def query_communities(
    request: CommunityQueryRequest,
    supabase: Client = Depends(get_supabase_client_dependency),
    redis_url: Optional[str] = None
):
    """
    Detect entity clusters/communities using Louvain algorithm.
    
    Use cases:
    - "Group related vendors/customers"
    - "Find operational silos"
    - "Identify business units from transaction patterns"
    - "Discover hidden entity relationships"
    """
    try:
        # FIX #1: Create new engine instance per request (no global state)
        engine = FinleyGraphEngine(supabase, redis_url)
        
        if not engine.graph:
            raise HTTPException(status_code=400, detail="Graph not built. Call /build first.")
        
        communities = engine.detect_communities()
        community_count = len(set(communities.values()))
        
        return CommunityQueryResponse(
            status="success",
            communities=communities,
            community_count=community_count
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("community_query_failed", error=str(e), user_id=request.user_id)
        raise HTTPException(status_code=500, detail=f"Community query failed: {str(e)}")


@router.get("/stats/{user_id}", response_model=GraphStats)
async def get_graph_stats(
    user_id: str,
    supabase: Client = Depends(get_supabase_client_dependency),
    redis_url: Optional[str] = None
):
    """Get current graph statistics for user"""
    try:
        # FIX #1: Create new engine instance per request (no global state)
        engine = FinleyGraphEngine(supabase, redis_url)
        
        if not engine.graph:
            raise HTTPException(status_code=400, detail="Graph not built. Call /build first.")
        
        return GraphStats(
            node_count=engine.graph.vcount(),
            edge_count=engine.graph.ecount(),
            avg_degree=2 * engine.graph.ecount() / max(1, engine.graph.vcount()),
            density=engine.graph.density(),
            connected_components=len(engine.graph.connected_components(mode='weak')),
            build_time_seconds=0.0,
            last_updated=engine.last_build_time or datetime.now()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("stats_query_failed", error=str(e), user_id=user_id)
        raise HTTPException(status_code=500, detail=f"Stats query failed: {str(e)}")


@router.post("/query/temporal-patterns", response_model=TemporalPatternQueryResponse)
async def query_temporal_patterns(
    request: TemporalPatternQueryRequest,
    supabase: Client = Depends(get_supabase_client_dependency),
    redis_url: Optional[str] = None
):
    """
    Query for recurring temporal patterns between entities.
    
    Returns:
    - Recurrence frequency (daily, weekly, monthly, yearly)
    - Recurrence score (0-1)
    - Next predicted occurrence
    - Pattern history
    """
    try:
        # FIX #1: Create new engine instance per request (no global state)
        engine = FinleyGraphEngine(supabase, redis_url)
        
        if not engine.graph:
            raise HTTPException(status_code=400, detail="Graph not built. Call /build first.")
        
        # Find path between entities
        path = engine.find_path(request.source_entity_id, request.target_entity_id)
        if not path:
            return TemporalPatternQueryResponse(
                status="success",
                patterns=[],
                message="No path found between entities"
            )
        
        # Extract temporal patterns from edges
        patterns = []
        for edge_data in path.path_edges:
            if edge_data.get('recurrence_frequency') and edge_data.get('recurrence_frequency') != 'none':
                recurrence_score = edge_data.get('recurrence_score', 0.0)
                if recurrence_score >= request.min_recurrence_score:
                    patterns.append({
                        'relationship_type': edge_data.get('relationship_type'),
                        'recurrence_frequency': edge_data.get('recurrence_frequency'),
                        'recurrence_score': recurrence_score,
                        'next_predicted_occurrence': edge_data.get('next_predicted_occurrence'),
                        'confidence': edge_data.get('confidence_score', 0.0)
                    })
        
        return TemporalPatternQueryResponse(
            status="success",
            patterns=patterns,
            message=f"Found {len(patterns)} temporal patterns"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("temporal_pattern_query_failed", error=str(e), user_id=request.user_id)
        raise HTTPException(status_code=500, detail=f"Temporal pattern query failed: {str(e)}")


@router.post("/query/seasonal-cycles", response_model=SeasonalCycleQueryResponse)
async def query_seasonal_cycles(
    request: SeasonalCycleQueryRequest,
    supabase: Client = Depends(get_supabase_client_dependency),
    redis_url: Optional[str] = None
):
    """
    Query for seasonal patterns between entities.
    
    Returns:
    - Seasonal months (1-12)
    - Seasonal strength (0-1)
    - Seasonal confidence
    """
    try:
        # FIX #1: Create new engine instance per request (no global state)
        engine = FinleyGraphEngine(supabase, redis_url)
        
        if not engine.graph:
            raise HTTPException(status_code=400, detail="Graph not built. Call /build first.")
        
        # Find path between entities
        path = engine.find_path(request.source_entity_id, request.target_entity_id)
        if not path:
            return SeasonalCycleQueryResponse(
                status="success",
                seasonal_cycles=[],
                message="No path found between entities"
            )
        
        # Extract seasonal patterns from edges
        cycles = []
        for edge_data in path.path_edges:
            if edge_data.get('seasonal_months'):
                seasonal_strength = edge_data.get('seasonal_strength', 0.0)
                if seasonal_strength >= request.min_seasonal_strength:
                    cycles.append({
                        'relationship_type': edge_data.get('relationship_type'),
                        'seasonal_months': edge_data.get('seasonal_months'),
                        'seasonal_strength': seasonal_strength,
                        'confidence': edge_data.get('confidence_score', 0.0)
                    })
        
        return SeasonalCycleQueryResponse(
            status="success",
            seasonal_cycles=cycles,
            message=f"Found {len(cycles)} seasonal cycles"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("seasonal_cycle_query_failed", error=str(e), user_id=request.user_id)
        raise HTTPException(status_code=500, detail=f"Seasonal cycle query failed: {str(e)}")


@router.post("/query/fraud-detection", response_model=FraudDetectionQueryResponse)
async def query_fraud_detection(
    request: FraudDetectionQueryRequest,
    supabase: Client = Depends(get_supabase_client_dependency),
    redis_url: Optional[str] = None
):
    """
    Query for duplicate/fraudulent transactions between entities.
    
    Returns:
    - Fraud alerts with duplicate confidence
    - Overall fraud risk score
    - Duplicate transaction details
    """
    try:
        # FIX #1: Create new engine instance per request (no global state)
        engine = FinleyGraphEngine(supabase, redis_url)
        
        if not engine.graph:
            raise HTTPException(status_code=400, detail="Graph not built. Call /build first.")
        
        # Find path between entities
        path = engine.find_path(request.source_entity_id, request.target_entity_id)
        if not path:
            return FraudDetectionQueryResponse(
                status="success",
                fraud_alerts=[],
                fraud_risk_score=0.0,
                message="No path found between entities"
            )
        
        # Extract fraud indicators from edges
        fraud_alerts = []
        total_fraud_score = 0.0
        for edge_data in path.path_edges:
            if edge_data.get('is_duplicate'):
                duplicate_confidence = edge_data.get('duplicate_confidence', 0.0)
                if duplicate_confidence >= request.min_fraud_confidence:
                    fraud_alerts.append({
                        'relationship_type': edge_data.get('relationship_type'),
                        'is_duplicate': True,
                        'duplicate_confidence': duplicate_confidence,
                        'reasoning': edge_data.get('reasoning', 'Duplicate detected')
                    })
                    total_fraud_score += duplicate_confidence
        
        fraud_risk_score = total_fraud_score / max(1, len(path.path_edges))
        
        return FraudDetectionQueryResponse(
            status="success",
            fraud_alerts=fraud_alerts,
            fraud_risk_score=fraud_risk_score,
            message=f"Found {len(fraud_alerts)} fraud alerts with risk score {fraud_risk_score:.2f}"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("fraud_detection_query_failed", error=str(e), user_id=request.user_id)
        raise HTTPException(status_code=500, detail=f"Fraud detection query failed: {str(e)}")


@router.post("/query/root-causes", response_model=RootCauseQueryResponse)
async def query_root_causes(
    request: RootCauseQueryRequest,
    supabase: Client = Depends(get_supabase_client_dependency),
    redis_url: Optional[str] = None
):
    """
    Query for root cause analysis between entities.
    
    Returns:
    - Root cause analysis text
    - Causal chain (path of causality)
    - Causal strength scores
    """
    try:
        # FIX #1: Create new engine instance per request (no global state)
        engine = FinleyGraphEngine(supabase, redis_url)
        
        if not engine.graph:
            raise HTTPException(status_code=400, detail="Graph not built. Call /build first.")
        
        # Find path between entities
        path = engine.find_path(request.source_entity_id, request.target_entity_id)
        if not path:
            return RootCauseQueryResponse(
                status="success",
                root_causes=[],
                causal_chain=[],
                message="No path found between entities"
            )
        
        # Extract root cause analysis
        root_causes = []
        causal_chain = []
        for i, edge_data in enumerate(path.path_edges):
            if edge_data.get('root_cause_analysis'):
                root_causes.append(edge_data['root_cause_analysis'])
            
            causal_chain.append({
                'step': i + 1,
                'relationship_type': edge_data.get('relationship_type'),
                'causal_strength': edge_data.get('causal_strength', 0.0),
                'causal_direction': edge_data.get('causal_direction', 'unknown'),
                'reasoning': edge_data.get('reasoning', '')
            })
        
        return RootCauseQueryResponse(
            status="success",
            root_causes=root_causes,
            causal_chain=causal_chain,
            message=f"Found {len(root_causes)} root causes in causal chain of {len(causal_chain)} steps"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("root_cause_query_failed", error=str(e), user_id=request.user_id)
        raise HTTPException(status_code=500, detail=f"Root cause query failed: {str(e)}")


@router.post("/query/predictions", response_model=PredictionQueryResponse)
async def query_predictions(
    request: PredictionQueryRequest,
    supabase: Client = Depends(get_supabase_client_dependency),
    redis_url: Optional[str] = None
):
    """
    Query for predicted future relationships between entities.
    
    Returns:
    - Prediction confidence scores
    - Prediction reasons
    - Next predicted entities/events
    """
    try:
        # FIX #1: Create new engine instance per request (no global state)
        engine = FinleyGraphEngine(supabase, redis_url)
        
        if not engine.graph:
            raise HTTPException(status_code=400, detail="Graph not built. Call /build first.")
        
        # Find path between entities
        path = engine.find_path(request.source_entity_id, request.target_entity_id)
        if not path:
            return PredictionQueryResponse(
                status="success",
                predictions=[],
                message="No path found between entities"
            )
        
        # Extract predictions from edges
        predictions = []
        for edge_data in path.path_edges:
            if edge_data.get('prediction_confidence'):
                prediction_confidence = edge_data.get('prediction_confidence', 0.0)
                if prediction_confidence >= request.min_prediction_confidence:
                    predictions.append({
                        'relationship_type': edge_data.get('relationship_type'),
                        'prediction_confidence': prediction_confidence,
                        'prediction_reason': edge_data.get('prediction_reason', 'Pattern-based prediction'),
                        'next_predicted_occurrence': edge_data.get('next_predicted_occurrence'),
                        'confidence': edge_data.get('confidence_score', 0.0)
                    })
        
        return PredictionQueryResponse(
            status="success",
            predictions=predictions,
            message=f"Found {len(predictions)} predictions"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("prediction_query_failed", error=str(e), user_id=request.user_id)
        raise HTTPException(status_code=500, detail=f"Prediction query failed: {str(e)}")


@router.delete("/cache/{user_id}")
async def clear_cache(user_id: str, redis_url: Optional[str] = None):
    """
    Clear cached graph from Redis for user (force rebuild on next request).
    
    FIX #1: No local cache to clear. This endpoint clears Redis cache only.
    """
    try:
        if redis_url:
            # Clear Redis cache for this user
            engine = FinleyGraphEngine(None, redis_url)
            await engine._clear_redis_cache(user_id)
            logger.info("redis_cache_cleared", user_id=user_id)
        return {"status": "success", "message": f"Cache cleared for user {user_id}"}
    except Exception as e:
        logger.warning("cache_clear_failed", user_id=user_id, error=str(e))
        return {"status": "warning", "message": f"Cache clear partially failed: {str(e)}"}
