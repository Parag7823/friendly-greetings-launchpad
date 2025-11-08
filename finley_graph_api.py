"""
FinleyGraph FastAPI Integration
Production-ready graph query endpoints
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from supabase import Client
import structlog

from finley_graph_engine import FinleyGraphEngine, PathResult, GraphStats

logger = structlog.get_logger(__name__)

# Global graph engine instance per user (cached)
_graph_engines: Dict[str, FinleyGraphEngine] = {}

router = APIRouter(prefix="/api/v1/graph", tags=["Knowledge Graph"])


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
    status: str
    path: Optional[PathResult]
    message: str


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
# HELPER FUNCTIONS
# ============================================================================

async def get_graph_engine(user_id: str, supabase: Client, redis_url: Optional[str] = None) -> FinleyGraphEngine:
    """Get or create graph engine for user"""
    if user_id not in _graph_engines:
        _graph_engines[user_id] = FinleyGraphEngine(supabase, redis_url)
    return _graph_engines[user_id]


def clear_graph_cache(user_id: str):
    """Clear cached graph engine for user"""
    if user_id in _graph_engines:
        del _graph_engines[user_id]
        logger.info("graph_cache_cleared", user_id=user_id)


# ============================================================================
# API ENDPOINTS
# ============================================================================

@router.post("/build", response_model=GraphBuildResponse)
async def build_graph(
    request: GraphBuildRequest,
    supabase: Client = Depends(),
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
        engine = await get_graph_engine(request.user_id, supabase, redis_url)
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
    supabase: Client = Depends(),
    redis_url: Optional[str] = None
):
    """
    Incrementally update graph with new data since last N minutes.
    
    - Fetches only new/modified entities and relationships
    - Updates existing graph without full rebuild
    - Much faster than full rebuild for small updates
    """
    try:
        engine = await get_graph_engine(request.user_id, supabase, redis_url)
        
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
    supabase: Client = Depends(),
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
        engine = await get_graph_engine(request.user_id, supabase, redis_url)
        
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
    supabase: Client = Depends(),
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
        engine = await get_graph_engine(request.user_id, supabase, redis_url)
        
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
    supabase: Client = Depends(),
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
        engine = await get_graph_engine(request.user_id, supabase, redis_url)
        
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
    supabase: Client = Depends(),
    redis_url: Optional[str] = None
):
    """Get current graph statistics for user"""
    try:
        engine = await get_graph_engine(user_id, supabase, redis_url)
        
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


@router.delete("/cache/{user_id}")
async def clear_cache(user_id: str):
    """Clear cached graph engine for user (force rebuild on next request)"""
    clear_graph_cache(user_id)
    return {"status": "success", "message": f"Cache cleared for user {user_id}"}
