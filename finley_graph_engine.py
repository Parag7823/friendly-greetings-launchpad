 """
FinleyGraph Engine - Production Knowledge Graph for Financial Intelligence
Uses igraph (13-32x faster than networkx) + Supabase + Redis caching
"""

import asyncio
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

import igraph as ig
import structlog
from pydantic import BaseModel, Field
from supabase import Client

logger = structlog.get_logger(__name__)


class GraphNode(BaseModel):
    """Node from normalized_entities"""
    id: str
    entity_type: str
    canonical_name: str
    confidence_score: float
    platform_sources: List[str] = Field(default_factory=list)
    first_seen_at: datetime
    last_seen_at: datetime
    email: Optional[str] = None
    phone: Optional[str] = None
    bank_account: Optional[str] = None


class GraphEdge(BaseModel):
    """Edge from relationship_instances + enrichments"""
    id: str
    source_id: str
    target_id: str
    relationship_type: str
    confidence_score: float
    detection_method: str
    reasoning: Optional[str] = None
    created_at: datetime
    causal_strength: Optional[float] = None
    causal_direction: Optional[str] = None
    temporal_pattern_id: Optional[str] = None
    recurrence_score: Optional[float] = None


class GraphStats(BaseModel):
    """Graph metrics"""
    node_count: int
    edge_count: int
    avg_degree: float
    density: float
    connected_components: int
    build_time_seconds: float
    last_updated: datetime


class PathResult(BaseModel):
    """Path query result"""
    source_name: str
    target_name: str
    path_length: int
    path_nodes: List[str]
    path_edges: List[Dict[str, Any]]
    total_causal_strength: float
    confidence: float


class FinleyGraphEngine:
    """Production-grade knowledge graph using igraph + Supabase"""
    
    def __init__(self, supabase: Client, redis_url: Optional[str] = None):
        self.supabase = supabase
        self.redis_url = redis_url
        self.graph: Optional[ig.Graph] = None
        self.node_id_to_index: Dict[str, int] = {}
        self.index_to_node_id: Dict[int, str] = {}
        self.last_build_time: Optional[datetime] = None
    
    async def build_graph(self, user_id: str, force_rebuild: bool = False) -> GraphStats:
        """Build graph from Supabase tables"""
        start_time = datetime.now()
        
        if not force_rebuild and self.redis_url:
            cached = await self._load_from_cache(user_id)
            if cached:
                return cached
        
        logger.info("building_graph", user_id=user_id)
        
        # Fetch data
        nodes = await self._fetch_nodes(user_id)
        edges = await self._fetch_edges(user_id)
        
        # Build igraph
        self.graph = ig.Graph(directed=True)
        self.node_id_to_index = {}
        self.index_to_node_id = {}
        
        # Add nodes
        for idx, node in enumerate(nodes):
            self.graph.add_vertex(
                name=node.id,
                entity_type=node.entity_type,
                canonical_name=node.canonical_name,
                confidence_score=node.confidence_score,
                platform_sources=node.platform_sources,
                first_seen_at=node.first_seen_at.isoformat(),
                last_seen_at=node.last_seen_at.isoformat()
            )
            self.node_id_to_index[node.id] = idx
            self.index_to_node_id[idx] = node.id
        
        # Add edges
        edge_list = []
        edge_attrs = {
            'edge_id': [], 'relationship_type': [], 'confidence_score': [],
            'causal_strength': [], 'causal_direction': [], 'reasoning': []
        }
        
        for edge in edges:
            src_idx = self.node_id_to_index.get(edge.source_id)
            tgt_idx = self.node_id_to_index.get(edge.target_id)
            
            if src_idx is None or tgt_idx is None:
                continue
            
            edge_list.append((src_idx, tgt_idx))
            edge_attrs['edge_id'].append(edge.id)
            edge_attrs['relationship_type'].append(edge.relationship_type)
            edge_attrs['confidence_score'].append(edge.confidence_score)
            edge_attrs['causal_strength'].append(edge.causal_strength or 0.0)
            edge_attrs['causal_direction'].append(edge.causal_direction or 'none')
            edge_attrs['reasoning'].append(edge.reasoning or '')
        
        self.graph.add_edges(edge_list)
        for attr, values in edge_attrs.items():
            self.graph.es[attr] = values
        
        # Stats
        build_time = (datetime.now() - start_time).total_seconds()
        self.last_build_time = datetime.now()
        
        stats = GraphStats(
            node_count=self.graph.vcount(),
            edge_count=self.graph.ecount(),
            avg_degree=2 * self.graph.ecount() / max(1, self.graph.vcount()),
            density=self.graph.density(),
            connected_components=len(self.graph.connected_components(mode='weak')),
            build_time_seconds=build_time,
            last_updated=self.last_build_time
        )
        
        logger.info("graph_built", **stats.dict())
        
        if self.redis_url:
            await self._save_to_cache(user_id, stats)
        
        return stats
    
    async def _fetch_nodes(self, user_id: str) -> List[GraphNode]:
        """Fetch normalized_entities"""
        resp = self.supabase.table('normalized_entities').select(
            'id, entity_type, canonical_name, confidence_score, '
            'platform_sources, first_seen_at, last_seen_at, email, phone, bank_account'
        ).eq('user_id', user_id).execute()
        return [GraphNode(**row) for row in resp.data]
    
    async def _fetch_edges(self, user_id: str) -> List[GraphEdge]:
        """Fetch relationship_instances + enrichments"""
        # Get relationships
        rel_resp = self.supabase.table('relationship_instances').select(
            'id, source_event_id, target_event_id, relationship_type, '
            'confidence_score, detection_method, reasoning, created_at'
        ).eq('user_id', user_id).execute()
        
        relationships = rel_resp.data
        event_ids = set()
        for rel in relationships:
            event_ids.add(rel['source_event_id'])
            event_ids.add(rel['target_event_id'])
        
        # Map events to entities
        entity_map = await self._fetch_entity_mappings(user_id, list(event_ids))
        
        # Get causal enrichments
        causal_map = await self._fetch_causal_enrichments(user_id, [r['id'] for r in relationships])
        
        # Build edges
        edges = []
        for rel in relationships:
            src_entity = entity_map.get(rel['source_event_id'])
            tgt_entity = entity_map.get(rel['target_event_id'])
            
            if not src_entity or not tgt_entity:
                continue
            
            causal = causal_map.get(rel['id'], {})
            
            edges.append(GraphEdge(
                id=rel['id'],
                source_id=src_entity,
                target_id=tgt_entity,
                relationship_type=rel['relationship_type'],
                confidence_score=rel['confidence_score'],
                detection_method=rel['detection_method'],
                reasoning=rel.get('reasoning'),
                created_at=datetime.fromisoformat(rel['created_at'].replace('Z', '+00:00')),
                causal_strength=causal.get('causal_score'),
                causal_direction=causal.get('causal_direction')
            ))
        
        return edges
    
    async def _fetch_entity_mappings(self, user_id: str, event_ids: List[str]) -> Dict[str, str]:
        """Map event IDs to entity IDs"""
        if not event_ids:
            return {}
        resp = self.supabase.table('entity_matches').select(
            'source_row_id, normalized_entity_id'
        ).eq('user_id', user_id).in_('source_row_id', event_ids).execute()
        return {row['source_row_id']: row['normalized_entity_id'] for row in resp.data}
    
    async def _fetch_causal_enrichments(self, user_id: str, rel_ids: List[str]) -> Dict[str, Dict]:
        """Fetch causal_relationships"""
        if not rel_ids:
            return {}
        resp = self.supabase.table('causal_relationships').select(
            'relationship_id, causal_score, causal_direction'
        ).eq('user_id', user_id).in_('relationship_id', rel_ids).execute()
        return {row['relationship_id']: row for row in resp.data}
    
    async def _save_to_cache(self, user_id: str, stats: GraphStats):
        """Save to Redis"""
        try:
            import redis.asyncio as aioredis
            client = await aioredis.from_url(self.redis_url)
            data = pickle.dumps({
                'graph': self.graph,
                'node_id_to_index': self.node_id_to_index,
                'index_to_node_id': self.index_to_node_id,
                'stats': stats.dict(),
                'last_build_time': self.last_build_time
            })
            await client.setex(f"finley_graph:{user_id}", 3600, data)
            logger.info("graph_cached", user_id=user_id)
        except Exception as e:
            logger.error("cache_failed", error=str(e))
    
    async def _load_from_cache(self, user_id: str) -> Optional[GraphStats]:
        """Load from Redis"""
        try:
            import redis.asyncio as aioredis
            client = await aioredis.from_url(self.redis_url)
            data = await client.get(f"finley_graph:{user_id}")
            if not data:
                return None
            obj = pickle.loads(data)
            self.graph = obj['graph']
            self.node_id_to_index = obj['node_id_to_index']
            self.index_to_node_id = obj['index_to_node_id']
            self.last_build_time = obj['last_build_time']
            return GraphStats(**obj['stats'])
        except:
            return None
    
    def find_path(self, source_id: str, target_id: str, max_len: int = 5) -> Optional[PathResult]:
        """Find shortest path using igraph"""
        if not self.graph:
            raise ValueError("Graph not built")
        
        src_idx = self.node_id_to_index.get(source_id)
        tgt_idx = self.node_id_to_index.get(target_id)
        
        if src_idx is None or tgt_idx is None:
            return None
        
        paths = self.graph.get_shortest_paths(src_idx, tgt_idx, mode='out', output='epath')
        
        if not paths or not paths[0] or len(paths[0]) > max_len:
            return None
        
        edge_indices = paths[0]
        path_nodes = [self.index_to_node_id[src_idx]]
        path_edges = []
        total_causal = 0.0
        total_conf = 0.0
        
        for eidx in edge_indices:
            e = self.graph.es[eidx]
            path_nodes.append(self.index_to_node_id[e.target])
            path_edges.append({
                'relationship_type': e['relationship_type'],
                'causal_strength': e['causal_strength'],
                'confidence_score': e['confidence_score'],
                'reasoning': e['reasoning']
            })
            total_causal += e['causal_strength']
            total_conf += e['confidence_score']
        
        return PathResult(
            source_name=self.graph.vs[src_idx]['canonical_name'],
            target_name=self.graph.vs[tgt_idx]['canonical_name'],
            path_length=len(edge_indices),
            path_nodes=[self.graph.vs[self.node_id_to_index[nid]]['canonical_name'] for nid in path_nodes],
            path_edges=path_edges,
            total_causal_strength=total_causal,
            confidence=total_conf / len(edge_indices) if edge_indices else 0.0
        )
    
    def get_entity_importance(self, algorithm: str = 'pagerank') -> Dict[str, float]:
        """Calculate entity importance"""
        if not self.graph:
            raise ValueError("Graph not built")
        
        if algorithm == 'pagerank':
            weights = [e['causal_strength'] for e in self.graph.es]
            scores = self.graph.pagerank(weights=weights, directed=True)
        elif algorithm == 'betweenness':
            scores = self.graph.betweenness(directed=True)
        else:
            scores = self.graph.closeness(mode='out')
        
        return {self.index_to_node_id[i]: s for i, s in enumerate(scores)}
    
    def detect_communities(self) -> Dict[str, int]:
        """Detect entity clusters using Louvain"""
        if not self.graph:
            raise ValueError("Graph not built")
        
        communities = self.graph.community_multilevel()
        result = {}
        for cid, members in enumerate(communities):
            for vidx in members:
                result[self.index_to_node_id[vidx]] = cid
        return result
    
    async def incremental_update(self, user_id: str, since: datetime) -> Dict[str, int]:
        """Incremental update - only fetch new data since timestamp"""
        if not self.graph:
            logger.warning("graph_not_built_forcing_full_rebuild")
            await self.build_graph(user_id)
            return {'nodes_added': self.graph.vcount(), 'edges_added': self.graph.ecount()}
        
        logger.info("incremental_update", user_id=user_id, since=since.isoformat())
        
        # Fetch new entities
        resp = self.supabase.table('normalized_entities').select(
            'id, entity_type, canonical_name, confidence_score, '
            'platform_sources, first_seen_at, last_seen_at, email, phone, bank_account'
        ).eq('user_id', user_id).gte('last_seen_at', since.isoformat()).execute()
        
        nodes_added = 0
        for row in resp.data:
            node = GraphNode(**row)
            if node.id not in self.node_id_to_index:
                idx = self.graph.vcount()
                self.graph.add_vertex(
                    name=node.id,
                    entity_type=node.entity_type,
                    canonical_name=node.canonical_name,
                    confidence_score=node.confidence_score,
                    platform_sources=node.platform_sources,
                    first_seen_at=node.first_seen_at.isoformat(),
                    last_seen_at=node.last_seen_at.isoformat()
                )
                self.node_id_to_index[node.id] = idx
                self.index_to_node_id[idx] = node.id
                nodes_added += 1
        
        # Fetch new relationships
        rel_resp = self.supabase.table('relationship_instances').select(
            'id, source_event_id, target_event_id, relationship_type, '
            'confidence_score, detection_method, reasoning, created_at'
        ).eq('user_id', user_id).gte('created_at', since.isoformat()).execute()
        
        # Map to entities
        event_ids = set()
        for rel in rel_resp.data:
            event_ids.add(rel['source_event_id'])
            event_ids.add(rel['target_event_id'])
        
        entity_map = await self._fetch_entity_mappings(user_id, list(event_ids))
        causal_map = await self._fetch_causal_enrichments(user_id, [r['id'] for r in rel_resp.data])
        
        edges_added = 0
        for rel in rel_resp.data:
            src_entity = entity_map.get(rel['source_event_id'])
            tgt_entity = entity_map.get(rel['target_event_id'])
            
            if not src_entity or not tgt_entity:
                continue
            
            src_idx = self.node_id_to_index.get(src_entity)
            tgt_idx = self.node_id_to_index.get(tgt_entity)
            
            if src_idx is not None and tgt_idx is not None:
                causal = causal_map.get(rel['id'], {})
                self.graph.add_edge(
                    src_idx, tgt_idx,
                    edge_id=rel['id'],
                    relationship_type=rel['relationship_type'],
                    confidence_score=rel['confidence_score'],
                    causal_strength=causal.get('causal_score', 0.0),
                    causal_direction=causal.get('causal_direction', 'none'),
                    reasoning=rel.get('reasoning', '')
                )
                edges_added += 1
        
        self.last_build_time = datetime.now()
        logger.info("incremental_update_complete", nodes_added=nodes_added, edges_added=edges_added)
        
        # Update cache
        if self.redis_url and (nodes_added > 0 or edges_added > 0):
            stats = GraphStats(
                node_count=self.graph.vcount(),
                edge_count=self.graph.ecount(),
                avg_degree=2 * self.graph.ecount() / max(1, self.graph.vcount()),
                density=self.graph.density(),
                connected_components=len(self.graph.connected_components(mode='weak')),
                build_time_seconds=0.0,
                last_updated=self.last_build_time
            )
            await self._save_to_cache(user_id, stats)
        
        return {'nodes_added': nodes_added, 'edges_added': edges_added}
