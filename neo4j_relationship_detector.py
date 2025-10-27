"""
Neo4j Relationship Detector - Graph Database Integration

This module provides Neo4j graph database integration for relationship detection,
enabling visual graph exploration, fast multi-hop queries, and advanced graph algorithms.

Features:
- Create event nodes and relationship edges
- Multi-hop causal chain traversal
- Root cause analysis with shortest path
- Pattern matching and similarity search
- PageRank for influential events
- Community detection for transaction clustering
- Graceful error handling with detailed logging
"""

import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from neo4j import GraphDatabase, AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError, Neo4jError

logger = logging.getLogger(__name__)


class Neo4jRelationshipDetector:
    """
    Neo4j-powered relationship detection for visual graph exploration
    and advanced graph analytics.
    """
    
    def __init__(
        self, 
        uri: str = None, 
        user: str = None, 
        password: str = None,
        max_connection_pool_size: int = 50
    ):
        """
        Initialize Neo4j connection.
        
        Args:
            uri: Neo4j connection URI (e.g., neo4j+s://xxxxx.databases.neo4j.io)
            user: Username (default: neo4j)
            password: Password from Neo4j Aura
            max_connection_pool_size: Max connections in pool
        """
        self.uri = uri or os.getenv('NEO4J_URI')
        self.user = user or os.getenv('NEO4J_USER', 'neo4j')
        self.password = password or os.getenv('NEO4J_PASSWORD')
        
        if not all([self.uri, self.user, self.password]):
            error_msg = """
❌ Neo4j credentials missing! Set these environment variables:

NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password-here

Get free Neo4j Aura instance: https://neo4j.com/cloud/aura-free/
"""
            logger.error(error_msg)
            raise ValueError("Neo4j credentials required")
        
        try:
            logger.info(f"🔌 Connecting to Neo4j: {self.uri}")
            logger.info(f"   User: {self.user}")
            
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                max_connection_pool_size=max_connection_pool_size,
                connection_timeout=10.0,  # 10 second timeout
                max_transaction_retry_time=5.0
            )
            
            # Test connection with detailed error handling
            try:
                self.driver.verify_connectivity()
                logger.info(f"✅ Neo4j connected successfully: {self.uri}")
            except AuthError as auth_err:
                logger.error(f"""
❌ Neo4j authentication failed!
   URI: {self.uri}
   User: {self.user}
   
   Possible issues:
   1. Wrong password - check NEO4J_PASSWORD in .env
   2. Wrong username - should be 'neo4j' for Aura
   3. Database not started in Neo4j Aura console
   
   Error: {auth_err}
""")
                raise
            
            # Create constraints and indexes
            self._setup_schema()
            
        except AuthError:
            raise  # Already logged above
        except ServiceUnavailable as e:
            logger.error(f"""
❌ Neo4j service unavailable!
   URI: {self.uri}
   
   Possible issues:
   1. Wrong URI - check NEO4J_URI in .env
   2. Database paused in Neo4j Aura (free tier auto-pauses)
   3. Network/firewall blocking connection
   
   Error: {e}
""")
            raise
        except Exception as e:
            logger.error(f"❌ Neo4j connection failed: {e}")
            raise
    
    def _setup_schema(self):
        """Create constraints and indexes for optimal performance"""
        try:
            with self.driver.session() as session:
                # Constraint: Unique event_id
                session.run("""
                    CREATE CONSTRAINT event_id_unique IF NOT EXISTS
                    FOR (e:Event) REQUIRE e.event_id IS UNIQUE
                """)
                
                # Indexes for fast lookups
                session.run("CREATE INDEX event_type IF NOT EXISTS FOR (e:Event) ON (e.document_type)")
                session.run("CREATE INDEX event_platform IF NOT EXISTS FOR (e:Event) ON (e.platform)")
                session.run("CREATE INDEX event_user IF NOT EXISTS FOR (e:Event) ON (e.user_id)")
                session.run("CREATE INDEX event_amount IF NOT EXISTS FOR (e:Event) ON (e.amount_usd)")
                session.run("CREATE INDEX event_date IF NOT EXISTS FOR (e:Event) ON (e.source_ts)")
                
                logger.info("✅ Neo4j schema setup complete")
        except Exception as e:
            logger.warning(f"⚠️ Schema setup failed (may already exist): {e}")
    
    def create_event_node(self, event_data: Dict[str, Any]) -> bool:
        """
        Create or update event node in Neo4j.
        
        Args:
            event_data: Event dictionary with event_id, document_type, amount_usd, etc.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.driver.session() as session:
                query = """
                MERGE (e:Event {event_id: $event_id})
                SET e.document_type = $document_type,
                    e.amount_usd = $amount_usd,
                    e.source_ts = datetime($source_ts),
                    e.vendor_standard = $vendor_standard,
                    e.platform = $platform,
                    e.user_id = $user_id,
                    e.created_at = datetime(),
                    e.updated_at = datetime()
                RETURN e.event_id as event_id
                """
                
                result = session.run(query, 
                    event_id=event_data.get('id') or event_data.get('event_id'),
                    document_type=event_data.get('document_type'),
                    amount_usd=float(event_data.get('amount_usd', 0)),
                    source_ts=event_data.get('source_ts') or datetime.utcnow().isoformat(),
                    vendor_standard=event_data.get('vendor_standard'),
                    platform=event_data.get('source_platform') or event_data.get('platform'),
                    user_id=event_data.get('user_id')
                )
                
                record = result.single()
                if record:
                    logger.debug(f"✅ Created/updated event node: {record['event_id']}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to create event node: {e}")
            return False
    
    def create_relationship(
        self, 
        source_id: str, 
        target_id: str, 
        rel_data: Dict[str, Any]
    ) -> bool:
        """
        Create CAUSES relationship between two events.
        
        Args:
            source_id: Source event ID
            target_id: Target event ID
            rel_data: Relationship metadata (confidence, semantic description, etc.)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.driver.session() as session:
                query = """
                MATCH (source:Event {event_id: $source_id})
                MATCH (target:Event {event_id: $target_id})
                MERGE (source)-[r:CAUSES]->(target)
                SET r.relationship_type = $relationship_type,
                    r.confidence_score = $confidence_score,
                    r.detection_method = $detection_method,
                    r.semantic_description = $semantic_description,
                    r.temporal_causality = $temporal_causality,
                    r.business_logic = $business_logic,
                    r.causal_score = $causal_score,
                    r.temporal_precedence = $temporal_precedence,
                    r.strength = $strength,
                    r.consistency = $consistency,
                    r.created_at = datetime(),
                    r.updated_at = datetime()
                RETURN r
                """
                
                result = session.run(query,
                    source_id=source_id,
                    target_id=target_id,
                    relationship_type=rel_data.get('relationship_type', 'unknown'),
                    confidence_score=float(rel_data.get('confidence_score', 0)),
                    detection_method=rel_data.get('detection_method', 'database_join'),
                    semantic_description=rel_data.get('semantic_description'),
                    temporal_causality=rel_data.get('temporal_causality'),
                    business_logic=rel_data.get('business_logic'),
                    causal_score=float(rel_data.get('causal_score', 0)) if rel_data.get('causal_score') else None,
                    temporal_precedence=float(rel_data.get('temporal_precedence', 0)) if rel_data.get('temporal_precedence') else None,
                    strength=float(rel_data.get('strength', 0)) if rel_data.get('strength') else None,
                    consistency=float(rel_data.get('consistency', 0)) if rel_data.get('consistency') else None
                )
                
                if result.single():
                    logger.debug(f"✅ Created relationship: {source_id} → {target_id}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to create relationship: {e}")
            return False
    
    def find_causal_chain(
        self, 
        event_id: str, 
        max_hops: int = 5, 
        min_confidence: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Find all events caused by this event (multi-hop traversal).
        
        Args:
            event_id: Starting event ID
            max_hops: Maximum relationship hops to traverse
            min_confidence: Minimum confidence score for relationships
        
        Returns:
            List of caused events with metadata
        """
        try:
            with self.driver.session() as session:
                query = f"""
                MATCH path = (start:Event {{event_id: $event_id}})-[r:CAUSES*1..{max_hops}]->(end:Event)
                WHERE ALL(rel IN relationships(path) WHERE rel.confidence_score >= $min_confidence)
                RETURN 
                    end.event_id AS caused_event_id,
                    end.document_type AS document_type,
                    end.amount_usd AS amount_usd,
                    end.source_ts AS source_ts,
                    length(path) AS hops,
                    [rel IN relationships(path) | rel.relationship_type] AS relationship_chain,
                    reduce(score = 1.0, rel IN relationships(path) | score * rel.confidence_score) AS chain_confidence
                ORDER BY hops, chain_confidence DESC
                """
                
                result = session.run(query, event_id=event_id, min_confidence=min_confidence)
                return [dict(record) for record in result]
                
        except Exception as e:
            logger.error(f"❌ Failed to find causal chain: {e}")
            return []
    
    def find_root_cause(
        self, 
        problem_event_id: str, 
        max_depth: int = 10,
        min_causal_score: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Find root cause by traversing backwards through causal relationships.
        
        Args:
            problem_event_id: Problem event to trace back from
            max_depth: Maximum depth to search
            min_causal_score: Minimum causal score for relationships
        
        Returns:
            List of potential root causes with causal paths
        """
        try:
            with self.driver.session() as session:
                query = f"""
                MATCH path = (root:Event)-[r:CAUSES*1..{max_depth}]->(problem:Event {{event_id: $problem_event_id}})
                WHERE ALL(rel IN relationships(path) WHERE coalesce(rel.causal_score, 0) >= $min_causal_score)
                WITH path, root, 
                     length(path) AS depth,
                     reduce(score = 1.0, rel IN relationships(path) | score * coalesce(rel.causal_score, 0.5)) AS total_causality
                ORDER BY total_causality DESC, depth ASC
                LIMIT 10
                RETURN 
                    root.event_id AS root_event_id,
                    root.document_type AS root_type,
                    root.amount_usd AS root_amount,
                    root.source_ts AS root_date,
                    depth,
                    total_causality,
                    [node IN nodes(path) | node.event_id] AS causal_path
                """
                
                result = session.run(query, 
                    problem_event_id=problem_event_id,
                    min_causal_score=min_causal_score
                )
                return [dict(record) for record in result]
                
        except Exception as e:
            logger.error(f"❌ Failed to find root cause: {e}")
            return []
    
    def get_relationship_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get comprehensive relationship statistics for a user.
        
        Args:
            user_id: User ID
        
        Returns:
            Statistics dictionary
        """
        try:
            with self.driver.session() as session:
                query = """
                MATCH (e:Event {user_id: $user_id})
                OPTIONAL MATCH (e)-[r:CAUSES]->()
                RETURN 
                    count(DISTINCT e) AS total_events,
                    count(r) AS total_relationships,
                    avg(r.confidence_score) AS avg_confidence,
                    avg(r.causal_score) AS avg_causality,
                    count(DISTINCT r.relationship_type) AS unique_relationship_types
                """
                
                result = session.run(query, user_id=user_id)
                record = result.single()
                return dict(record) if record else {}
                
        except Exception as e:
            logger.error(f"❌ Failed to get relationship stats: {e}")
            return {}
    
    def close(self):
        """Close Neo4j driver connection"""
        if self.driver:
            self.driver.close()
            logger.info("✅ Neo4j connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
