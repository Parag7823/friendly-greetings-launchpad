"""
Google CTO-Grade Locust Load Testing for Relationship Ecosystem
================================================================

This file implements comprehensive load testing for the relationship detection
and graph ecosystem, covering ALL components with realistic user behavior patterns.

Coverage:
- Enhanced Relationship Detector
- Semantic Relationship Extractor
- Temporal Pattern Learner
- Causal Inference Engine
- Finley Graph Engine
- ALL 11 Graph API endpoints

Target: 50+ concurrent users with 0.1% error rate
"""

from locust import HttpUser, task, between, events
import uuid
import time
import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Any

# CRITICAL: Load environment variables from .env.test
import dotenv
env_path = Path(__file__).parent.parent / '.env.test'
dotenv.load_dotenv(env_path)
print(f"[Locust Relationship] Loaded environment from {env_path}")
print(f"[Locust Relationship] TEST_API_URL: {os.getenv('TEST_API_URL')}")
print(f"[Locust Relationship] SUPABASE_URL: {os.getenv('SUPABASE_URL')[:30]}..." if os.getenv('SUPABASE_URL') else "[Locust Relationship] SUPABASE_URL: NOT SET")


# ============================================================================
# CONFIGURATION
# ============================================================================

# Backend URL - PRODUCTION TESTING (Railway deployment)
BACKEND_URL = os.getenv('TEST_API_URL', os.getenv('BACKEND_URL', 'http://localhost:8000'))

# Test user credentials - Use a dedicated test user with pre-built graph
TEST_USER_ID = os.getenv('TEST_USER_ID', str(uuid.uuid4()))
TEST_SESSION_TOKEN = "test_session_token_placeholder"

# Graph API base path
GRAPH_API_BASE = "/api/v1/graph"

# Test entity IDs - These should exist in your test database
# In production, these will be generated from your test data
TEST_ENTITY_IDS = []  # Will be populated during test


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_test_entity_ids(count: int = 20) -> List[str]:
    """Generate realistic entity IDs for testing"""
    return [str(uuid.uuid4()) for _ in range(count)]


def validate_confidence_score(score: Any, response) -> bool:
    """
    Validate confidence score is between 0.0 and 1.0
    
    GOOGLE CTO-GRADE: Strict validation, no lenient checks
    """
    if not isinstance(score, (int, float)):
        response.failure(f"❌ Confidence score must be numeric, got {type(score)}")
        return False
    
    if not (0.0 <= score <= 1.0):
        response.failure(f"❌ Confidence score must be 0-1, got {score}")
        return False
    
    return True


def validate_graph_stats(data: Dict, response) -> bool:
    """
    Validate GraphStats response structure
    
    Required fields:
    - node_count (int)
    - edge_count (int)
    - avg_degree (float)
    - density (float)
    - connected_components (int)
    """
    required_fields = {
        'node_count': int,
        'edge_count': int,
        'avg_degree': (int, float),
        'density': (int, float),
        'connected_components': int
    }
    
    for field, expected_type in required_fields.items():
        if field not in data:
            response.failure(f"❌ Missing required field '{field}' in GraphStats")
            return False
        
        if not isinstance(data[field], expected_type):
            response.failure(f"❌ Field '{field}' must be {expected_type}, got {type(data[field])}")
            return False
    
    # Validate logical constraints
    if data['node_count'] < 0:
        response.failure(f"❌ node_count cannot be negative: {data['node_count']}")
        return False
    
    if data['edge_count'] < 0:
        response.failure(f"❌ edge_count cannot be negative: {data['edge_count']}")
        return False
    
    if not (0.0 <= data['density'] <= 1.0):
        response.failure(f"❌ density must be 0-1, got {data['density']}")
        return False
    
    return True


def validate_path_result(data: Dict, response) -> bool:
    """
    Validate PathResult response structure
    
    Required fields:
    - path_nodes (list)
    - path_edges (list)
    - path_length (int)
    - confidence (float)
    """
    if 'path' not in data and data.get('path') is None:
        # No path found is acceptable
        return True
    
    path = data.get('path', {})
    
    required_fields = ['path_nodes', 'path_edges', 'path_length']
    for field in required_fields:
        if field not in path:
            response.failure(f"❌ Missing required field '{field}' in PathResult")
            return False
    
    if not isinstance(path['path_nodes'], list):
        response.failure(f"❌ path_nodes must be list, got {type(path['path_nodes'])}")
        return False
    
    if not isinstance(path['path_edges'], list):
        response.failure(f"❌ path_edges must be list, got {type(path['path_edges'])}")
        return False
    
    if not isinstance(path['path_length'], int):
        response.failure(f"❌ path_length must be int, got {type(path['path_length'])}")
        return False
    
    # Validate path consistency
    if len(path['path_nodes']) - 1 != len(path['path_edges']):
        response.failure(f"❌ Path inconsistent: {len(path['path_nodes'])} nodes but {len(path['path_edges'])} edges")
        return False
    
    return True


def validate_temporal_pattern(pattern: Dict, response) -> bool:
    """Validate temporal pattern structure"""
    required_fields = ['relationship_type', 'recurrence_frequency']
    
    for field in required_fields:
        if field not in pattern:
            response.failure(f"❌ Missing '{field}' in temporal pattern")
            return False
    
    if 'recurrence_score' in pattern:
        if not validate_confidence_score(pattern['recurrence_score'], response):
            return False
    
    return True


def validate_causal_chain(chain: List[Dict], response) -> bool:
    """Validate causal chain structure"""
    valid_directions = ['source_to_target', 'target_to_source', 'bidirectional', 'none', 'unknown']
    
    for step in chain:
        if 'causal_direction' in step:
            if step['causal_direction'] not in valid_directions:
                response.failure(f"❌ Invalid causal_direction: {step['causal_direction']}")
                return False
        
        if 'causal_strength' in step:
            if not validate_confidence_score(step['causal_strength'], response):
                return False
    
    return True


# ============================================================================
# USER BEHAVIOR CLASSES
# ============================================================================

class RelationshipDetectionUser(HttpUser):
    """
    Simulates a user focused on basic graph operations and relationship detection.
    
    Primary workflow:
    1. Build graph from normalized data
    2. Query graph statistics
    3. Find paths between entities
    4. Analyze entity importance
    5. Detect communities
    
    Weight: 40% of traffic (most common use case)
    """
    wait_time = between(2, 5)
    weight = 40
    
    def on_start(self):
        """Initialize user with test data"""
        self.user_id = TEST_USER_ID
        self.session_token = TEST_SESSION_TOKEN
        self.graph_built = False
        self.entity_ids = []
        
    @task(10)
    def build_graph(self):
        """
        Build Knowledge Graph - Core Operation
        
        Endpoint: POST /api/v1/graph/build
        Expected: 200 OK with GraphStats
        
        This tests:
        - Finley Graph Engine graph construction
        - Entity and relationship loading from Supabase
        - igraph performance
        - Redis caching
        """
        with self.client.post(
            f"{GRAPH_API_BASE}/build",
            json={
                "user_id": self.user_id,
                "force_rebuild": False  # Use cache when available
            },
            catch_response=True,
            name="POST /api/v1/graph/build"
        ) as response:
            if response.status_code != 200:
                response.failure(f"❌ Expected 200, got {response.status_code}: {response.text[:200]}")
            else:
                try:
                    data = response.json()
                    
                    # Validate response structure
                    if 'status' not in data:
                        response.failure("❌ Missing 'status' in response")
                        return
                    
                    if 'stats' not in data:
                        response.failure("❌ Missing 'stats' in response")
                        return
                    
                    # Validate GraphStats
                    if validate_graph_stats(data['stats'], response):
                        self.graph_built = True
                        response.success()
                        
                        # Log stats for monitoring
                        stats = data['stats']
                        print(f"[Graph Built] Nodes: {stats['node_count']}, Edges: {stats['edge_count']}, Density: {stats['density']:.4f}")
                    
                except json.JSONDecodeError as e:
                    response.failure(f"❌ Invalid JSON response: {e}")
    
    @task(8)
    def get_graph_stats(self):
        """
        Get Graph Statistics
        
        Endpoint: GET /api/v1/graph/stats/{user_id}
        Expected: 200 OK with GraphStats
        
        This tests:
        - Graph metrics calculation
        - Cache retrieval
        """
        with self.client.get(
            f"{GRAPH_API_BASE}/stats/{self.user_id}",
            catch_response=True,
            name="GET /api/v1/graph/stats/{user_id}"
        ) as response:
            if response.status_code == 400:
                # Graph not built yet - acceptable
                response.success()
            elif response.status_code != 200:
                response.failure(f"❌ Expected 200 or 400, got {response.status_code}: {response.text[:200]}")
            else:
                try:
                    data = response.json()
                    if validate_graph_stats(data, response):
                        response.success()
                except json.JSONDecodeError as e:
                    response.failure(f"❌ Invalid JSON response: {e}")
    
    @task(7)
    def query_shortest_path(self):
        """
        Find Shortest Path Between Entities
        
        Endpoint: POST /api/v1/graph/query/path
        Expected: 200 OK with PathResult or null
        
        This tests:
        - igraph shortest path algorithms
        - Path enrichment with intelligence layers
        - Edge data retrieval
        """
        # Generate or use existing entity IDs
        if not self.entity_ids:
            self.entity_ids = generate_test_entity_ids(10)
        
        source_id = random.choice(self.entity_ids)
        target_id = random.choice(self.entity_ids)
        
        with self.client.post(
            f"{GRAPH_API_BASE}/query/path",
            json={
                "user_id": self.user_id,
                "source_entity_id": source_id,
                "target_entity_id": target_id,
                "max_length": 5
            },
            catch_response=True,
            name="POST /api/v1/graph/query/path"
        ) as response:
            if response.status_code == 400:
                # Graph not built - acceptable
                response.success()
            elif response.status_code != 200:
                response.failure(f"❌ Expected 200, got {response.status_code}: {response.text[:200]}")
            else:
                try:
                    data = response.json()
                    
                    if 'status' not in data:
                        response.failure("❌ Missing 'status' in response")
                        return
                    
                    # Validate path result if present
                    if validate_path_result(data, response):
                        response.success()
                    
                except json.JSONDecodeError as e:
                    response.failure(f"❌ Invalid JSON response: {e}")
    
    @task(5)
    def query_entity_importance(self):
        """
        Calculate Entity Importance using Graph Algorithms
        
        Endpoint: POST /api/v1/graph/query/importance
        Expected: 200 OK with ranked entities
        
        This tests:
        - PageRank algorithm
        - Betweenness centrality
        - Closeness centrality
        """
        algorithms = ['pagerank', 'betweenness', 'closeness']
        algorithm = random.choice(algorithms)
        
        with self.client.post(
            f"{GRAPH_API_BASE}/query/importance",
            json={
                "user_id": self.user_id,
                "algorithm": algorithm,
                "top_n": 20
            },
            catch_response=True,
            name=f"POST /api/v1/graph/query/importance ({algorithm})"
        ) as response:
            if response.status_code == 400:
                # Graph not built - acceptable
                response.success()
            elif response.status_code != 200:
                response.failure(f"❌ Expected 200, got {response.status_code}: {response.text[:200]}")
            else:
                try:
                    data = response.json()
                    
                    required_fields = ['status', 'top_entities', 'algorithm']
                    for field in required_fields:
                        if field not in data:
                            response.failure(f"❌ Missing '{field}' in response")
                            return
                    
                    # Validate top_entities structure
                    if not isinstance(data['top_entities'], list):
                        response.failure(f"❌ top_entities must be list, got {type(data['top_entities'])}")
                        return
                    
                    # Validate each entity has required fields
                    for entity in data['top_entities'][:5]:  # Check first 5
                        if 'entity_id' not in entity:
                            response.failure("❌ Missing 'entity_id' in top entity")
                            return
                        if 'score' not in entity:
                            response.failure("❌ Missing 'score' in top entity")
                            return
                        if not isinstance(entity['score'], (int, float)):
                            response.failure(f"❌ Score must be numeric, got {type(entity['score'])}")
                            return
                    
                    response.success()
                    
                except json.JSONDecodeError as e:
                    response.failure(f"❌ Invalid JSON response: {e}")
    
    @task(3)
    def detect_communities(self):
        """
        Detect Entity Communities using Louvain Algorithm
        
        Endpoint: POST /api/v1/graph/query/communities
        Expected: 200 OK with community mapping
        
        This tests:
        - Community detection algorithms
        - Graph clustering
        """
        with self.client.post(
            f"{GRAPH_API_BASE}/query/communities",
            json={
                "user_id": self.user_id
            },
            catch_response=True,
            name="POST /api/v1/graph/query/communities"
        ) as response:
            if response.status_code == 400:
                # Graph not built - acceptable
                response.success()
            elif response.status_code != 200:
                response.failure(f"❌ Expected 200, got {response.status_code}: {response.text[:200]}")
            else:
                try:
                    data = response.json()
                    
                    required_fields = ['status', 'communities', 'community_count']
                    for field in required_fields:
                        if field not in data:
                            response.failure(f"❌ Missing '{field}' in response")
                            return
                    
                    if not isinstance(data['communities'], dict):
                        response.failure(f"❌ communities must be dict, got {type(data['communities'])}")
                        return
                    
                    if not isinstance(data['community_count'], int):
                        response.failure(f"❌ community_count must be int, got {type(data['community_count'])}")
                        return
                    
                    response.success()
                    
                except json.JSONDecodeError as e:
                    response.failure(f"❌ Invalid JSON response: {e}")


class GraphAnalyticsUser(HttpUser):
    """
    Simulates a user focused on intelligence layer queries.
    
    Primary workflow:
    1. Query temporal patterns
    2. Analyze seasonal cycles
    3. Perform incremental graph updates
    4. Community detection
    
    Weight: 30% of traffic
    """
    wait_time = between(3, 6)
    weight = 30
    
    def on_start(self):
        """Initialize analytics user"""
        self.user_id = TEST_USER_ID
        self.entity_ids = generate_test_entity_ids(15)
    
    @task(10)
    def query_temporal_patterns(self):
        """
        Query Recurring Temporal Patterns
        
        Endpoint: POST /api/v1/graph/query/temporal-patterns
        Expected: 200 OK with temporal patterns
        
        This tests:
        - Temporal Pattern Learner integration
        - Pattern frequency analysis
        - Next occurrence predictions
        """
        source_id = random.choice(self.entity_ids)
        target_id = random.choice(self.entity_ids)
        
        with self.client.post(
            f"{GRAPH_API_BASE}/query/temporal-patterns",
            json={
                "user_id": self.user_id,
                "source_entity_id": source_id,
                "target_entity_id": target_id,
                "min_recurrence_score": 0.5
            },
            catch_response=True,
            name="POST /api/v1/graph/query/temporal-patterns"
        ) as response:
            if response.status_code == 400:
                # Graph not built - acceptable
                response.success()
            elif response.status_code != 200:
                response.failure(f"❌ Expected 200, got {response.status_code}: {response.text[:200]}")
            else:
                try:
                    data = response.json()
                    
                    required_fields = ['status', 'patterns', 'message']
                    for field in required_fields:
                        if field not in data:
                            response.failure(f"❌ Missing '{field}' in response")
                            return
                    
                    if not isinstance(data['patterns'], list):
                        response.failure(f"❌ patterns must be list, got {type(data['patterns'])}")
                        return
                    
                    # Validate each pattern
                    for pattern in data['patterns'][:3]:  # Check first 3
                        if not validate_temporal_pattern(pattern, response):
                            return
                    
                    response.success()
                    
                except json.JSONDecodeError as e:
                    response.failure(f"❌ Invalid JSON response: {e}")
    
    @task(8)
    def query_seasonal_cycles(self):
        """
        Query Seasonal Patterns
        
        Endpoint: POST /api/v1/graph/query/seasonal-cycles
        Expected: 200 OK with seasonal cycles
        
        This tests:
        - Seasonal pattern detection
        - Month-based cycle analysis
        """
        source_id = random.choice(self.entity_ids)
        target_id = random.choice(self.entity_ids)
        
        with self.client.post(
            f"{GRAPH_API_BASE}/query/seasonal-cycles",
            json={
                "user_id": self.user_id,
                "source_entity_id": source_id,
                "target_entity_id": target_id,
                "min_seasonal_strength": 0.5
            },
            catch_response=True,
            name="POST /api/v1/graph/query/seasonal-cycles"
        ) as response:
            if response.status_code == 400:
                # Graph not built - acceptable
                response.success()
            elif response.status_code != 200:
                response.failure(f"❌ Expected 200, got {response.status_code}: {response.text[:200]}")
            else:
                try:
                    data = response.json()
                    
                    required_fields = ['status', 'seasonal_cycles', 'message']
                    for field in required_fields:
                        if field not in data:
                            response.failure(f"❌ Missing '{field}' in response")
                            return
                    
                    if not isinstance(data['seasonal_cycles'], list):
                        response.failure(f"❌ seasonal_cycles must be list, got {type(data['seasonal_cycles'])}")
                        return
                    
                    # Validate each cycle
                    for cycle in data['seasonal_cycles']:
                        if 'seasonal_strength' in cycle:
                            if not validate_confidence_score(cycle['seasonal_strength'], response):
                                return
                    
                    response.success()
                    
                except json.JSONDecodeError as e:
                    response.failure(f"❌ Invalid JSON response: {e}")
    
    @task(5)
    def incremental_update(self):
        """
        Incremental Graph Update
        
        Endpoint: POST /api/v1/graph/update
        Expected: 200 OK with update counts
        
        This tests:
        - Incremental graph updates
        - New node/edge integration
        - Cache invalidation
        """
        with self.client.post(
            f"{GRAPH_API_BASE}/update",
            json={
                "user_id": self.user_id,
                "since_minutes": 60  # Last hour
            },
            catch_response=True,
            name="POST /api/v1/graph/update"
        ) as response:
            if response.status_code != 200:
                response.failure(f"❌ Expected 200, got {response.status_code}: {response.text[:200]}")
            else:
                try:
                    data = response.json()
                    
                    required_fields = ['status', 'nodes_added', 'edges_added']
                    for field in required_fields:
                        if field not in data:
                            response.failure(f"❌ Missing '{field}' in response")
                            return
                    
                    if not isinstance(data['nodes_added'], int):
                        response.failure(f"❌ nodes_added must be int, got {type(data['nodes_added'])}")
                        return
                    
                    if not isinstance(data['edges_added'], int):
                        response.failure(f"❌ edges_added must be int, got {type(data['edges_added'])}")
                        return
                    
                    response.success()
                    
                except json.JSONDecodeError as e:
                    response.failure(f"❌ Invalid JSON response: {e}")


class CausalAnalysisUser(HttpUser):
    """
    Simulates a user focused on advanced causal analytics.
    
    Primary workflow:
    1. Root cause analysis
    2. Fraud detection
    3. Prediction queries
    4. Causal chain validation
    
    Weight: 20% of traffic
    """
    wait_time = between(4, 8)
    weight = 20
    
    def on_start(self):
        """Initialize causal analysis user"""
        self.user_id = TEST_USER_ID
        self.entity_ids = generate_test_entity_ids(12)
    
    @task(10)
    def query_root_causes(self):
        """
        Root Cause Analysis
        
        Endpoint: POST /api/v1/graph/query/root-causes
        Expected: 200 OK with causal chain
        
        This tests:
        - Causal Inference Engine integration
        - Bradford Hill criteria scoring
        - Causal chain construction
        """
        source_id = random.choice(self.entity_ids)
        target_id = random.choice(self.entity_ids)
        
        with self.client.post(
            f"{GRAPH_API_BASE}/query/root-causes",
            json={
                "user_id": self.user_id,
                "source_entity_id": source_id,
                "target_entity_id": target_id
            },
            catch_response=True,
            name="POST /api/v1/graph/query/root-causes"
        ) as response:
            if response.status_code == 400:
                # Graph not built - acceptable
                response.success()
            elif response.status_code != 200:
                response.failure(f"❌ Expected 200, got {response.status_code}: {response.text[:200]}")
            else:
                try:
                    data = response.json()
                    
                    required_fields = ['status', 'root_causes', 'causal_chain', 'message']
                    for field in required_fields:
                        if field not in data:
                            response.failure(f"❌ Missing '{field}' in response")
                            return
                    
                    if not isinstance(data['root_causes'], list):
                        response.failure(f"❌ root_causes must be list, got {type(data['root_causes'])}")
                        return
                    
                    if not isinstance(data['causal_chain'], list):
                        response.failure(f"❌ causal_chain must be list, got {type(data['causal_chain'])}")
                        return
                    
                    # Validate causal chain structure
                    if not validate_causal_chain(data['causal_chain'], response):
                        return
                    
                    response.success()
                    
                except json.JSONDecodeError as e:
                    response.failure(f"❌ Invalid JSON response: {e}")
    
    @task(8)
    def query_fraud_detection(self):
        """
        Fraud Detection Query
        
        Endpoint: POST /api/v1/graph/query/fraud-detection
        Expected: 200 OK with fraud alerts
        
        This tests:
        - Duplicate detection integration
        - Fraud risk scoring
        - Alert generation
        """
        source_id = random.choice(self.entity_ids)
        target_id = random.choice(self.entity_ids)
        
        with self.client.post(
            f"{GRAPH_API_BASE}/query/fraud-detection",
            json={
                "user_id": self.user_id,
                "source_entity_id": source_id,
                "target_entity_id": target_id,
                "min_fraud_confidence": 0.7
            },
            catch_response=True,
            name="POST /api/v1/graph/query/fraud-detection"
        ) as response:
            if response.status_code == 400:
                # Graph not built - acceptable
                response.success()
            elif response.status_code != 200:
                response.failure(f"❌ Expected 200, got {response.status_code}: {response.text[:200]}")
            else:
                try:
                    data = response.json()
                    
                    required_fields = ['status', 'fraud_alerts', 'fraud_risk_score', 'message']
                    for field in required_fields:
                        if field not in data:
                            response.failure(f"❌ Missing '{field}' in response")
                            return
                    
                    if not isinstance(data['fraud_alerts'], list):
                        response.failure(f"❌ fraud_alerts must be list, got {type(data['fraud_alerts'])}")
                        return
                    
                    # Validate fraud risk score
                    if not validate_confidence_score(data['fraud_risk_score'], response):
                        return
                    
                    response.success()
                    
                except json.JSONDecodeError as e:
                    response.failure(f"❌ Invalid JSON response: {e}")
    
    @task(6)
    def query_predictions(self):
        """
        Prediction Query
        
        Endpoint: POST /api/v1/graph/query/predictions
        Expected: 200 OK with predictions
        
        This tests:
        - Temporal Pattern Learner predictions
        - Future relationship predictions
        - Confidence scoring
        """
        source_id = random.choice(self.entity_ids)
        target_id = random.choice(self.entity_ids)
        
        with self.client.post(
            f"{GRAPH_API_BASE}/query/predictions",
            json={
                "user_id": self.user_id,
                "source_entity_id": source_id,
                "target_entity_id": target_id,
                "min_prediction_confidence": 0.6
            },
            catch_response=True,
            name="POST /api/v1/graph/query/predictions"
        ) as response:
            if response.status_code == 400:
                # Graph not built - acceptable
                response.success()
            elif response.status_code != 200:
                response.failure(f"❌ Expected 200, got {response.status_code}: {response.text[:200]}")
            else:
                try:
                    data = response.json()
                    
                    required_fields = ['status', 'predictions', 'message']
                    for field in required_fields:
                        if field not in data:
                            response.failure(f"❌ Missing '{field}' in response")
                            return
                    
                    if not isinstance(data['predictions'], list):
                        response.failure(f"❌ predictions must be list, got {type(data['predictions'])}")
                        return
                    
                    # Validate each prediction
                    for pred in data['predictions']:
                        if 'prediction_confidence' in pred:
                            if not validate_confidence_score(pred['prediction_confidence'], response):
                                return
                    
                    response.success()
                    
                except json.JSONDecodeError as e:
                    response.failure(f"❌ Invalid JSON response: {e}")


class PowerGraphUser(HttpUser):
    """
    Simulates a power user doing aggressive graph operations.
    
    Pattern:
    - Batch queries
    - Complex multi-hop paths
    - Parallel intelligence queries
    - Rapid updates
    
    Weight: 10% of traffic (stress testing)
    """
    wait_time = between(1, 3)
    weight = 10
    
    def on_start(self):
        """Initialize power user"""
        self.user_id = TEST_USER_ID
        self.entity_ids = generate_test_entity_ids(30)
    
    @task(10)
    def batch_path_queries(self):
        """Execute multiple path queries rapidly"""
        num_queries = random.randint(3, 5)
        
        for _ in range(num_queries):
            source_id = random.choice(self.entity_ids)
            target_id = random.choice(self.entity_ids)
            
            self.client.post(
                f"{GRAPH_API_BASE}/query/path",
                json={
                    "user_id": self.user_id,
                    "source_entity_id": source_id,
                    "target_entity_id": target_id,
                    "max_length": 5
                },
                name="Batch: Path Queries"
            )
            time.sleep(0.1)  # Small delay between queries
    
    @task(5)
    def complex_multi_hop_query(self):
        """Query long paths to stress test graph algorithms"""
        source_id = random.choice(self.entity_ids)
        target_id = random.choice(self.entity_ids)
        
        self.client.post(
            f"{GRAPH_API_BASE}/query/path",
            json={
                "user_id": self.user_id,
                "source_entity_id": source_id,
                "target_entity_id": target_id,
                "max_length": 10  # Long paths
            },
            name="Complex: Long Path Query"
        )
    
    @task(8)
    def parallel_intelligence_queries(self):
        """
        Query all intelligence layers in parallel
        
        This stress tests:
        - Concurrent database queries
        - Cache performance
        - API responsiveness under load
        """
        source_id = random.choice(self.entity_ids)
        target_id = random.choice(self.entity_ids)
        
        # Fire off multiple intelligence queries
        intelligence_endpoints = [
            '/query/temporal-patterns',
            '/query/seasonal-cycles',
            '/query/fraud-detection',
            '/query/root-causes',
            '/query/predictions'
        ]
        
        for endpoint in intelligence_endpoints:
            self.client.post(
                f"{GRAPH_API_BASE}{endpoint}",
                json={
                    "user_id": self.user_id,
                    "source_entity_id": source_id,
                    "target_entity_id": target_id
                },
                name=f"Intelligence: {endpoint}"
            )
            time.sleep(0.05)  # Tiny delay


# ============================================================================
# LOCUST EVENT HANDLERS
# ============================================================================

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when the load test starts"""
    print("\n" + "="*80)
    print("🚀 STARTING LOAD TEST: Relationship Ecosystem")
    print("="*80)
    print(f"Target: {environment.host}")
    print(f"Test User: {TEST_USER_ID}")
    print(f"Graph API Base: {GRAPH_API_BASE}")
    print("="*80)
    print("\nComponents Under Test:")
    print("  ✓ Enhanced Relationship Detector")
    print("  ✓ Semantic Relationship Extractor")
    print("  ✓ Temporal Pattern Learner")
    print("  ✓ Causal Inference Engine")
    print("  ✓ Finley Graph Engine")
    print("  ✓ 11 Graph API Endpoints")
    print("="*80 + "\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when the load test stops"""
    print("\n" + "="*80)
    print("✅ RELATIONSHIP ECOSYSTEM LOAD TEST COMPLETE")
    print("="*80)
    
    stats = environment.stats.total
    print(f"Total Requests: {stats.num_requests}")
    print(f"Total Failures: {stats.num_failures}")
    print(f"Failure Rate: {stats.fail_ratio:.2%}")
    print(f"Average Response Time: {stats.avg_response_time:.0f}ms")
    print(f"P95 Response Time: {stats.get_response_time_percentile(0.95):.0f}ms")
    print(f"P99 Response Time: {stats.get_response_time_percentile(0.99):.0f}ms")
    print(f"RPS: {stats.total_rps:.2f}")
    
    # PRODUCTION-GRADE: Validate against Google CTO targets
    print("\n" + "="*80)
    print("📊 GOOGLE CTO-GRADE VALIDATION")
    print("="*80)
    
    # Target: Error rate < 0.1%
    error_rate = stats.fail_ratio * 100
    error_status = "✅ PASS" if error_rate < 0.1 else "❌ FAIL"
    print(f"{error_status} Error Rate: {error_rate:.3f}% (target: < 0.1%)")
    
    # Target: P95 < 1000ms
    p95 = stats.get_response_time_percentile(0.95)
    p95_status = "✅ PASS" if p95 < 1000 else "❌ FAIL"
    print(f"{p95_status} P95 Latency: {p95:.0f}ms (target: < 1000ms)")
    
    # Target: P99 < 3000ms
    p99 = stats.get_response_time_percentile(0.99)
    p99_status = "✅ PASS" if p99 < 3000 else "❌ FAIL"
    print(f"{p99_status} P99 Latency: {p99:.0f}ms (target: < 3000ms)")
    
    # Target: Throughput > 50 RPS
    rps_status = "✅ PASS" if stats.total_rps > 50 else "❌ FAIL"
    print(f"{rps_status} Throughput: {stats.total_rps:.2f} RPS (target: > 50 RPS)")
    
    print("="*80 + "\n")


# ============================================================================
# INSTRUCTIONS
# ============================================================================
"""
To run this relationship ecosystem load test:

1. Ensure test data is prepared:
   - User must have normalized data in database
   - Relationships should be detected
   - Graph should be buildable

2. Set environment variables in .env.test:
   TEST_API_URL=https://your-railway-app.railway.app
   TEST_USER_ID=your-test-user-uuid
   SUPABASE_URL=your-supabase-url
   SUPABASE_KEY=your-supabase-key

3. Run Locust with Web UI:
   locust -f tests/locustfile_relationship.py --host=https://your-railway-app.railway.app

4. Open http://localhost:8089 in browser
   - Set users: 50
   - Spawn rate: 5 users/second
   - Run time: 10 minutes

5. Or run headless (CI/CD):
   locust -f tests/locustfile_relationship.py \\
          --host=https://your-railway-app.railway.app \\
          --users 50 --spawn-rate 5 --run-time 10m --headless \\
          --html=locust_relationship_report.html

Target Metrics (Google CTO-Grade):
- Error Rate: < 0.1%
- P50 Response Time: < 300ms
- P95 Response Time: < 1000ms
- P99 Response Time: < 3000ms
- Throughput: > 50 RPS
- Graph Build Time: < 5s (for 1000 nodes, 5000 edges)

CRITICAL: This tests REAL production code on Railway.
Any failures must trigger fixes in production code, NOT test adjustments!
"""
