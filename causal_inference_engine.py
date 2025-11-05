"""
Production-Grade Causal Inference Engine
=========================================

Implements Bradford Hill criteria for causal relationship detection,
root cause analysis, and counterfactual ("what-if") analysis.

This module transforms correlation-based relationship detection into
true causal reasoning using scientific causality criteria.

Features:
- Bradford Hill criteria evaluation (6 criteria)
- Causal graph construction and traversal
- Root cause analysis with backward tracing
- Counterfactual analysis for what-if scenarios
- Network analysis using igraph (13-32x faster than NetworkX)
- Production-ready with comprehensive error handling

Author: Senior Full-Stack Engineer
Version: 1.0.0
Date: 2025-01-21
"""

import logging
import igraph as ig  # REFACTORED: 13-32x faster than networkx
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)


class CausalDirection(Enum):
    """Direction of causal relationship"""
    SOURCE_TO_TARGET = "source_to_target"
    TARGET_TO_SOURCE = "target_to_source"
    BIDIRECTIONAL = "bidirectional"
    NONE = "none"


@dataclass
class BradfordHillScores:
    """Bradford Hill criteria scores for causality assessment"""
    temporal_precedence: float  # 0.0-1.0
    strength: float
    consistency: float
    specificity: float
    dose_response: float
    plausibility: float
    causal_score: float  # Average of all criteria
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class CausalRelationship:
    """Causal relationship with Bradford Hill analysis"""
    relationship_id: str
    source_event_id: str
    target_event_id: str
    bradford_hill_scores: BradfordHillScores
    is_causal: bool
    causal_direction: CausalDirection
    criteria_details: Dict[str, Any]


@dataclass
class RootCauseAnalysis:
    """Root cause analysis result"""
    problem_event_id: str
    root_event_id: str
    causal_path: List[str]
    path_length: int
    total_impact_usd: float
    affected_event_count: int
    affected_event_ids: List[str]
    root_cause_description: str
    confidence_score: float


@dataclass
class CounterfactualAnalysis:
    """Counterfactual analysis result"""
    intervention_event_id: str
    intervention_type: str
    original_value: Any
    counterfactual_value: Any
    affected_events: List[Dict[str, Any]]
    total_impact_delta_usd: float
    scenario_description: str


class CausalInferenceEngine:
    """
    Production-grade causal inference engine for financial event analysis.
    
    Implements:
    1. Bradford Hill criteria for causal detection
    2. Root cause analysis using graph traversal
    3. Counterfactual analysis for what-if scenarios
    """
    
    def __init__(self, supabase_client, config: Optional[Dict[str, Any]] = None):
        self.supabase = supabase_client
        self.config = config or self._get_default_config()
        
        # Causal graph (built on demand)
        self.causal_graph: Optional[ig.Graph] = None
        
        # Metrics
        self.metrics = {
            'causal_analyses': 0,
            'root_cause_analyses': 0,
            'counterfactual_analyses': 0,
            'causal_relationships_found': 0,
            'avg_causal_score': 0.0
        }
        
        logger.info("✅ CausalInferenceEngine initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'causal_threshold': 0.7,  # Minimum score to be considered causal
            'max_root_cause_depth': 10,  # Maximum depth for root cause tracing
            'consistency_threshold': 5,  # Minimum pattern count for consistency
            'temporal_window_days': 180,  # Maximum days between cause and effect
        }
    
    async def analyze_causal_relationships(
        self,
        user_id: str,
        relationship_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze relationships to determine causality using Bradford Hill criteria.
        
        Args:
            user_id: User ID
            relationship_ids: Optional list of specific relationship IDs to analyze
        
        Returns:
            Dictionary with causal analysis results and statistics
        """
        try:
            logger.info(f"Starting causal analysis for user_id={user_id}")
            
            # Fetch relationships to analyze
            relationships = await self._fetch_relationships(user_id, relationship_ids)
            
            if not relationships:
                return {
                    'causal_relationships': [],
                    'total_analyzed': 0,
                    'causal_count': 0,
                    'message': 'No relationships found to analyze'
                }
            
            causal_relationships = []
            causal_count = 0
            
            # Analyze each relationship
            for rel in relationships:
                try:
                    causal_rel = await self._analyze_single_relationship(rel, user_id)
                    
                    if causal_rel:
                        causal_relationships.append(causal_rel)
                        
                        # Store in database
                        await self._store_causal_relationship(causal_rel, user_id)
                        
                        if causal_rel.is_causal:
                            causal_count += 1
                        
                        self.metrics['causal_analyses'] += 1
                        
                except Exception as e:
                    logger.error(f"Failed to analyze relationship {rel.get('id')}: {e}")
                    continue
            
            # Update metrics
            if causal_relationships:
                avg_score = sum(cr.bradford_hill_scores.causal_score for cr in causal_relationships) / len(causal_relationships)
                self.metrics['avg_causal_score'] = avg_score
                self.metrics['causal_relationships_found'] = causal_count
            
            logger.info(f"✅ Causal analysis completed: {causal_count}/{len(relationships)} causal relationships found")
            
            return {
                'causal_relationships': [asdict(cr) for cr in causal_relationships],
                'total_analyzed': len(relationships),
                'causal_count': causal_count,
                'causal_percentage': (causal_count / len(relationships) * 100) if relationships else 0.0,
                'avg_causal_score': self.metrics['avg_causal_score'],
                'message': f'Causal analysis completed: {causal_count} causal relationships identified'
            }
            
        except Exception as e:
            logger.error(f"Causal analysis failed: {e}")
            return {
                'causal_relationships': [],
                'error': str(e),
                'message': 'Causal analysis failed'
            }
    
    async def _analyze_single_relationship(
        self,
        relationship: Dict[str, Any],
        user_id: str
    ) -> Optional[CausalRelationship]:
        """Analyze a single relationship using Bradford Hill criteria"""
        try:
            rel_id = relationship['id']
            source_id = relationship['source_event_id']
            target_id = relationship['target_event_id']
            
            # Call PostgreSQL function to calculate Bradford Hill scores
            result = self.supabase.rpc(
                'calculate_bradford_hill_scores',
                {
                    'p_relationship_id': rel_id,
                    'p_source_event_id': source_id,
                    'p_target_event_id': target_id,
                    'p_user_id': user_id
                }
            ).execute()
            
            if not result.data or 'error' in result.data:
                logger.warning(f"Failed to calculate Bradford Hill scores for {rel_id}")
                return None
            
            scores_data = result.data
            
            # Build Bradford Hill scores object
            bradford_hill_scores = BradfordHillScores(
                temporal_precedence=scores_data.get('temporal_precedence_score', 0.0),
                strength=scores_data.get('strength_score', 0.0),
                consistency=scores_data.get('consistency_score', 0.0),
                specificity=scores_data.get('specificity_score', 0.0),
                dose_response=scores_data.get('dose_response_score', 0.0),
                plausibility=scores_data.get('plausibility_score', 0.0),
                causal_score=scores_data.get('causal_score', 0.0)
            )
            
            # Determine if causal
            is_causal = bradford_hill_scores.causal_score >= self.config['causal_threshold']
            
            # Determine causal direction
            causal_direction = self._determine_causal_direction(
                bradford_hill_scores,
                relationship
            )
            
            # Build criteria details
            criteria_details = {
                'time_diff_days': scores_data.get('time_diff_days', 0),
                'similar_pattern_count': scores_data.get('similar_pattern_count', 0),
                'threshold_used': self.config['causal_threshold']
            }
            
            return CausalRelationship(
                relationship_id=rel_id,
                source_event_id=source_id,
                target_event_id=target_id,
                bradford_hill_scores=bradford_hill_scores,
                is_causal=is_causal,
                causal_direction=causal_direction,
                criteria_details=criteria_details
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze relationship: {e}")
            return None
    
    def _determine_causal_direction(
        self,
        scores: BradfordHillScores,
        relationship: Dict[str, Any]
    ) -> CausalDirection:
        """Determine the direction of causality"""
        
        # Use temporal precedence and semantic analysis
        if scores.temporal_precedence >= 0.7:
            # Strong temporal precedence suggests source causes target
            temporal_causality = relationship.get('temporal_causality', '')
            
            if temporal_causality == 'source_causes_target':
                return CausalDirection.SOURCE_TO_TARGET
            elif temporal_causality == 'target_causes_source':
                return CausalDirection.TARGET_TO_SOURCE
            elif temporal_causality == 'bidirectional':
                return CausalDirection.BIDIRECTIONAL
            else:
                return CausalDirection.SOURCE_TO_TARGET  # Default based on temporal precedence
        
        return CausalDirection.NONE
    
    async def perform_root_cause_analysis(
        self,
        user_id: str,
        problem_event_id: str
    ) -> Dict[str, Any]:
        """
        Perform root cause analysis to find the original cause of a problem.
        
        Args:
            user_id: User ID
            problem_event_id: ID of the problem event to analyze
        
        Returns:
            Dictionary with root cause analysis results
        """
        try:
            logger.info(f"Starting root cause analysis for event {problem_event_id}")
            
            # Call PostgreSQL function to find root causes
            result = self.supabase.rpc(
                'find_root_causes',
                {
                    'p_problem_event_id': problem_event_id,
                    'p_user_id': user_id,
                    'p_max_depth': self.config['max_root_cause_depth']
                }
            ).execute()
            
            if not result.data:
                return {
                    'root_causes': [],
                    'message': 'No root causes found'
                }
            
            root_causes = []
            
            for root_data in result.data:
                # Fetch event details
                root_event = await self._fetch_event_by_id(root_data['root_event_id'], user_id)
                problem_event = await self._fetch_event_by_id(problem_event_id, user_id)
                
                if not root_event or not problem_event:
                    continue
                
                # Calculate impact
                causal_path = root_data['causal_path']
                affected_events = await self._get_affected_events(causal_path, user_id)
                
                total_impact = sum(
                    event.get('amount_usd', 0.0) for event in affected_events
                )
                
                # Build root cause description
                description = self._build_root_cause_description(
                    root_event,
                    problem_event,
                    root_data['path_length']
                )
                
                root_cause = RootCauseAnalysis(
                    problem_event_id=problem_event_id,
                    root_event_id=root_data['root_event_id'],
                    causal_path=causal_path,
                    path_length=root_data['path_length'],
                    total_impact_usd=total_impact,
                    affected_event_count=len(affected_events),
                    affected_event_ids=[e['id'] for e in affected_events],
                    root_cause_description=description,
                    confidence_score=root_data['total_causal_score']
                )
                
                root_causes.append(root_cause)
                
                # Store in database
                await self._store_root_cause_analysis(root_cause, user_id)
            
            self.metrics['root_cause_analyses'] += 1
            
            logger.info(f"✅ Root cause analysis completed: {len(root_causes)} root causes found")
            
            return {
                'root_causes': [asdict(rc) for rc in root_causes],
                'total_found': len(root_causes),
                'message': f'Root cause analysis completed: {len(root_causes)} root causes identified'
            }
            
        except Exception as e:
            logger.error(f"Root cause analysis failed: {e}")
            return {
                'root_causes': [],
                'error': str(e),
                'message': 'Root cause analysis failed'
            }
    
    async def perform_counterfactual_analysis(
        self,
        user_id: str,
        intervention_event_id: str,
        intervention_type: str,
        counterfactual_value: Any,
        scenario_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform counterfactual "what-if" analysis.
        
        Args:
            user_id: User ID
            intervention_event_id: Event to modify
            intervention_type: Type of intervention (amount_change, date_change, etc.)
            counterfactual_value: New value for the intervention
            scenario_name: Optional name for the scenario
        
        Returns:
            Dictionary with counterfactual analysis results
        """
        try:
            logger.info(f"Starting counterfactual analysis for event {intervention_event_id}")
            
            # Fetch intervention event
            intervention_event = await self._fetch_event_by_id(intervention_event_id, user_id)
            
            if not intervention_event:
                return {
                    'error': 'Intervention event not found',
                    'message': 'Counterfactual analysis failed'
                }
            
            # Get original value
            original_value = self._get_event_value(intervention_event, intervention_type)
            
            # Build causal graph if not already built
            if not self.causal_graph:
                await self._build_causal_graph(user_id)
            
            # Find downstream affected events
            affected_events = await self._propagate_counterfactual(
                intervention_event_id,
                intervention_type,
                original_value,
                counterfactual_value,
                user_id
            )
            
            # Calculate total impact delta
            total_impact_delta = sum(
                event.get('impact_delta_usd', 0.0) for event in affected_events
            )
            
            # Build scenario description
            scenario_description = self._build_counterfactual_description(
                intervention_event,
                intervention_type,
                original_value,
                counterfactual_value,
                affected_events
            )
            
            counterfactual = CounterfactualAnalysis(
                intervention_event_id=intervention_event_id,
                intervention_type=intervention_type,
                original_value=original_value,
                counterfactual_value=counterfactual_value,
                affected_events=affected_events,
                total_impact_delta_usd=total_impact_delta,
                scenario_description=scenario_description
            )
            
            # Store in database
            await self._store_counterfactual_analysis(counterfactual, user_id, scenario_name)
            
            self.metrics['counterfactual_analyses'] += 1
            
            logger.info(f"✅ Counterfactual analysis completed: {len(affected_events)} events affected")
            
            return {
                'counterfactual_analysis': asdict(counterfactual),
                'affected_event_count': len(affected_events),
                'total_impact_delta_usd': total_impact_delta,
                'message': 'Counterfactual analysis completed successfully'
            }
            
        except Exception as e:
            logger.error(f"Counterfactual analysis failed: {e}")
            return {
                'error': str(e),
                'message': 'Counterfactual analysis failed'
            }
    
    async def _build_causal_graph(self, user_id: str):
        """Build directed causal graph from causal relationships"""
        try:
            # Fetch all causal relationships
            result = self.supabase.table('causal_relationships').select(
                'relationship_id, causal_score, is_causal, causal_direction'
            ).eq('user_id', user_id).eq('is_causal', True).execute()
            
            if not result.data:
                self.causal_graph = ig.Graph(directed=True)
                return
            
            # Fetch relationship details
            rel_ids = [r['relationship_id'] for r in result.data]
            relationships = self.supabase.table('relationship_instances').select(
                'id, source_event_id, target_event_id'
            ).in_('id', rel_ids).execute()
            
            # Build graph with igraph
            self.causal_graph = ig.Graph(directed=True)
            
            # Collect unique vertices
            vertices = set()
            edges = []
            edge_attrs = {'relationship_id': [], 'confidence': [], 'causal_score': []}
            
            for rel in relationships.data:
                source = rel['source_event_id']
                target = rel['target_event_id']
                vertices.add(source)
                vertices.add(target)
                edges.append((source, target))
                edge_attrs['relationship_id'].append(rel['id'])
                edge_attrs['confidence'].append(rel.get('confidence_score', 0.0))
                edge_attrs['causal_score'].append(rel.get('causal_score', 0.0))
            
            # Add vertices
            vertex_list = list(vertices)
            self.causal_graph.add_vertices(vertex_list)
            
            # Add edges with attributes
            self.causal_graph.add_edges(edges)
            for attr_name, attr_values in edge_attrs.items():
                self.causal_graph.es[attr_name] = attr_values
            
            logger.info(f"✅ Causal graph built: {self.causal_graph.vcount()} nodes, {self.causal_graph.ecount()} edges")
            
        except Exception as e:
            logger.error(f"Failed to build causal graph: {e}")
            self.causal_graph = ig.Graph(directed=True)
    
    async def _propagate_counterfactual(
        self,
        intervention_event_id: str,
        intervention_type: str,
        original_value: Any,
        counterfactual_value: Any,
        user_id: str
    ) -> List[Dict[str, Any]]:
        """Propagate counterfactual changes through causal graph"""
        try:
            if not self.causal_graph or intervention_event_id not in self.causal_graph:
                return []
            
            affected_events = []
            
            # Find all downstream events using BFS
            try:
                vertex_idx = self.causal_graph.vs.find(name=intervention_event_id).index
                descendants_idx = self.causal_graph.subcomponent(vertex_idx, mode='out')
                descendants = [self.causal_graph.vs[idx]['name'] for idx in descendants_idx if idx != vertex_idx]
            except ValueError:
                descendants = []
            
            for event_id in descendants:
                # Fetch event
                event = await self._fetch_event_by_id(event_id, user_id)
                
                if not event:
                    continue
                
                # Calculate impact based on intervention type
                impact_delta = self._calculate_counterfactual_impact(
                    intervention_type,
                    original_value,
                    counterfactual_value,
                    event
                )
                
                affected_events.append({
                    'event_id': event_id,
                    'document_type': event.get('document_type'),
                    'amount_usd': event.get('amount_usd'),
                    'impact_delta_usd': impact_delta,
                    'vendor': event.get('vendor_standard')
                })
            
            return affected_events
            
        except Exception as e:
            logger.error(f"Failed to propagate counterfactual: {e}")
            return []
    
    def _calculate_counterfactual_impact(
        self,
        intervention_type: str,
        original_value: Any,
        counterfactual_value: Any,
        affected_event: Dict[str, Any]
    ) -> float:
        """Calculate impact of counterfactual on downstream event"""
        
        if intervention_type == 'amount_change':
            # Simple proportional impact
            if isinstance(original_value, (int, float)) and original_value != 0:
                delta_ratio = (counterfactual_value - original_value) / original_value
                event_amount = affected_event.get('amount_usd', 0.0)
                return event_amount * delta_ratio
        
        elif intervention_type == 'date_change':
            # Date changes might affect interest, penalties, etc.
            # Simplified calculation
            return 0.0
        
        return 0.0
    
    def _get_event_value(self, event: Dict[str, Any], intervention_type: str) -> Any:
        """Get the value to be modified from an event"""
        
        if intervention_type == 'amount_change':
            return event.get('amount_usd', 0.0)
        elif intervention_type == 'date_change':
            return event.get('source_ts')
        
        return None
    
    def _build_root_cause_description(
        self,
        root_event: Dict[str, Any],
        problem_event: Dict[str, Any],
        path_length: int
    ) -> str:
        """Build natural language description of root cause"""
        
        root_type = root_event.get('document_type', 'event')
        problem_type = problem_event.get('document_type', 'event')
        root_vendor = root_event.get('vendor_standard', 'unknown')
        
        return (
            f"Root cause: {root_type} from {root_vendor} "
            f"led to {problem_type} through {path_length} causal steps"
        )
    
    def _build_counterfactual_description(
        self,
        intervention_event: Dict[str, Any],
        intervention_type: str,
        original_value: Any,
        counterfactual_value: Any,
        affected_events: List[Dict[str, Any]]
    ) -> str:
        """Build natural language description of counterfactual scenario"""
        
        event_type = intervention_event.get('document_type', 'event')
        
        if intervention_type == 'amount_change':
            return (
                f"If {event_type} amount was ${counterfactual_value:,.2f} instead of ${original_value:,.2f}, "
                f"it would affect {len(affected_events)} downstream events"
            )
        elif intervention_type == 'date_change':
            return (
                f"If {event_type} date was {counterfactual_value} instead of {original_value}, "
                f"it would affect {len(affected_events)} downstream events"
            )
        
        return f"Counterfactual scenario affecting {len(affected_events)} events"
    
    async def _fetch_relationships(
        self,
        user_id: str,
        relationship_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Fetch relationships to analyze"""
        try:
            query = self.supabase.table('relationship_instances').select(
                'id, source_event_id, target_event_id, relationship_type, '
                'confidence_score, temporal_causality, business_logic'
            ).eq('user_id', user_id)
            
            if relationship_ids:
                query = query.in_('id', relationship_ids)
            
            result = query.execute()
            return result.data if result.data else []
            
        except Exception as e:
            logger.error(f"Failed to fetch relationships: {e}")
            return []
    
    async def _fetch_event_by_id(
        self,
        event_id: str,
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        """Fetch single event by ID"""
        try:
            result = self.supabase.table('raw_events').select(
                'id, document_type, amount_usd, source_ts, vendor_standard, payload'
            ).eq('id', event_id).eq('user_id', user_id).execute()
            
            return result.data[0] if result.data else None
            
        except Exception as e:
            logger.error(f"Failed to fetch event: {e}")
            return None
    
    async def _get_affected_events(
        self,
        causal_path: List[str],
        user_id: str
    ) -> List[Dict[str, Any]]:
        """Get events in causal path"""
        try:
            result = self.supabase.table('raw_events').select(
                'id, document_type, amount_usd, vendor_standard'
            ).in_('id', causal_path).eq('user_id', user_id).execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            logger.error(f"Failed to fetch affected events: {e}")
            return []
    
    async def _store_causal_relationship(
        self,
        causal_rel: CausalRelationship,
        user_id: str
    ):
        """Store causal relationship in database"""
        try:
            data = {
                'user_id': user_id,
                'relationship_id': causal_rel.relationship_id,
                'temporal_precedence_score': causal_rel.bradford_hill_scores.temporal_precedence,
                'strength_score': causal_rel.bradford_hill_scores.strength,
                'consistency_score': causal_rel.bradford_hill_scores.consistency,
                'specificity_score': causal_rel.bradford_hill_scores.specificity,
                'dose_response_score': causal_rel.bradford_hill_scores.dose_response,
                'plausibility_score': causal_rel.bradford_hill_scores.plausibility,
                'causal_score': causal_rel.bradford_hill_scores.causal_score,
                'is_causal': causal_rel.is_causal,
                'causal_direction': causal_rel.causal_direction.value,
                'criteria_details': causal_rel.criteria_details
            }
            
            self.supabase.table('causal_relationships').upsert(data).execute()
            
        except Exception as e:
            logger.error(f"Failed to store causal relationship: {e}")
    
    async def _store_root_cause_analysis(
        self,
        root_cause: RootCauseAnalysis,
        user_id: str
    ):
        """Store root cause analysis in database"""
        try:
            data = {
                'user_id': user_id,
                'problem_event_id': root_cause.problem_event_id,
                'root_event_id': root_cause.root_event_id,
                'causal_path': root_cause.causal_path,
                'path_length': root_cause.path_length,
                'total_impact_usd': root_cause.total_impact_usd,
                'affected_event_count': root_cause.affected_event_count,
                'affected_event_ids': root_cause.affected_event_ids,
                'root_cause_description': root_cause.root_cause_description,
                'confidence_score': root_cause.confidence_score
            }
            
            self.supabase.table('root_cause_analyses').insert(data).execute()
            
        except Exception as e:
            logger.error(f"Failed to store root cause analysis: {e}")
    
    async def _store_counterfactual_analysis(
        self,
        counterfactual: CounterfactualAnalysis,
        user_id: str,
        scenario_name: Optional[str] = None
    ):
        """Store counterfactual analysis in database"""
        try:
            data = {
                'user_id': user_id,
                'intervention_event_id': counterfactual.intervention_event_id,
                'intervention_type': counterfactual.intervention_type,
                'original_value': counterfactual.original_value,
                'counterfactual_value': counterfactual.counterfactual_value,
                'affected_events': counterfactual.affected_events,
                'total_impact_delta_usd': counterfactual.total_impact_delta_usd,
                'scenario_description': counterfactual.scenario_description,
                'scenario_name': scenario_name or f"Scenario {datetime.utcnow().isoformat()}"
            }
            
            self.supabase.table('counterfactual_analyses').insert(data).execute()
            
        except Exception as e:
            logger.error(f"Failed to store counterfactual analysis: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get engine metrics"""
        return self.metrics.copy()
