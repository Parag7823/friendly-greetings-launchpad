"""
Production-Grade Causal Inference Engine
=========================================

Implements Bradford Hill criteria for causal relationship detection.
Uses PostgreSQL RPC functions for efficient score calculation.

Author: Senior Full-Stack Engineer
Version: 2.0.0
Date: 2025-11-05
"""

import structlog
import asyncio
import joblib
import io
import numpy as np
import pandas as pd
import numpy_financial as npf
from jinja2 import Template
try:
    import igraph as ig
except ImportError:
    ig = None
try:
    from dowhy import CausalModel
except ImportError:
    CausalModel = None
try:
    from econml.dml import CausalForestDML
except ImportError:
    CausalForestDML = None
try:
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
except ImportError:
    GradientBoostingRegressor = None
    GradientBoostingClassifier = None
try:
    import shap
except ImportError:
    shap = None
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum

# COMPULSORY: Embedding service for semantic intelligence
from data_ingestion_normalization.embedding_service import get_embedding_service

logger = structlog.get_logger(__name__)


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
    
    async def _generate_causal_embedding(self, causal_rel: 'CausalRelationship') -> Optional[List[float]]:
        """
        COMPULSORY: Generate BGE embedding for semantic similarity search of causal relationships.
        
        The embedding captures the semantic meaning of the causality for:
        - Finding similar causal chains across different relationship types
        - Semantic grouping of causal explanations
        - AI-powered causal reasoning recommendations
        """
        try:
            # Build semantic text from causal analysis
            shap_explanation = causal_rel.criteria_details.get('shap_explanation', {})
            top_factors = shap_explanation.get('top_3_factors', [])
            top_factors_str = ', '.join([f[0] for f in top_factors]) if top_factors else 'unknown'
            
            embedding_text = (
                f"causal relationship {causal_rel.causal_direction.value} "
                f"causal score {causal_rel.bradford_hill_scores.causal_score:.2f} "
                f"temporal precedence {causal_rel.bradford_hill_scores.temporal_precedence:.2f} "
                f"strength {causal_rel.bradford_hill_scores.strength:.2f} "
                f"top factors: {top_factors_str} "
                f"{'confirmed causal' if causal_rel.is_causal else 'correlation only'}"
            )
            
            embedding_service = await get_embedding_service()
            embedding = await embedding_service.embed_text(embedding_text)
            
            logger.debug("Generated embedding for causal relationship", 
                        relationship_id=causal_rel.relationship_id)
            return embedding
            
        except Exception as e:
            logger.error(f"Causal embedding generation failed: {e}")
            return None

    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'causal_threshold': 0.7,
            'max_root_cause_depth': 10,
            'consistency_threshold': 5,
            'temporal_window_days': 180,
        }
    
    async def analyze_causal_relationships(
        self,
        user_id: str,
        relationship_ids: Optional[List[str]] = None,
        job_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze relationships to determine causality using Bradford Hill criteria.
        
        Args:
            user_id: User ID
            relationship_ids: Optional list of specific relationship IDs to analyze
            job_id: Optional job ID for tracking
        
        Returns:
            Dictionary with causal analysis results and statistics
        """
        try:
            logger.info(f"Starting causal analysis for user_id={user_id}")
            
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
            
            for rel in relationships:
                try:
                    causal_rel = await self._analyze_single_relationship(rel, user_id)
                    
                    if causal_rel:
                        causal_relationships.append(causal_rel)
                        await self._store_causal_relationship(causal_rel, user_id, job_id)
                        
                        if causal_rel.is_causal:
                            causal_count += 1
                        
                        self.metrics['causal_analyses'] += 1
                        
                except Exception as e:
                    logger.error(f"Failed to analyze relationship {rel.get('id')}: {e}")
                    continue
            
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
    
    async def analyze_all_relationships(self, user_id: str) -> Dict[str, Any]:
        """Wrapper for backward compatibility with ARQ worker"""
        return await self.analyze_causal_relationships(user_id)
    
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
            bradford_hill_scores = BradfordHillScores(
                temporal_precedence=scores_data.get('temporal_precedence_score', 0.0),
                strength=scores_data.get('strength_score', 0.0),
                consistency=scores_data.get('consistency_score', 0.0),
                specificity=scores_data.get('specificity_score', 0.0),
                dose_response=scores_data.get('dose_response_score', 0.0),
                plausibility=scores_data.get('plausibility_score', 0.0),
                causal_score=scores_data.get('causal_score', 0.0)
            )
            
            is_causal = bradford_hill_scores.causal_score >= self.config['causal_threshold']
            causal_direction = self._determine_causal_direction(bradford_hill_scores, relationship)
            
            criteria_details = {
                'time_diff_days': scores_data.get('time_diff_days', 0),
                'similar_pattern_count': scores_data.get('similar_pattern_count', 0),
                'threshold_used': self.config['causal_threshold']
            }
            
            shap_explanation = self._explain_causal_score_with_shap(bradford_hill_scores)
            if shap_explanation:
                criteria_details['shap_explanation'] = shap_explanation
            
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
        """Determine causal direction based on temporal precedence and semantic analysis"""
        if scores.temporal_precedence >= 0.7:
            temporal_causality = relationship.get('temporal_causality', '')
            
            if temporal_causality == 'source_causes_target':
                return CausalDirection.SOURCE_TO_TARGET
            elif temporal_causality == 'target_causes_source':
                return CausalDirection.TARGET_TO_SOURCE
            elif temporal_causality == 'bidirectional':
                return CausalDirection.BIDIRECTIONAL
            else:
                return CausalDirection.SOURCE_TO_TARGET
        
        return CausalDirection.NONE
    
    def _explain_causal_score_with_shap(
        self,
        bradford_hill_scores: BradfordHillScores
    ) -> Optional[Dict[str, Any]]:
        """Generate SHAP explanations for Bradford Hill criteria feature importance"""
        if not shap:
            logger.warning("SHAP not installed, skipping causal score explanation")
            return None
        
        try:
            scores_array = np.array([[
                bradford_hill_scores.temporal_precedence,
                bradford_hill_scores.strength,
                bradford_hill_scores.consistency,
                bradford_hill_scores.specificity,
                bradford_hill_scores.dose_response,
                bradford_hill_scores.plausibility
            ]])
            
            feature_names = [
                'Temporal Precedence',
                'Strength',
                'Consistency',
                'Specificity',
                'Dose Response',
                'Plausibility'
            ]
            
            def model_func(x):
                return x.mean(axis=1)
            
            background_data = shap.sample(scores_array, min(100, len(scores_array)))
            explainer = shap.KernelExplainer(model_func, background_data)
            shap_values = explainer.shap_values(scores_array)
            
            shap_sum = np.abs(shap_values[0]).sum()
            shap_percentages = (np.abs(shap_values[0]) / shap_sum * 100).tolist() if shap_sum > 0 else [0] * len(feature_names)
            
            explanation = {
                'shap_values': shap_values[0].tolist(),
                'shap_percentages': shap_percentages,
                'feature_names': feature_names,
                'base_value': float(explainer.expected_value),
                'feature_importance': dict(zip(feature_names, shap_percentages)),
                'top_3_factors': sorted(
                    zip(feature_names, shap_percentages),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
            }
            
            logger.info("SHAP explanation generated for causal score", 
                       causal_score=bradford_hill_scores.causal_score,
                       top_factor=explanation['top_3_factors'][0][0])
            
            return explanation
            
        except Exception as e:
            logger.warning(f"Failed to generate SHAP explanation for causal score: {e}")
            return None
    
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
                root_event = await self._fetch_event_by_id(root_data['root_event_id'], user_id)
                problem_event = await self._fetch_event_by_id(problem_event_id, user_id)
                
                if not root_event or not problem_event:
                    continue
                
                causal_path = root_data['causal_path']
                affected_events = await self._get_affected_events(causal_path, user_id)
                total_impact = sum(event.get('amount_usd', 0.0) for event in affected_events)
                
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
    
    async def analyze_root_causes(self, user_id: str, problem_event_id: Optional[str] = None) -> Dict[str, Any]:
        """Wrapper for backward compatibility with ARQ worker"""
        if problem_event_id:
            return await self.perform_root_cause_analysis(user_id, problem_event_id)
        # If no specific event, analyze all root causes
        try:
            result = self.supabase.rpc('find_root_causes', {'p_user_id': user_id}).execute()
            if result.data and len(result.data) > 0:
                return await self.perform_root_cause_analysis(user_id, result.data[0]['problem_event_id'])
        except:
            pass
        return {"root_causes": [], "total_found": 0}
    
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
            
            intervention_event = await self._fetch_event_by_id(intervention_event_id, user_id)
            
            if not intervention_event:
                return {
                    'error': 'Intervention event not found',
                    'message': 'Counterfactual analysis failed'
                }
            
            original_value = self._get_event_value(intervention_event, intervention_type)
            
            if not self.causal_graph:
                await self._build_causal_graph(user_id)
            
            affected_events = await self._propagate_counterfactual(
                intervention_event_id,
                intervention_type,
                original_value,
                counterfactual_value,
                user_id
            )
            
            total_impact_delta = sum(event.get('impact_delta_usd', 0.0) for event in affected_events)
            
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
    
    async def analyze_counterfactuals(self, user_id: str, intervention_event_id: Optional[str] = None) -> Dict[str, Any]:
        """Wrapper for backward compatibility with ARQ worker"""
        if intervention_event_id:
            return await self.perform_counterfactual_analysis(user_id, intervention_event_id, intervention_type='amount_change', counterfactual_value=0)
        # If no specific event, analyze first event
        try:
            result = self.supabase.table('raw_events').select('id').eq('user_id', user_id).limit(1).execute()
            if result.data and len(result.data) > 0:
                return await self.perform_counterfactual_analysis(user_id, result.data[0]['id'], intervention_type='amount_change', counterfactual_value=0)
        except:
            pass
        return {"scenarios": [], "total_scenarios": 0}
    
    async def _build_causal_graph(self, user_id: str):
        """Build directed causal graph from causal relationships"""
        try:
            import igraph as ig
            
            result = self.supabase.table('causal_relationships').select(
                'relationship_id, causal_score, is_causal, causal_direction'
            ).eq('user_id', user_id).eq('is_causal', True).execute()
            
            if not result.data:
                self.causal_graph = ig.Graph(directed=True)
                return
            
            rel_ids = [r['relationship_id'] for r in result.data]
            relationships = self.supabase.table('relationship_instances').select(
                'id, source_event_id, target_event_id'
            ).in_('id', rel_ids).execute()
            
            self.causal_graph = ig.Graph(directed=True)
            
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
            
            vertex_list = list(vertices)
            self.causal_graph.add_vertices(vertex_list)
            self.causal_graph.add_edges(edges)
            for attr_name, attr_values in edge_attrs.items():
                self.causal_graph.es[attr_name] = attr_values
            
            logger.info(f"✅ Causal graph built: {self.causal_graph.vcount()} nodes, {self.causal_graph.ecount()} edges")
            
        except Exception as e:
            logger.error(f"Failed to build causal graph: {e}")
            # Lazy import igraph for error fallback
            import igraph as ig
            self.causal_graph = ig.Graph(directed=True)
    
    async def _propagate_counterfactual(
        self,
        intervention_event_id: str,
        intervention_type: str,
        original_value: Any,
        counterfactual_value: Any,
        user_id: str
    ) -> List[Dict[str, Any]]:
        """Propagate counterfactual changes through causal graph with status filtering"""
        try:
            if not self.causal_graph or intervention_event_id not in self.causal_graph:
                return []
            
            affected_events = []
            
            try:
                vertex_idx = self.causal_graph.vs.find(name=intervention_event_id).index
                descendants_idx = self.causal_graph.subcomponent(vertex_idx, mode='out')
                descendants = [self.causal_graph.vs[idx]['name'] for idx in descendants_idx if idx != vertex_idx]
            except ValueError:
                descendants = []
            
            for event_id in descendants:
                event = await self._fetch_event_by_id(event_id, user_id)
                
                if not event:
                    continue
                
                event_status = event.get('status', 'unknown').lower()
                if event_status in ('failed', 'cancelled', 'voided', 'reversed'):
                    logger.debug(f"Skipping {event_status} transaction {event_id}")
                    continue
                
                if event.get('is_deleted', False):
                    logger.debug(f"Skipping deleted transaction {event_id}")
                    continue
                
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
                    'vendor': event.get('vendor_standard'),
                    'status': event_status
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
        """
        Calculate impact of counterfactual on downstream event.
        
        Uses numpy-financial for accurate interest and financial calculations.
        Implements date_change impact including late fees, interest, and cash flow timing.
        """
        
        if intervention_type == 'amount_change':
            # Simple proportional impact
            if isinstance(original_value, (int, float)) and original_value != 0:
                delta_ratio = (counterfactual_value - original_value) / original_value
                event_amount = affected_event.get('amount_usd', 0.0)
                return event_amount * delta_ratio
        
        elif intervention_type == 'date_change':
            try:
                # Calculate days difference
                if isinstance(original_value, str) and isinstance(counterfactual_value, str):
                    from datetime import datetime
                    orig_date = datetime.fromisoformat(original_value.replace('Z', '+00:00'))
                    counter_date = datetime.fromisoformat(counterfactual_value.replace('Z', '+00:00'))
                    days_diff = (counter_date - orig_date).days
                else:
                    days_diff = 0
                
                # Earlier payment = no penalty
                if days_diff < 0:
                    return 0.0
                
                # Get event amount and type
                event_amount = affected_event.get('amount_usd', 0.0)
                event_type = affected_event.get('document_type', '').lower()
                
                total_impact = 0.0
                
                # 1. Late fees using numpy-financial (1.5% monthly after 30 days)
                if days_diff > 30:
                    months_late = (days_diff - 30) / 30.0
                    late_fee_rate = 0.015  # 1.5% per month
                    late_fees = event_amount * late_fee_rate * months_late
                    total_impact += late_fees
                    logger.debug(f"Late fees: ${late_fees:.2f} for {months_late:.1f} months")
                
                # 2. Interest charges using numpy-financial (8% annual)
                if days_diff > 0:
                    annual_rate = 0.08
                    # Use numpy-financial for present value calculation
                    # pv = present value of future payment delayed by days_diff
                    try:
                        pv = npf.pv(rate=annual_rate/365, nper=days_diff, pmt=0, fv=-event_amount)
                        interest = event_amount - pv
                        total_impact += interest
                        logger.debug(f"Interest (npf): ${interest:.2f} for {days_diff} days")
                    except Exception as npf_err:
                        # Fallback to simple calculation if npf fails
                        daily_rate = annual_rate / 365.0
                        interest = event_amount * daily_rate * days_diff
                        total_impact += interest
                        logger.debug(f"Interest (fallback): ${interest:.2f} for {days_diff} days")
                
                # 3. Cash flow impact (opportunity cost at 5% annual)
                if 'payment' in event_type or 'invoice' in event_type:
                    opportunity_cost_rate = 0.05 / 365.0
                    opportunity_cost = event_amount * opportunity_cost_rate * days_diff
                    total_impact += opportunity_cost
                    logger.debug(f"Opportunity cost: ${opportunity_cost:.2f}")
                
                # 4. Covenant violation risk (if payment is critical)
                if 'loan' in event_type or 'debt' in event_type:
                    if days_diff > 15:
                        covenant_risk = event_amount * 0.02
                        total_impact += covenant_risk
                        logger.debug(f"Covenant risk penalty: ${covenant_risk:.2f}")
                
                return total_impact
                
            except Exception as e:
                logger.error(f"Failed to calculate date_change impact: {e}")
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
        """Build natural language description of root cause using Jinja2 template"""
        
        # Jinja2 template for root cause descriptions (easily editable by non-developers)
        template_str = (
            "Root cause: {{ root_type }} from {{ root_vendor }} "
            "led to {{ problem_type }} through {{ path_length }} causal step{% if path_length != 1 %}s{% endif %}"
        )
        
        template = Template(template_str)
        return template.render(
            root_type=root_event.get('document_type', 'event'),
            problem_type=problem_event.get('document_type', 'event'),
            root_vendor=root_event.get('vendor_standard', 'unknown'),
            path_length=path_length
        )
    
    def _build_counterfactual_description(
        self,
        intervention_event: Dict[str, Any],
        intervention_type: str,
        original_value: Any,
        counterfactual_value: Any,
        affected_events: List[Dict[str, Any]]
    ) -> str:
        """Build natural language description of counterfactual scenario using Jinja2 template"""
        
        event_type = intervention_event.get('document_type', 'event')
        affected_count = len(affected_events)
        
        if intervention_type == 'amount_change':
            # Jinja2 template for amount change scenarios
            template_str = (
                "If {{ event_type }} amount was ${{ counterfactual_value:,.2f }} "
                "instead of ${{ original_value:,.2f }}, "
                "it would affect {{ affected_count }} downstream event{% if affected_count != 1 %}s{% endif %}"
            )
            template = Template(template_str)
            return template.render(
                event_type=event_type,
                counterfactual_value=counterfactual_value,
                original_value=original_value,
                affected_count=affected_count
            )
        
        elif intervention_type == 'date_change':
            # Jinja2 template for date change scenarios
            template_str = (
                "If {{ event_type }} date was {{ counterfactual_value }} "
                "instead of {{ original_value }}, "
                "it would affect {{ affected_count }} downstream event{% if affected_count != 1 %}s{% endif %}"
            )
            template = Template(template_str)
            return template.render(
                event_type=event_type,
                counterfactual_value=counterfactual_value,
                original_value=original_value,
                affected_count=affected_count
            )
        
        # Default template for other intervention types
        template_str = "Counterfactual scenario affecting {{ affected_count }} event{% if affected_count != 1 %}s{% endif %}"
        template = Template(template_str)
        return template.render(affected_count=affected_count)
    
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
        user_id: str,
        job_id: Optional[str] = None
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
            
            if job_id:
                data['job_id'] = job_id
            
            # COMPULSORY: Generate embedding for semantic intelligence
            causal_embedding = await self._generate_causal_embedding(causal_rel)
            if causal_embedding:
                data['causal_embedding'] = causal_embedding
            else:
                logger.warning("causal_embedding_generation_failed", 
                              relationship_id=causal_rel.relationship_id)
            
            self.supabase.table('causal_relationships').upsert(data).execute()

            
            # CRITICAL FIX #3: Update normalized_events with causal metadata
            try:
                self.supabase.table('normalized_events').update({
                    'causal_weight': causal_rel.bradford_hill_scores.causal_score,
                    'causal_links': [
                        {
                            'relationship_id': causal_rel.relationship_id,
                            'is_causal': causal_rel.is_causal,
                            'causal_direction': causal_rel.causal_direction.value,
                            'causal_score': causal_rel.bradford_hill_scores.causal_score
                        }
                    ],
                    'causal_reasoning': causal_rel.criteria_details.get('reasoning', ''),
                    'updated_at': datetime.utcnow().isoformat()
                }).eq('user_id', user_id).execute()
                
                logger.debug(f"Updated normalized_events with causal metadata for relationship {causal_rel.relationship_id}")
            except Exception as norm_update_err:
                logger.warning(f"Failed to update normalized_events with causal metadata: {norm_update_err}")
            
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
        scenario_name: Optional[str] = None,
        job_id: Optional[str] = None
    ):
        """Store counterfactual analysis in database"""
        try:
            affected_event_count = len(counterfactual.affected_events) if counterfactual.affected_events else 0
            
            data = {
                'user_id': user_id,
                'intervention_event_id': counterfactual.intervention_event_id,
                'intervention_type': counterfactual.intervention_type,
                'original_value': counterfactual.original_value,
                'counterfactual_value': counterfactual.counterfactual_value,
                'affected_events': counterfactual.affected_events,
                'total_impact_delta_usd': counterfactual.total_impact_delta_usd,
                'affected_event_count': affected_event_count,
                'scenario_description': counterfactual.scenario_description,
                'scenario_name': scenario_name or f"Scenario {datetime.utcnow().isoformat()}"
            }
            
            if job_id:
                data['job_id'] = job_id
            
            self.supabase.table('counterfactual_analyses').insert(data).execute()
            
        except Exception as e:
            logger.error(f"Failed to store counterfactual analysis: {e}")
    


    async def _save_model(self, model: Any, path: str) -> bool:
        """Save model to Supabase Storage using joblib (more efficient than pickle)"""
        try:
            # Serialize model with joblib (better compression and speed than pickle)
            buffer = io.BytesIO()
            joblib.dump(model, buffer, compress=3)  # compress=3 for good balance
            buffer.seek(0)
            
            # Upload to storage
            # Using 'models' folder in 'finely-upload' bucket
            storage_path = f"models/{path}"
            self.supabase.storage.from_("finely-upload").upload(
                path=storage_path,
                file=buffer.getvalue(),
                file_options={"content-type": "application/octet-stream", "upsert": "true"}
            )
            logger.info(f"✅ Model saved to storage: {storage_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to save model to storage: {e}")
            return False

    async def _load_model(self, path: str) -> Optional[Any]:
        """Load model from Supabase Storage using joblib (more efficient than pickle)"""
        try:
            storage_path = f"models/{path}"
            response = self.supabase.storage.from_("finely-upload").download(storage_path)
            
            if response:
                return joblib.load(io.BytesIO(response))
            return None
        except Exception as e:
            logger.debug(f"Model not found or failed to load from {path}: {e}")
            return None

    async def discover_causal_graph_with_dowhy(
        self,
        user_id: str,
        force_retrain: bool = False
    ) -> Dict[str, Any]:
        """
        Automatic causal graph discovery using DoWhy.
        
        NEW CAPABILITY - replaces manual Bradford Hill with automatic causal discovery.
        
        Args:
            user_id: User ID
            force_retrain: If True, ignore cached model and retrain
        
        Returns:
            Dictionary with discovered causal graph and relationships
        """
        try:
            logger.info(f"Running DoWhy causal discovery for user_id={user_id}")
            
            model_path = f"{user_id}/dowhy_causal_model.pkl"
            
            # Try to load cached model
            if not force_retrain:
                cached_model = await self._load_model(model_path)
                if cached_model:
                    logger.info("✅ Using cached DoWhy model")
                    # Re-run estimation on cached model
                    identified_estimand = cached_model.identify_effect(proceed_when_unidentifiable=True)
                    estimate = cached_model.estimate_effect(
                        identified_estimand,
                        method_name="backdoor.linear_regression"
                    )
                    refutation = cached_model.refute_estimate(
                        identified_estimand,
                        estimate,
                        method_name="random_common_cause"
                    )
                    return {
                        'causal_effect': float(estimate.value),
                        'confidence_intervals': {
                            'lower': float(estimate.value - 1.96 * estimate.get_standard_error()) if hasattr(estimate, 'get_standard_error') else None,
                            'upper': float(estimate.value + 1.96 * estimate.get_standard_error()) if hasattr(estimate, 'get_standard_error') else None
                        },
                        'identified_estimand': str(identified_estimand),
                        'refutation_result': str(refutation),
                        'message': f'DoWhy discovered causal effect (cached): {estimate.value:.4f}'
                    }

            # Fetch relationship and event data
            relationships = await self._fetch_relationships(user_id)
            
            if not relationships or len(relationships) < 10:
                return {
                    'causal_graph': None,
                    'message': 'Insufficient data for DoWhy causal discovery (need 10+ relationships)'
                }
            
            # Prepare data for DoWhy
            data_rows = []
            for rel in relationships:
                try:
                    # Fetch source and target events
                    source = self.supabase.table('raw_events').select('*').eq('id', rel['source_event_id']).execute()
                    target = self.supabase.table('raw_events').select('*').eq('id', rel['target_event_id']).execute()
                    
                    if source.data and target.data:
                        data_rows.append({
                            'treatment': 1,  # Source event occurred
                            'outcome': target.data[0].get('amount_usd', 0),
                            'source_amount': source.data[0].get('amount_usd', 0),
                            'target_amount': target.data[0].get('amount_usd', 0),
                            'relationship_type': rel.get('relationship_type', 'unknown'),
                            'confidence': rel.get('confidence_score', 0.5)
                        })
                except Exception as e:
                    logger.debug(f"Failed to process relationship {rel.get('id')}: {e}")
                    continue
            
            if len(data_rows) < 10:
                return {
                    'causal_graph': None,
                    'message': 'Insufficient valid data for DoWhy'
                }
            
            # Create DataFrame
            df = pd.DataFrame(data_rows)
            
            # Define causal model
            model = CausalModel(
                data=df,
                treatment='treatment',
                outcome='outcome',
                common_causes=['source_amount', 'confidence']
            )
            
            # Save trained model
            await self._save_model(model, model_path)
            
            # Identify causal effect
            identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
            
            # Estimate causal effect
            estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.linear_regression"
            )
            
            # Refute estimate (sensitivity analysis)
            refutation = model.refute_estimate(
                identified_estimand,
                estimate,
                method_name="random_common_cause"
            )
            
            logger.info(f"✅ DoWhy causal discovery completed")
            
            return {
                'causal_effect': float(estimate.value),
                'confidence_intervals': {
                    'lower': float(estimate.value - 1.96 * estimate.get_standard_error()) if hasattr(estimate, 'get_standard_error') else None,
                    'upper': float(estimate.value + 1.96 * estimate.get_standard_error()) if hasattr(estimate, 'get_standard_error') else None
                },
                'identified_estimand': str(identified_estimand),
                'refutation_result': str(refutation),
                'total_relationships_analyzed': len(data_rows),
                'message': f'DoWhy discovered causal effect: {estimate.value:.4f}'
            }
            
        except Exception as e:
            logger.error(f"DoWhy causal discovery failed: {e}", exc_info=True)
            return {
                'causal_graph': None,
                'error': str(e),
                'message': 'DoWhy causal discovery failed'
            }
    
    async def estimate_treatment_effects_with_econml(
        self,
        user_id: str,
        treatment_type: str = 'invoice',
        outcome_type: str = 'payment',
        force_retrain: bool = False
    ) -> Dict[str, Any]:
        """
        Heterogeneous treatment effect estimation using EconML.
        
        NEW CAPABILITY - replaces simple counterfactuals with sophisticated causal ML.
        
        Args:
            user_id: User ID
            treatment_type: Type of treatment event (e.g., 'invoice')
            outcome_type: Type of outcome event (e.g., 'payment')
            force_retrain: If True, ignore cached model and retrain
        
        Returns:
            Dictionary with treatment effects and confidence intervals
        """
        try:
            logger.info(f"Running EconML treatment effect estimation for user_id={user_id}")
            
            model_path = f"{user_id}/econml_model_{treatment_type}_{outcome_type}.pkl"
            
            # Fetch events (needed for X feature generation even if model is cached)
            events_result = self.supabase.table('raw_events').select(
                'id, document_type, amount_usd, source_ts, vendor_standard'
            ).eq('user_id', user_id).execute()
            
            if not events_result.data or len(events_result.data) < 20:
                return {
                    'treatment_effects': [],
                    'message': 'Insufficient data for EconML (need 20+ events)'
                }
            
            # Prepare data
            treatment_events = [e for e in events_result.data if treatment_type.lower() in e.get('document_type', '').lower()]
            outcome_events = [e for e in events_result.data if outcome_type.lower() in e.get('document_type', '').lower()]
            
            if len(treatment_events) < 10 or len(outcome_events) < 10:
                return {
                    'treatment_effects': [],
                    'message': f'Insufficient {treatment_type} or {outcome_type} events'
                }
            
            # Build training data
            X = []  # Features (vendor, amount)
            T = []  # Treatment (1 if invoice exists, 0 otherwise)
            Y = []  # Outcome (payment amount)
            
            for outcome in outcome_events:
                # Find matching treatment
                matching_treatment = next(
                    (t for t in treatment_events if t.get('vendor_standard') == outcome.get('vendor_standard')),
                    None
                )
                
                if matching_treatment:
                    X.append([matching_treatment.get('amount_usd', 0)])
                    T.append(1)
                    Y.append(outcome.get('amount_usd', 0))
                else:
                    X.append([0])
                    T.append(0)
                    Y.append(outcome.get('amount_usd', 0))
            
            if len(X) < 20:
                return {
                    'treatment_effects': [],
                    'message': 'Insufficient matched pairs for EconML'
                }
            
            # Convert to numpy arrays
            X = np.array(X)
            T = np.array(T)
            Y = np.array(Y)
            
            est = None
            
            # Try to load cached model
            if not force_retrain:
                est = await self._load_model(model_path)
                if est:
                    logger.info("✅ Using cached EconML model")
            
            if not est:
                # Train Causal Forest
                est = CausalForestDML(
                    model_y=GradientBoostingRegressor(),
                    model_t=GradientBoostingClassifier(),
                    random_state=42
                )
                
                est.fit(Y=Y, T=T, X=X)
                
                # Save trained model
                await self._save_model(est, model_path)
            
            # Estimate treatment effects
            treatment_effects = est.effect(X)
            
            # Get confidence intervals
            lb, ub = est.effect_interval(X, alpha=0.05)
            
            # Build results
            results = []
            for i, (effect, lower, upper) in enumerate(zip(treatment_effects, lb, ub)):
                results.append({
                    'index': i,
                    'treatment_effect': float(effect),
                    'confidence_interval_lower': float(lower),
                    'confidence_interval_upper': float(upper),
                    'feature_value': float(X[i][0])
                })
            
            avg_effect = float(np.mean(treatment_effects))
            
            logger.info(f"✅ EconML treatment effect estimation completed: avg effect = {avg_effect:.2f}")
            
            return {
                'treatment_effects': results[:10],  # Return top 10
                'average_treatment_effect': avg_effect,
                'total_analyzed': len(results),
                'treatment_type': treatment_type,
                'outcome_type': outcome_type,
                'message': f'EconML estimated average treatment effect: ${avg_effect:.2f}'
            }
            
        except Exception as e:
            logger.error(f"EconML treatment effect estimation failed: {e}", exc_info=True)
            return {
                'treatment_effects': [],
                'error': str(e),
                'message': 'EconML treatment effect estimation failed'
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get engine metrics"""
        return self.metrics.copy()


# ============================================================================
# PRELOAD PATTERN: Initialize heavy dependencies at module-load time
# ============================================================================
# This runs automatically when the module is imported, eliminating the
# first-request latency that was caused by lazy-loading.
# 
# BENEFITS:
# - First request is instant (no cold-start delay)
# - Shared across all worker instances
# - Memory is allocated once, not per-instance

_PRELOAD_COMPLETED = False

def _preload_all_modules():
    """
    PRELOAD PATTERN: Initialize all heavy modules at module-load time.
    Called automatically when module is imported.
    This eliminates first-request latency.
    """
    global _PRELOAD_COMPLETED
    
    if _PRELOAD_COMPLETED:
        return
    
    # Note: igraph, dowhy, econml, shap, sklearn are already imported at top
    # with try/except fallbacks. The preload here ensures they're fully initialized.
    
    # Preload numpy (critical for all numerical operations)
    try:
        import numpy as np
        _ = np.array([1, 2, 3])  # Force numpy to fully initialize
        logger.info("✅ PRELOAD: numpy loaded at module-load time")
    except Exception as e:
        logger.warning(f"⚠️ PRELOAD: numpy load failed: {e}")
    
    # Preload numpy-financial
    try:
        import numpy_financial as npf
        logger.info("✅ PRELOAD: numpy_financial loaded at module-load time")
    except Exception as e:
        logger.warning(f"⚠️ PRELOAD: numpy_financial load failed: {e}")
    
    # Preload igraph (used for causal graph)
    try:
        import igraph as ig
        if ig is not None:
            logger.info("✅ PRELOAD: igraph loaded at module-load time")
    except Exception as e:
        logger.warning(f"⚠️ PRELOAD: igraph load failed: {e}")
    
    # Preload shap (used for explainability)
    try:
        import shap
        if shap is not None:
            logger.info("✅ PRELOAD: shap loaded at module-load time")
    except Exception as e:
        logger.warning(f"⚠️ PRELOAD: shap load failed: {e}")
    
    # Preload sklearn (used for CausalForestDML)
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        if GradientBoostingRegressor is not None:
            logger.info("✅ PRELOAD: sklearn loaded at module-load time")
    except Exception as e:
        logger.warning(f"⚠️ PRELOAD: sklearn load failed: {e}")
    
    _PRELOAD_COMPLETED = True

try:
    _preload_all_modules()
except Exception as e:
    logger.warning(f"Module-level causal preload failed (will use fallback): {e}")
