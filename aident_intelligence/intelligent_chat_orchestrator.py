"""Intelligent Chat Orchestrator - Routes questions to intelligence engines.

Features: Question classification, intelligent routing, response formatting,
context management, and proactive insights.
"""

import structlog
import json
import os
import sys
import importlib.util
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
import asyncio
from groq import AsyncGroq  # CHANGED: Using Groq instead of Anthropic

from langgraph.graph import StateGraph, END
try:
    from langgraph.types import RetryPolicy
except ImportError:
    # Fallback for newer langgraph versions where RetryPolicy moved
    try:
        from langgraph.pregel import RetryPolicy
    except ImportError:
        # If still not available, create a simple fallback
        class RetryPolicy:
            def __init__(self, max_retries=3, backoff_factor=1.0):
                self.max_retries = max_retries
                self.backoff_factor = backoff_factor
from typing_extensions import TypedDict
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import spacy
from spacy.matcher import PhraseMatcher
from jinja2 import Template

_spacy_nlp = None  # Global spaCy model (loaded once to prevent 500MB+ overhead)

def _load_spacy_model():
    """Load spaCy model with EntityRuler for financial keyword detection."""
    global _spacy_nlp
    if _spacy_nlp is None:
        try:
            from pathlib import Path
            
            _spacy_nlp = spacy.load("en_core_web_sm")
            
            # Load EntityRuler from JSONL patterns file
            patterns_path = Path(__file__).parent / "config" / "financial_patterns.jsonl"
            
            if patterns_path.exists():
                # Add EntityRuler to pipeline
                if "entity_ruler" not in _spacy_nlp.pipe_names:
                    ruler = _spacy_nlp.add_pipe("entity_ruler", before="ner")
                    ruler.from_disk(patterns_path)
                    logger_temp = structlog.get_logger(__name__)
                    logger_temp.info(f"✅ spaCy EntityRuler loaded from {patterns_path}")
            else:
                logger_temp = structlog.get_logger(__name__)
                logger_temp.warning(f"Financial patterns file not found: {patterns_path}")
            
            logger_temp = structlog.get_logger(__name__)
            logger_temp.info("✅ spaCy model loaded with EntityRuler for financial keywords")
        except OSError:
            logger_temp = structlog.get_logger(__name__)
            logger_temp.error("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
    return _spacy_nlp

# Load spaCy model at module import time (one-time cost)
_load_spacy_model()

from aiocache import cached, Cache
from aiocache.serializers import JsonSerializer

# Pydantic models for LangChain's .with_structured_output()
from pydantic import BaseModel, Field
try:
    # Try package layout first
    from aident_cfo_brain.intent_and_guard_engine import (
        IntentClassifier,
        OutputGuard,
        UserIntent,
        get_intent_classifier,
        get_output_guard
    )
except ImportError:
    # Fallback to relative import
    from .intent_and_guard_engine import (
        IntentClassifier,
        OutputGuard,
        UserIntent,
        get_intent_classifier,
        get_output_guard
    )

logger = structlog.get_logger(__name__)

# Package imports with graceful fallback
FinleyGraphEngine = None
AidentMemoryManager = None
CausalInferenceEngine = None
TemporalPatternLearner = None
EnhancedRelationshipDetector = None

try:
    from aident_cfo_brain.finley_graph_engine import FinleyGraphEngine
    from aident_cfo_brain.aident_memory_manager import AidentMemoryManager
    from aident_cfo_brain.causal_inference_engine import CausalInferenceEngine
    from aident_cfo_brain.temporal_pattern_learner import TemporalPatternLearner
    from aident_cfo_brain.enhanced_relationship_detector import EnhancedRelationshipDetector
    logger.info("✅ Intelligence engines imported successfully")
except ImportError as e:
    # Fallback for direct module execution (not in package context)
    try:
        from finley_graph_engine import FinleyGraphEngine
        from aident_memory_manager import AidentMemoryManager
        from causal_inference_engine import CausalInferenceEngine
        from temporal_pattern_learner import TemporalPatternLearner
        from enhanced_relationship_detector import EnhancedRelationshipDetector
        logger.warning("Using direct imports (not in package context)")
    except ImportError as e2:
        logger.warning(f"Intelligence engines not fully available: {e2} - some features will be disabled")
        # Don't raise - allow graceful degradation

try:
    from data_ingestion_normalization.entity_resolver_optimized import EntityResolverOptimized as EntityResolver
except ImportError:
    from entity_resolver_optimized import EntityResolverOptimized as EntityResolver

try:
    from data_ingestion_normalization.embedding_service import EmbeddingService
except ImportError:
    from embedding_service import EmbeddingService

# REFACTOR: Import PromptLoader for externalized prompt management
try:
    from aident_cfo_brain.prompt_loader import get_prompt_loader
except ImportError:
    from prompt_loader import get_prompt_loader



class QuestionType(Enum):
    """Types of questions the system can handle"""
    CAUSAL = "causal"  # Why did X happen?
    TEMPORAL = "temporal"  # When will X happen?
    RELATIONSHIP = "relationship"  # Show connections
    WHAT_IF = "what_if"  # What if scenarios
    EXPLAIN = "explain"  # Explain this number
    GENERAL = "general"  # General financial questions
    DATA_QUERY = "data_query"  # Query raw data
    UNKNOWN = "unknown"  # Couldn't classify


# Pydantic models for type-safe extraction with instructor

class QuestionClassification(BaseModel):
        metrics: List[str] = Field(
            ...,
            description="Specific metrics to analyze (e.g., 'Q1 revenue', 'monthly expenses')"
        )
        time_periods: List[str] = Field(
            default_factory=list,
            description="Time periods mentioned (e.g., 'last quarter', 'this year', 'Q1 2024')"
        )
        confidence: float = Field(
            ...,
            ge=0.0,
            le=1.0,
            description="Confidence score for extraction"
        )
    
class ScenarioExtraction(BaseModel):
        """Type-safe scenario extraction from what-if questions"""
        scenario_type: str = Field(
            ...,
            description="Type of scenario: sensitivity_analysis, forecast, comparison, or impact_analysis"
        )
        base_metric: str = Field(
            ...,
            description="The main metric being analyzed (e.g., 'cash flow', 'revenue')"
        )
        variables: List[str] = Field(
            ...,
            description="Variables being changed in the scenario (e.g., 'payment delay', 'hiring cost')"
        )
        changes: List[str] = Field(
            ...,
            description="Specific changes to apply (e.g., 'delay by 30 days', 'increase by 20%')"
        )
        confidence: float = Field(
            ...,
            ge=0.0,
            le=1.0,
            description="Confidence score for scenario extraction"
        )
    
class QueryParameterExtraction(BaseModel):
        """Type-safe query parameter extraction from data queries"""
        filters: List[str] = Field(
            default_factory=list,
            description="Filter conditions (e.g., 'amount > 1000', 'date in Q1')"
        )
        sort_by: str = Field(
            default="date",
            description="Field to sort by (e.g., 'amount', 'date', 'vendor')"
        )
        limit: int = Field(
            default=100,
            ge=1,
            le=10000,
            description="Maximum number of results to return"
        )
        group_by: List[str] = Field(
            default_factory=list,
            description="Fields to group by (e.g., 'vendor', 'category')"
        )
        confidence: float = Field(
            ...,
            ge=0.0,
            le=1.0,
            description="Confidence score for parameter extraction"
        )
    
class EntityIDExtraction(BaseModel):
        """Type-safe entity ID extraction from questions"""
        entity_type: str = Field(
            ...,
            description="Type of entity (e.g., 'invoice', 'transaction', 'vendor', 'customer')"
        )
        entity_identifier: str = Field(
            ...,
            description="The specific identifier or value (e.g., 'INV-12345', 'Acme Corp', 'TXN-789')"
        )
        search_field: str = Field(
            default="id",
            description="Field to search in (e.g., 'id', 'name', 'reference_number')"
        )
        confidence: float = Field(
            ...,
            ge=0.0,
            le=1.0,
            description="Confidence score for entity identification"
        )
    
    # Intent handler response models
class GreetingResponse(BaseModel):
        """Type-safe greeting response"""
        greeting_message: str = Field(
            ...,
            description="Warm greeting message for the user"
        )
        follow_up: str = Field(
            ...,
            description="Follow-up question or suggestion (e.g., 'How can I help you today?')"
        )
        tone: str = Field(
            default="friendly",
            description="Tone of response (friendly, professional, warm)"
        )
    
class CapabilitySummaryResponse(BaseModel):
        """Type-safe capability summary response"""
        capabilities: List[str] = Field(
            ...,
            description="List of main capabilities (e.g., 'Causal analysis', 'Temporal patterns', 'What-if scenarios')"
        )
        key_features: List[str] = Field(
            ...,
            description="Key features that differentiate the platform"
        )
        next_step: str = Field(
            ...,
            description="Suggested next action for user"
        )
    
class SystemFlowResponse(BaseModel):
        """Type-safe system flow explanation"""
        flow_steps: List[str] = Field(
            ...,
            description="Steps in the system flow (e.g., 'Connect data', 'Ask questions', 'Get insights')"
        )
        current_step: str = Field(
            default="",
            description="Where user is in the flow"
        )
        next_step: str = Field(
            ...,
            description="What to do next"
        )
    
class DifferentiatorResponse(BaseModel):
        """Type-safe differentiator explanation"""
        differentiators: List[str] = Field(
            ...,
            description="Key differentiators from competitors"
        )
        unique_value: str = Field(
            ...,
            description="Unique value proposition"
        )
        proof_point: str = Field(
            ...,
            description="Evidence or proof of differentiation"
        )
    
class HelpResponse(BaseModel):
        """Type-safe help response"""
        help_topics: List[str] = Field(
            ...,
            description="Available help topics"
        )
        suggested_topic: str = Field(
            ...,
            description="Suggested help topic based on context"
        )
        contact_info: str = Field(
            default="",
            description="Contact information if needed"
        )
    
class SmalltalkResponse(BaseModel):
        """Type-safe smalltalk response"""
        response_text: str = Field(
            ...,
            description="Friendly smalltalk response"
        )
        tone: str = Field(
            default="friendly",
            description="Tone of response (friendly, warm, casual)"
        )
        engagement_level: int = Field(
            default=1,
            ge=1,
            le=5,
            description="Engagement level (1-5)"
        )
    
class MetaFeedbackResponse(BaseModel):
        """Type-safe meta feedback response"""
        acknowledgment: str = Field(
            ...,
            description="Acknowledgment of the feedback"
        )
        action_taken: str = Field(
            ...,
            description="What action will be taken based on feedback"
        )
        appreciation: str = Field(
            ...,
            description="Expression of appreciation for feedback"
        )


class DataMode(Enum):
    """Data availability modes for response differentiation"""
    NO_DATA = "no_data"  # User has no data connected yet
    LIMITED_DATA = "limited_data"  # User has <50 transactions
    RICH_DATA = "rich_data"  # User has >50 transactions


class OnboardingState(Enum):
    """Track onboarding state to prevent repetition"""
    FIRST_VISIT = "first_visit"  # User's first interaction
    ONBOARDED = "onboarded"  # User has seen onboarding
    DATA_CONNECTED = "data_connected"  # User has connected data
    ACTIVE = "active"  # User is actively using system


class OrchestratorState(TypedDict, total=False):
    """Unified state for LangGraph orchestrator with automatic persistence."""
    question: str
    user_id: str
    chat_id: Optional[str]
    chat_title: Optional[str]
    context: Optional[Dict[str, Any]]
    intent: str
    intent_confidence: float
    question_type: str
    question_confidence: float
    low_confidence_intent: bool
    low_confidence_question: bool
    confidence_threshold_breached: bool
    conversation_history: List[Dict[str, str]]
    memory_context: str
    memory_messages: List[Dict[str, str]]
    temporal_data: Optional[Dict[str, Any]]
    seasonal_data: Optional[Dict[str, Any]]
    fraud_data: Optional[Dict[str, Any]]
    root_cause_data: Optional[Dict[str, Any]]
    data_mode: str
    response: Any
    # Metadata
    processing_steps: List[str]
    errors: List[str]


def _create_error_response(operation: str, error: str, fallback: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create structured error response with fallback data."""
    return {
        "error": True,
        "operation": operation,
        "message": f"{operation} failed: {error}",
        **(fallback or {})
    }


def _get_fallback_entities(question: str) -> Dict[str, Any]:
    """Extract basic entities using simple keyword matching as fallback."""
    question_lower = question.lower()
    financial_keywords = ["revenue", "expense", "cash flow", "profit", "vendor", "customer", "invoice", "payment"]
    found_metrics = [kw for kw in financial_keywords if kw in question_lower]
    return {
        "entities": [],
        "metrics": found_metrics,
        "time_periods": [],
        "confidence": 0.3 if found_metrics else 0.0
    }


@dataclass
class ChatResponse:
    """Structured response from the orchestrator"""
    answer: str  # Human-readable answer
    question_type: QuestionType
    confidence: float  # 0.0-1.0
    data: Optional[Dict[str, Any]] = None  # Structured data
    actions: Optional[List[Dict[str, Any]]] = None  # Suggested actions
    visualizations: Optional[List[Dict[str, Any]]] = None  # Chart data
    follow_up_questions: Optional[List[str]] = None  # Suggested questions
    error: Optional[str] = None  # Error message if operation partially failed
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'answer': self.answer,
            'question_type': self.question_type.value,
            'confidence': self.confidence,
            'data': self.data,
            'actions': self.actions,
            'visualizations': self.visualizations,
            'follow_up_questions': self.follow_up_questions,
            'timestamp': datetime.utcnow().isoformat()
        }


class NullEmbeddingService:
    """Null object pattern for graceful degradation when EmbeddingService unavailable."""
    def __init__(self):
        self.available = False
    
    async def embed_text(self, text: str):
        """Return None embedding (graceful degradation)"""
        logger.debug("NullEmbeddingService: embed_text called but service unavailable")
        return None
    
    async def embed_batch(self, texts: List[str]):
        """Return None embeddings (graceful degradation)"""
        logger.debug(f"NullEmbeddingService: embed_batch called for {len(texts)} texts but service unavailable")
        return [None] * len(texts)
    
    def similarity(self, emb1, emb2) -> float:
        """Return 0.0 similarity (graceful degradation)"""
        logger.debug("NullEmbeddingService: similarity called but service unavailable")
        return 0.0


class IntelligentChatOrchestrator:
    """Routes questions to intelligence engines and manages orchestration."""
    
    def __init__(self, supabase_client, cache_client=None, groq_client=None, embedding_service=None):
        """Initialize orchestrator with all intelligence engines."""
        if groq_client:
            self.groq = groq_client
        else:
            groq_api_key = os.getenv('GROQ_API_KEY')
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY environment variable is required")
            self.groq = AsyncGroq(api_key=groq_api_key)
        
        self.openai = None  # Not used - using Groq internally
        self.supabase = supabase_client
        self.cache = cache_client
        
        if embedding_service is None:
            try:
                self.embedding_service = EmbeddingService(cache_client=cache_client)
                logger.info("✅ EmbeddingService initialized")
            except Exception as e:
                logger.warning("Failed to initialize EmbeddingService, using NullEmbeddingService fallback", error=str(e))
                self.embedding_service = NullEmbeddingService()
        else:
            self.embedding_service = embedding_service
        
        # Initialize intelligence engines
        self.causal_engine = CausalInferenceEngine(
            supabase_client=supabase_client
        )
        
        self.temporal_learner = TemporalPatternLearner(
            supabase_client=supabase_client
        )
        
        self.relationship_detector = EnhancedRelationshipDetector(
            supabase_client=supabase_client,
            cache_client=cache_client,
            embedding_service=self.embedding_service
        )
        
        self.entity_resolver = EntityResolver(
            supabase_client=supabase_client,
            cache_client=cache_client
        )
        
        self.graph_engine = FinleyGraphEngine(
            supabase=supabase_client,
            redis_url=os.getenv('ARQ_REDIS_URL') or os.getenv('REDIS_URL')
        )
        
        self.intent_classifier = get_intent_classifier()
        self.output_guard = get_output_guard(self.groq, self.embedding_service)
        
        self.prompt_loader = get_prompt_loader()
        logger.info("✅ PromptLoader initialized")
        
        from jinja2 import Environment, FileSystemLoader
        from pathlib import Path
        templates_dir = Path(__file__).parent / "templates"
        self.jinja_env = Environment(loader=FileSystemLoader(str(templates_dir)))
        logger.info("✅ Jinja2 environment initialized")
        
        logger.info("✅ IntelligentChatOrchestrator initialized")
        
        self.graph = self._build_langgraph()
        logger.info("✅ LangGraph state machine compiled")
    
    def _build_langgraph(self):
        """Build LangGraph state machine with retry policies for error recovery."""
        workflow = StateGraph(OrchestratorState)
        
        retry_policy_critical = RetryPolicy(max_attempts=3, backoff_multiplier=2.0)
        retry_policy_standard = RetryPolicy(max_attempts=2, backoff_multiplier=2.0)
        
        workflow.add_node("init_memory", self._node_init_memory)
        workflow.add_node("determine_data_mode", self._node_determine_data_mode)
        workflow.add_node("classify_intent", self._node_classify_intent, retry_policy=retry_policy_critical)
        workflow.add_node("route_by_intent", self._node_route_by_intent)
        workflow.add_node("ask_clarifying_question", self._node_ask_clarifying_question, retry_policy=retry_policy_standard)
        
        workflow.add_node("handle_greeting", self._node_handle_greeting, retry_policy=retry_policy_standard)
        workflow.add_node("handle_smalltalk", self._node_handle_smalltalk, retry_policy=retry_policy_standard)
        workflow.add_node("handle_capability_summary", self._node_handle_capability_summary, retry_policy=retry_policy_standard)
        workflow.add_node("handle_system_flow", self._node_handle_system_flow, retry_policy=retry_policy_standard)
        workflow.add_node("handle_differentiator", self._node_handle_differentiator, retry_policy=retry_policy_standard)
        workflow.add_node("handle_meta_feedback", self._node_handle_meta_feedback, retry_policy=retry_policy_standard)
        workflow.add_node("handle_help", self._node_handle_help, retry_policy=retry_policy_standard)
        
        workflow.add_node("classify_question", self._node_classify_question, retry_policy=retry_policy_critical)
        workflow.add_node("route_by_question_type", self._node_route_by_question_type)
        workflow.add_node("handle_causal", self._node_handle_causal, retry_policy=retry_policy_standard)
        workflow.add_node("handle_temporal", self._node_handle_temporal, retry_policy=retry_policy_standard)
        workflow.add_node("handle_relationship", self._node_handle_relationship, retry_policy=retry_policy_standard)
        workflow.add_node("handle_whatif", self._node_handle_whatif, retry_policy=retry_policy_standard)
        workflow.add_node("handle_explain", self._node_handle_explain, retry_policy=retry_policy_standard)
        workflow.add_node("handle_data_query", self._node_handle_data_query, retry_policy=retry_policy_standard)
        workflow.add_node("handle_general", self._node_handle_general, retry_policy=retry_policy_standard)
        
        workflow.add_node("fetch_temporal_data", self._node_fetch_temporal_data, retry_policy=retry_policy_standard)
        workflow.add_node("fetch_seasonal_data", self._node_fetch_seasonal_data, retry_policy=retry_policy_standard)
        workflow.add_node("fetch_fraud_data", self._node_fetch_fraud_data, retry_policy=retry_policy_standard)
        workflow.add_node("fetch_root_cause_data", self._node_fetch_root_cause_data, retry_policy=retry_policy_standard)
        workflow.add_node("aggregate_parallel_results", self._node_aggregate_parallel_results)
        
        workflow.add_node("validate_response", self._node_validate_response)
        workflow.add_node("enrich_with_graph_intelligence", self._node_enrich_with_graph_intelligence, retry_policy=retry_policy_standard)
        workflow.add_node("store_in_database", self._node_store_in_database, retry_policy=retry_policy_standard)
        
        workflow.add_node("determine_data_mode", self._node_determine_data_mode)
        workflow.add_node("onboarding_handler", self._node_onboarding_handler)
        workflow.add_node("exploration_handler", self._node_exploration_handler)
        workflow.add_node("advanced_handler", self._node_advanced_handler)
        
        workflow.add_node("apply_output_guard", self._node_apply_output_guard)
        workflow.add_node("save_memory", self._node_save_memory)
        
        workflow.set_entry_point("init_memory")
        
        workflow.add_edge("init_memory", "determine_data_mode")
        workflow.add_edge("determine_data_mode", "classify_intent")
        workflow.add_edge("classify_intent", "route_by_intent")
        workflow.add_conditional_edges(
            "route_by_intent",
            lambda state: {
                "greeting": "greeting",
                "smalltalk": "smalltalk",
                "capability_summary": "capability_summary",
                "system_flow": "system_flow",
                "differentiator": "differentiator",
                "meta_feedback": "meta_feedback",
                "help": "help",
            }.get(state.get("intent", "unknown"), "data_analysis"),
            {
                "clarify": "ask_clarifying_question",  # FEATURE #3: Low confidence path
                "greeting": "handle_greeting",
                "smalltalk": "handle_smalltalk",
                "capability_summary": "handle_capability_summary",
                "system_flow": "handle_system_flow",
                "differentiator": "handle_differentiator",
                "meta_feedback": "handle_meta_feedback",
                "help": "handle_help",
                "data_analysis": "classify_question",
            }
        )
        
        # Intent handlers → Output guard
        for handler in ["handle_greeting", "handle_smalltalk", "handle_capability_summary",
                       "handle_system_flow", "handle_differentiator", "handle_meta_feedback", "handle_help"]:
            workflow.add_edge(handler, "apply_output_guard")
        
        workflow.add_edge("ask_clarifying_question", "apply_output_guard")
        workflow.add_edge("classify_question", "route_by_question_type")
        
        workflow.add_conditional_edges(
            "route_by_question_type",
            lambda state: {
                "causal": "causal",
                "temporal": "temporal",
                "relationship": "relationship",
                "what_if": "what_if",
                "explain": "explain",
                "data_query": "data_query",
            }.get(state.get("question_type", "general"), "general"),
            {
                "causal": "handle_causal",
                "temporal": "handle_temporal",
                "relationship": "handle_relationship",
                "what_if": "handle_whatif",
                "explain": "handle_explain",
                "data_query": "handle_data_query",
                "general": "handle_general",
            }
        )
        
        from langgraph.types import Send
        
        def _route_temporal_parallel(state: OrchestratorState):
            """Route to all fetch nodes in parallel."""
            return [
                Send("fetch_temporal_data", state),
                Send("fetch_seasonal_data", state),
                Send("fetch_fraud_data", state),
                Send("fetch_root_cause_data", state),
            ]
        
        workflow.add_conditional_edges(
            "handle_temporal",
            _route_temporal_parallel,
            {
                "fetch_temporal_data": "fetch_temporal_data",
                "fetch_seasonal_data": "fetch_seasonal_data",
                "fetch_fraud_data": "fetch_fraud_data",
                "fetch_root_cause_data": "fetch_root_cause_data",
            }
        )
        
        workflow.add_edge("fetch_temporal_data", "aggregate_parallel_results")
        workflow.add_edge("fetch_seasonal_data", "aggregate_parallel_results")
        workflow.add_edge("fetch_fraud_data", "aggregate_parallel_results")
        workflow.add_edge("fetch_root_cause_data", "aggregate_parallel_results")
        
        workflow.add_edge("aggregate_parallel_results", "apply_output_guard")
        
        for handler in ["handle_causal", "handle_relationship",
                       "handle_whatif", "handle_explain", "handle_data_query", "handle_general"]:
            workflow.add_edge(handler, "apply_output_guard")
        
        for handler in ["handle_greeting", "handle_smalltalk", "handle_capability_summary",
                       "handle_system_flow", "handle_differentiator", "handle_meta_feedback", "handle_help"]:
            workflow.add_edge(handler, "apply_output_guard")
        
        workflow.add_edge("apply_output_guard", "validate_response")
        workflow.add_edge("validate_response", "enrich_with_graph_intelligence")
        workflow.add_edge("enrich_with_graph_intelligence", "store_in_database")
        workflow.add_edge("store_in_database", "save_memory")
        workflow.add_edge("save_memory", END)
        
        return workflow.compile()
    
    async def _node_init_memory(self, state: OrchestratorState) -> OrchestratorState:
        """Initialize memory using LangGraph checkpointing."""
        try:
            memory_manager = AidentMemoryManager(
                user_id=state["user_id"],
                redis_url=os.getenv('ARQ_REDIS_URL') or os.getenv('REDIS_URL')
            )
            
            # Load memory from LangGraph checkpoint
            memory_data = await memory_manager.load_memory()
            memory_context = memory_data.get("buffer", "")
            memory_messages = memory_data.get("messages", [])
            
            # Get conversation history from memory or database
            if memory_messages:
                conversation_history = memory_messages
            else:
                try:
                    conversation_history = await asyncio.wait_for(
                        self._load_conversation_history(state["user_id"], state.get("chat_id")) 
                        if state.get("chat_id") else asyncio.sleep(0),
                        timeout=5.0
                    ) if state.get("chat_id") else []
                except asyncio.TimeoutError:
                    logger.warning("Conversation history loading timed out")
                    conversation_history = []
                except Exception as e:
                    logger.warning(f"Failed to load conversation history: {e}")
                    conversation_history = []
            
            # Store in state for use by other nodes
            state["memory_manager"] = memory_manager
            state["memory_context"] = memory_context
            state["memory_messages"] = memory_messages
            state["conversation_history"] = conversation_history
            state["processing_steps"] = state.get("processing_steps", []) + ["init_memory"]
            
            logger.info("Memory initialized from LangGraph checkpoint",
                       user_id=state["user_id"],
                       message_count=len(memory_messages))
            
        except Exception as e:
            logger.error("Memory initialization failed", error=str(e))
            state["errors"] = state.get("errors", []) + [f"Memory init failed: {str(e)}"]
        
        return state
    
    async def _node_classify_intent(self, state: OrchestratorState) -> OrchestratorState:
        """Classify user intent with retry logic and confidence threshold."""
        try:
            intent_result = await self._classify_intent_with_retry(state["question"])
            state["intent"] = intent_result.intent.value
            state["intent_confidence"] = intent_result.confidence
            state["processing_steps"] = state.get("processing_steps", []) + ["classify_intent"]
            logger.info("Intent classified", intent=intent_result.intent.value, confidence=round(intent_result.confidence, 2))
            
            if intent_result.confidence < 0.6:
                logger.warning(f"Low confidence intent classification: {intent_result.confidence:.2f}")
                state["low_confidence_intent"] = True
                state["confidence_threshold_breached"] = True
            else:
                state["low_confidence_intent"] = False
                
        except Exception as e:
            logger.error("Intent classification failed after retries", error=str(e))
            state["errors"] = state.get("errors", []) + [f"Intent classification failed: {str(e)}"]
            # Fallback to UNKNOWN intent
            state["intent"] = "unknown"
            state["intent_confidence"] = 0.0
            state["low_confidence_intent"] = True
        return state
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((Exception,)),
        reraise=True
    )
    async def _classify_intent_with_retry(self, question: str):
        """Classify intent with automatic retry on failure."""
        return await self.intent_classifier.classify(question)
    
    async def _node_route_by_intent(self, state: OrchestratorState) -> OrchestratorState:
        """Route by intent (decision node)."""
        state["processing_steps"] = state.get("processing_steps", []) + ["route_by_intent"]
        return state
    
    async def _node_classify_question(self, state: OrchestratorState) -> OrchestratorState:
        """Classify question type with confidence threshold."""
        try:
            question_type, confidence = await self._classify_question(
                state["question"],
                state["user_id"],
                state.get("conversation_history", []),
                memory_context=state.get("memory_context", "")
            )
            state["question_type"] = question_type.value
            state["question_confidence"] = confidence
            state["processing_steps"] = state.get("processing_steps", []) + ["classify_question"]
            logger.info("Question type classified", question_type=question_type.value, confidence=round(confidence, 2))
            
            if confidence < 0.6:
                logger.warning(f"Low confidence question classification: {confidence:.2f}")
                state["low_confidence_question"] = True
                state["confidence_threshold_breached"] = True
            else:
                state["low_confidence_question"] = False
                
        except Exception as e:
            logger.error("Question classification failed", error=str(e))
            state["question_type"] = "general"
            state["question_confidence"] = 0.0
            state["low_confidence_question"] = True
            state["errors"] = state.get("errors", []) + [f"Question classification failed: {str(e)}"]
        
        return state
    
    async def _node_route_by_question_type(self, state: OrchestratorState) -> OrchestratorState:
        """Route by question type (decision node)."""
        state["processing_steps"] = state.get("processing_steps", []) + ["route_by_question_type"]
        return state
    
    async def _node_ask_clarifying_question(self, state: OrchestratorState) -> OrchestratorState:
        """Ask clarifying question when confidence is too low."""
        try:
            question = state.get("question", "")
            intent_confidence = state.get("intent_confidence", 0.0)
            question_confidence = state.get("question_confidence", 0.0)
            
            # Determine which classification was low confidence
            if intent_confidence < 0.6:
                clarification_prompt = f"""I'm not entirely sure what you're asking. Your question was: "{question}"
                
Could you clarify what you'd like to know? For example:
- Are you asking about your financial data?
- Do you want to know what I can do?
- Are you looking for general financial advice?
- Something else?"""
            else:
                clarification_prompt = f"""I understand you're asking about your financial data, but I'm not sure exactly what type of analysis you need.
                
Could you be more specific? For example:
- "Show me my cash flow trends"
- "Find duplicate transactions"
- "Analyze my expenses"
- "Predict when customers will pay"
- Something else?"""
            
            logger.info(f"Asking clarifying question (intent_conf={intent_confidence:.2f}, question_conf={question_confidence:.2f})")
            
            response = ChatResponse(
                answer=clarification_prompt,
                question_type=QuestionType.GENERAL,
                confidence=0.5,  # Lower confidence since we're asking for clarification
                data={
                    "clarification_needed": True,
                    "original_question": question,
                    "intent_confidence": intent_confidence,
                    "question_confidence": question_confidence
                }
            )
            
            state["response"] = response
            state["processing_steps"] = state.get("processing_steps", []) + ["ask_clarifying_question"]
            
        except Exception as e:
            logger.error("Clarifying question handler failed", error=str(e))
            state["errors"] = state.get("errors", []) + [f"Clarifying question failed: {str(e)}"]
            state["response"] = ChatResponse(
                answer="I'm having trouble understanding your question. Could you rephrase it?",
                question_type=QuestionType.GENERAL,
                confidence=0.0
            )
        
        return state
    
    async def _node_handle_greeting(self, state: OrchestratorState) -> OrchestratorState:
        """Handle greeting intent."""
        return await self._execute_handler(state, self._handle_greeting, "greeting", "handle_greeting")
    
    async def _node_handle_smalltalk(self, state: OrchestratorState) -> OrchestratorState:
        """Handle smalltalk intent."""
        return await self._execute_handler(state, self._handle_smalltalk, "smalltalk", "handle_smalltalk")
    
    async def _node_handle_capability_summary(self, state: OrchestratorState) -> OrchestratorState:
        """Handle capability summary intent."""
        return await self._execute_handler(state, self._handle_capability_summary, "capability_summary", "handle_capability_summary")
    
    async def _node_handle_system_flow(self, state: OrchestratorState) -> OrchestratorState:
        """Handle system flow intent."""
        return await self._execute_handler(state, self._handle_system_flow, "system_flow", "handle_system_flow")
    
    async def _node_handle_differentiator(self, state: OrchestratorState) -> OrchestratorState:
        """Handle differentiator intent."""
        return await self._execute_handler(state, self._handle_differentiator, "differentiator", "handle_differentiator")
    
    async def _node_handle_meta_feedback(self, state: OrchestratorState) -> OrchestratorState:
        """Handle meta feedback intent."""
        return await self._execute_handler(state, self._handle_meta_feedback, "meta_feedback", "handle_meta_feedback")
    
    async def _node_handle_help(self, state: OrchestratorState) -> OrchestratorState:
        """Handle help intent."""
        return await self._execute_handler(state, self._handle_help, "help", "handle_help")
    
    async def _node_handle_causal(self, state: OrchestratorState) -> OrchestratorState:
        """Handle causal question."""
        return await self._execute_question_handler(state, self._handle_causal_question, "causal", "handle_causal")
    
    async def _node_handle_temporal(self, state: OrchestratorState) -> OrchestratorState:
        """Handle temporal question."""
        return await self._execute_question_handler(state, self._handle_temporal_question, "temporal", "handle_temporal")
    
    async def _node_handle_relationship(self, state: OrchestratorState) -> OrchestratorState:
        """Handle relationship question."""
        return await self._execute_question_handler(state, self._handle_relationship_question, "relationship", "handle_relationship")
    
    async def _node_handle_whatif(self, state: OrchestratorState) -> OrchestratorState:
        """Handle what-if question."""
        return await self._execute_question_handler(state, self._handle_whatif_question, "whatif", "handle_whatif")
    
    async def _node_handle_explain(self, state: OrchestratorState) -> OrchestratorState:
        """Handle explain question."""
        return await self._execute_question_handler(state, self._handle_explain_question, "explain", "handle_explain")
    
    async def _node_handle_data_query(self, state: OrchestratorState) -> OrchestratorState:
        """Handle data query question."""
        return await self._execute_question_handler(state, self._handle_data_query, "data_query", "handle_data_query")
    
    async def _node_handle_general(self, state: OrchestratorState) -> OrchestratorState:
        """Handle general question."""
        try:
            response = await self._handle_general_question(state["question"], state["user_id"], state.get("context"), state.get("conversation_history", []), None)
            state["response"] = response
            state["processing_steps"] = state.get("processing_steps", []) + ["handle_general"]
        except Exception as e:
            logger.error("General handler failed", error=str(e))
            state["errors"] = state.get("errors", []) + [f"General handler failed: {str(e)}"]
        return state
    
    async def _execute_handler(self, state: OrchestratorState, handler_func, handler_name: str, step_name: str) -> OrchestratorState:
        """Generic handler executor for intent handlers."""
        try:
            response = await handler_func(state["question"], state["user_id"], state.get("conversation_history", []), None)
            state["response"] = response
            state["processing_steps"] = state.get("processing_steps", []) + [step_name]
        except Exception as e:
            logger.error(f"{handler_name} handler failed", error=str(e))
            state["errors"] = state.get("errors", []) + [f"{handler_name} handler failed: {str(e)}"]
        return state
    
    async def _execute_question_handler(self, state: OrchestratorState, handler_func, handler_name: str, step_name: str) -> OrchestratorState:
        """Generic handler executor for question type handlers."""
        try:
            response = await handler_func(state["question"], state["user_id"], state.get("context"), state.get("conversation_history", []))
            state["response"] = response
            state["processing_steps"] = state.get("processing_steps", []) + [step_name]
        except Exception as e:
            logger.error(f"{handler_name} handler failed", error=str(e))
            state["errors"] = state.get("errors", []) + [f"{handler_name} handler failed: {str(e)}"]
        return state
    
    async def _node_apply_output_guard(self, state: OrchestratorState) -> OrchestratorState:
        """Apply output guard to check for repetition."""
        try:
            if state.get("response"):
                safe_response = await self.output_guard.check_and_fix(
                    proposed_response=state["response"].answer,
                    recent_responses=state.get("memory_messages", [])[-5:],
                    question=state["question"],
                    frustration_level=0
                )
                state["response"].answer = safe_response
            state["processing_steps"] = state.get("processing_steps", []) + ["apply_output_guard"]
        except Exception as e:
            logger.error("Output guard failed", error=str(e))
            state["errors"] = state.get("errors", []) + [f"Output guard failed: {str(e)}"]
        return state
    
    async def _node_save_memory(self, state: OrchestratorState) -> OrchestratorState:
        """Save memory using LangGraph checkpointing."""
        try:
            if state.get("response"):
                memory_manager = state.get("memory_manager")
                
                if not memory_manager:
                    logger.warning("Memory manager not initialized, creating fallback instance")
                    try:
                        memory_manager = AidentMemoryManager(
                            user_id=state["user_id"],
                            redis_url=os.getenv('ARQ_REDIS_URL') or os.getenv('REDIS_URL')
                        )
                    except Exception as e:
                        logger.error("Failed to create fallback memory manager", error=str(e))
                        memory_manager = None
                
                if memory_manager:
                    await memory_manager.add_message(state["question"], state["response"].answer)
                    logger.info("Memory saved to LangGraph checkpoint",
                               user_id=state["user_id"],
                               question_length=len(state["question"]),
                               response_length=len(state["response"].answer))
                else:
                    logger.warning("Memory save skipped: memory manager unavailable")
            
            state["processing_steps"] = state.get("processing_steps", []) + ["save_memory"]
        except Exception as e:
            logger.error("Memory save failed", error=str(e))
            state["errors"] = state.get("errors", []) + [f"Memory save failed: {str(e)}"]
        return state
    
    async def _node_validate_response(self, state: OrchestratorState) -> OrchestratorState:
        """Validate response quality and safety."""
        try:
            if state.get("response"):
                # Validate response has required fields
                response = state["response"]
                if not response.answer:
                    state["errors"] = state.get("errors", []) + ["Response has empty answer"]
                    logger.warning("Response validation failed: empty answer")
                
                # Validate confidence score
                if not (0.0 <= response.confidence <= 1.0):
                    response.confidence = max(0.0, min(1.0, response.confidence))
                    logger.warning("Response confidence normalized", confidence=response.confidence)
                
                # Check for safety issues
                if len(response.answer) > 10000:
                    logger.warning("Response too long", length=len(response.answer))
                    response.answer = response.answer[:10000] + "..."
            
            state["processing_steps"] = state.get("processing_steps", []) + ["validate_response"]
            logger.info("Response validation completed")
        except Exception as e:
            logger.error("Response validation failed", error=str(e))
            state["errors"] = state.get("errors", []) + [f"Response validation failed: {str(e)}"]
        return state
    
    async def _node_enrich_with_graph_intelligence(self, state: OrchestratorState) -> OrchestratorState:
        """Enrich response with graph intelligence from FinleyGraph engine."""
        try:
            if state.get("response") and state.get("question"):
                response = state["response"]
                
                # Query graph engine for enrichment
                try:
                    enrichment = await self.graph_engine.enrich_response(
                        question=state["question"],
                        response_answer=response.answer,
                        user_id=state["user_id"],
                        question_type=response.question_type.value
                    )
                    
                    if enrichment:
                        # Add graph insights to response data
                        if response.data is None:
                            response.data = {}
                        response.data["graph_insights"] = enrichment
                        logger.info("Response enriched with graph intelligence")
                except Exception as e:
                    logger.warning("Graph enrichment failed (non-blocking)", error=str(e))
            
            state["processing_steps"] = state.get("processing_steps", []) + ["enrich_with_graph_intelligence"]
        except Exception as e:
            logger.error("Enrichment pipeline failed", error=str(e))
            state["errors"] = state.get("errors", []) + [f"Enrichment failed: {str(e)}"]
        return state
    
    async def _node_store_in_database(self, state: OrchestratorState) -> OrchestratorState:
        """Store response in database."""
        try:
            if state.get("response"):
                await self._store_chat_message(
                    user_id=state["user_id"],
                    chat_id=state.get("chat_id"),
                    question=state["question"],
                    response=state["response"],
                    chat_title=state.get("chat_title")
                )
            state["processing_steps"] = state.get("processing_steps", []) + ["store_in_database"]
            logger.info("Response stored in database")
        except Exception as e:
            logger.error("Database storage failed", error=str(e))
            state["errors"] = state.get("errors", []) + [f"Storage failed: {str(e)}"]
        return state
    
    async def _node_determine_data_mode(self, state: OrchestratorState) -> OrchestratorState:
        """Determine user's data availability mode."""
        try:
            data_mode = await self._determine_data_mode(state["user_id"])
            state["data_mode"] = data_mode.value
            state["processing_steps"] = state.get("processing_steps", []) + ["determine_data_mode"]
            logger.info("Data mode determined", mode=data_mode.value)
        except Exception as e:
            logger.error("Data mode determination failed", error=str(e))
            state["data_mode"] = "no_data"
            state["errors"] = state.get("errors", []) + [f"Data mode determination failed: {str(e)}"]
        return state
    
    async def _node_onboarding_handler(self, state: OrchestratorState) -> OrchestratorState:
        """Handle NO_DATA mode - user has no data connected yet"""
        try:
            onboarding_response = await self._generate_no_data_intelligence(state["user_id"])
            
            if state.get("response"):
                state["response"].answer = onboarding_response
                state["response"].confidence = 0.9
            
            state["processing_steps"] = state.get("processing_steps", []) + ["onboarding_handler"]
            logger.info("Onboarding guidance provided")
        except Exception as e:
            logger.error("Onboarding handler failed", error=str(e))
            state["errors"] = state.get("errors", []) + [f"Onboarding handler failed: {str(e)}"]
        return state
    
    async def _node_exploration_handler(self, state: OrchestratorState) -> OrchestratorState:
        """Handle LIMITED_DATA mode - user has <50 transactions"""
        try:
            exploration_questions = [
                "What are my top expense categories?",
                "Show me my cash flow trends",
                "Which vendors do I pay most often?",
                "What's my average transaction size?",
                "Are there any unusual transactions?"
            ]
            
            if state.get("response"):
                state["response"].follow_up_questions = exploration_questions
                state["response"].confidence = 0.85
            
            state["processing_steps"] = state.get("processing_steps", []) + ["exploration_handler"]
            logger.info("Exploration guidance provided")
        except Exception as e:
            logger.error("Exploration handler failed", error=str(e))
            state["errors"] = state.get("errors", []) + [f"Exploration handler failed: {str(e)}"]
        return state
    
    async def _node_advanced_handler(self, state: OrchestratorState) -> OrchestratorState:
        """Handle RICH_DATA mode - user has >50 transactions"""
        try:
            advanced_questions = [
                "What's driving my profitability changes?",
                "Predict my cash flow for next quarter",
                "Which relationships are most valuable?",
                "Detect anomalies in my transactions",
                "Compare my metrics to industry benchmarks"
            ]
            
            if state.get("response"):
                state["response"].follow_up_questions = advanced_questions
                state["response"].confidence = 0.95
            
            state["processing_steps"] = state.get("processing_steps", []) + ["advanced_handler"]
            logger.info("Advanced analysis guidance provided")
        except Exception as e:
            logger.error("Advanced handler failed", error=str(e))
            state["errors"] = state.get("errors", []) + [f"Advanced handler failed: {str(e)}"]
        return state
    
    async def _node_fetch_temporal_data(self, state: OrchestratorState) -> OrchestratorState:
        """Fetch temporal data in parallel with other data sources"""
        try:
            temporal_data = await self.temporal_learner.learn_all_patterns(state["user_id"])
            state["temporal_data"] = temporal_data
            state["processing_steps"] = state.get("processing_steps", []) + ["fetch_temporal_data"]
            logger.info("Temporal data fetched", patterns_count=len(temporal_data.get('patterns', [])))
        except Exception as e:
            logger.error("Temporal data fetch failed", error=str(e))
            state["temporal_data"] = {}
            state["errors"] = state.get("errors", []) + [f"Temporal data fetch failed: {str(e)}"]
        return state
    
    async def _node_fetch_seasonal_data(self, state: OrchestratorState) -> OrchestratorState:
        """Fetch seasonal data in parallel with other data sources"""
        try:
            temporal_data = state.get("temporal_data", {})
            seasonal_data = {
                "seasonal_patterns": temporal_data.get("seasonal_patterns", []),
                "seasonality_strength": temporal_data.get("seasonality_strength", 0.0),
                "seasonal_forecast": temporal_data.get("seasonal_forecast", {})
            }
            state["seasonal_data"] = seasonal_data
            state["processing_steps"] = state.get("processing_steps", []) + ["fetch_seasonal_data"]
            logger.info("Seasonal data fetched", seasonality_strength=seasonal_data.get("seasonality_strength"))
        except Exception as e:
            logger.error("Seasonal data fetch failed", error=str(e))
            state["seasonal_data"] = {}
            state["errors"] = state.get("errors", []) + [f"Seasonal data fetch failed: {str(e)}"]
        return state
    
    async def _node_fetch_fraud_data(self, state: OrchestratorState) -> OrchestratorState:
        """Fetch fraud/anomaly data in parallel with other data sources"""
        try:
            fraud_result = self.supabase.rpc(
                'detect_anomalies',
                {'p_user_id': state["user_id"]}
            ).execute()
            
            fraud_data = {
                "anomalies": fraud_result.data if fraud_result.data else [],
                "anomaly_count": len(fraud_result.data) if fraud_result.data else 0
            }
            state["fraud_data"] = fraud_data
            state["processing_steps"] = state.get("processing_steps", []) + ["fetch_fraud_data"]
            logger.info("Fraud data fetched", anomaly_count=fraud_data.get("anomaly_count"))
        except Exception as e:
            logger.error("Fraud data fetch failed", error=str(e))
            state["fraud_data"] = {}
            state["errors"] = state.get("errors", []) + [f"Fraud data fetch failed: {str(e)}"]
        return state
    
    async def _node_fetch_root_cause_data(self, state: OrchestratorState) -> OrchestratorState:
        """Fetch root cause analysis data in parallel with other data sources."""
        try:
            # Run causal analysis for root cause detection
            root_cause_data = await self.causal_engine.analyze_causal_relationships(
                user_id=state["user_id"]
            )
            state["root_cause_data"] = root_cause_data
            state["processing_steps"] = state.get("processing_steps", []) + ["fetch_root_cause_data"]
            logger.info("Root cause data fetched", relationships_count=len(root_cause_data.get('causal_relationships', [])))
        except Exception as e:
            logger.error("Root cause data fetch failed", error=str(e))
            state["root_cause_data"] = {}
            state["errors"] = state.get("errors", []) + [f"Root cause data fetch failed: {str(e)}"]
        return state
    
    async def _node_aggregate_parallel_results(self, state: OrchestratorState) -> OrchestratorState:
        """Aggregate results from parallel fetch nodes"""
        try:
            aggregated_data = {
                "temporal": state.get("temporal_data", {}),
                "seasonal": state.get("seasonal_data", {}),
                "fraud": state.get("fraud_data", {}),
                "root_cause": state.get("root_cause_data", {})
            }
            
            if state.get("response"):
                state["response"].data = aggregated_data
            
            state["processing_steps"] = state.get("processing_steps", []) + ["aggregate_parallel_results"]
            logger.info("Parallel results aggregated", data_sources=list(aggregated_data.keys()))
        except Exception as e:
            logger.error("Result aggregation failed", error=str(e))
            state["errors"] = state.get("errors", []) + [f"Result aggregation failed: {str(e)}"]
        return state
    
    async def _determine_data_mode(self, user_id: str) -> DataMode:
        """Determine user's data availability mode using business rules engine."""
        try:
            events_result = self.supabase.table('raw_events')\
                .select('id', count='exact')\
                .eq('user_id', user_id)\
                .execute()
            
            transaction_count = len(events_result.data) if events_result.data else 0
            
            try:
                from aident_cfo_brain.business_rules_engine import get_business_rules_engine
            except ImportError:
                from business_rules_engine import get_business_rules_engine
            
            rules_engine = get_business_rules_engine()
            mode_str = rules_engine.determine_data_mode(transaction_count)
            
            # Convert string to DataMode enum
            try:
                return DataMode(mode_str)
            except ValueError:
                logger.warning(f"Invalid data mode from rules: {mode_str}, defaulting to NO_DATA")
                return DataMode.NO_DATA
        
        except Exception as e:
            logger.warning(f"Failed to determine data mode: {e}")
            return DataMode.NO_DATA

    async def _get_onboarding_state(self, user_id: str) -> OnboardingState:
        """Get user's onboarding state from database"""
        try:
            prefs_result = self.supabase.table('user_preferences')\
                .select('onboarding_state')\
                .eq('user_id', user_id)\
                .limit(1)\
                .execute()
            
            if prefs_result.data and len(prefs_result.data) > 0:
                state_str = prefs_result.data[0].get('onboarding_state', 'first_visit')
                try:
                    return OnboardingState(state_str)
                except ValueError:
                    return OnboardingState.FIRST_VISIT
            else:
                return OnboardingState.FIRST_VISIT
        except Exception as e:
            logger.warning(f"Failed to get onboarding state: {e}")
            return OnboardingState.FIRST_VISIT

    async def _set_onboarding_state(self, user_id: str, state: OnboardingState) -> bool:
        """Save user's onboarding state to database"""
        try:
            self.supabase.table('user_preferences').upsert({
                'user_id': user_id,
                'onboarding_state': state.value,
                'updated_at': datetime.utcnow().isoformat()
            }, on_conflict='user_id').execute()
            
            logger.info(f"Onboarding state updated for user {user_id}: {state.value}")
            return True
        except Exception as e:
            logger.warning(f"Failed to set onboarding state: {e}")
            return False

    async def _generate_no_data_intelligence(self, user_id: str) -> str:
        """Generate intelligent onboarding questions using LLM"""
        try:
            # REFACTOR: Load prompt from external config
            prompt = self.prompt_loader.get_prompt('no_data_onboarding', 'system')
            
            response = await self.groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a financial intelligence expert generating insightful questions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            intelligent_questions = response.choices[0].message.content
            formatted_response = f"""🧠 **Here are some questions I can answer once you connect your data:**

{intelligent_questions}

**Ready to unlock these insights?** Click "Data Sources" to connect QuickBooks, Xero, or upload your financial files."""
            
            return formatted_response
            
        except Exception as e:
            logger.warning(f"Failed to generate intelligent questions: {e}")
            return """🧠 **Here are some questions I can answer once you connect your data:**

❓ What's causing my cash flow to fluctuate?
❓ When do my customers typically pay?
❓ Which vendors represent my biggest costs?

**Ready to unlock these insights?** Click "Data Sources" to connect QuickBooks, Xero, or upload your financial files."""

    async def _generate_dynamic_onboarding_message(self, user_id: str, onboarding_state: OnboardingState) -> str:
        """Generate personalized onboarding message based on user state"""
        try:
            # REFACTORED: Use PromptLoader instead of hard-coded strings
            if onboarding_state == OnboardingState.FIRST_VISIT:
                return self.prompt_loader.get_onboarding_message('first_visit')
            
            elif onboarding_state == OnboardingState.ONBOARDED:
                return self.prompt_loader.get_onboarding_message('onboarded')
            
            elif onboarding_state == OnboardingState.ACTIVE:
                return self.prompt_loader.get_onboarding_message('active')
            
            else:  # RETURNING or other states
                return self.prompt_loader.get_onboarding_message('returning')
                
        except Exception as e:
            logger.warning(f"Failed to load onboarding message: {e}")
            # Ultra-minimal fallback
            return "👋 Connect your data to get started with financial insights!"

    async def process_question(
        self,
        question: str,
        user_id: str,
        chat_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        chat_title: Optional[str] = None
    ) -> ChatResponse:
        """Process user question through LangGraph orchestration"""
        try:
            logger.info("Processing question", question=question, user_id=user_id, chat_id=chat_id)
            print(f"[ORCHESTRATOR] Starting process_question for user {user_id}", flush=True)
            
            initial_state = {
                "question": question,
                "user_id": user_id,
                "chat_id": chat_id,
                "chat_title": chat_title or "New Chat",
                "context": context or {},
                "response": None,
                "processing_steps": ["initialize_state"],
                "errors": []
            }
            
            try:
                final_state = await self.graph.ainvoke(initial_state)
                response = final_state.get("response")
                
                if not response:
                    logger.error("LangGraph returned no response")
                    response = ChatResponse(
                        answer="I encountered an error processing your question. Please try again.",
                        question_type=QuestionType.UNKNOWN,
                        confidence=0.0,
                        data={}
                    )
                
                logger.info("LangGraph execution completed", 
                           processing_steps=final_state.get("processing_steps", []),
                           errors=final_state.get("errors", []))
            except Exception as e:
                logger.error("LangGraph execution failed", error=str(e), exc_info=True)
                response = ChatResponse(
                    answer="I encountered an error processing your question. Please try rephrasing it or contact support if the issue persists.",
                    question_type=QuestionType.UNKNOWN,
                    confidence=0.0,
                    data={'error': str(e)}
                )
            
            logger.info("Question processed successfully",
                       question_type=response.question_type.value if response else "unknown",
                       confidence=response.confidence if response else 0.0)
            
            return response
            
        except Exception as e:
            logger.error("Error processing question", error=str(e), exc_info=True)
            return ChatResponse(
                answer=f"I encountered an error processing your question. Please try rephrasing it or contact support if the issue persists.",
                question_type=QuestionType.UNKNOWN,
                confidence=0.0,
                data={'error': str(e)}
            )
    
    async def _classify_question(
        self,
        question: str,
        user_id: str,
        conversation_history: list[Dict[str, str]] = None,
        memory_context: str = ""
    ) -> Tuple[QuestionType, float]:
        """
        Classify question type using SetFit (ML-based, no LLM calls).
        
        REFACTORED: Replaced LLM classification (100ms+ per call) with SetFit (1ms).
        - 100x faster
        - No API costs
        - Works offline
        - Better accuracy with training data
        """
        try:
            # Import SetFit classifier
            try:
                from aident_cfo_brain.question_classifier_setfit import get_question_classifier_setfit
            except ImportError:
                from question_classifier_setfit import get_question_classifier_setfit
            
            classifier = get_question_classifier_setfit()
            
            # Classify (fast, local, no API call)
            question_type_str, confidence = classifier.classify(question)
            
            logger.info("question_classified_setfit", 
                       type=question_type_str, 
                       confidence=confidence,
                       method="SetFit (ML-based)")
            
            try:
                question_type = QuestionType(question_type_str)
            except ValueError:
                question_type = QuestionType.UNKNOWN
                confidence = 0.0
            
            return question_type, confidence
            
        except Exception as e:
            logger.error("question_classification_failed", error=str(e), exc_info=True)
            return QuestionType.UNKNOWN, 0.0
    
    async def _handle_causal_question(
        self,
        question: str,
        user_id: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: list[Dict[str, str]] = None
    ) -> ChatResponse:
        """Handle causal questions (WHY questions)"""
        try:
            entities = await self._extract_entities_from_question(question, user_id)
            causal_results = await self.causal_engine.analyze_causal_relationships(user_id=user_id)
            
            if not causal_results.get('causal_relationships'):
                return ChatResponse(
                    answer="I haven't found enough data to perform causal analysis yet. Upload more financial data to enable this feature.",
                    question_type=QuestionType.CAUSAL,
                    confidence=0.8
                )
            
            answer = await self._format_causal_response(question, causal_results, entities)
            actions = self._generate_causal_actions(causal_results)
            follow_ups = [
                "What if I address the root cause?",
                "Show me the complete causal chain",
                "Are there other contributing factors?"
            ]
            
            return ChatResponse(
                answer=answer,
                question_type=QuestionType.CAUSAL,
                confidence=0.9,
                data=causal_results,
                actions=actions,
                follow_up_questions=follow_ups
            )
            
        except Exception as e:
            logger.error("Causal question handling failed", error=str(e))
            return ChatResponse(
                answer="I encountered an error analyzing the causal relationships. Please try again.",
                question_type=QuestionType.CAUSAL,
                confidence=0.0,
                data={'error': str(e)}
            )
    
    async def _handle_temporal_question(
        self,
        question: str,
        user_id: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: list[Dict[str, str]] = None
    ) -> ChatResponse:
        """Handle temporal questions (WHEN questions)"""
        try:
            patterns_result = await self.temporal_learner.learn_all_patterns(user_id)
            
            if not patterns_result.get('patterns'):
                return ChatResponse(
                    answer="I need more historical data to learn temporal patterns. Upload more data spanning multiple time periods.",
                    question_type=QuestionType.TEMPORAL,
                    confidence=0.8
                )
            
            answer = await self._format_temporal_response(question, patterns_result)
            visualizations = self._generate_temporal_visualizations(patterns_result)
            follow_ups = [
                "Are there any anomalies in the timeline?"
            ]
            
            return ChatResponse(
                answer=answer,
                question_type=QuestionType.TEMPORAL,
                confidence=0.9,
                data=patterns_result,
                visualizations=visualizations,
                follow_up_questions=follow_ups
            )
            
        except Exception as e:
            logger.error("Temporal question handling failed", error=str(e))
            return ChatResponse(
                answer="I encountered an error analyzing temporal patterns. Please try again.",
                question_type=QuestionType.TEMPORAL,
                confidence=0.0,
                data={'error': str(e)}
            )
    
    async def _handle_relationship_question(
        self,
        question: str,
        user_id: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: list[Dict[str, str]] = None
    ) -> ChatResponse:
        """Handle relationship questions (e.g., 'Show vendor relationships')."""
        try:
            # Detect relationships
            relationships_result = await self.relationship_detector.detect_all_relationships(
                user_id=user_id
            )
            
            if not relationships_result.get('relationships'):
                return ChatResponse(
                    answer="I haven't detected any relationships in your data yet. Upload more interconnected financial data (invoices, payments, etc.).",
                    question_type=QuestionType.RELATIONSHIP,
                    confidence=0.8
                )
            
            # Format response
            answer = await self._format_relationship_response(question, relationships_result)
            
            # Generate graph visualization data
            visualizations = self._generate_relationship_visualizations(relationships_result)
            
            # Actions
            actions = [
                {"type": "view_graph", "label": "View Relationship Graph", "data": relationships_result},
                {"type": "export", "label": "Export Relationships", "format": "csv"}
            ]
            
            # Follow-ups
            follow_ups = [
                "Show me cross-platform relationships",
                "Which relationships are strongest?",
                "Find missing relationships"
            ]
            
            return ChatResponse(
                answer=answer,
                question_type=QuestionType.RELATIONSHIP,
                confidence=0.9,
                data=relationships_result,
                actions=actions,
                visualizations=visualizations,
                follow_up_questions=follow_ups
            )
            
        except Exception as e:
            logger.error("Relationship question handling failed", error=str(e))
            return ChatResponse(
                answer="I encountered an error analyzing relationships. Please try again.",
                question_type=QuestionType.RELATIONSHIP,
                confidence=0.0,
                data={'error': str(e)}
            )
    
    async def _handle_whatif_question(
        self,
        question: str,
        user_id: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: list[Dict[str, str]] = None
    ) -> ChatResponse:
        """Handle what-if questions (e.g., 'What if I delay payment?')."""
        try:
            # Extract scenario parameters from question
            scenario = await self._extract_scenario_from_question(question, user_id)
            
            # Run counterfactual analysis (placeholder - needs implementation)
            answer = f"What-if analysis is being developed. Your scenario: {scenario}"
            
            return ChatResponse(
                answer=answer,
                question_type=QuestionType.WHAT_IF,
                confidence=0.7,
                data={'scenario': scenario},
                follow_up_questions=[
                    "What's the best case scenario?",
                    "What's the worst case scenario?",
                    "Show me alternative options"
                ]
            )
            
        except Exception as e:
            logger.error("What-if question handling failed", error=str(e))
            return ChatResponse(
                answer="I encountered an error running the what-if analysis. Please try again.",
                question_type=QuestionType.WHAT_IF,
                confidence=0.0,
                data={'error': str(e)}
            )
    
    async def _handle_explain_question(
        self,
        question: str,
        user_id: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: list[Dict[str, str]] = None
    ) -> ChatResponse:
        """Handle explain questions (e.g., 'Where did this number come from?')."""
        try:
            # Extract entity/number to explain
            entity_id = await self._extract_entity_id_from_question(question, user_id, context)
            
            if not entity_id:
                return ChatResponse(
                    answer="I need more specific information. Can you provide an invoice number, transaction ID, or specific amount?",
                    question_type=QuestionType.EXPLAIN,
                    confidence=0.6
                )
            
            # Get provenance data
            provenance_result = self.supabase.rpc(
                'get_event_provenance',
                {'p_user_id': user_id, 'p_event_id': entity_id}
            ).execute()
            
            if not provenance_result.data:
                return ChatResponse(
                    answer="I couldn't find detailed provenance information for that item.",
                    question_type=QuestionType.EXPLAIN,
                    confidence=0.5
                )
            
            # Format explanation
            answer = await self._format_provenance_explanation(provenance_result.data)
            
            # Actions
            actions = [
                {"type": "view_lineage", "label": "View Full Lineage", "data": provenance_result.data},
                {"type": "verify_integrity", "label": "Verify Data Integrity"}
            ]
            
            return ChatResponse(
                answer=answer,
                question_type=QuestionType.EXPLAIN,
                confidence=0.9,
                data=provenance_result.data,
                actions=actions
            )
            
        except Exception as e:
            logger.error("Explain question handling failed", error=str(e))
            return ChatResponse(
                answer="I encountered an error retrieving the explanation. Please try again.",
                question_type=QuestionType.EXPLAIN,
                confidence=0.0,
                data={'error': str(e)}
            )
    
    async def _handle_data_query(
        self,
        question: str,
        user_id: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: list[Dict[str, str]] = None
    ) -> ChatResponse:
        """Handle data query questions (e.g., 'Show me all invoices')"""
        try:
            query_params = await self._extract_query_params_from_question(question, user_id)
            result = self.supabase.table('raw_events').select('*').eq(
                'user_id', user_id
            ).limit(100).execute()
            
            if not result.data:
                return ChatResponse(
                    answer="No data found matching your query.",
                    question_type=QuestionType.DATA_QUERY,
                    confidence=0.8
                )
            
            answer = f"I found {len(result.data)} records matching your query."
            
            return ChatResponse(
                answer=answer,
                question_type=QuestionType.DATA_QUERY,
                confidence=0.85,
                data={'records': result.data[:10], 'total': len(result.data)},
                actions=[
                    {"type": "export", "label": "Export All Results", "format": "csv"},
                    {"type": "visualize", "label": "Visualize Data"}
                ]
            )
            
        except Exception as e:
            logger.error("Data query handling failed", error=str(e))
            return ChatResponse(
                answer="I encountered an error querying the data. Please try again.",
                question_type=QuestionType.DATA_QUERY,
                confidence=0.0,
                data={'error': str(e)}
            )
    
    async def _detect_follow_up_question(
        self,
        question: str,
        user_id: str,
        memory_manager: Optional[Any],
        conversation_history: list[Dict[str, str]] = None
    ) -> Tuple[bool, Optional[str]]:
        """Detect if this is a follow-up question"""
        try:
            if not conversation_history or len(conversation_history) < 2:
                return False, None
            
            last_messages = [m for m in conversation_history[-4:] if m.get('role') == 'assistant']
            if not last_messages:
                return False, None
            
            last_response = last_messages[-1].get('content', '').lower()
            is_last_onboarding = any([
                'connect your data' in last_response,
                'quickbooks' in last_response and 'xero' in last_response,
                'data sources' in last_response,
                'upload your financial files' in last_response,
                'let\'s get started' in last_response
            ])
            
            is_capability_question = any([
                'what can you do' in question.lower(),
                'what are your capabilities' in question.lower(),
                'what can you help with' in question.lower(),
                'what do you do' in question.lower()
            ])
            
            if is_last_onboarding and is_capability_question:
                return True, 'onboarding'
            
            follow_up_patterns = ['how?', 'why?', 'tell me more', 'explain', 'what do you mean']
            is_simple_followup = any(pattern in question.lower() for pattern in follow_up_patterns)
            if is_simple_followup and len(conversation_history) >= 2:
                return True, 'clarification'
            return False, None
            
        except Exception as e:
            logger.warning(f"Failed to detect follow-up question: {e}")
            return False, None

    async def _get_cached_response(self, question: str, user_id: str) -> Optional[str]:
        """Check semantic cache for similar questions"""
        try:
            cache_key = f"response:{user_id}:{question[:50]}"
            cached = await Cache.get(cache_key)
            if cached:
                logger.info(f"Cache hit for question: {question[:50]}")
                return cached
            return None
            
        except Exception as e:
            logger.debug(f"Cache lookup failed (non-blocking): {str(e)}")
            return None

    async def _cache_response(self, question: str, user_id: str, response: str, ttl: int = 3600) -> None:
        """Cache response for future similar questions"""
        try:
            cache_key = f"response:{user_id}:{question[:50]}"
            await Cache.set(cache_key, response, ttl=ttl)
            logger.info(f"Cached response for question: {question[:50]}")
            
        except Exception as e:
            logger.debug(f"Cache write failed (non-blocking): {str(e)}")

    async def _stream_groq_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        model: str = "llama-3.3-70b-versatile",
        max_tokens: int = 1000,
        temperature: float = 0.7
    ):
        """Stream Groq API responses for incremental frontend display"""
        try:
            stream = await self.groq.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    *messages
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )
            
            full_response = ""
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content
            
            logger.info(f"Streaming response completed. Total length: {len(full_response)} chars")
            
        except Exception as e:
            logger.error(f"Streaming response failed: {str(e)}")
            yield f"Error generating response: {str(e)}"

    async def _handle_general_question(
        self,
        question: str,
        user_id: str,
        context: Optional[Dict[str, Any]],
        conversation_history: list[Dict[str, str]] = None,
        memory_manager: Optional[Any] = None
    ) -> ChatResponse:
        """Handle general financial questions with semantic caching"""
        try:
            cached_response = await self._get_cached_response(question, user_id)
            if cached_response:
                logger.info(f"Returning cached response for: {question[:50]}")
                return ChatResponse(
                    answer=cached_response,
                    question_type=QuestionType.GENERAL,
                    confidence=0.95,
                    data={"cached": True}
                )
            
            is_follow_up, last_response_type = await self._detect_follow_up_question(
                question, user_id, memory_manager, conversation_history
            )
            user_context = await self._fetch_user_data_context(user_id)
            messages = []
            
            if conversation_history:
                for msg in conversation_history[-10:]:
                    messages.append({
                        "role": msg['role'],
                        "content": msg['content']
                    })
            
            enriched_question = f"""USER QUESTION: {question}

USER'S ACTUAL DATA CONTEXT:
{user_context}

Reference their ACTUAL data in your response. Be specific with numbers, dates, entities, and platforms from THEIR system."""
            
            messages.append({
                "role": "user",
                "content": enriched_question
            })
            
            # REFACTOR: Load system prompt from external config
            system_prompt = self.prompt_loader.get_prompt('general_question', 'system')
            
            # Stream response for better UX
            answer_chunks = []
            async for chunk in self._stream_groq_response(
                messages=messages,
                system_prompt=system_prompt,
                model="llama-3.3-70b-versatile",
                max_tokens=1000,
                temperature=0.7
            ):
                answer_chunks.append(chunk)
            
            answer = "".join(answer_chunks)
            
            # Cache response for future similar questions (non-blocking)
            asyncio.create_task(self._cache_response(question, user_id, answer))
            
            # Conversation history is persisted in database via _store_chat_message()
            
            # Generate intelligent follow-up questions based on context
            follow_ups = self._generate_intelligent_followups(user_id, question, answer, user_context)
            
            return ChatResponse(
                answer=answer,
                question_type=QuestionType.GENERAL,
                confidence=0.85,
                follow_up_questions=follow_ups,
                data={"streaming": True}
            )
            
        except Exception as e:
            logger.error("General question handling failed", error=str(e))
            return ChatResponse(
                answer="I encountered an error processing your question. Please try again.",
                question_type=QuestionType.GENERAL,
                confidence=0.0,
                data={'error': str(e)}
            )
    
    def _generate_intelligent_followups(self, user_id: str, question: str, answer: str, user_context: str) -> list[str]:
        """Generate contextually relevant follow-up questions based on conversation and user data."""
        
        # Check if user has data
        has_data = "connections" in user_context.lower() or "files" in user_context.lower()
        has_connections = "active connection" in user_context.lower()
        has_files = "uploaded file" in user_context.lower()
        
        # Intelligent follow-ups based on context
        if not has_data:
            # No data - guide to onboarding
            return [
                "What can you do once I connect my data?",
                "How do I connect QuickBooks or Xero?",
                "Can I upload Excel files instead?"
            ]
        elif has_connections and not has_files:
            # Has connections - encourage exploration
            return [
                "Analyze my cash flow patterns",
                "Show me my top expenses this month",
                "Predict when customers will pay"
            ]
        elif has_files and not has_connections:
            # Has files - suggest connections
            return [
                "What insights can you find in my uploaded data?",
                "Should I connect QuickBooks for real-time updates?",
                "Find duplicate transactions"
            ]
        else:
            # Has both - advanced questions
            return [
                "What are my biggest financial risks right now?",
                "Find cost-saving opportunities",
                "Compare this month vs. last month"
            ]
    
    # Intent handlers for specific question types
    
    async def _handle_greeting(
        self,
        question: str,
        user_id: str,
        conversation_history: list[Dict[str, str]] = None,
        memory_manager: Optional[Any] = None
    ) -> ChatResponse:
        """Handle greeting intent with warm, personalized response using .with_structured_output()."""
        try:
            
            # Build context about user
            user_context = ""
            if memory_manager:
                stats = await memory_manager.get_memory_stats()
                user_context = f"User has {stats.get('message_count', 0)} previous messages in conversation."
            
            # REFACTORED: Use PromptLoader instead of inline prompts
            system_prompt = self.prompt_loader.get_prompt('greeting', 'system')
            user_prompt_template = self.prompt_loader.get_prompt('greeting', 'user')
            user_prompt = user_prompt_template.replace('{{ question }}', question).replace('{{ user_context }}', user_context if user_context else '')
            
            
            # REFACTORED: Use native LangChain with PromptLoader
            structured_llm = self.groq.with_structured_output(GreetingResponse)
            response = await asyncio.wait_for(
                structured_llm.ainvoke([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]),
                timeout=10.0
            )
            
            answer = f"{response.greeting_message} {response.follow_up}"
            logger.info("Greeting handled with instructor", tone=response.tone)
            
            return ChatResponse(
                answer=answer,
                question_type=QuestionType.GENERAL,
                confidence=1.0,
                data={"intent": "greeting", "tone": response.tone}
            )
        
        except Exception as e:
            logger.error("Greeting handler failed", error=str(e))
            return ChatResponse(
                answer="Hello! I'm Finley, your AI finance assistant. How can I help you today?",
                question_type=QuestionType.GENERAL,
                confidence=0.8,
                data={}
            )
    
    async def _handle_smalltalk(
        self,
        question: str,
        user_id: str,
        conversation_history: list[Dict[str, str]] = None,
        memory_manager: Optional[Any] = None
    ) -> ChatResponse:
        """Handle smalltalk with friendly, brief response."""
        try:
            smalltalk_prompt = "You are Finley, a friendly AI finance assistant. Keep responses brief and warm (1-2 sentences)."
            response = await self.groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": smalltalk_prompt},
                    {"role": "user", "content": question}
                ],
                max_tokens=150,
                temperature=0.7
            )
            answer = response.choices[0].message.content
            await self._cache_response(question, user_id, answer)
            return ChatResponse(
                answer=answer,
                question_type=QuestionType.GENERAL,
                confidence=0.85,
                data={"cached": False}
            )
        
        except Exception as e:
            logger.error("Smalltalk handler failed", error=str(e))
            return ChatResponse(
                answer="I appreciate the chat! How can I help with your finances today?",
                question_type=QuestionType.GENERAL,
                confidence=0.7,
                data={}
            )
    
    async def _handle_capability_summary(
        self,
        question: str,
        user_id: str,
        conversation_history: list[Dict[str, str]] = None,
        memory_manager: Optional[Any] = None
    ) -> ChatResponse:
        """Handle capability summary request with structured response using .with_structured_output()."""
        try:
            # REFACTORED: Use PromptLoader instead of inline prompts
            system_prompt = self.prompt_loader.get_prompt('capability_summary', 'system')
            user_prompt = self.prompt_loader.get_prompt('capability_summary', 'user')
            
            # REFACTORED: Use native LangChain with PromptLoader
            structured_llm = self.groq.with_structured_output(CapabilitySummaryResponse)
            response = await asyncio.wait_for(
                structured_llm.ainvoke([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]),
                timeout=10.0
            )
            
            capabilities_text = "\n".join(f"• {cap}" for cap in response.capabilities)
            features_text = "\n".join(f"• {feat}" for feat in response.key_features)
            
            answer = f"""I can help you with:

{capabilities_text}

Key features:
{features_text}

{response.next_step}"""
            
            logger.info("Capability summary handled with instructor")
            
            return ChatResponse(
                answer=answer,
                question_type=QuestionType.GENERAL,
                confidence=1.0,
                data={"intent": "capability_summary"}
            )
        
        except Exception as e:
            logger.error("Capability summary handler failed", error=str(e))
            return ChatResponse(
                answer="I can help with causal analysis, temporal patterns, relationship mapping, what-if scenarios, anomaly detection, and predictive forecasting. What would you like to explore?",
                question_type=QuestionType.GENERAL,
                confidence=0.8,
                data={}
            )
    
    async def _handle_system_flow(
        self,
        question: str,
        user_id: str,
        conversation_history: list[Dict[str, str]] = None,
        memory_manager: Optional[Any] = None
    ) -> ChatResponse:
        """Handle system flow explanation with structured response using .with_structured_output()."""
        try:
            
            prompt = """Explain the system flow for using Finley AI finance assistant.

Include:
1. Main steps in the flow (connect data, ask questions, get insights, decide)
2. Where the user currently is (if possible from context)
3. What they should do next"""
            
            structured_llm = self.groq.with_structured_output(SystemFlowResponse)
            response = await asyncio.wait_for(
                structured_llm.ainvoke([
                    {"role": "system", "content": "You are Finley. Explain the system flow clearly."},
                    {"role": "user", "content": prompt}
                ]),
                timeout=10.0
            )
            
            steps_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(response.flow_steps))
            
            answer = f"""Here's how the system works:

{steps_text}

{response.next_step}"""
            
            logger.info("System flow handled with instructor")
            
            return ChatResponse(
                answer=answer,
                question_type=QuestionType.GENERAL,
                confidence=1.0,
                data={"intent": "system_flow"}
            )
        
        except Exception as e:
            logger.error("System flow handler failed", error=str(e))
            return ChatResponse(
                answer="The system works like this: 1) Connect your financial data 2) Ask me questions 3) Get AI-powered insights 4) Make better decisions. What would you like to know?",
                question_type=QuestionType.GENERAL,
                confidence=0.7,
                data={}
            )
    
    async def _handle_differentiator(
        self,
        question: str,
        user_id: str,
        conversation_history: list[Dict[str, str]] = None,
        memory_manager: Optional[Any] = None
    ) -> ChatResponse:
        """Handle differentiator explanation with structured response using instructor."""
        try:
            if not INSTRUCTOR_AVAILABLE:
                return ChatResponse(
                    answer="I use advanced AI with causal inference, temporal pattern learning, and semantic relationship extraction to provide insights that generic finance tools can't.",
                    question_type=QuestionType.GENERAL,
                    confidence=0.7,
                    data={}
                )
            
            prompt = """Explain what makes Finley different from other finance tools.

Include:
1. Key differentiators (AI-powered, causal inference, pattern detection, etc.)
2. Unique value proposition
3. Proof or evidence of differentiation"""
            
            structured_llm = self.groq.with_structured_output(DifferentiatorResponse)
            response = await asyncio.wait_for(
                structured_llm.ainvoke([
                        {"role": "system", "content": "You are Finley. Explain your unique value clearly and confidently."},
                        {"role": "user", "content": prompt}
                    ]),
                timeout=10.0
            )
            
            differentiators_text = "\n".join(f"• {diff}" for diff in response.differentiators)
            
            answer = f"""Here's what makes me different:

{differentiators_text}

Unique value: {response.unique_value}

Proof: {response.proof_point}"""
            
            logger.info("Differentiator handled with instructor")
            
            return ChatResponse(
                answer=answer,
                question_type=QuestionType.GENERAL,
                confidence=1.0,
                data={"intent": "differentiator"}
            )
        
        except Exception as e:
            logger.error("Differentiator handler failed", error=str(e))
            return ChatResponse(
                answer="I use advanced AI with causal inference, temporal pattern learning, and semantic relationship extraction to provide insights that generic finance tools can't.",
                question_type=QuestionType.GENERAL,
                confidence=0.7,
                data={}
            )
    
    async def _handle_meta_feedback(
        self,
        question: str,
        user_id: str,
        conversation_history: list[Dict[str, str]] = None,
        memory_manager: Optional[Any] = None
    ) -> ChatResponse:
        """Handle meta feedback and suggestions."""
        try:
            # For meta feedback, use simple LLM response
            messages = [
                {"role": "system", "content": "You are Finley. Acknowledge feedback warmly and thank the user for their input."},
                {"role": "user", "content": question}
            ]
            
            response = await asyncio.wait_for(
                self.groq.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    max_tokens=150,
                    temperature=0.7
                ),
                timeout=10.0
            )
            
            answer = response.choices[0].message.content.strip()
            logger.info("Meta feedback handled")
            
            return ChatResponse(
                answer=answer,
                question_type=QuestionType.GENERAL,
                confidence=0.9,
                data={"intent": "meta_feedback"}
            )
        
        except Exception as e:
            logger.error("Meta feedback handler failed", error=str(e))
            return ChatResponse(
                answer="Thank you for your feedback! I appreciate it and will continue to improve.",
                question_type=QuestionType.GENERAL,
                confidence=0.7,
                data={}
            )
    
    async def _handle_help(
        self,
        question: str,
        user_id: str,
        conversation_history: list[Dict[str, str]] = None,
        memory_manager: Optional[Any] = None
    ) -> ChatResponse:
        """Handle help requests with structured response using instructor."""
        try:
            if not INSTRUCTOR_AVAILABLE:
                help_topics = [
                    "Getting started - How to connect your data",
                    "Question types - What kinds of questions I can answer",
                    "Data sources - What financial platforms I support",
                    "Troubleshooting - Common issues and solutions"
                ]
                answer = "I can help with:\n" + "\n".join(f"• {topic}" for topic in help_topics)
                return ChatResponse(
                    answer=answer,
                    question_type=QuestionType.GENERAL,
                    confidence=0.8,
                    data={}
                )
            
            prompt = f"""Generate help information for a user asking: "{question}"

Include:
1. Available help topics
2. Suggested topic based on their question
3. Contact info if they need more help"""
            
            structured_llm = self.groq.with_structured_output(HelpResponse)
            response = await asyncio.wait_for(
                structured_llm.ainvoke([
                        {"role": "system", "content": "You are Finley's help system. Provide helpful information."},
                        {"role": "user", "content": prompt}
                    ]),
                timeout=10.0
            )
            
            topics_text = "\n".join(f"• {topic}" for topic in response.help_topics)
            
            answer = f"""I can help with:

{topics_text}

Based on your question, I suggest: {response.suggested_topic}"""
            
            if response.contact_info:
                answer += f"\n\nFor more help: {response.contact_info}"
            
            logger.info("Help handled with instructor")
            
            return ChatResponse(
                answer=answer,
                question_type=QuestionType.GENERAL,
                confidence=1.0,
                data={"intent": "help"}
            )
        
        except Exception as e:
            logger.error("Help handler failed", error=str(e))
            return ChatResponse(
                answer="I can help with getting started, question types, data sources, and troubleshooting. What do you need help with?",
                question_type=QuestionType.GENERAL,
                confidence=0.7,
                data={}
            )
    
    # Helper methods for formatting and data retrieval
    
    async def _extract_entities_from_question(
        self,
        question: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Extract entities from question using spaCy NER. Returns dict with entities, metrics, time_periods, confidence."""
        try:
            # Use global spaCy model (loaded once at module level)
            if _spacy_nlp is None:
                logger.warning("spaCy model not available - using fallback extraction")
                return _get_fallback_entities(question)
            
            # Run spaCy NER on question
            doc = await asyncio.to_thread(lambda: _spacy_nlp(question))
            
            # Extract entities by type
            entities = []
            metrics = []
            time_periods = []
            confidence_scores = []
            
            # Financial entity labels to extract
            financial_entity_types = {
                "MONEY": "metrics",      # Monetary amounts
                "DATE": "time_periods",  # Dates and time periods
                "ORG": "entities",       # Organizations (vendors, customers)
                "PERSON": "entities",    # People
                "GPE": "entities",       # Locations
                "PRODUCT": "entities",   # Products/services
                "EVENT": "entities",     # Events
            }
            
            # Extract entities from spaCy results
            for ent in doc.ents:
                entity_text = ent.text.lower()
                entity_label = ent.label_
                
                # spaCy provides confidence via ent._.confidence if available
                # Default to 0.85 for spaCy entities (high accuracy)
                entity_confidence = getattr(ent._, 'confidence', 0.85)
                confidence_scores.append(entity_confidence)
                
                # Categorize by entity type
                if entity_label in financial_entity_types:
                    category = financial_entity_types[entity_label]
                    if category == "metrics":
                        metrics.append(entity_text)
                    elif category == "time_periods":
                        time_periods.append(entity_text)
                    else:
                        entities.append(entity_text)
            
            
            # Extract financial keywords from EntityRuler patterns
            # REFACTORED: EntityRuler automatically tags financial keywords as FINANCIAL_KEYWORD entities
            nlp = _load_spacy_model()
            
            if nlp and "entity_ruler" in nlp.pipe_names:
                # EntityRuler already processed the question in the NER step above
                # Just extract FINANCIAL_KEYWORD entities
                for ent in doc.ents:
                    if ent.label_ == "FINANCIAL_KEYWORD":
                        keyword = ent.text.lower()
                        if keyword not in metrics:
                            metrics.append(keyword)
                            confidence_scores.append(0.95)  # High confidence for EntityRuler matches
            else:
                # Fallback to simple string matching if EntityRuler unavailable
                financial_keywords = [
                    "revenue", "expenses", "cash flow", "profit", "margin",
                    "inventory", "receivables", "payables", "assets", "liabilities",
                    "equity", "ebitda", "gross profit", "net income", "operating income",
                    "cost of goods sold", "operating expenses", "interest expense",
                    "tax expense", "depreciation", "amortization", "cash", "debt",
                    "vendor", "customer", "supplier", "transaction", "invoice",
                    "payment", "receipt", "balance", "account", "budget"
                ]
                
                question_lower = question.lower()
                for keyword in financial_keywords:
                    if keyword in question_lower and keyword not in metrics:
                        metrics.append(keyword)
                        confidence_scores.append(0.9)  # High confidence for known keywords
            
            # Remove duplicates while preserving order
            entities = list(dict.fromkeys(entities))
            metrics = list(dict.fromkeys(metrics))
            time_periods = list(dict.fromkeys(time_periods))
            
            # Calculate average confidence
            combined_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            
            result = {
                "entities": entities,
                "metrics": metrics,
                "time_periods": time_periods,
                "confidence": combined_confidence
            }
            
            logger.info("spaCy entity extraction completed",
                       entity_count=len(entities),
                       metric_count=len(metrics),
                       time_period_count=len(time_periods),
                       confidence=round(combined_confidence, 2))
            
            return result
        
        except asyncio.TimeoutError:
            logger.error("Entity extraction timeout - using fallback")
            return _create_error_response("entity_extraction", "timeout", _get_fallback_entities(question))
        except Exception as e:
            logger.error("Entity extraction failed", error=str(e))
            return _create_error_response("entity_extraction", str(e), _get_fallback_entities(question))
    
    async def _extract_scenario_from_question(
        self,
        question: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Extract scenario parameters from what-if question. Returns dict with scenario details."""
        try:
            if not INSTRUCTOR_AVAILABLE:
                logger.warning("instructor not available - scenario extraction returning basic dict")
                return {'question': question}
            
            # Build prompt for scenario extraction
            prompt = f"""Analyze this what-if question and extract scenario parameters.

QUESTION: {question}

Extract:
1. Scenario type: sensitivity_analysis (what if X changes?), forecast (predict future), comparison (compare scenarios), or impact_analysis (impact of change)
2. Base metric: The main metric being analyzed (e.g., cash flow, revenue, profit)
3. Variables: What variables are being changed (e.g., payment delay, hiring cost, price increase)
4. Changes: Specific changes to apply (e.g., "delay by 30 days", "increase by 20%", "reduce to $50k")

Be precise about the changes mentioned."""
            
            # Use instructor for type-safe extraction
            structured_llm = self.groq.with_structured_output(ScenarioExtraction)
            response = await asyncio.wait_for(
                structured_llm.ainvoke([
                        {"role": "system", "content": "You are a financial scenario analysis expert. Extract scenario parameters from what-if questions."},
                        {"role": "user", "content": prompt}
                    ]),
                timeout=30.0
            )
            
            logger.info("Scenario extracted with instructor",
                       scenario_type=response.scenario_type,
                       base_metric=response.base_metric,
                       variable_count=len(response.variables),
                       confidence=response.confidence)
            
            return {
                "scenario_type": response.scenario_type,
                "base_metric": response.base_metric,
                "variables": response.variables,
                "changes": response.changes,
                "confidence": response.confidence,
                "question": question
            }
        
        except asyncio.TimeoutError:
            logger.error("Scenario extraction timeout")
            return _create_error_response("scenario_extraction", "timeout", {'question': question, 'scenario_type': 'unknown', 'variables': [], 'changes': []})
        except Exception as e:
            logger.error("Scenario extraction failed", error=str(e))
            return _create_error_response("scenario_extraction", str(e), {'question': question, 'scenario_type': 'unknown', 'variables': [], 'changes': []})
    
    async def _extract_entity_id_from_question(
        self,
        question: str,
        user_id: str,
        context: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Extract entity ID from question. Returns entity identifier string or None."""
        try:
            if not INSTRUCTOR_AVAILABLE:
                logger.warning("instructor not available - entity ID extraction returning None")
                return None
            
            # Build context information if available
            context_info = ""
            if context:
                context_info = f"\nCONTEXT: {str(context)[:500]}"
            
            # Build prompt for entity ID extraction
            prompt = f"""Analyze this question and extract the specific entity being referenced.

QUESTION: {question}{context_info}

Extract:
1. Entity type: What kind of entity (invoice, transaction, vendor, customer, etc.)
2. Entity identifier: The specific ID or name (e.g., 'INV-12345', 'Acme Corp', 'TXN-789')
3. Search field: Which field to search in (id, name, reference_number, etc.)

If no specific entity is mentioned, return empty string for entity_identifier."""
            
            # Use instructor for type-safe extraction
            structured_llm = self.groq.with_structured_output(EntityIDExtraction)
            response = await asyncio.wait_for(
                structured_llm.ainvoke([
                        {"role": "system", "content": "You are a financial data extraction expert. Extract specific entity IDs from questions."},
                        {"role": "user", "content": prompt}
                    ]),
                timeout=30.0
            )
            
            # Return entity identifier if found and confidence is high
            if response.entity_identifier and response.confidence > 0.5:
                logger.info("Entity ID extracted with instructor",
                           entity_type=response.entity_type,
                           entity_identifier=response.entity_identifier,
                           confidence=response.confidence)
                return response.entity_identifier
            else:
                logger.debug("Entity ID extraction low confidence or empty", 
                            confidence=response.confidence)
                return None
        
        except asyncio.TimeoutError:
            logger.error("Entity ID extraction timeout")
            return None
        except Exception as e:
            logger.error("Entity ID extraction failed", error=str(e))
            return None
    
    async def _extract_query_params_from_question(
        self,
        question: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Extract query parameters from data query question. Returns dict with filters, sort_by, limit, group_by."""
        try:
            if not INSTRUCTOR_AVAILABLE:
                logger.warning("instructor not available - query parameter extraction using fallback")
                return {'filters': {}, 'limit': 100, 'error': False}
            
            # Build prompt for query parameter extraction
            prompt = f"""Analyze this data query question and extract query parameters.

QUESTION: {question}

Extract:
1. Filters: Conditions to filter data (e.g., "amount > 1000", "date in Q1", "vendor = Acme")
2. Sort by: Field to sort results (default: date)
3. Limit: Maximum number of results (default: 100, max: 10000)
4. Group by: Fields to group results (e.g., "vendor", "category", "month")

Be specific about filter conditions and sorting preferences."""
            
            # Use instructor for type-safe extraction
            structured_llm = self.groq.with_structured_output(QueryParameterExtraction)
            response = await asyncio.wait_for(
                structured_llm.ainvoke([
                        {"role": "system", "content": "You are a database query expert. Extract query parameters from natural language questions."},
                        {"role": "user", "content": prompt}
                    ]),
                timeout=30.0
            )
            
            logger.info("Query parameters extracted with instructor",
                       filter_count=len(response.filters),
                       sort_by=response.sort_by,
                       limit=response.limit,
                       group_by_count=len(response.group_by),
                       confidence=response.confidence)
            
            return {
                "filters": response.filters,
                "sort_by": response.sort_by,
                "limit": response.limit,
                "group_by": response.group_by,
                "confidence": response.confidence
            }
        
        except asyncio.TimeoutError:
            logger.error("Query parameter extraction timeout")
            return _create_error_response("query_extraction", "timeout", {'filters': {}, 'limit': 100})
        except Exception as e:
            logger.error("Query parameter extraction failed", error=str(e))
            return _create_error_response("query_extraction", str(e), {'filters': {}, 'limit': 100})
    
    async def _format_causal_response(
        self,
        question: str,
        causal_results: Dict[str, Any],
        entities: Dict[str, Any]
    ) -> str:
        """
        Format causal analysis results using Jinja2 template.
        
        REFACTORED: Replaced string concatenation with template rendering.
        """
        try:
            template = self.jinja_env.get_template('causal_response.j2')
            
            # Prepare actions
            actions = self._generate_causal_actions(causal_results)
            
            # Prepare follow-up questions
            follow_up_questions = [
                "What if I address the root cause?",
                "Show me the complete causal chain",
                "Are there other contributing factors?"
            ]
            
            return template.render(
                question=question,
                causal_results=causal_results,
                actions=actions,
                follow_up_questions=follow_up_questions
            )
        
        except Exception as e:
            logger.error(f"Template rendering failed: {e}")
            # Simple fallback
            causal_rels = causal_results.get('causal_relationships', [])
            return f"I analyzed your data and found {len(causal_rels)} causal relationships."
    
    async def _format_temporal_response(
        self,
        question: str,
        patterns_result: Dict[str, Any]
    ) -> str:
        """
        Format temporal pattern results using Jinja2 template.
        
        REFACTORED: Replaced string concatenation with template rendering.
        """
        try:
            template = self.jinja_env.get_template('temporal_response.j2')
            
            # Extract predictions and insights
            patterns = patterns_result.get('patterns', [])
            predictions = patterns_result.get('predictions', [])
            
            # Prepare follow-up questions
            follow_up_questions = [
                "When is the next expected occurrence?",
                "What's the forecast for next month?",
                "Show me the seasonal patterns"
            ]
            
            return template.render(
                question=question,
                predictions=predictions,
                temporal_insights=patterns_result.get('insights', {}),
                recommendations=patterns_result.get('recommendations', []),
                follow_up_questions=follow_up_questions
            )
        
        except Exception as e:
            logger.error(f"Template rendering failed: {e}")
            patterns = patterns_result.get('patterns', [])
            return f"I've learned {len(patterns)} temporal patterns from your data."
    
    async def _format_relationship_response(
        self,
        question: str,
        relationships_result: Dict[str, Any]
    ) -> str:
        """Format relationship detection using Jinja2 template."""
        try:
            template = self.jinja_env.get_template('relationship_response.j2')
            
            relationships = relationships_result.get('relationships', [])
            
            # Group by type
            by_type = {}
            for rel in relationships:
                rel_type = rel.get('relationship_type', 'unknown')
                by_type[rel_type] = by_type.get(rel_type, 0) + 1
            
            return template.render(
                question=question,
                relationships=relationships,
                relationship_counts=by_type,
                follow_up_questions=[
                    "Show me the strongest relationships",
                    "Which entities are most connected?",
                    "Are there any unusual patterns?"
                ]
            )
        except Exception as e:
            logger.error(f"Template rendering failed: {e}")
            relationships = relationships_result.get('relationships', [])
            return f"I found {len(relationships)} relationships in your financial data."
    
    async def _format_provenance_explanation(
        self,
        provenance_data: Dict[str, Any]
    ) -> str:
        """Format provenance data into human-readable explanation"""
        answer = "Here's the complete story of this data:\n\n"
        
        # Add source information
        source = provenance_data.get('source', {})
        answer += f"**Source**: {source.get('filename', 'Unknown')}\n"
        answer += f"**Uploaded**: {source.get('upload_date', 'Unknown')}\n\n"
        
        # Add transformation chain
        lineage = provenance_data.get('lineage_path', [])
        if lineage:
            answer += "**Transformation Chain**:\n"
            for i, step in enumerate(lineage, 1):
                answer += f"{i}. {step.get('step', 'Unknown')} - {step.get('operation', 'Unknown')}\n"
        
        return answer
    
    def _generate_causal_actions(
        self,
        causal_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate suggested actions based on causal analysis"""
        return [
            {"type": "view_details", "label": "View Detailed Analysis"},
            {"type": "run_whatif", "label": "Run What-If Scenario"},
            {"type": "export", "label": "Export Results"}
        ]
    
    def _generate_temporal_visualizations(
        self,
        patterns_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate visualization data for temporal patterns"""
        return [
            {
                "type": "line_chart",
                "title": "Temporal Patterns Over Time",
                "data": patterns_result.get('patterns', [])
            }
        ]
    
    def _generate_relationship_visualizations(
        self,
        relationships_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate visualization data for relationships"""
        return [
            {
                "type": "network_graph",
                "title": "Relationship Network",
                "data": relationships_result.get('relationships', [])
            }
        ]
    
    async def _load_user_long_term_memory(self, user_id: str) -> Dict[str, Any]:
        """
        Load user's long-term memory using Mem0.
        
        REFACTORED: Replaced 60 lines of SQL queries with Mem0 library.
        - Automatic memory extraction from conversations
        - Semantic search over preferences
        - Built-in persistence
        """
        try:
            from mem0 import Memory
            
            # Initialize Mem0 (lazy load)
            memory = Memory()
            
            # Get user memories
            memories = memory.get_all(user_id=user_id, limit=20)
            
            # Format for context
            preferences = []
            recurring_topics = []
            
            for mem in memories:
                content = mem.get('memory', '')
                if 'prefers' in content.lower() or 'likes' in content.lower():
                    preferences.append(content)
                else:
                    recurring_topics.append(content)
            
            return {
                'preferences': preferences[:10],
                'recurring_topics': recurring_topics[:10],
                'total_memories': len(memories)
            }
        
        except ImportError:
            logger.warning("Mem0 not available. Install: pip install mem0ai")
            # Fallback to empty
            return {'preferences': [], 'recurring_topics': [], 'total_memories': 0}
        except Exception as e:
            logger.error(f"Failed to load long-term memory: {e}")
            return {'preferences': [], 'recurring_topics': [], 'total_memories': 0}
    
    async def _save_user_insight(self, user_id: str, insight: str, category: str):
        """Save important insights to long-term memory for future reference"""
        try:
            # Store in user_preferences table (upsert)
            self.supabase.table('user_preferences').upsert({
                'user_id': user_id,
                'last_insight': insight,
                'last_insight_category': category,
                'last_insight_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat()
            }, on_conflict='user_id').execute()
        except Exception as e:
            logger.error("Failed to save insight", error=str(e))
            return {'error': True, 'message': f'Failed to save insight: {e}'}
    
    async def _fetch_user_data_context(self, user_id: str) -> str:
        """Fetch user's actual data context for personalized responses."""
        try:
            # Load long-term memory first
            long_term_memory = await self._load_user_long_term_memory(user_id)
            
            # Query active data sources
            connections_result = self.supabase.table('user_connections').select('*').eq('user_id', user_id).eq('status', 'active').execute()
            connected_sources = [conn['provider'] for conn in connections_result.data] if connections_result.data else []
            
            # Query uploaded files
            files_result = self.supabase.table('raw_records').select('file_name, created_at').eq('user_id', user_id).order('created_at', desc=True).limit(5).execute()
            recent_files = [f['file_name'] for f in files_result.data] if files_result.data else []
            
            # Query transaction summary (last 90 days)
            ninety_days_ago = (datetime.utcnow() - timedelta(days=90)).isoformat()
            events_result = self.supabase.table('raw_events')\
                .select('id, source_platform, ingest_ts, payload, classification_metadata')\
                .eq('user_id', user_id)\
                .gte('ingest_ts', ninety_days_ago)\
                .order('ingest_ts', desc=True)\
                .limit(1000)\
                .execute()
            
            total_transactions = len(events_result.data) if events_result.data else 0
            platforms = list(set([e.get('source_platform', 'unknown') for e in events_result.data])) if events_result.data else []
            
            # Calculate financial summary
            total_revenue = 0
            total_expenses = 0
            for event in (events_result.data or []):
                payload = event.get('payload', {})
                classification = event.get('classification_metadata', {})
                amount = float(payload.get('amount', 0) or payload.get('total_amount', 0) or 0)
                
                # Determine if revenue or expense based on classification
                kind = classification.get('kind', '').lower() if classification else ''
                if 'revenue' in kind or 'income' in kind or 'invoice' in kind:
                    total_revenue += abs(amount)
                elif 'expense' in kind or 'bill' in kind or 'payment' in kind:
                    total_expenses += abs(amount)
            
            # Query entities (vendors/customers)
            entities_result = self.supabase.table('normalized_entities').select('canonical_name, entity_type').eq('user_id', user_id).limit(20).execute()
            top_entities = [e['canonical_name'] for e in entities_result.data[:5]] if entities_result.data else []
            
            # Check if user has data
            has_data = total_transactions > 0
            
            # Build context string with financial summary AND long-term memory
            net_income = total_revenue - total_expenses
            
            # Add long-term memory context
            memory_context = ""
            if long_term_memory['recurring_questions']:
                topics = [f"{topic} ({count}x)" for topic, count in long_term_memory['recurring_questions']]
                memory_context = f"\n\nUSER'S RECURRING INTERESTS: {', '.join(topics)}"
            
            if long_term_memory['business_context']:
                biz = long_term_memory['business_context']
                if biz.get('industry'):
                    memory_context += f"\nBUSINESS TYPE: {biz.get('industry')}"
                if biz.get('size'):
                    memory_context += f" | SIZE: {biz.get('size')}"
            
            # Build context using Jinja2 templates (easily editable by non-developers)
            if not has_data:
                # NO DATA - Don't show zeros, show onboarding guidance
                no_data_template = Template("""CONNECTED DATA SOURCES: None yet
RECENT FILES UPLOADED: None yet
TOTAL TRANSACTIONS (Last 90 days): 0
PLATFORMS DETECTED: None
TOP ENTITIES: None yet

DATA STATUS: No data connected yet

NEXT STEPS FOR USER:
1. Connect a data source (QuickBooks, Xero, Stripe, Razorpay, PayPal, etc.)
2. OR upload financial files (CSV, Excel, PDF invoices/statements)
3. OR connect email (Gmail/Zoho Mail) to extract attachments

RECOMMENDATION: Guide user to connect data first before providing financial analysis{{ memory_context }}""")
                context = no_data_template.render(memory_context=memory_context)
            else:
                # HAS DATA - Show financial summary with actual numbers
                has_data_template = Template("""CONNECTED DATA SOURCES: {{ connected_sources }}
RECENT FILES UPLOADED: {{ recent_files }}
TOTAL TRANSACTIONS (Last 90 days): {{ total_transactions }}
PLATFORMS DETECTED: {{ platforms }}
TOP ENTITIES: {{ top_entities }}

FINANCIAL SUMMARY (Last 90 days):
- Total Revenue: ${{ total_revenue:,.2f }}
- Total Expenses: ${{ total_expenses:,.2f }}
- Net Income: ${{ net_income:,.2f }}
- Profit Margin: {{ profit_margin }}%{{ memory_context }}

DATA STATUS: {{ data_status }}""")
                
                profit_margin = (net_income / total_revenue * 100) if total_revenue > 0 else 0.0
                data_status = 'Rich data available - provide specific, quantified insights!' if total_transactions > 50 else 'Limited data - encourage user to connect more sources or upload files'
                
                context = has_data_template.render(
                    connected_sources=', '.join(connected_sources) if connected_sources else 'None yet',
                    recent_files=', '.join(recent_files) if recent_files else 'None yet',
                    total_transactions=total_transactions,
                    platforms=', '.join(platforms) if platforms else 'None',
                    top_entities=', '.join(top_entities) if top_entities else 'None yet',
                    total_revenue=total_revenue,
                    total_expenses=total_expenses,
                    net_income=net_income,
                    profit_margin=f"{profit_margin:.1f}",
                    memory_context=memory_context,
                    data_status=data_status
                )
            
            return context
            
        except Exception as e:
            logger.error("Failed to fetch user data context", error=str(e))
            return "DATA STATUS: Unable to fetch user data context"
    
    async def _load_conversation_history(
        self,
        user_id: str,
        chat_id: str,
        limit: int = 20
    ) -> list[Dict[str, str]]:
        """Load most recent conversation history from database."""
        try:
            # Query most recent N messages (descending = newest first)
            result = self.supabase.table('chat_messages')\
                .select('role, message, created_at')\
                .eq('user_id', user_id)\
                .eq('chat_id', chat_id)\
                .order('created_at', desc=True)\
                .limit(limit)\
                .execute()
            
            if not result.data:
                return []
            
            # Reverse to chronological order (oldest → newest for conversation flow)
            history = []
            for msg in reversed(result.data):
                history.append({
                    'role': msg['role'],
                    'content': msg['message']
                })
            
            logger.info("Loaded recent conversation history", message_count=len(history))
            return history
            
        except Exception as e:
            logger.error("Failed to load conversation history", error=str(e))
            return []
    
    # REMOVED: _summarize_conversation() - handled by LangChain's ConversationSummaryBufferMemory
    
    # Graph intelligence methods for temporal, seasonal, fraud, and predictive insights
    
    async def _get_temporal_insights(self, user_id: str, source_id: str, target_id: str) -> Optional[Dict[str, Any]]:
        """Get temporal pattern insights from FinleyGraph."""
        try:
            # Build graph if not already built
            if not self.graph_engine.graph:
                await self.graph_engine.build_graph(user_id)
            
            # Find path between entities
            path = self.graph_engine.find_path(source_id, target_id)
            if not path:
                return None
            
            # Extract temporal patterns
            temporal_patterns = []
            for edge_data in path.path_edges:
                if edge_data.get('recurrence_frequency') and edge_data.get('recurrence_frequency') != 'none':
                    temporal_patterns.append({
                        'frequency': edge_data.get('recurrence_frequency'),
                        'score': edge_data.get('recurrence_score', 0.0),
                        'next_occurrence': edge_data.get('next_predicted_occurrence'),
                        'relationship': edge_data.get('relationship_type')
                    })
            
            if temporal_patterns:
                return {
                    'patterns': temporal_patterns,
                    'count': len(temporal_patterns),
                    'confidence': sum(p['score'] for p in temporal_patterns) / max(1, len(temporal_patterns))
                }
            return None
        except Exception as e:
            logger.warning("temporal_insights_failed", error=str(e))
            return None
    
    async def _get_seasonal_insights(self, user_id: str, source_id: str, target_id: str) -> Optional[Dict[str, Any]]:
        """Get seasonal pattern insights from FinleyGraph."""
        try:
            if not self.graph_engine.graph:
                await self.graph_engine.build_graph(user_id)
            
            path = self.graph_engine.find_path(source_id, target_id)
            if not path:
                return None
            
            seasonal_cycles = []
            for edge_data in path.path_edges:
                if edge_data.get('seasonal_months'):
                    seasonal_cycles.append({
                        'months': edge_data.get('seasonal_months'),
                        'strength': edge_data.get('seasonal_strength', 0.0),
                        'relationship': edge_data.get('relationship_type')
                    })
            
            if seasonal_cycles:
                month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                              7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
                return {
                    'cycles': seasonal_cycles,
                    'count': len(seasonal_cycles),
                    'peak_months': [month_names.get(m, str(m)) for cycle in seasonal_cycles for m in cycle['months']],
                    'confidence': sum(c['strength'] for c in seasonal_cycles) / max(1, len(seasonal_cycles))
                }
            return None
        except Exception as e:
            logger.warning("seasonal_insights_failed", error=str(e))
            return None
    
    async def _get_fraud_warnings(self, user_id: str, source_id: str, target_id: str) -> Optional[Dict[str, Any]]:
        """Get fraud detection warnings from FinleyGraph."""
        try:
            if not self.graph_engine.graph:
                await self.graph_engine.build_graph(user_id)
            
            path = self.graph_engine.find_path(source_id, target_id)
            if not path:
                return None
            
            fraud_alerts = []
            total_fraud_score = 0.0
            for edge_data in path.path_edges:
                if edge_data.get('is_duplicate'):
                    duplicate_confidence = edge_data.get('duplicate_confidence', 0.0)
                    fraud_alerts.append({
                        'relationship': edge_data.get('relationship_type'),
                        'confidence': duplicate_confidence,
                        'reasoning': edge_data.get('reasoning', 'Duplicate detected')
                    })
                    total_fraud_score += duplicate_confidence
            
            if fraud_alerts:
                fraud_risk = total_fraud_score / max(1, len(path.path_edges))
                return {
                    'alerts': fraud_alerts,
                    'count': len(fraud_alerts),
                    'risk_score': fraud_risk,
                    'severity': 'HIGH' if fraud_risk > 0.7 else 'MEDIUM' if fraud_risk > 0.4 else 'LOW'
                }
            return None
        except Exception as e:
            logger.warning("fraud_detection_failed", error=str(e))
            return None
    
    async def _get_root_cause_analysis(self, user_id: str, source_id: str, target_id: str) -> Optional[Dict[str, Any]]:
        """Get root cause analysis from FinleyGraph."""
        try:
            if not self.graph_engine.graph:
                await self.graph_engine.build_graph(user_id)
            
            path = self.graph_engine.find_path(source_id, target_id)
            if not path:
                return None
            
            root_causes = []
            causal_chain = []
            total_causal_strength = 0.0
            
            for i, edge_data in enumerate(path.path_edges):
                if edge_data.get('root_cause_analysis'):
                    root_causes.append(edge_data['root_cause_analysis'])
                
                causal_strength = edge_data.get('causal_strength', 0.0)
                total_causal_strength += causal_strength
                causal_chain.append({
                    'step': i + 1,
                    'relationship': edge_data.get('relationship_type'),
                    'strength': causal_strength,
                    'direction': edge_data.get('causal_direction', 'unknown'),
                    'reasoning': edge_data.get('reasoning', '')
                })
            
            if root_causes or causal_chain:
                return {
                    'root_causes': root_causes,
                    'causal_chain': causal_chain,
                    'chain_length': len(causal_chain),
                    'total_causal_strength': total_causal_strength,
                    'avg_causal_strength': total_causal_strength / max(1, len(causal_chain))
                }
            return None
        except Exception as e:
            logger.warning("root_cause_analysis_failed", error=str(e))
            return None
    
    async def _get_predictions(self, user_id: str, source_id: str, target_id: str) -> Optional[Dict[str, Any]]:
        """Get future predictions from FinleyGraph."""
        try:
            if not self.graph_engine.graph:
                await self.graph_engine.build_graph(user_id)
            
            path = self.graph_engine.find_path(source_id, target_id)
            if not path:
                return None
            
            predictions = []
            for edge_data in path.path_edges:
                if edge_data.get('prediction_confidence'):
                    prediction_confidence = edge_data.get('prediction_confidence', 0.0)
                    predictions.append({
                        'relationship': edge_data.get('relationship_type'),
                        'confidence': prediction_confidence,
                        'reason': edge_data.get('prediction_reason', 'Pattern-based prediction'),
                        'next_occurrence': edge_data.get('next_predicted_occurrence')
                    })
            
            if predictions:
                return {
                    'predictions': predictions,
                    'count': len(predictions),
                    'avg_confidence': sum(p['confidence'] for p in predictions) / max(1, len(predictions))
                }
            return None
        except Exception as e:
            logger.warning("predictions_failed", error=str(e))
            return None
    
    async def _enrich_response_with_graph_intelligence(
        self,
        response: ChatResponse,
        user_id: str,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None
    ) -> ChatResponse:
        """Enrich response with FinleyGraph intelligence (temporal, seasonal, fraud, root cause, predictions)."""
        if not source_id or not target_id:
            return response
        
        try:
            # Fetch all intelligence insights in parallel
            temporal = await self._get_temporal_insights(user_id, source_id, target_id)
            seasonal = await self._get_seasonal_insights(user_id, source_id, target_id)
            fraud = await self._get_fraud_warnings(user_id, source_id, target_id)
            root_cause = await self._get_root_cause_analysis(user_id, source_id, target_id)
            predictions = await self._get_predictions(user_id, source_id, target_id)
            
            # Build enriched answer with insights
            insights = []
            
            if temporal:
                freq = temporal['patterns'][0]['frequency'] if temporal['patterns'] else 'unknown'
                insights.append(f"📊 **Temporal Pattern**: This occurs {freq} with {temporal['confidence']:.0%} confidence")
            
            if seasonal:
                months = ', '.join(seasonal['peak_months'][:3])
                insights.append(f"📅 **Seasonal Peak**: Strongest in {months}")
            
            if fraud and fraud['risk_score'] > 0.4:
                insights.append(f"⚠️ **Fraud Risk**: {fraud['severity']} risk detected ({fraud['risk_score']:.0%})")
            
            if root_cause:
                insights.append(f"🔍 **Root Cause**: {len(root_cause['root_causes'])} root causes identified in {root_cause['chain_length']}-step chain")
            
            if predictions:
                insights.append(f"🔮 **Prediction**: {predictions['count']} future connections predicted ({predictions['avg_confidence']:.0%} confidence)")
            
            # Append insights to answer
            if insights:
                response.answer += "\n\n" + "\n".join(insights)
                response.data = response.data or {}
                response.data['intelligence_insights'] = {
                    'temporal': temporal,
                    'seasonal': seasonal,
                    'fraud': fraud,
                    'root_cause': root_cause,
                    'predictions': predictions
                }
                logger.info("Response enriched with graph intelligence", insight_count=len(insights))
            
            return response
        except Exception as e:
            logger.warning("graph_intelligence_enrichment_failed", error=str(e))
            return response
    
    async def _store_chat_message(
        self,
        user_id: str,
        chat_id: Optional[str],
        question: str,
        response: ChatResponse,
        chat_title: Optional[str] = None
    ):
        """
        Store chat message and response using LangChain message store.
        
        REFACTORED: Replaced manual Supabase inserts with LangChain PostgresChatMessageHistory.
        - Standardized message schema
        - Automatic session handling
        - Type-safe message objects
        - Graceful fallback to Supabase if LangChain unavailable
        """
        try:
            # Generate chat_id once and reuse for both messages
            actual_chat_id = chat_id or f"chat_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            actual_chat_title = chat_title or "New Chat"
            
            # Import message store
            try:
                from aident_cfo_brain.chat_message_store import create_message_store
            except ImportError:
                from chat_message_store import create_message_store
            
            # Create message store (requires LangChain)
            import os
            postgres_url = os.getenv('DATABASE_URL') or os.getenv('SUPABASE_DB_URL')
            
            if not postgres_url:
                raise ValueError("DATABASE_URL or SUPABASE_DB_URL environment variable required")
            
            message_store = create_message_store(
                user_id=user_id,
                chat_id=actual_chat_id,
                connection_string=postgres_url
            )
            
            # Store messages using standardized interface
            await message_store.add_user_message(
                message=question,
                metadata={'chat_title': actual_chat_title}
            )
            
            await message_store.add_ai_message(
                message=response.answer,
                metadata={
                    'chat_title': actual_chat_title,
                    'question_type': response.question_type.value if response.question_type else 'unknown',
                    'confidence': response.confidence
                }
            )
        
        except Exception as e:
            logger.error("Failed to store chat message", error=str(e))


# ============================================================================
# PRELOAD PATTERN: Initialize heavy dependencies at module-load time
# ============================================================================
# This runs automatically when the module is imported, eliminating the
# first-request latency that was caused by lazy-loading.
# 
# Note: spaCy is already preloaded at line 78 via _load_spacy_model()
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
    
    # Note: spaCy already preloaded at line 78 via _load_spacy_model()
    
    # Preload LangGraph (used for state machine)
    try:
        from langgraph.graph import StateGraph, END
        from langgraph.types import RetryPolicy, Send
        logger.info("✅ PRELOAD: LangGraph loaded at module-load time")
    except Exception as e:
        logger.warning(f"⚠️ PRELOAD: LangGraph load failed: {e}")
    
    # Preload Groq (LLM client)
    try:
        from groq import AsyncGroq
        logger.info("✅ PRELOAD: AsyncGroq loaded at module-load time")
    except Exception as e:
        logger.warning(f"⚠️ PRELOAD: AsyncGroq load failed: {e}")
    
    # Preload aiocache (Redis caching)
    try:
        from aiocache import cached, Cache
        from aiocache.serializers import JsonSerializer
        logger.info("✅ PRELOAD: aiocache loaded at module-load time")
    except Exception as e:
        logger.warning(f"⚠️ PRELOAD: aiocache load failed: {e}")
    
    # Preload Jinja2 (response templating)
    try:
        from jinja2 import Template, Environment, FileSystemLoader
        logger.info("✅ PRELOAD: Jinja2 loaded at module-load time")
    except Exception as e:
        logger.warning(f"⚠️ PRELOAD: Jinja2 load failed: {e}")
    
    # Preload tenacity (retry logic)
    try:
        from tenacity import retry, stop_after_attempt, wait_exponential
        logger.info("✅ PRELOAD: tenacity loaded at module-load time")
    except Exception as e:
        logger.warning(f"⚠️ PRELOAD: tenacity load failed: {e}")
    
    # Force Pydantic model compilation
    try:
        # Validating a model forces schema compilation
        QuestionClassification.model_validate({
            'metrics': ['revenue'],
            'time_periods': ['Q1'],
            'confidence': 0.9
        })
        logger.info("✅ PRELOAD: Pydantic models compiled at module-load time")
    except Exception as e:
        logger.warning(f"⚠️ PRELOAD: Pydantic model compilation failed: {e}")
    
    _PRELOAD_COMPLETED = True

try:
    _preload_all_modules()
except Exception as e:
    logger.warning(f"Module-level orchestrator preload failed (will use fallback): {e}")

