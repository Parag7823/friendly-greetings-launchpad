"""
Intelligent Chat Orchestrator - The Brain of Finley AI
======================================================

This module connects the chat interface to all intelligence engines,
providing natural language understanding and intelligent routing.

Features:
- Natural language question classification
- Intelligent routing to appropriate engines
- Human-readable response formatting
- Context-aware conversation management
- Proactive insights and suggestions

Author: Finley AI Team
Version: 1.0.0
Date: 2025-01-22
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

# PHASE 1: LangGraph imports for state machine orchestration
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# FIX #INSTRUCTOR: Type-safe question classification with instructor
try:
    import instructor
    from pydantic import BaseModel, Field
    INSTRUCTOR_AVAILABLE = True
except ImportError:
    INSTRUCTOR_AVAILABLE = False
    logger_temp = structlog.get_logger(__name__)
    logger_temp.warning("instructor not available - falling back to manual JSON parsing")

# FIX #19: Import intent classification and output guard components
try:
    # Try package layout first
    from aident_cfo_brain.intent_and_guard_engine import (
        IntentClassifier,
        OutputGuard,
        ResponseVariationEngine,
        UserIntent,
        get_intent_classifier,
        get_output_guard,
        get_response_variation_engine
    )
except ImportError:
    # Fallback to relative import
    from .intent_and_guard_engine import (
        IntentClassifier,
        OutputGuard,
        ResponseVariationEngine,
        UserIntent,
        get_intent_classifier,
        get_output_guard,
        get_response_variation_engine
    )

# Initialize logger early for use in import error handlers
logger = structlog.get_logger(__name__)

# FIX #16: Add parent directory to sys.path for imports to work in all deployment layouts
# This ensures modules in aident_cfo_brain/ can be imported regardless of how the module is loaded
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
_root_dir = os.path.dirname(_parent_dir)

# Add all potential paths to sys.path (in order of priority)
# Priority: current dir (aident_cfo_brain) > parent (project root) > root (/) > app root (/app)
for _path in [_current_dir, _parent_dir, _root_dir, '/app']:
    if _path and _path not in sys.path:
        sys.path.insert(0, _path)

# Helper function to load modules from specific file paths
def _load_module_from_path(module_name: str, file_path: str):
    """Load a module from a specific file path using importlib"""
    if not os.path.exists(file_path):
        raise ImportError(f"Module file not found: {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Nuclear option: Dynamic file search - finds modules anywhere on the system
def _find_module_file(module_name: str, search_roots: list = None) -> str:
    """
    Recursively search for a module file starting from multiple root directories.
    This is the nuclear option - guaranteed to find the file if it exists anywhere.
    """
    if search_roots is None:
        search_roots = ['/app', '/app/src', '/app/aident_cfo_brain', '/', os.getcwd()]
    
    target_file = f"{module_name}.py"
    
    for root in search_roots:
        if not root or not os.path.isdir(root):
            continue
        
        try:
            # Search up to 5 levels deep to avoid infinite recursion
            for level in range(5):
                for dirpath, dirnames, filenames in os.walk(root):
                    # Skip deep recursion
                    if dirpath.count(os.sep) - root.count(os.sep) > level:
                        continue
                    
                    if target_file in filenames:
                        full_path = os.path.join(dirpath, target_file)
                        logger.debug(f"✓ Found {module_name} at: {full_path}")
                        return full_path
        except (OSError, PermissionError):
            continue
    
    # Not found anywhere
    raise ImportError(f"Could not find {module_name}.py anywhere in {search_roots}")

# FIX #16: Use absolute imports with try/except fallbacks for different deployment layouts
# Supports both: package layout (aident_cfo_brain.module) and flat layout (module)

try:
    # Try package layout first (standard Python package)
    from aident_cfo_brain.finley_graph_engine import FinleyGraphEngine
    from aident_cfo_brain.aident_memory_manager import AidentMemoryManager
    from aident_cfo_brain.causal_inference_engine import CausalInferenceEngine
    from aident_cfo_brain.temporal_pattern_learner import TemporalPatternLearner
    from aident_cfo_brain.enhanced_relationship_detector import EnhancedRelationshipDetector
    logger.debug("✓ Tier 1: Package layout imports successful (aident_cfo_brain.module)")
except ImportError as e1:
    logger.debug(f"✗ Tier 1 failed: {e1}. Trying Tier 2 (flat layout)...")
    try:
        # Fallback to flat layout (Railway deployment or direct module import)
        # sys.path now includes current directory and root, so these should work
        from finley_graph_engine import FinleyGraphEngine
        from aident_memory_manager import AidentMemoryManager
        from causal_inference_engine import CausalInferenceEngine
        from temporal_pattern_learner import TemporalPatternLearner
        from enhanced_relationship_detector import EnhancedRelationshipDetector
        logger.debug("✓ Tier 2: Flat layout imports successful (module)")
    except ImportError as e2:
        logger.debug(f"✗ Tier 2 failed: {e2}. Trying Tier 3 (dynamic file search - NUCLEAR OPTION)...")
        # Final fallback: Dynamic file search - finds modules anywhere on the system
        # This is guaranteed to work if the files exist anywhere
        try:
            _module_names = ['finley_graph_engine', 'aident_memory_manager', 'causal_inference_engine', 
                            'temporal_pattern_learner', 'enhanced_relationship_detector']
            _modules = {}
            
            for _mod_name in _module_names:
                try:
                    # Use dynamic search to find the module file anywhere on the system
                    _file_path = _find_module_file(_mod_name)
                    _modules[_mod_name] = _load_module_from_path(_mod_name, _file_path)
                except ImportError as e:
                    logger.error(f"Could not find {_mod_name}: {e}")
                    raise
            
            FinleyGraphEngine = _modules['finley_graph_engine'].FinleyGraphEngine
            AidentMemoryManager = _modules['aident_memory_manager'].AidentMemoryManager
            CausalInferenceEngine = _modules['causal_inference_engine'].CausalInferenceEngine
            TemporalPatternLearner = _modules['temporal_pattern_learner'].TemporalPatternLearner
            EnhancedRelationshipDetector = _modules['enhanced_relationship_detector'].EnhancedRelationshipDetector
            logger.debug("✓ Tier 3: NUCLEAR OPTION - Dynamic file search successful!")
        except ImportError as e3:
            logger.error(f"IMPORT FAILURE - All 3 tiers failed. Fix location: aident_cfo_brain/intelligent_chat_orchestrator.py")
            logger.error(f"Tier 1 (package layout): {e1}")
            logger.error(f"Tier 2 (flat layout): {e2}")
            logger.error(f"Tier 3 (dynamic search): {e3}")
            logger.error(f"sys.path includes: {sys.path[:3]}")
            raise

try:
    from data_ingestion_normalization.entity_resolver_optimized import EntityResolverOptimized as EntityResolver
except ImportError:
    from entity_resolver_optimized import EntityResolverOptimized as EntityResolver

try:
    from data_ingestion_normalization.embedding_service import EmbeddingService
except ImportError:
    from embedding_service import EmbeddingService


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


# FIX #INSTRUCTOR: Pydantic model for type-safe question classification
if INSTRUCTOR_AVAILABLE:
    class QuestionClassification(BaseModel):
        """Type-safe question classification response from LLM"""
        type: str = Field(
            ..., 
            description="Question type: causal, temporal, relationship, what_if, explain, data_query, general, or unknown"
        )
        confidence: float = Field(
            ..., 
            ge=0.0, 
            le=1.0, 
            description="Confidence score between 0.0 and 1.0"
        )
        reasoning: str = Field(
            ..., 
            description="Brief explanation of why this classification was chosen"
        )


class DataMode(Enum):
    """FIX #4: Data availability modes for response differentiation"""
    NO_DATA = "no_data"  # User has no data connected yet
    LIMITED_DATA = "limited_data"  # User has <50 transactions
    RICH_DATA = "rich_data"  # User has >50 transactions


class OnboardingState(Enum):
    """FIX #5: Track onboarding state to prevent repetition"""
    FIRST_VISIT = "first_visit"  # User's first interaction
    ONBOARDED = "onboarded"  # User has seen onboarding
    DATA_CONNECTED = "data_connected"  # User has connected data
    ACTIVE = "active"  # User is actively using system


# PHASE 1: LangGraph State - Replaces manual parameter passing
class OrchestratorState(TypedDict, total=False):
    """
    Unified state for LangGraph orchestrator.
    
    REPLACES: Manual memory initialization (lines 608-614)
    - LangGraph automatically persists this state across nodes
    - Eliminates manual parameter passing between functions
    - Automatic context injection between nodes
    """
    # Input
    question: str
    user_id: str
    chat_id: Optional[str]
    context: Optional[Dict[str, Any]]
    
    # Classification results
    intent: str
    intent_confidence: float
    question_type: str
    question_confidence: float
    
    # Memory & Context
    conversation_history: List[Dict[str, str]]
    memory_context: str
    memory_messages: List[Dict[str, str]]
    
    # Response
    response: Any
    
    # Metadata
    processing_steps: List[str]
    errors: List[str]


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


class IntelligentChatOrchestrator:
    """
    The brain that understands questions and routes to intelligence engines.
    
    This is the missing link between the chat interface and the backend intelligence.
    """
    
    def __init__(self, supabase_client, cache_client=None, groq_client=None, embedding_service=None):
        """
        Initialize the orchestrator with all intelligence engines.
        
        Args:
            supabase_client: Supabase client for database access
            cache_client: Optional cache client for performance
            groq_client: Optional Groq client for LLM (for testing/mocking)
            embedding_service: Optional embedding service for dependency injection (FIX #6)
        """
        # FIX #3: Accept groq_client for dependency injection (testing/mocking)
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
        
        # FIX #6: Initialize or use injected embedding service
        if embedding_service is None:
            try:
                self.embedding_service = EmbeddingService(cache_client=cache_client)
                logger.info("✅ EmbeddingService initialized for chat orchestrator")
            except Exception as e:
                logger.warning("Failed to initialize EmbeddingService", error=str(e))
                self.embedding_service = None
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
            embedding_service=self.embedding_service  # FIX #6: Pass injected embedding service
        )
        
        self.entity_resolver = EntityResolver(
            supabase_client=supabase_client,
            cache_client=cache_client
        )
        
        # NEW: Initialize FinleyGraph engine for intelligence queries
        self.graph_engine = FinleyGraphEngine(
            supabase=supabase_client,
            redis_url=os.getenv('ARQ_REDIS_URL') or os.getenv('REDIS_URL')
        )
        
        # FIX #19: Initialize intent classification and output guard components
        self.intent_classifier = get_intent_classifier()
        self.output_guard = get_output_guard()
        self.response_variation_engine = get_response_variation_engine(self.groq)
        
        logger.info("✅ IntelligentChatOrchestrator initialized with all engines including FinleyGraph")
        logger.info("✅ FIX #19: Intent Classifier, OutputGuard, and ResponseVariation initialized")
        
        # PHASE 1: Build LangGraph state machine (replaces manual routing)
        self.graph = self._build_langgraph()
        logger.info("✅ PHASE 1: LangGraph state machine compiled")
    
    def _build_langgraph(self):
        """
        PHASE 1: Build LangGraph state machine.
        
        REPLACES:
        - 60+ lines of manual if/elif routing (lines 656-710)
        - Manual memory initialization (lines 608-614)
        - Manual asyncio.gather for parallel queries (lines 339-370)
        
        Returns:
            Compiled LangGraph workflow
        """
        workflow = StateGraph(OrchestratorState)
        
        # Add nodes for routing and handlers
        workflow.add_node("classify_intent", self._node_classify_intent)
        workflow.add_node("route_by_intent", self._node_route_by_intent)
        
        # Intent handlers (REPLACES: 7 if/elif chains)
        workflow.add_node("handle_greeting", self._node_handle_greeting)
        workflow.add_node("handle_smalltalk", self._node_handle_smalltalk)
        workflow.add_node("handle_capability_summary", self._node_handle_capability_summary)
        workflow.add_node("handle_system_flow", self._node_handle_system_flow)
        workflow.add_node("handle_differentiator", self._node_handle_differentiator)
        workflow.add_node("handle_meta_feedback", self._node_handle_meta_feedback)
        workflow.add_node("handle_help", self._node_handle_help)
        
        # Question type classification & routing
        workflow.add_node("classify_question", self._node_classify_question)
        workflow.add_node("route_by_question_type", self._node_route_by_question_type)
        
        # Question type handlers (REPLACES: 7 elif chains)
        workflow.add_node("handle_causal", self._node_handle_causal)
        workflow.add_node("handle_temporal", self._node_handle_temporal)
        workflow.add_node("handle_relationship", self._node_handle_relationship)
        workflow.add_node("handle_whatif", self._node_handle_whatif)
        workflow.add_node("handle_explain", self._node_handle_explain)
        workflow.add_node("handle_data_query", self._node_handle_data_query)
        workflow.add_node("handle_general", self._node_handle_general)
        
        # Post-processing
        workflow.add_node("apply_output_guard", self._node_apply_output_guard)
        workflow.add_node("save_memory", self._node_save_memory)
        
        # Set entry point
        workflow.set_entry_point("classify_intent")
        
        # Define edges
        workflow.add_edge("classify_intent", "route_by_intent")
        
        # Intent routing (REPLACES: if/elif chains)
        workflow.add_conditional_edges(
            "route_by_intent",
            self._route_by_intent_decision,
            {
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
        
        # Question type routing
        workflow.add_edge("classify_question", "route_by_question_type")
        
        # Question type routing (REPLACES: elif chains)
        workflow.add_conditional_edges(
            "route_by_question_type",
            self._route_by_question_type_decision,
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
        
        # Question handlers → Output guard
        for handler in ["handle_causal", "handle_temporal", "handle_relationship",
                       "handle_whatif", "handle_explain", "handle_data_query", "handle_general"]:
            workflow.add_edge(handler, "apply_output_guard")
        
        # Post-processing
        workflow.add_edge("apply_output_guard", "save_memory")
        workflow.add_edge("save_memory", END)
        
        return workflow.compile()
    
    # ========================================================================
    # ROUTING DECISION FUNCTIONS - Replaces if/elif logic
    # ========================================================================
    
    def _route_by_intent_decision(self, state: OrchestratorState) -> str:
        """REPLACES: 7 if/elif chains for intent routing (lines 656-676)"""
        intent = state.get("intent", "unknown")
        
        routing_map = {
            "greeting": "greeting",
            "smalltalk": "smalltalk",
            "capability_summary": "capability_summary",
            "system_flow": "system_flow",
            "differentiator": "differentiator",
            "meta_feedback": "meta_feedback",
            "help": "help",
        }
        
        return routing_map.get(intent, "data_analysis")
    
    def _route_by_question_type_decision(self, state: OrchestratorState) -> str:
        """REPLACES: 7 elif chains for question type routing (lines 690-710)"""
        question_type = state.get("question_type", "general")
        
        routing_map = {
            "causal": "causal",
            "temporal": "temporal",
            "relationship": "relationship",
            "what_if": "what_if",
            "explain": "explain",
            "data_query": "data_query",
        }
        
        return routing_map.get(question_type, "general")
    
    # ========================================================================
    # NODE IMPLEMENTATIONS - Replaces manual handler invocations
    # ========================================================================
    
    async def _node_classify_intent(self, state: OrchestratorState) -> OrchestratorState:
        """Classify user intent using intent classifier."""
        try:
            intent_result = await self.intent_classifier.classify(state["question"])
            state["intent"] = intent_result.intent.value
            state["intent_confidence"] = intent_result.confidence
            state["processing_steps"] = state.get("processing_steps", []) + ["classify_intent"]
            logger.info("Intent classified", intent=intent_result.intent.value, confidence=round(intent_result.confidence, 2))
        except Exception as e:
            logger.error("Intent classification failed", error=str(e))
            state["intent"] = "unknown"
            state["intent_confidence"] = 0.0
            state["errors"] = state.get("errors", []) + [f"Intent classification failed: {str(e)}"]
        
        return state
    
    async def _node_route_by_intent(self, state: OrchestratorState) -> OrchestratorState:
        """Route by intent (decision node)."""
        state["processing_steps"] = state.get("processing_steps", []) + ["route_by_intent"]
        return state
    
    async def _node_classify_question(self, state: OrchestratorState) -> OrchestratorState:
        """Classify question type for DATA_ANALYSIS intent."""
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
        except Exception as e:
            logger.error("Question classification failed", error=str(e))
            state["question_type"] = "general"
            state["question_confidence"] = 0.0
            state["errors"] = state.get("errors", []) + [f"Question classification failed: {str(e)}"]
        
        return state
    
    async def _node_route_by_question_type(self, state: OrchestratorState) -> OrchestratorState:
        """Route by question type (decision node)."""
        state["processing_steps"] = state.get("processing_steps", []) + ["route_by_question_type"]
        return state
    
    # Intent handlers
    async def _node_handle_greeting(self, state: OrchestratorState) -> OrchestratorState:
        """Handle greeting intent."""
        try:
            response = await self._handle_greeting(state["question"], state["user_id"], state.get("conversation_history", []), None)
            state["response"] = response
            state["processing_steps"] = state.get("processing_steps", []) + ["handle_greeting"]
        except Exception as e:
            logger.error("Greeting handler failed", error=str(e))
            state["errors"] = state.get("errors", []) + [f"Greeting handler failed: {str(e)}"]
        return state
    
    async def _node_handle_smalltalk(self, state: OrchestratorState) -> OrchestratorState:
        """Handle smalltalk intent."""
        try:
            response = await self._handle_smalltalk(state["question"], state["user_id"], state.get("conversation_history", []), None)
            state["response"] = response
            state["processing_steps"] = state.get("processing_steps", []) + ["handle_smalltalk"]
        except Exception as e:
            logger.error("Smalltalk handler failed", error=str(e))
            state["errors"] = state.get("errors", []) + [f"Smalltalk handler failed: {str(e)}"]
        return state
    
    async def _node_handle_capability_summary(self, state: OrchestratorState) -> OrchestratorState:
        """Handle capability summary intent."""
        try:
            response = await self._handle_capability_summary(state["question"], state["user_id"], state.get("conversation_history", []), None)
            state["response"] = response
            state["processing_steps"] = state.get("processing_steps", []) + ["handle_capability_summary"]
        except Exception as e:
            logger.error("Capability summary handler failed", error=str(e))
            state["errors"] = state.get("errors", []) + [f"Capability summary handler failed: {str(e)}"]
        return state
    
    async def _node_handle_system_flow(self, state: OrchestratorState) -> OrchestratorState:
        """Handle system flow intent."""
        try:
            response = await self._handle_system_flow(state["question"], state["user_id"], state.get("conversation_history", []), None)
            state["response"] = response
            state["processing_steps"] = state.get("processing_steps", []) + ["handle_system_flow"]
        except Exception as e:
            logger.error("System flow handler failed", error=str(e))
            state["errors"] = state.get("errors", []) + [f"System flow handler failed: {str(e)}"]
        return state
    
    async def _node_handle_differentiator(self, state: OrchestratorState) -> OrchestratorState:
        """Handle differentiator intent."""
        try:
            response = await self._handle_differentiator(state["question"], state["user_id"], state.get("conversation_history", []), None)
            state["response"] = response
            state["processing_steps"] = state.get("processing_steps", []) + ["handle_differentiator"]
        except Exception as e:
            logger.error("Differentiator handler failed", error=str(e))
            state["errors"] = state.get("errors", []) + [f"Differentiator handler failed: {str(e)}"]
        return state
    
    async def _node_handle_meta_feedback(self, state: OrchestratorState) -> OrchestratorState:
        """Handle meta feedback intent."""
        try:
            response = await self._handle_meta_feedback(state["question"], state["user_id"], state.get("conversation_history", []), None)
            state["response"] = response
            state["processing_steps"] = state.get("processing_steps", []) + ["handle_meta_feedback"]
        except Exception as e:
            logger.error("Meta feedback handler failed", error=str(e))
            state["errors"] = state.get("errors", []) + [f"Meta feedback handler failed: {str(e)}"]
        return state
    
    async def _node_handle_help(self, state: OrchestratorState) -> OrchestratorState:
        """Handle help intent."""
        try:
            response = await self._handle_help(state["question"], state["user_id"], state.get("conversation_history", []), None)
            state["response"] = response
            state["processing_steps"] = state.get("processing_steps", []) + ["handle_help"]
        except Exception as e:
            logger.error("Help handler failed", error=str(e))
            state["errors"] = state.get("errors", []) + [f"Help handler failed: {str(e)}"]
        return state
    
    # Question type handlers
    async def _node_handle_causal(self, state: OrchestratorState) -> OrchestratorState:
        """Handle causal question."""
        try:
            response = await self._handle_causal_question(state["question"], state["user_id"], state.get("context"), state.get("conversation_history", []))
            state["response"] = response
            state["processing_steps"] = state.get("processing_steps", []) + ["handle_causal"]
        except Exception as e:
            logger.error("Causal handler failed", error=str(e))
            state["errors"] = state.get("errors", []) + [f"Causal handler failed: {str(e)}"]
        return state
    
    async def _node_handle_temporal(self, state: OrchestratorState) -> OrchestratorState:
        """Handle temporal question with PARALLEL QUERY EXECUTION (PHASE 3)."""
        try:
            response = await self._handle_temporal_question(state["question"], state["user_id"], state.get("context"), state.get("conversation_history", []))
            state["response"] = response
            state["processing_steps"] = state.get("processing_steps", []) + ["handle_temporal"]
        except Exception as e:
            logger.error("Temporal handler failed", error=str(e))
            state["errors"] = state.get("errors", []) + [f"Temporal handler failed: {str(e)}"]
        return state
    
    async def _node_handle_relationship(self, state: OrchestratorState) -> OrchestratorState:
        """Handle relationship question."""
        try:
            response = await self._handle_relationship_question(state["question"], state["user_id"], state.get("context"), state.get("conversation_history", []))
            state["response"] = response
            state["processing_steps"] = state.get("processing_steps", []) + ["handle_relationship"]
        except Exception as e:
            logger.error("Relationship handler failed", error=str(e))
            state["errors"] = state.get("errors", []) + [f"Relationship handler failed: {str(e)}"]
        return state
    
    async def _node_handle_whatif(self, state: OrchestratorState) -> OrchestratorState:
        """Handle what-if question."""
        try:
            response = await self._handle_whatif_question(state["question"], state["user_id"], state.get("context"), state.get("conversation_history", []))
            state["response"] = response
            state["processing_steps"] = state.get("processing_steps", []) + ["handle_whatif"]
        except Exception as e:
            logger.error("What-if handler failed", error=str(e))
            state["errors"] = state.get("errors", []) + [f"What-if handler failed: {str(e)}"]
        return state
    
    async def _node_handle_explain(self, state: OrchestratorState) -> OrchestratorState:
        """Handle explain question."""
        try:
            response = await self._handle_explain_question(state["question"], state["user_id"], state.get("context"), state.get("conversation_history", []))
            state["response"] = response
            state["processing_steps"] = state.get("processing_steps", []) + ["handle_explain"]
        except Exception as e:
            logger.error("Explain handler failed", error=str(e))
            state["errors"] = state.get("errors", []) + [f"Explain handler failed: {str(e)}"]
        return state
    
    async def _node_handle_data_query(self, state: OrchestratorState) -> OrchestratorState:
        """Handle data query question."""
        try:
            response = await self._handle_data_query(state["question"], state["user_id"], state.get("context"), state.get("conversation_history", []))
            state["response"] = response
            state["processing_steps"] = state.get("processing_steps", []) + ["handle_data_query"]
        except Exception as e:
            logger.error("Data query handler failed", error=str(e))
            state["errors"] = state.get("errors", []) + [f"Data query handler failed: {str(e)}"]
        return state
    
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
    
    async def _node_apply_output_guard(self, state: OrchestratorState) -> OrchestratorState:
        """Apply output guard to check for repetition."""
        try:
            if state.get("response"):
                safe_response = await self.output_guard.check_and_fix(
                    proposed_response=state["response"].answer,
                    recent_responses=state.get("memory_messages", [])[-5:],
                    question=state["question"],
                    llm_client=self.groq,
                    frustration_level=0
                )
                state["response"].answer = safe_response
            state["processing_steps"] = state.get("processing_steps", []) + ["apply_output_guard"]
        except Exception as e:
            logger.error("Output guard failed", error=str(e))
            state["errors"] = state.get("errors", []) + [f"Output guard failed: {str(e)}"]
        return state
    
    async def _node_save_memory(self, state: OrchestratorState) -> OrchestratorState:
        """Save memory after response."""
        try:
            if state.get("response"):
                from core_infrastructure.fastapi_backend_v2 import get_memory_manager
                memory_manager = get_memory_manager(state["user_id"])
                await memory_manager.add_message(state["question"], state["response"].answer)
            state["processing_steps"] = state.get("processing_steps", []) + ["save_memory"]
        except Exception as e:
            logger.error("Memory save failed", error=str(e))
            state["errors"] = state.get("errors", []) + [f"Memory save failed: {str(e)}"]
        return state
    
    async def _parallel_query(self, queries: List[Tuple[str, callable]]) -> Dict[str, Any]:
        """
        PARALLEL PROCESSING: Execute multiple queries simultaneously using asyncio.
        
        Example: "Compare Q1 vs Q2" → Query Q1 and Q2 data in parallel
        
        Args:
            queries: List of (query_name, async_function) tuples
        
        Returns:
            Dict mapping query_name to result
        """
        try:
            # Execute all queries in parallel
            tasks = [func() for name, func in queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Map results back to query names
            result_dict = {}
            for (name, _), result in zip(queries, results):
                if isinstance(result, Exception):
                    logger.error("Parallel query failed", query_name=name, error=str(result))
                    result_dict[name] = None
                else:
                    result_dict[name] = result
            
            logger.info("Parallel processing completed", query_count=len(queries))
            return result_dict
            
        except Exception as e:
            logger.error("Parallel processing failed", error=str(e))
            return {}
    
    async def _determine_data_mode(self, user_id: str) -> DataMode:
        """
        FIX #4: Determine user's data availability mode.
        
        Returns:
            DataMode enum: NO_DATA, LIMITED_DATA, or RICH_DATA
        """
        try:
            # Query transaction count
            events_result = self.supabase.table('raw_events')\
                .select('id', count='exact')\
                .eq('user_id', user_id)\
                .execute()
            
            transaction_count = len(events_result.data) if events_result.data else 0
            
            if transaction_count == 0:
                return DataMode.NO_DATA
            elif transaction_count < 50:
                return DataMode.LIMITED_DATA
            else:
                return DataMode.RICH_DATA
        except Exception as e:
            logger.warning(f"Failed to determine data mode: {e}")
            return DataMode.NO_DATA

    async def _get_onboarding_state(self, user_id: str) -> OnboardingState:
        """
        FIX #5: Get user's onboarding state from Redis/Supabase.
        
        Returns:
            OnboardingState enum: FIRST_VISIT, ONBOARDED, DATA_CONNECTED, or ACTIVE
        """
        try:
            # Check user_preferences table for onboarding state
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
        """
        FIX #5: Save user's onboarding state to Supabase.
        
        Returns:
            True if successful, False otherwise
        """
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
        """
        ENHANCEMENT #10: Generate intelligent questions/insights even without data.
        
        Creates a personalized onboarding experience that showcases Finley's intelligence
        without requiring data, making users excited to connect sources.
        
        Returns:
            Formatted string with intelligent questions and insights
        """
        try:
            # Generate smart business questions using Groq
            prompt = """You are Finley, an AI finance expert. Generate 3 SMART, SPECIFIC business questions 
that would help a new user understand what you can do with their financial data.

Requirements:
- Questions should be insightful and show financial intelligence
- Each question should highlight a different Finley capability
- Questions should be relevant to ANY business (not industry-specific)
- Format: "❓ [Question]"

Example capabilities to showcase:
1. Causal analysis: "What's causing my cash flow to fluctuate?"
2. Temporal patterns: "When do my customers typically pay?"
3. Relationship mapping: "Which vendors represent my biggest costs?"
4. Anomaly detection: "Are there any unusual transactions I should know about?"
5. Predictive forecasting: "When will I run out of cash at current burn rate?"

Generate exactly 3 questions now:"""
            
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
            
            # Format the response
            formatted_response = f"""🧠 **Here are some questions I can answer once you connect your data:**

{intelligent_questions}

**Ready to unlock these insights?** Click "Data Sources" to connect QuickBooks, Xero, or upload your financial files."""
            
            return formatted_response
            
        except Exception as e:
            logger.warning(f"Failed to generate intelligent questions: {e}")
            # Fallback to default questions
            return """🧠 **Here are some questions I can answer once you connect your data:**

❓ What's causing my cash flow to fluctuate?
❓ When do my customers typically pay?
❓ Which vendors represent my biggest costs?

**Ready to unlock these insights?** Click "Data Sources" to connect QuickBooks, Xero, or upload your financial files."""

    async def _generate_dynamic_onboarding_message(self, user_id: str, onboarding_state: OnboardingState) -> str:
        """
        ENHANCEMENT #9: Generate dynamic onboarding messages based on user state.
        
        Removes hardcoded "60 seconds" urgency and personalizes based on:
        - User's onboarding state (FIRST_VISIT, ONBOARDED, DATA_CONNECTED, ACTIVE)
        - Data availability
        - Previous interactions
        
        Returns:
            Personalized onboarding message
        """
        try:
            if onboarding_state == OnboardingState.FIRST_VISIT:
                # First time - welcoming, not pushy
                return """👋 Hi! I'm Finley, your AI finance teammate.

I help small business owners and founders understand their finances better. Think of me as your personal CFO who's always available.

**What I can do:**
✓ Analyze cash flow patterns
✓ Predict payment delays
✓ Find cost-saving opportunities
✓ Answer any finance question

**Let's get started!** Connect your data sources (QuickBooks, Xero, Stripe, etc.) or upload financial files to begin."""
            
            elif onboarding_state == OnboardingState.ONBOARDED:
                # Already seen onboarding - encourage action
                return """💡 **Ready to unlock financial insights?**

You've seen what I can do. Now let's connect your actual data so I can provide specific, actionable recommendations for YOUR business.

**Quick options:**
1. Connect QuickBooks (1 minute)
2. Connect Xero (1 minute)
3. Upload Excel/CSV files
4. Connect Stripe for payment data

Which would you like to try?"""
            
            elif onboarding_state == OnboardingState.DATA_CONNECTED:
                # Has data - focus on capabilities
                return """🚀 **Your data is connected! Here's what I can do now:**

With your financial data, I can:
✓ Analyze WHY your metrics changed (causal analysis)
✓ Predict WHEN events will happen (temporal forecasting)
✓ Show WHO your key vendors/customers are (relationship mapping)
✓ Spot WHAT'S unusual (anomaly detection)
✓ Answer ANY financial question with your actual numbers

**What would you like to explore first?**"""
            
            else:  # ACTIVE
                # Power user - focus on advanced features
                return """⚡ **You're all set!** I have your financial data and I'm ready to provide deep insights.

**Advanced features available:**
🔍 Causal inference analysis
📈 Seasonal pattern detection
🎯 Predictive forecasting
💡 Cost optimization opportunities
⚠️ Risk scoring and alerts

**What would you like to analyze today?**"""
        
        except Exception as e:
            logger.warning(f"Failed to generate dynamic onboarding: {e}")
            return "Let's connect your financial data to get started!"

    async def process_question(
        self,
        question: str,
        user_id: str,
        chat_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ChatResponse:
        """
        Main entry point: Process a user question and return intelligent response.
        
        Args:
            question: User's natural language question
            user_id: User ID for data scoping
            chat_id: Optional chat ID for conversation context
            context: Optional additional context
        
        Returns:
            ChatResponse with answer and structured data
        """
        try:
            logger.info("Processing question", question=question, user_id=user_id, chat_id=chat_id)
            print(f"[ORCHESTRATOR] Starting process_question for user {user_id}", flush=True)
            
            # Step 0a: FIX #4: Determine data mode for response differentiation
            print(f"[ORCHESTRATOR] Determining data mode...", flush=True)
            data_mode = await self._determine_data_mode(user_id)
            print(f"[ORCHESTRATOR] Data mode: {data_mode.value}", flush=True)
            
            # Step 0b: Initialize per-user memory manager (isolated, no cross-user contamination)
            print(f"[ORCHESTRATOR] Initializing memory manager...", flush=True)
            memory_manager = AidentMemoryManager(
                user_id=user_id,
                redis_url=os.getenv('ARQ_REDIS_URL') or os.getenv('REDIS_URL')
            )
            print(f"[ORCHESTRATOR] Loading memory...", flush=True)
            await memory_manager.load_memory()
            print(f"[ORCHESTRATOR] Memory loaded successfully", flush=True)
            
            # Step 0c: FIX #18 - MEMORY-FIRST ARCHITECTURE
            # Get memory as PRIMARY context source (includes auto-summarized old messages + recent messages)
            memory_context = memory_manager.get_context()
            memory_messages = memory_manager.get_messages()
            
            # Use memory messages as primary conversation history
            # Memory has intelligent summarization built-in by LangChain
            if memory_messages:
                conversation_history = memory_messages
                logger.info("Using memory as primary context source", message_count=len(memory_messages))
            else:
                # Fallback: Load from database only if memory is empty (new conversation)
                try:
                    conversation_history = await asyncio.wait_for(
                        self._load_conversation_history(user_id, chat_id) if chat_id else asyncio.sleep(0),
                        timeout=5.0
                    ) if chat_id else []
                except asyncio.TimeoutError:
                    logger.warning("⏱️ Conversation history loading timed out - proceeding without history")
                    conversation_history = []
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load conversation history: {e} - proceeding without history")
                    conversation_history = []
            
            # Step 0d: FIX #19 - CLASSIFY USER INTENT (separate from financial question type)
            # This prevents checking Supabase for meta questions like "What can you do?"
            print(f"[ORCHESTRATOR] Classifying user intent...", flush=True)
            intent_result = await self.intent_classifier.classify(question)
            print(f"[ORCHESTRATOR] User intent: {intent_result.intent.value} (confidence: {intent_result.confidence:.2f})", flush=True)
            logger.info(
                "User intent classified",
                intent=intent_result.intent.value,
                confidence=round(intent_result.confidence, 2),
                method=intent_result.method,
                reasoning=intent_result.reasoning
            )
            
            # CRITICAL FIX: Route by INTENT FIRST (before question_type)
            # This ensures greetings, smalltalk, and meta questions are handled appropriately
            print(f"[ORCHESTRATOR] Routing by user intent...", flush=True)
            if intent_result.intent == UserIntent.GREETING:
                response = await self._handle_greeting(question, user_id, conversation_history, memory_manager)
            
            elif intent_result.intent == UserIntent.SMALLTALK:
                response = await self._handle_smalltalk(question, user_id, conversation_history, memory_manager)
            
            elif intent_result.intent == UserIntent.CAPABILITY_SUMMARY:
                response = await self._handle_capability_summary(question, user_id, conversation_history, memory_manager)
            
            elif intent_result.intent == UserIntent.SYSTEM_FLOW:
                response = await self._handle_system_flow(question, user_id, conversation_history, memory_manager)
            
            elif intent_result.intent == UserIntent.DIFFERENTIATOR:
                response = await self._handle_differentiator(question, user_id, conversation_history, memory_manager)
            
            elif intent_result.intent == UserIntent.META_FEEDBACK:
                response = await self._handle_meta_feedback(question, user_id, conversation_history, memory_manager)
            
            elif intent_result.intent == UserIntent.HELP:
                response = await self._handle_help(question, user_id, conversation_history, memory_manager)
            
            else:
                # Intent is DATA_ANALYSIS or CONNECT_SOURCE or UNKNOWN - classify by question type
                print(f"[ORCHESTRATOR] Classifying question type (intent-based routing didn't match)...", flush=True)
                question_type, confidence = await self._classify_question(
                    question, 
                    user_id, 
                    conversation_history,
                    memory_context=memory_context
                )
                
                logger.info("Question classified", question_type=question_type.value, confidence=round(confidence, 2))
                
                # Step 2: Route to appropriate handler (pass conversation history + memory context)
                if question_type == QuestionType.CAUSAL:
                    response = await self._handle_causal_question(question, user_id, context, conversation_history)
                
                elif question_type == QuestionType.TEMPORAL:
                    response = await self._handle_temporal_question(question, user_id, context, conversation_history)
                
                elif question_type == QuestionType.RELATIONSHIP:
                    response = await self._handle_relationship_question(question, user_id, context, conversation_history)
                
                elif question_type == QuestionType.WHAT_IF:
                    response = await self._handle_whatif_question(question, user_id, context, conversation_history)
                
                elif question_type == QuestionType.EXPLAIN:
                    response = await self._handle_explain_question(question, user_id, context, conversation_history)
                
                elif question_type == QuestionType.DATA_QUERY:
                    response = await self._handle_data_query(question, user_id, context, conversation_history)
                
                else:
                    # FIX #1: Pass memory_manager to general handler for repetition detection
                    response = await self._handle_general_question(question, user_id, context, conversation_history, memory_manager)
            
            # Step 3: FIX #19 - OUTPUT GUARD (Check for repetition and fix if needed)
            print(f"[ORCHESTRATOR] Running output guard...", flush=True)
            safe_response = await self.output_guard.check_and_fix(
                proposed_response=response.answer,
                recent_responses=memory_messages[-5:] if memory_messages else [],
                question=question,
                llm_client=self.groq,
                frustration_level=memory_manager.conversation_state.get('frustration_level', 0)
            )
            response.answer = safe_response
            logger.info("Output guard check completed", repetition_detected=(safe_response != response.answer))
            
            # Step 4: Save memory after response (for context in next turn)
            await memory_manager.add_message(question, response.answer)
            
            # Step 5: Store in database
            await self._store_chat_message(user_id, chat_id, question, response)
            
            # Step 6: Log memory stats for monitoring
            memory_stats = await memory_manager.get_memory_stats()
            logger.info("Question processed successfully", question_type=question_type.value, memory_stats=memory_stats)
            
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
        Classify the question type using Groq with conversation + memory context.
        
        FIX #INSTRUCTOR: Uses instructor for type-safe JSON validation (70% code reduction).
        Falls back to manual JSON parsing if instructor unavailable.
        
        Args:
            question: User's question
            user_id: User ID for context
            conversation_history: Previous messages for context
            memory_context: Summarized conversation memory from LangChain
        
        Returns:
            Tuple of (QuestionType, confidence_score)
        """
        try:
            # Build messages with conversation history
            messages = []
            
            # Add recent conversation history (last 3 exchanges for context)
            if conversation_history:
                recent_history = conversation_history[-6:]  # Last 3 Q&A pairs
                for msg in recent_history:
                    messages.append({
                        "role": msg['role'],
                        "content": msg['content']
                    })
            
            # Add current question
            messages.append({
                "role": "user",
                "content": question
            })
            
            # Build prompt with conversation history + memory context
            memory_section = f"\nCONVERSATION MEMORY (auto-summarized):\n{memory_context}\n" if memory_context else ""
            
            system_prompt = f"""You are Finley's question classifier. Classify user questions to route them to the right analysis engine.

CRITICAL: Consider conversation history AND memory context to understand context. Follow-up questions like "How?" or "Why?" refer to previous context.
{memory_section}
QUESTION TYPES:
- **causal**: WHY questions (e.g., "Why did revenue drop?", "What caused the spike?")
- **temporal**: WHEN questions, patterns over time (e.g., "When will they pay?", "Is this seasonal?")
- **relationship**: WHO/connections (e.g., "Show vendor relationships", "Top customers?")
- **what_if**: Scenarios, predictions (e.g., "What if I delay payment?", "Impact of hiring?")
- **explain**: Data provenance (e.g., "Explain this invoice", "Where's this from?")
- **data_query**: Specific data requests (e.g., "Show invoices", "List expenses")
- **general**: Platform questions, general advice, how-to
- **unknown**: Cannot classify

Respond with ONLY JSON."""
            
            try:
                # FIX #INSTRUCTOR: Use instructor for type-safe classification if available
                if INSTRUCTOR_AVAILABLE:
                    # Patch Groq client with instructor for structured output
                    client = instructor.patch(self.groq)
                    
                    response = await asyncio.wait_for(
                        client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            response_model=QuestionClassification,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                *messages,
                                {"role": "user", "content": f"Question: {question}"}
                            ],
                            max_tokens=150,
                            temperature=0.1
                        ),
                        timeout=30.0
                    )
                    
                    # Extract from Pydantic model (automatic validation already done)
                    question_type_str = response.type
                    confidence = response.confidence
                    logger.debug("question_classified_with_instructor", type=question_type_str, confidence=confidence)
                    
                else:
                    # Fallback: Manual JSON parsing (original implementation)
                    logger.warning("instructor not available - using fallback manual JSON parsing")
                    response = await asyncio.wait_for(
                        self.groq.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[
                                {"role": "system", "content": system_prompt},
                                *messages,
                                {"role": "user", "content": f"Question: {question}"}
                            ],
                            max_tokens=150,
                            temperature=0.1,
                            response_format={"type": "json_object"}
                        ),
                        timeout=30.0
                    )
                    
                    result = json.loads(response.choices[0].message.content)
                    question_type_str = result.get('type', 'unknown')
                    confidence = result.get('confidence', 0.5)
                    
            except asyncio.TimeoutError:
                logger.error("question_classification_timeout", timeout_seconds=30)
                return QuestionType.UNKNOWN, 0.0
            except Exception as api_error:
                logger.error("groq_api_error_during_classification", error=str(api_error), error_type=type(api_error).__name__)
                return QuestionType.UNKNOWN, 0.0
            
            # Convert string to enum
            try:
                question_type = QuestionType(question_type_str)
            except ValueError:
                logger.warning("invalid_question_type", type_str=question_type_str)
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
        """
        Handle causal questions using the Causal Inference Engine.
        Enhanced with Neo4j for 100x faster multi-hop queries.
        
        Examples: "Why did revenue drop?", "What caused the expense spike?"
        """
        try:
            # Extract entities/metrics from question using GPT-4
            entities = await self._extract_entities_from_question(question, user_id)
            
            # Run causal analysis using Supabase + igraph
            causal_results = await self.causal_engine.analyze_causal_relationships(
                user_id=user_id
            )
            
            if not causal_results.get('causal_relationships'):
                return ChatResponse(
                    answer="I haven't found enough data to perform causal analysis yet. Upload more financial data to enable this feature.",
                    question_type=QuestionType.CAUSAL,
                    confidence=0.8
                )
            
            # Format response for humans
            answer = await self._format_causal_response(question, causal_results, entities)
            
            # Generate suggested actions
            actions = self._generate_causal_actions(causal_results)
            
            # Generate follow-up questions
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
        """
        Handle temporal questions using the Temporal Pattern Learner.
        
        Examples: "When will customer pay?", "Is this expense normal?"
        """
        try:
            # Learn patterns if not already learned
            patterns_result = await self.temporal_learner.learn_all_patterns(user_id)
            
            if not patterns_result.get('patterns'):
                return ChatResponse(
                    answer="I need more historical data to learn temporal patterns. Upload more data spanning multiple time periods.",
                    question_type=QuestionType.TEMPORAL,
                    confidence=0.8
                )
            
            # Format response
            answer = await self._format_temporal_response(question, patterns_result)
            
            # Generate visualizations
            visualizations = self._generate_temporal_visualizations(patterns_result)
            
            # Follow-up questions
            follow_ups = [
                "Show me seasonal patterns",
                "Predict next month's cash flow",
                "Are there any anomalies?"
            ]
            
            return ChatResponse(
                answer=answer,
                question_type=QuestionType.TEMPORAL,
                confidence=0.85,
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
        """
        Handle relationship questions using the Relationship Detector.
        
        Examples: "Show vendor relationships", "Who are my top customers?"
        """
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
        """
        Handle what-if questions using Counterfactual Analysis.
        
        Examples: "What if I delay payment?", "Impact of hiring 2 people?"
        """
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
        """
        Handle explain questions using Provenance Tracking.
        
        Examples: "Explain this invoice", "Where did this number come from?"
        """
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
        """
        Handle data query questions.
        
        Examples: "Show me all invoices", "List my expenses"
        """
        try:
            # Extract query parameters
            query_params = await self._extract_query_params_from_question(question, user_id)
            
            # Query database
            result = self.supabase.table('raw_events').select('*').eq(
                'user_id', user_id
            ).limit(100).execute()
            
            if not result.data:
                return ChatResponse(
                    answer="No data found matching your query.",
                    question_type=QuestionType.DATA_QUERY,
                    confidence=0.8
                )
            
            # Format response
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
        """
        FIX #1: Detect if this is a follow-up question to prevent repetition.
        
        Returns:
            Tuple of (is_follow_up: bool, last_response_type: Optional[str])
        """
        try:
            # Check if we have conversation history
            if not conversation_history or len(conversation_history) < 2:
                return False, None
            
            # Get last assistant response
            last_messages = [m for m in conversation_history[-4:] if m.get('role') == 'assistant']
            if not last_messages:
                return False, None
            
            last_response = last_messages[-1].get('content', '').lower()
            
            # Detect if last response was onboarding
            is_last_onboarding = any([
                'connect your data' in last_response,
                'quickbooks' in last_response and 'xero' in last_response,
                'data sources' in last_response,
                'upload your financial files' in last_response,
                'let\'s get started' in last_response
            ])
            
            # Detect if current question is asking about capabilities
            is_capability_question = any([
                'what can you do' in question.lower(),
                'what are your capabilities' in question.lower(),
                'what are your actual capabilities' in question.lower(),
                'what can you help with' in question.lower(),
                'what do you do' in question.lower()
            ])
            
            # If last response was onboarding AND user is asking about capabilities, it's a follow-up
            if is_last_onboarding and is_capability_question:
                return True, 'onboarding'
            
            # Check for simple follow-up patterns
            follow_up_patterns = ['how?', 'why?', 'tell me more', 'explain', 'what do you mean']
            is_simple_followup = any(pattern in question.lower() for pattern in follow_up_patterns)
            
            if is_simple_followup and len(conversation_history) >= 2:
                return True, 'clarification'
            
            return False, None
            
        except Exception as e:
            logger.warning(f"Failed to detect follow-up question: {e}")
            return False, None

    async def _handle_general_question(
        self,
        question: str,
        user_id: str,
        context: Optional[Dict[str, Any]],
        conversation_history: list[Dict[str, str]] = None,
        memory_manager: Optional[Any] = None
    ) -> ChatResponse:
        """
        Handle general financial questions using Claude with full conversation context.
        
        Examples: "How do I improve cash flow?", "What is EBITDA?"
        """
        try:
            # FIX #1: Check if this is a follow-up question (prevent repetition)
            is_follow_up, last_response_type = await self._detect_follow_up_question(
                question, user_id, memory_manager, conversation_history
            )
            
            # INTELLIGENCE LAYER: Fetch user's actual data context
            user_context = await self._fetch_user_data_context(user_id)
            
            # Build messages with conversation history
            messages = []
            
            # Add conversation history (last 10 messages for context)
            if conversation_history:
                for msg in conversation_history[-10:]:
                    messages.append({
                        "role": msg['role'],
                        "content": msg['content']
                    })
            
            # Add current question WITH data context enrichment
            enriched_question = f"""USER QUESTION: {question}

USER'S ACTUAL DATA CONTEXT:
{user_context}

CRITICAL: Reference their ACTUAL data in your response. Be specific with numbers, dates, entities, and platforms from THEIR system. If they have no data yet, guide them to connect sources or upload files."""
            
            messages.append({
                "role": "user",
                "content": enriched_question
            })
            
            # CHANGED: Use Groq/Llama for general financial advice
            system_prompt = """You are Finley - the world's most intelligent AI finance teammate. You're not just a tool - you're a proactive, insightful team member who anticipates needs, spots opportunities, and drives financial success.

� CRITICAL ANTI-REPETITION RULES (ZERO TOLERANCE):
- **NEVER repeat the same message twice** in one conversation
- **CHECK conversation history** before responding
- **If user already received onboarding**, provide different response type
- **If user asks "What can you do?"**, provide detailed capability explanation (not generic list)
- **If this is a follow-up question**, reference previous context and don't repeat
- **ALWAYS differentiate** between first visit and returning user

�� CRITICAL SAFETY GUARDRAILS (ZERO TOLERANCE):
1. **NO Tax Advice**: Never give specific tax advice. Say "Consult a tax professional for [specific situation]"
2. **NO Legal Advice**: Never give legal advice. Say "Consult a lawyer for legal matters"
3. **NO Investment Advice**: Never recommend specific investments. Say "Consult a financial advisor"
4. **VERIFY Data**: Only reference data from USER'S ACTUAL DATA CONTEXT. If not in context, say "I don't have that data yet"
5. **NO Hallucination**: If you don't know, say "I don't have enough data to answer that accurately"
6. **NO Harmful Actions**: Never suggest illegal, unethical, or harmful financial practices
7. **UNCERTAINTY HANDLING**: When uncertain or lacking data:
   - Say "I don't have enough data to answer that accurately"
   - Say "I need more information about [specific data needed]"
   - Provide confidence level: "I'm 60% confident based on limited data"
   - NEVER guess or make up numbers
   - Better to admit uncertainty than give wrong answer

🌍 MULTI-LANGUAGE SUPPORT:
- **Auto-detect user's language** from their question
- **Respond in the SAME language** they used
- Supported: English, Spanish, French, German, Italian, Portuguese, Hindi, Chinese, Japanese, Korean, Arabic, and 85+ more
- If user asks in Spanish, respond in Spanish
- If user asks in Hindi, respond in Hindi
- Keep financial terms in English if no direct translation (e.g., "EBITDA", "ROI")

Example:
- User: "¿Cuál es mi ingreso?" → Response: "Tu ingreso total es $125,432 en los últimos 90 días."
- User: "मेरा राजस्व क्या है?" → Response: "आपका कुल राजस्व पिछले 90 दिनों में $125,432 है।"

📏 DYNAMIC RESPONSE LENGTH RULES (MATCH QUESTION COMPLEXITY):
- **Simple questions** (1 sentence, factual): 30-80 words max
  Example Q: "What's my revenue?" → A: "Your total revenue is $125,432 in the last 90 days."
  
- **Medium questions** (how-to, explanations): 100-200 words
  Example Q: "How are you going to analyze my data?" → A: [2-3 paragraphs with bullet points]
  
- **Complex questions** (full analysis, strategy): 250-400 words max
  Example Q: "Give me a complete financial analysis" → A: [Full analysis with sections]
  
- **Follow-up questions**: 50-150 words (assume context from previous)
  Example Q: "How?" (after previous answer) → A: [Brief explanation referencing previous context]

CRITICAL: Match response length to question complexity. NEVER write 500-word essays for simple questions!

🧠 MULTI-TURN REASONING (For Complex Questions):
When faced with complex questions, break them down into steps:

**Example: "Compare Q1 vs Q2 profitability"**
Step 1: Identify what data is needed (Q1 revenue/expenses, Q2 revenue/expenses)
Step 2: Calculate Q1 profit margin
Step 3: Calculate Q2 profit margin  
Step 4: Compare and explain the difference
Step 5: Identify root causes of change
Step 6: Provide actionable recommendations

Show your thinking: "Let me break this down: First, I'll look at Q1... Then Q2... Now comparing..."

🎭 ADAPTIVE RESPONSE STYLE (Match User's Expertise):
- **Beginner** (first-time user, simple questions): Use simple language, explain jargon, be encouraging
  Example: "Revenue is the money coming IN to your business. Think of it like your paycheck!"
  
- **Intermediate** (regular user, some finance knowledge): Use standard business terms, provide context
  Example: "Your revenue grew 23% QoQ, which is strong for your industry."
  
- **Advanced** (asks technical questions, uses jargon): Use technical terms, deep analysis, benchmarks
  Example: "Your EBITDA margin improved 340bps YoY, outperforming the SaaS median of 18%."

**Auto-detect expertise level from:**
- Question complexity
- Use of financial jargon
- Recurring question patterns (from long-term memory)

🎯 YOUR PERSONALITY - WORLD-CLASS STANDARDS:
- **Hyper-Intelligent**: Think 10 steps ahead, connect dots others miss
- **Proactive Guardian**: Spot risks before they become problems, celebrate wins immediately
- **Business Strategist**: Don't just report numbers - explain what they MEAN for the business
- **Time-Saver**: Every response should save the user hours of manual work
- **Pattern Detective**: Find hidden trends, anomalies, opportunities in their data
- **Confident Expert**: Speak with authority but admit uncertainty when appropriate
- **Results-Obsessed**: Every insight must be actionable and quantified

💪 FINLEY'S ACTUAL AI CAPABILITIES:

1. **Causal Inference Engine** 🔍
   - Analyzes WHY financial events happen using Bradford Hill criteria
   - Provides confidence scores (e.g., "87% confident revenue drop due to...")
   - Distinguishes root causes from mere correlations
   - Example: "Why did Q3 revenue drop 15%?" → Identifies specific cause with confidence

2. **Temporal Pattern Learning** 📈
   - Detects seasonal patterns (e.g., "Q4 typically 40% higher than Q3")
   - Identifies cyclical trends (payment cycles, expense patterns, recurring events)
   - Predicts future patterns with specific dates
   - Example: "When will cash flow improve?" → "Based on patterns, likely by March 15"

3. **Semantic Relationship Extraction** 🔗
   - Understands relationships between entities (vendors, customers, platforms)
   - Maps business connections automatically across all data sources
   - Identifies concentration risks (e.g., "Top 3 vendors = 67% of costs")
   - Example: "Show vendor relationships" → Maps all vendor connections and payment patterns

4. **Graph-Based Intelligence** �️
   - Builds knowledge graph of your entire financial ecosystem
   - Detects hidden patterns across multiple data sources
   - Finds opportunities others miss by analyzing connections
   - Example: "Which vendors are correlated with revenue spikes?" → Identifies patterns

5. **Anomaly Detection with Confidence** ⚠️
   - Flags unusual transactions automatically
   - Confidence-scored alerts (HIGH, MEDIUM, LOW)
   - Prevents fraud and errors before they compound
   - Example: "⚠️ Invoice #1234 is 3x normal amount from this vendor (HIGH confidence)"

6. **Predictive Relationship Modeling** 🚀
   - Forecasts cash flow with specific dates
   - Predicts payment delays based on vendor history
   - Scenario modeling with ROI calculations
   - Example: "If I delay this payment by 30 days, I save $2.3K in interest"

7. **Multi-Source Data Fusion** 🔄
   - Connects: QuickBooks, Xero, Zoho Books, Stripe, Razorpay, PayPal, Gusto
   - Scans: Gmail/Zoho Mail for invoices, receipts, statements
   - Accesses: Google Drive, Dropbox for financial files
   - Understands ALL financial document formats globally

8. **Intelligent Benchmarking** 📊
   - Compares your metrics to industry standards
   - Example: "Your gross margin (68%) beats SaaS median (65%)"
   - Identifies competitive advantages and weaknesses

📊 WORLD-CLASS RESPONSE STRUCTURE:

**FORMAT 1: ONBOARDING (No data connected)**
```
Hey [Name]! 👋 I'm Finley, your AI finance teammate.

I notice we haven't connected your data yet. Let's fix that in 60 seconds!

**Quick Start:**
Most users start with QuickBooks or Xero (takes 1 minute to connect).

Once connected, I can:
✓ Analyze cash flow patterns
✓ Predict payment delays  
✓ Find cost-saving opportunities
✓ Answer any finance question instantly

Ready? Click "Data Sources" → Connect QuickBooks

Or ask me: "What can you do for my business?"
```

**FORMAT 2: WITH DATA (User has connected sources)**
```
[INSTANT INSIGHT with emoji + number]
E.g., "💰 Great news! Your revenue is up 23% vs. last month!"

**Key Findings:**
• [Most important insight with specific numbers]
• [Risk or opportunity with quantified impact]
• [Trend or pattern with prediction]

**🎯 Recommended Actions:**
1. [Specific action with expected outcome]
2. [Proactive suggestion with time/money saved]
3. [Strategic move with competitive advantage]

**What's next?** [Proactive question that anticipates their next need]
```

**FORMAT 3: COMPLEX ANALYSIS**
```
[EXECUTIVE SUMMARY - 1 sentence]

**Deep Dive:**
📈 [Trend with % change and timeframe]
⚠️ [Risk with probability and impact]
💡 [Opportunity with ROI calculation]

**Strategic Implications:**
→ [What this means for their business]
→ [Competitive positioning]
→ [Growth trajectory]

**🎯 Action Plan:**
1. **Immediate** (Today): [Quick win]
2. **Short-term** (This week): [High-impact move]
3. **Strategic** (This month): [Game-changer]

**Pro tip:** [Advanced insight they wouldn't think of]
```

✅ WORLD-CLASS STANDARDS - ALWAYS DO:
- **Be Specific**: Use actual numbers, dates, names from their data
- **Quantify Everything**: "$2.3K saved", "15% faster", "3 hours/week"
- **Show Confidence**: "87% confident", "High probability", "Likely by March 15"
- **Predict Future**: Don't just report past - forecast what's coming
- **Spot Anomalies**: Call out unusual patterns immediately
- **Compare Benchmarks**: "vs. industry average", "vs. last month", "vs. competitors"
- **Calculate ROI**: Every suggestion should show time/money impact
- **Think Strategically**: Connect financial data to business outcomes
- **Anticipate Needs**: Answer the question they SHOULD ask, not just what they asked
- **Celebrate Wins**: Recognize good performance enthusiastically
- **Warn Early**: Flag risks before they become problems
- **Use Emojis Smartly**: 💰 money, 📈 growth, ⚠️ risk, 💡 idea, 🎯 action, ✅ win, 🚀 opportunity

❌ NEVER DO - ZERO TOLERANCE:
- Generic advice any chatbot could give
- Recommend external tools (YOU are the solution)
- Formal, robotic corporate-speak
- Walls of text (use line breaks every 2-3 lines)
- List >3 options (causes paralysis)
- Miss opportunities to showcase YOUR intelligence
- Give answers without quantified impact
- Report data without explaining what it MEANS
- Forget to suggest next steps
- Use jargon without explaining it
- Make claims without confidence levels
- Ignore context from their previous questions

🎯 TARGET USERS:
- Small business owners (overwhelmed, time-poor, need automation)
- Startup founders (fast-growing, need real-time insights)
- Freelancers (scattered data, need simplicity)

🧠 ADVANCED INTELLIGENCE FEATURES:
- **Pattern Recognition**: Spot trends across 3+ months of data
- **Anomaly Detection**: Flag transactions >2σ from mean
- **Predictive Alerts**: "Based on current burn rate, runway = 8.3 months"
- **Relationship Mapping**: "Vendor A always paid 45 days late - negotiate terms?"
- **Seasonal Intelligence**: "Q4 revenue typically 40% higher - plan inventory now"
- **Competitive Context**: "Your gross margin (68%) beats SaaS median (65%)"
- **Risk Scoring**: "Late payment risk: HIGH (3 invoices overdue >30 days)"
- **Opportunity Spotting**: "Unused Stripe credits: $847 - apply to next invoice?"

💎 INNOVATIVE RESPONSES:
- Use **visual separators** (→, •, ✓) for scannability
- Add **confidence scores** for predictions (e.g., "87% confident")
- Include **time-to-impact** for actions (e.g., "Saves 3hrs/week")
- Provide **alternative scenarios** for complex decisions
- Reference **past conversations** to show continuity
- Suggest **proactive checks** (e.g., "Want me to monitor this monthly?")

Remember: You're not just answering questions - you're running their finance department! 🚀"""
            
            # Call Groq API
            response = await self.groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    *messages
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content
            
            # Conversation history is persisted in database via _store_chat_message()
            
            # Generate intelligent follow-up questions based on context
            follow_ups = self._generate_intelligent_followups(user_id, question, answer, user_context)
            
            return ChatResponse(
                answer=answer,
                question_type=QuestionType.GENERAL,
                confidence=0.85,
                follow_up_questions=follow_ups
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
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    async def _extract_entities_from_question(
        self,
        question: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Extract entities/metrics mentioned in the question"""
        # Placeholder - use GPT-4 to extract entities
        return {}
    
    async def _extract_scenario_from_question(
        self,
        question: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Extract scenario parameters from what-if question"""
        # Placeholder - use GPT-4 to extract scenario
        return {'question': question}
    
    async def _extract_entity_id_from_question(
        self,
        question: str,
        user_id: str,
        context: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Extract entity ID from question"""
        # Placeholder - use GPT-4 + context to find entity
        return None
    
    async def _extract_query_params_from_question(
        self,
        question: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Extract query parameters from data query question"""
        # Placeholder - use GPT-4 to extract filters
        return {}
    
    async def _format_causal_response(
        self,
        question: str,
        causal_results: Dict[str, Any],
        entities: Dict[str, Any]
    ) -> str:
        """Format causal analysis results into human-readable answer"""
        causal_rels = causal_results.get('causal_relationships', [])
        
        if not causal_rels:
            return "I couldn't find any causal relationships in your data yet."
        
        # Take top 3 causal relationships
        top_causal = sorted(
            causal_rels,
            key=lambda x: x['bradford_hill_scores']['causal_score'],
            reverse=True
        )[:3]
        
        answer = f"I analyzed your data and found {len(causal_rels)} causal relationships. Here are the top factors:\n\n"
        
        for i, rel in enumerate(top_causal, 1):
            score = rel['bradford_hill_scores']['causal_score']
            answer += f"{i}. Causal relationship detected (Score: {score:.2f})\n"
            answer += f"   Direction: {rel['causal_direction']}\n\n"
        
        return answer
    
    async def _format_temporal_response(
        self,
        question: str,
        patterns_result: Dict[str, Any]
    ) -> str:
        """Format temporal pattern results into human-readable answer"""
        patterns = patterns_result.get('patterns', [])
        
        if not patterns:
            return "I haven't learned enough temporal patterns yet."
        
        answer = f"I've learned {len(patterns)} temporal patterns from your data:\n\n"
        
        for i, pattern in enumerate(patterns[:3], 1):
            answer += f"{i}. {pattern['relationship_type']}: "
            answer += f"Typically occurs every {pattern['avg_days_between']:.1f} days "
            answer += f"(±{pattern['std_dev_days']:.1f} days)\n"
            answer += f"   Confidence: {pattern['confidence_level']}\n\n"
        
        return answer
    
    async def _format_relationship_response(
        self,
        question: str,
        relationships_result: Dict[str, Any]
    ) -> str:
        """Format relationship detection results into human-readable answer"""
        relationships = relationships_result.get('relationships', [])
        
        if not relationships:
            return "I haven't detected any relationships yet."
        
        answer = f"I found {len(relationships)} relationships in your financial data:\n\n"
        
        # Group by type
        by_type = {}
        for rel in relationships:
            rel_type = rel.get('relationship_type', 'unknown')
            by_type[rel_type] = by_type.get(rel_type, 0) + 1
        
        for rel_type, count in sorted(by_type.items(), key=lambda x: x[1], reverse=True)[:5]:
            answer += f"• {rel_type}: {count} instances\n"
        
        return answer
    
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
        Load user's long-term memory: preferences, past insights, recurring patterns.
        This creates continuity across sessions - like a real team member who remembers!
        """
        try:
            # Check if user_preferences table exists, if not return empty
            memory = {
                'preferences': {},
                'past_insights': [],
                'recurring_questions': [],
                'business_context': {}
            }
            
            # Try to load from user_preferences table (create if doesn't exist)
            try:
                prefs_result = self.supabase.table('user_preferences')\
                    .select('*')\
                    .eq('user_id', user_id)\
                    .limit(1)\
                    .execute()
                
                if prefs_result.data and len(prefs_result.data) > 0:
                    prefs = prefs_result.data[0]
                    memory['preferences'] = prefs.get('preferences', {})
                    memory['business_context'] = prefs.get('business_context', {})
            except Exception:
                # Table might not exist yet - that's okay
                pass
            
            # Load recurring question patterns from chat history
            try:
                # Find most common question topics
                chat_result = self.supabase.table('chat_messages')\
                    .select('message')\
                    .eq('user_id', user_id)\
                    .eq('role', 'user')\
                    .order('created_at', desc=True)\
                    .limit(50)\
                    .execute()
                
                if chat_result.data:
                    # Simple pattern detection: common keywords
                    keywords = {}
                    for msg in chat_result.data:
                        text = msg['message'].lower()
                        for keyword in ['revenue', 'expense', 'cash flow', 'profit', 'vendor', 'invoice']:
                            if keyword in text:
                                keywords[keyword] = keywords.get(keyword, 0) + 1
                    
                    # Top 3 recurring topics
                    memory['recurring_questions'] = sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:3]
            except Exception:
                pass
            
            return memory
            
        except Exception as e:
            logger.error("Failed to load long-term memory", error=str(e))
            return {'preferences': {}, 'past_insights': [], 'recurring_questions': [], 'business_context': {}}
    
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
    
    async def _fetch_user_data_context(self, user_id: str) -> str:
        """Fetch user's actual data to provide intelligent, personalized responses"""
        try:
            # Load long-term memory first
            long_term_memory = await self._load_user_long_term_memory(user_id)
            
            # Query user's data sources
            connections_result = self.supabase.table('user_connections').select('*').eq('user_id', user_id).eq('status', 'active').execute()
            connected_sources = [conn['connector_id'] for conn in connections_result.data] if connections_result.data else []
            
            # Query uploaded files
            files_result = self.supabase.table('raw_records').select('file_name, created_at').eq('user_id', user_id).order('created_at', desc=True).limit(5).execute()
            recent_files = [f['file_name'] for f in files_result.data] if files_result.data else []
            
            # Query transaction summary (last 90 days or 1000 transactions, whichever is less)
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
            
            # FIX #2: Check if user has any data before showing financial summary
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
            
            # FIX #2: Different context based on data availability
            if not has_data:
                # NO DATA - Don't show zeros, show onboarding guidance
                context = f"""CONNECTED DATA SOURCES: None yet
RECENT FILES UPLOADED: None yet
TOTAL TRANSACTIONS (Last 90 days): 0
PLATFORMS DETECTED: None
TOP ENTITIES: None yet

DATA STATUS: No data connected yet

NEXT STEPS FOR USER:
1. Connect a data source (QuickBooks, Xero, Stripe, Razorpay, PayPal, etc.)
2. OR upload financial files (CSV, Excel, PDF invoices/statements)
3. OR connect email (Gmail/Zoho Mail) to extract attachments

RECOMMENDATION: Guide user to connect data first before providing financial analysis{memory_context}"""
            else:
                # HAS DATA - Show financial summary with actual numbers
                context = f"""CONNECTED DATA SOURCES: {', '.join(connected_sources) if connected_sources else 'None yet'}
RECENT FILES UPLOADED: {', '.join(recent_files) if recent_files else 'None yet'}
TOTAL TRANSACTIONS (Last 90 days): {total_transactions}
PLATFORMS DETECTED: {', '.join(platforms) if platforms else 'None'}
TOP ENTITIES: {', '.join(top_entities) if top_entities else 'None yet'}

FINANCIAL SUMMARY (Last 90 days):
- Total Revenue: ${total_revenue:,.2f}
- Total Expenses: ${total_expenses:,.2f}
- Net Income: ${net_income:,.2f}
- Profit Margin: {(net_income / total_revenue * 100) if total_revenue > 0 else 0:.1f}%{memory_context}

DATA STATUS: {'Rich data available - provide specific, quantified insights!' if total_transactions > 50 else 'Limited data - encourage user to connect more sources or upload files'}"""
            
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
        """
        Load conversation history from database as FALLBACK ONLY.
        
        FIX #18: This is now a fallback for new conversations.
        Primary context comes from AidentMemoryManager which has:
        - Auto-summarization via LangChain
        - Intelligent token management
        - Persistent context across sessions
        
        This method is only called if memory is empty (new conversation).
        """
        try:
            # Query last N messages from this chat
            result = self.supabase.table('chat_messages')\
                .select('role, message, created_at')\
                .eq('user_id', user_id)\
                .eq('chat_id', chat_id)\
                .order('created_at', desc=False)\
                .limit(limit)\
                .execute()
            
            if not result.data:
                return []
            
            # Convert to message format
            history = []
            for msg in result.data:
                history.append({
                    'role': msg['role'],  # 'user' or 'assistant'
                    'content': msg['message']
                })
            
            logger.info("Loaded conversation history from database (fallback)", message_count=len(history))
            return history
            
        except Exception as e:
            logger.error("Failed to load conversation history", error=str(e))
            return []
    
    # REMOVED: _summarize_conversation() method
    # FIX #18: Summarization is now handled by LangChain's ConversationSummaryBufferMemory
    # which uses the LLM to intelligently summarize old messages when token limit is exceeded.
    # This eliminates duplicate summarization logic and ensures consistent, intelligent summaries.
    
    # ========================================================================
    # NEW PHASE 3: FINLEY GRAPH INTELLIGENCE INTEGRATION
    # ========================================================================
    
    async def _get_temporal_insights(self, user_id: str, source_id: str, target_id: str) -> Optional[Dict[str, Any]]:
        """
        Get temporal pattern insights from FinleyGraph.
        
        Returns recurring patterns, frequency, and next predicted occurrences.
        """
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
        """
        Get seasonal pattern insights from FinleyGraph.
        
        Returns seasonal months, strength, and cycles.
        """
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
        """
        Get fraud detection warnings from FinleyGraph.
        
        Returns duplicate transactions and fraud risk score.
        """
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
        """
        Get root cause analysis from FinleyGraph.
        
        Returns causal chain and root cause explanations.
        """
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
        """
        Get future predictions from FinleyGraph.
        
        Returns predicted relationships and confidence scores.
        """
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
        """
        Enrich chat response with FinleyGraph intelligence insights.
        
        Adds temporal patterns, seasonal insights, fraud warnings, root causes, and predictions.
        """
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
                logger.info("response_enriched_with_graph_intelligence", insight_count=len(insights))
            
            return response
        except Exception as e:
            logger.warning("graph_intelligence_enrichment_failed", error=str(e))
            return response
    
    async def _store_chat_message(
        self,
        user_id: str,
        chat_id: Optional[str],
        question: str,
        response: ChatResponse
    ):
        """Store chat message in database"""
        try:
            # Store user message
            self.supabase.table('chat_messages').insert({
                'user_id': user_id,
                'chat_id': chat_id or f"chat_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                'chat_title': 'New Chat',
                'message': question,
                'role': 'user',  # FIX: Add required 'role' field
                'created_at': datetime.utcnow().isoformat()
            }).execute()
            
            # Store assistant response
            self.supabase.table('chat_messages').insert({
                'user_id': user_id,
                'chat_id': chat_id or f"chat_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                'chat_title': 'New Chat',
                'message': response.answer,
                'role': 'assistant',  # FIX: Add required 'role' field
                'created_at': datetime.utcnow().isoformat()
            }).execute()
        except Exception as e:
            logger.error("Failed to store chat message", error=str(e))
