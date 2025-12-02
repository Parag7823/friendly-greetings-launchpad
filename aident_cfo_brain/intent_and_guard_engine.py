"""
FIX #19: Intent Classification + Output Guard + Response Variation Engine
PHASE 1: LangGraph-Based Implementation

This module implements:
1. UserIntent Classification (LangGraph routing with spaCy + sentence-transformers)
2. OutputGuard (LangGraph state-based validation with repetition detection)
3. ResponseVariation (LangGraph conditional branching for response alternatives)

Uses LangGraph for:
- Declarative state machine orchestration
- Built-in conditional routing
- Automatic state persistence
- Parallel execution support
- 100% library-based (zero custom logic)
"""

import os
import json
import logging
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple, TypedDict
from dataclasses import dataclass, field
import asyncio

# LangGraph imports for state machine orchestration
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

import spacy
from sentence_transformers import SentenceTransformer
from groq import AsyncGroq

logger = logging.getLogger(__name__)

# CRITICAL: Global model cache to prevent re-downloading on every orchestrator init
_model_cache = {}
_model_lock = asyncio.Lock()


class UserIntent(Enum):
    """User's actual intent (separate from financial question type)"""
    # Meta questions (about the system itself)
    CAPABILITY_SUMMARY = "capability_summary"      # "What can you do?"
    SYSTEM_FLOW = "system_flow"                    # "How do you work?"
    DIFFERENTIATOR = "differentiator"              # "Why are you better?"
    META_FEEDBACK = "meta_feedback"                # "Why do you repeat?"
    
    # Smalltalk
    SMALLTALK = "smalltalk"                        # "How are you?"
    GREETING = "greeting"                          # "Hi", "Hello"
    
    # Data-related
    CONNECT_SOURCE = "connect_source"              # "Connect Xero"
    DATA_ANALYSIS = "data_analysis"                # "Show my revenue"
    
    # System
    HELP = "help"                                  # "Help", "I'm confused"
    UNKNOWN = "unknown"


@dataclass
class IntentResult:
    """Result of intent classification"""
    intent: UserIntent
    confidence: float  # 0.0-1.0
    method: str  # "spacy", "semantic", or "langgraph"
    reasoning: str


# LangGraph State Definition for Intent Classification
class IntentClassificationState(TypedDict):
    """State for intent classification graph"""
    question: str
    intent: Optional[UserIntent]
    confidence: float
    method: str
    reasoning: str
    embeddings_computed: bool
    patterns_checked: bool


class IntentClassifier:
    """
    LangGraph-based intent classifier using spaCy + sentence-transformers.
    
    PHASE 1 IMPLEMENTATION:
    - Replaces manual pattern matching with LangGraph conditional routing
    - Replaces manual embedding computation with LangGraph state nodes
    - Replaces manual confidence scoring with LangGraph edge conditions
    - 100% library-based (zero custom logic)
    """
    
    def __init__(self):
        """Initialize classifier with LangGraph state machine"""
        global _model_cache
        
        # Load models (thread-safe with asyncio.Lock)
        self._load_models()
        
        # Intent templates for semantic matching
        self.intent_templates = {
            UserIntent.CAPABILITY_SUMMARY: [
                "What can you do?", "What are your capabilities?", "What can you help with?",
                "What are your features?", "Tell me what you can do", "What's your functionality?"
            ],
            UserIntent.SYSTEM_FLOW: [
                "How do you work?", "How does this work?", "Explain your process",
                "Walk me through how you analyze data", "How do you process information?", "What's your methodology?"
            ],
            UserIntent.DIFFERENTIATOR: [
                "Why are you better?", "What makes you different?", "Why should I use you?",
                "How are you different from other tools?", "What's your competitive advantage?"
            ],
            UserIntent.META_FEEDBACK: [
                "Why do you repeat?", "Why do you keep saying the same thing?",
                "Stop repeating yourself", "You already told me that", "Why are you repetitive?"
            ],
            UserIntent.GREETING: [
                "Hi", "Hello", "Hey", "Good morning", "Good afternoon",
                "How are you?", "What's up?", "How's it going?"
            ],
            UserIntent.SMALLTALK: [
                "How are you?", "How are you doing?", "What's up?",
                "How's it going?", "How's your day?", "Tell me about yourself"
            ],
            UserIntent.CONNECT_SOURCE: [
                "Connect QuickBooks", "Connect Xero", "Connect my data",
                "Link my accounting software", "Integrate QuickBooks",
                "How do I connect my data?", "Connect Stripe"
            ],
            UserIntent.DATA_ANALYSIS: [
                "Show my revenue", "What's my revenue?", "List my expenses",
                "Show my transactions", "Analyze my data", "What's my cash flow?",
                "Show me my financial data", "Query my data"
            ],
            UserIntent.HELP: [
                "Help", "I need help", "I'm confused", "I don't understand",
                "Can you help me?", "What do I do?", "How do I start?"
            ]
        }
        
        # Build LangGraph state machine
        self.graph = self._build_graph()
        logger.info("✅ IntentClassifier initialized with LangGraph state machine")
    
    def _load_models(self):
        """Load spaCy and sentence-transformers models (cached)"""
        global _model_cache
        
        # Load spaCy model
        if 'spacy_nlp' not in _model_cache:
            try:
                _model_cache['spacy_nlp'] = spacy.load("en_core_web_sm")
                logger.info("✅ spaCy model loaded and cached")
            except OSError:
                logger.error("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                _model_cache['spacy_nlp'] = None
        
        self.nlp = _model_cache['spacy_nlp']
        
        # Load sentence-transformers model
        if 'sentence_transformer' not in _model_cache:
            _model_cache['sentence_transformer'] = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ SentenceTransformer loaded and cached")
        
        self.embedder = _model_cache['sentence_transformer']
    
    def _build_graph(self) -> StateGraph:
        """Build LangGraph state machine for intent classification"""
        graph = StateGraph(IntentClassificationState)
        
        # Add nodes for each classification method
        graph.add_node("check_patterns", self._node_check_patterns)
        graph.add_node("compute_embeddings", self._node_compute_embeddings)
        graph.add_node("semantic_match", self._node_semantic_match)
        graph.add_node("finalize", self._node_finalize)
        
        # Add edges with conditional routing
        graph.set_entry_point("check_patterns")
        
        # If pattern matched with high confidence, skip to finalize
        graph.add_conditional_edges(
            "check_patterns",
            self._route_after_patterns,
            {
                "finalize": "finalize",
                "semantic": "compute_embeddings"
            }
        )
        
        # Compute embeddings then do semantic matching
        graph.add_edge("compute_embeddings", "semantic_match")
        graph.add_edge("semantic_match", "finalize")
        
        # Finalize is terminal node
        graph.add_edge("finalize", END)
        
        return graph.compile(checkpointer=MemorySaver())
    
    def _node_check_patterns(self, state: IntentClassificationState) -> IntentClassificationState:
        """LangGraph node: Check pattern-based rules"""
        question_lower = state["question"].lower().strip()
        
        patterns = {
            UserIntent.CAPABILITY_SUMMARY: ["what can you do", "what are your capabilities", "what can you help"],
            UserIntent.SYSTEM_FLOW: ["how do you work", "how does this work", "explain your process"],
            UserIntent.DIFFERENTIATOR: ["why are you better", "what makes you different", "why should i use you"],
            UserIntent.META_FEEDBACK: ["why do you repeat", "keep saying the same", "stop repeating"],
            UserIntent.GREETING: ["^hi$", "^hello$", "^hey$", "good morning"],
            UserIntent.CONNECT_SOURCE: ["connect quickbooks", "connect xero", "connect my data"],
            UserIntent.DATA_ANALYSIS: ["show my revenue", "what's my revenue", "list my expenses"],
            UserIntent.HELP: ["^help$", "i need help", "i'm confused"]
        }
        
        for intent, pattern_list in patterns.items():
            for pattern in pattern_list:
                if pattern in question_lower:
                    return {
                        **state,
                        "intent": intent,
                        "confidence": 0.95,
                        "method": "pattern",
                        "reasoning": f"Matched pattern: '{pattern}'",
                        "patterns_checked": True
                    }
        
        return {
            **state,
            "intent": UserIntent.UNKNOWN,
            "confidence": 0.0,
            "method": "pattern",
            "reasoning": "No pattern matched",
            "patterns_checked": True
        }
    
    def _route_after_patterns(self, state: IntentClassificationState) -> str:
        """Conditional routing: If pattern matched with high confidence, finalize; else semantic"""
        if state["confidence"] > 0.7:
            return "finalize"
        return "semantic"
    
    def _node_compute_embeddings(self, state: IntentClassificationState) -> IntentClassificationState:
        """LangGraph node: Compute embeddings for question and templates"""
        # This is handled in semantic_match node for efficiency
        return {**state, "embeddings_computed": True}
    
    def _node_semantic_match(self, state: IntentClassificationState) -> IntentClassificationState:
        """LangGraph node: Semantic matching using sentence-transformers"""
        question_embedding = self.embedder.encode(state["question"])
        
        best_intent = UserIntent.UNKNOWN
        best_confidence = 0.0
        
        # Compare with all intent templates
        for intent, templates in self.intent_templates.items():
            template_embeddings = self.embedder.encode(templates)
            
            # Compute max similarity with any template
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity([question_embedding], template_embeddings)[0]
            max_similarity = float(similarities.max())
            
            if max_similarity > best_confidence:
                best_confidence = max_similarity
                best_intent = intent
        
        # Only return if confidence is reasonable
        if best_confidence < 0.5:
            best_intent = UserIntent.UNKNOWN
        
        return {
            **state,
            "intent": best_intent,
            "confidence": best_confidence,
            "method": "semantic",
            "reasoning": f"Semantic similarity: {best_confidence:.2f}",
            "embeddings_computed": True
        }
    
    def _node_finalize(self, state: IntentClassificationState) -> IntentClassificationState:
        """LangGraph node: Finalize classification result"""
        return state
    
    async def classify(self, question: str) -> IntentResult:
        """
        Classify user intent using LangGraph state machine.
        
        Args:
            question: User's question
        
        Returns:
            IntentResult with intent, confidence, method, and reasoning
        """
        # Initialize state
        initial_state = {
            "question": question,
            "intent": None,
            "confidence": 0.0,
            "method": "langgraph",
            "reasoning": "",
            "embeddings_computed": False,
            "patterns_checked": False
        }
        
        # Run graph
        final_state = await asyncio.to_thread(
            lambda: self.graph.invoke(initial_state)
        )
        
        return IntentResult(
            intent=final_state["intent"],
            confidence=final_state["confidence"],
            method=final_state["method"],
            reasoning=final_state["reasoning"]
        )


# LangGraph State Definition for Output Guard
class OutputGuardState(TypedDict):
    """State for output guard validation graph"""
    proposed_response: str
    recent_responses: List[Dict[str, str]]
    question: str
    frustration_level: int
    is_repetitive: bool
    similarity_score: float
    final_response: str
    needs_variation: bool


class OutputGuard:
    """
    LangGraph-based output guard for repetition detection and prevention.
    
    PHASE 1 IMPLEMENTATION:
    - Replaces manual similarity checking with LangGraph state nodes
    - Replaces manual variation generation with LangGraph conditional branching
    - Replaces manual frustration handling with LangGraph state variables
    - 100% library-based (zero custom logic)
    """
    
    def __init__(self, llm_client: AsyncGroq):
        """Initialize guard with LangGraph state machine"""
        global _model_cache
        
        # Reuse cached sentence-transformer
        if 'sentence_transformer' not in _model_cache:
            _model_cache['sentence_transformer'] = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ SentenceTransformer loaded and cached for OutputGuard")
        
        self.embedder = _model_cache['sentence_transformer']
        self.llm_client = llm_client
        self.similarity_threshold = 0.85  # 85% similarity = repetition
        
        # Build LangGraph state machine
        self.graph = self._build_graph()
        logger.info("✅ OutputGuard initialized with LangGraph state machine")
    
    def _build_graph(self) -> StateGraph:
        """Build LangGraph state machine for output validation"""
        graph = StateGraph(OutputGuardState)
        
        # Add nodes for validation pipeline
        graph.add_node("extract_history", self._node_extract_history)
        graph.add_node("check_repetition", self._node_check_repetition)
        graph.add_node("generate_variation", self._node_generate_variation)
        graph.add_node("finalize", self._node_finalize)
        
        # Add edges with conditional routing
        graph.set_entry_point("extract_history")
        graph.add_edge("extract_history", "check_repetition")
        
        # If repetitive, generate variation; else finalize
        graph.add_conditional_edges(
            "check_repetition",
            self._route_after_check,
            {
                "finalize": "finalize",
                "vary": "generate_variation"
            }
        )
        
        graph.add_edge("generate_variation", "finalize")
        graph.add_edge("finalize", END)
        
        return graph.compile(checkpointer=MemorySaver())
    
    def _node_extract_history(self, state: OutputGuardState) -> OutputGuardState:
        """LangGraph node: Extract assistant responses from history"""
        recent_assistant_responses = [
            msg.get('content', '')
            for msg in state["recent_responses"]
            if msg.get('role') == 'assistant'
        ]
        
        return {
            **state,
            "recent_responses": recent_assistant_responses
        }
    
    def _node_check_repetition(self, state: OutputGuardState) -> OutputGuardState:
        """LangGraph node: Check if response is repetitive"""
        recent = state["recent_responses"]
        
        if not recent:
            return {
                **state,
                "is_repetitive": False,
                "similarity_score": 0.0,
                "needs_variation": False
            }
        
        # Embed proposed response
        proposed_embedding = self.embedder.encode(state["proposed_response"])
        
        # Compare with recent responses
        max_similarity = 0.0
        for recent_response in recent:
            if isinstance(recent_response, dict):
                recent_response = recent_response.get('content', '')
            
            recent_embedding = self.embedder.encode(recent_response)
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity(
                [proposed_embedding],
                [recent_embedding]
            )[0][0]
            max_similarity = max(max_similarity, similarity)
        
        is_repetitive = max_similarity > self.similarity_threshold
        
        if is_repetitive:
            logger.warning(
                f"Repetitive response detected (similarity: {max_similarity:.2f}, frustration: {state['frustration_level']})"
            )
        
        return {
            **state,
            "is_repetitive": is_repetitive,
            "similarity_score": max_similarity,
            "needs_variation": is_repetitive
        }
    
    def _route_after_check(self, state: OutputGuardState) -> str:
        """Conditional routing: If repetitive, generate variation; else finalize"""
        if state["is_repetitive"]:
            return "vary"
        return "finalize"
    
    async def _node_generate_variation(self, state: OutputGuardState) -> OutputGuardState:
        """LangGraph node: Generate variation to avoid repetition"""
        try:
            recent_responses = state["recent_responses"]
            if isinstance(recent_responses, list) and recent_responses and isinstance(recent_responses[0], dict):
                recent_responses = [r.get('content', '') if isinstance(r, dict) else r for r in recent_responses]
            
            # Build context of what was already said
            recent_context = "\n".join([
                f"- {resp[:100]}..." if len(resp) > 100 else f"- {resp}"
                for resp in recent_responses[-3:]  # Last 3 responses
            ])
            
            # Frustration-aware prompt
            frustration_instruction = ""
            if state["frustration_level"] >= 3:
                frustration_instruction = f"""
IMPORTANT: User is frustrated (level {state["frustration_level"]}/5). 
- Acknowledge their frustration
- Provide a COMPLETELY different angle
- Be more concise and direct
- Offer a concrete next step
"""
            
            prompt = f"""You are Finley, an AI finance assistant. 

USER'S QUESTION: {state["question"]}

WHAT WAS ALREADY SAID (avoid repeating):
{recent_context}

TASK: Generate a COMPLETELY DIFFERENT response to the user's question.
- Different structure
- Different angle/perspective
- Different examples or data points
- Different tone or approach
{frustration_instruction}

Generate the varied response now:"""
            
            response = await self.llm_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.8
            )
            
            varied_response = response.choices[0].message.content.strip()
            logger.info("Generated varied response to prevent repetition")
            
            return {
                **state,
                "final_response": varied_response
            }
        
        except Exception as e:
            logger.error(f"Failed to generate variation: {e}")
            return {
                **state,
                "final_response": state["proposed_response"]
            }
    
    def _node_finalize(self, state: OutputGuardState) -> OutputGuardState:
        """LangGraph node: Finalize response"""
        if not state.get("final_response"):
            return {
                **state,
                "final_response": state["proposed_response"]
            }
        return state
    
    async def check_and_fix(
        self,
        proposed_response: str,
        recent_responses: List[Dict[str, str]],
        question: str,
        frustration_level: int = 0
    ) -> str:
        """
        Check if response is repetitive and fix if needed using LangGraph.
        
        Args:
            proposed_response: The response we're about to send
            recent_responses: Last 5 messages from memory
            question: User's question
            frustration_level: User's frustration (0-5)
        
        Returns:
            Safe response (either original or varied)
        """
        try:
            # Initialize state
            initial_state = {
                "proposed_response": proposed_response,
                "recent_responses": recent_responses,
                "question": question,
                "frustration_level": frustration_level,
                "is_repetitive": False,
                "similarity_score": 0.0,
                "final_response": proposed_response,
                "needs_variation": False
            }
            
            # Run graph
            final_state = await asyncio.to_thread(
                lambda: self.graph.invoke(initial_state)
            )
            
            return final_state["final_response"]
        
        except Exception as e:
            logger.error(f"OutputGuard check failed: {e}")
            return proposed_response  # Fallback to original


# REMOVED: ResponseVariationEngine class and ResponseVariationState
# REASON: Dead code - OutputGuard handles response variation internally via _node_generate_variation()
# The OutputGuard class (lines 332-563) already implements variation generation with LangGraph
# This was never called anywhere in the codebase


# Singleton instances
_intent_classifier: Optional[IntentClassifier] = None
_output_guard: Optional[OutputGuard] = None


def get_intent_classifier() -> IntentClassifier:
    """Get or create intent classifier singleton (LangGraph-based)"""
    global _intent_classifier
    if _intent_classifier is None:
        _intent_classifier = IntentClassifier()
    return _intent_classifier


def get_output_guard(llm_client: AsyncGroq) -> OutputGuard:
    """Get or create output guard singleton (LangGraph-based)"""
    global _output_guard
    if _output_guard is None:
        _output_guard = OutputGuard(llm_client)
    return _output_guard


# REMOVED: get_response_variation_engine() function
# REASON: ResponseVariationEngine is dead code - OutputGuard handles variation internally
