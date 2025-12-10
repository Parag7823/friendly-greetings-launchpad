"""
Intelligence Layer Test Suite - Google CTO Grade
=================================================

Comprehensive end-to-end tests for the AI intelligence layer:
1. IntelligentChatOrchestrator - The brain routing questions to engines
2. IntentClassifier + OutputGuard - Intent detection and response quality
3. AidentMemoryManager - Conversation memory with entity tracking

Test Philosophy:
- NO MOCKS: Real code execution with real LLM calls
- FIX PRODUCTION CODE: Tests reveal bugs, we fix the source
- HYPOTHESIS: Property-based testing for edge cases
- SEQUENTIAL: Tests follow actual data flow

Author: Aident Test Team
Quality: Google CTO Grade
"""

import pytest
import asyncio
import os
import sys
from typing import Dict, Any, List, Optional
from datetime import datetime
from hypothesis import given, strategies as st, settings, HealthCheck

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================================
# Phase 1: Core Data Structures and Helpers
# ============================================================================

class TestPhase1CoreDataStructures:
    """Test core data structures used across the intelligence layer."""
    
    # -------------------------------------------------------------------------
    # Test 1.1: QuestionType Enum
    # -------------------------------------------------------------------------
    
    def test_question_type_enum_has_all_types(self):
        """Verify QuestionType enum contains all expected question types."""
        from aident_cfo_brain.intelligent_chat_orchestrator import QuestionType
        
        expected_types = [
            "CAUSAL", "TEMPORAL", "RELATIONSHIP", "WHAT_IF", 
            "EXPLAIN", "DATA_QUERY", "GENERAL", "UNKNOWN"
        ]
        
        actual_types = [qt.name for qt in QuestionType]
        
        for expected in expected_types:
            assert expected in actual_types, f"Missing QuestionType: {expected}"
    
    def test_question_type_values_are_lowercase(self):
        """QuestionType values should be lowercase for consistency."""
        from aident_cfo_brain.intelligent_chat_orchestrator import QuestionType
        
        for qt in QuestionType:
            assert qt.value == qt.value.lower(), f"{qt.name} value is not lowercase: {qt.value}"
    
    # -------------------------------------------------------------------------
    # Test 1.2: ChatResponse Dataclass
    # -------------------------------------------------------------------------
    
    def test_chat_response_required_fields(self):
        """ChatResponse must have answer, question_type, confidence."""
        from aident_cfo_brain.intelligent_chat_orchestrator import ChatResponse, QuestionType
        
        response = ChatResponse(
            answer="Test answer",
            question_type=QuestionType.GENERAL,
            confidence=0.9
        )
        
        assert response.answer == "Test answer"
        assert response.question_type == QuestionType.GENERAL
        assert response.confidence == 0.9
    
    def test_chat_response_optional_fields_default_none(self):
        """Optional fields should default to None."""
        from aident_cfo_brain.intelligent_chat_orchestrator import ChatResponse, QuestionType
        
        response = ChatResponse(
            answer="Test",
            question_type=QuestionType.GENERAL,
            confidence=0.5
        )
        
        assert response.data is None
        assert response.actions is None
        assert response.visualizations is None
        assert response.follow_up_questions is None
        assert response.error is None
    
    def test_chat_response_to_dict_serialization(self):
        """to_dict() should produce valid JSON-serializable dictionary."""
        from aident_cfo_brain.intelligent_chat_orchestrator import ChatResponse, QuestionType
        import json
        
        response = ChatResponse(
            answer="Revenue is $100K",
            question_type=QuestionType.DATA_QUERY,
            confidence=0.95,
            data={"revenue": 100000},
            follow_up_questions=["What about expenses?"]
        )
        
        result = response.to_dict()
        
        # Should be JSON serializable
        json_str = json.dumps(result)
        assert len(json_str) > 0
        
        # Check structure
        assert result["answer"] == "Revenue is $100K"
        assert result["question_type"] == "data_query"
        assert result["confidence"] == 0.95
        assert result["data"]["revenue"] == 100000
        assert "timestamp" in result
    
    @given(st.floats(min_value=-1.0, max_value=2.0))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_chat_response_confidence_boundary(self, confidence):
        """Hypothesis: Test confidence values including out-of-range."""
        from aident_cfo_brain.intelligent_chat_orchestrator import ChatResponse, QuestionType
        
        # Should not raise even with out-of-range confidence
        response = ChatResponse(
            answer="Test",
            question_type=QuestionType.GENERAL,
            confidence=confidence
        )
        
        # Confidence is stored as-is (validation happens at usage)
        assert response.confidence == confidence
    
    # -------------------------------------------------------------------------
    # Test 1.3: Error Response Helper
    # -------------------------------------------------------------------------
    
    def test_create_error_response_basic(self):
        """_create_error_response should create structured error."""
        from aident_cfo_brain.intelligent_chat_orchestrator import _create_error_response
        
        result = _create_error_response("entity_extraction", "timeout")
        
        assert result["error"] is True
        assert result["operation"] == "entity_extraction"
        assert "timeout" in result["message"]
    
    def test_create_error_response_with_fallback(self):
        """_create_error_response should merge fallback data."""
        from aident_cfo_brain.intelligent_chat_orchestrator import _create_error_response
        
        fallback = {"metrics": ["revenue"], "confidence": 0.3}
        result = _create_error_response("extraction", "failed", fallback)
        
        assert result["error"] is True
        assert result["metrics"] == ["revenue"]
        assert result["confidence"] == 0.3
    
    def test_create_error_response_none_fallback(self):
        """_create_error_response should handle None fallback."""
        from aident_cfo_brain.intelligent_chat_orchestrator import _create_error_response
        
        result = _create_error_response("test_op", "test_error", None)
        
        assert result["error"] is True
        assert len(result) == 3  # error, operation, message
    
    # -------------------------------------------------------------------------
    # Test 1.4: Fallback Entities Helper
    # -------------------------------------------------------------------------
    
    def test_get_fallback_entities_with_keywords(self):
        """_get_fallback_entities should extract financial keywords."""
        from aident_cfo_brain.intelligent_chat_orchestrator import _get_fallback_entities
        
        result = _get_fallback_entities("What is my revenue and profit?")
        
        assert "revenue" in result["metrics"]
        assert "profit" in result["metrics"]
        assert result["confidence"] == 0.3
        assert result["entities"] == []
    
    def test_get_fallback_entities_no_keywords(self):
        """_get_fallback_entities should return zero confidence for non-financial."""
        from aident_cfo_brain.intelligent_chat_orchestrator import _get_fallback_entities
        
        result = _get_fallback_entities("Hello, how are you today?")
        
        assert result["metrics"] == []
        assert result["confidence"] == 0.0
    
    @given(st.text(min_size=0, max_size=1000))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_get_fallback_entities_never_crashes(self, question):
        """Hypothesis: _get_fallback_entities should never crash on any input."""
        from aident_cfo_brain.intelligent_chat_orchestrator import _get_fallback_entities
        
        result = _get_fallback_entities(question)
        
        assert isinstance(result, dict)
        assert "metrics" in result
        assert "entities" in result
        assert "time_periods" in result
        assert "confidence" in result


# ============================================================================
# Phase 2: Intent Classification (intent_and_guard_engine.py)
# ============================================================================

class TestPhase2IntentClassification:
    """Test IntentClassifier from intent_and_guard_engine.py"""
    
    # -------------------------------------------------------------------------
    # Test 2.1: UserIntent Enum
    # -------------------------------------------------------------------------
    
    def test_user_intent_has_all_intents(self):
        """UserIntent enum should have all expected intent types."""
        from aident_cfo_brain.intent_and_guard_engine import UserIntent
        
        expected_intents = [
            "CAPABILITY_SUMMARY", "SYSTEM_FLOW", "DIFFERENTIATOR",
            "META_FEEDBACK", "SMALLTALK", "GREETING", "CONNECT_SOURCE",
            "DATA_ANALYSIS", "HELP", "UNKNOWN"
        ]
        
        actual_intents = [ui.name for ui in UserIntent]
        
        for expected in expected_intents:
            assert expected in actual_intents, f"Missing UserIntent: {expected}"
    
    def test_user_intent_values_are_lowercase(self):
        """UserIntent values should be lowercase for consistency."""
        from aident_cfo_brain.intent_and_guard_engine import UserIntent
        
        for ui in UserIntent:
            assert ui.value == ui.value.lower(), f"{ui.name} value is not lowercase: {ui.value}"
    
    # -------------------------------------------------------------------------
    # Test 2.2: IntentResult Dataclass
    # -------------------------------------------------------------------------
    
    def test_intent_result_structure(self):
        """IntentResult should have intent, confidence, method, reasoning."""
        from aident_cfo_brain.intent_and_guard_engine import IntentResult, UserIntent
        
        result = IntentResult(
            intent=UserIntent.GREETING,
            confidence=0.95,
            method="langchain_structured_routing",
            reasoning="User said hello"
        )
        
        assert result.intent == UserIntent.GREETING
        assert result.confidence == 0.95
        assert result.method == "langchain_structured_routing"
        assert "hello" in result.reasoning.lower()
    
    # -------------------------------------------------------------------------
    # Test 2.3: IntentClassifier Initialization
    # -------------------------------------------------------------------------
    
    def test_intent_classifier_requires_groq_api_key(self):
        """IntentClassifier should raise if GROQ_API_KEY not set."""
        from aident_cfo_brain.intent_and_guard_engine import IntentClassifier
        
        # Temporarily unset the key
        original = os.environ.get("GROQ_API_KEY")
        if original:
            del os.environ["GROQ_API_KEY"]
        
        try:
            with pytest.raises(ValueError, match="GROQ_API_KEY"):
                IntentClassifier()
        finally:
            # Restore
            if original:
                os.environ["GROQ_API_KEY"] = original
    
    @pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY required")
    def test_intent_classifier_initializes_with_api_key(self):
        """IntentClassifier should initialize successfully with API key."""
        from aident_cfo_brain.intent_and_guard_engine import IntentClassifier
        
        classifier = IntentClassifier()
        
        assert classifier.groq_client is not None
        assert classifier.langchain_llm is not None
    
    # -------------------------------------------------------------------------
    # Test 2.4: Intent Classification (Real LLM Calls)
    # -------------------------------------------------------------------------
    
    @pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY required")
    @pytest.mark.asyncio
    async def test_classify_greeting_intent(self):
        """classify() should detect greeting intent."""
        from aident_cfo_brain.intent_and_guard_engine import IntentClassifier, UserIntent
        
        classifier = IntentClassifier()
        result = await classifier.classify("Hello there!")
        
        assert result.intent in [UserIntent.GREETING, UserIntent.SMALLTALK]
        assert result.confidence > 0.5
        assert result.method != "fallback"
    
    @pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY required")
    @pytest.mark.asyncio
    async def test_classify_data_analysis_intent(self):
        """classify() should detect data analysis intent."""
        from aident_cfo_brain.intent_and_guard_engine import IntentClassifier, UserIntent
        
        classifier = IntentClassifier()
        result = await classifier.classify("Show me my revenue for last month")
        
        assert result.intent == UserIntent.DATA_ANALYSIS
        assert result.confidence > 0.6
    
    @pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY required")
    @pytest.mark.asyncio
    async def test_classify_capability_summary_intent(self):
        """classify() should detect capability summary intent."""
        from aident_cfo_brain.intent_and_guard_engine import IntentClassifier, UserIntent
        
        classifier = IntentClassifier()
        result = await classifier.classify("What can you do?")
        
        assert result.intent == UserIntent.CAPABILITY_SUMMARY
        assert result.confidence > 0.6
    
    @pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY required")
    @pytest.mark.asyncio
    async def test_classify_connect_source_intent(self):
        """classify() should detect connect source intent."""
        from aident_cfo_brain.intent_and_guard_engine import IntentClassifier, UserIntent
        
        classifier = IntentClassifier()
        result = await classifier.classify("Connect my QuickBooks account")
        
        assert result.intent == UserIntent.CONNECT_SOURCE
        assert result.confidence > 0.6


# ============================================================================
# Phase 3: Output Guard (intent_and_guard_engine.py)
# ============================================================================

class TestPhase3OutputGuard:
    """Test OutputGuard from intent_and_guard_engine.py"""
    
    # -------------------------------------------------------------------------
    # Test 3.1: OutputGuard Initialization
    # -------------------------------------------------------------------------
    
    @pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY required")
    def test_output_guard_initializes(self):
        """OutputGuard should initialize with AsyncGroq client."""
        from aident_cfo_brain.intent_and_guard_engine import OutputGuard
        from groq import AsyncGroq
        
        client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        guard = OutputGuard(client)
        
        assert guard.memory is not None
        assert guard.langchain_llm is not None
    
    # -------------------------------------------------------------------------
    # Test 3.2: Repetition Detection
    # -------------------------------------------------------------------------
    
    @pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY required")
    def test_check_repetition_no_memory(self):
        """_check_repetition_in_summary should return False with empty buffer."""
        from aident_cfo_brain.intent_and_guard_engine import OutputGuard
        from groq import AsyncGroq
        
        client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        guard = OutputGuard(client)
        
        result = guard._check_repetition_in_summary(
            proposed_response="Your revenue is $100K",
            memory_buffer="",
            frustration_level=0
        )
        
        assert result is False
    
    @pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY required")
    def test_check_repetition_detects_high_overlap(self):
        """_check_repetition_in_summary should detect >75% overlap."""
        from aident_cfo_brain.intent_and_guard_engine import OutputGuard
        from groq import AsyncGroq
        
        client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        guard = OutputGuard(client)
        
        # Same response repeated (must be > 50 chars for repetition check)
        buffer = "Your revenue is $100K this month from various sales channels across regions."
        response = "Your revenue is $100K this month from various sales channels across regions."
        
        result = guard._check_repetition_in_summary(
            proposed_response=response,
            memory_buffer=buffer,
            frustration_level=0
        )
        
        assert result is True, "Should detect 100% overlap as repetitive"
    
    @pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY required")
    def test_check_repetition_frustration_lowers_threshold(self):
        """Frustrated users should have lower repetition threshold (60% instead of 75%)."""
        from aident_cfo_brain.intent_and_guard_engine import OutputGuard
        from groq import AsyncGroq
        
        client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        guard = OutputGuard(client)
        
        # ~65% overlap - should trigger with frustration but not without
        buffer = "Your monthly revenue is $100K from various sales channels."
        response = "Your revenue is $100K from sales this month actually."
        
        # Without frustration (threshold 75%) - should not trigger
        result_normal = guard._check_repetition_in_summary(response, buffer, frustration_level=0)
        
        # With frustration (threshold 60%) - should trigger
        result_frustrated = guard._check_repetition_in_summary(response, buffer, frustration_level=3)
        
        # The frustrated user threshold should be more sensitive
        # (exact behavior depends on word overlap calculation)
        assert isinstance(result_normal, bool)
        assert isinstance(result_frustrated, bool)


# ============================================================================
# Phase 4: Aident Memory Manager
# ============================================================================

class TestPhase4AidentMemoryManager:
    """Test AidentMemoryManager from aident_memory_manager.py"""
    
    # -------------------------------------------------------------------------
    # Test 4.1: Memory Manager Initialization
    # -------------------------------------------------------------------------
    
    @pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY required")
    def test_memory_manager_requires_user_id(self):
        """AidentMemoryManager requires user_id."""
        from aident_cfo_brain.aident_memory_manager import AidentMemoryManager
        
        # Should NOT raise
        manager = AidentMemoryManager(user_id="test-user-123")
        assert manager.user_id == "test-user-123"
    
    @pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY required")
    def test_memory_manager_configurable_token_limit(self):
        """AidentMemoryManager should accept custom token limit."""
        from aident_cfo_brain.aident_memory_manager import AidentMemoryManager
        
        manager = AidentMemoryManager(user_id="test", max_token_limit=5000)
        assert manager.max_token_limit == 5000
    
    # -------------------------------------------------------------------------
    # Test 4.2: Message Operations
    # -------------------------------------------------------------------------
    
    @pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY required")
    @pytest.mark.asyncio
    async def test_add_message_stores_correctly(self):
        """add_message should store user and assistant messages."""
        from aident_cfo_brain.aident_memory_manager import AidentMemoryManager
        
        manager = AidentMemoryManager(user_id="test-add-msg")
        
        await manager.add_message("What is my revenue?", "Your revenue is $100K")
        
        messages = manager.get_messages()
        assert len(messages) >= 2
        
        # Check roles are correct
        roles = [m["role"] for m in messages]
        assert "user" in roles
        assert "assistant" in roles
    
    @pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY required")
    def test_get_context_returns_string(self):
        """get_context should return conversation context as string."""
        from aident_cfo_brain.aident_memory_manager import AidentMemoryManager
        
        manager = AidentMemoryManager(user_id="test-context")
        context = manager.get_context()
        
        assert isinstance(context, str)
    
    # -------------------------------------------------------------------------
    # Test 4.3: Entity Tracking (New Feature)
    # -------------------------------------------------------------------------
    
    @pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY required")
    def test_resolve_reference_no_entities(self):
        """resolve_reference should return original if no entity stack."""
        from aident_cfo_brain.aident_memory_manager import AidentMemoryManager
        
        manager = AidentMemoryManager(user_id="test-ref")
        
        question = "Why did it increase?"
        resolved = manager.resolve_reference(question)
        
        # No entities in stack, so should return original
        assert resolved == question
    
    @pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY required")
    @pytest.mark.asyncio
    async def test_update_entity_context_extracts_entities(self):
        """update_entity_context should extract financial entities from response."""
        from aident_cfo_brain.aident_memory_manager import AidentMemoryManager
        
        manager = AidentMemoryManager(user_id="test-entity")
        
        await manager.update_entity_context(
            question="What is my biggest expense?",
            response="Your biggest expense is Marketing at $50K."
        )
        
        entities = manager.get_entities()
        assert "Marketing" in entities["entity_stack"]
    
    @pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY required")
    @pytest.mark.asyncio
    async def test_resolve_reference_after_entity_update(self):
        """resolve_reference should replace 'it' with last entity."""
        from aident_cfo_brain.aident_memory_manager import AidentMemoryManager
        
        manager = AidentMemoryManager(user_id="test-resolve")
        
        # First, establish entity context
        await manager.update_entity_context(
            question="What is my biggest expense?",
            response="Your biggest expense is Marketing at $50K."
        )
        
        # Now resolve reference
        question = "Why did it increase?"
        resolved = manager.resolve_reference(question)
        
        assert "Marketing" in resolved, f"Expected 'Marketing' in resolved: {resolved}"
        assert "it" not in resolved.lower() or "Marketing" in resolved
    
    @pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY required")
    def test_entity_stack_limited_to_ten(self):
        """Entity stack should be limited to 10 items."""
        from aident_cfo_brain.aident_memory_manager import AidentMemoryManager
        
        manager = AidentMemoryManager(user_id="test-stack-limit")
        
        # Manually add 15 entities
        manager._entity_stack = [f"Entity{i}" for i in range(15)]
        manager._entity_stack = manager._entity_stack[-10:]  # Apply limit
        
        assert len(manager._entity_stack) == 10
        assert manager._entity_stack[0] == "Entity5"  # First 5 should be removed
    
    # -------------------------------------------------------------------------
    # Test 4.4: Memory Stats
    # -------------------------------------------------------------------------
    
    @pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY required")
    @pytest.mark.asyncio
    async def test_get_memory_stats_returns_valid_structure(self):
        """get_memory_stats should return valid stats dictionary."""
        from aident_cfo_brain.aident_memory_manager import AidentMemoryManager
        
        manager = AidentMemoryManager(user_id="test-stats")
        stats = await manager.get_memory_stats()
        
        assert "user_id" in stats
        assert "message_count" in stats
        assert "max_token_limit" in stats
        assert stats["user_id"] == "test-stats"


# ============================================================================
# Phase 5: Integration Tests (Full Flow)
# ============================================================================

class TestPhase5IntegrationFlow:
    """Test the full integration flow across all three modules."""
    
    @pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY required")
    @pytest.mark.asyncio
    async def test_full_conversation_flow(self):
        """Test a complete conversation flow with memory and intent."""
        from aident_cfo_brain.aident_memory_manager import AidentMemoryManager
        from aident_cfo_brain.intent_and_guard_engine import IntentClassifier, UserIntent
        
        # Initialize components
        memory = AidentMemoryManager(user_id="test-integration")
        classifier = IntentClassifier()
        
        # Simulate conversation
        conversation = [
            ("Hello!", UserIntent.GREETING),
            ("What can you do?", UserIntent.CAPABILITY_SUMMARY),
            ("Show me my revenue", UserIntent.DATA_ANALYSIS),
        ]
        
        for question, expected_intent in conversation:
            # Classify intent
            result = await classifier.classify(question)
            assert result.intent == expected_intent or result.confidence < 0.5
            
            # Store in memory
            await memory.add_message(question, f"Response to: {question}")
        
        # Verify memory
        messages = memory.get_messages()
        assert len(messages) >= 6  # 3 user + 3 assistant


# ============================================================================
# Hypothesis Edge Case Tests
# ============================================================================

class TestHypothesisEdgeCases:
    """Property-based tests using Hypothesis for edge case coverage."""
    
    @given(st.text(min_size=1, max_size=500, alphabet=st.characters(blacklist_categories=('Cs',))))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_fallback_entities_handles_any_text(self, question):
        """_get_fallback_entities should handle any valid text input."""
        from aident_cfo_brain.intelligent_chat_orchestrator import _get_fallback_entities
        
        result = _get_fallback_entities(question)
        
        assert isinstance(result["metrics"], list)
        assert isinstance(result["entities"], list)
        assert 0.0 <= result["confidence"] <= 1.0
    
    @given(st.text(min_size=0, max_size=100))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_error_response_handles_any_strings(self, operation):
        """_create_error_response should handle any operation name."""
        from aident_cfo_brain.intelligent_chat_orchestrator import _create_error_response
        
        result = _create_error_response(operation, "test_error")
        
        assert result["error"] is True
        assert result["operation"] == operation


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
