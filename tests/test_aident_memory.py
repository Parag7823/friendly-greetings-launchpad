"""
Test Suite for Aident Memory Manager
====================================

Tests for LangChain ConversationSummaryBufferMemory integration.
Verifies per-user isolation, persistence, and concurrent access.

Author: Aident Team
Version: 1.0.0
"""

import pytest
import asyncio
from datetime import datetime
from aident_cfo_brain.aident_memory_manager import AidentMemoryManager


class TestAidentMemoryManager:
    """Test suite for AidentMemoryManager"""
    
    @pytest.mark.asyncio
    async def test_memory_initialization(self):
        """Test memory manager initialization"""
        memory = AidentMemoryManager(user_id="test_user_1")
        assert memory.user_id == "test_user_1"
        assert memory.memory is not None
        assert memory.max_token_limit == 2000
    
    @pytest.mark.asyncio
    async def test_add_single_message(self):
        """Test adding a single message to memory"""
        memory = AidentMemoryManager(user_id="test_user_2")
        
        success = await memory.add_message(
            user_message="What is my revenue?",
            assistant_response="Your revenue is $100,000 in Q1."
        )
        
        assert success is True
        messages = memory.get_messages()
        assert len(messages) >= 1
    
    @pytest.mark.asyncio
    async def test_add_multiple_messages(self):
        """Test adding multiple messages (multi-turn conversation)"""
        memory = AidentMemoryManager(user_id="test_user_3")
        
        # Add 5 messages
        for i in range(5):
            success = await memory.add_message(
                user_message=f"Question {i+1}",
                assistant_response=f"Answer {i+1}"
            )
            assert success is True
        
        messages = memory.get_messages()
        assert len(messages) >= 5
    
    @pytest.mark.asyncio
    async def test_get_context(self):
        """Test retrieving conversation context"""
        memory = AidentMemoryManager(user_id="test_user_4")
        
        await memory.add_message(
            user_message="Why did revenue drop?",
            assistant_response="Revenue dropped due to seasonal factors."
        )
        
        context = memory.get_context()
        assert len(context) > 0
        assert "revenue" in context.lower() or "drop" in context.lower()
    
    @pytest.mark.asyncio
    async def test_memory_stats(self):
        """Test memory statistics"""
        memory = AidentMemoryManager(user_id="test_user_5")
        
        await memory.add_message(
            user_message="Test question",
            assistant_response="Test answer"
        )
        
        stats = await memory.get_memory_stats()
        assert stats["user_id"] == "test_user_5"
        assert stats["message_count"] >= 1
        assert stats["buffer_size"] > 0
        assert "max_token_limit" in stats
    
    @pytest.mark.asyncio
    async def test_per_user_isolation(self):
        """Test that different users have isolated memory"""
        memory_user_a = AidentMemoryManager(user_id="user_a")
        memory_user_b = AidentMemoryManager(user_id="user_b")
        
        # Add different messages to each user
        await memory_user_a.add_message(
            user_message="User A question",
            assistant_response="User A answer"
        )
        
        await memory_user_b.add_message(
            user_message="User B question",
            assistant_response="User B answer"
        )
        
        # Verify isolation
        context_a = memory_user_a.get_context()
        context_b = memory_user_b.get_context()
        
        assert "User A" in context_a
        assert "User B" in context_b
        assert "User B" not in context_a
        assert "User A" not in context_b
    
    @pytest.mark.asyncio
    async def test_clear_memory(self):
        """Test clearing memory"""
        memory = AidentMemoryManager(user_id="test_user_6")
        
        # Add messages
        await memory.add_message(
            user_message="Test question",
            assistant_response="Test answer"
        )
        
        # Clear memory
        success = await memory.clear_memory()
        assert success is True
        
        # Verify cleared
        messages = memory.get_messages()
        assert len(messages) == 0
    
    @pytest.mark.asyncio
    async def test_concurrent_users(self):
        """Test concurrent access from multiple users"""
        async def create_and_use_memory(user_id: str, message_count: int):
            memory = AidentMemoryManager(user_id=user_id)
            
            for i in range(message_count):
                await memory.add_message(
                    user_message=f"User {user_id} - Question {i+1}",
                    assistant_response=f"User {user_id} - Answer {i+1}"
                )
            
            stats = await memory.get_memory_stats()
            return stats
        
        # Simulate 5 concurrent users
        tasks = [
            create_and_use_memory(f"concurrent_user_{i}", 3)
            for i in range(5)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Verify all users completed successfully
        assert len(results) == 5
        for stats in results:
            assert stats["message_count"] >= 3
    
    @pytest.mark.asyncio
    async def test_long_conversation_summarization(self):
        """Test auto-summarization on long conversations"""
        # Create memory with small token limit to trigger summarization
        memory = AidentMemoryManager(
            user_id="test_user_7",
            max_token_limit=500  # Small limit to trigger summarization
        )
        
        # Add many messages
        for i in range(10):
            await memory.add_message(
                user_message=f"Question about financial metrics {i}: What is the impact of changing vendor payment terms?",
                assistant_response=f"Answer {i}: The impact depends on cash flow timing, discount rates, and working capital requirements."
            )
        
        stats = await memory.get_memory_stats()
        
        # Verify memory was created and summarization occurred
        assert stats["message_count"] > 0
        assert stats["buffer_size"] > 0
        # Buffer should be summarized (not all messages stored as-is)
        assert stats["buffer_tokens"] < stats["message_count"] * 50  # Rough estimate
    
    @pytest.mark.asyncio
    async def test_memory_key_format(self):
        """Test that memory keys follow correct format"""
        user_id = "test_user_8"
        memory = AidentMemoryManager(user_id=user_id)
        
        assert memory.memory_key == f"memory:{user_id}"
        assert memory.lock_key == f"memory_lock:{user_id}"
    
    @pytest.mark.asyncio
    async def test_get_messages_format(self):
        """Test that messages are returned in correct format"""
        memory = AidentMemoryManager(user_id="test_user_9")
        
        await memory.add_message(
            user_message="What is revenue?",
            assistant_response="Revenue is $100,000."
        )
        
        messages = memory.get_messages()
        
        # Verify format
        for msg in messages:
            assert "role" in msg
            assert "content" in msg
            assert msg["role"] in ["user", "assistant"]
            assert len(msg["content"]) > 0


class TestMemoryIntegration:
    """Integration tests for memory with orchestrator"""
    
    @pytest.mark.asyncio
    async def test_memory_context_in_classification(self):
        """Test that memory context is used in question classification"""
        memory = AidentMemoryManager(user_id="integration_test_1")
        
        # Simulate conversation history
        await memory.add_message(
            user_message="Why did revenue drop in Q2?",
            assistant_response="Revenue dropped due to seasonal factors and vendor delays."
        )
        
        # Get context for classification
        context = memory.get_context()
        
        # Verify context contains relevant information
        assert len(context) > 0
        # Context should mention revenue or Q2
        assert any(word in context.lower() for word in ["revenue", "q2", "seasonal"])
    
    @pytest.mark.asyncio
    async def test_memory_persistence_simulation(self):
        """Test memory persistence across sessions"""
        user_id = "persistence_test_1"
        
        # Session 1: Add messages
        memory1 = AidentMemoryManager(user_id=user_id)
        await memory1.add_message(
            user_message="First question",
            assistant_response="First answer"
        )
        
        # Session 2: Load memory
        memory2 = AidentMemoryManager(user_id=user_id)
        await memory2.load_memory()
        
        # Verify memory was loaded
        messages = memory2.get_messages()
        assert len(messages) > 0


class TestMemoryEdgeCases:
    """Test edge cases and error handling"""
    
    @pytest.mark.asyncio
    async def test_empty_memory(self):
        """Test behavior with empty memory"""
        memory = AidentMemoryManager(user_id="edge_case_1")
        
        context = memory.get_context()
        messages = memory.get_messages()
        
        assert context == "" or len(context) == 0
        assert len(messages) == 0
    
    @pytest.mark.asyncio
    async def test_very_long_message(self):
        """Test handling of very long messages"""
        memory = AidentMemoryManager(user_id="edge_case_2")
        
        long_message = "x" * 10000  # 10k character message
        
        success = await memory.add_message(
            user_message=long_message,
            assistant_response="Response to long message"
        )
        
        assert success is True
    
    @pytest.mark.asyncio
    async def test_special_characters(self):
        """Test handling of special characters"""
        memory = AidentMemoryManager(user_id="edge_case_3")
        
        special_message = "Question with émojis 🚀 and spëcial çhars: @#$%^&*()"
        
        success = await memory.add_message(
            user_message=special_message,
            assistant_response="Answer with spëcial çhars"
        )
        
        assert success is True
        context = memory.get_context()
        assert len(context) > 0
    
    @pytest.mark.asyncio
    async def test_unicode_handling(self):
        """Test handling of unicode characters"""
        memory = AidentMemoryManager(user_id="edge_case_4")
        
        unicode_message = "Question in Chinese: 你好世界 and Arabic: مرحبا بالعالم"
        
        success = await memory.add_message(
            user_message=unicode_message,
            assistant_response="Response with unicode"
        )
        
        assert success is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
