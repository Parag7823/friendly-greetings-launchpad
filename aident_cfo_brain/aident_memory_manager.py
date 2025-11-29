"""
Aident Memory Manager - Production-Ready Conversational Memory
==============================================================

Integrates LangChain's ConversationSummaryBufferMemory with Redis persistence
to maintain deep context across multi-turn conversations for 50+ concurrent users.

Features:
- Per-user isolated memory instances (no cross-user contamination)
- Auto-summarization of older messages to prevent token explosion
- Redis-backed persistence for conversation continuity
- Async-safe operations for concurrent user handling
- Configurable token limits and summary strategies

Author: Aident Team
Version: 1.0.0
Date: 2025-01-26
"""

import asyncio
import json
import pickle
import structlog
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from langchain.memory import ConversationSummaryBufferMemory
from langchain_groq import ChatGroq
import redis.asyncio as redis

logger = structlog.get_logger(__name__)


class AidentMemoryManager:
    """
    Production-grade memory manager for Aident conversations.
    
    Each user gets their own isolated memory instance that:
    - Retains last 10-20 messages with full detail
    - Auto-summarizes older messages to prevent context overflow
    - Persists to Redis for conversation continuity
    - Supports 50+ concurrent users without conflicts
    """
    
    def __init__(
        self,
        user_id: str,
        redis_url: Optional[str] = None,
        max_token_limit: int = 2000,
        groq_api_key: Optional[str] = None
    ):
        """
        Initialize memory manager for a specific user.
        
        Args:
            user_id: Unique user identifier (scopes memory to this user)
            redis_url: Redis connection URL (defaults to REDIS_URL env var)
            max_token_limit: Max tokens before auto-summarization (default 2000)
            groq_api_key: Groq API key for summary generation (defaults to env var)
        """
        self.user_id = user_id
        self.max_token_limit = max_token_limit
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.memory_key = f"memory:{user_id}"
        self.lock_key = f"memory_lock:{user_id}"
        
        # FIX #18: Conversation state tracking for repetition detection
        self.conversation_state = {
            'topics_discussed': set(),
            'response_types_used': [],
            'phrases_used': set(),
            'frustration_level': 0,
            'last_response_type': None
        }
        
        # Initialize LLM for summary generation
        import os
        api_key = groq_api_key or os.getenv('GROQ_API_KEY')
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is required for memory summarization")
        
        self.summarizer = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            api_key=api_key
        )
        
        # Initialize LangChain memory
        self.memory = ConversationSummaryBufferMemory(
            llm=self.summarizer,
            max_token_limit=max_token_limit,
            memory_key="chat_history",
            ai_prefix="Aident",
            human_prefix="User",
            return_messages=True
        )
        
        logger.info(
            "memory_manager_initialized",
            user_id=user_id,
            max_token_limit=max_token_limit
        )
    
    async def initialize_redis(self):
        """Initialize async Redis connection (call this in async context)"""
        if self.redis_client is None:
            import os
            url = self.redis_url or os.getenv('REDIS_URL') or os.getenv('ARQ_REDIS_URL')
            if not url:
                logger.warning("No Redis URL provided, memory will not persist across sessions")
                return
            
            try:
                self.redis_client = await redis.from_url(url, decode_responses=False)
                await self.redis_client.ping()
                logger.info("redis_connection_established", user_id=self.user_id)
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self.redis_client = None
    
    async def load_memory(self) -> Dict[str, Any]:
        """
        Load memory from Redis for this user.
        
        Returns:
            Dict with 'buffer' and 'messages' keys
        """
        try:
            if self.redis_client is None:
                await self.initialize_redis()
            
            if self.redis_client is None:
                logger.debug("Redis unavailable, starting with empty memory", user_id=self.user_id)
                return {"buffer": "", "messages": []}
            
            # Acquire lock to prevent concurrent modifications
            async with self._get_lock():
                raw = await self.redis_client.get(self.memory_key)
                
                if raw:
                    try:
                        saved = pickle.loads(raw)
                        # Load messages into the memory object
                        messages = saved.get("messages", [])
                        if messages:
                            self.memory.chat_memory.messages = messages
                        
                        logger.info(
                            "memory_loaded_from_redis",
                            user_id=self.user_id,
                            message_count=len(self.memory.chat_memory.messages)
                        )
                        return saved
                    except Exception as e:
                        logger.error(f"Failed to deserialize memory: {e}")
                        # Continue with empty memory instead of failing
                        return {"buffer": "", "messages": []}
                else:
                    logger.debug("No existing memory found in Redis", user_id=self.user_id)
                    return {"buffer": "", "messages": []}
        
        except Exception as e:
            logger.error(f"Failed to load memory from Redis: {e}")
            return {"buffer": "", "messages": []}
    
    async def save_memory(self) -> bool:
        """
        Save memory to Redis for persistence.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.redis_client is None:
                await self.initialize_redis()
            
            if self.redis_client is None:
                logger.debug("Redis unavailable, memory not persisted", user_id=self.user_id)
                return False
            
            # Acquire lock to prevent concurrent modifications
            async with self._get_lock():
                # Get buffer safely - handle both string and list types
                buffer = getattr(self.memory, 'buffer', '')
                if isinstance(buffer, list):
                    buffer = str(buffer)
                
                obj = {
                    "buffer": buffer,
                    "messages": self.memory.chat_memory.messages,
                    "saved_at": datetime.utcnow().isoformat()
                }
                
                # Serialize and store with 24-hour expiration
                serialized = pickle.dumps(obj)
                await self.redis_client.setex(
                    self.memory_key,
                    86400,  # 24 hours
                    serialized
                )
                
                logger.info(
                    "memory_saved_to_redis",
                    user_id=self.user_id,
                    message_count=len(self.memory.chat_memory.messages),
                    buffer_size=len(str(buffer))
                )
                return True
        
        except Exception as e:
            logger.error(f"Failed to save memory to Redis: {e}")
            return False
    
    async def add_message(self, user_message: str, assistant_response: str) -> bool:
        """
        Add a user message and assistant response to memory.
        
        Args:
            user_message: User's question/input
            assistant_response: Aident's response
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Save context (this triggers auto-summarization if needed)
            self.memory.save_context(
                {"input": user_message},
                {"output": assistant_response}
            )
            
            # Persist to Redis
            await self.save_memory()
            
            logger.info(
                "message_added_to_memory",
                user_id=self.user_id,
                message_length=len(user_message)
            )
            return True
        
        except Exception as e:
            logger.error(f"Failed to add message to memory: {e}")
            return False
    
    def get_context(self) -> str:
        """
        Get current conversation context (buffer + recent messages).
        
        Returns:
            Formatted conversation history string
        """
        try:
            variables = self.memory.load_memory_variables({})
            return variables.get("chat_history", "")
        except Exception as e:
            logger.error(f"Failed to get memory context: {e}")
            return ""
    
    def get_messages(self) -> List[Dict[str, str]]:
        """
        Get all messages in memory.
        
        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        try:
            messages = []
            for msg in self.memory.chat_memory.messages:
                messages.append({
                    "role": "user" if msg.type == "human" else "assistant",
                    "content": msg.content
                })
            return messages
        except Exception as e:
            logger.error(f"Failed to get messages: {e}")
            return []
    
    async def clear_memory(self) -> bool:
        """
        Clear all memory for this user (useful for new conversations).
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.memory.clear()
            
            if self.redis_client is None:
                await self.initialize_redis()
            
            if self.redis_client:
                await self.redis_client.delete(self.memory_key)
            
            logger.info("memory_cleared", user_id=self.user_id)
            return True
        
        except Exception as e:
            logger.error(f"Failed to clear memory: {e}")
            return False
    
    def _get_lock(self):
        """Get async lock for Redis operations (prevents race conditions)"""
        return _AsyncLock(self.redis_client, self.lock_key) if self.redis_client else _NoOpLock()
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about current memory usage.
        
        Returns:
            Dict with message count, buffer size, and other metrics
        """
        try:
            messages = self.memory.chat_memory.messages
            
            # Get buffer - handle both string and list types
            buffer = self.memory.buffer
            if isinstance(buffer, list):
                buffer_size = len(buffer)
                buffer_tokens = sum(len(str(item).split()) for item in buffer)
            else:
                buffer_size = len(str(buffer))
                buffer_tokens = len(str(buffer).split())
            
            return {
                "user_id": self.user_id,
                "message_count": len(messages),
                "buffer_size": buffer_size,
                "buffer_tokens": buffer_tokens,
                "max_token_limit": self.max_token_limit,
                "recent_messages": [
                    {"role": "user" if m.type == "human" else "assistant", "content": m.content[:100]}
                    for m in messages[-5:]  # Last 5 messages
                ]
            }
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {"error": str(e)}
    
    def detect_frustration(self, user_message: str) -> int:
        """
        Detect frustration signals in user message.
        
        Returns:
            Frustration level (0-5)
        """
        frustration_signals = [
            'why', 'again', 'same', 'repeat', 'still', 'not helping',
            'confused', 'don\'t understand', 'what?', 'huh?', 'seriously',
            'come on', 'enough', 'stop', 'annoyed', 'frustrated'
        ]
        
        message_lower = user_message.lower()
        signal_count = sum(1 for signal in frustration_signals if signal in message_lower)
        
        # Increment frustration level if signals detected
        if signal_count > 0:
            self.conversation_state['frustration_level'] = min(
                self.conversation_state['frustration_level'] + signal_count,
                5  # Cap at 5
            )
            logger.info(
                "frustration_detected",
                user_id=self.user_id,
                frustration_level=self.conversation_state['frustration_level']
            )
        
        return self.conversation_state['frustration_level']
    
    def get_conversation_state(self) -> Dict[str, Any]:
        """
        Get current conversation state for repetition detection.
        
        Returns:
            Dict with topics, response types, phrases, and frustration level
        """
        return {
            'topics_discussed': list(self.conversation_state['topics_discussed']),
            'response_types_used': self.conversation_state['response_types_used'][-5:],  # Last 5
            'phrases_used': list(self.conversation_state['phrases_used']),
            'frustration_level': self.conversation_state['frustration_level'],
            'last_response_type': self.conversation_state['last_response_type']
        }
    
    def update_conversation_state(self, user_message: str, assistant_response: str) -> None:
        """
        Update conversation state after each exchange.
        
        Tracks topics, response types, and phrases for repetition detection.
        """
        # Extract topics from user message
        topic_keywords = {
            'revenue': ['revenue', 'sales', 'income', 'earnings'],
            'expenses': ['expense', 'cost', 'spending', 'outflow'],
            'cash_flow': ['cash flow', 'liquidity', 'cash position'],
            'profitability': ['profit', 'margin', 'ebitda', 'net income'],
            'vendors': ['vendor', 'supplier', 'payment', 'invoice'],
            'trends': ['trend', 'pattern', 'growth', 'decline'],
            'comparison': ['compare', 'vs', 'versus', 'difference'],
        }
        
        user_lower = user_message.lower()
        for topic, keywords in topic_keywords.items():
            if any(kw in user_lower for kw in keywords):
                self.conversation_state['topics_discussed'].add(topic)
        
        # Extract response type from assistant response
        response_type = 'general'
        response_lower = assistant_response.lower()
        
        if any(phrase in response_lower for phrase in ['here are', 'let me break', 'step 1', 'first,']):
            response_type = 'explanation'
        elif any(phrase in response_lower for phrase in ['your', 'data shows', 'based on', 'analysis']):
            response_type = 'data_query'
        elif any(phrase in response_lower for phrase in ['recommend', 'suggest', 'should', 'consider']):
            response_type = 'strategy'
        elif any(phrase in response_lower for phrase in ['why', 'because', 'reason', 'caused']):
            response_type = 'causal'
        
        self.conversation_state['response_types_used'].append(response_type)
        self.conversation_state['last_response_type'] = response_type
        
        # Extract opening phrase from response (first 15 words)
        words = assistant_response.split()[:15]
        phrase = ' '.join(words).lower()
        self.conversation_state['phrases_used'].add(phrase)


class _AsyncLock:
    """Simple async lock using Redis for distributed locking"""
    
    def __init__(self, redis_client, lock_key: str, timeout: int = 5):
        self.redis_client = redis_client
        self.lock_key = lock_key
        self.timeout = timeout
        self.acquired = False
    
    async def __aenter__(self):
        """Acquire lock"""
        for attempt in range(self.timeout * 10):  # Try for ~5 seconds
            acquired = await self.redis_client.set(
                self.lock_key,
                "1",
                nx=True,  # Only set if not exists
                ex=1  # Expire after 1 second
            )
            if acquired:
                self.acquired = True
                return self
            await asyncio.sleep(0.1)
        
        logger.warning(f"Failed to acquire lock: {self.lock_key}")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Release lock"""
        if self.acquired:
            try:
                await self.redis_client.delete(self.lock_key)
            except Exception as e:
                logger.warning(f"Failed to release lock: {e}")


class _NoOpLock:
    """No-op lock when Redis is unavailable"""
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
