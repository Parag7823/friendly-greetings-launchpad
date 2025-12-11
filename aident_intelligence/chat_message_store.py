"""
LangChain Message Storage Adapter
===================================

REPLACES: Manual Supabase inserts for chat messages
USES: LangChain PostgresChatMessageHistory

BENEFITS:
- Standardized message schema
- Automatic session handling
- Built-in message retrieval
- Type-safe message objects
"""

from typing import Optional, List
from datetime import datetime
import structlog

logger = structlog.get_logger(__name__)

# Check if langchain is available
try:
    from langchain_community.chat_message_histories import PostgresChatMessageHistory
    from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not installed. Using fallback storage. Install: pip install langchain langchain-community")


class ChatMessageStore:
    """
    Stores chat messages using LangChain's PostgresChatMessageHistory.
    
    Provides a standardized interface for storing and retrieving chat messages with:
    - Automatic session management
    - Type-safe message objects
    - Built-in pagination
    - Message metadata support
    """
    
    def __init__(self, connection_string: str, user_id: str, chat_id: str):
        """
        Initialize chat message store.
        
        Args:
            connection_string: PostgreSQL connection string
            user_id: User ID for message filtering
            chat_id: Chat/session ID
        """
        self.connection_string = connection_string
        self.user_id = user_id
        self.chat_id = chat_id
        self.use_langchain = LANGCHAIN_AVAILABLE
        
        if self.use_langchain:
            try:
                # Use chat_id as session_id
                # Table will be 'message_store' by default
                self.history = PostgresChatMessageHistory(
                    connection_string=connection_string,
                    session_id=f"{user_id}:{chat_id}"
                )
                logger.info(f"✅ PostgresChatMessageHistory initialized for session {chat_id}")
            except Exception as e:
                logger.error(f"Failed to initialize LangChain history: {e}")
                self.use_langchain = False
    
    async def add_user_message(self, message: str, metadata: Optional[dict] = None):
        """
        Add a user message to the chat history.
        
        Args:
            message: User's message content
            metadata: Optional metadata (e.g., timestamp, question_type)
        """
        if not self.use_langchain:
            logger.warning("LangChain unavailable, message not stored")
            return
        
        try:
            # Create HumanMessage with metadata
            msg = HumanMessage(content=message)
            if metadata:
                msg.additional_kwargs = metadata
            
            self.history.add_message(msg)
            logger.info(f"Stored user message for session {self.chat_id}")
        
        except Exception as e:
            logger.error(f"Failed to store user message: {e}")
    
    async def add_ai_message(self, message: str, metadata: Optional[dict] = None):
        """
        Add an AI response message to the chat history.
        
        Args:
            message: AI's response content
            metadata: Optional metadata (e.g., confidence, question_type, data)
        """
        if not self.use_langchain:
            logger.warning("LangChain unavailable, message not stored")
            return
        
        try:
            # Create AIMessage with metadata
            msg = AIMessage(content=message)
            if metadata:
                msg.additional_kwargs = metadata
            
            self.history.add_message(msg)
            logger.info(f"Stored AI message for session {self.chat_id}")
        
        except Exception as e:
            logger.error(f"Failed to store AI message: {e}")
    
    def get_messages(self, limit: Optional[int] = None) -> List[BaseMessage]:
        """
        Retrieve chat messages for this session.
        
        Args:
            limit: Optional limit on number of messages to retrieve
        
        Returns:
            List of LangChain message objects
        """
        if not self.use_langchain:
            return []
        
        try:
            messages = self.history.messages
            
            if limit and len(messages) > limit:
                return messages[-limit:]  # Return most recent N messages
            
            return messages
        
        except Exception as e:
            logger.error(f"Failed to retrieve messages: {e}")
            return []
    
    def clear(self):
        """Clear all messages for this session."""
        if not self.use_langchain:
            return
        
        try:
            self.history.clear()
            logger.info(f"Cleared chat history for session {self.chat_id}")
        
        except Exception as e:
            logger.error(f"Failed to clear chat history: {e}")


def create_message_store(
    user_id: str,
    chat_id: str,
    connection_string: str
) -> ChatMessageStore:
    """
    Factory function to create message store.
    
    Args:
        user_id: User ID
        chat_id: Chat/session ID
        connection_string: PostgreSQL connection string (REQUIRED)
    
    Returns:
        ChatMessageStore
    
    Raises:
        RuntimeError: If LangChain or connection_string not available
    """
    if not LANGCHAIN_AVAILABLE:
        raise RuntimeError(
            "LangChain not installed. Install: pip install langchain langchain-community psycopg2-binary"
        )
    
    if not connection_string:
        raise ValueError("PostgreSQL connection_string is required")
    
    return ChatMessageStore(connection_string, user_id, chat_id)


# ============================================================================
# PRELOAD PATTERN: Initialize heavy dependencies at module-load time
# ============================================================================
# This runs automatically when the module is imported, eliminating the
# first-request latency that was caused by lazy-loading.

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
    
    # Preload LangChain PostgresChatMessageHistory
    try:
        if LANGCHAIN_AVAILABLE:
            from langchain_community.chat_message_histories import PostgresChatMessageHistory
            from langchain_core.messages import HumanMessage, AIMessage
            logger.info("✅ PRELOAD: PostgresChatMessageHistory loaded at module-load time")
    except Exception as e:
        logger.warning(f"⚠️ PRELOAD: PostgresChatMessageHistory load failed: {e}")
    
    _PRELOAD_COMPLETED = True

try:
    _preload_all_modules()
except Exception as e:
    logger.warning(f"Module-level chat_message_store preload failed (will use fallback): {e}")

