"""
Aident CFO Brain Configuration
==============================

Centralized configuration for all intelligence modules.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMConfig:
    MODEL: str = "llama-3.3-70b-versatile"
    TIMEOUT_CLASSIFICATION: float = 30.0
    TIMEOUT_RESPONSE: float = 60.0
    TIMEOUT_INTENT: float = 10.0
    MAX_TOKENS_CLASSIFICATION: int = 200
    MAX_TOKENS_RESPONSE: int = 1000
    MAX_TOKENS_SMALLTALK: int = 150
    TEMPERATURE_CLASSIFICATION: float = 0.1
    TEMPERATURE_RESPONSE: float = 0.7


@dataclass(frozen=True)
class MemoryConfig:
    MAX_TOKEN_LIMIT: int = 2000
    REDIS_KEY_PREFIX: str = "aident:memory"
    REDIS_TTL_SECONDS: int = 86400 * 7  # 7 days
    MAX_MESSAGES_IN_CONTEXT: int = 20


@dataclass(frozen=True)
class ConfidenceConfig:
    INTENT_THRESHOLD: float = 0.6
    QUESTION_THRESHOLD: float = 0.6
    REPETITION_THRESHOLD: float = 0.85


@dataclass(frozen=True)
class RetryConfig:
    MAX_ATTEMPTS_CRITICAL: int = 3
    MAX_ATTEMPTS_STANDARD: int = 2
    BACKOFF_MULTIPLIER: float = 2.0


LLM = LLMConfig()
MEMORY = MemoryConfig()
CONFIDENCE = ConfidenceConfig()
RETRY = RetryConfig()


def get_redis_url() -> str:
    return os.getenv('ARQ_REDIS_URL') or os.getenv('REDIS_URL') or ''


def get_groq_api_key() -> str:
    key = os.getenv('GROQ_API_KEY')
    if not key:
        raise ValueError("GROQ_API_KEY environment variable is required")
    return key
