"""
Aident Intelligence Module
===========================

Core intelligence engines for the Finley AI financial assistant.

Components:
- aident_memory_manager: Conversational memory with Redis persistence
- intelligent_chat_orchestrator: Routes questions to intelligence engines
- intent_and_guard_engine: Intent classification and output quality control
- prompt_loader: Manages LangChain prompts from YAML configuration
- question_classifier_setfit: Fast few-shot question classification
- train_question_classifier: Training script for question classifier

Usage:
    from aident_intelligence import (
        AidentMemoryManager,
        IntelligentChatOrchestrator,
        IntentClassifier,
        PromptLoader,
        QuestionClassifierSetFit
    )
"""

__version__ = "1.0.0"
__author__ = "Aident Team"

# Import main classes for convenience
try:
    from .aident_memory_manager import AidentMemoryManager, get_memory_manager
    from .intelligent_chat_orchestrator import IntelligentChatOrchestrator
    from .intent_and_guard_engine import IntentClassifier, OutputGuard, UserIntent
    from .prompt_loader import PromptLoader, get_prompt_loader
    from .question_classifier_setfit import QuestionClassifierSetFit
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import some Aident Intelligence components: {e}")

__all__ = [
    "AidentMemoryManager",
    "get_memory_manager",
    "IntelligentChatOrchestrator",
    "IntentClassifier",
    "OutputGuard",
    "UserIntent",
    "PromptLoader",
    "get_prompt_loader",
    "QuestionClassifierSetFit",
]
