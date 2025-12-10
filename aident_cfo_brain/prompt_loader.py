"""
Prompt Loader Utility
======================

Loads and manages LangChain PromptTemplates from external YAML configuration.
Provides centralized prompt management with hot-reload capability.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from jinja2 import Template
import structlog

logger = structlog.get_logger(__name__)

class PromptLoader:
    """
    Loads prompts from YAML config and provides Jinja2 template rendering.
    
    Replaces hard-coded prompts with externalized configuration for:
    - Easy editing without code changes
    - Version control of prompt evolution
    - A/B testing of different prompts
    - Non-technical team members can update prompts
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize prompt loader with config file path.
        
        Args:
            config_path: Path to prompts.yaml file. If None, uses default location.
        """
        if config_path is None:
            # Default to aident_cfo_brain/config/prompts.yaml
            config_path = Path(__file__).parent / "config" / "prompts.yaml"
        
        self.config_path = Path(config_path)
        self.prompts: Dict[str, Any] = {}
        self._load_prompts()
    
    def _load_prompts(self) -> None:
        """Load prompts from YAML file."""
        try:
            if not self.config_path.exists():
                logger.error(f"Prompts config file not found: {self.config_path}")
                raise FileNotFoundError(f"Prompts config not found: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.prompts = yaml.safe_load(f)
            
            logger.info(f"✅ Loaded {len(self.prompts)} prompt categories from {self.config_path}")
        
        except Exception as e:
            logger.error(f"Failed to load prompts: {e}")
            # Use empty dict to prevent cascading failures
            self.prompts = {}
    
    def reload(self) -> None:
        """Reload prompts from YAML (for hot-reload during development)."""
        logger.info("Reloading prompts from config...")
        self._load_prompts()
    
    def get_prompt(self, category: str, key: str, **kwargs) -> str:
        """
        Get a prompt template and render it with variables.
        
        Args:
            category: Prompt category (e.g., 'general_question', 'greeting')
            key: Prompt key within category (e.g., 'system', 'user')
            **kwargs: Variables to substitute in the template
        
        Returns:
            Rendered prompt string
        
        Examples:
            >>> loader = PromptLoader()
            >>> prompt = loader.get_prompt('greeting', 'user', question="Hello", user_context="...")
        """
        try:
            # Navigate to prompt
            if category not in self.prompts:
                logger.warning(f"Prompt category '{category}' not found in config")
                return ""
            
            category_data = self.prompts[category]
            
            if key not in category_data:
                logger.warning(f"Prompt key '{key}' not found in category '{category}'")
                return ""
            
            prompt_template = category_data[key]
            
            # Render with Jinja2
            template = Template(prompt_template)
            rendered = template.render(**kwargs)
            
            return rendered.strip()
        
        except Exception as e:
            logger.error(f"Failed to render prompt {category}.{key}: {e}")
            return ""
    
    def get_fallback(self, fallback_key: str) -> str:
        """
        Get a fallback response for error cases.
        
        Args:
            fallback_key: Key in the 'fallbacks' category
        
        Returns:
            Fallback response string
        """
        try:
            if 'fallbacks' not in self.prompts:
                return "I encountered an error. Please try again."
            
            fallbacks = self.prompts['fallbacks']
            return fallbacks.get(fallback_key, "I encountered an error. Please try again.")
        
        except Exception as e:
            logger.error(f"Failed to get fallback {fallback_key}: {e}")
            return "I encountered an error. Please try again."
    
    def get_onboarding_message(self, state: str) -> str:
        """
        Get onboarding message for a specific user state.
        
        Args:
            state: Onboarding state ('first_visit', 'onboarded', 'data_connected', 'active')
        
        Returns:
            Onboarding message string
        """
        try:
            if 'onboarding' not in self.prompts:
                return "Let's connect your financial data to get started!"
            
            onboarding = self.prompts['onboarding']
            return onboarding.get(state, onboarding.get('first_visit', ''))
        
        except Exception as e:
            logger.error(f"Failed to get onboarding message for state {state}: {e}")
            return "Let's connect your financial data to get started!"


# Global singleton instance
_prompt_loader: Optional[PromptLoader] = None


def get_prompt_loader(config_path: Optional[str] = None) -> PromptLoader:
    """
    Get or create the global PromptLoader singleton.
    
    Args:
        config_path: Optional custom config path
    
    Returns:
        PromptLoader instance
    """
    global _prompt_loader
    
    if _prompt_loader is None:
        _prompt_loader = PromptLoader(config_path)
    
    return _prompt_loader
