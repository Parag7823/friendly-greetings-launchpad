"""
Business Rules Engine
=====================

REPLACES: Hard-coded if/elif logic for onboarding thresholds
USES: json-logic-py for dynamic rule evaluation

BENEFITS:
- Change thresholds without code changes
- Version control for business rules
- A/B test different thresholds
- Non-developers can modify rules
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import structlog

logger = structlog.get_logger(__name__)

# Check if json-logic is available
try:
    from json_logic import jsonLogic
    JSON_LOGIC_AVAILABLE = True
except ImportError:
    JSON_LOGIC_AVAILABLE = False
    logger.warning("json-logic-py not installed. Using fallback rules. Install: pip install json-logic")


class BusinessRulesEngine:
    """
    Evaluates business rules defined in JSON configuration.
    
    Replaces hard-coded if/elif logic with externalized rules:
    - Onboarding thresholds
    - Feature access rules
    - Data mode determination
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize business rules engine.
        
        Args:
            config_path: Path to business_rules.json. If None, uses default location.
        """
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "business_rules.json"
        
        self.config_path = Path(config_path)
        self.rules: Dict[str, Any] = {}
        self._load_rules()
    
    def _load_rules(self) -> None:
        """Load rules from JSON configuration."""
        try:
            if not self.config_path.exists():
                logger.error(f"Business rules config not found: {self.config_path}")
                raise FileNotFoundError(f"Business rules config not found: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.rules = json.load(f)
            
            logger.info(f"✅ Loaded business rules from {self.config_path}")
        
        except Exception as e:
            logger.error(f"Failed to load business rules: {e}")
            # Use empty dict to prevent cascading failures
            self.rules = {}
    
    def reload(self) -> None:
        """Reload rules from JSON (for hot-reload during development)."""
        logger.info("Reloading business rules from config...")
        self._load_rules()
    
    def determine_data_mode(self, transaction_count: int) -> str:
        """
        Determine data mode based on transaction count.
        
        Args:
            transaction_count: Number of transactions
        
        Returns:
            Data mode string: 'NO_DATA', 'LIMITED_DATA', or 'RICH_DATA'
        """
        if not JSON_LOGIC_AVAILABLE:
            raise RuntimeError(
                "json-logic-py not installed. Install it: pip install json-logic"
            )
        
        try:
            rules = self.rules.get('data_mode_rules', {}).get('rules', [])
            data = {"transaction_count": transaction_count}
            
            # Evaluate rules in order
            for rule in rules:
                condition = rule.get('condition')
                result = rule.get('result')
                
                if jsonLogic(condition, data):
                    logger.info(f"Data mode determined: {result} (transaction_count={transaction_count})")
                    return result
            
            # Default fallback
            return "NO_DATA"
        
        except Exception as e:
            logger.error(f"Failed to evaluate data mode rules: {e}")
            raise RuntimeError(f"Rule evaluation failed: {e}") from e
    
    def get_threshold(self, threshold_name: str) -> Optional[int]:
        """
        Get a threshold value from configuration.
        
        Args:
            threshold_name: Name of the threshold
        
        Returns:
            Threshold value or None
        """
        thresholds = self.rules.get('onboarding_thresholds', {})
        return thresholds.get(threshold_name)
    
    def check_feature_access(self, feature: str, transaction_count: int) -> bool:
        """
        Check if a feature is accessible based on transaction count.
        
        Args:
            feature: Feature name
            transaction_count: Number of transactions
        
        Returns:
            True if feature is accessible, False otherwise
        """
        try:
            feature_rules = self.rules.get('feature_access_rules', {}).get('rules', [])
            
            for rule in feature_rules:
                if rule.get('feature') == feature:
                    min_transactions = rule.get('min_transactions', 0)
                    enabled = rule.get('enabled', True)
                    
                    if not enabled:
                        return False
                    
                    return transaction_count >= min_transactions
            
            # Default: allow if feature not found in rules
            return True
        
        except Exception as e:
            logger.error(f"Failed to check feature access: {e}")
            return True  # Fail open
    
    def evaluate_custom_rule(self, rule_name: str, data: Dict[str, Any]) -> Any:
        """
        Evaluate a custom rule by name.
        
        Args:
            rule_name: Name of the rule category in config
            data: Data to evaluate against the rules
        
        Returns:
            Result of rule evaluation
        """
        if not JSON_LOGIC_AVAILABLE:
            logger.warning(f"Cannot evaluate custom rule '{rule_name}': json-logic not available")
            return None
        
        try:
            rule_category = self.rules.get(rule_name, {})
            rules = rule_category.get('rules', [])
            
            for rule in rules:
                condition = rule.get('condition')
                result = rule.get('result')
                
                if jsonLogic(condition, data):
                    return result
            
            return None
        
        except Exception as e:
            logger.error(f"Failed to evaluate custom rule '{rule_name}': {e}")
            return None


# Singleton instance
_business_rules_engine: Optional[BusinessRulesEngine] = None


def get_business_rules_engine(config_path: Optional[str] = None) -> BusinessRulesEngine:
    """Get or create the global BusinessRulesEngine singleton."""
    global _business_rules_engine
    
    if _business_rules_engine is None:
        _business_rules_engine = BusinessRulesEngine(config_path)
    
    return _business_rules_engine
