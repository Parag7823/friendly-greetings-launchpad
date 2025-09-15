"""
Accuracy Enhancement System
Provides confidence scoring, validation rules, and idempotency for data enrichment and document analysis.
"""

import hashlib
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, date
from dataclasses import dataclass, asdict
from enum import Enum
import re
import difflib
from decimal import Decimal, InvalidOperation

logger = logging.getLogger(__name__)

class ConfidenceLevel(Enum):
    """Confidence level enumeration"""
    VERY_LOW = 0.0
    LOW = 0.25
    MEDIUM = 0.5
    HIGH = 0.75
    VERY_HIGH = 1.0

@dataclass
class ValidationRule:
    """Validation rule definition"""
    field_name: str
    rule_type: str  # 'regex', 'range', 'enum', 'custom'
    rule_value: Any
    error_message: str
    confidence_impact: float = 0.1  # How much this rule affects confidence

@dataclass
class ConfidenceScore:
    """Confidence score with breakdown"""
    overall: float
    breakdown: Dict[str, float]
    factors: List[str]
    timestamp: datetime
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

@dataclass
class ValidationResult:
    """Validation result with details"""
    is_valid: bool
    confidence_score: float
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]

class IdempotencyManager:
    """
    Manages idempotency for operations to ensure same inputs produce same outputs.
    """
    
    def __init__(self):
        self.operation_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 3600  # 1 hour
    
    def generate_operation_id(self, operation_type: str, inputs: Dict[str, Any]) -> str:
        """Generate deterministic operation ID from inputs"""
        # Create deterministic hash from inputs
        input_string = json.dumps(inputs, sort_keys=True, default=str)
        operation_hash = hashlib.sha256(f"{operation_type}:{input_string}".encode()).hexdigest()
        return f"{operation_type}:{operation_hash[:16]}"
    
    def get_cached_result(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Get cached result for operation"""
        if operation_id in self.operation_cache:
            cached_data = self.operation_cache[operation_id]
            # Check if cache is still valid
            if time.time() - cached_data['timestamp'] < self.cache_ttl:
                return cached_data['result']
            else:
                # Remove expired cache
                del self.operation_cache[operation_id]
        return None
    
    def cache_result(self, operation_id: str, result: Dict[str, Any]) -> None:
        """Cache operation result"""
        self.operation_cache[operation_id] = {
            'result': result,
            'timestamp': time.time()
        }
    
    def clear_expired_cache(self) -> int:
        """Clear expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, data in self.operation_cache.items()
            if current_time - data['timestamp'] >= self.cache_ttl
        ]
        
        for key in expired_keys:
            del self.operation_cache[key]
        
        return len(expired_keys)

class ValidationEngine:
    """
    Advanced validation engine with confidence scoring.
    """
    
    def __init__(self):
        self.validation_rules: Dict[str, List[ValidationRule]] = {}
        self.custom_validators: Dict[str, callable] = {}
        self._load_default_rules()
    
    def _load_default_rules(self):
        """Load default validation rules"""
        # Amount validation rules
        self.add_validation_rule(
            field_name="amount",
            rule_type="range",
            rule_value={"min": 0, "max": 1000000},
            error_message="Amount must be between 0 and 1,000,000",
            confidence_impact=0.2
        )
        
        # Date validation rules
        self.add_validation_rule(
            field_name="date",
            rule_type="custom",
            rule_value=self._validate_date_format,
            error_message="Invalid date format",
            confidence_impact=0.15
        )
        
        # Email validation rules
        self.add_validation_rule(
            field_name="email",
            rule_type="regex",
            rule_value=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            error_message="Invalid email format",
            confidence_impact=0.1
        )
        
        # Vendor name validation rules
        self.add_validation_rule(
            field_name="vendor_name",
            rule_type="custom",
            rule_value=self._validate_vendor_name,
            error_message="Invalid vendor name",
            confidence_impact=0.1
        )
        
        # Platform ID validation rules
        self.add_validation_rule(
            field_name="platform_id",
            rule_type="custom",
            rule_value=self._validate_platform_id,
            error_message="Invalid platform ID format",
            confidence_impact=0.15
        )
    
    def add_validation_rule(self, field_name: str, rule_type: str, rule_value: Any, 
                           error_message: str, confidence_impact: float = 0.1):
        """Add a validation rule"""
        rule = ValidationRule(
            field_name=field_name,
            rule_type=rule_type,
            rule_value=rule_value,
            error_message=error_message,
            confidence_impact=confidence_impact
        )
        
        if field_name not in self.validation_rules:
            self.validation_rules[field_name] = []
        
        self.validation_rules[field_name].append(rule)
    
    def add_custom_validator(self, field_name: str, validator_func: callable):
        """Add custom validator function"""
        self.custom_validators[field_name] = validator_func
    
    def validate_field(self, field_name: str, value: Any) -> ValidationResult:
        """Validate a single field"""
        errors = []
        warnings = []
        suggestions = []
        confidence_penalty = 0.0
        
        if field_name not in self.validation_rules:
            return ValidationResult(
                is_valid=True,
                confidence_score=1.0,
                errors=errors,
                warnings=warnings,
                suggestions=suggestions
            )
        
        for rule in self.validation_rules[field_name]:
            try:
                if rule.rule_type == "regex":
                    if not re.match(rule.rule_value, str(value)):
                        errors.append(rule.error_message)
                        confidence_penalty += rule.confidence_impact
                
                elif rule.rule_type == "range":
                    if isinstance(value, (int, float, Decimal)):
                        if value < rule.rule_value["min"] or value > rule.rule_value["max"]:
                            errors.append(rule.error_message)
                            confidence_penalty += rule.confidence_impact
                    else:
                        warnings.append(f"Value {value} is not numeric for range validation")
                        confidence_penalty += rule.confidence_impact * 0.5
                
                elif rule.rule_type == "enum":
                    if value not in rule.rule_value:
                        errors.append(rule.error_message)
                        confidence_penalty += rule.confidence_impact
                
                elif rule.rule_type == "custom":
                    if callable(rule.rule_value):
                        custom_result = rule.rule_value(value)
                        if not custom_result["is_valid"]:
                            errors.append(rule.error_message)
                            confidence_penalty += rule.confidence_impact
                        
                        if "warnings" in custom_result:
                            warnings.extend(custom_result["warnings"])
                        if "suggestions" in custom_result:
                            suggestions.extend(custom_result["suggestions"])
                
            except Exception as e:
                logger.warning(f"Validation rule failed for {field_name}: {e}")
                warnings.append(f"Validation rule failed: {str(e)}")
                confidence_penalty += rule.confidence_impact * 0.5
        
        # Apply custom validator if available
        if field_name in self.custom_validators:
            try:
                custom_result = self.custom_validators[field_name](value)
                if not custom_result["is_valid"]:
                    errors.append(f"Custom validation failed for {field_name}")
                    confidence_penalty += 0.1
            except Exception as e:
                logger.warning(f"Custom validator failed for {field_name}: {e}")
                warnings.append(f"Custom validator failed: {str(e)}")
        
        confidence_score = max(0.0, 1.0 - confidence_penalty)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            confidence_score=confidence_score,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def validate_data(self, data: Dict[str, Any]) -> Dict[str, ValidationResult]:
        """Validate entire data structure"""
        results = {}
        
        for field_name, value in data.items():
            results[field_name] = self.validate_field(field_name, value)
        
        return results
    
    def _validate_date_format(self, value: Any) -> Dict[str, Any]:
        """Custom date format validator"""
        if value is None:
            return {"is_valid": False, "warnings": ["Date is null"]}
        
        try:
            if isinstance(value, str):
                # Try common date formats
                date_formats = [
                    "%Y-%m-%d",
                    "%m/%d/%Y",
                    "%d/%m/%Y",
                    "%Y-%m-%d %H:%M:%S",
                    "%m/%d/%Y %H:%M:%S"
                ]
                
                for fmt in date_formats:
                    try:
                        datetime.strptime(value, fmt)
                        return {"is_valid": True}
                    except ValueError:
                        continue
                
                return {"is_valid": False, "warnings": ["Unrecognized date format"]}
            
            elif isinstance(value, (datetime, date)):
                return {"is_valid": True}
            
            else:
                return {"is_valid": False, "warnings": ["Invalid date type"]}
        
        except Exception as e:
            return {"is_valid": False, "warnings": [f"Date validation error: {str(e)}"]}
    
    def _validate_vendor_name(self, value: Any) -> Dict[str, Any]:
        """Custom vendor name validator"""
        if value is None or value == "":
            return {"is_valid": False, "warnings": ["Vendor name is empty"]}
        
        if not isinstance(value, str):
            return {"is_valid": False, "warnings": ["Vendor name must be a string"]}
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'^\d+$',  # Only numbers
            r'^[^a-zA-Z]*$',  # No letters
            r'^.{1,2}$',  # Too short
            r'^.{100,}$'  # Too long
        ]
        
        warnings = []
        for pattern in suspicious_patterns:
            if re.match(pattern, value):
                warnings.append(f"Vendor name '{value}' has suspicious pattern")
        
        # Check for common vendor name patterns
        if len(value) < 3:
            warnings.append("Vendor name is very short")
        
        if len(value) > 100:
            warnings.append("Vendor name is very long")
        
        return {
            "is_valid": len(warnings) == 0,
            "warnings": warnings,
            "suggestions": ["Consider standardizing vendor name format"] if warnings else []
        }
    
    def _validate_platform_id(self, value: Any) -> Dict[str, Any]:
        """Custom platform ID validator"""
        if value is None or value == "":
            return {"is_valid": False, "warnings": ["Platform ID is empty"]}
        
        if not isinstance(value, str):
            return {"is_valid": False, "warnings": ["Platform ID must be a string"]}
        
        # Check for common platform ID patterns
        platform_patterns = [
            r'^[A-Z]{2,4}_\d+$',  # Platform prefix + number
            r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$',  # UUID
            r'^\d{10,}$',  # Long numeric ID
            r'^[A-Z0-9]{10,}$'  # Alphanumeric ID
        ]
        
        for pattern in platform_patterns:
            if re.match(pattern, value):
                return {"is_valid": True}
        
        return {
            "is_valid": False,
            "warnings": ["Platform ID doesn't match expected patterns"],
            "suggestions": ["Check platform ID format"]
        }

class ConfidenceCalculator:
    """
    Calculates confidence scores for data enrichment and document analysis results.
    """
    
    def __init__(self):
        self.validation_engine = ValidationEngine()
        self.confidence_weights = {
            'validation': 0.3,
            'ai_confidence': 0.25,
            'data_quality': 0.2,
            'consistency': 0.15,
            'completeness': 0.1
        }
    
    def calculate_enrichment_confidence(self, enriched_data: Dict[str, Any], 
                                      validation_results: Dict[str, ValidationResult],
                                      ai_confidence: float = 0.8) -> ConfidenceScore:
        """Calculate confidence score for enrichment results"""
        factors = []
        breakdown = {}
        
        # Validation confidence
        validation_scores = [result.confidence_score for result in validation_results.values()]
        avg_validation_score = sum(validation_scores) / len(validation_scores) if validation_scores else 1.0
        breakdown['validation'] = avg_validation_score
        factors.append(f"Validation score: {avg_validation_score:.2f}")
        
        # AI confidence
        breakdown['ai_confidence'] = ai_confidence
        factors.append(f"AI confidence: {ai_confidence:.2f}")
        
        # Data quality assessment
        data_quality_score = self._assess_data_quality(enriched_data)
        breakdown['data_quality'] = data_quality_score
        factors.append(f"Data quality: {data_quality_score:.2f}")
        
        # Consistency check
        consistency_score = self._check_consistency(enriched_data)
        breakdown['consistency'] = consistency_score
        factors.append(f"Consistency: {consistency_score:.2f}")
        
        # Completeness check
        completeness_score = self._check_completeness(enriched_data)
        breakdown['completeness'] = completeness_score
        factors.append(f"Completeness: {completeness_score:.2f}")
        
        # Calculate weighted overall score
        overall_score = sum(
            breakdown[factor] * self.confidence_weights[factor]
            for factor in breakdown
        )
        
        return ConfidenceScore(
            overall=overall_score,
            breakdown=breakdown,
            factors=factors,
            timestamp=datetime.utcnow()
        )
    
    def calculate_document_analysis_confidence(self, analysis_result: Dict[str, Any],
                                             document_features: Dict[str, Any]) -> ConfidenceScore:
        """Calculate confidence score for document analysis results"""
        factors = []
        breakdown = {}
        
        # Feature-based confidence
        feature_confidence = self._assess_document_features(document_features)
        breakdown['feature_confidence'] = feature_confidence
        factors.append(f"Feature confidence: {feature_confidence:.2f}")
        
        # Pattern matching confidence
        pattern_confidence = self._assess_pattern_matching(analysis_result)
        breakdown['pattern_confidence'] = pattern_confidence
        factors.append(f"Pattern confidence: {pattern_confidence:.2f}")
        
        # AI classification confidence
        ai_confidence = analysis_result.get('ai_confidence', 0.8)
        breakdown['ai_confidence'] = ai_confidence
        factors.append(f"AI confidence: {ai_confidence:.2f}")
        
        # OCR confidence (if applicable)
        ocr_confidence = analysis_result.get('ocr_confidence', 1.0)
        breakdown['ocr_confidence'] = ocr_confidence
        factors.append(f"OCR confidence: {ocr_confidence:.2f}")
        
        # Calculate overall score
        overall_score = (
            feature_confidence * 0.3 +
            pattern_confidence * 0.25 +
            ai_confidence * 0.25 +
            ocr_confidence * 0.2
        )
        
        return ConfidenceScore(
            overall=overall_score,
            breakdown=breakdown,
            factors=factors,
            timestamp=datetime.utcnow()
        )
    
    def _assess_data_quality(self, data: Dict[str, Any]) -> float:
        """Assess overall data quality"""
        quality_indicators = []
        
        # Check for null values
        null_count = sum(1 for value in data.values() if value is None or value == "")
        null_ratio = null_count / len(data) if data else 0
        quality_indicators.append(1.0 - null_ratio)
        
        # Check for data types consistency
        type_consistency = self._check_type_consistency(data)
        quality_indicators.append(type_consistency)
        
        # Check for outliers (for numeric fields)
        outlier_score = self._check_outliers(data)
        quality_indicators.append(outlier_score)
        
        return sum(quality_indicators) / len(quality_indicators)
    
    def _check_type_consistency(self, data: Dict[str, Any]) -> float:
        """Check data type consistency"""
        type_consistency_score = 1.0
        
        # Check amount fields
        amount_fields = ['amount', 'total', 'value', 'price', 'cost']
        for field in amount_fields:
            if field in data and data[field] is not None:
                try:
                    float(data[field])
                except (ValueError, TypeError):
                    type_consistency_score -= 0.1
        
        # Check date fields
        date_fields = ['date', 'created_at', 'updated_at', 'timestamp']
        for field in date_fields:
            if field in data and data[field] is not None:
                if not isinstance(data[field], (str, datetime, date)):
                    type_consistency_score -= 0.1
        
        return max(0.0, type_consistency_score)
    
    def _check_outliers(self, data: Dict[str, Any]) -> float:
        """Check for outliers in numeric data"""
        numeric_fields = ['amount', 'total', 'value', 'price', 'cost']
        outlier_score = 1.0
        
        for field in numeric_fields:
            if field in data and data[field] is not None:
                try:
                    value = float(data[field])
                    # Simple outlier detection (values > 100k or < 0)
                    if value > 100000 or value < 0:
                        outlier_score -= 0.1
                except (ValueError, TypeError):
                    outlier_score -= 0.05
        
        return max(0.0, outlier_score)
    
    def _check_consistency(self, data: Dict[str, Any]) -> float:
        """Check data consistency"""
        consistency_score = 1.0
        
        # Check currency consistency
        if 'currency' in data and 'amount' in data:
            # Simple currency validation
            valid_currencies = ['USD', 'EUR', 'GBP', 'CAD', 'AUD']
            if data['currency'] not in valid_currencies:
                consistency_score -= 0.1
        
        # Check date consistency
        if 'date' in data and 'created_at' in data:
            try:
                if isinstance(data['date'], str) and isinstance(data['created_at'], str):
                    date_obj = datetime.strptime(data['date'], '%Y-%m-%d')
                    created_obj = datetime.strptime(data['created_at'], '%Y-%m-%d')
                    if date_obj > created_obj:
                        consistency_score -= 0.1  # Date in future
            except (ValueError, TypeError):
                consistency_score -= 0.05
        
        return max(0.0, consistency_score)
    
    def _check_completeness(self, data: Dict[str, Any]) -> float:
        """Check data completeness"""
        required_fields = ['amount', 'date', 'vendor_name']
        present_fields = sum(1 for field in required_fields if field in data and data[field] is not None)
        return present_fields / len(required_fields)
    
    def _assess_document_features(self, features: Dict[str, Any]) -> float:
        """Assess document features for confidence"""
        feature_score = 0.0
        total_features = 0
        
        # Column count confidence
        if 'column_count' in features:
            col_count = features['column_count']
            if 3 <= col_count <= 20:  # Reasonable range
                feature_score += 1.0
            elif col_count < 3:
                feature_score += 0.5  # Too few columns
            else:
                feature_score += 0.7  # Too many columns
            total_features += 1
        
        # Row count confidence
        if 'row_count' in features:
            row_count = features['row_count']
            if 1 <= row_count <= 10000:  # Reasonable range
                feature_score += 1.0
            elif row_count == 0:
                feature_score += 0.0  # No data
            else:
                feature_score += 0.8  # Very large dataset
            total_features += 1
        
        # Data type confidence
        if 'data_types' in features:
            type_score = len(features['data_types']) / 5  # Expect ~5 different types
            feature_score += min(1.0, type_score)
            total_features += 1
        
        return feature_score / total_features if total_features > 0 else 0.5
    
    def _assess_pattern_matching(self, analysis_result: Dict[str, Any]) -> float:
        """Assess pattern matching confidence"""
        pattern_score = 0.0
        
        # Check for recognized patterns
        if 'recognized_patterns' in analysis_result:
            patterns = analysis_result['recognized_patterns']
            if patterns:
                pattern_score += 0.5
        
        # Check for platform detection
        if 'platform' in analysis_result and analysis_result['platform']:
            pattern_score += 0.3
        
        # Check for document type detection
        if 'document_type' in analysis_result and analysis_result['document_type']:
            pattern_score += 0.2
        
        return min(1.0, pattern_score)

class AccuracyEnhancementSystem:
    """
    Main accuracy enhancement system that coordinates all accuracy features.
    """
    
    def __init__(self):
        self.validation_engine = ValidationEngine()
        self.confidence_calculator = ConfidenceCalculator()
        self.idempotency_manager = IdempotencyManager()
        self.operation_history: List[Dict[str, Any]] = []
    
    async def enhance_enrichment_accuracy(self, row_data: Dict[str, Any], 
                                        enrichment_result: Dict[str, Any],
                                        file_context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance enrichment accuracy with validation and confidence scoring"""
        
        # Check idempotency
        operation_id = self.idempotency_manager.generate_operation_id(
            "enrichment", 
            {"row_data": row_data, "file_context": file_context}
        )
        
        cached_result = self.idempotency_manager.get_cached_result(operation_id)
        if cached_result:
            logger.info(f"Returning cached enrichment result for {operation_id}")
            return cached_result
        
        # Validate enriched data
        validation_results = self.validation_engine.validate_data(enrichment_result)
        
        # Calculate confidence score
        confidence_score = self.confidence_calculator.calculate_enrichment_confidence(
            enrichment_result, validation_results
        )
        
        # Enhance result with accuracy information
        enhanced_result = enrichment_result.copy()
        enhanced_result.update({
            'accuracy_enhancement': {
                'confidence_score': confidence_score.overall,
                'confidence_breakdown': confidence_score.breakdown,
                'confidence_factors': confidence_score.factors,
                'validation_results': {
                    field: {
                        'is_valid': result.is_valid,
                        'confidence_score': result.confidence_score,
                        'errors': result.errors,
                        'warnings': result.warnings,
                        'suggestions': result.suggestions
                    }
                    for field, result in validation_results.items()
                },
                'timestamp': confidence_score.timestamp.isoformat(),
                'operation_id': operation_id
            }
        })
        
        # Cache result for idempotency
        self.idempotency_manager.cache_result(operation_id, enhanced_result)
        
        # Log operation
        self.operation_history.append({
            'operation_id': operation_id,
            'operation_type': 'enrichment',
            'confidence_score': confidence_score.overall,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        return enhanced_result
    
    async def enhance_document_analysis_accuracy(self, df_hash: str, filename: str,
                                               analysis_result: Dict[str, Any],
                                               document_features: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance document analysis accuracy with validation and confidence scoring"""
        
        # Check idempotency
        operation_id = self.idempotency_manager.generate_operation_id(
            "document_analysis",
            {"df_hash": df_hash, "filename": filename}
        )
        
        cached_result = self.idempotency_manager.get_cached_result(operation_id)
        if cached_result:
            logger.info(f"Returning cached analysis result for {operation_id}")
            return cached_result
        
        # Calculate confidence score
        confidence_score = self.confidence_calculator.calculate_document_analysis_confidence(
            analysis_result, document_features
        )
        
        # Enhance result with accuracy information
        enhanced_result = analysis_result.copy()
        enhanced_result.update({
            'accuracy_enhancement': {
                'confidence_score': confidence_score.overall,
                'confidence_breakdown': confidence_score.breakdown,
                'confidence_factors': confidence_score.factors,
                'document_features': document_features,
                'timestamp': confidence_score.timestamp.isoformat(),
                'operation_id': operation_id
            }
        })
        
        # Cache result for idempotency
        self.idempotency_manager.cache_result(operation_id, enhanced_result)
        
        # Log operation
        self.operation_history.append({
            'operation_id': operation_id,
            'operation_type': 'document_analysis',
            'confidence_score': confidence_score.overall,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        return enhanced_result
    
    def get_accuracy_statistics(self) -> Dict[str, Any]:
        """Get accuracy enhancement statistics"""
        if not self.operation_history:
            return {
                'total_operations': 0,
                'average_confidence': 0.0,
                'confidence_distribution': {},
                'operation_types': {}
            }
        
        # Calculate statistics
        total_operations = len(self.operation_history)
        confidence_scores = [op['confidence_score'] for op in self.operation_history]
        average_confidence = sum(confidence_scores) / len(confidence_scores)
        
        # Confidence distribution
        confidence_distribution = {
            'very_high': sum(1 for score in confidence_scores if score >= 0.9),
            'high': sum(1 for score in confidence_scores if 0.8 <= score < 0.9),
            'medium': sum(1 for score in confidence_scores if 0.6 <= score < 0.8),
            'low': sum(1 for score in confidence_scores if 0.4 <= score < 0.6),
            'very_low': sum(1 for score in confidence_scores if score < 0.4)
        }
        
        # Operation types
        operation_types = {}
        for op in self.operation_history:
            op_type = op['operation_type']
            if op_type not in operation_types:
                operation_types[op_type] = {'count': 0, 'avg_confidence': 0.0}
            operation_types[op_type]['count'] += 1
        
        # Calculate average confidence per operation type
        for op_type in operation_types:
            type_scores = [op['confidence_score'] for op in self.operation_history 
                          if op['operation_type'] == op_type]
            operation_types[op_type]['avg_confidence'] = sum(type_scores) / len(type_scores)
        
        return {
            'total_operations': total_operations,
            'average_confidence': average_confidence,
            'confidence_distribution': confidence_distribution,
            'operation_types': operation_types,
            'cache_size': len(self.idempotency_manager.operation_cache)
        }
    
    def cleanup_expired_cache(self) -> int:
        """Clean up expired cache entries"""
        return self.idempotency_manager.clear_expired_cache()

# Global accuracy enhancement system instance
_global_accuracy_system: Optional[AccuracyEnhancementSystem] = None

def get_global_accuracy_system() -> AccuracyEnhancementSystem:
    """Get or create global accuracy enhancement system"""
    global _global_accuracy_system
    
    if _global_accuracy_system is None:
        _global_accuracy_system = AccuracyEnhancementSystem()
    
    return _global_accuracy_system
