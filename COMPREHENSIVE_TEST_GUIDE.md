# Comprehensive Test Guide for Finley AI Platform

## Overview
This guide provides comprehensive testing instructions for all Finley AI platform features, including the new AI-powered relationship detection system.

## Prerequisites
1. **Environment Setup**: Ensure all environment variables are configured
2. **Database**: Supabase instance with proper tables and migrations
3. **API Keys**: OpenAI API key for AI features
4. **Test Data**: Sample financial files for testing

## Test Categories

### 1. Basic System Tests
- **Health Check**: Verify API connectivity
- **Environment Debug**: Check configuration
- **Database Connection**: Test Supabase connectivity

### 2. Core Processing Tests
- **Platform Detection**: Test multi-platform recognition
- **AI Classification**: Test intelligent row classification
- **Batch Processing**: Test bulk data processing
- **Entity Resolution**: Test entity matching and linking

### 3. Data Enrichment Tests
- **Currency Normalization**: Test currency detection and conversion
- **Vendor Standardization**: Test vendor name cleaning
- **Platform ID Extraction**: Test ID extraction from various platforms
- **Data Enrichment Pipeline**: Test complete enrichment workflow

### 4. **NEW: AI-Powered Relationship Detection Tests** 🚀

#### **27. Test AI Relationship Detection**
**Purpose**: Test comprehensive AI-powered relationship detection for ANY financial data

**What it tests**:
- Universal relationship detection across all platforms
- AI-powered relationship type discovery
- Multi-dimensional relationship scoring
- Dynamic pattern learning from user data
- Relationship validation and confidence scoring

**Expected Response**:
```json
{
  "message": "AI Relationship Detection Test Completed",
  "result": {
    "relationships": [
      {
        "source_event_id": "uuid",
        "target_event_id": "uuid", 
        "relationship_type": "invoice_to_payment",
        "confidence_score": 0.85,
        "source_platform": "stripe",
        "target_platform": "razorpay",
        "amount_match": true,
        "date_match": true,
        "entity_match": true,
        "detection_method": "ai_discovered"
      }
    ],
    "total_relationships": 15,
    "relationship_types": ["invoice_to_payment", "fee_to_transaction", "refund_to_original"],
    "ai_discovered_count": 8,
    "message": "Comprehensive AI-powered relationship analysis completed"
  }
}
```

#### **28. Test Relationship Type Discovery**
**Purpose**: Test AI-powered discovery of relationship types in financial data

**What it tests**:
- AI analysis of financial events to identify relationship patterns
- Dynamic discovery of new relationship types
- Platform-agnostic relationship type detection
- Context-aware relationship classification

**Expected Response**:
```json
{
  "message": "Relationship Type Discovery Test Completed",
  "discovered_types": [
    "invoice_to_payment",
    "fee_to_transaction", 
    "refund_to_original",
    "payroll_to_payout",
    "tax_to_income",
    "expense_to_reimbursement"
  ],
  "total_events": 50
}
```

#### **29. Test AI Relationship Scoring**
**Purpose**: Test multi-dimensional relationship scoring with AI-powered analysis

**What it tests**:
- Amount matching with intelligent tolerance
- Date matching with configurable windows
- Entity matching with fuzzy logic
- ID matching with pattern recognition
- Context matching with semantic analysis
- Comprehensive scoring algorithm

**Expected Response**:
```json
{
  "message": "AI Relationship Scoring Test Completed",
  "scoring_results": [
    {
      "relationship_type": "invoice_to_payment",
      "comprehensive_score": 0.85,
      "amount_score": 1.0,
      "date_score": 0.8,
      "entity_score": 0.9,
      "id_score": 0.6,
      "context_score": 0.7,
      "event1_id": "uuid",
      "event2_id": "uuid"
    }
  ]
}
```

#### **30. Test Relationship Validation**
**Purpose**: Test relationship validation and filtering with confidence scoring

**What it tests**:
- Relationship existence validation
- Confidence score filtering
- Event validation in database
- Relationship metadata validation
- Quality assurance for discovered relationships

**Expected Response**:
```json
{
  "message": "Relationship Validation Test Completed",
  "total_relationships": 10,
  "validated_relationships": 8,
  "validation_results": [
    {
      "source_event_id": "uuid",
      "target_event_id": "uuid",
      "relationship_type": "test_relationship",
      "confidence_score": 0.8,
      "validated": true,
      "validation_score": 0.8
    }
  ]
}
```

### **Key Features of AI Relationship Detection** 🧠

#### **1. Universal Relationship Types**
- **Invoice ↔ Payment** (any platform)
- **Fee ↔ Transaction** (any platform)
- **Refund ↔ Original** (any platform)
- **Payroll ↔ Payout** (any platform)
- **Tax ↔ Income** (any platform)
- **Expense ↔ Reimbursement** (any platform)
- **Subscription ↔ Payment** (any platform)
- **Loan ↔ Payment** (any platform)
- **Investment ↔ Return** (any platform)
- **Custom relationships** discovered by AI

#### **2. Multi-Dimensional Scoring**
- **Amount Matching** (30% weight): Intelligent tolerance per relationship type
- **Date Matching** (20% weight): Configurable windows (1 day to 1 year)
- **Entity Matching** (20% weight): Fuzzy logic with semantic similarity
- **ID Matching** (15% weight): Pattern recognition for platform IDs
- **Context Matching** (15% weight): Semantic analysis of descriptions

#### **3. AI-Powered Discovery**
- **Dynamic Relationship Types**: AI discovers new relationship patterns
- **Pattern Learning**: Learns from user's historical data
- **Context Awareness**: Understands financial context and terminology
- **Platform Agnostic**: Works with any financial platform or data source

#### **4. Intelligent Tolerance**
- **Invoice to Payment**: Exact amount match (0.001 tolerance)
- **Fee to Transaction**: 1% tolerance
- **Refund to Original**: Exact amount match
- **Payroll to Payout**: 5% tolerance
- **Tax to Income**: 2% tolerance
- **Investment to Return**: 10% tolerance

#### **5. Configurable Date Windows**
- **Invoice to Payment**: 7 days
- **Fee to Transaction**: Same day
- **Refund to Original**: 30 days
- **Payroll to Payout**: 3 days
- **Tax to Income**: 90 days
- **Investment to Return**: 1 year

### **Testing Strategy** 📋

#### **Phase 1: Basic Functionality**
1. Run **Test 27** to verify AI relationship detection works
2. Check for any errors in relationship discovery
3. Verify that relationships are being detected

#### **Phase 2: Relationship Discovery**
1. Run **Test 28** to test relationship type discovery
2. Verify AI is discovering appropriate relationship types
3. Check for new relationship types not in predefined list

#### **Phase 3: Scoring Analysis**
1. Run **Test 29** to test relationship scoring
2. Analyze scoring results for different relationship types
3. Verify scoring logic is working correctly

#### **Phase 4: Validation**
1. Run **Test 30** to test relationship validation
2. Verify relationships are being properly validated
3. Check confidence scores are reasonable

#### **Phase 5: Integration Testing**
1. Upload sample financial files
2. Run complete relationship detection
3. Verify relationships are discovered and stored
4. Test with different platforms and data types

### **Expected Behaviors** ✅

#### **For Invoice to Payment Relationships**:
- Should detect when invoice amount matches payment amount
- Should consider date proximity (within 7 days)
- Should match vendor/entity names
- Should have high confidence for exact matches

#### **For Fee to Transaction Relationships**:
- Should detect fee amounts within transaction amounts
- Should match same-day transactions
- Should consider platform-specific patterns

#### **For AI-Discovered Relationships**:
- Should find relationships not in predefined types
- Should provide confidence scores for new relationships
- Should learn from user's data patterns

### **Troubleshooting** 🔧

#### **Common Issues**:
1. **No relationships found**: Check if user has sufficient data
2. **Low confidence scores**: Verify data quality and completeness
3. **AI errors**: Check OpenAI API key and connectivity
4. **Database errors**: Verify Supabase connection and permissions

#### **Debug Steps**:
1. Run environment debug test first
2. Check database connectivity
3. Verify OpenAI API key is working
4. Test with sample data if needed

### **Performance Considerations** ⚡

#### **Optimization Features**:
- **Caching**: Relationship patterns are cached for performance
- **Batch Processing**: Multiple relationships processed together
- **Intelligent Filtering**: Only high-confidence relationships stored
- **Lazy Loading**: Relationships computed on-demand

#### **Scalability**:
- **Platform Agnostic**: Works with any financial platform
- **Dynamic Discovery**: Adapts to new relationship types
- **Configurable Thresholds**: Adjustable confidence levels
- **Efficient Algorithms**: Optimized for large datasets

## Advanced Testing

### Cross-Platform Testing
Test with data from multiple platforms:
- Stripe payments with Razorpay invoices
- QuickBooks payroll with bank statements
- Multiple payment processors
- Different currencies and formats

### Edge Cases
- Missing data fields
- Inconsistent date formats
- Currency conversion issues
- Duplicate transactions
- Partial matches

### Performance Testing
- Large datasets (1000+ events)
- Multiple relationship types
- Concurrent processing
- Memory usage optimization

## Conclusion

The AI-powered relationship detection system provides a **completely flexible and extensible** solution for discovering ANY type of financial relationship across different platforms and data sources. It uses advanced AI techniques to:

1. **Discover new relationship types** automatically
2. **Score relationships** using multiple dimensions
3. **Learn from user data** to improve over time
4. **Handle any financial platform** or data format
5. **Provide confidence scores** for relationship quality

This system eliminates the limitations of hardcoded relationship types and provides a truly intelligent solution for early-stage finance teams who need to understand complex financial relationships across their entire data ecosystem. 