# Claude API Integration Guide

## 🎯 Overview

This document describes the Claude API integration implemented in the Finley AI backend, providing automatic fallback between OpenAI and Claude providers for enhanced reliability and performance.

## 🏗️ Architecture

### **Unified AI Call System**

The system uses a unified interface that automatically handles:
- **Primary Provider**: OpenAI (gpt-4o-mini)
- **Fallback Provider**: Claude (claude-3-opus-20240229)
- **Automatic Switching**: When primary provider fails
- **Error Handling**: Graceful degradation with fallback results

### **Key Components**

1. **`try_openai_request()`** - OpenAI-specific API calls
2. **`try_claude_request()`** - Claude-specific API calls  
3. **`unified_ai_call()`** - Main orchestration function
4. **`safe_openai_call()`** - Legacy compatibility wrapper

## 🔧 Implementation Details

### **Environment Variables**

```bash
# Required for OpenAI
OPENAI_API_KEY=your_openai_key_here

# Required for Claude fallback
CLAUDE_API_KEY=your_claude_key_here
```

### **Client Initialization**

```python
# Global clients initialized at startup
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
claude_client = anthropic.Anthropic(api_key=os.getenv('CLAUDE_API_KEY'))
```

### **Usage Example**

```python
# Unified AI call with automatic fallback
result = await unified_ai_call(
    openai_client,           # OpenAI client
    claude_client,           # Claude client
    "openai",                # Primary provider preference
    [{"role": "user", "content": "Your message"}],  # Messages
    0.1,                     # Temperature
    200,                     # Max tokens
    "Fallback response"      # Fallback if both fail
)
```

## 🔄 Fallback Logic

### **Primary Provider (OpenAI)**
1. Attempt OpenAI API call
2. If successful → Return result
3. If quota/rate limit error → Try Claude
4. If other error → Try Claude

### **Fallback Provider (Claude)**
1. Convert OpenAI message format to Claude format
2. Attempt Claude API call
3. If successful → Return result
4. If fails → Return fallback result

### **Error Handling**

```python
# Quota/Rate Limit Detection
if any(keyword in str(e).lower() for keyword in 
       ['429', 'quota', 'insufficient_quota', 'rate_limit']):
    # Try fallback provider
```

## 📊 Provider Comparison

| Feature | OpenAI | Claude |
|---------|--------|--------|
| **Model** | gpt-4o-mini | claude-3-opus-20240229 |
| **Speed** | Fast | Slower but more accurate |
| **Cost** | Lower | Higher |
| **Reasoning** | Good | Excellent |
| **Fallback** | Primary | Secondary |

## 🚀 Deployment

### **Render Environment Variables**

1. Go to your Render dashboard
2. Navigate to your service
3. Go to Environment tab
4. Add the following variables:

```bash
OPENAI_API_KEY=sk-...
CLAUDE_API_KEY=sk-ant-...
```

### **Requirements Update**

The `anthropic` library has been added to `requirements.txt`:

```txt
anthropic==0.7.8
```

### **Deployment Steps**

1. ✅ Environment variables configured
2. ✅ Requirements.txt updated
3. ✅ Code deployed
4. ✅ Test the integration

## 🧪 Testing

### **Run Integration Tests**

```bash
python test_claude_integration.py
```

### **Test Scenarios**

1. **Normal Operation**: Both providers working
2. **OpenAI Failure**: Claude fallback
3. **Claude Failure**: OpenAI primary
4. **Both Failure**: Fallback response
5. **No Providers**: Error handling

## 🔮 Extending to More Providers

### **Adding Gemini Support**

1. **Add Provider Function**:
```python
async def try_gemini_request(client, model, messages, temperature, max_tokens):
    # Gemini-specific implementation
    pass
```

2. **Update Unified Call**:
```python
# Add Gemini to fallback chain
if fallback_provider == "gemini":
    fallback_result = await try_gemini_request(...)
```

3. **Add Environment Variable**:
```python
gemini_api_key = os.getenv('GEMINI_API_KEY')
```

4. **Update Requirements**:
```txt
google-generativeai==0.3.0
```

### **Provider Priority System**

```python
# Configurable provider priority
PROVIDER_PRIORITY = ["openai", "claude", "gemini"]
```

## 📈 Monitoring & Logging

### **Log Messages**

```python
# Success
logger.info(f"AI request successful using {provider} ({model})")

# Fallback
logger.warning(f"Primary provider ({provider}) failed: {error}")
logger.info(f"Attempting fallback to {fallback_provider}...")

# Complete failure
logger.error(f"All AI providers failed. Primary: {error1}, Fallback: {error2}")
```

### **Metrics to Monitor**

- Provider success rates
- Fallback frequency
- Response times per provider
- Error rates by provider

## 🛡️ Security Considerations

### **API Key Management**

- ✅ Keys stored in environment variables
- ✅ No hardcoded credentials
- ✅ Proper error handling for missing keys

### **Rate Limiting**

- ✅ Automatic retry with exponential backoff
- ✅ Graceful degradation on limits
- ✅ Fallback to alternative provider

## 🔧 Troubleshooting

### **Common Issues**

1. **"No AI providers available"**
   - Check environment variables
   - Verify API keys are valid

2. **"Claude fallback disabled"**
   - Install anthropic library
   - Check CLAUDE_API_KEY

3. **"All providers failed"**
   - Check API quotas
   - Verify network connectivity
   - Check API key validity

### **Debug Mode**

```python
# Enable detailed logging
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## 📚 API Reference

### **unified_ai_call()**

```python
async def unified_ai_call(
    openai_client,           # OpenAI client instance
    claude_client,           # Claude client instance
    model_preference,        # "openai" or "claude"
    messages,                # List of message dicts
    temperature=0.1,         # Generation temperature
    max_tokens=200,          # Max tokens to generate
    fallback_result=None     # Fallback if all fail
) -> str:                    # Returns AI response content
```

### **try_openai_request()**

```python
async def try_openai_request(
    client,                  # OpenAI client
    model,                   # Model name
    messages,                # Message list
    temperature=0.1,         # Temperature
    max_tokens=200           # Max tokens
) -> dict:                   # Returns success/error dict
```

### **try_claude_request()**

```python
async def try_claude_request(
    client,                  # Claude client
    model,                   # Model name
    messages,                # Message list
    temperature=0.1,         # Temperature
    max_tokens=200           # Max tokens
) -> dict:                   # Returns success/error dict
```

## 🎉 Benefits

1. **Enhanced Reliability**: Automatic fallback prevents service outages
2. **Better Performance**: Choose optimal provider per use case
3. **Cost Optimization**: Use cheaper provider when possible
4. **Future-Proof**: Easy to add more providers
5. **Seamless Integration**: No frontend changes required

---

**Status**: ✅ Production Ready  
**Last Updated**: January 2025  
**Version**: 1.0.0
