# 🧪 Finley AI Testing Guide

This guide covers the BDD (Behavior-Driven Development) testing setup for Finley AI using both Behave and pytest frameworks.

## 📁 Project Structure

```
friendly-greetings-launchpad/
├── features/                          # BDD feature files
│   ├── environment.py                 # Behave environment setup
│   ├── steps/                         # Step definitions
│   │   ├── common_steps.py           # Common step definitions
│   │   └── api_steps.py              # API-specific steps
│   ├── health_check.feature          # Health check scenarios
│   ├── file_processing.feature       # File processing scenarios
│   └── chat_functionality.feature    # Chat functionality scenarios
├── tests/                            # Unit tests
│   ├── __init__.py
│   └── test_api.py                   # API unit tests
├── behave.ini                        # Behave configuration
├── pytest.ini                       # Pytest configuration
├── run_tests.py                      # Test runner script
└── requirements.txt                  # Updated with BDD dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Tests

#### Using Behave (BDD)
```bash
# Run all BDD tests
behave

# Run specific feature
behave features/health_check.feature

# Run with tags
behave --tags @smoke

# Run with verbose output
behave --verbose
```

#### Using Pytest
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_api.py

# Run with markers
pytest -m "api"

# Run with coverage
pytest --cov=fastapi_backend
```

#### Using the Test Runner Script
```bash
# Run both frameworks
python run_tests.py

# Run only Behave
python run_tests.py --framework behave

# Run only Pytest
python run_tests.py --framework pytest

# Run specific features
python run_tests.py --features features/health_check.feature

# Run with verbose output
python run_tests.py --verbose
```

## 📋 Available Test Commands

### Behave Commands
- `behave` - Run all BDD tests
- `behave features/health_check.feature` - Run specific feature
- `behave --tags @smoke` - Run tests with specific tags
- `behave --verbose` - Verbose output
- `behave --format json` - JSON output format

### Pytest Commands
- `pytest` - Run all tests
- `pytest tests/` - Run unit tests only
- `pytest features/` - Run BDD tests via pytest
- `pytest -m "api"` - Run tests with specific markers
- `pytest --cov=fastapi_backend` - Run with coverage
- `pytest -v` - Verbose output

## 🎯 Test Categories

### BDD Features
1. **Health Check** (`health_check.feature`)
   - API health verification
   - Endpoint accessibility

2. **File Processing** (`file_processing.feature`)
   - Excel file upload and processing
   - Entity resolution testing
   - Relationship detection testing
   - AI classification testing

3. **Chat Functionality** (`chat_functionality.feature`)
   - Chat message sending
   - Chat history retrieval
   - Chat rename/delete operations

### Unit Tests
1. **API Endpoints** (`test_api.py`)
   - Health check endpoint
   - Chat endpoints
   - File processing endpoints

## 🔧 Configuration

### Behave Configuration (`behave.ini`)
- Format: Pretty output with colors
- Steps directory: `features/steps`
- Environment file: `features/environment.py`
- Logging level: INFO

### Pytest Configuration (`pytest.ini`)
- Test paths: `tests` and `features`
- BDD integration enabled
- Coverage reporting
- Async support enabled

## 📝 Writing New Tests

### Adding New BDD Scenarios

1. Create or edit a `.feature` file in `features/`
2. Write scenarios using Gherkin syntax
3. Implement step definitions in `features/steps/`

Example:
```gherkin
Feature: New Feature
  As a user
  I want to do something
  So that I can achieve a goal

  Scenario: Test something
    Given I have a valid API client
    When I make a GET request to /new-endpoint
    Then I should receive a 200 status code
```

### Adding New Unit Tests

1. Create or edit test files in `tests/`
2. Use pytest conventions
3. Mark tests with appropriate markers

Example:
```python
import pytest
from fastapi.testclient import TestClient

def test_new_endpoint():
    client = TestClient(app)
    response = client.get("/new-endpoint")
    assert response.status_code == 200
```

## 🏷️ Tags and Markers

### Behave Tags
- `@smoke` - Smoke tests
- `@api` - API tests
- `@slow` - Slow running tests

### Pytest Markers
- `unit` - Unit tests
- `integration` - Integration tests
- `e2e` - End-to-end tests
- `api` - API tests
- `bdd` - BDD tests

## 📊 Coverage Reports

Coverage reports are generated in HTML format:
```bash
pytest --cov=fastapi_backend --cov-report=html
```

Reports are saved to `htmlcov/index.html`

## 🐛 Debugging Tests

### Verbose Output
```bash
behave --verbose
pytest -v
```

### Debug Mode
```bash
pytest --pdb  # Drop into debugger on failure
```

### Logging
Tests use the configured logging level from `behave.ini` and `pytest.ini`

## 🔄 Continuous Integration

The test setup is ready for CI/CD integration:

```yaml
# Example GitHub Actions workflow
- name: Run Tests
  run: |
    pip install -r requirements.txt
    python run_tests.py --framework both
```

## 📚 Best Practices

1. **Write descriptive scenarios** - Use clear, business-focused language
2. **Keep steps reusable** - Create common step definitions
3. **Use appropriate tags** - Mark tests for different environments
4. **Test edge cases** - Include error scenarios and boundary conditions
5. **Maintain test data** - Keep test data clean and consistent
6. **Document test scenarios** - Add comments for complex test logic

## 🆘 Troubleshooting

### Common Issues

1. **Import errors**: Ensure project root is in Python path
2. **Missing dependencies**: Run `pip install -r requirements.txt`
3. **Test client issues**: Check FastAPI app initialization
4. **Environment variables**: Set required env vars for testing

### Getting Help

- Check the logs for detailed error messages
- Use verbose output to see step-by-step execution
- Review the step definitions for correct implementation
- Ensure all required dependencies are installed
