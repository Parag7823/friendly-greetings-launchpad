"""
Common step definitions for Finley AI BDD tests
"""
from behave import given, when, then, step
import json
import time
from pathlib import Path

@given('I have a valid API client')
def step_valid_api_client(context):
    """Ensure we have a valid API client"""
    assert hasattr(context, 'client'), "Test client not initialized"
    context.base_url = "http://testserver"

@given('the API is running')
def step_api_running(context):
    """Verify the API is running and accessible"""
    response = context.client.get("/health")
    assert response.status_code == 200, f"API not running: {response.status_code}"
    context.api_healthy = True

@when('I make a {method} request to {endpoint}')
def step_make_request(context, method, endpoint):
    """Make a request to the specified endpoint"""
    method = method.upper()
    url = f"{context.base_url}{endpoint}"
    
    if method == "GET":
        context.response = context.client.get(endpoint)
    elif method == "POST":
        data = getattr(context, 'request_data', {})
        context.response = context.client.post(endpoint, json=data)
    elif method == "PUT":
        data = getattr(context, 'request_data', {})
        context.response = context.client.put(endpoint, json=data)
    elif method == "DELETE":
        context.response = context.client.delete(endpoint)
    else:
        raise ValueError(f"Unsupported HTTP method: {method}")

@when('I send the following JSON data')
def step_send_json_data(context):
    """Send JSON data from the step text"""
    context.request_data = json.loads(context.text)

@when('I wait for {seconds:d} seconds')
def step_wait_seconds(context, seconds):
    """Wait for specified number of seconds"""
    time.sleep(seconds)

@then('I should receive a {status_code:d} status code')
def step_should_receive_status(context, status_code):
    """Verify the response status code"""
    assert context.response is not None, "No response received"
    assert context.response.status_code == status_code, \
        f"Expected {status_code}, got {context.response.status_code}: {context.response.text}"

@then('the response should contain the key {key}')
def step_response_should_contain_key(context, key):
    """Verify the response contains a specific key"""
    assert context.response is not None, "No response received"
    data = context.response.json()
    assert key in data, f"Key '{key}' not found in response: {data}"

@then('the response should contain the message "{message}"')
def step_response_should_contain_message(context, message):
    """Verify the response contains a specific message"""
    assert context.response is not None, "No response received"
    data = context.response.json()
    assert 'message' in data, "No 'message' field in response"
    assert message in data['message'], f"Expected '{message}' in '{data['message']}'"

@then('the response should be valid JSON')
def step_response_valid_json(context):
    """Verify the response is valid JSON"""
    assert context.response is not None, "No response received"
    try:
        context.response.json()
    except json.JSONDecodeError as e:
        assert False, f"Response is not valid JSON: {e}"

@then('I should see an error message')
def step_should_see_error(context):
    """Verify an error message is present"""
    assert context.response is not None, "No response received"
    assert context.response.status_code >= 400, "Expected an error status code"
    data = context.response.json()
    assert 'detail' in data or 'error' in data, "No error message in response"

@then('the response should contain HTML content')
def step_response_should_contain_html(context):
    """Verify the response contains HTML content"""
    assert context.response is not None, "No response received"
    content = context.response.text
    assert '<html' in content.lower() or '<!doctype' in content.lower(), "Response does not contain HTML content"
