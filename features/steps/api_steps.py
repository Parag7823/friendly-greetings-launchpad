"""
API-specific step definitions for Finley AI BDD tests
"""
from behave import given, when, then, step
import json
import os
from pathlib import Path

@given('I have a test Excel file')
def step_have_test_excel_file(context):
    """Set up a test Excel file"""
    # Create a simple test file path
    context.test_file_path = "test_files/test_data.xlsx"
    context.scenario_data['file_path'] = context.test_file_path

@given('I have uploaded a file with {file_type} data')
def step_have_uploaded_file(context, file_type):
    """Simulate having uploaded a file"""
    context.scenario_data['file_type'] = file_type
    context.scenario_data['file_uploaded'] = True

@when('I upload the file to the processing endpoint')
def step_upload_file(context):
    """Upload file to the processing endpoint"""
    file_path = context.scenario_data.get('file_path', 'test_files/test_data.xlsx')
    
    # Check if file exists, if not create a mock upload
    if not os.path.exists(file_path):
        # For testing, we'll simulate a file upload
        files = {'file': ('test_data.xlsx', b'mock excel content', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
        data = {'user_id': 'test_user_123'}
        context.response = context.client.post("/upload-and-process", files=files, data=data)
    else:
        with open(file_path, 'rb') as f:
            files = {'file': (file_path, f, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
            data = {'user_id': 'test_user_123'}
            context.response = context.client.post("/upload-and-process", files=files, data=data)

@when('I send a chat message {message}')
def step_send_chat_message(context, message):
    """Send a chat message"""
    chat_data = {
        "message": message,
        "user_id": "test_user_123",
        "chat_id": context.scenario_data.get('chat_id', 'test_chat_123')
    }
    context.request_data = chat_data
    context.response = context.client.post("/chat", json=chat_data)

@when('I request chat history for user {user_id}')
def step_request_chat_history(context, user_id):
    """Request chat history for a user"""
    context.response = context.client.get(f"/chat-history/{user_id}")

@when('I rename chat {chat_id} to {new_title}')
def step_rename_chat(context, chat_id, new_title):
    """Rename a chat"""
    rename_data = {
        "chat_id": chat_id,
        "new_title": new_title,
        "user_id": "test_user_123"
    }
    context.request_data = rename_data
    context.response = context.client.put("/chat/rename", json=rename_data)

@when('I delete chat {chat_id}')
def step_delete_chat(context, chat_id):
    """Delete a chat"""
    delete_data = {
        "chat_id": chat_id,
        "user_id": "test_user_123"
    }
    context.request_data = delete_data
    context.response = context.client.request("DELETE", "/chat/delete", json=delete_data)

@given('I have a chat ID {chat_id}')
def step_have_chat_id(context, chat_id):
    """Set up a chat ID for testing"""
    context.scenario_data['chat_id'] = chat_id

@given('I have sent messages to chat {chat_id}')
def step_have_sent_messages(context, chat_id):
    """Simulate having sent messages to a chat"""
    context.scenario_data['chat_id'] = chat_id
    context.scenario_data['messages_sent'] = True


@then('the file should be processed successfully')
def step_file_processed_successfully(context):
    """Verify file processing was successful"""
    assert context.response is not None, "No response received"
    assert context.response.status_code == 200, f"File processing failed: {context.response.text}"
    
    data = context.response.json()
    assert 'message' in data, "No success message in response"
    assert 'processing_id' in data, "No processing ID returned"

@then('I should receive a chat response')
def step_should_receive_chat_response(context):
    """Verify a chat response was received"""
    assert context.response is not None, "No response received"
    assert context.response.status_code == 200, f"Chat request failed: {context.response.text}"
    
    data = context.response.json()
    assert 'response' in data, "No chat response in data"
    assert 'chat_id' in data, "No chat ID in response"

@then('the chat should be saved with ID {chat_id}')
def step_chat_saved_with_id(context, chat_id):
    """Verify chat was saved with specific ID"""
    data = context.response.json()
    assert data.get('chat_id') == chat_id, f"Expected chat_id {chat_id}, got {data.get('chat_id')}"
    context.scenario_data['chat_id'] = chat_id

@then('the processing should include {feature}')
def step_processing_should_include(context, feature):
    """Verify processing includes specific feature"""
    data = context.response.json()
    
    if feature == "entity resolution":
        assert 'entities' in data or 'normalized_entities' in data, "Entity resolution not included"
    elif feature == "relationship detection":
        assert 'relationships' in data or 'relationship_instances' in data, "Relationship detection not included"
    elif feature == "AI classification":
        assert 'classification' in data or 'ai_analysis' in data, "AI classification not included"
    elif feature == "data normalization":
        assert 'normalized_data' in data or 'processed_data' in data, "Data normalization not included"

@then('the response should contain "chats"')
def step_response_should_contain_chats(context):
    """Verify the response contains chats"""
    assert context.response is not None, "No response received"
    data = context.response.json()
    assert 'chats' in data, "No 'chats' field in response"
