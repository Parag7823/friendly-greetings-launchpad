Feature: Chat Functionality
  As a user
  I want to chat with the AI about my financial data
  So that I can get insights and answers

  Background:
    Given I have a valid API client
    And the API is running

  Scenario: Send a chat message
    Given I have a chat ID "test_chat_123"
    When I send a chat message "What are my top expenses?"
    Then I should receive a 200 status code
    And I should receive a chat response
    And the chat should be saved with ID "test_chat_123"

  Scenario: Request chat history
    Given I have sent messages to chat "test_chat_123"
    When I request chat history for user "test_user_123"
    Then I should receive a 200 status code
    And the response should contain "chats"

  Scenario: Rename a chat
    Given I have a chat ID "test_chat_123"
    When I rename chat "test_chat_123" to "Expense Analysis"
    Then I should receive a 200 status code
    And the response should contain the message "Chat renamed successfully"

  Scenario: Delete a chat
    Given I have a chat ID "test_chat_123"
    When I delete chat "test_chat_123"
    Then I should receive a 200 status code
    And the response should contain the message "Chat deleted successfully"

  Scenario: Handle empty chat message
    When I send a chat message ""
    Then I should receive a 400 status code
    And I should see an error message
