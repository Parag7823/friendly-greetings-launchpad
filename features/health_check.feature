Feature: Health Check
  As a developer
  I want to verify the API is running
  So that I can ensure the service is operational

  Background:
    Given I have a valid API client

  Scenario: API health check
    When I make a GET request to /health
    Then I should receive a 200 status code
    And the response should contain the key status
    And the response should contain the message "healthy"

  Scenario: API root endpoint
    When I make a GET request to /
    Then I should receive a 200 status code
    And the response should contain HTML content
