Feature: File Processing
  As a user
  I want to upload and process financial files
  So that I can analyze my financial data

  Background:
    Given I have a valid API client
    And the API is running

  Scenario: Upload and process Excel file
    Given I have a test Excel file
    When I upload the file to the processing endpoint
    Then I should receive a 200 status code
    And the file should be processed successfully
    And the processing should include "entity resolution"
    And the processing should include "relationship detection"
    And the processing should include "AI classification"
    And the processing should include "data normalization"

  Scenario: Handle unsupported file type
    Given I have uploaded a file with "unsupported" data
    When I upload the file to the processing endpoint
    Then I should receive a 400 status code
    And I should see an error message

  Scenario: Process file with missing data
    Given I have uploaded a file with "empty" data
    When I upload the file to the processing endpoint
    Then I should receive a 400 status code
    And I should see an error message
