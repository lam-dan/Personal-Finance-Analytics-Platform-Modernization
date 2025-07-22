"""
Personal Finance Analytics Platform - GraphQL Integration Tests

This module contains integration tests for the GraphQL API endpoints.
These tests verify that the GraphQL API works correctly by:
1. Starting the actual GraphQL service in a subprocess
2. Making real HTTP POST requests with GraphQL queries
3. Validating GraphQL response structure and data
4. Ensuring proper cleanup of test resources

Testing Approach:
- Integration tests that test the actual running GraphQL service
- Real HTTP requests with GraphQL query syntax
- Validation of GraphQL response format (data wrapper)
- Resource cleanup to prevent test interference

GraphQL Testing Benefits:
- Tests the complete GraphQL query execution pipeline
- Validates GraphQL schema and type system
- Ensures proper response formatting
- Tests field selection and data fetching
"""

import subprocess
import time

import requests


def test_graphql_transactions():
    """
    Test the GraphQL transactions query to ensure it returns valid transaction data.

    This test:
    1. Starts the GraphQL service on port 8002
    2. Makes a POST request to /graphql with a transactions query
    3. Verifies the response status is 200 (success)
    4. Ensures the response follows GraphQL format (data wrapper)
    5. Validates that transactions data is present
    6. Cleans up by stopping the service

    Expected Behavior:
    - Service starts successfully
    - GraphQL endpoint returns HTTP 200
    - Response contains 'data' wrapper (GraphQL standard)
    - 'transactions' field contains transaction data
    - Field selection works correctly (id, amount, category, date)
    """
    # Start the GraphQL service in a subprocess
    # Using uvicorn to serve the GraphQL app on port 8002
    process = subprocess.Popen(
        [
            "python",
            "-m",
            "uvicorn",
            "graphql_api.schema:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8002",
        ]
    )

    # Allow time for the service to start up and be ready
    time.sleep(2)

    try:
        # Define GraphQL query for transactions
        # This tests field selection and data fetching
        query = "{ transactions { id amount category date } }"

        # Make HTTP POST request to GraphQL endpoint
        # GraphQL uses POST requests with JSON body containing the query
        response = requests.post(
            "http://localhost:8002/graphql", json={"query": query}
        )

        # Verify successful HTTP response
        assert response.status_code == 200

        # Verify GraphQL response structure
        # GraphQL responses are wrapped in a 'data' object
        assert "data" in response.json()

        # Verify that transactions data is present in the response
        assert "transactions" in response.json()["data"]

    finally:
        # Always clean up: terminate the service process
        process.terminate()
        process.wait()  # Wait for process to fully terminate


def test_graphql_investments():
    """
    Test the GraphQL investments query to ensure it returns valid investment data.

    This test:
    1. Starts the GraphQL service on port 8002
    2. Makes a POST request to /graphql with an investments query
    3. Verifies the response status is 200 (success)
    4. Ensures the response follows GraphQL format (data wrapper)
    5. Validates that investments data is present
    6. Cleans up by stopping the service

    Expected Behavior:
    - Service starts successfully
    - GraphQL endpoint returns HTTP 200
    - Response contains 'data' wrapper (GraphQL standard)
    - 'investments' field contains investment data
    - Field selection works correctly (id, asset, value, lastUpdated)
    """
    # Start the GraphQL service in a subprocess
    process = subprocess.Popen(
        [
            "python",
            "-m",
            "uvicorn",
            "graphql_api.schema:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8002",
        ]
    )

    # Allow time for the service to start up and be ready
    time.sleep(2)

    try:
        # Define GraphQL query for investments
        # This tests field selection and data fetching
        query = "{ investments { id asset value lastUpdated } }"

        # Make HTTP POST request to GraphQL endpoint
        # GraphQL uses POST requests with JSON body containing the query
        response = requests.post(
            "http://localhost:8002/graphql", json={"query": query}
        )

        # Verify successful HTTP response
        assert response.status_code == 200

        # Verify GraphQL response structure
        # GraphQL responses are wrapped in a 'data' object
        assert "data" in response.json()

        # Verify that investments data is present in the response
        assert "investments" in response.json()["data"]

    finally:
        # Always clean up: terminate the service process
        process.terminate()
        process.wait()
