"""
Personal Finance Analytics Platform - FastAPI Integration Tests

This module contains integration tests for the FastAPI service endpoints.
These tests verify that the FastAPI service works correctly by:
1. Starting the actual FastAPI service in a subprocess
2. Making real HTTP requests to the service endpoints
3. Validating HTTP response status codes and data structure
4. Ensuring proper cleanup of test resources

Testing Approach:
- Integration tests that test the actual running FastAPI service
- Real HTTP requests to verify endpoint functionality
- Validation of response status codes and data formats
- Resource cleanup to prevent test interference

FastAPI Testing Benefits:
- Tests the complete HTTP request/response pipeline
- Validates endpoint routing and request handling
- Ensures proper JSON response formatting
- Tests parameter handling and validation
"""

import subprocess
import time

import requests


def test_get_transactions():
    """
    Test the /transactions endpoint to ensure it returns valid transaction data.

    This test:
    1. Starts the FastAPI service on port 8000
    2. Makes a GET request to /transactions
    3. Verifies the response status is 200 (success)
    4. Ensures the response is a list of transactions
    5. Cleans up by stopping the service

    Expected Behavior:
    - Service starts successfully
    - Endpoint returns HTTP 200
    - Response contains a list of transaction objects
    """
    # Start the FastAPI service in a subprocess
    process = subprocess.Popen(
        [
            "python",
            "-m",
            "uvicorn",
            "python_service.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ]
    )

    # Allow time for the service to start up and be ready
    time.sleep(2)

    try:
        # Make HTTP request to the transactions endpoint
        response = requests.get("http://localhost:8000/transactions")

        # Verify successful HTTP response
        assert response.status_code == 200

        # Verify response structure is a list
        assert isinstance(response.json(), list)

    finally:
        # Always clean up: terminate the service process
        process.terminate()
        process.wait()  # Wait for process to fully terminate


def test_get_investments():
    """
    Test the /investments endpoint to ensure it returns valid investment data.

    This test:
    1. Starts the FastAPI service on port 8000
    2. Makes a GET request to /investments
    3. Verifies the response status is 200 (success)
    4. Ensures the response is a list of investments
    5. Cleans up by stopping the service

    Expected Behavior:
    - Service starts successfully
    - Endpoint returns HTTP 200
    - Response contains a list of investment objects
    """
    # Start the FastAPI service in a subprocess
    process = subprocess.Popen(
        [
            "python",
            "-m",
            "uvicorn",
            "python_service.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ]
    )

    # Allow time for the service to start up and be ready
    time.sleep(2)

    try:
        # Make HTTP request to the investments endpoint
        response = requests.get("http://localhost:8000/investments")

        # Verify successful HTTP response
        assert response.status_code == 200

        # Verify response structure is a list
        assert isinstance(response.json(), list)

    finally:
        # Always clean up: terminate the service process
        process.terminate()
        process.wait()


def test_spending_trends():
    """
    Test the /spending_trends endpoint with date range parameters.

    This test:
    1. Starts the FastAPI service on port 8000
    2. Makes a GET request to /spending_trends with date parameters
    3. Verifies the response status is 200 (success)
    4. Ensures the response is a dictionary with spending summaries
    5. Cleans up by stopping the service

    Expected Behavior:
    - Service starts successfully
    - Endpoint returns HTTP 200
    - Response contains spending analysis by category
    - Date filtering works correctly
    """
    # Start the FastAPI service in a subprocess
    process = subprocess.Popen(
        [
            "python",
            "-m",
            "uvicorn",
            "python_service.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ]
    )

    # Allow time for the service to start up and be ready
    time.sleep(2)

    try:
        # Make HTTP request to the spending trends endpoint with date range
        response = requests.get(
            "http://localhost:8000/spending_trends?"
            "start_date=2024-07-01&end_date=2024-07-31"
        )

        # Verify successful HTTP response
        assert response.status_code == 200

        # Verify response structure is a dictionary with spending summaries
        assert isinstance(response.json(), dict)

    finally:
        # Always clean up: terminate the service process
        process.terminate()
        process.wait()
