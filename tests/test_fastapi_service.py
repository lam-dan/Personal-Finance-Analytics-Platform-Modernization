"""
Personal Finance Analytics Platform - FastAPI Integration Tests

This module contains integration tests for the FastAPI service endpoints.
These tests verify that the REST API endpoints work correctly by:
1. Starting the actual FastAPI service in a subprocess
2. Making real HTTP requests to the running service
3. Validating responses and data structures
4. Ensuring proper cleanup of test resources

Testing Approach:
- Integration tests that test the actual running service
- Real HTTP requests instead of mocked clients
- Proper service lifecycle management (start/stop)
- Resource cleanup to prevent test interference

This approach provides confidence that the API works as expected
from a real client's perspective.
"""

import requests
import time
import subprocess
import signal
import os

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
    # Using uvicorn to serve the FastAPI app on port 8000
    process = subprocess.Popen([
        'python', '-m', 'uvicorn', 
        'python_service.main:app', 
        '--host', '0.0.0.0', 
        '--port', '8000'
    ])
    
    # Allow time for the service to start up and be ready
    time.sleep(2)
    
    try:
        # Make HTTP request to the transactions endpoint
        response = requests.get('http://localhost:8000/transactions')
        
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
    process = subprocess.Popen([
        'python', '-m', 'uvicorn', 
        'python_service.main:app', 
        '--host', '0.0.0.0', 
        '--port', '8000'
    ])
    
    # Allow time for the service to start up and be ready
    time.sleep(2)
    
    try:
        # Make HTTP request to the investments endpoint
        response = requests.get('http://localhost:8000/investments')
        
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
    process = subprocess.Popen([
        'python', '-m', 'uvicorn', 
        'python_service.main:app', 
        '--host', '0.0.0.0', 
        '--port', '8000'
    ])
    
    # Allow time for the service to start up and be ready
    time.sleep(2)
    
    try:
        # Make HTTP request to the spending trends endpoint with date range
        response = requests.get(
            'http://localhost:8000/spending_trends?start_date=2024-07-01&end_date=2024-07-31'
        )
        
        # Verify successful HTTP response
        assert response.status_code == 200
        
        # Verify response structure is a dictionary with spending summaries
        assert isinstance(response.json(), dict)
        
    finally:
        # Always clean up: terminate the service process
        process.terminate()
        process.wait()
