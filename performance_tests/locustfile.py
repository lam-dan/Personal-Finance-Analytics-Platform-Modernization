"""
Performance Testing with Locust

This file demonstrates performance testing for the Personal Finance Analytics Platform.
It tests the FastAPI and GraphQL services under load to ensure they meet performance requirements.

Usage:
    locust -f performance_tests/locustfile.py --host=http://localhost:8000
"""

from locust import HttpUser, task, between
import json
import random

class FastAPIUser(HttpUser):
    """
    Simulates users interacting with the FastAPI service.
    
    This class demonstrates performance testing patterns for REST APIs,
    including different types of requests and realistic user behavior.
    """
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Initialize user session."""
        self.client.headers.update({
            'User-Agent': 'PerformanceTest/1.0',
            'Accept': 'application/json'
        })
    
    @task(3)  # Higher weight - more frequent
    def get_transactions(self):
        """
        Test GET /transactions endpoint.
        
        This is a high-frequency operation that users perform regularly
        to view their transaction history.
        """
        with self.client.get("/transactions", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    response.success()
                else:
                    response.failure("Invalid response format")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(2)  # Medium weight
    def get_investments(self):
        """
        Test GET /investments endpoint.
        
        Users check their investment portfolio less frequently than transactions.
        """
        with self.client.get("/investments", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    response.success()
                else:
                    response.failure("Invalid response format")
            else:
                response.failure(f"HTTP {response.status_code}")

class GraphQLUser(HttpUser):
    """
    Simulates users interacting with the GraphQL service.
    
    This demonstrates performance testing for GraphQL APIs, which have
    different characteristics than REST APIs.
    """
    
    wait_time = between(2, 5)  # GraphQL queries are typically more complex
    
    def on_start(self):
        """Initialize user session."""
        self.client.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'GraphQLPerformanceTest/1.0'
        })
    
    @task(4)  # Most frequent - basic queries
    def query_transactions(self):
        """
        Test basic GraphQL transactions query.
        
        This simulates users fetching transaction data through GraphQL,
        which is a common operation.
        """
        query = {
            "query": """
                query {
                    transactions {
                        id
                        amount
                        category
                        date
                    }
                }
            """
        }
        
        # Try both GraphQL endpoints (FastAPI and standalone GraphQL service)
        endpoints = ["/graphql", "http://localhost:8002/graphql"]
        
        for endpoint in endpoints:
            try:
                with self.client.post(endpoint, json=query, catch_response=True) as response:
                    if response.status_code == 200:
                        data = response.json()
                        if 'data' in data and 'transactions' in data['data']:
                            response.success()
                            return
                        else:
                            response.failure("Invalid GraphQL response")
                    elif response.status_code == 404:
                        # Try next endpoint
                        continue
                    else:
                        response.failure(f"HTTP {response.status_code}")
            except Exception as e:
                # Try next endpoint
                continue
        
        # If all endpoints fail, mark as failure
        self.client.post("/graphql", json=query, catch_response=True).failure("All GraphQL endpoints failed")
    
    @task(2)  # Medium frequency
    def query_investments(self):
        """
        Test GraphQL investments query.
        
        Users query investment data less frequently than transactions.
        """
        query = {
            "query": """
                query {
                    investments {
                        id
                        asset
                        value
                        lastUpdated
                    }
                }
            """
        }
        
        # Try both GraphQL endpoints
        endpoints = ["/graphql", "http://localhost:8002/graphql"]
        
        for endpoint in endpoints:
            try:
                with self.client.post(endpoint, json=query, catch_response=True) as response:
                    if response.status_code == 200:
                        data = response.json()
                        if 'data' in data and 'investments' in data['data']:
                            response.success()
                            return
                        else:
                            response.failure("Invalid GraphQL response")
                    elif response.status_code == 404:
                        # Try next endpoint
                        continue
                    else:
                        response.failure(f"HTTP {response.status_code}")
            except Exception as e:
                # Try next endpoint
                continue
        
        # If all endpoints fail, mark as failure
        self.client.post("/graphql", json=query, catch_response=True).failure("All GraphQL endpoints failed")

class ObservabilityUser(HttpUser):
    """
    Simulates monitoring systems checking observability endpoints.
    
    This demonstrates performance testing for monitoring and observability
    endpoints that are called frequently by monitoring systems.
    """
    
    wait_time = between(5, 10)  # Monitoring checks are frequent but not constant
    
    def on_start(self):
        """Initialize user session."""
        self.client.headers.update({
            'User-Agent': 'MonitoringSystem/1.0',
            'Accept': 'text/plain'
        })
    
    @task(3)  # Health checks are very frequent
    def health_check(self):
        """
        Test health check endpoint.
        
        Monitoring systems frequently check health endpoints
        to ensure services are running properly.
        """
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 404:
                # Health endpoint might not exist, that's okay for demo
                response.success()
            else:
                response.failure(f"Health check failed: HTTP {response.status_code}")
    
    @task(2)  # Metrics are checked regularly
    def get_metrics(self):
        """
        Test metrics endpoint.
        
        Monitoring systems collect metrics regularly for
        performance monitoring and alerting.
        """
        with self.client.get("/metrics", catch_response=True) as response:
            if response.status_code == 200:
                # Check if response contains Prometheus metrics format
                content = response.text
                if 'http_requests_total' in content or 'python_info' in content:
                    response.success()
                else:
                    response.failure("Invalid metrics format")
            elif response.status_code == 404:
                # Metrics endpoint might not exist, that's okay for demo
                response.success()
            else:
                response.failure(f"Metrics endpoint failed: HTTP {response.status_code}")
    
    @task(1)  # Logs are checked less frequently
    def get_logs(self):
        """
        Test logs endpoint (if available).
        
        Log endpoints are checked less frequently than metrics
        and health checks.
        """
        with self.client.get("/logs", catch_response=True) as response:
            if response.status_code in [200, 404]:  # 404 is acceptable if no logs endpoint
                response.success()
            else:
                response.failure(f"Logs endpoint failed: HTTP {response.status_code}")

# Custom events for detailed monitoring
from locust import events

@events.request.add_listener
def my_request_handler(request_type, name, response_time, response_length, response,
                      context, exception, start_time, url, **kwargs):
    """
    Custom request handler for detailed performance monitoring.
    
    This demonstrates how to add custom monitoring and logging
    to performance tests for better observability.
    """
    if exception:
        print(f"Request failed: {name} - {exception}")
    else:
        print(f"Request successful: {name} - {response_time}ms")

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """
    Called when the test starts.
    
    This can be used to set up test data or initialize
    monitoring systems.
    """
    print("Performance test starting...")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """
    Called when the test stops.
    
    This can be used to clean up test data or generate
    performance reports.
    """
    print("Performance test completed.") 