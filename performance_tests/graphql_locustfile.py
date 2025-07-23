"""
GraphQL Performance Testing with Locust

This file specifically tests the GraphQL service on port 8002.
It should be run with: locust -f performance_tests/graphql_locustfile.py --host=http://localhost:8002

Usage:
    locust -f performance_tests/graphql_locustfile.py --host=http://localhost:8002 --users=5 --spawn-rate=1 --run-time=30s --headless
"""

from locust import HttpUser, task, between
import json

class GraphQLUser(HttpUser):
    """
    Simulates users interacting with the GraphQL service on port 8002.
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
        
        with self.client.post("/graphql", json=query, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and 'transactions' in data['data']:
                    response.success()
                else:
                    response.failure("Invalid GraphQL response")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(2)  # Medium frequency
    def query_investments(self):
        """
        Test GraphQL investments query.
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
        
        with self.client.post("/graphql", json=query, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and 'investments' in data['data']:
                    response.success()
                else:
                    response.failure("Invalid GraphQL response")
            else:
                response.failure(f"HTTP {response.status_code}")

# Custom events for detailed monitoring
from locust import events

@events.request.add_listener
def my_request_handler(request_type, name, response_time, response_length, response,
                      context, exception, start_time, url, **kwargs):
    """
    Custom request handler for detailed performance monitoring.
    """
    if exception:
        print(f"GraphQL request failed: {name} - {exception}")
    else:
        print(f"GraphQL request successful: {name} - {response_time}ms")

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when the test starts."""
    print("GraphQL performance test starting...")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when the test stops."""
    print("GraphQL performance test completed.") 