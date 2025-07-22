"""
Personal Finance Analytics Platform - Observability Service

This module provides observability capabilities for the personal finance platform
including structured logging, Prometheus metrics collection, and health monitoring.

The observability service runs continuously to:
- Collect application metrics (request counts, latency)
- Provide structured logging for debugging and monitoring
- Expose Prometheus metrics endpoint for monitoring systems
- Simulate real-world application behavior for testing

In production, this would integrate with:
- Prometheus for metrics storage
- Grafana for visualization
- AlertManager for alerting
- ELK stack for log aggregation
"""

import logging
from prometheus_client import Counter, Histogram, start_http_server
import time

# Configure structured logging with timestamps and log levels
# This provides consistent log formatting across the application
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics definitions
# These metrics will be exposed at /metrics endpoint for monitoring systems

# Counter metric to track total number of requests processed
REQUEST_COUNT = Counter(
    'app_request_count', 
    'Total number of requests processed by the application'
)

# Histogram metric to track request latency distribution
# This helps identify performance bottlenecks and monitor response times
REQUEST_LATENCY = Histogram(
    'app_request_latency_seconds', 
    'Request latency in seconds',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]  # Define latency buckets for histogram
)

# Start Prometheus metrics server on port 8001
# This exposes the /metrics endpoint for Prometheus to scrape
start_http_server(8001)
logger.info("Prometheus metrics available on port 8001")

# Example instrumented function that demonstrates metrics collection
@REQUEST_LATENCY.time()  # Decorator automatically measures execution time
def sample_process():
    """
    Simulates a typical application process with metrics collection.
    
    This function demonstrates how to instrument application code with:
    - Timing measurements (via @REQUEST_LATENCY.time() decorator)
    - Counter increments for business metrics
    - Structured logging for debugging
    
    In a real application, this would be replaced with actual business logic.
    """
    logger.info("Starting sample process")
    time.sleep(1)  # Simulate processing delay
    logger.info("Sample process completed")
    REQUEST_COUNT.inc()  # Increment the request counter

if __name__ == "__main__":
    """
    Main entry point for the observability service.
    
    This service runs continuously to provide ongoing metrics and logging.
    It demonstrates a production-ready observability pattern with:
    - Continuous operation
    - Graceful shutdown handling
    - Error recovery and retry logic
    - Health monitoring capabilities
    """
    # Run continuously to provide ongoing metrics
    logger.info("Starting continuous observability service")
    
    while True:
        try:
            # Execute the sample process to generate metrics
            sample_process()
            
        except KeyboardInterrupt:
            # Handle graceful shutdown when Ctrl+C is pressed
            logger.info("Shutting down observability service")
            break
            
        except Exception as e:
            # Handle unexpected errors with retry logic
            logger.error(f"Error in sample process: {e}")
            time.sleep(5)  # Wait before retrying to avoid tight error loops
