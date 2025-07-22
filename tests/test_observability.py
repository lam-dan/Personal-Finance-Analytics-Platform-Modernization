import subprocess  # To spawn the observability script as a separate process
import time  # For introducing delays to allow service startup

import requests  # To make HTTP requests to the metrics endpoint


def test_observability_metrics():
    """
    Test that the observability metrics endpoint starts and is reachable.
    Validates that Prometheus metrics are exposed correctly.
    """
    # Start the observability module in a subprocess
    process = subprocess.Popen(["python", "observability/logging_config.py"])
    time.sleep(3)  # Allow time for the metrics server to start

    try:
        # Attempt to access the base observability URL
        response = requests.get("http://localhost:8001")
        # Prometheus returns 404 at root, so accept either 200 or 404
        assert response.status_code == 200 or response.status_code == 404

        # Access the /metrics endpoint to verify Prometheus metrics
        metrics_response = requests.get("http://localhost:8001/metrics")
        # Ensure endpoint is reachable
        assert metrics_response.status_code == 200
        # Verify metric is present
        assert "app_request_count" in metrics_response.text
    finally:
        # Ensure the subprocess is terminated after the test
        process.terminate()
