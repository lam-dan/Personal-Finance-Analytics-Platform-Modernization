# File: python_service/main.py
"""
Personal Finance Analytics Platform - FastAPI Service

This module provides REST API endpoints for financial data including:
- Transaction management and retrieval
- Investment portfolio tracking
- Spending trend analysis with date filtering

The service simulates a real-world financial API with mock data for
demonstration. In production, this would connect to actual databases
and include authentication.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException

# Initialize FastAPI application with metadata
app = FastAPI(
    title="Personal Finance Analytics API",
    description="REST API for personal finance data management and analytics",
    version="1.0.0",
)

# Sample mock data for demonstration purposes
# In production, this would be fetched from a database
TRANSACTIONS = [
    {
        "id": 1,
        "amount": -120.50,
        "category": "Groceries",
        "date": "2024-07-01",
    },
    {
        "id": 2,
        "amount": -75.20,
        "category": "Transportation",
        "date": "2024-07-03",
    },
    {"id": 3, "amount": 5000.00, "category": "Salary", "date": "2024-07-05"},
]

INVESTMENTS = [
    {"id": 1, "asset": "Stocks", "value": 15000, "last_updated": "2024-07-07"},
    {"id": 2, "asset": "Bonds", "value": 5000, "last_updated": "2024-07-07"},
]

# Configure structured logging for observability
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.get("/transactions", response_model=List[Dict[str, Any]])
def get_transactions():
    """
    Retrieve all financial transactions.

    Returns:
        List[Dict]: List of transaction objects with id, amount,
        category, and date
    """
    logger.info("Fetching all transactions")
    return TRANSACTIONS


@app.get("/investments", response_model=List[Dict[str, Any]])
def get_investments():
    """
    Retrieve all investment portfolio data.

    Returns:
        List[Dict]: List of investment objects with id, asset, value,
        and last_updated
    """
    logger.info("Fetching all investments")
    return INVESTMENTS


@app.get("/spending_trends")
def spending_trends(start_date: str, end_date: str) -> Dict[str, float]:
    """
    Analyze spending trends by category within a specified date range.

    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format

    Returns:
        Dict[str, float]: Spending summary by category with total amounts

    Raises:
        HTTPException: If date format is invalid (400 Bad Request)
    """
    logger.info(
        f"Calculating spending trends between {start_date} " f"and {end_date}"
    )

    # Validate and parse date inputs
    try:
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Invalid date format. Use YYYY-MM-DD."
        )

    # Filter transactions within the specified date range
    filtered = [
        t
        for t in TRANSACTIONS
        if start <= datetime.fromisoformat(t["date"]) <= end
    ]

    # Calculate spending totals by category
    category_sum = {}
    for tx in filtered:
        category = tx["category"]
        amount = tx["amount"]
        category_sum[category] = category_sum.get(category, 0) + amount

    return category_sum
