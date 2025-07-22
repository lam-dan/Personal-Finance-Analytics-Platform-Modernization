"""
Personal Finance Analytics Platform - GraphQL API

This module defines the GraphQL schema for the personal finance analytics
platform. It provides a unified interface for querying financial data
including transactions and investments using GraphQL's type-safe query
language.

The GraphQL API offers several advantages over REST:
- Single endpoint for all queries
- Type-safe queries with introspection
- Efficient data fetching with field selection
- Real-time capabilities with subscriptions (future enhancement)
"""

from typing import List

import strawberry
from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter


# Define the Transaction GraphQL type with its fields
@strawberry.type
class Transaction:
    """
    GraphQL type representing a financial transaction.

    Fields:
        id: Unique identifier for the transaction
        amount: Transaction amount (negative for expenses, positive for income)
        category: Spending category (e.g., Groceries, Transportation, Salary)
        date: Transaction date in ISO format (YYYY-MM-DD)
    """

    id: int
    amount: float
    category: str
    date: str


# Define the Investment GraphQL type with its fields
@strawberry.type
class Investment:
    """
    GraphQL type representing an investment asset.

    Fields:
        id: Unique identifier for the investment
        asset: Type of investment (e.g., Stocks, Bonds, ETFs)
        value: Current market value of the investment
        last_updated: Date when the value was last updated
    """

    id: int
    asset: str
    value: float
    last_updated: str


# Define the root Query type for GraphQL API
@strawberry.type
class Query:
    """
    Root GraphQL query type that defines all available queries.

    This is the entry point for all GraphQL queries. Each field represents
    a different query that clients can execute against the API.
    """

    @strawberry.field
    def transactions(self) -> List[Transaction]:
        """
        Query to fetch all financial transactions.

        Returns:
            List[Transaction]: List of all transactions in the system

        Example GraphQL query:
            query {
                transactions {
                    id
                    amount
                    category
                    date
                }
            }
        """
        # Mock data - in production, this would query a database
        return [
            Transaction(
                id=1, amount=-120.50, category="Groceries", date="2024-07-01"
            ),
            Transaction(
                id=2,
                amount=-75.20,
                category="Transportation",
                date="2024-07-03",
            ),
        ]

    @strawberry.field
    def investments(self) -> List[Investment]:
        """
        Query to fetch all investment portfolio data.

        Returns:
            List[Investment]: List of all investments in the portfolio

        Example GraphQL query:
            query {
                investments {
                    id
                    asset
                    value
                    lastUpdated
                }
            }
        """
        # Mock data - in production, this would query a database
        return [
            Investment(
                id=1, asset="Stocks", value=15000, last_updated="2024-07-07"
            ),
            Investment(
                id=2, asset="Bonds", value=5000, last_updated="2024-07-07"
            ),
        ]


# Create the GraphQL schema with the defined Query type
# The schema defines the complete GraphQL API structure
schema = strawberry.Schema(Query)

# Create the FastAPI app instance to serve the GraphQL API
app = FastAPI(
    title="Personal Finance Analytics GraphQL API",
    description="GraphQL API for personal finance data queries",
    version="1.0.0",
)

# Create the GraphQL router with the schema
# This integrates the GraphQL schema with FastAPI
graphql_app = GraphQLRouter(schema)

# Include the GraphQL router in the FastAPI app under the /graphql endpoint
# This makes the GraphQL API available at http://localhost:8002/graphql
app.include_router(graphql_app, prefix="/graphql")
