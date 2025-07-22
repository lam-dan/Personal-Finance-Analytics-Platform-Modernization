import strawberry
from typing import List
from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter

# Define the Transaction GraphQL type with its fields
@strawberry.type
class Transaction:
    id: int
    amount: float
    category: str
    date: str

# Define the Investment GraphQL type with its fields
@strawberry.type
class Investment:
    id: int
    asset: str
    value: float
    last_updated: str

# Define the root Query type for GraphQL API
@strawberry.type
class Investment:
    id: int
    asset: str
    value: float
    last_updated: str

# Define the root Query type for GraphQL API
@strawberry.type
class Query:
    # GraphQL query to fetch all transactions
    @strawberry.field
    def transactions(self) -> List[Transaction]:
        return [
            Transaction(id=1, amount=-120.50, category="Groceries", date="2024-07-01"),
            Transaction(id=2, amount=-75.20, category="Transportation", date="2024-07-03")
        ]

    # GraphQL query to fetch all investments
    @strawberry.field
    def investments(self) -> List[Investment]:
        return [
            Investment(id=1, asset="Stocks", value=15000, last_updated="2024-07-07"),
            Investment(id=2, asset="Bonds", value=5000, last_updated="2024-07-07")
        ]

# Create the GraphQL schema with the defined Query type
schema = strawberry.Schema(Query)

# Create the FastAPI app instance
app = FastAPI()

# Create the GraphQL router with the schema
graphql_app = GraphQLRouter(schema)

# Include the GraphQL router in the FastAPI app under the /graphql endpoint
app.include_router(graphql_app, prefix="/graphql")