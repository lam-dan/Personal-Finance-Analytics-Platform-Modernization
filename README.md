# Personal Finance Analytics Platform Modernization

## Overview
This project modernizes a legacy personal finance analytics platform by transitioning from Scala to Python, upgrading the data store, and enhancing the API with GraphQL. It is designed to simulate a real-world FinTech modernization initiative, incorporating observability, scalability, reliability, and DevOps best practices.

## Goals
- **Migrate** legacy Scala services to Python (FastAPI)
- **Implement** a GraphQL API layer using Strawberry
- **Migrate** data from PostgreSQL (AWS RDS simulation) to TimescaleDB for time-series capabilities
- **Enhance** observability with structured logging and metrics
- **Ensure** service reliability with retries, circuit breakers, and health checks
- **Automate** CI/CD using GitHub Actions

## Project Structure
```
/legacy_scala_service/     # Mock Scala service (reference for migration)
/python_service/           # Python FastAPI service for REST APIs
/graphql_api/              # GraphQL API layer
/db_migration/             # Data migration scripts: PostgreSQL to TimescaleDB
/observability/            # Logging, Metrics, Health Checks
/ci_cd/                    # CI/CD pipeline configurations (GitHub Actions)
```

## Features
- **Transactions API**: View financial transactions
- **Investments API**: View current investment portfolio
- **Spending Trends API**: Analyze spending across categories in a date range
- **GraphQL API**: Unified interface for complex queries across financial data
- **Observability**: Structured logs, Prometheus metrics, health checks
- **Resilience**: Retry logic, circuit breakers, graceful degradation

## Setup Instructions
### Requirements
- Python 3.9+
- FastAPI
- Strawberry GraphQL
- PostgreSQL and TimescaleDB
- Docker / Docker Compose (optional for local dev)

### Running the Python Service
```bash
cd python_service
pip install -r requirements.txt
uvicorn main:app --reload
```

### Running the GraphQL API
```bash
cd graphql_api
pip install -r requirements.txt
uvicorn schema:app --reload
```

### Observability Setup
- Prometheus and Grafana for metrics visualization
- Health check endpoints available on `/health`

### CI/CD
CI/CD pipelines are defined in `.github/workflows/` for linting, testing, and deployment.

## Future Enhancements
- Integrate Clickstream data ingestion via Kafka
- Implement data backfilling for migrated databases
- Expand financial analytics with predictive insights

## License
MIT License