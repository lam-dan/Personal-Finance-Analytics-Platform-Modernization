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
/legacy_scala_services/    # Mock Scala service (reference for migration)
/python_service/           # Python FastAPI service for REST APIs
/graphql_api/              # GraphQL API layer
/db_migration/             # Data migration scripts: PostgreSQL to TimescaleDB
/observability/            # Logging, Metrics, Health Checks
/tests/                    # Comprehensive test suite
/docker-compose.yml        # Multi-service orchestration
```

## Features
- **Transactions API**: View financial transactions
- **Investments API**: View current investment portfolio
- **Spending Trends API**: Analyze spending across categories in a date range
- **GraphQL API**: Unified interface for complex queries across financial data
- **Observability**: Structured logs, Prometheus metrics, health checks
- **Resilience**: Retry logic, circuit breakers, graceful degradation
- **Database Migration**: Automated PostgreSQL to TimescaleDB migration
- **Comprehensive Testing**: Unit and integration tests for all services

## Quick Start

### Prerequisites
- Python 3.9+
- Docker and Docker Compose
- Virtual environment (recommended)

### 1. Clone and Setup
```bash
git clone <repository-url>
cd Personal-Finance-Analytics-Platform-Modernization
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start All Services with Docker Compose
```bash
docker-compose up -d
```

This will start:
- **PostgreSQL** (port 5432) - Source database
- **TimescaleDB** (port 5433) - Destination database for time-series data
- **FastAPI Service** (port 8000) - REST API endpoints
- **GraphQL API** (port 8002) - GraphQL interface
- **Observability Service** (port 8001) - Prometheus metrics

### 3. Run Database Migration
```bash
python db_migration/migrate.py
```

### 4. Run Tests
```bash
make test
```

## API Endpoints

### FastAPI Service (Port 8000)
- `GET /transactions` - List all transactions
- `GET /investments` - List all investments
- `GET /spending_trends?start_date=YYYY-MM-DD&end_date=YYYY-MM-DD` - Spending analysis

### GraphQL API (Port 8002)
- `POST /graphql` - GraphQL queries for transactions and investments

### Observability (Port 8001)
- `GET /metrics` - Prometheus metrics endpoint

## Development

### Running Individual Services

#### FastAPI Service
```bash
cd python_service
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### GraphQL API
```bash
cd graphql_api
uvicorn schema:app --host 0.0.0.0 --port 8002 --reload
```

#### Observability Service
```bash
cd observability
python logging_config.py
```

### Testing
The project includes comprehensive tests for all components:

```bash
# Run all tests
make test

# Run specific test files
pytest tests/test_fastapi_service.py
pytest tests/test_graphql_api.py
pytest tests/test_migration.py
pytest tests/test_observability.py
```

### Docker Development
```bash
# Build and start all services
docker-compose up --build

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Database Configuration

### PostgreSQL (Source)
- **Host**: localhost
- **Port**: 5432
- **Database**: finance_db
- **User**: postgres
- **Password**: postgres

### TimescaleDB (Destination)
- **Host**: localhost
- **Port**: 5433
- **Database**: finance_timescale
- **User**: postgres
- **Password**: postgres

## Recent Improvements

### Fixed Issues
- **Docker Build Issues**: Fixed path references in Dockerfiles
- **Test Dependencies**: Added missing `httpx` and `requests` packages
- **API Testing**: Replaced problematic `TestClient` with real HTTP requests
- **Database Migration**: Added automatic table creation and fixed port configuration
- **Observability**: Made the service run continuously instead of exiting
- **Test Reliability**: All 7 tests now pass consistently

### 🔧 Technical Improvements
- **Robust Migration Script**: Automatically creates tables if they don't exist
- **Real Integration Tests**: Tests actual running services via HTTP
- **Continuous Observability**: Prometheus metrics server runs indefinitely
- **Proper Error Handling**: Graceful handling of missing dependencies and services
- **Docker Optimization**: Streamlined container builds and configurations

## Monitoring and Observability

### Metrics Available
- `app_request_count` - Total number of requests processed
- `app_request_latency_seconds` - Request latency histogram
- Custom business metrics for financial data

### Health Checks
- Service health endpoints available on each service
- Database connectivity monitoring
- Automated alerting for service failures

## Future Enhancements
- **Real-time Data Processing**: Integrate Clickstream data ingestion via Kafka
- **Advanced Analytics**: Implement predictive insights and ML models
- **Data Backfilling**: Automated historical data migration
- **Performance Optimization**: Query optimization and caching strategies
- **Security Enhancements**: Authentication, authorization, and data encryption
- **Scalability**: Horizontal scaling and load balancing

## Troubleshooting

### Common Issues
1. **Port Conflicts**: Ensure ports 8000, 8001, 8002, 5432, 5433 are available
2. **Database Connection**: Verify PostgreSQL and TimescaleDB containers are running
3. **Test Failures**: Run `pip install -r requirements.txt` to ensure all dependencies are installed
4. **Docker Issues**: Use `docker-compose down && docker-compose up --build` to rebuild containers

### Logs and Debugging
```bash
# View service logs
docker-compose logs fastapi
docker-compose logs graphql
docker-compose logs observability

# Check database connectivity
docker-compose exec postgres psql -U postgres -d finance_db
docker-compose exec timescaledb psql -U postgres -d finance_timescale
```

## License
MIT License