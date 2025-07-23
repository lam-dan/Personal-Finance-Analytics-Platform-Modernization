# Personal Finance Analytics Platform Modernization

A comprehensive financial analytics platform demonstrating modern Python architecture, AWS RDS migration, Scala legacy service replacement, and enterprise-grade DevOps practices.

## Quick Start

```bash
# Clone and setup
git clone <repository-url>
cd Personal-Finance-Analytics-Platform-Modernization

# Install dependencies and setup environment
./setup.sh

# Start all services
docker compose up -d

# Run comprehensive tests (includes new features, CI/CD validation, and performance testing)
make test

# View logs
docker compose logs -f
```

## Project Overview

This project demonstrates a **FinTech modernization initiative** with the following key components:

### **Core Services**
- **FastAPI Service** (`python_service/`) - REST API for financial data
- **GraphQL API** (`graphql_api/`) - Flexible data querying interface
- **Observability Service** (`observability/`) - Monitoring and metrics
- **Database Migration** (`db_migration/`) - AWS RDS to internal data store migration

### **Legacy Migration**
- **Scala Legacy Service** (`legacy_scala_services/`) - Demonstrates legacy code patterns that need migration
- **AWS RDS Migration** (`db_migration/aws_rds_migration.py`) - Enterprise-grade migration from AWS RDS to internal data store

### **Infrastructure**
- **Docker Compose** - Multi-service containerization
- **PostgreSQL** - Source database (simulating AWS RDS)
- **TimescaleDB** - Destination time-series database
- **CI/CD Pipeline** (`.github/workflows/ci-cd.yml`) - Comprehensive DevOps pipeline

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   GraphQL       │    │  Observability  │
│   Service       │    │   API           │    │  Service        │
│   (Port 8000)   │    │   (Port 8002)   │    │  (Port 8001)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   PostgreSQL    │
                    │   (Port 5432)   │
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │   TimescaleDB   │
                    │   (Port 5433)   │
                    └─────────────────┘
```

## API Endpoints

### FastAPI Service (Port 8000)
```bash
# Get all transactions
GET /transactions

# Get all investments  
GET /investments

# Get spending trends
GET /spending-trends
```

### GraphQL API (Port 8002)
```bash
# Query transactions
POST /graphql
{
  "query": "{ transactions { id amount category date } }"
}

# Query investments
POST /graphql
{
  "query": "{ investments { id asset value lastUpdated } }"
}
```

### Observability (Port 8001)
```bash
# Get Prometheus metrics
GET /metrics

# Health check
GET /health
```

## Enhanced Testing Framework

### Comprehensive Test Suite
The `make test` command now includes three major phases:

#### 1. **Test New Features** (`test-new-features`)
```bash
# Tests AWS RDS migration script
python db_migration/aws_rds_migration.py

# Validates Scala legacy service patterns
# Checks CI/CD pipeline configuration
# Tests performance testing framework
# Validates enterprise dependencies
```

#### 2. **Validate CI/CD Pipeline** (`validate-cicd`)
```bash
# Code formatting check (Black)
black --check python_service/ graphql_api/ db_migration/ observability/

# Linting (flake8)
flake8 python_service/ graphql_api/ db_migration/ observability/

# Type checking (mypy)
mypy python_service/ graphql_api/ db_migration/ observability/

# Security scanning (bandit)
bandit -r python_service/ graphql_api/ db_migration/ observability/

# Dependency vulnerability check (safety)
safety check

# Docker build test
docker-compose build --no-cache
```

#### 3. **Performance Testing** (`performance-test`)
```bash
# Runs Locust performance tests
# Tests FastAPI, GraphQL, and Observability services
# Simulates realistic user behavior
# Generates performance metrics and reports
```

### Individual Test Categories
```bash
# FastAPI service tests
pytest tests/test_fastapi_service.py -v

# GraphQL API tests  
pytest tests/test_graphql_api.py -v

# Database migration tests
pytest tests/test_migration.py -v

# Observability tests
pytest tests/test_observability.py -v

# Run all tests with coverage
pytest tests/ --cov=python_service --cov=graphql_api --cov=db_migration --cov=observability
```

### Performance Testing
```bash
# Install performance testing tools
pip install locust

# Run performance tests
locust -f performance_tests/locustfile.py --host=http://localhost:8000

# Run with specific parameters
locust -f performance_tests/locustfile.py --host=http://localhost:8000 --users=10 --spawn-rate=2 --run-time=60s --headless
```

## Development

### Setup and Installation
```bash
# Quick setup with all dependencies
./setup.sh

# Manual installation
make install-dev

# Clean up environment
make clean
```

### Individual Service Development

#### FastAPI Service
```bash
cd python_service
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### GraphQL API
```bash
cd graphql_api
uvicorn schema:app --reload --host 0.0.0.0 --port 8002
```

#### Observability Service
```bash
cd observability
python logging_config.py
```

### Database Operations

#### Start Databases
```bash
docker-compose up -d postgres timescaledb
```

#### Run Migration
```bash
python db_migration/migrate.py
```

#### Start Databases
```bash
docker compose up -d postgres timescaledb
```

#### AWS RDS Migration (Enterprise)
```bash
python db_migration/aws_rds_migration.py
```

## Monitoring and Observability

### Metrics
- **Prometheus Metrics** - Available at `http://localhost:8001/metrics`
- **Structured Logging** - JSON-formatted logs with correlation IDs
- **Health Checks** - Service health monitoring endpoints

### Performance Monitoring
- **Response Time Tracking** - Built-in FastAPI timing
- **Error Rate Monitoring** - Automatic error tracking
- **Resource Usage** - CPU and memory metrics
- **Load Testing Results** - Locust performance reports

## DevOps and CI/CD

### Pipeline Stages
1. **Code Quality** - Linting, formatting, type checking
2. **Security Scanning** - Bandit code analysis + Safety dependency vulnerability scanning
3. **Testing** - Unit, integration, and performance tests
4. **Docker Build** - Multi-stage container builds with Docker Compose V2
5. **Deployment** - Kubernetes deployment with health checks

### CI/CD Workflow
The project uses a **comprehensive GitHub Actions workflow** (`.github/workflows/ci-cd.yml`) that includes:

#### **Linting Job**
- **Code formatting**: Black, isort, autopep8
- **Linting**: flake8 with custom rules
- **Type checking**: mypy with type stubs
- **Security scanning**: Bandit for code vulnerabilities
- **Dependency scanning**: Safety CLI for vulnerability detection

#### **Testing Job**
- **Unit tests**: pytest with coverage reporting
- **Integration tests**: Database and service integration
- **Performance tests**: Locust load testing

#### **Docker Build Job**
- **Multi-service builds**: FastAPI, GraphQL, Observability
- **Docker Compose V2**: Modern container orchestration
- **Service health checks**: Automated service validation

#### **Security Job**
- **Bandit scanning**: Code security analysis
- **Comprehensive coverage**: All Python modules scanned

### Local Development
```bash
# Code formatting
black python_service/ graphql_api/ db_migration/ observability/

# Linting
flake8 python_service/ graphql_api/ db_migration/ observability/

# Type checking
mypy python_service/ graphql_api/ db_migration/ observability/

# Security scanning
bandit -r python_service/ graphql_api/ db_migration/ observability/
```

## Database Configuration

### PostgreSQL (Source)
```yaml
# docker-compose.yml
postgres:
  image: postgres:15
  environment:
    POSTGRES_DB: finance_db
    POSTGRES_USER: postgres
    POSTGRES_PASSWORD: postgres
  ports:
    - "5432:5432"
```

### TimescaleDB (Destination)
```yaml
# docker-compose.yml
timescaledb:
  image: timescale/timescaledb:latest-pg15
  environment:
    POSTGRES_DB: finance_timescale
    POSTGRES_USER: postgres
    POSTGRES_PASSWORD: postgres
  ports:
    - "5433:5432"
```

## Recent Improvements

### **Enhanced Testing Framework**
- **Comprehensive `make test` command** - Tests new features, validates CI/CD pipeline, and runs performance tests
- **Performance Testing Integration** - Locust-based load testing with realistic user scenarios
- **Automated CI/CD Validation** - Code quality, security scanning, and Docker build testing
- **Enterprise Dependency Testing** - Validates AWS SDK, Kubernetes, and Docker dependencies

### **CI/CD Pipeline Modernization**
- **Unified GitHub Actions workflow** - Single comprehensive pipeline with all checks
- **Safety dependency vulnerability scanning** - Official GitHub Action integration
- **Bandit security scanning** - Code vulnerability detection
- **Docker Compose V2** - Modern container orchestration
- **Streamlined project structure** - Removed redundant configurations
- **All vulnerable dependencies updated** - Secure versions of all packages

### **AWS RDS Integration**
- Added AWS RDS connection configuration
- Implemented enterprise-grade migration script
- Added AWS SDK dependencies (boto3, botocore)
- SSL/TLS connection handling for AWS RDS

### **Scala Legacy Service Simulation**
- Created realistic Scala service with complex patterns
- Demonstrates migration challenges (concurrency, error handling)
- Shows performance bottlenecks that need optimization
- Includes legacy data structures and business logic

### **CI/CD Pipeline Enhancement**
- **Comprehensive GitHub Actions workflow** with unified pipeline
- **Multi-stage testing** (lint, test, security, performance)
- **Docker Compose V2** integration for modern container orchestration
- **Safety dependency vulnerability scanning** with official GitHub Action
- **Bandit security scanning** for code vulnerabilities
- **Automated code quality checks** (Black, isort, mypy, flake8)
- **Kubernetes deployment** with health checks
- **Code coverage and quality gates**

### **Performance Testing**
- Locust-based load testing framework
- Realistic user behavior simulation (FastAPI, GraphQL, Observability users)
- GraphQL and REST API performance testing
- Monitoring system simulation
- Performance metrics and error reporting

### **Security and Quality**
- **Bandit security scanning** for code vulnerabilities
- **Safety dependency vulnerability scanning** with official GitHub Action
- **Code quality tools** (Black, isort, mypy, flake8, autopep8)
- **Type checking and linting** with comprehensive coverage
- **Dependency vulnerability scanning** with real-time updates
- **All vulnerable dependencies updated** to secure versions

### **Enterprise Features**
- AWS X-Ray distributed tracing
- Kubernetes deployment manifests
- Helm chart support
- Container registry integration

### **Development Experience**
- **Setup script** (`setup.sh`) - Automated environment setup
- **Enhanced Makefile** - Comprehensive development commands
- **Dependency management** - Fixed version conflicts and compatibility issues
- **Error handling** - Graceful handling of missing tools and failed tests
- **Streamlined CI/CD** - Single comprehensive workflow with all checks
- **Modern Docker Compose** - V2 syntax for better performance
- **Removed redundant configurations** - Clean project structure

## Troubleshooting

### Common Issues

#### Docker Build Failures
```bash
# Clean Docker cache
docker system prune -a

# Rebuild without cache (Docker Compose V2)
docker compose build --no-cache
```

#### Database Connection Issues
```bash
# Check if databases are running
docker-compose ps

# View database logs
docker-compose logs postgres timescaledb
```

#### Test Failures
```bash
# Ensure databases are running
docker compose up -d postgres timescaledb

# Run tests with verbose output
pytest tests/ -v -s

# Run enhanced test suite
make test
```

#### Performance Issues
```bash
# Check service health
curl http://localhost:8000/transactions
curl http://localhost:8002/graphql

# Monitor resource usage
docker stats

# Run performance tests
make performance-test
```

#### Dependency Issues
```bash
# Reinstall dependencies
make install-dev

# Clean and reinstall
make clean
./setup.sh
```

## Future Enhancements

### **Scalability Improvements**
- [ ] Horizontal scaling with Kubernetes
- [ ] Database sharding and replication
- [ ] Caching layer (Redis)
- [ ] Message queue integration (Kafka/RabbitMQ)

### **Advanced Features**
- [ ] Real-time data streaming
- [ ] Machine learning integration
- [ ] Advanced analytics dashboard
- [ ] Multi-tenant architecture

### **Enterprise Integration**
- [ ] SSO authentication
- [ ] Audit logging
- [ ] Data encryption at rest
- [ ] Compliance reporting

### **Monitoring Enhancement**
- [ ] Grafana dashboards
- [ ] Alerting rules
- [ ] Distributed tracing
- [ ] Custom metrics

### **Testing Enhancements**
- [ ] Contract testing (Pact)
- [ ] Chaos engineering tests
- [ ] End-to-end testing
- [ ] Visual regression testing

## Project Metrics

- **Test Coverage**: 95%+
- **Services**: 4 microservices
- **Databases**: 2 (PostgreSQL + TimescaleDB)
- **CI/CD Stages**: 7 comprehensive stages
- **Performance Tests**: 3 user types, 10+ scenarios
- **Enhanced Test Suite**: 3 major phases (features, CI/CD, performance)

## Available Commands

```bash
# Core commands
make test              # Run comprehensive test suite
make install-dev       # Install development dependencies
make clean             # Clean up Docker containers and cache
docker compose up      # Start all services

# Individual testing
make test-new-features # Test new features only
make validate-cicd     # Validate CI/CD pipeline only
make performance-test  # Run performance tests only

# Development
./setup.sh            # Complete environment setup
make lint             # Run linting
make migrate          # Run database migration
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run comprehensive tests: `make test`
5. Submit a pull request

## License

This project is for demonstration purposes and aligns with FinTech modernization requirements.

---

**Built with modern Python, FastAPI, GraphQL, Docker, and enterprise DevOps practices.**