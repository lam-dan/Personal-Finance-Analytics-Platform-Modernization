.PHONY: install lint test migrate run-graphql run-fastapi observability

install:
	pip install -r requirements.txt

lint:
	flake8 graphql_api python_service db_migration observability

test:
	pytest

migrate:
	python db_migration/migrate.py

run-graphql:
	uvicorn graphql_api.schema:app --reload

run-fastapi:
	uvicorn python_service.main:app --reload

observability:
	python observability/logging_config.py

start-all:
	@echo "Starting all services..."
	make run-fastapi &
	make run-graphql &
	make observability
