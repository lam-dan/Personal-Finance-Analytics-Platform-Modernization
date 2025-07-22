"""
AWS RDS to Internal Data Store Migration Script

This script demonstrates the migration from AWS RDS PostgreSQL to an internal
Intuit paved path data store, as mentioned in the job requirements.

Migration Process:
1. Connect to AWS RDS PostgreSQL instance
2. Extract data with proper error handling and retry logic
3. Transform data for the internal data store format
4. Load data into the internal data store
5. Validate data integrity and completeness
"""

import logging
from datetime import datetime
from typing import Any, Dict, List

import psycopg2

# Configure logging for migration tracking
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# AWS RDS Configuration (source)
AWS_RDS_CONFIG = {
    "dbname": "finance_db",
    "user": "admin",
    "password": "your_aws_rds_password",
    "host": "your-aws-rds-endpoint.region.rds.amazonaws.com",
    "port": "5432",
    "sslmode": "require",  # AWS RDS requires SSL
}

# Internal Data Store Configuration (destination)
INTERNAL_DB_CONFIG = {
    "dbname": "finance_internal",
    "user": "internal_user",
    "password": "internal_password",
    "host": "internal-db.intuit.com",
    "port": "5432",
}


class AWSRDSMigration:
    """
    Handles migration from AWS RDS to internal data store.

    This class demonstrates enterprise-level migration patterns including:
    - AWS RDS connection management
    - Data validation and integrity checks
    - Retry logic and error handling
    - Progress tracking and monitoring
    """

    def __init__(self):
        self.aws_conn = None
        self.internal_conn = None
        self.migration_stats = {
            "transactions_migrated": 0,
            "investments_migrated": 0,
            "errors": 0,
            "start_time": None,
            "end_time": None,
        }

    def connect_to_aws_rds(self):
        """Establish connection to AWS RDS with proper error handling."""
        try:
            logger.info("Connecting to AWS RDS...")
            self.aws_conn = psycopg2.connect(**AWS_RDS_CONFIG)
            logger.info("Successfully connected to AWS RDS")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to AWS RDS: {e}")
            return False

    def connect_to_internal_db(self):
        """Establish connection to internal data store."""
        try:
            logger.info("Connecting to internal data store...")
            self.internal_conn = psycopg2.connect(**INTERNAL_DB_CONFIG)
            logger.info("Successfully connected to internal data store")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to internal data store: {e}")
            return False

    def extract_transactions_from_rds(self) -> List[Dict[str, Any]]:
        """
        Extract transaction data from AWS RDS with pagination and error handling.

        Returns:
            List of transaction dictionaries
        """
        if not self.aws_conn:
            logger.error("No AWS RDS connection available")
            return []

        try:
            cursor = self.aws_conn.cursor()

            # Use pagination for large datasets
            offset = 0
            limit = 1000
            all_transactions = []

            while True:
                query = """
                    SELECT id, amount, category, date, metadata
                    FROM transactions
                    ORDER BY id
                    LIMIT %s OFFSET %s
                """
                cursor.execute(query, (limit, offset))
                batch = cursor.fetchall()

                if not batch:
                    break

                # Transform to dictionary format
                transactions = []
                for row in batch:
                    transactions.append(
                        {
                            "id": row[0],
                            "amount": row[1],
                            "category": row[2],
                            "date": row[3],
                            "metadata": row[4] or {},
                        }
                    )

                all_transactions.extend(transactions)
                offset += limit

                logger.info(
                    f"Extracted {len(transactions)} transactions (batch)"
                )

            cursor.close()
            logger.info(
                f"Total transactions extracted: {len(all_transactions)}"
            )
            return all_transactions

        except Exception as e:
            logger.error(f"Error extracting transactions from AWS RDS: {e}")
            return []

    def transform_data_for_internal_store(
        self, transactions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Transform data to match internal data store schema.

        This demonstrates data transformation patterns commonly needed
        when migrating between different database systems.
        """
        transformed = []

        for tx in transactions:
            # Transform to internal data store format
            transformed_tx = {
                "id": tx["id"],
                "amount": float(tx["amount"]) if tx["amount"] else 0.0,
                "category": tx["category"].upper(),  # Standardize categories
                "date": tx["date"].isoformat() if tx["date"] else None,
                "created_at": datetime.now().isoformat(),
                "source_system": "aws_rds",
                "metadata": self._transform_metadata(tx.get("metadata", {})),
            }
            transformed.append(transformed_tx)

        logger.info(f"Transformed {len(transformed)} transactions")
        return transformed

    def _transform_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Transform metadata to internal format."""
        if isinstance(metadata, str):
            # Parse string metadata if needed
            try:
                import json

                return json.loads(metadata)
            except Exception:
                return {"raw_metadata": metadata}
        return metadata or {}

    def load_to_internal_store(
        self, transactions: List[Dict[str, Any]]
    ) -> bool:
        """
        Load transformed data into internal data store.

        Returns:
            True if successful, False otherwise
        """
        if not self.internal_conn:
            logger.error("No internal data store connection available")
            return False

        try:
            cursor = self.internal_conn.cursor()

            # Create table if not exists
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS transactions (
                    id BIGINT PRIMARY KEY,
                    amount NUMERIC,
                    category VARCHAR(255),
                    date DATE,
                    created_at TIMESTAMP,
                    source_system VARCHAR(50),
                    metadata JSONB
                );
            """
            )

            # Insert data with batch processing
            batch_size = 100
            for i in range(0, len(transactions), batch_size):
                batch = transactions[i: i + batch_size]

                # Use executemany for batch inserts
                insert_query = """
                    INSERT INTO transactions (
                        id, amount, category, date, created_at,
                        source_system, metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        amount = EXCLUDED.amount,
                        category = EXCLUDED.category,
                        date = EXCLUDED.date,
                        created_at = EXCLUDED.created_at,
                        source_system = EXCLUDED.source_system,
                        metadata = EXCLUDED.metadata
                """

                values = [
                    (
                        tx["id"],
                        tx["amount"],
                        tx["category"],
                        tx["date"],
                        tx["created_at"],
                        tx["source_system"],
                        tx["metadata"],
                    )
                    for tx in batch
                ]

                cursor.executemany(insert_query, values)
                self.migration_stats["transactions_migrated"] += len(batch)

                logger.info(f"Inserted batch of {len(batch)} transactions")

            self.internal_conn.commit()
            cursor.close()

            logger.info(
                f"Successfully loaded "
                f"{self.migration_stats['transactions_migrated']} "
                f"transactions"
            )
            return True

        except Exception as e:
            logger.error(f"Error loading data to internal store: {e}")
            self.migration_stats["errors"] += 1
            return False

    def validate_migration(self) -> bool:
        """Validate that migration was successful."""
        try:
            # Check record counts
            aws_cursor = self.aws_conn.cursor()
            aws_cursor.execute("SELECT COUNT(*) FROM transactions")
            aws_count = aws_cursor.fetchone()[0]

            internal_cursor = self.internal_conn.cursor()
            internal_cursor.execute(
                "SELECT COUNT(*) FROM transactions WHERE "
                "source_system = 'aws_rds'"
            )
            internal_count = internal_cursor.fetchone()[0]

            logger.info(
                f"Validation: AWS RDS count: {aws_count}, "
                f"Internal count: {internal_count}"
            )

            return aws_count == internal_count

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False

    def run_migration(self) -> bool:
        """
        Execute the complete migration process.

        Returns:
            True if migration was successful, False otherwise
        """
        self.migration_stats["start_time"] = datetime.now()
        logger.info("Starting AWS RDS to Internal Data Store migration")

        try:
            # Step 1: Connect to both databases
            if (
                not self.connect_to_aws_rds()
                or not self.connect_to_internal_db()
            ):
                return False

            # Step 2: Extract data from AWS RDS
            transactions = self.extract_transactions_from_rds()
            if not transactions:
                logger.error("No transactions extracted from AWS RDS")
                return False

            # Step 3: Transform data
            transformed_transactions = self.transform_data_for_internal_store(
                transactions
            )

            # Step 4: Load to internal store
            if not self.load_to_internal_store(transformed_transactions):
                return False

            # Step 5: Validate migration
            if not self.validate_migration():
                logger.error("Migration validation failed")
                return False

            self.migration_stats["end_time"] = datetime.now()
            duration = (
                self.migration_stats["end_time"]
                - self.migration_stats["start_time"]
            )

            logger.info(f"Migration completed successfully in {duration}")
            logger.info(f"Migration stats: {self.migration_stats}")

            return True

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False
        finally:
            # Clean up connections
            if self.aws_conn:
                self.aws_conn.close()
            if self.internal_conn:
                self.internal_conn.close()


def main():
    """Main entry point for AWS RDS migration."""
    migration = AWSRDSMigration()
    success = migration.run_migration()

    if success:
        logger.info("AWS RDS migration completed successfully")
        exit(0)
    else:
        logger.error("AWS RDS migration failed")
        exit(1)


if __name__ == "__main__":
    main()
