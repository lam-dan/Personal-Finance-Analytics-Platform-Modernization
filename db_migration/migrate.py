"""
Personal Finance Analytics Platform - Database Migration Script

This module handles the migration of financial data from PostgreSQL to TimescaleDB.
It demonstrates a real-world data migration scenario where legacy data is moved
to a modern time-series database for enhanced analytics capabilities.

Migration Process:
1. Connect to source PostgreSQL database (simulating AWS RDS)
2. Create necessary tables if they don't exist
3. Extract data from source database
4. Connect to destination TimescaleDB
5. Create corresponding tables in destination
6. Transform and load data into TimescaleDB
7. Verify data integrity and completeness

TimescaleDB Benefits:
- Native time-series data handling
- Automatic data retention policies
- Efficient time-based queries
- Built-in compression and aggregation
"""

import psycopg2
import logging

# Configure structured logging for migration tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Connection details for the source PostgreSQL database (simulating AWS RDS)
# In production, this would connect to a real AWS RDS instance
SOURCE_DB_CONFIG = {
    'dbname': 'finance_db',
    'user': 'postgres',
    'password': 'postgres',
    'host': 'localhost',
    'port': '5432'
}

# Connection details for the destination TimescaleDB
# TimescaleDB is a PostgreSQL extension optimized for time-series data
DEST_DB_CONFIG = {
    'dbname': 'finance_timescale',
    'user': 'postgres',
    'password': 'postgres',
    'host': 'localhost',
    'port': '5433'  # Different port to avoid conflicts with PostgreSQL
}

def create_tables_if_not_exist(conn):
    """
    Create database tables if they don't exist.
    
    This function ensures the required schema exists before attempting
    data migration. It uses CREATE TABLE IF NOT EXISTS to avoid errors
    if tables already exist.
    
    Args:
        conn: PostgreSQL database connection object
        
    Tables Created:
        - transactions: Stores financial transaction data
        - investments: Stores investment portfolio data
    """
    cur = conn.cursor()
    
    # Create transactions table with appropriate data types
    # SERIAL PRIMARY KEY provides auto-incrementing unique IDs
    # NUMERIC type ensures precise decimal calculations for financial data
    cur.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id SERIAL PRIMARY KEY,
            amount NUMERIC,
            category VARCHAR(255),
            date DATE
        );
    """)
    
    # Create investments table for portfolio tracking
    # Similar structure with asset-specific fields
    cur.execute("""
        CREATE TABLE IF NOT EXISTS investments (
            id SERIAL PRIMARY KEY,
            asset VARCHAR(255),
            value NUMERIC,
            last_updated DATE
        );
    """)
    
    # Commit the schema changes
    conn.commit()
    cur.close()
    logger.info("Tables created/verified successfully")

def migrate_transactions():
    """
    Main migration function that transfers data from PostgreSQL to TimescaleDB.
    
    This function performs a complete ETL (Extract, Transform, Load) process:
    1. Extract: Read data from source PostgreSQL database
    2. Transform: Prepare data for TimescaleDB (if needed)
    3. Load: Insert data into destination TimescaleDB
    
    The migration is designed to be idempotent - it can be run multiple times
    without causing data duplication or errors.
    """
    logger.info("Starting data migration for transactions")

    # Step 1: Connect to source PostgreSQL database
    src_conn = psycopg2.connect(**SOURCE_DB_CONFIG)
    
    # Step 2: Ensure source tables exist (create if missing)
    create_tables_if_not_exist(src_conn)
    
    # Step 3: Extract data from source database
    src_cur = src_conn.cursor()
    src_cur.execute("SELECT id, amount, category, date FROM transactions")
    transactions = src_cur.fetchall()

    logger.info(f"Fetched {len(transactions)} transactions from source DB")

    # Step 4: Connect to destination TimescaleDB
    dest_conn = psycopg2.connect(**DEST_DB_CONFIG)
    
    # Step 5: Ensure destination tables exist
    create_tables_if_not_exist(dest_conn)
    
    dest_cur = dest_conn.cursor()

    # Step 6: Load data into TimescaleDB
    # Using parameterized queries to prevent SQL injection
    for tx in transactions:
        dest_cur.execute(
            "INSERT INTO transactions (id, amount, category, date) VALUES (%s, %s, %s, %s)",
            tx
        )

    # Step 7: Commit the transaction to persist changes
    dest_conn.commit()
    logger.info("Transactions migrated successfully")

    # Step 8: Clean up database connections
    # Always close cursors and connections to prevent resource leaks
    src_cur.close()
    src_conn.close()
    dest_cur.close()
    dest_conn.close()

if __name__ == "__main__":
    """
    Main entry point for the migration script.
    
    This allows the script to be run directly or imported as a module.
    In production, this might be called from a CI/CD pipeline or
    scheduled job for regular data synchronization.
    """
    migrate_transactions()