"""
PostgreSQL Database Connectivity Test
"""

import psycopg2
from psycopg2 import OperationalError

# Database connection details
DB_CONFIG = {
    "host": "192.168.21.188",
    "database": "photo_validation",
    "user": "usraiphoval",
    "password": "dS4!s2k6",
    "port": 5432
}


def test_connection():
    """Test PostgreSQL database connectivity."""
    connection = None
    try:
        print("Connecting to PostgreSQL database...")
        connection = psycopg2.connect(**DB_CONFIG)

        cursor = connection.cursor()

        # Get PostgreSQL version
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"Connection successful!")
        print(f"PostgreSQL version: {version[0]}")

        # Get current database
        cursor.execute("SELECT current_database();")
        db_name = cursor.fetchone()
        print(f"Connected to database: {db_name[0]}")

        # Get current user
        cursor.execute("SELECT current_user;")
        user = cursor.fetchone()
        print(f"Connected as user: {user[0]}")

        # Get list of tables in the database
        print("\n--- Tables in database ---")
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        tables = cursor.fetchall()

        if tables:
            for i, table in enumerate(tables, 1):
                print(f"  {i}. {table[0]}")
            print(f"\nTotal tables: {len(tables)}")
        else:
            print("  No tables found in 'public' schema.")

        cursor.close()
        print("\nDatabase connectivity test PASSED!")
        return True

    except OperationalError as e:
        print(f"Connection FAILED!")
        print(f"Error: {e}")
        return False

    finally:
        if connection:
            connection.close()
            print("Connection closed.")


if __name__ == "__main__":
    test_connection()
