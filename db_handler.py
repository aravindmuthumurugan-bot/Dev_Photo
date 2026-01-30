"""
PostgreSQL Database Handler for AI Photo Validation

Database: photo_validation
Table: AI_Photo_Validation
"""

import psycopg2
from psycopg2 import sql, pool
from psycopg2.extras import Json, RealDictCursor
from typing import Dict, Optional
from datetime import datetime
import json

# Database Configuration
DB_CONFIG = {
    "host": "192.168.21.188",
    "port": 5432,
    "database": "photo_validation",
    "user": "usraiphoval",
    "password": "dS4!s2k6"
}

# Connection pool for efficient connection management
connection_pool = None


def init_connection_pool(min_conn: int = 1, max_conn: int = 10):
    """Initialize the database connection pool"""
    global connection_pool
    try:
        connection_pool = psycopg2.pool.ThreadedConnectionPool(
            min_conn,
            max_conn,
            **DB_CONFIG
        )
        print(f"[DB] Connection pool initialized (min={min_conn}, max={max_conn})")
        return True
    except Exception as e:
        print(f"[DB] Failed to initialize connection pool: {e}")
        return False


def get_connection():
    """Get a connection from the pool"""
    global connection_pool
    if connection_pool is None:
        init_connection_pool()
    return connection_pool.getconn()


def release_connection(conn):
    """Release a connection back to the pool"""
    global connection_pool
    if connection_pool and conn:
        connection_pool.putconn(conn)


def close_all_connections():
    """Close all connections in the pool"""
    global connection_pool
    if connection_pool:
        connection_pool.closeall()
        print("[DB] All connections closed")


def create_table_if_not_exists():
    """
    Create the AI_Photo_Validation table if it doesn't exist

    Schema:
    - validation_id: UUID (PRIMARY KEY)
    - matri_id: VARCHAR (FOREIGN KEY - user provided)
    - batch_id: UUID (optional - for batch validations)
    - timestamp: TIMESTAMP WITH TIME ZONE
    - photo_type: VARCHAR (PRIMARY/SECONDARY)
    - image_filename: VARCHAR
    - final_status: VARCHAR (ACCEPTED/REJECTED/SUSPENDED/MANUAL_REVIEW/ERROR)
    - final_decision: VARCHAR (APPROVE/REJECT/SUSPEND/MANUAL_REVIEW/ERROR)
    - final_action: VARCHAR
    - final_reason: TEXT
    - image_was_cropped: BOOLEAN
    - cropped_image_base64: TEXT (optional - stores base64 of cropped image)
    - checklist_summary: JSONB (stores checklist data)
    - stage1_checks: JSONB (stores stage 1 check results)
    - stage2_checks: JSONB (stores stage 2 check results)
    - library_usage: JSONB (stores library usage info)
    - validation_approach: VARCHAR (hybrid/stage1_only)
    - response_time_seconds: DECIMAL
    - gpu_used: BOOLEAN
    - gpu_info: JSONB (stores GPU configuration info)
    - created_at: TIMESTAMP WITH TIME ZONE (auto-generated)
    """

    create_table_sql = """
    CREATE TABLE IF NOT EXISTS AI_Photo_Validation (
        validation_id VARCHAR(36) PRIMARY KEY,
        matri_id VARCHAR(50) NOT NULL,
        batch_id VARCHAR(36),
        timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
        photo_type VARCHAR(20) NOT NULL,
        image_filename VARCHAR(255),
        final_status VARCHAR(30) NOT NULL,
        final_decision VARCHAR(30) NOT NULL,
        final_action VARCHAR(50),
        final_reason TEXT,
        image_was_cropped BOOLEAN DEFAULT FALSE,
        cropped_image_base64 TEXT,
        checklist_summary JSONB,
        stage1_checks JSONB,
        stage2_checks JSONB,
        library_usage JSONB,
        validation_approach VARCHAR(30),
        response_time_seconds DECIMAL(10, 3),
        gpu_used BOOLEAN DEFAULT FALSE,
        gpu_info JSONB,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

        CONSTRAINT fk_matri_id_check CHECK (matri_id IS NOT NULL AND matri_id != '')
    );

    -- Create indexes for faster queries
    CREATE INDEX IF NOT EXISTS idx_ai_photo_validation_matri_id
        ON AI_Photo_Validation(matri_id);

    CREATE INDEX IF NOT EXISTS idx_ai_photo_validation_batch_id
        ON AI_Photo_Validation(batch_id);

    CREATE INDEX IF NOT EXISTS idx_ai_photo_validation_final_status
        ON AI_Photo_Validation(final_status);

    CREATE INDEX IF NOT EXISTS idx_ai_photo_validation_timestamp
        ON AI_Photo_Validation(timestamp);

    CREATE INDEX IF NOT EXISTS idx_ai_photo_validation_photo_type
        ON AI_Photo_Validation(photo_type);
    """

    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute(create_table_sql)
            conn.commit()
        print("[DB] Table AI_Photo_Validation created/verified successfully")
        return True
    except Exception as e:
        print(f"[DB] Error creating table: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            release_connection(conn)


def insert_validation_result(validation_data: Dict, batch_id: Optional[str] = None,
                             response_time: Optional[float] = None,
                             gpu_info: Optional[Dict] = None) -> bool:
    """
    Insert a validation result into the database

    Args:
        validation_data: The formatted validation result from format_validation_result()
        batch_id: Optional batch ID for batch validations
        response_time: Response time in seconds
        gpu_info: GPU configuration info

    Returns:
        True if successful, False otherwise
    """

    insert_sql = """
    INSERT INTO AI_Photo_Validation (
        validation_id,
        matri_id,
        batch_id,
        timestamp,
        photo_type,
        image_filename,
        final_status,
        final_decision,
        final_action,
        final_reason,
        image_was_cropped,
        cropped_image_base64,
        checklist_summary,
        stage1_checks,
        stage2_checks,
        library_usage,
        validation_approach,
        response_time_seconds,
        gpu_used,
        gpu_info
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
    )
    """

    conn = None
    try:
        # Extract matri_id from stage2_checks if available, otherwise use a placeholder
        matri_id = None
        if validation_data.get("stage2_checks") and isinstance(validation_data["stage2_checks"], dict):
            matri_id = validation_data["stage2_checks"].get("matri_id")

        # If matri_id not found in stage2_checks, it should be passed separately
        # This will be handled by the caller

        # Determine if GPU was used
        gpu_used = False
        if validation_data.get("library_usage"):
            gpu_used = validation_data["library_usage"].get("gpu_used", False)

        values = (
            validation_data.get("validation_id"),
            validation_data.get("matri_id", "UNKNOWN"),  # This will be overridden by caller
            batch_id,
            validation_data.get("timestamp", datetime.utcnow().isoformat()),
            validation_data.get("photo_type"),
            validation_data.get("image_filename"),
            validation_data.get("final_status"),
            validation_data.get("final_decision"),
            validation_data.get("final_action"),
            validation_data.get("final_reason"),
            validation_data.get("image_was_cropped", False),
            validation_data.get("cropped_image_base64"),
            Json(validation_data.get("checklist_summary")) if validation_data.get("checklist_summary") else None,
            Json(validation_data.get("stage1_checks")) if validation_data.get("stage1_checks") else None,
            Json(validation_data.get("stage2_checks")) if validation_data.get("stage2_checks") else None,
            Json(validation_data.get("library_usage")) if validation_data.get("library_usage") else None,
            validation_data.get("validation_approach"),
            response_time,
            gpu_used,
            Json(gpu_info) if gpu_info else None
        )

        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute(insert_sql, values)
            conn.commit()

        print(f"[DB] Validation result saved: {validation_data.get('validation_id')}")
        return True

    except Exception as e:
        print(f"[DB] Error inserting validation result: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            release_connection(conn)


def insert_validation_with_matri_id(validation_data: Dict, matri_id: str,
                                     batch_id: Optional[str] = None,
                                     response_time: Optional[float] = None,
                                     gpu_info: Optional[Dict] = None) -> bool:
    """
    Insert a validation result with explicit matri_id

    Args:
        validation_data: The formatted validation result
        matri_id: The matri_id provided by the user
        batch_id: Optional batch ID
        response_time: Response time in seconds
        gpu_info: GPU configuration info

    Returns:
        True if successful, False otherwise
    """

    insert_sql = """
    INSERT INTO AI_Photo_Validation (
        validation_id,
        matri_id,
        batch_id,
        timestamp,
        photo_type,
        image_filename,
        final_status,
        final_decision,
        final_action,
        final_reason,
        image_was_cropped,
        cropped_image_base64,
        checklist_summary,
        stage1_checks,
        stage2_checks,
        library_usage,
        validation_approach,
        response_time_seconds,
        gpu_used,
        gpu_info
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
    )
    """

    conn = None
    try:
        # Determine if GPU was used
        gpu_used = False
        if validation_data.get("library_usage"):
            gpu_used = validation_data["library_usage"].get("gpu_used", False)

        values = (
            validation_data.get("validation_id"),
            matri_id,
            batch_id,
            validation_data.get("timestamp", datetime.utcnow().isoformat()),
            validation_data.get("photo_type"),
            validation_data.get("image_filename"),
            validation_data.get("final_status"),
            validation_data.get("final_decision"),
            validation_data.get("final_action"),
            validation_data.get("final_reason"),
            validation_data.get("image_was_cropped", False),
            validation_data.get("cropped_image_base64"),
            Json(validation_data.get("checklist_summary")) if validation_data.get("checklist_summary") else None,
            Json(validation_data.get("stage1_checks")) if validation_data.get("stage1_checks") else None,
            Json(validation_data.get("stage2_checks")) if validation_data.get("stage2_checks") else None,
            Json(validation_data.get("library_usage")) if validation_data.get("library_usage") else None,
            validation_data.get("validation_approach"),
            response_time,
            gpu_used,
            Json(gpu_info) if gpu_info else None
        )

        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute(insert_sql, values)
            conn.commit()

        print(f"[DB] Validation result saved: {validation_data.get('validation_id')} for matri_id: {matri_id}")
        return True

    except Exception as e:
        print(f"[DB] Error inserting validation result: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            release_connection(conn)


def get_validation_by_id(validation_id: str) -> Optional[Dict]:
    """Get a validation result by validation_id"""

    select_sql = """
    SELECT * FROM AI_Photo_Validation
    WHERE validation_id = %s
    """

    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(select_sql, (validation_id,))
            result = cursor.fetchone()
            return dict(result) if result else None
    except Exception as e:
        print(f"[DB] Error fetching validation: {e}")
        return None
    finally:
        if conn:
            release_connection(conn)


def get_validations_by_matri_id(matri_id: str, limit: int = 100) -> list:
    """Get all validation results for a matri_id"""

    select_sql = """
    SELECT * FROM AI_Photo_Validation
    WHERE matri_id = %s
    ORDER BY timestamp DESC
    LIMIT %s
    """

    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(select_sql, (matri_id, limit))
            results = cursor.fetchall()
            return [dict(row) for row in results]
    except Exception as e:
        print(f"[DB] Error fetching validations: {e}")
        return []
    finally:
        if conn:
            release_connection(conn)


def get_validations_by_batch_id(batch_id: str) -> list:
    """Get all validation results for a batch_id"""

    select_sql = """
    SELECT * FROM AI_Photo_Validation
    WHERE batch_id = %s
    ORDER BY timestamp
    """

    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(select_sql, (batch_id,))
            results = cursor.fetchall()
            return [dict(row) for row in results]
    except Exception as e:
        print(f"[DB] Error fetching batch validations: {e}")
        return []
    finally:
        if conn:
            release_connection(conn)


def get_validation_statistics(start_date: Optional[str] = None,
                               end_date: Optional[str] = None) -> Dict:
    """Get validation statistics"""

    base_sql = """
    SELECT
        COUNT(*) as total_validations,
        COUNT(CASE WHEN final_status = 'ACCEPTED' THEN 1 END) as approved,
        COUNT(CASE WHEN final_status = 'REJECTED' THEN 1 END) as rejected,
        COUNT(CASE WHEN final_status = 'SUSPENDED' THEN 1 END) as suspended,
        COUNT(CASE WHEN final_status = 'MANUAL_REVIEW' THEN 1 END) as manual_review,
        COUNT(CASE WHEN photo_type = 'PRIMARY' THEN 1 END) as primary_photos,
        COUNT(CASE WHEN photo_type = 'SECONDARY' THEN 1 END) as secondary_photos,
        AVG(response_time_seconds) as avg_response_time,
        COUNT(CASE WHEN gpu_used = true THEN 1 END) as gpu_processed
    FROM AI_Photo_Validation
    """

    conditions = []
    params = []

    if start_date:
        conditions.append("timestamp >= %s")
        params.append(start_date)

    if end_date:
        conditions.append("timestamp <= %s")
        params.append(end_date)

    if conditions:
        base_sql += " WHERE " + " AND ".join(conditions)

    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(base_sql, params if params else None)
            result = cursor.fetchone()
            return dict(result) if result else {}
    except Exception as e:
        print(f"[DB] Error fetching statistics: {e}")
        return {}
    finally:
        if conn:
            release_connection(conn)


def test_connection() -> bool:
    """Test database connection"""
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            print(f"[DB] Connection test successful: {result}")
            return True
    except Exception as e:
        print(f"[DB] Connection test failed: {e}")
        return False
    finally:
        if conn:
            release_connection(conn)


# Initialize on module import
def initialize_database():
    """Initialize database connection pool and create table"""
    print("[DB] Initializing database...")
    if init_connection_pool():
        if test_connection():
            create_table_if_not_exists()
            print("[DB] Database initialization complete")
            return True
    print("[DB] Database initialization failed")
    return False
