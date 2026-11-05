"""LLMetrics SQLite Persistence Layer.

Provides connection pooling, WAL mode initialization, schema creation,
and transactional CRUD helpers for requests, incidents, settings, and analytical caches.
"""

import json
import os
import sqlite3
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd

def get_db_path(db_path: Optional[str] = None) -> str:
    """Returns the effective database path, prioritizing explicit parameter or environment variable."""
    if db_path is not None:
        return db_path
    return os.getenv("LLMETRICS_DB_PATH", "backend/llmetrics.db")

DEFAULT_DB_PATH = get_db_path()

def get_db_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Creates a SQLite connection with row factory and WAL journal mode."""
    target_path = get_db_path(db_path)
    os.makedirs(os.path.dirname(target_path) if os.path.dirname(target_path) else ".", exist_ok=True)
    conn = sqlite3.connect(target_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn

def init_db(db_path: Optional[str] = None) -> None:
    """Initializes tables for requests, incidents, analysis cache, and settings."""
    target_path = get_db_path(db_path)
    conn = get_db_connection(target_path)
    with conn:
        # Requests table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                request_id TEXT UNIQUE NOT NULL,
                feature TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL,
                latency_ms REAL NOT NULL,
                status TEXT NOT NULL,
                retry_count INTEGER NOT NULL DEFAULT 0,
                cache_hit INTEGER NOT NULL DEFAULT 0,
                prompt_text TEXT,
                cost_usd REAL NOT NULL
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON requests(timestamp);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_requests_model ON requests(model);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_requests_status ON requests(status);")

        # Incidents table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                start_day INTEGER NOT NULL,
                end_day INTEGER NOT NULL,
                target_model TEXT NOT NULL,
                description TEXT
            );
        """)

        # Metadata / Settings key-value
        conn.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)

        # Analysis cache table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analysis_cache (
                dataset_version TEXT NOT NULL,
                section_name TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (dataset_version, section_name)
            );
        """)

        # Default settings if missing
        conn.execute("""
            INSERT OR IGNORE INTO settings (key, value) VALUES ('monthly_budget', '1200.0');
        """)
        conn.execute("""
            INSERT OR IGNORE INTO settings (key, value) VALUES ('dataset_version', 'none');
        """)
    conn.close()

def save_dataframe_to_db(df: pd.DataFrame, db_path: Optional[str] = None) -> int:
    """Saves cleaned DataFrame of requests into SQLite, replacing previous dataset."""
    conn = get_db_connection(db_path)
    with conn:
        conn.execute("DELETE FROM requests;")
        df.to_sql("requests", conn, if_exists="append", index=False)
        # Invalidate old cache
        conn.execute("DELETE FROM analysis_cache;")
        # Generate new dataset version hash from count + first/last timestamp
        ver_str = f"v_{len(df)}_{df['timestamp'].min()}_{df['timestamp'].max()}"
        conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('dataset_version', ?);", (ver_str,))
    conn.close()
    return len(df)

def load_dataframe_from_db(db_path: Optional[str] = None) -> pd.DataFrame:
    """Loads all request records from database as pandas DataFrame."""
    conn = get_db_connection(db_path)
    df = pd.read_sql_query("SELECT * FROM requests ORDER BY timestamp ASC;", conn)
    conn.close()
    return df

def save_incidents_to_db(incidents: List[Dict[str, Any]], db_path: Optional[str] = None) -> None:
    """Saves ground truth incidents to DB."""
    conn = get_db_connection(db_path)
    with conn:
        conn.execute("DELETE FROM incidents;")
        for inc in incidents:
            conn.execute(
                """INSERT OR REPLACE INTO incidents (id, name, type, start_day, end_day, target_model, description)
                   VALUES (?, ?, ?, ?, ?, ?, ?);""",
                (inc["id"], inc["name"], inc["type"], inc["start_day"], inc["end_day"], inc["target_model"], inc["description"])
            )
    conn.close()

def load_incidents_from_db(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves ground truth incidents from DB."""
    conn = get_db_connection(db_path)
    rows = conn.execute("SELECT * FROM incidents ORDER BY start_day ASC;").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_setting(key: str, default: Optional[str] = None, db_path: Optional[str] = None) -> Optional[str]:
    """Gets setting value by key."""
    conn = get_db_connection(db_path)
    row = conn.execute("SELECT value FROM settings WHERE key = ?;", (key,)).fetchone()
    conn.close()
    return row["value"] if row else default

def set_setting(key: str, value: str, db_path: Optional[str] = None) -> None:
    """Sets setting key-value pair."""
    conn = get_db_connection(db_path)
    with conn:
        conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?);", (key, value))
    conn.close()

def get_cached_analysis(section_name: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves cached analysis JSON for current dataset version."""
    conn = get_db_connection(db_path)
    ver = get_setting("dataset_version", default="none", db_path=db_path)
    if ver == "none":
        conn.close()
        return None

    row = conn.execute(
        "SELECT payload_json FROM analysis_cache WHERE dataset_version = ? AND section_name = ?;",
        (ver, section_name)
    ).fetchone()
    conn.close()

    if row:
        return json.loads(row["payload_json"])
    return None

def set_cached_analysis(section_name: str, data: Dict[str, Any], db_path: Optional[str] = None) -> None:
    """Stores analytical result JSON in cache for current dataset version."""
    ver = get_setting("dataset_version", default="none", db_path=db_path)
    if ver == "none":
        return

    conn = get_db_connection(db_path)
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO analysis_cache (dataset_version, section_name, payload_json) VALUES (?, ?, ?);",
            (ver, section_name, json.dumps(data, default=str))
        )
    conn.close()
