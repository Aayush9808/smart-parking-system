"""
Database — SQLite persistence for occupancy history.
"""

import sqlite3
import json
from config.settings import DB_PATH


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = _connect()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS occupancy_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            total_slots INTEGER NOT NULL,
            occupied    INTEGER NOT NULL,
            empty       INTEGER NOT NULL,
            occupancy_rate REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS slot_snapshots (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            slot_id     INTEGER NOT NULL,
            slot_name   TEXT NOT NULL,
            status      TEXT NOT NULL,
            confidence  REAL DEFAULT 0.0
        );
    """)
    conn.commit()
    conn.close()


def save_occupancy(record: dict):
    """Save a single occupancy record."""
    conn = _connect()
    conn.execute(
        "INSERT INTO occupancy_history (timestamp, total_slots, occupied, empty, occupancy_rate) "
        "VALUES (?, ?, ?, ?, ?)",
        (record["timestamp"], record["total_slots"], record["occupied"],
         record["empty"], record["occupancy_rate"]),
    )
    conn.commit()
    conn.close()


def save_occupancy_batch(records: list[dict]):
    """Save multiple occupancy records at once."""
    conn = _connect()
    conn.executemany(
        "INSERT INTO occupancy_history (timestamp, total_slots, occupied, empty, occupancy_rate) "
        "VALUES (?, ?, ?, ?, ?)",
        [(r["timestamp"], r["total_slots"], r["occupied"],
          r["empty"], r["occupancy_rate"]) for r in records],
    )
    conn.commit()
    conn.close()


def save_slot_snapshot(timestamp: str, slots: list[dict]):
    """Save slot states for a detection event."""
    conn = _connect()
    conn.executemany(
        "INSERT INTO slot_snapshots (timestamp, slot_id, slot_name, status, confidence) "
        "VALUES (?, ?, ?, ?, ?)",
        [(timestamp, s["id"], s["name"], s["status"],
          s.get("confidence", 0.0)) for s in slots],
    )
    conn.commit()
    conn.close()


def get_history(limit: int = 336) -> list[dict]:
    """Get recent occupancy history (default: 14 days × 24 hours)."""
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM occupancy_history ORDER BY timestamp DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in reversed(rows)]


def get_heatmap_data() -> list[dict]:
    """Get occupancy grouped by day-of-week and hour for heatmap."""
    conn = _connect()
    rows = conn.execute("""
        SELECT
            CAST(strftime('%w', timestamp) AS INTEGER) AS day,
            CAST(strftime('%H', timestamp) AS INTEGER) AS hour,
            ROUND(AVG(occupancy_rate), 3) AS avg_rate,
            COUNT(*) AS samples
        FROM occupancy_history
        GROUP BY day, hour
        ORDER BY day, hour
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_record_count() -> int:
    conn = _connect()
    count = conn.execute("SELECT COUNT(*) FROM occupancy_history").fetchone()[0]
    conn.close()
    return count


def clear_history():
    """Clear all historical data."""
    conn = _connect()
    conn.execute("DELETE FROM occupancy_history")
    conn.execute("DELETE FROM slot_snapshots")
    conn.commit()
    conn.close()
