"""
SQLite database layer for storing parking snapshots and occupancy summaries.
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

DB_PATH = Path(__file__).parent.parent / "data" / "parking.db"


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS slot_snapshots (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            slot_id     INTEGER NOT NULL,
            slot_name   TEXT    NOT NULL,
            status      TEXT    NOT NULL CHECK(status IN ('occupied','empty')),
            confidence  REAL    DEFAULT 1.0,
            created_at  TEXT    DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS occupancy_summary (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL UNIQUE,
            total_occupied  INTEGER NOT NULL,
            total_empty     INTEGER NOT NULL,
            total_slots     INTEGER NOT NULL,
            occupancy_rate  REAL    NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_snap_ts   ON slot_snapshots(timestamp);
        CREATE INDEX IF NOT EXISTS idx_snap_slot ON slot_snapshots(slot_id);
        CREATE INDEX IF NOT EXISTS idx_sum_ts    ON occupancy_summary(timestamp);
        """
    )
    conn.commit()
    conn.close()


# ── Write helpers ────────────────────────────────────────────────────────────

def store_snapshot(timestamp: str, slot_statuses: List[Dict]):
    """Insert one detection snapshot (all slots at a single point in time)."""
    conn = get_connection()
    cur = conn.cursor()
    total_occ = sum(1 for s in slot_statuses if s["status"] == "occupied")
    total = len(slot_statuses)

    for s in slot_statuses:
        cur.execute(
            "INSERT INTO slot_snapshots (timestamp,slot_id,slot_name,status,confidence) "
            "VALUES (?,?,?,?,?)",
            (timestamp, s["id"], s["name"], s["status"], s.get("confidence", 1.0)),
        )

    cur.execute(
        "INSERT OR REPLACE INTO occupancy_summary "
        "(timestamp,total_occupied,total_empty,total_slots,occupancy_rate) "
        "VALUES (?,?,?,?,?)",
        (timestamp, total_occ, total - total_occ, total,
         round(total_occ / max(total, 1), 4)),
    )
    conn.commit()
    conn.close()


def store_bulk_data(historical_data: List[Dict]):
    """Bulk-insert simulated historical data (clears old rows first)."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM slot_snapshots")
    cur.execute("DELETE FROM occupancy_summary")

    for rec in historical_data:
        ts = rec["timestamp"]
        for s in rec["slots"]:
            cur.execute(
                "INSERT INTO slot_snapshots (timestamp,slot_id,slot_name,status) "
                "VALUES (?,?,?,?)",
                (ts, s["id"], s["name"], s["status"]),
            )
        cur.execute(
            "INSERT OR REPLACE INTO occupancy_summary "
            "(timestamp,total_occupied,total_empty,total_slots,occupancy_rate) "
            "VALUES (?,?,?,?,?)",
            (ts, rec["total_occupied"], rec["total_empty"], rec["total_slots"],
             round(rec["total_occupied"] / max(rec["total_slots"], 1), 4)),
        )

    conn.commit()
    conn.close()


# ── Read helpers ─────────────────────────────────────────────────────────────

def get_latest_snapshot() -> Optional[List[Dict]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT MAX(timestamp) AS ts FROM slot_snapshots")
    row = cur.fetchone()
    if not row or not row["ts"]:
        conn.close()
        return None
    latest = row["ts"]
    cur.execute(
        "SELECT slot_id, slot_name, status, confidence "
        "FROM slot_snapshots WHERE timestamp=? ORDER BY slot_id",
        (latest,),
    )
    results = [dict(r) for r in cur.fetchall()]
    conn.close()
    return results


def get_occupancy_history(hours: int = 24) -> List[Dict]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT timestamp,total_occupied,total_empty,total_slots,occupancy_rate "
        "FROM occupancy_summary ORDER BY timestamp DESC LIMIT ?",
        (hours * 4,),
    )
    results = [dict(r) for r in cur.fetchall()]
    results.reverse()
    conn.close()
    return results


def get_heatmap_data() -> List[Dict]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            CAST(strftime('%w', timestamp) AS INTEGER) AS day_of_week,
            CAST(strftime('%H', timestamp) AS INTEGER) AS hour,
            AVG(occupancy_rate) AS avg_rate,
            COUNT(*) AS count
        FROM occupancy_summary
        GROUP BY day_of_week, hour
        ORDER BY day_of_week, hour
        """
    )
    results = [dict(r) for r in cur.fetchall()]
    conn.close()
    return results


def get_summary_for_training() -> List[Dict]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT timestamp,total_occupied,total_empty,total_slots,occupancy_rate "
        "FROM occupancy_summary ORDER BY timestamp"
    )
    results = [dict(r) for r in cur.fetchall()]
    conn.close()
    return results
