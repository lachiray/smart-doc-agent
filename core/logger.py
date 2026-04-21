"""
SQLite observability logger.

Records every request so the admin panel and evaluation scripts
have a complete audit trail of routing decisions, latency, and errors.
"""
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

DB_PATH = Path("data/logs.db")


def init_db() -> None:
    """Create the database and table if they don't exist. Safe to call repeatedly."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS requests (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp           TEXT    NOT NULL,
                input_text          TEXT    NOT NULL,
                input_length        INTEGER,
                selected_tool       TEXT,
                confidence          REAL,
                needs_clarification INTEGER,
                rationale           TEXT,
                tool_output         TEXT,
                router_latency_ms   REAL,
                tool_latency_ms     REAL,
                total_latency_ms    REAL,
                error               TEXT,
                user_feedback       INTEGER DEFAULT 0
            )
        """)
        conn.commit()


def log_request(
    input_text: str,
    selected_tool: Optional[str],
    confidence: Optional[float],
    needs_clarification: bool,
    rationale: Optional[str],
    tool_output: Optional[str],
    router_latency_ms: float,
    tool_latency_ms: float,
    error: Optional[str] = None,
) -> int:
    """Insert one request record and return its auto-increment ID."""
    total_ms = router_latency_ms + tool_latency_ms
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            """
            INSERT INTO requests (
                timestamp, input_text, input_length, selected_tool, confidence,
                needs_clarification, rationale, tool_output,
                router_latency_ms, tool_latency_ms, total_latency_ms,
                error, user_feedback
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                datetime.utcnow().isoformat(),
                input_text[:2000],
                len(input_text),
                selected_tool,
                confidence,
                int(needs_clarification),
                rationale,
                (tool_output or "")[:3000],
                round(router_latency_ms, 1),
                round(tool_latency_ms, 1),
                round(total_ms, 1),
                error,
            ),
        )
        conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]


def update_feedback(row_id: int, feedback: int) -> None:
    """Record user thumbs-up (1) or thumbs-down (-1) for a request."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE requests SET user_feedback = ? WHERE id = ?",
            (feedback, row_id),
        )
        conn.commit()


def get_recent_requests(limit: int = 50) -> list[dict]:
    """Return the most recent requests as a list of dicts."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM requests ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_metrics_summary() -> dict:
    """Compute aggregate operational metrics over all logged requests."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        total = conn.execute("SELECT COUNT(*) AS n FROM requests").fetchone()["n"]
        if total == 0:
            return {}

        avg_latency = conn.execute(
            "SELECT AVG(total_latency_ms) AS avg FROM requests WHERE error IS NULL"
        ).fetchone()["avg"]

        tool_counts = conn.execute(
            """
            SELECT selected_tool, COUNT(*) AS cnt
            FROM requests
            WHERE selected_tool IS NOT NULL
            GROUP BY selected_tool
            ORDER BY cnt DESC
            """
        ).fetchall()

        error_count = conn.execute(
            "SELECT COUNT(*) AS n FROM requests WHERE error IS NOT NULL"
        ).fetchone()["n"]

        clarification_count = conn.execute(
            "SELECT COUNT(*) AS n FROM requests WHERE needs_clarification = 1"
        ).fetchone()["n"]

        avg_confidence = conn.execute(
            "SELECT AVG(confidence) AS avg FROM requests WHERE confidence IS NOT NULL"
        ).fetchone()["avg"]

        thumbs_up = conn.execute(
            "SELECT COUNT(*) AS n FROM requests WHERE user_feedback = 1"
        ).fetchone()["n"]

    return {
        "total_requests": total,
        "avg_total_latency_ms": round(avg_latency or 0, 1),
        "tool_distribution": {r["selected_tool"]: r["cnt"] for r in tool_counts},
        "error_count": error_count,
        "error_rate_pct": round(error_count / total * 100, 1),
        "clarification_count": clarification_count,
        "clarification_rate_pct": round(clarification_count / total * 100, 1),
        "avg_confidence": round(avg_confidence or 0, 3),
        "thumbs_up_count": thumbs_up,
    }
