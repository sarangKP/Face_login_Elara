"""
timeline.py — Persistent timeline of emotion + scene snapshots.

Storage : SQLite at faces/timeline.db
Strategy: only write a row when something meaningfully changed —
          emotion changed, affected-flag flipped, or >50% of subjects swapped.
          This prevents thousands of identical rows per day.

Query results:
    GET /timeline          → all users, newest first
    GET /timeline/<user>   → one user, newest first
"""
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from analyzer import Analysis

log = logging.getLogger(__name__)

DB_PATH = Path("faces/timeline.db")
DB_PATH.parent.mkdir(exist_ok=True)


# ── Schema ────────────────────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


with _connect() as _c:
    _c.execute("""
        CREATE TABLE IF NOT EXISTS snapshots (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT    NOT NULL,
            user        TEXT    NOT NULL,
            emotion     TEXT    NOT NULL,
            confidence  REAL    NOT NULL,
            scene       TEXT    NOT NULL,
            subjects    TEXT    NOT NULL,
            affected    INTEGER NOT NULL,
            reason      TEXT    NOT NULL,
            summary     TEXT    NOT NULL,
            thumbnail   BLOB
        )
    """)
    _c.execute("CREATE INDEX IF NOT EXISTS idx_user_ts ON snapshots(user, ts)")


# ── Change detection ──────────────────────────────────────────────────────────

def _last_row(user: str) -> Optional[sqlite3.Row]:
    with _connect() as conn:
        return conn.execute(
            "SELECT * FROM snapshots WHERE user=? ORDER BY id DESC LIMIT 1",
            (user,)
        ).fetchone()


def _is_change(prev: Optional[sqlite3.Row], curr: Analysis) -> bool:
    if prev is None:
        return True                             # always write first snapshot
    if prev["emotion"] != curr["emotion"]:
        return True                             # emotion changed
    if bool(prev["affected"]) != curr["emotion_affected_by_scene"]:
        return True                             # affected flag flipped
    ps = {s.lower() for s in json.loads(prev["subjects"])}
    cs = {s.lower() for s in curr["subjects"]}
    if not ps and not cs:
        return False
    union = ps | cs
    jaccard = len(ps & cs) / len(union)
    return jaccard < 0.5                        # >50% subjects changed


# ── Serialization ─────────────────────────────────────────────────────────────

def _row_to_dict(r: sqlite3.Row) -> dict:
    return {
        "id":               r["id"],
        "ts":               r["ts"],
        "user":             r["user"],
        "emotion":          r["emotion"],
        "confidence":       round(r["confidence"], 3),
        "scene":            r["scene"],
        "subjects":         json.loads(r["subjects"]),
        "emotion_affected": bool(r["affected"]),
        "reason":           r["reason"],
        "summary":          r["summary"],
        "has_thumbnail":    r["thumbnail"] is not None,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def last_for(user: str) -> Optional[dict]:
    """Most recent snapshot for this user, or None."""
    row = _last_row(user)
    return _row_to_dict(row) if row else None


def maybe_record(
    user:      str,
    analysis:  Analysis,
    thumbnail: Optional[bytes] = None,
) -> Optional[int]:
    """
    Write a snapshot row only if it differs from the last one for this user.
    Returns the new row id, or None if nothing was written.
    """
    prev = _last_row(user)
    if not _is_change(prev, analysis):
        return None

    ts = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO snapshots "
            "(ts, user, emotion, confidence, scene, subjects, affected, reason, summary, thumbnail) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                ts, user,
                analysis["emotion"],
                analysis["emotion_confidence"],
                analysis["scene_description"],
                json.dumps(analysis["subjects"]),
                int(analysis["emotion_affected_by_scene"]),
                analysis["reason"],
                analysis["summary"],
                thumbnail,
            )
        )
        new_id = cur.lastrowid

    log.info("timeline: #%d [%s] %s", new_id, user, analysis["summary"])
    return new_id


def recent(user: Optional[str] = None, limit: int = 50) -> list:
    """Return recent snapshots as plain dicts, newest first."""
    with _connect() as conn:
        if user:
            rows = conn.execute(
                "SELECT * FROM snapshots WHERE user=? ORDER BY id DESC LIMIT ?",
                (user, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM snapshots ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_thumbnail(snapshot_id: int) -> Optional[bytes]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT thumbnail FROM snapshots WHERE id=?", (snapshot_id,)
        ).fetchone()
    return row["thumbnail"] if row else None