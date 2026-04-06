from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def list_tables(conn: sqlite3.Connection) -> List[str]:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    return [r[0] for r in cur.fetchall()]


def normalize_token(token) -> str:
    if isinstance(token, (bytes, bytearray, memoryview)):
        return bytes(token).hex()
    return str(token)


def maybe_db_token(token):
    if isinstance(token, str):
        try:
            return bytes.fromhex(token)
        except ValueError:
            return token
    return token


def _safe_query(conn: sqlite3.Connection, query: str):
    return conn.execute(query).fetchall()


def fetch_log_meta(conn: sqlite3.Connection) -> Dict:
    row = conn.execute("SELECT * FROM log LIMIT 1").fetchone()
    return dict(row) if row else {}


def fetch_lidar_pcs(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    query = """
    SELECT lp.token, lp.timestamp, lp.scene_token, lp.ego_pose_token, s.name AS scene_name
    FROM lidar_pc AS lp
    LEFT JOIN scene AS s ON s.token = lp.scene_token
    ORDER BY lp.timestamp ASC
    """
    return conn.execute(query).fetchall()


def fetch_ego_poses(conn: sqlite3.Connection) -> Dict[str, sqlite3.Row]:
    rows = conn.execute("SELECT * FROM ego_pose ORDER BY timestamp ASC").fetchall()
    return {normalize_token(row['token']): row for row in rows}


def fetch_boxes_for_lidar_token(conn: sqlite3.Connection, lidar_pc_token) -> List[sqlite3.Row]:
    """Load tracked boxes for one lidar frame using the official track->category join."""
    q = """
    SELECT
        lb.*,
        COALESCE(c.name, 'unknown') AS category,
        t.width AS track_width,
        t.length AS track_length,
        t.height AS track_height
    FROM lidar_box AS lb
    INNER JOIN track AS t ON t.token = lb.track_token
    LEFT JOIN category AS c ON c.token = t.category_token
    WHERE lb.lidar_pc_token = ?
    ORDER BY lb.track_token ASC
    """
    return conn.execute(q, [maybe_db_token(lidar_pc_token)]).fetchall()


def fetch_scenario_tags(conn: sqlite3.Connection) -> Dict[str, List[str]]:
    rows = conn.execute("SELECT lidar_pc_token, type FROM scenario_tag").fetchall()
    tags: Dict[str, List[str]] = {}
    for row in rows:
        key = normalize_token(row['lidar_pc_token'])
        tags.setdefault(key, []).append(str(row['type']))
    return tags


def fetch_traffic_light_status(conn: sqlite3.Connection) -> Dict[str, List[sqlite3.Row]]:
    tables = set(list_tables(conn))
    if 'traffic_light_status' not in tables:
        return {}
    rows = conn.execute("SELECT * FROM traffic_light_status").fetchall()
    out: Dict[str, List[sqlite3.Row]] = {}
    for row in rows:
        out.setdefault(normalize_token(row['lidar_pc_token']), []).append(row)
    return out


def fetch_boxes_in_time_window(
    conn: sqlite3.Connection,
    start_ts: int,
    end_ts: int,
    track_tokens: Sequence[str] | None = None,
) -> List[sqlite3.Row]:
    base = """
    SELECT
        lb.*,
        lp.timestamp AS lidar_timestamp,
        COALESCE(c.name, 'unknown') AS category,
        t.width AS track_width,
        t.length AS track_length,
        t.height AS track_height
    FROM lidar_box AS lb
    INNER JOIN lidar_pc AS lp ON lp.token = lb.lidar_pc_token
    INNER JOIN track AS t ON t.token = lb.track_token
    LEFT JOIN category AS c ON c.token = t.category_token
    WHERE lp.timestamp >= ? AND lp.timestamp <= ?
    """
    args: List[object] = [start_ts, end_ts]
    if track_tokens:
        placeholders = ','.join(['?'] * len(track_tokens))
        base += f" AND lb.track_token IN ({placeholders})"
        args.extend(maybe_db_token(t) for t in track_tokens)
    base += " ORDER BY lp.timestamp ASC"
    return conn.execute(base, args).fetchall()
