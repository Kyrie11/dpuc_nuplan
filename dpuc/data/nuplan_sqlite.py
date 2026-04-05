
from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def list_tables(conn: sqlite3.Connection) -> List[str]:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return [r[0] for r in cur.fetchall()]


def fetch_log_meta(conn: sqlite3.Connection) -> Dict:
    row = conn.execute("SELECT * FROM log LIMIT 1").fetchone()
    return dict(row) if row else {}


def fetch_lidar_pcs(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    query = """
    SELECT lp.token, lp.timestamp, lp.scene_token, lp.ego_pose_token, s.name as scene_name
    FROM lidar_pc lp
    LEFT JOIN scene s ON lp.scene_token = s.token
    ORDER BY lp.timestamp ASC
    """
    return conn.execute(query).fetchall()


def fetch_ego_poses(conn: sqlite3.Connection) -> Dict[str, sqlite3.Row]:
    rows = conn.execute("SELECT * FROM ego_pose").fetchall()
    return {bytes(row['token']).hex() if isinstance(row['token'], (bytes, bytearray)) else row['token']: row for row in rows}


def fetch_boxes_for_lidar_token(conn: sqlite3.Connection, lidar_pc_token) -> List[sqlite3.Row]:
    q = """
    SELECT lb.*, t.category_name as category
    FROM lidar_box lb
    LEFT JOIN track t ON lb.track_token = t.token
    WHERE lb.lidar_pc_token = ?
    """
    param = bytes.fromhex(lidar_pc_token) if isinstance(lidar_pc_token, str) and len(lidar_pc_token) > 20 else lidar_pc_token
    return conn.execute(q, [param]).fetchall()


def fetch_scenario_tags(conn: sqlite3.Connection) -> Dict[str, List[str]]:
    rows = conn.execute("SELECT lidar_pc_token, type FROM scenario_tag").fetchall()
    tags: Dict[str, List[str]] = {}
    for row in rows:
        key = bytes(row['lidar_pc_token']).hex() if isinstance(row['lidar_pc_token'], (bytes, bytearray)) else row['lidar_pc_token']
        tags.setdefault(key, []).append(row['type'])
    return tags


def normalize_token(token) -> str:
    if isinstance(token, (bytes, bytearray)):
        return bytes(token).hex()
    return str(token)
