from __future__ import annotations

import argparse
import bisect
import math
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from tqdm import tqdm

from dpuc.config import load_config
from dpuc.data.nuplan_sqlite import (
    connect,
    fetch_boxes_for_lidar_token,
    fetch_boxes_in_time_window,
    fetch_ego_poses,
    fetch_lidar_pcs,
    fetch_log_meta,
    fetch_scenario_tags,
    normalize_token,
)
from dpuc.data.schema import ActionSample, AgentState, EgoState, PrefixSample
from dpuc.data.features import build_candidate_bank, build_slots, build_witnesses, infer_action_library
from dpuc.utils.io import ensure_dir, save_pickle


def _quat_to_yaw(row) -> float:
    if all(k in row.keys() for k in ('qw', 'qx', 'qy', 'qz')):
        qw, qx, qy, qz = float(row['qw']), float(row['qx']), float(row['qy']), float(row['qz'])
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return float(math.atan2(siny_cosp, cosy_cosp))
    return 0.0


def _get_row_value(row, key: str, default=0.0):
    return row[key] if key in row.keys() and row[key] is not None else default


def _row_to_ego(row) -> EgoState:
    return EgoState(
        x=float(_get_row_value(row, 'x', 0.0)),
        y=float(_get_row_value(row, 'y', 0.0)),
        yaw=_quat_to_yaw(row),
        vx=float(_get_row_value(row, 'vx', 0.0)),
        vy=float(_get_row_value(row, 'vy', 0.0)),
        ax=float(_get_row_value(row, 'acceleration_x', 0.0)),
        ay=float(_get_row_value(row, 'acceleration_y', 0.0)),
        t=float(_get_row_value(row, 'timestamp', 0)) * 1e-6,
    )


def _rows_to_agents(rows, ts_us: int) -> List[AgentState]:
    agents: List[AgentState] = []
    for r in rows:
        token = normalize_token(r['track_token']) if 'track_token' in r.keys() else normalize_token(r['token'])
        yaw = float(r['yaw']) if 'yaw' in r.keys() else 0.0
        vx = float(r['vx']) if 'vx' in r.keys() else 0.0
        vy = float(r['vy']) if 'vy' in r.keys() else 0.0
        length = float(_get_row_value(r, 'length', _get_row_value(r, 'track_length', 0.0)) or 0.0)
        width = float(_get_row_value(r, 'width', _get_row_value(r, 'track_width', 0.0)) or 0.0)
        agents.append(
            AgentState(
                track_id=token,
                category=str(r['category'] or 'unknown'),
                x=float(r['x']),
                y=float(r['y']),
                yaw=yaw,
                vx=vx,
                vy=vy,
                length=max(length, 0.1),
                width=max(width, 0.1),
                t=ts_us * 1e-6,
            )
        )
    return agents


def _oracle_value(action_name: str, ego: EgoState, future: List[EgoState], agents: List[AgentState]) -> float:
    progress = 0.0
    if future:
        progress = np.hypot(future[-1].x - ego.x, future[-1].y - ego.y)
    nearest = min((np.hypot(a.x - ego.x, a.y - ego.y) for a in agents), default=50.0)
    brake_bias = {
        'lane_follow': 0.0,
        'mild_accel': 0.4,
        'comfort_brake': -0.8,
        'strong_brake': -1.5,
        'creep': -0.4,
        'stop_hold': -1.2,
        'lane_change_left': 0.1,
        'lane_change_right': 0.1,
        'route_commit': 0.0,
    }[action_name]
    interaction_penalty = max(0.0, 10.0 - nearest) * 0.2
    return -progress + interaction_penalty + brake_bias


def _slice_rows_by_time(lidar_pcs, timestamps: List[int], start_ts: int, end_ts: int):
    left = bisect.bisect_left(timestamps, start_ts)
    right = bisect.bisect_right(timestamps, end_ts)
    return lidar_pcs[left:right]


def build_prefixes_for_db(db_path: Path, split: str, cfg) -> List[PrefixSample]:
    conn = connect(db_path)
    meta = fetch_log_meta(conn)
    lidar_pcs = fetch_lidar_pcs(conn)
    ego_map = fetch_ego_poses(conn)
    tags = fetch_scenario_tags(conn)

    interval_us = int(cfg.data.sample_interval_sec * 1e6)
    history_us = int(cfg.data.history_sec * 1e6)
    horizon_us = int(cfg.data.horizon_sec * 1e6)

    timestamps = [int(r['timestamp']) for r in lidar_pcs]
    if not timestamps:
        conn.close()
        return []

    prefixes: List[PrefixSample] = []
    for row in lidar_pcs:
        ts = int(row['timestamp'])
        if ts - timestamps[0] < history_us or timestamps[-1] - ts < horizon_us:
            continue
        if (ts - timestamps[0]) % interval_us != 0:
            continue

        ego = _row_to_ego(ego_map[normalize_token(row['ego_pose_token'])])
        cur_agents = _rows_to_agents(fetch_boxes_for_lidar_token(conn, normalize_token(row['token'])), ts)

        hist_rows = _slice_rows_by_time(lidar_pcs, timestamps, ts - history_us, ts - interval_us)
        fut_rows = _slice_rows_by_time(lidar_pcs, timestamps, ts + interval_us, ts + horizon_us)
        ego_history = [_row_to_ego(ego_map[normalize_token(r['ego_pose_token'])]) for r in hist_rows if (ts - int(r['timestamp'])) % interval_us == 0]
        ego_history.append(ego)
        future_ego = [ego] + [_row_to_ego(ego_map[normalize_token(r['ego_pose_token'])]) for r in fut_rows if (int(r['timestamp']) - ts) % interval_us == 0]

        selected_track_tokens = [a.track_id for a in cur_agents]
        history_boxes = fetch_boxes_in_time_window(conn, ts - history_us, ts, selected_track_tokens)
        agents_history = {}
        for hist_box in history_boxes:
            hist_ts = int(_get_row_value(hist_box, 'lidar_timestamp', ts))
            if hist_ts > ts or ((ts - hist_ts) % interval_us != 0):
                continue
            hist_agents = _rows_to_agents([hist_box], hist_ts)
            if not hist_agents:
                continue
            hist_agent = hist_agents[0]
            agents_history.setdefault(hist_agent.track_id, []).append(hist_agent)
        for track_id in list(agents_history.keys()):
            agents_history[track_id] = sorted(agents_history[track_id], key=lambda x: x.t)
        for a in cur_agents:
            agents_history.setdefault(a.track_id, []).append(a)
            agents_history[a.track_id] = sorted(agents_history[a.track_id], key=lambda x: x.t)
        action_specs = infer_action_library(ego)
        public_values: List[float] = []
        for action_idx, _ in action_specs:
            action_name = ['lane_follow','mild_accel','comfort_brake','strong_brake','creep','stop_hold','lane_change_left','lane_change_right','route_commit'][action_idx]
            public_values.append(_oracle_value(action_name, ego, future_ego, cur_agents) + (0.2 if action_idx > 5 else 0.0))

        actions: List[ActionSample] = []
        for action_idx, action_feats in action_specs:
            action_name = ['lane_follow','mild_accel','comfort_brake','strong_brake','creep','stop_hold','lane_change_left','lane_change_right','route_commit'][action_idx]
            slots = build_slots(ego, cur_agents, action_idx, max_agents=cfg.data.max_agents)
            witnesses = build_witnesses(public_values, action_idx, top_k=cfg.model.max_witnesses, tau_g=cfg.planner.witness_gap_temp)
            candidate_structures = build_candidate_bank(
                ego,
                action_index=action_idx,
                slots=slots,
                witnesses=witnesses,
                bank_cap=cfg.planner.bank_cap,
                beam_width=cfg.planner.beam_width,
                residual_components=cfg.model.residual_components,
            )
            actions.append(
                ActionSample(
                    action_index=action_idx,
                    action_name=action_name,
                    action_features=action_feats,
                    slots=slots,
                    witnesses=witnesses,
                    candidate_structures=candidate_structures,
                    oracle_value=_oracle_value(action_name, ego, future_ego, cur_agents),
                    public_value=public_values[action_idx],
                )
            )

        scenario_type = (tags.get(normalize_token(row['token']), ['unknown']) or ['unknown'])[0]
        prefixes.append(
            PrefixSample(
                sample_id=f"{db_path.stem}:{ts}",
                split=split,
                log_name=db_path.stem,
                location=str(meta.get('location', 'unknown')),
                scenario_type=scenario_type,
                timestamp_us=ts,
                ego_history=sorted(ego_history, key=lambda x: x.t),
                agents_history=agents_history,
                future_ego=sorted(future_ego, key=lambda x: x.t),
                actions=actions,
            )
        )
    conn.close()
    return prefixes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--limit-db', type=int, default=0)
    args = parser.parse_args()
    cfg = load_config(args.config)
    raw_root = Path(cfg.data.raw_root)
    out_root = ensure_dir(Path(cfg.data.processed_dir))
    if not raw_root.exists():
        raise FileNotFoundError(f'Raw nuPlan root not found: {raw_root}')
    for split_name, subdirs in [('train', cfg.data.train_dirs), ('val', cfg.data.val_dirs)]:
        out_split = ensure_dir(out_root / split_name)
        db_files = []
        for subdir in subdirs:
            db_files.extend(sorted((raw_root / subdir).glob('*.db')))
        if args.limit_db > 0:
            db_files = db_files[: args.limit_db]
        manifest = []
        for db_path in tqdm(db_files, desc=f'preprocess-{split_name}'):
            prefixes = build_prefixes_for_db(db_path, split_name, cfg)
            dst = out_split / f'{db_path.stem}.pkl'
            save_pickle(prefixes, dst)
            manifest.append({'db': str(db_path), 'samples': len(prefixes), 'output': str(dst)})
        save_pickle(manifest, out_root / f'{split_name}_manifest.pkl')


if __name__ == '__main__':
    main()
