
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Iterable, List
import numpy as np
from tqdm import tqdm

from dpuc.config import load_config
from dpuc.data.nuplan_sqlite import connect, fetch_boxes_for_lidar_token, fetch_ego_poses, fetch_lidar_pcs, fetch_log_meta, fetch_scenario_tags, normalize_token
from dpuc.data.schema import ActionSample, AgentState, EgoState, PrefixSample
from dpuc.data.features import build_slots, build_witnesses, infer_action_library
from dpuc.utils.io import save_pickle, ensure_dir


def _row_to_ego(row) -> EgoState:
    return EgoState(
        x=float(row['x']), y=float(row['y']), yaw=0.0, vx=float(row['vx']), vy=float(row['vy']),
        ax=float(row['acceleration_x']), ay=float(row['acceleration_y']), t=float(row['timestamp']) * 1e-6,
    )


def _rows_to_agents(rows, ts_us: int) -> List[AgentState]:
    agents = []
    for r in rows:
        token = normalize_token(r['track_token']) if 'track_token' in r.keys() else str(r['token'])
        yaw = float(r['yaw']) if 'yaw' in r.keys() else 0.0
        vx = float(r['vx']) if 'vx' in r.keys() else 0.0
        vy = float(r['vy']) if 'vy' in r.keys() else 0.0
        agents.append(AgentState(
            track_id=token, category=str(r['category'] or 'unknown'), x=float(r['x']), y=float(r['y']), yaw=yaw,
            vx=vx, vy=vy, length=float(r['length']), width=float(r['width']), t=ts_us * 1e-6,
        ))
    return agents


def _oracle_value(action_name: str, ego: EgoState, future: List[EgoState], agents: List[AgentState]) -> float:
    progress = 0.0
    if future:
        progress = np.hypot(future[-1].x - ego.x, future[-1].y - ego.y)
    nearest = min((np.hypot(a.x - ego.x, a.y - ego.y) for a in agents), default=50.0)
    brake_bias = 0.0
    if action_name == 'strong_brake':
        brake_bias = -1.5
    elif action_name == 'comfort_brake':
        brake_bias = -0.8
    elif action_name == 'mild_accel':
        brake_bias = 0.4
    elif action_name == 'stop_hold':
        brake_bias = -1.2
    interaction_penalty = max(0.0, 10.0 - nearest) * 0.2
    return -progress + interaction_penalty + brake_bias


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
    token_rows = {normalize_token(r['token']): r for r in lidar_pcs}

    prefixes: List[PrefixSample] = []
    for row in lidar_pcs:
        ts = int(row['timestamp'])
        if ts - timestamps[0] < history_us or timestamps[-1] - ts < horizon_us:
            continue
        if (ts - timestamps[0]) % interval_us != 0:
            continue
        ego_row = ego_map[normalize_token(row['ego_pose_token'])]
        ego = _row_to_ego(ego_row)
        cur_agents = _rows_to_agents(fetch_boxes_for_lidar_token(conn, normalize_token(row['token'])), ts)
        ego_history = [ego]
        future_ego = [ego]
        agents_history = {a.track_id: [a] for a in cur_agents}
        # Lightweight replay construction; uses nearby timestamps and nearest ego poses only.
        for other in lidar_pcs:
            ots = int(other['timestamp'])
            if ts - history_us <= ots < ts and (ts - ots) % interval_us == 0:
                ego_history.append(_row_to_ego(ego_map[normalize_token(other['ego_pose_token'])]))
            if ts < ots <= ts + horizon_us and (ots - ts) % interval_us == 0:
                future_ego.append(_row_to_ego(ego_map[normalize_token(other['ego_pose_token'])]))
        ego_history = sorted(ego_history, key=lambda x: x.t)
        future_ego = sorted(future_ego, key=lambda x: x.t)

        actions = []
        public_values = []
        action_specs = infer_action_library(ego)
        for action_idx, action_feats in action_specs:
            action_name = ['lane_follow','mild_accel','comfort_brake','strong_brake','creep','stop_hold','lane_change_left','lane_change_right','route_commit'][action_idx]
            public_values.append(_oracle_value(action_name, ego, future_ego, cur_agents) + (0.2 if action_idx > 5 else 0.0))
        for action_idx, action_feats in action_specs:
            action_name = ['lane_follow','mild_accel','comfort_brake','strong_brake','creep','stop_hold','lane_change_left','lane_change_right','route_commit'][action_idx]
            slots = build_slots(ego, cur_agents, action_idx, max_agents=cfg.data.max_agents)
            witnesses = build_witnesses(public_values, action_idx, top_k=cfg.model.max_witnesses, tau_g=cfg.planner.witness_gap_temp)
            oracle_value = _oracle_value(action_name, ego, future_ego, cur_agents)
            candidate_structures = []
            for slot in slots[: min(8, len(slots))]:
                candidate_structures.append({
                    'slot_id': slot.slot_id,
                    'structure_id': f"{action_idx}:{slot.slot_id}:{slot.label}",
                    'answers': {slot.slot_id: slot.label},
                    'prob': 1.0 / max(1, len(slots[:8])),
                    'coverage': [min(1.0, 1.0 / (1.0 + abs(w.gap) + j)) for j, w in enumerate(witnesses)],
                    'residual_mu': [0.0, 0.0],
                    'residual_logvar': [0.0, 0.0],
                })
            actions.append(ActionSample(
                action_index=action_idx,
                action_name=action_name,
                action_features=action_feats,
                slots=slots,
                witnesses=witnesses,
                candidate_structures=candidate_structures,
                oracle_value=oracle_value,
                public_value=public_values[action_idx],
            ))
        scenario_type = (tags.get(normalize_token(row['token']), ['unknown']) or ['unknown'])[0]
        prefixes.append(PrefixSample(
            sample_id=f"{db_path.stem}:{ts}", split=split, log_name=db_path.stem,
            location=str(meta.get('location', 'unknown')), scenario_type=scenario_type,
            timestamp_us=ts, ego_history=ego_history, agents_history=agents_history,
            future_ego=future_ego, actions=actions,
        ))
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
