
from __future__ import annotations
import math
from typing import Dict, Iterable, List, Tuple
from .schema import ACTION_NAMES, AgentState, EgoState, Slot, Witness


def angle_wrap(x: float) -> float:
    while x > math.pi:
        x -= 2 * math.pi
    while x < -math.pi:
        x += 2 * math.pi
    return x


def distance(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def infer_action_library(ego: EgoState) -> List[Tuple[int, List[float]]]:
    base_v = math.hypot(ego.vx, ego.vy)
    feats = []
    for idx, name in enumerate(ACTION_NAMES):
        accel = {
            'lane_follow': 0.0,
            'mild_accel': 1.0,
            'comfort_brake': -1.0,
            'strong_brake': -2.5,
            'creep': -0.5,
            'stop_hold': -3.5,
            'lane_change_left': 0.0,
            'lane_change_right': 0.0,
            'route_commit': 0.0,
        }[name]
        lateral = -1.0 if 'left' in name else (1.0 if 'right' in name else 0.0)
        feats.append((idx, [base_v, accel, lateral, ego.ax, ego.ay]))
    return feats


def build_slots(ego: EgoState, agents: List[AgentState], action_index: int, max_agents: int = 20) -> List[Slot]:
    slots: List[Slot] = []
    sorted_agents = sorted(agents, key=lambda a: distance(a.x, a.y, ego.x, ego.y))[:max_agents]
    for owner_index, agent in enumerate(sorted_agents):
        dx, dy = agent.x - ego.x, agent.y - ego.y
        d = math.hypot(dx, dy) + 1e-6
        rel_v = math.hypot(agent.vx - ego.vx, agent.vy - ego.vy)
        bearing = math.atan2(dy, dx) - ego.yaw
        heading_diff = angle_wrap(agent.yaw - ego.yaw)
        if d < 15:
            slot_type = 'precedence'
        elif abs(bearing) < 0.5 and d < 30:
            slot_type = 'occupancy'
        elif abs(bearing) < 1.2:
            slot_type = 'merge_gap'
        elif rel_v < 1.0:
            slot_type = 'release'
        else:
            slot_type = 'route_branch'
        features = [dx, dy, d, rel_v, bearing, heading_diff, agent.length, agent.width, float(action_index)]
        answer_vocab = {
            'precedence': ['ego_first', 'other_first', 'joint'],
            'merge_gap': ['open_ahead', 'open_behind', 'closed'],
            'release': ['release_before', 'release_after', 'joint_release'],
            'route_branch': ['keep', 'branch_left', 'branch_right'],
            'occupancy': ['free', 'occupied'],
        }[slot_type]
        label = 0 if d > 20 else min(1, len(answer_vocab)-1)
        slots.append(Slot(
            slot_id=f"{agent.track_id}:{slot_type}:{action_index}", owner_id=agent.track_id, slot_type=slot_type,
            anchor_x=(ego.x + agent.x) / 2, anchor_y=(ego.y + agent.y) / 2, owner_index=owner_index,
            answer_vocab=answer_vocab, features=features, label=label,
        ))
    return slots


def build_witnesses(public_values: List[float], action_index: int, top_k: int = 5, tau_g: float = 0.15) -> List[Witness]:
    gaps = []
    base = public_values[action_index]
    for idx, value in enumerate(public_values):
        if idx == action_index:
            continue
        gaps.append((idx, base - value))
    gaps = sorted(gaps, key=lambda x: abs(x[1]))[:top_k]
    denom = sum(math.exp(-abs(g)/tau_g) for _, g in gaps) + 1e-6
    out = []
    for rival_idx, gap in gaps:
        w = math.exp(-abs(gap)/tau_g) / denom
        out.append(Witness(
            witness_id=f"{action_index}->{rival_idx}", rival_action_index=rival_idx, gap=gap, weight=w,
            features=[float(action_index), float(rival_idx), gap, abs(gap), w],
        ))
    return out
