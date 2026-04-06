from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .schema import ACTION_NAMES, AgentState, EgoState, Slot, Witness


SLOT_TYPE_TO_ID = {
    'precedence': 0,
    'merge_gap': 1,
    'release': 2,
    'route_branch': 3,
    'occupancy': 4,
}


ACTION_TO_ACCEL = {
    'lane_follow': 0.0,
    'mild_accel': 1.0,
    'comfort_brake': -1.0,
    'strong_brake': -2.5,
    'creep': -0.5,
    'stop_hold': -3.5,
    'lane_change_left': 0.0,
    'lane_change_right': 0.0,
    'route_commit': 0.0,
}


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
    feats: List[Tuple[int, List[float]]] = []
    for idx, name in enumerate(ACTION_NAMES):
        lateral = -1.0 if 'left' in name else (1.0 if 'right' in name else 0.0)
        feats.append((idx, [base_v, ACTION_TO_ACCEL[name], lateral, ego.ax, ego.ay]))
    return feats


def _answer_vocab_for(slot_type: str) -> List[str]:
    return {
        'precedence': ['ego_first', 'other_first', 'joint'],
        'merge_gap': ['open_ahead', 'open_behind', 'closed'],
        'release': ['release_before', 'release_after', 'joint_release'],
        'route_branch': ['keep', 'branch_left', 'branch_right'],
        'occupancy': ['free', 'occupied'],
    }[slot_type]


def build_slots(ego: EgoState, agents: List[AgentState], action_index: int, max_agents: int = 20) -> List[Slot]:
    action_name = ACTION_NAMES[action_index]
    slots: List[Slot] = []
    sorted_agents = sorted(agents, key=lambda a: distance(a.x, a.y, ego.x, ego.y))[:max_agents]
    for owner_index, agent in enumerate(sorted_agents):
        dx, dy = agent.x - ego.x, agent.y - ego.y
        d = math.hypot(dx, dy) + 1e-6
        rel_vx, rel_vy = agent.vx - ego.vx, agent.vy - ego.vy
        rel_v = math.hypot(rel_vx, rel_vy)
        bearing = angle_wrap(math.atan2(dy, dx) - ego.yaw)
        heading_diff = angle_wrap(agent.yaw - ego.yaw)

        if d < 15.0:
            slot_type = 'precedence'
        elif 'lane_change' in action_name and abs(bearing) < 0.9 and d < 35.0:
            slot_type = 'merge_gap'
        elif abs(bearing) < 0.45 and d < 30.0:
            slot_type = 'occupancy'
        elif rel_v < 1.0 and d < 25.0:
            slot_type = 'release'
        else:
            slot_type = 'route_branch'

        answer_vocab = _answer_vocab_for(slot_type)
        if slot_type == 'occupancy':
            label = 1 if d < 12.0 else 0
        elif slot_type == 'precedence':
            label = 1 if d < 10.0 else 0
        elif slot_type == 'merge_gap':
            label = 2 if d < 8.0 else (0 if dy > 0 else 1)
        elif slot_type == 'release':
            label = 0 if rel_v > 0.5 else 2
        else:
            label = 1 if dy > 2.0 else (2 if dy < -2.0 else 0)

        features = [
            dx,
            dy,
            d,
            rel_vx,
            rel_vy,
            rel_v,
            bearing,
            heading_diff,
            agent.length,
            agent.width,
            float(action_index),
            float(SLOT_TYPE_TO_ID[slot_type]),
        ]
        slots.append(
            Slot(
                slot_id=f"{agent.track_id}:{slot_type}:{action_index}",
                owner_id=agent.track_id,
                slot_type=slot_type,
                anchor_x=(ego.x + agent.x) / 2.0,
                anchor_y=(ego.y + agent.y) / 2.0,
                owner_index=owner_index,
                answer_vocab=answer_vocab,
                features=features,
                label=label,
            )
        )
    return slots


def build_witnesses(public_values: List[float], action_index: int, top_k: int = 5, tau_g: float = 0.15) -> List[Witness]:
    base = float(public_values[action_index])
    gaps = [(idx, base - float(value)) for idx, value in enumerate(public_values) if idx != action_index]
    gaps = sorted(gaps, key=lambda x: abs(x[1]))[:top_k]
    denom = sum(math.exp(-abs(gap) / max(tau_g, 1e-6)) for _, gap in gaps) + 1e-6
    out: List[Witness] = []
    for rival_idx, gap in gaps:
        weight = math.exp(-abs(gap) / max(tau_g, 1e-6)) / denom
        out.append(
            Witness(
                witness_id=f"{action_index}->{rival_idx}",
                rival_action_index=rival_idx,
                gap=gap,
                weight=weight,
                features=[float(action_index), float(rival_idx), gap, abs(gap), weight],
            )
        )
    return out


def _slot_answer_probs(slot: Slot) -> np.ndarray:
    d = max(0.1, float(slot.features[2]))
    if slot.slot_type == 'precedence':
        logits = np.array([2.5 / d, 3.0 / max(0.2, 20.0 - d), 0.2], dtype=np.float32)
    elif slot.slot_type == 'merge_gap':
        logits = np.array([slot.features[1] / max(d, 1.0), -slot.features[1] / max(d, 1.0), 1.5 / d], dtype=np.float32)
    elif slot.slot_type == 'release':
        logits = np.array([0.4 + slot.features[5], 0.4, 0.2], dtype=np.float32)
    elif slot.slot_type == 'route_branch':
        logits = np.array([0.6, max(0.0, slot.features[1]), max(0.0, -slot.features[1])], dtype=np.float32)
    else:
        logits = np.array([1.5 if d > 15.0 else 0.2, 1.2 if d <= 15.0 else 0.2], dtype=np.float32)
    logits = logits[: len(slot.answer_vocab)]
    logits = logits - logits.max()
    probs = np.exp(logits)
    return probs / probs.sum()


def build_candidate_bank(
    ego: EgoState,
    action_index: int,
    slots: List[Slot],
    witnesses: List[Witness],
    bank_cap: int = 64,
    beam_width: int = 32,
    residual_components: int = 2,
) -> List[Dict]:
    """Approximate Appendix A bank builder with beam search over joint slot assignments."""
    if not slots:
        return [
            {
                'structure_id': f'{action_index}:empty',
                'answers': {},
                'slot_ids': [],
                'prob': 1.0,
                'coverage': [0.0 for _ in witnesses],
                'residual_mu': [0.0, 0.0],
                'residual_logvar': [-0.5, -0.5],
            }
        ]

    slot_probs = {slot.slot_id: _slot_answer_probs(slot) for slot in slots}
    ordered_slots = sorted(slots, key=lambda s: float(slot_probs[s.slot_id].max()), reverse=True)

    beams: List[Tuple[float, Dict[str, int]]] = [(0.0, {})]
    for slot in ordered_slots[: min(len(ordered_slots), 12)]:
        probs = slot_probs[slot.slot_id]
        expansions: List[Tuple[float, Dict[str, int]]] = []
        top_answers = np.argsort(-probs)[: min(2, len(probs))]
        for score, partial in beams:
            for answer_idx in top_answers:
                next_partial = dict(partial)
                next_partial[slot.slot_id] = int(answer_idx)
                compat_penalty = 0.0
                if slot.slot_type == 'precedence' and answer_idx == 2:
                    compat_penalty += 0.35
                if slot.slot_type == 'occupancy' and answer_idx == 1 and slot.features[2] > 20.0:
                    compat_penalty += 0.5
                expansions.append((score + math.log(float(probs[answer_idx]) + 1e-9) - compat_penalty, next_partial))
        expansions.sort(key=lambda x: x[0], reverse=True)
        beams = expansions[:beam_width]

    structures: List[Dict] = []
    for rank, (score, answers) in enumerate(beams[:bank_cap]):
        probs = []
        precedence_ct = 0
        occupancy_ct = 0
        merge_ct = 0
        for slot in ordered_slots[: min(len(ordered_slots), 12)]:
            answer_idx = answers.get(slot.slot_id, 0)
            probs.append(float(slot_probs[slot.slot_id][answer_idx]))
            if slot.slot_type == 'precedence' and answer_idx == 1:
                precedence_ct += 1
            if slot.slot_type == 'occupancy' and answer_idx == 1:
                occupancy_ct += 1
            if slot.slot_type == 'merge_gap' and answer_idx == 2:
                merge_ct += 1
        struct_prob = float(np.exp(np.mean(np.log(np.clip(np.asarray(probs), 1e-9, 1.0)))))
        coverage: List[float] = []
        for j, witness in enumerate(witnesses):
            witness_score = 0.35 / (1.0 + abs(witness.gap))
            witness_score += 0.25 * precedence_ct + 0.15 * merge_ct + 0.1 * occupancy_ct
            witness_score += 0.03 * (j + 1)
            coverage.append(float(max(0.0, min(1.0, witness_score))))
        residual_mu = [
            0.25 * precedence_ct + 0.15 * merge_ct,
            0.2 * occupancy_ct + 0.05 * rank,
        ]
        residual_logvar = [
            math.log(0.25 + 0.05 * (1 + precedence_ct)),
            math.log(0.25 + 0.05 * (1 + occupancy_ct + merge_ct)),
        ]
        structures.append(
            {
                'structure_id': f"{action_index}:{rank}:{'|'.join(f'{k}={v}' for k, v in sorted(answers.items()))}",
                'answers': answers,
                'slot_ids': sorted(answers.keys()),
                'prob': struct_prob,
                'coverage': coverage,
                'residual_mu': residual_mu,
                'residual_logvar': residual_logvar,
            }
        )

    prob_sum = sum(s['prob'] for s in structures) + 1e-9
    for s in structures:
        s['prob'] = float(s['prob'] / prob_sum)
    return structures
