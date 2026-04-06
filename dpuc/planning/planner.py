from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from dpuc.planning.bridge import frozen_support_value
from dpuc.planning.support import greedy_select


def _normalize_probs(raw_probs: np.ndarray) -> np.ndarray:
    if raw_probs.size == 0:
        return raw_probs
    total = float(raw_probs.sum())
    if total <= 0.0:
        return np.ones_like(raw_probs, dtype=np.float32) / max(1, raw_probs.size)
    return raw_probs / total


def _agent_dbi_scores(prefix_sample) -> List[Tuple[str, float]]:
    ego = prefix_sample.ego_history[-1]
    scores: Dict[str, float] = {}
    for action in prefix_sample.actions:
        witness_mass = float(sum(w.weight for w in action.witnesses))
        for slot in action.slots:
            dx = float(slot.features[0])
            dy = float(slot.features[1])
            dist = max(1.0, float(slot.features[2]))
            rel_v = abs(float(slot.features[5]))
            overlap = 1.0 if slot.slot_type in ('precedence', 'merge_gap', 'occupancy', 'release') else 0.35
            boundary = 1.0 + witness_mass
            scores[slot.owner_id] = scores.get(slot.owner_id, 0.0) + overlap * boundary * (1.5 / dist + 0.1 * rel_v + 0.02 * abs(dy))
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def _apply_agent_refinement(action, selected: List[Dict], refined_agents: Sequence[str]) -> List[float]:
    refined = set(refined_agents)
    adjusted = []
    for cand in selected:
        slot_ids = cand.get('slot_ids', [])
        touched = sum(1 for slot_id in slot_ids if str(slot_id).split(':', 1)[0] in refined)
        boost = 1.0 + 0.08 * touched + 0.04 * float(np.mean(cand.get('coverage', [0.0]) or [0.0]))
        adjusted.append(float(cand.get('prob', 0.0)) * boost)
    return adjusted


def _evaluate_actions(prefix_sample, planner_cfg, retained_k: int, refined_agents: Sequence[str] | None = None) -> List[Dict]:
    action_outputs = []
    refined_agents = tuple(refined_agents or ())
    for action in prefix_sample.actions:
        witness_weights = [w.weight for w in action.witnesses]
        selected = greedy_select(action.candidate_structures, witness_weights, retained_k)
        raw_probs = np.asarray([c.get('prob', 0.0) for c in selected], dtype=np.float32)
        retained_mass = float(raw_probs.sum())
        if refined_agents:
            raw_probs = np.asarray(_apply_agent_refinement(action, selected, refined_agents), dtype=np.float32)
        probs = _normalize_probs(raw_probs)
        value, ess, max_norm_weight = frozen_support_value(
            action.action_name,
            selected,
            probs.tolist(),
            planner_cfg.bridge_n,
            seed=hash((prefix_sample.sample_id, action.action_name, retained_k)) % (2**31 - 1),
            weight_clip=planner_cfg.weight_clip,
        )
        if refined_agents:
            affected_slots = sum(1 for slot in action.slots if slot.owner_id in refined_agents)
            value -= 0.02 * affected_slots
        uncertainty_margin = 0.05 * max_norm_weight + 0.02 * max(0.0, planner_cfg.ess_min - ess)
        action_outputs.append(
            {
                'action_index': action.action_index,
                'action_name': action.action_name,
                'value': value,
                'oracle_value': action.oracle_value,
                'public_value': action.public_value,
                'ess': ess,
                'max_norm_weight': max_norm_weight,
                'retained_mass': retained_mass,
                'selected_structures': selected,
                'refined_agents': list(refined_agents),
                'uncertainty_margin': uncertainty_margin,
            }
        )
    return action_outputs


def _diagnostics(action_outputs: List[Dict], planner_cfg) -> Tuple[bool, Dict]:
    sorted_actions = sorted(action_outputs, key=lambda x: x['value'])
    top_two = sorted_actions[:2]
    top_gap = float('inf')
    if len(top_two) == 2:
        top_gap = abs(top_two[1]['value'] - top_two[0]['value'])
    flagged = False
    reasons = []
    for action in top_two:
        if action['retained_mass'] < planner_cfg.retained_mass_min:
            flagged = True
            reasons.append(f"low_retained_mass:{action['action_name']}")
        if action['ess'] < planner_cfg.ess_min:
            flagged = True
            reasons.append(f"low_ess:{action['action_name']}")
        if action['max_norm_weight'] > planner_cfg.max_norm_weight:
            flagged = True
            reasons.append(f"high_weight:{action['action_name']}")
    if np.isfinite(top_gap) and top_gap < planner_cfg.boundary_gap:
        flagged = True
        reasons.append('tiny_top_gap')
    if len(top_two) == 2:
        conf_gap = top_gap
        unc_gap = top_two[0].get('uncertainty_margin', 0.0) + top_two[1].get('uncertainty_margin', 0.0)
        if conf_gap < unc_gap:
            flagged = True
            reasons.append('gap_below_uncertainty_interval')
    return flagged, {'top_gap': top_gap, 'reasons': reasons}


def run_planner(prefix_sample, planner_cfg) -> Dict:
    public_outputs = _evaluate_actions(prefix_sample, planner_cfg, planner_cfg.retained_k)
    flagged, diag = _diagnostics(public_outputs, planner_cfg)

    ranked_agents = _agent_dbi_scores(prefix_sample)
    refined_agents = [agent_id for agent_id, _ in ranked_agents[: max(0, int(planner_cfg.agent_budget))]]
    mixed_outputs = _evaluate_actions(prefix_sample, planner_cfg, planner_cfg.retained_k, refined_agents=refined_agents) if refined_agents else public_outputs
    final_outputs = mixed_outputs
    fallback_used = False

    if flagged:
        fallback_used = True
        final_outputs = _evaluate_actions(prefix_sample, planner_cfg, planner_cfg.fallback_k)
        for item in final_outputs:
            item['value'] += item.get('uncertainty_margin', 0.0)

    best = min(final_outputs, key=lambda x: x['value'])
    oracle_best = min(final_outputs, key=lambda x: x['oracle_value'])
    return {
        'public_actions': public_outputs,
        'actions': final_outputs,
        'best': best,
        'oracle_best': oracle_best,
        'flagged': flagged,
        'fallback_used': fallback_used,
        'diagnostics': diag,
        'refined_agents': refined_agents,
        'agent_dbi_ranking': ranked_agents,
    }
