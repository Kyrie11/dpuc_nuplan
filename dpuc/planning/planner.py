from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from dpuc.planning.bridge import frozen_support_value
from dpuc.planning.runtime import dbi_scores, interface_slot_predictions, load_runtime_models, structure_probabilities_from_interface, support_scores
from dpuc.planning.support import select_support


def _normalize_probs(raw_probs: np.ndarray) -> np.ndarray:
    if raw_probs.size == 0:
        return raw_probs
    total = float(raw_probs.sum())
    if total <= 0.0:
        return np.ones_like(raw_probs, dtype=np.float32) / max(1, raw_probs.size)
    return raw_probs / total


def _heuristic_agent_ranking(prefix_sample) -> List[Tuple[str, float]]:
    ego = prefix_sample.ego_history[-1]
    scores: Dict[str, float] = {}
    for action in prefix_sample.actions:
        witness_mass = float(sum(w.weight for w in action.witnesses))
        for slot in action.slots:
            dy = float(slot.features[1])
            dist = max(1.0, float(slot.features[2]))
            rel_v = abs(float(slot.features[5]))
            overlap = 1.0 if slot.slot_type in ('precedence', 'merge_gap', 'occupancy', 'release') else 0.35
            boundary = 1.0 + witness_mass
            scores[slot.owner_id] = scores.get(slot.owner_id, 0.0) + overlap * boundary * (1.5 / dist + 0.1 * rel_v + 0.02 * abs(dy))
    for track_id, hist in prefix_sample.agents_history.items():
        agent = hist[-1]
        dist = max(1.0, float(np.hypot(agent.x - ego.x, agent.y - ego.y)))
        scores[track_id] = scores.get(track_id, 0.0) + 1.0 / dist
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def _agent_ranking(prefix_sample, planner_cfg, runtime) -> List[Tuple[str, float]]:
    heur = dict(_heuristic_agent_ranking(prefix_sample))
    learned = dbi_scores(runtime, prefix_sample) if planner_cfg.dbi_exact else {}
    merged = {k: 0.35 * heur.get(k, 0.0) + 0.65 * learned.get(k, 0.0) for k in set(heur) | set(learned)}
    return sorted(merged.items(), key=lambda kv: kv[1], reverse=True)


def _select_agents(prefix_sample, planner_cfg, runtime, oracle_gain: Dict[str, float] | None = None) -> List[str]:
    budget = max(0, int(planner_cfg.agent_budget))
    if budget <= 0:
        return []
    variant = str(planner_cfg.individualization_variant).lower()
    ego = prefix_sample.ego_history[-1]
    if variant == 'allind':
        return list(prefix_sample.agents_history.keys())
    if variant == 'random-b':
        rng = np.random.default_rng(0)
        keys = sorted(prefix_sample.agents_history.keys())
        if len(keys) <= budget:
            return keys
        return [keys[i] for i in rng.choice(len(keys), size=budget, replace=False)]
    if variant == 'nearest-b':
        ranked = sorted(prefix_sample.agents_history.keys(), key=lambda k: np.hypot(prefix_sample.agents_history[k][-1].x - ego.x, prefix_sample.agents_history[k][-1].y - ego.y))
        return ranked[:budget]
    if variant == 'ttc-b':
        def ttc(k: str) -> float:
            a = prefix_sample.agents_history[k][-1]
            rel_x = a.x - ego.x
            rel_v = max(1e-3, ego.vx - a.vx)
            return abs(rel_x / rel_v)
        ranked = sorted(prefix_sample.agents_history.keys(), key=ttc)
        return ranked[:budget]
    if variant == 'entropy-b':
        ent = {}
        for action in prefix_sample.actions:
            for slot in action.slots:
                p = np.ones(len(slot.answer_vocab), dtype=np.float32) / max(1, len(slot.answer_vocab))
                ent[slot.owner_id] = ent.get(slot.owner_id, 0.0) + float(-(p * np.log(np.clip(p, 1e-8, 1.0))).sum())
        return [k for k, _ in sorted(ent.items(), key=lambda kv: kv[1], reverse=True)[:budget]]
    if variant == 'boundarysens-b':
        sens = {}
        for action in prefix_sample.actions:
            witness_mass = float(sum(w.weight for w in action.witnesses))
            for slot in action.slots:
                if slot.slot_type in ('precedence', 'merge_gap', 'release', 'occupancy'):
                    sens[slot.owner_id] = sens.get(slot.owner_id, 0.0) + witness_mass / max(1.0, float(slot.features[2]))
        return [k for k, _ in sorted(sens.items(), key=lambda kv: kv[1], reverse=True)[:budget]]
    if variant == 'resamplevoi' and oracle_gain:
        return [k for k, _ in sorted(oracle_gain.items(), key=lambda kv: kv[1], reverse=True)[:budget]]
    ranked = _agent_ranking(prefix_sample, planner_cfg, runtime)
    return [agent_id for agent_id, _ in ranked[:budget]]


def _apply_agent_refinement(action, selected: List[Dict], refined_agents: Sequence[str], planner_cfg) -> List[float]:
    refined = set(refined_agents)
    adjusted = []
    for cand in selected:
        slot_ids = cand.get('slot_ids', [])
        touched = sum(1 for slot_id in slot_ids if str(slot_id).split(':', 1)[0] in refined)
        local_closure = float(np.mean(cand.get('coverage', [0.0]) or [0.0])) if planner_cfg.use_local_closure_refresh else 0.0
        boost = 1.0 + 0.08 * touched + 0.04 * local_closure
        adjusted.append(float(cand.get('runtime_prob', cand.get('prob', 0.0))) * boost)
    return adjusted


def _oracle_agent_gain(prefix_sample) -> Dict[str, float]:
    gains: Dict[str, float] = {}
    for action in prefix_sample.actions:
        witness_mass = float(sum(w.weight for w in action.witnesses))
        for slot in action.slots:
            gain = witness_mass * (1.0 if slot.slot_type in ('precedence', 'merge_gap', 'release') else 0.5) / max(1.0, float(slot.features[2]))
            gains[slot.owner_id] = gains.get(slot.owner_id, 0.0) + gain
    return gains


def _evaluate_actions(prefix_sample, planner_cfg, runtime, retained_k: int, refined_agents: Sequence[str] | None = None) -> List[Dict]:
    action_outputs = []
    refined_agents = tuple(refined_agents or ())
    for action in prefix_sample.actions:
        slot_pred = interface_slot_predictions(runtime, action.action_features, action.slots, variant=planner_cfg.interface_variant)
        runtime_probs = structure_probabilities_from_interface(action.candidate_structures, slot_pred)
        for cand, p in zip(action.candidate_structures, runtime_probs):
            cand['runtime_prob'] = float(p)
        witness_weights = [w.weight for w in action.witnesses]
        learned_scores = support_scores(runtime, action.action_features, action.candidate_structures) if planner_cfg.use_uplift_term else None
        selected = select_support(
            action.candidate_structures,
            witness_weights,
            retained_k,
            variant=planner_cfg.support_variant,
            rescue=planner_cfg.use_rescue_support,
            learned_scores=learned_scores,
        )
        raw_probs = np.asarray([c.get('runtime_prob', c.get('prob', 0.0)) for c in selected], dtype=np.float32)
        retained_mass = float(raw_probs.sum())
        if refined_agents:
            raw_probs = np.asarray(_apply_agent_refinement(action, selected, refined_agents, planner_cfg), dtype=np.float32)
        probs = _normalize_probs(raw_probs)
        value, ess, max_norm_weight, value_parts = frozen_support_value(
            action.action_name,
            selected,
            probs.tolist(),
            planner_cfg.bridge_n,
            seed=hash((prefix_sample.sample_id, action.action_name, retained_k, refined_agents, getattr(planner_cfg, 'seed_offset', 0))) % (2**31 - 1),
            weight_clip=planner_cfg.weight_clip,
            variant=planner_cfg.bridge_variant,
            interface_variant=planner_cfg.interface_variant,
        )
        if refined_agents:
            affected_slots = sum(1 for slot in action.slots if slot.owner_id in refined_agents)
            value -= 0.02 * affected_slots
        uncertainty_margin = 0.05 * max_norm_weight + 0.02 * max(0.0, planner_cfg.ess_min - ess)
        boundary_slots = {slot.slot_id for slot in action.slots if slot.slot_type in ('precedence', 'merge_gap', 'release', 'occupancy')}
        selected_ids = {sid for cand in selected for sid in cand.get('slot_ids', [])}
        mass_full = np.asarray([c.get('runtime_prob', c.get('prob', 0.0)) for c in action.candidate_structures], dtype=np.float32)
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
                'slot_predictions': slot_pred,
                'full_structure_probs': mass_full,
                'value_parts': value_parts,
                'selected_boundary_ids': selected_ids & boundary_slots,
                'oracle_boundary_ids': boundary_slots,
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


def run_planner(prefix_sample, planner_cfg, runtime=None) -> Dict:
    runtime = runtime or load_runtime_models(type('CfgWrap', (), {'output_dir': 'outputs/default', 'device': 'cpu', 'model': type('M', (), {'hidden_dim':256,'num_heads':8,'num_layers':4,'residual_components':2})})())
    public_outputs = _evaluate_actions(prefix_sample, planner_cfg, runtime, planner_cfg.retained_k)
    flagged, diag = _diagnostics(public_outputs, planner_cfg)

    oracle_gain = _oracle_agent_gain(prefix_sample)
    refined_agents = _select_agents(prefix_sample, planner_cfg, runtime, oracle_gain=oracle_gain)
    mixed_outputs = _evaluate_actions(prefix_sample, planner_cfg, runtime, planner_cfg.retained_k, refined_agents=refined_agents) if refined_agents else public_outputs
    final_outputs = mixed_outputs
    fallback_used = False

    if flagged and planner_cfg.use_correction_fallback:
        fallback_used = True
        final_outputs = _evaluate_actions(prefix_sample, planner_cfg, runtime, planner_cfg.fallback_k)
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
        'agent_dbi_ranking': _agent_ranking(prefix_sample, planner_cfg, runtime),
        'oracle_agent_gain': oracle_gain,
    }
