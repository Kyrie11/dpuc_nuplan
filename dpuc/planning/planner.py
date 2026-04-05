
from __future__ import annotations
from typing import Dict, List
import numpy as np
from dpuc.planning.support import greedy_select
from dpuc.planning.bridge import frozen_support_value


def run_planner(prefix_sample, planner_cfg) -> Dict:
    action_outputs = []
    for action in prefix_sample.actions:
        witness_weights = [w.weight for w in action.witnesses]
        selected = greedy_select(action.candidate_structures, witness_weights, planner_cfg.retained_k)
        probs = np.asarray([c.get('prob', 0.0) for c in selected], dtype=np.float32)
        if probs.sum() <= 0:
            probs = np.ones(len(selected), dtype=np.float32) / max(1, len(selected))
        else:
            probs = probs / probs.sum()
        value, ess = frozen_support_value(action.action_name, selected, probs.tolist(), planner_cfg.bridge_n)
        action_outputs.append({
            'action_index': action.action_index,
            'action_name': action.action_name,
            'value': value,
            'oracle_value': action.oracle_value,
            'public_value': action.public_value,
            'ess': ess,
            'retained_mass': float(probs.sum()),
        })
    best = min(action_outputs, key=lambda x: x['value'])
    oracle_best = min(action_outputs, key=lambda x: x['oracle_value'])
    return {'actions': action_outputs, 'best': best, 'oracle_best': oracle_best}
