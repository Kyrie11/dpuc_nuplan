from __future__ import annotations

from typing import Dict, List

import numpy as np


def greedy_select(candidate_structures: List[Dict], witness_weights: List[float], k: int, rescue: bool = True) -> List[Dict]:
    selected: List[Dict] = []
    covered = np.zeros(len(witness_weights), dtype=np.float32)
    remaining = list(candidate_structures)
    rescue_used = False

    while remaining and len(selected) < k:
        best_idx = 0
        best_score = -1e18
        for i, cand in enumerate(remaining):
            cov = np.asarray(cand.get('coverage', [0.0] * len(witness_weights)), dtype=np.float32)
            marginal = float(np.sum(np.asarray(witness_weights, dtype=np.float32) * np.maximum(cov - covered, 0.0)))
            diversity = 0.0
            if selected:
                cur_slots = set(cand.get('slot_ids', []))
                diversity = min(
                    len(cur_slots.symmetric_difference(set(prev.get('slot_ids', []))))
                    for prev in selected
                )
            score = marginal + 1e-3 * diversity + 1e-4 * float(cand.get('prob', 0.0))
            if score > best_score:
                best_idx = i
                best_score = score

        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        covered = np.maximum(covered, np.asarray(chosen.get('coverage', covered), dtype=np.float32))

        if rescue and (not rescue_used) and len(selected) < k and len(remaining) > 0:
            uncovered_idx = int(np.argmin(covered)) if len(covered) else -1
            if uncovered_idx >= 0 and covered[uncovered_idx] < 1e-4:
                rescue_idx = max(
                    range(len(remaining)),
                    key=lambda i: float(np.asarray(remaining[i].get('coverage', [0.0] * len(witness_weights)))[uncovered_idx]),
                )
                rescue_cand = remaining.pop(rescue_idx)
                selected.append(rescue_cand)
                covered = np.maximum(covered, np.asarray(rescue_cand.get('coverage', covered), dtype=np.float32))
                rescue_used = True

    return selected[:k]
