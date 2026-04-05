
from __future__ import annotations
from typing import Dict, List
import numpy as np


def greedy_select(candidate_structures: List[Dict], witness_weights: List[float], k: int, rescue: bool = True) -> List[Dict]:
    selected: List[Dict] = []
    covered = np.zeros(len(witness_weights), dtype=np.float32)
    remaining = list(candidate_structures)
    for _ in range(min(k, len(remaining))):
        best_idx, best_gain = 0, -1.0
        for i, cand in enumerate(remaining):
            cov = np.asarray(cand.get('coverage', [0.0] * len(witness_weights)), dtype=np.float32)
            gain = float(np.sum(np.asarray(witness_weights) * np.maximum(cov - covered, 0.0)))
            if gain > best_gain:
                best_idx, best_gain = i, gain
        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        covered = np.maximum(covered, np.asarray(chosen.get('coverage', covered), dtype=np.float32))
    if rescue and len(selected) < k and remaining:
        uncovered = int(np.argmin(covered)) if len(covered) > 0 else 0
        rescue_idx = max(range(len(remaining)), key=lambda i: remaining[i].get('coverage', [0.0])[uncovered] if remaining[i].get('coverage') else 0.0)
        selected.append(remaining[rescue_idx])
    return selected[:k]
