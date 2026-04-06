from __future__ import annotations

from typing import Dict, List

import numpy as np


def _coverage_array(cand: Dict, witness_count: int) -> np.ndarray:
    return np.asarray(cand.get('coverage', [0.0] * witness_count), dtype=np.float32)


def select_support(candidate_structures: List[Dict], witness_weights: List[float], k: int, variant: str = 'ours', rescue: bool = True, learned_scores: np.ndarray | None = None) -> List[Dict]:
    if k <= 0:
        return []
    if not candidate_structures:
        return []
    witness_weights = np.asarray(witness_weights, dtype=np.float32)
    k = min(k, len(candidate_structures))

    if variant == 'masstopk':
        ordered = sorted(candidate_structures, key=lambda c: float(c.get('prob', 0.0)), reverse=True)
        return ordered[:k]
    if variant == 'structtopk':
        ordered = sorted(candidate_structures, key=lambda c: len(c.get('slot_ids', [])), reverse=True)
        return ordered[:k]
    if variant == 'random':
        rng = np.random.default_rng(0)
        idx = rng.choice(len(candidate_structures), size=k, replace=False)
        return [candidate_structures[i] for i in idx]
    if variant == 'diversetopk':
        selected: List[Dict] = []
        remaining = list(candidate_structures)
        while remaining and len(selected) < k:
            if not selected:
                best = max(remaining, key=lambda c: len(c.get('slot_ids', [])) + 0.1 * float(c.get('prob', 0.0)))
                remaining.remove(best)
                selected.append(best)
                continue
            def diversity(c):
                cur = set(c.get('slot_ids', []))
                return min(len(cur.symmetric_difference(set(s.get('slot_ids', [])))) for s in selected)
            best = max(remaining, key=diversity)
            remaining.remove(best)
            selected.append(best)
        return selected
    if variant == 'uncunion':
        selected = []
        oracle_idx = np.argsort(-witness_weights)[: min(len(witness_weights), k)]
        remaining = list(candidate_structures)
        for wid in oracle_idx:
            if not remaining or len(selected) >= k:
                break
            best = max(remaining, key=lambda c: float(_coverage_array(c, len(witness_weights))[wid]))
            remaining.remove(best)
            selected.append(best)
        if len(selected) < k:
            leftover = [c for c in candidate_structures if c not in selected]
            leftover.sort(key=lambda c: float(c.get('prob', 0.0)), reverse=True)
            selected.extend(leftover[: k - len(selected)])
        return selected[:k]

    selected: List[Dict] = []
    covered = np.zeros(len(witness_weights), dtype=np.float32)
    remaining = list(candidate_structures)
    rescue_used = False
    score_lookup = {}
    if learned_scores is not None:
        for cand, score in zip(candidate_structures, learned_scores):
            score_lookup[id(cand)] = float(score)

    while remaining and len(selected) < k:
        best_idx = 0
        best_score = -1e18
        for i, cand in enumerate(remaining):
            cov = _coverage_array(cand, len(witness_weights))
            marginal = float(np.sum(witness_weights * np.maximum(cov - covered, 0.0)))
            diversity = 0.0
            if selected:
                cur_slots = set(cand.get('slot_ids', []))
                diversity = min(len(cur_slots.symmetric_difference(set(prev.get('slot_ids', [])))) for prev in selected)
            uplift = score_lookup.get(id(cand), 0.0)
            score = marginal + 1e-3 * diversity + 1e-4 * float(cand.get('prob', 0.0)) + 0.15 * uplift
            if score > best_score:
                best_idx = i
                best_score = score
        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        covered = np.maximum(covered, _coverage_array(chosen, len(witness_weights)))

        if rescue and (not rescue_used) and len(selected) < k and len(remaining) > 0 and len(covered):
            uncovered_idx = int(np.argmin(covered))
            if covered[uncovered_idx] < 1e-4:
                rescue_idx = max(range(len(remaining)), key=lambda i: float(_coverage_array(remaining[i], len(witness_weights))[uncovered_idx]))
                rescue_cand = remaining.pop(rescue_idx)
                selected.append(rescue_cand)
                covered = np.maximum(covered, _coverage_array(rescue_cand, len(witness_weights)))
                rescue_used = True
    return selected[:k]
