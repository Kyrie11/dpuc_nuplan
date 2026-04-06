from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def gaussian_logpdf(x: np.ndarray, mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
    return -0.5 * (np.log(2.0 * np.pi) + logvar + ((x - mu) ** 2) / np.exp(logvar)).sum(axis=-1)


def draw_bridge_samples(structure: Dict, n: int, seed: int = 0, frozen_bank: np.ndarray | None = None) -> np.ndarray:
    if frozen_bank is not None:
        return np.asarray(frozen_bank, dtype=np.float32)
    rng = np.random.default_rng(seed)
    mu = np.asarray(structure.get('runtime_residual_mu', structure.get('residual_mu', [0.0, 0.0])), dtype=np.float32)
    std = np.exp(0.5 * np.asarray(structure.get('runtime_residual_logvar', structure.get('residual_logvar', [0.0, 0.0])), dtype=np.float32))
    return rng.normal(mu, std, size=(n, mu.shape[0])).astype(np.float32)


def _target_residual_params(action_name: str, structure: Dict, interface_variant: str = 'ours') -> Tuple[np.ndarray, np.ndarray]:
    mu = np.asarray(structure.get('runtime_residual_mu', structure.get('residual_mu', [0.0, 0.0])), dtype=np.float32).copy()
    logvar = np.asarray(structure.get('runtime_residual_logvar', structure.get('residual_logvar', [0.0, 0.0])), dtype=np.float32).copy()
    if action_name in ('strong_brake', 'comfort_brake', 'stop_hold'):
        mu += np.array([-0.15, 0.1], dtype=np.float32)
    elif action_name in ('lane_change_left', 'lane_change_right', 'route_commit'):
        mu += np.array([0.1, 0.15], dtype=np.float32)
        logvar += np.log(1.15)
    else:
        mu += np.array([0.05, 0.0], dtype=np.float32)
    if interface_variant == 'public_only':
        logvar += np.log(1.3)
    elif interface_variant == 'single_latent':
        logvar += np.log(1.15)
    return mu, logvar


def evaluate_structure_cost(action_name: str, samples: np.ndarray, structure: Dict | None = None) -> np.ndarray:
    prog_cost = {
        'lane_follow': 0.2,
        'mild_accel': 0.0,
        'comfort_brake': 0.5,
        'strong_brake': 0.8,
        'creep': 0.6,
        'stop_hold': 1.0,
        'lane_change_left': 0.3,
        'lane_change_right': 0.3,
        'route_commit': 0.25,
    }[action_name]
    interaction = np.linalg.norm(samples, axis=1)
    structure_penalty = 0.0
    if structure is not None:
        answers = structure.get('answers', {})
        structure_penalty = 0.05 * len(answers)
        structure_penalty += 0.1 * sum(1 for v in answers.values() if int(v) > 0)
    return prog_cost + 0.2 * interaction + structure_penalty


def frozen_support_value(action_name: str, structures: List[Dict], probs: List[float], bridge_n: int, seed: int = 0, weight_clip: float = 20.0, variant: str = 'ours', interface_variant: str = 'ours'):
    total = 0.0
    ess_list = []
    max_norm_weight = 0.0
    value_parts = []
    frozen_banks = [draw_bridge_samples(structure, bridge_n, seed + i) for i, structure in enumerate(structures)]
    for i, (structure, prob) in enumerate(zip(structures, probs)):
        frozen_bank = frozen_banks[i]
        samples = draw_bridge_samples(None, bridge_n, frozen_bank=frozen_bank)
        bridge_mu = np.asarray(structure.get('residual_mu', [0.0, 0.0]), dtype=np.float32)
        bridge_logvar = np.asarray(structure.get('residual_logvar', [0.0, 0.0]), dtype=np.float32)
        target_mu, target_logvar = _target_residual_params(action_name, structure, interface_variant=interface_variant)
        if variant == 'perifacemc':
            samples = draw_bridge_samples(structure, bridge_n, seed + 1009 * (i + 1))
            weights = np.ones(bridge_n, dtype=np.float32)
        elif variant == 'frozen_nobridge':
            weights = np.ones(bridge_n, dtype=np.float32)
        else:
            logw = gaussian_logpdf(samples, target_mu, target_logvar) - gaussian_logpdf(samples, bridge_mu, bridge_logvar)
            if variant == 'directis':
                weights = np.exp(np.clip(logw, -10.0, np.log(weight_clip)))
            else:
                weights = np.exp(np.clip(logw, -6.0, np.log(weight_clip)))
        clipped = np.clip(weights, 0.0, weight_clip)
        normalized = clipped / max(clipped.sum(), 1e-6)
        max_norm_weight = max(max_norm_weight, float(normalized.max(initial=0.0)))
        ess = float((clipped.sum() ** 2) / (np.square(clipped).sum() + 1e-6))
        ess_list.append(ess)
        costs = evaluate_structure_cost(action_name, samples, structure)
        part = float(prob) * float(np.sum(normalized * costs))
        total += part
        value_parts.append(part)
    min_ess = min(ess_list) if ess_list else float(bridge_n)
    return total, min_ess, max_norm_weight, np.asarray(value_parts, dtype=np.float32)
