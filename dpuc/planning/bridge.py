
from __future__ import annotations
from typing import Dict, List
import numpy as np


def gaussian_logpdf(x: np.ndarray, mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
    return -0.5 * (np.log(2 * np.pi) + logvar + ((x - mu) ** 2) / np.exp(logvar))


def draw_bridge_samples(structure: Dict, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mu = np.asarray(structure.get('residual_mu', [0.0, 0.0]), dtype=np.float32)
    std = np.exp(0.5 * np.asarray(structure.get('residual_logvar', [0.0, 0.0]), dtype=np.float32))
    return rng.normal(mu, std, size=(n, mu.shape[0]))


def evaluate_structure_cost(action_name: str, samples: np.ndarray) -> np.ndarray:
    prog_cost = {
        'lane_follow': 0.2, 'mild_accel': 0.0, 'comfort_brake': 0.5, 'strong_brake': 0.8, 'creep': 0.6,
        'stop_hold': 1.0, 'lane_change_left': 0.3, 'lane_change_right': 0.3, 'route_commit': 0.25,
    }[action_name]
    interaction = np.linalg.norm(samples, axis=1)
    return prog_cost + 0.2 * interaction


def frozen_support_value(action_name: str, structures: List[Dict], probs: List[float], bridge_n: int, seed: int = 0):
    total = 0.0
    ess_list = []
    for i, (s, p) in enumerate(zip(structures, probs)):
        z = draw_bridge_samples(s, bridge_n, seed + i)
        costs = evaluate_structure_cost(action_name, z)
        w = np.ones_like(costs)
        ess = (w.sum() ** 2) / (np.square(w).sum() + 1e-6)
        ess_list.append(float(ess))
        total += float(p) * float(costs.mean())
    return total, min(ess_list) if ess_list else bridge_n
