
from __future__ import annotations
from typing import Dict, Iterable, List
import numpy as np


def gap_mae(ref: np.ndarray, pred: np.ndarray) -> float:
    errs = []
    for i in range(len(ref)):
        for j in range(len(ref)):
            if i == j:
                continue
            errs.append(abs((pred[i] - pred[j]) - (ref[i] - ref[j])))
    return float(np.mean(errs)) if errs else 0.0


def pair_acc(ref: np.ndarray, pred: np.ndarray, tie_eps: float = 0.01) -> float:
    def sgn(x):
        if x > tie_eps:
            return 1
        if x < -tie_eps:
            return -1
        return 0
    acc = []
    for i in range(len(ref)):
        for j in range(len(ref)):
            if i == j:
                continue
            acc.append(float(sgn(pred[i] - pred[j]) == sgn(ref[i] - ref[j])))
    return float(np.mean(acc)) if acc else 0.0


def top1(ref: np.ndarray, pred: np.ndarray) -> float:
    return float(int(np.argmin(ref) == np.argmin(pred)))


def dir_metric(ref: np.ndarray, pred: np.ndarray) -> float:
    return float(ref[np.argmin(pred)] - ref.min())


def mass_recall(selected_probs: np.ndarray, full_probs: np.ndarray) -> float:
    return float(selected_probs.sum() / (full_probs.sum() + 1e-6))


def boundary_rec(selected_ids: set, oracle_boundary_ids: set) -> float:
    if not oracle_boundary_ids:
        return 0.0
    return len(selected_ids & oracle_boundary_ids) / len(oracle_boundary_ids)


def gap_preservation(ref: np.ndarray, pred: np.ndarray, gamma_bd: float = 0.10) -> float:
    total, hit = 0, 0
    for i in range(len(ref)):
        for j in range(i + 1, len(ref)):
            if abs(ref[i] - ref[j]) <= gamma_bd:
                total += 1
                if np.sign(ref[i] - ref[j]) == np.sign(pred[i] - pred[j]):
                    hit += 1
    return float(hit / max(1, total))


def spearman_rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def aurc(risk: np.ndarray, confidence: np.ndarray) -> float:
    order = np.argsort(-confidence)
    risk = risk[order]
    curve = np.cumsum(risk) / np.arange(1, len(risk) + 1)
    return float(curve.mean())
