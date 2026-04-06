from __future__ import annotations
from typing import Iterable, List
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


def slot_nll(labels: List[int], probs: List[np.ndarray]) -> float:
    if not labels:
        return 0.0
    vals = []
    for y, p in zip(labels, probs):
        idx = int(y) % len(p)
        vals.append(-float(np.log(np.clip(p[idx], 1e-8, 1.0))))
    return float(np.mean(vals))


def slot_ece(labels: List[int], probs: List[np.ndarray], n_bins: int = 10) -> float:
    if not labels:
        return 0.0
    conf = np.asarray([float(np.max(p)) for p in probs], dtype=np.float32)
    pred = np.asarray([int(np.argmax(p)) for p in probs], dtype=np.int64)
    labels_arr = np.asarray(labels, dtype=np.int64)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        mask = (conf >= lo) & (conf < hi if b < n_bins - 1 else conf <= hi)
        if not np.any(mask):
            continue
        acc = float(np.mean(pred[mask] == labels_arr[mask]))
        c = float(np.mean(conf[mask]))
        ece += float(np.mean(mask)) * abs(acc - c)
    return float(ece)


def relative_variance(values: List[np.ndarray]) -> float:
    if not values:
        return 0.0
    arr = np.stack(values)
    denom = np.mean(np.abs(arr), axis=0) + 1e-6
    return float(np.mean(np.var(arr, axis=0) / denom))


def flip_rate(values: List[np.ndarray]) -> float:
    if len(values) < 2:
        return 0.0
    winners = [int(np.argmin(v)) for v in values]
    base = winners[0]
    return float(np.mean([w != base for w in winners[1:]]))


def voi_mae(pred_gain: float, oracle_gain: float) -> float:
    return float(abs(pred_gain - oracle_gain))


def spearman_rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def topk_recall(pred_scores: np.ndarray, oracle_scores: np.ndarray, k: int) -> float:
    if len(pred_scores) == 0 or len(oracle_scores) == 0 or k <= 0:
        return 0.0
    pred_idx = set(np.argsort(-pred_scores)[: min(k, len(pred_scores))].tolist())
    oracle_idx = set(np.argsort(-oracle_scores)[: min(k, len(oracle_scores))].tolist())
    return float(len(pred_idx & oracle_idx) / max(1, len(oracle_idx)))


def aurc(risk: np.ndarray, confidence: np.ndarray) -> float:
    order = np.argsort(-confidence)
    risk = risk[order]
    curve = np.cumsum(risk) / np.arange(1, len(risk) + 1)
    return float(curve.mean())


def worst_k_mean(vals: Iterable[float], frac: float = 0.05) -> float:
    vals = np.asarray(list(vals), dtype=np.float32)
    if vals.size == 0:
        return 0.0
    k = max(1, int(np.ceil(frac * vals.size)))
    return float(np.mean(np.sort(vals)[-k:]))
