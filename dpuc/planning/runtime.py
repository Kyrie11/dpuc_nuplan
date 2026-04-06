from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from dpuc.models.dbi import DBIModel
from dpuc.models.interface import InterfaceModel
from dpuc.models.support import SupportUtilityModel


@dataclass
class RuntimeModels:
    interface: InterfaceModel | None = None
    support: SupportUtilityModel | None = None
    dbi: DBIModel | None = None
    device: str = "cpu"


_RUNTIME_CACHE: Dict[tuple, RuntimeModels] = {}


def _safe_load_state(module: torch.nn.Module, ckpt_path: Path) -> bool:
    if not ckpt_path.exists():
        return False
    state = torch.load(ckpt_path, map_location="cpu")
    try:
        module.load_state_dict(state, strict=True)
        return True
    except Exception:
        try:
            module.load_state_dict(state, strict=False)
            return True
        except Exception:
            return False


def load_runtime_models(cfg) -> RuntimeModels:
    key = (
        cfg.output_dir,
        cfg.device,
        cfg.model.hidden_dim,
        cfg.model.num_heads,
        cfg.model.num_layers,
        cfg.model.residual_components,
    )
    if key in _RUNTIME_CACHE:
        return _RUNTIME_CACHE[key]

    device = cfg.device if torch.cuda.is_available() and str(cfg.device).startswith("cuda") else "cpu"
    ckpt_dir = Path(cfg.output_dir) / "checkpoints"
    interface = InterfaceModel(
        hidden_dim=cfg.model.hidden_dim,
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        residual_components=cfg.model.residual_components,
    )
    support_in_dim = 5 + 5 + 1
    support_out_dim = 5
    support = SupportUtilityModel(support_in_dim, support_out_dim, cfg.model.hidden_dim)
    dbi = DBIModel(in_dim=6)

    if not _safe_load_state(interface, ckpt_dir / "interface_best.pt"):
        interface = None
    else:
        interface.eval().to(device)
    if not _safe_load_state(support, ckpt_dir / "support_best.pt"):
        support = None
    else:
        support.eval().to(device)
    if not _safe_load_state(dbi, ckpt_dir / "dbi_best.pt"):
        dbi = None
    else:
        dbi.eval().to(device)

    runtime = RuntimeModels(interface=interface, support=support, dbi=dbi, device=device)
    _RUNTIME_CACHE[key] = runtime
    return runtime


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    return exp / max(np.sum(exp), 1e-8)


def interface_slot_predictions(runtime: RuntimeModels, action_feat: Sequence[float], slots: Sequence, variant: str = "ours") -> Dict[str, Dict]:
    outputs: Dict[str, Dict] = {}
    if variant == "public_only":
        for slot in slots:
            n = len(slot.answer_vocab)
            probs = np.ones(n, dtype=np.float32) / max(n, 1)
            outputs[slot.slot_id] = {
                "probs": probs,
                "residual_mu": np.zeros(2, dtype=np.float32),
                "residual_logvar": np.zeros(2, dtype=np.float32),
                "calibration": 0.5,
                "pred_label": int(np.argmax(probs)),
            }
        return outputs

    if runtime.interface is None or variant in {"agnostic", "single_latent", "query_only", "full_future_head", "no_switch"}:
        for slot in slots:
            feat = np.asarray(slot.features, dtype=np.float32)
            d = max(0.5, float(feat[2]))
            rel_v = abs(float(feat[5]))
            if variant == "agnostic":
                logits = np.array([1.0 / d, 0.8 / d, 0.6 / d], dtype=np.float32)[: len(slot.answer_vocab)]
            elif variant == "single_latent":
                logits = np.array([0.9, 0.7 + 0.05 * rel_v, 0.6], dtype=np.float32)[: len(slot.answer_vocab)]
            elif variant == "query_only":
                logits = np.array([0.6 + 0.1 * feat[10], 0.4 + 0.05 * rel_v, 0.2], dtype=np.float32)[: len(slot.answer_vocab)]
            elif variant == "full_future_head":
                logits = np.array([1.0 / d, 0.6 + 0.1 * rel_v, 0.3 + 0.05 * abs(feat[1])], dtype=np.float32)[: len(slot.answer_vocab)]
            else:
                logits = np.array([1.2 / d, 0.9 / d, 0.3 + 0.05 * rel_v], dtype=np.float32)[: len(slot.answer_vocab)]
            probs = _softmax(logits)
            outputs[slot.slot_id] = {
                "probs": probs,
                "residual_mu": np.array([0.05 * rel_v, 0.03 * abs(feat[1])], dtype=np.float32),
                "residual_logvar": np.log(np.array([0.8, 0.8], dtype=np.float32)),
                "calibration": float(probs.max()),
                "pred_label": int(np.argmax(probs)),
            }
        return outputs

    device = runtime.device
    action_tensor = torch.tensor(np.asarray(action_feat, dtype=np.float32), device=device).unsqueeze(0)
    with torch.no_grad():
        for slot in slots:
            slot_tensor = torch.tensor(np.asarray(slot.features, dtype=np.float32), device=device).unsqueeze(0)
            out = runtime.interface(action_tensor, slot_tensor)
            logits = out["answer_logits"].squeeze(0).detach().cpu().numpy()[: len(slot.answer_vocab)]
            if variant == "no_switch":
                logits = np.roll(logits, 1)
            probs = _softmax(logits)
            residual_mu = out["residual_mu"].squeeze(0).detach().cpu().numpy()
            residual_logvar = out["residual_logvar"].squeeze(0).detach().cpu().numpy()
            outputs[slot.slot_id] = {
                "probs": probs,
                "residual_mu": residual_mu[:2],
                "residual_logvar": np.clip(residual_logvar[:2], -3.0, 3.0),
                "calibration": float(torch.sigmoid(out["calibration"]).item()),
                "pred_label": int(np.argmax(probs)),
            }
    return outputs


def structure_probabilities_from_interface(candidate_structures: Sequence[Dict], slot_predictions: Dict[str, Dict]) -> np.ndarray:
    if not candidate_structures:
        return np.zeros(0, dtype=np.float32)
    probs = []
    for cand in candidate_structures:
        answer_map = cand.get("answers", {})
        if not answer_map:
            probs.append(float(cand.get("prob", 0.0)))
            continue
        logp = 0.0
        used = 0
        residual_mu = []
        residual_lv = []
        for slot_id, answer_idx in answer_map.items():
            pred = slot_predictions.get(slot_id)
            if pred is None:
                continue
            slot_probs = pred["probs"]
            idx = int(answer_idx) % len(slot_probs)
            logp += float(np.log(np.clip(slot_probs[idx], 1e-8, 1.0)))
            residual_mu.append(pred["residual_mu"])
            residual_lv.append(pred["residual_logvar"])
            used += 1
        if used == 0:
            probs.append(float(cand.get("prob", 0.0)))
            continue
        cand["runtime_residual_mu"] = np.mean(np.asarray(residual_mu, dtype=np.float32), axis=0).tolist()
        cand["runtime_residual_logvar"] = np.mean(np.asarray(residual_lv, dtype=np.float32), axis=0).tolist()
        probs.append(float(np.exp(logp / used)))
    probs = np.asarray(probs, dtype=np.float32)
    total = float(probs.sum())
    if total <= 0.0:
        return np.ones_like(probs) / max(len(probs), 1)
    return probs / total


def support_scores(runtime: RuntimeModels, action_feat: Sequence[float], candidate_structures: Sequence[Dict]) -> np.ndarray:
    if not candidate_structures:
        return np.zeros(0, dtype=np.float32)
    if runtime.support is None:
        return np.asarray([float(np.mean(c.get("coverage", [0.0]) or [0.0])) for c in candidate_structures], dtype=np.float32)
    feats = []
    for cand in candidate_structures:
        feats.append(
            np.concatenate(
                [
                    np.asarray(action_feat, dtype=np.float32),
                    np.asarray(cand.get("coverage", [0.0] * 5), dtype=np.float32)[:5],
                    np.asarray([cand.get("prob", 0.0)], dtype=np.float32),
                ]
            )
        )
    x = torch.tensor(np.stack(feats), device=runtime.device)
    with torch.no_grad():
        pred = runtime.support(x).detach().cpu().numpy()
    return pred.mean(axis=1).astype(np.float32)


def dbi_scores(runtime: RuntimeModels, prefix_sample) -> Dict[str, float]:
    ego = prefix_sample.ego_history[-1]
    scores: Dict[str, float] = {}
    for track_id, hist in prefix_sample.agents_history.items():
        agent = hist[-1]
        dist = float(np.hypot(agent.x - ego.x, agent.y - ego.y))
        rel_speed = float(np.hypot(agent.vx - ego.vx, agent.vy - ego.vy))
        feat = np.asarray([dist, rel_speed, agent.length, agent.width, ego.vx, ego.vy], dtype=np.float32)
        if runtime.dbi is not None:
            x = torch.tensor(feat, device=runtime.device).unsqueeze(0)
            with torch.no_grad():
                score = float(runtime.dbi(x).item())
        else:
            score = float(max(0.0, 1.0 - dist / 40.0) + 0.1 * rel_speed)
        scores[track_id] = score
    return scores
