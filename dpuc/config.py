
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List
import yaml

@dataclass
class DataConfig:
    raw_root: str = "/dataset/nuplan/data/cache"
    processed_dir: str = "data/processed"
    train_dirs: List[str] = field(default_factory=lambda: [
        "train_boston", "train_pittsburgh", "train_singapore", "train_vegas_2"
    ])
    val_dirs: List[str] = field(default_factory=lambda: ["public_set_val"])
    history_sec: float = 2.0
    horizon_sec: float = 6.0
    sample_interval_sec: float = 0.5
    max_agents: int = 20
    radius_m: float = 60.0

@dataclass
class ModelConfig:
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    residual_components: int = 2
    max_actions: int = 9
    max_slots: int = 64
    max_witnesses: int = 5

@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 30
    warmup_epochs: int = 2
    lr: float = 2e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    num_workers: int = 0
    seed: int = 42
    ans_weight: float = 1.0
    res_weight: float = 0.5
    rank_weight: float = 0.2
    cal_weight: float = 0.1
    selector_epochs: int = 10

@dataclass
class PlannerConfig:
    bank_cap: int = 64
    oracle_bank_cap: int = 256
    retained_k: int = 8
    fallback_k: int = 12
    bridge_n: int = 16
    oracle_bridge_n: int = 64
    action_budget: int = 9
    agent_budget: int = 3
    beam_width: int = 32
    lambda_viol: float = 5.0
    witness_gap_temp: float = 0.15
    tie_eps: float = 0.01
    boundary_gap: float = 0.10
    retained_mass_min: float = 0.8
    ess_min: float = 6.0
    max_norm_weight: float = 0.35
    weight_clip: float = 20.0

@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    output_dir: str = "outputs/default"
    device: str = "cuda"

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)


def _deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    cfg = ExperimentConfig()
    if path is None:
        return cfg
    with open(path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f) or {}
    base = asdict(cfg)
    merged = _deep_update(base, raw)
    return ExperimentConfig(
        data=DataConfig(**merged['data']),
        model=ModelConfig(**merged['model']),
        train=TrainConfig(**merged['train']),
        planner=PlannerConfig(**merged['planner']),
        output_dir=merged['output_dir'],
        device=merged['device'],
    )
