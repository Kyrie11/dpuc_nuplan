
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional

ACTION_NAMES = [
    'lane_follow', 'mild_accel', 'comfort_brake', 'strong_brake', 'creep', 'stop_hold',
    'lane_change_left', 'lane_change_right', 'route_commit'
]
SLOT_TYPES = ['precedence', 'merge_gap', 'release', 'route_branch', 'occupancy']
SLOT_ANSWERS = {
    'precedence': ['ego_first', 'other_first', 'joint'],
    'merge_gap': ['open_ahead', 'open_behind', 'closed'],
    'release': ['release_before', 'release_after', 'joint_release'],
    'route_branch': ['keep', 'branch_left', 'branch_right'],
    'occupancy': ['free', 'occupied'],
}

@dataclass
class AgentState:
    track_id: str
    category: str
    x: float
    y: float
    yaw: float
    vx: float
    vy: float
    length: float
    width: float
    t: float

@dataclass
class EgoState:
    x: float
    y: float
    yaw: float
    vx: float
    vy: float
    ax: float
    ay: float
    t: float

@dataclass
class Slot:
    slot_id: str
    owner_id: str
    slot_type: str
    anchor_x: float
    anchor_y: float
    owner_index: int
    answer_vocab: List[str]
    features: List[float]
    label: int = 0

@dataclass
class Witness:
    witness_id: str
    rival_action_index: int
    gap: float
    weight: float
    features: List[float]

@dataclass
class ActionSample:
    action_index: int
    action_name: str
    action_features: List[float]
    slots: List[Slot] = field(default_factory=list)
    witnesses: List[Witness] = field(default_factory=list)
    candidate_structures: List[Dict] = field(default_factory=list)
    oracle_value: float = 0.0
    public_value: float = 0.0

@dataclass
class PrefixSample:
    sample_id: str
    split: str
    log_name: str
    location: str
    scenario_type: str
    timestamp_us: int
    ego_history: List[EgoState]
    agents_history: Dict[str, List[AgentState]]
    future_ego: List[EgoState]
    actions: List[ActionSample]
