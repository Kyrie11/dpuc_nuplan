
from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import Dataset

from dpuc.utils.io import load_pickle


def _flatten_processed(split_dir: str | Path) -> List:
    split_dir = Path(split_dir)
    items = []
    for pkl in sorted(split_dir.glob('*.pkl')):
        items.extend(load_pickle(pkl))
    return items


class InterfaceDataset(Dataset):
    def __init__(self, split_dir: str | Path):
        self.samples = _flatten_processed(split_dir)
        self.rows = []
        for sample in self.samples:
            oracle_values = [a.oracle_value for a in sample.actions]
            for action in sample.actions:
                for slot in action.slots:
                    self.rows.append({
                        'sample_id': sample.sample_id,
                        'action_index': action.action_index,
                        'action_feat': np.asarray(action.action_features, dtype=np.float32),
                        'slot_feat': np.asarray(slot.features, dtype=np.float32),
                        'slot_type': slot.slot_type,
                        'label': slot.label,
                        'oracle_values': np.asarray(oracle_values, dtype=np.float32),
                        'public_value': np.float32(action.public_value),
                        'oracle_value': np.float32(action.oracle_value),
                    })

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        return {
            'action_feat': torch.tensor(row['action_feat']),
            'slot_feat': torch.tensor(row['slot_feat']),
            'label': torch.tensor(row['label']).long(),
            'oracle_values': torch.tensor(row['oracle_values']),
            'public_value': torch.tensor(row['public_value']),
            'oracle_value': torch.tensor(row['oracle_value']),
        }


class SupportDataset(Dataset):
    def __init__(self, split_dir: str | Path):
        self.samples = _flatten_processed(split_dir)
        self.rows = []
        for sample in self.samples:
            for action in sample.actions:
                witness_weights = np.asarray([w.weight for w in action.witnesses], dtype=np.float32)
                for cand in action.candidate_structures:
                    feat = np.concatenate([
                        np.asarray(action.action_features, dtype=np.float32),
                        np.asarray(cand['coverage'], dtype=np.float32),
                        np.asarray([cand['prob']], dtype=np.float32),
                    ])
                    target = np.asarray(cand['coverage'], dtype=np.float32)
                    self.rows.append({'feat': feat, 'target': target, 'weights': witness_weights})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        return {
            'feat': torch.tensor(row['feat']),
            'target': torch.tensor(row['target']),
            'weights': torch.tensor(row['weights']),
        }


class DBIDataset(Dataset):
    def __init__(self, split_dir: str | Path):
        self.samples = _flatten_processed(split_dir)
        self.rows = []
        for sample in self.samples:
            ego = sample.ego_history[-1]
            # heuristic oracle-VOI label from nearest agents.
            for track_id, hist in sample.agents_history.items():
                agent = hist[-1]
                dist = float(np.hypot(agent.x - ego.x, agent.y - ego.y))
                rel_speed = float(np.hypot(agent.vx - ego.vx, agent.vy - ego.vy))
                target = float(max(0.0, 1.0 - dist / 40.0) + 0.1 * rel_speed)
                feat = np.asarray([dist, rel_speed, agent.length, agent.width, ego.vx, ego.vy], dtype=np.float32)
                self.rows.append({'feat': feat, 'target': target})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        return {'feat': torch.tensor(row['feat']), 'target': torch.tensor(row['target'])}
