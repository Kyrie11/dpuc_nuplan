
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

from dpuc.config import load_config
from dpuc.data.dataset import _flatten_processed
from dpuc.eval.metrics import gap_mae, pair_acc, top1, dir_metric, gap_preservation
from dpuc.planning.planner import run_planner
from dpuc.utils.io import save_json, ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--split', type=str, default='val')
    args = parser.parse_args()
    cfg = load_config(args.config)
    samples = _flatten_processed(Path(cfg.data.processed_dir) / args.split)
    metrics = {'GapMAE': [], 'PairAcc': [], 'Top1': [], 'DIR': [], 'GapPres@K': [], 'MinESS': []}
    for sample in tqdm(samples, desc='offline-eval'):
        result = run_planner(sample, cfg.planner)
        ref = np.asarray([a['oracle_value'] for a in result['actions']], dtype=np.float32)
        pred = np.asarray([a['value'] for a in result['actions']], dtype=np.float32)
        metrics['GapMAE'].append(gap_mae(ref, pred))
        metrics['PairAcc'].append(pair_acc(ref, pred, cfg.planner.tie_eps))
        metrics['Top1'].append(top1(ref, pred))
        metrics['DIR'].append(dir_metric(ref, pred))
        metrics['GapPres@K'].append(gap_preservation(ref, pred, cfg.planner.boundary_gap))
        metrics['MinESS'].append(min(a['ess'] for a in result['actions']))
    summary = {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}
    out_dir = ensure_dir(Path(cfg.output_dir) / 'eval')
    save_json(summary, out_dir / f'offline_{args.split}_metrics.json')
    print(summary)

if __name__ == '__main__':
    main()
