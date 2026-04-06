from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
from pathlib import Path as P
from dpuc.config import load_config
from dpuc.data.dataset import _flatten_processed
from dpuc.eval.offline_eval import run_experiment_suite
from dpuc.utils.io import ensure_dir, save_json

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=None)
parser.add_argument('--split', type=str, default='val')
parser.add_argument('--limit', type=int, default=0)
args = parser.parse_args()
cfg = load_config(args.config)
samples = _flatten_processed(P(cfg.data.processed_dir) / args.split)
if args.limit > 0:
    samples = samples[:args.limit]
result = run_experiment_suite(cfg, samples, split=args.split)
out_dir = ensure_dir(P(cfg.output_dir) / 'eval')
save_json(result, out_dir / f'experiments_{args.split}.json')
print(result)
