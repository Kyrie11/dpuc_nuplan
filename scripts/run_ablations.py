from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
from dpuc.config import load_config

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=None)
args = parser.parse_args()
cfg = load_config(args.config)
variants = {'ours': {}, 'w_o_action_conditioning': {'planner': {'agent_budget': 0}}, 'w_o_rescue_support': {'planner': {'retained_k': 6}}, 'w_o_bridge_bank': {'planner': {'bridge_n': 1}}, 'w_o_exact_dbi': {'planner': {'agent_budget': 1}}, 'w_o_correction_fallback': {'planner': {'fallback_k': 8}}}
print('Ablation variants defined in README; launch train/eval per variant as needed.')
for name, override in variants.items():
    print(name, override)
