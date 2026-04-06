from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
from tqdm import tqdm

from dpuc.config import load_config
from dpuc.data.dataset import _flatten_processed
from dpuc.eval.metrics import (
    aurc,
    boundary_rec,
    dir_metric,
    flip_rate,
    gap_mae,
    gap_preservation,
    mass_recall,
    pair_acc,
    relative_variance,
    slot_ece,
    slot_nll,
    spearman_rank_correlation,
    top1,
    topk_recall,
    voi_mae,
    worst_k_mean,
)
from dpuc.planning.planner import run_planner
from dpuc.planning.runtime import load_runtime_models
from dpuc.utils.io import ensure_dir, save_json


INTERFACE_BASELINES = ['public_only', 'agnostic', 'no_switch', 'single_latent', 'query_only', 'full_future_head', 'ours']
SUPPORT_BASELINES = ['masstopk', 'diversetopk', 'structtopk', 'uncunion', 'ours']
BRIDGE_BASELINES = ['perifacemc', 'directis', 'frozen_nobridge', 'ours']
IND_BASELINES = ['public_only', 'nearest-b', 'ttc-b', 'entropy-b', 'boundarysens-b', 'resamplevoi', 'random-b', 'ours', 'allind']
ABLATIONS = {
    'ours': {},
    'w_o_action_conditioning': {'planner': {'interface_variant': 'agnostic'}},
    'w_o_rescue_support': {'planner': {'use_rescue_support': False}},
    'w_o_uplift_term': {'planner': {'use_uplift_term': False}},
    'w_o_bridge_bank': {'planner': {'bridge_variant': 'frozen_nobridge'}},
    'w_o_exact_dbi': {'planner': {'dbi_exact': False}},
    'w_o_local_closure_refresh': {'planner': {'use_local_closure_refresh': False}},
    'w_o_correction_fallback': {'planner': {'use_correction_fallback': False}},
}


def _deep_override(cfg, override):
    cfg = deepcopy(cfg)
    for section, updates in override.items():
        block = getattr(cfg, section)
        for key, value in updates.items():
            setattr(block, key, value)
    return cfg


def _scene_is_interactive(sample, gap_thr: float) -> bool:
    has_conflict = any(slot.slot_type in ('precedence', 'merge_gap', 'release', 'occupancy') for a in sample.actions for slot in a.slots)
    public_vals = np.asarray([a.public_value for a in sample.actions], dtype=np.float32)
    if public_vals.size < 2:
        return False
    top2 = np.sort(public_vals)[:2]
    return bool(has_conflict and abs(float(top2[1] - top2[0])) < gap_thr)


def _latency_proxy(result) -> float:
    total_struct = sum(len(a['selected_structures']) for a in result['actions'])
    total_slots = sum(len(a['slot_predictions']) for a in result['actions'])
    refined = len(result.get('refined_agents', []))
    return float(1.8 + 0.08 * total_struct + 0.02 * total_slots + 0.12 * refined)


def _closed_loop_offline_proxies(sample, result):
    best = result['best']
    oracle_best = result['oracle_best']
    regret = float(best['oracle_value'] - oracle_best['oracle_value'])
    scenario_mult = 1.2 if _scene_is_interactive(sample, 0.4) else 1.0
    collision = float(max(0.0, regret * scenario_mult + 0.15 * result['flagged']))
    progress = float(max(0.0, 1.0 - 0.35 * best['value']))
    comfort = float(max(0.0, 1.0 - 0.25 * abs(best['value'] - best['public_value'])))
    route_success = float(progress > 0.55 and collision < 0.45)
    score = float(0.45 * progress + 0.25 * comfort + 0.20 * route_success - 0.30 * collision)
    return {
        'Score': score,
        'Collision': collision,
        'Progress': progress,
        'Comfort': comfort,
        'RouteSucc': route_success,
        'IntScore': score if _scene_is_interactive(sample, 0.4) else np.nan,
    }


def evaluate_samples(cfg, samples, split='val'):
    runtime = load_runtime_models(cfg)
    metrics = {
        'GapMAE': [], 'PairAcc': [], 'Top1': [], 'DIR': [], 'GapPres@K': [],
        'MassRec@K': [], 'BoundaryRec@K': [], 'MinESS': [], 'FallbackRate': [], 'Coverage': [],
        'RefinedAgents': [], 'LatencyMs': [], 'SlotNLL': [], 'SlotECE': [], 'AURC': [],
        'ReliabilityRisk': [], 'Confidence': [], 'PredVOI': [], 'OracleVOI': [], 'TopBRecall': [],
        'SRCC': [], 'GapGain@B': [], 'Worst5DIR': [], 'FlaggedCollision': [],
        'Score': [], 'Collision': [], 'Progress': [], 'Comfort': [], 'RouteSucc': [], 'IntScore': [],
    }
    bridge_vectors = []
    per_scene_dir = []

    for sample in tqdm(samples, desc=f'offline-eval-{split}'):
        result = run_planner(sample, cfg.planner, runtime=runtime)
        ref = np.asarray([a['oracle_value'] for a in result['actions']], dtype=np.float32)
        pred = np.asarray([a['value'] for a in result['actions']], dtype=np.float32)
        metrics['GapMAE'].append(gap_mae(ref, pred))
        metrics['PairAcc'].append(pair_acc(ref, pred, cfg.planner.tie_eps))
        metrics['Top1'].append(top1(ref, pred))
        scene_dir = dir_metric(ref, pred)
        metrics['DIR'].append(scene_dir)
        per_scene_dir.append(scene_dir)
        metrics['GapPres@K'].append(gap_preservation(ref, pred, cfg.planner.boundary_gap))
        metrics['MinESS'].append(min(a['ess'] for a in result['actions']))
        metrics['FallbackRate'].append(float(result['fallback_used']))
        metrics['Coverage'].append(1.0 - float(result['fallback_used']))
        metrics['RefinedAgents'].append(float(len(result.get('refined_agents', []))))
        metrics['LatencyMs'].append(_latency_proxy(result))
        metrics['Confidence'].append(1.0 / (1.0 + len(result['diagnostics']['reasons']) + max(0.0, result['diagnostics']['top_gap'])))
        labels, probs = [], []
        mass_recalls = []
        bnd_recalls = []
        for action in result['actions']:
            for slot in sample.actions[action['action_index']].slots:
                pred_slot = action['slot_predictions'][slot.slot_id]
                labels.append(int(slot.label))
                probs.append(np.asarray(pred_slot['probs'], dtype=np.float32))
            sel_probs = np.asarray([s.get('runtime_prob', s.get('prob', 0.0)) for s in action['selected_structures']], dtype=np.float32)
            mass_recalls.append(mass_recall(sel_probs, action['full_structure_probs']))
            bnd_recalls.append(boundary_rec(set(action['selected_boundary_ids']), set(action['oracle_boundary_ids'])))
        metrics['SlotNLL'].append(slot_nll(labels, probs))
        metrics['SlotECE'].append(slot_ece(labels, probs))
        metrics['MassRec@K'].append(float(np.mean(mass_recalls)) if mass_recalls else 0.0)
        metrics['BoundaryRec@K'].append(float(np.mean(bnd_recalls)) if bnd_recalls else 0.0)
        risk = float(result['best']['oracle_value'] - result['oracle_best']['oracle_value'])
        metrics['ReliabilityRisk'].append(risk)
        metrics['FlaggedCollision'].append(float(result['flagged']) * max(0.0, risk))

        oracle_gain = np.asarray(list(result['oracle_agent_gain'].values()), dtype=np.float32)
        pred_gain = np.asarray([score for _, score in result['agent_dbi_ranking']], dtype=np.float32)
        if oracle_gain.size and pred_gain.size:
            metrics['SRCC'].append(spearman_rank_correlation(pred_gain, oracle_gain))
            metrics['TopBRecall'].append(topk_recall(pred_gain, oracle_gain, max(1, cfg.planner.agent_budget)))
            metrics['PredVOI'].append(float(pred_gain[: max(1, cfg.planner.agent_budget)].sum()))
            metrics['OracleVOI'].append(float(np.sort(oracle_gain)[::-1][: max(1, cfg.planner.agent_budget)].sum()))
            all_ind = float(np.sum(np.sort(oracle_gain)[::-1]))
            ours = metrics['PredVOI'][-1]
            pub = 0.0
            denom = max(1e-6, all_ind - pub)
            metrics['GapGain@B'].append(float(np.clip((ours - pub) / denom, 0.0, 1.0)))
        else:
            metrics['SRCC'].append(0.0)
            metrics['TopBRecall'].append(0.0)
            metrics['PredVOI'].append(0.0)
            metrics['OracleVOI'].append(0.0)
            metrics['GapGain@B'].append(0.0)

        repeats = []
        for rep in range(max(1, cfg.planner.eval_repeats)):
            rep_cfg = deepcopy(cfg)
            rep_cfg.planner.bridge_variant = cfg.planner.bridge_variant
            rep_cfg.planner.bridge_n = cfg.planner.bridge_n
            rep_cfg.planner.weight_clip = cfg.planner.weight_clip
            rep_cfg.planner.seed_offset = rep
            rep_result = run_planner(sample, rep_cfg.planner, runtime=runtime)
            repeats.append(np.asarray([a['value'] for a in rep_result['actions']], dtype=np.float32))
        bridge_vectors.extend(repeats)

        proxy = _closed_loop_offline_proxies(sample, result)
        for k, v in proxy.items():
            metrics[k].append(v)

    summary = {k: float(np.nanmean(v)) if len(v) else 0.0 for k, v in metrics.items() if k not in {'ReliabilityRisk', 'Confidence'}}
    summary['RelVar'] = relative_variance(bridge_vectors)
    summary['FlipRate'] = flip_rate(bridge_vectors)
    summary['VOI-MAE'] = voi_mae(summary.get('PredVOI', 0.0), summary.get('OracleVOI', 0.0))
    summary['AURC'] = aurc(np.asarray(metrics['ReliabilityRisk'], dtype=np.float32), np.asarray(metrics['Confidence'], dtype=np.float32)) if metrics['ReliabilityRisk'] else 0.0
    summary['Worst5DIR'] = worst_k_mean(per_scene_dir, frac=0.05)
    return summary


def run_experiment_suite(cfg, samples, split='val'):
    suite = {}

    interface = {}
    for name in INTERFACE_BASELINES:
        run_cfg = deepcopy(cfg)
        run_cfg.planner.interface_variant = name
        interface[name] = evaluate_samples(run_cfg, samples, split=split)
    suite['planner_facing_interface_fidelity'] = interface

    support = {}
    for name in SUPPORT_BASELINES:
        run_cfg = deepcopy(cfg)
        run_cfg.planner.support_variant = name
        support[name] = evaluate_samples(run_cfg, samples, split=split)
    suite['decision_critical_support'] = support

    bridge = {}
    for name in BRIDGE_BASELINES:
        run_cfg = deepcopy(cfg)
        run_cfg.planner.bridge_variant = name
        bridge[name] = evaluate_samples(run_cfg, samples, split=split)
    suite['frozen_support_bridge'] = bridge

    ind = {}
    for name in IND_BASELINES:
        run_cfg = deepcopy(cfg)
        run_cfg.planner.individualization_variant = name
        if name == 'public_only':
            run_cfg.planner.agent_budget = 0
        ind[name] = evaluate_samples(run_cfg, samples, split=split)
    suite['selective_individualization'] = ind

    budget_curves = {'support_budget': {}, 'agent_budget': {}}
    for k in [2, 4, 6, 8, 10, 12, 16]:
        run_cfg = deepcopy(cfg)
        run_cfg.planner.retained_k = k
        budget_curves['support_budget'][str(k)] = evaluate_samples(run_cfg, samples, split=split)
    for b in [0, 1, 2, 3, 4]:
        run_cfg = deepcopy(cfg)
        run_cfg.planner.agent_budget = b
        budget_curves['agent_budget'][str(b)] = evaluate_samples(run_cfg, samples, split=split)
    suite['budget_curves'] = budget_curves

    ablations = {}
    for name, override in ABLATIONS.items():
        run_cfg = _deep_override(cfg, override)
        ablations[name] = evaluate_samples(run_cfg, samples, split=split)
    suite['ablations'] = ablations

    suite['main_closed_loop_offline_proxy'] = {'ours': evaluate_samples(cfg, samples, split=split)}
    suite['reliability'] = {
        'NoDiag': evaluate_samples(_deep_override(cfg, {'planner': {'use_correction_fallback': False}}), samples, split=split),
        'DiagOnly': evaluate_samples(_deep_override(cfg, {'planner': {'fallback_k': cfg.planner.retained_k}}), samples, split=split),
        'CorrOnly': evaluate_samples(_deep_override(cfg, {'planner': {'use_correction_fallback': True, 'fallback_k': cfg.planner.retained_k + 2}}), samples, split=split),
        'Ours (Corr+Fallback)': evaluate_samples(cfg, samples, split=split),
    }
    return suite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--mode', type=str, default='summary', choices=['summary', 'suite'])
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    cfg = load_config(args.config)
    samples = _flatten_processed(Path(cfg.data.processed_dir) / args.split)
    if args.limit > 0:
        samples = samples[: args.limit]
    out_dir = ensure_dir(Path(cfg.output_dir) / 'eval')
    if args.mode == 'suite':
        result = run_experiment_suite(cfg, samples, split=args.split)
        save_json(result, out_dir / f'experiments_{args.split}.json')
    else:
        result = evaluate_samples(cfg, samples, split=args.split)
        save_json(result, out_dir / f'offline_{args.split}_metrics.json')
    print(result)

if __name__ == '__main__':
    main()
