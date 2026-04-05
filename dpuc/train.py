
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dpuc.config import load_config
from dpuc.data.dataset import InterfaceDataset, SupportDataset, DBIDataset
from dpuc.models.interface import InterfaceModel
from dpuc.models.support import SupportUtilityModel
from dpuc.models.dbi import DBIModel
from dpuc.utils.io import ensure_dir
from dpuc.utils.seed import set_seed


def cosine_lr(optimizer, base_lr, step, total_steps, warmup_steps):
    if step < warmup_steps:
        lr = base_lr * (step + 1) / max(1, warmup_steps)
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        lr = 0.5 * base_lr * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535))).item()
    for group in optimizer.param_groups:
        group['lr'] = lr


def train_interface(cfg):
    set_seed(cfg.train.seed)
    train_ds = InterfaceDataset(Path(cfg.data.processed_dir) / 'train')
    val_ds = InterfaceDataset(Path(cfg.data.processed_dir) / 'val')
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    model = InterfaceModel(hidden_dim=cfg.model.hidden_dim, num_heads=cfg.model.num_heads, num_layers=cfg.model.num_layers, residual_components=cfg.model.residual_components).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    total_steps = cfg.train.epochs * max(1, len(train_loader))
    warmup_steps = cfg.train.warmup_epochs * max(1, len(train_loader))
    step = 0
    best = float('inf')
    ckpt_dir = ensure_dir(Path(cfg.output_dir) / 'checkpoints')
    for epoch in range(cfg.train.epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f'interface-train-{epoch}'):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch['action_feat'], batch['slot_feat'])
            rank_target = batch['oracle_value']
            loss = (
                cfg.train.ans_weight * ce(out['answer_logits'], batch['label']) +
                cfg.train.res_weight * mse(out['residual_mu'], torch.zeros_like(out['residual_mu'])) +
                cfg.train.rank_weight * mse(out['value'], rank_target) +
                cfg.train.cal_weight * mse(torch.sigmoid(out['calibration']), torch.ones_like(out['calibration']) * 0.8)
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            cosine_lr(optimizer, cfg.train.lr, step, total_steps, warmup_steps)
            step += 1
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(batch['action_feat'], batch['slot_feat'])
                val_loss += ce(out['answer_logits'], batch['label']).item()
        val_loss /= max(1, len(val_loader))
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), ckpt_dir / 'interface_best.pt')
    return ckpt_dir / 'interface_best.pt'


def train_support(cfg):
    set_seed(cfg.train.seed)
    ds = SupportDataset(Path(cfg.data.processed_dir) / 'train')
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True)
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    sample = ds[0]
    model = SupportUtilityModel(sample['feat'].numel(), sample['target'].numel(), cfg.model.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    mse = nn.MSELoss()
    ckpt_dir = ensure_dir(Path(cfg.output_dir) / 'checkpoints')
    for epoch in range(cfg.train.selector_epochs):
        model.train()
        for batch in tqdm(loader, desc=f'support-train-{epoch}'):
            feat = batch['feat'].to(device)
            target = batch['target'].to(device)
            pred = model(feat)
            loss = mse(pred, target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    torch.save(model.state_dict(), ckpt_dir / 'support_best.pt')
    return ckpt_dir / 'support_best.pt'


def train_dbi(cfg):
    set_seed(cfg.train.seed)
    ds = DBIDataset(Path(cfg.data.processed_dir) / 'train')
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True)
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    model = DBIModel(in_dim=ds[0]['feat'].numel()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    mse = nn.MSELoss()
    ckpt_dir = ensure_dir(Path(cfg.output_dir) / 'checkpoints')
    for epoch in range(10):
        for batch in tqdm(loader, desc=f'dbi-train-{epoch}'):
            feat = batch['feat'].to(device)
            target = batch['target'].to(device)
            pred = model(feat)
            loss = mse(pred, target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    torch.save(model.state_dict(), ckpt_dir / 'dbi_best.pt')
    return ckpt_dir / 'dbi_best.pt'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--stage', type=str, choices=['interface', 'support', 'dbi', 'all'], default='all')
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.stage in ('interface', 'all'):
        train_interface(cfg)
    if args.stage in ('support', 'all'):
        train_support(cfg)
    if args.stage in ('dbi', 'all'):
        train_dbi(cfg)

if __name__ == '__main__':
    main()
