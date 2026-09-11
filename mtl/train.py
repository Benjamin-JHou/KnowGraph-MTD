"""Train and evaluate the multitask bioactivity model (CLI entry point).

Examples::

    # Full model (DWA + gradient surgery + adapters), scaffold split, one seed
    python -m mtl.train --data-dir data/bioactivity --features-dir data/features \
        --split scaffold --ablation full --seed 22

    # Five-seed reproducibility run (manuscript protocol)
    python -m mtl.train --ablation full --seeds 22,42,62,82,102

    # Ablation sweep
    for a in full no_dwa no_gs no_adapter dwa_gs dwa_adapter gs_adapter; do
        python -m mtl.train --ablation $a --seeds 22,42,62,82,102
    done

    # Single-task baseline (seven independent models, one per target)
    python -m mtl.train --ablation stl --seeds 22,42,62,82,102

Outputs (under ``--out-dir``): per-seed ``*_metrics.json`` (per-target and
aggregate test metrics), ``*_predictions.csv`` (pIC50 true/predicted), and
``*_history.json`` (training curves).
"""

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config, resolve_flags
from .data import build_all_task_data
from .model import MTLModel
from .trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Multitask bioactivity model (KnowGraph-MTD).")
    parser.add_argument("--data-dir", default=Config().data_dir)
    parser.add_argument("--features-dir", default=Config().features_dir)
    parser.add_argument("--prepared-dir", default=Config().prepared_dir)
    parser.add_argument("--split", choices=["random", "scaffold"], default=Config().split)
    parser.add_argument("--ablation", choices=[
        "full", "no_dwa", "no_gs", "no_adapter", "dwa_gs", "dwa_adapter", "gs_adapter", "stl",
    ], default=Config().ablation)
    parser.add_argument("--gs-order", choices=["desc", "asc", "random"], default=Config().gs_order)
    parser.add_argument("--seed", type=int, default=Config().seed)
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seed list (overrides --seed), e.g. 22,42,62,82,102")
    parser.add_argument("--targets", type=str, default=None,
                        help="Comma-separated target names (default: all seven targets)")
    parser.add_argument("--epochs", type=int, default=Config().epochs)
    parser.add_argument("--batch-size", type=int, default=Config().batch_size)
    parser.add_argument("--device", default=Config().device)
    parser.add_argument("--out-dir", default=Config().out_dir)
    return parser.parse_args()


def _make_config(args) -> Config:
    cfg = Config(
        data_dir=args.data_dir,
        features_dir=args.features_dir,
        prepared_dir=args.prepared_dir,
        split=args.split,
        ablation=args.ablation,
        gs_order=args.gs_order,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        out_dir=args.out_dir,
    )
    if args.seeds:
        cfg.seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.targets:
        cfg.targets = [t for t in args.targets.split(",") if t.strip()]
    return cfg


def _resolve_device(cfg: Config) -> torch.device:
    if cfg.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.device)


def _set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loaders(datasets: Dict[str, tuple], cfg: Config, seed: int):
    """Build per-task DataLoaders with seeded worker behavior."""
    from torch.utils.data import DataLoader
    g = torch.Generator()
    g.manual_seed(seed)

    def make(ds):
        return DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, generator=g, drop_last=False,
        )

    train = {t: make(d[0]) for t, d in datasets.items()}
    val = {t: DataLoader(d[1], batch_size=cfg.batch_size, shuffle=False) for t, d in datasets.items()}
    test = {t: DataLoader(d[2], batch_size=cfg.batch_size, shuffle=False) for t, d in datasets.items()}
    return train, val, test


def run_seed(cfg: Config, seed: int, device: torch.device, flags: dict) -> dict:
    _set_seed(seed)
    datasets = build_all_task_data(cfg)
    task_names = list(datasets.keys())
    train_ld, val_ld, test_ld = _loaders(datasets, cfg, seed)

    model = MTLModel(cfg, task_names, use_adapters=flags["use_adapters"])
    print(f"[train] ablation={cfg.ablation} seed={seed} params={model.n_parameters:,} "
          f"device={device}")

    if cfg.ablation == "stl":
        # Seven independent single-task models with the same backbone.
        results = {}
        for t in task_names:
            stl_model = MTLModel(cfg, [t], use_adapters=False)
            trainer = Trainer(cfg, stl_model, [t], device)
            trainer.configure(use_dwa=False, use_gs=False, use_adapters=False)
            t_train = {t: train_ld[t]}
            t_val = {t: val_ld[t]}
            t_test = {t: test_ld[t]}
            prefix = os.path.join(cfg.out_dir, f"{cfg.split}_{cfg.ablation}_seed{seed}_{t}")
            m, agg, _ = trainer.fit(t_train, t_val, t_test, prefix)
            results[t] = m[t]
        agg = Trainer._aggregate(results)
    else:
        trainer = Trainer(cfg, model, task_names, device)
        trainer.configure(
            use_dwa=flags["use_dwa"], use_gs=flags["use_gs"], use_adapters=flags["use_adapters"]
        )
        prefix = os.path.join(cfg.out_dir, f"{cfg.split}_{cfg.ablation}_seed{seed}")
        m, agg, _ = trainer.fit(train_ld, val_ld, test_ld, prefix)

    return {"per_task": m, "aggregate": agg}


def main():
    args = parse_args()
    cfg = _make_config(args)
    device = _resolve_device(cfg)
    flags = resolve_flags(cfg.ablation)

    seeds = cfg.seeds or [cfg.seed]
    os.makedirs(cfg.out_dir, exist_ok=True)

    all_agg = []
    for seed in seeds:
        res = run_seed(cfg, seed, device, flags)
        all_agg.append(res["aggregate"])
        print(f"[train] seed={seed} aggregate: " +
              ", ".join(f"{k}={v:.4f}" for k, v in res["aggregate"].items()))

    if len(seeds) > 1:
        keys = list(all_agg[0].keys())
        summary = {}
        for k in keys:
            vals = [a[k] for a in all_agg if np.isfinite(a[k])]
            summary[k] = {
                "mean": float(np.mean(vals)) if vals else None,
                "std": float(np.std(vals)) if vals else None,
                "cv": float(np.std(vals) / np.mean(vals)) if vals and np.mean(vals) else None,
            }
        with open(os.path.join(cfg.out_dir, f"{cfg.split}_{cfg.ablation}_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print("[train] multi-seed summary:")
        for k, v in summary.items():
            print(f"  {k}: mean={v['mean']:.4f} std={v['std']:.4f} cv={v['cv']:.4f}")


if __name__ == "__main__":
    main()
