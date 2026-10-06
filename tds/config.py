"""YAML configs layered over DEFAULTS (a config file may `inherit:` another)."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS = {
    "seed": 0,
    "device": "auto",
    "model": {"hidden": 64, "layers": 3, "encoder": "mpnn", "readout": "meanmax"},
    "env": {"reward_mode": "unit", "shaped_beta": 0.5, "paper_alpha": 1.0,
            "use_forced": True, "mask_mode": "gain"},
    "train": {
        "total_steps": 300_000, "num_envs": 8, "buffer_size": 100_000, "batch_size": 64,
        "warmup": 2_000, "train_every": 2, "lr": 1.0e-4, "gamma": 1.0, "n_step": 1,
        "loss": "huber", "grad_clip": 10.0, "target_update": "soft", "tau": 0.005,
        "hard_every": 1_000, "eps_start": 1.0, "eps_end": 0.05, "eps_frac": 0.3,
        # curriculum: list of [fraction_of_steps_until, n_min, n_max]
        "curriculum": [[0.4, 20, 50], [1.0, 20, 100]],
        "families": {"er": 0.30, "ba": 0.25, "ws": 0.15, "rgg": 0.15, "lattice": 0.15},
        "weighted": False, "log_every": 1_000,
    },
    "val": {"every": 5_000, "path": "data/val_set.npz", "count": 200, "n_min": 50,
            "n_max": 100, "ilp_time": 60.0},
    "out": {"name": "default", "ckpt_dir": "checkpoints", "run_dir": "runs"},
}


def deep_update(base: dict, upd: dict) -> dict:
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path=None, overrides: dict = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        parent = raw.pop("inherit", None)
        if parent:
            cfg = load_config(Path(path).parent / parent)
        deep_update(cfg, raw)
    deep_update(cfg, overrides or {})
    return cfg


def resolve_device(name: str):
    import torch
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name
