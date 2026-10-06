"""Copy the best checkpoints of the 2 delivered seeds and write checkpoints/model_info.json.

    python scripts/finalize_models.py
Every value is read from the training runs (runs/default_seed*/val.csv and the checkpoints).
"""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tds.model import QNet  # noqa: E402

TARGETS = {0: ROOT / "checkpoints" / "best.pt", 1: ROOT / "checkpoints" / "seed1_best.pt"}


def main():
    seeds = []
    cfg = None
    for s, dst in TARGETS.items():
        run = ROOT / "runs" / f"default_seed{s}"
        src = ROOT / "checkpoints" / f"default_seed{s}" / "best.pt"
        val = pd.read_csv(run / "val.csv")
        best = val.loc[val.gap_rr.idxmin()]
        ck = torch.load(src, map_location="cpu", weights_only=False)
        assert abs(ck["val"]["gap_rr"] - best.gap_rr) < 1e-9, "best.pt does not match the best validation row"
        shutil.copy2(src, dst)
        cfg = ck["model_cfg"]
        seeds.append({"seed": s, "checkpoint": str(dst.relative_to(ROOT)).replace("\\", "/"),
                      "train_steps": int(val.step.max()), "best_step": int(best.step),
                      "best_val_gap_rr": float(best.gap_rr), "best_val_gap_raw": float(best.gap_raw),
                      "best_val_frac_optimal_rr": float(best.frac_opt_rr),
                      "val_gap_greedy_rr": float(best.greedy_rr_gap), "val_gap_random_rr": float(best.random_rr_gap),
                      "episodes": int(val.episodes.max()), "finished": (run / "summary.json").exists()})
    m = QNet(**cfg)
    info = {"encoder": cfg["encoder"], "hidden": cfg["hidden"], "layers": cfg["layers"], "readout": cfg["readout"],
            "parameters": int(sum(p.numel() for p in m.parameters())), "seeds": seeds,
            "validation_note": "Validation: 200 fixed graphs (n = 50–100; ER, BA, WS, random geometric, lattices), "
                               "all optima proven by the exact ILP. Gap = mean (|S| − optimum) / optimum. "
                               "best.pt = checkpoint with the lowest gap after redundancy removal (RR)."}
    (ROOT / "checkpoints" / "model_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
