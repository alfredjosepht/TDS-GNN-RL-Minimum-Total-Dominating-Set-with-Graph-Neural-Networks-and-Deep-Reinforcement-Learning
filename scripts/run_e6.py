"""E6: greedy rollout vs best of k in {4, 16} sampled rollouts (softmax over Q / T).

Step 1 picks the temperature T on the VALIDATION set (data/val_set.npz), using the
seed-0 model and k = 4, never the test sets. Step 2 evaluates all 5 seeds on the
held-out E5 sets with the chosen T and writes results/e6/.

    python scripts/run_e6.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tds.postprocess import redundancy_removal  # noqa: E402
from tds.rollout import rollout  # noqa: E402
from tds.solve import default_device, get_model  # noqa: E402
from tds.validation import get_validation_set  # noqa: E402

TEMPS = [0.01, 0.03, 0.1, 0.3, 1.0]
SEEDS = [0, 1, 2, 3, 4]


def pick_temperature(device):
    graphs, info = get_validation_set()
    opt = np.asarray(info["opt"], float)
    model = get_model("checkpoints/default_seed0/best.pt", device)
    scores = {}
    for T in TEMPS:
        gaps = []
        for i, g in enumerate(graphs):
            runs = rollout(model, [g] * 4, device, mode="sample", temperature=T,
                           rng=np.random.default_rng(i), batch_size=4)
            best = min(len(redundancy_removal(g, r["solution"], scores=r["q"], protected=r["forced"])) for r in runs)
            gaps.append((best - opt[i]) / opt[i])
        scores[T] = float(np.mean(gaps))
        print(f"  T={T}: validation gap (best of 4, +RR) = {scores[T]:.4f}", flush=True)
    T = min(scores, key=scores.get)
    return T, scores


def main():
    from _runner import run_sharded, tables
    device = default_device()
    T, scores = pick_temperature(device)
    Path("results/e6").mkdir(parents=True, exist_ok=True)
    Path("results/e6/temperature_selection.json").write_text(json.dumps({"chosen": T, "val_gap": scores}, indent=2))
    cks = [f"checkpoints/default_seed{s}/best.pt" for s in SEEDS]
    cfg = {"name": "e6", "sets": ["data/test/e5/*.npz"], "out": "results/e6/raw.csv",
           "methods": [{"type": "ilp", "time_limit": 60, "threads": 8},
                       {"type": "rl", "label": "ours", "checkpoints": cks, "seeds": SEEDS},
                       {"type": "rl", "label": "ours", "checkpoints": cks, "seeds": SEEDS, "samples": 4, "temperature": T},
                       {"type": "rl", "label": "ours", "checkpoints": cks, "seeds": SEEDS, "samples": 16, "temperature": T}]}
    Path("configs/eval_e6.yaml").write_text(
        "# Written by scripts/run_e6.py (temperature chosen on the validation set).\n" + yaml.safe_dump(cfg, sort_keys=False))
    run_sharded("configs/eval_e6.yaml", 3)
    tables("e6")


if __name__ == "__main__":
    main()
