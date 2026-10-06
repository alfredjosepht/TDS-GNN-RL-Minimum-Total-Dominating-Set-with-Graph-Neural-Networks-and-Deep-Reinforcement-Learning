"""E5 ablations. Requires the ablation checkpoints:
    for c in configs/ablation_*.yaml: python -m tds.train --config $c --seed 0
then: python scripts/run_e5.py"""
import argparse

from _runner import run_sharded, tables

ap = argparse.ArgumentParser()
ap.add_argument("--shards", type=int, default=3)
a = ap.parse_args()
run_sharded("configs/eval_e5.yaml", a.shards)
tables("e5")
