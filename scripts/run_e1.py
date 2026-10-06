"""E1: see configs/eval_e1.yaml.  python scripts/run_e1.py [--shards N]"""
import argparse

from _runner import run_sharded, tables

ap = argparse.ArgumentParser()
ap.add_argument("--shards", type=int, default=3)
a = ap.parse_args()
run_sharded("configs/eval_e1.yaml", a.shards)
tables("e1")
