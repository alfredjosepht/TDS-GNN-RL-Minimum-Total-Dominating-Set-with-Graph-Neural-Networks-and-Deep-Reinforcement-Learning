"""E3: see configs/eval_e3.yaml.  python scripts/run_e3.py [--shards N]"""
import argparse

from _runner import run_sharded, tables

ap = argparse.ArgumentParser()
ap.add_argument("--shards", type=int, default=1)
a = ap.parse_args()
run_sharded("configs/eval_e3.yaml", a.shards)
tables("e3")
