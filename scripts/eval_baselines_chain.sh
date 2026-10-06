#!/bin/sh
# Non-RL rows (greedy variants + ILP) for E2, E5 and E3, run after E1's baselines, at low priority.
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
until grep -q "child exited" results/e1_baselines.log; do sleep 60; done
for e in e2 e5 e3; do
  $PY scripts/guarded_run.py --low-priority --min-free-gb 1.0 -- python -m tds.evaluate --config configs/eval_$e.yaml --methods greedy,greedy_rr,greedy_rr_ls,ilp >> results/${e}_baselines.log 2>&1
done
