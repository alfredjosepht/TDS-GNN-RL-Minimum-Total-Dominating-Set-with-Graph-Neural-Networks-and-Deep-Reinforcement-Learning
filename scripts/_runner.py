"""Run one evaluation config in parallel shards, then build its tables."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run_sharded(config: str, shards: int, extra=()):
    procs = [subprocess.Popen([PY, "-m", "tds.evaluate", "--config", config, "--shard", str(i),
                               "--num-shards", str(shards), *extra], cwd=ROOT)
             for i in range(shards)]
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"evaluation failed: exit codes {codes}")


def tables(*exps):
    subprocess.check_call([PY, "scripts/make_tables.py", *exps], cwd=ROOT)
