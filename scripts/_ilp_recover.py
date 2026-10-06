"""Re-solve (60 s) E1 instances whose optimality was proven by the earlier 60 s run but not at 30 s."""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import run_final as rf  # noqa: E402
from tds.baselines import ilp  # noqa: E402
from tds.checker import is_total_dominating_set  # noqa: E402


def job(a):
    exp, s, i, g = a
    r = ilp(g, time_limit=60.0, threads=4)
    assert is_total_dominating_set(g.adj, r.vertices)
    return exp, s, i, r.size, bool(r.info["optimal"]), float(r.info["bound"]), r.runtime, r.vertices


if __name__ == "__main__":
    df = pd.read_csv(rf.OUT / "ilp.csv")
    if "time_limit" not in df:
        df["time_limit"] = rf.ILP_TIME
    old = pd.read_csv(ROOT / "results/e1/raw.csv")
    old = old[(old.method == "ilp") & (old.optimal.astype(str) == "True")]
    proven60 = {(r.set.split("/")[-1].replace(".npz", ""), int(r.idx)): int(r.size) for r in old.itertuples()}
    G = {(e, s, i): g for e, s, i, g in rf.instances()}
    todo = [(r.exp, r.set, int(r.idx), G[(r.exp, r.set, int(r.idx))]) for r in df.itertuples()
            if r.exp == "E1" and not r.optimal and (r.set, int(r.idx)) in proven60]
    print("re-solving", len(todo), "instances at 60 s", flush=True)
    with ProcessPoolExecutor(max_workers=3) as ex, open(rf.OUT / "solutions_ilp.jsonl", "a") as f:
        for exp, s, i, size, opt, bound, rt, S in ex.map(job, todo):
            if opt:
                assert size == proven60[(s, i)]
            m = (df.exp == exp) & (df.set == s) & (df.idx == i)
            df.loc[m, ["size", "optimal", "bound", "runtime", "time_limit"]] = [size, opt, bound, rt, 60.0]
            f.write(json.dumps(dict(exp=exp, set=s, idx=i, method="ilp", seed=None, vertices=S, time_limit=60)) + "\n")
    df.to_csv(rf.OUT / "ilp.csv", index=False)
    print("proven now:", int(df[df.exp == "E1"].optimal.sum()), "/", int((df.exp == "E1").sum()), flush=True)
