"""E4: real / benchmark graphs.

1. Download instances yourself and put them in data/benchmarks/:
     - DIMACS graph colouring instances (.col), e.g. from
       https://mat.tepper.cmu.edu/COLOR/instances.html
     - DIMACS clique instances (.clq) from the 2nd DIMACS challenge
     - Network Repository (https://networkrepository.com) .mtx files
     - SNAP (https://snap.stanford.edu/data/) edge lists (.txt / .edges)
2. python scripts/run_e4.py            (converts, then evaluates configs/eval_e4.yaml)

A TDS exists only if there is no isolated vertex. Graphs with isolated vertices are
SKIPPED and listed, unless --drop-isolated is given, in which case isolated vertices
are deleted (and this is recorded in the graph metadata; report it in the paper).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tds.graph import Graph  # noqa: E402
from tds.io import read_graph, save_graphs  # noqa: E402

SRC = Path("data/benchmarks")
DST = Path("data/benchmarks_npz")
EXTS = {".col", ".clq", ".mtx", ".txt", ".edges", ".el", ".dimacs"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop-isolated", action="store_true")
    ap.add_argument("--convert-only", action="store_true")
    a = ap.parse_args()
    files = sorted(p for p in SRC.glob("**/*") if p.suffix.lower() in EXTS)
    if not files:
        print(__doc__)
        print(f"No benchmark files found in {SRC.resolve()}. Nothing to do.")
        return
    DST.mkdir(parents=True, exist_ok=True)
    for p in files:
        g = read_graph(p)
        iso = g.isolated_vertices()
        if len(iso):
            if not a.drop_isolated:
                print(f"SKIP {p.name}: {len(iso)} isolated vertices (no TDS exists); use --drop-isolated")
                continue
            keep = np.setdiff1d(np.arange(g.n), iso)
            remap = -np.ones(g.n, dtype=np.int64)
            remap[keep] = np.arange(len(keep))
            g = Graph.from_edges(len(keep), remap[g.edges()], name=g.name,
                                 meta={"source": str(p), "dropped_isolated": int(len(iso))})
        g.meta.setdefault("family", "benchmark")
        save_graphs([g], DST / f"{p.stem}.npz")
        print(f"converted {p.name}: n={g.n} m={g.num_edges}")
    if a.convert_only:
        return
    from _runner import run_sharded, tables  # noqa: F401
    run_sharded("configs/eval_e4.yaml", 1)
    print("raw results in results/e4/raw.csv")


if __name__ == "__main__":
    main()
