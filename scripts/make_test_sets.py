"""Generate the fixed test sets for E1, E2, E3, E5/E6 into data/test/.

Seeds are derived from a base (5_000_000) that no training run (seeds 0..9 ->
sampler seed 1_000_003*(s+1)) or the validation set (777_000_777) uses.

    python scripts/make_test_sets.py [--only e1,e2,e3,e5]
"""
import argparse
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tds.generators import GraphSampler, lattice_dims, make_graph  # noqa: E402
from tds.io import save_graphs  # noqa: E402

OUT = Path("data/test")
BASE = 5_000_000

E1_FAMILIES = {
    "er_d4": dict(family="er", avg_deg=4), "er_d8": dict(family="er", avg_deg=8),
    "er_d16": dict(family="er", avg_deg=16), "ba_m2": dict(family="ba", m=2),
    "ba_m4": dict(family="ba", m=4), "ba_m8": dict(family="ba", m=8),
    "ws_k6": dict(family="ws", k=6, p=0.1),
}
E1_SIZES = [20, 50, 100, 200, 500, 1000]
E1_PER_CELL = 50

E2_GRID_K = [5, 8, 10, 12, 15, 20, 25, 30, 40, 50]          # k x k grids, n = 25 .. 2500
E2_TARGETS = [25, 64, 100, 144, 225, 400, 625, 900, 1600, 2500]

E3 = {"er_d8": dict(family="er", avg_deg=8), "ba_m3": dict(family="ba", m=3)}
E3_SIZES = [10_000, 50_000, 100_000]
E3_PER_CELL = 3


def seed_for(*parts) -> int:
    return BASE + zlib.crc32("|".join(map(str, parts)).encode()) % 1_000_000_000


def e1():
    for fam, params in E1_FAMILIES.items():
        for n in E1_SIZES:
            gs = [make_graph({**params, "n": n, "seed": seed_for("e1", fam, n, i)}) for i in range(E1_PER_CELL)]
            save_graphs(gs, OUT / "e1" / f"{fam}_n{n}.npz")
            print("e1", fam, n, len(gs))


def e2():
    gs = [make_graph(dict(family="grid", rows=k, cols=k)) for k in E2_GRID_K]
    save_graphs(gs, OUT / "e2" / "grid.npz")
    for kind in ("tri", "hex"):
        gs = []
        for t in E2_TARGETS:
            a, b = lattice_dims(kind, t)
            gs.append(make_graph(dict(family=kind, rows=a, cols=b)))
        save_graphs(gs, OUT / "e2" / f"{kind}.npz")
    print("e2 done")


def e3():
    for fam, params in E3.items():
        for n in E3_SIZES:
            gs = [make_graph({**params, "n": n, "seed": seed_for("e3", fam, n, i)}) for i in range(E3_PER_CELL)]
            save_graphs(gs, OUT / "e3" / f"{fam}_n{n}.npz")
            print("e3", fam, n)


def e5():
    """Mixed-family held-out set for ablations / inference variants (same distribution
    as validation but disjoint seed), plus a larger-n generalisation set."""
    s = GraphSampler(seed_for("e5", "mixed"))
    save_graphs([s.sample(50, 100) for _ in range(200)], OUT / "e5" / "mixed_50_100.npz")
    s = GraphSampler(seed_for("e5", "large"))
    save_graphs([s.sample(200, 500) for _ in range(100)], OUT / "e5" / "mixed_200_500.npz")
    print("e5 done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="e1,e2,e3,e5")
    sel = ap.parse_args().only.split(",")
    for name in sel:
        globals()[name]()
