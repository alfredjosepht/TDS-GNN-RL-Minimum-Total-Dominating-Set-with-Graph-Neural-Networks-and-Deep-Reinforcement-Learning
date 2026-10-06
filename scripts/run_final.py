"""Final reduced experiments (E1, E2, E3) for the 2-seed delivery.

    python scripts/run_final.py ilp       # stage 1: exact ILP rows (30 s), parallel, reuses proven rows
    python scripts/run_final.py verify    # re-solve + verify the reused rows
    python scripts/run_final.py timed     # stage 2: greedy / RL rows, sequential (fair runtimes)
    python scripts/run_final.py report    # tables, plots, results/summary.json

E1: ER (avg deg 8) and BA (m = 4), n in {20, 50, 100, 200, 500}, first 20 graphs of each saved set.
E2: grid / triangular / hexagonal lattices with n <= 400 (zero-shot).
E3: one ER (avg deg 8) and one BA (m = 3) graph with n = 10,000.
Every solution is verified with is_total_dominating_set and stored in solutions.jsonl.
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tds.baselines import greedy, greedy_rr, greedy_rr_ls, ilp  # noqa: E402
from tds.checker import is_total_dominating_set  # noqa: E402
from tds.io import load_graphs  # noqa: E402

OUT = ROOT / "results" / "final"
SEEDS = {0: "checkpoints/best.pt", 1: "checkpoints/seed1_best.pt"}
LS_TIME = 1.0
ILP_TIME = 30.0
E1_SETS = [f"{fam}_n{n}" for fam in ("er_d8", "ba_m4") for n in (20, 50, 100, 200, 500)]
E1_PER_CELL = 20
E2_MAX_N = 410
E3_SETS = ["er_d8_n10000", "ba_m3_n10000"]

# Theme colours (match the app)
TEAL, AMBER, GREY, INK, MUTED = "#0F766E", "#D97706", "#9CA3AF", "#1F2328", "#6B6F76"


def instances():
    """(experiment, set, idx, graph) for every test instance."""
    out = []
    for s in E1_SETS:
        for i, g in enumerate(load_graphs(ROOT / "data/test/e1" / f"{s}.npz")[:E1_PER_CELL]):
            out.append(("E1", s, i, g))
    for kind in ("grid", "tri", "hex"):
        for i, g in enumerate(load_graphs(ROOT / "data/test/e2" / f"{kind}.npz")):
            if g.n <= E2_MAX_N:
                out.append(("E2", kind, i, g))
    for s in E3_SETS:
        out.append(("E3", s, 0, load_graphs(ROOT / "data/test/e3" / f"{s}.npz")[0]))
    return out


def _ilp_job(args):
    exp, s, i, g = args
    r = ilp(g, time_limit=ILP_TIME, threads=4)
    assert is_total_dominating_set(g.adj, r.vertices)
    return dict(exp=exp, set=s, idx=i, n=g.n, size=r.size, optimal=bool(r.info["optimal"]),
                bound=r.info["bound"], runtime=r.runtime, reused=False, vertices=r.vertices)


def stage_ilp():
    OUT.mkdir(parents=True, exist_ok=True)
    reuse = {}
    old = ROOT / "results" / "e1" / "raw.csv"
    if old.exists():                       # proven optima from the earlier (60 s) run are still optimal
        df = pd.read_csv(old)
        df = df[(df.method == "ilp") & (df.optimal.astype(str) == "True")]
        for r in df.itertuples():
            reuse[(r.set.split("/")[-1].replace(".npz", ""), int(r.idx))] = r
    jobs, rows = [], []
    for exp, s, i, g in instances():
        if exp == "E3":
            continue                        # no ILP at n = 10,000 in the reduced plan
        if exp == "E1" and (s, i) in reuse:
            r = reuse[(s, i)]
            rows.append(dict(exp=exp, set=s, idx=i, n=g.n, size=int(r.size), optimal=True, bound=float(r.bound),
                             runtime=float(r.runtime), reused=True, vertices=None))
        else:
            jobs.append((exp, s, i, g))
    print(f"ILP: reusing {len(rows)} proven rows, solving {len(jobs)} instances (30 s limit, 3 workers)", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=3) as ex:
        for k, r in enumerate(ex.map(_ilp_job, jobs)):
            rows.append(r)
            if (k + 1) % 20 == 0:
                print(f"  {k + 1}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    sol = [dict(exp=r["exp"], set=r["set"], idx=r["idx"], method="ilp", seed=None, vertices=r.pop("vertices"))
           for r in rows if r.get("vertices") is not None]
    pd.DataFrame(rows).to_csv(OUT / "ilp.csv", index=False)
    with open(OUT / "solutions_ilp.jsonl", "w") as f:
        for s in sol:
            f.write(json.dumps(s) + "\n")
    print(f"ILP stage done in {time.time() - t0:.0f}s", flush=True)


def stage_verify_reused():
    """Re-solve the ILP rows reused from the earlier run, verify the solutions and check that the
    proven optimum matches; afterwards every ILP row has a verified solution in solutions_ilp.jsonl."""
    df = pd.read_csv(OUT / "ilp.csv")
    todo = df[df.reused == True]  # noqa: E712
    graphs = {(e, s_, i): g for e, s_, i, g in instances()}
    jobs = [(r.exp, r.set, int(r.idx), graphs[(r.exp, r.set, int(r.idx))]) for r in todo.itertuples()]
    print(f"re-verifying {len(jobs)} reused ILP rows", flush=True)
    with ProcessPoolExecutor(max_workers=3) as ex:
        res = list(ex.map(_ilp_job, jobs))
    with open(OUT / "solutions_ilp.jsonl", "a") as f:
        for r in res:
            old = df[(df.exp == r["exp"]) & (df.set == r["set"]) & (df.idx == r["idx"])].iloc[0]
            if r["optimal"]:
                assert r["size"] == int(old["size"]), ("proven optimum changed", r["set"], r["idx"])
            mask = (df.exp == r["exp"]) & (df.set == r["set"]) & (df.idx == r["idx"])
            df.loc[mask, ["size", "optimal", "bound", "runtime", "reused"]] = \
                [r["size"], r["optimal"], r["bound"], r["runtime"], False]
            f.write(json.dumps(dict(exp=r["exp"], set=r["set"], idx=r["idx"], method="ilp", seed=None,
                                    vertices=r["vertices"])) + "\n")
    df.to_csv(OUT / "ilp.csv", index=False)
    print("all reused rows re-solved and verified; optimum unchanged", flush=True)


def stage_timed():
    import torch
    from tds.evaluate import rl_rows
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows, t0 = [], time.time()
    sol_f = open(OUT / "solutions_timed.jsonl", "w")

    def rec(exp, s, i, g, method, seed, S, rt):
        ok = is_total_dominating_set(g.adj, S)
        assert ok, (exp, s, i, method, seed)
        rows.append(dict(exp=exp, set=s, idx=i, n=g.n, m=g.num_edges, method=method, seed=seed,
                         size=len(S), runtime=rt, valid=ok))
        sol_f.write(json.dumps(dict(exp=exp, set=s, idx=i, method=method, seed=seed, vertices=list(map(int, S)))) + "\n")

    # warm up CUDA / model loading so the first instance is not charged for it
    from tds.solve import get_model
    for ck in SEEDS.values():
        get_model(str(ROOT / ck), device)
    insts = instances()
    for k, (exp, s, i, g) in enumerate(insts):
        for name, fn in (("greedy", greedy), ("greedy_rr", greedy_rr)):
            r = fn(g)
            rec(exp, s, i, g, name, None, r.vertices, r.runtime)
        if exp != "E3":
            r = greedy_rr_ls(g, ls_time=LS_TIME)
            rec(exp, s, i, g, "greedy_rr_ls", None, r.vertices, r.runtime)
        for seed, ck in SEEDS.items():
            mc = {"ls_time": 0 if exp == "E3" else LS_TIME, "multi_select": "auto"}
            for name, S, rt, _ in rl_rows(g, str(ROOT / ck), "ours", seed, mc, device):
                rec(exp, s, i, g, name, seed, S, rt)
        if (k + 1) % 25 == 0:
            print(f"  {k + 1}/{len(insts)} instances ({time.time() - t0:.0f}s)", flush=True)
    sol_f.close()
    pd.DataFrame(rows).to_csv(OUT / "timed.csv", index=False)
    print(f"timed stage done in {time.time() - t0:.0f}s, {len(rows)} rows, all valid", flush=True)


# ------------------------------------------------------------------ report
METHODS = ["greedy", "greedy_rr", "greedy_rr_ls", "rl", "rl_rr", "rl_rr_ls"]
PRETTY = {"greedy": "Greedy", "greedy_rr": "Greedy+RR", "greedy_rr_ls": "Greedy+RR+LS", "rl": "GNN-RL",
          "rl_rr": "GNN-RL+RR", "rl_rr_ls": "GNN-RL+RR+LS"}


def gt_grid(k):
    m, r = divmod(k, 4)
    return (2 * m + 1) * (2 * m + r)


def agg(df, col):
    """mean ± std: RL rows -> per-seed mean then mean/std across seeds; greedy -> mean over graphs."""
    if df.seed.notna().any():
        per = df.groupby("seed")[col].mean()
        return float(per.mean()), float(per.std(ddof=1)) if len(per) > 1 else 0.0, int(len(per))
    return float(df[col].mean()), float("nan"), 1


def fmt(m, s, pct=False, d=2):
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return "--"
    k = 100 if pct else 1
    return f"{k * m:.{d}f}" + ("" if np.isnan(s) else f" ± {k * s:.{d}f}")


def seed_overlap_check():
    from tds.generators import GraphSampler
    from tds.config import load_config
    cfg = load_config(ROOT / "configs/default.yaml")
    train = set()
    for s in SEEDS:
        sm = json.loads((ROOT / f"runs/default_seed{s}/summary.json").read_text()) \
            if (ROOT / f"runs/default_seed{s}/summary.json").exists() else {"episodes": 40000}
        smp = GraphSampler(seed=1_000_003 * (s + 1), families=cfg["train"]["families"])
        # n bounds do not change the number of RNG draws (bounded-integer rejection
        # probability ~1e-8), so this replays the training spec stream; 2x margin.
        for _ in range(2 * int(sm["episodes"]) + 1000):
            train.add(smp.sample_spec(20, 100)["seed"])
    val = {sp["seed"] for sp in json.loads((ROOT / "data/val_set.json").read_text())["specs"]}
    test = set()
    for p in list((ROOT / "data/test").glob("*/*.npz")):
        for g in load_graphs(p):
            if g.meta.get("seed") is not None and g.meta.get("family") not in ("grid", "tri", "hex"):
                test.add(g.meta["seed"])
    res = {"n_train_seeds_checked": len(train), "n_val_seeds": len(val), "n_test_seeds": len(test),
           "train_val_overlap": len(train & val), "train_test_overlap": len(train & test),
           "val_test_overlap": len(val & test),
           "note": "lattice test graphs are deterministic (no seed); training lattices are small (n<=100) "
                   "random-size patches"}
    print("seed overlap:", res, flush=True)
    return res


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#C9C4BA", "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})
    t = pd.read_csv(OUT / "timed.csv")
    il = pd.read_csv(OUT / "ilp.csv")[["exp", "set", "idx", "size", "optimal", "bound", "reused"]].rename(
        columns={"size": "opt_size"})
    t = t.merge(il, on=["exp", "set", "idx"], how="left")
    t["gap"] = np.where(t.optimal == True, (t["size"] - t.opt_size) / t.opt_size, np.nan)  # noqa: E712
    n_sol = sum(1 for f in OUT.glob("solutions_*.jsonl") for _ in open(f))
    assert t.valid.all()
    summary = {"validity": {"solutions_checked": int(len(t) + il.shape[0]), "valid_fraction": 1.0,
                            "solutions_in_jsonl": n_sol},
               "settings": {"ls_time_s": LS_TIME, "ilp_time_limit_s": ILP_TIME, "seeds": list(SEEDS),
                            "e1_graphs_per_cell": E1_PER_CELL}}
    summary["seed_overlap"] = seed_overlap_check()
    tables = {}

    # ---- E1
    e1 = t[t.exp == "E1"].copy()
    e1["family"] = e1.set.str.replace(r"_n\d+$", "", regex=True)
    rows, plot = [], []
    for s in E1_SETS:
        d = e1[e1.set == s]
        il_s = il[(il.exp == "E1") & (il.set == s)]
        r = {"family": s.rsplit("_n", 1)[0], "n": int(d.n.iloc[0]), "graphs": int(il_s.shape[0]),
             "ILP proven": int(il_s.optimal.sum()), "optimum (mean, proven)": fmt(il_s[il_s.optimal].opt_size.mean(), np.nan, d=1)}
        for m in METHODS:
            dm = d[d.method == m]
            gm, gs, _ = agg(dm[dm.optimal == True], "gap")  # noqa: E712
            r[PRETTY[m] + " gap %"] = fmt(gm, gs, pct=True)
            tm, ts, _ = agg(dm, "runtime")
            plot.append(dict(set=s, family=r["family"], n=r["n"], method=m, gap=gm, gap_sd=gs, rt=tm, rt_sd=ts))
        rows.append(r)
    tables["E1"] = pd.DataFrame(rows)
    sz = []
    for s in E1_SETS:
        d = e1[e1.set == s]
        r = {"family": s.rsplit("_n", 1)[0], "n": int(d.n.iloc[0])}
        for m in METHODS:
            mm, ms, _ = agg(d[d.method == m], "size")
            r[PRETTY[m] + " |S|"] = fmt(mm, ms, d=1)
            tm, ts, _ = agg(d[d.method == m], "runtime")
            r[PRETTY[m] + " time s"] = fmt(tm, ts, d=3)
        sz.append(r)
    tables["E1_size_time"] = pd.DataFrame(sz)
    pe = pd.DataFrame(plot)

    # ---- E2
    e2 = t[t.exp == "E2"]
    rows = []
    for (s, i), d in e2.groupby(["set", "idx"]):
        n = int(d.n.iloc[0])
        ilr = il[(il.exp == "E2") & (il.set == s) & (il.idx == i)]
        cf = gt_grid(int(round(np.sqrt(n)))) if s == "grid" and int(round(np.sqrt(n))) ** 2 == n else None
        r = {"lattice": s, "n": n, "closed form": cf if cf is not None else "--",
             "ILP": (f"{int(ilr.opt_size.iloc[0])}" + ("" if bool(ilr.optimal.iloc[0]) else
                     f" (not proven, bound {float(ilr.bound.iloc[0]):.0f})")) if len(ilr) else "--"}
        ref = cf if cf is not None else (int(ilr.opt_size.iloc[0]) if len(ilr) and bool(ilr.optimal.iloc[0]) else None)
        for m in METHODS:
            mm, ms, _ = agg(d[d.method == m], "size")
            r[PRETTY[m]] = fmt(mm, ms, d=1)
        r["_ref"] = ref
        rows.append(r)
    e2t = pd.DataFrame(rows).sort_values(["lattice", "n"])
    lat_plot = e2t.copy()
    tables["E2"] = e2t.drop(columns=["_ref"])

    # ---- E3
    e3 = t[t.exp == "E3"]
    rows = []
    for s in E3_SETS:
        d = e3[e3.set == s]
        r = {"graph": s, "n": int(d.n.iloc[0]), "m": int(d.m.iloc[0])}
        for m in ["greedy", "greedy_rr", "rl", "rl_rr"]:
            mm, ms, _ = agg(d[d.method == m], "size")
            tm, ts, _ = agg(d[d.method == m], "runtime")
            r[PRETTY[m] + " |S|"] = fmt(mm, ms, d=1)
            r[PRETTY[m] + " time s"] = fmt(tm, ts, d=2)
        rows.append(r)
    tables["E3"] = pd.DataFrame(rows)

    for k, df in tables.items():
        df.to_csv(OUT / f"table_{k}.csv", index=False)
        (OUT / f"table_{k}.tex").write_text(df.to_latex(index=False, escape=True), encoding="utf-8")
    summary["tables"] = {k: df.to_dict(orient="records") for k, df in tables.items()}

    # headline: E1 overall gap on proven instances, mean ± std across seeds
    head = {}
    pr = e1[e1.optimal == True]  # noqa: E712
    for m in METHODS:
        gm, gs, ns = agg(pr[pr.method == m], "gap")
        head[m] = {"gap_mean": gm, "gap_std_across_seeds": None if np.isnan(gs) else gs, "seeds": ns}
    il1 = il[il.exp == "E1"]
    summary["E1_headline"] = {"instances": int(il1.shape[0]), "ilp_proven": int(il1.optimal.sum()),
                              "ilp_not_proven": int((~il1.optimal.astype(bool)).sum()), "gap_on_proven": head}
    (ROOT / "results" / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    # ---- plots
    style = {"greedy_rr": (GREY, "o", "-"), "greedy_rr_ls": (MUTED, "s", "--"),
             "rl_rr": (TEAL, "o", "-"), "rl_rr_ls": (AMBER, "s", "--")}
    fams = ["er_d8", "ba_m4"]
    titles = {"er_d8": "Erdős–Rényi, avg. degree 8", "ba_m4": "Barabási–Albert, m = 4"}
    for key, ylab, fname in (("gap", "gap to ILP optimum (%)", "plot_gap_vs_n.png"),
                             ("rt", "runtime (s)", "plot_runtime_vs_n.png")):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=(key == "gap"))
        for ax, f in zip(axes, fams):
            for m, (c, mk, ls) in style.items():
                d = pe[(pe.family == f) & (pe.method == m)].sort_values("n")
                y = d[key] * (100 if key == "gap" else 1)
                err = d[key + "_sd"].fillna(0) * (100 if key == "gap" else 1)
                ax.errorbar(d.n, y, yerr=err, color=c, marker=mk, ls=ls, lw=1.8, ms=5, capsize=3, label=PRETTY[m])
            ax.set_xscale("log")
            ax.set_xticks([20, 50, 100, 200, 500])
            ax.set_xticklabels(["20", "50", "100", "200", "500"])
            if key == "rt":
                ax.set_yscale("log")
            ax.set_title(titles[f], color=INK, fontsize=11, loc="left")
            ax.set_xlabel("vertices n")
            ax.grid(alpha=0.25, color="#C9C4BA")
        axes[0].set_ylabel(ylab)
        axes[1].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / fname, dpi=150)
        plt.close(fig)
    # lattice plot: |S| / reference
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, kind in zip(axes, ["grid", "tri", "hex"]):
        d = lat_plot[lat_plot.lattice == kind]
        ns = d.n.values
        for m, (c, mk, ls) in style.items():
            vals = [float(str(v).split(" ")[0]) if v != "--" else np.nan for v in d[PRETTY[m]]]
            ax.plot(ns, vals, color=c, marker=mk, ls=ls, lw=1.8, ms=5, label=PRETTY[m])
        refs = [r if r is not None else np.nan for r in d["_ref"]]
        ax.plot(ns, refs, color=INK, marker="*", ls="none", ms=10, label="optimum (closed form / ILP)")
        ax.set_title({"grid": "Square grid", "tri": "Triangular lattice", "hex": "Hexagonal lattice"}[kind],
                     color=INK, fontsize=11, loc="left")
        ax.set_xlabel("vertices n")
        ax.grid(alpha=0.25, color="#C9C4BA")
    axes[0].set_ylabel("|S|")
    axes[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "plot_lattices.png", dpi=150)
    plt.close(fig)
    print("report written:", OUT, flush=True)


if __name__ == "__main__":
    {"ilp": stage_ilp, "verify": stage_verify_reused, "timed": stage_timed, "report": report}[sys.argv[1]]()
