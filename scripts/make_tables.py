"""Aggregate raw evaluation CSVs into CSV + LaTeX tables and plots.

    python scripts/make_tables.py e1 [e2 e3 e5 e6]

Conventions
- size: mean ± std over the graphs of a cell. For RL methods with several training
  seeds, the per-seed cell mean is computed first and the table reports
  mean ± std ACROSS SEEDS (std over graphs is in the *_full.csv files).
- gap: mean of (|S| - ref) / ref, ref = ILP solution value. If the ILP did not
  prove optimality on every graph of a cell, ref is the ILP's best incumbent and the
  cell is marked with † (gap is then relative to best known, not to the optimum);
  the ILP lower bound is reported in the ILP columns.
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RES = Path(os.environ.get("TDS_RESULTS", "results"))
METHOD_ORDER = ["greedy", "greedy_rr", "greedy_rr_ls", "rl", "rl_rr", "rl_rr_ls",
                "rl_k4", "rl_rr_k4", "rl_k16", "rl_rr_k16", "ilp"]
PRETTY = {"greedy": "Greedy", "greedy_rr": "Greedy+RR", "greedy_rr_ls": "Greedy+RR+LS",
          "rl": "Ours", "rl_rr": "Ours+RR", "rl_rr_ls": "Ours+RR+LS", "ilp": "ILP",
          "rl_k4": "Ours k=4", "rl_rr_k4": "Ours+RR k=4", "rl_k16": "Ours k=16", "rl_rr_k16": "Ours+RR k=16"}


def load(exp):
    files = sorted((RES / exp).glob("raw*.csv"))
    if not files:
        raise FileNotFoundError(f"no results/{exp}/raw*.csv; run the evaluation first")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(subset=["set", "idx", "method", "label", "seed"], keep="last")
    assert df["valid"].astype(bool).all(), "found a row with an invalid solution"
    df["cell"] = df["set"].str.replace(r"^.*/", "", regex=True).str.replace(".npz", "", regex=False)
    cfg_path = Path("configs") / f"eval_{exp}.yaml"
    if cfg_path.exists():                     # apply the same per-set graph cap as the evaluation
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text())
        large, lim = cfg.get("max_graphs_large"), cfg.get("max_graphs_per_set")
        if lim:
            df = df[df.idx < lim]
        if large:
            df = df[~((df.n >= large["min_n"]) & (df.idx >= large["count"]))]
    return df


def attach_ref(df):
    ilp = df[df.method == "ilp"][["set", "idx", "size", "optimal", "bound"]].rename(
        columns={"size": "ref", "optimal": "ref_opt", "bound": "ref_bound"})
    out = df.merge(ilp, on=["set", "idx"], how="left")
    out["gap"] = (out["size"] - out["ref"]) / out["ref"]
    return out


def seed_aggregate(df, keys, value):
    """mean/std across seeds of per-seed means (std over graphs if single seed)."""
    per_seed = df.groupby(keys + ["seed"])[value].mean().reset_index()
    nseeds = per_seed.groupby(keys)["seed"].nunique()
    agg = per_seed.groupby(keys)[value].agg(["mean", "std"]).reset_index()
    over_graphs = df.groupby(keys)[value].std().rename("std_graphs").reset_index()
    agg = agg.merge(over_graphs, on=keys).merge(nseeds.rename("nseeds").reset_index(), on=keys)
    agg["std"] = np.where(agg["nseeds"] > 1, agg["std"], agg["std_graphs"])
    return agg


def fmt(m, s, digits=1, pct=False):
    if pd.isna(m):
        return "--"
    if pct:
        return f"{100 * m:.{digits}f} ± {100 * (0 if pd.isna(s) else s):.{digits}f}"
    return f"{m:.{digits}f} ± {0 if pd.isna(s) else s:.{digits}f}"


def write_tables(table: pd.DataFrame, name: str, caption: str):
    out = RES / name
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out.with_suffix(".csv"), index=False)
    latex = table.to_latex(index=False, escape=True, caption=caption, label=f"tab:{out.stem}")
    out.with_suffix(".tex").write_text(latex, encoding="utf-8")
    print(f"wrote {out}.csv/.tex")


def cell_table(df, label_filter=None, value="size", pct=False, digits=1):
    d = df if label_filter is None else df[df.label.isin(label_filter) | ~df.method.str.startswith("rl")]
    agg = seed_aggregate(d, ["cell", "method"], value)
    proven = df[df.method == "ilp"].groupby("cell")["optimal"].apply(lambda s: s.astype(str).eq("True").mean())
    rows = []
    for cell in sorted(agg.cell.unique(), key=_cell_sort):
        r = {"cell": cell, "n": int(df[df.cell == cell].n.mean()),
             "ILP proven": f"{100 * proven.get(cell, np.nan):.0f}%"}
        for m in METHOD_ORDER:
            a = agg[(agg.cell == cell) & (agg.method == m)]
            if len(a):
                txt = fmt(a["mean"].iloc[0], a["std"].iloc[0], digits, pct)
                if value == "gap" and proven.get(cell, 1) < 1:
                    txt += " †"
                r[PRETTY[m]] = txt
        rows.append(r)
    return pd.DataFrame(rows), agg


def _cell_sort(c):
    import re
    m = re.match(r"(.*)_n(\d+)$", c)
    return (m.group(1), int(m.group(2))) if m else (c, 0)


# --------------------------------------------------------------------- E1
def e1():
    df = attach_ref(load("e1"))
    t_size, _ = cell_table(df)
    write_tables(t_size, "e1/table_size", "E1: |S| (mean ± std). RL: mean ± std across 5 training seeds.")
    t_gap, agg_gap = cell_table(df, value="gap", pct=True, digits=2)
    write_tables(t_gap, "e1/table_gap", "E1: gap to ILP in % († = ILP not proven optimal on all graphs: gap to best known).")
    t_time, agg_t = cell_table(df, value="runtime", digits=3)
    write_tables(t_time, "e1/table_time", "E1: runtime in seconds.")
    # plot gap vs n per family
    agg_gap["n"] = pd.to_numeric(agg_gap.cell.str.extract(r"_n(\d+)$")[0], errors="coerce")
    agg_gap = agg_gap.dropna(subset=["n"])
    agg_gap["family"] = agg_gap.cell.str.replace(r"_n\d+$", "", regex=True)
    fams = sorted(agg_gap.family.unique())
    fig, axes = plt.subplots(1, len(fams), figsize=(3.2 * len(fams), 3.2), sharey=True)
    for ax, f in zip(np.atleast_1d(axes), fams):
        for m in ["greedy", "greedy_rr", "greedy_rr_ls", "rl_rr", "rl_rr_ls"]:
            a = agg_gap[(agg_gap.family == f) & (agg_gap.method == m)].sort_values("n")
            if len(a):
                ax.errorbar(a.n, 100 * a["mean"], yerr=100 * a["std"].fillna(0), marker="o", ms=3, label=PRETTY[m], capsize=2)
        ax.set_xscale("log")
        ax.set_title(f)
        ax.set_xlabel("n")
        ax.grid(alpha=.3)
    np.atleast_1d(axes)[0].set_ylabel("gap to ILP (%)")
    np.atleast_1d(axes)[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RES / "e1" / "gap_vs_n.png", dpi=130)
    print("wrote results/e1/gap_vs_n.png")


# --------------------------------------------------------------------- E2
def gt_grid(k):
    m, r = divmod(k, 4)
    return (2 * m + 1) * (2 * m + r)


def e2():
    df = attach_ref(load("e2"))
    rows = []
    for (cell, idx), g in df.groupby(["cell", "idx"]):
        name = g.name.iloc[0]
        r = {"lattice": cell, "graph": name, "n": int(g.n.iloc[0])}
        if cell == "grid":
            k = int(round(np.sqrt(g.n.iloc[0])))
            r["closed form (A302488)"] = gt_grid(k)
        il = g[g.method == "ilp"]
        if len(il):
            r["ILP"] = f"{int(il['size'].iloc[0])}" + ("" if str(il.optimal.iloc[0]) == "True" else
                                                       f" (bound {float(il.bound.iloc[0]):.0f})")
        for m in ["greedy", "greedy_rr", "greedy_rr_ls", "rl", "rl_rr", "rl_rr_ls"]:
            gm = g[g.method == m]
            if len(gm):
                r[PRETTY[m]] = f"{gm['size'].mean():.1f}" + (f" ± {gm['size'].std():.1f}" if len(gm) > 1 else "")
        rows.append(r)
    t = pd.DataFrame(rows).sort_values(["lattice", "n"])
    write_tables(t, "e2/table", "E2: lattices (zero-shot). RL: mean ± std across seeds.")


# --------------------------------------------------------------------- E3
def e3():
    df = attach_ref(load("e3"))
    t, _ = cell_table(df, digits=1)
    write_tables(t, "e3/table_size", "E3: |S| at scale. ILP limited to 600 s (incumbent; † = not proven).")
    tt, _ = cell_table(df, value="runtime", digits=2)
    write_tables(tt, "e3/table_time", "E3: runtime (s).")


# --------------------------------------------------------------------- E5
def e5():
    df = attach_ref(load("e5"))
    rl = df[df.method.str.startswith("rl")]
    rows = []
    for label in rl.label.unique():
        for cell in sorted(df.cell.unique()):
            r = {"model": label, "set": cell}
            for m in ["rl", "rl_rr", "rl_rr_ls"]:
                d = rl[(rl.label == label) & (rl.cell == cell) & (rl.method == m)]
                if len(d):
                    a = seed_aggregate(d, ["cell"], "gap")
                    r[{"rl": "raw", "rl_rr": "+RR", "rl_rr_ls": "+RR+LS"}[m]] = fmt(a["mean"].iloc[0], a["std"].iloc[0], 2, True)
            rows.append(r)
    for m in ["greedy_rr", "greedy_rr_ls"]:
        for cell in sorted(df.cell.unique()):
            d = df[(df.method == m) & (df.cell == cell)]
            if len(d):
                rows.append({"model": PRETTY[m], "set": cell, "raw": fmt(d.gap.mean(), d.gap.std(), 2, True)})
    write_tables(pd.DataFrame(rows), "e5/table", "E5: ablations, gap to ILP (%) on held-out mixed sets.")


# --------------------------------------------------------------------- E6
def e6():
    df = attach_ref(load("e6"))
    rows = []
    for (cell, m), d in df[df.method != "ilp"].groupby(["cell", "method"]):
        a = seed_aggregate(d, ["cell"], "gap")
        rt = d.runtime.mean()
        rows.append({"set": cell, "variant": PRETTY.get(m, m), "gap %": fmt(a["mean"].iloc[0], a["std"].iloc[0], 2, True),
                     "time (s)": f"{rt:.3f}"})
    write_tables(pd.DataFrame(rows), "e6/table", "E6: greedy rollout vs best of k sampled rollouts.")


if __name__ == "__main__":
    for name in sys.argv[1:] or ["e1", "e2", "e3", "e5", "e6"]:
        try:
            globals()[name]()
        except FileNotFoundError as ex:
            print(f"{name}: {ex}")
