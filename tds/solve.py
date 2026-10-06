"""Inference API and CLI.

    from tds import solve
    res = solve(G, method="rl", checkpoint="checkpoints/best.pt", rr=True, ls_time=1.0)

    python -m tds.solve --graph path/to/edges.txt --method rl --out solution.json --plot solution.png

Every returned solution has been verified with the TDS checker (`res.is_valid`).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np

from .graph import Graph
from .postprocess import local_search, redundancy_removal
from .result import SolveResult

METHODS = ("rl", "greedy", "greedy_rr", "greedy_rr_ls", "ilp", "paper_faithful")
DEFAULT_CHECKPOINT = "checkpoints/best.pt"
_MODEL_CACHE: dict = {}


def as_graph(G) -> Graph:
    if isinstance(G, Graph):
        return G
    if isinstance(G, (str, Path)):
        from .io import read_graph
        return read_graph(G)
    try:
        import networkx as nx
        if isinstance(G, nx.Graph):
            return Graph.from_networkx(G)
    except ImportError:
        pass
    if isinstance(G, (list, tuple)):
        return Graph.from_adj_list(G)
    raise TypeError(f"cannot interpret {type(G)} as a graph")


def get_model(checkpoint, device):
    from .agent import load_model
    key = (str(Path(checkpoint).resolve()), str(device))
    if key not in _MODEL_CACHE:
        if not Path(checkpoint).exists():
            raise FileNotFoundError(f"checkpoint {checkpoint} not found; train one with "
                                    "`python -m tds.train --config configs/default.yaml`")
        _MODEL_CACHE[key] = load_model(checkpoint, device)
    return _MODEL_CACHE[key]


def default_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def solve_rl(graph: Graph, checkpoint=DEFAULT_CHECKPOINT, rr=True, ls_time=1.0, samples=1,
             temperature=1.0, seed=0, device=None, multi_select="auto", trace=False,
             env_kwargs=None, method_name="rl") -> SolveResult:
    from .rollout import rollout
    graph.require_tds_exists()
    device = device or default_device()
    model = get_model(checkpoint, device)
    ek = dict(getattr(model, "env_kwargs", {}) or {})
    ek.update(env_kwargs or {})
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    if samples <= 1:
        runs = rollout(model, [graph], device, env_kwargs=ek, multi_select=multi_select, trace=trace)
    else:
        runs = rollout(model, [graph] * samples, device, env_kwargs=ek, mode="sample",
                       temperature=temperature, rng=rng, batch_size=min(samples, 16), trace=trace)
    t_policy = time.perf_counter() - t0
    best = None
    for r in runs:
        S_raw = r["solution"]
        S_rr = redundancy_removal(graph, S_raw, scores=r["q"], protected=r["forced"]) if rr else S_raw
        key = (float(graph.w[S_rr].sum()), float(graph.w[S_raw].sum()))
        if best is None or key < best[0]:
            best = (key, S_raw, S_rr, r)
    _, S_raw, S_rr, r = best
    t_rr = time.perf_counter() - t0 - t_policy
    S = S_rr
    if ls_time and ls_time > 0:
        S = local_search(graph, S_rr, time_limit=ls_time, seed=seed)
    info = dict(size_raw=len(S_raw), size_rr=len(S_rr), size_ls=len(S), policy_time=t_policy,
                rr_time=t_rr, samples=samples, checkpoint=str(checkpoint), steps=r["steps"],
                forced=len(r["forced"]), rr=rr, ls_time=ls_time, order=r["order"])
    if trace:
        info["trace"] = r["trace"]
    name = method_name + ("_rr" if rr else "") + ("_ls" if ls_time else "") + (f"_k{samples}" if samples > 1 else "")
    return SolveResult.build(graph, S, time.perf_counter() - t0, name, **info)


def solve(G, method: str = "rl", checkpoint=DEFAULT_CHECKPOINT, rr: bool = True, ls_time: float = 1.0,
          samples: int = 1, temperature: float = 1.0, seed: int = 0, device=None,
          ilp_time: float = 60.0, multi_select="auto", trace: bool = False) -> SolveResult:
    """Solve minimum TDS on G (Graph, networkx graph, adjacency list, or file path)."""
    graph = as_graph(G)
    graph.require_tds_exists()
    from . import baselines as B
    if method == "rl":
        return solve_rl(graph, checkpoint, rr, ls_time, samples, temperature, seed, device, multi_select, trace)
    if method == "greedy":
        return B.greedy(graph)
    if method == "greedy_rr":
        return B.greedy_rr(graph)
    if method == "greedy_rr_ls":
        return B.greedy_rr_ls(graph, ls_time=ls_time, seed=seed)
    if method == "ilp":
        return B.ilp(graph, time_limit=ilp_time)
    if method == "paper_faithful":
        from .baselines.paper_faithful import paper_faithful
        return paper_faithful(graph, checkpoint, device=device)
    raise ValueError(f"unknown method {method!r}; choose from {METHODS}")


def plot_solution(graph: Graph, S, path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    G = graph.to_networkx()
    if graph.n > 3000:
        raise ValueError("graph too large to draw (> 3000 vertices)")
    pos = nx.kamada_kawai_layout(G) if graph.n <= 300 else nx.spring_layout(G, seed=0)
    Sset = set(S)
    colors = ["#d62728" if v in Sset else "#c7d7e8" for v in G.nodes]
    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.35, width=0.8)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=max(10, 3000 // max(graph.n, 1)),
                           edgecolors="k", linewidths=0.3)
    ax.set_title(title or f"TDS |S|={len(S)} (red), n={graph.n}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Minimum total dominating set solver")
    ap.add_argument("--graph", required=True, help="edge list / .col / .clq / .mtx file")
    ap.add_argument("--method", default="rl", choices=METHODS)
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--no-rr", action="store_true")
    ap.add_argument("--ls-time", type=float, default=1.0)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--ilp-time", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="write solution JSON here")
    ap.add_argument("--plot", default=None, help="write a PNG drawing here")
    a = ap.parse_args()
    from .io import read_graph
    g = read_graph(a.graph)
    res = solve(g, a.method, a.checkpoint, rr=not a.no_rr, ls_time=a.ls_time, samples=a.samples,
                temperature=a.temperature, seed=a.seed, device=a.device, ilp_time=a.ilp_time)
    labels = g.meta.get("labels")
    d = res.to_dict()
    d["info"].pop("order", None)
    d.update(graph=a.graph, n=g.n, m=g.num_edges)
    if labels:
        d["vertex_labels"] = [labels[v] for v in res.vertices]
    print(f"{res.method}: |S| = {res.size}, valid = {res.is_valid}, time = {res.runtime:.3f}s")
    if a.out:
        Path(a.out).write_text(json.dumps(d, indent=2))
        print("wrote", a.out)
    if a.plot:
        plot_solution(g, res.vertices, a.plot, title=f"{res.method}: |S|={res.size}")
        print("wrote", a.plot)


class _CallableModule(types.ModuleType):
    """Lets `from tds import solve` (which yields this module) be called like the function."""

    def __call__(self, *args, **kwargs):
        return solve(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableModule

if __name__ == "__main__":
    main()
