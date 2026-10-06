"""Solver calls for the app. Every result is verified with is_total_dominating_set here."""
import json
import time
from pathlib import Path

import streamlit as st

from tds.checker import is_total_dominating_set

ROOT = Path(__file__).resolve().parents[2]
MODELS = {"Seed 0 (default)": ROOT / "checkpoints" / "best.pt", "Seed 1": ROOT / "checkpoints" / "seed1_best.pt"}
METHOD_LABELS = {"rl": "GNN + RL (ours)", "greedy": "Greedy", "greedy_rr": "Greedy + RR", "ilp": "Exact ILP"}
ILP_MAX_N = 300
SAMPLE_TEMPERATURE = 0.1        # not tuned on a held-out set; stated in the UI


def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    from tds.agent import load_model as _load
    return _load(path, device())


def available_models():
    return {k: v for k, v in MODELS.items() if v.exists()}


def model_info():
    p = ROOT / "checkpoints" / "model_info.json"
    return json.loads(p.read_text()) if p.exists() else None


def _verified(graph, S, method, runtime, **info):
    ok = is_total_dominating_set(graph.adj, S)
    if not ok:                                     # never show an invalid solution
        raise RuntimeError(f"{method} returned a set that is not a total dominating set")
    return {"method": method, "S": sorted(int(v) for v in S), "size": len(set(S)), "runtime": runtime,
            "valid": ok, **info}


def run_rl(graph, model_key, rr=True, ls_time=0.0, samples=1, trace=False):
    from tds.solve import _MODEL_CACHE, solve_rl
    path = str(available_models()[model_key])
    m = load_model(path)
    from pathlib import Path as _P
    _MODEL_CACHE[(str(_P(path).resolve()), str(device()))] = m      # share the cached model with tds.solve
    res = solve_rl(graph, path, rr=rr, ls_time=ls_time, samples=samples, temperature=SAMPLE_TEMPERATURE,
                   device=device(), multi_select="auto", trace=trace and samples == 1)
    info = {k: res.info[k] for k in ("size_raw", "size_rr", "size_ls", "policy_time", "forced", "steps")}
    if trace and "trace" in res.info:
        info["trace"] = res.info["trace"]
    return _verified(graph, res.vertices, METHOD_LABELS["rl"], res.runtime, **info)


def run_baseline(graph, key, ls_time=0.0, ilp_time=30.0):
    from tds.baselines import greedy, greedy_rr, greedy_rr_ls, ilp
    if key == "greedy":
        r = greedy(graph)
    elif key == "greedy_rr":
        r = greedy_rr_ls(graph, ls_time=ls_time) if ls_time > 0 else greedy_rr(graph)
    elif key == "ilp":
        if graph.n > ILP_MAX_N:
            raise ValueError(f"Exact ILP is limited to {ILP_MAX_N} vertices in the app")
        r = ilp(graph, time_limit=ilp_time)
        return _verified(graph, r.vertices, METHOD_LABELS["ilp"], r.runtime, optimal=bool(r.info["optimal"]),
                         bound=float(r.info["bound"]))
    else:
        raise ValueError(key)
    label = METHOD_LABELS[key] + (" + LS" if key == "greedy_rr" and ls_time > 0 else "")
    return _verified(graph, r.vertices, label, r.runtime)


@st.cache_data(show_spinner=False, max_entries=64, hash_funcs={})
def small_optimum(edges_key: tuple, n: int):
    """Proven optimum by ILP for small graphs (n <= 150, 10 s), else None."""
    from tds.baselines import ilp
    from tds.graph import Graph
    g = Graph.from_edges(n, list(edges_key))
    r = ilp(g, time_limit=10.0)
    return r.size if r.info["optimal"] else None


def optimum_for(lg):
    from .graph_input import known_optimum
    k = known_optimum(lg)
    if k:
        return k
    g = lg.graph
    if g.n <= 150 and not len(g.isolated_vertices()):
        v = small_optimum(tuple(map(tuple, g.edges().tolist())), g.n)
        if v is not None:
            return v, "proven optimal by the exact ILP"
    return None


def solution_json(lg, res):
    return json.dumps({"method": res["method"], "size": res["size"], "valid_total_dominating_set": res["valid"],
                       "runtime_s": round(res["runtime"], 6), "n": lg.graph.n, "m": lg.graph.num_edges,
                       "vertices": [lg.label(v) for v in res["S"]]}, indent=2, default=str)


def solution_png(lg, S):
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    from .network import layout
    from .theme import GREY, TEAL
    g = lg.graph
    pos = layout(lg)
    G = g.to_networkx()
    if pos is None:
        pos = nx.spring_layout(G, seed=0)
    else:
        pos = {k: (x, -y) for k, (x, y) in pos.items()}
    Sset = set(S)
    fig, ax = plt.subplots(figsize=(7, 7), dpi=140)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#CFCAC0", width=0.7)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=[TEAL if v in Sset else GREY for v in G.nodes],
                           node_size=[60 if v in Sset else 22 for v in G.nodes], linewidths=0)
    ax.set_title(f"Total dominating set: |S| = {len(Sset)}  (n = {g.n})", fontsize=11, color="#1F2328", loc="left")
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def timed(fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    return out, time.perf_counter() - t0
