"""Graph input/output.

Readers: plain edge lists (`u v [w]` per line, '#'/'%' comments, arbitrary integer
or string labels), DIMACS `.clq/.col` (`p edge n m`, `e u v`, 1-based) and
Matrix Market `.mtx` (pattern/real, symmetric or general; treated as undirected).
Test sets are stored as `.npz` (edges + weights + JSON spec per graph).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .graph import Graph


def read_edge_list(path, name: str = None) -> Graph:
    labels, edges = {}, []

    def lab(t):
        if t not in labels:
            labels[t] = len(labels)
        return labels[t]

    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line[0] in "#%":
                continue
            parts = line.replace(",", " ").split()
            if len(parts) == 1:          # a lone vertex
                lab(parts[0])
                continue
            edges.append((lab(parts[0]), lab(parts[1])))
    # Relabel so integer labels keep their natural order.
    keys = list(labels)
    try:
        order = sorted(keys, key=lambda k: int(k))
    except ValueError:
        order = keys
    remap = {labels[k]: i for i, k in enumerate(order)}
    e = [(remap[a], remap[b]) for a, b in edges]
    return Graph.from_edges(len(order), e, name=name or Path(path).stem,
                            meta={"source": str(path), "labels": order if len(order) <= 10000 else None})


def read_dimacs(path, name: str = None) -> Graph:
    n, edges = None, []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            p = line.split()
            if not p or p[0] == "c":
                continue
            if p[0] == "p":
                n = int(p[2])
            elif p[0] in ("e", "a"):
                edges.append((int(p[1]) - 1, int(p[2]) - 1))
    if n is None:
        raise ValueError(f"{path}: missing 'p edge n m' line")
    return Graph.from_edges(n, edges, name=name or Path(path).stem, meta={"source": str(path)})


def read_mtx(path, name: str = None) -> Graph:
    from scipy.io import mmread
    from scipy.sparse import coo_matrix
    try:
        A = mmread(str(path), spmatrix=False)
    except TypeError:                     # older SciPy
        A = mmread(str(path))
    A = coo_matrix(A)
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"{path}: adjacency matrix must be square, got {A.shape}")
    return Graph.from_edges(A.shape[0], np.stack([A.row, A.col], 1), name=name or Path(path).stem,
                            meta={"source": str(path)})


def read_graph(path) -> Graph:
    ext = Path(path).suffix.lower()
    if ext in (".clq", ".col", ".dimacs"):
        return read_dimacs(path)
    if ext == ".mtx":
        return read_mtx(path)
    return read_edge_list(path)


def write_edge_list(graph: Graph, path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# n={graph.n} m={graph.num_edges} name={graph.name}\n")
        for u, v in graph.edges().tolist():
            f.write(f"{u} {v}\n")


def save_graphs(graphs, path) -> None:
    """Save a list of Graphs (with their generator specs) to a single .npz."""
    arrs = {}
    metas = []
    for i, g in enumerate(graphs):
        arrs[f"e{i}"] = g.edges().astype(np.int32)
        if g.weights is not None:
            arrs[f"w{i}"] = g.weights
        metas.append({"n": g.n, "name": g.name, "meta": g.meta})
    arrs["index"] = np.array(json.dumps(metas))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrs)


def load_graphs(path) -> list:
    z = np.load(path, allow_pickle=False)
    metas = json.loads(str(z["index"]))
    out = []
    for i, m in enumerate(metas):
        w = z[f"w{i}"] if f"w{i}" in z.files else None
        out.append(Graph.from_edges(m["n"], z[f"e{i}"], weights=w, name=m["name"], meta=m["meta"]))
    return out
