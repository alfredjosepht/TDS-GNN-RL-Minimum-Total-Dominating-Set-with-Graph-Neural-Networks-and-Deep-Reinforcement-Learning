"""Graph input for the app: tolerant parsing (with warnings) and cached generation."""
from dataclasses import dataclass, field
from pathlib import Path
import tempfile

import numpy as np
import streamlit as st

from tds.generators import lattice_dims, make_graph
from tds.graph import Graph

SAMPLE_EDGE_LIST = """# Sample edge list: one edge "u v" per line; lines starting with # are comments.
# This is the Petersen graph (10 vertices, 15 edges); its total domination number is 4.
0 1
1 2
2 3
3 4
4 0
0 5
1 6
2 7
3 8
4 9
5 7
7 9
9 6
6 8
8 5
"""


@dataclass
class LoadedGraph:
    graph: Graph
    labels: list                       # original label of internal vertex i
    source: str
    warnings: list = field(default_factory=list)
    spec: dict = None                  # generator spec (for generated graphs)

    def label(self, v):
        return self.labels[v] if self.labels is not None else v


class GraphInputError(ValueError):
    pass


def parse_edge_text(text: str, source: str = "pasted text") -> LoadedGraph:
    """Parse 'u v' lines. Comments (#, %) and blank lines are skipped; a line with a
    single token declares a vertex (possibly isolated). Labels may be any tokens."""
    if not text or not text.strip():
        raise GraphInputError("The input is empty. Provide at least one edge such as `0 1`.")
    labels, edges, bad, loops = {}, set(), [], 0
    dup = 0

    def lab(tok):
        if tok not in labels:
            labels[tok] = len(labels)
        return labels[tok]

    lines = text.splitlines()
    # DIMACS text ('p edge n m' header, 'c' comments, 'e u v' edges) only when the header is present
    dimacs = any(l.split()[:2] in (["p", "edge"], ["p", "col"]) for l in lines if l.strip())
    for ln, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line[0] in "#%":
            continue
        parts = line.replace(",", " ").replace("\t", " ").split()
        if dimacs:
            if parts[0] in ("c", "p"):
                continue
            if parts[0] == "e":
                parts = parts[1:]
        if len(parts) == 1:
            lab(parts[0])
            continue
        if len(parts) > 3:
            bad.append(ln)
            continue
        u, v = lab(parts[0]), lab(parts[1])
        if u == v:
            loops += 1
            continue
        e = (min(u, v), max(u, v))
        if e in edges:
            dup += 1
        edges.add(e)
    if not labels:
        raise GraphInputError("No vertices were found. Each line should look like `u v`.")
    order = list(labels)
    try:                                   # keep natural numeric order when all labels are integers
        order = sorted(order, key=lambda t: int(t))
    except ValueError:
        pass
    remap = {labels[t]: i for i, t in enumerate(order)}
    E = [(remap[a], remap[b]) for a, b in edges]
    g = Graph.from_edges(len(order), E, name=Path(source).stem)
    warns = []
    if bad:
        warns.append(f"Skipped {len(bad)} malformed line(s) (e.g. line {bad[0]}).")
    if loops:
        warns.append(f"Dropped {loops} self-loop(s): total domination uses open neighbourhoods, so a loop never helps.")
    if dup:
        warns.append(f"Merged {dup} duplicate edge(s).")
    as_int = all(t.lstrip("-").isdigit() for t in order)
    if not as_int or [int(t) for t in order] != list(range(len(order))):
        warns.append("Vertex labels were relabelled internally to 0..n-1; results show your original labels.")
    return LoadedGraph(g, [int(t) if as_int else t for t in order], source, warns)


def parse_upload(name: str, data: bytes) -> LoadedGraph:
    ext = Path(name).suffix.lower()
    if not data:
        raise GraphInputError(f"`{name}` is empty.")
    if ext in (".col", ".clq", ".mtx", ".dimacs"):
        from tds.io import read_dimacs, read_mtx
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
            f.write(data)
        try:
            g = read_mtx(f.name) if ext == ".mtx" else read_dimacs(f.name)
        except Exception as ex:                       # malformed file
            raise GraphInputError(f"Could not read `{name}` as {ext} ({type(ex).__name__}: {ex}).")
        g.name = Path(name).stem
        return LoadedGraph(g, None, name, [])
    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:
        raise GraphInputError(f"`{name}` is not a text file.")
    return parse_edge_text(text, name)


GEN_TYPES = {"Erdős–Rényi": "er", "Barabási–Albert": "ba", "Watts–Strogatz": "ws",
             "Grid": "grid", "Triangular lattice": "tri", "Hexagonal lattice": "hex"}


@st.cache_data(show_spinner=False, max_entries=32)
def generate(kind: str, n: int, density: float, seed: int, side: int = 10) -> LoadedGraph:
    if kind == "er":
        spec = dict(family="er", n=n, avg_deg=float(density), seed=seed)
    elif kind == "ba":
        spec = dict(family="ba", n=n, m=int(density), seed=seed)
    elif kind == "ws":
        spec = dict(family="ws", n=n, k=int(density), p=0.1, seed=seed)
    elif kind == "grid":
        spec = dict(family="grid", rows=side, cols=side)
    else:
        a, b = lattice_dims(kind, n)
        spec = dict(family=kind, rows=a, cols=b)
    g = make_graph(spec)
    return LoadedGraph(g, None, g.name, [], spec)


def known_optimum(lg: LoadedGraph):
    """(value, source) when the optimum is known in closed form, else None."""
    sp = lg.spec or {}
    if sp.get("family") == "grid" and sp.get("rows") == sp.get("cols"):
        k = sp["rows"]
        m, r = divmod(k, 4)
        return (2 * m + 1) * (2 * m + r), f"closed form, {k}×{k} grid"
    return None
