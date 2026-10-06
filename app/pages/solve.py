import streamlit as st

from components import solver
from components.graph_input import GEN_TYPES, SAMPLE_EDGE_LIST, GraphInputError, generate, parse_edge_text, parse_upload
from components.network import MAX_DRAW, solution_html
from components.state import current, graph_pills, isolated_card, set_current
from components.theme import card_title, notice, stat_tiles


def input_card():
    card_title("1 · Graph")
    t_gen, t_up, t_paste = st.tabs(["Generate", "Upload", "Paste"])
    with t_gen:
        kind_label = st.selectbox("Graph type", list(GEN_TYPES), index=3, key="gen_kind")
        kind = GEN_TYPES[kind_label]
        side, n, dens = 10, 100, 0
        if kind == "grid":
            side = st.slider("Side length (grid is side × side)", 2, 60, 10, key="gen_side")
        else:
            n = st.slider("Vertices", 10, 5000, 100, step=10, key="gen_n")
            if kind == "er":
                dens = st.slider("Average degree", 2.0, 20.0, 6.0, 0.5, key="gen_er")
            elif kind == "ba":
                dens = st.slider("Edges per new vertex (m)", 1, 8, 3, key="gen_ba")
            elif kind == "ws":
                dens = st.select_slider("Ring neighbours (k)", [2, 4, 6, 8], 4, key="gen_ws")
        seed = st.number_input("Seed", 0, 10**6, 0, key="gen_seed") if kind in ("er", "ba", "ws") else 0
        if st.button("Use this graph", key="gen_go", width="stretch"):
            lg = generate(kind, int(n), float(dens), int(seed), side=int(side))
            set_current(lg, ("gen", kind, n, dens, seed, side))
    with t_up:
        st.caption("Edge list `.txt` (one `u v` per line, labels may be any tokens), DIMACS `.col`/`.clq`, or Matrix Market `.mtx`.")
        up = st.file_uploader("Graph file", type=["txt", "edges", "el", "col", "clq", "mtx"], label_visibility="collapsed")
        st.download_button("Download a sample edge list", SAMPLE_EDGE_LIST, "sample_petersen.txt", "text/plain",
                           width="stretch")
        if up is not None and st.session_state.get("last_upload") != (up.name, up.size, up.file_id):
            st.session_state.last_upload = (up.name, up.size, up.file_id)
            try:
                lg = parse_upload(up.name, up.getvalue())
                set_current(lg, ("up", up.name, up.size))
            except GraphInputError as ex:
                notice(f"<b>Could not read the file.</b> {ex}", "warn")
    with t_paste:
        txt = st.text_area("Edges, one per line", "0 1\n1 2\n2 3\n3 0\n0 2", height=150,
                           help="Each line `u v`. Lines starting with # are ignored.")
        if st.button("Use pasted edges", width="stretch"):
            try:
                lg = parse_edge_text(txt, "pasted text")
                set_current(lg, ("paste", txt))
            except GraphInputError as ex:
                notice(f"<b>Could not read the edges.</b> {ex}", "warn")


def method_card(lg):
    card_title("2 · Method")
    n = lg.graph.n
    method = st.segmented_control("Method", ["GNN + RL (ours)", "Greedy", "Greedy + RR", "Exact ILP"],
                                  default="GNN + RL (ours)", label_visibility="collapsed", key="method")
    method = method or "GNN + RL (ours)"
    ilp_off = n > solver.ILP_MAX_N
    if method == "Exact ILP" and ilp_off:
        notice(f"Exact ILP is disabled above {solver.ILP_MAX_N} vertices: the problem is NP-hard and the solver could "
               f"run for minutes. Use the GNN or greedy methods for this graph.", "warn")
    opts = {}
    if method == "GNN + RL (ours)":
        models = solver.available_models()
        if not models:
            notice("No trained checkpoint found in <span class='mono'>checkpoints/</span>.", "warn")
        opts["model"] = st.radio("Model", list(models) or ["—"], horizontal=True, key="model_choice")
        opts["rr"] = st.toggle("Redundancy removal", True, help="Drop vertices whose removal keeps S a TDS.")
        k = st.select_slider("Sampled rollouts k", [1, 4, 16], 1,
                             help=f"k = 1 runs the greedy policy. k > 1 samples k rollouts from softmax(Q / {solver.SAMPLE_TEMPERATURE}) "
                                  "and keeps the best (temperature not tuned).")
        opts["samples"] = k
    if method in ("GNN + RL (ours)", "Greedy + RR"):
        ls = st.toggle("Local search", False, help="Time-limited 2-for-1 and swap moves after redundancy removal.")
        opts["ls_time"] = st.slider("Local search time (s)", 0.2, 5.0, 1.0, 0.2) if ls else 0.0
    go = st.button("Find total dominating set", type="primary", width="stretch",
                   disabled=(method == "Exact ILP" and ilp_off),
                   help=f"Exact ILP is limited to {solver.ILP_MAX_N} vertices." if ilp_off and method == "Exact ILP" else None)
    return method, opts, go


def run(lg, method, opts):
    g = lg.graph
    if method == "GNN + RL (ours)":
        return solver.run_rl(g, opts["model"], rr=opts["rr"], ls_time=opts.get("ls_time", 0.0), samples=opts["samples"])
    key = {"Greedy": "greedy", "Greedy + RR": "greedy_rr", "Exact ILP": "ilp"}[method]
    return solver.run_baseline(g, key, ls_time=opts.get("ls_time", 0.0))


def result_view(lg, res):
    g = lg.graph
    opt = solver.optimum_for(lg)
    items = [("|S|", res["size"], res["method"], "teal")]
    if opt:
        val, src = opt
        gap = 100.0 * (res["size"] - val) / val
        items.append(("Optimum", val, src, ""))
        items.append(("Gap", f"{gap:.1f}%", "0% means optimal", "amber" if gap > 0 else "teal"))
    elif res.get("optimal") is not None:
        items.append(("ILP status", "optimal" if res["optimal"] else "time limit",
                      f"lower bound {res['bound']:.1f}", ""))
    items.append(("Runtime", f"{res['runtime']:.3f}s", "wall clock", ""))
    items.append(("Valid", "✓", "checked by is_total_dominating_set", "teal"))
    stat_tiles(items)
    notice("<b>Valid total dominating set:</b> every vertex, including every vertex of S, has a neighbour in S.")
    if "size_raw" in res:
        st.markdown(f'<div class="caption">Construction picked {res["size_raw"]} vertices ({res["forced"]} forced by the '
                    f'degree-1 rule) → {res["size_rr"]} after redundancy removal → {res["size_ls"]} after local search.</div>',
                    unsafe_allow_html=True)
    if g.n <= MAX_DRAW:
        st.iframe(solution_html(lg, res["S"], height=560), height=600)
    else:
        notice(f"The graph has {g.n} vertices; drawing is skipped above {MAX_DRAW} so the page stays responsive. "
               "Download the solution below.", "")
    c1, c2, _ = st.columns([1, 1, 2])
    c1.download_button("solution.json", solver.solution_json(lg, res), "solution.json", "application/json",
                       width="stretch")
    if g.n <= MAX_DRAW:
        if c2.button("Prepare PNG", width="stretch"):
            st.session_state.png = solver.solution_png(lg, res["S"])
        if st.session_state.get("png"):
            c2.download_button("solution.png", st.session_state.png, "solution.png", "image/png", width="stretch")


def render():
    left, right = st.columns([1, 2.5], gap="large")
    with left:
        with st.container(border=True):
            input_card()
        lg = current()
        with st.container(border=True):
            method, opts, go = method_card(lg)
    with right:
        st.markdown('<div class="section-title">Result</div>', unsafe_allow_html=True)
        graph_pills(lg)
        for w in lg.warnings:
            notice(w, "warn")
        if isolated_card(lg):
            return
        if go:
            try:
                with st.spinner("Finding a total dominating set…"):
                    st.session_state.result = run(lg, method, opts)
                    st.session_state.pop("png", None)
            except Exception as ex:
                notice(f"<b>Could not solve:</b> {ex}", "err")
        res = st.session_state.get("result")
        if res is None:
            st.markdown('<div class="caption" style="margin-top:14px;">Choose a method on the left and press '
                        '<b>Find total dominating set</b>. The default graph is a 10×10 grid, whose optimum is 30.</div>',
                        unsafe_allow_html=True)
            if lg.graph.n <= MAX_DRAW:
                st.iframe(solution_html(lg, [], height=520), height=560)
        else:
            result_view(lg, res)
