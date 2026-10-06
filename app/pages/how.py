import streamlit as st

from components.theme import table

PIPE = [
    ("01", "Graph", "edge list, DIMACS, Matrix Market, or generated"),
    ("02", "Checks & reduction", "reject isolated vertices; the neighbour of every leaf is forced into S"),
    ("03", "GNN encoder", "3-layer message-passing network on sparse edges, hidden size 64"),
    ("04", "Q-values", "graph summary (mean ‖ max) joined with each vertex embedding"),
    ("05", "Pick a vertex", "highest Q among legal actions: not in S and gain > 0"),
    ("06", "Post-process", "redundancy removal, then optional local search"),
    ("07", "Verified TDS", "is_total_dominating_set on every answer"),
]

ROWS = [
    {"Aspect": "Effect of picking v", "Base paper (dominating set)": "covers N[v] = v and its neighbours",
     "Our system (total dominating set)": "covers only N(v); v itself still needs a chosen neighbour"},
    {"Aspect": "Action rule", "Base paper (dominating set)": "after picking v, v and all of N(v) are blocked",
     "Our system (total dominating set)": "blocking rule removed; legal = not in S and gain(v) > 0"},
    {"Aspect": "State", "Base paper (dominating set)": "one bit per vertex (dominated or not)",
     "Our system (total dominating set)": "separate selected / covered channels + 8 more features"},
    {"Aspect": "Reward", "Base paper (dominating set)": "change in the number of dominated vertices",
     "Our system (total dominating set)": "−1 per chosen vertex, γ = 1, so the return is −|S|"},
    {"Aspect": "Preprocessing", "Base paper (dominating set)": "none",
     "Our system (total dominating set)": "degree-1 rule: the neighbour of a leaf is forced"},
    {"Aspect": "Post-processing", "Base paper (dominating set)": "none",
     "Our system (total dominating set)": "redundancy removal + time-limited local search"},
    {"Aspect": "Message passing", "Base paper (dominating set)": "dense n × n matrices",
     "Our system (total dominating set)": "sparse edge lists (10⁴–10⁵ vertices fit)"},
    {"Aspect": "Graph summary", "Base paper (dominating set)": "sum over vertices",
     "Our system (total dominating set)": "mean ‖ max (does not grow with n)"},
]


def render():
    st.markdown('<div class="section-title" style="font-size:1.6rem;">How it works</div>', unsafe_allow_html=True)
    st.markdown("""<p>A set S of vertices is a <b>total dominating set</b> when every vertex, including the vertices of S
themselves, has at least one neighbour in S. Finding the smallest one is NP-hard. The agent builds S one vertex at a
time; a graph neural network scores every candidate, and the score was learned with Double DQN by rewarding −1 for every
vertex added, so a higher return means a smaller set.</p>""", unsafe_allow_html=True)
    cells = []
    for i, (n, t, s) in enumerate(PIPE):
        cls = "step loop" if n == "05" else "step"
        cells.append(f'<div class="{cls}"><div class="n">{n}</div><div class="t">{t}</div><div class="s">{s}</div></div>')
        if i < len(PIPE) - 1:
            cells.append('<div class="arrow">→</div>')
    st.markdown(f'<div class="pipe">{"".join(cells)}</div>', unsafe_allow_html=True)
    st.markdown('<div class="caption">Steps 04–05 repeat (dashed) until every vertex has a chosen neighbour.</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="section-title">Base paper vs. our system</div>', unsafe_allow_html=True)
    table(ROWS, ["Aspect", "Base paper (dominating set)", "Our system (total dominating set)"])

    st.markdown('<div class="section-title">Why the gain &gt; 0 mask never loses the optimum</div>', unsafe_allow_html=True)
    st.markdown("""<p>Take any minimum total dominating set S*. Because it is minimal, every v in S* has a <i>private
neighbour</i> p: a vertex whose only neighbour in S* is v (otherwise v could be dropped). Add the forced vertices first
(they belong to every total dominating set), then the rest of S* in any order. When v is added, its private neighbour p
is still uncovered, because only v can cover it, so gain(v) ≥ 1 and v is a legal action. Every minimum set is therefore
reachable under the mask, and with reward −1 per step the optimal policy reaches the optimum. The base paper's rule,
by contrast, keeps chosen vertices pairwise non-adjacent, so it can never produce a total dominating set.</p>""",
                unsafe_allow_html=True)

    st.markdown('<div class="section-title">Reference</div>', unsafe_allow_html=True)
    st.markdown("""<p class="caption">Base paper: M. Chen, S. Liu, W. He (2024). <i>Learn to solve dominating set problem
with GNN and reinforcement learning.</i> Applied Mathematics and Computation 474:128717.</p>""", unsafe_allow_html=True)
