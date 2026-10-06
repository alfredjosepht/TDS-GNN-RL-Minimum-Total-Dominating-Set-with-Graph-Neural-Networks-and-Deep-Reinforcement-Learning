import json
from pathlib import Path

import streamlit as st

from components import solver
from components.theme import card_title, notice, stat_tiles, table

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "results" / "final"


def pending(what):
    notice(f"<b>Experiment pending.</b> {what} has not been produced yet; it appears here automatically once "
           f"<span class='mono'>scripts/run_final.py</span> has run.", "warn")


def model_card():
    info = solver.model_info()
    st.markdown('<div class="section-title">Model card</div>', unsafe_allow_html=True)
    if not info:
        pending("checkpoints/model_info.json")
        return
    seeds = info["seeds"]
    tiles = []
    for s in seeds:
        tiles.append((f"Seed {s['seed']} · val gap", f"{100 * s['best_val_gap_rr']:.2f}%",
                      f"after RR, step {s['best_step']:,} of {s['train_steps']:,}", "teal"))
    tiles.append(("Greedy + RR · val gap", f"{100 * seeds[0]['val_gap_greedy_rr']:.2f}%", "same 200 validation graphs", ""))
    tiles.append(("Parameters", f"{info['parameters']:,}", f"{info['encoder'].upper()}, {info['layers']} layers, "
                                                          f"hidden {info['hidden']}", ""))
    stat_tiles(tiles)
    st.markdown(f'<div class="caption">{info["validation_note"]}</div>', unsafe_allow_html=True)


def render():
    st.markdown('<div class="section-title" style="font-size:1.6rem;">Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="caption">Every number on this page is read from files written by runs in this repository '
                '(<span class="mono">checkpoints/model_info.json</span>, <span class="mono">results/summary.json</span>). '
                'Every reported solution was verified with <span class="mono">is_total_dominating_set</span>.</div>',
                unsafe_allow_html=True)
    model_card()
    sp = ROOT / "results" / "summary.json"
    if not sp.exists():
        st.markdown('<div class="section-title">Experiments</div>', unsafe_allow_html=True)
        pending("results/summary.json")
        return
    s = json.loads(sp.read_text())
    v = s["validity"]
    h = s["E1_headline"]
    ov = s.get("seed_overlap", {})
    stat_tiles([("Solutions checked", f"{v['solutions_checked']:,}", "all methods, all experiments", ""),
                ("Valid", f"{100 * v['valid_fraction']:.0f}%", "is_total_dominating_set", "teal"),
                ("E1 ILP proven", f"{h['ilp_proven']} / {h['instances']}", f"{h['ilp_not_proven']} not proven (excluded from gaps)", ""),
                ("Seed overlap", f"{ov.get('train_val_overlap', '?')}/{ov.get('train_test_overlap', '?')}/{ov.get('val_test_overlap', '?')}",
                 "train–val / train–test / val–test", "")])
    tabs = st.tabs(["E1 · random graphs", "E2 · lattices", "E3 · scale"])
    with tabs[0]:
        rows = s["tables"]["E1"]
        cols = list(rows[0].keys())
        table(rows, cols, numeric=[c for c in cols if c not in ("family",)])
        st.markdown(f'<div class="caption">Gap to the ILP optimum in %, computed only on instances where the ILP '
                    f'(limit {s["settings"]["ilp_time_limit_s"]:.0f} s) proved optimality. GNN rows: mean ± std across '
                    f'the {len(s["settings"]["seeds"])} training seeds. Local search: {s["settings"]["ls_time_s"]:.0f} s '
                    f'for both greedy and GNN.</div>', unsafe_allow_html=True)
        for f in ("plot_gap_vs_n.png", "plot_runtime_vs_n.png"):
            if (FINAL / f).exists():
                st.image(str(FINAL / f), width="stretch")
        rows = s["tables"]["E1_size_time"]
        with st.expander("Solution sizes and runtimes"):
            table(rows, list(rows[0].keys()), numeric=[c for c in rows[0] if c != "family"])
    with tabs[1]:
        rows = s["tables"]["E2"]
        table(rows, list(rows[0].keys()), numeric=[c for c in rows[0] if c != "lattice"])
        st.markdown('<div class="caption">Zero-shot: the model never saw lattices this large in training. Closed form '
                    'for n×n grids: γ<sub>t</sub> = (2m+1)(2m+r) with n = 4m+r. GNN columns: mean ± std across seeds.</div>',
                    unsafe_allow_html=True)
        if (FINAL / "plot_lattices.png").exists():
            st.image(str(FINAL / "plot_lattices.png"), width="stretch")
    with tabs[2]:
        rows = s["tables"]["E3"]
        table(rows, list(rows[0].keys()), numeric=[c for c in rows[0] if c != "graph"])
        st.markdown('<div class="caption">One ER (average degree 8) and one BA (m = 3) graph with 10,000 vertices. '
                    'At this size the GNN adds up to ⌈n/1000⌉ vertices per forward pass (each pick re-checked).</div>',
                    unsafe_allow_html=True)
