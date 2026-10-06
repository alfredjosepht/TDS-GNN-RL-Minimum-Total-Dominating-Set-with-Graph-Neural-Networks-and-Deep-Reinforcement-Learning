"""Write the README 'Results' section from results/summary.json and checkpoints/model_info.json.
No number is typed by hand: everything below is read from files produced by runs in this repository."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md_table(rows, cols=None):
    cols = cols or list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def pct(m, s=None):
    if m is None:
        return "--"
    return f"{100 * m:.2f}%" + ("" if s is None else f" ± {100 * s:.2f}")


def main():
    info = json.loads((ROOT / "checkpoints/model_info.json").read_text())
    s = json.loads((ROOT / "results/summary.json").read_text())
    L = []
    L.append("All numbers below were produced by `scripts/finalize_models.py` and `scripts/run_final.py` "
             "(see [Reproducing the results](#reproducing-the-results)). GNN results are mean ± std across the "
             f"{len(info['seeds'])} training seeds; greedy methods are deterministic. "
             f"**{s['validity']['solutions_checked']:,} solutions were checked with `is_total_dominating_set`: "
             f"{100 * s['validity']['valid_fraction']:.0f}% valid.**")
    L.append("\n### Model\n")
    rows = [{"seed": x["seed"], "training steps": f"{x['train_steps']:,}", "best checkpoint (step)": f"{x['best_step']:,}",
             "val. gap, GNN+RR": pct(x["best_val_gap_rr"]), "val. gap, GNN raw": pct(x["best_val_gap_raw"]),
             "val. optimal (GNN+RR)": f"{100 * x['best_val_frac_optimal_rr']:.1f}%",
             "val. gap, greedy+RR": pct(x["val_gap_greedy_rr"])} for x in info["seeds"]]
    L.append(md_table(rows))
    L.append(f"\n{info['encoder'].upper()} encoder, {info['layers']} layers, hidden size {info['hidden']}, "
             f"{info['readout']} readout, {info['parameters']:,} parameters. {info['validation_note']}")
    h = s["E1_headline"]
    L.append("\n### Headline (E1, ER + BA, n = 20–500)\n")
    L.append(f"Mean gap to the proven optimum over the {h['ilp_proven']} of {h['instances']} E1 instances where the "
             f"ILP proved optimality ({h['ilp_not_proven']} not proven, excluded):\n")
    names = {"greedy": "Greedy", "greedy_rr": "Greedy + RR", "greedy_rr_ls": "Greedy + RR + LS",
             "rl": "GNN-RL", "rl_rr": "GNN-RL + RR", "rl_rr_ls": "GNN-RL + RR + LS"}
    rows = []
    for k, nm in names.items():
        g = h["gap_on_proven"][k]
        rows.append({"method": nm, "mean gap": pct(g["gap_mean"], g["gap_std_across_seeds"])})
    L.append(md_table(rows))
    L.append(f"\nLocal search time limit: {s['settings']['ls_time_s']:.0f} s for both greedy and GNN.")
    L.append("\n### E1 — random graphs (gap to ILP optimum, proven instances only)\n")
    L.append(md_table(s["tables"]["E1"]))
    L.append("\n![gap vs n](results/final/plot_gap_vs_n.png)\n\n![runtime vs n](results/final/plot_runtime_vs_n.png)")
    L.append("\n<details><summary>E1 solution sizes and runtimes</summary>\n")
    L.append(md_table(s["tables"]["E1_size_time"]))
    L.append("\n</details>")
    L.append("\n### E2 — lattices, zero-shot (|S|)\n")
    L.append(md_table(s["tables"]["E2"]))
    L.append("\nGrid closed form: γ_t(n×n) = (2m+1)(2m+r) with n = 4m+r (OEIS A302488; verified by the ILP for "
             "n ≤ 12 in `tests/test_closed_forms.py`).\n\n![lattices](results/final/plot_lattices.png)")
    L.append("\n### E3 — scale (n = 10,000)\n")
    L.append(md_table(s["tables"]["E3"]))
    ov = s["seed_overlap"]
    L.append(f"\n**Data separation.** Training-graph seeds checked: {ov['n_train_seeds_checked']:,}; validation: "
             f"{ov['n_val_seeds']}; test: {ov['n_test_seeds']:,}. Overlaps: train–val {ov['train_val_overlap']}, "
             f"train–test {ov['train_test_overlap']}, val–test {ov['val_test_overlap']}. ({ov['note']}.)")
    if (ROOT / "results/final/training_curves.png").exists():
        L.append("\n**Training curves** (both seeds):\n\n![training](results/final/training_curves.png)")
    block = "\n".join(L)
    p = ROOT / "README.md"
    txt = p.read_text(encoding="utf-8")
    txt = re.sub(r"(<!-- RESULTS:BEGIN[^>]*-->)(.*?)(<!-- RESULTS:END -->)",
                 lambda m: m.group(1) + "\n" + block + "\n" + m.group(3), txt, flags=re.S)
    p.write_text(txt, encoding="utf-8")
    print("README results section written")


if __name__ == "__main__":
    main()
