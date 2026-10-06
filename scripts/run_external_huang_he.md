# Running the Huang & He (2026) baseline on our test graphs

Huang & He, "A Reinforcement Learning Framework for Minimize Total Dominating Set
Problem", *Advances in Applied Mathematics* 15(3):338–350, 2026, is the closest
prior work. Its code is at <https://github.com/Hyelow0/RL-Total-Dominating-main>.
**Their code is not included in this repository.** Do not copy it in; run it in its
own folder and compare via files.

## 1. Export our test graphs as edge lists

```bash
python - <<'EOF'
from pathlib import Path
from tds.io import load_graphs, write_edge_list
for npz in Path("data/test").glob("*/*.npz"):
    out = Path("data/export_edgelists") / npz.parent.name / npz.stem
    out.mkdir(parents=True, exist_ok=True)
    for i, g in enumerate(load_graphs(npz)):
        write_edge_list(g, out / f"{i:03d}.txt")      # 0-based "u v" per line
EOF
```

## 2. Run their solver

Clone their repository separately, then follow its README to load each exported
edge list. Adapt only their I/O, and record exactly what you changed. Save one
solution per graph as a JSON list of 0-based vertex ids:
`data/external_huang_he/<set>/<stem>/<idx>.json`.

## 3. Verify and import

Every external solution must pass our checker before it is reported:

```python
import json
from pathlib import Path
from tds.io import load_graphs
from tds.checker import is_total_dominating_set
for npz in Path("data/test").glob("*/*.npz"):
    graphs = load_graphs(npz)
    d = Path("data/external_huang_he") / npz.parent.name / npz.stem
    for i, g in enumerate(graphs):
        f = d / f"{i:03d}.json"
        if f.exists():
            S = json.loads(f.read_text())
            print(npz.stem, i, len(S), is_total_dominating_set(g.adj, S))
```

Report invalid solutions as invalid. Do not repair them silently.

## Caution

Their paper reports greedy = 85 on ER(200, p = 0.1). On our ER(200, p = 0.1)
graphs, a correct greedy gives 17–18, greedy+RR 17–18 and the ILP 15–16 (seeds
0–4, 60 s limit; see the Phase 1 notes in the README). Compare against our verified
greedy, not against their reported greedy numbers.
