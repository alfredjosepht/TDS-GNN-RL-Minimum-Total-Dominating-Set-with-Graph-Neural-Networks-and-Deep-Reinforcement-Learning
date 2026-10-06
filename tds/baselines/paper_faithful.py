"""Base-paper-faithful ablation (Section 8.1, item 5).

Uses a checkpoint trained with configs/ablation_paper_faithful.yaml: the paper's
reward (Eq. 5, adapted to open neighbourhoods), no action pruning beyond v ∉ S, no
forced-vertex preprocessing, sum readout and MSE loss. At inference: no redundancy
removal and no local search.

Note: the base paper's own action rule (forbid v ∪ N(v) after picking v) is NOT
used: it makes chosen vertices pairwise non-adjacent, so it can never produce a
total dominating set (change C2 in the README).
"""
from __future__ import annotations

PAPER_ENV = dict(reward_mode="paper", mask_mode="notin", use_forced=False)
DEFAULT_PAPER_CHECKPOINT = "checkpoints/paper_faithful_seed0/best.pt"


def paper_faithful(graph, checkpoint=None, device=None):
    from ..solve import solve_rl
    ck = checkpoint if checkpoint and "paper" in str(checkpoint) else DEFAULT_PAPER_CHECKPOINT
    return solve_rl(graph, ck, rr=False, ls_time=0.0, device=device, multi_select=1,
                    env_kwargs=PAPER_ENV, method_name="paper_faithful")
