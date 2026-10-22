"""
RGI: Regret-Guided Insertion
===============================================================================
Construction heuristic producing initial solutions for SA on the single- and
multi-runway Aircraft Landing Problem.  One code path serves both: m = 1 is
the single-runway problem.

    (C1) r_j ≤ x_j ≤ d_j
    (C2) x_k ≥ x_j + s_jk     for all j before k on the same runway
    (C3) min Σ_j g_j·max(δ_j − x_j, 0) + h_j·max(x_j − δ_j, 0)

RGI is construction only (no improvement phase): it inserts one aircraft at a
time and hands the result to SA.  Every returned schedule is feasible under
all pairwise separations and costed with the exact Stage-2 timing optimum.

ALGORITHM
---------
§1  Candidate set
    Unscheduled aircraft are kept in target-time order.  The candidate set is
    the first `cand` of them.

§2  Exact pricing and regret
    For each candidate j, every insertion slot on every runway is priced with
    the exact Stage-2 timing evaluator: c1 is j's cheapest insertion
    increase, c2 its second cheapest.  The regret of j is c2 − c1, what j
    stands to lose if its best slot is taken first.  A candidate with exactly
    one feasible slot has infinite regret.

§3  Selection (the only source of randomness)
    rng is None  →  insert the candidate with the largest regret, ties broken
                    by target order, at its cheapest slot.
    rng given    →  draw a candidate with probability
                        ∝ exp((regret_j − regret_max) / (temperature · c_unit)),
                    c_unit = mean_j max(g_j, h_j) · s̄.  Infinite regrets are
                    placed one c_unit above the largest finite regret, so
                    they dominate the draw without making it degenerate.

§4  Commit
    The drawn aircraft is inserted at its cheapest slot; the runway's new
    cost is already exact from §2.  If no candidate has a feasible slot, the
    first unscheduled aircraft in target order that has one is inserted
    instead; if none does, construction fails.

USAGE
-----
    from const_heuristics.rgi import rgi, rgi_pool, RGIParams

    seqs, cost = rgi(inst, m)                                    # deterministic
    seqs, cost = rgi(inst, m, rng=np.random.default_rng(7))      # randomised
    pool = rgi_pool(inst, m, k=16, seed=0)                       # [(cost, seqs), ...]

`inst` is any object with arrays r, d, delta, g, h, s, e.g.
alp_core.load_instance("data/airland9.txt").  seqs is a list of m runway
sequences; for the single-runway solver use seqs[0].

Benchmark: set the CONFIGURE HERE block below and run  python const_heuristics/rgi.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    from .alp_core import INF, Problem, Seqs, build_pool, run_benchmark
except ImportError:                                   # run as a script: python const_heuristics/rgi.py
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from const_heuristics.alp_core import INF, Problem, Seqs, build_pool, run_benchmark

# ═════════════════════════════════════════════════════════════════════════════
#   CONFIGURE HERE  (benchmark driver, §6)
#
#   JOBS       — list of (instance_name, m) pairs; None = every OR Library job
#                with a nonzero reference value (plus all m = 1 jobs).
#   POOL_SIZE  — distinct schedules requested per job from rgi_pool.
#   SEED       — first rng seed used by rgi_pool.
#   N_WORKERS  — processes used by rgi_pool (1 = serial).
# ═════════════════════════════════════════════════════════════════════════════
JOBS: Optional[List[Tuple[str, int]]] = None
POOL_SIZE = 8
SEED      = 0
N_WORKERS = 1
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class RGIParams:
    """Construction knobs."""
    cand:        int   = 10    # size of the candidate set (first unscheduled by target time)
    temperature: float = 0.2   # selection temperature in units of c_unit (rng only)


# ─────────────────────────────────────────────────────────────────────────────
# §2  Exact insertion pricing
# ─────────────────────────────────────────────────────────────────────────────

def _best_two(P: Problem, seqs: Seqs, costs: np.ndarray, j: int):
    """(c1, argbest, c2): cheapest and second-cheapest insertion increases of j."""
    c1 = c2 = INF
    arg = None
    for rho in range(len(seqs)):
        sq = seqs[rho]
        base = costs[rho]
        for pos in range(len(sq) + 1):
            c = P.exact_cost(sq[:pos] + [j] + sq[pos:])
            if c >= INF:
                continue
            inc = c - base
            if inc < c1:
                c2 = c1
                c1 = inc
                arg = (rho, pos, c)
            elif inc < c2:
                c2 = inc
    return c1, arg, c2


# ─────────────────────────────────────────────────────────────────────────────
# §1, §3-4  Candidate set, selection and commit
# ─────────────────────────────────────────────────────────────────────────────

def _construct(P: Problem, m: int, prm: RGIParams, rng) -> Optional[Tuple[Seqs, float]]:
    seqs: Seqs = [[] for _ in range(m)]
    costs = np.zeros(m)
    scheduled = np.zeros(P.n, dtype=bool)
    target_order = [int(j) for j in np.argsort(P.dl, kind="stable")]
    remaining = P.n
    while remaining:
        # Bound alp_core's exact-cost memo: RGI prices ~cand x slots distinct
        # sequences per step, and at n = 500 the unbounded memo grows to
        # millions of sequence tuples and exhausts memory (observed: the
        # benchmark process was OOM-killed twice on airland13).  Current
        # runway costs are tracked in `costs`, so clearing loses only
        # cross-step reuse.
        if len(P._memo) > 50_000:
            P._memo.clear()
        pend = [j for j in target_order if not scheduled[j]]
        cands = pend[:prm.cand]
        scored = []                       # (aircraft, regret, argbest)
        for j in cands:
            c1, arg, c2 = _best_two(P, seqs, costs, j)
            if arg is None:
                continue
            regret = (c2 - c1) if c2 < INF else np.inf
            scored.append((j, regret, arg))
        if not scored:
            # ragged fallback: any unscheduled aircraft with a feasible slot
            placed = False
            for j in pend:
                c1, arg, _ = _best_two(P, seqs, costs, j)
                if arg is not None:
                    scored = [(j, 0.0, arg)]
                    placed = True
                    break
            if not placed:
                return None
        if rng is None or len(scored) == 1:
            # argmax regret, ties by target order (scored is in target order)
            pick = max(scored, key=lambda t: t[1])
        else:
            regs = np.array([t[1] for t in scored])
            finite = regs[np.isfinite(regs)]
            rmax = finite.max() if len(finite) else 0.0
            regs = np.where(np.isfinite(regs), regs, rmax + P.c_unit)
            w = np.exp((regs - regs.max()) / (prm.temperature * P.c_unit))
            pick = scored[int(rng.choice(len(scored), p=w / w.sum()))]
        j, _, (rho, pos, c) = pick
        seqs[rho].insert(pos, j)
        costs[rho] = c
        scheduled[j] = True
        remaining -= 1
    return seqs, float(costs.sum())


# ─────────────────────────────────────────────────────────────────────────────
# §5  Public API
# ─────────────────────────────────────────────────────────────────────────────

def rgi(inst, m: int = 1, rng: Optional[np.random.Generator] = None,
        params: Optional[RGIParams] = None,
        _problem: Optional[Problem] = None) -> Tuple[Optional[Seqs], float]:
    """
    Build one schedule.  rng=None → deterministic.

    Returns (seqs, cost): seqs is a list of m runway sequences, cost the exact
    Stage-2 objective.  Returns (None, inf) only if no aircraft can be placed.
    """
    prm = params or RGIParams()
    P = _problem or Problem(inst)
    res = _construct(P, m, prm, rng)
    if res is None:
        return None, INF
    return res


def rgi_pool(inst, m: int = 1, k: int = 8, seed: int = 0,
             params: Optional[RGIParams] = None, include_deterministic: bool = True,
             max_tries: Optional[int] = None, n_workers: int = 1,
             escalate: float = 1.5, temperature_max: float = 10.0) -> List[Tuple[float, Seqs]]:
    """Up to k distinct RGI schedules, sorted by exact cost (see alp_core.build_pool)."""
    return build_pool(rgi, inst, m, k, seed, params or RGIParams(), include_deterministic,
                      max_tries, n_workers, escalate, temperature_max)


# ─────────────────────────────────────────────────────────────────────────────
# §6  Benchmark driver (settings in CONFIGURE HERE)
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    run_benchmark(rgi, RGIParams(), JOBS, POOL_SIZE, SEED, N_WORKERS)


if __name__ == "__main__":
    main()
