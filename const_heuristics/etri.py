"""
ETRI: Exact-Timing Randomised Insertion
===============================================================================
Construction heuristic producing initial solutions for SA on the single- and
multi-runway Aircraft Landing Problem.  One code path serves both: m = 1 is
the single-runway problem.

    (C1) r_j ≤ x_j ≤ d_j
    (C2) x_k ≥ x_j + s_jk     for all j before k on the same runway
    (C3) min Σ_j g_j·max(δ_j − x_j, 0) + h_j·max(x_j − δ_j, 0)

ETRI is construction only (no improvement phase): it builds the schedule one
aircraft at a time and hands the result to SA.  Every returned schedule is
feasible under all pairwise separations and costed with the exact Stage-2
timing optimum.

ALGORITHM
---------
§1  Exact timing (alp_core.cet, PAVA)
    For a fixed runway sequence substitute y_k = x_k − S_k, S_k = Σ_{t<k} s_{t,t+1}.
    The consecutive-separation timing LP becomes isotonic regression
    (y nondecreasing) with convex piecewise-linear losses and box bounds, which
    the pool-adjacent-violators algorithm solves exactly.  That LP relaxes the
    full pairwise LP, so when its optimum also satisfies every non-consecutive
    separation it is the exact optimum; otherwise the committed runway is
    re-priced with the full pairwise LP (§4).  Validated against HiGHS
    (0 mismatches), ~150x faster at n = 500.

§2  Candidate set and pilot score
    Pending aircraft are kept in target-time order.  The candidate set is the
    first `candidates` pending aircraft.  For each candidate j and each of its
    `slots` cheapest insertion slots (runway, position) in a window around its
    target-order position, the pilot inserts j there and then greedily inserts
    the remaining first `horizon` pending aircraft (cheapest slot each).  The
    pilot score of j is the lowest total cost so reached.  Candidates with no
    feasible slot or pilot completion are dropped: what remains is the
    feasible set.

§3  Selection (the only source of randomness)
    rng is None  →  insert the feasible candidate with the lowest pilot score.
    rng given    →  draw a feasible candidate with probability
                        ∝ exp(−(score_j − score_min) / (temperature · c_unit)),
                    c_unit = mean_j max(g_j, h_j) · s̄  (cost of delaying an average
                    aircraft by one average separation).  Close calls are randomised,
                    clearly worse candidates are almost never drawn.
    The drawn aircraft is committed to the slot that gave its pilot score.

§4  Commit
    The new runway sequence is re-priced exactly (cet, or the full pairwise LP
    when a non-consecutive separation binds).  A slot that the LP rejects as
    infeasible is skipped in favour of the next-best one.

USAGE
-----
    from const_heuristics.etri import etri, etri_pool, ETRIParams

    seqs, cost = etri(inst, m)                                    # deterministic
    seqs, cost = etri(inst, m, rng=np.random.default_rng(7))      # randomised
    pool = etri_pool(inst, m, k=16, seed=0)                       # [(cost, seqs), ...]

`inst` is any object with arrays r, d, delta, g, h, s, e.g.
alp_core.load_instance("data/airland9.txt").  seqs is a list of m runway
sequences; for the single-runway solver use seqs[0].

Benchmark: set the CONFIGURE HERE block below and run  python const_heuristics/etri.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numba import njit

try:
    from .alp_core import INF, Problem, Seqs, build_pool, canonical, commit, run_benchmark, slot_options
except ImportError:                                   # run as a script: python const_heuristics/etri.py
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from const_heuristics.alp_core import INF, Problem, Seqs, build_pool, canonical, commit, run_benchmark, slot_options

# ═════════════════════════════════════════════════════════════════════════════
#   CONFIGURE HERE  (benchmark driver, §6)
#
#   JOBS       — list of (instance_name, m) pairs; None = every OR Library job
#                with a nonzero reference value (plus all m = 1 jobs).
#   POOL_SIZE  — distinct schedules requested per job from etri_pool.
#   SEED       — first rng seed used by etri_pool.
#   N_WORKERS  — processes used by etri_pool (1 = serial).
# ═════════════════════════════════════════════════════════════════════════════
JOBS: Optional[List[Tuple[str, int]]] = None
POOL_SIZE = 8
SEED      = 0
N_WORKERS = 1
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class ETRIParams:
    """Construction knobs.  Defaults chosen on airland1-13 with m = 1..5."""
    candidates:  int   = 4     # size of the candidate set (first pending by target time)
    horizon:     int   = 8     # pending aircraft inserted by each pilot run
    slots:       int   = 4     # cheapest slots per candidate scored by the pilot
    w_back:      int   = 6     # insertion window: positions before the target-order slot
    w_fwd:       int   = 1     # insertion window: positions after the target-order slot
    temperature: float = 0.3   # selection temperature in units of c_unit (rng only)


# ─────────────────────────────────────────────────────────────────────────────
# §2  Pilot score  (§1 exact timing, slots and commit live in alp_core.py)
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _pilot(seqs, lens, costs, j, rho, p, c, look, r, d, dl, g, h, s, s_max, wb, wf):
    """Total cost after placing j at (rho, p) and greedily inserting the rest of `look`."""
    S = seqs.copy(); Ln = lens.copy(); C = costs.copy()
    commit(S, Ln, C, rho, p, j, c)
    for t in range(look.shape[0]):
        k = look[t]
        if k == j:
            continue
        o = slot_options(S, Ln, C, k, r, d, dl, g, h, s, s_max, wb, wf)
        if o.shape[0] == 0:
            return INF
        commit(S, Ln, C, int(o[0, 1]), int(o[0, 2]), k, o[0, 3])
    return C.sum()


# ─────────────────────────────────────────────────────────────────────────────
# §3-4  Candidate scoring, selection and commit
# ─────────────────────────────────────────────────────────────────────────────

def _score_candidates(P: Problem, seqs, lens, costs, pending, prm: ETRIParams, wb: int, wf: int):
    """[(pilot score, j, [(score, rho, p) per scored slot, best first]), ...] for the feasible set."""
    look = np.array(pending[:prm.horizon], np.int64)
    feasible = []
    for j in pending[:prm.candidates]:
        opts = slot_options(seqs, lens, costs, j, *P.A, wb, wf)
        scored = []
        for inc, rho, p, c in opts[:prm.slots]:
            sc = _pilot(seqs, lens, costs, j, int(rho), int(p), c, look, *P.A, wb, wf)
            if sc < INF:
                scored.append((sc, int(rho), int(p)))
        if scored:
            scored.sort()
            feasible.append((scored[0][0], j, scored))
    return feasible


def _construct(P: Problem, m: int, prm: ETRIParams, rng) -> Optional[Tuple]:
    n = P.n
    pending = [int(j) for j in np.argsort(P.dl, kind="stable")]
    seqs = np.zeros((m, n), np.int64); lens = np.zeros(m, np.int64); costs = np.zeros(m)
    while pending:
        feasible = _score_candidates(P, seqs, lens, costs, pending, prm, prm.w_back, prm.w_fwd)
        if not feasible:                       # no slot in the window: search every position
            feasible = _score_candidates(P, seqs, lens, costs, pending, prm, n, n)
        if not feasible:
            return None
        sc = np.array([f[0] for f in feasible])
        if rng is not None and prm.temperature > 0 and len(feasible) > 1:
            w = np.exp(-(sc - sc.min()) / (prm.temperature * P.c_unit))
            first = int(rng.choice(len(feasible), p=w / w.sum()))
            order = [first] + [int(i) for i in np.argsort(sc, kind="stable") if i != first]
        else:
            order = [int(i) for i in np.argsort(sc, kind="stable")]
        for i in order:                        # falls through only if the LP rejects every slot
            _, j, slots = feasible[i]
            for _, rho, p in slots:
                new = seqs[rho, :lens[rho]].tolist()
                new.insert(p, j)
                c = P.exact_cost(new)
                if c < INF:
                    commit(seqs, lens, costs, rho, p, j, c)
                    break
            else:
                continue
            pending.remove(j)
            break
        else:
            return None
    return seqs, lens, costs


# ─────────────────────────────────────────────────────────────────────────────
# §5  Public API
# ─────────────────────────────────────────────────────────────────────────────

def etri(inst, m: int = 1, rng: Optional[np.random.Generator] = None,
         params: Optional[ETRIParams] = None,
         _problem: Optional[Problem] = None) -> Tuple[Optional[Seqs], float]:
    """
    Build one schedule.  rng=None → deterministic.

    Returns (seqs, cost): seqs is a list of m runway sequences, cost the exact
    Stage-2 objective.  Returns (None, inf) if no feasible slot was found for
    some aircraft.
    """
    prm = params or ETRIParams()
    P = _problem or Problem(inst)
    res = _construct(P, m, prm, rng)
    if res is None:
        return None, INF
    seqs, lens, costs = res
    return [seqs[k, :lens[k]].tolist() for k in range(m)], float(costs.sum())


def etri_pool(inst, m: int = 1, k: int = 8, seed: int = 0,
              params: Optional[ETRIParams] = None, include_deterministic: bool = True,
              max_tries: Optional[int] = None, n_workers: int = 1,
              escalate: float = 1.5, temperature_max: float = 10.0) -> List[Tuple[float, Seqs]]:
    """
    Up to k distinct ETRI schedules, sorted by exact cost (see alp_core.build_pool).

    Fewer than k come back when the instance admits fewer (airland6 with m = 1
    has essentially one feasible sequence).
    """
    return build_pool(etri, inst, m, k, seed, params or ETRIParams(), include_deterministic,
                      max_tries, n_workers, escalate, temperature_max)


# ─────────────────────────────────────────────────────────────────────────────
# §6  Benchmark driver (settings in CONFIGURE HERE)
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    run_benchmark(etri, ETRIParams(), JOBS, POOL_SIZE, SEED, N_WORKERS)


if __name__ == "__main__":
    main()
