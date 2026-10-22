"""
SBC: Stochastic Beam Construction
===============================================================================
Construction heuristic producing initial solutions for SA on the single- and
multi-runway Aircraft Landing Problem.  One code path serves both: m = 1 is
the single-runway problem.

    (C1) r_j ≤ x_j ≤ d_j
    (C2) x_k ≥ x_j + s_jk     for all j before k on the same runway
    (C3) min Σ_j g_j·max(δ_j − x_j, 0) + h_j·max(x_j − δ_j, 0)

SBC is construction only (no improvement phase): it grows a beam of partial
schedules level by level and hands the best complete schedule to SA.  Every
returned schedule is feasible under all pairwise separations and costed with
the exact Stage-2 timing optimum.

ALGORITHM
---------
§1  Beam and decision order
    The beam holds up to `width` partial schedules.  Aircraft are decided one
    level at a time in target-time order.  In randomised runs the order is
    jittered once per construction, keys δ_j + N(0, jitter · s̄), so aircraft
    with close target times can swap while the broad order survives.

§2  Expansion
    Each state expands by placing the level's aircraft at up to `slots`
    cheapest feasible slots per runway (alp_core.slot_options, exact cost
    increases, inside the w_back/w_fwd window around its target-order
    position; the full window is searched when the window offers no slot).
    A child whose runway sequence the exact evaluator prices at infinity is
    dropped.

§3  Pricing and retention
    States are priced by exact partial cost, the sum of the per-runway exact
    costs.  Every state at a level has scheduled the same set of aircraft, so
    the prices compare directly.  Children are deduplicated by canonical
    partial form, cheapest copy kept.
    rng is None  →  keep the `width` cheapest children.
    rng given    →  sample `width` of the top oversample · `width` children
                    without replacement with probability
                        ∝ exp(−(price − price_min) / (temperature · c_unit)),
                    c_unit = mean_j max(g_j, h_j) · s̄.

§4  Commit and completion
    A child is committed with alp_core.commit at its exact cost as it is
    created, so every beam state is exactly priced at all times.  When no
    aircraft remain pending, the cheapest state of the final beam is the
    returned schedule.

USAGE
-----
    from const_heuristics.sbc import sbc, sbc_pool, SBCParams

    seqs, cost = sbc(inst, m)                                    # deterministic
    seqs, cost = sbc(inst, m, rng=np.random.default_rng(7))      # randomised
    pool = sbc_pool(inst, m, k=16, seed=0)                       # [(cost, seqs), ...]

`inst` is any object with arrays r, d, delta, g, h, s, e.g.
alp_core.load_instance("data/airland9.txt").  seqs is a list of m runway
sequences; for the single-runway solver use seqs[0].

Benchmark: set the CONFIGURE HERE block below and run  python const_heuristics/sbc.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    from .alp_core import INF, Problem, Seqs, build_pool, canonical, commit, run_benchmark, slot_options
except ImportError:                                   # run as a script: python const_heuristics/sbc.py
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from const_heuristics.alp_core import INF, Problem, Seqs, build_pool, canonical, commit, run_benchmark, slot_options

# ═════════════════════════════════════════════════════════════════════════════
#   CONFIGURE HERE  (benchmark driver, §6)
#
#   JOBS       — list of (instance_name, m) pairs; None = every OR Library job
#                with a nonzero reference value (plus all m = 1 jobs).
#   POOL_SIZE  — distinct schedules requested per job from sbc_pool.
#   SEED       — first rng seed used by sbc_pool.
#   N_WORKERS  — processes used by sbc_pool (1 = serial).
# ═════════════════════════════════════════════════════════════════════════════
JOBS: Optional[List[Tuple[str, int]]] = None
POOL_SIZE = 8
SEED      = 0
N_WORKERS = 1
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class SBCParams:
    """Construction knobs."""
    width:       int   = 16    # beam width: states retained per level
    slots:       int   = 2     # cheapest slots branched on, per runway
    oversample:  int   = 2     # retention draws from the top oversample * width children
    w_back:      int   = 6     # insertion window: positions before the target-order slot
    w_fwd:       int   = 1     # insertion window: positions after the target-order slot
    temperature: float = 0.3   # retention temperature in units of c_unit (rng only)
    jitter:      float = 0.5   # decision-order jitter in units of s_bar (rng only)


# ─────────────────────────────────────────────────────────────────────────────
# §1  Beam state
# ─────────────────────────────────────────────────────────────────────────────

class _State:
    __slots__ = ("seqs", "lens", "costs", "pending")
    def __init__(self, seqs, lens, costs, pending):
        self.seqs, self.lens, self.costs, self.pending = seqs, lens, costs, pending
    def clone(self):
        return _State(self.seqs.copy(), self.lens.copy(), self.costs.copy(), list(self.pending))
    def partial(self) -> Seqs:
        return [self.seqs[rho, :self.lens[rho]].tolist() for rho in range(self.seqs.shape[0])]


# ─────────────────────────────────────────────────────────────────────────────
# §2  Expansion
# ─────────────────────────────────────────────────────────────────────────────

def _expand(P: Problem, st: _State, prm: SBCParams) -> List[_State]:
    """Children of one beam state: the next aircraft (target order) placed at
    its up-to-`slots` cheapest feasible slots per runway."""
    j = st.pending[0]
    for wb, wf in ((prm.w_back, prm.w_fwd), (P.n, P.n)):
        o = slot_options(st.seqs, st.lens, st.costs, j, *P.A, wb, wf)
        if len(o):
            break
    children: List[_State] = []
    per_rho: dict = {}
    for inc, rho, p, cst in o:
        rho, p = int(rho), int(p)
        if per_rho.get(rho, 0) >= prm.slots:
            continue
        per_rho[rho] = per_rho.get(rho, 0) + 1
        child = st.clone()
        new = child.seqs[rho, :child.lens[rho]].tolist()
        new.insert(p, j)
        c = P.exact_cost(new)
        if c >= INF:
            continue
        commit(child.seqs, child.lens, child.costs, rho, p, j, c)
        child.pending.remove(j)
        children.append(child)
    return children


# ─────────────────────────────────────────────────────────────────────────────
# §1, §3-4  Decision order, retention and completion
# ─────────────────────────────────────────────────────────────────────────────

def _construct(P: Problem, m: int, prm: SBCParams, rng) -> Optional[Tuple[Seqs, float]]:
    n = P.n
    if rng is not None and prm.jitter > 0:
        keys = P.dl + rng.normal(0.0, prm.jitter * P.s_bar, size=n)
    else:
        keys = P.dl
    init = _State(np.zeros((m, n), np.int64), np.zeros(m, np.int64), np.zeros(m),
                  [int(j) for j in np.argsort(keys, kind="stable")])
    beam = [init]
    while beam and beam[0].pending:
        children: List[_State] = []
        for st in beam:
            children.extend(_expand(P, st, prm))
        if not children:
            return None
        # dedupe by canonical partial form, keep the cheapest copy
        priced = {}
        for ch in children:
            key = canonical(ch.partial())
            price = float(ch.costs.sum())
            if key not in priced or price < priced[key][0]:
                priced[key] = (price, ch)
        pool = sorted(priced.values(), key=lambda t: t[0])
        W = prm.width
        if rng is not None and prm.temperature > 0 and len(pool) > W:
            top = pool[:min(prm.oversample * W, len(pool))]
            sc = np.array([t[0] for t in top])
            w = np.exp(-(sc - sc.min()) / (prm.temperature * P.c_unit))
            pick = rng.choice(len(top), size=W, replace=False, p=w / w.sum())
            beam = [top[i][1] for i in sorted(pick.tolist())]
        else:
            beam = [t[1] for t in pool[:W]]
    if not beam:
        return None
    best = min(beam, key=lambda st: float(st.costs.sum()))
    return best.partial(), float(best.costs.sum())


# ─────────────────────────────────────────────────────────────────────────────
# §5  Public API
# ─────────────────────────────────────────────────────────────────────────────

def sbc(inst, m: int = 1, rng: Optional[np.random.Generator] = None,
        params: Optional[SBCParams] = None,
        _problem: Optional[Problem] = None) -> Tuple[Optional[Seqs], float]:
    """
    Build one schedule.  rng=None → deterministic.

    Returns (seqs, cost): seqs is a list of m runway sequences, cost the exact
    Stage-2 objective.  Returns (None, inf) if some level produced no child.
    """
    prm = params or SBCParams()
    P = _problem or Problem(inst)
    res = _construct(P, m, prm, rng)
    if res is None:
        return None, INF
    return res


def sbc_pool(inst, m: int = 1, k: int = 8, seed: int = 0,
             params: Optional[SBCParams] = None, include_deterministic: bool = True,
             max_tries: Optional[int] = None, n_workers: int = 1,
             escalate: float = 1.5, temperature_max: float = 10.0) -> List[Tuple[float, Seqs]]:
    """Up to k distinct SBC schedules, sorted by exact cost (see alp_core.build_pool)."""
    return build_pool(sbc, inst, m, k, seed, params or SBCParams(), include_deterministic,
                      max_tries, n_workers, escalate, temperature_max)


# ─────────────────────────────────────────────────────────────────────────────
# §6  Benchmark driver (settings in CONFIGURE HERE)
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    run_benchmark(sbc, SBCParams(), JOBS, POOL_SIZE, SEED, N_WORKERS)


if __name__ == "__main__":
    main()
