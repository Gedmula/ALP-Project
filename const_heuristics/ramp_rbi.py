"""
TC-RBI: Target-Conflict Regret-Based Insertion
==============================================
Construction heuristic producing initial solutions for SA on the single- and
multi-runway Aircraft Landing Problem.  One code path serves both: m = 1 is
the single-runway problem.

    (C1) r_j ≤ x_j ≤ d_j
    (C2) x_k ≥ x_j + s_jk     for all j before k on the same runway
    (C3) min Σ_j g_j·max(δ_j − x_j, 0) + h_j·max(x_j − δ_j, 0)

Every quantity the heuristic compares is a cost in the units of (C3): there are
no weights to tune and no search phase.  Every returned schedule is feasible
under all pairwise separations and costed with the exact Stage-2 timing
optimum (alp_core.cet, PAVA; full pairwise LP when a non-consecutive
separation binds).

ALGORITHM
---------
§1  Conflict queue (which aircraft compete now)
    Pending aircraft are kept in target-time order.  They are projected
    first-come-first-served onto the current runway frontiers: aircraft k lands
    at max(δ_k, t_last(ρ) + s_{last(ρ),k}) on the runway ρ where that is earliest.
    The queue Q is the first pending aircraft, then each next one that is
    delayed past its target in this projection or in target conflict with its
    predecessor (δ_k − δ_prev < s_prev,k).  It always holds at least m + 1
    aircraft (with m runways the next m + 1 must put two on one runway) and at
    most `queue_cap` (a computational bound).

§2  Regret by queue resolution
    For a member j of Q and a slot σ (runway, position) in the window around
    j's target-order position, the resolution cost

        D(j, σ) = total schedule cost after inserting j at σ and then every other
                  member of Q, in target order, at its cheapest slot

    measures what committing j to σ now does to the whole queue.  For a queue
    of two this is exactly the regret rule: with c(j) the best insertion cost of
    j and c(j | k) its best cost after k is placed, D(j,·) ≤ D(k,·)  ⇔
    c(j) + c(k|j) ≤ c(k) + c(j|k)  ⇔  r(k|j) ≤ r(j|k), where r(a|b) = c(a|b) − c(a)
    is the regret a suffers when b goes first.  The member that would suffer
    most from waiting goes first.

§3  Selection (the only source of randomness)
    Each member's score is min_σ D(j, σ).
    rng is None  →  insert the member with the lowest score at its best slot.
    rng given    →  draw a member with probability
                        ∝ exp(−(score_j − score_min) / (temperature · c_unit)),
                    c_unit = mean_j max(g_j, h_j) · s̄, then use its best slot.

§4  Commit
    The new runway sequence is re-priced exactly.  A slot the LP rejects is
    skipped in favour of the member's next-best slot, then the next member.

USAGE
-----
    from const_heuristics.ramp_rbi import tc_rbi, tc_rbi_pool, TCRBIParams

    seqs, cost = tc_rbi(inst, m)                                    # deterministic
    seqs, cost = tc_rbi(inst, m, rng=np.random.default_rng(7))      # randomised
    pool = tc_rbi_pool(inst, m, k=16, seed=0)                       # [(cost, seqs), ...]

`inst` is any object with arrays r, d, delta, g, h, s, e.g.
alp_core.load_instance("data/airland9.txt").  seqs is a list of m runway
sequences; for the single-runway solver use seqs[0].

Benchmark: set the CONFIGURE HERE block below and run  python const_heuristics/ramp_rbi.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numba import njit

try:
    from .alp_core import INF, Problem, Seqs, build_pool, cet, commit, run_benchmark, slot_options
except ImportError:                                   # run as a script: python const_heuristics/ramp_rbi.py
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from const_heuristics.alp_core import INF, Problem, Seqs, build_pool, cet, commit, run_benchmark, slot_options


# ═════════════════════════════════════════════════════════════════════════════
#   CONFIGURE HERE  (benchmark driver, §6)
#
#   JOBS       — list of (instance_name, m) pairs; None = every OR Library job
#                with a nonzero reference value (plus all m = 1 jobs).
#   POOL_SIZE  — distinct schedules requested per job from tc_rbi_pool.
#   SEED       — first rng seed used by tc_rbi_pool.
#   N_WORKERS  — processes used by tc_rbi_pool (1 = serial).
# ═════════════════════════════════════════════════════════════════════════════
JOBS: Optional[List[Tuple[str, int]]] = None
POOL_SIZE = 8
SEED      = 0
N_WORKERS = 1
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class TCRBIParams:
    """Construction knobs.  None of them weighs one cost against another."""
    queue_cap:   int   = 14    # maximum conflict-queue length (computational bound)
    w_back:      int   = 6     # insertion window: positions before the target-order slot
    w_fwd:       int   = 1     # insertion window: positions after the target-order slot
    temperature: float = 0.3   # selection temperature in units of c_unit (rng only)


# ─────────────────────────────────────────────────────────────────────────────
# §1  Conflict queue
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _queue(seqs, lens, pending, kmin, cap, r, d, dl, g, h, s, s_max):
    """First aircraft of `pending` (target order) that compete for the runways now (§1)."""
    m = seqs.shape[0]
    last = np.full(m, -1, np.int64)
    t_last = np.full(m, -INF)
    for rho in range(m):
        L = lens[rho]
        if L > 0:
            x = np.empty(L)
            cet(seqs[rho, :L], L, r, d, dl, g, h, s, s_max, x)
            last[rho] = seqs[rho, L - 1]
            t_last[rho] = x[L - 1]
    out = np.empty(min(cap, pending.shape[0]), np.int64)
    c = 0
    prev = -1
    for t in range(pending.shape[0]):
        k = pending[t]
        best = INF; br = 0
        for rho in range(m):
            e = dl[k]
            if last[rho] >= 0 and t_last[rho] + s[last[rho], k] > e:
                e = t_last[rho] + s[last[rho], k]
            if e < best:
                best = e; br = rho
        delayed = best > dl[k] + 1e-9
        conflict = prev >= 0 and dl[k] - dl[prev] < s[prev, k]
        if c >= kmin and not (delayed or conflict):
            break
        out[c] = k; c += 1
        prev = k
        last[br] = k; t_last[br] = best
        if c >= out.shape[0]:
            break
    return out[:c]


# ─────────────────────────────────────────────────────────────────────────────
# §2  Regret by queue resolution
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _resolve(seqs, lens, costs, Q, a, rho, p, c, r, d, dl, g, h, s, s_max, wb, wf):
    """D: total cost after placing Q[a] at (rho, p), then the rest of Q in order at cheapest slots."""
    S = seqs.copy(); Ln = lens.copy(); C = costs.copy()
    commit(S, Ln, C, rho, p, Q[a], c)
    for b in range(Q.shape[0]):
        if b == a:
            continue
        o = slot_options(S, Ln, C, Q[b], r, d, dl, g, h, s, s_max, wb, wf)
        if o.shape[0] == 0:
            return INF
        commit(S, Ln, C, int(o[0, 1]), int(o[0, 2]), Q[b], o[0, 3])
    return C.sum()


@njit(cache=True)
def _queue_scores(seqs, lens, costs, Q, r, d, dl, g, h, s, s_max, wb, wf):
    """Rows (member index a, D, runway, position) for every finite slot of every member,
    members in queue order and slots in alp_core.slot_options order."""
    m = seqs.shape[0]
    out = np.empty((Q.shape[0] * m * (min(wb, seqs.shape[1]) + min(wf, seqs.shape[1]) + 2), 4))
    no = 0
    for a in range(Q.shape[0]):
        o = slot_options(seqs, lens, costs, Q[a], r, d, dl, g, h, s, s_max, wb, wf)
        for t in range(o.shape[0]):
            D = _resolve(seqs, lens, costs, Q, a, int(o[t, 1]), int(o[t, 2]), o[t, 3],
                         r, d, dl, g, h, s, s_max, wb, wf)
            if D < INF:
                out[no, 0] = a; out[no, 1] = D; out[no, 2] = o[t, 1]; out[no, 3] = o[t, 2]
                no += 1
    return out[:no]


# ─────────────────────────────────────────────────────────────────────────────
# §3-4  Selection and commit
# ─────────────────────────────────────────────────────────────────────────────

def _construct(P: Problem, m: int, prm: TCRBIParams, rng,
               start: Optional[Tuple] = None, pending: Optional[List[int]] = None) -> Optional[Tuple]:
    """
    Build a schedule.  By default from empty runways with every aircraft pending.
    With `start` = (seqs, lens, costs) and `pending` (aircraft not in `start`),
    the partial schedule is completed instead (used by ruin-and-recreate).
    """
    n = P.n
    if start is None:
        pending = [int(j) for j in np.argsort(P.dl, kind="stable")]
        seqs = np.zeros((m, n), np.int64); lens = np.zeros(m, np.int64); costs = np.zeros(m)
    else:
        seqs, lens, costs = (a.copy() for a in start)
        pending = sorted((int(j) for j in pending), key=lambda j: (P.dl[j], j))
    while pending:
        Q = _queue(seqs, lens, np.array(pending, np.int64), min(m + 1, len(pending)),
                   prm.queue_cap, *P.A)
        rows = _queue_scores(seqs, lens, costs, Q, *P.A, prm.w_back, prm.w_fwd)
        if len(rows) == 0:                     # no slot in the window: search every position
            rows = _queue_scores(seqs, lens, costs, Q, *P.A, n, n)
        if len(rows) == 0:
            return None
        # per member: its slots by (D, slot_options order); score = best D
        slots = {}
        for a, D, rho, p in rows:
            slots.setdefault(int(a), []).append((D, int(rho), int(p)))
        members = sorted(slots)                # queue order = target order
        for a in members:
            slots[a].sort(key=lambda t: t[0])  # stable: ties keep slot_options order
        sc = np.array([slots[a][0][0] for a in members])
        if rng is not None and prm.temperature > 0 and len(members) > 1:
            w = np.exp(-(sc - sc.min()) / (prm.temperature * P.c_unit))
            first = int(rng.choice(len(members), p=w / w.sum()))
            order = [first] + [int(i) for i in np.argsort(sc, kind="stable") if i != first]
        else:
            order = [int(i) for i in np.argsort(sc, kind="stable")]
        for i in order:                        # falls through only if the LP rejects every slot
            j = int(Q[members[i]])
            for _, rho, p in slots[members[i]]:
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

def tc_rbi(inst, m: int = 1, rng: Optional[np.random.Generator] = None,
           params: Optional[TCRBIParams] = None,
           _problem: Optional[Problem] = None) -> Tuple[Optional[Seqs], float]:
    """
    Build one schedule.  rng=None → deterministic.

    Returns (seqs, cost): seqs is a list of m runway sequences, cost the exact
    Stage-2 objective.  Returns (None, inf) if no feasible slot was found for
    some aircraft.
    """
    prm = params or TCRBIParams()
    P = _problem or Problem(inst)
    res = _construct(P, m, prm, rng)
    if res is None:
        return None, INF
    seqs, lens, costs = res
    return [seqs[k, :lens[k]].tolist() for k in range(m)], float(costs.sum())


def tc_rbi_pool(inst, m: int = 1, k: int = 8, seed: int = 0,
                params: Optional[TCRBIParams] = None, include_deterministic: bool = True,
                max_tries: Optional[int] = None, n_workers: int = 1,
                escalate: float = 1.5, temperature_max: float = 10.0) -> List[Tuple[float, Seqs]]:
    """Up to k distinct TC-RBI schedules, sorted by exact cost (see alp_core.build_pool)."""
    return build_pool(tc_rbi, inst, m, k, seed, params or TCRBIParams(), include_deterministic,
                      max_tries, n_workers, escalate, temperature_max)


# ─────────────────────────────────────────────────────────────────────────────
# §6  Benchmark driver (settings in CONFIGURE HERE)
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    run_benchmark(tc_rbi, TCRBIParams(), JOBS, POOL_SIZE, SEED, N_WORKERS)


if __name__ == "__main__":
    main()
