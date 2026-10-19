"""
operators.py — SA neighbourhood operators and phase-dependent selection (§16, §17).

Operator summary
────────────────
N1   Adjacent swap         — swap positions p and p+1 on one runway.
N2   Arbitrary swap        — swap any two positions on one runway.
N3b  Best re-insertion     — remove aircraft at p, reinsert at cheapest
                             feasible position on same runway by proxy.
N4   Block relocation      — move b consecutive aircraft to a new position
                             on the same runway.
X1   Transfer              — move one aircraft to any position on another runway.
X2   Cross-runway swap     — swap one aircraft from each of two runways.
X3   Best transfer         — move one aircraft to its globally best feasible
                             position on a different runway by proxy delta.
X4   Block transfer        — transfer a consecutive block to another runway.
X7   TC-guided repair      — target X3/N3b at the highest-impact aircraft.
XE   Ejection alias        — alias for X3 used in late-phase table.

Phase tables
────────────
f < 0.30 → EARLY : emphasise cross-runway diversification (X1–X4).
f < 0.75 → MID   : balanced; X7 (targeted repair) activated.
f ≥ 0.75 → LATE  : emphasise within-runway intensification (N3b, N2).
m = 2    → LATE_M2: cross-runway operators kept active through late phase,
            because m=2 forces the critical assignment decision right up to
            the end.
m = 1    → SINGLE: within-runway operators only.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from misc.mr_alp_2.config  import EPS
from misc.mr_alp_2.models  import Instance, HeuristicParams, MRSAParams
from misc.mr_alp_2.proxy   import (
    runway_feasible,
    _rwy_proxy_components,
    init_proxy_arrays,
    compute_proxy,
)


# ═══════════════════════════════════════════════════════════════════════════
#   Move result container
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MoveResult:
    """Proposed new sequences and the runway indices that were modified."""
    seqs:     List[List[int]]
    affected: List[int]


# ═══════════════════════════════════════════════════════════════════════════
#   Within-runway operators  N1 / N2 / N3b / N4
# ═══════════════════════════════════════════════════════════════════════════

def op_n1_adjacent_swap(
    seqs: List[List[int]], rho: int, p: int, inst: Instance,
) -> Optional[MoveResult]:
    """N1 — swap positions p and p+1 on runway ρ."""
    seq = seqs[rho]
    if p >= len(seq) - 1:
        return None
    ns = seq[:]
    ns[p], ns[p + 1] = ns[p + 1], ns[p]
    if not runway_feasible(ns, inst):
        return None
    r = [s[:] for s in seqs]
    r[rho] = ns
    return MoveResult(r, [rho])


def op_n2_swap(
    seqs: List[List[int]], rho: int, p: int, q: int, inst: Instance,
) -> Optional[MoveResult]:
    """N2 — swap positions p and q on runway ρ (arbitrary distance)."""
    seq = seqs[rho]
    if p == q or p >= len(seq) or q >= len(seq):
        return None
    ns = seq[:]
    ns[p], ns[q] = ns[q], ns[p]
    if not runway_feasible(ns, inst):
        return None
    r = [s[:] for s in seqs]
    r[rho] = ns
    return MoveResult(r, [rho])


def op_n3b_best_insertion(
    seqs:   List[List[int]],
    rho:    int,
    p:      int,
    inst:   Instance,
    params: HeuristicParams,
) -> Optional[MoveResult]:
    """
    N3b — remove aircraft at position p, reinsert at the cheapest feasible
    position on the same runway by proxy cost.
    """
    seq = seqs[rho]
    L   = len(seq)
    if L < 2:
        return None
    ac  = seq[p]
    sm  = seq[:p] + seq[p + 1:]
    if not runway_feasible(sm, inst):
        return None

    best_score = math.inf
    best_q     = -1
    for q in range(L):
        if q == p:
            continue
        ns = sm[:q] + [ac] + sm[q:]
        if not runway_feasible(ns, inst):
            continue
        tc, lbt, sep = _rwy_proxy_components(ns, inst)
        s = params.mu_tc * tc + params.mu_late * lbt + params.mu_sep * sep
        if s < best_score:
            best_score, best_q = s, q

    if best_q == -1:
        return None
    ns = sm[:best_q] + [ac] + sm[best_q:]
    r  = [s[:] for s in seqs]
    r[rho] = ns
    return MoveResult(r, [rho])


def op_n4_block_reloc(
    seqs: List[List[int]], rho: int, p: int, b: int, q: int, inst: Instance,
) -> Optional[MoveResult]:
    """N4 — move block seq[p:p+b] to position q on the same runway."""
    seq = seqs[rho]
    if p + b > len(seq) or b < 1:
        return None
    blk  = seq[p:p + b]
    rest = seq[:p] + seq[p + b:]
    ins  = q % (len(rest) + 1)
    ns   = rest[:ins] + blk + rest[ins:]
    if not runway_feasible(ns, inst):
        return None
    r = [s[:] for s in seqs]
    r[rho] = ns
    return MoveResult(r, [rho])


# ═══════════════════════════════════════════════════════════════════════════
#   Cross-runway operators  X1 / X2 / X3 / X4 / X7
# ═══════════════════════════════════════════════════════════════════════════

def op_x1_transfer(
    seqs: List[List[int]], rho_a: int, p: int, rho_b: int, q: int,
    inst: Instance,
) -> Optional[MoveResult]:
    """X1 — transfer aircraft at (ρa, p) to position q on runway ρb."""
    if rho_a == rho_b:
        return None
    sa = seqs[rho_a][:]
    sb = seqs[rho_b][:]
    ac = sa.pop(p)
    sb.insert(min(q, len(sb)), ac)
    if not runway_feasible(sa, inst) or not runway_feasible(sb, inst):
        return None
    r = [s[:] for s in seqs]
    r[rho_a] = sa
    r[rho_b] = sb
    return MoveResult(r, [rho_a, rho_b])


def op_x2_swap(
    seqs: List[List[int]], rho_a: int, p: int, rho_b: int, q: int,
    inst: Instance,
) -> Optional[MoveResult]:
    """X2 — swap aircraft at (ρa, p) and (ρb, q)."""
    if rho_a == rho_b or not seqs[rho_a] or not seqs[rho_b]:
        return None
    if p >= len(seqs[rho_a]) or q >= len(seqs[rho_b]):
        return None
    sa = seqs[rho_a][:]
    sb = seqs[rho_b][:]
    sa[p], sb[q] = sb[q], sa[p]
    if not runway_feasible(sa, inst) or not runway_feasible(sb, inst):
        return None
    r = [s[:] for s in seqs]
    r[rho_a] = sa
    r[rho_b] = sb
    return MoveResult(r, [rho_a, rho_b])


def op_x3_best_transfer(
    seqs:    List[List[int]],
    rho_a:   int,
    p:       int,
    rho_b:   int,
    inst:    Instance,
    params:  HeuristicParams,
    tc_rwy:  np.ndarray,
    lbt_rwy: np.ndarray,
    sep_rwy: np.ndarray,
) -> Optional[MoveResult]:
    """
    X3 — transfer aircraft from (ρa, p) to the best feasible position on ρb.

    Evaluates the proxy delta for each feasible position on ρb and returns
    the move with the minimum delta (most improvement).
    """
    if rho_a == rho_b:
        return None
    sa = seqs[rho_a][:]
    ac = sa.pop(p)
    if not runway_feasible(sa, inst):
        return None

    m  = len(seqs)
    n  = inst.n
    bs = float(inst.Pen_bar) / max((n / m) ** 2, 1.0)
    t  = sum(len(s) for s in seqs)
    ob = ((len(seqs[rho_a]) - t / m) ** 2
          + (len(seqs[rho_b]) - t / m) ** 2)

    best_delta = math.inf
    best_sb:   Optional[List[int]] = None

    for q in range(len(seqs[rho_b]) + 1):
        sb = seqs[rho_b][:]
        sb.insert(q, ac)
        if not runway_feasible(sb, inst):
            continue
        ta, la, ea = _rwy_proxy_components(sa, inst)
        tb, lb, eb = _rwy_proxy_components(sb, inst)
        nb = ((len(sa) - t / m) ** 2 + (len(sb) - t / m) ** 2)
        delta = (
            params.mu_tc    * ((ta + tb) - (tc_rwy[rho_a] + tc_rwy[rho_b]))
            + params.mu_late  * ((la + lb) - (lbt_rwy[rho_a] + lbt_rwy[rho_b]))
            + params.mu_count * (nb - ob) * bs
            + params.mu_sep   * ((ea + eb) - (sep_rwy[rho_a] + sep_rwy[rho_b]))
        )
        if delta < best_delta:
            best_delta, best_sb = delta, sb

    if best_sb is None:
        return None
    r = [s[:] for s in seqs]
    r[rho_a] = sa
    r[rho_b] = best_sb
    return MoveResult(r, [rho_a, rho_b])


def op_x4_block_transfer(
    seqs:  List[List[int]],
    rho_a: int,
    p:     int,
    b:     int,
    rho_b: int,
    q:     int,
    inst:  Instance,
) -> Optional[MoveResult]:
    """X4 — transfer block seq_a[p:p+b] to position q on runway ρb."""
    if rho_a == rho_b or p + b > len(seqs[rho_a]):
        return None
    blk = seqs[rho_a][p:p + b]
    sa  = seqs[rho_a][:p] + seqs[rho_a][p + b:]
    sb  = seqs[rho_b][:]
    sb[q:q] = blk
    if not runway_feasible(sa, inst) or not runway_feasible(sb, inst):
        return None
    r = [s[:] for s in seqs]
    r[rho_a] = sa
    r[rho_b] = sb
    return MoveResult(r, [rho_a, rho_b])


def op_x7_tc_repair(
    seqs:    List[List[int]],
    tc_rwy:  np.ndarray,
    lbt_rwy: np.ndarray,
    sep_rwy: np.ndarray,
    inst:    Instance,
    params:  HeuristicParams,
    rng:     random.Random,
    impact:  Optional[np.ndarray],
) -> Optional[MoveResult]:
    """
    X7 — TC-guided repair (§17).

    Selects the highest-impact aircraft (by LP impact or by tc/load score)
    and attempts X3 transfer to another runway; falls back to N3b if
    only one runway exists or all X3 positions are infeasible.
    """
    m = len(seqs)
    if impact is not None:
        scored = sorted(
            [(impact[seqs[rho][pos]], rho, pos)
             for rho in range(m) for pos in range(len(seqs[rho]))],
            key=lambda x: -x[0],
        )
    else:
        scored = sorted(
            [(tc_rwy[rho] / max(len(seqs[rho]), 1), rho, pos)
             for rho in range(m) for pos in range(len(seqs[rho]))],
            key=lambda x: -x[0],
        )
    if not scored:
        return None

    top_k    = max(1, len(scored) // 5)
    _, rho_a, p = rng.choice(scored[:top_k])

    others = [r for r in range(m) if r != rho_a]
    if others:
        res = op_x3_best_transfer(seqs, rho_a, p, rng.choice(others),
                                   inst, params, tc_rwy, lbt_rwy, sep_rwy)
        if res is not None:
            return res
    if len(seqs[rho_a]) >= 2:
        return op_n3b_best_insertion(seqs, rho_a, p, inst, params)
    return None


# ═══════════════════════════════════════════════════════════════════════════
#   Targeted aircraft selector
# ═══════════════════════════════════════════════════════════════════════════

def pick_aircraft_targeted(
    seqs:   List[List[int]],
    inst:   Instance,
    rng:    random.Random,
    pa_tc:  Optional[np.ndarray] = None,
    pa_lbt: Optional[np.ndarray] = None,
    impact: Optional[np.ndarray] = None,
) -> Tuple[int, int]:
    """
    Select a (runway, position) pair for operator application.

    Selection probabilities:
      60%  uniform random over all (ρ, pos) pairs.
      25%  top-20% by impact / pa_tc score.
      15%  top-20% by pa_lbt score.

    Falls back to uniform when no score arrays are available.
    """
    m    = len(seqs)
    flat = [(rho, pos)
            for rho in range(m) for pos in range(len(seqs[rho]))]
    if not flat:
        return 0, 0

    r      = rng.random()
    scores = impact if impact is not None else pa_tc

    if r < 0.60 or scores is None:
        return rng.choice(flat)

    if r < 0.85:
        ranked = sorted(
            ((scores[seqs[rho][pos]], rho, pos) for rho, pos in flat),
            key=lambda x: -x[0],
        )
        top  = max(1, len(ranked) // 5)
        _, rho, pos = rng.choice(ranked[:top])
        return rho, pos

    lbt_arr = pa_lbt if pa_lbt is not None else scores
    ranked  = sorted(
        ((lbt_arr[seqs[rho][pos]], rho, pos) for rho, pos in flat),
        key=lambda x: -x[0],
    )
    top = max(1, len(ranked) // 5)
    _, rho, pos = rng.choice(ranked[:top])
    return rho, pos


# ═══════════════════════════════════════════════════════════════════════════
#   Phase-dependent operator tables (§17.16)
# ═══════════════════════════════════════════════════════════════════════════

_OPS_EARLY   = [("X1",.18),("X2",.18),("X3",.18),("X4",.10),
                ("N2",.15),("N3b",.12),("N1",.09)]
_OPS_MID     = [("X1",.12),("X2",.12),("X3",.12),("X7",.14),
                ("N2",.14),("N3b",.18),("N1",.10),("XE",.08)]
_OPS_LATE    = [("N1",.18),("N2",.17),("N3b",.23),("X2",.14),
                ("X3",.10),("X7",.10),("XE",.08)]
_OPS_LATE_M2 = [("X2",.22),("X3",.18),("X7",.14),("XE",.10),
                ("N3b",.16),("N2",.12),("N1",.08)]
_OPS_SINGLE  = [("N1",.25),("N2",.28),("N3b",.30),("N4",.17)]


def select_op(f: float, m: int, rng: random.Random) -> str:
    """
    Sample an operator name from the phase-appropriate table.

    f : iteration fraction t / N_iter ∈ [0, 1].
    m : runway count.
    """
    if m == 1:
        table = _OPS_SINGLE
    elif m == 2:
        if   f < 0.30: table = _OPS_EARLY
        elif f < 0.75: table = _OPS_MID
        else:          table = _OPS_LATE_M2
    elif f < 0.30:     table = _OPS_EARLY
    elif f < 0.75:     table = _OPS_MID
    else:              table = _OPS_LATE
    ops, weights = zip(*table)
    return rng.choices(ops, weights=weights, k=1)[0]


def apply_op(
    op:      str,
    seqs:    List[List[int]],
    tc_rwy:  np.ndarray,
    lbt_rwy: np.ndarray,
    sep_rwy: np.ndarray,
    inst:    Instance,
    params:  HeuristicParams,
    p_sa:    MRSAParams,
    rng:     random.Random,
    stag:    int,
    N_iter:  int,
    pa_tc:   Optional[np.ndarray] = None,
    pa_lbt:  Optional[np.ndarray] = None,
    impact:  Optional[np.ndarray] = None,
    C_lp:    Optional[np.ndarray] = None,
) -> Optional[MoveResult]:
    """
    Dispatch operator op and return a MoveResult (or None on failure).

    The targeted aircraft selector is called first to obtain (rho_a, pos_a);
    operators requiring two runways sample rho_b uniformly from the rest.
    Block size respects B_stag cap under stagnation and B_max otherwise.
    """
    m        = len(seqs)
    rho_a, pos_a = pick_aircraft_targeted(
        seqs, inst, rng, pa_tc, pa_lbt, impact)
    L_a      = len(seqs[rho_a])
    b_cap    = (p_sa.B_stag
                if stag >= int(p_sa.M_stag * 0.75) else p_sa.B_max)

    if op == "N1":
        return op_n1_adjacent_swap(seqs, rho_a,
                                    rng.randint(0, max(L_a - 2, 0)), inst)
    if op == "N2":
        if L_a < 2: return None
        return op_n2_swap(seqs, rho_a,
                           rng.randint(0, L_a - 1),
                           rng.randint(0, L_a - 1), inst)
    if op == "N3b":
        if L_a < 2: return None
        return op_n3b_best_insertion(seqs, rho_a,
                                      rng.randint(0, L_a - 1), inst, params)
    if op == "N4":
        if L_a < 2: return None
        b = rng.randint(1, min(b_cap, L_a))
        return op_n4_block_reloc(seqs, rho_a,
                                  rng.randint(0, max(L_a - b, 0)), b,
                                  rng.randint(0, max(L_a - b, 0)), inst)
    if op == "X1":
        if m < 2: return None
        rho_b = rng.choice([r for r in range(m) if r != rho_a])
        return op_x1_transfer(seqs, rho_a, pos_a, rho_b,
                               rng.randint(0, len(seqs[rho_b])), inst)
    if op == "X2":
        if m < 2 or not seqs[rho_a]: return None
        rho_b = rng.choice([r for r in range(m) if r != rho_a])
        if not seqs[rho_b]: return None
        return op_x2_swap(seqs, rho_a, pos_a, rho_b,
                           rng.randint(0, len(seqs[rho_b]) - 1), inst)
    if op in ("X3", "XE"):
        if m < 2: return None
        rho_b = rng.choice([r for r in range(m) if r != rho_a])
        return op_x3_best_transfer(seqs, rho_a, pos_a, rho_b,
                                    inst, params, tc_rwy, lbt_rwy, sep_rwy)
    if op == "X4":
        if m < 2 or L_a < 1: return None
        b     = rng.randint(1, min(b_cap, L_a))
        rho_b = rng.choice([r for r in range(m) if r != rho_a])
        return op_x4_block_transfer(seqs, rho_a,
                                     rng.randint(0, max(L_a - b, 0)), b,
                                     rho_b,
                                     rng.randint(0, len(seqs[rho_b])), inst)
    if op == "X7":
        return op_x7_tc_repair(seqs, tc_rwy, lbt_rwy, sep_rwy,
                                inst, params, rng, impact)
    return None


# ═══════════════════════════════════════════════════════════════════════════
#   Candidate pool generation (§22)
# ═══════════════════════════════════════════════════════════════════════════

def generate_candidate_pool(
    f:       float,
    seqs:    List[List[int]],
    tc_rwy:  np.ndarray,
    lbt_rwy: np.ndarray,
    sep_rwy: np.ndarray,
    inst:    Instance,
    params:  HeuristicParams,
    p_sa:    MRSAParams,
    rng:     random.Random,
    stag:    int,
    N_iter:  int,
    R:       int,
    pa_tc:   Optional[np.ndarray],
    pa_lbt:  Optional[np.ndarray],
    impact:  Optional[np.ndarray],
    C_lp:    Optional[np.ndarray],
) -> list:
    """
    Generate R candidate moves, evaluate their proxy objectives, and return
    the pool sorted ascending (best first) by proxy value.

    Each pool entry: (proxy_new, MoveResult, tc_n, lbt_n, sep_n).
    """
    pool = []
    for _ in range(R):
        op  = select_op(f, len(seqs), rng)
        res = apply_op(op, seqs, tc_rwy, lbt_rwy, sep_rwy,
                        inst, params, p_sa, rng, stag, N_iter,
                        pa_tc=pa_tc, pa_lbt=pa_lbt,
                        impact=impact, C_lp=C_lp)
        if res is None:
            continue
        tc_n  = tc_rwy.copy()
        lbt_n = lbt_rwy.copy()
        sep_n = sep_rwy.copy()
        for rho in res.affected:
            tc_n[rho], lbt_n[rho], sep_n[rho] = _rwy_proxy_components(
                res.seqs[rho], inst)
        pool.append((
            compute_proxy(res.seqs, tc_n, lbt_n, sep_n, inst, params),
            res, tc_n, lbt_n, sep_n,
        ))
    pool.sort(key=lambda x: x[0])
    return pool