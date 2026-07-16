"""
operators.py — MR-ALP Solver: SA Neighbourhood Operators and Phase Selection
=============================================================================
§16  Move operators  N1/N2/N3b/N4 (within-runway), X1–X4/X7/XE (cross-runway)
§17  Phase-dependent operator selection table and apply_op dispatcher
§22  Candidate pool generation (_generate_candidate_pool)

Operator summary
----------------
N1   Adjacent swap         — swap positions p and p+1 on runway ρ.
N2   Arbitrary swap        — swap any two positions on runway ρ.
N3b  Best re-insertion     — remove aircraft from position p, re-insert at
                             cheapest feasible position on same runway.
N4   Block relocation      — move a block of b consecutive aircraft to a new
                             position on the same runway.
X1   Transfer              — move one aircraft to any position on a different runway.
X2   Cross-runway swap     — swap one aircraft from each of two different runways.
X3   Best transfer         — move one aircraft to its globally best feasible
                             position on a different runway.
X4   Block transfer        — transfer a block of aircraft to a different runway.
X7   TC-guided repair      — target X3 at the highest-impact aircraft.
XE   Ejection (alias X3)   — alias used in late-phase table for readability.

Phase tables (_OPS_EARLY / _OPS_MID / _OPS_LATE / _OPS_SINGLE)
----------------------------------------------------------------
Operator weights vary by iteration fraction f = t / N_iter:
  f < 0.30 → EARLY : emphasis on cross-runway diversification (X1–X4).
  f < 0.75 → MID   : balanced; X7 (targeted repair) activated.
  f ≥ 0.75 → LATE  : emphasis on within-runway intensification (N3b, N2).
  m = 1    → SINGLE: only within-runway operators.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from mr_alp.config      import DELTA_EVAL
from mr_alp.models      import Instance, HeuristicParams, MRSAParams
from mr_alp.instance    import runway_feasible
from mr_alp.proxy       import (
    _rwy_proxy_components, init_proxy_arrays, compute_proxy,
)
from mr_alp.delta       import best_insertion_position


# ═══════════════════════════════════════════════════════════════════════════
#   §16  MOVE RESULT CONTAINER
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MoveResult:
    """Carries the proposed new sequences and the runway indices that changed."""
    seqs:     List[List[int]]
    affected: List[int]


# ═══════════════════════════════════════════════════════════════════════════
#   §16  WITHIN-RUNWAY OPERATORS
# ═══════════════════════════════════════════════════════════════════════════

def op_n1_adjacent_swap(
    seqs: List[List[int]], rho: int, p: int, inst: Instance
) -> Optional[MoveResult]:
    """N1 — swap positions p and p+1 on runway ρ."""
    seq = seqs[rho]
    if p >= len(seq) - 1:
        return None
    ns = seq[:]
    ns[p], ns[p + 1] = ns[p + 1], ns[p]
    if not runway_feasible(ns, inst):
        return None
    r = [s[:] for s in seqs]; r[rho] = ns
    return MoveResult(r, [rho])


def op_n2_swap(
    seqs: List[List[int]], rho: int, p: int, q: int, inst: Instance
) -> Optional[MoveResult]:
    """N2 — swap positions p and q on runway ρ (arbitrary distance)."""
    seq = seqs[rho]
    if p == q or p >= len(seq) or q >= len(seq):
        return None
    ns = seq[:]
    ns[p], ns[q] = ns[q], ns[p]
    if not runway_feasible(ns, inst):
        return None
    r = [s[:] for s in seqs]; r[rho] = ns
    return MoveResult(r, [rho])


def op_n3b_best_insertion(
    seqs: List[List[int]], rho: int, p: int,
    inst: Instance, params: HeuristicParams,
) -> Optional[MoveResult]:
    """
    N3b — remove aircraft at position p, re-insert at the cheapest feasible
    position on the same runway by proxy objective.

    With config.DELTA_EVAL the position search uses the staged delta
    evaluation in mr_alp.delta (O(L) scoring + exact check on top-k);
    otherwise the legacy full scan evaluates every position.
    """
    seq = seqs[rho]
    L   = len(seq)
    if L < 2:
        return None
    ac  = seq[p]
    sm  = seq[:p] + seq[p + 1:]
    if not runway_feasible(sm, inst):
        return None

    if DELTA_EVAL:
        best_q = best_insertion_position(sm, ac, inst, params)
        if best_q < 0:
            return None
        ns = sm[:best_q] + [ac] + sm[best_q:]
        r  = [s[:] for s in seqs]; r[rho] = ns
        return MoveResult(r, [rho])

    best_score, best_q = math.inf, -1
    for q in range(L):
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
    r  = [s[:] for s in seqs]; r[rho] = ns
    return MoveResult(r, [rho])


def op_n4_block_reloc(
    seqs: List[List[int]], rho: int, p: int, b: int, q: int, inst: Instance
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
    r = [s[:] for s in seqs]; r[rho] = ns
    return MoveResult(r, [rho])


# ═══════════════════════════════════════════════════════════════════════════
#   §16  CROSS-RUNWAY OPERATORS
# ═══════════════════════════════════════════════════════════════════════════

def op_x1_transfer(
    seqs: List[List[int]], rho_a: int, p: int, rho_b: int, q: int, inst: Instance
) -> Optional[MoveResult]:
    """X1 — transfer aircraft at (ρa, p) to position q on runway ρb."""
    if rho_a == rho_b:
        return None
    sa = seqs[rho_a][:]; sb = seqs[rho_b][:]
    ac = sa.pop(p)
    sb.insert(min(q, len(sb)), ac)
    if not runway_feasible(sa, inst) or not runway_feasible(sb, inst):
        return None
    r = [s[:] for s in seqs]; r[rho_a] = sa; r[rho_b] = sb
    return MoveResult(r, [rho_a, rho_b])


def op_x2_swap(
    seqs: List[List[int]], rho_a: int, p: int, rho_b: int, q: int, inst: Instance
) -> Optional[MoveResult]:
    """X2 — swap aircraft at (ρa, p) and (ρb, q)."""
    if rho_a == rho_b or not seqs[rho_a] or not seqs[rho_b]:
        return None
    if p >= len(seqs[rho_a]) or q >= len(seqs[rho_b]):
        return None
    sa = seqs[rho_a][:]; sb = seqs[rho_b][:]
    sa[p], sb[q] = sb[q], sa[p]
    if not runway_feasible(sa, inst) or not runway_feasible(sb, inst):
        return None
    r = [s[:] for s in seqs]; r[rho_a] = sa; r[rho_b] = sb
    return MoveResult(r, [rho_a, rho_b])


def op_x3_best_transfer(
    seqs: List[List[int]], rho_a: int, p: int, rho_b: int,
    inst: Instance, params: HeuristicParams,
    tc_rwy: np.ndarray, lbt_rwy: np.ndarray, sep_rwy: np.ndarray,
) -> Optional[MoveResult]:
    """
    X3 — transfer aircraft from (ρa, p) to the best feasible position on ρb.

    With config.DELTA_EVAL the target-position search uses the staged delta
    evaluation in mr_alp.delta.  This is ranking-equivalent for the terms
    that vary with the insertion position: the source-runway components and
    the balance (mu_count) term are constant across q on a fixed runway
    pair, so they cannot change the argmin and are omitted from the search.
    """
    if rho_a == rho_b:
        return None
    sa = seqs[rho_a][:]; ac = sa.pop(p)
    if not runway_feasible(sa, inst):
        return None

    if DELTA_EVAL:
        best_q = best_insertion_position(seqs[rho_b], ac, inst, params)
        if best_q < 0:
            return None
        sb = seqs[rho_b][:]; sb.insert(best_q, ac)
        r = [s[:] for s in seqs]; r[rho_a] = sa; r[rho_b] = sb
        return MoveResult(r, [rho_a, rho_b])

    n  = inst.n; m = len(seqs)
    bs = float(inst.Pen_bar) / max((n / m) ** 2, 1.0)
    t  = sum(len(s) for s in seqs)
    ob = (len(seqs[rho_a]) - t / m) ** 2 + (len(seqs[rho_b]) - t / m) ** 2

    # Source-runway components are loop-invariant: hoisted out of the scan
    # (was recomputed per position — a free 2x on the legacy path).
    ta, la, ea = _rwy_proxy_components(sa, inst)

    best_delta, best_sb = math.inf, None
    for q in range(len(seqs[rho_b]) + 1):
        sb = seqs[rho_b][:]; sb.insert(q, ac)
        if not runway_feasible(sb, inst):
            continue
        tb, lb, eb = _rwy_proxy_components(sb, inst)
        nb = (len(sa) - t / m) ** 2 + (len(sb) - t / m) ** 2
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
    r = [s[:] for s in seqs]; r[rho_a] = sa; r[rho_b] = best_sb
    return MoveResult(r, [rho_a, rho_b])


def op_x4_block_transfer(
    seqs: List[List[int]], rho_a: int, p: int, b: int,
    rho_b: int, q: int, inst: Instance,
) -> Optional[MoveResult]:
    """X4 — transfer block seq_a[p:p+b] to position q on runway ρb."""
    if rho_a == rho_b or p + b > len(seqs[rho_a]):
        return None
    blk = seqs[rho_a][p:p + b]
    sa  = seqs[rho_a][:p] + seqs[rho_a][p + b:]
    sb  = seqs[rho_b][:]; sb[q:q] = blk
    if not runway_feasible(sa, inst) or not runway_feasible(sb, inst):
        return None
    r = [s[:] for s in seqs]; r[rho_a] = sa; r[rho_b] = sb
    return MoveResult(r, [rho_a, rho_b])


def op_x7_tc_repair(
    seqs: List[List[int]],
    tc_rwy: np.ndarray, lbt_rwy: np.ndarray,
    inst: Instance, params: HeuristicParams,
    rng: random.Random,
    impact: Optional[np.ndarray],
    pa_tc: Optional[np.ndarray] = None,
    pa_lbt: Optional[np.ndarray] = None,
) -> Optional[MoveResult]:
    """
    X7 — TC-guided repair.

    Selects the highest-impact aircraft (by LP impact or per-aircraft TC/load
    score) and attempts X3 transfer to another runway; falls back to N3b
    re-insertion.
    """
    m = len(seqs)
    cands = []
    for rho in range(m):
        for pos, ac in enumerate(seqs[rho]):
            if impact is not None:
                score = float(impact[ac])
            elif pa_tc is not None or pa_lbt is not None:
                score = (float(pa_tc[ac]) if pa_tc is not None else 0.0)
                score += (float(pa_lbt[ac]) if pa_lbt is not None else 0.0)
            else:
                score = float(tc_rwy[rho] + lbt_rwy[rho]) / max(len(seqs[rho]), 1)
            cands.append((score, rho, pos))
    if not cands:
        return None
    cands.sort(key=lambda x: -x[0])
    _, rho_a, p = rng.choice(cands[:max(1, len(cands) // 5)])

    others = [r for r in range(m) if r != rho_a]
    if others:
        res = op_x3_best_transfer(seqs, rho_a, p, rng.choice(others),
                                   inst, params, tc_rwy, lbt_rwy,
                                   np.zeros(m))
        if res is not None:
            return res
    if len(seqs[rho_a]) >= 2:
        return op_n3b_best_insertion(seqs, rho_a, p, inst, params)
    return None


# ═══════════════════════════════════════════════════════════════════════════
#   §17  TARGETED AIRCRAFT SELECTOR
# ═══════════════════════════════════════════════════════════════════════════

def pick_aircraft_targeted(
    seqs: List[List[int]],
    inst: Instance,
    rng: random.Random,
    pa_tc: Optional[np.ndarray] = None,
    pa_lbt: Optional[np.ndarray] = None,
    impact: Optional[np.ndarray] = None,
) -> Tuple[int, int]:
    """
    Select a (runway, position) pair for operator application.

    Selection probabilities:
      60% uniform random across all (ρ, pos) pairs.
      25% top-20% by impact / pa_tc score.
      15% top-20% by pa_lbt score.
    """
    m    = len(seqs)
    flat = [(rho, pos) for rho in range(m) for pos in range(len(seqs[rho]))]
    if not flat:
        return 0, 0
    r      = rng.random()
    scores = impact if impact is not None else pa_tc
    if r < 0.60 or scores is None:
        return rng.choice(flat)
    if r < 0.85:
        scored = sorted(
            ((scores[seqs[rho][pos]], rho, pos) for rho, pos in flat),
            key=lambda x: -x[0])
        top  = max(1, len(scored) // 5)
        _, rho, pos = rng.choice(scored[:top])
        return rho, pos
    lbt_arr = pa_lbt if pa_lbt is not None else scores
    scored  = sorted(
        ((lbt_arr[seqs[rho][pos]], rho, pos) for rho, pos in flat),
        key=lambda x: -x[0])
    top = max(1, len(scored) // 5)
    _, rho, pos = rng.choice(scored[:top])
    return rho, pos


# ═══════════════════════════════════════════════════════════════════════════
#   §17  PHASE-DEPENDENT OPERATOR TABLES AND DISPATCHER
# ═══════════════════════════════════════════════════════════════════════════

_OPS_EARLY  = [("X1",.18),("X2",.18),("X3",.18),("X4",.10),("N2",.15),("N3b",.12),("N1",.09)]
_OPS_MID    = [("X1",.12),("X2",.12),("X3",.12),("X7",.14),("N2",.14),("N3b",.18),("N1",.10),("XE",.08)]
_OPS_LATE   = [("N1",.18),("N2",.17),("N3b",.23),("X2",.14),("X3",.10),("X7",.10),("XE",.08)]
# For m=2 the key decision is WHICH runway each aircraft lands on, so keep
# cross-runway operators active through the late phase instead of switching
# to within-runway intensification.
_OPS_LATE_M2 = [("X2",.22),("X3",.18),("X7",.14),("XE",.10),("N3b",.16),("N2",.12),("N1",.08)]
_OPS_SINGLE = [("N1",.25),("N2",.28),("N3b",.30),("N4",.17)]


def select_op(f: float, m: int, rng: random.Random) -> str:
    """
    Sample an operator name from the phase-appropriate probability table.

    f : iteration fraction t / N_iter ∈ [0, 1].
    m : runway count (selects single-runway table when m=1; m=2 uses a
        late-phase table that keeps cross-runway operators active longer).
    """
    if m == 1:
        table = _OPS_SINGLE
    elif m == 2:
        if f < 0.30:   table = _OPS_EARLY
        elif f < 0.75: table = _OPS_MID
        else:          table = _OPS_LATE_M2
    elif f < 0.30:   table = _OPS_EARLY
    elif f < 0.75:   table = _OPS_MID
    else:            table = _OPS_LATE
    ops, weights = zip(*table)
    return rng.choices(ops, weights=weights, k=1)[0]


def apply_op(
    op: str,
    seqs: List[List[int]],
    tc_rwy: np.ndarray, lbt_rwy: np.ndarray, sep_rwy: np.ndarray,
    inst: Instance, params: HeuristicParams, p_sa: MRSAParams,
    rng: random.Random, stag: int, N_iter: int,
    pa_tc: Optional[np.ndarray] = None,
    pa_lbt: Optional[np.ndarray] = None,
    impact: Optional[np.ndarray] = None,
    C_lp: Optional[np.ndarray] = None,
) -> Optional[MoveResult]:
    """
    Dispatch operator op and return a MoveResult (or None on failure).

    The targeted aircraft selector is called first to obtain (rho_a, pos_a);
    operators that require two runways pick rho_b uniformly from the rest.
    """
    m       = len(seqs)
    rho_a, pos_a = pick_aircraft_targeted(seqs, inst, rng, pa_tc, pa_lbt, impact)
    L_a     = len(seqs[rho_a])

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
        b_cap = p_sa.B_stag if stag >= int(p_sa.M_stag_frac * N_iter) else p_sa.B_max
        b     = rng.randint(1, min(b_cap, L_a))
        return op_n4_block_reloc(seqs, rho_a,
                                  rng.randint(0, L_a - b), b,
                                  rng.randint(0, L_a - b), inst)
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
        b     = rng.randint(1, min(p_sa.B_max, L_a))
        rho_b = rng.choice([r for r in range(m) if r != rho_a])
        return op_x4_block_transfer(seqs, rho_a,
                                     rng.randint(0, L_a - b), b,
                                     rho_b, rng.randint(0, len(seqs[rho_b])),
                                     inst)
    if op == "X7":
        return op_x7_tc_repair(seqs, tc_rwy, lbt_rwy,
                                inst, params, rng, impact, pa_tc, pa_lbt)
    return None


# ═══════════════════════════════════════════════════════════════════════════
#   §22  CANDIDATE POOL GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_candidate_pool(
    f: float,
    seqs: List[List[int]],
    tc_rwy: np.ndarray, lbt_rwy: np.ndarray, sep_rwy: np.ndarray,
    inst: Instance, params: HeuristicParams, p_sa: MRSAParams,
    rng: random.Random, stag: int, N_iter: int, R: int,
    pa_tc: Optional[np.ndarray],
    pa_lbt: Optional[np.ndarray],
    impact: Optional[np.ndarray],
    C_lp: Optional[np.ndarray],
    op_stats: Optional[Dict[str, list]] = None,
) -> list:
    """
    Generate R candidate moves, evaluate their proxy objectives, and return
    the pool sorted ascending by proxy value.

    Each pool entry: (proxy_new, MoveResult, tc_n, lbt_n, sep_n).

    op_stats (optional): per-operator accumulator mutated in place,
    op -> [n_selected, n_valid, time_s].  Timing covers apply_op plus the
    candidate's proxy re-evaluation, so time_s reflects the full cost of
    proposing one candidate with that operator.
    """
    pool = []
    for _ in range(R):
        op  = select_op(f, len(seqs), rng)
        t_op = time.perf_counter() if op_stats is not None else 0.0
        res = apply_op(op, seqs, tc_rwy, lbt_rwy, sep_rwy,
                        inst, params, p_sa, rng, stag, N_iter,
                        pa_tc=pa_tc, pa_lbt=pa_lbt,
                        impact=impact, C_lp=C_lp)
        if res is None:
            if op_stats is not None:
                rec = op_stats.setdefault(op, [0, 0, 0.0])
                rec[0] += 1
                rec[2] += time.perf_counter() - t_op
            continue
        tc_n = tc_rwy.copy(); lbt_n = lbt_rwy.copy(); sep_n = sep_rwy.copy()
        for rho in res.affected:
            tc_n[rho], lbt_n[rho], sep_n[rho] = _rwy_proxy_components(
                res.seqs[rho], inst)
        pool.append((
            compute_proxy(res.seqs, tc_n, lbt_n, sep_n, inst, params),
            res, tc_n, lbt_n, sep_n,
        ))
        if op_stats is not None:
            rec = op_stats.setdefault(op, [0, 0, 0.0])
            rec[0] += 1
            rec[1] += 1
            rec[2] += time.perf_counter() - t_op
    pool.sort(key=lambda x: x[0])
    return pool
