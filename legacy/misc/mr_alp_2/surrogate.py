"""
surrogate.py — Construction surrogate (§7, §8).

Consecutive-predecessor propagation provides fast (O(n)) landing-time
estimates used to rank insertions during construction and inter-runway
improvement.  The surrogate is not used to certify feasibility; all final
feasibility checks are performed by the Stage-2 LP (§11).

Surrogate time propagation (§7):
    Ĉ_{u1}  = r_{u1}
    Ĉ_{uq}  = max(r_{uq}, Ĉ_{u(q-1)} + s[u(q-1), uq])

Insertion cost (§8):
    Δ(j, ρ, p) = ΔP̂ + λ₁·h̄·ΔSep_raw + λ₂·h̄·(B'_ρ − B̄)
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from misc.mr_alp_2.models import Instance, HybridParams
from misc.mr_alp_2.config import EPS


# ─────────────────────────────────────────────────────────────────────────────
#   Surrogate time propagation
# ─────────────────────────────────────────────────────────────────────────────

def surrogate_times(seq: List[int], inst: Instance) -> List[float]:
    """
    Consecutive-separation surrogate landing times Ĉ_u for seq (§7).
    Returns an empty list for an empty sequence.
    """
    if not seq:
        return []
    C    = [0.0] * len(seq)
    C[0] = float(inst.r[seq[0]])
    for q in range(1, len(seq)):
        C[q] = max(float(inst.r[seq[q]]),
                   C[q - 1] + float(inst.s[seq[q - 1], seq[q]]))
    return C


def surrogate_penalty(seq: List[int], inst: Instance) -> float:
    """Surrogate runway penalty P̂(π_ρ) = Σ [g_u·E_u + h_u·T_u] (§7)."""
    if not seq:
        return 0.0
    C = surrogate_times(seq, inst)
    total = 0.0
    for q, j in enumerate(seq):
        total += (float(inst.g[j]) * max(float(inst.delta[j]) - C[q], 0.0)
                  + float(inst.h[j]) * max(C[q] - float(inst.delta[j]), 0.0))
    return total


def committed_time(seq: List[int], inst: Instance) -> float:
    """Committed time B_ρ = Ĉ_{last(ρ)} (§7).  Returns 0 for empty runway."""
    if not seq:
        return 0.0
    return surrogate_times(seq, inst)[-1]


# ─────────────────────────────────────────────────────────────────────────────
#   Incremental insertion cost (optimised for construction inner loop)
# ─────────────────────────────────────────────────────────────────────────────

def _delta_sep_raw(j: int, seq: List[int], p: int, inst: Instance) -> float:
    """
    ΔSep_raw(j, ρ, p): incremental separation burden (§8.2).

    Interior (a, j, b): max(0, s[a,j] + s[j,b] − s[a,b])
    First     (j, b)  : s[j, b]
    Last      (a, j)  : s[a, j]
    Empty              : 0
    """
    L = len(seq)
    if L == 0:
        return 0.0
    has_pred = p > 0
    has_succ = p < L
    if has_pred and has_succ:
        a, b = seq[p - 1], seq[p]
        return max(0.0,
                   float(inst.s[a, j])
                   + float(inst.s[j, b])
                   - float(inst.s[a, b]))
    elif has_succ:
        return float(inst.s[j, seq[0]])
    else:
        return float(inst.s[seq[-1], j])


def insertion_cost_incremental(
    j:      int,
    seq:    List[int],
    p:      int,
    C_orig: List[float],
    B_bar:  float,
    inst:   Instance,
    params: HybridParams,
) -> float:
    """
    Δ(j, ρ, p) using precomputed surrogate times C_orig for seq (§8).

    Propagates only from position p onwards, reusing the prefix.
    Returns +inf if the insertion is surrogate-infeasible (any Ĉ_u > d_u).

    This is the preferred variant for inner loops where C_orig is cached.
    """
    # ── Compute C_j (inserted aircraft) ──────────────────────────────────
    if p > 0:
        t_prev = C_orig[p - 1]
        s_pj   = float(inst.s[seq[p - 1], j])
    else:
        t_prev = 0.0
        s_pj   = 0.0
    C_j = max(float(inst.r[j]), t_prev + s_pj)
    if C_j > float(inst.d[j]):
        return math.inf

    pen_j = (float(inst.g[j]) * max(float(inst.delta[j]) - C_j, 0.0)
             + float(inst.h[j]) * max(C_j - float(inst.delta[j]), 0.0))

    # ── Propagate downstream effect on seq[p:] ────────────────────────────
    delta_pen = pen_j
    t_cur     = C_j
    prev      = j
    for q in range(p, len(seq)):
        k     = seq[q]
        C_new = max(float(inst.r[k]),
                    t_cur + float(inst.s[prev, k]))
        if C_new > float(inst.d[k]):
            return math.inf
        C_old = C_orig[q]
        pen_new = (float(inst.g[k]) * max(float(inst.delta[k]) - C_new, 0.0)
                   + float(inst.h[k]) * max(C_new - float(inst.delta[k]), 0.0))
        pen_old = (float(inst.g[k]) * max(float(inst.delta[k]) - C_old, 0.0)
                   + float(inst.h[k]) * max(C_old - float(inst.delta[k]), 0.0))
        delta_pen += pen_new - pen_old
        t_cur = C_new
        prev  = k

    # ── Separation burden and balance terms ───────────────────────────────
    d_sep = inst.h_bar * _delta_sep_raw(j, seq, p, inst)
    d_bal = inst.h_bar * (t_cur - B_bar)     # B'_ρ = t_cur after suffix

    return delta_pen + params.lambda1 * d_sep + params.lambda2 * d_bal


def insertion_cost(
    j:      int,
    seq:    List[int],
    p:      int,
    B_bar:  float,
    inst:   Instance,
    params: HybridParams,
) -> float:
    """
    Δ(j, ρ, p) with full surrogate recomputation (§8).

    Convenience wrapper over insertion_cost_incremental for callers that
    do not cache C_orig.
    """
    C_orig = surrogate_times(seq, inst)
    return insertion_cost_incremental(j, seq, p, C_orig, B_bar, inst, params)


# ─────────────────────────────────────────────────────────────────────────────
#   Best insertion across all runways
# ─────────────────────────────────────────────────────────────────────────────

def best_insertion(
    j:           int,
    seqs:        List[List[int]],
    inst:        Instance,
    params:      HybridParams,
    exclude_rho: Optional[int] = None,
) -> Tuple[float, int, int]:
    """
    Find the minimum-cost feasible insertion of aircraft j across all (ρ, p).

    Returns (best_cost, best_rho, best_p).
    If exclude_rho is given, that runway is skipped (used for regret, §9.3).
    Returns (inf, -1, -1) when no feasible insertion exists.
    """
    m     = len(seqs)
    B_bar = float(np.mean([committed_time(seqs[rho], inst) for rho in range(m)]))

    best_cost = math.inf
    best_rho  = -1
    best_p    = -1

    for rho in range(m):
        if rho == exclude_rho:
            continue
        seq    = seqs[rho]
        C_orig = surrogate_times(seq, inst)
        for p in range(len(seq) + 1):
            c = insertion_cost_incremental(j, seq, p, C_orig, B_bar, inst, params)
            if c < best_cost:
                best_cost = c
                best_rho  = rho
                best_p    = p

    return best_cost, best_rho, best_p