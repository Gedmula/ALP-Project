"""
lp.py — Phase 4: Stage-2 LP for fixed multi-runway sequences (§11).

Variables  : C_j ∈ [r_j, d_j], E_j ≥ 0, T_j ≥ 0  for all j ∈ J.
Objective  : min Σ_j (g_j E_j + h_j T_j).
Separation : C_j ≥ C_i + s[i,j]  for all i ≺_ρ j on runway ρ,
             enforced for ALL predecessor pairs (not only consecutive),
             as required when the triangle inequality is violated
             (Beasley et al. 2000, §2.2, Eqs. 7/12).
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog

from misc.mr_alp_2.models import Instance


# ─────────────────────────────────────────────────────────────────────────────
#   LP matrix construction
# ─────────────────────────────────────────────────────────────────────────────

def _build_lp(
    seqs: List[List[int]], inst: Instance
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Build the Stage-2 LP coefficient structures for fixed multi-runway sequences.

    Decision vector: [C_0, …, C_{n-1}, E_0, …, E_{n-1}, T_0, …, T_{n-1}]
    Length: 3n.

    Separation constraints: for each runway ρ and each ordered pair (l, m)
    with l appearing before m in π_ρ:
        C_{seq[m]} − C_{seq[l]} ≥ s[seq[l], seq[m]]
      →  C_{seq[l]} − C_{seq[m]} ≤ −s[seq[l], seq[m]]    (A_ub row form)

    Earliness: E_j ≥ δ_j − C_j  →  −C_j − E_j ≤ −δ_j
    Tardiness: T_j ≥ C_j − δ_j  →   C_j − T_j ≤  δ_j

    Returns
    -------
    c   : (3n,) cost vector.
    A   : (n_constraints, 3n) inequality matrix.
    b   : (n_constraints,) RHS vector.
    bnd : list of (lo, hi) variable bounds length 3n.
    """
    n   = inst.n
    c   = np.concatenate([np.zeros(n), inst.g, inst.h])
    bnd = ([(float(inst.r[j]), float(inst.d[j])) for j in range(n)]
           + [(0.0, None)] * 2 * n)

    rows_A: list = []
    rows_b: list = []

    # Full pairwise separation per runway
    for rho in range(inst.m):
        seq = seqs[rho]
        L   = len(seq)
        for l_idx in range(L - 1):
            for m_idx in range(l_idx + 1, L):
                row             = np.zeros(3 * n)
                row[seq[l_idx]] =  1.0
                row[seq[m_idx]] = -1.0
                rows_A.append(row)
                rows_b.append(-float(inst.s[seq[l_idx], seq[m_idx]]))

    # Earliness linearisation
    for j in range(n):
        row = np.zeros(3 * n)
        row[j]     = -1.0
        row[n + j] = -1.0
        rows_A.append(row)
        rows_b.append(-float(inst.delta[j]))

    # Tardiness linearisation
    for j in range(n):
        row = np.zeros(3 * n)
        row[j]         =  1.0
        row[2 * n + j] = -1.0
        rows_A.append(row)
        rows_b.append(float(inst.delta[j]))

    A = np.array(rows_A, dtype=float)
    b = np.array(rows_b, dtype=float)
    return c, A, b, bnd


# ─────────────────────────────────────────────────────────────────────────────
#   Public LP interface
# ─────────────────────────────────────────────────────────────────────────────

def stage2_lp(
    seqs: List[List[int]], inst: Instance
) -> Tuple[float, Optional[np.ndarray]]:
    """
    Solve the Stage-2 LP for fixed multi-runway sequences.

    Returns
    -------
    (obj, landing_times)  on success.
    (inf, None)           if the LP is infeasible or the solver fails.
    """
    c, A, b, bnd = _build_lp(seqs, inst)
    res = linprog(c, A_ub=A, b_ub=b, bounds=bnd,
                  method='highs',
                  options={'disp': False, 'presolve': True})
    if res.status != 0:
        return math.inf, None
    return float(res.fun), res.x[: inst.n]


def stage2_lp_objective(seqs: List[List[int]], inst: Instance) -> float:
    """Return the LP objective scalar only (convenience wrapper)."""
    obj, _ = stage2_lp(seqs, inst)
    return obj


def lp_recheck(
    seqs: List[List[int]], inst: Instance
) -> Tuple[float, Optional[np.ndarray]]:
    """
    Independent LP re-solve with tightened primal and dual tolerances.

    Used exclusively by verify.py (§18) as a cross-check (constraint C9).
    Uses an identical constraint matrix to stage2_lp so that any objective
    discrepancy reflects a genuine solver inconsistency rather than a
    constraint-set difference.
    """
    c, A, b, bnd = _build_lp(seqs, inst)
    res = linprog(c, A_ub=A, b_ub=b, bounds=bnd,
                  method='highs',
                  options={
                      'disp':                           False,
                      'presolve':                       True,
                      'dual_feasibility_tolerance':     1e-9,
                      'primal_feasibility_tolerance':   1e-9,
                  })
    if res.status != 0:
        return math.inf, None
    return float(res.fun), res.x[: inst.n]