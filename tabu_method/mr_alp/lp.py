"""
lp.py — MR-ALP Solver: LP Objective and Exact Feasibility Verification
=======================================================================
§12  stage2_lp_objective  — HiGHS-backed LP for fixed landing sequences
§13  verify_and_exact_obj — earliest-time propagation + constraint audit

Correctness note
----------------
OR Library separation matrices violate the triangle inequality.  Both
functions enforce ALL ordered pairs (i, j) with i appearing before j on
the same runway — giving n(n-1)/2 constraints per runway, not O(n).  Any
solver that uses only consecutive-predecessor constraints will produce
infeasible or sub-optimal results on these instances.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

from mr_alp.models import Instance


# ═══════════════════════════════════════════════════════════════════════════
#   §12  STAGE-2 LP: EXACT LANDING-TIME OPTIMISATION
# ═══════════════════════════════════════════════════════════════════════════

def stage2_lp_objective(
    sequences: List[List[int]],
    inst: Instance,
    eps_tol: float = 1e-6,
) -> Tuple[float, Optional[np.ndarray], bool, List[str]]:
    """
    Minimise total weighted earliness / tardiness for fixed landing sequences.

    Formulation
    -----------
    Variables : C_j ∈ [r_j, d_j]  (landing time)
                E_j ≥ 0             (earliness)
                T_j ≥ 0             (tardiness)

    Minimize  : Σ_j  g_j · E_j + h_j · T_j

    Subject to:
      C1  C_j − E_j ≥ δ_j     (for all j)
      C2  C_j + T_j ≥ δ_j     (for all j; direction reversed in ≤ form)
      C3  C_j − C_i ≥ s[i,j]  (for ALL ordered pairs i≺j on the same runway)
      C4  r_j ≤ C_j ≤ d_j     (via variable bounds)

    Constraint set C3 has O(n²) entries (not O(n)) because the separation
    matrix does not satisfy the triangle inequality.

    Parameters
    ----------
    sequences : list of per-runway aircraft index sequences.
    inst      : parsed Instance.
    eps_tol   : violation tolerance for post-solve audit.

    Returns
    -------
    (obj, C_lp, feasible, violations)
    obj        : optimal LP objective; math.inf if infeasible.
    C_lp       : optimal landing times array (shape n,); None if infeasible.
    feasible   : True iff solver reports success and no bound/sep violations.
    violations : list of violation description strings (empty when feasible).
    """
    n  = inst.n
    C0 = 0;  E0 = n;  T0 = 2 * n;  nv = 3 * n

    c_obj = np.zeros(nv)
    c_obj[E0:E0 + n] = inst.g
    c_obj[T0:T0 + n] = inst.h

    # All ordered pairs (i,j) where i precedes j on the same runway
    sep_pairs = [
        (seq[a], seq[b])
        for seq in sequences
        for a in range(len(seq))
        for b in range(a + 1, len(seq))
    ]
    n_ineq = 2 * n + len(sep_pairs)
    rows, cols, vals = [], [], []
    b_ub = np.empty(n_ineq)
    r = 0

    # C1: −C_j − E_j ≤ −δ_j
    for j in range(n):
        rows += [r, r]; cols += [C0 + j, E0 + j]; vals += [-1., -1.]
        b_ub[r] = -float(inst.delta[j]); r += 1

    # C2: C_j − T_j ≤ δ_j
    for j in range(n):
        rows += [r, r]; cols += [C0 + j, T0 + j]; vals += [1., -1.]
        b_ub[r] = float(inst.delta[j]); r += 1

    # C3: C_i − C_j ≤ −s[i,j]
    for i, j in sep_pairs:
        rows += [r, r]; cols += [C0 + i, C0 + j]; vals += [1., -1.]
        b_ub[r] = -float(inst.s[i, j]); r += 1

    A_ub = csr_matrix((vals, (rows, cols)), shape=(n_ineq, nv))
    bounds = ([(float(inst.r[j]), float(inst.d[j])) for j in range(n)]
              + [(0., None)] * (2 * n))

    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if not res.success:
        return math.inf, None, False, [f"LP solver: {res.message}"]

    C_lp = res.x[C0:C0 + n]
    obj  = float(res.fun)
    viol: List[str] = []

    for j in range(n):
        if C_lp[j] < inst.r[j] - eps_tol:
            viol.append(f"Ac {j}: C={C_lp[j]:.4f} < r={inst.r[j]:.4f}")
        if C_lp[j] > inst.d[j] + eps_tol:
            viol.append(f"Ac {j}: C={C_lp[j]:.4f} > d={inst.d[j]:.4f}")

    for seq in sequences:
        for a in range(len(seq)):
            for b in range(a + 1, len(seq)):
                i, j = seq[a], seq[b]
                gap = C_lp[j] - C_lp[i]
                if gap < inst.s[i, j] - eps_tol:
                    viol.append(
                        f"sep({i},{j}): {gap:.4f} < {inst.s[i,j]:.4f}")

    return obj, C_lp, len(viol) == 0, viol


# ═══════════════════════════════════════════════════════════════════════════
#   §13  EXACT FEASIBILITY VERIFICATION + EARLIEST-TIME OBJECTIVE
# ═══════════════════════════════════════════════════════════════════════════

def verify_and_exact_obj(
    sequences: List[List[int]],
    inst: Instance,
    eps_tol: float = 1e-6,
) -> Tuple[bool, List[str], float, dict]:
    """
    Compute earliest feasible landing times via full pairwise propagation,
    evaluate the penalty objective, and audit all constraint groups.

    Constraint groups audited
    -------------------------
    C1  All n aircraft are scheduled exactly once.
    C2  C_j ≥ r_j  and  C_j ≤ d_j  for all j.
    C3  C_j − C_i ≥ s[i,j]  for ALL ordered pairs i≺j (O(n²) per runway).

    Parameters
    ----------
    sequences : list of per-runway aircraft index sequences.
    inst      : parsed Instance.
    eps_tol   : constraint violation tolerance.

    Returns
    -------
    (feasible, violations, obj, C_exact)
    feasible   : True iff all constraint groups pass.
    violations : list of violation description strings.
    obj        : weighted earliness/tardiness objective under exact times.
    C_exact    : dict mapping aircraft index → exact landing time.
    """
    C_exact: dict = {}

    for rho, seq in enumerate(sequences):
        if not seq:
            continue
        L    = len(seq)
        C_r  = [0.0] * L
        C_r[0] = float(inst.r[seq[0]])
        for q in range(1, L):
            j  = seq[q]
            t  = float(inst.r[j])
            for h in range(q):
                t = max(t, C_r[h] + float(inst.s[seq[h], j]))
            C_r[q] = t
        for q, j in enumerate(seq):
            C_exact[j] = C_r[q]

    viol: List[str] = []

    # C1: coverage
    for j in range(inst.n):
        if j not in C_exact:
            viol.append(f"Aircraft {j} not scheduled")

    # C2: window bounds
    for j, Cj in C_exact.items():
        if Cj < inst.r[j] - eps_tol:
            viol.append(f"Ac {j}: C={Cj:.2f} < r={inst.r[j]:.2f}")
        if Cj > inst.d[j] + eps_tol:
            viol.append(f"Ac {j}: C={Cj:.2f} > d={inst.d[j]:.2f}")

    # C3: all ordered-pair separations
    for rho, seq in enumerate(sequences):
        for qi in range(len(seq)):
            for qj in range(qi + 1, len(seq)):
                i, j   = seq[qi], seq[qj]
                Ci, Cj = C_exact.get(i, 0.0), C_exact.get(j, 0.0)
                if Cj - Ci < inst.s[i, j] - eps_tol:
                    viol.append(
                        f"Rwy{rho+1} sep({i},{j}): "
                        f"{Cj-Ci:.4f} < {inst.s[i,j]:.4f}")

    obj = sum(
        float(inst.g[j]) * max(float(inst.delta[j]) - Cj, 0.0)
        + float(inst.h[j]) * max(Cj - float(inst.delta[j]), 0.0)
        for j, Cj in C_exact.items()
    )

    return len(viol) == 0, viol, obj, C_exact