"""
proxy.py — Fast O(n) proxy objective and runway feasibility (§20.1).

Provides surrogate-quality components used in operator screening during
the multi-runway SA, avoiding LP calls on every candidate move.

Components for one runway
─────────────────────────
tc  : Σ max(s[i,j] − (δ_j − δ_i), 0) for consecutive pairs.
      Captures wake-separation conflicts relative to target-time order.
lbt : Σ h_j · max(Ĉ_j − δ_j, 0) — lower-bound tardiness.
sep : Σ s[i,j] for consecutive pairs — raw separation overhead.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from misc.mr_alp_2.config   import EPS
from misc.mr_alp_2.models   import Instance, HeuristicParams
from misc.mr_alp_2.surrogate import surrogate_times


def runway_feasible(seq: List[int], inst: Instance) -> bool:
    """
    Return True iff every surrogate landing time respects the hard deadline.

    Used in operator implementations to screen proposed moves before
    computing proxy costs, avoiding proxy computation on infeasible
    candidates.
    """
    if not seq:
        return True
    C = surrogate_times(seq, inst)
    for q, j in enumerate(seq):
        if C[q] > float(inst.d[j]) + 1e-9:
            return False
    return True


def _rwy_proxy_components(
    seq: List[int], inst: Instance,
) -> Tuple[float, float, float]:
    """
    Proxy objective components for one runway sequence.

    Returns (tc, lbt, sep); all three are non-negative.
    """
    if not seq:
        return 0.0, 0.0, 0.0

    C   = surrogate_times(seq, inst)
    lbt = sum(float(inst.h[j]) * max(C[q] - float(inst.delta[j]), 0.0)
              for q, j in enumerate(seq))
    tc  = 0.0
    sep = 0.0
    for q in range(len(seq) - 1):
        i, j  = seq[q], seq[q + 1]
        s_ij  = float(inst.s[i, j])
        tc   += max(0.0, s_ij - (float(inst.delta[j]) - float(inst.delta[i])))
        sep  += s_ij
    return tc, lbt, sep


def init_proxy_arrays(
    seqs: List[List[int]], inst: Instance,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute (tc_rwy, lbt_rwy, sep_rwy) for all runways, shape (m,).

    Call once before the SA loop and update incrementally after accepted
    moves by recomputing only the affected runway entries.
    """
    m   = len(seqs)
    tc  = np.zeros(m)
    lbt = np.zeros(m)
    sep = np.zeros(m)
    for rho in range(m):
        tc[rho], lbt[rho], sep[rho] = _rwy_proxy_components(seqs[rho], inst)
    return tc, lbt, sep


def compute_proxy(
    seqs:    List[List[int]],
    tc_rwy:  np.ndarray,
    lbt_rwy: np.ndarray,
    sep_rwy: np.ndarray,
    inst:    Instance,
    params:  HeuristicParams,
) -> float:
    """
    Weighted aggregate proxy across all runways.

    proxy = μ_tc·Σtc + μ_late·Σlbt + μ_count·balance_penalty + μ_sep·Σsep

    The balance penalty converts runway-load imbalance into cost units
    using Pen_bar as a scale factor.
    """
    m   = len(seqs)
    n   = inst.n
    t   = sum(len(s) for s in seqs)
    bal = sum((len(seqs[r]) - t / m) ** 2 for r in range(m))
    bs  = float(inst.Pen_bar) / max((n / m) ** 2, 1.0)
    return (params.mu_tc    * float(tc_rwy.sum())
            + params.mu_late  * float(lbt_rwy.sum())
            + params.mu_count * bal * bs
            + params.mu_sep   * float(sep_rwy.sum()))


def lp_impact_scores(
    seqs:  List[List[int]],
    C_lp:  np.ndarray,
    inst:  Instance,
) -> np.ndarray:
    """
    LP-derived impact score for each aircraft: g_j·E_j + h_j·T_j.

    Returns an (n,) array indexed by aircraft index.  Used by
    pick_aircraft_targeted to bias operator selection toward high-penalty
    aircraft after each LP evaluation.
    """
    E = np.maximum(inst.delta - C_lp, 0.0)
    T = np.maximum(C_lp - inst.delta, 0.0)
    return inst.g * E + inst.h * T