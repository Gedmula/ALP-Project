"""
=============================================================================
proxy.py — MR-ALP Solver: Surrogate Proxy Objective and Per-Aircraft Scoring
=============================================================================
§10  total_target_conflict  — full pairwise TC sum (GPU when available)
§14  Proxy component arrays and compute_proxy
§15  Per-aircraft impact scoring for targeted operator selection

The proxy objective F̂ is used for SA Metropolis acceptance decisions only.
It is a fast surrogate for the LP objective and must NOT be compared against
LP values or BKS references.  All gap reporting uses the LP objective.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from mr_alp.models    import Instance, HeuristicParams
from mr_alp.instance  import surrogate_times
from mr_alp.config    import USE_GPU, GPU_MIN_N

try:
    import torch as _torch
    _GPU_AVAIL   = _torch.cuda.is_available()
    _CUDA_DEVICE = _torch.device("cuda") if _GPU_AVAIL else None
except ImportError:
    _torch = None; _GPU_AVAIL = False; _CUDA_DEVICE = None


# ═══════════════════════════════════════════════════════════════════════════
#   §10  TOTAL TARGET-CONFLICT  (GPU-accelerated for large instances)
# ═══════════════════════════════════════════════════════════════════════════

def total_target_conflict(sequences: List[List[int]], inst: Instance) -> float:
    """
    Compute total pairwise target-time conflict across all runways.

    TC = Σ_{rho} Σ_{i≺j on rho}  0.5·(p_i+p_j) · max(s[i,j]−(δ_j−δ_i), 0)

    Uses PyTorch CUDA when USE_GPU=True, a GPU is available, and n ≥ GPU_MIN_N.
    Falls back to NumPy vectorised computation otherwise.

    This quantity is the Optuna tuning objective for instances with n > 100
    (where calling the LP for every trial is too expensive).
    """
    if USE_GPU and _GPU_AVAIL and _torch is not None and inst.n >= GPU_MIN_N:
        total = 0.0
        for seq in sequences:
            L = len(seq)
            if L < 2:
                continue
            sa = np.asarray(seq, dtype=np.intp)
            ii, jj = np.triu_indices(L, k=1)
            i_t = _torch.from_numpy(sa[ii].astype(np.int64)).to(_CUDA_DEVICE)
            j_t = _torch.from_numpy(sa[jj].astype(np.int64)).to(_CUDA_DEVICE)
            v   = (inst._s_gpu[i_t, j_t]
                   - (inst._delta_gpu[j_t] - inst._delta_gpu[i_t]))
            total += float(
                _torch.sum(
                    0.5 * (inst._p_arr_gpu[i_t] + inst._p_arr_gpu[j_t])
                    * _torch.clamp(v, min=0.0)
                )
            )
        return total

    total = 0.0
    for seq in sequences:
        L = len(seq)
        if L < 2:
            continue
        sa = np.asarray(seq, dtype=np.intp)
        ii, jj = np.triu_indices(L, k=1)
        v = inst.s[sa[ii], sa[jj]] - (inst.delta[sa[jj]] - inst.delta[sa[ii]])
        total += float(
            (0.5 * (inst.p_arr[sa[ii]] + inst.p_arr[sa[jj]])
             * np.maximum(v, 0.0)).sum()
        )
    return total


# ═══════════════════════════════════════════════════════════════════════════
#   §14  PROXY COMPONENT ARRAYS AND AGGREGATE COMPUTE
# ═══════════════════════════════════════════════════════════════════════════

def _rwy_proxy_components(
    seq: List[int], inst: Instance
) -> Tuple[float, float, float]:
    """
    Compute (tc, lbt, sep) proxy components for one runway sequence.

    tc  : pairwise target-conflict sum for this runway.
    lbt : surrogate tardiness lower bound Σ h_j · max(C̃_j − δ_j, 0).
    sep : h̄ · Σ_{consecutive} s[seq[q-1], seq[q]]  (separation burden).
    """
    if not seq:
        return 0.0, 0.0, 0.0
    L      = len(seq)
    s_arr  = np.asarray(seq, dtype=np.intp)

    if L >= 2:
        ii, jj = np.triu_indices(L, k=1)
        i_ac   = s_arr[ii]; j_ac = s_arr[jj]
        v      = inst.s[i_ac, j_ac] - (inst.delta[j_ac] - inst.delta[i_ac])
        tc     = float(
            (0.5 * (inst.p_arr[i_ac] + inst.p_arr[j_ac])
             * np.maximum(v, 0.0)).sum()
        )
    else:
        tc = 0.0

    C_hat = np.asarray(surrogate_times(seq, inst))
    lbt   = float((inst.h[s_arr] * np.maximum(C_hat - inst.delta[s_arr], 0.0)).sum())
    sep   = (float(inst.s[s_arr[:-1], s_arr[1:]].sum()) * inst.h_bar
             if L >= 2 else 0.0)

    return tc, lbt, sep


def _balance_term(seqs: List[List[int]], inst: Instance) -> float:
    """
    Runway-balance penalty: deviation of per-runway load from mean load n/m,
    normalised by the expected squared deviation (n/m)².
    """
    n = inst.n; m = len(seqs)
    return (
        sum((len(seqs[r]) - n / m) ** 2 for r in range(m))
        * float(inst.Pen_bar) / max((n / m) ** 2, 1.0)
    )


def init_proxy_arrays(
    seqs: List[List[int]], inst: Instance
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialise per-runway proxy component arrays (tc_rwy, lbt_rwy, sep_rwy).
    Call once per SA solution; update only affected runways after each move.
    """
    m = len(seqs)
    tc_rwy = np.zeros(m); lbt_rwy = np.zeros(m); sep_rwy = np.zeros(m)
    for rho in range(m):
        tc_rwy[rho], lbt_rwy[rho], sep_rwy[rho] = _rwy_proxy_components(
            seqs[rho], inst)
    return tc_rwy, lbt_rwy, sep_rwy


def compute_proxy(
    seqs: List[List[int]],
    tc_rwy: np.ndarray,
    lbt_rwy: np.ndarray,
    sep_rwy: np.ndarray,
    inst: Instance,
    params: HeuristicParams,
) -> float:
    """
    Aggregate proxy objective F̂ = μ_TC·TC + μ_late·LBT + μ_count·Bal + μ_sep·Sep.
    """
    return (
        params.mu_tc    * float(tc_rwy.sum())
        + params.mu_late  * float(lbt_rwy.sum())
        + params.mu_count * _balance_term(seqs, inst)
        + params.mu_sep   * float(sep_rwy.sum())
    )


# ═══════════════════════════════════════════════════════════════════════════
#   §15  PER-AIRCRAFT IMPACT SCORING
# ═══════════════════════════════════════════════════════════════════════════

def compute_per_aircraft_scores(
    seqs: List[List[int]], inst: Instance
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-aircraft target-conflict (pa_tc) and tardiness-lower-bound
    (pa_lbt) scores used by the targeted aircraft selector.

    Returns (pa_tc, pa_lbt), each of shape (n,).
    """
    n = inst.n
    pa_tc  = np.zeros(n)
    pa_lbt = np.zeros(n)

    for seq in seqs:
        if not seq:
            continue
        L     = len(seq)
        s_arr = np.asarray(seq, dtype=np.intp)
        if L >= 2:
            ii, jj  = np.triu_indices(L, k=1)
            i_ac    = s_arr[ii]; j_ac = s_arr[jj]
            v       = inst.s[i_ac, j_ac] - (inst.delta[j_ac] - inst.delta[i_ac])
            c       = (0.5 * (inst.p_arr[i_ac] + inst.p_arr[j_ac])
                       * np.maximum(v, 0.0))
            np.add.at(pa_tc, i_ac, c)
            np.add.at(pa_tc, j_ac, c)
        C_hat         = np.asarray(surrogate_times(seq, inst))
        pa_lbt[s_arr] = inst.h[s_arr] * np.maximum(C_hat - inst.delta[s_arr], 0.0)

    return pa_tc, pa_lbt


def lp_impact_scores(
    seqs: List[List[int]],
    C_lp: np.ndarray,
    inst: Instance,
    lambda_b: float = 0.5,
    eps_tight: float = 1e-4,
) -> np.ndarray:
    """
    Combined LP-impact score for aircraft selection in LP-guided repairs.

    Impact_j = P_j + λ_b · binding_count_j

    P_j           : LP penalty  g_j·E_j + h_j·T_j.
    binding_count_j : number of separation constraints C_j − C_i = s[i,j]
                      (within eps_tight) in which j participates.

    Higher-impact aircraft are prioritised for relocation and ejection.
    """
    n = inst.n
    E = np.maximum(inst.delta - C_lp, 0.0)
    T = np.maximum(C_lp - inst.delta, 0.0)
    P = inst.g * E + inst.h * T
    binding = np.zeros(n)

    for seq in seqs:
        L = len(seq)
        for qi in range(L):
            for qj in range(qi + 1, L):
                i, j = seq[qi], seq[qj]
                if C_lp[j] - C_lp[i] - inst.s[i, j] <= eps_tight:
                    binding[i] += 1.0
                    binding[j] += 1.0

    return P + lambda_b * binding


