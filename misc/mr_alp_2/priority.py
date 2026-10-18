"""
priority.py — Aircraft priority measures for construction (§9).

Implements:
  Regret_j   : Δ^(2)_j − Δ^(1)_j  (§9.3)
  Priority_j : η·Regret̃_j + (1−η)·CR̃_j  (§9.4)
"""
from __future__ import annotations

import math
from typing import Dict, List, Set, Tuple

import numpy as np

from misc.mr_alp_2.models import Instance, HybridParams
from misc.mr_alp_2.surrogate import best_insertion
from misc.mr_alp_2.config import EPS


def _compute_regret_single(
    j:      int,
    seqs:   List[List[int]],
    inst:   Instance,
    params: HybridParams,
) -> Tuple[float, float, int, int]:
    """
    Compute Δ^(1)_j, Δ^(2)_j, best_rho, best_p for aircraft j (§9.3).

    Δ^(1)_j : minimum insertion cost across all (ρ, p).
    Δ^(2)_j : minimum insertion cost excluding the best runway ρ*_j.
    Regret_j : Δ^(2)_j − Δ^(1)_j.
    """
    delta1, rho_star, p_star = best_insertion(j, seqs, inst, params)
    delta2, _, _             = best_insertion(j, seqs, inst, params,
                                              exclude_rho=rho_star)
    return delta1, delta2, rho_star, p_star


def compute_priority_scores(
    remaining: Set[int],
    seqs:      List[List[int]],
    inst:      Instance,
    params:    HybridParams,
) -> Dict[int, Tuple[float, int, int]]:
    """
    Compute Priority_j for all unscheduled aircraft (§9.4).

    Priority_j = η · Regret̃_j + (1 − η) · CR̃_j

    where CR̃ and Regret̃ are min-max normalised to [0, 1].

    Returns
    -------
    dict : j → (priority_score, best_rho, best_p)
        best_rho and best_p are the argmin insertion destination for j
        (used directly by the caller to execute the insertion).
    """
    if not remaining:
        return {}

    rem  = list(remaining)

    # Collect raw values
    regret_raw:   Dict[int, float] = {}
    best_rho_map: Dict[int, int]   = {}
    best_p_map:   Dict[int, int]   = {}

    for j in rem:
        d1, d2, rho, p = _compute_regret_single(j, seqs, inst, params)
        regret_raw[j]  = (d2 - d1) if not math.isinf(d2) else 0.0
        best_rho_map[j] = rho
        best_p_map[j]   = p

    # Min-max normalise CR (§9.4)
    cr_arr  = inst.CR[[j for j in rem]]
    cr_min, cr_max = cr_arr.min(), cr_arr.max()
    cr_norm = (cr_arr - cr_min) / max(cr_max - cr_min, EPS)

    # Min-max normalise Regret (§9.4)
    reg_arr  = np.array([regret_raw[j] for j in rem])
    reg_min, reg_max = reg_arr.min(), reg_arr.max()
    reg_norm = (reg_arr - reg_min) / max(reg_max - reg_min, EPS)

    eta    = params.eta
    result: Dict[int, Tuple[float, int, int]] = {}
    for idx, j in enumerate(rem):
        prio       = eta * float(reg_norm[idx]) + (1.0 - eta) * float(cr_norm[idx])
        result[j]  = (prio, best_rho_map[j], best_p_map[j])

    return result