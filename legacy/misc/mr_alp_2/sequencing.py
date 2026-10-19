"""
sequencing.py — Phase 2: Per-runway initial sequence generation (§12).

Given a fixed runway assignment (ordered or unordered list of aircraft
indices per runway), five ordering rules are tried and the one with the
lowest surrogate penalty is returned.

Rules:
  EDD   : nondecreasing latest landing time d_j.
  ERD   : nondecreasing earliest landing time r_j.
  TARGET: nondecreasing target time δ_j.
  MPDS  : exponential priority greedy (single-runway variant).
  WAKE  : minimum consecutive separation greedy.
"""
from __future__ import annotations

import math
from typing import List

import numpy as np

from misc.mr_alp_2.models import Instance
from misc.mr_alp_2.surrogate import surrogate_times, surrogate_penalty
from misc.mr_alp_2.config import EPS


def _seq_edd(aircraft: List[int], inst: Instance) -> List[int]:
    return sorted(aircraft, key=lambda j: float(inst.d[j]))


def _seq_erd(aircraft: List[int], inst: Instance) -> List[int]:
    return sorted(aircraft, key=lambda j: float(inst.r[j]))


def _seq_target(aircraft: List[int], inst: Instance) -> List[int]:
    return sorted(aircraft, key=lambda j: float(inst.delta[j]))


def _seq_mpds(aircraft: List[int], inst: Instance) -> List[int]:
    """
    Single-runway MPDS greedy sequence.

    MPDS index: exp(−slack/K1) · exp(−s[k,j]/(K2·s̄)) ·
                exp(−wait/K3)  · exp(−pen/K4)

    Parameters K1..K4 are computed from the subset statistics, following
    the formulation in Beasley et al. (2000) §4 and the single-runway
    adaptation in the existing codebase.
    """
    if not aircraft:
        return []
    rem  = list(aircraft)
    sb   = inst.s_bar
    d_arr = inst.delta[rem]
    d_max, d_min = float(d_arr.max()), float(d_arr.min())
    C_max = max(d_min + len(rem) * sb, d_max)
    R     = (d_max - d_min) / C_max if C_max > 0 else 0.0
    K1    = 4.5 + R if R <= 0.5 else 6.0 - 2.0 * R
    K2    = max((1.0 - float(inst.delta[rem].sum()) / (len(rem) * C_max))
                / (2.0 * sb), EPS) if sb > 0 else 1.0
    K3    = max((float(inst.r[rem].max()) - float(inst.r[rem].min()))
                / sb, EPS) if len(rem) > 1 and sb > 0 else 1.0
    K4    = max(float((inst.g[rem] + inst.h[rem]).mean()), EPS)

    seq  : List[int] = []
    t    = 0.0
    last = -1

    while rem:
        best_score = -math.inf
        best_j     = rem[0]
        for j in rem:
            s_kj  = float(inst.s[last, j]) if last >= 0 else 0.0
            t_hat = max(float(inst.r[j]), t + s_kj)
            slack = max(float(inst.delta[j]) - t_hat, 0.0)
            wait  = max(float(inst.r[j]) - t, 0.0)
            pen   = (float(inst.g[j]) * max(float(inst.delta[j]) - t_hat, 0.0)
                     + float(inst.h[j]) * max(t_hat - float(inst.delta[j]), 0.0))
            score = (math.exp(-slack / (K1 + EPS))
                     * math.exp(-s_kj  / (K2 * sb + EPS))
                     * math.exp(-wait  / (K3 + EPS))
                     * math.exp(-pen   / (K4 + EPS)))
            if score > best_score:
                best_score = score
                best_j     = j
        seq.append(best_j)
        s_kj = float(inst.s[last, best_j]) if last >= 0 else 0.0
        t    = max(float(inst.r[best_j]), t + s_kj)
        last = best_j
        rem.remove(best_j)

    return seq


def _seq_wake(aircraft: List[int], inst: Instance) -> List[int]:
    """
    Minimum consecutive separation greedy (§12).
    At each step, select the unscheduled aircraft minimising s[last, j].
    """
    if not aircraft:
        return []
    rem  = list(aircraft)
    seq  : List[int] = []
    last = -1

    while rem:
        if last < 0:
            j_star = min(rem, key=lambda j: float(inst.s[j, :].max()))
        else:
            j_star = min(rem, key=lambda j: float(inst.s[last, j]))
        seq.append(j_star)
        last = j_star
        rem.remove(j_star)

    return seq


def best_initial_sequence(aircraft: List[int], inst: Instance) -> List[int]:
    """
    Generate sequences by all five rules; return the one with minimum
    surrogate penalty (§12).
    """
    if not aircraft:
        return []
    candidates = [
        _seq_edd(aircraft, inst),
        _seq_erd(aircraft, inst),
        _seq_target(aircraft, inst),
        _seq_mpds(aircraft, inst),
        _seq_wake(aircraft, inst),
    ]
    return min(candidates, key=lambda seq: surrogate_penalty(seq, inst))


def generate_runway_sequences(
    seqs: List[List[int]], inst: Instance
) -> List[List[int]]:
    """
    Phase 2 entry point: apply best_initial_sequence to each runway (§12).
    The input seqs[rho] may already be ordered from Phase 1; this function
    re-evaluates all five orderings and picks the surrogate-best.
    """
    return [best_initial_sequence(list(seqs[rho]), inst)
            for rho in range(inst.m)]