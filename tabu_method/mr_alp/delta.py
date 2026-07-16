"""
delta.py — MR-ALP Solver: Staged Delta Evaluation for Insertion Scans
======================================================================
Replaces the O(L³) full-scan position search inside N3b / X3 (and therefore
X7 / XE, which delegate to them) with a two-stage search:

Stage 1 — vectorised scoring of ALL L+1 insertion positions in O(L) total:
    score1(q) = mu_tc · ΔTC(q) + mu_sep · h̄ · Δsep(q)
  ΔTC(q) is exact: only pairs involving the moved aircraft change, and a
  prefix/suffix cumsum gives every position at once.  Δsep(q) is exact and
  O(1) per position (one consecutive edge is replaced by two).  The lbt
  term is deferred to stage 2 (it requires landing times).  Terms that are
  constant across q (source-runway components, X3's balance term) cancel
  out of the ranking and are omitted.

Stage 2 — exact evaluation of positions in stage-1 order:
    1. surrogate-time insertion check (O(L), numba kernel when available)
       — surrogate times are lower bounds, so surrogate-infeasible implies
       truly infeasible: a sound fast reject;
    2. full pairwise feasibility check (O(L²), required because s violates
       the triangle inequality);
    3. surrogate lbt from the stage-2 times — the same approximation that
       _rwy_proxy_components uses, so the final ranking metric matches the
       legacy full scan:  full(q) = score1(q) + mu_late · lbt(q).
  Stops after DELTA_FINALISTS_K fully-feasible positions have been scored.

Behaviour note: results can differ from the legacy scan when the best
position by lbt is not among the top-k by score1.  This is an accepted,
flag-gated (config.DELTA_EVAL) algorithm change — A/B it, don't assume.

Cost per operator call: O(L) + k·O(L²)  vs  legacy 2L·O(L²).
"""
from __future__ import annotations

import math
from typing import List

import numpy as np

from mr_alp.config       import DELTA_FINALISTS_K
from mr_alp.models       import Instance, HeuristicParams
from mr_alp.instance     import runway_feasible, surrogate_times
from mr_alp.construction import _compute_insert_times


def _stage1_scores(
    base: np.ndarray, ac: int, inst: Instance, params: HeuristicParams,
) -> np.ndarray:
    """
    Exact ΔTC and Δsep for inserting ac at every position of base, O(L).

    base : aircraft-id array of the runway sequence WITHOUT ac, length L.
    Returns score array of length L+1 (position q = insert before base[q]).
    """
    L = int(base.shape[0])
    if L == 0:
        return np.zeros(1)

    d_ac = float(inst.delta[ac])
    p_ac = float(inst.p_arr[ac])
    dj   = inst.delta[base]
    w    = 0.5 * (inst.p_arr[base] + p_ac)

    # Pair (j, ac): j lands before ac  → conflict if s[j,ac] > δ_ac − δ_j.
    cb = w * np.maximum(inst.s[base, ac] - (d_ac - dj), 0.0)
    # Pair (ac, j): ac lands before j  → conflict if s[ac,j] > δ_j − δ_ac.
    ca = w * np.maximum(inst.s[ac, base] - (dj - d_ac), 0.0)

    pre = np.empty(L + 1); pre[0] = 0.0
    np.cumsum(cb, out=pre[1:])                    # pre[q] = Σ cb[:q]
    suf = np.empty(L + 1); suf[L] = 0.0
    suf[:L] = np.cumsum(ca[::-1])[::-1]           # suf[q] = Σ ca[q:]
    tc_delta = pre + suf

    sep_delta = np.empty(L + 1)
    s_in  = inst.s[ac, base]                      # ac immediately before j
    s_out = inst.s[base, ac]                      # j immediately before ac
    sep_delta[0] = s_in[0]
    sep_delta[L] = s_out[L - 1]
    if L > 1:
        sep_delta[1:L] = (s_out[:L - 1] + s_in[1:]
                          - inst.s[base[:L - 1], base[1:]])

    return params.mu_tc * tc_delta + params.mu_sep * inst.h_bar * sep_delta


def best_insertion_position(
    sm: List[int], ac: int,
    inst: Instance, params: HeuristicParams,
    k: int = DELTA_FINALISTS_K,
) -> int:
    """
    Best feasible insertion position for ac into sequence sm (ac excluded).

    Walks positions in stage-1 score order; scores the first k positions
    that pass full pairwise feasibility; returns the best of those by
    score1 + mu_late·lbt.  Returns -1 if no feasible position exists.
    """
    L       = len(sm)
    base    = np.asarray(sm, dtype=np.intp) if L else np.empty(0, dtype=np.intp)
    scores1 = _stage1_scores(base, ac, inst, params)
    order   = np.argsort(scores1, kind="stable")
    C_prev  = surrogate_times(sm, inst)

    best_q, best_full, n_scored = -1, math.inf, 0
    for q in order:
        q = int(q)
        # Fast sound reject: surrogate times are lower bounds.
        C_new, sur_ok = _compute_insert_times(ac, q, sm, C_prev, inst)
        if not sur_ok:
            continue
        ns = sm[:q] + [ac] + sm[q:]
        if not runway_feasible(ns, inst):
            continue
        ids = np.insert(base, q, ac)
        lbt = float((inst.h[ids]
                     * np.maximum(np.asarray(C_new) - inst.delta[ids],
                                  0.0)).sum())
        full = float(scores1[q]) + params.mu_late * lbt
        if full < best_full:
            best_full, best_q = full, q
        n_scored += 1
        if n_scored >= k:
            break
    return best_q
