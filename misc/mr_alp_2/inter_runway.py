"""
inter_runway.py — Phase 5: Multi-runway dual-track SA improvement (§14–§21).

Architecture
────────────
The previous implementation called single-runway SA after every inter-runway
move, spending the entire time budget on ~200 expensive iterations while the
LP objective never improved.  This rewrite follows the dual-track SA design:

  Proxy track   : cheap O(n) proxy acceptance on every iteration.
  LP incumbent  : exact Stage-2 LP called only every Nfull iterations
                  (adaptive: 20 early / 50 mid / 100 late) or when
                  proxy falls below best proxy seen so far.

This allows thousands of proxy iterations in the same wall time, yielding
dramatically more exploration.  A post-SA LP-VND polish further tightens
the incumbent before returning.

Adaptive move-pressure weights (§16.6) are updated at each LP improvement
event, reinforcing the component that was the primary driver of the aircraft
that triggered the improvement.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import replace
from typing import List, Optional, Tuple

import numpy as np

from misc.mr_alp_2.config      import EPS, ADAPT_W_XI, ADAPT_W_BETA
from misc.mr_alp_2.models      import Instance, HybridParams, SingleSAParams, HeuristicParams, MRSAParams
from misc.mr_alp_2.lp          import stage2_lp
from misc.mr_alp_2.sa_single   import run_single_runway_sa
from misc.mr_alp_2.proxy       import (
    init_proxy_arrays, compute_proxy, lp_impact_scores,
)
from misc.mr_alp_2.operators   import generate_candidate_pool
from misc.mr_alp_2.repair      import ElitePool, lp_vnd_polish, path_relink
from misc.mr_alp_2.move_pressure import (
    compute_move_pressure, AdaptiveWeightUpdater,
)


# ─────────────────────────────────────────────────────────────────────────────
#   Temperature calibration
# ─────────────────────────────────────────────────────────────────────────────

def _calibrate_proxy_T(
    seqs:      List[List[int]],
    inst:      Instance,
    h_params:  HeuristicParams,
    mr_sa_p:   MRSAParams,
    rng:       random.Random,
    T_fallback: float,
) -> float:
    """
    Sample 120 random moves and calibrate T so average worsening proxy
    moves are accepted with probability chi0.

    Returns T_fallback when no worsening moves are observed.
    """
    tc_rwy, lbt_rwy, sep_rwy = init_proxy_arrays(seqs, inst)
    proxy_curr = compute_proxy(seqs, tc_rwy, lbt_rwy, sep_rwy, inst, h_params)
    deltas: List[float] = []

    for _ in range(120):
        pool = generate_candidate_pool(
            0.1, seqs, tc_rwy, lbt_rwy, sep_rwy,
            inst, h_params, mr_sa_p, rng, 0, 1000, 1,
            None, None, None, None,
        )
        if not pool:
            continue
        d = pool[0][0] - proxy_curr
        if d > 0:
            deltas.append(d)

    if not deltas:
        return T_fallback
    return max(
        -float(np.mean(deltas)) / math.log(max(mr_sa_p.chi0, 1e-12)),
        T_fallback * 0.01,
    )


# ─────────────────────────────────────────────────────────────────────────────
#   Adaptive LP call interval
# ─────────────────────────────────────────────────────────────────────────────

def _nfull(f: float, mr_sa_p: MRSAParams) -> int:
    if   f < 0.20: return mr_sa_p.lp_interval_early
    elif f < 0.80: return mr_sa_p.lp_interval_mid
    return mr_sa_p.lp_interval_late


# ─────────────────────────────────────────────────────────────────────────────
#   Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def inter_runway_improve(
    seqs_init:  List[List[int]],
    lp_init:    float,
    x_init:     Optional[np.ndarray],
    inst:        Instance,
    params:      HybridParams,
    sa_p:        SingleSAParams,
    T_outer:     float,
    n_outer:     int                    = 200,
    seed:        int                    = 0,
    h_params:    Optional[HeuristicParams] = None,
    mr_sa_p:     Optional[MRSAParams]      = None,
) -> Tuple[List[List[int]], float, Optional[np.ndarray]]:
    """
    Dual-track multi-runway SA improvement with LP-VND post-processing.

    The number of proxy iterations is max(n_outer × 25, 5000) so the caller
    can scale effort via n_outer without changing internal SA parameters.
    Signature is backward-compatible with the previous implementation.

    Parameters
    ----------
    seqs_init  : incumbent multi-runway sequences entering Phase 5.
    lp_init    : LP objective of the incumbent.
    x_init     : LP landing times of the incumbent.
    inst       : preprocessed Instance.
    params     : HybridParams (move-pressure weights, B_max, alpha_outer, K).
    sa_p       : SingleSAParams used for within-runway SA in post-processing.
    T_outer    : initial SA temperature (caller-supplied; re-calibrated here
                 if the proxy calibration succeeds).
    n_outer    : controls proxy SA budget; N_iter = max(n_outer × 25, 5000).
    seed       : master RNG seed.
    h_params   : HeuristicParams for proxy weights; defaults to HeuristicParams().
    mr_sa_p    : MRSAParams for SA schedule; defaults to MRSAParams().

    Returns
    -------
    (best_seqs, best_lp, best_x)
    """
    if h_params is None:
        h_params = HeuristicParams()
    if mr_sa_p is None:
        mr_sa_p = MRSAParams()

    rng     = random.Random(seed)
    N_iter  = max(n_outer * 25, 5000)
    t0      = time.perf_counter()

    # ── Initialise proxy state ────────────────────────────────────────────
    current_seqs              = [s[:] for s in seqs_init]
    tc_rwy, lbt_rwy, sep_rwy = init_proxy_arrays(current_seqs, inst)
    proxy_curr = compute_proxy(current_seqs, tc_rwy, lbt_rwy, sep_rwy,
                                inst, h_params)

    best_seqs  = [s[:] for s in current_seqs]
    best_lp    = lp_init
    best_x     = x_init.copy() if x_init is not None else None
    tc_b, lbt_b, sep_b = init_proxy_arrays(best_seqs, inst)
    proxy_best = compute_proxy(best_seqs, tc_b, lbt_b, sep_b, inst, h_params)

    # ── LP impact scores and adaptive weight tracker ──────────────────────
    impact  = lp_impact_scores(best_seqs, best_x, inst) if best_x is not None else None
    updater = AdaptiveWeightUpdater(xi=ADAPT_W_XI, beta=ADAPT_W_BETA)

    # ── Elite pool ────────────────────────────────────────────────────────
    pool_elite = ElitePool()
    if best_x is not None:
        pool_elite.try_add(best_seqs, best_lp, best_x)

    # ── Temperature calibration ───────────────────────────────────────────
    T = _calibrate_proxy_T(current_seqs, inst, h_params,
                            mr_sa_p, rng, T_outer)

    stag     = 0
    n_reheat = 0
    MAX_REHEAT = 3

    # ── Main proxy SA loop ────────────────────────────────────────────────
    for t_it in range(N_iter):
        f = t_it / N_iter

        # Adaptive move-pressure weights
        w_current = updater.weights()
        params_w  = replace(params, w=w_current)

        pool_cands = generate_candidate_pool(
            f, current_seqs, tc_rwy, lbt_rwy, sep_rwy,
            inst, h_params, mr_sa_p, rng, stag, N_iter,
            mr_sa_p.R_pool, None, None, impact, best_x,
        )

        if not pool_cands:
            stag += 1
            T    *= mr_sa_p.alpha
            continue

        proxy_new, res, tc_n, lbt_n, sep_n = pool_cands[0]
        d_proxy = proxy_new - proxy_curr

        # SA acceptance on proxy delta
        accepted = (d_proxy <= 0 or
                    (T > 1e-15
                     and rng.random() < math.exp(-d_proxy / T)))
        if accepted:
            current_seqs              = res.seqs
            tc_rwy, lbt_rwy, sep_rwy = tc_n, lbt_n, sep_n
            proxy_curr                = proxy_new
            stag                      = 0 if d_proxy < 0 else stag + 1
        else:
            stag += 1

        # Adaptive LP evaluation
        Nfull = _nfull(f, mr_sa_p)
        call_lp = (proxy_new < proxy_best or t_it % Nfull == 0)

        if call_lp:
            lp_val, lp_x = stage2_lp(current_seqs, inst)
            if not math.isinf(lp_val) and lp_x is not None:
                pool_elite.try_add(current_seqs, lp_val, lp_x)
                if lp_val < best_lp - 1e-9:
                    # Identify primary move-pressure driver for weight update
                    if best_x is not None:
                        try:
                            MP_raw = compute_move_pressure(
                                best_seqs, best_x, inst, params_w)
                            component_idx = int(np.argmax(MP_raw))
                            updater.reward(component_idx,
                                           best_lp - lp_val)
                        except Exception:
                            pass

                    best_seqs  = [s[:] for s in current_seqs]
                    best_lp    = lp_val
                    best_x     = lp_x.copy()
                    tc_b, lbt_b, sep_b = init_proxy_arrays(best_seqs, inst)
                    proxy_best = compute_proxy(
                        best_seqs, tc_b, lbt_b, sep_b, inst, h_params)
                    impact = lp_impact_scores(best_seqs, best_x, inst)
                else:
                    updater.decay()

        T *= mr_sa_p.alpha

        # Stagnation: reheat and restart from best
        if stag >= mr_sa_p.M_stag:
            if n_reheat < MAX_REHEAT:
                T         *= mr_sa_p.beta_reheat
                stag       = 0
                n_reheat  += 1
                current_seqs              = [s[:] for s in best_seqs]
                tc_rwy, lbt_rwy, sep_rwy = init_proxy_arrays(current_seqs, inst)
                proxy_curr                = proxy_best
            else:
                break

    # ── Path relinking across elite pool ─────────────────────────────────
    sol_a, sol_b = pool_elite.best_quality_pair()
    if sol_a is not None and sol_b is not None:
        pr_seqs, pr_lp = path_relink(sol_a, sol_b, inst, h_params)
        if pr_lp < best_lp - 1e-9:
            best_seqs = pr_seqs
            best_lp   = pr_lp
            _, best_x = stage2_lp(best_seqs, inst)

    # ── Within-runway SA refinement on best assignment ────────────────────
    refined = [s[:] for s in best_seqs]
    for rho in range(inst.m):
        if refined[rho]:
            refined[rho], _ = run_single_runway_sa(
                refined[rho], inst, sa_p, seed=seed + rho)
    ref_lp, ref_x = stage2_lp(refined, inst)
    if not math.isinf(ref_lp) and ref_x is not None and ref_lp < best_lp - 1e-9:
        best_seqs = refined
        best_lp   = ref_lp
        best_x    = ref_x

    # ── LP-VND polish ─────────────────────────────────────────────────────
    if best_x is not None:
        elapsed    = time.perf_counter() - t0
        vnd_budget = min(90.0, max(10.0, 300.0 - elapsed))
        pol_seqs, pol_lp = lp_vnd_polish(
            best_seqs, best_lp, best_x,
            inst, h_params, mr_sa_p,
            max_rounds=8, t_limit=vnd_budget,
        )
        if pol_lp < best_lp - 1e-9:
            best_seqs = pol_seqs
            best_lp   = pol_lp
            _, best_x = stage2_lp(best_seqs, inst)

    return best_seqs, best_lp, best_x