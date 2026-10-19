"""
sa.py — MR-ALP Solver: Single SA Chain, Spawn-Safe Worker, Optuna Tuning
=========================================================================
§11  Optuna hyperparameter tuning (optimize_rbi_params, optimize_sa_params)
§18  SA helper functions (iteration counts, calibration, adaptive sizing)
§23  run_mr_sa — single SA chain with reactive cooling, LP-guided repair,
     elite pool seeding, and LP-timeline tracking.
§26  _sa_worker — module-level entry point for ProcessPoolExecutor.

Chain return contract
---------------------
run_mr_sa returns:
    best_p_seqs  : best-proxy sequences found.
    best_proxy   : proxy value at best_p_seqs.
    best_lp_seqs : LP-best sequences found.
    best_lp      : LP value at best_lp_seqs; math.inf if no LP solution found.
    best_C_lp    : LP solution vector (shape n,) or None.
    stats dict   : {label, history, alpha_history, t_best_proxy,
                    t_best_lp, wall, lp_timeline}
                   lp_timeline is a list of (chain_relative_seconds, lp_val)
                   pairs recording each LP improvement event on this chain.
                   All times are relative to the start of this chain, not the
                   job; solver.py converts them to job-relative times.

_sa_worker packs the 11-tuple consumed by ms_mr_sa.
"""
from __future__ import annotations

import math
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mr_alp.config    import (
    N_RBI_TRIALS_BASE, RBI_OPTUNA_SEED, SA_N_TRIALS_BASE, SA_OPTUNA_SEED,
    SA_N_OPTUNA_JOBS, N_OPTUNA_WORKERS,
)
from mr_alp.models    import Instance, HeuristicParams, MRSAParams
from mr_alp.lp        import stage2_lp_objective
from mr_alp.proxy     import (
    init_proxy_arrays, compute_proxy,
    compute_per_aircraft_scores, lp_impact_scores,
    _rwy_proxy_components,
)
from mr_alp.operators import (
    select_op, apply_op, generate_candidate_pool,
)
from mr_alp.repair    import (
    lp_guided_penalty_repair, target_conflict_repair,
    ejection_chain_transfer, lns_remove_reinsert, _lp_repair_params,
)
from mr_alp.instance  import runway_feasible

try:
    import optuna as _optuna
    _optuna.logging.set_verbosity(_optuna.logging.WARNING)
    _OPTUNA = True
except ImportError:
    _optuna = None; _OPTUNA = False


# ═══════════════════════════════════════════════════════════════════════════
#   §18  SA HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def _n_iter(n: int) -> int:
    """SA iterations per chain, scaled by instance size."""
    if n <= 50:   return 2_000
    if n <= 250:  return 5_000
    return 8_000


def _R_candidates(n: int) -> int:
    """Candidate pool size R per SA step."""
    if n <= 100:  return 10
    if n <= 250:  return 20
    return 30


def _vnd_max_rounds(n: int) -> int:
    """Maximum LP-VND rounds, scaled by instance size."""
    if n <= 100:  return 15
    if n <= 250:  return 10
    return 5


def _n_full(t: int, N_iter: int) -> int:
    """
    Interval for full reactive-cooling and per-aircraft score refresh.
    More frequent early in the search, less so later.
    """
    f = t / max(N_iter, 1)
    if f <= 0.25:  return 20
    if f <= 0.75:  return 50
    return 100


def _adaptive_t_limit(n: int, m: int, seed_lp: float, bks: Optional[float]) -> float:
    """
    Map seed LP gap to a wall-time budget for the SA+VND+PR phase.

    Zero-objective instances (bks=0) need little time.  Instances with large
    gaps get MAX_T_LIMIT.  Unknown BKS falls back to the default T_LIMIT.
    """
    from mr_alp.config import T_LIMIT, MAX_T_LIMIT
    if bks is None:           return T_LIMIT
    if bks == 0.0:            return 60.0
    if math.isinf(seed_lp):  return MAX_T_LIMIT
    gap = 100.0 * (seed_lp - bks) / bks
    if gap <= 0.0:   return 60.0
    if gap <= 2.0:   return T_LIMIT * 0.5
    if gap <= 5.0:   return T_LIMIT
    if gap <= 10.0:  return min(2000.0, MAX_T_LIMIT)
    return MAX_T_LIMIT


def _calibrate_t0(
    seqs: List[List[int]],
    inst: Instance,
    params: HeuristicParams,
    p_sa: MRSAParams,
    seed: int,
    N_iter: int,
) -> float:
    """
    Estimate an initial SA temperature T₀ such that p_sa.chi0 fraction of
    worsening moves are accepted.

    Samples n_cal random moves, collects positive deltas, and solves:
        chi0 = exp(−mean(Δ⁺) / T₀)  ⟹  T₀ = −mean(Δ⁺) / ln(chi0).
    """
    rng   = random.Random(seed)
    m     = len(seqs)
    tc_r, lbt_r, sep_r = init_proxy_arrays(seqs, inst)
    proxy_cur = compute_proxy(seqs, tc_r, lbt_r, sep_r, inst, params)
    pa_tc, pa_lbt = compute_per_aircraft_scores(seqs, inst)
    deltas_pos = []

    for _ in range(p_sa.n_cal):
        op  = select_op(0.5, m, rng)
        res = apply_op(op, seqs, tc_r, lbt_r, sep_r, inst, params, p_sa,
                        rng, 0, N_iter, pa_tc, pa_lbt)
        if res is None:
            continue
        tc_n, lbt_n, sep_n = init_proxy_arrays(res.seqs, inst)
        d = compute_proxy(res.seqs, tc_n, lbt_n, sep_n, inst, params) - proxy_cur
        if d > 1e-9:
            deltas_pos.append(d)

    if not deltas_pos:
        return max(abs(proxy_cur) * 0.01, 1.0)
    return max(
        -float(np.mean(deltas_pos)) / math.log(p_sa.chi0 + 1e-12),
        1e-3,
    )


def _ils_perturb(
    seqs:   List[List[int]],
    inst:   Instance,
    rng:    random.Random,
    k:      int = 4,
) -> List[List[int]]:
    """
    ILS perturbation: simultaneously relocate k aircraft across runways.

    Picks aircraft one at a time, removes from source runway, inserts at a
    random feasible position on a different runway.  Larger k creates a bigger
    structural jump to escape deep local-optima basins.  Falls back to the
    original seqs if no valid move is found.
    """
    m      = len(seqs)
    if m < 2:
        return [s[:] for s in seqs]
    total  = sum(len(s) for s in seqs)
    k      = min(k, max(1, total // 3))
    result = [s[:] for s in seqs]
    moved  = 0
    for _ in range(k * 10):
        if moved >= k:
            break
        rho_a = rng.randrange(m)
        if not result[rho_a]:
            continue
        pos_a = rng.randrange(len(result[rho_a]))
        ac    = result[rho_a][pos_a]
        sm    = result[rho_a][:pos_a] + result[rho_a][pos_a + 1:]
        if not runway_feasible(sm, inst):
            continue
        rho_b = rng.choice([r for r in range(m) if r != rho_a])
        q     = rng.randint(0, len(result[rho_b]))
        cand  = result[rho_b][:q] + [ac] + result[rho_b][q:]
        if not runway_feasible(cand, inst):
            continue
        result[rho_a] = sm
        result[rho_b] = cand
        moved += 1
    return result


# ═══════════════════════════════════════════════════════════════════════════
#   §11  OPTUNA HYPERPARAMETER TUNING
# ═══════════════════════════════════════════════════════════════════════════

def _n_rbi_trials(n: int, base: int) -> int:
    if n <= 100:  return base
    if n <= 250:  return max(10, base // 3)
    return max(5, base // 6)


def _sa_n_trials(n: int, base: int) -> int:
    if n <= 50:   return base
    if n <= 100:  return max(10, base // 2)
    if n <= 250:  return max(6,  base // 4)
    return max(3, base // 7)


def optimize_rbi_params(
    inst: Instance, m: int,
    n_trials: int, seed: int, n_jobs: int = 1,
) -> HeuristicParams:
    """
    Tune HeuristicParams for (inst, m) using Optuna TPE.

    Objective: stage2_lp_objective for n ≤ 100; total_target_conflict proxy
    otherwise (LP is too expensive per trial for large instances).
    """
    from mr_alp.construction import ramp_rbi
    from mr_alp.proxy        import total_target_conflict
    if not _OPTUNA or n_trials == 0:
        return HeuristicParams()
    use_lp = (inst.n <= 100)

    def objective(trial):
        p = HeuristicParams(
            eta      = trial.suggest_float('eta',      0.20, 0.80),
            mu_tc    = trial.suggest_float('mu_tc',    0.10, 5.00),
            mu_late  = trial.suggest_float('mu_late',  0.01, 2.00),
            mu_count = trial.suggest_float('mu_count', 0.10, 3.00),
            mu_sep   = trial.suggest_float('mu_sep',   0.00, 0.50),
        )
        seqs, _ = ramp_rbi(inst, m, p)
        if use_lp:
            obj, _, feas, _ = stage2_lp_objective(seqs, inst)
            return obj if feas else 1e12
        return total_target_conflict(seqs, inst)

    sampler = _optuna.samplers.TPESampler(seed=seed)
    study   = _optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials,
                   n_jobs=min(n_jobs, n_trials), show_progress_bar=False)
    bp = study.best_params
    return HeuristicParams(eta=bp['eta'], mu_tc=bp['mu_tc'],
                           mu_late=bp['mu_late'], mu_count=bp['mu_count'],
                           mu_sep=bp['mu_sep'])


def optimize_sa_params(
    inst: Instance, m: int,
    params: HeuristicParams,
    n_trials: int, seed: int, n_jobs: int = 1,
) -> MRSAParams:
    """
    Tune MRSAParams for (inst, m) using Optuna TPE.

    Each trial runs a shortened SA chain (N_iter/6) with lp_repair_interval=0
    for speed.  Objective is the final LP value of that chain.
    """
    from mr_alp.construction import ramp_rbi
    if not _OPTUNA or n_trials == 0:
        return MRSAParams()
    N_tune = max(300, _n_iter(inst.n) // 6)

    def objective(trial):
        p_sa = MRSAParams(
            chi0         = trial.suggest_float('chi0',         0.50, 0.95),
            M_stag_frac  = trial.suggest_float('M_stag_frac',  0.05, 0.30),
            beta         = trial.suggest_float('beta',         1.20, 2.50),
            lp_gamma     = trial.suggest_float('lp_gamma',     0.01, 0.20),
            chi_target   = trial.suggest_float('chi_target',   0.10, 0.35),
            lp_repair_interval=0,
        )
        seqs, _ = ramp_rbi(inst, m, params)
        _, _, blp_seqs, best_lp, _, _ = run_mr_sa(
            seqs, math.inf, inst, params, p_sa, N_tune,
            label="sa_tune", seed=trial.number * 13 + seed)
        if math.isinf(best_lp):
            lp_val, _, feas, _ = stage2_lp_objective(blp_seqs or seqs, inst)
            best_lp = lp_val if feas else 1e12
        return best_lp

    sampler = _optuna.samplers.TPESampler(seed=seed)
    study   = _optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials,
                   n_jobs=min(n_jobs, n_trials), show_progress_bar=False)
    bp = study.best_params
    return MRSAParams(chi0=bp['chi0'], M_stag_frac=bp['M_stag_frac'],
                      beta=bp['beta'], lp_gamma=bp['lp_gamma'],
                      chi_target=bp['chi_target'])


# ═══════════════════════════════════════════════════════════════════════════
#   §23  SINGLE SA CHAIN
# ═══════════════════════════════════════════════════════════════════════════

def run_mr_sa(
    init_seqs:   List[List[int]],
    init_lp:     float,
    inst:        Instance,
    params:      HeuristicParams,
    p_sa:        MRSAParams,
    N_iter:      int,
    label:       str  = "chain",
    seed:        int  = 0,
    T0:          Optional[float] = None,
    t_deadline:  Optional[float] = None,
) -> Tuple[List[List[int]], float, List[List[int]], float,
           Optional[np.ndarray], Dict[str, Any]]:
    """
    Execute one SA chain with reactive cooling, LP-triggered checks,
    and LP-guided repair operators.

    Time tracking
    -------------
    All timestamps inside this function are CHAIN-RELATIVE (from the moment
    run_mr_sa is called).  solver.py converts them to job-relative times by
    adding the SA dispatch offset.

    stats['lp_timeline'] records (chain_relative_seconds, lp_val) for every
    LP improvement seen by this chain.  stats['t_best_lp'] is the chain-relative
    time at which the best LP was first recorded.

    Returns
    -------
    (best_p_seqs, best_proxy, best_lp_seqs, best_lp, best_C_lp, stats)
    """
    CHI_TARGET  = p_sa.chi_target;    ALPHA_STEP  = p_sa.alpha_step
    ALPHA_LO    = p_sa.alpha_lo;      ALPHA_HI    = p_sa.alpha_hi
    MAX_REHEATS = p_sa.max_reheats;   M_STAG      = max(1, int(p_sa.M_stag_frac * N_iter))
    GAMMA       = p_sa.lp_gamma;      LP_REPAIR   = p_sa.lp_repair_interval
    NZ_THRESH   = p_sa.near_zero_threshold
    EC_DEPTH    = min(p_sa.ejection_chain_depth,
                      2 if len(init_seqs) < 3 else p_sa.ejection_chain_depth)
    R           = _R_candidates(inst.n)
    q_lp, K     = _lp_repair_params(inst.n)

    rng   = random.Random(seed)
    m     = len(init_seqs)
    t0    = time.perf_counter()
    seqs  = [s[:] for s in init_seqs]

    tc_rwy, lbt_rwy, sep_rwy = init_proxy_arrays(seqs, inst)
    proxy     = compute_proxy(seqs, tc_rwy, lbt_rwy, sep_rwy, inst, params)
    pa_tc, pa_lbt = compute_per_aircraft_scores(seqs, inst)
    impact    = None

    best_p_seqs   = [s[:] for s in seqs]; best_proxy = proxy; t_best_proxy = 0.0
    best_lp_seqs  = [s[:] for s in seqs]; best_lp    = init_lp
    best_C_lp     = None;                 t_best_lp  = 0.0
    lp_timeline   = [(0.0, init_lp)] if not math.isinf(init_lp) else []
    best_proxy_lp_checked = proxy

    T     = T0 or _calibrate_t0(seqs, inst, params, p_sa, seed, N_iter)
    T_min = T * p_sa.T_min_frac
    alpha = (ALPHA_HI + ALPHA_LO) / 2.0

    history: List[float] = []; alpha_history: List[float] = []
    stag = 0; n_reheats = 0; n_accepted = 0; n_tried = 0; n_ils_restarts = 0

    for t in range(1, N_iter + 1):
        if t_deadline is not None and time.perf_counter() >= t_deadline:
            break
        f    = t / N_iter
        pool = generate_candidate_pool(
            f, seqs, tc_rwy, lbt_rwy, sep_rwy,
            inst, params, p_sa, rng, stag, N_iter, R,
            pa_tc, pa_lbt, impact, best_C_lp)

        if not pool:
            history.append(best_proxy); alpha_history.append(alpha); continue

        if rng.random() < 0.80:
            proxy_new, res, tc_n, lbt_n, sep_n = pool[0]
        else:
            proxy_new, res, tc_n, lbt_n, sep_n = rng.choice(pool[:min(5, len(pool))])

        n_tried += 1
        dlt    = proxy_new - proxy
        accept = (dlt <= 0 or rng.random() < math.exp(-dlt / max(T, 1e-15)))

        if accept:
            seqs = res.seqs; tc_rwy = tc_n; lbt_rwy = lbt_n; sep_rwy = sep_n
            proxy = proxy_new; n_accepted += 1
            stag  = max(stag - 1, 0) if dlt < 0 else stag + 1
            if proxy < best_proxy - 1e-9:
                best_p_seqs   = [s[:] for s in seqs]; best_proxy = proxy
                t_best_proxy  = time.perf_counter() - t0; stag = 0
        else:
            stag += 1

        # ── LP trigger ───────────────────────────────────────────────────
        call_lp = (t % _n_full(t, N_iter) == 0
                   or proxy_new < (1.0 - GAMMA) * best_proxy_lp_checked)
        if call_lp:
            lp_val, C_cur, lp_feas, _ = stage2_lp_objective(seqs, inst)
            best_proxy_lp_checked = proxy
            if lp_feas and lp_val < best_lp - 1e-9:
                best_lp_seqs = [s[:] for s in seqs]; best_lp = lp_val
                best_C_lp    = C_cur;  t_best_lp = time.perf_counter() - t0
                stag         = 0
                impact       = lp_impact_scores(seqs, C_cur, inst,
                                                 p_sa.lambda_binding,
                                                 p_sa.eps_tight)
                lp_timeline.append((t_best_lp, lp_val))

        # ── Periodic LP-guided repair ─────────────────────────────────────
        if LP_REPAIR > 0 and t % LP_REPAIR == 0 and best_C_lp is not None:
            cand, cand_lp = lp_guided_penalty_repair(
                best_lp_seqs, best_C_lp, inst, params, K=K, q_lp=q_lp)
            if cand is not None and cand_lp < best_lp - 1e-9:
                best_lp_seqs = cand; best_lp = cand_lp
                _, best_C_lp, _, _ = stage2_lp_objective(best_lp_seqs, inst)
                if best_C_lp is not None:
                    impact = lp_impact_scores(best_lp_seqs, best_C_lp, inst,
                                              p_sa.lambda_binding, p_sa.eps_tight)
                lp_timeline.append((time.perf_counter() - t0, cand_lp)); stag = 0

        if LP_REPAIR > 0 and t % (LP_REPAIR * 2) == 0 and best_lp < NZ_THRESH:
            cand, cand_lp = target_conflict_repair(
                best_lp_seqs, inst, params, K=max(K // 2, 3))
            if cand is not None and cand_lp < best_lp - 1e-9:
                best_lp_seqs = cand; best_lp = cand_lp
                _, C_new, feas_new, _ = stage2_lp_objective(best_lp_seqs, inst)
                if feas_new:
                    best_C_lp = C_new
                    lp_timeline.append((time.perf_counter() - t0, cand_lp))

        if (LP_REPAIR > 0 and t % (LP_REPAIR * 3) == 0
                and best_C_lp is not None and m >= 2):
            cand, cand_lp = ejection_chain_transfer(
                best_lp_seqs, best_C_lp, inst, params,
                depth=EC_DEPTH, K=max(K // 2, 3))
            if cand is not None and cand_lp < best_lp - 1e-9:
                best_lp_seqs = cand; best_lp = cand_lp
                _, C_new, feas_new, _ = stage2_lp_objective(best_lp_seqs, inst)
                if feas_new:
                    best_C_lp = C_new
                    impact     = lp_impact_scores(best_lp_seqs, best_C_lp, inst,
                                                   p_sa.lambda_binding, p_sa.eps_tight)
                lp_timeline.append((time.perf_counter() - t0, cand_lp)); stag = 0

        # ── LNS destroy-repair (every 4×LP_REPAIR iters, m >= 2) ──────────
        if (LP_REPAIR > 0 and t % (LP_REPAIR * 4) == 0
                and best_C_lp is not None and m >= 2):
            k_lns = max(3, min(5, inst.n // (m * 10)))
            cand, cand_lp = lns_remove_reinsert(
                best_lp_seqs, best_C_lp, inst, params,
                k=k_lns, K=max(K // 2, 4))
            if cand is not None and cand_lp < best_lp - 1e-9:
                best_lp_seqs = cand; best_lp = cand_lp
                _, C_new, feas_new, _ = stage2_lp_objective(best_lp_seqs, inst)
                if feas_new:
                    best_C_lp = C_new
                    impact    = lp_impact_scores(best_lp_seqs, best_C_lp, inst,
                                                  p_sa.lambda_binding, p_sa.eps_tight)
                lp_timeline.append((time.perf_counter() - t0, cand_lp)); stag = 0

        # ── Reactive cooling ──────────────────────────────────────────────
        if t % _n_full(t, N_iter) == 0:
            chi   = n_accepted / max(n_tried, 1)
            alpha = (max(ALPHA_LO, alpha - ALPHA_STEP) if chi > CHI_TARGET
                     else min(ALPHA_HI, alpha + ALPHA_STEP))
            n_accepted = n_tried = 0
            pa_tc, pa_lbt = compute_per_aircraft_scores(seqs, inst)

        T = max(T * alpha, T_min)

        # ── Stagnation restart ─────────────────────────────────────────────
        if stag >= M_STAG:
            if n_reheats >= MAX_REHEATS:
                if n_ils_restarts >= p_sa.max_ils_restarts:
                    break
                # ILS restart: structural kick from best LP solution then
                # warm-restart SA with reduced temperature
                base = best_lp_seqs if not math.isinf(best_lp) else best_p_seqs
                k_kick = max(3, inst.n // (m * 8))
                seqs = _ils_perturb(base, inst, rng, k=k_kick)
                tc_rwy, lbt_rwy, sep_rwy = init_proxy_arrays(seqs, inst)
                proxy = compute_proxy(seqs, tc_rwy, lbt_rwy, sep_rwy, inst, params)
                T = (T0 or T) * 0.5
                stag = 0; n_reheats = 0; n_ils_restarts += 1
            else:
                T = min(T * p_sa.t_reheat, T0 or T)
                perturbed = False
                for _ in range(5):
                    pres = apply_op(
                        rng.choice(["X4", "X2"]), seqs, tc_rwy, lbt_rwy, sep_rwy,
                        inst, params, p_sa, rng, M_STAG + 1, N_iter,
                        pa_tc=pa_tc, pa_lbt=pa_lbt, impact=impact)
                    if pres is not None:
                        for rho in pres.affected:
                            tc_rwy[rho], lbt_rwy[rho], sep_rwy[rho] = (
                                _rwy_proxy_components(pres.seqs[rho], inst))
                        seqs  = pres.seqs
                        proxy = compute_proxy(seqs, tc_rwy, lbt_rwy, sep_rwy,
                                              inst, params)
                        perturbed = True; break
                # On second+ reheat, also apply ILS kick if single-op failed
                if not perturbed and n_reheats >= 1 and m >= 2:
                    k_kick = 2 + n_reheats
                    seqs = _ils_perturb(seqs, inst, rng, k=k_kick)
                    tc_rwy, lbt_rwy, sep_rwy = init_proxy_arrays(seqs, inst)
                    proxy = compute_proxy(seqs, tc_rwy, lbt_rwy, sep_rwy,
                                         inst, params)
                stag = 0; n_reheats += 1

        history.append(best_proxy); alpha_history.append(alpha)

    # ── Final LP check on best-proxy solution ──────────────────────────────
    if math.isinf(best_lp):
        lp_val, C_cur, lp_feas, _ = stage2_lp_objective(best_p_seqs, inst)
        if lp_feas:
            best_lp_seqs = [s[:] for s in best_p_seqs]; best_lp = lp_val
            best_C_lp    = C_cur; t_best_lp = time.perf_counter() - t0
            lp_timeline.append((t_best_lp, lp_val))

    return best_p_seqs, best_proxy, best_lp_seqs, best_lp, best_C_lp, {
        'label':         label,
        'history':       history,
        'alpha_history': alpha_history,
        't_best_proxy':  t_best_proxy,
        't_best_lp':     t_best_lp,        # chain-relative
        'wall':          time.perf_counter() - t0,
        'lp_timeline':   lp_timeline,       # chain-relative timestamps
    }


# ═══════════════════════════════════════════════════════════════════════════
#   §26  SPAWN-SAFE SA WORKER
# ═══════════════════════════════════════════════════════════════════════════

def _sa_worker(args: tuple) -> tuple:
    """
    Module-level entry point for one SA chain in a worker process.

    Must be at module level for ProcessPoolExecutor pickle compatibility.
    CUDA tensors on Instance are re-initialised by Instance.__setstate__
    in the receiving process.

    Return tuple (11 elements)
    ---------------------------
    0  label
    1  bp_seqs       best proxy sequences
    2  b_proxy       best proxy value
    3  blp_seqs      best LP sequences
    4  b_lp          best LP value (math.inf if none found)
    5  b_C_lp        LP solution vector or None
    6  history        per-iteration best_proxy trace
    7  t_best_proxy   chain-relative wall time at proxy best
    8  t_best_lp      chain-relative wall time at LP best
    9  alpha_history  per-iteration α trace
    10 lp_timeline    list of (chain_relative_s, lp_val) improvement events
    """
    label, init_seqs, init_lp, inst, params, p_sa, N_iter, seed, t_deadline = args
    bp_seqs, b_proxy, blp_seqs, b_lp, b_C_lp, st = run_mr_sa(
        init_seqs, init_lp, inst, params, p_sa, N_iter,
        label=label, seed=seed, t_deadline=t_deadline)
    return (label, bp_seqs, b_proxy, blp_seqs, b_lp, b_C_lp,
            st['history'], st['t_best_proxy'], st['t_best_lp'],
            st['alpha_history'], st['lp_timeline'])