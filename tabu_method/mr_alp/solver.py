"""
solver.py — MR-ALP Solver: Parallel Multi-Start SA  (main solver)
==================================================================
§27  ms_mr_sa — parallel K-chain SA with seed portfolio, elite pool,
     path relinking, LP-VND polish, and unified job-relative time tracking.

Time-tracking design
--------------------
All LP improvement events are recorded on a single unified job_lp_timeline
where every timestamp is JOB-RELATIVE (wall seconds from the start of the
ms_mr_sa call).  The timeline covers four improvement sources:

  1. Seed LP events    : recorded by _build_seed_portfolio; already job-relative.
  2. SA chain events   : recorded by run_mr_sa as chain-relative times;
                         converted here by adding t_sa_dispatch_offset, the
                         job-relative time at which SA tasks were submitted.
  3. VND events        : appended with time.perf_counter() − t_job_start.
  4. Path-relinking    : appended with time.perf_counter() − t_job_start.

total_t_best is computed as the earliest time in job_lp_timeline at which
the final best_lp was first achieved.  Concretely:

  total_t_best = seed phase time   ← if best came from seed LP
               = t_sa_offset + chain_t_best_lp  ← if SA improved on seed LP
               = VND / PR timestamp             ← if VND or PR found the best

This is derived by scanning job_lp_timeline forward and finding the first
entry whose lp_val ≤ best_lp + ε.
"""
from __future__ import annotations

import math
import platform
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import multiprocessing as _mp

from mr_alp.config        import (
    N_CHAINS, N_WORKERS, T_LIMIT, MAX_T_LIMIT,
    ELITE_POOL_MAX, ELITE_MIN_DIV, USE_ALL_SEEDS,
    SA_CHAIN_DIVERSITY, SET_PARTITION_RECOMBINE,
)
from mr_alp.models        import Instance, HeuristicParams, MRSAParams
from mr_alp.lp            import stage2_lp_objective
from mr_alp.construction  import _build_seed_portfolio
from mr_alp.repair        import (
    ElitePool, path_relink, lp_vnd_polish, set_partition_recombine,
    _lp_repair_params,
)
from mr_alp.sa            import _sa_worker, _n_iter, _vnd_max_rounds

try:
    import torch as _torch
    _GPU_AVAIL = _torch.cuda.is_available()
except ImportError:
    _GPU_AVAIL = False

_MP_CTX = _mp.get_context(
    "spawn" if (platform.system() == "Windows" or _GPU_AVAIL) else "fork"
)


def _chain_sa_params(base: MRSAParams, chain_idx: int) -> MRSAParams:
    """Return a chain-specific SA profile for parallel search diversity."""
    if not SA_CHAIN_DIVERSITY:
        return base
    profiles = (
        {},
        {
            "chi0": min(0.95, base.chi0 + 0.10),
            "chi_target": min(0.35, base.chi_target + 0.08),
            "T_min_frac": max(base.T_min_frac, 0.02),
            "M_stag_frac": max(0.05, base.M_stag_frac * 0.75),
            "max_ils_restarts": max(base.max_ils_restarts, 3),
        },
        {
            "chi0": max(0.55, base.chi0 - 0.15),
            "chi_target": max(0.10, base.chi_target - 0.05),
            "lp_gamma": min(0.20, base.lp_gamma * 1.8),
            "lp_repair_interval": max(40, base.lp_repair_interval // 2),
        },
        {
            "M_stag_frac": max(0.04, base.M_stag_frac * 0.50),
            "max_reheats": max(base.max_reheats, 5),
            "max_ils_restarts": max(base.max_ils_restarts, 4),
            "B_max": max(base.B_max, 4),
            "B_stag": max(base.B_stag, 7),
        },
    )
    return replace(base, **profiles[chain_idx % len(profiles)])


def ms_mr_sa(
    inst:     Instance,
    m:        int,
    params:   HeuristicParams,
    p_sa:     Optional[MRSAParams] = None,
    n_chains: int   = N_CHAINS,
    t_limit:  float = T_LIMIT,
    seed:     int   = 0,
) -> Tuple[List[List[int]], float, Dict[str, Any]]:
    """
    Run K parallel SA chains, collect an elite pool, apply path relinking,
    LP-VND polish, and return the best overall solution.

    Workflow
    --------
    1. _build_seed_portfolio generates and LP-evaluates all construction
       heuristics; timing data is captured per seed.
    2. SA chains run in parallel via ProcessPoolExecutor (spawn context).
    3. The best LP-feasible chain result seeds the elite pool.
    4. LP-VND polish refines the best solution.
    5. Path relinking between elite pairs produces additional candidates.
    6. A final LP check confirms the reported objective.

    Time tracking
    -------------
    portfolio_timing (from _build_seed_portfolio) provides:
      seed_lp_events  : job-relative timestamps of seed LP improvements.
      t_best_seed_lp  : job-relative time of best seed LP event.

    SA chain lp_timeline entries are chain-relative; they are converted to
    job-relative by adding t_sa_dispatch_offset (= time.perf_counter() − t_job
    when SA tasks are submitted to the executor).

    VND and path-relinking improvements are appended with current job time.

    total_t_best is the earliest job-relative time the final best_lp was seen.

    Returns
    -------
    best_seqs : list of per-runway aircraft index sequences.
    best_lp   : LP-optimal objective (math.inf if no feasible solution found).
    stats     : dict containing timing, pool, portfolio, and history data.
    """
    p_sa   = p_sa or MRSAParams()
    N_iter = _n_iter(inst.n)
    t_job  = time.perf_counter()
    t_dead = t_job + t_limit

    # ── Stage 1: build and LP-screen seed portfolio ───────────────────────
    starts, portfolio_info, seed_lps, seed_raw_objs, portfolio_timing = (
        _build_seed_portfolio(inst, m, params, n_chains, seed))

    best_seed_lp  = min(seed_lps)      if seed_lps      else math.inf
    best_seed_raw = min(seed_raw_objs) if seed_raw_objs else math.inf

    # Seed improvement events are already job-relative (from portfolio start,
    # which coincides with t_job for our purposes).
    job_lp_timeline: List[Tuple[float, float]] = list(
        portfolio_timing['seed_lp_events'])

    print(f"  [{inst.name} m={m}] {len(portfolio_info)} seeds evaluated | "
          f"{'all' if USE_ALL_SEEDS else n_chains} selected | "
          f"N_iter={N_iter} | t_limit={t_limit:.0f}s")
    print(f"  SA params: {p_sa}")
    for label, raw_obj, lp_val, selected in sorted(portfolio_info, key=lambda x: x[2]):
        tag     = " ← selected" if selected else ""
        raw_str = f"{raw_obj:.4f}" if not math.isinf(raw_obj) else "inf"
        lp_str  = f"{lp_val:.4f}" if not math.isinf(lp_val)  else "inf"
        print(f"    {label:<12} raw={raw_str}  LP={lp_str}{tag}")

    # ── Stage 2: parallel SA chains ───────────────────────────────────────
    task_params = [_chain_sa_params(p_sa, i) for i in range(len(starts))]
    tasks = [
        (lbl, s, seed_lps[i], inst, params, task_params[i], N_iter,
         seed + i * 31, t_dead)
        for i, (lbl, s) in enumerate(starts)
    ]
    n_sa_workers = (min(len(tasks), N_WORKERS) if USE_ALL_SEEDS
                    else min(n_chains, len(tasks)))

    # Record job-relative time when SA tasks are submitted; used to convert
    # chain-relative LP timeline entries to job-relative ones.
    t_sa_dispatch = time.perf_counter()
    t_sa_dispatch_offset = t_sa_dispatch - t_job

    with ProcessPoolExecutor(max_workers=n_sa_workers,
                              mp_context=_MP_CTX) as ex:
        results = list(ex.map(_sa_worker, tasks))

    # ── Select best chain result ──────────────────────────────────────────
    feas_rs = [r for r in results if not math.isinf(r[4])]
    if feas_rs:
        best_r    = min(feas_rs, key=lambda r: r[4])
        best_seqs = best_r[3]; best_lp = best_r[4]; best_C = best_r[5]
    else:
        warnings.warn(f"{inst.name} m={m}: no LP-feasible SA solution found.")
        best_r    = min(results, key=lambda r: r[2])
        best_seqs = best_r[1]; best_lp = math.inf; best_C = None

    # Convert best chain's LP timeline to job-relative and append
    for chain_t, lp_v in best_r[10]:   # index 10 = lp_timeline
        job_lp_timeline.append((t_sa_dispatch_offset + chain_t, lp_v))

    # ── Build elite pool from all feasible chain results ──────────────────
    # min_diversity scales with instance size so large instances get structurally
    # distinct solutions rather than near-duplicates.
    dyn_min_div = max(ELITE_MIN_DIV, inst.n // 30)
    pool = ElitePool(ELITE_POOL_MAX, dyn_min_div)
    for r in feas_rs:
        pool.try_add(r[3], r[4], r[5])

    # ── Confirm best LP and seed pool ─────────────────────────────────────
    final_lp, final_C, final_feas, _ = stage2_lp_objective(best_seqs, inst)
    if final_feas and final_lp < best_lp - 1e-9:
        best_lp = final_lp; best_C = final_C
        job_lp_timeline.append((time.perf_counter() - t_job, best_lp))
    if final_feas and final_C is not None:
        pool.try_add(best_seqs, best_lp, final_C)

    # ── LP-VND polish ─────────────────────────────────────────────────────
    vnd_lp_prev = best_lp
    if best_C is not None and not math.isinf(best_lp):
        best_seqs, best_lp = lp_vnd_polish(
            best_seqs, best_lp, best_C, inst, params, p_sa,
            max_rounds=_vnd_max_rounds(inst.n),
            t_limit=max(30.0, t_limit * 0.15))
        final_lp, final_C, final_feas, _ = stage2_lp_objective(best_seqs, inst)
        if final_feas and final_lp < best_lp - 1e-9:
            best_lp = final_lp; best_C = final_C
        if best_lp < vnd_lp_prev - 1e-9:
            job_lp_timeline.append((time.perf_counter() - t_job, best_lp))
        if final_feas and final_C is not None:
            pool.try_add(best_seqs, best_lp, final_C)

    # ── Path relinking between elite pairs ───────────────────────────────
    sp_improved = False
    sp_seqs, sp_lp = (set_partition_recombine(
        pool, inst, params, m, best_lp, max_columns=80)
        if SET_PARTITION_RECOMBINE else (None, math.inf))
    if sp_seqs is not None and sp_lp < best_lp - 1e-9:
        best_seqs = sp_seqs
        best_lp = sp_lp
        _, sp_C, sp_feas, _ = stage2_lp_objective(best_seqs, inst)
        if sp_feas:
            best_C = sp_C
            pool.try_add(best_seqs, best_lp, best_C)
        job_lp_timeline.append((time.perf_counter() - t_job, best_lp))
        sp_improved = True

    relink_improved = False
    pr_t_limit = max(20.0, t_limit * 0.10)
    pr_t0      = time.perf_counter()
    for pair_fn in [pool.best_quality_pair, pool.most_diverse_pair]:
        if time.perf_counter() - pr_t0 > pr_t_limit:
            break
        sol_a, sol_b = pair_fn()
        if sol_a is None:
            continue
        for a, b in [(sol_a, sol_b), (sol_b, sol_a)]:
            if time.perf_counter() - pr_t0 > pr_t_limit:
                break
            pr_seqs, pr_lp = path_relink(
                a, b, inst, params, max_steps=40, eval_interval=3, K_lp=12)
            if pr_lp < best_lp - 1e-9:
                best_seqs = pr_seqs; best_lp = pr_lp
                _, pr_C, pr_feas, _ = stage2_lp_objective(best_seqs, inst)
                if pr_feas:
                    best_C = pr_C
                    pool.try_add(best_seqs, best_lp, best_C)
                job_lp_timeline.append((time.perf_counter() - t_job, best_lp))
                relink_improved = True

    # ── Final LP confirmation ─────────────────────────────────────────────
    final_lp, _, final_feas, final_viols = stage2_lp_objective(best_seqs, inst)
    if final_feas and final_lp < best_lp - 1e-9:
        best_lp = final_lp
        job_lp_timeline.append((time.perf_counter() - t_job, best_lp))

    # ── Sort timeline and compute total_t_best ─────────────────────────────
    # Remove duplicate timestamps (keep lowest lp at each time point).
    job_lp_timeline.sort(key=lambda x: x[0])

    # Walk forward and find the first time the final best_lp was achieved.
    total_t_best = next(
        (t for t, v in job_lp_timeline if v <= best_lp + 1e-9),
        time.perf_counter() - t_job,
    )

    elite_solutions = [
        (s.lp_obj, [seq[:] for seq in s.seqs])
        for s in sorted(pool.solutions, key=lambda s: s.lp_obj)
    ]
    chain_iters = [
        {
            "label": r[0],
            "n_iter_done": r[11],
            "n_iter_budget": r[12],
            "wall": r[13],
            "best_lp": r[4],
            "tabu": r[15],
            "sa_params": {
                "chi0": task_params[i].chi0,
                "M_stag_frac": task_params[i].M_stag_frac,
                "lp_gamma": task_params[i].lp_gamma,
                "chi_target": task_params[i].chi_target,
                "lp_repair_interval": task_params[i].lp_repair_interval,
                "max_reheats": task_params[i].max_reheats,
                "max_ils_restarts": task_params[i].max_ils_restarts,
            },
        }
        for i, r in enumerate(results)
    ]

    # ── Aggregate per-operator stats across all chains ─────────────────────
    op_stats_total: Dict[str, Dict[str, float]] = {}
    for r in results:
        for op, d in (r[14] or {}).items():
            agg = op_stats_total.setdefault(
                op, {"selected": 0, "valid": 0, "time_s": 0.0})
            agg["selected"] += d["selected"]
            agg["valid"]    += d["valid"]
            agg["time_s"]   += d["time_s"]
    for agg in op_stats_total.values():
        agg["time_s"] = round(agg["time_s"], 3)

    return best_seqs, best_lp, {
        # ── Seed portfolio timing ──────────────────────────────────────
        't_seed_construct':  portfolio_timing['t_seed_construct'],
        't_seed_lp_eval':    portfolio_timing['t_seed_lp_eval'],
        't_portfolio':       portfolio_timing['t_portfolio'],
        't_best_seed_lp':    portfolio_timing['t_best_seed_lp'],
        'seed_timing':       portfolio_timing['seed_timing'],
        # ── SA phase timing ────────────────────────────────────────────
        't_sa_start':        t_sa_dispatch_offset,
        # ── Unified time-to-best (job-relative) ───────────────────────
        't_best_lp':         total_t_best,
        # ── Standard stats ────────────────────────────────────────────
        'seed_lps':          seed_lps,
        'seed_raw_objs':     seed_raw_objs,
        'best_seed_raw':     best_seed_raw,
        'seed_portfolio':    portfolio_info,
        'all_results':       results,
        'wall':              time.perf_counter() - t_job,
        'final_feas':        final_feas,
        'final_viols':       final_viols,
        'history':           best_r[6],
        'alpha_history':     best_r[9],
        'n_iter_done':       best_r[11],
        'n_iter_budget':     best_r[12],
        'chain_iters':       chain_iters,
        'sa_chain_diversity': SA_CHAIN_DIVERSITY,
        'op_stats_total':    op_stats_total,
        'op_stats_best_chain': best_r[14],
        'tabu_stats_best_chain': best_r[15],
        'tabu_stats_all_chains': [r[15] for r in results],
        'elite_pool_size':   len(pool.solutions),
        'relinking_improved': relink_improved,
        'set_partition_improved': sp_improved,
        'elite_solutions':   elite_solutions,
        'job_lp_timeline':   job_lp_timeline,
    }
