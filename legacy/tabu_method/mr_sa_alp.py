"""
main.py — MR-ALP Solver: Job Entry Point and Batch Runner
==========================================================
§31  _run_one_job — executes one (instance, runway-count) pair end-to-end.
§32  main         — discovers instances, submits jobs, writes outputs.

Usage
-----
Configure §0 in mr_alp/config.py, then:
    python main.py

No command-line arguments — all configuration lives in config.py.
"""
from __future__ import annotations

import contextlib
import io
import math
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import multiprocessing as _mp

from mr_alp.config import (
    BATCH_MODE, INSTANCE_PATH, FOLDER, INSTANCE_RUNWAYS, KNOWN_OPTIMA,
    N_WORKERS, N_CHAINS, T_LIMIT, MAX_T_LIMIT, OUTPUT_DIR,
    SAVE_RESULTS, SAVE_PLOTS,
    RUN_RBI_OPTUNA, RUN_SA_OPTUNA,
    N_RBI_TRIALS_BASE, SA_N_TRIALS_BASE, SA_OPTUNA_SEED, SA_N_OPTUNA_JOBS,
    RBI_OPTUNA_SEED, N_OPTUNA_WORKERS,
    ATC_K, ATCS_K1, ATCS_K2, GRASP_K_VALUES, MPDS_MAX_N,
)
from mr_alp.models import (
    HeuristicParams, MRSAParams,
    RBI_PARAM_BANK, SA_PARAM_BANK, _DEFAULT_RBI,
)
from mr_alp.instance      import load_instance
from mr_alp.lp            import stage2_lp_objective, verify_and_exact_obj
from mr_alp.construction  import ramp_rbi
from mr_alp.sa            import (
    _n_iter, _sa_n_trials, _n_rbi_trials,
    optimize_rbi_params, optimize_sa_params,
)
from mr_alp.solver        import ms_mr_sa
from mr_alp.repair        import _lp_repair_params
from mr_alp.output        import (
    print_mr_result, print_summary_table,
    save_run_results, generate_plots,
    _gap_str, _is_new_bks,
)
from mr_alp.config import ELITE_POOL_MAX, ELITE_MIN_DIV  

try:
    import torch as _torch
    _GPU_AVAIL = _torch.cuda.is_available()
except ImportError:
    _GPU_AVAIL = False

_MP_CTX = _mp.get_context(
    "spawn" if (platform.system() == "Windows" or _GPU_AVAIL) else "fork"
)


# ═══════════════════════════════════════════════════════════════════════════
#   §31  JOB ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def _adaptive_t_limit(n: int, m: int, seed_lp: float,
                       bks: Optional[float]) -> float:
    """Delegate to the sa module helper (avoids circular import via re-export)."""
    from mr_alp.sa import _adaptive_t_limit as _atl
    return _atl(n, m, seed_lp, bks)


def _run_one_job(fp: str, m: int, seed: int = 0) -> dict:
    """
    Execute one (instance, runway-count) job and return a complete result dict.

    Workflow
    --------
    1. Parse instance file.
    2. Resolve TC-RBI params  (RBI_PARAM_BANK → Optuna → defaults).
    3. Resolve SA params       (SA_PARAM_BANK  → Optuna → defaults).
    4. Evaluate seed LP for adaptive time-budget computation.
    5. Run ms_mr_sa  (parallel SA + VND + path relinking).
    6. Verify feasibility; assemble result dict with Gantt arrays and timing.

    All console output produced inside this function is captured into
    result['output'] so that the batch runner can flush it in arrival order.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        inst = load_instance(fp)

        # ── TC-RBI parameter resolution ────────────────────────────────
        params = RBI_PARAM_BANK.get((inst.name, m))
        if params is None:
            if RUN_RBI_OPTUNA:
                n_t = _n_rbi_trials(inst.n, N_RBI_TRIALS_BASE)
                print(f"\n  [RBI Optuna] {inst.name.upper()} m={m} "
                      f"→ {n_t} trials ...")
                t_rbi  = time.perf_counter()
                params = optimize_rbi_params(inst, m, n_t, RBI_OPTUNA_SEED,
                                              n_jobs=N_OPTUNA_WORKERS)
                print(f"  [RBI Optuna] done in "
                      f"{time.perf_counter()-t_rbi:.1f}s  best: {params}")
            else:
                print(f"  [WARN] ({inst.name}, m={m}) not in RBI_PARAM_BANK "
                      f"— using defaults.")
                params = _DEFAULT_RBI

        # ── SA parameter resolution ────────────────────────────────────
        p_sa_tuned = False
        p_sa       = SA_PARAM_BANK.get((inst.name, m))
        if p_sa is None and RUN_SA_OPTUNA:
            from mr_alp.sa import _OPTUNA as _optuna_available
            if not _optuna_available:
                # Previously this path silently fell back to MRSAParams()
                # defaults while still recording optuna_tuned=True, which
                # contaminated the 2026-06-11 hard-case comparison.
                print(f"  [WARN] SA Optuna requested but optuna is NOT "
                      f"installed — using MRSAParams() defaults for "
                      f"({inst.name}, m={m}).")
                p_sa = MRSAParams()
            else:
                n_t   = _sa_n_trials(inst.n, SA_N_TRIALS_BASE)
                print(f"\n  [SA Optuna] {inst.name.upper()} m={m} "
                      f"→ {n_t} trials ...")
                t_opt  = time.perf_counter()
                p_sa   = optimize_sa_params(inst, m, params, n_trials=n_t,
                                             seed=SA_OPTUNA_SEED,
                                             n_jobs=SA_N_OPTUNA_JOBS)
                print(f"  [SA Optuna] done in "
                      f"{time.perf_counter()-t_opt:.1f}s  best: {p_sa}")
                p_sa_tuned = True
        if p_sa is None:
            p_sa = MRSAParams()

        # ── Adaptive time budget ───────────────────────────────────────
        base_seqs, _ = ramp_rbi(inst, m, params)
        base_lp, _, base_feas, _ = stage2_lp_objective(base_seqs, inst)
        seed_lp_est = base_lp if base_feas else math.inf
        bks         = KNOWN_OPTIMA.get(inst.name, {}).get(m)
        job_t_limit = _adaptive_t_limit(inst.n, m, seed_lp_est, bks)
        print(f"  Adaptive T_LIMIT: {job_t_limit:.0f}s  "
              f"(seed_LP={seed_lp_est:.2f}  BKS={bks})")

        # ── Main solve ─────────────────────────────────────────────────
        t0 = time.perf_counter()
        best_seqs, best_lp, stats = ms_mr_sa(
            inst, m, params, p_sa=p_sa,
            n_chains=N_CHAINS, t_limit=job_t_limit, seed=seed)
        elapsed  = time.perf_counter() - t0
        seed_lp  = min(stats['seed_lps']) if stats['seed_lps'] else math.inf
        feasible = stats['final_feas']
        _, viol_e, _, _ = verify_and_exact_obj(best_seqs, inst)

        print_mr_result(inst, m, best_seqs, best_lp, elapsed,
                        stats['seed_lps'], params, p_sa, stats)

    opt = KNOWN_OPTIMA.get(inst.name, {}).get(m)
    # Retrieve final C_lp for Gantt chart (outside redirect, no stdout impact)
    _, C_lp_final, _, _ = stage2_lp_objective(best_seqs, inst)

    return dict(
        name=inst.name, n=inst.n, m=m,
        seed_lp=seed_lp, sa_lp=best_lp,
        best_seed_raw=stats.get("best_seed_raw", math.inf),
        opt=opt, feasible=feasible, time=elapsed,
        # ── Timing breakdown ────────────────────────────────────────────
        t_seed_construct=stats.get("t_seed_construct", 0.0),
        t_seed_lp_eval  =stats.get("t_seed_lp_eval",  0.0),
        t_best_seed_lp  =stats.get("t_best_seed_lp",  0.0),
        t_sa_start      =stats.get("t_sa_start",       0.0),
        t_best_lp       =stats.get("t_best_lp",        0.0),
        # ── Per-seed timing list (for verification.txt) ────────────────
        seed_timing     =stats.get("seed_timing", []),
        # ── Other stats ─────────────────────────────────────────────────
        p_sa=p_sa, p_sa_tuned=p_sa_tuned,
        best_seqs=best_seqs,
        elite_solutions=stats.get("elite_solutions", []),
        job_lp_timeline=stats.get("job_lp_timeline", []),
        elite_pool_size=stats.get("elite_pool_size", 0),
        relinking_improved=stats.get("relinking_improved", False),
        set_partition_improved=stats.get("set_partition_improved", False),
        seed_portfolio=stats.get("seed_portfolio", []),
        history=stats.get("history", []),
        alpha_history=stats.get("alpha_history", []),
        n_iter_done=stats.get("n_iter_done", 0),
        n_iter_budget=stats.get("n_iter_budget", 0),
        chain_iters=stats.get("chain_iters", []),
        sa_chain_diversity=stats.get("sa_chain_diversity", False),
        op_stats_total=stats.get("op_stats_total", {}),
        op_stats_best_chain=stats.get("op_stats_best_chain", {}),
        tabu_stats_best_chain=stats.get("tabu_stats_best_chain", {}),
        tabu_stats_all_chains=stats.get("tabu_stats_all_chains", []),
        violations=viol_e,
        # ── Arrays for Gantt chart ────────────────────────────────────
        C_lp      =C_lp_final,
        r_arr     =inst.r.copy(),
        delta_arr =inst.delta.copy(),
        d_arr     =inst.d.copy(),
        s_mat     =inst.s.copy(),
        output    =buf.getvalue(),
    )


# ═══════════════════════════════════════════════════════════════════════════
#   §32  MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Entry point for the MR-ALP solver.

    BATCH_MODE=True   : discovers all airland*.txt files in FOLDER and submits
                        one job per (instance, runway-count) pair.
    BATCH_MODE=False  : runs all configured runway counts for INSTANCE_PATH.
    """
    print("=" * 74)
    print("  MR-ALP Solver — TC-RBI + Parallel SA + VND + Path Relinking")
    print(f"  Workers       : {N_WORKERS} processes | {N_CHAINS} chains/job")
    print(f"  T_LIMIT       : {T_LIMIT:.0f}s (adaptive, max {MAX_T_LIMIT:.0f}s)")
    print(f"  Elite pool    : max {ELITE_POOL_MAX} solutions, "
          f"min diversity {ELITE_MIN_DIV}")
    print(f"  Seed heuristics: FCFS, EDD, WEDD, ATC(K={ATC_K}), "
          f"ATCS(K1={ATCS_K1},K2={ATCS_K2}), TC-RBI, CAF, WCC, "
          f"GRASP{list(GRASP_K_VALUES)}"
          + (f", MPDS(n≤{MPDS_MAX_N})" if MPDS_MAX_N > 0 else ""))
    print(f"  Output dir    : {OUTPUT_DIR}")
    print(f"  Save results  : {SAVE_RESULTS}  |  Save plots: {SAVE_PLOTS}")

    try:
        import numba;   nb_str = "Numba JIT"
    except ImportError: nb_str = "no Numba"
    gpu_str = "PyTorch GPU" if _GPU_AVAIL else "no GPU"
    try:
        import matplotlib; mpl_str = "matplotlib"
    except ImportError:    mpl_str = "no matplotlib"
    rbi_opt = (f"RBI Optuna ON ({N_RBI_TRIALS_BASE} base trials)"
               if RUN_RBI_OPTUNA else "RBI Optuna OFF")
    sa_opt  = (f"SA Optuna ON ({SA_N_TRIALS_BASE} base trials)"
               if RUN_SA_OPTUNA  else "SA Optuna OFF")
    print(f"  Accel         : {nb_str}, {gpu_str}, {mpl_str}")
    print(f"  RBI tuning    : {rbi_opt}")
    print(f"  SA tuning     : {sa_opt}")
    print("=" * 74)

    if BATCH_MODE:
        folder = Path(FOLDER)
        files  = sorted(
            folder.glob("airland*.txt"),
            key=lambda p: int(''.join(filter(str.isdigit, p.stem)) or 0))
        if not files:
            print(f"No airland*.txt files found in {folder.resolve()}"); return
        jobs = [(str(fp), m)
                for fp in files
                for m in INSTANCE_RUNWAYS.get(fp.stem.lower(), [1])]
        print(f"  Submitting {len(jobs)} jobs to {N_WORKERS} workers...\n")

        results = []
        with ProcessPoolExecutor(max_workers=N_WORKERS,
                                  mp_context=_MP_CTX) as ex:
            futs = {ex.submit(_run_one_job, fp, m): (fp, m) for fp, m in jobs}
            for fut in as_completed(futs):
                fp, m = futs[fut]
                try:
                    r = fut.result(); results.append(r)
                    print(r["output"], end="")
                    bks_tag  = " ★NEW BKS★" if _is_new_bks(r["sa_lp"], r["opt"]) else ""
                    tune_tag = " [tuned]"   if r.get("p_sa_tuned") else ""
                    portfolio    = r.get("seed_portfolio", [])
                    sel_seeds    = [(lp, lbl) for lbl, _, lp, sel in portfolio
                                    if sel and not math.isinf(lp)]
                    best_seed_lbl = (f"  best_seed={min(sel_seeds,key=lambda x:x[0])[1]}"
                                     if sel_seeds else "")
                    print(f"  ↳ {Path(fp).stem:<12} m={m}  "
                          f"seed={r['seed_lp']:.2f}  SA={r['sa_lp']:.2f}  "
                          f"gap={_gap_str(r['sa_lp'],r['opt'])}  "
                          f"{'✓' if r['feasible'] else '✗'}  "
                          f"ttb={r.get('t_best_lp',0):.1f}s  "
                          f"({r['time']:.1f}s)"
                          f"{tune_tag}{bks_tag}{best_seed_lbl}")
                except Exception as exc:
                    print(f"  ERROR {Path(fp).stem} m={m}: {exc}")

        print_summary_table(results)
        if SAVE_RESULTS: save_run_results(results, OUTPUT_DIR)
        if SAVE_PLOTS:   generate_plots(results, OUTPUT_DIR)

    else:
        fp  = Path(INSTANCE_PATH)
        cfg = INSTANCE_RUNWAYS.get(fp.stem.lower(), [1])
        res = []
        for m in cfg:
            r = _run_one_job(str(fp), m)
            print(r["output"], end="")
            bks_tag = " ★NEW BKS★" if _is_new_bks(r["sa_lp"], r["opt"]) else ""
            tune_tag = " [tuned]"   if r.get("p_sa_tuned") else ""
            portfolio    = r.get("seed_portfolio", [])
            sel_seeds    = [(lp, lbl) for lbl, _, lp, sel in portfolio
                            if sel and not math.isinf(lp)]
            best_seed_lbl = (f"  best_seed={min(sel_seeds,key=lambda x:x[0])[1]}"
                             if sel_seeds else "")
            print(f"  ↳ {fp.stem:<12} m={m}  "
                  f"seed={r['seed_lp']:.2f}  SA={r['sa_lp']:.2f}  "
                  f"gap={_gap_str(r['sa_lp'],r['opt'])}  "
                  f"{'✓' if r['feasible'] else '✗'}  "
                  f"ttb={r.get('t_best_lp',0):.1f}s  "
                  f"({r['time']:.1f}s){bks_tag}")
            res.append(r)
        if len(cfg) > 1: print_summary_table(res)
        if SAVE_RESULTS: save_run_results(res, OUTPUT_DIR)
        if SAVE_PLOTS:   generate_plots(res, OUTPUT_DIR)


if __name__ == "__main__":
    main()


