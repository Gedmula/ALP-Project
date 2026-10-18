"""
solver.py — Main solver entry point (§19, Algorithm 3).

Workflow
────────
Phase 0 : Instance preprocessing         (Instance.__post_init__).
Phase 1 : Construction portfolio          (construction.py, §10).
Phase 2 : Per-runway initial sequences    (sequencing.py, §12).
Phase 3 : Single-runway SA per runway     (sa_single.py, §11).
Phase 4 : Stage-2 LP evaluation           (lp.py, §11).
Phase 5 : Inter-runway improvement        (inter_runway.py, §13–§16).
Phase 6 : Final Stage-2 LP + verification (lp.py + verify.py, §18).

Parallelism
───────────
Phases 2–4 are embarrassingly parallel across portfolio candidates: each
candidate's sequencing, per-runway SA, and LP evaluation are independent.
A ProcessPoolExecutor with a fork-based context runs all portfolio items
concurrently.  The worker count is controlled by N_WORKERS in config.py
and can be overridden per-call via the n_workers argument to solve().

Fork context is used on Linux because it avoids re-importing the module
tree in each worker, making startup negligible for the short-lived tasks
that portfolio evaluation represents.
"""
from __future__ import annotations

import math
import multiprocessing as _mp
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from misc.mr_alp_2.config import T_OUTER_INIT_FRAC, EPS, N_WORKERS
from misc.mr_alp_2.models import Instance, HybridParams, SingleSAParams
from misc.mr_alp_2.construction import build_assignment_portfolio
from misc.mr_alp_2.sequencing import generate_runway_sequences
from misc.mr_alp_2.sa_single import run_single_runway_sa
from misc.mr_alp_2.surrogate import surrogate_times, surrogate_penalty
from misc.mr_alp_2.lp import stage2_lp
from misc.mr_alp_2.inter_runway import inter_runway_improve
from misc.mr_alp_2.verify import verify_schedule


# ─────────────────────────────────────────────────────────────────────────────
#   Multiprocessing context
#   Fork is preferred on Linux: child processes inherit the parent's address
#   space via copy-on-write, eliminating re-import overhead.
#   Spawn is used on Windows and when CUDA is active to avoid fork hazards.
# ─────────────────────────────────────────────────────────────────────────────

def _mp_context():
    try:
        import torch as _torch
        _cuda = _torch.cuda.is_available()
    except ImportError:
        _cuda = False
    use_spawn = platform.system() == "Windows" or _cuda
    return _mp.get_context("spawn" if use_spawn else "fork")


# ─────────────────────────────────────────────────────────────────────────────
#   Top-level portfolio worker
#   Must be defined at module level so it is picklable when spawn is used.
# ─────────────────────────────────────────────────────────────────────────────

def _portfolio_worker(
    args: Tuple[int, List[List[int]], Instance, SingleSAParams, int],
) -> Tuple[int, float, Optional[List[List[int]]], Optional[np.ndarray]]:
    """
    Execute Phases 2–4 for one portfolio candidate.

    Parameters (packed into args for ProcessPoolExecutor.map)
    ----------------------------------------------------------
    idx        : portfolio index (returned for logging).
    assignment : runway assignment produced by the construction rule.
    inst       : ALP instance.
    sa_p       : single-runway SA parameters.
    seed       : master RNG seed; per-runway offsets are derived from it.

    Returns
    -------
    (idx, lp_val, seqs, x)
    lp_val is inf and seqs/x are None when the LP is infeasible.
    """
    idx, assignment, inst, sa_p, seed = args

    # Phase 2: choose the best initial sequence per runway
    seqs = generate_runway_sequences(assignment, inst)

    # Phase 3: independent single-runway SA for each runway
    for rho in range(inst.m):
        if not seqs[rho]:
            continue
        seqs[rho], _ = run_single_runway_sa(
            seqs[rho], inst, sa_p,
            seed=seed + idx * inst.m + rho)

    # Phase 4: Stage-2 LP evaluation of the combined assignment
    lp_val, x = stage2_lp(seqs, inst)
    if math.isinf(lp_val) or x is None:
        return idx, math.inf, None, None
    return idx, lp_val, seqs, x


# ─────────────────────────────────────────────────────────────────────────────
#   Size-adaptive SA parameter selection
# ─────────────────────────────────────────────────────────────────────────────

def _sa_params_for_size(n: int) -> SingleSAParams:
    """
    Return SingleSAParams calibrated to instance size n.

    Larger instances require a slower cooling rate and more moves per
    temperature level to explore the wider feasible landscape adequately.
    """
    if n <= 20:
        return SingleSAParams(alpha=0.97,  N_iter=100, T_min=1e-4,
                              I_max=400,   M_stag=50,  chi0=0.50)
    if n <= 50:
        return SingleSAParams(alpha=0.98,  N_iter=200, T_min=1e-4,
                              I_max=700,   M_stag=80,  chi0=0.50)
    if n <= 150:
        return SingleSAParams(alpha=0.995, N_iter=350, T_min=1e-5,
                              I_max=1200,  M_stag=120, chi0=0.40)
    return SingleSAParams(alpha=0.997, N_iter=500, T_min=1e-6,
                          I_max=2000,  M_stag=200, chi0=0.35)


# ─────────────────────────────────────────────────────────────────────────────
#   Main solver
# ─────────────────────────────────────────────────────────────────────────────

def solve(
    inst:          Instance,
    params:        Optional[HybridParams]   = None,
    sa_p:          Optional[SingleSAParams] = None,
    n_outer_iters: int   = 200,
    seed:          int   = 0,
    t_limit:       float = 3600.0,
    n_workers:     int   = N_WORKERS,
    verbose:       bool  = True,
) -> Dict[str, Any]:
    """
    Run the hybrid decomposition solver on a preprocessed Instance.

    Parameters
    ----------
    inst          : preprocessed Instance (Phase 0 constants already set).
    params        : HybridParams for construction and inter-runway layers;
                   defaults to HybridParams() if None.
    sa_p          : SingleSAParams for runway-level SA; size-adaptive
                   defaults applied if None.
    n_outer_iters : maximum Phase 5 inter-runway improvement iterations.
    seed          : master RNG seed; individual phases derive offsets from it.
    t_limit       : wall-clock budget in seconds.
    n_workers     : number of parallel worker processes for portfolio
                   evaluation (Phases 2–4).  Defaults to N_WORKERS from
                   config.py; override here for per-call control.
    verbose       : whether to print phase-level progress.

    Returns
    -------
    dict with keys:
        seqs     : best multi-runway schedule  List[List[int]]
                   (None if no feasible schedule was found).
        lp_obj   : LP objective of the best schedule (math.inf if none).
        x        : LP landing times np.ndarray (None if infeasible).
        wall_s   : total elapsed wall time in seconds.
        t_best   : wall time at which the best LP objective was first
                   achieved.
        verified : True if the final schedule passed all verification checks.
        report   : VerificationReport (None if no feasible schedule).
    """
    if params is None:
        params = HybridParams()
    if sa_p is None:
        sa_p = _sa_params_for_size(inst.n)

    t0       = time.perf_counter()
    deadline = t0 + t_limit

    def _elapsed() -> float:
        return time.perf_counter() - t0

    def _over_budget() -> bool:
        return time.perf_counter() >= deadline

    if verbose:
        print(f"\n{'═'*64}")
        print(f"  Hybrid MR-ALP   instance={inst.name}  "
              f"n={inst.n}  m={inst.m}")
        print(f"  Budget: {t_limit:.0f}s   seed={seed}   "
              f"workers={n_workers}")
        print(f"{'═'*64}")

    # ─────────────────────────────────────────────────────────────────────
    #   Phase 1: Construction portfolio
    # ─────────────────────────────────────────────────────────────────────
    if verbose:
        print(f"  [Phase 1] Building construction portfolio …")
    portfolio = build_assignment_portfolio(inst, params)
    if verbose:
        print(f"           {len(portfolio)} assignments generated  "
              f"({_elapsed():.1f}s)")

    # ─────────────────────────────────────────────────────────────────────
    #   Phases 2–4: Parallel portfolio evaluation
    #   Each portfolio candidate runs sequencing + per-runway SA + LP
    #   in its own worker process.
    # ─────────────────────────────────────────────────────────────────────
    if verbose:
        print(f"  [Phase 2–4] Sequencing → SA → LP  "
              f"({len(portfolio)} items  ×  {n_workers} workers) …")

    tasks = [
        (idx, assignment, inst, sa_p, seed)
        for idx, assignment in enumerate(portfolio)
    ]
    n_w = min(n_workers, len(tasks))

    best_seqs: Optional[List[List[int]]] = None
    best_lp:   float                     = math.inf
    best_x:    Optional[np.ndarray]      = None
    t_best:    float                     = 0.0

    ctx = _mp_context()
    with ProcessPoolExecutor(max_workers=n_w, mp_context=ctx) as ex:
        for idx, lp_val, seqs, x in ex.map(_portfolio_worker, tasks):
            if not math.isinf(lp_val) and seqs is not None and x is not None:
                if lp_val < best_lp - EPS:
                    best_seqs = [s[:] for s in seqs]
                    best_lp   = lp_val
                    best_x    = x.copy()
                    t_best    = _elapsed()
                    if verbose:
                        print(f"           Portfolio {idx}: "
                              f"LP={best_lp:.4f}  "
                              f"({_elapsed():.1f}s)  ★")
                elif verbose:
                    print(f"           Portfolio {idx}: "
                          f"LP={lp_val:.4f}  "
                          f"({_elapsed():.1f}s)")
            elif verbose:
                print(f"           Portfolio {idx}: "
                      f"LP=inf  ({_elapsed():.1f}s)")

    if best_seqs is None:
        if verbose:
            print("  No feasible schedule found in construction phase.")
        return {
            'seqs': None, 'lp_obj': math.inf, 'x': None,
            'wall_s': _elapsed(), 't_best': 0.0,
            'verified': False, 'report': None,
        }

    # ─────────────────────────────────────────────────────────────────────
    #   Phase 5: Inter-runway improvement (sequential — each iteration
    #   depends on the result of the previous one)
    # ─────────────────────────────────────────────────────────────────────
    remaining = max(0.0, deadline - time.perf_counter())
    if remaining > 1.0 and not _over_budget():
        if verbose:
            print(f"  [Phase 5] Inter-runway improvement  "
                  f"(budget remaining: {remaining:.0f}s) …")
        T_outer = max(best_lp * T_OUTER_INIT_FRAC, 1.0)
        seqs_imp, lp_imp, x_imp = inter_runway_improve(
            best_seqs, best_lp, best_x,
            inst, params, sa_p,
            T_outer=T_outer,
            n_outer=n_outer_iters,
            seed=seed + 9999,
        )
        if not math.isinf(lp_imp) and lp_imp < best_lp - EPS:
            best_seqs = seqs_imp
            best_lp   = lp_imp
            best_x    = x_imp
            t_best    = _elapsed()
            if verbose:
                print(f"           LP improved to {best_lp:.4f}  "
                      f"({_elapsed():.1f}s)  ★")
        elif verbose:
            print(f"           No LP improvement  ({_elapsed():.1f}s)")


    # ─────────────────────────────────────────────────────────────────────
    #   Phase 5.5: Assignment balance repair
    #   Triggered only when max−min runway counts exceed _skew_threshold.
    #   Accepts transfers that do not increase LP beyond tolerance so
    #   solution quality is preserved while load skew is reduced.
    # ─────────────────────────────────────────────────────────────────────
    counts_now = [len(s) for s in best_seqs]
    imbalance  = max(counts_now) - min(counts_now)
    thresh     = _skew_threshold(inst.n, inst.m)

    if inst.m > 1 and imbalance > thresh and not _over_budget():
        if verbose:
            print(f"  [Phase 5.5] Balance repair  "
                  f"(imbalance={imbalance}, threshold={thresh}) …")
        best_seqs, best_lp, best_x = _rebalance_assignment(
            best_seqs, best_lp, best_x, inst, verbose=verbose)
        final_counts  = [len(s) for s in best_seqs]
        final_imbal   = max(final_counts) - min(final_counts)
        if verbose:
            dist_str = "/".join(str(c) for c in final_counts)
            print(f"           Distribution: [{dist_str}]  "
                  f"imbalance={final_imbal}  "
                  f"LP={best_lp:.4f}  ({_elapsed():.1f}s)")

    # ─────────────────────────────────────────────────────────────────────
    #   Phase 6: Final LP re-solve + verification
    # ─────────────────────────────────────────────────────────────────────
    if verbose:
        print(f"  [Phase 6] Final LP re-solve + verification …")

    final_lp, final_x = stage2_lp(best_seqs, inst)
    if not math.isinf(final_lp) and final_lp < best_lp - EPS:
        best_lp = final_lp
        best_x  = final_x
        t_best  = _elapsed()

    passed, report = verify_schedule(best_seqs, best_x, best_lp, inst)

    wall = _elapsed()
    if verbose:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  Final LP obj : {best_lp:.6f}  [{status}]  "
              f"wall={wall:.1f}s  t_best={t_best:.1f}s")
        print(f"{'═'*64}\n")

    return {
        'seqs':     best_seqs,
        'lp_obj':   best_lp,
        'x':        best_x,
        'wall_s':   wall,
        't_best':   t_best,
        'verified': passed,
        'report':   report,
    }

# ─────────────────────────────────────────────────────────────────────────────
#   Phase 5.5: Assignment balance check and repair
# ─────────────────────────────────────────────────────────────────────────────

def _skew_threshold(n: int, m: int) -> int:
    """
    Maximum tolerated imbalance: max(runway count) − min(runway count).

    Allows a deviation of up to n // m aircraft (one full runway's worth),
    with a hard minimum of 3 so small instances are not over-constrained.
    This flags extreme cases like the 4/40 split seen in airland7_m2
    (threshold = 22, imbalance = 36 → triggers) while accepting the
    natural ±1–2 aircraft variation that arises from n % m ≠ 0.
    """
    return max(3, n // m)


def _rebalance_assignment(
    seqs:    List[List[int]],
    lp_init: float,
    x_init:  Optional[np.ndarray],
    inst:    Instance,
    verbose: bool = True,
) -> Tuple[List[List[int]], float, Optional[np.ndarray]]:
    """
    Iteratively transfer aircraft from the most overloaded runway to the
    most underloaded runway until imbalance ≤ _skew_threshold(n, m).

    At each step:
      1. Identify rho_heavy (max count) and rho_light (min count).
      2. For every aircraft on rho_heavy, find the best feasible
         insertion position on rho_light by surrogate penalty.
      3. LP-evaluate the top K = min(5, n_candidates) proxy candidates.
      4. Accept the move if LP_new ≤ LP_current + tolerance, where:
             tolerance = 0                      when LP_init = 0
             tolerance = max(1.0, LP_init×0.01) otherwise (≤1% increase)

    The surrogate feasibility screen (surrogate_times deadline check)
    prevents any LP call on proposals that already violate hard deadlines.
    At most n transfer attempts are made (safety cap).

    Returns (seqs, best_lp, best_x) — rebalanced where possible, unchanged
    when no valid balanced assignment can be found within tolerance.
    """
    m         = inst.m
    n         = inst.n
    threshold = _skew_threshold(n, m)
    tol       = 0.0 if lp_init < 1e-9 else max(1.0, lp_init * 0.01)

    best_seqs = [s[:] for s in seqs]
    best_lp   = lp_init
    best_x    = x_init.copy() if x_init is not None else None

    for _attempt in range(n):   # at most n transfers
        counts    = [len(s) for s in best_seqs]
        imbalance = max(counts) - min(counts)
        if imbalance <= threshold:
            break

        rho_heavy = int(np.argmax(counts))
        rho_light = int(np.argmin(counts))

        # ── Surrogate pre-screen: best position on rho_light per aircraft ──
        candidates: List[Tuple[float, int, int]] = []  # (score, j, best_pos)

        for j in best_seqs[rho_heavy]:
            seq_h_new = [k for k in best_seqs[rho_heavy] if k != j]

            # Removing j from rho_heavy can only decrease downstream
            # surrogate times so seq_h_new is always surrogate-feasible;
            # check anyway as a defensive guard.
            if seq_h_new:
                C_h = surrogate_times(seq_h_new, inst)
                if any(C_h[q] > float(inst.d[seq_h_new[q]]) + 1e-9
                       for q in range(len(seq_h_new))):
                    continue  # should never trigger

            # Find the best (lowest surrogate penalty) feasible position
            # for j on rho_light.
            best_pos_score = math.inf
            best_pos       = -1
            for p in range(len(best_seqs[rho_light]) + 1):
                seq_l_new = (best_seqs[rho_light][:p]
                             + [j]
                             + best_seqs[rho_light][p:])
                C_l = surrogate_times(seq_l_new, inst)
                if any(C_l[q] > float(inst.d[seq_l_new[q]]) + 1e-9
                       for q in range(len(seq_l_new))):
                    continue
                score = surrogate_penalty(seq_l_new, inst)
                if score < best_pos_score:
                    best_pos_score = score
                    best_pos       = p

            if best_pos >= 0:
                candidates.append((best_pos_score, j, best_pos))

        if not candidates:
            if verbose:
                print(f"           Rebalance: no feasible transfer found "
                      f"(imbalance {imbalance} remains)")
            break

        # ── LP evaluation on top K proxy candidates ─────────────────────
        candidates.sort(key=lambda x: x[0])
        K_eval = min(5, len(candidates))

        best_move_lp   = best_lp + tol + 1.0   # sentinel
        best_move_seqs = None
        best_move_x    = None

        for _, j, best_pos in candidates[:K_eval]:
            cand = [s[:] for s in best_seqs]
            cand[rho_heavy] = [k for k in cand[rho_heavy] if k != j]
            cand[rho_light] = (cand[rho_light][:best_pos]
                               + [j]
                               + cand[rho_light][best_pos:])
            cand_lp, cand_x = stage2_lp(cand, inst)
            if (not math.isinf(cand_lp)
                    and cand_x is not None
                    and cand_lp < best_move_lp):
                best_move_lp   = cand_lp
                best_move_seqs = [s[:] for s in cand]
                best_move_x    = cand_x.copy()

        # ── Accept or stop ───────────────────────────────────────────────
        if (best_move_seqs is not None
                and best_move_lp <= best_lp + tol + 1e-9):
            new_counts    = [len(s) for s in best_move_seqs]
            new_imbalance = max(new_counts) - min(new_counts)
            if verbose:
                lp_note = (f"  LP {best_lp:.4f} → {best_move_lp:.4f}"
                           if abs(best_move_lp - best_lp) > 1e-9 else "")
                print(f"           Rebalance: imbalance "
                      f"{imbalance} → {new_imbalance}{lp_note}")
            best_seqs = best_move_seqs
            best_lp   = best_move_lp
            best_x    = best_move_x
        else:
            if verbose:
                print(f"           Rebalance: no valid transfer within "
                      f"tolerance (imbalance {imbalance} remains)")
            break

    return best_seqs, best_lp, best_x