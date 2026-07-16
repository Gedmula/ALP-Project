"""
=============================================================================
construction.py — MR-ALP Solver: TC-RBI, Seed Heuristics H1–H9, Portfolio
=============================================================================
§8   TC-RBI priority measures and insertion cost components
§9   ramp_rbi (TC-RBI construction) and inter_runway_repair
§25  Seed heuristics H1–H9  (FCFS, EDD, WEDD, ATC, ATCS, CAF, MPDS, WCC, GRASP)
§25b _build_seed_portfolio — LP-screened portfolio with per-seed wall-time tracking

Timing contract
---------------
_build_seed_portfolio measures, for each seed heuristic:
  t_construct   wall time for the construction call.
  t_lp_eval     wall time for the stage2_lp_objective call.
  t_job_relative cumulative time from portfolio start when this seed's LP was computed.

It returns a portfolio_timing dict (see docstring) consumed by ms_mr_sa
in solver.py to form the unified job_lp_timeline and total_t_best.
"""


import math
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Internal imports (construction depends on: config, models, instance, lp, proxy)
from mr_alp.config import (
    ATC_K, ATCS_K1, ATCS_K2, ELITE_MIN_DIV, GRASP_K_VALUES,
    MPDS_MAX_N, USE_ALL_SEEDS, N_CHAINS,
)
from mr_alp.models    import Instance, HeuristicParams
from mr_alp.instance  import (
    surrogate_times, surrogate_penalty, runway_feasible,
    _insert_times_kernel,
)
from mr_alp.lp        import stage2_lp_objective, verify_and_exact_obj
from mr_alp.proxy     import (
    _rwy_proxy_components, init_proxy_arrays, compute_proxy,
)

try:
    import numba as nb
    _NUMBA = True
except ImportError:
    _NUMBA = False

# ═══════════════════════════════════════════════════════════════════════════
#   §8  TC-RBI PRIORITY MEASURES AND INSERTION COST COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════

def compute_priorities(inst: Instance) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute aircraft priority measures for TC-RBI screening.

    AF_j  = (d_j − r_j) − mean_bilateral_separation_j
    CR_j  = (g_j + h_j) / max(AF_j, ε)

    CR_j (critical ratio) combines penalty rate and time-pressure; aircraft
    with high CR are inserted first in the RBI loop.
    """
    s_sym = (inst.s + inst.s.T) / 2.0
    np.fill_diagonal(s_sym, 0.0)
    s_b = s_sym.sum(axis=1) / max(inst.n - 1, 1)
    AF  = (inst.d - inst.r) - s_b
    CR  = (inst.g + inst.h) / np.maximum(AF, inst.eps)
    return AF, CR


def minmax_norm(arr: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + eps)


def _compute_insert_times(
    j: int, p: int,
    seq: List[int], C_hat_seq: List[float],
    inst: Instance,
) -> Tuple[List[float], bool]:
    """Surrogate times after inserting aircraft j at position p; dispatches to Numba."""
    if _NUMBA:
        L  = len(seq)
        sa = np.asarray(seq,       dtype=np.int32) if L else np.empty(0, dtype=np.int32)
        Ca = np.asarray(C_hat_seq, dtype=np.float64) if C_hat_seq else np.empty(0, dtype=np.float64)
        C_n, ok = _insert_times_kernel(j, p, sa, Ca, inst.r, inst.s, inst.d)
        return list(C_n), bool(ok)

    seq_n = seq[:p] + [j] + seq[p:]
    L_n   = len(seq_n)
    C_n   = list(C_hat_seq[:p])
    if p == 0:
        C_n.append(float(inst.r[j]))
    else:
        C_n.append(max(float(inst.r[j]),
                       C_n[p - 1] + float(inst.s[seq_n[p - 1], j])))
    for q in range(p + 1, L_n):
        prev, cur = seq_n[q - 1], seq_n[q]
        C_n.append(max(float(inst.r[cur]),
                       C_n[q - 1] + float(inst.s[prev, cur])))
    for q in range(p, L_n):
        if C_n[q] > inst.d[seq_n[q]] + 1e-9:
            return C_n, False
    return C_n, True


def _is_feasible_anywhere(
    j: int,
    sequences: List[List[int]],
    C_hats: List[List[float]],
    inst: Instance,
) -> bool:
    """Return True iff aircraft j has at least one feasible insertion position."""
    m = len(sequences)
    for rho in range(m):
        _, ok = _compute_insert_times(j, len(sequences[rho]),
                                       sequences[rho], C_hats[rho], inst)
        if ok:
            return True
    for rho in range(m):
        for p in range(len(sequences[rho])):
            _, ok = _compute_insert_times(j, p, sequences[rho], C_hats[rho], inst)
            if ok:
                return True
    return False


def target_conflict_insert(j: int, p: int, seq: List[int], inst: Instance) -> float:
    """
    Incremental weighted target-time conflict from inserting j at position p.

    ΔTC = Σ_{i∈pred} 0.5·(p_i+p_j)·max(s[i,j]−(δ_j−δ_i),0)
        + Σ_{k∈succ} 0.5·(p_j+p_k)·max(s[j,k]−(δ_k−δ_j),0)
    """
    pj = float(inst.p_arr[j]); cost = 0.0
    pred = seq[:p]
    if pred:
        pa = np.asarray(pred, dtype=np.intp)
        v  = inst.s[pa, j] - (inst.delta[j] - inst.delta[pa])
        cost += float((0.5 * (inst.p_arr[pa] + pj) * np.maximum(v, 0.0)).sum())
    succ = seq[p:]
    if succ:
        sa = np.asarray(succ, dtype=np.intp)
        v  = inst.s[j, sa] - (inst.delta[sa] - inst.delta[j])
        cost += float((0.5 * (pj + inst.p_arr[sa]) * np.maximum(v, 0.0)).sum())
    return cost


def lower_bound_tardiness(
    seq: List[int], C_hat: List[float], inst: Instance
) -> float:
    if not seq:
        return 0.0
    sa = np.asarray(seq, dtype=np.intp)
    Ca = np.asarray(C_hat)
    return float((inst.h[sa] * np.maximum(Ca - inst.delta[sa], 0.0)).sum())


def count_balance_delta(
    rho: int, sequences: List[List[int]], inst: Instance
) -> float:
    m  = len(sequences); n = inst.n
    t  = sum(len(s) for s in sequences)
    ol = len(sequences[rho])
    return (((ol + 1 - (t + 1) / m) ** 2 - (ol - t / m) ** 2)
            * float(inst.Pen_bar) / max((n / m) ** 2, 1.0))


def evaluate_insertion(
    j: int, rho: int, p: int,
    sequences: List[List[int]], C_hats: List[List[float]],
    B_bar: float, inst: Instance, params: HeuristicParams,
) -> Tuple[float, List[float]]:
    """
    Composite insertion cost = μ_TC·ΔTC + μ_late·ΔLate + μ_count·Δcount + μ_sep·ΔSep.
    Returns (cost, C_new_for_runway_rho).  Returns (inf, []) if infeasible.
    """
    seq, C_hat_seq = sequences[rho], C_hats[rho]
    L = len(seq)
    C_n, ok = _compute_insert_times(j, p, seq, C_hat_seq, inst)
    if not ok:
        return math.inf, []

    seq_n  = seq[:p] + [j] + seq[p:]
    dTC    = target_conflict_insert(j, p, seq, inst)
    dLate  = (lower_bound_tardiness(seq_n, C_n, inst)
              - lower_bound_tardiness(seq, C_hat_seq, inst))
    dCount = count_balance_delta(rho, sequences, inst)

    if L == 0:        dSep_raw = 0.0
    elif p == 0:      dSep_raw = float(inst.s[j, seq[0]])
    elif p == L:      dSep_raw = float(inst.s[seq[-1], j])
    else:
        a, b     = seq[p - 1], seq[p]
        dSep_raw = max(0.0, float(inst.s[a, j]) + float(inst.s[j, b])
                          - float(inst.s[a, b]))

    cost = (params.mu_tc * dTC + params.mu_late * dLate
            + params.mu_count * dCount
            + params.mu_sep * inst.h_bar * dSep_raw)
    return cost, C_n


def _candidate_positions(
    j: int, rho: int,
    sequences: List[List[int]], inst: Instance,
) -> List[int]:
    """All positions for n ≤ 100; a centred window around δ_j-order for larger instances."""
    seq = sequences[rho]; L = len(seq)
    if inst.n <= 100:
        return list(range(L + 1))
    p0 = next((p for p, u in enumerate(seq)
               if inst.delta[u] >= inst.delta[j]), L)
    return sorted(set(range(max(0, p0 - 2), min(L + 1, p0 + 3))) | {0, L})


def _best_insertions(
    j: int, m: int,
    sequences: List[List[int]], C_hats: List[List[float]],
    B_bar: float, inst: Instance, params: HeuristicParams,
) -> Tuple[Tuple[float, int, int, List[float]], float]:
    """
    Best (runway, position) for j and regret (cost of second-best runway).
    Returns (best1=(c1,rho1,p1,C1), c2).
    """
    per_rho = []
    for rho in range(m):
        rc, rp, rC = math.inf, 0, []
        for p in _candidate_positions(j, rho, sequences, inst):
            c, Cn = evaluate_insertion(j, rho, p, sequences, C_hats, B_bar, inst, params)
            if c < rc:
                rc, rp, rC = c, p, Cn
        per_rho.append((rc, rp, rC))

    sr = sorted(range(m), key=lambda r: per_rho[r][0])
    c1, p1, C1 = per_rho[sr[0]]
    best1 = (c1, sr[0], p1, C1)

    if m > 1:
        c2 = per_rho[sr[1]][0]
    else:
        all_c = sorted(
            evaluate_insertion(j, 0, p, sequences, C_hats, B_bar, inst, params)[0]
            for p in range(len(sequences[0]) + 1)
        )
        c2 = all_c[1] if len(all_c) > 1 else math.inf

    return best1, c2


def min_violation_insert(
    j: int, sequences: List[List[int]], inst: Instance
) -> Tuple[int, int, List[float]]:
    """Least-infeasible insertion for aircraft j (forced-set fallback)."""
    best_V, best_rho, best_p, best_C = math.inf, 0, 0, []
    for rho, seq in enumerate(sequences):
        for p in range(len(seq) + 1):
            seq_n = seq[:p] + [j] + seq[p:]
            C_t   = surrogate_times(seq_n, inst)
            V     = sum(max(C_t[q] - inst.d[seq_n[q]], 0.0)
                        for q in range(len(seq_n)))
            if V < best_V:
                best_V, best_rho, best_p, best_C = V, rho, p, C_t
    return best_rho, best_p, best_C


def inter_runway_repair(
    sequences: List[List[int]], C_hats: List[List[float]],
    inst: Instance, params: HeuristicParams,
    max_iterations: int = 150,
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    Improve runway-load balance by relocating high-TC aircraft from overloaded
    to underloaded runways.  Applied as a post-processing step after RBI.
    """
    m = len(sequences)
    if m == 1:
        return sequences, C_hats

    sequences = [list(s) for s in sequences]
    C_hats    = [list(c) for c in C_hats]
    mean_load = inst.n / m

    for _ in range(max_iterations):
        rho_src = max(range(m), key=lambda r: len(sequences[r]))
        if len(sequences[rho_src]) <= mean_load + 1:
            break
        B_bar = sum(C_hats[r][-1] if C_hats[r] else 0.0 for r in range(m)) / m

        best_tc, best_sp = -1.0, -1
        for sp, j in enumerate(sequences[rho_src]):
            seq_no = sequences[rho_src][:sp] + sequences[rho_src][sp + 1:]
            tc = target_conflict_insert(j, sp, seq_no, inst)
            if tc > best_tc:
                best_tc, best_sp = tc, sp
        if best_sp == -1:
            break

        j_move = sequences[rho_src][best_sp]
        best_c, best_rd, best_dp, best_Cn = math.inf, -1, -1, []
        for rd in range(m):
            if rd == rho_src:
                continue
            for dp in range(len(sequences[rd]) + 1):
                c, Cn = evaluate_insertion(j_move, rd, dp, sequences, C_hats,
                                            B_bar, inst, params)
                if c < best_c:
                    best_c, best_rd, best_dp, best_Cn = c, rd, dp, Cn
        if best_rd == -1 or math.isinf(best_c):
            break

        sequences[rho_src].pop(best_sp)
        C_hats[rho_src] = surrogate_times(sequences[rho_src], inst)
        sequences[best_rd].insert(best_dp, j_move)
        C_hats[best_rd] = best_Cn

    return sequences, C_hats


# ═══════════════════════════════════════════════════════════════════════════
#   §9  TC-RBI CONSTRUCTION HEURISTIC
# ═══════════════════════════════════════════════════════════════════════════

def ramp_rbi(
    inst: Instance, m: int, params: HeuristicParams
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    TC-RBI: Target-Conflict Regret-Based Insertion.

    Iteratively inserts aircraft by minimising a composite insertion cost
    (ΔTC + ΔLate + Δcount + ΔSep), breaking ties by regret (second-best
    minus best runway cost) and criticality ratio CR_j.  Aircraft with no
    feasible position are handled by minimum-violation insertion.

    Returns (sequences, C_hats).
    """
    n, eps = inst.n, inst.eps
    _, CR  = compute_priorities(inst)
    sequences = [[] for _ in range(m)]
    C_hats    = [[] for _ in range(m)]
    B         = [0.0] * m; B_bar = 0.0
    U         = list(range(n)); F = set()

    def committed(rho):
        return C_hats[rho][-1] if C_hats[rho] else 0.0

    def refresh_forced():
        for j in list(U):
            if j in F:
                continue
            if not _is_feasible_anywhere(j, sequences, C_hats, inst):
                F.add(j)

    def do_insert(j, rho, p, C_new):
        nonlocal B_bar
        sequences[rho].insert(p, j); C_hats[rho] = C_new
        B[rho] = committed(rho); B_bar = sum(B) / m
        U.remove(j); F.discard(j)

    while U:
        while F & set(U):
            j_star = max([j for j in U if j in F], key=lambda j: CR[j])
            rho, p, C_new = min_violation_insert(j_star, sequences, inst)
            seq_new = sequences[rho][:p] + [j_star] + sequences[rho][p:]
            sequences[rho] = seq_new; C_hats[rho] = C_new
            B[rho] = committed(rho); B_bar = sum(B) / m
            U.remove(j_star); F.discard(j_star); refresh_forced()
        if not U:
            break
        U_avail = [j for j in U if j not in F]
        if not U_avail:
            break

        tau   = min(B)
        urg   = np.array([1.0 / max(float(inst.delta[j]) - tau, eps)
                           for j in U_avail])
        cr_arr = np.array([CR[j] for j in U_avail])
        screen = (params.eta * minmax_norm(cr_arr, eps)
                  + (1 - params.eta) * minmax_norm(urg, eps))

        q_eff = (len(U_avail) if n <= 100
                 else min(150, max(50, int(0.25 * len(U_avail)))))
        top   = np.argsort(screen)[::-1][:q_eff]
        U_q   = [U_avail[i] for i in top]

        info = {}
        for j in U_q:
            (c1, rho1, p1, Cn1), c2 = _best_insertions(
                j, m, sequences, C_hats, B_bar, inst, params)
            info[j] = (c1, rho1, p1, Cn1, c2)

        finite = [info[j][4] - info[j][0] for j in U_q
                  if info[j][4] < math.inf]
        R_max  = ((max(finite) + inst.h_bar * inst.T_span)
                  if finite else inst.h_bar * inst.T_span)

        best_c  = np.array([info[j][0] for j in U_q])
        regret  = np.array([
            (info[j][4] - info[j][0]) if info[j][4] < math.inf else R_max
            for j in U_q
        ])
        cr_uq   = np.array([CR[j] for j in U_q])
        score   = (minmax_norm(best_c, eps)
                   - 0.20 * minmax_norm(regret, eps)
                   - 0.10 * minmax_norm(cr_uq, eps))

        j_star             = U_q[int(np.argmin(score))]
        c_s, rho_s, p_s, Cn_s, _ = info[j_star]
        if math.isinf(c_s):
            F.add(j_star); continue
        do_insert(j_star, rho_s, p_s, Cn_s); refresh_forced()

    sequences, C_hats = inter_runway_repair(sequences, C_hats, inst, params)
    return sequences, C_hats


# ═══════════════════════════════════════════════════════════════════════════
#   §25  SEED CONSTRUCTION HEURISTICS  H1 – H9
# ═══════════════════════════════════════════════════════════════════════════

def seed_fcfs(inst: Instance, m: int) -> Tuple[List[List[int]], List[List[float]]]:
    """
    H1 — First-Come First-Served.

    Processes aircraft in non-decreasing r_j order.  Each aircraft is assigned
    to the runway with the smallest current committed time; the committed time
    is updated respecting the consecutive-predecessor separation from the last
    landed aircraft on that runway.

    Complexity: O(n log n + n·m).
    Reference: Beasley et al. (2000).
    """
    order     = list(np.argsort(inst.r))
    seqs      = [[] for _ in range(m)]
    committed = [0.0] * m
    for j in order:
        rho  = int(np.argmin(committed))
        prev = seqs[rho][-1] if seqs[rho] else -1
        sep  = float(inst.s[prev, j]) if prev >= 0 else 0.0
        seqs[rho].append(j)
        committed[rho] = max(float(inst.r[j]), committed[rho] + sep)
    return seqs, [surrogate_times(seq, inst) for seq in seqs]


def seed_edd_balanced(inst: Instance, m: int) -> Tuple[List[List[int]], List[List[float]]]:
    """
    H2 — EDD-Balanced (Earliest Target-Time, round-robin assignment).

    Sorts aircraft by δ_j and assigns them round-robin across runways,
    giving each runway a balanced target-time-ordered sequence.

    Complexity: O(n log n).
    Reference: Pinedo (2016), §3.2 (EDD rule optimal for 1 ‖ L_max).
    """
    order = list(np.argsort(inst.delta))
    seqs  = [[] for _ in range(m)]
    for k, j in enumerate(order):
        seqs[k % m].append(j)
    return seqs, [surrogate_times(seq, inst) for seq in seqs]


def seed_wedd(inst: Instance, m: int) -> Tuple[List[List[int]], List[List[float]]]:
    """
    H3 — Weighted EDD.

    Sorts by δ_j / (g_j + h_j): aircraft with small target time relative to
    their combined penalty rate are scheduled first.  Assignment is to the
    runway with the smallest current committed time.

    Complexity: O(n log n + n·m).
    Reference: Ernst, Krishnamoorthy & Storer (1999).
    """
    scores    = inst.delta / np.maximum(inst.g + inst.h, inst.eps)
    order     = list(np.argsort(scores))
    seqs      = [[] for _ in range(m)]
    C_hats    = [[] for _ in range(m)]
    committed = [0.0] * m
    for j in order:
        rho = int(np.argmin(committed))
        seqs[rho].append(j)
        C_hats[rho] = surrogate_times(seqs[rho], inst)
        committed[rho] = C_hats[rho][-1] if C_hats[rho] else 0.0
    return seqs, C_hats


def seed_atc(
    inst: Instance, m: int, K: float = ATC_K
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    H4 — Apparent Tardiness Cost (ATC).

    When a runway opens at time t, scores each unscheduled aircraft j as:
        I_j(t) = h_j/s̄_j · exp(−max(δ_j − s̄_j − t, 0) / (K·s̄))

    Large K → WSPT rule; small K → minimum-slack rule (Pinedo 2016, §14.2).
    Complexity: O(n²·m).
    Reference: Vepsäläinen & Morton (1987).
    """
    s_sym  = (inst.s + inst.s.T) / 2.0; np.fill_diagonal(s_sym, 0.0)
    s_mean = s_sym.sum(axis=1) / max(inst.n - 1, 1)
    s_bar  = float(np.mean(s_mean[s_mean > 0])) if np.any(s_mean > 0) else 1.0
    Ks     = max(K * s_bar, 1e-9)

    seqs        = [[] for _ in range(m)]
    C_hats      = [[] for _ in range(m)]
    committed   = [0.0] * m
    unscheduled = list(range(inst.n))

    while unscheduled:
        rho = int(np.argmin(committed)); t = committed[rho]
        best_j, best_idx = -1, -math.inf
        for j in unscheduled:
            slack = max(float(inst.delta[j]) - s_mean[j] - t, 0.0)
            idx   = (float(inst.h[j]) / max(s_mean[j], 1e-9)) * math.exp(-slack / Ks)
            if idx > best_idx:
                best_idx, best_j = idx, j
        seqs[rho].append(best_j)
        C_hats[rho]  = surrogate_times(seqs[rho], inst)
        committed[rho] = C_hats[rho][-1] if C_hats[rho] else 0.0
        unscheduled.remove(best_j)

    return seqs, C_hats


def seed_atcs(
    inst: Instance, m: int, K1: float = ATCS_K1, K2: float = ATCS_K2
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    H5 — Apparent Tardiness Cost with Setups (ATCS).

    Extends ATC with a separation discount for the last aircraft ℓ on the
    candidate runway:
        I_j(t,ℓ) = h_j/s̄_j · exp(−slack/(K₁·s̄)) · exp(−s[ℓ,j]/(K₂·s̄))

    Complexity: O(n²·m).
    Reference: Lee, Bhaskaran & Pinedo (1997); Pinedo (2016), §14.2.
    """
    s_sym  = (inst.s + inst.s.T) / 2.0; np.fill_diagonal(s_sym, 0.0)
    s_mean = s_sym.sum(axis=1) / max(inst.n - 1, 1)
    s_bar  = float(np.mean(s_mean[s_mean > 0])) if np.any(s_mean > 0) else 1.0
    K1s, K2s = max(K1 * s_bar, 1e-9), max(K2 * s_bar, 1e-9)

    seqs        = [[] for _ in range(m)]
    C_hats      = [[] for _ in range(m)]
    committed   = [0.0] * m
    last        = [-1]  * m
    unscheduled = list(range(inst.n))

    while unscheduled:
        rho = int(np.argmin(committed)); t = committed[rho]; ell = last[rho]
        best_j, best_idx = -1, -math.inf
        for j in unscheduled:
            slack   = max(float(inst.delta[j]) - s_mean[j] - t, 0.0)
            sep_val = float(inst.s[ell, j]) if ell >= 0 else 0.0
            idx     = ((float(inst.h[j]) / max(s_mean[j], 1e-9))
                       * math.exp(-slack / K1s)
                       * math.exp(-sep_val / K2s))
            if idx > best_idx:
                best_idx, best_j = idx, j
        seqs[rho].append(best_j); last[rho] = best_j
        C_hats[rho]    = surrogate_times(seqs[rho], inst)
        committed[rho] = C_hats[rho][-1] if C_hats[rho] else 0.0
        unscheduled.remove(best_j)

    return seqs, C_hats


def seed_caf(
    inst: Instance, m: int, params: HeuristicParams
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    H6 — Critical-Aircraft-First Insertion (CAF).

    Schedules aircraft in strict decreasing CR_j order, inserting each at
    the globally cheapest feasible (runway, position) using the TC-RBI
    composite cost.

    Complexity: O(n²·m).
    """
    _, CR     = compute_priorities(inst)
    order     = list(np.argsort(CR)[::-1])
    seqs      = [[] for _ in range(m)]
    C_hats    = [[] for _ in range(m)]
    committed = [0.0] * m; B_bar = 0.0

    for j in order:
        best_cost, best_rho, best_p, best_Cn = math.inf, 0, 0, []
        for rho in range(m):
            for p in _candidate_positions(j, rho, seqs, inst):
                cost, Cn = evaluate_insertion(j, rho, p, seqs, C_hats,
                                               B_bar, inst, params)
                if cost < best_cost:
                    best_cost, best_rho, best_p, best_Cn = cost, rho, p, Cn
        if math.isinf(best_cost):
            best_rho, best_p, best_Cn = min_violation_insert(j, seqs, inst)
        seqs[best_rho].insert(best_p, j); C_hats[best_rho] = best_Cn
        committed[best_rho] = C_hats[best_rho][-1] if C_hats[best_rho] else 0.0
        B_bar = sum(committed) / m

    return seqs, C_hats


def seed_mpds(
    inst: Instance, m: int, params: HeuristicParams
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    H7 — Most-Penalised-Displacement-First (MPDS).

    At each step scores each unscheduled aircraft j by (g_j+h_j)/min_cost_j.
    Skipped when n > MPDS_MAX_N (default 150) due to O(n³/m) complexity.

    Reference: Beasley et al. (2000) greedy construction variants.
    """
    seqs        = [[] for _ in range(m)]
    C_hats      = [[] for _ in range(m)]
    committed   = [0.0] * m; B_bar = 0.0
    unscheduled = list(range(inst.n))

    while unscheduled:
        best_score                       = -math.inf
        sel_idx, sel_rho, sel_p, sel_Cn  = 0, 0, 0, []

        for ji, j in enumerate(unscheduled):
            min_cost = math.inf; b_rho, b_p, b_Cn = 0, 0, []
            for rho in range(m):
                for p in _candidate_positions(j, rho, seqs, inst):
                    cost, Cn = evaluate_insertion(j, rho, p, seqs, C_hats,
                                                   B_bar, inst, params)
                    if cost < min_cost:
                        min_cost, b_rho, b_p, b_Cn = cost, rho, p, Cn
            score = ((float(inst.g[j]) + float(inst.h[j])) / max(min_cost, 1e-9)
                     if min_cost < math.inf else -math.inf)
            if score > best_score:
                best_score, sel_idx, sel_rho, sel_p, sel_Cn = (
                    score, ji, b_rho, b_p, b_Cn)

        j = unscheduled.pop(sel_idx)
        if best_score == -math.inf:
            sel_rho, sel_p, sel_Cn = min_violation_insert(j, seqs, inst)
        seqs[sel_rho].insert(sel_p, j); C_hats[sel_rho] = sel_Cn
        committed[sel_rho] = C_hats[sel_rho][-1] if C_hats[sel_rho] else 0.0
        B_bar = sum(committed) / m

    return seqs, C_hats


def seed_wcc(
    inst: Instance, m: int, params: HeuristicParams
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    H8 — Wake-Compatibility Chain Construction (WCC).

    Groups aircraft with low mutual separation into chains using a savings
    metric W_ij; chains are built greedily from the most critical aircraft
    (highest CR_j) and assigned to runways in decreasing length order.

    Complexity: O(n²) chain extraction + O(n²·m) assignment.
    Reference: Clarke & Wright (1964) savings concept.
    """
    n = inst.n; _, CR = compute_priorities(inst)
    s_bar = inst.s_bar; T_span = max(inst.T_span, inst.eps)
    W = np.zeros((n, n))
    for i in range(n):
        savings   = (s_bar - inst.s[i]) / max(s_bar, inst.eps)
        tgt_gap   = np.abs(inst.delta - inst.delta[i]) / T_span
        idle_wait = np.maximum(inst.r - inst.delta[i], 0.0) / T_span
        W[i]      = savings - tgt_gap - idle_wait
    np.fill_diagonal(W, -np.inf)

    unassigned = set(range(n)); chains = []
    while unassigned:
        j0    = max(unassigned, key=lambda j: CR[j])
        chain = [j0]; unassigned.remove(j0)
        while True:
            tail   = chain[-1]
            best_k = max(
                (k for k in unassigned
                 if inst.delta[k] >= inst.delta[tail] and W[tail, k] > 0),
                key=lambda k: W[tail, k], default=None)
            if best_k is None:
                break
            chain.append(best_k); unassigned.remove(best_k)
        chains.append(chain)
    chains.sort(key=len, reverse=True)

    seqs      = [[] for _ in range(m)]
    C_hats    = [[] for _ in range(m)]
    committed = [0.0] * m; B_bar = 0.0

    for chain in chains:
        for j in chain:
            best_cost, best_rho, best_p, best_Cn = math.inf, 0, len(seqs[0]), []
            for rho in range(m):
                p    = len(seqs[rho])
                cost, Cn = evaluate_insertion(j, rho, p, seqs, C_hats,
                                               B_bar, inst, params)
                if cost < best_cost:
                    best_cost, best_rho, best_p, best_Cn = cost, rho, p, Cn
            if math.isinf(best_cost):
                best_rho, best_p, best_Cn = min_violation_insert(j, seqs, inst)
            seqs[best_rho].insert(best_p, j); C_hats[best_rho] = best_Cn
            committed[best_rho] = C_hats[best_rho][-1] if C_hats[best_rho] else 0.0
            B_bar = sum(committed) / m

    return seqs, C_hats


def seed_grasp(
    inst: Instance, m: int, params: HeuristicParams,
    k: int = 3, rng_seed: int = 0,
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    H9 — GRASP (Greedy Randomised Adaptive Search Procedure).

    At each step: score all unscheduled aircraft by CR_j × urgency_j; build
    a Restricted Candidate List (RCL) of the top k; pick one at random;
    insert at the cheapest feasible (runway, position).

    k=1 → deterministic greedy (equivalent to CAF without regret).
    Complexity: O(n²·m).
    Reference: Feo & Resende (1995), Journal of Global Optimization 6(2): 109–133.
    """
    rng         = random.Random(rng_seed)
    _, CR       = compute_priorities(inst)
    seqs        = [[] for _ in range(m)]
    C_hats      = [[] for _ in range(m)]
    committed   = [0.0] * m; B_bar = 0.0
    unscheduled = list(range(inst.n))

    while unscheduled:
        tau    = min(committed)
        scores = sorted(
            ((float(CR[j]) / max(float(inst.delta[j]) - tau, inst.eps), j)
             for j in unscheduled),
            reverse=True)
        j = rng.choice([jj for _, jj in scores[:min(k, len(scores))]])

        best_cost, best_rho, best_p, best_Cn = math.inf, 0, 0, []
        for rho in range(m):
            for p in _candidate_positions(j, rho, seqs, inst):
                cost, Cn = evaluate_insertion(j, rho, p, seqs, C_hats,
                                               B_bar, inst, params)
                if cost < best_cost:
                    best_cost, best_rho, best_p, best_Cn = cost, rho, p, Cn
        if math.isinf(best_cost):
            best_rho, best_p, best_Cn = min_violation_insert(j, seqs, inst)

        seqs[best_rho].insert(best_p, j); C_hats[best_rho] = best_Cn
        committed[best_rho] = C_hats[best_rho][-1] if C_hats[best_rho] else 0.0
        B_bar = sum(committed) / m
        unscheduled.remove(j)

    return seqs, C_hats


# ═══════════════════════════════════════════════════════════════════════════
#   §25b  LP-SCREENED SEED PORTFOLIO BUILDER  (with per-seed timing)
# ═══════════════════════════════════════════════════════════════════════════

def _build_seed_portfolio(
    inst: Instance,
    m: int,
    params: HeuristicParams,
    n_chains: int,
    seed: int,
) -> Tuple[
    List[Tuple[str, List[List[int]]]],            # selected_starts
    List[Tuple[str, float, float, bool]],          # portfolio_info: (label,raw,lp,selected)
    List[float],                                    # selected_lps
    List[float],                                    # selected_raw_objs
    Dict[str, Any],                                 # portfolio_timing
]:
    """
    Generate, LP-evaluate, and select construction heuristic seeds.

    Heuristics evaluated
    --------------------
    FCFS, EDD, WEDD, ATC, ATCS, TC-RBI, CAF, WCC, GRASP-k1, GRASP-k2,
    and MPDS (only when n ≤ MPDS_MAX_N).

    Per-seed timing
    ---------------
    Each heuristic is individually timed for construction and LP evaluation.
    The portfolio_timing dict returned contains:

      t_seed_construct  : total wall time for all seed construction calls.
      t_seed_lp_eval    : total wall time for all LP evaluation calls.
      t_portfolio       : total wall time for the full portfolio phase.
      t_best_seed_lp    : job-relative time when the best seed LP was first
                          achieved (used by solver.py to compute total_t_best).
      seed_lp_events    : list of (t_job_relative, lp_val) for each new best.
      seed_timing       : per-seed list of dicts with keys label, t_construct,
                          t_lp_eval, t_job_relative, lp_val, raw_obj.

    Selection
    ---------
    When USE_ALL_SEEDS=True  : every feasibly evaluated seed is forwarded to SA.
    When USE_ALL_SEEDS=False : diversity-aware top-N_CHAINS selection by LP value.
    """
    k1, k2         = GRASP_K_VALUES
    t_port_start   = time.perf_counter()
    seed_timing_list: List[Dict] = []
    seed_lp_events:   List[Tuple[float, float]] = []
    evaluated:        List[Tuple[float, float, str, List[List[int]]]] = []
    current_best_lp   = math.inf
    t_construct_total = 0.0
    t_lp_total        = 0.0

    def _timed_seed(label: str, build_fn) -> None:
        nonlocal t_construct_total, t_lp_total, current_best_lp

        t_c0   = time.perf_counter()
        seqs   = build_fn()
        t_c    = time.perf_counter() - t_c0
        t_construct_total += t_c

        feas_e, _, raw_obj, _ = verify_and_exact_obj(seqs, inst)
        raw_obj = raw_obj if feas_e else math.inf

        t_lp0  = time.perf_counter()
        lp, _, lp_feas, _ = stage2_lp_objective(seqs, inst)
        t_lp   = time.perf_counter() - t_lp0
        t_lp_total += t_lp

        lp_val     = lp if lp_feas else math.inf
        t_job_rel  = time.perf_counter() - t_port_start   # job-relative

        if not math.isinf(lp_val) and lp_val < current_best_lp:
            current_best_lp = lp_val
            seed_lp_events.append((t_job_rel, lp_val))

        seed_timing_list.append(dict(
            label=label, t_construct=t_c, t_lp_eval=t_lp,
            t_job_relative=t_job_rel, lp_val=lp_val, raw_obj=raw_obj,
        ))
        evaluated.append((lp_val, raw_obj, label, seqs))

    # ── Evaluate all heuristics in sequence ──────────────────────────────
    _timed_seed("FCFS",        lambda: seed_fcfs(inst, m)[0])
    _timed_seed("EDD",         lambda: seed_edd_balanced(inst, m)[0])
    _timed_seed("WEDD",        lambda: seed_wedd(inst, m)[0])
    _timed_seed("ATC",         lambda: seed_atc(inst, m, K=ATC_K)[0])
    _timed_seed("ATCS",        lambda: seed_atcs(inst, m, K1=ATCS_K1, K2=ATCS_K2)[0])
    _timed_seed("TC-RBI",      lambda: ramp_rbi(inst, m, params)[0])
    _timed_seed("CAF",         lambda: seed_caf(inst, m, params)[0])
    _timed_seed("WCC",         lambda: seed_wcc(inst, m, params)[0])
    _timed_seed(f"GRASP-{k1}", lambda: seed_grasp(inst, m, params, k=k1, rng_seed=seed)[0])
    _timed_seed(f"GRASP-{k2}", lambda: seed_grasp(inst, m, params, k=k2, rng_seed=seed+999)[0])
    if inst.n <= MPDS_MAX_N:
        _timed_seed("MPDS",    lambda: seed_mpds(inst, m, params)[0])

    t_port_total = time.perf_counter() - t_port_start

    # ── Sort by LP value ──────────────────────────────────────────────────
    evaluated.sort(key=lambda x: x[0])

    # ── Diversity-aware selection ─────────────────────────────────────────
    def _rwy_hamming(sa, sb):
        ma = len(sa)
        aa = {sa[r][p]: r for r in range(ma) for p in range(len(sa[r]))}
        return sum(1 for r in range(len(sb)) for j in sb[r] if aa.get(j) != r)

    if USE_ALL_SEEDS:
        selected = [(label, seqs, lp_val, raw_obj)
                    for lp_val, raw_obj, label, seqs in evaluated]
    else:
        selected:  List[Tuple[str, List, float, float]] = []
        used_lbl: set = set()
        # First pass: prefer diversity
        for lp_val, raw_obj, label, seqs in evaluated:
            if len(selected) >= n_chains: break
            if label in used_lbl: continue
            diverse = (not selected
                       or all(_rwy_hamming(seqs, s) >= ELITE_MIN_DIV
                              for _, s, _, _ in selected))
            if diverse:
                selected.append((label, seqs, lp_val, raw_obj))
                used_lbl.add(label)
        # Second pass: fill by LP regardless of diversity
        for lp_val, raw_obj, label, seqs in evaluated:
            if len(selected) >= n_chains: break
            if label in used_lbl: continue
            selected.append((label, seqs, lp_val, raw_obj))
            used_lbl.add(label)

    sel_labels        = {lbl for lbl, _, _, _ in selected}
    selected_starts   = [(lbl, s)       for lbl, s, _,  _  in selected]
    selected_lps      = [lp             for _,   _, lp, _  in selected]
    selected_raw_objs = [ro             for _,   _, _,  ro in selected]
    portfolio_info    = [
        (label, raw_obj, lp_val, label in sel_labels)
        for lp_val, raw_obj, label, _ in evaluated
    ]

    portfolio_timing = dict(
        t_seed_construct=t_construct_total,
        t_seed_lp_eval=t_lp_total,
        t_portfolio=t_port_total,
        t_best_seed_lp=seed_lp_events[-1][0] if seed_lp_events else 0.0,
        seed_lp_events=seed_lp_events,
        seed_timing=seed_timing_list,
    )

    return selected_starts, portfolio_info, selected_lps, selected_raw_objs, portfolio_timing