"""
construction.py — Phase 1: Runway-assignment construction portfolio.

Provides two tiers of heuristics:

Tier A — Original six-rule assignment portfolio (§10)
──────────────────────────────────────────────────────
  RBI    : Regret-Based Insertion        (§10.1)
  MPDS   : MPDS-Balanced                 (§10.2)
  EDD    : EDD-Balanced                  (§10.3)
  TARGET : Target-Time Balanced          (§10.4)
  CRIT   : Critical-Aircraft-First       (§10.5)
  WAKE   : Wake-Compatibility Chain      (§10.6)

Tier B — TC-RBI / RAMP-RBI and LP-screened seed portfolio (§10, §25, §25b)
─────────────────────────────────────────────────────────────────────────────
  ramp_rbi        : TC-RBI (Target-Conflict Regret-Based Insertion)
  seed_fcfs       : H1 — First-Come First-Served
  seed_edd_balanced : H2 — EDD round-robin balanced
  seed_wedd       : H3 — Weighted EDD
  seed_atc        : H4 — Apparent Tardiness Cost
  seed_atcs       : H5 — ATC with Setups
  seed_caf        : H6 — Critical-Aircraft-First Insertion
  seed_mpds       : H7 — Most-Penalised-Displacement-First
  seed_wcc        : H8 — Wake-Compatibility Chain Construction
  seed_grasp      : H9 — GRASP

Entry points
────────────
build_assignment_portfolio : Tier A portfolio — returns raw assignments.
build_seed_portfolio       : Tier B portfolio — LP-screened, with timing.
"""
from __future__ import annotations

import math
import random
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from misc.mr_alp_2.config import (
    EPS,
    ATC_K, ATCS_K1, ATCS_K2,
    ELITE_MIN_DIV, GRASP_K_VALUES,
    MPDS_MAX_N, N_CHAINS, USE_ALL_SEEDS,
)
from misc.mr_alp_2.models import Instance, HybridParams, HeuristicParams
from misc.mr_alp_2.surrogate import (
    surrogate_times, surrogate_penalty, committed_time, best_insertion,
)
from misc.mr_alp_2.priority import compute_priority_scores
from misc.mr_alp_2.lp import stage2_lp_objective


# ═══════════════════════════════════════════════════════════════════════════
#   §8 / §9  TC-RBI PRIORITY AND INSERTION-COST COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════

def compute_priorities(inst: Instance) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute AF_j and CR_j for all aircraft (§9.1, §9.2).

    AF_j = (d_j − r_j) − mean_bilateral_separation_j
    CR_j = (g_j + h_j) / max(AF_j, ε)

    Returns (AF, CR) as (n,) arrays.  Identical to inst.AF / inst.CR
    computed by Instance.__post_init__; exposed here for direct use in
    heuristic implementations that do not hold an Instance reference.
    """
    s_sym = (inst.s + inst.s.T) / 2.0
    np.fill_diagonal(s_sym, 0.0)
    s_b = s_sym.sum(axis=1) / max(inst.n - 1, 1)
    AF  = (inst.d - inst.r) - s_b
    CR  = (inst.g + inst.h) / np.maximum(AF, EPS)
    return AF, CR


def _minmax_norm(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise arr to [0, 1].  Constant arrays map to 0."""
    lo, hi = float(arr.min()), float(arr.max())
    return (arr - lo) / (hi - lo + EPS)


def _compute_insert_times(
    j:         int,
    p:         int,
    seq:       List[int],
    C_hat_seq: List[float],
    inst:      Instance,
) -> Tuple[List[float], bool]:
    """
    Surrogate landing times after inserting aircraft j at position p.

    Reuses the unchanged prefix C_hat_seq[:p] and propagates the
    consecutive-predecessor recursion from position p onward.

    Returns (C_new, feasible) where feasible is False when any C_new[q]
    exceeds the hard deadline d[seq_n[q]].
    """
    seq_n = seq[:p] + [j] + seq[p:]
    L_n   = len(seq_n)
    C_n   = list(C_hat_seq[:p])

    if p == 0:
        C_n.append(float(inst.r[j]))
    else:
        C_n.append(max(float(inst.r[j]),
                       C_n[p - 1] + float(inst.s[seq_n[p - 1], j])))

    for q in range(p + 1, L_n):
        prev = seq_n[q - 1]
        cur  = seq_n[q]
        C_n.append(max(float(inst.r[cur]),
                       C_n[q - 1] + float(inst.s[prev, cur])))

    for q in range(p, L_n):
        if C_n[q] > float(inst.d[seq_n[q]]) + 1e-9:
            return C_n, False

    return C_n, True


def _is_feasible_anywhere(
    j:         int,
    sequences: List[List[int]],
    C_hats:    List[List[float]],
    inst:      Instance,
) -> bool:
    """Return True iff aircraft j has at least one feasible insertion position."""
    m = len(sequences)
    # Fast path: try appending to each runway tail first
    for rho in range(m):
        _, ok = _compute_insert_times(
            j, len(sequences[rho]), sequences[rho], C_hats[rho], inst)
        if ok:
            return True
    # Full scan over interior positions
    for rho in range(m):
        for p in range(len(sequences[rho])):
            _, ok = _compute_insert_times(
                j, p, sequences[rho], C_hats[rho], inst)
            if ok:
                return True
    return False


def _target_conflict_insert(
    j: int, p: int, seq: List[int], inst: Instance
) -> float:
    """
    Incremental weighted target-time conflict from inserting j at position p.

    ΔTC = Σ_{i∈pred} 0.5·(p_i+p_j)·max(s[i,j]−(δ_j−δ_i), 0)
        + Σ_{k∈succ} 0.5·(p_j+p_k)·max(s[j,k]−(δ_k−δ_j), 0)
    """
    pj   = float(inst.p_arr[j])
    cost = 0.0
    pred = seq[:p]
    if pred:
        pa    = np.asarray(pred, dtype=np.intp)
        delta = inst.s[pa, j] - (inst.delta[j] - inst.delta[pa])
        cost += float((0.5 * (inst.p_arr[pa] + pj)
                       * np.maximum(delta, 0.0)).sum())
    succ = seq[p:]
    if succ:
        sa    = np.asarray(succ, dtype=np.intp)
        delta = inst.s[j, sa] - (inst.delta[sa] - inst.delta[j])
        cost += float((0.5 * (pj + inst.p_arr[sa])
                       * np.maximum(delta, 0.0)).sum())
    return cost


def _lower_bound_tardiness(
    seq: List[int], C_hat: List[float], inst: Instance
) -> float:
    """Lower-bound tardiness Σ h_j · max(Ĉ_j − δ_j, 0) for one runway."""
    if not seq:
        return 0.0
    sa = np.asarray(seq, dtype=np.intp)
    Ca = np.asarray(C_hat)
    return float((inst.h[sa] * np.maximum(Ca - inst.delta[sa], 0.0)).sum())


def _count_balance_delta(
    rho: int, sequences: List[List[int]], inst: Instance
) -> float:
    """
    Marginal squared-imbalance cost of adding one aircraft to runway rho.

    Δcount = [(ol+1 − (t+1)/m)² − (ol − t/m)²] · Pen_bar / (n/m)²
    """
    m  = len(sequences)
    t  = sum(len(s) for s in sequences)
    ol = len(sequences[rho])
    return (((ol + 1 - (t + 1) / m) ** 2 - (ol - t / m) ** 2)
            * float(inst.Pen_bar) / max((inst.n / m) ** 2, 1.0))


def _evaluate_insertion(
    j:         int,
    rho:       int,
    p:         int,
    sequences: List[List[int]],
    C_hats:    List[List[float]],
    B_bar:     float,
    inst:      Instance,
    params:    HeuristicParams,
) -> Tuple[float, List[float]]:
    """
    Composite TC-RBI insertion cost (§8).

    cost = μ_TC·ΔTC + μ_late·ΔLate + μ_count·Δcount + μ_sep·h̄·ΔSep_raw

    Returns (cost, C_new_for_rho).  Returns (inf, []) when the insertion
    violates any hard deadline d_j on the modified runway.
    """
    seq       = sequences[rho]
    C_hat_seq = C_hats[rho]
    L         = len(seq)

    C_n, ok = _compute_insert_times(j, p, seq, C_hat_seq, inst)
    if not ok:
        return math.inf, []

    seq_n  = seq[:p] + [j] + seq[p:]
    dTC    = _target_conflict_insert(j, p, seq, inst)
    dLate  = (_lower_bound_tardiness(seq_n, C_n, inst)
               - _lower_bound_tardiness(seq, C_hat_seq, inst))
    dCount = _count_balance_delta(rho, sequences, inst)

    if L == 0:
        dSep_raw = 0.0
    elif p == 0:
        dSep_raw = float(inst.s[j, seq[0]])
    elif p == L:
        dSep_raw = float(inst.s[seq[-1], j])
    else:
        a, b     = seq[p - 1], seq[p]
        dSep_raw = max(0.0,
                       float(inst.s[a, j])
                       + float(inst.s[j, b])
                       - float(inst.s[a, b]))

    cost = (params.mu_tc    * dTC
            + params.mu_late  * dLate
            + params.mu_count * dCount
            + params.mu_sep   * inst.h_bar * dSep_raw)
    return cost, C_n


def _candidate_positions(
    j:         int,
    rho:       int,
    sequences: List[List[int]],
    inst:      Instance,
) -> List[int]:
    """
    Return candidate insertion positions for j on runway rho (§10.2).

    For n ≤ 100: all L+1 positions.
    For n > 100: a ±2 window around the δ_j-ordered position, always
    including 0 and L so head and tail are never excluded.
    """
    seq = sequences[rho]
    L   = len(seq)
    if inst.n <= 100:
        return list(range(L + 1))
    p0 = next((p for p, u in enumerate(seq)
               if float(inst.delta[u]) >= float(inst.delta[j])), L)
    return sorted(set(range(max(0, p0 - 2), min(L + 1, p0 + 3))) | {0, L})


def _best_insertions(
    j:         int,
    m:         int,
    sequences: List[List[int]],
    C_hats:    List[List[float]],
    B_bar:     float,
    inst:      Instance,
    params:    HeuristicParams,
) -> Tuple[Tuple[float, int, int, List[float]], float]:
    """
    Best (runway, position) for aircraft j and the regret value (§9.4).

    Returns (best1=(c1, rho1, p1, C1), c2) where c2 is the cheapest
    insertion cost on the second-best runway.  For m=1, c2 is the cost
    of the second-cheapest position on the sole runway.
    """
    per_rho: List[Tuple[float, int, List[float]]] = []
    for rho in range(m):
        rc, rp, rC = math.inf, 0, []
        for p in _candidate_positions(j, rho, sequences, inst):
            c, Cn = _evaluate_insertion(
                j, rho, p, sequences, C_hats, B_bar, inst, params)
            if c < rc:
                rc, rp, rC = c, p, Cn
        per_rho.append((rc, rp, rC))

    sr    = sorted(range(m), key=lambda r: per_rho[r][0])
    c1, p1, C1 = per_rho[sr[0]]
    best1 = (c1, sr[0], p1, C1)

    if m > 1:
        c2 = per_rho[sr[1]][0]
    else:
        all_c = sorted(
            _evaluate_insertion(
                j, 0, p, sequences, C_hats, B_bar, inst, params)[0]
            for p in range(len(sequences[0]) + 1)
        )
        c2 = all_c[1] if len(all_c) > 1 else math.inf

    return best1, c2


def _min_violation_insert(
    j:         int,
    sequences: List[List[int]],
    inst:      Instance,
) -> Tuple[int, int, List[float]]:
    """
    Least-infeasible insertion for aircraft j (forced-set fallback, §9.3).

    Minimises total deadline violation Σ max(Ĉ'_u − d_u, 0) across
    all aircraft on the modified runway.
    """
    best_V   = math.inf
    best_rho = 0
    best_p   = 0
    best_C:  List[float] = []

    for rho, seq in enumerate(sequences):
        for p in range(len(seq) + 1):
            seq_n = seq[:p] + [j] + seq[p:]
            C_t   = surrogate_times(seq_n, inst)
            V     = sum(max(C_t[q] - float(inst.d[seq_n[q]]), 0.0)
                        for q in range(len(seq_n)))
            if V < best_V:
                best_V, best_rho, best_p, best_C = V, rho, p, C_t

    return best_rho, best_p, best_C


def _inter_runway_repair(
    sequences:      List[List[int]],
    C_hats:         List[List[float]],
    inst:           Instance,
    params:         HeuristicParams,
    max_iterations: int = 150,
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    Post-construction balance repair: relocate high-TC aircraft from the
    most overloaded runway to underloaded runways.

    Runs after ramp_rbi to correct any load skew created during insertion.
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

        B_bar = sum(
            C_hats[r][-1] if C_hats[r] else 0.0 for r in range(m)
        ) / m

        # Identify the aircraft on rho_src with the highest TC score
        best_tc = -1.0
        best_sp = -1
        for sp, j in enumerate(sequences[rho_src]):
            seq_no = (sequences[rho_src][:sp]
                      + sequences[rho_src][sp + 1:])
            tc = _target_conflict_insert(j, sp, seq_no, inst)
            if tc > best_tc:
                best_tc, best_sp = tc, sp
        if best_sp == -1:
            break

        j_move   = sequences[rho_src][best_sp]
        best_c   = math.inf
        best_rd  = -1
        best_dp  = 0
        best_Cn: List[float] = []

        for rd in range(m):
            if rd == rho_src:
                continue
            for dp in range(len(sequences[rd]) + 1):
                c, Cn = _evaluate_insertion(
                    j_move, rd, dp, sequences, C_hats, B_bar, inst, params)
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
#   TC-RBI / RAMP-RBI (§10, Algorithm 1)
# ═══════════════════════════════════════════════════════════════════════════

def ramp_rbi(
    inst:   Instance,
    m:      int,
    params: HeuristicParams,
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    TC-RBI / RAMP-RBI: Target-Conflict Regret-Based Insertion (§10).

    Iteratively inserts aircraft by minimising the composite TC-RBI cost
    (μ_TC·ΔTC + μ_late·ΔLate + μ_count·Δcount + μ_sep·ΔSep), breaking
    ties via regret (second-best minus best runway cost) and criticality CR_j.
    Aircraft with no feasible position enter the forced-insertion queue and
    are placed by minimum-deadline-violation (§9.3).

    A post-construction inter_runway_repair pass corrects load skew.

    Returns (sequences, C_hats).
    """
    n   = inst.n
    CR  = inst.CR          # precomputed by Instance.__post_init__ (§9.2)

    sequences: List[List[int]]   = [[] for _ in range(m)]
    C_hats:    List[List[float]] = [[] for _ in range(m)]
    B          = [0.0] * m
    B_bar      = 0.0
    U:  List[int] = list(range(n))
    F:  Set[int]  = set()

    def _committed(rho: int) -> float:
        return C_hats[rho][-1] if C_hats[rho] else 0.0

    def _refresh_forced() -> None:
        """Scan remaining aircraft against the current state; add to F if infeasible."""
        for j in list(U):
            if j in F:
                continue
            if not _is_feasible_anywhere(j, sequences, C_hats, inst):
                F.add(j)

    def _do_insert(j: int, rho: int, p: int, C_new: List[float]) -> None:
        nonlocal B_bar
        sequences[rho].insert(p, j)
        C_hats[rho] = C_new
        B[rho]  = _committed(rho)
        B_bar   = sum(B) / m
        U.remove(j)
        F.discard(j)

    while U:
        # ── 1.  Process forced aircraft before normal selection (§9.3) ──
        while F & set(U):
            j_star        = max([j for j in U if j in F], key=lambda j: CR[j])
            rho, p, C_new = _min_violation_insert(j_star, sequences, inst)
            sequences[rho] = sequences[rho][:p] + [j_star] + sequences[rho][p:]
            C_hats[rho]    = C_new
            B[rho]  = _committed(rho)
            B_bar   = sum(B) / m
            U.remove(j_star)
            F.discard(j_star)
            _refresh_forced()

        if not U:
            break

        U_avail = [j for j in U if j not in F]
        if not U_avail:
            break

        # ── 2.  Candidate screening by urgency × criticality (§10.1) ───
        tau    = min(B)
        urg    = np.array([
            1.0 / max(float(inst.delta[j]) - tau, EPS)
            for j in U_avail
        ])
        cr_arr = np.array([CR[j] for j in U_avail])
        screen = (params.eta       * _minmax_norm(cr_arr)
                  + (1.0 - params.eta) * _minmax_norm(urg))

        q_eff = (len(U_avail) if n <= 100
                 else min(150, max(50, int(0.25 * len(U_avail)))))
        top   = np.argsort(screen)[::-1][:q_eff]
        U_q   = [U_avail[i] for i in top]

        # ── 3.  Compute regret for each screened candidate (§9.4) ──────
        info: Dict[int, Tuple[float, int, int, List[float], float]] = {}
        for j in U_q:
            best1, c2 = _best_insertions(
                j, m, sequences, C_hats, B_bar, inst, params)
            c1, rho1, p1, Cn1 = best1
            info[j] = (c1, rho1, p1, Cn1, c2)

        finite = [info[j][4] - info[j][0] for j in U_q
                  if info[j][4] < math.inf]
        R_max  = ((max(finite) + inst.h_bar * inst.T_span)
                  if finite else inst.h_bar * inst.T_span)

        best_c_arr = np.array([info[j][0] for j in U_q])
        regret_arr = np.array([
            (info[j][4] - info[j][0]) if info[j][4] < math.inf else R_max
            for j in U_q
        ])
        cr_uq = np.array([CR[j] for j in U_q])

        # Selection rule: argmin(norm_cost − 0.20·norm_regret − 0.10·norm_CR)
        score  = (_minmax_norm(best_c_arr)
                  - 0.20 * _minmax_norm(regret_arr)
                  - 0.10 * _minmax_norm(cr_uq))
        j_star = U_q[int(np.argmin(score))]

        c_s, rho_s, p_s, Cn_s, _ = info[j_star]
        if math.isinf(c_s):
            F.add(j_star)
            continue

        _do_insert(j_star, rho_s, p_s, Cn_s)
        _refresh_forced()

    sequences, C_hats = _inter_runway_repair(sequences, C_hats, inst, params)
    return sequences, C_hats


# ═══════════════════════════════════════════════════════════════════════════
#   Seed Heuristics H1 – H9 (§25)
# ═══════════════════════════════════════════════════════════════════════════

def seed_fcfs(
    inst: Instance, m: int
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    H1 — First-Come First-Served (§25.1).

    Assigns aircraft in non-decreasing r_j order to the runway with the
    smallest current committed time.

    Complexity: O(n log n + n·m).
    Reference: Beasley et al. (2000).
    """
    order:     List[int]   = list(np.argsort(inst.r))
    seqs:      List[List[int]]   = [[] for _ in range(m)]
    committed: List[float]       = [0.0] * m

    for j in order:
        rho  = int(np.argmin(committed))
        prev = seqs[rho][-1] if seqs[rho] else -1
        sep  = float(inst.s[prev, j]) if prev >= 0 else 0.0
        seqs[rho].append(j)
        committed[rho] = max(float(inst.r[j]), committed[rho] + sep)

    return seqs, [surrogate_times(seq, inst) for seq in seqs]


def seed_edd_balanced(
    inst: Instance, m: int
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    H2 — EDD-Balanced: sort by δ_j, assign round-robin across runways (§25.2).

    Complexity: O(n log n).
    Reference: Pinedo (2016), §3.2.
    """
    order = list(np.argsort(inst.delta))
    seqs: List[List[int]] = [[] for _ in range(m)]
    for k, j in enumerate(order):
        seqs[k % m].append(j)
    return seqs, [surrogate_times(seq, inst) for seq in seqs]


def seed_wedd(
    inst: Instance, m: int
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    H3 — Weighted EDD (§25.3).

    Sorts by δ_j / (g_j + h_j) and assigns to the runway with the smallest
    committed time.

    Complexity: O(n log n + n·m).
    Reference: Ernst, Krishnamoorthy & Storer (1999).
    """
    scores    = inst.delta / np.maximum(inst.g + inst.h, EPS)
    order     = list(np.argsort(scores))
    seqs:     List[List[int]]   = [[] for _ in range(m)]
    C_hats:   List[List[float]] = [[] for _ in range(m)]
    committed = [0.0] * m

    for j in order:
        rho = int(np.argmin(committed))
        seqs[rho].append(j)
        C_hats[rho]    = surrogate_times(seqs[rho], inst)
        committed[rho] = C_hats[rho][-1] if C_hats[rho] else 0.0

    return seqs, C_hats


def seed_atc(
    inst: Instance, m: int, K: float = ATC_K,
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    H4 — Apparent Tardiness Cost (§25.4).

    On the next-to-open runway at time t, scores each unscheduled aircraft:
        I_j(t) = (h_j / s̄_j) · exp(−max(δ_j − s̄_j − t, 0) / (K · s̄))

    Large K → WSPT; small K → minimum-slack.

    Complexity: O(n² · m).
    Reference: Vepsäläinen & Morton (1987).
    """
    s_sym  = (inst.s + inst.s.T) / 2.0
    np.fill_diagonal(s_sym, 0.0)
    s_mean = s_sym.sum(axis=1) / max(inst.n - 1, 1)
    s_bar  = (float(np.mean(s_mean[s_mean > 0]))
               if np.any(s_mean > 0) else 1.0)
    Ks     = max(K * s_bar, 1e-9)

    seqs:       List[List[int]]   = [[] for _ in range(m)]
    C_hats:     List[List[float]] = [[] for _ in range(m)]
    committed   = [0.0] * m
    unscheduled = list(range(inst.n))

    while unscheduled:
        rho      = int(np.argmin(committed))
        t        = committed[rho]
        best_j   = -1
        best_idx = -math.inf

        for j in unscheduled:
            slack = max(float(inst.delta[j]) - s_mean[j] - t, 0.0)
            idx   = (float(inst.h[j]) / max(s_mean[j], 1e-9)
                     * math.exp(-slack / Ks))
            if idx > best_idx:
                best_idx, best_j = idx, j

        seqs[rho].append(best_j)
        C_hats[rho]    = surrogate_times(seqs[rho], inst)
        committed[rho] = C_hats[rho][-1] if C_hats[rho] else 0.0
        unscheduled.remove(best_j)

    return seqs, C_hats


def seed_atcs(
    inst: Instance, m: int,
    K1: float = ATCS_K1, K2: float = ATCS_K2,
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    H5 — Apparent Tardiness Cost with Setups (§25.5).

    Extends ATC with a separation discount for the last aircraft ℓ on the
    candidate runway:
        I_j(t, ℓ) = (h_j/s̄_j)·exp(−slack/(K₁·s̄))·exp(−s[ℓ,j]/(K₂·s̄))

    Complexity: O(n² · m).
    Reference: Lee, Bhaskaran & Pinedo (1997); Pinedo (2016), §14.2.
    """
    s_sym  = (inst.s + inst.s.T) / 2.0
    np.fill_diagonal(s_sym, 0.0)
    s_mean = s_sym.sum(axis=1) / max(inst.n - 1, 1)
    s_bar  = (float(np.mean(s_mean[s_mean > 0]))
               if np.any(s_mean > 0) else 1.0)
    K1s, K2s = max(K1 * s_bar, 1e-9), max(K2 * s_bar, 1e-9)

    seqs:       List[List[int]]   = [[] for _ in range(m)]
    C_hats:     List[List[float]] = [[] for _ in range(m)]
    committed   = [0.0] * m
    last        = [-1]  * m
    unscheduled = list(range(inst.n))

    while unscheduled:
        rho      = int(np.argmin(committed))
        t        = committed[rho]
        ell      = last[rho]
        best_j   = -1
        best_idx = -math.inf

        for j in unscheduled:
            slack   = max(float(inst.delta[j]) - s_mean[j] - t, 0.0)
            sep_val = float(inst.s[ell, j]) if ell >= 0 else 0.0
            idx     = (float(inst.h[j]) / max(s_mean[j], 1e-9)
                       * math.exp(-slack / K1s)
                       * math.exp(-sep_val / K2s))
            if idx > best_idx:
                best_idx, best_j = idx, j

        seqs[rho].append(best_j)
        last[rho]      = best_j
        C_hats[rho]    = surrogate_times(seqs[rho], inst)
        committed[rho] = C_hats[rho][-1] if C_hats[rho] else 0.0
        unscheduled.remove(best_j)

    return seqs, C_hats


def seed_caf(
    inst: Instance, m: int, params: HeuristicParams,
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    H6 — Critical-Aircraft-First Insertion (§25.6).

    Inserts aircraft in strict decreasing CR_j order; each is placed at the
    globally cheapest feasible (runway, position) under the TC-RBI composite
    cost.  Infeasible insertions fall back to _min_violation_insert.

    Complexity: O(n² · m).
    """
    CR        = inst.CR
    order     = list(np.argsort(CR)[::-1])
    seqs:      List[List[int]]   = [[] for _ in range(m)]
    C_hats:    List[List[float]] = [[] for _ in range(m)]
    committed  = [0.0] * m
    B_bar      = 0.0

    for j in order:
        best_cost = math.inf
        best_rho  = 0
        best_p    = 0
        best_Cn:  List[float] = []

        for rho in range(m):
            for p in _candidate_positions(j, rho, seqs, inst):
                cost, Cn = _evaluate_insertion(
                    j, rho, p, seqs, C_hats, B_bar, inst, params)
                if cost < best_cost:
                    best_cost, best_rho, best_p, best_Cn = cost, rho, p, Cn

        if math.isinf(best_cost):
            best_rho, best_p, best_Cn = _min_violation_insert(j, seqs, inst)

        seqs[best_rho].insert(best_p, j)
        C_hats[best_rho] = best_Cn
        committed[best_rho] = (C_hats[best_rho][-1]
                                if C_hats[best_rho] else 0.0)
        B_bar = sum(committed) / m

    return seqs, C_hats


def seed_mpds(
    inst: Instance, m: int, params: HeuristicParams,
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    H7 — Most-Penalised-Displacement-First (§25.7).

    Scores each unscheduled aircraft by (g_j + h_j) / min_insertion_cost_j
    and inserts the highest scorer.  Skipped by build_seed_portfolio when
    n > MPDS_MAX_N due to O(n³/m) complexity.

    Reference: Beasley et al. (2000).
    """
    seqs:       List[List[int]]   = [[] for _ in range(m)]
    C_hats:     List[List[float]] = [[] for _ in range(m)]
    committed   = [0.0] * m
    B_bar       = 0.0
    unscheduled = list(range(inst.n))

    while unscheduled:
        best_score                           = -math.inf
        sel_idx  = 0
        sel_rho  = 0
        sel_p    = 0
        sel_Cn:  List[float] = []

        for ji, j in enumerate(unscheduled):
            min_cost = math.inf
            b_rho, b_p, b_Cn = 0, 0, []
            for rho in range(m):
                for p in _candidate_positions(j, rho, seqs, inst):
                    cost, Cn = _evaluate_insertion(
                        j, rho, p, seqs, C_hats, B_bar, inst, params)
                    if cost < min_cost:
                        min_cost, b_rho, b_p, b_Cn = cost, rho, p, Cn
            score = ((float(inst.g[j]) + float(inst.h[j]))
                     / max(min_cost, 1e-9)
                     if min_cost < math.inf else -math.inf)
            if score > best_score:
                best_score = score
                sel_idx, sel_rho, sel_p, sel_Cn = ji, b_rho, b_p, b_Cn

        j = unscheduled.pop(sel_idx)
        if best_score == -math.inf:
            sel_rho, sel_p, sel_Cn = _min_violation_insert(j, seqs, inst)

        seqs[sel_rho].insert(sel_p, j)
        C_hats[sel_rho] = sel_Cn
        committed[sel_rho] = (C_hats[sel_rho][-1]
                               if C_hats[sel_rho] else 0.0)
        B_bar = sum(committed) / m

    return seqs, C_hats


def seed_wcc(
    inst: Instance, m: int, params: HeuristicParams,
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    H8 — Wake-Compatibility Chain Construction (§25.8).

    Builds chains of aircraft with mutually low separation using a savings
    metric W_ij; chains start from the most critical aircraft (highest CR_j)
    and extend greedily.  Chains are assigned to runways in decreasing length
    order, inserting each aircraft at the cheapest tail position.

    Complexity: O(n²) chain extraction + O(n² · m) assignment.
    Reference: Clarke & Wright (1964) savings concept.
    """
    n      = inst.n
    CR     = inst.CR
    s_bar  = inst.s_bar
    T_span = max(inst.T_span, EPS)

    # Savings matrix (§11.3 of spec)
    W = np.zeros((n, n))
    for i in range(n):
        savings   = (s_bar - inst.s[i]) / max(s_bar, EPS)
        tgt_gap   = np.abs(inst.delta - inst.delta[i]) / T_span
        idle_wait = np.maximum(inst.r - inst.delta[i], 0.0) / T_span
        W[i]      = savings - tgt_gap - idle_wait
    np.fill_diagonal(W, -np.inf)

    unassigned: Set[int]        = set(range(n))
    chains:     List[List[int]] = []

    while unassigned:
        j0    = max(unassigned, key=lambda j: CR[j])
        chain = [j0]
        unassigned.remove(j0)
        while True:
            tail   = chain[-1]
            best_k = max(
                (k for k in unassigned
                 if float(inst.delta[k]) >= float(inst.delta[tail])
                 and W[tail, k] > 0.0),
                key=lambda k: W[tail, k],
                default=None,
            )
            if best_k is None:
                break
            chain.append(best_k)
            unassigned.remove(best_k)
        chains.append(chain)

    chains.sort(key=len, reverse=True)

    seqs:     List[List[int]]   = [[] for _ in range(m)]
    C_hats:   List[List[float]] = [[] for _ in range(m)]
    committed = [0.0] * m
    B_bar     = 0.0

    for chain in chains:
        for j in chain:
            best_cost = math.inf
            best_rho  = 0
            best_p    = 0
            best_Cn:  List[float] = []

            for rho in range(m):
                p = len(seqs[rho])          # append only — preserves chain order
                cost, Cn = _evaluate_insertion(
                    j, rho, p, seqs, C_hats, B_bar, inst, params)
                if cost < best_cost:
                    best_cost, best_rho, best_p, best_Cn = cost, rho, p, Cn

            if math.isinf(best_cost):
                best_rho, best_p, best_Cn = _min_violation_insert(
                    j, seqs, inst)

            seqs[best_rho].insert(best_p, j)
            C_hats[best_rho] = best_Cn
            committed[best_rho] = (C_hats[best_rho][-1]
                                   if C_hats[best_rho] else 0.0)
            B_bar = sum(committed) / m

    return seqs, C_hats


def seed_grasp(
    inst:     Instance,
    m:        int,
    params:   HeuristicParams,
    k:        int = 3,
    rng_seed: int = 0,
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    H9 — GRASP (Greedy Randomised Adaptive Search Procedure) (§25.9).

    At each step scores all unscheduled aircraft by CR_j / max(δ_j − τ, ε),
    builds a Restricted Candidate List (RCL) of the top k, picks one
    uniformly at random, and inserts it at the cheapest feasible position.

    k=1 → deterministic (equivalent to CAF without regret).

    Complexity: O(n² · m).
    Reference: Feo & Resende (1995), J. Global Optim. 6(2): 109–133.
    """
    rng         = random.Random(rng_seed)
    CR          = inst.CR
    seqs:       List[List[int]]   = [[] for _ in range(m)]
    C_hats:     List[List[float]] = [[] for _ in range(m)]
    committed   = [0.0] * m
    B_bar       = 0.0
    unscheduled = list(range(inst.n))

    while unscheduled:
        tau    = min(committed)
        scored = sorted(
            ((float(CR[j]) / max(float(inst.delta[j]) - tau, EPS), j)
             for j in unscheduled),
            reverse=True,
        )
        j = rng.choice([jj for _, jj in scored[:min(k, len(scored))]])

        best_cost = math.inf
        best_rho  = 0
        best_p    = 0
        best_Cn:  List[float] = []

        for rho in range(m):
            for p in _candidate_positions(j, rho, seqs, inst):
                cost, Cn = _evaluate_insertion(
                    j, rho, p, seqs, C_hats, B_bar, inst, params)
                if cost < best_cost:
                    best_cost, best_rho, best_p, best_Cn = cost, rho, p, Cn

        if math.isinf(best_cost):
            best_rho, best_p, best_Cn = _min_violation_insert(j, seqs, inst)

        seqs[best_rho].insert(best_p, j)
        C_hats[best_rho] = best_Cn
        committed[best_rho] = (C_hats[best_rho][-1]
                                if C_hats[best_rho] else 0.0)
        B_bar = sum(committed) / m
        unscheduled.remove(j)

    return seqs, C_hats


# ═══════════════════════════════════════════════════════════════════════════
#   Tier A — Original Six-Rule Assignment Portfolio (§10)
#   Retained verbatim; used by solver.py via build_assignment_portfolio.
# ═══════════════════════════════════════════════════════════════════════════

def _mean_committed(seqs: List[List[int]], inst: Instance) -> float:
    return float(np.mean([committed_time(seqs[rho], inst)
                           for rho in range(len(seqs))]))


def _fallback_insert(j: int, seqs: List[List[int]]) -> None:
    """Append j to the shortest runway when no feasible insertion is found."""
    rho = min(range(len(seqs)), key=lambda r: len(seqs[r]))
    seqs[rho].append(j)


def rbi_construction(inst: Instance, params: HybridParams) -> List[List[int]]:
    """
    Regret-Based Insertion (§10.1, Algorithm 1).

    Priority_j = η·Regret̃_j + (1−η)·CR̃_j.
    """
    seqs:      List[List[int]] = [[] for _ in range(inst.m)]
    remaining: Set[int]        = set(range(inst.n))

    while remaining:
        scores = compute_priority_scores(remaining, seqs, inst, params)
        if not scores:
            for j in list(remaining):
                _, rho, p = best_insertion(j, seqs, inst, params)
                if rho < 0:
                    _fallback_insert(j, seqs)
                else:
                    seqs[rho].insert(p, j)
            break
        j_star    = max(scores, key=lambda j: scores[j][0])
        _, rho, p = scores[j_star]
        if rho < 0:
            _fallback_insert(j_star, seqs)
        else:
            seqs[rho].insert(p, j_star)
        remaining.discard(j_star)

    return seqs


def mpds_balanced_construction(inst: Instance) -> List[List[int]]:
    """
    MPDS-Balanced: greedy (j*, ρ*) = argmax I_{jρ} (§10.2).

    I_{jρ} = exp(−Slack/W̄)·exp(−s[last,j]/s̄)·exp(−Wait/W̄)·exp(−Pen/Pen_bar)
    """
    seqs:      List[List[int]] = [[] for _ in range(inst.m)]
    remaining: Set[int]        = set(range(inst.n))

    while remaining:
        best_score = -math.inf
        best_j     = -1
        best_rho   = 0

        for j in remaining:
            for rho in range(inst.m):
                seq   = seqs[rho]
                B_rho = committed_time(seq, inst)
                last  = seq[-1] if seq else -1
                s_kj  = float(inst.s[last, j]) if last >= 0 else 0.0
                t_hat = max(float(inst.r[j]), B_rho + s_kj)
                slack = max(float(inst.delta[j]) - t_hat, 0.0)
                wait  = max(float(inst.r[j]) - B_rho, 0.0)
                pen   = (float(inst.g[j]) * max(float(inst.delta[j]) - t_hat, 0.0)
                         + float(inst.h[j]) * max(t_hat - float(inst.delta[j]), 0.0))
                score = (math.exp(-slack / (inst.W_bar + EPS))
                         * math.exp(-s_kj  / (inst.s_bar + EPS))
                         * math.exp(-wait  / (inst.W_bar + EPS))
                         * math.exp(-pen   / (inst.Pen_bar + EPS)))
                if score > best_score:
                    best_score = score
                    best_j     = j
                    best_rho   = rho

        seqs[best_rho].append(best_j)
        remaining.discard(best_j)

    return seqs


def edd_balanced_construction(
    inst: Instance, params: HybridParams
) -> List[List[int]]:
    """EDD-Balanced: sort by d_j, insert at min-cost (ρ, p) (§10.3)."""
    seqs  = [[] for _ in range(inst.m)]
    order = sorted(range(inst.n), key=lambda j: float(inst.d[j]))
    for j in order:
        _, rho, p = best_insertion(j, seqs, inst, params)
        if rho < 0:
            _fallback_insert(j, seqs)
        else:
            seqs[rho].insert(p, j)
    return seqs


def target_time_balanced_construction(
    inst: Instance, params: HybridParams
) -> List[List[int]]:
    """
    Target-Time Balanced: sort by δ_j, assign minimising (§10.4):
      Cost = |Ĉ−δ|/Tspan + s[last,j]/s̄ + (B'−B̄)/Tspan
    """
    seqs  = [[] for _ in range(inst.m)]
    order = sorted(range(inst.n), key=lambda j: float(inst.delta[j]))

    for j in order:
        best_cost = math.inf
        best_rho  = 0
        B_bar     = _mean_committed(seqs, inst)

        for rho in range(inst.m):
            seq   = seqs[rho]
            last  = seq[-1] if seq else -1
            s_kj  = float(inst.s[last, j]) if last >= 0 else 0.0
            B_rho = committed_time(seq, inst)
            t_hat = max(float(inst.r[j]), B_rho + s_kj)
            B_new = max(B_rho, t_hat)
            cost  = (abs(t_hat - float(inst.delta[j])) / (inst.T_span + EPS)
                     + s_kj / (inst.s_bar + EPS)
                     + (B_new - B_bar) / (inst.T_span + EPS))
            if cost < best_cost:
                best_cost = cost
                best_rho  = rho

        seqs[best_rho].append(j)

    return seqs


def critical_first_construction(
    inst: Instance, params: HybridParams
) -> List[List[int]]:
    """Critical-Aircraft-First: sort by decreasing CR_j, insert at best (ρ, p) (§10.5)."""
    seqs  = [[] for _ in range(inst.m)]
    order = sorted(range(inst.n), key=lambda j: float(inst.CR[j]), reverse=True)
    for j in order:
        _, rho, p = best_insertion(j, seqs, inst, params)
        if rho < 0:
            _fallback_insert(j, seqs)
        else:
            seqs[rho].insert(p, j)
    return seqs


def wake_compatibility_construction(inst: Instance) -> List[List[int]]:
    """
    Wake-Compatibility Chain: seeds each runway with a high-compatibility
    aircraft, extends in round-robin by Comp[last,j]·exp(−|Ĉ−δ|/W̄) (§10.6).
    """
    seqs:      List[List[int]] = [[] for _ in range(inst.m)]
    remaining: Set[int]        = set(range(inst.n))

    avg_comp   = np.array([
        float(np.mean(1.0 / (inst.s[j, :] + EPS)))
        for j in range(inst.n)
    ])
    seed_order = list(np.argsort(-avg_comp))
    for rho in range(inst.m):
        for j in seed_order:
            if j in remaining:
                seqs[rho].append(j)
                remaining.discard(j)
                break

    while remaining:
        for rho in range(inst.m):
            if not remaining:
                break
            seq    = seqs[rho]
            last   = seq[-1] if seq else -1
            C_last = surrogate_times(seq, inst)[-1] if seq else 0.0

            best_score = -math.inf
            best_j     = next(iter(remaining))

            for j in remaining:
                s_lj  = float(inst.s[last, j]) if last >= 0 else 0.0
                comp  = 1.0 / (s_lj + EPS)
                t_hat = max(float(inst.r[j]), C_last + s_lj)
                fit   = math.exp(-abs(t_hat - float(inst.delta[j]))
                                  / (inst.W_bar + EPS))
                if comp * fit > best_score:
                    best_score = comp * fit
                    best_j     = j

            seqs[rho].append(best_j)
            remaining.discard(best_j)

    return seqs


def build_assignment_portfolio(
    inst: Instance, params: HybridParams,
) -> List[List[List[int]]]:
    """
    Tier A portfolio: six construction rules (§10).

    Returns [RBI, MPDS, EDD, TARGET, CRITICAL, WAKE], each a
    List[List[int]] of shape [m][n_rho].  Used by solver.py Phases 2–4.
    """
    return [
        rbi_construction(inst, params),
        mpds_balanced_construction(inst),
        edd_balanced_construction(inst, params),
        target_time_balanced_construction(inst, params),
        critical_first_construction(inst, params),
        wake_compatibility_construction(inst),
    ]


# ═══════════════════════════════════════════════════════════════════════════
#   Tier B — LP-Screened Seed Portfolio Builder (§25b)
# ═══════════════════════════════════════════════════════════════════════════

def build_seed_portfolio(
    inst:     Instance,
    params:   HeuristicParams,
    m:        Optional[int] = None,
    n_chains: int           = N_CHAINS,
    seed:     int           = 0,
) -> Tuple[
    List[Tuple[str, List[List[int]]]],
    List[Tuple[str, float, float, bool]],
    List[float],
    List[float],
    Dict[str, Any],
]:
    """
    Generate, LP-evaluate, and select construction seeds for SA initialisation.

    Heuristics evaluated (up to 11)
    ────────────────────────────────
    RAMP-RBI, FCFS, EDD, WEDD, ATC, ATCS, CAF, WCC, GRASP-k1, GRASP-k2,
    and MPDS (only when n ≤ MPDS_MAX_N).

    Selection
    ---------
    USE_ALL_SEEDS=False (default): diversity-aware top n_chains seeds by
      LP value.  Diversity is measured by runway-assignment Hamming distance;
      seeds with distance < ELITE_MIN_DIV from an already-selected seed are
      deferred to a second pass that fills by LP value alone.
    USE_ALL_SEEDS=True: every feasibly-evaluated seed is forwarded.

    Parameters
    ----------
    inst     : preprocessed Instance.
    params   : HeuristicParams for RAMP-RBI and insertion-cost evaluation.
    m        : number of runways; defaults to inst.m.
    n_chains : maximum number of seeds to forward to SA (ignored when
               USE_ALL_SEEDS is True).
    seed     : RNG seed for GRASP variants.

    Returns
    -------
    selected_starts   : [(label, sequences)] forwarded to SA chains.
    portfolio_info    : [(label, raw_obj, lp_val, selected)] for all seeds.
    selected_lps      : LP objectives of selected seeds.
    selected_raw_objs : surrogate objectives of selected seeds.
    portfolio_timing  : timing dict with keys:
                          t_seed_construct, t_seed_lp_eval, t_portfolio,
                          t_best_seed_lp, seed_lp_events, seed_timing.
    """
    if m is None:
        m = inst.m

    k1, k2 = GRASP_K_VALUES

    t_port_start      = time.perf_counter()
    seed_timing_list: List[Dict[str, Any]]         = []
    seed_lp_events:   List[Tuple[float, float]]    = []
    evaluated:        List[Tuple[float, float, str, List[List[int]]]] = []
    current_best_lp   = math.inf
    t_construct_total = 0.0
    t_lp_total        = 0.0

    def _timed_seed(label: str, build_fn) -> None:
        nonlocal t_construct_total, t_lp_total, current_best_lp

        t_c0    = time.perf_counter()
        seqs    = build_fn()
        t_c     = time.perf_counter() - t_c0
        t_construct_total += t_c

        raw_obj = sum(surrogate_penalty(seq, inst) for seq in seqs)

        t_lp0   = time.perf_counter()
        lp_val  = stage2_lp_objective(seqs, inst)   # float; math.inf if infeasible
        t_lp    = time.perf_counter() - t_lp0
        t_lp_total += t_lp

        t_job_rel = time.perf_counter() - t_port_start

        if not math.isinf(lp_val) and lp_val < current_best_lp:
            current_best_lp = lp_val
            seed_lp_events.append((t_job_rel, lp_val))

        seed_timing_list.append(dict(
            label=label,
            t_construct=t_c,
            t_lp_eval=t_lp,
            t_job_relative=t_job_rel,
            lp_val=lp_val,
            raw_obj=raw_obj,
        ))
        evaluated.append((lp_val, raw_obj, label, seqs))

    # ── Evaluate all heuristics in sequence ──────────────────────────────
    _timed_seed("RAMP-RBI",      lambda: ramp_rbi(inst, m, params)[0])
    _timed_seed("FCFS",          lambda: seed_fcfs(inst, m)[0])
    _timed_seed("EDD",           lambda: seed_edd_balanced(inst, m)[0])
    _timed_seed("WEDD",          lambda: seed_wedd(inst, m)[0])
    _timed_seed("ATC",           lambda: seed_atc(inst, m, K=ATC_K)[0])
    _timed_seed("ATCS",          lambda: seed_atcs(inst, m, K1=ATCS_K1, K2=ATCS_K2)[0])
    _timed_seed("CAF",           lambda: seed_caf(inst, m, params)[0])
    _timed_seed("WCC",           lambda: seed_wcc(inst, m, params)[0])
    _timed_seed(f"GRASP-{k1}",   lambda: seed_grasp(inst, m, params, k=k1,
                                                      rng_seed=seed)[0])
    _timed_seed(f"GRASP-{k2}",   lambda: seed_grasp(inst, m, params, k=k2,
                                                      rng_seed=seed + 999)[0])
    if inst.n <= MPDS_MAX_N:
        _timed_seed("MPDS",      lambda: seed_mpds(inst, m, params)[0])

    t_port_total = time.perf_counter() - t_port_start

    # ── Sort by LP value (inf last) ───────────────────────────────────────
    evaluated.sort(key=lambda x: x[0])

    # ── Diversity-aware selection ─────────────────────────────────────────
    def _rwy_hamming(sa: List[List[int]], sb: List[List[int]]) -> int:
        """Runway-assignment Hamming distance between two complete schedules."""
        ma = len(sa)
        aa = {sa[r][p]: r
              for r in range(ma) for p in range(len(sa[r]))}
        return sum(1 for r in range(len(sb)) for j in sb[r]
                   if aa.get(j) != r)

    selected: List[Tuple[str, List[List[int]], float, float]] = []

    if USE_ALL_SEEDS:
        selected = [
            (label, seqs, lp_val, raw_obj)
            for lp_val, raw_obj, label, seqs in evaluated
            if not math.isinf(lp_val)
        ]
    else:
        used_lbl: Set[str] = set()

        # First pass: diversity-preferred
        for lp_val, raw_obj, label, seqs in evaluated:
            if len(selected) >= n_chains:
                break
            if label in used_lbl or math.isinf(lp_val):
                continue
            diverse = (not selected
                       or all(_rwy_hamming(seqs, s) >= ELITE_MIN_DIV
                              for _, s, _, _ in selected))
            if diverse:
                selected.append((label, seqs, lp_val, raw_obj))
                used_lbl.add(label)

        # Second pass: fill remaining slots by LP value
        for lp_val, raw_obj, label, seqs in evaluated:
            if len(selected) >= n_chains:
                break
            if label in used_lbl or math.isinf(lp_val):
                continue
            selected.append((label, seqs, lp_val, raw_obj))
            used_lbl.add(label)

    sel_labels        = {lbl for lbl, _, _, _ in selected}
    selected_starts   = [(lbl, s)  for lbl, s,  _,  _  in selected]
    selected_lps      = [lp        for _,   _,  lp, _  in selected]
    selected_raw_objs = [ro        for _,   _,  _,  ro in selected]
    portfolio_info    = [
        (label, raw_obj, lp_val, label in sel_labels)
        for lp_val, raw_obj, label, _ in evaluated
    ]

    portfolio_timing: Dict[str, Any] = dict(
        t_seed_construct=t_construct_total,
        t_seed_lp_eval=t_lp_total,
        t_portfolio=t_port_total,
        t_best_seed_lp=(seed_lp_events[-1][0] if seed_lp_events else 0.0),
        seed_lp_events=seed_lp_events,
        seed_timing=seed_timing_list,
    )

    return (selected_starts, portfolio_info,
            selected_lps, selected_raw_objs, portfolio_timing)