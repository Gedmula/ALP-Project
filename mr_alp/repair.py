"""
repair.py — MR-ALP Solver: LP-Guided Repair, VND Polish, Elite Pool, Path Relinking
=====================================================================================
§19  LP-guided repair operators
       lp_guided_penalty_repair  — relocate top-penalty aircraft globally.
       lp_guided_pair_swap       — cross-runway swap of near-δ-time aircraft.
       target_conflict_repair    — deterministic repair for near-zero objectives.
       ejection_chain_transfer   — depth-D ejection chain from high-impact aircraft.
       lns_remove_reinsert       — LNS: remove top-k simultaneously, reinsert jointly.
§20  ElitePool — fixed-size pool with runway-Hamming diversity guard.
§21  path_relink — walk from one elite solution toward another, evaluating LP at intervals.
§24  lp_vnd_polish — monotone LP-VND combining all five repair operators.
"""
from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from mr_alp.models     import Instance, HeuristicParams, MRSAParams
from mr_alp.instance   import runway_feasible
from mr_alp.lp         import stage2_lp_objective
from mr_alp.proxy      import (
    _rwy_proxy_components, init_proxy_arrays, compute_proxy, lp_impact_scores,
)
from mr_alp.config     import ELITE_POOL_MAX, ELITE_MIN_DIV


# ═══════════════════════════════════════════════════════════════════════════
#   §19  LP-GUIDED REPAIR OPERATORS
# ═══════════════════════════════════════════════════════════════════════════

def _top_penalty_aircraft(
    C_lp: np.ndarray, inst: Instance, q: int
) -> List[int]:
    """Return indices of the q highest-penalty aircraft under C_lp."""
    E = np.maximum(inst.delta - C_lp, 0.0)
    T = np.maximum(C_lp - inst.delta, 0.0)
    return list(np.argsort(inst.g * E + inst.h * T)[::-1][:q])


def _lp_repair_params(n: int) -> Tuple[int, int]:
    """Return (q_lp, K): number of target aircraft and candidate pool cap."""
    if n <= 50:   return 20, 20
    if n <= 100:  return 15, 15
    if n <= 250:  return 10, 10
    return 12, 10


def lp_guided_penalty_repair(
    seqs: List[List[int]], C_lp: np.ndarray,
    inst: Instance, params: HeuristicParams,
    K: int = 15, q_lp: int = 15,
) -> Tuple[Optional[List[List[int]]], float]:
    """
    Attempt to reduce LP penalty by globally relocating each of the q_lp
    highest-penalty aircraft to any feasible (runway, position) pair.

    Candidates are ranked by proxy objective; the top K are LP-evaluated.
    Returns (best_seqs, best_lp) or (None, inf) if no improvement is found.
    """
    m   = len(seqs)
    loc = {seqs[rho][pos]: (rho, pos)
           for rho in range(m) for pos in range(len(seqs[rho]))}
    H   = _top_penalty_aircraft(C_lp, inst, q_lp)
    candidates = []

    for j in H:
        rho_src, pos_src = loc[j]
        sm = seqs[rho_src][:pos_src] + seqs[rho_src][pos_src + 1:]
        if not runway_feasible(sm, inst):
            continue
        base = [s[:] for s in seqs]; base[rho_src] = sm
        for rho_dst in range(m):
            for p_dst in range(len(base[rho_dst]) + 1):
                cand = [s[:] for s in base]
                cand[rho_dst] = cand[rho_dst][:p_dst] + [j] + cand[rho_dst][p_dst:]
                if not runway_feasible(cand[rho_dst], inst):
                    continue
                tc_n, lbt_n, sep_n = init_proxy_arrays(cand, inst)
                candidates.append((
                    compute_proxy(cand, tc_n, lbt_n, sep_n, inst, params), cand))
        if len(candidates) > K * 20:
            candidates.sort(key=lambda x: x[0])
            candidates = candidates[:K * 5]

    if not candidates:
        return None, math.inf
    candidates.sort(key=lambda x: x[0])
    best_lp = math.inf; best_cand = None
    for _, cand in candidates[:K]:
        lp, _, feas, _ = stage2_lp_objective(cand, inst)
        if feas and lp < best_lp:
            best_lp, best_cand = lp, cand
    return best_cand, best_lp


def lp_guided_pair_swap(
    seqs: List[List[int]], C_lp: np.ndarray,
    inst: Instance, params: HeuristicParams,
    q_lp: int = 15, K: int = 30, kappa: float = 0.25,
) -> Tuple[Optional[List[List[int]]], float]:
    """
    Cross-runway swap of high-penalty aircraft with target-time-compatible
    partners (|δ_i − δ_j| ≤ κ·W_bar).

    Returns (best_seqs, best_lp) or (None, inf).
    """
    from mr_alp.operators import op_x2_swap
    m   = len(seqs); W_bar = inst.W_bar
    H   = _top_penalty_aircraft(C_lp, inst, q_lp)
    loc = {seqs[rho][pos]: (rho, pos)
           for rho in range(m) for pos in range(len(seqs[rho]))}
    candidates = []

    for i in H:
        rho_i, pos_i = loc[i]
        for rho_j in range(m):
            if rho_j == rho_i:
                continue
            for pos_j, j in enumerate(seqs[rho_j]):
                if abs(inst.delta[i] - inst.delta[j]) > kappa * W_bar:
                    continue
                res = op_x2_swap(seqs, rho_i, pos_i, rho_j, pos_j, inst)
                if res is None:
                    continue
                tc_n, lbt_n, sep_n = init_proxy_arrays(res.seqs, inst)
                candidates.append((
                    compute_proxy(res.seqs, tc_n, lbt_n, sep_n, inst, params),
                    res.seqs))

    if not candidates:
        return None, math.inf
    candidates.sort(key=lambda x: x[0])
    best_lp = math.inf; best_cand = None
    for _, cand in candidates[:K]:
        lp, _, feas, _ = stage2_lp_objective(cand, inst)
        if feas and lp < best_lp:
            best_lp, best_cand = lp, cand
    return best_cand, best_lp


def target_conflict_repair(
    seqs: List[List[int]], inst: Instance, params: HeuristicParams, K: int = 15
) -> Tuple[Optional[List[List[int]]], float]:
    """
    Deterministic repair for near-zero-objective instances.

    Identifies the most conflicting pairs (large s[i,j] − (δ_j − δ_i))
    and attempts to relocate the contributing aircraft to a better global
    position.  Returns (best_seqs, best_lp) or (None, inf).
    """
    m = len(seqs); conflicts = []
    for rho, seq in enumerate(seqs):
        for qi in range(len(seq)):
            for qj in range(qi + 1, len(seq)):
                i, j = seq[qi], seq[qj]
                tc = max(0.0, float(inst.s[i, j]) - (float(inst.delta[j])
                                                       - float(inst.delta[i])))
                if tc > 1e-9:
                    conflicts.append((tc, rho, qi, i, j))
    if not conflicts:
        return None, math.inf
    conflicts.sort(reverse=True)

    loc = {seqs[rho][pos]: (rho, pos)
           for rho in range(m) for pos in range(len(seqs[rho]))}
    candidates = []
    for _, _, _, i, j in conflicts[:8]:
        for ac in [i, j]:
            rho_src, pos_src = loc[ac]
            sm = seqs[rho_src][:pos_src] + seqs[rho_src][pos_src + 1:]
            if not runway_feasible(sm, inst):
                continue
            base = [s[:] for s in seqs]; base[rho_src] = sm
            for rho_dst in range(m):
                for p_dst in range(len(base[rho_dst]) + 1):
                    if rho_dst == rho_src and p_dst == pos_src:
                        continue
                    cand = [s[:] for s in base]
                    cand[rho_dst] = (cand[rho_dst][:p_dst] + [ac]
                                     + cand[rho_dst][p_dst:])
                    if not runway_feasible(cand[rho_dst], inst):
                        continue
                    tc_n, lbt_n, sep_n = init_proxy_arrays(cand, inst)
                    candidates.append((
                        compute_proxy(cand, tc_n, lbt_n, sep_n, inst, params), cand))
            if len(candidates) > K * 15:
                candidates.sort(key=lambda x: x[0])
                candidates = candidates[:K * 4]

    if not candidates:
        return None, math.inf
    candidates.sort(key=lambda x: x[0])
    best_lp = math.inf; best_cand = None
    for _, cand in candidates[:K]:
        lp, _, feas, _ = stage2_lp_objective(cand, inst)
        if feas and lp < best_lp:
            best_lp, best_cand = lp, cand
    return best_cand, best_lp


def ejection_chain_transfer(
    seqs: List[List[int]], C_lp: np.ndarray,
    inst: Instance, params: HeuristicParams,
    depth: int = 2, K: int = 15,
) -> Tuple[Optional[List[List[int]]], float]:
    """
    Depth-D ejection chain starting from high-impact aircraft.

    Depth 1: simple X3 transfer  j1: ρ1 → ρ2.
    Depth 2: j1: ρ1 → ρ2, then j2 (displaced from ρ2): ρ2 → ρ3.
    Depth is capped at 1 when m < 3.

    Returns (best_seqs, best_lp) or (None, inf).
    """
    m = len(seqs)
    if m < 3:
        depth = 1
    q_lp, _ = _lp_repair_params(inst.n)
    H   = _top_penalty_aircraft(C_lp, inst, min(q_lp, 6))
    loc = {seqs[rho][pos]: (rho, pos)
           for rho in range(m) for pos in range(len(seqs[rho]))}
    candidates = []

    for j1 in H:
        rho1, pos1 = loc[j1]
        sm1 = seqs[rho1][:pos1] + seqs[rho1][pos1 + 1:]
        if not runway_feasible(sm1, inst):
            continue
        for rho2 in range(m):
            if rho2 == rho1:
                continue
            best_q2, best_seq2, best_s2 = -1, None, math.inf
            for q2 in range(len(seqs[rho2]) + 1):
                c2 = seqs[rho2][:q2] + [j1] + seqs[rho2][q2:]
                if not runway_feasible(c2, inst):
                    continue
                tc, lbt, sep = _rwy_proxy_components(c2, inst)
                s = params.mu_tc * tc + params.mu_late * lbt + params.mu_sep * sep
                if s < best_s2:
                    best_s2, best_q2, best_seq2 = s, q2, c2
            if best_seq2 is None:
                continue
            st1 = [s[:] for s in seqs]; st1[rho1] = sm1; st1[rho2] = best_seq2

            if depth == 1:
                tc_n, lbt_n, sep_n = init_proxy_arrays(st1, inst)
                candidates.append((
                    compute_proxy(st1, tc_n, lbt_n, sep_n, inst, params),
                    [s[:] for s in st1]))
            else:
                for j2 in seqs[rho2]:
                    try:
                        j2_pos = best_seq2.index(j2)
                    except ValueError:
                        continue
                    sm2 = best_seq2[:j2_pos] + best_seq2[j2_pos + 1:]
                    if not runway_feasible(sm2, inst):
                        continue
                    for rho3 in range(m):
                        if rho3 == rho2:
                            continue
                        best_q3, best_seq3, best_s3 = -1, None, math.inf
                        for q3 in range(len(st1[rho3]) + 1):
                            c3 = st1[rho3][:q3] + [j2] + st1[rho3][q3:]
                            if not runway_feasible(c3, inst):
                                continue
                            tc, lbt, sep = _rwy_proxy_components(c3, inst)
                            s = params.mu_tc * tc + params.mu_late * lbt + params.mu_sep * sep
                            if s < best_s3:
                                best_s3, best_q3, best_seq3 = s, q3, c3
                        if best_seq3 is None:
                            continue
                        st2 = [s[:] for s in st1]; st2[rho2] = sm2; st2[rho3] = best_seq3
                        tc_n, lbt_n, sep_n = init_proxy_arrays(st2, inst)
                        candidates.append((
                            compute_proxy(st2, tc_n, lbt_n, sep_n, inst, params),
                            [s[:] for s in st2]))
                    if len(candidates) >= K * 20: break
                if len(candidates) >= K * 20: break
            if len(candidates) >= K * 20: break
        if len(candidates) >= K * 20: break

    if not candidates:
        return None, math.inf
    candidates.sort(key=lambda x: x[0])
    best_lp = math.inf; best_cand = None
    for _, cand in candidates[:K]:
        lp, _, feas, _ = stage2_lp_objective(cand, inst)
        if feas and lp < best_lp:
            best_lp, best_cand = lp, cand
    return best_cand, best_lp


def lns_remove_reinsert(
    seqs:    List[List[int]],
    C_lp:    np.ndarray,
    inst:    Instance,
    params:  HeuristicParams,
    k:       int = 3,
    K:       int = 8,
    n_perms: int = 5,
) -> Tuple[Optional[List[List[int]]], float]:
    """
    LNS destroy-repair: remove the top-k highest-penalty aircraft simultaneously,
    then reinsert them using multiple greedy orderings ranked by proxy and LP-
    evaluated.  Explores joint assignment interactions that sequential single-
    aircraft relocations miss (e.g. swapping two aircraft between overloaded
    runways simultaneously reduces both their penalties).

    Returns (best_seqs, best_lp) or (None, inf).
    """
    m       = len(seqs)
    targets = _top_penalty_aircraft(C_lp, inst, k)

    # Build base schedule with all k targets removed simultaneously
    base = [s[:] for s in seqs]
    for ac in targets:
        for rho in range(m):
            if ac in base[rho]:
                pos = base[rho].index(ac)
                sm  = base[rho][:pos] + base[rho][pos + 1:]
                if not runway_feasible(sm, inst):
                    return None, math.inf
                base[rho] = sm
                break

    def _greedy_insert(order: List[int]) -> Optional[List[List[int]]]:
        """Insert aircraft in `order` one-by-one, each at best proxy position."""
        r = [s[:] for s in base]
        for ac in order:
            best_score, best_rho, best_q = math.inf, -1, -1
            for rho in range(m):
                for q in range(len(r[rho]) + 1):
                    cand = r[rho][:q] + [ac] + r[rho][q:]
                    if not runway_feasible(cand, inst):
                        continue
                    tc, lbt, sep = _rwy_proxy_components(cand, inst)
                    s = params.mu_tc * tc + params.mu_late * lbt + params.mu_sep * sep
                    if s < best_score:
                        best_score, best_rho, best_q = s, rho, q
            if best_rho == -1:
                return None
            r[best_rho] = r[best_rho][:best_q] + [ac] + r[best_rho][best_q:]
        return r

    # Collect candidate solutions for all insertion orderings (up to n_perms)
    candidates: list = []
    all_perms  = list(itertools.permutations(targets))
    step       = max(1, len(all_perms) // n_perms)
    orderings  = [list(all_perms[i])
                  for i in range(0, len(all_perms), step)][:n_perms]

    for order in orderings:
        r = _greedy_insert(order)
        if r is None:
            continue
        tc_n, lbt_n, sep_n = init_proxy_arrays(r, inst)
        candidates.append((
            compute_proxy(r, tc_n, lbt_n, sep_n, inst, params),
            [s[:] for s in r]))

    if not candidates:
        return None, math.inf
    candidates.sort(key=lambda x: x[0])
    best_lp = math.inf; best_cand = None
    for _, cand in candidates[:K]:
        lp, _, feas, _ = stage2_lp_objective(cand, inst)
        if feas and lp < best_lp:
            best_lp, best_cand = lp, cand
    return best_cand, best_lp


# ═══════════════════════════════════════════════════════════════════════════
#   §20  ELITE SOLUTION POOL
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class _EliteSolution:
    seqs:   List[List[int]]
    lp_obj: float
    C_lp:   Optional[np.ndarray]


class ElitePool:
    """
    Fixed-size pool of LP-certified schedules with a runway-Hamming diversity
    admission guard.

    Admission policy
    ----------------
    A candidate is admitted if:
      (a) its LP objective is strictly better than the worst incumbent, OR
      (b) it has runway-Hamming distance ≥ min_diversity to every incumbent.
    When the pool exceeds max_size, solutions are sorted by LP and trimmed.

    Methods
    -------
    try_add(seqs, lp_obj, C_lp) → bool
    runway_distance(sa, sb) → int
    best → _EliteSolution | None
    most_diverse_pair() → (_EliteSolution, _EliteSolution) | (None, None)
    best_quality_pair() → (_EliteSolution, _EliteSolution) | (None, None)
    """
    def __init__(self, max_size: int = ELITE_POOL_MAX,
                 min_diversity: int = ELITE_MIN_DIV):
        self.solutions:     List[_EliteSolution] = []
        self.max_size:      int = max_size
        self.min_diversity: int = min_diversity

    def runway_distance(
        self, seqs_a: List[List[int]], seqs_b: List[List[int]]
    ) -> int:
        """Runway-Hamming distance: number of aircraft assigned to different runways."""
        m = len(seqs_a)
        aa = {seqs_a[r][p]: r for r in range(m) for p in range(len(seqs_a[r]))}
        return sum(1 for r in range(len(seqs_b))
                   for j in seqs_b[r] if aa.get(j) != r)

    def try_add(
        self,
        seqs: List[List[int]],
        lp_obj: float,
        C_lp: Optional[np.ndarray],
    ) -> bool:
        if math.isinf(lp_obj):
            return False
        if not self.solutions:
            self.solutions.append(
                _EliteSolution([s[:] for s in seqs], lp_obj,
                               C_lp.copy() if C_lp is not None else None))
            return True
        diverse  = all(self.runway_distance(seqs, s.seqs) >= self.min_diversity
                       for s in self.solutions)
        worst_lp = max(s.lp_obj for s in self.solutions)
        if lp_obj < worst_lp or diverse:
            self.solutions.append(
                _EliteSolution([s[:] for s in seqs], lp_obj,
                               C_lp.copy() if C_lp is not None else None))
            if len(self.solutions) > self.max_size:
                self.solutions.sort(key=lambda s: s.lp_obj)
                self.solutions = self.solutions[:self.max_size]
            return True
        return False

    @property
    def best(self) -> Optional[_EliteSolution]:
        return min(self.solutions, key=lambda s: s.lp_obj) if self.solutions else None

    def most_diverse_pair(
        self,
    ) -> Tuple[Optional[_EliteSolution], Optional[_EliteSolution]]:
        if len(self.solutions) < 2:
            return None, None
        best_d = -1; best_a = best_b = None
        for i in range(len(self.solutions)):
            for j in range(i + 1, len(self.solutions)):
                d = self.runway_distance(self.solutions[i].seqs,
                                          self.solutions[j].seqs)
                if d > best_d:
                    best_d, best_a, best_b = d, self.solutions[i], self.solutions[j]
        return best_a, best_b

    def best_quality_pair(
        self,
    ) -> Tuple[Optional[_EliteSolution], Optional[_EliteSolution]]:
        if len(self.solutions) < 2:
            return None, None
        ss = sorted(self.solutions, key=lambda s: s.lp_obj)
        for i in range(len(ss)):
            for j in range(i + 1, len(ss)):
                if self.runway_distance(ss[i].seqs, ss[j].seqs) >= self.min_diversity:
                    return ss[i], ss[j]
        return self.most_diverse_pair()


# ═══════════════════════════════════════════════════════════════════════════
#   §21  PATH RELINKING
# ═══════════════════════════════════════════════════════════════════════════

def path_relink(
    sol_a: _EliteSolution,
    sol_b: _EliteSolution,
    inst: Instance,
    params: HeuristicParams,
    max_steps: int = 40,
    eval_interval: int = 5,
    K_lp: int = 8,
) -> Tuple[List[List[int]], float]:
    """
    Walk from sol_a toward sol_b by iteratively moving differing aircraft to
    their target runway in sol_b.  LP is evaluated every eval_interval steps
    on the best proxy candidates seen so far.

    Returns (best_seqs, best_lp).
    """
    m       = len(sol_a.seqs)
    current = [s[:] for s in sol_a.seqs]
    best_seqs = [s[:] for s in sol_a.seqs]; best_lp = sol_a.lp_obj
    assign_b  = {sol_b.seqs[r][p]: r
                 for r in range(m) for p in range(len(sol_b.seqs[r]))}
    proxy_buffer = []

    def _flush_buffer():
        nonlocal best_seqs, best_lp
        proxy_buffer.sort(key=lambda x: x[0])
        for _, cand in proxy_buffer[:K_lp]:
            lp, _, feas, _ = stage2_lp_objective(cand, inst)
            if feas and lp < best_lp - 1e-9:
                best_seqs, best_lp = cand, lp
        proxy_buffer.clear()

    for step in range(max_steps):
        assign_cur = {current[r][p]: r
                      for r in range(m) for p in range(len(current[r]))}
        differing  = [(j, assign_b[j]) for j in assign_b
                      if assign_cur.get(j) != assign_b[j]]
        if not differing:
            break

        if sol_a.C_lp is not None:
            impact = lp_impact_scores(current, sol_a.C_lp, inst)
            differing.sort(key=lambda x: -impact[x[0]])
        else:
            differing.sort(key=lambda x: -(inst.g[x[0]] + inst.h[x[0]]))

        moved = False
        for j, rho_target in differing[:5]:
            rho_cur = assign_cur.get(j)
            if rho_cur is None or rho_cur == rho_target:
                continue
            pos_cur = current[rho_cur].index(j)
            sm = current[rho_cur][:pos_cur] + current[rho_cur][pos_cur + 1:]
            if not runway_feasible(sm, inst):
                continue
            best_q, best_score = -1, math.inf
            for q in range(len(current[rho_target]) + 1):
                cs = current[rho_target][:q] + [j] + current[rho_target][q:]
                if not runway_feasible(cs, inst):
                    continue
                tc, lbt, sep = _rwy_proxy_components(cs, inst)
                s = params.mu_tc * tc + params.mu_late * lbt + params.mu_sep * sep
                if s < best_score:
                    best_score, best_q = s, q
            if best_q == -1:
                continue
            current[rho_cur] = sm
            current[rho_target] = (current[rho_target][:best_q]
                                   + [j]
                                   + current[rho_target][best_q:])
            moved = True; break

        if not moved:
            break
        tc_n, lbt_n, sep_n = init_proxy_arrays(current, inst)
        px = compute_proxy(current, tc_n, lbt_n, sep_n, inst, params)
        proxy_buffer.append((px, [s[:] for s in current]))
        if (step + 1) % eval_interval == 0:
            _flush_buffer()

    if proxy_buffer:
        _flush_buffer()
    return best_seqs, best_lp


# ═══════════════════════════════════════════════════════════════════════════
#   §24  LP-VND POLISH
# ═══════════════════════════════════════════════════════════════════════════

def lp_vnd_polish(
    seqs: List[List[int]],
    init_lp: float,
    C_lp: Optional[np.ndarray],
    inst: Instance,
    params: HeuristicParams,
    p_sa: Optional[MRSAParams] = None,
    max_rounds: int = 10,
    t_limit: float = 90.0,
) -> Tuple[List[List[int]], float]:
    """
    Monotone LP-VND using five neighbourhoods applied in order:
      N1  lp_guided_penalty_repair
      N2  lp_guided_pair_swap
      N3  target_conflict_repair   (only when LP obj < 200)
      N4  ejection_chain_transfer
      N5  lns_remove_reinsert      (only when m >= 2)

    Restarts from N1 on any LP improvement.  Terminates when no operator
    improves within one full pass or t_limit is exceeded.
    """
    import time
    p_sa     = p_sa or MRSAParams()
    m        = len(seqs)
    ec_depth = min(p_sa.ejection_chain_depth, 2 if m < 3 else p_sa.ejection_chain_depth)
    q_lp, K  = _lp_repair_params(inst.n)
    t0       = time.perf_counter()
    best_seqs = [s[:] for s in seqs]; best_lp = init_lp; best_C = (
        C_lp.copy() if C_lp is not None else None)

    for _ in range(max_rounds):
        if time.perf_counter() - t0 > t_limit:
            break
        improved = False

        # N1
        if best_C is not None:
            cand, cand_lp = lp_guided_penalty_repair(
                best_seqs, best_C, inst, params, K=K, q_lp=q_lp)
            if cand is not None and cand_lp < best_lp - 1e-9:
                best_seqs, best_lp = cand, cand_lp
                _, best_C, _, _ = stage2_lp_objective(best_seqs, inst)
                improved = True; continue

        # N2
        if best_C is not None:
            cand, cand_lp = lp_guided_pair_swap(
                best_seqs, best_C, inst, params, q_lp=q_lp, K=K)
            if cand is not None and cand_lp < best_lp - 1e-9:
                best_seqs, best_lp = cand, cand_lp
                _, best_C, _, _ = stage2_lp_objective(best_seqs, inst)
                improved = True; continue

        # N3 (near-zero only)
        if best_lp < 200.0:
            cand, cand_lp = target_conflict_repair(
                best_seqs, inst, params, K=max(K // 2, 3))
            if cand is not None and cand_lp < best_lp - 1e-9:
                best_seqs, best_lp = cand, cand_lp
                _, C_new, feas_new, _ = stage2_lp_objective(best_seqs, inst)
                if feas_new:
                    best_C = C_new
                improved = True; continue

        # N4
        if best_C is not None and m >= 2:
            cand, cand_lp = ejection_chain_transfer(
                best_seqs, best_C, inst, params,
                depth=ec_depth, K=max(K // 2, 3))
            if cand is not None and cand_lp < best_lp - 1e-9:
                best_seqs, best_lp = cand, cand_lp
                _, best_C, _, _ = stage2_lp_objective(best_seqs, inst)
                improved = True; continue

        # N5: LNS destroy-repair (m >= 2 only)
        if best_C is not None and m >= 2:
            k_lns = max(3, min(5, inst.n // (m * 10)))
            cand, cand_lp = lns_remove_reinsert(
                best_seqs, best_C, inst, params, k=k_lns, K=K)
            if cand is not None and cand_lp < best_lp - 1e-9:
                best_seqs, best_lp = cand, cand_lp
                _, best_C, _, _ = stage2_lp_objective(best_seqs, inst)
                improved = True; continue

        if not improved:
            break

    return best_seqs, best_lp