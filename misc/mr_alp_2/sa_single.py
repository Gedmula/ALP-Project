"""
sa_single.py — Phase 3: Single-runway simulated annealing (§11).

The surrogate objective (consecutive-propagation penalty) is used for SA
acceptance decisions; the Stage-2 LP is invoked only by the outer solver
after each accepted inter-runway move (§17.2).

Neighbourhood operators (§11.1):
  N1: adjacent swap       — swap positions p and p+1.
  N2: general swap        — swap any two positions.
  N3: insertion           — remove aircraft at i, re-insert at j.
  N4: block reversal      — reverse subsequence [i..j].
  N5: critical insertion  — move highest-penalty aircraft to best position.

Temperature control uses reactive α adaptation:
  if acceptance rate χ > χ*: α decreases (faster cooling)
  if χ < χ*:                 α increases (slower cooling)
Stagnation triggers reheating up to MAX_REHEATS times.
"""
from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple

import numpy as np

from misc.mr_alp_2.models import Instance, SingleSAParams
from misc.mr_alp_2.surrogate import surrogate_times, surrogate_penalty


# ─────────────────────────────────────────────────────────────────────────────
#   §11.1  Neighbourhood operators
# ─────────────────────────────────────────────────────────────────────────────

def _op_adjacent_swap(seq: List[int], p: int) -> List[int]:
    s = seq[:]; s[p], s[p + 1] = s[p + 1], s[p]; return s


def _op_general_swap(seq: List[int], i: int, j: int) -> List[int]:
    s = seq[:]; s[i], s[j] = s[j], s[i]; return s


def _op_insertion(seq: List[int], i: int, j: int) -> List[int]:
    s   = seq[:]
    ac  = s.pop(i)
    ins = j if j <= i else j - 1
    s.insert(ins, ac)
    return s


def _op_block_reversal(seq: List[int], i: int, j: int) -> List[int]:
    if i > j:
        i, j = j, i
    s = seq[:]
    s[i:j + 1] = s[i:j + 1][::-1]
    return s


def _op_critical_insertion(seq: List[int], inst: Instance) -> Optional[List[int]]:
    """
    N5: move the aircraft with the highest surrogate penalty to its best
    position in the same runway (§11.1).
    """
    if len(seq) < 2:
        return None
    C    = surrogate_times(seq, inst)
    pens = [
        float(inst.g[j]) * max(float(inst.delta[j]) - C[q], 0.0)
        + float(inst.h[j]) * max(C[q] - float(inst.delta[j]), 0.0)
        for q, j in enumerate(seq)
    ]
    src = int(np.argmax(pens))

    best_pen = math.inf
    best_seq = None
    for dst in range(len(seq) + 1):
        if dst in (src, src + 1):
            continue
        cand = _op_insertion(seq, src, dst)
        pen  = surrogate_penalty(cand, inst)
        if pen < best_pen:
            best_pen = pen
            best_seq = cand
    return best_seq


def _random_neighbour(
    seq: List[int], rng: random.Random, n: int, inst: Instance
) -> List[int]:
    op = rng.randint(0, 4)
    if op == 0 and n >= 2:
        return _op_adjacent_swap(seq, rng.randint(0, n - 2))
    if op == 1 and n >= 2:
        i, j = rng.sample(range(n), 2)
        return _op_general_swap(seq, i, j)
    if op == 2 and n >= 2:
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n)
        return _op_insertion(seq, i, j)
    if op == 3 and n >= 2:
        i, j = sorted(rng.sample(range(n), 2))
        return _op_block_reversal(seq, i, j)
    crit = _op_critical_insertion(seq, inst)
    return crit if crit is not None else seq[:]


# ─────────────────────────────────────────────────────────────────────────────
#   T₀ calibration
# ─────────────────────────────────────────────────────────────────────────────

def _calibrate_T0(
    seq:  List[int],
    inst: Instance,
    p:    SingleSAParams,
    seed: int,
) -> float:
    """
    Sample neighbourhood moves and set T₀ so worsening moves of typical
    magnitude are accepted with probability chi0.
    """
    rng  = random.Random(seed)
    n    = len(seq)
    f0   = surrogate_penalty(seq, inst)
    dps  = []
    for _ in range(150):
        fn = surrogate_penalty(_random_neighbour(seq, rng, n, inst), inst)
        if fn > f0:
            dps.append(fn - f0)
    if not dps:
        return max(f0 * 0.01, 1.0)
    return max(-float(np.mean(dps)) / math.log(p.chi0 + 1e-12), 1e-3)


# ─────────────────────────────────────────────────────────────────────────────
#   §11  Single-runway SA (Algorithm 2)
# ─────────────────────────────────────────────────────────────────────────────

def run_single_runway_sa(
    seq0: List[int],
    inst: Instance,
    p:    SingleSAParams,
    seed: int = 0,
    T0:   Optional[float] = None,
) -> Tuple[List[int], float]:
    """
    Single-runway SA minimising surrogate_penalty (§11, Algorithm 2).

    Reactive α adaptation adjusts the cooling rate each outer iteration
    to keep the acceptance rate near χ* = 0.20.  Stagnation triggers
    reheating (temperature multiplied by T_REHEAT) up to MAX_REHEATS.

    Returns (best_sequence, best_surrogate_objective).
    """
    if not seq0:
        return [], 0.0

    # Reactive-adaptation constants
    CHI_TARGET  = 0.20
    ALPHA_STEP  = 0.005
    ALPHA_LO    = 0.80
    ALPHA_HI    = 0.999
    MAX_REHEATS = 2
    T_REHEAT    = 2.0

    rng   = random.Random(seed)
    n     = len(seq0)
    T     = T0 if T0 is not None else _calibrate_T0(seq0, inst, p, seed)
    alpha = p.alpha

    pi      = seq0[:]
    f       = surrogate_penalty(pi, inst)
    pi_best = pi[:]
    f_best  = f
    stag    = 0
    n_rht   = 0

    for _ in range(p.I_max):
        n_accepted = 0
        n_tried    = 0
        improved   = False

        for _ in range(p.N_iter):
            pi2 = _random_neighbour(pi, rng, n, inst)
            f2  = surrogate_penalty(pi2, inst)
            n_tried += 1
            dlt = f2 - f
            if dlt <= 0 or rng.random() < math.exp(-dlt / max(T, 1e-15)):
                pi, f = pi2, f2
                n_accepted += 1
            if f < f_best - 1e-9:
                pi_best, f_best = pi[:], f
                improved = True

        # Reactive α update (§11.2)
        chi   = n_accepted / max(n_tried, 1)
        alpha = (max(ALPHA_LO, alpha - ALPHA_STEP)
                 if chi > CHI_TARGET
                 else min(ALPHA_HI, alpha + ALPHA_STEP))
        T    *= alpha

        if improved:
            stag = 0
        else:
            stag += 1

        if T < p.T_min:
            break

        if stag >= p.M_stag:
            if n_rht >= MAX_REHEATS:
                break
            T     *= T_REHEAT
            pi, f  = pi_best[:], f_best
            stag   = 0
            n_rht += 1

    return pi_best, f_best