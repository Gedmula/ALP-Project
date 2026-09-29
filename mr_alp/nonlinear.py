"""
nonlinear.py — Pinol & Beasley (2006) non-linear objective (maximisation).

    maximise Σ_i D_i,   D_i = −d_i² if d_i ≥ 0 else +d_i²,   d_i = x_i − T_i

D_i is strictly decreasing in x_i, so for a fixed sequence the componentwise
earliest feasible schedule is optimal ("close-up property"): no LP needed.
Separation is enforced for all ordered pairs on a runway (triangle inequality
fails on OR-Library data); since earliest times are nondecreasing along a
runway, the backward scan can stop once x_h + max(S) ≤ x_q.

Σ_i D_i(E_i) is an upper bound on the optimum for any number of runways.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not (a and callable(a[0])) else a[0]


@njit(cache=True)
def _earliest(seq, r, d, s, s_max):
    L = seq.shape[0]
    x = np.empty(L)
    for q in range(L):
        j = seq[q]
        t = r[j]
        h = q - 1
        while h >= 0:
            v = x[h] + s[seq[h], j]
            if v > t:
                t = v
            if x[h] + s_max <= t:
                break
            h -= 1
        if t > d[j] + 1e-9:
            return False, x
        x[q] = t
    return True, x


@njit(cache=True)
def _score(seq, x, delta):
    tot = 0.0
    for q in range(seq.shape[0]):
        dv = x[q] - delta[seq[q]]
        tot += -dv * dv if dv >= 0 else dv * dv
    return tot


def score_of(x: np.ndarray, delta: np.ndarray) -> float:
    dv = x - delta
    return float(np.where(dv >= 0, -dv * dv, dv * dv).sum())


class NonlinearEvaluator:
    """cost(seq) = −(runway score); inf if infeasible. Same interface as RunwayEvaluator."""

    def __init__(self, inst, cache_max: int = 400_000):
        self.r = np.asarray(inst.r, dtype=np.float64)
        self.d = np.asarray(inst.d, dtype=np.float64)
        self.delta = np.asarray(inst.delta, dtype=np.float64)
        self.s = np.ascontiguousarray(inst.s, dtype=np.float64)
        self.s_max = float(self.s.max())
        self.cache: Dict[Tuple[int, ...], float] = {}
        self.cache_max = cache_max
        self.n_lp = 0

    def times(self, seq: Sequence[int]):
        if len(seq) == 0:
            return 0.0, np.empty(0)
        a = np.asarray(seq, dtype=np.int64)
        ok, x = _earliest(a, self.r, self.d, self.s, self.s_max)
        if not ok:
            return math.inf, None
        self.n_lp += 1
        return -_score(a, x, self.delta), x

    def cost(self, seq: Sequence[int]) -> float:
        key = tuple(seq)
        c = self.cache.get(key)
        if c is None:
            c = self.times(key)[0]
            if len(self.cache) >= self.cache_max:
                self.cache.clear()
            self.cache[key] = c
        return c

    def local(self, new: List[int], old: List[int], x_old, W: int = 8):
        return self.times(new)

    def total(self, seqs: List[List[int]]) -> float:
        return sum(self.cost(sq) for sq in seqs)


def upper_bound(inst) -> float:
    dv = np.asarray(inst.r, dtype=np.float64) - np.asarray(inst.delta, dtype=np.float64)
    return float(np.where(dv >= 0, -dv * dv, dv * dv).sum())


# Best known values, non-linear objective. Sources: Faye (2015) Tables 4, 6
# (small instances proven optimal except airland7 m=1) and Pinol & Beasley
# (2006) Tables 1, 2; the larger of the two is kept.
NONLINEAR_BKS: Dict[str, Dict[int, Tuple[float, bool]]] = {
    "airland1":  {1: (4849, True), 2: (5924, True), 3: (6185, True), 4: (6237, True)},
    "airland2":  {1: (18337, True), 2: (19948, True), 3: (20078, True)},
    "airland3":  {1: (35632, True), 2: (38524, True), 3: (38664, True)},
    "airland4":  {1: (20001, True), 2: (22888, True), 3: (23659, True),
                  4: (23955, True), 5: (24140, True)},
    "airland5":  {1: (19381, True), 2: (26021, True), 3: (26495, True),
                  4: (26699, True), 5: (26732, True)},
    "airland6":  {1: (-2847013, True), 2: (-8943, True), 3: (0, True)},
    "airland7":  {1: (-23266, False), 2: (644749, True), 3: (646432, True)},
    "airland8":  {1: (728837, False), 2: (797116, False), 3: (799417, False)},
    "airland9":  {1: (10926459, False), 2: (13690376, False), 3: (14037508, False),
                  4: (14090232, False)},
    "airland10": {1: (14713752, False), 2: (19829588, False), 3: (20126522, False),
                  4: (20136384, False)},
    "airland11": {1: (21827542, False), 2: (26862823, False), 3: (27409004, False),
                  4: (27484328, False), 5: (27506007, False)},
    "airland12": {1: (27619476, False), 2: (33809839, False), 3: (34343826, False),
                  4: (34401946, False), 5: (34405915, False)},
    "airland13": {1: (45677264, False), 2: (62149293, False), 3: (63666346, False),
                  4: (63886852, False), 5: (63891976, False)},
}
