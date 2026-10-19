"""
exact_eval.py — per-runway exact Stage-2 evaluation.

The joint Stage-2 LP has no inter-runway rows, so it separates:
    F(π_1..π_m) = Σ_ρ f(π_ρ),
where f(π) is the single-runway timing LP. A move touching runways a, b
changes only f(π_a) and f(π_b).

Row pruning (exact): with all consecutive rows present, the pair row
x_j - x_i ≥ s_ij (i at position a, j at position b > a+1) is implied when
    Σ_{t=a}^{b-1} s[π_t, π_{t+1}] ≥ s_ij        (chain implication), or
    r_j - d_i ≥ s_ij                               (window implication).
Chain sums are nondecreasing in b, so the scan stops once they exceed max(s).
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import highspy

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not (a and callable(a[0])) else a[0]


@njit(cache=True)
def _prune_and_propagate(seq, r, d, s, s_max):
    """Return (feasible, ia, ib) where (ia[k], ib[k]) are the kept pair rows (positions)."""
    L = seq.shape[0]
    P = np.zeros(L)
    for t in range(1, L):
        P[t] = P[t - 1] + s[seq[t - 1], seq[t]]
    cap = L * 6
    ia = np.empty(cap, np.int64)
    ib = np.empty(cap, np.int64)
    k = 0
    for a in range(L - 1):
        i = seq[a]
        for b in range(a + 1, L):
            if P[b] - P[a] >= s_max and b > a + 1:
                break
            j = seq[b]
            sij = s[i, j]
            if b > a + 1 and (P[b] - P[a] >= sij or r[j] - d[i] >= sij):
                continue
            if k == cap:
                cap *= 2
                ia2 = np.empty(cap, np.int64); ib2 = np.empty(cap, np.int64)
                ia2[:k] = ia[:k]; ib2[:k] = ib[:k]
                ia = ia2; ib = ib2
            ia[k] = a; ib[k] = b; k += 1
    # earliest-time propagation over kept rows (exact, since pruned rows are implied)
    x = np.empty(L)
    for t in range(L):
        x[t] = r[seq[t]]
    order = np.argsort(ib[:k], kind="mergesort")
    for q in order:
        a = ia[q]; b = ib[q]
        v = x[a] + s[seq[a], seq[b]]
        if v > x[b]:
            x[b] = v
    for t in range(L):
        if x[t] > d[seq[t]] + 1e-9:
            return False, ia[:k], ib[:k]
    return True, ia[:k], ib[:k]


class RunwayEvaluator:
    """Exact single-runway timing LP with row pruning and memoisation."""

    def __init__(self, inst, cache_max: int = 400_000):
        self.inst = inst
        self.r = np.asarray(inst.r, dtype=np.float64)
        self.d = np.asarray(inst.d, dtype=np.float64)
        self.delta = np.asarray(inst.delta, dtype=np.float64)
        self.g = np.asarray(inst.g, dtype=np.float64)
        self.h = np.asarray(inst.h, dtype=np.float64)
        self.s = np.ascontiguousarray(inst.s, dtype=np.float64)
        self.s_max = float(self.s.max())
        self.cache: Dict[Tuple[int, ...], float] = {}
        self.cache_max = cache_max
        self.n_lp = 0
        self._h = highspy.Highs()
        self._h.setOptionValue("output_flag", False)

    def cost(self, seq: Sequence[int]) -> float:
        key = tuple(seq)
        c = self.cache.get(key)
        if c is not None:
            return c
        c = self._solve(key)
        if len(self.cache) >= self.cache_max:
            self.cache.clear()
        self.cache[key] = c
        return c

    def times(self, seq: Sequence[int]) -> Tuple[float, np.ndarray]:
        return self._solve(tuple(seq), want_x=True)

    def _solve(self, key, want_x: bool = False):
        L = len(key)
        if L == 0:
            return (0.0, np.empty(0)) if want_x else 0.0
        seq = np.asarray(key, dtype=np.int64)
        feas, ia, ib = _prune_and_propagate(seq, self.r, self.d, self.s, self.s_max)
        if not feas:
            return (math.inf, None) if want_x else math.inf
        obj, x = self._lp(seq, self.r[seq], self.d[seq], ia, ib)
        if want_x:
            return obj, x
        return obj

    def penalties(self, seq: Sequence[int], x: np.ndarray) -> np.ndarray:
        idx = np.asarray(seq, dtype=np.int64)
        dv = self.delta[idx] - x
        return self.g[idx] * np.maximum(dv, 0.0) + self.h[idx] * np.maximum(-dv, 0.0)

    def local(self, new: List[int], old: List[int], x_old: np.ndarray, W: int = 8):
        """
        Upper bound on f(new): re-optimise only positions within W of the edit,
        holding all other landing times at x_old. Returns (cost, x_new) or (inf, None).
        """
        Ln, Lo = len(new), len(old)
        if Ln == 0:
            return 0.0, np.empty(0)
        lo = 0
        mn = min(Ln, Lo)
        while lo < mn and new[lo] == old[lo]:
            lo += 1
        suf = 0
        while suf < mn - lo and new[Ln - 1 - suf] == old[Lo - 1 - suf]:
            suf += 1
        f0 = max(0, lo - W)
        f1 = min(Ln - 1, Ln - 1 - suf + W)
        if f1 - f0 + 1 >= Ln - 2:
            c, x = self.times(new)
            return c, x
        x = np.empty(Ln)
        x[:f0] = x_old[:f0]
        shift = Lo - Ln
        x[f1 + 1:] = x_old[f1 + 1 + shift:]
        sub = np.asarray(new[f0:f1 + 1], dtype=np.int64)
        K = sub.shape[0]
        lb = self.r[sub].copy(); ub = self.d[sub].copy()
        s, sm = self.s, self.s_max
        for q in range(K):
            j = sub[q]
            a = f0 - 1
            while a >= 0:
                v = x[a] + s[new[a], j]
                if v > lb[q]:
                    lb[q] = v
                if x[a] + sm <= lb[q]:
                    break
                a -= 1
            b = f1 + 1
            while b < Ln:
                v = x[b] - s[j, new[b]]
                if v < ub[q]:
                    ub[q] = v
                if x[b] - sm >= ub[q]:
                    break
                b += 1
        if np.any(lb > ub + 1e-9):
            return math.inf, None
        feas, ia, ib = _prune_and_propagate(sub, self.r, self.d, s, sm)
        if not feas:
            return math.inf, None
        obj, xs = self._lp(sub, lb, ub, ia, ib)
        if xs is None:
            return math.inf, None
        x[f0:f1 + 1] = xs
        fixed = np.concatenate([self.penalties(new[:f0], x[:f0]),
                                self.penalties(new[f1 + 1:], x[f1 + 1:])]).sum()
        return float(fixed + obj), x

    def _lp(self, seq, lo_b, hi_b, ia, ib):
        L = seq.shape[0]
        self.n_lp += 1
        inf = highspy.kHighsInf
        K = ia.shape[0]
        nrow = L + K
        row_lo = np.empty(nrow); row_hi = np.empty(nrow)
        row_lo[:L] = self.delta[seq]; row_hi[:L] = self.delta[seq]
        row_lo[L:] = self.s[seq[ia], seq[ib]]; row_hi[L:] = inf
        starts = np.empty(nrow + 1, dtype=np.int32)
        idx = np.empty(3 * L + 2 * K, dtype=np.int32)
        val = np.empty(3 * L + 2 * K)
        ar = np.arange(L)
        starts[:L] = 3 * ar
        idx[0:3 * L:3] = ar; idx[1:3 * L:3] = L + ar; idx[2:3 * L:3] = 2 * L + ar
        val[0:3 * L:3] = 1.0; val[1:3 * L:3] = 1.0; val[2:3 * L:3] = -1.0
        off = 3 * L
        starts[L:nrow] = off + 2 * np.arange(K)
        idx[off::2] = ib; idx[off + 1::2] = ia
        val[off::2] = 1.0; val[off + 1::2] = -1.0
        starts[nrow] = 3 * L + 2 * K
        lp = highspy.HighsLp()
        lp.num_col_ = 3 * L; lp.num_row_ = nrow
        lp.col_cost_ = np.concatenate([np.zeros(L), self.g[seq], self.h[seq]])
        lp.col_lower_ = np.concatenate([lo_b, np.zeros(2 * L)])
        lp.col_upper_ = np.concatenate([hi_b, np.full(2 * L, inf)])
        lp.row_lower_ = row_lo; lp.row_upper_ = row_hi
        lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
        lp.a_matrix_.start_ = starts; lp.a_matrix_.index_ = idx; lp.a_matrix_.value_ = val
        hh = self._h
        hh.clearModel(); hh.passModel(lp); hh.run()
        if hh.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            return math.inf, None
        return hh.getInfo().objective_function_value, np.asarray(hh.getSolution().col_value[:L])

    def total(self, seqs: List[List[int]]) -> float:
        return sum(self.cost(sq) for sq in seqs)
