"""
alp_core.py — shared kernels for the construction heuristics in const_heuristics/
===============================================================================
Used by ETRI (etri.py) and TC-RBI (ramp_rbi.py).

    cet            exact optimal timing of one runway sequence (PAVA) + exactness flag
    slot_options   priced insertion slots of an aircraft in a window around its target slot
    commit         insert an aircraft into the (m, n) runway state arrays
    Problem        instance arrays, exact runway cost (cet, full pairwise LP when flagged)
    canonical      runway-order-free schedule key
    build_pool     distinct schedules from repeated randomised constructions
    Instance       instance data;  load_instance parses an OR Library file
    REFERENCE      benchmark reference values per (instance, runway count)
    audit          independent feasibility and cost check of a schedule
    run_benchmark  OR Library benchmark table for a heuristic

Exact timing (§1).  For a runway sequence substitute y_k = x_k − S_k with
S_k = Σ_{t<k} s_{t,t+1}.  The consecutive-separation timing LP becomes isotonic
regression (y nondecreasing) with convex piecewise-linear losses and box
bounds, which the pool-adjacent-violators algorithm (PAVA) solves exactly.
That LP relaxes the full pairwise LP, so when its optimum also satisfies every
non-consecutive separation it is the exact optimum; otherwise Problem prices
the sequence with the full pairwise LP.
"""
from __future__ import annotations

import math
import multiprocessing as _mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numba import njit
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

INF = np.inf
Seqs = List[List[int]]


# ─────────────────────────────────────────────────────────────────────────────
# §1  Exact consecutive timing via PAVA
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def cet(seq, L, r, d, dl, g, h, s, s_max, x):
    """
    Optimal timing of `seq[:L]` under consecutive separations.

    Returns (cost, exact).  cost = inf if the runway is infeasible.  x[:L]
    receives landing times.  exact is True iff x also satisfies every
    non-consecutive separation, in which case cost is the full pairwise LP
    optimum (an optimal solution of a relaxation that is feasible for the
    original problem).  When exact is False, cost is a lower bound.
    """
    if L == 0:
        return 0.0, True
    S = np.empty(L)
    S[0] = 0.0
    for k in range(1, L):
        S[k] = S[k - 1] + s[seq[k - 1], seq[k]]
    # Each block [bs, be] keeps its breakpoints sorted in key/wg/wh[bs..be].
    key = np.empty(L); wg = np.empty(L); wh = np.empty(L)
    tk = np.empty(L); tg = np.empty(L); th = np.empty(L)
    bs = np.empty(L, np.int64); be = np.empty(L, np.int64)
    blo = np.empty(L); bhi = np.empty(L); by = np.empty(L)
    nb = 0
    for k in range(L):
        j = seq[k]
        lo = r[j] - S[k]; hi = d[j] - S[k]
        if lo > hi + 1e-9:
            return INF, False
        t = dl[j] - S[k]
        key[k] = t; wg[k] = g[j]; wh[k] = h[j]
        bs[nb] = k; be[nb] = k; blo[nb] = lo; bhi[nb] = hi
        by[nb] = min(max(t, lo), hi)
        nb += 1
        while nb >= 2 and by[nb - 2] > by[nb - 1] + 1e-12:
            a0 = bs[nb - 2]; a1 = be[nb - 2]; b1 = be[nb - 1]
            i1 = a0; i2 = a1 + 1; o = 0
            while i1 <= a1 and i2 <= b1:
                if key[i1] <= key[i2]:
                    tk[o] = key[i1]; tg[o] = wg[i1]; th[o] = wh[i1]; i1 += 1
                else:
                    tk[o] = key[i2]; tg[o] = wg[i2]; th[o] = wh[i2]; i2 += 1
                o += 1
            while i1 <= a1:
                tk[o] = key[i1]; tg[o] = wg[i1]; th[o] = wh[i1]; i1 += 1; o += 1
            while i2 <= b1:
                tk[o] = key[i2]; tg[o] = wg[i2]; th[o] = wh[i2]; i2 += 1; o += 1
            slope = 0.0
            for q in range(o):
                key[a0 + q] = tk[q]; wg[a0 + q] = tg[q]; wh[a0 + q] = th[q]
                slope -= tg[q]
            lo = max(blo[nb - 2], blo[nb - 1]); hi = min(bhi[nb - 2], bhi[nb - 1])
            if lo > hi + 1e-9:
                return INF, False
            # Smallest minimiser of the block's summed piecewise-linear loss.
            if slope >= 0.0:
                y = lo
            else:
                y = tk[o - 1]
                for q in range(o):
                    slope += tg[q] + th[q]
                    if slope >= 0.0:
                        y = tk[q]
                        break
            be[nb - 2] = b1; blo[nb - 2] = lo; bhi[nb - 2] = hi
            by[nb - 2] = min(max(y, lo), hi)
            nb -= 1
    cost = 0.0
    for b in range(nb):
        for k in range(bs[b], be[b] + 1):
            xv = by[b] + S[k]
            x[k] = xv
            j = seq[k]
            dv = dl[j] - xv
            if dv > 0:
                cost += g[j] * dv
            else:
                cost -= h[j] * dv
    for k in range(2, L):
        jk = seq[k]
        for a in range(k - 2, -1, -1):
            if S[k] - S[a] >= s_max:
                break
            if x[k] - x[a] < s[seq[a], jk] - 1e-7:
                return cost, False
    return cost, True


# ─────────────────────────────────────────────────────────────────────────────
# §2  Insertion slots
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def ins(src, L, j, p, out):
    for k in range(p):
        out[k] = src[k]
    out[p] = j
    for k in range(p, L):
        out[k + 1] = src[k]


@njit(cache=True)
def target_pos(seq, L, j, dl):
    """First position whose aircraft has a later target than j."""
    for k in range(L):
        if dl[seq[k]] > dl[j]:
            return k
    return L


@njit(cache=True)
def slot_options(seqs, lens, costs, j, r, d, dl, g, h, s, s_max, wb, wf):
    """
    Feasible insertion slots of j, sorted by cost increase.
    Rows: (increase, runway, position, new runway cost).
    """
    m = seqs.shape[0]; n = seqs.shape[1]
    buf = np.empty(n + 1, np.int64); x = np.empty(n + 1)
    out = np.empty((m * (min(wb, n) + min(wf, n) + 2), 4)); no = 0
    for rho in range(m):
        L = lens[rho]
        q0 = target_pos(seqs[rho], L, j, dl)
        for p in range(max(0, q0 - wb), min(L, q0 + wf) + 1):
            ins(seqs[rho], L, j, p, buf)
            c, _ = cet(buf, L + 1, r, d, dl, g, h, s, s_max, x)
            if c < INF:
                out[no, 0] = c - costs[rho]; out[no, 1] = rho; out[no, 2] = p
                out[no, 3] = c; no += 1
    o = out[:no]
    return o[np.argsort(o[:, 0], kind="mergesort")]


@njit(cache=True)
def commit(seqs, lens, costs, rho, p, j, c):
    L = lens[rho]
    for k in range(L, p, -1):
        seqs[rho, k] = seqs[rho, k - 1]
    seqs[rho, p] = j
    lens[rho] = L + 1
    costs[rho] = c


# ─────────────────────────────────────────────────────────────────────────────
# §3  Exact runway cost (PAVA fast path, full pairwise LP otherwise)
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def lp_rows(seq, r, d, s, s_max):
    """Pairwise separation rows (a, b) not implied by the consecutive chain or windows."""
    L = seq.shape[0]
    P = np.zeros(L)
    for t in range(1, L):
        P[t] = P[t - 1] + s[seq[t - 1], seq[t]]
    cap = 8 * L + 8
    ia = np.empty(cap, np.int64); ib = np.empty(cap, np.int64)
    k = 0
    for a in range(L - 1):
        for b in range(a + 1, L):
            if b > a + 1:
                if P[b] - P[a] >= s_max:
                    break
                sij = s[seq[a], seq[b]]
                if P[b] - P[a] >= sij or r[seq[b]] - d[seq[a]] >= sij:
                    continue
            if k == cap:
                cap *= 2
                ia2 = np.empty(cap, np.int64); ib2 = np.empty(cap, np.int64)
                ia2[:k] = ia[:k]; ib2[:k] = ib[:k]
                ia = ia2; ib = ib2
            ia[k] = a; ib[k] = b; k += 1
    return ia[:k], ib[:k]


class Problem:
    def __init__(self, inst):
        self.r  = np.ascontiguousarray(inst.r, dtype=np.float64)
        self.d  = np.ascontiguousarray(inst.d, dtype=np.float64)
        self.dl = np.ascontiguousarray(inst.delta, dtype=np.float64)
        self.g  = np.ascontiguousarray(inst.g, dtype=np.float64)
        self.h  = np.ascontiguousarray(inst.h, dtype=np.float64)
        self.s  = np.ascontiguousarray(inst.s, dtype=np.float64).copy()
        np.fill_diagonal(self.s, 0.0)
        self.n = len(self.r)
        self.s_max = float(self.s.max())
        off = self.s[~np.eye(self.n, dtype=bool)]
        self.s_bar = float(off[off > 0].mean()) if np.any(off > 0) else 1.0
        self.c_unit = max(float(np.mean(np.maximum(self.g, self.h))) * self.s_bar, 1e-9)
        self.A = (self.r, self.d, self.dl, self.g, self.h, self.s, self.s_max)
        self._memo: Dict[Tuple[int, ...], float] = {}

    def exact_cost(self, seq: Sequence[int]) -> float:
        """Full pairwise Stage-2 LP for one runway (cet fast path, HiGHS otherwise)."""
        key = tuple(seq)
        c = self._memo.get(key)
        if c is not None:
            return c
        arr = np.asarray(key, dtype=np.int64)
        c, ok = cet(arr, len(arr), *self.A, np.empty(max(len(arr), 1)))
        if not ok and c < INF:
            c = self._lp(arr)
        self._memo[key] = c
        return c

    def times(self, seq: Sequence[int]) -> Tuple[float, Optional[np.ndarray]]:
        """Exact cost and optimal landing times of one runway sequence ((inf, None) if infeasible)."""
        arr = np.asarray(seq, dtype=np.int64)
        x = np.empty(max(len(arr), 1))
        c, ok = cet(arr, len(arr), *self.A, x)
        if c == INF:
            return INF, None
        if ok:
            return c, x[:len(arr)].copy()
        return self._lp(arr, want_x=True)

    def _lp(self, seq: np.ndarray, want_x: bool = False):
        L = len(seq)
        ia, ib = lp_rows(seq, self.r, self.d, self.s, self.s_max)
        K = len(ia)
        # vars: x (L), E (L), T (L);  rows: x_a − x_b ≤ −s_ab;  −x − E ≤ −δ;  x − T ≤ δ
        rows = np.concatenate([np.repeat(np.arange(K), 2),
                               K + np.repeat(np.arange(L), 2),
                               K + L + np.repeat(np.arange(L), 2)])
        cols = np.concatenate([np.column_stack([ia, ib]).ravel(),
                               np.column_stack([np.arange(L), L + np.arange(L)]).ravel(),
                               np.column_stack([np.arange(L), 2 * L + np.arange(L)]).ravel()])
        vals = np.concatenate([np.tile([1.0, -1.0], K), np.full(2 * L, -1.0),
                               np.tile([1.0, -1.0], L)])
        A = csr_matrix((vals, (rows, cols)), shape=(K + 2 * L, 3 * L))
        b = np.concatenate([-self.s[seq[ia], seq[ib]], -self.dl[seq], self.dl[seq]])
        c = np.concatenate([np.zeros(L), self.g[seq], self.h[seq]])
        bounds = [(self.r[j], self.d[j]) for j in seq] + [(0, None)] * (2 * L)
        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
        if want_x:
            return (float(res.fun), res.x[:L].copy()) if res.status == 0 else (INF, None)
        return float(res.fun) if res.status == 0 else INF


# ─────────────────────────────────────────────────────────────────────────────
# §4  Pools of distinct schedules
# ─────────────────────────────────────────────────────────────────────────────

def canonical(seqs: Seqs) -> Tuple[Tuple[int, ...], ...]:
    """Runways are identical, so two schedules are the same up to runway order."""
    return tuple(sorted(tuple(sq) for sq in seqs))


# construct(inst, m, rng, params, problem) -> (seqs or None, exact cost); rng None = deterministic
Constructor = Callable[..., Tuple[Optional[Seqs], float]]


def _pool_worker(args):
    construct, inst, m, prm, seeds = args
    P = Problem(inst)
    return [construct(inst, m, np.random.default_rng(sd), prm, P) for sd in seeds]


def build_pool(construct: Constructor, inst, m: int, k: int, seed: int, params,
               include_deterministic: bool = True, max_tries: Optional[int] = None,
               n_workers: int = 1, escalate: float = 1.5,
               temperature_max: float = 10.0) -> List[Tuple[float, Seqs]]:
    """
    Up to k distinct schedules from `construct`, sorted by exact cost.

    The deterministic schedule is included first when include_deterministic.
    Randomised runs use seeds seed, seed+1, ... until k distinct schedules are
    found or max_tries (default 4k) runs have been made.  Each batch that
    returns a duplicate multiplies params.temperature by `escalate` (capped at
    temperature_max), trading some quality for diversity on instances where
    most selection steps have one clear winner.  Fewer than k schedules come
    back when the instance admits fewer.
    """
    prm = params
    max_tries = max_tries or 4 * k
    found: Dict[tuple, Tuple[float, Seqs]] = {}
    P = Problem(inst)

    def add(res) -> bool:
        """Store a new schedule; return True if it duplicates one already found."""
        sq, c = res
        if sq is None or not math.isfinite(c):
            return False
        key = canonical(sq)
        if key in found:
            return True
        found[key] = (c, sq)
        return False

    if include_deterministic:
        add(construct(inst, m, None, prm, P))
    tries, nxt = 0, seed
    while len(found) < k and tries < max_tries:
        batch = min(max_tries - tries, max(k - len(found), n_workers))
        seeds = list(range(nxt, nxt + batch)); nxt += batch; tries += batch
        if n_workers > 1:
            chunks = [(construct, inst, m, prm, seeds[i::n_workers]) for i in range(n_workers)]
            with ProcessPoolExecutor(n_workers, mp_context=_mp.get_context("fork")) as ex:
                results = [res for out in ex.map(_pool_worker, chunks) for res in out]
        else:
            results = [construct(inst, m, np.random.default_rng(sd), prm, P) for sd in seeds]
        if sum(add(res) for res in results):
            prm = replace(prm, temperature=min(prm.temperature * escalate, temperature_max))
    return sorted(found.values(), key=lambda t: t[0])[:k]


# ─────────────────────────────────────────────────────────────────────────────
# §5  Instances, reference values and the schedule audit
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Instance:
    """ALP instance; s[i, j] is the separation when i lands before j on the same runway."""
    name: str
    n: int
    r: np.ndarray
    delta: np.ndarray
    d: np.ndarray
    g: np.ndarray
    h: np.ndarray
    s: np.ndarray


def load_instance(path, name: Optional[str] = None) -> Instance:
    """
    Parse an OR Library ALP file (Beasley et al. 2000): `n freeze_time`, then per
    aircraft `appearance r delta d g h s[i,0] .. s[i,n-1]`.  Appearance and freeze
    times are not used.  The separation diagonal is set to 0.
    """
    path = Path(path)
    tok = path.read_text().split()
    n = int(tok[0])
    if len(tok) != 2 + n * (6 + n):
        raise ValueError(f"{path.name}: {len(tok)} tokens, expected {2 + n * (6 + n)}")
    rows = np.array(tok[2:], dtype=np.float64).reshape(n, 6 + n)
    s = rows[:, 6:].copy()
    np.fill_diagonal(s, 0.0)
    r, delta, d = rows[:, 1].copy(), rows[:, 2].copy(), rows[:, 3].copy()
    if np.any(r > delta + 1e-6) or np.any(delta > d + 1e-6):
        raise ValueError(f"{path.name}: requires r <= delta <= d for every aircraft")
    return Instance(name or path.stem.lower(), n, r, delta, d,
                    rows[:, 4].copy(), rows[:, 5].copy(), s)


# Reference values (instance -> {runways: value}): optimal for airland1-8 and for the
# zero-cost entries; best known (CPLEX with a time limit, Salehipour 2020) otherwise.
REFERENCE: Dict[str, Dict[int, float]] = {
    "airland1":  {1: 700.00,    2: 90.00,    3: 0.00},
    "airland2":  {1: 1480.00,   2: 210.00,   3: 0.00},
    "airland3":  {1: 820.00,    2: 60.00,    3: 0.00},
    "airland4":  {1: 2520.00,   2: 640.00,   3: 130.00,  4: 0.00},
    "airland5":  {1: 3100.00,   2: 650.00,   3: 170.00,  4: 0.00},
    "airland6":  {1: 24442.00,  2: 554.00,   3: 0.00},
    "airland7":  {1: 1550.00,   2: 0.00},
    "airland8":  {1: 1950.00,   2: 135.00,   3: 0.00},
    "airland9":  {1: 5611.70,   2: 444.10,   3: 75.75,   4: 0.00},
    "airland10": {1: 12310.70,  2: 1143.70,  3: 205.21,  4: 34.22,  5: 0.00},
    "airland11": {1: 12418.32,  2: 1330.91,  3: 253.07,  4: 54.53,  5: 0.00},
    "airland12": {1: 16152.73,  2: 1695.62,  3: 221.97,  4: 2.44,   5: 0.00},
    "airland13": {1: 37268.12,  2: 3920.39,  3: 673.85,  4: 89.95,  5: 0.00},
}


def audit(seqs: Seqs, inst, claimed_cost: Optional[float] = None,
          tol: float = 1e-6) -> Tuple[bool, List[str], float]:
    """
    Independent check of a schedule.  Returns (ok, violations, cost).

    A1  every aircraft appears exactly once;
    A2  feasibility by earliest-time propagation over ALL predecessor pairs
        (plain Python, independent of PAVA): x_k = max(r_k, max_{a<k} x_a + s_ak) ≤ d_k;
    A3  at the optimal landing times (Problem.times) every window and every
        ordered-pair separation holds, and the cost recomputed from those
        times matches `claimed_cost` (when given).
    """
    viol: List[str] = []
    seen = sorted(j for sq in seqs for j in sq)
    if seen != list(range(inst.n)):
        viol.append("A1: schedule is not a permutation of all aircraft")
    r, d, s, dl, g, h = inst.r, inst.d, inst.s, inst.delta, inst.g, inst.h
    for rho, sq in enumerate(seqs):
        x = []
        for k, j in enumerate(sq):
            t = float(r[j])
            for a in range(k):
                t = max(t, x[a] + float(s[sq[a], j]))
            x.append(t)
            if t > d[j] + tol:
                viol.append(f"A2: runway {rho + 1}: aircraft {j} earliest time {t:.4f} > d={d[j]:.4f}")
    P = Problem(inst)
    total = 0.0
    for rho, sq in enumerate(seqs):
        if not sq:
            continue
        c, x = P.times(sq)
        if x is None:
            viol.append(f"A3: runway {rho + 1}: timing LP infeasible")
            total = INF
            continue
        for k, j in enumerate(sq):
            if x[k] < r[j] - tol or x[k] > d[j] + tol:
                viol.append(f"A3: aircraft {j} lands at {x[k]:.4f} outside [{r[j]}, {d[j]}]")
            for b in range(k + 1, len(sq)):
                if x[b] - x[k] < s[j, sq[b]] - tol:
                    viol.append(f"A3: runway {rho + 1}: sep({j},{sq[b]}) {x[b] - x[k]:.4f} < {s[j, sq[b]]:.4f}")
        dev = dl[np.asarray(sq)] - x
        total += float(np.sum(g[np.asarray(sq)] * np.maximum(dev, 0) + h[np.asarray(sq)] * np.maximum(-dev, 0)))
    if claimed_cost is not None and math.isfinite(total) and abs(total - claimed_cost) > 1e-6 * max(1.0, abs(total)):
        viol.append(f"A3: recomputed cost {total:.6f} differs from claimed {claimed_cost:.6f}")
    return not viol, viol, total


# ─────────────────────────────────────────────────────────────────────────────
# §6  Benchmark driver
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(construct: Constructor, params, jobs: Optional[List[Tuple[str, int]]],
                  pool_size: int, seed: int, n_workers: int) -> None:
    """Deterministic run + pool per OR Library job, every schedule checked with audit()."""
    import time

    root = Path(__file__).resolve().parents[1]
    jobs = jobs or [(nm, m) for nm, tab in REFERENCE.items()
                    for m, v in tab.items() if m == 1 or v > 0]
    print(f"{'job':14s} {'n':>4s} {'BKS':>10s} {'det':>10s} {'gap':>7s} "
          f"{'pool best':>10s} {'gap':>7s} {'pool mean':>10s} {'gap':>7s} {'uniq':>5s} {'audit':>5s} {'t':>6s}")
    for nm, m in jobs:
        inst = load_instance(str(root / "data" / f"{nm}.txt"))
        bks = REFERENCE.get(nm, {}).get(m)
        gap = lambda v: f"{100 * (v - bks) / bks:+.2f}%" if bks else "  -"
        t = time.perf_counter()
        det_seqs, det = construct(inst, m, None, params, Problem(inst))
        pool = build_pool(construct, inst, m, pool_size, seed, params, n_workers=n_workers)
        wall = time.perf_counter() - t
        ok = all(audit(sq, inst, c)[0] for c, sq in pool + [(det, det_seqs)])
        costs = [c for c, _ in pool]
        print(f"{nm + ':' + str(m):14s} {inst.n:4d} {bks or 0:10.2f} {det:10.2f} {gap(det):>7s} "
              f"{costs[0]:10.2f} {gap(costs[0]):>7s} {np.mean(costs):10.2f} {gap(np.mean(costs)):>7s} "
              f"{len(pool):5d} {'PASS' if ok else 'FAIL':>5s} {wall:5.1f}s", flush=True)
