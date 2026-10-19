"""
exact_sa.py — multi-runway SA driven by the exact Stage-2 objective.

This carries over what made the single-runway solver work: every acceptance
decision uses the true LP objective, not a surrogate. It is affordable here
because F = Σ_ρ f(π_ρ) separates by runway, so a move only re-solves the one
or two runway LPs it touches (see exact_eval.py), and repeated sequences are
memoised.

Search: time-based cyclic SA. Each cycle restarts from the incumbent after an
ILS kick (cross-runway transfers + a per-runway double bridge), then cools
geometrically from T0 to T0·1e-3 over the cycle length.
"""
from __future__ import annotations

import bisect
import math
import multiprocessing as mp
import random
import time
from typing import List, Optional, Tuple

import numpy as np

from mr_alp.exact_eval import RunwayEvaluator
from mr_alp.nonlinear import NonlinearEvaluator


Seqs = List[List[int]]


def _delta_pos(seq: List[int], j: int, delta: np.ndarray) -> int:
    keys = [delta[a] for a in seq]
    return bisect.bisect_left(keys, delta[j])


def _propose(seqs: Seqs, rng: random.Random, delta: np.ndarray, m: int):
    """Return (touched_runways, {rho: new_seq}) or None."""
    u = rng.random()
    nonempty = [k for k in range(m) if seqs[k]]
    if m >= 2 and u < 0.45:
        a = rng.choice(nonempty)
        b = rng.choice([k for k in range(m) if k != a])
        sa, sb = seqs[a], seqs[b]
        p = rng.randrange(len(sa))
        j = sa[p]
        if rng.random() < 0.55:
            q = _delta_pos(sb, j, delta) + rng.randint(-2, 2)
            q = min(max(q, 0), len(sb))
            return (a, b), {a: sa[:p] + sa[p + 1:], b: sb[:q] + [j] + sb[q:]}
        if not sb:
            return None
        q = _delta_pos(sb, j, delta) + rng.randint(-2, 1)
        q = min(max(q, 0), len(sb) - 1)
        i = sb[q]
        na = sa[:]; nb = sb[:]
        na[p] = i; nb[q] = j
        return (a, b), {a: na, b: nb}
    a = rng.choice(nonempty)
    sa = seqs[a]
    L = len(sa)
    if L < 2:
        return None
    v = rng.random()
    p = rng.randrange(L)
    if v < 0.45:
        q = min(max(p + rng.choice((-4, -3, -2, -1, 1, 2, 3, 4)), 0), L - 1)
        if q == p:
            return None
        ns = sa[:p] + sa[p + 1:]
        ns.insert(q, sa[p])
    elif v < 0.85:
        q = min(max(p + rng.choice((-3, -2, -1, 1, 2, 3)), 0), L - 1)
        if q == p:
            return None
        ns = sa[:]
        ns[p], ns[q] = ns[q], ns[p]
    else:
        k = rng.choice((2, 3))
        if p + k > L:
            return None
        blk = sa[p:p + k]
        rest = sa[:p] + sa[p + k:]
        q = min(max(p + rng.randint(-4, 4), 0), len(rest))
        ns = rest[:q] + blk + rest[q:]
    return (a,), {a: ns}


def _kick(seqs: Seqs, ev: RunwayEvaluator, rng: random.Random, k: int) -> Seqs:
    m = len(seqs)
    out = [s[:] for s in seqs]
    moved = 0
    for _ in range(k * 20):
        if moved >= k:
            break
        prop = _propose(out, rng, ev.delta, m)
        if prop is None:
            continue
        touched, new = prop
        if len(touched) < 2 and m >= 2:
            continue
        if any(math.isinf(ev.cost(new[r])) for r in touched):
            continue
        for r, sq in new.items():
            out[r] = sq
        moved += 1
    return out


def exact_sa(
    inst,
    init: Seqs,
    t_limit: float,
    seed: int = 0,
    cycle_s: Optional[float] = None,
    chi0: float = 0.3,
    restart_T_frac: float = 0.3,
    window: Optional[int] = None,
    objective: str = "linear",
    verbose: bool = False,
) -> Tuple[Seqs, float, dict]:
    rng = random.Random(seed)
    ev = (NonlinearEvaluator if objective == "nonlinear" else RunwayEvaluator)(inst)
    m = len(init)
    use_local = window is not None and window > 0

    def full(sq):
        return ev.times(sq) if use_local else (ev.cost(sq), None)

    def evaluate(r, new_sq):
        if use_local:
            return ev.local(new_sq, seqs[r], xs[r], window)
        return ev.cost(new_sq), None

    def reset(sol):
        s_ = [q[:] for q in sol]
        cx = [full(q) for q in s_]
        return s_, [c for c, _ in cx], [x for _, x in cx]

    seqs, costs, xs = reset(init)
    cur = sum(costs)
    if math.isinf(cur):
        raise ValueError("initial solution infeasible")
    best, best_seqs = cur, [s[:] for s in seqs]
    t0 = time.perf_counter()
    deadline = t0 + t_limit

    pos = []
    for _ in range(300):
        prop = _propose(seqs, rng, ev.delta, m)
        if prop is None:
            continue
        touched, new = prop
        dv = sum(evaluate(r, new[r])[0] for r in touched) - sum(costs[r] for r in touched)
        if 0 < dv < math.inf:
            pos.append(dv)
    base_T0 = (float(np.median(pos)) if pos else max(cur * 0.01, 1.0)) / -math.log(chi0)
    cycle_s = cycle_s or max(10.0, t_limit / 8)

    timeline = [(0.0, best)]
    n_it = 0; n_cycles = 0
    while time.perf_counter() < deadline:
        n_cycles += 1
        c_start = time.perf_counter()
        c_len = min(cycle_s, deadline - c_start)
        T0 = base_T0 * (1.0 if n_cycles == 1 else restart_T_frac)
        T_end = T0 * 1e-3
        if n_cycles > 1:
            seqs, costs, xs = reset(
                _kick(best_seqs, ev, rng, k=rng.randint(2, max(3, inst.n // (10 * m)))))
            cur = sum(costs)
        T = T0
        t_refresh = time.perf_counter()
        while True:
            n_it += 1
            if (n_it & 63) == 0:
                now = time.perf_counter()
                el = now - c_start
                if el >= c_len:
                    break
                T = T0 * (T_end / T0) ** (el / c_len)
                if use_local and now - t_refresh > 2.0:
                    seqs, costs, xs = reset(seqs)
                    cur = sum(costs); t_refresh = now
            prop = _propose(seqs, rng, ev.delta, m)
            if prop is None:
                continue
            touched, new = prop
            newc = {r: evaluate(r, new[r]) for r in touched}
            dv = sum(c for c, _ in newc.values()) - sum(costs[r] for r in touched)
            if math.isinf(dv):
                continue
            if dv <= 0 or rng.random() < math.exp(-dv / T):
                for r in touched:
                    seqs[r] = new[r]; costs[r], xs[r] = newc[r]
                cur += dv
                if cur < best - 1e-7:
                    if use_local:
                        seqs, costs, xs = reset(seqs)
                    cur = sum(costs)
                    if cur < best - 1e-7:
                        best = cur; best_seqs = [s[:] for s in seqs]
                        timeline.append((time.perf_counter() - t0, best))
                        if verbose:
                            print(f"    [seed {seed}] t={time.perf_counter()-t0:7.1f}s  best={best:.4f}", flush=True)
    return best_seqs, best, {
        "iters": n_it, "cycles": n_cycles, "lp_solves": ev.n_lp,
        "timeline": timeline, "wall": time.perf_counter() - t0,
    }


def _worker(args):
    inst, init, t_limit, seed, kw = args
    return exact_sa(inst, init, t_limit, seed=seed, **kw)


def ms_exact_sa(inst, starts: List[Seqs], t_limit: float, n_workers: int = 4,
                seed: int = 0, **kw):
    tasks = [(inst, starts[i % len(starts)], t_limit, seed + 97 * i, kw)
             for i in range(n_workers)]
    ctx = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    with mp.get_context(ctx).Pool(n_workers) as pool:
        res = pool.map(_worker, tasks)
    return min(res, key=lambda r: r[1]), res
