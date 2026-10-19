"""
search.py — parallel-tempering simulated annealing for the single-runway ALP
===============================================================================
R replicas (one process each) hold fixed temperatures on a geometric ladder
between T_cold and T_hot.  Each round every replica runs a batch of Metropolis
moves (kernels.run_batch); the coordinator then

  1. exchanges temperatures between neighbouring replicas with probability
     min(1, exp((1/T_i − 1/T_j)(E_i − E_j)))  (even/odd pairs alternate),
  2. records every replica's best in the elite archive and the global best,
  3. reseeds the hottest replica with a mutated elite (ruin-and-recreate) when
     the global best has not improved for the stagnation window, and mutates the
     hottest replica's own state with probability hot_mutation_prob,
  4. lets the coldest replica re-price, with the exact LP, the moves that PAVA
     flagged as possibly infeasible-or-inexact but promising.

Temperatures are calibrated from the uphill moves of the best starting
schedule: T_hot accepts the median uphill cost increase with probability
hot_accept, T_cold accepts the 5th-percentile (small, refining) increase with
probability cold_accept: T = −quantile / ln(acceptance).

Stopping: the iteration budget (budget_per_aircraft · n moves per replica) or
the wall-clock limit, whichever comes first.  Known optima are never used.
"""
from __future__ import annotations

import math
import multiprocessing as mp
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from const_heuristics.alp_core import INF, Problem, audit, canonical, cet
from const_heuristics.etri import etri_pool
from const_heuristics.ramp_rbi import tc_rbi_pool
from sa_single.kernels import prices, run_batch, sample_uphill, seed_rng, span_cdf
from sa_single.mutation import ruin_recreate
from sa_single.params import SAParams

Start = Tuple[float, List[int]]


# ─────────────────────────────────────────────────────────────────────────────
# Starting schedules
# ─────────────────────────────────────────────────────────────────────────────

def build_starts(inst, k: int, seed: int = 0, n_workers: int = 1) -> List[Start]:
    """
    Distinct TC-RBI and ETRI schedules (single runway), best first.  The
    target-order sequence is added when feasible, so instances where the greedy
    constructions dead-end (tight windows) still get a start.
    """
    pool = tc_rbi_pool(inst, 1, k=(k + 1) // 2, seed=seed, n_workers=n_workers)
    pool += etri_pool(inst, 1, k=(k + 1) // 2, seed=seed, n_workers=n_workers)
    edd = [int(j) for j in np.argsort(np.asarray(inst.delta), kind="stable")]
    c_edd = Problem(inst).exact_cost(edd)
    if math.isfinite(c_edd):
        pool.append((c_edd, [edd]))
    seen, out = set(), []
    for c, sq in sorted(pool, key=lambda t: t[0]):
        key = canonical(sq)
        if key not in seen:
            seen.add(key)
            out.append((c, sq[0]))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Replica worker process
# ─────────────────────────────────────────────────────────────────────────────

class _Replica:
    """State of one replica inside its worker process."""

    def __init__(self, inst, prm: SAParams, seed: int):
        self.P = Problem(inst)
        self.prm = prm
        n = self.P.n
        self.n = n
        self.seq = np.zeros(n, np.int64)
        self.best_seq = np.zeros(n, np.int64)
        self.x = np.zeros(n)
        self.lam = np.zeros(max(n - 1, 1))
        self.clam = np.zeros(max(n - 1, 1))
        self.buf = np.zeros(n, np.int64)
        self.xbuf = np.zeros(n)
        self.flag_seqs = np.zeros((64, n), np.int64)
        self.flag_lb = np.zeros(64)
        self.dcdf = span_cdf(prm.window)
        self.cost = INF
        self.best_cost = INF
        seed_rng(seed)

    def set(self, seq) -> None:
        self.seq[:] = seq
        self._refresh(self.P.exact_cost(self.seq.tolist()))
        if self.cost < self.best_cost:
            self.best_cost = self.cost
            self.best_seq[:] = self.seq

    def _refresh(self, cost: float) -> None:
        P = self.P
        cet(self.seq, self.n, *P.A, self.x)
        prices(self.seq, self.n, self.x, P.r, P.d, P.dl, P.g, P.h, P.s, self.lam, self.clam)
        self.cost = cost

    def run(self, T: float, n_moves: int, lp_check: bool) -> Dict:
        P, prm = self.P, self.prm
        best_before = self.best_cost
        cost, best, n_acc, n_priced, it_best, n_flag = run_batch(
            self.seq, self.x, self.lam, self.clam, self.best_seq, self.cost, self.best_cost, T,
            n_moves, prm.window, prm.guide, prm.p_insert, prm.p_swap, self.dcdf,
            P.r, P.d, P.dl, P.g, P.h, P.s, P.s_max, self.buf, self.xbuf, self.flag_seqs, self.flag_lb)
        self.cost, self.best_cost = cost, best
        n_lp = 0
        if lp_check and n_flag:
            for idx in np.argsort(self.flag_lb[:n_flag], kind="stable")[:prm.lp_checks]:
                if self.flag_lb[idx] >= self.cost - 1e-9:
                    break
                c = P.exact_cost(self.flag_seqs[idx].tolist())
                if c < self.cost - 1e-9:
                    self.seq[:] = self.flag_seqs[idx]
                    self._refresh(c)
                    n_lp += 1
                    if c < self.best_cost - 1e-9:
                        self.best_cost = c
                        self.best_seq[:] = self.seq
                        it_best = n_moves - 1
                    break
        improved = self.best_cost < best_before - 1e-9
        return dict(cost=self.cost, best_cost=self.best_cost, seq=self.seq.copy(),
                    best_seq=self.best_seq.copy() if improved else None,
                    it_best=it_best if improved else -1, n_acc=n_acc, n_priced=n_priced,
                    n_flag=n_flag, n_lp=n_lp)

    def sample(self, n_samples: int) -> np.ndarray:
        P, prm = self.P, self.prm
        out = np.zeros(n_samples)
        m = sample_uphill(self.seq, self.x, self.clam, self.cost, n_samples, prm.window, prm.guide,
                          prm.p_insert, prm.p_swap, self.dcdf, P.r, P.d, P.dl, P.g, P.h, P.s, P.s_max,
                          self.buf, self.xbuf, out)
        return out[:m]


def _worker(conn, inst, prm: SAParams, seed: int) -> None:
    rep = _Replica(inst, prm, seed)
    while True:
        cmd, arg = conn.recv()
        if cmd == "set":
            rep.set(arg)
            conn.send(rep.cost)
        elif cmd == "run":
            conn.send(rep.run(*arg))
        elif cmd == "sample":
            conn.send(rep.sample(arg))
        elif cmd == "stop":
            conn.send(None)
            break


# ─────────────────────────────────────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Result:
    name: str
    n: int
    best_cost: float
    best_seq: List[int]
    best_x: np.ndarray
    ttb_iter: int                      # moves per replica when the final best was first found
    ttb_time: float                    # seconds (search only) when the final best was first found
    wall_time: float                   # search seconds (excludes building the starting schedules)
    iterations: int                    # moves per replica performed
    budget: int                        # iteration budget (moves per replica)
    replicas: int
    temperatures: np.ndarray           # ladder, coldest first
    start_best: float                  # best starting schedule
    audit_ok: bool
    audit_msgs: List[str]
    params: SAParams
    history: Dict[str, np.ndarray] = field(default_factory=dict)
    swap_attempts: np.ndarray = None   # per adjacent ladder pair
    swap_accepts: np.ndarray = None
    replica_start_cost: np.ndarray = None   # per replica, in initial ladder order (coldest first)
    replica_best_cost: np.ndarray = None
    events: List[Tuple[int, str]] = field(default_factory=list)   # (iteration, "stagnation" | "hot")
    lp_accepts: int = 0
    stop_reason: str = ""


def _diversity(archive: List[Tuple[float, tuple]], n: int) -> float:
    """Mean absolute position difference (per aircraft) over all pairs of archive members."""
    if len(archive) < 2:
        return 0.0
    pos = []
    for _, sq in archive:
        p = np.empty(n)
        p[np.asarray(sq)] = np.arange(n)
        pos.append(p)
    tot, cnt = 0.0, 0
    for a in range(len(pos)):
        for b in range(a + 1, len(pos)):
            tot += float(np.mean(np.abs(pos[a] - pos[b])))
            cnt += 1
    return tot / cnt


# ─────────────────────────────────────────────────────────────────────────────
# Parallel tempering
# ─────────────────────────────────────────────────────────────────────────────

def solve(inst, params: Optional[SAParams] = None, seed: int = 0,
          starts: Optional[List[Start]] = None, verbose: bool = False) -> Result:
    """
    Minimise the single-runway landing cost of `inst` (arrays r, delta, d, g, h, s).
    `starts`: list of (cost, sequence), best first; built with TC-RBI and ETRI if None.
    """
    prm = params or SAParams()
    R = prm.replicas or max(1, (os.cpu_count() or 2) - 2)
    P = Problem(inst)
    n = P.n
    rng = np.random.default_rng(seed)
    if starts is None:
        starts = build_starts(inst, R, seed, n_workers=R)
    starts = sorted((st for st in starts if math.isfinite(st[0])), key=lambda t: t[0])
    if not starts:
        raise ValueError(f"{getattr(inst, 'name', 'instance')}: no feasible starting schedule")
    init = [starts[i % len(starts)] for i in range(R)]       # best schedules on the coldest levels

    budget = prm.iteration_budget(n)
    batch = max(1, int(round(prm.exchange_per_aircraft * n)))
    stag_rounds = max(1, math.ceil(prm.stagnation_per_aircraft * n / batch))

    # compile the kernels once here so the forked workers inherit them
    warm = _Replica(inst, prm, seed)
    warm.set(init[0][1])
    warm.run(1.0, 10, True)
    warm.sample(5)
    del warm

    ctx = mp.get_context("fork")
    pipes, procs = [], []
    for w in range(R):
        a, b = ctx.Pipe()
        pr = ctx.Process(target=_worker, args=(b, inst, prm, seed * 1009 + w), daemon=True)
        pr.start()
        pipes.append(a)
        procs.append(pr)
    try:
        for w in range(R):
            pipes[w].send(("set", init[w][1]))
        start_cost = np.array([pipes[w].recv() for w in range(R)])

        # temperature ladder from the uphill moves of the best start (worker 0)
        pipes[0].send(("sample", prm.calib_samples))
        ups = pipes[0].recv()
        q_hot = float(np.quantile(ups, prm.hot_quantile)) if len(ups) else P.c_unit
        q_cold = float(np.quantile(ups, prm.cold_quantile)) if len(ups) else 0.1 * P.c_unit
        t_hot = -q_hot / math.log(prm.hot_accept)
        t_cold = min(-q_cold / math.log(prm.cold_accept), t_hot)
        temps = (np.array([t_cold]) if R == 1 else
                 t_cold * (t_hot / t_cold) ** (np.arange(R) / (R - 1)))
        level_of = list(range(R))          # worker -> ladder level (0 = coldest)
        worker_at = list(range(R))         # ladder level -> worker

        cur_cost = start_cost.copy()
        cur_seq = [np.asarray(init[w][1], np.int64) for w in range(R)]
        rep_best = start_cost.copy()
        g_best = float(start_cost.min())
        g_seq = list(cur_seq[int(np.argmin(start_cost))])
        archive: List[Tuple[float, tuple]] = []

        def add_archive(c: float, sq) -> None:
            key = tuple(int(v) for v in sq)
            if any(k == key for _, k in archive):
                return
            archive.append((c, key))
            archive.sort(key=lambda t: t[0])
            del archive[prm.archive_size:]

        for w in range(R):
            add_archive(float(start_cost[w]), cur_seq[w])

        ttb_iter, ttb_time = 0, 0.0
        swap_att = np.zeros(max(R - 1, 1), np.int64)
        swap_acc = np.zeros(max(R - 1, 1), np.int64)
        hist = {k: [] for k in ("iters", "time", "global_best", "archive_size", "archive_div")}
        hist["level_cost"] = []
        events: List[Tuple[int, str]] = []
        lp_accepts, stag, done, rnd = 0, 0, 0, 0
        stop_reason = "iterations"
        t0 = time.perf_counter()

        while done < budget:
            if time.perf_counter() - t0 >= prm.time_limit:
                stop_reason = "time limit"
                break
            nm = min(batch, budget - done)
            t_round = time.perf_counter()
            for w in range(R):
                pipes[w].send(("run", (float(temps[level_of[w]]), nm, level_of[w] == 0)))
            res = [pipes[w].recv() for w in range(R)]
            t_end = time.perf_counter()
            improved = False
            for w, rr in enumerate(res):
                cur_cost[w] = rr["cost"]
                cur_seq[w] = rr["seq"]
                lp_accepts += rr["n_lp"]
                if rr["best_seq"] is not None:
                    rep_best[w] = min(rep_best[w], rr["best_cost"])
                    add_archive(rr["best_cost"], rr["best_seq"])
                    if rr["best_cost"] < g_best - 1e-9:
                        g_best = rr["best_cost"]
                        g_seq = list(rr["best_seq"])
                        frac = (rr["it_best"] + 1) / nm
                        ttb_iter = done + rr["it_best"] + 1
                        ttb_time = (t_round - t0) + frac * (t_end - t_round)
                        improved = True
            done += nm
            rnd += 1
            stag = 0 if improved else stag + 1

            # replica exchange on alternating neighbour pairs
            for l in range(rnd % 2, R - 1, 2):
                wi, wj = worker_at[l], worker_at[l + 1]
                arg = (1.0 / temps[l] - 1.0 / temps[l + 1]) * (cur_cost[wi] - cur_cost[wj])
                swap_att[l] += 1
                if arg >= 0 or rng.random() < math.exp(arg):
                    swap_acc[l] += 1
                    worker_at[l], worker_at[l + 1] = wj, wi
                    level_of[wi], level_of[wj] = l + 1, l

            # diversification of the hottest replica
            w_hot = worker_at[R - 1]
            new = None
            if stag >= stag_rounds and archive:
                src = archive[int(rng.integers(len(archive)))][1]
                new = ruin_recreate(P, list(src), prm.mutation_size, rng)
                events.append((done, "stagnation"))
                stag = 0
            elif R > 1 and rng.random() < prm.hot_mutation_prob:
                new = ruin_recreate(P, cur_seq[w_hot].tolist(), prm.mutation_size, rng)
                events.append((done, "hot"))
            if new is not None:
                pipes[w_hot].send(("set", new[0]))
                cur_cost[w_hot] = pipes[w_hot].recv()
                cur_seq[w_hot] = np.asarray(new[0], np.int64)
                add_archive(float(cur_cost[w_hot]), new[0])
                if cur_cost[w_hot] < g_best - 1e-9:
                    g_best = float(cur_cost[w_hot])
                    g_seq = list(new[0])
                    ttb_iter, ttb_time = done, time.perf_counter() - t0
                    stag = 0

            hist["iters"].append(done)
            hist["time"].append(time.perf_counter() - t0)
            hist["global_best"].append(g_best)
            hist["level_cost"].append([cur_cost[worker_at[l]] for l in range(R)])
            hist["archive_size"].append(len(archive))
            hist["archive_div"].append(_diversity(archive, n))
            if verbose and rnd % 20 == 0:
                print(f"  round {rnd:5d}  iter {done:9d}/{budget}  best {g_best:.2f}  "
                      f"t {time.perf_counter() - t0:.1f}s", flush=True)
        wall = time.perf_counter() - t0
    finally:
        for p in pipes:
            try:
                p.send(("stop", None))
                p.recv()
            except (BrokenPipeError, EOFError, OSError):
                pass
        for pr in procs:
            pr.join(timeout=5)
            if pr.is_alive():
                pr.terminate()

    best_cost, x = P.times(g_seq)
    ok, msgs, _ = audit([g_seq], inst, best_cost)
    return Result(
        name=getattr(inst, "name", ""), n=n, best_cost=float(best_cost), best_seq=[int(v) for v in g_seq],
        best_x=x, ttb_iter=int(ttb_iter), ttb_time=float(ttb_time), wall_time=float(wall),
        iterations=int(done), budget=budget, replicas=R, temperatures=temps,
        start_best=float(start_cost.min()), audit_ok=ok, audit_msgs=msgs, params=prm,
        history={k: np.asarray(v) for k, v in hist.items()}, swap_attempts=swap_att, swap_accepts=swap_acc,
        replica_start_cost=start_cost, replica_best_cost=rep_best, events=events,
        lp_accepts=int(lp_accepts), stop_reason=stop_reason)


def checkpoints(result: Result, k: int = 101) -> np.ndarray:
    """Global best after each of k evenly spaced fractions of the iteration budget (0 … 1)."""
    it = result.history["iters"]
    gb = result.history["global_best"]
    out = np.empty(k)
    for i, f in enumerate(np.linspace(0.0, 1.0, k)):
        idx = np.searchsorted(it, f * result.budget, side="right") - 1
        out[i] = result.start_best if idx < 0 else gb[idx]
    return out
