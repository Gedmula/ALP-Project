"""
Optimality proofs for the multi-runway linear ALP (zero inter-runway separation).

    python prove_linear.py airland10:2 --time 21600 --solver highs
    python prove_linear.py airland10:2 airland11:2 --time 21600 --solver gurobi

MIP (Beasley et al. 2000 style):
    x_j in [r_j, d_j], E_j, T_j >= 0,  E_j >= delta_j - x_j,  T_j >= x_j - delta_j
    y_jr in {0,1}, sum_r y_jr = 1
    o_ij in {0,1}  (i lands before j),  z_ij in {0,1}  (same runway), i < j
    z_ij >= y_ir + y_jr - 1
    x_j >= x_i + s_ij - M_ij (2 - o_ij - z_ij),   M_ij = d_i + s_ij - r_j
    x_i >= x_j + s_ji - M_ji (1 + o_ij - z_ij)
Reductions: a direction with d_i + s_ij <= r_j is implied by the windows; a pair
with d_i < r_j has o_ij = 1 fixed. Runways are identical, so aircraft k (in
index order) may only use runways 0..k.

The best solution from our heuristic runs is passed as a MIP start, so the
incumbent is the best known value and the run is about raising the lower bound.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import time
from collections import defaultdict

import numpy as np

from mr_alp.config import KNOWN_OPTIMA
from mr_alp.exact_eval import RunwayEvaluator
from mr_alp.instance import load_instance
from mr_alp.lp import stage2_lp_objective

SCHEDULE_FILES = ["exact_sa_bench/schedules.csv", "exact_sa_bench_a13m2_900s/schedules.csv",
                  "exact_sa_results/schedules.csv", "MR_results/schedules.csv"]


def best_known_schedule(inst, name, m):
    """Best feasible schedule for (name, m) across saved result files."""
    best, best_seqs = math.inf, None
    for path in SCHEDULE_FILES:
        if not os.path.exists(path):
            continue
        blocks, cur = [], None
        with open(path) as f:
            for r in csv.DictReader(f):
                if r["instance"] != name or int(r["m"]) != m:
                    cur = None
                    continue
                rho, pos, j = int(r["rho"]), int(r["position"]), int(r["aircraft_j"])
                if cur is None or (rho == 1 and pos == 1):
                    cur = defaultdict(list)
                    blocks.append(cur)
                cur[rho].append((pos, j))
        for b in blocks:
            seqs = [[j for _, j in sorted(b[k])] for k in sorted(b)]
            seqs += [[] for _ in range(m - len(seqs))]
            if sorted(j for s in seqs for j in s) != list(range(inst.n)):
                continue
            obj, _, feas, _ = stage2_lp_objective(seqs, inst)
            if feas and obj < best:
                best, best_seqs = obj, seqs
    return best, best_seqs


class Model:
    """Backend-neutral container: columns, rows, integrality, objective."""

    def __init__(self):
        self.lb, self.ub, self.cost, self.integer, self.names = [], [], [], [], []
        self.rows = []  # (lo, hi, [(col, coef)], lazy)
        self.n_tri = 0

    def var(self, lb, ub, cost=0.0, integer=False, name=""):
        self.lb.append(lb); self.ub.append(ub); self.cost.append(cost)
        self.integer.append(integer); self.names.append(name)
        return len(self.lb) - 1

    def row(self, lo, hi, terms, lazy=False):
        self.rows.append((lo, hi, terms, lazy))


def build(inst, m, ub=math.inf, cuts=(), tri_k=0):
    n = inst.n
    r, d, dl = inst.r.astype(float), inst.d.astype(float), inst.delta.astype(float)
    if ub < math.inf:
        # each aircraft's penalty is at most the total, so any solution with cost <= ub
        # keeps x_j within ub/g_j before and ub/h_j after its target
        g, h = inst.g.astype(float), inst.h.astype(float)
        with np.errstate(divide="ignore"):
            r = np.maximum(r, np.where(g > 0, dl - ub / g, r))
            d = np.minimum(d, np.where(h > 0, dl + ub / h, d))
    s = inst.s.astype(float)
    M = Model()
    X = [M.var(r[j], d[j], name=f"x{j}") for j in range(n)]
    E = [M.var(0, math.inf, float(inst.g[j]), name=f"E{j}") for j in range(n)]
    T = [M.var(0, math.inf, float(inst.h[j]), name=f"T{j}") for j in range(n)]
    for j in range(n):
        M.row(dl[j], math.inf, [(X[j], 1), (E[j], 1)])
        M.row(-math.inf, dl[j], [(X[j], 1), (T[j], -1)])
    order = np.argsort(r, kind="stable")
    rank = np.empty(n, int); rank[order] = np.arange(n)
    Y = {}
    for j in range(n):
        allowed = range(min(m, rank[j] + 1))
        for k in range(m):
            Y[j, k] = M.var(0, 1 if k in allowed else 0, integer=True, name=f"y{j}_{k}")
        M.row(1, 1, [(Y[j, k], 1) for k in range(m)])
    O, Z, n_pairs = {}, {}, 0
    for i in range(n):
        for j in range(i + 1, n):
            need_ij = d[i] + s[i, j] > r[j]
            need_ji = d[j] + s[j, i] > r[i]
            if not (need_ij or need_ji):
                continue
            n_pairs += 1
            lo_o = 1 if d[i] < r[j] else 0
            hi_o = 0 if d[j] < r[i] else 1
            o = M.var(lo_o, hi_o, integer=True, name=f"o{i}_{j}")
            z = M.var(0, 1, integer=True, name=f"z{i}_{j}")
            O[i, j], Z[i, j] = o, z
            for k in range(m):
                M.row(-1, math.inf, [(z, 1), (Y[i, k], -1), (Y[j, k], -1)])
            if need_ij and hi_o == 1:
                Mij = d[i] + s[i, j] - r[j]
                # x_j - x_i - Mij*o - Mij*z >= s_ij - 2 Mij
                M.row(s[i, j] - 2 * Mij, math.inf,
                      [(X[j], 1), (X[i], -1), (o, -Mij), (z, -Mij)])
            if need_ji and lo_o == 0:
                Mji = d[j] + s[j, i] - r[i]
                # x_i - x_j + Mji*o - Mji*z >= s_ji - Mji
                M.row(s[j, i] - Mji, math.inf,
                      [(X[i], 1), (X[j], -1), (o, Mji), (z, -Mji)])
    # block cuts: sum_{j in S} (g_j E_j + h_j T_j) >= LB(S)
    for S, lb in cuts:
        M.row(lb, math.inf, [(E[j], float(inst.g[j])) for j in S] +
                            [(T[j], float(inst.h[j])) for j in S])
    # transitivity on a runway: no cycle a->b->c->a among aircraft sharing a runway
    #   bef(a,b) + bef(b,c) + bef(c,a) <= 2 + (3 - z_ab - z_bc - z_ac)
    if tri_k > 0 and m >= 1:
        def bef(a, b):
            return ([(O[a, b], 1.0)], 0.0) if a < b else ([(O[b, a], -1.0)], 1.0)
        def zz(a, b):
            return Z[min(a, b), max(a, b)]
        byT = np.argsort(dl, kind="stable")
        n_tri = 0
        for p in range(n):
            for q in range(p + 1, min(n, p + tri_k)):
                for w in range(q + 1, min(n, p + tri_k)):
                    a, b, c = int(byT[p]), int(byT[q]), int(byT[w])
                    pairs = [(min(u, v), max(u, v)) for u, v in ((a, b), (b, c), (a, c))]
                    if any(pr not in O for pr in pairs):
                        continue
                    for (u, v, w2) in ((a, b, c), (a, c, b)):
                        terms, const = [], 0.0
                        for x1, x2 in ((u, v), (v, w2), (w2, u)):
                            tt, cc = bef(x1, x2); terms += tt; const += cc
                        terms += [(zz(a, b), 1.0), (zz(b, c), 1.0), (zz(a, c), 1.0)]
                        M.row(-math.inf, 5.0 - const, terms, lazy=True)
                        n_tri += 1
        M.n_tri = n_tri
    return M, X, Y, O, Z, n_pairs


class SubInstance:
    def __init__(self, inst, idx):
        self.n = len(idx)
        self.r, self.d, self.delta = inst.r[idx], inst.d[idx], inst.delta[idx]
        self.g, self.h = inst.g[idx], inst.h[idx]
        self.s = inst.s[np.ix_(idx, idx)]


def block_cuts(inst, m, ub, solve, block, t_block, threads):
    """Sliding blocks of `block` aircraft by target time; each block's proven
    lower bound bounds that block's share of the cost in any solution."""
    order = np.argsort(inst.delta, kind="stable")
    stride = max(1, block // 2)
    cuts, starts = [], list(range(0, max(1, inst.n - block + 1), stride))
    if starts[-1] + block < inst.n:
        starts.append(inst.n - block)
    for st in starts:
        idx = np.sort(order[st:st + block])
        sub = SubInstance(inst, idx)
        Ms, *_ = build(sub, m, ub)
        _, obj, bound, _ = solve(Ms, None, t_block, threads, None, quiet=True)
        lb = bound if (bound is not None and math.isfinite(bound)) else 0.0
        if lb > 1e-6:
            cuts.append((list(map(int, idx)), lb * (1 - 1e-9)))
    return cuts


def start_vector(M, X, Y, O, Z, inst, seqs):
    ev = RunwayEvaluator(inst)
    x = np.zeros(inst.n); rwy = np.zeros(inst.n, int); pos = np.zeros(inst.n, int)
    for k, sq in enumerate(seqs):
        if sq:
            _, xs = ev.times(sq)
            x[sq] = xs
        for p, j in enumerate(sq):
            rwy[j], pos[j] = k, p
    # relabel runways by first aircraft in r-order to satisfy the symmetry cut
    order = np.argsort(inst.r, kind="stable")
    relabel, nxt = {}, 0
    for j in order:
        if rwy[j] not in relabel:
            relabel[rwy[j]] = nxt; nxt += 1
    rwy = np.array([relabel[k] for k in rwy])
    v = np.zeros(len(M.lb))
    for j in range(inst.n):
        v[X[j]] = x[j]
        v[M.names.index(f"E{j}")] = max(inst.delta[j] - x[j], 0)
        v[M.names.index(f"T{j}")] = max(x[j] - inst.delta[j], 0)
        v[Y[j, rwy[j]]] = 1
    for (i, j), o in O.items():
        same = rwy[i] == rwy[j]
        v[Z[i, j]] = 1 if same else 0
        before = (pos[i] < pos[j]) if same else (x[i] <= x[j])
        lo, hi = M.lb[o], M.ub[o]
        v[o] = min(max(1 if before else 0, lo), hi)
    return v


def solve_highs(M, start, t_limit, threads, log, quiet=False):
    import highspy
    highspy.Highs.resetGlobalScheduler(True)
    h = highspy.Highs()
    h.setOptionValue("output_flag", not quiet)
    h.setOptionValue("log_to_console", not quiet)
    h.setOptionValue("time_limit", float(t_limit))
    h.setOptionValue("threads", int(threads))
    h.setOptionValue("mip_rel_gap", 1e-9)
    if log:
        h.setOptionValue("log_file", log)
    inf = highspy.kHighsInf
    nc = len(M.lb)
    fix = lambda v: inf if v == math.inf else (-inf if v == -math.inf else v)
    h.addVars(nc, np.array([fix(v) for v in M.lb]), np.array([fix(v) for v in M.ub]))
    h.changeColsCost(nc, np.arange(nc, dtype=np.int32), np.array(M.cost))
    ints = np.array([i for i, b in enumerate(M.integer) if b], dtype=np.int32)
    h.changeColsIntegrality(len(ints), ints,
                            np.array([highspy.HighsVarType.kInteger] * len(ints)))
    for lo, hi, terms, _lazy in M.rows:
        idx = np.array([c for c, _ in terms], dtype=np.int32)
        val = np.array([a for _, a in terms], dtype=float)
        h.addRow(fix(lo), fix(hi), len(idx), idx, val)
    if start is not None:
        sol = highspy.HighsSolution()
        sol.col_value = list(start)
        sol.value_valid = True
        h.setSolution(sol)
    h.run()
    info = h.getInfo()
    return (str(h.getModelStatus()), info.objective_function_value,
            info.mip_dual_bound, info.mip_gap)


def solve_gurobi(M, start, t_limit, threads, log, quiet=False):
    import gurobipy as gp
    g = gp.Model()
    g.Params.OutputFlag = 0 if quiet else 1
    g.Params.TimeLimit = t_limit
    g.Params.Threads = threads
    g.Params.MIPGap = 1e-9
    if log:
        g.Params.LogFile = log
    v = [g.addVar(lb=M.lb[i], ub=M.ub[i], obj=M.cost[i],
                  vtype=gp.GRB.BINARY if M.integer[i] else gp.GRB.CONTINUOUS)
         for i in range(len(M.lb))]
    for lo, hi, terms, lazy in M.rows:
        e = gp.quicksum(a * v[c] for c, a in terms)
        cons = []
        if lo == hi:
            cons.append(g.addConstr(e == lo))
        else:
            if lo != -math.inf:
                cons.append(g.addConstr(e >= lo))
            if hi != math.inf:
                cons.append(g.addConstr(e <= hi))
        if lazy:
            for c in cons:
                c.Lazy = 1
    if start is not None:
        for var, val in zip(v, start):
            var.Start = val
    g.optimize()
    status = {2: "Optimal", 9: "TimeLimit"}.get(g.Status, str(g.Status))
    return status, g.ObjVal if g.SolCount else math.inf, g.ObjBound, g.MIPGap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="+", help="instance:m, e.g. airland10:2")
    ap.add_argument("--time", type=float, default=21600, help="seconds per job")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--solver", choices=["highs", "gurobi"], default="highs")
    ap.add_argument("--no-start", action="store_true", help="skip the MIP start")
    ap.add_argument("--no-tighten", dest="tighten", action="store_false",
                    help="keep the original time windows")
    ap.add_argument("--block", type=int, default=0,
                    help="add block lower-bound cuts with blocks of this many aircraft (e.g. 30)")
    ap.add_argument("--block-time", type=float, default=60, help="seconds per block solve")
    ap.add_argument("--tri", type=int, default=0,
                    help="add transitivity cuts among aircraft within this many target-time neighbours (e.g. 12)")
    ap.add_argument("--out", default="proofs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for job in a.jobs:
        name, m = job.split(":"); m = int(m)
        inst = load_instance(f"data/{name}.txt")
        t0 = time.perf_counter()
        ub, seqs = (math.inf, None) if a.no_start else best_known_schedule(inst, name, m)
        ub_t = ub if a.tighten else math.inf
        solve = solve_highs if a.solver == "highs" else solve_gurobi
        cuts = []
        if a.block > 0:
            tb = time.perf_counter()
            cuts = block_cuts(inst, m, ub_t, solve, a.block, a.block_time, a.threads)
            print(f"{name} m={m}: {len(cuts)} block cuts, sum of disjoint-block bounds "
                  f"~{sum(lb for _, lb in cuts[::2]):.2f}, in {time.perf_counter()-tb:.0f}s", flush=True)
        M, X, Y, O, Z, n_pairs = build(inst, m, ub_t, cuts, a.tri)
        start = start_vector(M, X, Y, O, Z, inst, seqs) if seqs else None
        print(f"{name} m={m}: {len(M.lb)} cols, {len(M.rows)} rows, {n_pairs} pairs, "
              f"{M.n_tri} transitivity cuts, start={ub:.2f}, "
              f"built in {time.perf_counter()-t0:.1f}s", flush=True)
        tag = (f"_b{a.block}" if a.block else "") + (f"_t{a.tri}" if a.tri else "")
        log = os.path.join(a.out, f"{name}_m{m}_{a.solver}{tag}.log")
        status, obj, bound, gap = solve(M, start, a.time, a.threads, log)
        wall = time.perf_counter() - t0
        bks = KNOWN_OPTIMA.get(name, {}).get(m)
        proven = gap is not None and gap <= 1e-6
        print(f"RESULT {name} m={m}: status={status} best={obj:.4f} bound={bound:.4f} "
              f"gap={100*gap:.4f}% bks={bks} proven_optimal={proven} wall={wall:.0f}s", flush=True)
        path = os.path.join(a.out, "summary_v2.csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["instance", "m", "solver", "status", "best", "bound", "gap_pct",
                            "bks", "proven_optimal", "time_limit", "wall_s", "start",
                            "block", "tri"])
            w.writerow([name, m, a.solver, status, obj, bound, 100 * gap, bks, proven,
                        a.time, round(wall, 1), ub, a.block, a.tri])


if __name__ == "__main__":
    main()
