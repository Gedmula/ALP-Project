"""
Exact-objective multi-runway SA runner.

    python run_exact_sa.py airland9:2 airland12:2 --t 300 --workers 4

Seeds come from the existing construction portfolio. Every reported value is
re-checked with the joint Stage-2 LP and the full pairwise audit.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import csv
import io
import math
import os
import time

from mr_alp.config import KNOWN_OPTIMA
from mr_alp.construction import _build_seed_portfolio
from mr_alp.exact_sa import ms_exact_sa
from mr_alp.instance import load_instance
from mr_alp.lp import stage2_lp_objective, verify_and_exact_obj
from mr_alp.models import RBI_PARAM_BANK, HeuristicParams


def load_schedule(path: str, name: str, m: int):
    rows = collections.defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["instance"] == name and int(r["m"]) == m:
                rows[int(r["rho"])].append((int(r["position"]), int(r["aircraft_j"])))
    return [[j for _, j in sorted(rows[k])] for k in sorted(rows)] or None


def run(name: str, m: int, t_limit: float, workers: int, seed: int,
        warm: str = None, **kw):
    inst = load_instance(f"data/{name}.txt")
    params = RBI_PARAM_BANK.get((name, m), HeuristicParams())
    with contextlib.redirect_stdout(io.StringIO()):
        starts, info, seed_lps, _, _ = _build_seed_portfolio(inst, m, params, workers, seed)
    order = sorted(range(len(starts)), key=lambda i: seed_lps[i])
    starts = [starts[i][1] for i in order if not math.isinf(seed_lps[i])]
    seed_best = min(seed_lps)
    warm_seqs = load_schedule(warm, name, m) if warm else None
    if warm_seqs:
        wlp, _, wfeas, _ = stage2_lp_objective(warm_seqs, inst)
        if wfeas:
            print(f"  warm start {name} m={m}: {wlp:.2f} (seed best {seed_best:.2f})", flush=True)
            starts = [warm_seqs] * max(1, workers - 1) + starts[:1]
            seed_best = min(seed_best, wlp)
    t = time.perf_counter()
    (seqs, obj, st), allr = ms_exact_sa(inst, starts, t_limit, n_workers=workers, seed=seed, **kw)
    wall = time.perf_counter() - t
    lp, _, feas, _ = stage2_lp_objective(seqs, inst)
    ok, viol, _, _ = verify_and_exact_obj(seqs, inst)
    bks = KNOWN_OPTIMA.get(name, {}).get(m)
    gap = 100 * (lp - bks) / bks if bks else float("nan")
    print(f"{name} m={m}: seed={seed_best:.2f}  exactSA={lp:.2f}  bks={bks}  gap={gap:+.4f}%  "
          f"lp_feas={feas} audit={'PASS' if ok else 'FAIL'}  wall={wall:.0f}s  "
          f"chains={[round(r[1], 2) for r in allr]}  iters/chain~{st['iters']}", flush=True)
    return dict(instance=name, m=m, n=inst.n, seed_lp=seed_best, exact_sa_lp=lp, bks=bks,
                gap_pct=gap, feasible=feas and ok, wall_s=round(wall, 1), seqs=seqs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="+", help="instance:m, e.g. airland9:2")
    ap.add_argument("--t", type=float, default=300.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="exact_sa_results")
    ap.add_argument("--warm", default=None, help="schedules.csv to warm-start from")
    ap.add_argument("--chi0", type=float, default=0.3)
    ap.add_argument("--cycle", type=float, default=None, help="seconds per SA cycle")
    ap.add_argument("--restart-frac", type=float, default=0.3)
    ap.add_argument("--window", type=int, default=0,
                    help="re-optimise only +/-W positions around each move (0 = full runway LP)")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for job in a.jobs:
        name, m = job.split(":")
        rows.append(run(name, int(m), a.t, a.workers, a.seed, warm=a.warm, chi0=a.chi0,
                        cycle_s=a.cycle, restart_T_frac=a.restart_frac, window=a.window,
                        verbose=a.verbose))
    with open(os.path.join(a.out, "summary.csv"), "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != "seqs"] + ["t_limit"])
        if f.tell() == 0:
            w.writeheader()
        for r in rows:
            w.writerow({**{k: v for k, v in r.items() if k != "seqs"}, "t_limit": a.t})
    with open(os.path.join(a.out, "schedules.csv"), "a", newline="") as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(["instance", "m", "rho", "position", "aircraft_j"])
        for r in rows:
            for rho, sq in enumerate(r["seqs"], 1):
                for p, j in enumerate(sq, 1):
                    w.writerow([r["instance"], r["m"], rho, p, j])


if __name__ == "__main__":
    main()
