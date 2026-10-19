"""
run.py — solve OR Library single-runway instances with the parallel-tempering SA
===============================================================================
Set the CONFIGURE HERE block and run from the repository root:

    python sa_single/run.py

Outputs (OUT_DIR):
    results.csv     one row per (instance, seed): best cost, gap to reference,
                    time to best (iterations and seconds), wall time, audit
    schedules.csv   landing order and times of the best schedule per instance
    plots/<instance>/  gantt, convergence, seed improvement, penalty profile,
                       ladder, archive        (best seed of each instance)
    plots/gap_summary.png
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from const_heuristics.alp_core import REFERENCE, load_instance
from sa_single import plots
from sa_single.params import SAParams
from sa_single.search import build_starts, solve

# ═════════════════════════════════════════════════════════════════════════════
#   CONFIGURE HERE
#
#   INSTANCES    OR Library instance names (files in data/)
#   SEEDS        independent runs per instance
#   PARAMS_FILE  SA settings recommended by the DOE (sa_single/doe/analyze.py);
#                the centre-point defaults of SAParams are used if it is missing
#   TIME_LIMIT   wall-clock cap per run, seconds
#   OUT_DIR      results folder
# ═════════════════════════════════════════════════════════════════════════════
INSTANCES   = [f"airland{i}" for i in range(1, 14)]
SEEDS       = [0]
PARAMS_FILE = ROOT / "sa_single" / "doe" / "results" / "recommended_params.json"
TIME_LIMIT  = 3600.0
OUT_DIR     = ROOT / "sa_single" / "results"
# ═════════════════════════════════════════════════════════════════════════════


def main() -> None:
    prm = SAParams.from_json(PARAMS_FILE) if Path(PARAMS_FILE).exists() else SAParams()
    prm.time_limit = TIME_LIMIT
    print(f"parameters: {prm}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res_path, sch_path = OUT_DIR / "results.csv", OUT_DIR / "schedules.csv"
    summary = []
    with open(res_path, "w", newline="") as fr, open(sch_path, "w", newline="") as fs:
        wr, ws = csv.writer(fr), csv.writer(fs)
        wr.writerow(["instance", "n", "seed", "reference", "start_best", "best", "gap_pct", "ttb_iter",
                     "ttb_s", "iterations", "budget", "wall_s", "starts_s", "replicas", "stop_reason", "audit"])
        ws.writerow(["instance", "seed", "position", "aircraft", "r", "delta", "d", "x",
                     "earliness", "tardiness", "penalty"])
        for name in INSTANCES:
            inst = load_instance(ROOT / "data" / f"{name}.txt")
            ref = REFERENCE.get(name, {}).get(1)
            best_run = None
            for seed in SEEDS:
                t = time.perf_counter()
                R = prm.replicas or max(1, (os.cpu_count() or 2) - 2)
                starts = build_starts(inst, R, seed=seed, n_workers=R)
                t_starts = time.perf_counter() - t
                res = solve(inst, prm, seed=seed, starts=starts)
                gap = 100 * (res.best_cost - ref) / ref if ref else float("nan")
                wr.writerow([name, inst.n, seed, ref, f"{res.start_best:.4f}", f"{res.best_cost:.4f}",
                             f"{gap:.4f}", res.ttb_iter, f"{res.ttb_time:.2f}", res.iterations, res.budget,
                             f"{res.wall_time:.2f}", f"{t_starts:.2f}", res.replicas, res.stop_reason,
                             "PASS" if res.audit_ok else "FAIL"])
                fr.flush()
                print(f"{name} seed {seed}: best {res.best_cost:.2f}  ref {ref}  gap {gap:+.4f}%  "
                      f"ttb {res.ttb_time:.1f}s  wall {res.wall_time:.1f}s  audit {'PASS' if res.audit_ok else 'FAIL'}",
                      flush=True)
                if best_run is None or res.best_cost < best_run[1].best_cost:
                    best_run = (seed, res)
            seed, res = best_run
            for q, j in enumerate(res.best_seq):
                xq = res.best_x[q]
                e, tdy = max(inst.delta[j] - xq, 0.0), max(xq - inst.delta[j], 0.0)
                ws.writerow([name, seed, q, j, inst.r[j], inst.delta[j], inst.d[j], f"{xq:.4f}",
                             f"{e:.4f}", f"{tdy:.4f}", f"{inst.g[j] * e + inst.h[j] * tdy:.4f}"])
            pdir = OUT_DIR / "plots" / name
            plots.gantt(inst, res.best_seq, res.best_x, pdir / "gantt.png", f"Landing schedule — {name}")
            plots.convergence(res, pdir / "convergence.png", reference=ref)
            plots.seed_improvement(res, pdir / "seed_improvement.png")
            plots.penalty_profile(inst, res.best_seq, res.best_x, pdir / "penalty_profile.png",
                                  f"Penalty profile — {name}: {res.best_cost:.2f}")
            plots.ladder(res, pdir / "ladder.png")
            plots.archive(res, pdir / "archive.png")
            if ref is not None:
                summary.append(dict(instance=name, best=res.best_cost, reference=ref))
    if summary:
        plots.gap_summary(summary, OUT_DIR / "plots" / "gap_summary.png")
    print(f"results: {res_path}")


if __name__ == "__main__":
    main()
