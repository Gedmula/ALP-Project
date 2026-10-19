"""
run_doe.py — execute one phase of the design of experiments
===============================================================================
Set the CONFIGURE HERE block and run from the repository root:

    python sa_single/doe/run_doe.py

Phases
    screening   2^(7−3) resolution IV + 4 centre points           (tune set)
    foldover    single-factor foldover of the screening fraction   (tune set)
    ccd         face-centred CCD on the active factors              (tune set)
    anchor      long runs (ANCHOR_MULT × budget) that strengthen the reference
                value of every instance of ANCHOR_SET           (tune or holdout)
    confirm     recommended settings on unseen instances            (holdout set)

Every (configuration, instance, seed) run uses the same fixed iteration budget
(SAParams.budget_per_aircraft · n moves per replica), so configurations are
compared at equal effort.  Runs execute one after another (each uses every
replica process) in a randomised, reproducible order and the phase can be
resumed: finished runs in records.csv are skipped.

Outputs (sa_single/doe/results/<phase>/)
    design.csv    configurations, coded and natural levels
    records.csv   one row per run: best cost, time to best (iterations and s), …
    curves.csv    global best at 101 evenly spaced fractions of the budget
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from sa_single.doe import design as D
from sa_single.doe.instances import load, manifest
from sa_single.params import FACTORS, SAParams, decode
from sa_single.search import build_starts, checkpoints, solve

# ═════════════════════════════════════════════════════════════════════════════
#   CONFIGURE HERE
#
#   PHASE            "screening" | "foldover" | "ccd" | "anchor" | "confirm"
#   N_SEEDS          independent runs per (configuration, instance)
#   FOLDOVER_FACTOR  factor whose sign is reversed in the foldover phase
#   CCD_FACTORS      active factors from the screening analysis (letters)
#   CCD_FIXED        coded levels of the inactive factors during the CCD
#   ANCHOR_MULT      budget multiplier of the anchor runs
#   ANCHOR_SET       instance set of the anchor runs: "tune" or "holdout"
#   REPLICAS         replica processes per run (None = logical CPUs − 2)
#   ORDER_SEED       seed of the randomised run order
# ═════════════════════════════════════════════════════════════════════════════
PHASE           = "screening"
N_SEEDS         = 3
FOLDOVER_FACTOR = "A"
CCD_FACTORS     = ["A", "B", "C", "E"]
CCD_FIXED       = {}
ANCHOR_MULT     = 10
ANCHOR_SET      = "tune"
REPLICAS        = None
ORDER_SEED      = 2026
# ═════════════════════════════════════════════════════════════════════════════

RESULTS = ROOT / "sa_single" / "doe" / "results"
CACHE = ROOT / "sa_single" / "doe" / "cache"
RECOMMENDED = RESULTS / "recommended_params.json"


def phase_design() -> pd.DataFrame:
    if PHASE == "screening":
        return D.screening(4)
    if PHASE == "foldover":
        return D.foldover(FOLDOVER_FACTOR)
    if PHASE == "ccd":
        return D.ccd(CCD_FACTORS, 6, CCD_FIXED)
    if PHASE in ("anchor", "confirm"):
        p = SAParams.from_json(RECOMMENDED) if RECOMMENDED.exists() else SAParams()
        coded = {f.code: f.coded(getattr(p, f.name)) for f in FACTORS}
        return pd.DataFrame([dict(config="X01", **coded, point=PHASE)])
    raise ValueError(f"unknown PHASE {PHASE!r}")


def params_for(row, base: SAParams) -> SAParams:
    p = replace(decode({c: float(row[c]) for c in D.LETTERS}, base), budget_a=None)   # fixed DOE budget
    if PHASE == "anchor":
        p = replace(p, budget_per_aircraft=base.budget_per_aircraft * ANCHOR_MULT)
    return replace(p, replicas=REPLICAS)


def starts_for(inst, R: int):
    """Starting schedules of an instance, built once with every core and cached."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{inst.name}.json"
    if path.exists():
        return [(c, s) for c, s in json.loads(path.read_text())]
    st = build_starts(inst, R, seed=0, n_workers=R)
    path.write_text(json.dumps(st))
    return st


def main() -> None:
    out = RESULTS / (f"anchor_{ANCHOR_SET}" if PHASE == "anchor" else PHASE)
    out.mkdir(parents=True, exist_ok=True)
    des = phase_design()
    base = SAParams.from_json(RECOMMENDED) if (PHASE in ("anchor", "confirm") and RECOMMENDED.exists()) else SAParams()
    nat = pd.DataFrame([asdict(params_for(r, base)) for _, r in des.iterrows()])[[f.name for f in FACTORS]]
    pd.concat([des, nat], axis=1).to_csv(out / "design.csv", index=False)

    set_name = "holdout" if PHASE == "confirm" else (ANCHOR_SET if PHASE == "anchor" else "tune")
    insts = manifest(set_name)
    n_seeds = N_SEEDS
    runs = [(c, i, s) for c in range(len(des)) for i in range(len(insts)) for s in range(n_seeds)]
    order = np.random.default_rng(ORDER_SEED).permutation(len(runs))

    rec_path, cur_path = out / "records.csv", out / "curves.csv"
    done = set(pd.read_csv(rec_path)["run_id"]) if rec_path.exists() else set()
    R = REPLICAS or max(1, (os.cpu_count() or 2) - 2)
    print(f"phase {PHASE}: {len(des)} configurations × {len(insts)} instances × {n_seeds} seeds "
          f"= {len(runs)} runs ({len(done)} already done), {R} replicas per run", flush=True)

    rec_cols = (["run_id", "phase", "config", "point"] + D.LETTERS + [f.name for f in FACTORS] +
                ["instance", "n", "congestion", "windows", "seed", "budget", "iterations", "replicas",
                 "start_best", "best_cost", "ttb_iter", "ttb_s", "wall_s", "stop_reason", "stagnation_events",
                 "hot_events", "swap_rate_mean", "lp_accepts", "audit"])
    new_rec = not rec_path.exists()
    with open(rec_path, "a", newline="") as fr, open(cur_path, "a", newline="") as fc:
        wr, wc = csv.writer(fr), csv.writer(fc)
        if new_rec:
            wr.writerow(rec_cols)
            wc.writerow(["run_id"] + [f"c{k:03d}" for k in range(101)])
        t_phase = time.perf_counter()
        for idx_done, k in enumerate(order):
            c, i, s = runs[k]
            row, irow = des.iloc[c], insts.iloc[i]
            run_id = f"{PHASE}|{row['config']}|{irow['instance']}|{s}"
            if run_id in done:
                continue
            inst = load(irow)
            prm = params_for(row, base)
            res = solve(inst, prm, seed=1000 * s + c, starts=starts_for(inst, R))
            swap = (res.swap_accepts / np.maximum(res.swap_attempts, 1)).mean() if res.replicas > 1 else float("nan")
            wr.writerow([run_id, PHASE, row["config"], row["point"]] + [row[L] for L in D.LETTERS] +
                        [getattr(prm, f.name) for f in FACTORS] +
                        [irow["instance"], irow["n"], irow["congestion"], irow["windows"], s, res.budget,
                         res.iterations, res.replicas, res.start_best, res.best_cost, res.ttb_iter,
                         f"{res.ttb_time:.4f}", f"{res.wall_time:.4f}", res.stop_reason,
                         sum(1 for _, e in res.events if e == "stagnation"),
                         sum(1 for _, e in res.events if e == "hot"), f"{swap:.4f}", res.lp_accepts,
                         "PASS" if res.audit_ok else "FAIL"])
            wc.writerow([run_id] + [f"{v:.6f}" for v in checkpoints(res)])
            fr.flush()
            fc.flush()
            el = time.perf_counter() - t_phase
            print(f"[{idx_done + 1}/{len(runs)}] {run_id}: best {res.best_cost:.2f}  ttb {res.ttb_iter} it "
                  f"({res.ttb_time:.1f}s)  wall {res.wall_time:.1f}s  audit {'PASS' if res.audit_ok else 'FAIL'}  "
                  f"[phase {el / 3600:.2f} h]", flush=True)


if __name__ == "__main__":
    main()
