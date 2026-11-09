"""
run_doe.py — execute one stage of the design of experiments
===============================================================================
Set the CONFIGURE HERE block and run from the repository root, in the pinned
environment of the screening (see sa_single/doe/DOE_RUNBOOK.md):

    python sa_single/doe/run_doe.py

With DRY_RUN = True it shows the design, instances, run count and time estimate
and runs nothing.  The stage (run configuration) is STAGE below; every stage is
defined in stages.py:

    screening       2^(7-3) resolution IV + 4 centre points             (done)
    foldover        single-factor foldover of the screening fraction     (optional)
    ccd             face-centred CCD on A-E, F and G fixed               (next)
    anchor_tune     10x-budget runs at the recommended settings          (tune instances)
    check           optimum vs F and G moved one at a time               (tune instances)
    confirm         recommended settings on unseen instances             (holdout_large)
    anchor_holdout  10x-budget runs on the holdout instances             (budget rule)

Every (configuration, instance, seed) run uses the same fixed iteration budget
(SAParams.budget_per_aircraft x n moves per replica), so configurations are
compared at equal effort.  Runs execute one after another (each uses every
replica process) in a randomised, reproducible order and a stage can be
resumed: finished runs in records.csv are skipped.  A stage's settings are
saved in its results folder (stage.json) and a resumed run must match them (the number of seeds may grow).

Outputs (sa_single/doe/results/<out>/)
    stage.json    the run configuration actually used
    design.csv    configurations, coded and natural levels
    instances.csv the instances of the stage
    records.csv   one row per run: best cost, time to best (iterations and s), ...
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
from sa_single.doe import stages as ST
from sa_single.doe.instances import manifest
from sa_single.params import FACTOR_SETS, SAParams, decode, encode

# ═════════════════════════════════════════════════════════════════════════════
#   CONFIGURE HERE
#
#   STAGE     a key of stages.STAGES
#   N_SEEDS   None = the stage's own number of seeds
#   REPLICAS  replica processes per run (None = logical CPUs - 2; the screening used 10 on 12 logical CPUs)
#   DRY_RUN   True = print the plan and stop
# ═════════════════════════════════════════════════════════════════════════════
STAGE    = "anchor_tune"
N_SEEDS  = None
REPLICAS = None
DRY_RUN  = False
# ═════════════════════════════════════════════════════════════════════════════

RESULTS = ROOT / "sa_single" / "doe" / "results"
CACHE = ROOT / "sa_single" / "doe" / "cache"
RECOMMENDED = RESULTS / "recommended_params.json"


# ─────────────────────────────────────────────────────────────────────────────
# Stage → design, parameters
# ─────────────────────────────────────────────────────────────────────────────

_WARNED: list = []


def _recommended_base(cfg: dict) -> SAParams:
    if RECOMMENDED.exists():
        return SAParams.from_json(RECOMMENDED)
    if cfg.get("needs_recommended"):
        raise SystemExit(f"stage {cfg['out']!r} needs {RECOMMENDED} - set ANALYSIS = \"ccd\" in analyze.py and run it first")
    if not _WARNED:
        print(f"WARNING: {RECOMMENDED.name} not found, using the default (centre) settings", flush=True)
        _WARNED.append(True)
    return SAParams()


def phase_design(cfg: dict) -> pd.DataFrame:
    ph = cfg["phase"]
    if ph == "screening":
        return D.screening(4)
    if ph == "foldover":
        return D.foldover(cfg["fold_factor"])
    if ph == "ccd":
        return D.ccd(cfg["ccd_factors"], cfg.get("n_center", 6), cfg.get("ccd_fixed", {}))
    base = _recommended_base(cfg)
    coded = encode(base, cfg["coding"])
    if ph == "check":
        rows = [dict(config="X01", **coded, point="check_base")]
        for k, move in enumerate(cfg["check_moves"], start=2):
            rows.append(dict(config=f"X{k:02d}", **{**coded, **move},
                             point="check_" + "_".join(f"{c}{v:+g}" for c, v in move.items())))
        return pd.DataFrame(rows)
    if ph in ("anchor", "confirm"):
        return pd.DataFrame([dict(config="X01", **coded, point=ph)])
    raise ValueError(f"unknown phase {ph!r}")


def base_params(cfg: dict) -> SAParams:
    if cfg["phase"] in ("anchor", "confirm", "check"):
        return _recommended_base(cfg)
    return SAParams()


def params_for(row, base: SAParams, cfg: dict, replicas) -> SAParams:
    p = replace(decode({c: float(row[c]) for c in D.LETTERS}, base, cfg["coding"]), budget_a=None)   # fixed DOE budget
    if cfg["phase"] == "anchor":
        p = replace(p, budget_per_aircraft=base.budget_per_aircraft * cfg["anchor_mult"])
    return replace(p, replicas=replicas)


def starts_for(inst, R: int):
    """Starting schedules of an instance, built once with every core and cached."""
    from sa_single.search import build_starts
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{inst.name}.json"
    if path.exists():
        return [(c, s) for c, s in json.loads(path.read_text())]
    st = build_starts(inst, R, seed=0, n_workers=R)
    path.write_text(json.dumps(st))
    return st


# ─────────────────────────────────────────────────────────────────────────────
# Time estimate from the screening runs
# ─────────────────────────────────────────────────────────────────────────────

def estimate_hours(des: pd.DataFrame, insts: pd.DataFrame, n_seeds: int, cfg: dict) -> float | None:
    """Seconds per run = screening wall time of the instance at the configuration's F level (F drives the cost:
    mutation size 16 made runs ~2x slower), times the anchor multiplier.  None if the screening records are absent."""
    path = RESULTS / "screening" / "records.csv"
    if not path.exists():
        return None
    rec = pd.read_csv(path, usecols=["instance", "n", "F", "wall_s"])
    by_inst = rec.groupby(["instance", "F"])["wall_s"].mean()
    by_inst_all = rec.groupby("instance")["wall_s"].mean()
    by_n = rec.groupby(["n", "F"])["wall_s"].mean()          # for instances the screening did not run (holdout)
    by_n_all = rec.groupby("n")["wall_s"].mean()
    tot = 0.0
    for _, r in des.iterrows():
        f = float(np.sign(r["F"])) if abs(r["F"]) > 0.5 else 0.0
        for _, ir in insts.iterrows():
            if ir["instance"] in by_inst_all.index:
                tot += by_inst.get((ir["instance"], f), by_inst_all[ir["instance"]])
            else:
                tot += by_n.get((ir["n"], f), by_n_all.get(ir["n"], np.nan))
    mult = cfg.get("anchor_mult", 1) if cfg["phase"] == "anchor" else 1
    return tot * n_seeds * mult / 3600.0


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if STAGE not in ST.STAGES:
        raise SystemExit(f"unknown STAGE {STAGE!r}; choose one of {list(ST.STAGES)}")
    cfg = dict(ST.STAGES[STAGE])
    cfg["stage"] = STAGE
    n_seeds = N_SEEDS or cfg["n_seeds"]
    cfg["n_seeds"] = n_seeds
    out = RESULTS / cfg["out"]
    des = phase_design(cfg)
    base = base_params(cfg)
    insts = manifest(cfg["instance_set"], cfg.get("subset"))
    nat = pd.DataFrame([asdict(params_for(r, base, cfg, REPLICAS)) for _, r in des.iterrows()])[
        [f.name for f in FACTOR_SETS[cfg["coding"]]]]
    full = pd.concat([des, nat], axis=1)
    runs = [(c, i, s) for c in range(len(des)) for i in range(len(insts)) for s in range(n_seeds)]
    est = estimate_hours(des, insts, n_seeds, cfg)

    print(f"stage {STAGE}: {cfg.get('note', '')}")
    print(f"coding {cfg['coding']}, instances {cfg['instance_set']}"
          f" ({'all' if not cfg.get('subset') else cfg['subset']}): {len(insts)}, seeds {n_seeds}")
    print(full.to_string(index=False))
    print(f"{len(des)} configurations x {len(insts)} instances x {n_seeds} seeds = {len(runs)} runs"
          + (f", about {est:.0f} h at the screening's seconds per run" if est is not None else ""), flush=True)
    if DRY_RUN:
        return

    out.mkdir(parents=True, exist_ok=True)
    saved = out / "stage.json"
    plan = {k: v for k, v in cfg.items() if k != "note"}
    same = lambda d: {k: v for k, v in json.loads(json.dumps(d)).items() if k != "n_seeds"}   # seeds may be added on resume
    if saved.exists() and same(json.loads(saved.read_text())) != same(plan):
        raise SystemExit(f"{saved} holds a different run configuration; resume with the same settings or use a new "
                         f"results folder (compare with stages.py)")
    saved.write_text(json.dumps(plan, indent=1))
    full.to_csv(out / "design.csv", index=False)
    insts.to_csv(out / "instances.csv", index=False)

    from sa_single.doe.instances import load
    from sa_single.search import checkpoints, solve

    order = np.random.default_rng(ST.ORDER_SEED).permutation(len(runs))
    rec_path, cur_path = out / "records.csv", out / "curves.csv"
    done = set(pd.read_csv(rec_path)["run_id"]) if rec_path.exists() else set()
    R = REPLICAS or max(1, (os.cpu_count() or 2) - 2)
    print(f"{len(done)} runs already done, {R} replicas per run", flush=True)

    rec_cols = (["run_id", "phase", "config", "point"] + D.LETTERS + list(nat.columns) +
                ["instance", "n", "congestion", "windows", "seed", "budget", "iterations", "replicas",
                 "start_best", "best_cost", "ttb_iter", "ttb_s", "wall_s", "stop_reason", "stagnation_events",
                 "hot_events", "swap_rate_mean", "lp_accepts", "audit"])
    new_rec = not rec_path.exists()
    phase_tag = cfg["out"] if cfg["phase"] == "anchor" else cfg["phase"]
    with open(rec_path, "a", newline="") as fr, open(cur_path, "a", newline="") as fc:
        wr, wc = csv.writer(fr), csv.writer(fc)
        if new_rec:
            wr.writerow(rec_cols)
            wc.writerow(["run_id"] + [f"c{k:03d}" for k in range(101)])
        t_phase = time.perf_counter()
        for idx_done, k in enumerate(order):
            c, i, s = runs[k]
            row, irow = des.iloc[c], insts.iloc[i]
            run_id = f"{phase_tag}|{row['config']}|{irow['instance']}|{s}"
            if run_id in done:
                continue
            inst = load(irow)
            prm = params_for(row, base, cfg, REPLICAS)
            res = solve(inst, prm, seed=1000 * s + c, starts=starts_for(inst, R))
            swap = (res.swap_accepts / np.maximum(res.swap_attempts, 1)).mean() if res.replicas > 1 else float("nan")
            wr.writerow([run_id, phase_tag, row["config"], row["point"]] + [row[L] for L in D.LETTERS] +
                        [getattr(prm, n) for n in nat.columns] +
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
                  f"[stage {el / 3600:.2f} h]", flush=True)


if __name__ == "__main__":
    main()
