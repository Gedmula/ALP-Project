"""
doe_experiments.py
==================
Design of Experiments for the Aircraft Landing Problem (ALP) Solver.

Three experiments
-----------------
1. Heuristic DOE  : how each initial heuristic (ERD / EDD / MDD / ATC_k2 /
                    ATC_k4 / MPDS) influences SA convergence speed and final
                    solution quality across all 13 OR-Library instances.

2. ILS DOE        : how the number of ILS restarts (n_ils) affects solution
                    quality and wall-clock time on instances with ≥ 50 aircraft.

3. Parameter DOE  : One-Factor-At-a-Time (OFAT) sweep + Latin Hypercube Sample
                    (LHS) over SA hyperparameters (α, N_iter, M_stag, I_max,
                    chi0) on representative small / medium / large instances,
                    with a Pearson-correlation sensitivity ranking at the end.

Usage
-----
    python doe_experiments.py               # run all three experiments
    python doe_experiments.py --exp 1       # heuristic experiment only
    python doe_experiments.py --exp 2       # ILS experiment only
    python doe_experiments.py --exp 3       # parameter experiment only
    python doe_experiments.py --data-dir ./data --out-dir ./doe_results
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False
    print("WARNING: pandas not available — results written as plain CSV via numpy.")

# ---------------------------------------------------------------------------
# Import from the main solver
# ---------------------------------------------------------------------------
from Single_runway_SA import (
    ALPInstance,
    SAParams,
    adaptive_params,
    evaluate,
    evaluate_semi,
    gen_atc,
    gen_edd,
    gen_erd,
    gen_mdd,
    gen_mpds,
    load_orlib,
    run_ils,
    run_sa,
)

warnings.filterwarnings("ignore")


# ===========================================================================
# GLOBAL CONFIGURATION
# ===========================================================================

# Known optima from Zhang et al. (2020) for the 13 OR-Library benchmarks
KNOWN_OPTIMA: Dict[str, float] = {
    "airland1":  700.0,
    "airland2":  1480.0,
    "airland3":  820.0,
    "airland4":  2520.0,
    "airland5":  3100.0,
    "airland6":  24442.0,
    "airland7":  1550.0,
    "airland8":  1950.0,
    "airland9":  1837.14,
    "airland10": 16026.0,
    "airland11": 16448.0,
    "airland12": 25479.0,
    "airland13": 39287.52,
}

# Heuristics tested in Experiment 1 — label → callable(inst) → sequence
HEURISTICS: Dict[str, callable] = {
    "ERD":    lambda inst: gen_erd(inst),
    "EDD":    lambda inst: gen_edd(inst),
    "MDD":    lambda inst: gen_mdd(inst),
    "ATC_k2": lambda inst: gen_atc(inst, K=2.0),
    "ATC_k4": lambda inst: gen_atc(inst, K=4.0),
    "MPDS":   lambda inst: gen_mpds(inst),
}

# ILS restart levels tested in Experiment 2
ILS_LEVELS: List[int] = [0, 1, 2, 3, 5]

# SA hyperparameter search bounds for Experiment 3
PARAM_SPACE: Dict[str, Tuple[float, float]] = {
    "alpha":  (0.90,  0.999),
    "N_iter": (40,    400),
    "M_stag": (20,    200),
    "I_max":  (100,   1500),
    "chi0":   (0.25,  0.80),
}

# OFAT levels — 5 values per parameter, straddling the adaptive default
OFAT_LEVELS: Dict[str, List] = {
    "alpha":  [0.90, 0.93, 0.97, 0.99, 0.999],
    "N_iter": [40,   80,   150,  250,  400],
    "M_stag": [20,   40,   80,   120,  200],
    "I_max":  [100,  300,  600,  900,  1500],
    "chi0":   [0.25, 0.35, 0.50, 0.65, 0.80],
}

# Number of LHS points for Experiment 3b
N_LHS: int = 50

# Repetitions per (instance, n_ils) combination in Experiment 2
N_REPS: int = 3

# Threshold for instances considered "large" in Experiment 2
LARGE_N: int = 50

# Base random seed — all experiments derive per-run seeds from this
BASE_SEED: int = 42

# Output directory roots (can be overridden via CLI)
DATA_DIR:  Path = Path("data")
DOE_DIR:   Path = Path("doe_results")
EXP1_DIR:  Path = DOE_DIR / "exp1_heuristic"
EXP2_DIR:  Path = DOE_DIR / "exp2_ils"
EXP3_DIR:  Path = DOE_DIR / "exp3_params"


# ===========================================================================
# SHARED UTILITIES
# ===========================================================================

def _load_all_instances() -> List[ALPInstance]:
    """Load all airland{1..13}.txt benchmarks found in DATA_DIR."""
    instances = []
    for i in range(1, 14):
        path = DATA_DIR / f"airland{i}.txt"
        if path.exists():
            inst = load_orlib(str(path), f"airland{i}")
            instances.append(inst)
    if not instances:
        raise FileNotFoundError(
            f"No airland*.txt files found in '{DATA_DIR}'. "
            "Pass --data-dir to specify the correct directory."
        )
    return instances


def _gap_pct(obj: float, inst_name: str) -> float:
    """Percentage gap to known optimum; NaN if optimum unknown."""
    opt = KNOWN_OPTIMA.get(inst_name)
    if opt is None or opt == 0.0:
        return float("nan")
    return 100.0 * (obj - opt) / opt


def _best_seq(inst: ALPInstance) -> List[int]:
    """
    Return the best feasible heuristic sequence for inst.
    Tries EDD first; falls back through MDD → ERD.
    """
    for fn in (gen_edd, gen_mdd, gen_erd):
        try:
            seq = fn(inst)
            if evaluate(seq, inst) < float("inf"):
                return seq
        except Exception:
            pass
    # Last resort: identity permutation
    return list(range(inst.n))


def _run_single_sa(seq0: List[int], inst: ALPInstance,
                   p: SAParams, seed: int = 0) -> Dict:
    """
    Run one SA chain from seq0 and return a normalised result dict:
        init_obj      float   — objective of the heuristic seed
        final_obj     float   — best fully-feasible objective found
        history       list    — best semi-feasible obj per outer iteration
        alpha_history list    — reactive α per outer iteration
        t_best        float   — wall-clock seconds to best fully-feasible
        wall_time     float   — total wall-clock seconds
        n_alt_seqs    int     — alternate near-optimal sequences found
    """
    t0 = time.perf_counter()
    pb_semi, fb_semi, stats = run_sa(seq0, inst, p, seed=seed)
    wall = time.perf_counter() - t0

    obj_feas = stats.get("obj_feas", float("inf"))
    if obj_feas is None or obj_feas >= 1e18:
        obj_feas = stats.get("obj", float("inf"))

    ttb = stats.get("t_best_feas", None)
    if ttb is None:
        ttb = stats.get("t_best", float("nan"))

    return {
        "init_obj":      stats.get("init_obj", float("nan")),
        "final_obj":     obj_feas,
        "history":       stats.get("history", []),
        "alpha_history": stats.get("alpha_history", []),
        "t_best":        float(ttb) if ttb is not None else float("nan"),
        "wall_time":     wall,
        "n_alt_seqs":    stats.get("n_alt_seqs", 0),
    }


def _make_params(base: SAParams, overrides: Dict) -> SAParams:
    """Return a new SAParams with selected fields overridden."""
    return SAParams(
        alpha=overrides.get("alpha",  base.alpha),
        N_iter=int(round(overrides.get("N_iter", base.N_iter))),
        T_min=overrides.get("T_min",  base.T_min),
        I_max=int(round(overrides.get("I_max",  base.I_max))),
        M_stag=int(round(overrides.get("M_stag", base.M_stag))),
        chi0=overrides.get("chi0",   base.chi0),
    )


def _lhs_sample(n_points: int, bounds: Dict[str, Tuple],
                seed: int = 0) -> List[Dict]:
    """
    Generate n_points Latin Hypercube samples over the given parameter
    bounds dict.  Integer-typed parameters (N_iter, M_stag, I_max) are
    rounded to the nearest integer.
    """
    rng = np.random.default_rng(seed)
    keys = list(bounds.keys())
    k = len(keys)

    # Stratified LHS: divide [0,1]^k into n_points equal strata per axis
    lhs = np.zeros((n_points, k))
    for j in range(k):
        perm = rng.permutation(n_points)
        lhs[:, j] = (perm + rng.uniform(size=n_points)) / n_points

    int_keys = {"N_iter", "M_stag", "I_max"}
    samples = []
    for row in lhs:
        point = {}
        for j, key in enumerate(keys):
            lo, hi = bounds[key]
            val = lo + row[j] * (hi - lo)
            point[key] = int(round(val)) if key in int_keys else float(val)
        samples.append(point)
    return samples


def _save_csv(records: List[Dict], path: Path) -> None:
    """Write a list of dicts to CSV — uses pandas if available."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if _PANDAS:
        import pandas as pd
        pd.DataFrame(records).to_csv(path, index=False)
    else:
        import csv
        if not records:
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)


def _section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ===========================================================================
# EXPERIMENT 1 — Heuristic Influence on SA Convergence
# ===========================================================================

def run_exp1(instances: List[ALPInstance], out_dir: Path) -> List[Dict]:
    """
    Evaluate how each initial heuristic influences SA convergence.

    Design
    ------
    Full factorial: 6 heuristics × 13 instances × 1 SA chain each.
    SA budget is fixed to min(adaptive_I_max, 600) outer iterations so that
    all heuristics run under the same computational envelope.

    Response variables
    ------------------
    init_obj        Objective of the raw heuristic sequence (Stage-2 LP value)
    final_obj       Best fully-feasible objective after SA
    improvement_pct (init_obj − final_obj) / init_obj × 100
    gap_pct         (final_obj − known_opt) / known_opt × 100
    ttb_s           Wall-clock seconds to best fully-feasible solution
    iter_to_best    Outer SA iteration at which best was first reached
    conv_speed      Normalised area under the convergence curve
                    (AUC of (history − min) / (max − min)); lower = faster
    n_alt_seqs      Number of distinct near-optimal sequences found
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    _section("EXPERIMENT 1: Heuristic Influence on SA Convergence")

    records: List[Dict] = []

    for inst in instances:
        p_base, _ = adaptive_params(inst.n)
        p = _make_params(p_base, {"I_max": min(p_base.I_max, 600)})

        print(f"\n  Instance: {inst.name}  (n={inst.n})")

        chain_results: Dict[str, Dict] = {}

        for h_name, h_fn in HEURISTICS.items():
            seed = BASE_SEED + abs(hash(h_name)) % 997

            # --- Generate heuristic seed ---
            try:
                seq0 = h_fn(inst)
                init_obj = evaluate(seq0, inst)
                if init_obj >= 1e18:
                    init_obj = evaluate_semi(seq0, inst)
            except Exception as exc:
                print(f"    [{h_name:8s}] heuristic FAILED: {exc}")
                continue

            print(f"    [{h_name:8s}] seed={init_obj:>10.2f}", end="", flush=True)

            # --- Run SA from this seed ---
            try:
                res = _run_single_sa(seq0, inst, p, seed=seed)
            except Exception as exc:
                print(f"  SA FAILED: {exc}")
                continue

            final_obj = res["final_obj"]
            chain_results[h_name] = res

            # Improvement %
            if init_obj > 0 and init_obj < 1e18:
                improvement = 100.0 * (init_obj - final_obj) / init_obj
            else:
                improvement = float("nan")

            # Normalised convergence area (lower = faster convergence)
            hist = res["history"]
            conv_speed = float("nan")
            iter_to_best = -1
            if hist:
                arr = np.array(hist, dtype=float)
                lo, hi = arr.min(), arr.max()
                if hi > lo:
                    arr_norm = (arr - lo) / (hi - lo)
                    conv_speed = float(np.trapz(arr_norm) / len(arr))
                iter_to_best = int(np.argmin(arr))

            records.append({
                "instance":       inst.name,
                "n":              inst.n,
                "heuristic":      h_name,
                "init_obj":       init_obj,
                "final_obj":      final_obj,
                "improvement_pct": improvement,
                "gap_pct":        _gap_pct(final_obj, inst.name),
                "ttb_s":          res["t_best"],
                "wall_time_s":    res["wall_time"],
                "iter_to_best":   iter_to_best,
                "conv_speed":     conv_speed,
                "n_alt_seqs":     res["n_alt_seqs"],
            })

            print(f"  →  final={final_obj:>10.2f}  Δ={improvement:>+6.1f}%"
                  f"  gap={_gap_pct(final_obj, inst.name):>+6.2f}%")

        _plot_exp1_convergence(inst, chain_results, out_dir)

    _save_csv(records, out_dir / "results.csv")
    _plot_exp1_summary(records, out_dir)

    print(f"\n  [Exp 1] Done. Results → {out_dir}/results.csv")
    return records


# --- Exp 1 plots -----------------------------------------------------------

def _plot_exp1_convergence(inst: ALPInstance,
                           chain_results: Dict[str, Dict],
                           out_dir: Path) -> None:
    """Convergence curves for every heuristic on a single instance."""
    if not chain_results:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = cm.tab10(np.linspace(0, 0.9, len(chain_results)))

    for (h_name, res), color in zip(chain_results.items(), colors):
        hist = res.get("history", [])
        if not hist:
            continue
        ax.plot(hist, label=h_name, linewidth=1.6, color=color)

    ax.set_xlabel("SA Outer Iteration")
    ax.set_ylabel("Best Objective (semi-feasible)")
    ax.set_title(f"SA Convergence by Initial Heuristic — {inst.name} (n={inst.n})")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Log scale if range spans more than one order of magnitude
    vals = [v for r in chain_results.values()
            for v in r.get("history", []) if v > 0]
    if vals and max(vals) / max(min(vals), 1e-9) > 10:
        ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(out_dir / f"convergence_{inst.name}.png", dpi=150)
    plt.close(fig)


def _plot_exp1_summary(records: List[Dict], out_dir: Path) -> None:
    """Four summary plots for Experiment 1."""
    if not records:
        return

    h_order = list(HEURISTICS.keys())

    def _group_mean_std(key: str):
        from collections import defaultdict
        groups = defaultdict(list)
        for r in records:
            v = r.get(key)
            if v is not None and not (isinstance(v, float) and (
                    v != v or abs(v) > 1e17)):
                groups[r["heuristic"]].append(v)
        means = {h: np.mean(groups[h]) if groups[h] else float("nan")
                 for h in h_order}
        stds  = {h: np.std(groups[h])  if groups[h] else float("nan")
                 for h in h_order}
        return means, stds

    # --- Plot 1: Mean improvement % and mean gap % side by side ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    m_impr, s_impr = _group_mean_std("improvement_pct")
    m_gap,  s_gap  = _group_mean_std("gap_pct")

    axes[0].bar(h_order, [m_impr[h] for h in h_order],
                yerr=[s_impr[h] for h in h_order],
                capsize=4, color="steelblue", alpha=0.85)
    axes[0].set_xlabel("Initial Heuristic")
    axes[0].set_ylabel("SA Improvement (%)")
    axes[0].set_title("Mean SA Improvement by Initial Heuristic")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(h_order, [m_gap[h] for h in h_order],
                yerr=[s_gap[h] for h in h_order],
                capsize=4, color="tomato", alpha=0.85)
    axes[1].set_xlabel("Initial Heuristic")
    axes[1].set_ylabel("Gap to Known Optimum (%)")
    axes[1].set_title("Mean Optimality Gap by Initial Heuristic")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "summary_improvement_gap.png", dpi=150)
    plt.close(fig)

    # --- Plot 2: Heatmap — improvement % per heuristic × instance ---
    instances = sorted({r["instance"] for r in records})
    heat = np.full((len(h_order), len(instances)), float("nan"))
    inst_idx = {name: i for i, name in enumerate(instances)}
    h_idx    = {h: i for i, h in enumerate(h_order)}
    for r in records:
        v = r.get("improvement_pct")
        if v is not None and not (isinstance(v, float) and v != v):
            heat[h_idx[r["heuristic"]], inst_idx[r["instance"]]] = v

    fig, ax = plt.subplots(figsize=(14, 5))
    masked = np.ma.masked_invalid(heat)
    im = ax.imshow(masked, aspect="auto", cmap="YlGn")
    ax.set_xticks(range(len(instances)))
    ax.set_xticklabels(instances, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(h_order)))
    ax.set_yticklabels(h_order, fontsize=9)
    plt.colorbar(im, ax=ax, label="SA Improvement (%)")
    ax.set_title("SA Improvement (%) — Heuristic × Instance Heatmap")
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap_improvement.png", dpi=150)
    plt.close(fig)

    # --- Plot 3: Convergence speed per heuristic ---
    m_spd, s_spd = _group_mean_std("conv_speed")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(h_order, [m_spd[h] for h in h_order],
           yerr=[s_spd[h] for h in h_order],
           capsize=4, color="mediumpurple", alpha=0.85)
    ax.set_xlabel("Initial Heuristic")
    ax.set_ylabel("Normalised Area Under Convergence Curve")
    ax.set_title("Convergence Speed by Heuristic  (lower = faster)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "convergence_speed.png", dpi=150)
    plt.close(fig)

    # --- Plot 4: Heuristic seed quality vs SA final quality scatter ---
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = cm.tab10(np.linspace(0, 0.9, len(h_order)))
    for h_name, color in zip(h_order, colors):
        sub = [r for r in records if r["heuristic"] == h_name]
        xs = [r["init_obj"]  for r in sub if r["init_obj"]  < 1e17]
        ys = [r["final_obj"] for r in sub if r["final_obj"] < 1e17]
        ax.scatter(xs, ys, label=h_name, s=60, alpha=0.75, color=color)

    all_vals = [r[k] for r in records
                for k in ("init_obj", "final_obj") if r[k] < 1e17]
    if all_vals:
        lo, hi = min(all_vals) * 0.95, max(all_vals) * 1.05
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="No improvement")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

    ax.set_xlabel("Heuristic Seed Objective")
    ax.set_ylabel("SA Final Objective (fully-feasible)")
    ax.set_title("Seed Quality vs SA Output — All Heuristics")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "seed_vs_final.png", dpi=150)
    plt.close(fig)


# ===========================================================================
# EXPERIMENT 2 — ILS Influence on Large Instances (n ≥ 50)
# ===========================================================================

def run_exp2(instances: List[ALPInstance], out_dir: Path) -> List[Dict]:
    """
    Evaluate how ILS restarts improve solution quality for large instances.

    Design
    ------
    Full factorial: ILS_LEVELS × large instances (n ≥ 50) × N_REPS reps.
    EDD is used as the fixed starting heuristic so that heuristic differences
    are not confounded with ILS differences.

    Response variables
    ------------------
    final_obj               Best fully-feasible objective
    gap_pct                 Percentage gap to known optimum
    wall_time_s             Total wall-clock seconds (SA + perturbations)
    ttb_s                   Wall-clock seconds to best fully-feasible
    improvement_vs_sa_pct   Improvement of n_ils>0 over n_ils=0 baseline
                            (computed per instance using mean of N_REPS reps)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    _section(f"EXPERIMENT 2: ILS Influence on Large Instances (n ≥ {LARGE_N})")

    large = [inst for inst in instances if inst.n >= LARGE_N]
    if not large:
        print(f"  No instances with n ≥ {LARGE_N} found — skipping Experiment 2.")
        return []

    print(f"  Large instances selected: {[i.name for i in large]}")
    records: List[Dict] = []

    for inst in large:
        p, _ = adaptive_params(inst.n)
        seq0 = _best_seq(inst)

        print(f"\n  Instance: {inst.name}  (n={inst.n})")

        # Collect baseline (n_ils=0) mean for relative improvement
        baseline_objs: List[float] = []

        for n_ils in ILS_LEVELS:
            rep_objs:  List[float] = []
            rep_walls: List[float] = []
            rep_ttbs:  List[float] = []

            for rep in range(N_REPS):
                seed = BASE_SEED + rep * 137 + n_ils * 31 + inst.n
                tag  = f"n_ils={n_ils}, rep={rep + 1}/{N_REPS}"
                print(f"    [{tag}]", end="", flush=True)

                t0 = time.perf_counter()
                try:
                    if n_ils == 0:
                        pb, fb, stats = run_sa(seq0, inst, p, seed=seed)
                    else:
                        pb, fb, stats = run_ils(seq0, inst, p,
                                                n_restarts=n_ils, seed=seed)

                    obj = stats.get("obj_feas", float("inf"))
                    if obj is None or obj >= 1e18:
                        obj = stats.get("obj", float("inf"))
                    ttb = stats.get("t_best_feas") or stats.get("t_best",
                                                                  float("nan"))
                except Exception as exc:
                    print(f"  FAILED: {exc}")
                    obj, ttb = float("inf"), float("nan")

                wall = time.perf_counter() - t0
                rep_objs.append(obj)
                rep_walls.append(wall)
                rep_ttbs.append(float(ttb) if ttb is not None else float("nan"))

                print(f"  obj={obj:.2f}  wall={wall:.1f}s")

            if n_ils == 0:
                baseline_objs = [v for v in rep_objs if v < 1e18]

            baseline_mean = (np.mean(baseline_objs)
                             if baseline_objs else float("nan"))

            for rep_i, (obj, wall, ttb) in enumerate(
                    zip(rep_objs, rep_walls, rep_ttbs)):
                vs_sa = float("nan")
                if baseline_mean > 0 and baseline_mean < 1e18 and obj < 1e18:
                    vs_sa = 100.0 * (baseline_mean - obj) / baseline_mean

                records.append({
                    "instance":             inst.name,
                    "n":                    inst.n,
                    "n_ils":                n_ils,
                    "rep":                  rep_i + 1,
                    "final_obj":            obj,
                    "gap_pct":              _gap_pct(obj, inst.name),
                    "wall_time_s":          wall,
                    "ttb_s":                ttb,
                    "improvement_vs_sa_pct": vs_sa,
                })

    _save_csv(records, out_dir / "results.csv")
    _plot_exp2_summary(records, out_dir)

    print(f"\n  [Exp 2] Done. Results → {out_dir}/results.csv")
    return records


# --- Exp 2 plots -----------------------------------------------------------

def _plot_exp2_summary(records: List[Dict], out_dir: Path) -> None:
    if not records:
        return

    instances = sorted({r["instance"] for r in records})
    colors    = cm.Set2(np.linspace(0, 0.9, len(instances)))

    # --- Plot 1: Mean objective and wall-time vs n_ils ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for inst_name, color in zip(instances, colors):
        sub = [r for r in records if r["instance"] == inst_name]
        for ax, key, label in [
            (ax1, "final_obj",   "Mean Final Objective"),
            (ax2, "wall_time_s", "Mean Wall Time (s)"),
        ]:
            from collections import defaultdict
            grp = defaultdict(list)
            for r in sub:
                if r[key] < 1e17:
                    grp[r["n_ils"]].append(r[key])
            xs = sorted(grp)
            ys = [np.mean(grp[x]) for x in xs]
            es = [np.std(grp[x])  for x in xs]
            ax.errorbar(xs, ys, yerr=es, marker="o", linewidth=1.6,
                        capsize=4, label=inst_name, color=color)

    ax1.set_xlabel("ILS Restarts (n_ils)")
    ax1.set_ylabel("Mean Final Objective")
    ax1.set_title("Solution Quality vs ILS Restarts")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("ILS Restarts (n_ils)")
    ax2.set_ylabel("Mean Wall Time (s)")
    ax2.set_title("Computation Time vs ILS Restarts")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "quality_and_time_vs_ils.png", dpi=150)
    plt.close(fig)

    # --- Plot 2: Optimality gap vs n_ils ---
    fig, ax = plt.subplots(figsize=(9, 5))
    for inst_name, color in zip(instances, colors):
        sub = [r for r in records if r["instance"] == inst_name]
        from collections import defaultdict
        grp = defaultdict(list)
        for r in sub:
            if not (r["gap_pct"] != r["gap_pct"]):  # exclude NaN
                grp[r["n_ils"]].append(r["gap_pct"])
        xs = sorted(grp)
        if not xs:
            continue
        ys = [np.mean(grp[x]) for x in xs]
        es = [np.std(grp[x])  for x in xs]
        ax.errorbar(xs, ys, yerr=es, marker="o", linewidth=1.6,
                    capsize=4, label=inst_name, color=color)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("ILS Restarts (n_ils)")
    ax.set_ylabel("Gap to Known Optimum (%)")
    ax.set_title(f"Optimality Gap vs ILS Restarts  (n ≥ {LARGE_N})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "gap_vs_ils.png", dpi=150)
    plt.close(fig)

    # --- Plot 3: Improvement over SA baseline — boxplot per n_ils level ---
    ils_pos = [k for k in ILS_LEVELS if k > 0]
    data_boxes = []
    for k in ils_pos:
        vals = [r["improvement_vs_sa_pct"] for r in records
                if r["n_ils"] == k
                and r["improvement_vs_sa_pct"] == r["improvement_vs_sa_pct"]]
        data_boxes.append(vals)

    if any(data_boxes):
        fig, ax = plt.subplots(figsize=(9, 5))
        bp = ax.boxplot(data_boxes, labels=[f"n_ils={k}" for k in ils_pos],
                        patch_artist=True)
        pal = plt.cm.Blues(np.linspace(0.35, 0.85, len(ils_pos)))
        for patch, c in zip(bp["boxes"], pal):
            patch.set_facecolor(c)
        ax.axhline(0, color="tomato", linewidth=1.2, linestyle="--",
                   label="No improvement")
        ax.set_xlabel("ILS Restarts")
        ax.set_ylabel("% Improvement over SA (n_ils=0)")
        ax.set_title("ILS Improvement over Baseline SA")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "ils_improvement_boxplot.png", dpi=150)
        plt.close(fig)

    # --- Plot 4: TTB vs n_ils ---
    fig, ax = plt.subplots(figsize=(9, 5))
    for inst_name, color in zip(instances, colors):
        sub = [r for r in records if r["instance"] == inst_name]
        from collections import defaultdict
        grp = defaultdict(list)
        for r in sub:
            v = r["ttb_s"]
            if v == v and v < 1e17:  # finite, non-NaN
                grp[r["n_ils"]].append(v)
        xs = sorted(grp)
        if not xs:
            continue
        ys = [np.mean(grp[x]) for x in xs]
        ax.plot(xs, ys, marker="s", linewidth=1.6,
                label=inst_name, color=color)

    ax.set_xlabel("ILS Restarts (n_ils)")
    ax.set_ylabel("Mean Time to Best (s)")
    ax.set_title("Time to Best Solution vs ILS Restarts")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "ttb_vs_ils.png", dpi=150)
    plt.close(fig)


# ===========================================================================
# EXPERIMENT 3 — SA Parameter Design of Experiments
# ===========================================================================

def run_exp3(instances: List[ALPInstance], out_dir: Path) -> List[Dict]:
    """
    Detailed SA parameter DOE.

    Sub-experiments
    ---------------
    3a  OFAT (One-Factor-At-a-Time): vary each of the 5 SA parameters
        across OFAT_LEVELS while holding the other four at their
        adaptive defaults.  Generates a per-parameter response curve
        for each representative instance.

    3b  LHS (Latin Hypercube Sample): N_LHS space-filling points over
        the full 5-dimensional parameter space.  Used for correlation-
        based sensitivity ranking and to find jointly good parameters.

    3c  Sensitivity ranking: Pearson |r| between each parameter and
        the final objective across LHS runs, aggregated over instances.

    Representative instances
    ------------------------
    Three instances are selected — the smallest, median-sized, and
    largest — to capture small / medium / large regimes without
    exhausting the computational budget.

    Response variables
    ------------------
    final_obj   Best fully-feasible objective from run_sa
    gap_pct     Percentage gap to known optimum
    ttb_s       Time to best fully-feasible solution
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    _section("EXPERIMENT 3: SA Parameter Design of Experiments")

    sorted_insts = sorted(instances, key=lambda i: i.n)
    rep_insts = _select_representative(sorted_insts, n=3)
    print(f"  Representative instances: {[(i.name, i.n) for i in rep_insts]}")

    all_records: List[Dict] = []

    for inst in rep_insts:
        p_default, _ = adaptive_params(inst.n)
        seq0 = _best_seq(inst)

        print(f"\n  --- {inst.name}  (n={inst.n}) ---")

        print("  3a  OFAT sweep ...")
        ofat_recs = _run_ofat(inst, seq0, p_default, out_dir)
        all_records.extend(ofat_recs)

        print("  3b  LHS sweep ...")
        lhs_recs = _run_lhs(inst, seq0, p_default, out_dir)
        all_records.extend(lhs_recs)

    _save_csv(all_records, out_dir / "results.csv")
    _plot_exp3_sensitivity(all_records, out_dir)
    _plot_ofat_combined(all_records, rep_insts, out_dir)

    print(f"\n  [Exp 3] Done. Results → {out_dir}/results.csv")
    return all_records


def _select_representative(instances: List[ALPInstance],
                            n: int = 3) -> List[ALPInstance]:
    """Return n instances evenly spaced by index after sorting by n."""
    if len(instances) <= n:
        return instances
    idx = np.round(np.linspace(0, len(instances) - 1, n)).astype(int)
    return [instances[i] for i in idx]


# --- Exp 3a: OFAT ----------------------------------------------------------

def _run_ofat(inst: ALPInstance, seq0: List[int],
              p_default: SAParams, out_dir: Path) -> List[Dict]:
    """OFAT sweep for all parameters on one instance."""
    records: List[Dict] = []

    default_vals = {
        "alpha":  p_default.alpha,
        "N_iter": p_default.N_iter,
        "M_stag": p_default.M_stag,
        "I_max":  p_default.I_max,
        "chi0":   p_default.chi0,
    }

    for param_name, levels in OFAT_LEVELS.items():
        level_data: List[Tuple[float, float]] = []   # (param_val, final_obj)

        for lv_i, level_val in enumerate(levels):
            p = _make_params(p_default, {param_name: level_val})
            seed = BASE_SEED + abs(hash(f"{inst.name}{param_name}{lv_i}")) % 997

            res = {}
            try:
                res = _run_single_sa(seq0, inst, p, seed=seed)
                final_obj = res["final_obj"]
            except Exception as exc:
                print(f"    OFAT [{param_name}={level_val}] FAILED: {exc}")
                final_obj = float("inf")

            is_default = (abs(level_val - default_vals[param_name]) < 1e-9
                          if isinstance(level_val, float)
                          else level_val == default_vals[param_name])

            records.append({
                "experiment":   "OFAT",
                "instance":     inst.name,
                "n":            inst.n,
                "param_varied": param_name,
                "param_value":  level_val,
                "is_default":   is_default,
                "final_obj":    final_obj,
                "gap_pct":      _gap_pct(final_obj, inst.name),
                "ttb_s":        res.get("t_best", float("nan")),
                "wall_time_s":  res.get("wall_time", float("nan")),
                # Freeze other parameters at their defaults for reference
                "alpha":        p.alpha,
                "N_iter":       p.N_iter,
                "M_stag":       p.M_stag,
                "I_max":        p.I_max,
                "chi0":         p.chi0,
            })
            level_data.append((level_val, final_obj))
            print(f"    OFAT [{param_name:8s}={level_val:8.4g}]"
                  f"  obj={final_obj:.2f}")

        _plot_ofat_single(inst, param_name, level_data,
                          default_vals[param_name], out_dir)

    return records


def _plot_ofat_single(inst: ALPInstance, param_name: str,
                      data: List[Tuple[float, float]],
                      default_val: float, out_dir: Path) -> None:
    """Response curve for one parameter on one instance."""
    xs = [d[0] for d in data]
    ys = [d[1] for d in data]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, "o-", linewidth=1.8, markersize=8, color="steelblue")
    ax.axvline(default_val, color="tomato", linewidth=1.5,
               linestyle="--", label=f"Adaptive default ({default_val:.4g})")
    ax.set_xlabel(param_name)
    ax.set_ylabel("Final Objective (post-SA)")
    ax.set_title(f"OFAT: {param_name} — {inst.name} (n={inst.n})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"ofat_{param_name}_{inst.name}.png", dpi=150)
    plt.close(fig)


# --- Exp 3b: LHS -----------------------------------------------------------

def _run_lhs(inst: ALPInstance, seq0: List[int],
             p_default: SAParams, out_dir: Path) -> List[Dict]:
    """LHS sweep over full parameter space on one instance."""
    samples = _lhs_sample(N_LHS, PARAM_SPACE,
                          seed=BASE_SEED + inst.n * 17)
    records: List[Dict] = []

    for si, sample in enumerate(samples):
        p = _make_params(p_default, sample)
        seed = BASE_SEED + si * 7 + inst.n * 1000

        res = {}
        try:
            res = _run_single_sa(seq0, inst, p, seed=seed)
            final_obj = res["final_obj"]
        except Exception as exc:
            print(f"    LHS [{si}] FAILED: {exc}")
            final_obj = float("inf")

        records.append({
            "experiment":  "LHS",
            "instance":    inst.name,
            "n":           inst.n,
            "sample_i":    si,
            "final_obj":   final_obj,
            "gap_pct":     _gap_pct(final_obj, inst.name),
            "ttb_s":       res.get("t_best", float("nan")),
            "wall_time_s": res.get("wall_time", float("nan")),
            **sample,
        })

        if (si + 1) % 10 == 0:
            valid = [r["final_obj"] for r in records
                     if r["final_obj"] < 1e17]
            best  = min(valid) if valid else float("nan")
            print(f"    LHS [{si + 1:3d}/{N_LHS}]"
                  f"  best so far={best:.2f}")

    _plot_lhs_scatter(inst, records, out_dir)
    return records


def _plot_lhs_scatter(inst: ALPInstance,
                      records: List[Dict], out_dir: Path) -> None:
    """Scatter of each LHS parameter vs final objective."""
    params = list(PARAM_SPACE.keys())
    fig, axes = plt.subplots(1, len(params),
                             figsize=(4 * len(params), 4),
                             squeeze=False)

    for ax, param in zip(axes[0], params):
        xs = np.array([r[param] for r in records], dtype=float)
        ys = np.array([r["final_obj"] for r in records], dtype=float)
        mask = np.isfinite(xs) & np.isfinite(ys) & (ys < 1e17)

        ax.scatter(xs[mask], ys[mask], alpha=0.55, s=28, c="steelblue")

        # Linear trend
        if mask.sum() > 2:
            z = np.polyfit(xs[mask], ys[mask], 1)
            xline = np.linspace(xs[mask].min(), xs[mask].max(), 60)
            ax.plot(xline, np.polyval(z, xline), "r--", linewidth=1.2)

        ax.set_xlabel(param, fontsize=9)
        if param == params[0]:
            ax.set_ylabel("Final Objective")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"LHS Parameter Scatter — {inst.name} (n={inst.n})",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / f"lhs_scatter_{inst.name}.png", dpi=150)
    plt.close(fig)


# --- Exp 3c: Sensitivity ranking -------------------------------------------

def _plot_exp3_sensitivity(records: List[Dict], out_dir: Path) -> None:
    """Pearson |r| sensitivity ranking from LHS results."""
    lhs = [r for r in records if r.get("experiment") == "LHS"]
    if not lhs:
        return

    params    = list(PARAM_SPACE.keys())
    instances = sorted({r["instance"] for r in lhs})

    # |r| per parameter × instance
    corr_matrix = np.full((len(instances), len(params)), float("nan"))

    for ri, inst_name in enumerate(instances):
        sub  = [r for r in lhs
                if r["instance"] == inst_name and r["final_obj"] < 1e17]
        objs = np.array([r["final_obj"] for r in sub], dtype=float)

        for ci, param in enumerate(params):
            xs = np.array([r[param] for r in sub], dtype=float)
            mask = np.isfinite(xs) & np.isfinite(objs)
            if mask.sum() < 4 or xs[mask].std() < 1e-12:
                continue
            corr = np.corrcoef(xs[mask], objs[mask])[0, 1]
            corr_matrix[ri, ci] = abs(corr)

    mean_corr = np.nanmean(corr_matrix, axis=0)
    std_corr  = np.nanstd(corr_matrix,  axis=0)
    order     = np.argsort(mean_corr)[::-1]
    ordered_params = [params[i] for i in order]

    # Bar chart: mean |r| sorted by importance
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(ordered_params, mean_corr[order], yerr=std_corr[order],
           capsize=4, color="mediumpurple", alpha=0.85)
    ax.set_xlabel("SA Parameter")
    ax.set_ylabel("|Pearson r| with Final Objective")
    ax.set_title("Parameter Sensitivity Ranking  (mean |r| across instances)")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "sensitivity_ranking.png", dpi=150)
    plt.close(fig)

    # Heatmap: |r| per instance × parameter
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(corr_matrix[:, order], aspect="auto",
                   cmap="Oranges", vmin=0, vmax=1)
    ax.set_yticks(range(len(instances)))
    ax.set_yticklabels(instances, fontsize=8)
    ax.set_xticks(range(len(ordered_params)))
    ax.set_xticklabels(ordered_params, fontsize=9)
    plt.colorbar(im, ax=ax, label="|Pearson r|")
    ax.set_title("Parameter Sensitivity Heatmap  (|r| per instance)")
    fig.tight_layout()
    fig.savefig(out_dir / "sensitivity_heatmap.png", dpi=150)
    plt.close(fig)

    # Print ranking table to stdout
    print("\n  Parameter sensitivity ranking (mean |Pearson r|):")
    for i in order:
        print(f"    {params[i]:8s}  |r| = {mean_corr[i]:.3f} ± {std_corr[i]:.3f}")


def _plot_ofat_combined(records: List[Dict],
                        rep_insts: List[ALPInstance],
                        out_dir: Path) -> None:
    """
    Grid plot: each row = one representative instance,
               each column = one SA parameter.
    Shows the OFAT response curve with the adaptive default marked.
    """
    ofat = [r for r in records if r.get("experiment") == "OFAT"]
    if not ofat:
        return

    params    = list(OFAT_LEVELS.keys())
    inst_names = [i.name for i in rep_insts]
    n_rows, n_cols = len(inst_names), len(params)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 3.2 * n_rows),
                             squeeze=False)

    for ri, inst_name in enumerate(inst_names):
        for ci, param in enumerate(params):
            ax = axes[ri][ci]
            sub = sorted(
                [r for r in ofat
                 if r["instance"] == inst_name and r["param_varied"] == param],
                key=lambda r: r["param_value"],
            )
            if not sub:
                ax.set_visible(False)
                continue

            xs = np.array([r["param_value"] for r in sub], dtype=float)
            ys = np.array([r["final_obj"]   for r in sub], dtype=float)
            mask = np.isfinite(ys) & (ys < 1e17)

            ax.plot(xs[mask], ys[mask], "o-", color="steelblue",
                    linewidth=1.6, markersize=6)

            # Highlight the default value
            def_mask = np.array([r["is_default"] for r in sub], dtype=bool)
            if def_mask.any():
                ax.scatter(xs[def_mask], ys[def_mask],
                           color="tomato", zorder=5, s=90)

            ax.set_xlabel(param, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
            if ci == 0:
                ax.set_ylabel(inst_name, fontsize=8)
            if ri == 0:
                ax.set_title(param, fontsize=9)

    fig.suptitle(
        "OFAT Response Curves: SA Parameters × Representative Instances\n"
        "(red dot = adaptive default)",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "ofat_combined.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# SUMMARY REPORT
# ===========================================================================

def _write_doe_summary(results: Dict[str, List[Dict]], out_dir: Path) -> None:
    """Write a JSON summary of key findings from all three experiments."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                     time.gmtime())}

    # Exp 1 — best heuristic by mean improvement
    recs1 = results.get("exp1", [])
    if recs1:
        from collections import defaultdict
        impr = defaultdict(list)
        for r in recs1:
            v = r.get("improvement_pct")
            if v is not None and v == v:
                impr[r["heuristic"]].append(v)
        mean_impr = {h: float(np.mean(vs)) for h, vs in impr.items() if vs}
        best_h = max(mean_impr, key=mean_impr.get) if mean_impr else None
        summary["exp1"] = {
            "best_heuristic_by_mean_improvement": best_h,
            "mean_improvement_pct": mean_impr,
        }

    # Exp 2 — best n_ils by mean gap
    recs2 = results.get("exp2", [])
    if recs2:
        from collections import defaultdict
        gap_by_ils = defaultdict(list)
        for r in recs2:
            v = r.get("gap_pct")
            if v is not None and v == v:
                gap_by_ils[r["n_ils"]].append(v)
        mean_gap = {k: float(np.mean(vs))
                    for k, vs in gap_by_ils.items() if vs}
        best_ils = min(mean_gap, key=mean_gap.get) if mean_gap else None
        summary["exp2"] = {
            "best_n_ils_by_mean_gap": best_ils,
            "mean_gap_pct_by_n_ils": {str(k): v for k, v in mean_gap.items()},
        }

    # Exp 3 — best LHS point and parameter ranking
    recs3 = results.get("exp3", [])
    lhs3  = [r for r in recs3 if r.get("experiment") == "LHS"]
    if lhs3:
        valid = [r for r in lhs3 if r["final_obj"] < 1e17]
        if valid:
            best = min(valid, key=lambda r: r["final_obj"])
            summary["exp3"] = {
                "best_lhs_params": {k: best[k] for k in PARAM_SPACE
                                    if k in best},
                "best_lhs_obj":    best["final_obj"],
                "best_lhs_gap_pct": best.get("gap_pct"),
                "best_lhs_instance": best["instance"],
            }

    with open(out_dir / "doe_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print(f"\n  Summary → {out_dir}/doe_summary.json")
    print(json.dumps(summary, indent=2, default=str))


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ALP Design of Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--exp", type=int, choices=[1, 2, 3], default=None,
        help="Run only experiment 1, 2, or 3. Default: run all three.",
    )
    parser.add_argument(
        "--data-dir", default="data",
        help="Directory containing airland*.txt files (default: ./data).",
    )
    parser.add_argument(
        "--out-dir", default="doe_results",
        help="Root output directory for results and plots (default: ./doe_results).",
    )
    parser.add_argument(
        "--large-n", type=int, default=LARGE_N,
        help=f"Instance size threshold for Experiment 2 (default: {LARGE_N}).",
    )
    parser.add_argument(
        "--n-lhs", type=int, default=N_LHS,
        help=f"Number of LHS points in Experiment 3 (default: {N_LHS}).",
    )
    parser.add_argument(
        "--n-reps", type=int, default=N_REPS,
        help=f"Repetitions per (instance, n_ils) in Experiment 2 (default: {N_REPS}).",
    )
    parser.add_argument(
        "--seed", type=int, default=BASE_SEED,
        help=f"Base random seed (default: {BASE_SEED}).",
    )
    args = parser.parse_args()

    # Apply CLI overrides to module-level globals
    global DATA_DIR, DOE_DIR, EXP1_DIR, EXP2_DIR, EXP3_DIR
    global LARGE_N, N_LHS, N_REPS, BASE_SEED
    DATA_DIR  = Path(args.data_dir)
    DOE_DIR   = Path(args.out_dir)
    EXP1_DIR  = DOE_DIR / "exp1_heuristic"
    EXP2_DIR  = DOE_DIR / "exp2_ils"
    EXP3_DIR  = DOE_DIR / "exp3_params"
    LARGE_N   = args.large_n
    N_LHS     = args.n_lhs
    N_REPS    = args.n_reps
    BASE_SEED = args.seed

    print("=" * 70)
    print("ALP Design of Experiments")
    print(f"  Data dir   : {DATA_DIR}")
    print(f"  Output dir : {DOE_DIR}")
    print(f"  Base seed  : {BASE_SEED}")
    print("=" * 70)

    instances = _load_all_instances()
    print(f"Loaded {len(instances)} instances: "
          f"{[i.name for i in instances]}")

    DOE_DIR.mkdir(parents=True, exist_ok=True)
    run_all = args.exp is None
    results: Dict[str, List[Dict]] = {}

    if run_all or args.exp == 1:
        results["exp1"] = run_exp1(instances, EXP1_DIR)

    if run_all or args.exp == 2:
        results["exp2"] = run_exp2(instances, EXP2_DIR)

    if run_all or args.exp == 3:
        results["exp3"] = run_exp3(instances, EXP3_DIR)

    _write_doe_summary(results, DOE_DIR)

    _section("ALL EXPERIMENTS COMPLETE")
    print(f"  All results and plots saved under: {DOE_DIR}/")


if __name__ == "__main__":
    main()
