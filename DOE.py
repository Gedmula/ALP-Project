"""
doe_alp.py
Aircraft Landing Problem — Design of Experiments
=================================================
Three controlled experiments that import directly from Single_runway_SA.py.

  Exp-1  Heuristic Seeding Study
         Isolates the effect of each initial solution generator (EDD, ERD,
         MDD, MPDS, ATC_k2, ATC_k4) on SA convergence quality and speed.
         Design : single SA chain per heuristic; R replications with distinct
                  seeds; all chains share identical SAParams and time budget.
         Parallelism: all (heuristic × rep) chains per instance submitted to
                  a single ProcessPoolExecutor of N_CPU workers.
         Metrics: initial/final objective (semi + fully-feasible), gap to
                  known optimum, time-to-best, convergence speed (outer
                  iterations to within 0.5% of chain final).

  Exp-2  ILS Depth Study  (instances with n ≥ exp2_min_n only)
         Tests whether increasing ILS restart count improves solution quality
         on larger instances where MS-SA is most strained.
         Design : full MS-SA with n_ils ∈ {0, 2, 4, 6}; R replications via
                  per-replication seed offset forwarded to every chain worker.
         Parallelism: each MS-SA call already uses all N_CPU cores internally
                  via its own ProcessPoolExecutor; the outer (n_ils × rep)
                  loop remains sequential to avoid nested pool contention.
         Metrics: mean/std gap per (instance, n_ils), time-to-best.

  Exp-3  Parameter Sensitivity DOE
         2^4 full factorial on {alpha, N_iter, I_max, M_stag} plus one
         center-point replicate.  A single fixed heuristic seed (EDD) is
         used so parameter effects are not confounded with seeding effects.
         Design : 16 corner combinations + 1 center × R replications.
         Parallelism: all (combo × rep) chains per instance submitted to
                  a single ProcessPoolExecutor of N_CPU workers.
         Metrics: main effects and two-factor interactions on gap to optimum.

Usage
-----
    Set RUN_EXP1 / RUN_EXP2 / RUN_EXP3 = True/False in the __main__ block,
    then run:  python doe_alp.py

Output
------
    doe_results/
        exp1_heuristic/  records.csv, summary.csv, plots/
        exp2_ils_depth/  records.csv, summary.csv, plots/
        exp3_parameter/  records.csv, main_effects.csv, interactions.csv, plots/
"""

# ═══════════════════════════════════════════════════════════════════════════
# 0.  IMPORTS
# ═══════════════════════════════════════════════════════════════════════════

import csv, math, os, time, warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from itertools import product as iproduct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from Single_runway_SA import (
    ALPInstance, SAParams, load_orlib,
    gen_edd, gen_erd, gen_mdd, gen_mpds, gen_atc,
    run_sa, evaluate, evaluate_semi,
    adaptive_params, _build_starts, _sa_worker, _CTX,
    N_CPU,
)

try:
    from tqdm import tqdm as _tqdm
    def _progress(it, **kw): return _tqdm(it, **kw)
except ImportError:
    def _progress(it, **kw): return it

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
# 1.  CONSTANTS & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

OR_DATA: Dict[str, float] = {
    "airland1":  700.0,    "airland2":  1480.0,
    "airland3":  820.0,    "airland4":  2520.0,
    "airland5":  3100.0,   "airland6":  24442.0,
    "airland7":  1550.0,   "airland8":  1950.0,
    "airland9":  5611.70,  "airland10": 12640.42,
    "airland11": 12462.18, "airland12": 16629.10,
    "airland13": 39287.52,
}

HEURISTICS: Dict[str, callable] = {
    "EDD":    gen_edd,
    "ERD":    gen_erd,
    "MDD":    gen_mdd,
    "MPDS":   gen_mpds,
    "ATC_k2": lambda inst: gen_atc(inst, K=2.0),
    "ATC_k4": lambda inst: gen_atc(inst, K=4.0),
}

FACTOR_LABELS = ["alpha", "N_iter", "I_max", "M_stag"]


@dataclass
class DOEConfig:
    # ── Paths ───────────────────────────────────────────────────────────────
    data_dir:              str   = "data"
    results_dir:           str   = "doe_results"

    # ── Experiment scope ────────────────────────────────────────────────────
    exp1_instances:        Optional[List[str]] = None
    exp2_min_n:            int   = 50
    exp3_instances:        Optional[List[str]] = None

    # ── Replications ────────────────────────────────────────────────────────
    exp1_reps:             int   = 5
    exp2_reps:             int   = 3
    exp3_reps:             int   = 2

    # ── Time budgets per run (seconds) ──────────────────────────────────────
    exp1_t_small:          float = 60.0     # n <= 20
    exp1_t_med:            float = 120.0    # 20 < n <= 50
    exp1_t_large:          float = 240.0    # n > 50
    exp2_t_limit:          float = 300.0    # full MS-SA per (n_ils, rep)
    exp3_t_small:          float = 60.0     # n <= 50
    exp3_t_large:          float = 180.0    # n > 50

    # ── Exp-2: ILS depth levels ─────────────────────────────────────────────
    exp2_n_ils_levels:     List[int] = field(
        default_factory=lambda: [0, 2, 4, 6])

    # ── Exp-3: 2^4 factor levels ────────────────────────────────────────────
    exp3_alpha_lo:         float = 0.950
    exp3_alpha_hi:         float = 0.995
    exp3_n_iter_lo:        int   = 80
    exp3_n_iter_hi:        int   = 400
    exp3_i_max_lo:         int   = 200
    exp3_i_max_hi:         int   = 1200
    exp3_m_stag_lo:        int   = 30
    exp3_m_stag_hi:        int   = 180
    exp3_center_alpha:     float = 0.980
    exp3_center_n_iter:    int   = 200
    exp3_center_i_max:     int   = 600
    exp3_center_m_stag:    int   = 100

    # ── Hardware ────────────────────────────────────────────────────────────
    n_workers:             int   = N_CPU
    seed_base:             int   = 42
    verbose:               bool  = True

    # ── Helpers ─────────────────────────────────────────────────────────────
    def exp1_t_limit(self, n: int) -> float:
        if n <= 20: return self.exp1_t_small
        if n <= 50: return self.exp1_t_med
        return self.exp1_t_large

    def exp3_t_limit(self, n: int) -> float:
        return self.exp3_t_small if n <= 50 else self.exp3_t_large


# ═══════════════════════════════════════════════════════════════════════════
# 2.  SHARED UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def load_instances(cfg: DOEConfig) -> Dict[str, Tuple[ALPInstance, float]]:
    """Load all available OR-Library files. Returns {name: (inst, known_opt)}."""
    data_dir = Path(cfg.data_dir)
    out: Dict[str, Tuple[ALPInstance, float]] = {}
    for name, opt in OR_DATA.items():
        p = data_dir / f"{name}.txt"
        if p.exists():
            try:
                inst = load_orlib(str(p), name)
                out[name] = (inst, opt)
                if cfg.verbose:
                    print(f"  [LOADED]  {name:12s}  n={inst.n:4d}  opt={opt}")
            except Exception as exc:
                print(f"  [ERROR]   {name}: {exc}")
        elif cfg.verbose:
            print(f"  [MISSING] {name}")
    return out


def gap_pct(f: float, opt: float) -> float:
    """Percentage gap to known optimum. Returns nan if either value is invalid."""
    if math.isinf(f) or math.isinf(opt) or opt <= 0 or math.isnan(f):
        return float("nan")
    return (f - opt) / opt * 100.0


def _convergence_speed(history: List[float], final_obj: float,
                       threshold: float = 0.005) -> int:
    """
    Return the outer-iteration index at which the history first reaches
    <= final_obj * (1 + threshold).
    """
    target = final_obj * (1.0 + threshold) + 1e-6
    for i, v in enumerate(history):
        if not math.isinf(v) and v <= target:
            return i
    return len(history)


def _exp1_sa_params(n: int) -> SAParams:
    """Fixed SAParams for Exp-1 and Exp-3 single-chain runs."""
    if n <= 20:
        return SAParams(alpha=0.97,  N_iter=100, T_min=1e-4, I_max=400,  M_stag=60)
    if n <= 50:
        return SAParams(alpha=0.98,  N_iter=150, T_min=1e-4, I_max=600,  M_stag=80)
    return SAParams(alpha=0.995, N_iter=300, T_min=1e-5, I_max=1200, M_stag=120)


def _write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# 3.  EXPERIMENT 1 — PARALLEL WORKERS
# ═══════════════════════════════════════════════════════════════════════════

def _exp1_chain_worker(args: tuple) -> dict:
    """
    Spawn-safe worker for one (heuristic × replication) SA chain.

    Computes init values and runs run_sa entirely inside the worker process
    so that no shared state is required.  t_deadline is computed locally
    after the process starts so that scheduling lag does not eat into the
    actual search budget.

    Args tuple layout:
        (inst, seq0, p, seed, t_lim, name, h_name, rep, opt)
    """
    inst, seq0, p, seed, t_lim, name, h_name, rep, opt = args
    init_semi  = evaluate_semi(seq0, inst)
    init_feas  = evaluate(seq0, inst)
    t_deadline = time.perf_counter() + t_lim
    _, fb_semi, stats = run_sa(seq0, inst, p, seed=seed, t_deadline=t_deadline)
    fb_feas  = stats.get("obj_feas",    float("inf"))
    history  = stats.get("history",     [])
    g        = gap_pct(fb_feas, opt)
    return {
        "instance":    name,
        "n":           inst.n,
        "known_opt":   opt,
        "heuristic":   h_name,
        "rep":         rep,
        "seed":        seed,
        "init_semi":   round(init_semi, 4),
        "init_feas":   round(init_feas, 4) if not math.isinf(init_feas) else None,
        "final_semi":  round(fb_semi,   4),
        "final_feas":  round(fb_feas,   4) if not math.isinf(fb_feas)  else None,
        "gap_pct":     round(g,         4) if not math.isnan(g)        else None,
        "t_best_feas": round(stats.get("t_best_feas", 0.0), 4),
        "t_best_semi": round(stats.get("t_best",      0.0), 4),
        "wall_s":      round(stats.get("time",        0.0), 4),
        "conv_itr":    _convergence_speed(history, fb_semi),
        "history":     history,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3.  EXPERIMENT 1 — HEURISTIC SEEDING STUDY
# ═══════════════════════════════════════════════════════════════════════════

def run_heuristic_study(
    instances: Dict[str, Tuple[ALPInstance, float]],
    cfg: DOEConfig,
) -> List[dict]:
    """
    Parallel heuristic seeding study.

    All (heuristic × replication) chains for a given instance are submitted
    simultaneously to a ProcessPoolExecutor of cfg.n_workers workers.  With
    6 heuristics × cfg.exp1_reps replications = 30 chains per instance, a
    32-core machine runs all 30 chains concurrently — up to 30× faster than
    the sequential version.

    SAParams and time budget are held constant across all treatments within
    an instance so that outcomes are attributable solely to the starting
    sequence.
    """
    records: List[dict] = []
    inst_names = cfg.exp1_instances or list(instances.keys())
    seeds      = [cfg.seed_base + i * 17 for i in range(cfg.exp1_reps)]

    n_chains = len(HEURISTICS) * cfg.exp1_reps
    print(f"\n{'═'*72}")
    print(f"  EXP-1  HEURISTIC SEEDING STUDY  ({cfg.n_workers} workers, "
          f"{n_chains} chains/instance)")
    print(f"  Instances: {len(inst_names)}  |  Heuristics: {len(HEURISTICS)}"
          f"  |  Reps: {cfg.exp1_reps}")
    print(f"{'═'*72}")

    for name in inst_names:
        if name not in instances:
            print(f"  ⚠  {name} not available — skipped"); continue
        inst, opt = instances[name]
        p     = _exp1_sa_params(inst.n)
        t_lim = cfg.exp1_t_limit(inst.n)

        # Build task list: one entry per (heuristic × replication)
        tasks: List[tuple] = []
        for h_name, h_fn in HEURISTICS.items():
            seq0 = h_fn(inst)
            for rep, seed in enumerate(seeds):
                tasks.append((inst, seq0, p, seed, t_lim,
                               name, h_name, rep + 1, opt))

        print(f"\n  ── {name}  (n={inst.n}, opt={opt})  "
              f"SAParams: α={p.alpha} N_iter={p.N_iter} "
              f"t_lim={t_lim:.0f}s ──")
        print(f"  Submitting {len(tasks)} chains to {cfg.n_workers} workers...")

        with ProcessPoolExecutor(max_workers=cfg.n_workers,
                                 mp_context=_CTX) as ex:
            batch = list(_progress(
                ex.map(_exp1_chain_worker, tasks),
                total=len(tasks), desc=f"  {name}", leave=False))

        # Sort for deterministic console output: heuristic → rep
        batch.sort(key=lambda r: (r["heuristic"], r["rep"]))

        if cfg.verbose:
            print(f"  {'Heuristic':8s} {'Rep':>3} {'Init(semi)':>11}"
                  f" {'Init(feas)':>11} {'Final(feas)':>12}"
                  f" {'Gap%':>8} {'T_best(s)':>10} {'Conv_itr':>9}")
            print(f"  {'─'*8} {'─'*3} {'─'*11} {'─'*11} {'─'*12}"
                  f" {'─'*8} {'─'*10} {'─'*9}")
            for r in batch:
                fin_s = f"{r['final_feas']:.2f}" if r["final_feas"] is not None else "inf"
                g_s   = f"{r['gap_pct']:+.2f}"   if r["gap_pct"]   is not None else "N/A"
                i_f_s = f"{r['init_feas']:.2f}"  if r["init_feas"] is not None else "inf"
                print(f"  {r['heuristic']:8s} {r['rep']:>3d} {r['init_semi']:>11.2f}"
                      f" {i_f_s:>11s} {fin_s:>12s} {g_s:>8s}"
                      f" {r['t_best_feas']:>10.2f} {r['conv_itr']:>9d}")

        records.extend(batch)

    print(f"\n  Exp-1 complete — {len(records)} records collected.")
    return records


def _aggregate_exp1(records: List[dict]) -> Dict[str, Dict[str, dict]]:
    """
    Aggregate Exp-1 records into:
        agg[instance][heuristic] = {mean_gap, std_gap, mean_conv_itr, ...}
    """
    bucket: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        bucket[r["instance"]][r["heuristic"]].append(r)

    agg: Dict[str, Dict[str, dict]] = {}
    for inst_name, h_dict in bucket.items():
        agg[inst_name] = {}
        for h_name, rows in h_dict.items():
            f_feas = [r["final_feas"] for r in rows if r["final_feas"] is not None]
            gaps   = [r["gap_pct"]    for r in rows if r["gap_pct"]    is not None]
            convs  = [r["conv_itr"]   for r in rows]
            i_feas = [r["init_feas"]  for r in rows if r["init_feas"]  is not None]
            agg[inst_name][h_name] = {
                "n_feasible":      len(f_feas),
                "mean_final_feas": float(np.mean(f_feas)) if f_feas else float("inf"),
                "std_final_feas":  float(np.std(f_feas))  if len(f_feas) > 1 else 0.0,
                "mean_gap_pct":    float(np.mean(gaps))   if gaps  else float("nan"),
                "std_gap_pct":     float(np.std(gaps))    if len(gaps) > 1 else 0.0,
                "mean_init_feas":  float(np.mean(i_feas)) if i_feas else float("inf"),
                "mean_conv_itr":   float(np.mean(convs)),
                "std_conv_itr":    float(np.std(convs))   if len(convs) > 1 else 0.0,
            }
    return agg


def plot_heuristic_study(records: List[dict], out_dir: Path) -> None:
    """Three figures: (a) gap box plots, (b) convergence curves,
    (c) initial vs final paired bars — one subplot grid per instance."""
    agg       = _aggregate_exp1(records)
    instances = list(agg.keys())
    h_names   = list(HEURISTICS.keys())
    palette   = plt.cm.tab10.colors

    # ── (a) Gap box plots ────────────────────────────────────────────────
    fig_rows  = math.ceil(len(instances) / 3)
    fig, axes = plt.subplots(fig_rows, min(3, len(instances)),
                             figsize=(5 * min(3, len(instances)), 4 * fig_rows),
                             squeeze=False)
    axes_flat = axes.flatten()
    gap_bucket: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r["gap_pct"] is not None:
            gap_bucket[r["instance"]][r["heuristic"]].append(r["gap_pct"])

    for ax_idx, name in enumerate(instances):
        ax   = axes_flat[ax_idx]
        data = [gap_bucket[name].get(h, []) for h in h_names]
        bp   = ax.boxplot(data, labels=h_names, patch_artist=True, widths=0.55)
        for patch, color in zip(bp["boxes"], palette):
            patch.set_facecolor(color); patch.set_alpha(0.75)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_title(name, fontsize=9); ax.set_ylabel("Gap %", fontsize=8)
        ax.tick_params(axis="x", labelsize=7, rotation=20)
        ax.grid(axis="y", alpha=0.25)
    for ax in axes_flat[len(instances):]:
        ax.set_visible(False)
    fig.suptitle("Exp-1 — Optimality Gap by Heuristic Seed", fontsize=11, y=1.01)
    plt.tight_layout()
    p = out_dir / "exp1_gap_boxplots.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")

    # ── (b) Convergence curves (mean over reps, one line per heuristic) ──
    conv_bucket: Dict[str, Dict[str, List[List[float]]]] = \
        defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.get("history"):
            conv_bucket[r["instance"]][r["heuristic"]].append(r["history"])

    for name in instances:
        if name not in conv_bucket: continue
        fig, ax = plt.subplots(figsize=(10, 4))
        for ci, h_name in enumerate(h_names):
            histories = conv_bucket[name][h_name]
            if not histories: continue
            max_len = max(len(h) for h in histories)
            padded  = [h + [h[-1]] * (max_len - len(h)) for h in histories]
            mean_h  = np.nanmean([[v for v in row] for row in padded], axis=0)
            ax.plot(mean_h, label=h_name, color=palette[ci], linewidth=1.5)
        ax.set_xlabel("Outer Iteration", fontsize=10)
        ax.set_ylabel("Mean fb_semi", fontsize=10)
        ax.set_title(f"Exp-1 Convergence — {name}", fontsize=11)
        ax.legend(fontsize=8, ncol=3); ax.grid(alpha=0.22)
        plt.tight_layout()
        p = out_dir / f"exp1_convergence_{name}.png"
        plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  Saved: {p}")

    # ── (c) Initial vs final objective (mean, paired bars) ───────────────
    for name in instances:
        h_agg = agg.get(name, {})
        if not h_agg: continue
        labels   = list(h_agg.keys())
        init_obj = [h_agg[h]["mean_init_feas"] for h in labels]
        fin_obj  = [h_agg[h]["mean_final_feas"] for h in labels]
        valid = [(l, i, f) for l, i, f in zip(labels, init_obj, fin_obj)
                 if not (math.isinf(i) or math.isinf(f))]
        if not valid: continue
        labels, init_obj, fin_obj = zip(*valid)
        x = np.arange(len(labels)); w = 0.38
        fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.1), 4))
        ax.bar(x - w/2, init_obj, w, color="#aec6e8", edgecolor="#1a6faf",
               linewidth=0.8, hatch="///", alpha=0.75, label="Heuristic seed (pre-SA)")
        ax.bar(x + w/2, fin_obj,  w, color="#1a6faf", alpha=0.85, label="SA final (feas)")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, fontsize=9)
        opt_val = OR_DATA.get(name)
        if opt_val:
            ax.axhline(opt_val, color="black", lw=0.9, ls="--",
                       label=f"Known opt ({opt_val})")
        ax.set_ylabel("Objective", fontsize=10)
        ax.set_title(f"Exp-1 Seed vs SA Final — {name}", fontsize=11)
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.22)
        plt.tight_layout()
        p = out_dir / f"exp1_seed_vs_final_{name}.png"
        plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  Saved: {p}")


def export_exp1(records: List[dict], out_dir: Path) -> None:
    agg        = _aggregate_exp1(records)
    fields_raw = ["instance", "n", "known_opt", "heuristic", "rep", "seed",
                  "init_semi", "init_feas", "final_semi", "final_feas",
                  "gap_pct", "t_best_feas", "t_best_semi", "wall_s", "conv_itr"]
    _write_csv(out_dir / "records.csv",
               [{k: r.get(k) for k in fields_raw} for r in records],
               fields_raw)
    rows = []
    for inst_name, h_dict in agg.items():
        for h_name, vals in h_dict.items():
            rows.append({"instance": inst_name, "heuristic": h_name, **{
                k: (f"{v:.4f}" if isinstance(v, float) and not math.isnan(v)
                    else ("nan" if isinstance(v, float) and math.isnan(v) else v))
                for k, v in vals.items()}})
    _write_csv(out_dir / "summary.csv", rows,
               ["instance", "heuristic", "n_feasible",
                "mean_final_feas", "std_final_feas",
                "mean_gap_pct", "std_gap_pct",
                "mean_init_feas", "mean_conv_itr", "std_conv_itr"])


# ═══════════════════════════════════════════════════════════════════════════
# 4.  EXPERIMENT 2 — ILS DEPTH STUDY
# ═══════════════════════════════════════════════════════════════════════════

def _safe_n_workers(n: int, n_workers_max: int) -> int:
    """
    Scale down the worker count for large instances to avoid OOM.

    For n aircraft, each LP call in _build_lp_matrices allocates a dense
    matrix of shape (n(n-1)/2, 3n) in float64.  The memory cost per worker
    is approximately:

        bytes ≈ n(n-1)/2 × 3n × 8  =  12n³  (bytes, leading term)

    Keeping total concurrent LP memory below ~12 GB (conservative for a
    32-core workstation with ≥32 GB RAM):

        n_workers ≤ 12 × 10⁹  /  (12 × n³)  =  10⁹ / n³

    The table below maps this formula to a practical per-tier cap.
    """
    if n <= 100:  return n_workers_max
    if n <= 200:  return min(n_workers_max, 12)
    if n <= 300:  return min(n_workers_max,  6)
    if n <= 400:  return min(n_workers_max,  3)
    return min(n_workers_max, 2)   # n > 400  (airland13: n=500)


def _ms_sa_seeded(inst: ALPInstance, p: SAParams, n_workers: int,
                  n_ils: int, rep: int, t_limit: float
                  ) -> Tuple[float, float, float, float]:
    """
    Full MS-SA with a replication-specific seed offset.

    Each chain's seed is shifted by (rep * 1000) so that independent
    replications explore distinct trajectories from the same heuristic
    starts.  This function already uses all workers internally via
    ProcessPoolExecutor, so the outer (n_ils × rep) loop is kept sequential
    to avoid nested pool contention.

    Workers are scaled down for large instances via _safe_n_workers to
    prevent OOM.  If the pool crashes anyway (e.g. on
    an unusually memory-constrained node) the call falls back to sequential
    execution so the experiment does not abort.

    Returns: (fb_feas, fb_semi, t_best_feas, t_best_semi)
    """
    w           = _safe_n_workers(inst.n, n_workers)
    seed_offset = rep * 1000
    starts      = _build_starts(inst, n_starts=w)
    t0          = time.perf_counter()
    t_deadline  = t0 + t_limit
    tasks       = [(lbl, seq, inst, p, sd + seed_offset, n_ils, t_deadline)
                   for lbl, seq, sd in starts]

    try:
        with ProcessPoolExecutor(max_workers=w, mp_context=_CTX) as ex:
            results = list(ex.map(_sa_worker, tasks))
    except Exception as exc:
        warnings.warn(
            f"_ms_sa_seeded: pool failed for {inst.name} n_ils={n_ils} rep={rep} "
            f"({type(exc).__name__}: {exc}).  Falling back to sequential execution."
        )
        results = [_sa_worker(t) for t in tasks]

    # Field layout: 0=lbl,1=pb_semi,2=fb_semi,3=pi_feas,4=fb_feas,
    #               5=hist,6=t_best_sa,7=t_best_feas,8=n_alt,9=init,10=alpha_hist
    feas = [r for r in results if not math.isinf(r[4])]
    if feas:
        best = min(feas, key=lambda r: r[4])
        return best[4], best[2], best[7], best[6]
    best = min(results, key=lambda r: r[2])
    return float("inf"), best[2], float("inf"), best[6]


def run_ils_depth_study(
    instances: Dict[str, Tuple[ALPInstance, float]],
    cfg: DOEConfig,
) -> List[dict]:
    """
    ILS depth study on all instances with n >= cfg.exp2_min_n.

    Each _ms_sa_seeded call already saturates all cfg.n_workers cores
    internally, so the outer (n_ils × rep) loop is sequential.  The
    SAParams come from adaptive_params(n), matching the production solver.
    """
    records: List[dict] = []
    large = {k: v for k, v in instances.items() if v[0].n >= cfg.exp2_min_n}

    print(f"\n{'═'*72}")
    print(f"  EXP-2  ILS DEPTH STUDY  (n ≥ {cfg.exp2_min_n}, "
          f"{cfg.n_workers} workers/call — already saturating all cores)")
    print(f"  Instances: {len(large)}  |  n_ils levels: {cfg.exp2_n_ils_levels}"
          f"  |  Reps: {cfg.exp2_reps}")
    print(f"{'═'*72}")

    for name, (inst, opt) in large.items():
        sa_p, _ = adaptive_params(inst.n)
        print(f"\n  ── {name}  (n={inst.n}, opt={opt})  "
              f"workers={_safe_n_workers(inst.n, cfg.n_workers)} ──")
        print(f"  {'n_ils':>6} {'Rep':>4} {'Obj(feas)':>12} {'Gap%':>8}"
              f" {'T_best(s)':>10} {'Wall(s)':>8}")
        print(f"  {'─'*6} {'─'*4} {'─'*12} {'─'*8} {'─'*10} {'─'*8}")

        for n_ils in cfg.exp2_n_ils_levels:
            for rep in range(cfg.exp2_reps):
                t_wall0 = time.perf_counter()
                fb_feas, fb_semi, t_bf, t_bs = _ms_sa_seeded(
                    inst, sa_p, cfg.n_workers, n_ils, rep, cfg.exp2_t_limit)
                wall = time.perf_counter() - t_wall0
                g    = gap_pct(fb_feas, opt)

                if cfg.verbose:
                    f_s = f"{fb_feas:.2f}" if not math.isinf(fb_feas) else "inf"
                    g_s = f"{g:+.2f}"      if not math.isnan(g)       else "N/A"
                    print(f"  {n_ils:>6d} {rep+1:>4d} {f_s:>12s} {g_s:>8s}"
                          f" {t_bf:>10.2f} {wall:>8.2f}")

                records.append({
                    "instance":    name,
                    "n":           inst.n,
                    "known_opt":   opt,
                    "n_ils":       n_ils,
                    "rep":         rep + 1,
                    "final_feas":  round(fb_feas, 4) if not math.isinf(fb_feas) else None,
                    "final_semi":  round(fb_semi, 4),
                    "gap_pct":     round(g,        4) if not math.isnan(g)       else None,
                    "t_best_feas": round(t_bf,     4) if not math.isinf(t_bf)    else None,
                    "t_best_semi": round(t_bs,     4),
                    "wall_s":      round(wall,     4),
                })

    print(f"\n  Exp-2 complete — {len(records)} records collected.")
    return records


def plot_ils_depth_study(records: List[dict], out_dir: Path) -> None:
    """(a) mean gap vs n_ils per instance with ±1 std bands,
    (b) mean time-to-best vs n_ils."""
    instances  = sorted({r["instance"] for r in records})
    ils_levels = sorted({r["n_ils"]    for r in records})
    palette    = plt.cm.tab10.colors

    def _collect(inst_name, metric):
        means, stds = [], []
        for n_ils in ils_levels:
            vals = [r[metric] for r in records
                    if r["instance"] == inst_name and r["n_ils"] == n_ils
                    and r[metric] is not None]
            means.append(float(np.mean(vals)) if vals else float("nan"))
            stds.append( float(np.std(vals))  if len(vals) > 1 else 0.0)
        return np.array(means), np.array(stds)

    fig, ax = plt.subplots(figsize=(9, 5))
    for ci, name in enumerate(instances):
        means, stds = _collect(name, "gap_pct")
        valid = ~np.isnan(means)
        if not valid.any(): continue
        x = np.array(ils_levels)[valid]
        ax.plot(x, means[valid], marker="o", color=palette[ci % 10],
                label=name, linewidth=1.8)
        ax.fill_between(x, means[valid] - stds[valid],
                        means[valid] + stds[valid],
                        color=palette[ci % 10], alpha=0.15)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("n_ils (ILS restarts per chain)", fontsize=10)
    ax.set_ylabel("Mean gap to optimum (%)", fontsize=10)
    ax.set_title("Exp-2 — ILS Depth vs Optimality Gap", fontsize=11)
    ax.set_xticks(ils_levels); ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.22)
    plt.tight_layout()
    p = out_dir / "exp2_gap_vs_nils.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")

    fig, ax = plt.subplots(figsize=(9, 5))
    for ci, name in enumerate(instances):
        means, _ = _collect(name, "t_best_feas")
        valid = ~np.isnan(means)
        if not valid.any(): continue
        x = np.array(ils_levels)[valid]
        ax.plot(x, means[valid], marker="s", color=palette[ci % 10],
                linestyle="--", label=name, linewidth=1.5)
    ax.set_xlabel("n_ils (ILS restarts per chain)", fontsize=10)
    ax.set_ylabel("Mean time-to-best (s)", fontsize=10)
    ax.set_title("Exp-2 — ILS Depth vs Time-to-Best", fontsize=11)
    ax.set_xticks(ils_levels); ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.22)
    plt.tight_layout()
    p = out_dir / "exp2_ttb_vs_nils.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")


def export_exp2(records: List[dict], out_dir: Path) -> None:
    fields = ["instance", "n", "known_opt", "n_ils", "rep",
              "final_feas", "final_semi", "gap_pct",
              "t_best_feas", "t_best_semi", "wall_s"]
    _write_csv(out_dir / "records.csv", records, fields)

    instances  = sorted({r["instance"] for r in records})
    ils_levels = sorted({r["n_ils"]    for r in records})
    summary    = []
    for name in instances:
        for n_ils in ils_levels:
            rows = [r for r in records
                    if r["instance"] == name and r["n_ils"] == n_ils]
            gaps = [r["gap_pct"]     for r in rows if r["gap_pct"]     is not None]
            ttbs = [r["t_best_feas"] for r in rows if r["t_best_feas"] is not None]
            summary.append({
                "instance":     name,
                "n":            rows[0]["n"]         if rows else "",
                "known_opt":    rows[0]["known_opt"] if rows else "",
                "n_ils":        n_ils,
                "n_reps":       len(rows),
                "mean_gap_pct": f"{np.mean(gaps):.4f}" if gaps else "nan",
                "std_gap_pct":  f"{np.std(gaps):.4f}"  if len(gaps) > 1 else "0.0",
                "mean_ttb_s":   f"{np.mean(ttbs):.3f}" if ttbs else "nan",
                "std_ttb_s":    f"{np.std(ttbs):.3f}"  if len(ttbs) > 1 else "0.0",
            })
    _write_csv(out_dir / "summary.csv", summary,
               ["instance", "n", "known_opt", "n_ils", "n_reps",
                "mean_gap_pct", "std_gap_pct", "mean_ttb_s", "std_ttb_s"])


# ═══════════════════════════════════════════════════════════════════════════
# 5.  EXPERIMENT 3 — PARALLEL WORKER
# ═══════════════════════════════════════════════════════════════════════════

def _exp3_chain_worker(args: tuple) -> dict:
    """
    Spawn-safe worker for one (combo × replication) SA chain in Exp-3.

    t_deadline is computed locally after process start so scheduling lag
    does not eat into the search budget.

    Args tuple layout:
        (inst, seq0, p, label, coded, seed, t_lim, name, opt, combo_id, rep)
    """
    inst, seq0, p, label, coded, seed, t_lim, name, opt, combo_id, rep = args
    t_deadline = time.perf_counter() + t_lim
    _, fb_semi, stats = run_sa(seq0, inst, p, seed=seed, t_deadline=t_deadline)
    fb_feas = stats.get("obj_feas", float("inf"))
    g       = gap_pct(fb_feas, opt)
    return {
        "instance":    name,
        "n":           inst.n,
        "known_opt":   opt,
        "combo_id":    combo_id,
        "combo_label": label,
        "coded_alpha": coded[0],
        "coded_Ni":    coded[1],
        "coded_Imax":  coded[2],
        "coded_Mstag": coded[3],
        "alpha":       p.alpha,
        "N_iter":      p.N_iter,
        "I_max":       p.I_max,
        "M_stag":      p.M_stag,
        "rep":         rep,
        "seed":        seed,
        "final_semi":  round(fb_semi, 4),
        "final_feas":  round(fb_feas, 4) if not math.isinf(fb_feas) else None,
        "gap_pct":     round(g,        4) if not math.isnan(g)       else None,
        "t_best_feas": round(stats.get("t_best_feas", 0.0), 4),
        "wall_s":      round(stats.get("time",        0.0), 4),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5.  EXPERIMENT 3 — PARAMETER SENSITIVITY DOE
# ═══════════════════════════════════════════════════════════════════════════

def _build_factorial_design(cfg: DOEConfig) -> List[Tuple[SAParams, str, List[int]]]:
    """
    Generate 2^4 full factorial + center point.

    Returns a list of (SAParams, combo_label, coded_vector) tuples.
    coded_vector entries are +1 (high) / -1 (low) / 0 (center) for each of
    {alpha, N_iter, I_max, M_stag}; used for main-effect calculations.
    """
    lo = (cfg.exp3_alpha_lo, cfg.exp3_n_iter_lo,
          cfg.exp3_i_max_lo, cfg.exp3_m_stag_lo)
    hi = (cfg.exp3_alpha_hi, cfg.exp3_n_iter_hi,
          cfg.exp3_i_max_hi, cfg.exp3_m_stag_hi)
    combos = []
    for bits in iproduct((-1, +1), repeat=4):
        vals = [h if b == +1 else l for b, l, h in zip(bits, lo, hi)]
        a, ni, im, ms = vals
        p     = SAParams(alpha=a, N_iter=int(ni), T_min=1e-4,
                         I_max=int(im), M_stag=int(ms))
        label = (f"a{'+' if bits[0]>0 else '-'}"
                 f"Ni{'+' if bits[1]>0 else '-'}"
                 f"Im{'+' if bits[2]>0 else '-'}"
                 f"Ms{'+' if bits[3]>0 else '-'}")
        combos.append((p, label, list(bits)))
    pc = SAParams(alpha=cfg.exp3_center_alpha, N_iter=cfg.exp3_center_n_iter,
                  T_min=1e-4, I_max=cfg.exp3_center_i_max,
                  M_stag=cfg.exp3_center_m_stag)
    combos.append((pc, "CENTER", [0, 0, 0, 0]))
    return combos


def run_parameter_doe(
    instances: Dict[str, Tuple[ALPInstance, float]],
    cfg: DOEConfig,
) -> List[dict]:
    """
    Parallel 2^4 full factorial parameter study.

    All (combo × rep) chains for a given instance are submitted simultaneously
    to a ProcessPoolExecutor of cfg.n_workers workers.  With 17 combos ×
    cfg.exp3_reps replications = 34 chains per instance, a 32-core machine
    runs them nearly concurrently — up to 32× faster than sequential.

    EDD is the fixed starting heuristic to isolate parameter effects from
    seeding effects.
    """
    records: List[dict] = []
    combos     = _build_factorial_design(cfg)
    inst_names = cfg.exp3_instances or list(instances.keys())
    seeds      = [cfg.seed_base + i * 13 for i in range(cfg.exp3_reps)]

    n_chains = len(combos) * cfg.exp3_reps
    print(f"\n{'═'*72}")
    print(f"  EXP-3  PARAMETER SENSITIVITY DOE  (2^4 + center, "
          f"{cfg.n_workers} workers, {n_chains} chains/instance)")
    print(f"  Instances: {len(inst_names)}  |  Combos: {len(combos)}"
          f"  |  Reps: {cfg.exp3_reps}  |  Seed heuristic: EDD")
    print(f"{'═'*72}")
    print(f"  Factor levels:")
    print(f"    alpha : lo={cfg.exp3_alpha_lo}   hi={cfg.exp3_alpha_hi}")
    print(f"    N_iter: lo={cfg.exp3_n_iter_lo}     hi={cfg.exp3_n_iter_hi}")
    print(f"    I_max : lo={cfg.exp3_i_max_lo}    hi={cfg.exp3_i_max_hi}")
    print(f"    M_stag: lo={cfg.exp3_m_stag_lo}     hi={cfg.exp3_m_stag_hi}")

    for name in inst_names:
        if name not in instances:
            print(f"  ⚠  {name} not available — skipped"); continue
        inst, opt = instances[name]
        t_lim = cfg.exp3_t_limit(inst.n)
        seq0  = gen_edd(inst)

        # Build task list: one entry per (combo × replication)
        tasks: List[tuple] = []
        for combo_id, (p, label, coded) in enumerate(combos):
            for rep, seed in enumerate(seeds):
                tasks.append((inst, seq0, p, label, coded, seed,
                               t_lim, name, opt, combo_id, rep + 1))

        print(f"\n  ── {name}  (n={inst.n}, opt={opt}, t_limit={t_lim:.0f}s) ──")
        print(f"  Submitting {len(tasks)} chains to {cfg.n_workers} workers...")

        with ProcessPoolExecutor(max_workers=cfg.n_workers,
                                 mp_context=_CTX) as ex:
            batch = list(_progress(
                ex.map(_exp3_chain_worker, tasks),
                total=len(tasks), desc=f"  {name}", leave=False))

        batch.sort(key=lambda r: (r["combo_id"], r["rep"]))

        if cfg.verbose:
            print(f"  {'Combo':20s} {'Rep':>3} {'Final(feas)':>12}"
                  f" {'Gap%':>8} {'Wall(s)':>8}")
            print(f"  {'─'*20} {'─'*3} {'─'*12} {'─'*8} {'─'*8}")
            for r in batch:
                f_s = f"{r['final_feas']:.2f}" if r["final_feas"] is not None else "inf"
                g_s = f"{r['gap_pct']:+.2f}"   if r["gap_pct"]   is not None else "N/A"
                print(f"  {r['combo_label']:20s} {r['rep']:>3d}"
                      f" {f_s:>12s} {g_s:>8s} {r['wall_s']:>8.2f}")

        records.extend(batch)

    print(f"\n  Exp-3 complete — {len(records)} records collected.")
    return records


def _compute_main_effects(records: List[dict]) -> Dict[str, Dict[str, float]]:
    """
    Compute 2^4 factorial main effects and two-factor interactions per instance.

    Main effect of F:   ME(F) = mean(y | F=+1) − mean(y | F=−1)
    Interaction A×B:    mean(coded_A × coded_B × y) over all corner runs
    Curvature:          center-point mean − corner mean

    Center point (coded = 0) is excluded from main-effect / interaction
    calculations and used only for the curvature estimate.
    """
    instances  = sorted({r["instance"] for r in records})
    effects: Dict[str, Dict[str, float]] = {}
    coded_keys = ["coded_alpha", "coded_Ni", "coded_Imax", "coded_Mstag"]

    for name in instances:
        rows = [r for r in records
                if r["instance"] == name
                and r["gap_pct"] is not None
                and all(r[ck] != 0 for ck in coded_keys)]
        if not rows:
            effects[name] = {}; continue

        y   = np.array([r["gap_pct"] for r in rows])
        X   = np.array([[r[ck] for ck in coded_keys] for r in rows])
        eff: Dict[str, float] = {}

        for fi, fname in enumerate(FACTOR_LABELS):
            hi = y[X[:, fi] == +1]; lo = y[X[:, fi] == -1]
            eff[fname] = float(np.mean(hi) - np.mean(lo)) if len(hi) and len(lo) else float("nan")

        for i in range(len(FACTOR_LABELS)):
            for j in range(i + 1, len(FACTOR_LABELS)):
                eff[f"{FACTOR_LABELS[i]}×{FACTOR_LABELS[j]}"] = \
                    float(np.mean(X[:, i] * X[:, j] * y))

        corner_mean = float(np.mean(y))
        center_rows = [r for r in records
                       if r["instance"] == name and r["gap_pct"] is not None
                       and all(r[ck] == 0 for ck in coded_keys)]
        if center_rows:
            eff["curvature"] = (float(np.mean([r["gap_pct"] for r in center_rows]))
                                - corner_mean)
        effects[name] = eff

    return effects


def plot_parameter_doe(records: List[dict], out_dir: Path) -> None:
    """(a) main effects plots, (b) factor importance bar chart,
    (c) two-factor interaction bars."""
    effects   = _compute_main_effects(records)
    instances = sorted({r["instance"] for r in records})
    palette   = plt.cm.tab10.colors

    # ── (a) Main effects ─────────────────────────────────────────────────
    n_inst    = len(instances)
    coded_keys = ["coded_alpha", "coded_Ni", "coded_Imax", "coded_Mstag"]
    fig, axes = plt.subplots(n_inst, len(FACTOR_LABELS),
                             figsize=(4 * len(FACTOR_LABELS), 3.5 * n_inst),
                             squeeze=False)
    cfg_obj = DOEConfig()
    lo_lbls = [cfg_obj.exp3_alpha_lo, cfg_obj.exp3_n_iter_lo,
               cfg_obj.exp3_i_max_lo, cfg_obj.exp3_m_stag_lo]
    hi_lbls = [cfg_obj.exp3_alpha_hi, cfg_obj.exp3_n_iter_hi,
               cfg_obj.exp3_i_max_hi, cfg_obj.exp3_m_stag_hi]

    for ri, name in enumerate(instances):
        for ci, (fname, ck, lo_lbl, hi_lbl) in enumerate(
                zip(FACTOR_LABELS, coded_keys, lo_lbls, hi_lbls)):
            ax = axes[ri][ci]
            lo_vals = [r["gap_pct"] for r in records
                       if r["instance"] == name and r[ck] == -1
                       and r["gap_pct"] is not None]
            hi_vals = [r["gap_pct"] for r in records
                       if r["instance"] == name and r[ck] == +1
                       and r["gap_pct"] is not None]
            if not lo_vals or not hi_vals:
                ax.set_visible(False); continue
            lo_mean, hi_mean = np.mean(lo_vals), np.mean(hi_vals)
            ax.plot(["Low", "High"], [lo_mean, hi_mean],
                    marker="o", color=palette[ri % 10], linewidth=2)
            ax.scatter(["Low", "High"], [lo_mean, hi_mean],
                       color=palette[ri % 10], s=60, zorder=3)
            ax.set_title(f"{name} — {fname}", fontsize=8)
            ax.set_ylabel("Mean gap (%)", fontsize=7)
            ax.set_xticklabels([str(lo_lbl), str(hi_lbl)], fontsize=7)
            ax.axhline(0, color="gray", lw=0.6, ls=":")
            ax.grid(alpha=0.25)

    fig.suptitle("Exp-3 — Main Effects (mean gap % per factor level)", fontsize=11)
    plt.tight_layout()
    p = out_dir / "exp3_main_effects.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")

    # ── (b) Factor importance ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, max(4, n_inst * 0.9)))
    y_pos   = np.arange(len(instances)); bar_w = 0.18
    for fi, fname in enumerate(FACTOR_LABELS):
        me_vals = [abs(effects.get(n, {}).get(fname, 0.0)) for n in instances]
        ax.barh(y_pos + fi * bar_w, me_vals, bar_w,
                label=fname, color=palette[fi], alpha=0.80)
    ax.set_yticks(y_pos + bar_w * 1.5)
    ax.set_yticklabels(instances, fontsize=9)
    ax.set_xlabel("|Main Effect| on gap%", fontsize=10)
    ax.set_title("Exp-3 — Factor Importance per Instance", fontsize=11)
    ax.legend(fontsize=9); ax.grid(axis="x", alpha=0.22)
    plt.tight_layout()
    p = out_dir / "exp3_factor_importance.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")

    # ── (c) Interaction bars ──────────────────────────────────────────────
    int_keys  = [f"{a}×{b}" for i, a in enumerate(FACTOR_LABELS)
                 for b in FACTOR_LABELS[i+1:]]
    fig, axes = plt.subplots(1, max(1, n_inst),
                             figsize=(4.5 * n_inst, 4.5), squeeze=False)
    for ci, name in enumerate(instances):
        ax   = axes[0][ci]
        eff  = effects.get(name, {})
        vals = sorted([(k, eff[k]) for k in int_keys if k in eff],
                      key=lambda x: abs(x[1]), reverse=True)
        if not vals:
            ax.set_visible(False); continue
        labels_i = [v[0] for v in vals]
        sizes    = [v[1] for v in vals]
        colors   = ["#c0392b" if s > 0 else "#1a6faf" for s in sizes]
        ax.barh(range(len(labels_i)), sizes, color=colors, alpha=0.80)
        ax.set_yticks(range(len(labels_i)))
        ax.set_yticklabels(labels_i, fontsize=7)
        ax.axvline(0, color="black", lw=0.7)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Interaction effect on gap%", fontsize=7)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Exp-3 — Two-Factor Interactions on Gap%", fontsize=11)
    plt.tight_layout()
    p = out_dir / "exp3_interactions.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")


def export_exp3(records: List[dict], out_dir: Path) -> None:
    fields_raw = ["instance", "n", "known_opt", "combo_id", "combo_label",
                  "coded_alpha", "coded_Ni", "coded_Imax", "coded_Mstag",
                  "alpha", "N_iter", "I_max", "M_stag",
                  "rep", "seed", "final_semi", "final_feas",
                  "gap_pct", "t_best_feas", "wall_s"]
    _write_csv(out_dir / "records.csv", records, fields_raw)

    effects = _compute_main_effects(records)
    me_rows = []
    for name, eff in effects.items():
        for fname, val in eff.items():
            me_rows.append({
                "instance":              name,
                "factor_or_interaction": fname,
                "effect":                f"{val:.4f}" if not math.isnan(val) else "nan",
            })
    _write_csv(out_dir / "main_effects.csv", me_rows,
               ["instance", "factor_or_interaction", "effect"])


# ═══════════════════════════════════════════════════════════════════════════
# 6.  MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

def main(run_exps: Optional[List[int]] = None) -> None:
    """
    Run all or a subset of the three DOE experiments.

    Parameters
    ----------
    run_exps : list of int, optional
        Experiments to run (1, 2, and/or 3). None runs all three.
    """
    cfg      = DOEConfig()
    run_exps = set(run_exps or [1, 2, 3])

    print("\n" + "═" * 72)
    print("  ALP — DESIGN OF EXPERIMENTS")
    print(f"  Experiments requested : {sorted(run_exps)}")
    print(f"  Workers               : {cfg.n_workers}  (all cores used)")
    print(f"  Data dir              : {Path(cfg.data_dir).resolve()}")
    print(f"  Results dir           : {Path(cfg.results_dir).resolve()}")
    print("═" * 72)

    print("\nLoading OR-Library instances...")
    instances = load_instances(cfg)
    if not instances:
        print("  No instances found — check data_dir. Exiting."); return

    if 1 in run_exps:
        out1 = Path(cfg.results_dir) / "exp1_heuristic"
        out1.mkdir(parents=True, exist_ok=True)
        rec1 = run_heuristic_study(instances, cfg)
        export_exp1(rec1, out1)
        plot_heuristic_study(rec1, out1)

    if 2 in run_exps:
        out2 = Path(cfg.results_dir) / "exp2_ils_depth"
        out2.mkdir(parents=True, exist_ok=True)
        rec2 = run_ils_depth_study(instances, cfg)
        export_exp2(rec2, out2)
        if rec2:
            plot_ils_depth_study(rec2, out2)
        else:
            print(f"  ⚠  No instances with n ≥ {cfg.exp2_min_n} — Exp-2 produced no records.")

    if 3 in run_exps:
        out3 = Path(cfg.results_dir) / "exp3_parameter"
        out3.mkdir(parents=True, exist_ok=True)
        rec3 = run_parameter_doe(instances, cfg)
        export_exp3(rec3, out3)
        if rec3:
            plot_parameter_doe(rec3, out3)

    print(f"\n{'═'*72}")
    print(f"  DOE complete.  All outputs in: {Path(cfg.results_dir).resolve()}")
    print(f"{'═'*72}\n")


if __name__ == "__main__":
    # ── Configure which experiments to run ──────────────────────────────
    # Set each flag to True or False.
    RUN_EXP1 = True    # Heuristic Seeding Study
    RUN_EXP2 = True    # ILS Depth Study  (n >= exp2_min_n only)
    RUN_EXP3 = True    # Parameter Sensitivity DOE  (2^4 factorial)

    exps = [i for i, run in [(1, RUN_EXP1), (2, RUN_EXP2), (3, RUN_EXP3)] if run]
    main(run_exps=exps or None)