"""
doe_analysis.py
Aircraft Landing Problem — DOE Result Analysis & Optimised Benchmark
=====================================================================
Reads the CSV outputs written by doe_alp.py, derives the best parameter
configuration per instance from the three experiments, and re-runs the
main solver using those instance-specific settings.

Pipeline
--------
  Step 1  Load DOE records from doe_results/
  Step 2  Exp-1 analysis  → best heuristic seed per instance
  Step 3  Exp-2 analysis  → best n_ils per instance
  Step 4  Exp-3 analysis  → best SAParams per instance (factorial main
                            effects + best observed corner)
  Step 5  Merge per-instance recommendations into an OptConfig
  Step 6  Run ms_sa on each instance with its OptConfig
  Step 7  Compare optimised vs adaptive-default results
  Step 8  Export analysis report + optimised benchmark results

Outputs
-------
  doe_results/analysis/
      recommendations.csv   — one row per instance, best settings
      benchmark_opt.csv     — optimised run objective, gap, wall time
      benchmark_cmp.csv     — side-by-side comparison vs default params
      analysis_report.txt   — narrative per-instance findings
      plots/
          heuristic_ranking_<instance>.png
          main_effects_summary.png
          nils_gain.png
          opt_vs_default.png

Usage
-----
  python doe_analysis.py
"""

# ═══════════════════════════════════════════════════════════════════════════
# 0.  IMPORTS
# ═══════════════════════════════════════════════════════════════════════════

import csv, math, time, warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from Single_runway_SA import (
    ALPInstance, SAParams, load_orlib,
    gen_edd, gen_erd, gen_mdd, gen_mpds, gen_atc,
    adaptive_params, _build_starts, _sa_worker, _CTX,
    N_CPU, _fmt_obj,
)
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
# 1.  CONSTANTS
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

HEURISTIC_GEN = {
    "EDD":    gen_edd,
    "ERD":    gen_erd,
    "MDD":    gen_mdd,
    "MPDS":   gen_mpds,
    "ATC_k2": lambda inst: gen_atc(inst, K=2.0),
    "ATC_k4": lambda inst: gen_atc(inst, K=4.0),
}

FACTOR_LABELS = ["alpha", "N_iter", "I_max", "M_stag"]

DOE_DIR  = Path("doe_results")
OUT_DIR  = DOE_DIR / "analysis"
DATA_DIR = Path("data")


# ═══════════════════════════════════════════════════════════════════════════
# 2.  DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class OptConfig:
    """Per-instance optimised configuration derived from DOE analysis."""
    instance:       str
    n:              int
    known_opt:      float

    # From Exp-1
    best_heuristic: str   = "EDD"
    exp1_mean_gap:  float = float("nan")   # mean gap of best heuristic

    # From Exp-2
    best_n_ils:     int   = 0
    exp2_mean_gap:  float = float("nan")   # mean gap at best n_ils

    # From Exp-3
    best_alpha:     float = 0.99
    best_N_iter:    int   = 120
    best_I_max:     int   = 600
    best_M_stag:    int   = 60
    exp3_best_gap:  float = float("nan")   # best gap observed in factorial

    # Analysis flags
    heuristic_significant: bool  = False   # large spread across heuristics
    ils_significant:       bool  = False   # ILS restart count matters
    notes:                 str   = ""

    def sa_params(self) -> SAParams:
        return SAParams(
            alpha  = self.best_alpha,
            N_iter = self.best_N_iter,
            T_min  = 1e-5 if self.n > 100 else 1e-4,
            I_max  = self.best_I_max,
            M_stag = self.best_M_stag,
        )


# ═══════════════════════════════════════════════════════════════════════════
# 3.  CSV LOADERS
# ═══════════════════════════════════════════════════════════════════════════

def _load_csv(path: Path) -> List[dict]:
    if not path.exists():
        print(f"  [MISSING] {path}")
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _float(val: str) -> float:
    """Parse a CSV value to float; return nan on failure."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return float("nan")


def _int(val: str) -> int:
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0


# ═══════════════════════════════════════════════════════════════════════════
# 4.  EXPERIMENT 1 — HEURISTIC ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def analyse_exp1(records: List[dict]) -> Dict[str, Tuple[str, float, float]]:
    """
    Determine the best heuristic seed per instance.

    Selection criterion: lowest mean gap_pct across replications.
    When two heuristics tie within 0.1 percentage points, prefer MPDS
    (the method's own dispatching rule) then MDD (often best single-point
    schedule on ALP instances) then EDD.

    Returns: {instance: (best_heuristic, mean_gap, gap_range)}
      gap_range = max_mean_gap - min_mean_gap, used as a significance proxy.
    """
    TIEBREAK = ["MPDS", "MDD", "EDD", "ATC_k2", "ATC_k4", "ERD"]

    # Aggregate: mean gap per (instance, heuristic)
    agg: Dict[str, Dict[str, List[float]]] = {}
    for r in records:
        inst = r["instance"]
        h    = r["heuristic"]
        g    = _float(r.get("gap_pct", ""))
        if math.isnan(g): continue
        agg.setdefault(inst, {}).setdefault(h, []).append(g)

    results: Dict[str, Tuple[str, float, float]] = {}
    for inst, h_dict in agg.items():
        mean_gaps = {h: float(np.mean(v)) for h, v in h_dict.items() if v}
        if not mean_gaps:
            results[inst] = ("EDD", float("nan"), 0.0)
            continue

        gap_range = max(mean_gaps.values()) - min(mean_gaps.values())
        min_gap   = min(mean_gaps.values())

        # Candidates within 0.1% of best
        candidates = [h for h, g in mean_gaps.items() if g <= min_gap + 0.1]
        best = next((h for h in TIEBREAK if h in candidates), candidates[0])
        results[inst] = (best, mean_gaps[best], gap_range)

    return results


def plot_heuristic_ranking(records: List[dict], out_dir: Path) -> None:
    """Horizontal bar chart: mean gap ± std per heuristic, one panel per instance."""
    agg: Dict[str, Dict[str, List[float]]] = {}
    for r in records:
        g = _float(r.get("gap_pct", ""))
        if math.isnan(g): continue
        agg.setdefault(r["instance"], {}).setdefault(r["heuristic"], []).append(g)

    palette = plt.cm.tab10.colors
    for inst, h_dict in agg.items():
        h_names  = sorted(h_dict.keys(),
                          key=lambda h: float(np.mean(h_dict[h])))
        means    = [float(np.mean(h_dict[h])) for h in h_names]
        stds     = [float(np.std(h_dict[h])) if len(h_dict[h]) > 1 else 0.0
                    for h in h_names]
        colors   = [palette[i % 10] for i in range(len(h_names))]

        fig, ax = plt.subplots(figsize=(8, max(3, len(h_names) * 0.7)))
        bars = ax.barh(h_names, means, xerr=stds, color=colors,
                       alpha=0.82, capsize=4, error_kw={"linewidth": 1.0})
        ax.axvline(0, color="black", lw=0.8, ls="--")
        for bar, m in zip(bars, means):
            ax.text(m + (max(means) - min(means)) * 0.01 + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{m:+.3f}%", va="center", fontsize=8)
        ax.set_xlabel("Mean gap to optimum (%)", fontsize=10)
        ax.set_title(f"Heuristic Seed Ranking — {inst}", fontsize=11)
        ax.grid(axis="x", alpha=0.25)
        plt.tight_layout()
        p = out_dir / f"heuristic_ranking_{inst}.png"
        plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  Saved: {p}")


# ═══════════════════════════════════════════════════════════════════════════
# 5.  EXPERIMENT 2 — ILS DEPTH ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def analyse_exp2(records: List[dict],
                 significance_threshold: float = 0.2
                 ) -> Dict[str, Tuple[int, float, bool]]:
    """
    Determine the best n_ils per instance.

    Selection criterion: lowest mean gap.  Diminishing-returns check: if
    the improvement from n_ils=0 to n_ils=best is less than
    significance_threshold percentage points, the ILS effect is flagged as
    non-significant and n_ils=0 is preferred (faster).

    Returns: {instance: (best_n_ils, mean_gap_at_best, is_significant)}
    """
    agg: Dict[str, Dict[int, List[float]]] = {}
    for r in records:
        inst  = r["instance"]
        n_ils = _int(r["n_ils"])
        g     = _float(r.get("gap_pct", ""))
        if math.isnan(g): continue
        agg.setdefault(inst, {}).setdefault(n_ils, []).append(g)

    results: Dict[str, Tuple[int, float, bool]] = {}
    for inst, ils_dict in agg.items():
        mean_gaps  = {k: float(np.mean(v)) for k, v in ils_dict.items() if v}
        if not mean_gaps:
            results[inst] = (0, float("nan"), False); continue

        best_k     = min(mean_gaps, key=mean_gaps.get)
        best_gap   = mean_gaps[best_k]
        base_gap   = mean_gaps.get(0, mean_gaps[min(mean_gaps)])
        gain       = base_gap - best_gap          # positive = improvement
        significant = gain >= significance_threshold

        # Prefer n_ils=0 when gain is negligible (it is faster)
        chosen_k = best_k if significant else 0
        results[inst] = (chosen_k, mean_gaps.get(chosen_k, best_gap), significant)

    return results


def plot_nils_gain(records: List[dict], out_dir: Path) -> None:
    """Gap improvement vs n_ils per instance, annotated with significance."""
    agg: Dict[str, Dict[int, List[float]]] = {}
    for r in records:
        g = _float(r.get("gap_pct", ""))
        if math.isnan(g): continue
        agg.setdefault(r["instance"], {}).setdefault(_int(r["n_ils"]), []).append(g)

    if not agg: return
    palette    = plt.cm.tab10.colors
    ils_levels = sorted({k for d in agg.values() for k in d})
    fig, ax    = plt.subplots(figsize=(9, 5))

    for ci, (inst, ils_dict) in enumerate(sorted(agg.items())):
        means = [float(np.mean(ils_dict[k])) if k in ils_dict else float("nan")
                 for k in ils_levels]
        valid = [not math.isnan(m) for m in means]
        x_v   = [ils_levels[i] for i, v in enumerate(valid) if v]
        y_v   = [means[i]      for i, v in enumerate(valid) if v]
        if not x_v: continue
        ax.plot(x_v, y_v, marker="o", color=palette[ci % 10],
                label=inst, linewidth=1.8)

    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("n_ils (ILS restarts per chain)", fontsize=10)
    ax.set_ylabel("Mean gap to optimum (%)", fontsize=10)
    ax.set_title("ILS Depth — Mean Gap per Instance", fontsize=11)
    ax.set_xticks(ils_levels)
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.22)
    plt.tight_layout()
    p = out_dir / "nils_gain.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")


# ═══════════════════════════════════════════════════════════════════════════
# 6.  EXPERIMENT 3 — PARAMETER ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def analyse_exp3(records: List[dict]
                 ) -> Dict[str, Tuple[SAParams, float]]:
    """
    Derive best SAParams per instance from the 2^4 factorial records.

    Two-stage selection:
      Stage A — Main effects: for each factor, choose the level (lo/hi) that
                minimises mean gap.  This gives a recommended direction
                without overfitting to a single corner.
      Stage B — Best observed corner: find the factorial corner whose mean
                gap (across reps) is lowest.  Use Stage-A as a tiebreaker
                when two corners are within 0.05% of each other.

    The Stage-B winner is reported as the recommended SAParams.

    Returns: {instance: (SAParams, best_mean_gap)}
    """
    coded_keys  = ["coded_alpha", "coded_Ni", "coded_Imax", "coded_Mstag"]
    param_keys  = ["alpha", "N_iter", "I_max", "M_stag"]

    # Bucket by (instance, combo_label)
    combos: Dict[str, Dict[str, Dict]] = {}
    for r in records:
        inst  = r["instance"]
        label = r["combo_label"]
        g     = _float(r.get("gap_pct", ""))
        if math.isnan(g) or label == "CENTER": continue
        combos.setdefault(inst, {}).setdefault(label, {"gaps": [], "params": {}})
        combos[inst][label]["gaps"].append(g)
        # Store the concrete parameter values (same across reps for a label)
        if not combos[inst][label]["params"]:
            combos[inst][label]["params"] = {
                k: _float(r[k]) for k in param_keys
            }
            combos[inst][label]["coded"] = {
                ck: _int(r[ck]) for ck in coded_keys
            }

    # Stage A — main effect direction per factor
    def _stage_a_direction(inst_combos: dict) -> Dict[str, int]:
        """Return +1 (high) or -1 (low) for each factor, based on mean gap."""
        directions = {}
        for fi, (fname, ck) in enumerate(zip(FACTOR_LABELS, coded_keys)):
            hi_gaps = [np.mean(d["gaps"]) for d in inst_combos.values()
                       if d["coded"].get(ck) == +1 and d["gaps"]]
            lo_gaps = [np.mean(d["gaps"]) for d in inst_combos.values()
                       if d["coded"].get(ck) == -1 and d["gaps"]]
            if not hi_gaps or not lo_gaps:
                directions[fname] = 0
            else:
                directions[fname] = -1 if np.mean(lo_gaps) < np.mean(hi_gaps) else +1
        return directions

    results: Dict[str, Tuple[SAParams, float]] = {}

    for inst, inst_combos in combos.items():
        if not inst_combos:
            continue
        mean_gaps = {lbl: float(np.mean(d["gaps"]))
                     for lbl, d in inst_combos.items() if d["gaps"]}
        if not mean_gaps:
            continue

        stage_a = _stage_a_direction(inst_combos)

        # Stage B — best corner
        best_gap   = min(mean_gaps.values())
        candidates = [lbl for lbl, g in mean_gaps.items()
                      if g <= best_gap + 0.05]

        # Tiebreak: choose candidate whose coded vector best agrees with Stage-A
        def _agreement(lbl):
            coded = inst_combos[lbl]["coded"]
            return sum(1 for fn, ck in zip(FACTOR_LABELS, coded_keys)
                       if stage_a.get(fn, 0) != 0
                       and coded.get(ck) == stage_a[fn])

        best_lbl = max(candidates, key=_agreement)
        bp       = inst_combos[best_lbl]["params"]

        sa_p = SAParams(
            alpha  = bp["alpha"],
            N_iter = int(bp["N_iter"]),
            T_min  = 1e-5 if _float(
                next(r["n"] for r in records if r["instance"] == inst)
            ) > 100 else 1e-4,
            I_max  = int(bp["I_max"]),
            M_stag = int(bp["M_stag"]),
        )
        results[inst] = (sa_p, mean_gaps[best_lbl])

    return results


def plot_main_effects_summary(records: List[dict], out_dir: Path) -> None:
    """
    Single figure: for each factor, plot mean gap at low vs high level
    across all instances, faceted by factor.  Shows universal trends.
    """
    coded_keys = ["coded_alpha", "coded_Ni", "coded_Imax", "coded_Mstag"]
    instances  = sorted({r["instance"] for r in records
                         if r.get("combo_label") != "CENTER"})

    fig, axes = plt.subplots(1, len(FACTOR_LABELS),
                             figsize=(4.5 * len(FACTOR_LABELS), 4),
                             sharey=False)
    palette = plt.cm.tab10.colors

    for ci, (fname, ck) in enumerate(zip(FACTOR_LABELS, coded_keys)):
        ax = axes[ci]
        for ri, inst in enumerate(instances):
            lo = [_float(r.get("gap_pct", "")) for r in records
                  if r["instance"] == inst and _int(r.get(ck, "0")) == -1
                  and r.get("combo_label") != "CENTER"
                  and not math.isnan(_float(r.get("gap_pct", "")))]
            hi = [_float(r.get("gap_pct", "")) for r in records
                  if r["instance"] == inst and _int(r.get(ck, "0")) == +1
                  and r.get("combo_label") != "CENTER"
                  and not math.isnan(_float(r.get("gap_pct", "")))]
            if not lo or not hi: continue
            ax.plot(["Low", "High"], [np.mean(lo), np.mean(hi)],
                    marker="o", color=palette[ri % 10],
                    linewidth=1.4, alpha=0.85, label=inst)

        ax.set_title(fname, fontsize=10)
        ax.set_ylabel("Mean gap (%)", fontsize=9)
        ax.axhline(0, color="gray", lw=0.6, ls=":")
        ax.grid(alpha=0.25)
        if ci == 0:
            ax.legend(fontsize=7, ncol=1, loc="best")

    fig.suptitle("Exp-3 — Main Effects Summary (all instances)", fontsize=11)
    plt.tight_layout()
    p = out_dir / "main_effects_summary.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")


# ═══════════════════════════════════════════════════════════════════════════
# 7.  MERGE INTO OptConfig PER INSTANCE
# ═══════════════════════════════════════════════════════════════════════════

def build_opt_configs(
    exp1_results: Dict[str, Tuple[str, float, float]],
    exp2_results: Dict[str, Tuple[int, float, bool]],
    exp3_results: Dict[str, Tuple[SAParams, float]],
    instances:    Dict[str, Tuple[ALPInstance, float]],
) -> Dict[str, OptConfig]:
    """
    Merge analysis results into one OptConfig per instance.

    When an experiment produced no records for an instance (e.g. Exp-2
    only covers n >= 50), the corresponding fields retain their dataclass
    defaults (adaptive_params values).
    """
    configs: Dict[str, OptConfig] = {}

    for name, (inst, opt) in instances.items():
        sa_default, n_ils_default = adaptive_params(inst.n)

        h_name, h_gap, h_range = exp1_results.get(name, ("EDD", float("nan"), 0.0))
        n_ils, ils_gap, ils_sig = exp2_results.get(name, (n_ils_default, float("nan"), False))
        sa_opt, e3_gap          = exp3_results.get(name,
                                      (sa_default, float("nan")))

        notes = []
        if not math.isnan(h_range):
            if h_range > 1.0:
                notes.append(f"strong heuristic effect (range={h_range:.2f}%)")
            elif h_range < 0.1:
                notes.append("heuristic seed has negligible influence")
        if ils_sig:
            notes.append(f"ILS restarts helpful (n_ils={n_ils})")
        else:
            notes.append("ILS restarts have marginal effect")

        cfg = OptConfig(
            instance              = name,
            n                     = inst.n,
            known_opt             = opt,
            best_heuristic        = h_name,
            exp1_mean_gap         = h_gap,
            best_n_ils            = n_ils,
            exp2_mean_gap         = ils_gap,
            best_alpha            = sa_opt.alpha,
            best_N_iter           = sa_opt.N_iter,
            best_I_max            = sa_opt.I_max,
            best_M_stag           = sa_opt.M_stag,
            exp3_best_gap         = e3_gap,
            heuristic_significant = h_range > 1.0,
            ils_significant       = ils_sig,
            notes                 = "; ".join(notes),
        )
        configs[name] = cfg

    return configs


# ═══════════════════════════════════════════════════════════════════════════
# 8.  OPTIMISED BENCHMARK RUN
# ═══════════════════════════════════════════════════════════════════════════

def _safe_n_workers(n: int, n_workers_max: int) -> int:
    """Scale down workers for large instances to avoid OOM on the LP matrix."""
    if n <= 100:  return n_workers_max
    if n <= 200:  return min(n_workers_max, 12)
    if n <= 300:  return min(n_workers_max,  6)
    if n <= 400:  return min(n_workers_max,  3)
    return min(n_workers_max, 2)


def _gap_pct(f: float, opt: float) -> float:
    if math.isinf(f) or opt <= 0: return float("nan")
    return (f - opt) / opt * 100.0


def run_optimised_benchmark(
    configs:   Dict[str, OptConfig],
    instances: Dict[str, Tuple[ALPInstance, float]],
    t_limit:   float = 600.0,
    n_workers: int   = N_CPU,
    reps:      int   = 3,
) -> List[dict]:
    """
    Run ms_sa on every instance using its OptConfig-derived settings.

    Each (instance × replication) uses:
      - SAParams from OptConfig.sa_params()
      - n_ils from OptConfig.best_n_ils
      - Starting sequence pool seeded partly by the best heuristic
        (via a modified _build_starts call that prioritises the
         recommended heuristic start)
      - Per-instance worker cap to prevent OOM on large instances

    Multiple replications are averaged to give stable estimates.

    Returns a list of benchmark result dicts.
    """
    all_results = []

    for name, cfg in configs.items():
        if name not in instances:
            continue
        inst, opt   = instances[name]
        sa_p        = cfg.sa_params()
        n_ils       = cfg.best_n_ils
        w           = _safe_n_workers(inst.n, n_workers)
        h_fn        = HEURISTIC_GEN.get(cfg.best_heuristic, gen_edd)

        print(f"\n  ── {name}  (n={inst.n}, opt={opt}) ──")
        print(f"     Heuristic: {cfg.best_heuristic}   n_ils: {n_ils}   "
              f"α: {sa_p.alpha}   N_iter: {sa_p.N_iter}   "
              f"I_max: {sa_p.I_max}   M_stag: {sa_p.M_stag}   "
              f"workers: {w}")
        print(f"  {'Rep':>4} {'Obj (feas)':>18} {'Gap%':>8} "
              f"{'T_best(s)':>10} {'Wall(s)':>8}")
        print(f"  {'─'*4} {'─'*18} {'─'*8} {'─'*10} {'─'*8}")

        rep_objs, rep_gaps, rep_ttbs = [], [], []

        for rep in range(reps):
            # Build starts: ensure the recommended heuristic is first in the
            # pool so it always gets a chain regardless of pool ordering.
            starts  = _build_starts(inst, n_starts=w)
            # Inject the recommended heuristic as the very first start
            h_seq   = h_fn(inst)
            starts  = [(cfg.best_heuristic, h_seq, (rep + 1) * 97)] + \
                      [s for s in starts
                       if s[0] != cfg.best_heuristic][:w - 1]

            t0         = time.perf_counter()
            t_deadline = t0 + t_limit
            tasks      = [(lbl, seq, inst, sa_p, sd + rep * 1000, n_ils, t_deadline)
                          for lbl, seq, sd in starts]

            try:
                with ProcessPoolExecutor(max_workers=w, mp_context=_CTX) as ex:
                    results_raw = list(ex.map(_sa_worker, tasks))
            except Exception as exc:
                warnings.warn(f"Pool failed for {name} rep {rep+1}: {exc}. "
                              "Falling back to sequential.")
                results_raw = [_sa_worker(t) for t in tasks]

            wall = time.perf_counter() - t0

            feas = [r for r in results_raw if not math.isinf(r[4])]
            if feas:
                best_r   = min(feas, key=lambda r: r[4])
                fb_feas  = best_r[4]
                fb_semi  = best_r[2]
                t_best   = best_r[7]
                has_feas = True
            else:
                best_r   = min(results_raw, key=lambda r: r[2])
                fb_feas  = float("inf")
                fb_semi  = best_r[2]
                t_best   = best_r[6]
                has_feas = False

            g = _gap_pct(fb_feas, opt)
            rep_objs.append(fb_feas)
            rep_gaps.append(g)
            rep_ttbs.append(t_best)

            f_disp = _fmt_obj(fb_feas, fb_semi)
            g_disp = f"{g:+.3f}%" if not math.isnan(g) else "N/A"
            print(f"  {rep+1:>4d} {f_disp:>18} {g_disp:>8} "
                  f"{t_best:>10.2f} {wall:>8.2f}")

        # Aggregate across reps
        valid_objs = [o for o in rep_objs if not math.isinf(o)]
        valid_gaps = [g for g in rep_gaps if not math.isnan(g)]
        mean_obj   = float(np.mean(valid_objs)) if valid_objs else float("inf")
        mean_gap   = float(np.mean(valid_gaps)) if valid_gaps else float("nan")
        std_gap    = float(np.std(valid_gaps))  if len(valid_gaps) > 1 else 0.0
        best_obj   = min(valid_objs) if valid_objs else float("inf")

        print(f"  ── Summary: mean_obj={_fmt_obj(mean_obj, fb_semi)}"
              f"  best={_fmt_obj(best_obj, fb_semi)}"
              f"  mean_gap={mean_gap:+.3f}% ──")

        all_results.append({
            "instance":   name,
            "n":          inst.n,
            "known_opt":  opt,
            "heuristic":  cfg.best_heuristic,
            "n_ils":      n_ils,
            "alpha":      sa_p.alpha,
            "N_iter":     sa_p.N_iter,
            "I_max":      sa_p.I_max,
            "M_stag":     sa_p.M_stag,
            "mean_obj":   mean_obj,
            "best_obj":   best_obj,
            "mean_gap":   mean_gap,
            "std_gap":    std_gap,
            "mean_ttb":   float(np.mean(rep_ttbs)) if rep_ttbs else 0.0,
            "reps":       reps,
        })

    return all_results


# ═══════════════════════════════════════════════════════════════════════════
# 9.  COMPARISON VS ADAPTIVE-DEFAULT
# ═══════════════════════════════════════════════════════════════════════════

def run_default_benchmark(
    instances: Dict[str, Tuple[ALPInstance, float]],
    t_limit:   float = 600.0,
    n_workers: int   = N_CPU,
    reps:      int   = 3,
) -> List[dict]:
    """
    Run ms_sa with adaptive_params defaults for a fair comparison baseline.
    Identical t_limit, reps, and worker cap as the optimised run.
    """
    all_results = []

    for name, (inst, opt) in instances.items():
        sa_p, n_ils = adaptive_params(inst.n)
        w           = _safe_n_workers(inst.n, n_workers)

        print(f"\n  ── {name}  (default)  n_ils={n_ils} ──")
        rep_objs, rep_gaps = [], []

        for rep in range(reps):
            starts     = _build_starts(inst, n_starts=w)
            t0         = time.perf_counter()
            t_deadline = t0 + t_limit
            tasks      = [(lbl, seq, inst, sa_p, sd + rep * 1000, n_ils, t_deadline)
                          for lbl, seq, sd in starts]

            try:
                with ProcessPoolExecutor(max_workers=w, mp_context=_CTX) as ex:
                    results_raw = list(ex.map(_sa_worker, tasks))
            except Exception as exc:
                results_raw = [_sa_worker(t) for t in tasks]

            feas = [r for r in results_raw if not math.isinf(r[4])]
            fb_feas = min(feas, key=lambda r: r[4])[4] if feas else float("inf")
            fb_semi = min(results_raw, key=lambda r: r[2])[2]

            rep_objs.append(fb_feas)
            rep_gaps.append(_gap_pct(fb_feas, opt))

        valid_objs = [o for o in rep_objs if not math.isinf(o)]
        valid_gaps = [g for g in rep_gaps if not math.isnan(g)]
        mean_obj   = float(np.mean(valid_objs)) if valid_objs else float("inf")
        mean_gap   = float(np.mean(valid_gaps)) if valid_gaps else float("nan")
        std_gap    = float(np.std(valid_gaps))  if len(valid_gaps) > 1 else 0.0
        best_obj   = min(valid_objs) if valid_objs else float("inf")

        all_results.append({
            "instance":  name,
            "n":         inst.n,
            "known_opt": opt,
            "mean_obj":  mean_obj,
            "best_obj":  best_obj,
            "mean_gap":  mean_gap,
            "std_gap":   std_gap,
        })

    return all_results


def plot_opt_vs_default(
    opt_results:     List[dict],
    default_results: List[dict],
    out_dir:         Path,
) -> None:
    """Side-by-side mean gap: optimised vs adaptive-default."""
    def_map = {r["instance"]: r for r in default_results}
    names, gaps_opt, gaps_def = [], [], []
    for r in opt_results:
        d = def_map.get(r["instance"])
        if d is None: continue
        names.append(r["instance"])
        gaps_opt.append(r["mean_gap"])
        gaps_def.append(d["mean_gap"])

    if not names: return
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.1), 5))

    ax.bar(x - w/2, gaps_def, w, color="#aec6e8", edgecolor="#1a6faf",
           linewidth=0.8, alpha=0.80, label="Adaptive default")
    ax.bar(x + w/2, gaps_opt, w, color="#1a6faf", alpha=0.85,
           label="DOE-optimised")

    for xi, (gd, go) in enumerate(zip(gaps_def, gaps_opt)):
        delta = gd - go
        if not (math.isnan(gd) or math.isnan(go)) and abs(delta) > 0.05:
            color = "#2ecc71" if delta > 0 else "#e74c3c"
            ax.annotate(f"Δ{delta:+.2f}%",
                        xy=(xi + w/2, go),
                        xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=7, color=color, fontweight="bold")

    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Mean gap to optimum (%)", fontsize=10)
    ax.set_title("DOE-Optimised vs Adaptive Default — Mean Gap", fontsize=11)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.22)
    plt.tight_layout()
    p = out_dir / "opt_vs_default.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")


# ═══════════════════════════════════════════════════════════════════════════
# 10.  EXPORT
# ═══════════════════════════════════════════════════════════════════════════

def _write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"  Saved: {path}")


def export_recommendations(
    configs:     Dict[str, OptConfig],
    opt_results: List[dict],
    out_dir:     Path,
) -> None:
    rows = []
    opt_map = {r["instance"]: r for r in opt_results}
    for name, cfg in configs.items():
        r = opt_map.get(name, {})
        rows.append({
            "instance":              name,
            "n":                     cfg.n,
            "known_opt":             cfg.known_opt,
            "best_heuristic":        cfg.best_heuristic,
            "exp1_mean_gap":         f"{cfg.exp1_mean_gap:.4f}" if not math.isnan(cfg.exp1_mean_gap) else "nan",
            "heuristic_significant": cfg.heuristic_significant,
            "best_n_ils":            cfg.best_n_ils,
            "ils_significant":       cfg.ils_significant,
            "best_alpha":            cfg.best_alpha,
            "best_N_iter":           cfg.best_N_iter,
            "best_I_max":            cfg.best_I_max,
            "best_M_stag":           cfg.best_M_stag,
            "exp3_best_gap":         f"{cfg.exp3_best_gap:.4f}" if not math.isnan(cfg.exp3_best_gap) else "nan",
            "optimised_mean_gap":    f"{r.get('mean_gap', float('nan')):.4f}" if not math.isnan(r.get("mean_gap", float("nan"))) else "nan",
            "optimised_best_obj":    f"{r.get('best_obj', float('inf')):.4f}" if not math.isinf(r.get("best_obj", float("inf"))) else "inf",
            "notes":                 cfg.notes,
        })
    _write_csv(out_dir / "recommendations.csv", rows, list(rows[0].keys()) if rows else [])


def export_comparison(
    opt_results:     List[dict],
    default_results: List[dict],
    out_dir:         Path,
) -> None:
    def_map = {r["instance"]: r for r in default_results}
    rows = []
    for r in opt_results:
        d = def_map.get(r["instance"], {})
        g_opt = r.get("mean_gap", float("nan"))
        g_def = d.get("mean_gap", float("nan"))
        delta = g_def - g_opt if not (math.isnan(g_opt) or math.isnan(g_def)) else float("nan")
        rows.append({
            "instance":      r["instance"],
            "n":             r["n"],
            "known_opt":     r["known_opt"],
            "default_gap":   f"{g_def:.4f}" if not math.isnan(g_def) else "nan",
            "optimised_gap": f"{g_opt:.4f}" if not math.isnan(g_opt) else "nan",
            "delta_gap":     f"{delta:+.4f}" if not math.isnan(delta) else "nan",
            "default_obj":   f"{d.get('mean_obj', float('inf')):.4f}" if not math.isinf(d.get("mean_obj", float("inf"))) else "inf",
            "optimised_obj": f"{r.get('mean_obj', float('inf')):.4f}" if not math.isinf(r.get("mean_obj", float("inf"))) else "inf",
        })
    _write_csv(out_dir / "benchmark_cmp.csv", rows,
               ["instance", "n", "known_opt",
                "default_gap", "optimised_gap", "delta_gap",
                "default_obj", "optimised_obj"])


def write_analysis_report(
    configs:         Dict[str, OptConfig],
    opt_results:     List[dict],
    default_results: List[dict],
    out_dir:         Path,
) -> None:
    opt_map = {r["instance"]: r for r in opt_results}
    def_map = {r["instance"]: r for r in default_results}
    lines   = ["ALP — DOE Analysis Report", "=" * 68, ""]

    for name, cfg in sorted(configs.items(), key=lambda x: x[1].n):
        o = opt_map.get(name, {}); d = def_map.get(name, {})
        g_opt = o.get("mean_gap", float("nan"))
        g_def = d.get("mean_gap", float("nan"))
        delta = g_def - g_opt if not (math.isnan(g_opt) or math.isnan(g_def)) else float("nan")

        lines += [
            f"Instance : {name}  (n={cfg.n}, known_opt={cfg.known_opt})",
            "─" * 60,
            f"  Heuristic seed : {cfg.best_heuristic}"
            + (f"  [Δrange={cfg.exp1_mean_gap:.3f}%, SIGNIFICANT]"
               if cfg.heuristic_significant else "  [effect negligible]"),
            f"  ILS restarts   : {cfg.best_n_ils}"
            + (" [SIGNIFICANT]" if cfg.ils_significant else " [marginal]"),
            f"  SAParams       : α={cfg.best_alpha}  N_iter={cfg.best_N_iter}"
            f"  I_max={cfg.best_I_max}  M_stag={cfg.best_M_stag}",
            f"  Notes          : {cfg.notes}",
            "",
            f"  Default  mean gap : {g_def:+.3f}%" if not math.isnan(g_def) else "  Default  mean gap : N/A",
            f"  Optimised mean gap: {g_opt:+.3f}%" if not math.isnan(g_opt) else "  Optimised mean gap: N/A",
        ]
        if not math.isnan(delta):
            direction = "improvement" if delta > 0 else "regression"
            lines.append(f"  Δ gap             : {delta:+.3f}% ({direction})")
        lines += [""]

    path = out_dir / "analysis_report.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# 11.  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── User settings ────────────────────────────────────────────────────
    T_LIMIT   = 600.0   # wall-clock budget per instance per rep (seconds)
    REPS      = 3       # replications for each benchmark run
    N_WORKERS = N_CPU   # maximum parallel chains

    # Set to True to re-run the default benchmark for fair comparison.
    # Set to False to skip if you already have default results elsewhere.
    RUN_DEFAULT_BENCHMARK = True

    # ── Paths ────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_dir = OUT_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "═" * 72)
    print("  ALP — DOE ANALYSIS & OPTIMISED BENCHMARK")
    print(f"  DOE results dir : {DOE_DIR.resolve()}")
    print(f"  Analysis out dir: {OUT_DIR.resolve()}")
    print(f"  t_limit/rep     : {T_LIMIT:.0f}s   reps: {REPS}   workers: {N_WORKERS}")
    print("═" * 72)

    # ── Load OR-Library instances ────────────────────────────────────────
    print("\nLoading instances...")
    instances: Dict[str, Tuple[ALPInstance, float]] = {}
    for name, opt in OR_DATA.items():
        p = DATA_DIR / f"{name}.txt"
        if p.exists():
            try:
                inst = load_orlib(str(p), name)
                instances[name] = (inst, opt)
                print(f"  [LOADED]  {name:12s}  n={inst.n:4d}")
            except Exception as exc:
                print(f"  [ERROR]   {name}: {exc}")
        else:
            print(f"  [MISSING] {name}")

    if not instances:
        print("No instances found — check DATA_DIR.  Exiting."); return

    # ── Load DOE records ─────────────────────────────────────────────────
    print("\nLoading DOE records...")
    exp1_records = _load_csv(DOE_DIR / "exp1_heuristic" / "records.csv")
    exp2_records = _load_csv(DOE_DIR / "exp2_ils_depth" / "records.csv")
    exp3_records = _load_csv(DOE_DIR / "exp3_parameter" / "records.csv")
    print(f"  Exp-1: {len(exp1_records)} records")
    print(f"  Exp-2: {len(exp2_records)} records")
    print(f"  Exp-3: {len(exp3_records)} records")

    # ── Per-experiment analysis ──────────────────────────────────────────
    print("\n── Exp-1: Heuristic analysis ──")
    exp1_res = analyse_exp1(exp1_records)
    for name, (h, g, rng) in sorted(exp1_res.items()):
        g_s = f"{g:+.3f}%" if not math.isnan(g) else "N/A"
        print(f"  {name:12s}  best: {h:8s}  mean_gap={g_s}  range={rng:.3f}%")

    print("\n── Exp-2: ILS depth analysis ──")
    exp2_res = analyse_exp2(exp2_records)
    for name, (k, g, sig) in sorted(exp2_res.items()):
        g_s = f"{g:+.3f}%" if not math.isnan(g) else "N/A"
        print(f"  {name:12s}  best n_ils={k}  mean_gap={g_s}  significant={sig}")

    print("\n── Exp-3: Parameter analysis ──")
    exp3_res = analyse_exp3(exp3_records)
    for name, (sa_p, g) in sorted(exp3_res.items()):
        g_s = f"{g:+.3f}%" if not math.isnan(g) else "N/A"
        print(f"  {name:12s}  α={sa_p.alpha}  N_iter={sa_p.N_iter}"
              f"  I_max={sa_p.I_max}  M_stag={sa_p.M_stag}  best_gap={g_s}")

    # ── Build OptConfig per instance ─────────────────────────────────────
    print("\n── Building per-instance OptConfigs ──")
    configs = build_opt_configs(exp1_res, exp2_res, exp3_res, instances)

    # ── Analysis plots ───────────────────────────────────────────────────
    print("\nGenerating analysis plots...")
    if exp1_records: plot_heuristic_ranking(exp1_records, plot_dir)
    if exp2_records: plot_nils_gain(exp2_records, plot_dir)
    if exp3_records: plot_main_effects_summary(exp3_records, plot_dir)

    # ── Optimised benchmark ──────────────────────────────────────────────
    print("\n" + "═" * 72)
    print("  OPTIMISED BENCHMARK RUN")
    print("═" * 72)
    opt_results = run_optimised_benchmark(
        configs, instances, t_limit=T_LIMIT, n_workers=N_WORKERS, reps=REPS)

    # ── Default benchmark ────────────────────────────────────────────────
    default_results = []
    if RUN_DEFAULT_BENCHMARK:
        print("\n" + "═" * 72)
        print("  DEFAULT BENCHMARK RUN  (adaptive_params baseline)")
        print("═" * 72)
        default_results = run_default_benchmark(
            instances, t_limit=T_LIMIT, n_workers=N_WORKERS, reps=REPS)

    # ── Comparison plot ──────────────────────────────────────────────────
    if default_results:
        plot_opt_vs_default(opt_results, default_results, plot_dir)

    # ── Print comparison table ────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print("  RESULTS COMPARISON")
    print(f"  {'Instance':<14} {'Default gap%':>14} {'Optimised gap%':>16} {'Δgap%':>8}")
    print("  " + "─" * 56)
    def_map = {r["instance"]: r for r in default_results}
    for r in sorted(opt_results, key=lambda x: x["n"]):
        d     = def_map.get(r["instance"], {})
        g_opt = r.get("mean_gap", float("nan"))
        g_def = d.get("mean_gap", float("nan"))
        delta = g_def - g_opt if not (math.isnan(g_opt) or math.isnan(g_def)) else float("nan")
        g_o_s = f"{g_opt:+.3f}%" if not math.isnan(g_opt) else "N/A"
        g_d_s = f"{g_def:+.3f}%" if not math.isnan(g_def) else "N/A"
        d_s   = f"{delta:+.3f}%" if not math.isnan(delta) else "N/A"
        print(f"  {r['instance']:<14} {g_d_s:>14} {g_o_s:>16} {d_s:>8}")
    print("═" * 72)

    # ── Export ───────────────────────────────────────────────────────────
    print("\nExporting results...")
    export_recommendations(configs, opt_results, OUT_DIR)
    if default_results:
        export_comparison(opt_results, default_results, OUT_DIR)
    write_analysis_report(configs, opt_results, default_results, OUT_DIR)

    print(f"\n{'═'*72}")
    print(f"  Analysis complete.  All outputs in: {OUT_DIR.resolve()}")
    print(f"{'═'*72}\n")


if __name__ == "__main__":
    main()