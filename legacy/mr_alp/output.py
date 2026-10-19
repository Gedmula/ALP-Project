"""
output.py — MR-ALP Solver: Reporting, Persistence, and Visualisation
=====================================================================
§28  BKS-aware console reporting  (print_mr_result, print_summary_table)
§29  File persistence  (save_run_results and its constituent writers)
§30  Visualisation  (generate_plots and seven plot functions)

Output directory layout
-----------------------
OUTPUT_DIR/
  summary.csv
  schedules.csv
  alternatives.csv
  verification.txt
  run_metadata.json
  plots/
    gap/           gap_summary.png
    convergence/   convergence_{inst}_{m}.png
    lp_timeline/   lp_timeline_{inst}_{m}.png
    time_to_best/  time_to_best.png
    elite_pool/    elite_pool_{inst}_{m}.png
    gantt/         gantt_{inst}_{m}.png
    seeds/         seeds_{inst}_{m}.png

Timing columns added to summary.csv (§29)
-----------------------------------------
t_seed_construct_s : wall time for all seed constructions.
t_seed_lp_eval_s   : wall time for all seed LP evaluations.
t_best_seed_lp_s   : job-relative time when the best seed LP was achieved.
t_sa_start_s       : job-relative time when SA chains were dispatched.
t_best_lp_s        : job-relative total time-to-best (earliest of seed or SA).
"""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from mr_alp.config  import (
    KNOWN_OPTIMA, OUTPUT_DIR,
    N_WORKERS, N_CHAINS, T_LIMIT, MAX_T_LIMIT,
    ELITE_POOL_MAX, ELITE_MIN_DIV,
    RUN_RBI_OPTUNA, RUN_SA_OPTUNA,
    N_RBI_TRIALS_BASE, SA_N_TRIALS_BASE,
    ATC_K, ATCS_K1, ATCS_K2, GRASP_K_VALUES, MPDS_MAX_N,
)
from mr_alp.models  import MRSAParams

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    _MPL = True
except ImportError:
    plt = mticker = None; _MPL = False

_PLOT_STYLE = {
    "figure.facecolor": "white", "axes.facecolor": "#f7f7f7",
    "axes.grid": True,  "grid.color": "white", "grid.linewidth": 0.8,
    "font.size": 10,    "axes.titlesize": 11,  "axes.labelsize": 10,
}


# ═══════════════════════════════════════════════════════════════════════════
#   §28  BKS-AWARE REPORTING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def _gap_str(obj: float, ref: Optional[float], mark_new: bool = True) -> str:
    if ref is None:  return "N/A"
    if ref == 0.0:   return "0.00%" if obj < 1e-6 else "∞"
    gap = 100.0 * (obj - ref) / ref
    if gap < -0.001 and mark_new:
        return f"{gap:.2f}% ★"
    return f"{gap:.2f}%"


def _is_new_bks(obj: float, ref: Optional[float]) -> bool:
    if ref is None or ref <= 0.0:
        return False
    return obj < ref - 1e-6


def print_mr_result(
    inst, m, seqs, lp_obj, elapsed, seed_lps, params, p_sa, stats=None
) -> None:
    """
    Print a BKS-aware per-instance result report to stdout.

    Reports a three-stage improvement chain:
      best_seed_raw  →  best_seed_LP  →  SA+VND+PR final
    with corresponding gap-to-BKS at each stage, plus timing breakdown.
    """
    from mr_alp.lp import verify_and_exact_obj
    feas_e, viol_e, earliest_obj, _ = verify_and_exact_obj(seqs, inst)
    ref     = KNOWN_OPTIMA.get(inst.name, {}).get(m)
    new_bks = _is_new_bks(lp_obj, ref)
    sep     = "=" * 74

    print(f"\n{sep}")
    print(f"  {inst.name.upper()}  |  n={inst.n}  |  m={m} runway(s)"
          + ("  ★ NEW BKS CANDIDATE ★" if new_bks else ""))
    print(sep)
    print(f"  Runtime (SA+PR+VND total)  : {elapsed:.2f} s")
    print(f"  TC-RBI params              : {params}")
    print(f"  SA params                  : {p_sa}")

    best_seed_raw = (stats.get('best_seed_raw', math.inf)
                     if stats else math.inf)
    best_seed_lp  = min(seed_lps) if seed_lps else math.inf

    print(f"  Best seed raw obj          : "
          + (f"{best_seed_raw:.4f}" if not math.isinf(best_seed_raw) else "inf"))
    print(f"  Best seed LP obj           : "
          + (f"{best_seed_lp:.4f}"  if not math.isinf(best_seed_lp)  else "inf"))
    print(f"  SA+VND+PR final LP         : {lp_obj:.4f}")
    print(f"  Earliest-time objective    : {earliest_obj:.4f}")

    if ref is not None:
        label = "BKS (opt=0)" if ref == 0.0 else "Reference / BKS"
        print(f"  {label:<26}: {ref:.4f}")
        print(f"  Gap chain (raw→seed LP→SA) : "
              f"{_gap_str(best_seed_raw, ref, False)} → "
              f"{_gap_str(best_seed_lp,  ref, False)} → "
              f"{_gap_str(lp_obj, ref)}")
    else:
        print(f"  Reference / BKS            : not available for m={m}")

    if stats:
        t_sc  = stats.get('t_seed_construct', 0.0)
        t_lpe = stats.get('t_seed_lp_eval',  0.0)
        t_bsl = stats.get('t_best_seed_lp',  0.0)
        t_sa  = stats.get('t_sa_start',      0.0)
        t_bl  = stats.get('t_best_lp',       0.0)
        print(f"  Seed construct time        : {t_sc:.2f} s")
        print(f"  Seed LP eval time          : {t_lpe:.2f} s")
        print(f"  Best seed LP achieved at   : {t_bsl:.2f} s (job-relative)")
        print(f"  SA dispatch offset         : {t_sa:.2f} s")
        print(f"  Total time to best LP      : {t_bl:.2f} s (job-relative)")
        print(f"  Elite pool size            : {stats.get('elite_pool_size', 'N/A')}")
        print(f"  Path relinking improved    : "
              f"{'Yes' if stats.get('relinking_improved') else 'No'}")
        portfolio = stats.get('seed_portfolio', [])
        if portfolio:
            n_sel = sum(1 for _, _, _, sel in portfolio if sel)
            print(f"  Seed portfolio             : {len(portfolio)} evaluated,"
                  f" {n_sel} selected")

    print(f"  Sequence feasibility       : {'PASS ✓' if feas_e else 'FAIL ✗'}")
    if not feas_e:
        for v in viol_e[:6]: print(f"    ✗ {v}")
        if len(viol_e) > 6:  print(f"    ... and {len(viol_e)-6} more")

    print("  Runway load:")
    for rho, seq in enumerate(seqs):
        print(f"    Runway {rho+1}: {len(seq):4d} aircraft  "
              f"seq=[{', '.join(str(j) for j in seq[:6])}"
              f"{',...' if len(seq) > 6 else ''}]")
    print(sep)


def print_summary_table(results: List[dict]) -> None:
    """Print a BKS-aware batch results table; flag new BKS candidates with ★."""
    col = ["Instance","n","m","Seed LP","Final LP","Reference",
           "Gap(seed)","Gap(SA)","BKS?","Feas","ttb(s)","Time(s)"]
    w   = [12,5,4,12,12,12,10,10,5,6,8,9]
    hdr = "  " + "".join(f"{c:>{w[i]}}" for i, c in enumerate(col))
    bar = "=" * len(hdr)
    print(f"\n{bar}\n  MR-ALP Solver — BATCH RESULTS\n{bar}")
    print(hdr); print("-" * len(hdr))

    for r in sorted(results, key=lambda x: (x["name"], x["m"])):
        ref = r["opt"]
        row = [r["name"], r["n"], r["m"],
               f"{r['seed_lp']:.4f}" if not math.isinf(r['seed_lp']) else "inf",
               f"{r['sa_lp']:.4f}"   if not math.isinf(r['sa_lp'])   else "inf",
               f"{ref:.4f}" if ref is not None else "N/A",
               _gap_str(r['seed_lp'], ref, False), _gap_str(r['sa_lp'], ref),
               "★" if _is_new_bks(r['sa_lp'], ref) else "",
               "✓" if r["feasible"] else "✗",
               f"{r.get('t_best_lp', 0.0):.1f}",
               f"{r['time']:.2f}"]
        print("  " + "".join(f"{str(v):>{w[i]}}" for i, v in enumerate(row)))

    print(bar)
    pos = [r for r in results if r["opt"] is not None and r["opt"] > 0
           and not math.isinf(r["sa_lp"])]
    if pos:
        sg  = [100.0 * (r["seed_lp"] - r["opt"]) / r["opt"] for r in pos
               if not math.isinf(r["seed_lp"])]
        ag  = [100.0 * (r["sa_lp"] - r["opt"]) / r["opt"] for r in pos]
        fc  = sum(1 for r in results if r["feasible"])
        nb  = sum(1 for r in results if _is_new_bks(r["sa_lp"], r["opt"]))
        print(f"  Feasible         : {fc}/{len(results)}")
        print(f"  New BKS cands    : {nb}")
        if sg: print(f"  Avg seed gap     : {np.mean(sg):.2f}%  Max: {max(sg):.2f}%")
        if ag: print(f"  Avg final gap    : {np.mean(ag):.2f}%  Max: {max(ag):.2f}%")
        if sg and ag: print(f"  Avg improvement  : {np.mean(sg)-np.mean(ag):.2f}pp")
    print(bar)


# ═══════════════════════════════════════════════════════════════════════════
#   §29  RESULT PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════════

def _ensure_dirs(output_dir: Path) -> None:
    for sub in ["gap","convergence","lp_timeline","time_to_best",
                "elite_pool","gantt","seeds"]:
        (output_dir / "plots" / sub).mkdir(parents=True, exist_ok=True)


def _save_summary_csv(results: List[dict], output_dir: Path) -> None:
    """Write summary.csv with timing columns for seed and SA phases."""
    path   = output_dir / "summary.csv"
    fields = [
        "instance","n","m",
        "seed_raw_obj","seed_lp","sa_lp","bks",
        "gap_seed_raw_pct","gap_seed_lp_pct","gap_sa_pct",
        "new_bks","feasible",
        "t_seed_construct_s","t_seed_lp_eval_s","t_best_seed_lp_s",
        "t_sa_start_s","t_best_lp_s","time_s",
        "elite_pool_size","relinking_improved",
        "best_seed_label","n_seeds_evaluated",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in sorted(results, key=lambda x: (x["name"], x["m"])):
            ref     = r["opt"]
            raw_obj = r.get("best_seed_raw", math.inf)
            gr  = (100.0*(raw_obj - ref)/ref
                   if ref and ref > 0 and not math.isinf(raw_obj) else None)
            gs  = (100.0*(r["seed_lp"] - ref)/ref
                   if ref and ref > 0 and not math.isinf(r["seed_lp"]) else None)
            ga  = (100.0*(r["sa_lp"] - ref)/ref
                   if ref and ref > 0 and not math.isinf(r["sa_lp"])   else None)
            portfolio = r.get("seed_portfolio", [])
            sel_seeds = [(lp, lbl) for lbl, _, lp, sel in portfolio
                         if sel and not math.isinf(lp)]
            best_label = (min(sel_seeds, key=lambda x: x[0])[1]
                          if sel_seeds else "")
            w.writerow({
                "instance":            r["name"],
                "n":                   r["n"],
                "m":                   r["m"],
                "seed_raw_obj":        "" if math.isinf(raw_obj) else f"{raw_obj:.6f}",
                "seed_lp":             "" if math.isinf(r["seed_lp"]) else f"{r['seed_lp']:.6f}",
                "sa_lp":               "" if math.isinf(r["sa_lp"])   else f"{r['sa_lp']:.6f}",
                "bks":                 "" if ref is None else ref,
                "gap_seed_raw_pct":    "" if gr is None else f"{gr:.4f}",
                "gap_seed_lp_pct":     "" if gs is None else f"{gs:.4f}",
                "gap_sa_pct":          "" if ga is None else f"{ga:.4f}",
                "new_bks":             _is_new_bks(r["sa_lp"], ref),
                "feasible":            r["feasible"],
                "t_seed_construct_s":  f"{r.get('t_seed_construct', 0.0):.3f}",
                "t_seed_lp_eval_s":    f"{r.get('t_seed_lp_eval',  0.0):.3f}",
                "t_best_seed_lp_s":    f"{r.get('t_best_seed_lp',  0.0):.3f}",
                "t_sa_start_s":        f"{r.get('t_sa_start',      0.0):.3f}",
                "t_best_lp_s":         f"{r.get('t_best_lp',       0.0):.3f}",
                "time_s":              f"{r['time']:.2f}",
                "elite_pool_size":     r.get("elite_pool_size", ""),
                "relinking_improved":  r.get("relinking_improved", False),
                "best_seed_label":     best_label,
                "n_seeds_evaluated":   len(portfolio),
            })
    print(f"  Saved {path}")


def _save_schedules_csv(results: List[dict], output_dir: Path) -> None:
    path = output_dir / "schedules.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance","m","rho","position","aircraft_j"])
        for r in sorted(results, key=lambda x: (x["name"], x["m"])):
            for rho, seq in enumerate(r.get("best_seqs", [])):
                for pos, j in enumerate(seq):
                    w.writerow([r["name"], r["m"], rho+1, pos+1, j])
    print(f"  Saved {path}")


def _save_alternatives_csv(results: List[dict], output_dir: Path) -> None:
    path = output_dir / "alternatives.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance","m","rank","lp_obj","rho","position","aircraft_j"])
        for r in sorted(results, key=lambda x: (x["name"], x["m"])):
            for rank, (lp_obj, seqs) in enumerate(r.get("elite_solutions",[]), 1):
                for rho, seq in enumerate(seqs):
                    for pos, j in enumerate(seq):
                        w.writerow([r["name"],r["m"],rank,f"{lp_obj:.6f}",
                                    rho+1,pos+1,j])
    print(f"  Saved {path}")


def _save_verification_txt(results: List[dict], output_dir: Path) -> None:
    """Feasibility audit + LP timeline + seed portfolio log for every job."""
    path = output_dir / "verification.txt"; sep = "=" * 72
    with open(path, "w") as f:
        f.write("MR-ALP Solver — FEASIBILITY VERIFICATION REPORT\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for r in sorted(results, key=lambda x: (x["name"], x["m"])):
            ref = r["opt"]
            f.write(f"{sep}\n  {r['name'].upper()}  |  n={r['n']}  |  m={r['m']}\n{sep}\n")
            f.write(f"  SA+VND+PR LP         : {r['sa_lp']:.6f}\n")
            if ref is not None:
                f.write(f"  Reference / BKS      : {ref}\n")
                f.write(f"  Gap to BKS           : {_gap_str(r['sa_lp'], ref)}\n")
                if _is_new_bks(r["sa_lp"], ref):
                    f.write("  *** NEW BKS CANDIDATE ***\n")
            f.write(f"  Sequence feasible    : {'YES' if r['feasible'] else 'NO'}\n")

            # Timing breakdown
            f.write(f"  Seed construct (s)   : {r.get('t_seed_construct', 0.0):.3f}\n")
            f.write(f"  Seed LP eval (s)     : {r.get('t_seed_lp_eval', 0.0):.3f}\n")
            f.write(f"  Best seed LP at (s)  : {r.get('t_best_seed_lp', 0.0):.3f}\n")
            f.write(f"  SA dispatch at (s)   : {r.get('t_sa_start', 0.0):.3f}\n")
            f.write(f"  Total time-to-best   : {r.get('t_best_lp', 0.0):.3f} s\n")
            f.write(f"  Total runtime (s)    : {r['time']:.2f}\n")

            viols = r.get("violations", [])
            if viols:
                f.write(f"  Violations ({len(viols)}):\n")
                for v in viols[:10]: f.write(f"    ✗ {v}\n")
                if len(viols) > 10:  f.write(f"    ... and {len(viols)-10} more\n")

            portfolio = r.get("seed_portfolio", [])
            if portfolio:
                f.write(f"  Seed portfolio ({len(portfolio)} heuristics evaluated):\n")
                for label, raw_obj, lp_val, selected in sorted(portfolio, key=lambda x: x[2]):
                    tag     = "  [SELECTED]" if selected else ""
                    raw_str = f"{raw_obj:.6f}" if not math.isinf(raw_obj) else "inf"
                    lp_str  = f"{lp_val:.6f}"  if not math.isinf(lp_val)  else "inf"
                    f.write(f"    {label:<12} raw={raw_str}  LP={lp_str}{tag}\n")

            # Per-seed timing block (new)
            seed_timing = r.get("seed_timing", [])
            if seed_timing:
                f.write("  Per-seed timing (construct_s / lp_eval_s / job_rel_s):\n")
                for st in seed_timing:
                    f.write(f"    {st['label']:<12}  "
                            f"construct={st['t_construct']:.3f}s  "
                            f"lp_eval={st['t_lp_eval']:.3f}s  "
                            f"@{st['t_job_relative']:.3f}s  "
                            f"LP={st['lp_val']:.4f}\n")

            tl = r.get("job_lp_timeline", [])
            if tl:
                f.write("  LP improvement timeline (job_rel_s, lp_val):\n")
                for t_s, lp_v in tl:
                    f.write(f"    t={t_s:8.3f}s  LP={lp_v:.6f}\n")
            f.write("\n")
    print(f"  Saved {path}")


def _save_run_metadata_json(results: List[dict], output_dir: Path) -> None:
    path = output_dir / "run_metadata.json"
    payload: Dict[str, Any] = {
        "run_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "N_WORKERS": N_WORKERS, "N_CHAINS": N_CHAINS,
            "T_LIMIT": T_LIMIT, "MAX_T_LIMIT": MAX_T_LIMIT,
            "ELITE_POOL_MAX": ELITE_POOL_MAX, "ELITE_MIN_DIV": ELITE_MIN_DIV,
            "RUN_RBI_OPTUNA": RUN_RBI_OPTUNA, "RUN_SA_OPTUNA": RUN_SA_OPTUNA,
            "ATC_K": ATC_K, "ATCS_K1": ATCS_K1, "ATCS_K2": ATCS_K2,
            "GRASP_K_VALUES": list(GRASP_K_VALUES), "MPDS_MAX_N": MPDS_MAX_N,
        },
        "results": [],
    }
    for r in sorted(results, key=lambda x: (x["name"], x["m"])):
        ref = r["opt"]
        gap = (100.0 * (r["sa_lp"] - ref) / ref
               if ref and ref > 0 and not math.isinf(r["sa_lp"]) else None)
        p   = r.get("p_sa", MRSAParams())
        portfolio = r.get("seed_portfolio", [])
        payload["results"].append({
            "instance": r["name"], "n": r["n"], "m": r["m"],
            "seed_lp":  None if math.isinf(r["seed_lp"]) else r["seed_lp"],
            "sa_lp":    None if math.isinf(r["sa_lp"])   else r["sa_lp"],
            "bks": ref, "gap_pct": round(gap, 4) if gap is not None else None,
            "new_bks":          _is_new_bks(r["sa_lp"], ref),
            "feasible":         r["feasible"],
            "timing": {
                "t_seed_construct_s": round(r.get("t_seed_construct", 0.0), 4),
                "t_seed_lp_eval_s":   round(r.get("t_seed_lp_eval",  0.0), 4),
                "t_best_seed_lp_s":   round(r.get("t_best_seed_lp",  0.0), 4),
                "t_sa_start_s":       round(r.get("t_sa_start",      0.0), 4),
                "t_best_lp_s":        round(r.get("t_best_lp",       0.0), 4),
                "total_s":            round(r["time"], 2),
            },
            "elite_pool_size":    r.get("elite_pool_size", 0),
            "n_elite_solutions":  len(r.get("elite_solutions", [])),
            "relinking_improved": r.get("relinking_improved", False),
            "n_seeds_evaluated":  len(portfolio),
            "seed_portfolio": [
                {"label":   lbl,
                 "raw_obj": None if math.isinf(raw) else round(raw, 6),
                 "lp_obj":  None if math.isinf(lp)  else round(lp,  6),
                 "selected": sel}
                for lbl, raw, lp, sel in portfolio
            ],
            "sa_params": {
                "chi0": p.chi0, "M_stag_frac": p.M_stag_frac,
                "lp_gamma": p.lp_gamma, "chi_target": p.chi_target,
                "optuna_tuned": r.get("p_sa_tuned", False),
            },
        })
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved {path}")


def save_run_results(results: List[dict], output_dir: Path) -> None:
    """Write all result files; create plot subdirectories if needed."""
    _ensure_dirs(output_dir)
    _save_summary_csv(results, output_dir)
    _save_schedules_csv(results, output_dir)
    _save_alternatives_csv(results, output_dir)
    _save_verification_txt(results, output_dir)
    _save_run_metadata_json(results, output_dir)


# ═══════════════════════════════════════════════════════════════════════════
#   §30  VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════

def _plot_gap_summary(results: List[dict], output_dir: Path) -> None:
    """Grouped bar chart: seed LP gap vs final SA+VND+PR gap."""
    if not _MPL: return
    pos_r = sorted(
        [r for r in results
         if r["opt"] is not None and r["opt"] > 0
         and not math.isinf(r["sa_lp"])],
        key=lambda x: (x["name"], x["m"]))
    if not pos_r: return
    labels    = [f"{r['name']}\nm={r['m']}" for r in pos_r]
    seed_gaps = [100*(r["seed_lp"]-r["opt"])/r["opt"]
                 if not math.isinf(r["seed_lp"]) else 0 for r in pos_r]
    sa_gaps   = [100*(r["sa_lp"]-r["opt"])/r["opt"] for r in pos_r]
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(max(10, len(labels)*0.55+2), 5))
    with plt.rc_context(_PLOT_STYLE):
        ax.bar(x-w/2, seed_gaps, w, label="Seed LP gap",  color="#4878CF", alpha=0.85)
        ax.bar(x+w/2, sa_gaps,   w, label="Final LP gap", color="#D65F5F", alpha=0.85)
        for xi, r, sg in zip(x, pos_r, sa_gaps):
            if _is_new_bks(r["sa_lp"], r["opt"]):
                ax.text(xi+w/2, max(sg,0)+0.3, "★", ha="center", va="bottom",
                        color="goldenrod", fontsize=13, fontweight="bold")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
        ax.set_ylabel("Gap to BKS (%)"); ax.set_title("MR-ALP — Seed vs Final Gap to BKS")
        ax.legend(loc="upper right")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
        plt.tight_layout()
        out = output_dir / "plots" / "gap" / "gap_summary.png"
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {out}")


def _plot_convergence(result: dict, output_dir: Path) -> None:
    """SA proxy convergence history for the best chain (normalised to start)."""
    if not _MPL: return
    history = result.get("history", [])
    if not history: return
    name = result["name"]; m = result["m"]
    hist = np.asarray(history, dtype=float)
    if hist[0] != 0: hist = hist / hist[0]
    fig, ax = plt.subplots(figsize=(7, 4))
    with plt.rc_context(_PLOT_STYLE):
        ax.plot(hist, linewidth=0.8, color="#4878CF", alpha=0.9)
        ax.set_xlabel("Iteration"); ax.set_ylabel("Best proxy (relative)")
        ax.set_title(f"{name.upper()} m={m} — SA proxy convergence")
        plt.tight_layout()
        out = output_dir / "plots" / "convergence" / f"convergence_{name}_{m}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {out}")


def _plot_lp_timeline(result: dict, output_dir: Path) -> None:
    """
    Step-function LP objective vs job-relative wall time.

    Vertical dashed line at total_t_best; horizontal dashed line at BKS.
    The step function covers the full period from seed phase through SA,
    VND, and path relinking, giving a complete picture of when each
    improvement was achieved.
    """
    if not _MPL: return
    tl = result.get("job_lp_timeline", [])
    if len(tl) < 2: return
    name = result["name"]; m = result["m"]; ref = result.get("opt")
    ts   = [t for t, _ in tl] + [result["time"]]
    lps  = [v for _, v in tl] + [tl[-1][1]]
    fig, ax = plt.subplots(figsize=(8, 4))
    with plt.rc_context(_PLOT_STYLE):
        ax.step(ts, lps, where="post", linewidth=1.5, color="#D65F5F",
                label="LP objective")
        ax.scatter([t for t,_ in tl], [v for _,v in tl],
                   s=30, color="#D65F5F", zorder=5)
        # Shade seed vs SA regions
        t_sa = result.get("t_sa_start", 0.0)
        if t_sa > 0:
            ax.axvspan(0, t_sa, alpha=0.07, color="#4878CF", label="Seed phase")
            ax.axvspan(t_sa, result["time"], alpha=0.05, color="#D65F5F",
                       label="SA phase")
        if ref is not None and ref > 0:
            ax.axhline(ref, color="goldenrod", linewidth=1.2, linestyle="--",
                       label=f"BKS ({ref:.2f})")
        t_best = result.get("t_best_lp")
        if t_best and t_best > 0:
            ax.axvline(t_best, color="steelblue", linewidth=0.9, linestyle=":",
                       label=f"t-to-best ({t_best:.1f}s)")
        ax.set_xlabel("Wall time — job-relative (s)")
        ax.set_ylabel("LP objective")
        ax.set_title(f"{name.upper()} m={m} — LP improvement timeline")
        ax.legend(fontsize=8); plt.tight_layout()
        out = output_dir / "plots" / "lp_timeline" / f"lp_timeline_{name}_{m}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {out}")


def _plot_time_to_best(results: List[dict], output_dir: Path) -> None:
    """Scatter: job-relative time-to-best-LP vs final BKS gap; colour = runway count."""
    if not _MPL: return
    pos = [r for r in results
           if r["opt"] is not None and r["opt"] > 0
           and not math.isinf(r["sa_lp"]) and r.get("t_best_lp") is not None]
    if len(pos) < 2: return
    ts   = [r["t_best_lp"] for r in pos]
    gaps = [100*(r["sa_lp"]-r["opt"])/r["opt"] for r in pos]
    ms   = [r["m"] for r in pos]; m_vals = sorted(set(ms))
    cmap = plt.get_cmap("tab10")
    colours = {mv: cmap(i / max(len(m_vals)-1, 1)) for i, mv in enumerate(m_vals)}
    fig, ax = plt.subplots(figsize=(7, 5))
    with plt.rc_context(_PLOT_STYLE):
        for mv in m_vals:
            idx = [i for i, r in enumerate(pos) if r["m"] == mv]
            ax.scatter([ts[i] for i in idx], [gaps[i] for i in idx],
                       s=60, color=colours[mv], label=f"m={mv}",
                       alpha=0.85, edgecolors="white")
        for r, t, g in zip(pos, ts, gaps):
            if abs(g) > 5:
                ax.annotate(f"{r['name']}\nm={r['m']}", (t, g), fontsize=7,
                            ha="left", va="bottom", xytext=(4,4),
                            textcoords="offset points")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Total time to best LP — job-relative (s)")
        ax.set_ylabel("Final gap to BKS (%)")
        ax.set_title("MR-ALP — Time to best LP vs BKS gap")
        ax.legend(title="Runways", fontsize=9); plt.tight_layout()
        out = output_dir / "plots" / "time_to_best" / "time_to_best.png"
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {out}")


def _plot_elite_pool(result: dict, output_dir: Path) -> None:
    """Horizontal bar chart of elite pool LP objectives."""
    if not _MPL: return
    elite = result.get("elite_solutions", [])
    if len(elite) < 2: return
    name = result["name"]; m = result["m"]; ref = result.get("opt")
    lp_vals = [lp for lp, _ in elite[:20]]; ranks = list(range(1, len(lp_vals)+1))
    fig, ax = plt.subplots(figsize=(6, max(3, len(lp_vals)*0.28)))
    with plt.rc_context(_PLOT_STYLE):
        colours = ["#D65F5F" if (ref and ref > 0 and _is_new_bks(lp, ref))
                   else "#4878CF" for lp in lp_vals]
        bars = ax.barh(ranks, lp_vals, color=colours, alpha=0.85, edgecolor="white")
        if ref is not None and ref > 0:
            ax.axvline(ref, color="goldenrod", linewidth=1.2, linestyle="--",
                       label=f"BKS ({ref:.2f})"); ax.legend(fontsize=9)
        ax.set_yticks(ranks)
        ax.set_yticklabels([f"Rank {r}" for r in ranks], fontsize=8)
        ax.invert_yaxis(); ax.set_xlabel("LP objective")
        ax.set_title(f"{name.upper()} m={m} — Elite pool LP distribution")
        for bar, lp in zip(bars, lp_vals):
            ax.text(bar.get_width()*1.002, bar.get_y()+bar.get_height()/2,
                    f"{lp:.2f}", va="center", fontsize=7.5)
        plt.tight_layout()
        out = output_dir / "plots" / "elite_pool" / f"elite_pool_{name}_{m}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {out}")


def _plot_gantt(result: dict, output_dir: Path) -> None:
    """
    Gantt chart of the LP-optimal landing schedule.

    Three visual layers per aircraft:
      1. Grey time-window background span  [r_j, d_j].
      2. Colour-coded landing bar          [C_j, C_j + sep_width]:
           blue  = early  (C_j < δ_j − 1)
           green = on-time
           red   = late   (C_j > δ_j + 1)
      3. Black target tick at δ_j.

    Aircraft index labels shown when n ≤ 80, suppressed for larger instances.
    """
    if not _MPL: return
    C_lp      = result.get("C_lp")
    seqs      = result.get("best_seqs", [])
    r_arr     = result.get("r_arr")
    delta_arr = result.get("delta_arr")
    d_arr     = result.get("d_arr")
    s_mat     = result.get("s_mat")
    name      = result["name"]; m = result["m"]
    if C_lp is None or not seqs or r_arr is None: return

    n_rwy    = len(seqs)
    annotate = result.get("n", 0) <= 80
    fig, ax  = plt.subplots(figsize=(14, max(3.0, n_rwy*1.8+1.2)))
    with plt.rc_context(_PLOT_STYLE):
        for rho, seq in enumerate(seqs):
            if not seq: continue
            L = len(seq)
            for qi, j in enumerate(seq):
                cj  = float(C_lp[j])
                rj  = float(r_arr[j]);   dj  = float(d_arr[j])
                dej = float(delta_arr[j])
                ax.barh(rho, dj-rj, left=rj, height=0.65,
                        color="#cccccc", alpha=0.30, linewidth=0, zorder=1)
                min_bw = max((dj-rj)*0.04, 5.0)
                bw = (max(float(s_mat[j, seq[qi+1]]), min_bw)
                      if qi < L-1 else min_bw)
                color = ("#4878CF" if cj < dej - 1.0
                         else "#D65F5F" if cj > dej + 1.0
                         else "#6ACC65")
                ax.barh(rho, bw, left=cj, height=0.50, color=color,
                        alpha=0.90, linewidth=0.5, edgecolor="white", zorder=3)
                ax.plot(dej, rho, marker="|", color="black",
                        markersize=9, markeredgewidth=1.5, zorder=5)
                if annotate:
                    ax.text(cj+bw*0.5, rho, str(j), ha="center", va="center",
                            fontsize=6.5, color="white", fontweight="bold",
                            zorder=6)
        ax.set_yticks(range(n_rwy))
        ax.set_yticklabels(
            [f"Runway {rho+1}  (n={len(seqs[rho])})" for rho in range(n_rwy)],
            fontsize=9)
        ax.set_xlabel("Time")
        ax.set_title(f"{name.upper()}  |  m={m} — LP-optimal landing schedule")
        ax.set_ylim(-0.7, n_rwy - 0.3)
        from matplotlib.patches import Patch
        from matplotlib.lines   import Line2D
        ax.legend(handles=[
            Patch(facecolor="#cccccc", alpha=0.50, label="Time window [r_j,d_j]"),
            Patch(facecolor="#4878CF", label="Early  (C_j < δ_j)"),
            Patch(facecolor="#6ACC65", label="On-time"),
            Patch(facecolor="#D65F5F", label="Late   (C_j > δ_j)"),
            Line2D([0],[0], marker="|", color="black", linewidth=0,
                   markersize=10, markeredgewidth=1.5, label="Target δ_j"),
        ], loc="upper right", fontsize=8, framealpha=0.85)
        plt.tight_layout()
        out = output_dir / "plots" / "gantt" / f"gantt_{name}_{m}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {out}")


def _plot_seed_comparison(result: dict, output_dir: Path) -> None:
    """
    Horizontal bar chart comparing LP objectives across all evaluated seed
    heuristics.  Selected seeds are highlighted in dark orange; non-selected
    in grey.  A vertical dashed line marks the BKS reference when available.
    Bars sorted ascending so the best seed (lowest LP) appears at the top
    after invert_yaxis.

    Saved to: plots/seeds/seeds_{inst}_{m}.png
    """
    if not _MPL: return
    portfolio = result.get("seed_portfolio", [])
    if not portfolio: return
    name = result["name"]; m = result["m"]; ref = result.get("opt")

    finite = sorted([(lp, label, sel) for label, lp, sel in
                     [(lbl, lp, sel) for lbl, _, lp, sel in portfolio]
                     if not math.isinf(lp)], key=lambda x: x[0])
    inf_entries = [(lbl, sel) for lbl, _, lp, sel in portfolio if math.isinf(lp)]
    if not finite and not inf_entries: return

    labels   = [lbl for _, lbl, _   in finite] + [lbl for lbl, _   in inf_entries]
    lp_vals  = [lp  for lp, _,  _   in finite] + [None]*len(inf_entries)
    selected = [sel for _, _,  sel  in finite] + [sel for _,  sel  in inf_entries]
    bar_vals = [lp if lp is not None else 0.0 for lp in lp_vals]
    colours  = ["#D65F5F" if sel else "#aaaaaa" for sel in selected]

    fig, ax = plt.subplots(figsize=(8, max(3.0, len(labels)*0.45+1.5)))
    with plt.rc_context(_PLOT_STYLE):
        y_pos = list(range(len(labels)))
        bars  = ax.barh(y_pos, bar_vals, color=colours, alpha=0.88, edgecolor="white")
        if ref is not None and ref > 0:
            ax.axvline(ref, color="goldenrod", linewidth=1.3, linestyle="--",
                       label=f"BKS ({ref:.2f})")
        ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis(); ax.set_xlabel("LP objective")
        ax.set_title(f"{name.upper()} m={m} — Seed portfolio LP comparison")
        for bar, lp in zip(bars, lp_vals):
            if lp is not None:
                ax.text(bar.get_width()*1.002, bar.get_y()+bar.get_height()/2,
                        f"{lp:.2f}", va="center", fontsize=8)
            else:
                ax.text(0.01, bar.get_y()+bar.get_height()/2,
                        "infeasible", va="center", fontsize=8, color="#888888")
        from matplotlib.patches import Patch
        from matplotlib.lines   import Line2D
        handles = [
            Patch(facecolor="#D65F5F", alpha=0.88, label="Selected for SA"),
            Patch(facecolor="#aaaaaa", alpha=0.88, label="Not selected"),
        ]
        if ref is not None and ref > 0:
            handles.append(Line2D([0],[0], color="goldenrod", linewidth=1.3,
                                  linestyle="--", label=f"BKS ({ref:.2f})"))
        ax.legend(handles=handles, fontsize=8, loc="lower right")
        plt.tight_layout()
        out = output_dir / "plots" / "seeds" / f"seeds_{name}_{m}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {out}")


def generate_plots(results: List[dict], output_dir: Path) -> None:
    """
    Generate all plots for a completed batch run.

    Global:  gap_summary.png, time_to_best.png
    Per-job: convergence, lp_timeline, elite_pool, gantt, seeds
    """
    if not _MPL:
        print("  [plots] matplotlib not available — skipping."); return
    _ensure_dirs(output_dir)
    _plot_gap_summary(results, output_dir)
    _plot_time_to_best(results, output_dir)
    for r in results:
        _plot_convergence(r, output_dir)
        _plot_lp_timeline(r, output_dir)
        _plot_elite_pool(r, output_dir)
        _plot_gantt(r, output_dir)
        _plot_seed_comparison(r, output_dir)