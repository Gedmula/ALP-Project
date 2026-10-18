"""
output.py — Result formatting, CSV/JSON export, and visualisation.

Public interface
────────────────
print_result            : single-result console line including gap.
print_gap_report        : detailed gap report for one (instance, m) result.
print_runway_assignment : per-runway aircraft count, landing order, and
                          optional per-aircraft penalty breakdown.
print_summary_table     : formatted aggregate table across all results.
save_results            : writes summary.csv, verification.txt,
                          run_metadata.json.
plot_gantt              : runway-grouped Gantt chart with time windows,
                          separation zones, and landing/target markers.
plot_penalty_profile    : runway-grouped stacked earliness/tardiness chart.

Gap convention
──────────────
gap_pct = (lp_obj − known_opt) / known_opt × 100

A negative gap means the solver found a schedule better than the stored
reference value (valid — reference values for m > 1 are heuristic BKS,
not certified optima).  A gap of exactly 0.00 % means the known optimum
was matched.  "N/A" is shown when known_opt is None or lp_obj is inf.
"""
from __future__ import annotations

import csv
import json
import math
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from misc.mr_alp_2.config import OUTPUT_DIR
from misc.mr_alp_2.lp     import stage2_lp
from misc.mr_alp_2.models import Instance


# ─────────────────────────────────────────────────────────────────────────────
#   Gap helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_gap(lp_obj: float, known_opt: Optional[float]) -> float:
    """
    Return (lp_obj − known_opt) / known_opt × 100.

    Returns nan when known_opt is None, zero, or lp_obj is inf.
    """
    if known_opt is None or known_opt == 0.0 or math.isinf(lp_obj):
        return float('nan')
    return (lp_obj - known_opt) / abs(known_opt) * 100.0


def _gap_str(gap_pct: float) -> str:
    """Format a gap value for console or CSV output."""
    if math.isnan(gap_pct):
        return "N/A"
    return f"{gap_pct:+.4f}%"


def _gap_label(gap_pct: float) -> str:
    """Return a short qualitative label for the gap."""
    if math.isnan(gap_pct):
        return ""
    if gap_pct < -1e-4:
        return "  ← NEW BEST"
    if abs(gap_pct) < 1e-4:
        return "  ← OPTIMAL"
    return ""


# ─────────────────────────────────────────────────────────────────────────────
#   Internal formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _format_ac_list(seq: List[int], line_width: int = 56) -> List[str]:
    """
    Format a sequence of aircraft indices as wrapped text lines.

    Each item is rendered as 'A{j}'.  Lines are wrapped at line_width
    characters; continuation lines are returned as separate strings.
    """
    if not seq:
        return ["(empty)"]
    labels: List[str] = [f"A{j}" for j in seq]
    lines:  List[str] = []
    current            = ""
    for lbl in labels:
        piece = ("  " + lbl) if current else lbl
        if current and len(current) + len(piece) > line_width:
            lines.append(current)
            current = lbl
        else:
            current += piece
    if current:
        lines.append(current)
    return lines


# ─────────────────────────────────────────────────────────────────────────────
#   Console output — single-line summary
# ─────────────────────────────────────────────────────────────────────────────

def print_result(
    result:    Dict[str, Any],
    known_opt: Optional[float] = None,
) -> None:
    """Print a single-line result summary including the optimality gap."""
    lp       = result.get('lp_obj', math.inf)
    wall     = result.get('wall_s', 0.0)
    t_best   = result.get('t_best', 0.0)
    verified = result.get('verified', False)
    opt      = result.get('known_opt', known_opt)
    gap_pct  = result.get('gap_pct', compute_gap(lp, opt))
    lp_s     = f"{lp:.4f}" if not math.isinf(lp) else "inf"
    print(f"  {result.get('name', '?'):<16}  "
          f"LP={lp_s}  "
          f"gap={_gap_str(gap_pct)}  "
          f"wall={wall:.1f}s  t_best={t_best:.1f}s  "
          f"[{'✓' if verified else '✗'}]"
          f"{_gap_label(gap_pct)}")


# ─────────────────────────────────────────────────────────────────────────────
#   Console output — runway assignment table
# ─────────────────────────────────────────────────────────────────────────────

def print_runway_assignment(
    seqs: List[List[int]],
    inst: Optional[Instance]    = None,
    x:    Optional[np.ndarray]  = None,
) -> None:
    """
    Print a runway assignment table and, when inst and x are supplied,
    a per-aircraft penalty breakdown.

    Parameters
    ----------
    seqs : multi-runway sequences from the solver.
    inst : Instance data (required for per-aircraft detail).
    x    : LP landing times (required for per-aircraft detail).
    """
    m          = len(seqs)
    W          = 68          # table width
    CONT_PAD   = " " * 19   # indent for wrapped continuation lines

    print(f"\n  Runway Assignment")
    print(f"  {'─' * W}")
    print(f"  {'Runway':<10} {'Count':>5}  Landing order")
    print(f"  {'─' * W}")

    for rho in range(m):
        seq   = seqs[rho]
        count = len(seq)
        lines = _format_ac_list(seq)
        print(f"  R{rho:<9} {count:>5}  {lines[0]}")
        for cont in lines[1:]:
            print(f"{CONT_PAD}{cont}")

    print(f"  {'─' * W}")

    # Per-aircraft penalty detail — requires inst and x
    if inst is not None and x is not None:
        W2 = 78
        print(f"\n  Per-aircraft detail")
        print(f"  {'─' * W2}")
        print(f"  {'Rwy':<5} {'AC':<5} {'Land t':>9} {'Target':>9} "
              f"{'Early(s)':>9} {'Late(s)':>9} {'Penalty':>9}  Status")
        print(f"  {'─' * W2}")
        total_pen = 0.0
        for rho in range(m):
            for j in seqs[rho]:
                xj  = float(x[j])
                tgt = float(inst.delta[j])
                E   = max(tgt - xj, 0.0)
                T   = max(xj - tgt, 0.0)
                ep  = float(inst.g[j]) * E
                tp  = float(inst.h[j]) * T
                pen = ep + tp
                total_pen += pen
                if abs(xj - tgt) < 0.5:
                    status = "on time"
                elif xj < tgt:
                    status = f"early {E:,.0f}s"
                else:
                    status = f"late  {T:,.0f}s"
                print(f"  R{rho:<4} A{j:<4} {xj:>9.1f} {tgt:>9.1f} "
                      f"{E:>9.1f} {T:>9.1f} {pen:>9.2f}  {status}")
        print(f"  {'─' * W2}")
        print(f"  {'Total penalty':>{W2 - 12}} {total_pen:>9.2f}")
        print(f"  {'─' * W2}")

    print()


# ─────────────────────────────────────────────────────────────────────────────
#   Console output — detailed gap report
# ─────────────────────────────────────────────────────────────────────────────

def print_gap_report(
    result: Dict[str, Any],
    inst:   Optional[Instance] = None,
) -> None:
    """
    Print a detailed gap report for one (instance, m) result.

    If inst is provided the per-aircraft landing-time and penalty breakdown
    is included in the runway assignment table.

    Parameters
    ----------
    result : dict returned by solve() and enriched by run_all().
    inst   : Instance (optional); enables per-aircraft penalty output.
    """
    name     = result.get('name', '?')
    lp       = result.get('lp_obj', math.inf)
    opt      = result.get('known_opt')
    gap_pct  = result.get('gap_pct', compute_gap(lp, opt))
    verified = result.get('verified', False)
    wall     = result.get('wall_s', 0.0)
    t_best   = result.get('t_best', 0.0)
    seqs     = result.get('seqs')
    x        = result.get('x')

    lp_s      = f"{lp:.6f}" if not math.isinf(lp) else "inf"
    opt_s     = f"{opt:.6f}" if opt is not None else "N/A"
    abs_gap   = ((lp - opt)
                 if (opt is not None and not math.isinf(lp))
                 else float('nan'))
    abs_gap_s = f"{abs_gap:+.6f}" if not math.isnan(abs_gap) else "N/A"

    if math.isnan(gap_pct):
        quality = "No reference available"
    elif gap_pct < -1e-4:
        quality = f"NEW BEST — improves reference by {abs(gap_pct):.4f}%"
    elif abs(gap_pct) < 1e-4:
        quality = "Matches known optimum"
    elif gap_pct < 1.0:
        quality = "Within 1% of reference"
    elif gap_pct < 5.0:
        quality = "Within 5% of reference"
    else:
        quality = f">{gap_pct:.1f}% above reference"

    print(f"\n  {'─' * 60}")
    print(f"  Gap Report — {name}")
    print(f"  {'─' * 60}")
    print(f"  LP objective    : {lp_s}")
    print(f"  Known reference : {opt_s}")
    print(f"  Absolute gap    : {abs_gap_s}")
    print(f"  Gap (%)         : {_gap_str(gap_pct)}")
    print(f"  Assessment      : {quality}")
    print(f"  Verified        : {'✓ PASS' if verified else '✗ FAIL'}")
    print(f"  Wall time       : {wall:.1f}s  (t_best={t_best:.1f}s)")
    print(f"  {'─' * 60}")

    # Runway assignment — always printed when sequences are available
    if seqs is not None:
        print_runway_assignment(seqs, inst=inst, x=x)


# ─────────────────────────────────────────────────────────────────────────────
#   Console output — aggregate summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(results: List[Dict[str, Any]]) -> None:
    """
    Print a formatted aggregate results table across all (instance, m) pairs.

    Columns: instance, n, m, known_opt, LP obj, gap (%), wall, t_best, verified.
    """
    print(f"\n{'═' * 84}")
    print(f"  {'Instance':<18} {'n':>5} {'m':>3} {'Known opt':>12} "
          f"{'LP obj':>12} {'Gap (%)':>10} {'Wall(s)':>8} "
          f"{'t_best(s)':>10} {'Verified':>9}")
    print(f"  {'─' * 18} {'─' * 5} {'─' * 3} {'─' * 12} {'─' * 12} "
          f"{'─' * 10} {'─' * 8} {'─' * 10} {'─' * 9}")

    for r in results:
        lp      = r.get('lp_obj', math.inf)
        opt     = r.get('known_opt')
        gap_pct = r.get('gap_pct', compute_gap(lp, opt))
        wall    = r.get('wall_s', 0.0)
        tb      = r.get('t_best', 0.0)
        vf      = "✓ PASS" if r.get('verified') else "✗ FAIL"
        lp_s    = f"{lp:.4f}"  if not math.isinf(lp) else "inf"
        opt_s   = f"{opt:.4f}" if opt is not None     else "N/A"
        label   = _gap_label(gap_pct).strip()

        print(f"  {r.get('name', '?'):<18} "
              f"{r.get('n', '?'):>5} "
              f"{r.get('m', '?'):>3} "
              f"{opt_s:>12} "
              f"{lp_s:>12} "
              f"{_gap_str(gap_pct):>10} "
              f"{wall:>8.1f} "
              f"{tb:>10.1f} "
              f"{vf:>9}"
              + (f"  {label}" if label else ""))

    print(f"{'═' * 84}\n")


# ─────────────────────────────────────────────────────────────────────────────
#   File export
# ─────────────────────────────────────────────────────────────────────────────

def save_results(
    results: List[Dict[str, Any]],
    out_dir: str = str(OUTPUT_DIR),
) -> None:
    """
    Write three output files to out_dir:
      summary.csv       : one row per (instance, m) with key metrics and gap.
      verification.txt  : full VerificationReport for each result.
      run_metadata.json : machine and run provenance.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Summary CSV ───────────────────────────────────────────────────────
    summary_path = out / "summary.csv"
    fields = [
        "instance", "n", "m",
        "known_opt", "lp_obj",
        "gap_abs", "gap_pct",
        "wall_s", "t_best_s",
        "verified",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in results:
            lp      = r.get("lp_obj", math.inf)
            opt     = r.get("known_opt")
            gap_pct = r.get("gap_pct", compute_gap(lp, opt))
            gap_abs = ((lp - opt)
                       if opt is not None and not math.isinf(lp)
                       else float("nan"))
            w.writerow({
                "instance":  r.get("name", ""),
                "n":         r.get("n", ""),
                "m":         r.get("m", ""),
                "known_opt": "" if opt is None         else f"{opt:.6f}",
                "lp_obj":    "" if math.isinf(lp)      else f"{lp:.6f}",
                "gap_abs":   "" if math.isnan(gap_abs) else f"{gap_abs:+.6f}",
                "gap_pct":   "" if math.isnan(gap_pct) else f"{gap_pct:+.6f}",
                "wall_s":    f"{r.get('wall_s',  0.0):.3f}",
                "t_best_s":  f"{r.get('t_best',  0.0):.3f}",
                "verified":  "PASS" if r.get("verified") else "FAIL",
            })
    print(f"  Saved: {summary_path}")

    # ── Verification text ─────────────────────────────────────────────────
    verif_path = out / "verification.txt"
    with open(verif_path, "w", encoding="utf-8") as fh:
        fh.write(f"Hybrid MR-ALP Verification Report\n"
                 f"Generated : {ts}\n"
                 f"Results   : {len(results)}\n")
        for r in results:
            rpt = r.get("report")
            fh.write("\n")
            fh.write(rpt.summary() if rpt is not None
                     else f"  {r.get('name', '?')}: no report available.\n")
    print(f"  Saved: {verif_path}")

    # ── JSON metadata ─────────────────────────────────────────────────────
    meta_path = out / "run_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump({
            "timestamp": ts,
            "hostname":  socket.gethostname(),
            "python":    sys.version,
            "results": [
                {
                    "name":      r.get("name"),
                    "n":         r.get("n"),
                    "m":         r.get("m"),
                    "known_opt": r.get("known_opt"),
                    "lp_obj": (
                        None if math.isinf(r.get("lp_obj", math.inf))
                        else r.get("lp_obj")),
                    "gap_pct": (
                        None if math.isnan(
                            r.get("gap_pct",
                                  compute_gap(r.get("lp_obj", math.inf),
                                              r.get("known_opt"))))
                        else r.get("gap_pct",
                                   compute_gap(r.get("lp_obj", math.inf),
                                               r.get("known_opt")))),
                    "verified": r.get("verified"),
                    "wall_s":   r.get("wall_s"),
                    "t_best_s": r.get("t_best"),
                }
                for r in results
            ],
        }, fh, indent=2)
    print(f"  Saved: {meta_path}")


# ─────────────────────────────────────────────────────────────────────────────
#   Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

# Consistent colour palette — up to 10 runways (tab10 cycles for more)
_RWY_COLOURS = list(plt.cm.tab10.colors)

# Runway band background fill colours (one per runway, cycling)
_BAND_FILLS = [
    '#EBF3FB',   # light blue   — R0
    '#FFF8E1',   # light amber  — R1
    '#F1F8E9',   # light green  — R2
    '#FCE4EC',   # light pink   — R3
    '#F3E5F5',   # light purple — R4
    '#E0F2F1',   # light teal   — R5
    '#FBE9E7',   # light deep-orange — R6
    '#E8EAF6',   # light indigo — R7
    '#E0F7FA',   # light cyan   — R8
    '#F9FBE7',   # light lime   — R9
]


def _build_runway_layout(
    seqs:    List[List[int]],
    row_h:   float = 1.0,
    gap:     float = 1.5,
) -> Tuple[Dict[int, float],
           Dict[int, Tuple[float, float]],
           Dict[int, float]]:
    """
    Assign y-coordinates for the runway-grouped Gantt layout.

    Within each runway, aircraft appear in landing order (first landing has
    the smallest y-value and will appear at the top after invert_yaxis).

    Returns
    -------
    y_ac      : aircraft_index → y_centre.
    rwy_band  : runway_index  → (y_lo, y_hi) for axhspan.
    rwy_centre: runway_index  → y_centre for ytick placement.
    """
    y_ac:       Dict[int, float]               = {}
    rwy_band:   Dict[int, Tuple[float, float]] = {}
    rwy_centre: Dict[int, float]               = {}

    current_y = 0.0
    for rho, seq in enumerate(seqs):
        if not seq:
            rwy_band[rho]   = (current_y - 0.5, current_y + 0.5)
            rwy_centre[rho] = current_y
            current_y      += gap
            continue
        y_lo = current_y - 0.5
        for j in seq:
            y_ac[j]    = current_y
            current_y += row_h
        y_hi           = current_y - row_h + 0.5
        rwy_band[rho]  = (y_lo, y_hi)
        rwy_centre[rho] = (y_lo + y_hi) / 2.0
        current_y      += gap

    return y_ac, rwy_band, rwy_centre


# ─────────────────────────────────────────────────────────────────────────────
#   Gantt chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_gantt(
    seqs:      List[List[int]],
    inst:      Instance,
    x:         Optional[np.ndarray] = None,
    obj:       Optional[float]      = None,
    known_opt: Optional[float]      = None,
    save_dir:  str                  = "plots",
) -> None:
    """
    Multi-runway Gantt chart with runway-grouped layout.

    Layout
    ──────
    Aircraft are grouped by runway.  Each runway occupies a distinct
    colour-shaded horizontal band separated by a gap.  Within each band,
    aircraft are shown in landing order from top to bottom.

    Visual elements per aircraft
    ────────────────────────────
    ───  grey line   : time window [r_j, d_j].
    ▓    orange zone : wake-separation zone after each landing
                       [x_j, x_j + s[j, j_next]] (consecutive pair only).
    ◇    open diamond: target landing time δ_j.
    ●    filled dot  : actual LP landing time x_j (runway colour).
    A{j} label       : aircraft index centred below the landing marker.
    """
    if x is None:
        _, x = stage2_lp(seqs, inst)
    if x is None:
        print(f"  plot_gantt: LP infeasible for {inst.name}, skipped.")
        return

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    m              = inst.m
    ROW_H          = 1.0     # vertical space per aircraft (data units)
    GAP            = 1.8     # gap between runway bands

    y_ac, rwy_band, rwy_centre = _build_runway_layout(seqs, ROW_H, GAP)

    # ── Figure size ───────────────────────────────────────────────────────
    n_rows  = sum(len(s) for s in seqs)
    fig_h   = max(5.0, n_rows * 0.55 + m * 0.9 + 2.0)
    t_range = float(inst.d.max()) - float(inst.r.min())
    fig_w   = max(14.0, min(24.0, t_range / 55.0 + 5.0))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # ── Runway band backgrounds ───────────────────────────────────────────
    for rho in range(m):
        y_lo, y_hi = rwy_band[rho]
        ax.axhspan(y_lo, y_hi,
                   color=_BAND_FILLS[rho % len(_BAND_FILLS)],
                   alpha=0.65, zorder=0)
        # Thin dashed separator at the top of each band
        ax.axhline(y_lo, color='#aaaaaa', linewidth=0.9,
                   linestyle='--', alpha=0.6, zorder=1)

    # ── Aircraft elements ─────────────────────────────────────────────────
    for rho, seq in enumerate(seqs):
        c = _RWY_COLOURS[rho % len(_RWY_COLOURS)]
        L = len(seq)
        for pos, j in enumerate(seq):
            yc      = y_ac[j]
            xj      = float(x[j])
            rj      = float(inst.r[j])
            dj      = float(inst.d[j])
            delta_j = float(inst.delta[j])

            # Time window [r_j, d_j] — thin grey line
            ax.hlines(yc, rj, dj,
                      colors='#999999', linewidth=2.0,
                      linestyles='-', zorder=2, alpha=0.8)

            # Wake-separation zone [x_j, x_j + s[j, j_next]] — orange fill
            if pos < L - 1:
                j_next  = seq[pos + 1]
                sep_end = xj + float(inst.s[j, j_next])
                ax.barh(yc, width=max(sep_end - xj, 0.0), left=xj,
                        height=0.30, color='#FF7043', alpha=0.40,
                        zorder=3, linewidth=0)

            # Target time δ_j — open diamond (dark)
            ax.plot(delta_j, yc,
                    marker='D', markersize=6.5,
                    color='#333333', markerfacecolor='white',
                    markeredgewidth=1.4, linestyle='none', zorder=4)

            # Actual landing time x_j — filled circle (runway colour)
            ax.plot(xj, yc,
                    marker='o', markersize=9,
                    color=c, markerfacecolor=c,
                    markeredgewidth=0.8, markeredgecolor='white',
                    linestyle='none', zorder=5)

            # Aircraft label below the landing marker
            ax.text(xj, yc + 0.34, f"A{j}",
                    fontsize=6.5, ha='center', va='bottom',
                    color=c, fontweight='bold', zorder=6)

    # ── Y-axis: runway labels at band centres ─────────────────────────────
    ytick_pos    = []
    ytick_labels = []
    for rho in range(m):
        ytick_pos.append(rwy_centre[rho])
        n_ac = len(seqs[rho])
        ytick_labels.append(f"Runway {rho}\n({n_ac} aircraft)")
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_labels, fontsize=9.5, fontweight='bold')
    ax.tick_params(axis='y', length=0, pad=6)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)

    # Invert y so runway 0 / first landing appears at the top
    ax.invert_yaxis()

    # ── Y limits: tight around plotted content ────────────────────────────
    if y_ac:
        y_vals = list(y_ac.values())
        ax.set_ylim(max(y_vals) + 0.8, min(y_vals) - 0.8)

    # ── X limits: slight padding around time windows ──────────────────────
    t_min = float(inst.r.min())
    t_max = float(inst.d.max())
    t_pad = max(30.0, (t_max - t_min) * 0.03)
    ax.set_xlim(t_min - t_pad, t_max + t_pad)

    # ── Legend ────────────────────────────────────────────────────────────
    legend_els = [
        plt.Line2D([0], [0], color='#999999', linewidth=2.0,
                   label='Time window  [r_j, d_j]'),
        plt.Line2D([0], [0], marker='D', markersize=6.5, linestyle='none',
                   color='#333333', markerfacecolor='white',
                   markeredgewidth=1.4, label='Target time  δ_j'),
        plt.Line2D([0], [0], marker='o', markersize=9, linestyle='none',
                   color='#555555', markerfacecolor='#555555',
                   label='Landing time  x_j  (LP)'),
        mpatches.Patch(facecolor='#FF7043', alpha=0.40,
                       label='Wake separation zone'),
    ]
    for rho in range(m):
        c = _RWY_COLOURS[rho % len(_RWY_COLOURS)]
        legend_els.append(
            plt.Line2D([0], [0], marker='o', markersize=8, linestyle='none',
                       color=c, markerfacecolor=c,
                       label=f'Runway {rho}  ({len(seqs[rho])} aircraft)'))
    ax.legend(handles=legend_els, loc='upper right',
              fontsize=8.0, framealpha=0.92, ncol=2,
              borderpad=0.8, labelspacing=0.5)

    # ── Labels, grid, and title ───────────────────────────────────────────
    title = f"{inst.name}  (m = {m})"
    if obj is not None:
        title += f"     LP obj = {obj:.4f}"
    if known_opt is not None and obj is not None:
        gap_pct = compute_gap(obj, known_opt)
        title  += f"     gap = {_gap_str(gap_pct)}"
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.grid(True, axis='x', linestyle=':', linewidth=0.5,
            color='#cccccc', alpha=0.8, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fname = Path(save_dir) / f"gantt_{inst.name}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
#   Penalty profile chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_penalty_profile(
    seqs:      List[List[int]],
    inst:      Instance,
    x:         np.ndarray,
    obj:       Optional[float]  = None,
    known_opt: Optional[float]  = None,
    save_dir:  str              = "plots",
) -> None:
    """
    Stacked per-aircraft earliness/tardiness bar chart, grouped by runway.

    Aircraft are ordered by landing position within each runway.  Runways
    are separated by visible gaps and background shading.  The stacked bars
    show earliness penalty (blue, bottom) and tardiness penalty (red, top).
    Aircraft indices are shown on the x-axis.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    m = inst.m

    # Build ordered list: (rho, j) in landing order per runway
    ordered:       List[Tuple[int, int]] = []
    runway_bounds: List[Tuple[int, int]] = []   # (start_bar, end_bar) in ordered

    for rho, seq in enumerate(seqs):
        start = len(ordered)
        for j in seq:
            ordered.append((rho, j))
        runway_bounds.append((start, len(ordered)))

    n_bars = len(ordered)
    if n_bars == 0:
        return

    # Penalty arrays
    early_p = np.array([
        float(inst.g[j]) * max(float(inst.delta[j]) - float(x[j]), 0.0)
        for _, j in ordered
    ])
    late_p = np.array([
        float(inst.h[j]) * max(float(x[j]) - float(inst.delta[j]), 0.0)
        for _, j in ordered
    ])
    total_per_ac = early_p + late_p

    # ── Figure ────────────────────────────────────────────────────────────
    fig_w = max(10.0, n_bars * 0.45 + 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))

    # x-positions (1-based)
    xs = np.arange(1, n_bars + 1)

    # ── Runway band backgrounds ───────────────────────────────────────────
    for rho, (s_idx, e_idx) in enumerate(runway_bounds):
        if s_idx < e_idx:
            ax.axvspan(s_idx + 0.5, e_idx + 0.5,
                       color=_BAND_FILLS[rho % len(_BAND_FILLS)],
                       alpha=0.55, zorder=0)

    # ── Stacked bars ──────────────────────────────────────────────────────
    ax.bar(xs, early_p, color='#1565C0', alpha=0.85,
           label='Earliness  g_j · E_j', zorder=2)
    ax.bar(xs, late_p, bottom=early_p, color='#B71C1C', alpha=0.85,
           label='Tardiness  h_j · T_j', zorder=2)

    # ── Runway separator lines ────────────────────────────────────────────
    for rho, (s_idx, e_idx) in enumerate(runway_bounds[:-1]):
        ax.axvline(e_idx + 0.5, color='#555555', linewidth=1.3,
                   linestyle='--', alpha=0.7, zorder=3)

    # ── Per-bar aircraft labels on x-axis ────────────────────────────────
    x_labels = [f"A{j}" for _, j in ordered]
    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels, fontsize=7.0, rotation=45, ha='right')

    # ── Runway annotations above the bars ────────────────────────────────
    y_top = float(total_per_ac.max()) if total_per_ac.size > 0 else 1.0
    ax.set_ylim(bottom=0.0, top=max(y_top * 1.22, 1.0))  # headroom for labels

    for rho, (s_idx, e_idx) in enumerate(runway_bounds):
        if s_idx < e_idx:
            mid_x = (s_idx + e_idx) / 2.0 + 1.0
            n_ac  = e_idx - s_idx
            c     = _RWY_COLOURS[rho % len(_RWY_COLOURS)]
            ax.text(mid_x, y_top * 1.12,
                    f"Runway {rho}  ({n_ac} ac)",
                    ha='center', va='bottom',
                    fontsize=8.5, fontweight='bold', color=c, zorder=4)

    # ── Totals above each bar (only when penalty is nonzero) ──────────────
    for xi, total in zip(xs, total_per_ac):
        if total > 1e-6:
            ax.text(xi, total + y_top * 0.01, f"{total:.0f}",
                    ha='center', va='bottom', fontsize=5.5,
                    color='#333333', zorder=4)

    # ── Labels, legend, grid ──────────────────────────────────────────────
    title = f"Per-aircraft penalty profile — {inst.name}"
    if obj is not None:
        title += f"     total = {obj:.4f}"
    if known_opt is not None and obj is not None:
        gap_pct = compute_gap(obj, known_opt)
        title  += f"     gap = {_gap_str(gap_pct)}"
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel("Aircraft  (landing order per runway)", fontsize=10)
    ax.set_ylabel("Penalty", fontsize=10)
    ax.legend(fontsize=9, loc='upper right', framealpha=0.92)
    ax.grid(axis='y', linestyle=':', linewidth=0.5,
            color='#cccccc', alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    fname = Path(save_dir) / f"penalty_{inst.name}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")