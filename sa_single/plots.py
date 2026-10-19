"""
plots.py — figures for single-runway SA runs
===============================================================================
gantt            landing schedule: windows, targets, landing times, separation
convergence      global best vs iterations and vs time; current cost per ladder level
seed_improvement starting cost vs final best per replica
penalty_profile  per-aircraft earliness / tardiness cost and deviation from target
ladder           temperature ladder and exchange acceptance per neighbour pair
archive          elite-archive size and diversity over the run
gap_summary      gap to the reference value across instances
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRAY, INK, INK2, GRID, SURF = "#9a9890", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ_BLUES = LinearSegmentedColormap.from_list(
    "seq_blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": "#c9c8c2", "axes.facecolor": SURF,
    "figure.facecolor": SURF, "savefig.facecolor": SURF, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "axes.titlelocation": "left", "axes.titleweight": "bold", "axes.titlesize": 10,
})


def _save(fig, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def gantt(inst, seq: List[int], x: np.ndarray, path, title: str = "") -> None:
    n = len(seq)
    fig, ax = plt.subplots(figsize=(11, max(4.0, min(0.11 * n + 1.5, 60.0))))
    for q, j in enumerate(seq):
        y = n - 1 - q
        ax.plot([inst.r[j], inst.d[j]], [y, y], color=GRAY, lw=0.8, zorder=1)
        if q + 1 < n:
            ax.barh(y, inst.s[j, seq[q + 1]], left=x[q], height=0.6, color=ORANGE, alpha=0.18, lw=0, zorder=1)
        dev = x[q] - inst.delta[j]
        col = ORANGE if dev > 1e-6 else (AQUA if dev < -1e-6 else BLUE)
        ax.plot(inst.delta[j], y, "s", mfc="none", mec=INK2, ms=4, zorder=2)
        ax.plot(x[q], y, "o", color=col, ms=4, zorder=3)
    lo = min(float(np.min(x)), float(np.min(inst.delta[seq]))) - 200
    hi = max(float(np.max(x)), float(np.max(inst.delta[seq]))) + 200
    ax.set_xlim(lo, hi)
    ax.set_ylim(-1, n)
    ax.set_yticks([n - 1 - q for q in range(0, n, max(1, n // 25))])
    ax.set_yticklabels([str(seq[q]) for q in range(0, n, max(1, n // 25))])
    ax.set_xlabel("time")
    ax.set_ylabel("aircraft (landing order, top first)")
    ax.set_title(title or "Landing schedule")
    handles = [plt.Line2D([], [], color=GRAY, lw=1, label="window [r, d]"),
               plt.Line2D([], [], marker="s", ls="", mfc="none", mec=INK2, label="target δ"),
               plt.Line2D([], [], marker="o", ls="", color=BLUE, label="landing, on target"),
               plt.Line2D([], [], marker="o", ls="", color=AQUA, label="landing, early"),
               plt.Line2D([], [], marker="o", ls="", color=ORANGE, label="landing, late"),
               plt.Rectangle((0, 0), 1, 1, color=ORANGE, alpha=0.18, label="separation to next")]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    _save(fig, path)


def convergence(result, path, reference: Optional[float] = None) -> None:
    h = result.history
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    it, tm, gb = h["iters"], h["time"], h["global_best"]
    ax = axes[0]
    ax.step(np.r_[0, it], np.r_[result.start_best, gb], where="post", color=BLUE, lw=2, label="global best")
    for i, kind in result.events:
        ax.axvline(i, color=ORANGE if kind == "stagnation" else YELLOW, lw=0.6, alpha=0.5)
    if reference is not None:
        ax.axhline(reference, color=INK2, ls=(0, (4, 3)), lw=1, label=f"reference {reference:g}")
    ax.axvline(result.ttb_iter, color=AQUA, lw=1.2, label=f"time to best: {result.ttb_iter:,} moves")
    ax.set_xlabel("iterations (moves per replica)")
    ax.set_ylabel("cost")
    ax.set_title(f"Global best — {result.name}  (lines: orange = stagnation reseed, yellow = hot mutation)")
    ax.legend(loc="upper right")
    ax = axes[1]
    ax.step(np.r_[0, tm], np.r_[result.start_best, gb], where="post", color=BLUE, lw=2)
    ax.axvline(result.ttb_time, color=AQUA, lw=1.2, label=f"time to best: {result.ttb_time:.1f} s")
    ax.set_xlabel("search time (s)")
    ax.set_ylabel("cost")
    ax.set_title("Global best against time")
    ax.legend(loc="upper right")
    ax = axes[2]
    lc = h["level_cost"]
    R = lc.shape[1]
    for l in range(R):
        ax.plot(it, lc[:, l], color=SEQ_BLUES(1 - l / max(R - 1, 1)), lw=0.8)
    ax.plot(it, gb, color=ORANGE, lw=1.6, label="global best")
    sm = plt.cm.ScalarMappable(cmap=SEQ_BLUES.reversed(), norm=plt.Normalize(0, R - 1))
    fig.colorbar(sm, ax=ax, label="ladder level (0 = coldest)")
    ax.set_xlabel("iterations (moves per replica)")
    ax.set_ylabel("current cost")
    ax.set_title("Current cost at each temperature level")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, path)


def seed_improvement(result, path) -> None:
    R = len(result.replica_start_cost)
    idx = np.arange(R)
    fig, ax = plt.subplots(figsize=(max(6, 0.35 * R + 2), 4))
    ax.bar(idx - 0.2, result.replica_start_cost, width=0.38, color=GRAY, label="starting schedule")
    ax.bar(idx + 0.2, result.replica_best_cost, width=0.38, color=BLUE, label="replica best")
    ax.set_xticks(idx)
    ax.set_xlabel("replica (initial ladder level, 0 = coldest)")
    ax.set_ylabel("cost")
    ax.set_title(f"Starting cost vs final best per replica — {result.name}")
    ax.legend()
    _save(fig, path)


def penalty_profile(inst, seq: List[int], x: np.ndarray, path, title: str = "") -> None:
    seq = np.asarray(seq)
    dev = x - inst.delta[seq]
    early = inst.g[seq] * np.maximum(-dev, 0)
    late = inst.h[seq] * np.maximum(dev, 0)
    q = np.arange(len(seq))
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax = axes[0]
    ax.bar(q, early, color=AQUA, width=0.9, label="earliness cost g·E")
    ax.bar(q, late, bottom=early, color=ORANGE, width=0.9, label="tardiness cost h·T")
    ax.set_ylabel("penalty")
    ax.set_title(title or f"Penalty profile — total {early.sum() + late.sum():.2f}")
    ax.legend(loc="upper right")
    ax = axes[1]
    ax.vlines(q, 0, dev, color=np.where(dev > 0, ORANGE, AQUA), lw=1.2)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("landing position")
    ax.set_ylabel("landing − target")
    fig.tight_layout()
    _save(fig, path)


def ladder(result, path) -> None:
    R = result.replicas
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.semilogy(np.arange(R), result.temperatures, "o-", color=BLUE, lw=2, ms=5)
    ax.set_xlabel("ladder level (0 = coldest)")
    ax.set_ylabel("temperature (log scale)")
    ax.set_title("Temperature ladder")
    ax = axes[1]
    if R > 1:
        rate = result.swap_accepts / np.maximum(result.swap_attempts, 1)
        ax.bar(np.arange(R - 1) + 0.5, rate, width=0.8, color=BLUE)
        ax.axhspan(0.2, 0.4, color=AQUA, alpha=0.12, label="commonly targeted 0.2–0.4")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right")
    ax.set_xlabel("neighbour pair (between levels l and l+1)")
    ax.set_ylabel("exchange acceptance rate")
    ax.set_title("Replica-exchange acceptance")
    fig.tight_layout()
    _save(fig, path)


def archive(result, path) -> None:
    h = result.history
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    axes[0].step(h["iters"], h["archive_size"], where="post", color=BLUE, lw=2)
    axes[0].set_ylabel("schedules")
    axes[0].set_title(f"Elite archive — {result.name}")
    axes[1].plot(h["iters"], h["archive_div"], color=BLUE, lw=2)
    axes[1].set_ylabel("mean position difference\nper aircraft")
    axes[1].set_xlabel("iterations (moves per replica)")
    fig.tight_layout()
    _save(fig, path)


def gap_summary(rows: List[Dict], path) -> None:
    """rows: dicts with instance, best (best over seeds), reference."""
    names = [r["instance"] for r in rows]
    gaps = np.array([100 * (r["best"] - r["reference"]) / r["reference"] if r["reference"] else 0.0
                     for r in rows])
    col = [BLUE if g < -1e-6 else (GRAY if abs(g) <= 1e-6 else ORANGE) for g in gaps]
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(rows) + 2), 4))
    ax.bar(np.arange(len(rows)), gaps, color=col, width=0.7)
    ax.axhline(0, color=INK2, lw=0.8)
    for i, g in enumerate(gaps):
        ax.text(i, g, f"{g:+.2f}%", ha="center", va="bottom" if g >= 0 else "top", fontsize=8, color=INK)
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("gap to reference (%)")
    ax.set_title("Best found vs reference value")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE, label="better than reference"),
                       plt.Rectangle((0, 0), 1, 1, color=GRAY, label="equal"),
                       plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="worse")], loc="upper right")
    _save(fig, path)
