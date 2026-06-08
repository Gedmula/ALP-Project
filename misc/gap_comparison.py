"""
gap_comparison_plot.py
Comparative gap analysis: Adaptive parameters vs DOE-calibrated parameters.
Produces two figures:
  Fig 1 — Side-by-side gap bar chart (adaptive vs DOE) for all 13 instances
  Fig 2 — Best-of-both gap vs known reference (with improvement annotations)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Paths — update these before running ─────────────────────────────────────
ADAPTIVE_CSV = "adaptive_results/summary.csv"
DOE_CSV      = "doe_results/summary.csv"
OUT_DIR      = "doe_results/analysis/plots"   # directory for saved figures
# ─────────────────────────────────────────────────────────────────────────────

import os
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
adp = pd.read_csv(ADAPTIVE_CSV).dropna(subset=["instance"])
doe = pd.read_csv(DOE_CSV).dropna(subset=["instance"])

# Merge on instance
df = adp[["instance", "n", "known_opt", "ms_sa_obj", "gap_pct"]].copy()
df = df.rename(columns={"ms_sa_obj": "obj_adp", "gap_pct": "gap_adp"})
df = df.merge(
    doe[["instance", "ms_sa_obj", "gap_pct"]].rename(
        columns={"ms_sa_obj": "obj_doe", "gap_pct": "gap_doe"}
    ),
    on="instance",
)

# Best-of-both objective and gap
df["obj_best"] = df[["obj_adp", "obj_doe"]].min(axis=1)
df["gap_best"] = (df["obj_best"] - df["known_opt"]) / df["known_opt"] * 100

# Sort by n, restrict to airland9 onwards (instances where not all are trivially optimal)
df = df.sort_values("n").reset_index(drop=True)
df = df[df["n"] >= 100].reset_index(drop=True)
labels = [f"{r['instance']}\n(n={r['n']})" for _, r in df.iterrows()]
x = np.arange(len(df))

# ── Colour scheme ─────────────────────────────────────────────────────────────
C_ADP  = "#4C72B0"   # blue  — adaptive
C_DOE  = "#DD8452"   # orange — DOE
C_BEST = "#55A868"   # green  — best-of-both
C_REF  = "#C44E52"   # red dashed — zero-gap reference

# ════════════════════════════════════════════════════════════════════════
# Figure 1 — Adaptive vs DOE gap comparison
# ════════════════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(10, 5))

bw = 0.35
bars_adp = ax1.bar(x - bw / 2, df["gap_adp"], width=bw,
                   color=C_ADP, label="Adaptive params", edgecolor="white")
bars_doe = ax1.bar(x + bw / 2, df["gap_doe"], width=bw,
                   color=C_DOE, label="DOE-calibrated params", edgecolor="white")

ax1.axhline(0, color="black", linewidth=0.8, linestyle="-")

# Annotate only non-zero bars
for bar, val in zip(bars_adp, df["gap_adp"]):
    if abs(val) > 0.001:
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 val - 0.05 if val < 0 else val + 0.05,
                 f"{val:.2f}%", ha="center", va="top" if val < 0 else "bottom",
                 fontsize=7, color=C_ADP, fontweight="bold")

for bar, val in zip(bars_doe, df["gap_doe"]):
    if abs(val) > 0.001:
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 val - 0.05 if val < 0 else val + 0.05,
                 f"{val:.2f}%", ha="center", va="top" if val < 0 else "bottom",
                 fontsize=7, color=C_DOE, fontweight="bold")

# No separator needed — all instances shown are n ≥ 100

ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=8)
ax1.set_ylabel("Gap to known reference (%)", fontsize=10)
ax1.set_title("MS-SA Gap Comparison: Adaptive vs DOE-Calibrated Parameters\n",
              fontsize=10)
ax1.legend(fontsize=9)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.grid(axis="y", linestyle=":", alpha=0.5)

plt.tight_layout()
p1 = f"{OUT_DIR}/gap_adaptive_vs_doe.png"
fig1.savefig(p1, dpi=200, bbox_inches="tight")
print(f"Saved: {p1}")

# ════════════════════════════════════════════════════════════════════════
# Figure 2 — Best-of-both vs known reference
# ════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(10, 5))

bar_colors = [C_BEST if g <= 0 else "#E8A838" for g in df["gap_best"]]
bars_best = ax2.bar(x, df["gap_best"], color=bar_colors, edgecolor="white", width=0.55)

ax2.axhline(0, color="black", linewidth=0.8)

for bar, val, row in zip(bars_best, df["gap_best"], df.itertuples()):
    if abs(val) > 0.001:
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 val - 0.06 if val < 0 else val + 0.06,
                 f"{val:.2f}%", ha="center",
                 va="top" if val < 0 else "bottom",
                 fontsize=8, fontweight="bold",
                 color="#2d6a4f" if val < 0 else "#8B4513")

# Which param set was best?
for i, row in df.iterrows():
    if abs(row["gap_adp"] - row["gap_doe"]) > 0.001:
        better = "Adp" if row["obj_adp"] < row["obj_doe"] else "DOE"
        ax2.text(x[i], 0.08, better, ha="center", va="bottom",
                 fontsize=7, color="gray", style="italic")

ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=8)
ax2.set_ylabel("Gap to known reference (%)", fontsize=10)
ax2.set_title("MS-SA Best-of-Both Results vs Known Reference\n",
              fontsize=10)

green_patch  = mpatches.Patch(color=C_BEST,   label="New best-known (gap < 0)")
amber_patch  = mpatches.Patch(color="#E8A838", label="Matches known optimal (gap = 0)")
ax2.legend(handles=[green_patch], fontsize=9)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.grid(axis="y", linestyle=":", alpha=0.5)

plt.tight_layout()
p2 = f"{OUT_DIR}/gap_best_of_both.png"
fig2.savefig(p2, dpi=200, bbox_inches="tight")
print(f"Saved: {p2}")

# ── Console summary table ─────────────────────────────────────────────────────
print("\n" + "═" * 72)
print(f"  {'Instance':<12} {'n':>5}  {'Adp gap%':>9}  {'DOE gap%':>9}  "
      f"{'Best gap%':>10}  {'Best obj':>12}  {'f*':>12}")
print("  " + "─" * 68)
for _, r in df.iterrows():
    better_tag = ("←Adp" if r["obj_adp"] < r["obj_doe"]
                  else "←DOE" if r["obj_doe"] < r["obj_adp"]
                  else "  tie")
    print(f"  {r['instance']:<12} {int(r['n']):>5}  "
          f"{r['gap_adp']:>+9.4f}  {r['gap_doe']:>+9.4f}  "
          f"{r['gap_best']:>+10.4f}  {r['obj_best']:>12.2f}  "
          f"{r['known_opt']:>12.2f}  {better_tag}")
print("═" * 72)