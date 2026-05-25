"""
exp3_factor_importance_replot.py
Replot Exp-3 factor importance (|main effect| on gap%) with instances
ordered correctly by n and annotated by size band.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data ────────────────────────────────────────────────────────────────────


df = pd.read_csv("DOE\\exp3_parameter\\main_effects.csv")

# ── Filter to main effects only ──────────────────────────────────────────────
main_effects = ["alpha", "N_iter", "I_max", "M_stag"]
df_me = df[df["factor_or_interaction"].isin(main_effects)].copy()
df_me["abs_effect"] = df_me["effect"].abs()

# ── Instance ordering by n ───────────────────────────────────────────────────
instance_n = {
    "airland1": 10,  "airland2": 15,  "airland3": 20,
    "airland4": 20,  "airland5": 20,  "airland6": 30,
    "airland7": 44,  "airland8": 50,  "airland9": 100,
    "airland10": 150,"airland11": 200,"airland12": 250,
    "airland13": 500,
}
ordered = sorted(instance_n, key=instance_n.get)   # airland1 … airland13 by n

# ── Pivot ────────────────────────────────────────────────────────────────────
pivot = (
    df_me.pivot(index="instance", columns="factor_or_interaction", values="abs_effect")
         .loc[ordered, main_effects]          # enforce row & column order
         .fillna(0)
)

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))

n_inst   = len(ordered)
n_factor = len(main_effects)
bar_h    = 0.18          # height of each bar
gap      = 0.06          # gap between factor bars in a group
group_h  = n_factor * bar_h + gap   # total height per instance group

colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]   # blue, orange, green, red
factor_labels = [r"$\alpha$", r"$N_\mathrm{iter}$", r"$I_\mathrm{max}$", r"$M_\mathrm{stag}$"]

y_centers = np.arange(n_inst) * (group_h + 0.10)  # centre of each instance group

for f_idx, (factor, color, label) in enumerate(zip(main_effects, colors, factor_labels)):
    offsets = y_centers + (f_idx - (n_factor - 1) / 2) * bar_h
    vals    = pivot[factor].values
    ax.barh(offsets, vals, height=bar_h * 0.85,
            color=color, label=label, edgecolor="white", linewidth=0.4)

# ── y-axis labels with (n=…) ─────────────────────────────────────────────────
ax.set_yticks(y_centers)
ax.set_yticklabels(
    [f"{inst}  (n={instance_n[inst]})" for inst in ordered],
    fontsize=9,
)

# ── Size-band separator lines ────────────────────────────────────────────────
# boundaries: after airland7 (n≤44) and after airland9 (n≤100)
band_after = {"airland7": "small  (n ≤ 44)",
              "airland9": "medium  (n = 50–100)"}
for inst, label_text in band_after.items():
    idx  = ordered.index(inst)
    # midpoint between this group and the next
    y_line = (y_centers[idx] + y_centers[idx + 1]) / 2
    ax.axhline(y_line, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)

# Annotate bands on the right margin
band_ranges = [
    (0,          ordered.index("airland7"),  "Small\n(n ≤ 44)"),
    (ordered.index("airland8"), ordered.index("airland9"),  "Medium\n(n = 50–100)"),
    (ordered.index("airland10"), n_inst - 1, "Large\n(n ≥ 150)"),
]
x_ann = pivot.values.max() * 1.02
for i_lo, i_hi, txt in band_ranges:
    y_mid = (y_centers[i_lo] + y_centers[i_hi]) / 2
    ax.text(x_ann, y_mid, txt, va="center", ha="left",
            fontsize=7.5, color="gray", style="italic")

# ── Formatting ────────────────────────────────────────────────────────────────
ax.set_xlabel(r"|Main Effect| on gap (%)", fontsize=11)
ax.set_title("Exp-3 — Factor Importance by Instance\n"
             r"($2^4$ Factorial DOE, EDD seed, bars show $|\hat{\beta}|$)",
             fontsize=11)
ax.legend(title="Factor", bbox_to_anchor=(1.18, 1), loc="upper right",
          fontsize=9, title_fontsize=9)
ax.axvline(0, color="black", linewidth=0.6)
ax.set_xlim(left=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", linestyle=":", alpha=0.5)

plt.tight_layout()
plt.savefig("DOE/analysis/plots/exp3_factor_importance_replot.png",
            dpi=200, bbox_inches="tight")
plt.show()
print("Saved: DOE/analysis/plots/exp3_factor_importance_replot.png")