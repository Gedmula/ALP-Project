"""
analyze.py — statistical analysis and plots of the DOE phases
===============================================================================
Set the CONFIGURE HERE block and run from the repository root:

    python sa_single/doe/analyze.py

ANALYSIS
    "screening"  effects, ANOVA, curvature test and active factors of the
                 2^(7−3) fraction (+ foldover runs if present)
    "ccd"        second-order response surface, lack of fit, desirability
                 optimisation → results/recommended_params.json
    "confirm"    held-out instances against the CCD prediction intervals
    "budget"     iteration budget read off the best-so-far curves of the long
                 anchor runs on the held-out set (confirm runs if absent) → budget
                 rule written into results/recommended_params.json

Responses (per run)
    gap       100 · (best − reference) / reference, reference = best value found
              for the instance in any phase (anchor runs included)
    log_gap   log10(gap + 0.01)                                     lower is better
    ttb_frac  iterations to the final best / iteration budget       lower is better
    ttb_s     seconds to the final best                             (reported)
    mean_gap  mean of the gap over the 101 budget checkpoints       lower is better
    ttt       budget fraction to come within GAP_TOL % (1 = never)  lower is better
    D         Derringer–Suich desirability of quality and speed     higher is better
              d_q = clip(1 − gap / U, 0, 1)   U = 95th percentile of gap
              d_t = max(1 − ttb_frac, 0.05)
              D   = (d_q^QW · d_t^SW)^(1 / (QW + SW))
"""
from __future__ import annotations

import json
import sys
import warnings
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from scipy.optimize import minimize
from statsmodels.stats.anova import anova_lm

from sa_single import plots as P            # shared style and colours
from sa_single.doe import design as Dsg
from sa_single.params import FACTOR_BY_CODE, SAParams, decode

warnings.filterwarnings("ignore")

# ═════════════════════════════════════════════════════════════════════════════
#   CONFIGURE HERE
# ═════════════════════════════════════════════════════════════════════════════
ANALYSIS         = "screening"
SCREENING_PHASES = ["screening", "foldover"]       # pooled in the screening analysis (missing ones skipped)
GAP_TOL          = 0.1        # %: threshold of the time-to-target response
QUALITY_WEIGHT   = 2.0        # desirability weights
SPEED_WEIGHT     = 1.0
ALPHA            = 0.05
BUDGET_TOL       = 0.01       # percentage points of excess gap tolerated by the budget rule
# ═════════════════════════════════════════════════════════════════════════════

RESULTS = ROOT / "sa_single" / "doe" / "results"
LETTERS = Dsg.LETTERS
FCOL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]   # factor A..G
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
RESP = [  # key, label, better
    ("log_gap", "log10(gap % + 0.01)", "min"),
    ("ttb_frac", "time to best (fraction of budget)", "min"),
    ("mean_gap", "mean gap over the run (%)", "min"),
    ("ttt", f"budget fraction to reach {GAP_TOL}% gap", "min"),
    ("D", "desirability (quality × speed)", "max"),
]
RESP_LABEL = {k: l for k, l, _ in RESP}
TUNE_PHASES = ["screening", "foldover", "ccd", "anchor_tune"]
HOLD_PHASES = ["confirm", "anchor_holdout"]


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def load_phase(name: str) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    d = RESULTS / name
    if not (d / "records.csv").exists():
        return None
    rec = pd.read_csv(d / "records.csv")
    cur = pd.read_csv(d / "curves.csv").set_index("run_id")
    return rec, cur


def references(phases: List[str]) -> pd.Series:
    best = []
    for ph in phases:
        x = load_phase(ph)
        if x is not None:
            best.append(x[0].groupby("instance")["best_cost"].min())
    return pd.concat(best, axis=1).min(axis=1)


def gap_pct(cost, ref):
    ref = np.asarray(ref, dtype=float)
    return np.where(ref > 1e-9, 100.0 * (np.asarray(cost) - ref) / np.maximum(ref, 1e-9), np.asarray(cost) - ref)


def responses(rec: pd.DataFrame, cur: pd.DataFrame, ref: pd.Series) -> pd.DataFrame:
    df = rec.copy()
    df["reference"] = df["instance"].map(ref)
    df["gap"] = np.maximum(gap_pct(df["best_cost"], df["reference"]), 0.0)
    df["log_gap"] = np.log10(df["gap"] + 0.01)
    df["ttb_frac"] = df["ttb_iter"] / df["budget"]
    cv = cur.loc[df["run_id"]].to_numpy()
    gt = np.maximum(gap_pct(cv, df["reference"].to_numpy()[:, None]), 0.0)
    df["mean_gap"] = gt.mean(axis=1)
    reach = gt <= GAP_TOL
    df["ttt"] = np.where(reach.any(axis=1), reach.argmax(axis=1) / (gt.shape[1] - 1), 1.0)
    u = max(float(np.quantile(df["gap"], 0.95)), GAP_TOL)
    dq = np.clip(1.0 - df["gap"] / u, 0.0, 1.0)
    dt = np.maximum(1.0 - df["ttb_frac"], 0.05)
    df["D"] = (dq ** QUALITY_WEIGHT * dt ** SPEED_WEIGHT) ** (1.0 / (QUALITY_WEIGHT + SPEED_WEIGHT))
    df["ctr"] = (df["point"] == "center").astype(float)
    df["n_class"] = df["n"].astype(int)
    return df


def gap_curves(df: pd.DataFrame, cur: pd.DataFrame) -> np.ndarray:
    cv = cur.loc[df["run_id"]].to_numpy()
    return np.maximum(gap_pct(cv, df["reference"].to_numpy()[:, None]), 0.0)


def outdir(name: str) -> Path:
    d = RESULTS / f"analysis_{name}"
    (d / "plots").mkdir(parents=True, exist_ok=True)
    return d


def _save(fig, path) -> None:
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Effects
# ─────────────────────────────────────────────────────────────────────────────

def estimable_2fi(fact: pd.DataFrame) -> Tuple[List[str], Dict[str, List[str]]]:
    """Two-factor interactions that can be estimated together with the main effects,
    and for each the interactions aliased with it (identical or opposite contrast)."""
    cols = Dsg.model_columns(fact)
    X = np.column_stack([np.ones(len(fact))] + [cols[c] for c in LETTERS])
    chosen, alias = [], {}
    for a, b in combinations(LETTERS, 2):
        t = a + b
        v = cols[t].to_numpy()
        if np.linalg.matrix_rank(np.column_stack([X, v])) > np.linalg.matrix_rank(X):
            X = np.column_stack([X, v])
            chosen.append(t)
    for t in chosen:
        v = cols[t].to_numpy()
        alias[t] = [u for u in cols.columns if len(u) == 2 and u != t and
                    (np.array_equal(cols[u].to_numpy(), v) or np.array_equal(cols[u].to_numpy(), -v))]
    return chosen, alias


def effect_label(t: str, alias: Dict[str, List[str]]) -> str:
    return "=".join([t] + alias.get(t, [])) if len(t) == 2 else t


def contrast_effects(fact: pd.DataFrame, resp: str, terms: List[str]) -> pd.Series:
    """Effect = mean(resp | +1) − mean(resp | −1), over all factorial runs."""
    cols = Dsg.model_columns(fact)
    y = fact[resp].to_numpy()
    return pd.Series({t: y[cols[t] > 0].mean() - y[cols[t] < 0].mean() for t in terms})


def lenth(effects: pd.Series) -> Tuple[float, float, float]:
    """Lenth's pseudo standard error, margin of error and simultaneous margin of error."""
    c = np.abs(effects.to_numpy())
    m = len(c)
    s0 = 1.5 * np.median(c)
    small = c[c < 2.5 * s0]
    pse = 1.5 * np.median(small) if len(small) else s0
    dfree = max(m / 3.0, 1.0)
    me = stats.t.ppf(0.975, dfree) * pse
    gamma = (1 + 0.95 ** (1.0 / m)) / 2
    sme = stats.t.ppf(gamma, dfree) * pse
    return pse, me, sme


def fit_screen_model(df: pd.DataFrame, resp: str, terms2: List[str]):
    data = df.copy()
    cols = Dsg.model_columns(data)
    for t in terms2:
        data[t] = cols[t]
    rhs = ["C(instance)"] + LETTERS + terms2 + (["ctr"] if data["ctr"].sum() > 0 else [])
    model = smf.ols(f"{resp} ~ " + " + ".join(rhs), data=data).fit()
    try:
        aov = anova_lm(model, typ=2)
    except Exception:
        aov = anova_lm(model, typ=1)
    return model, aov


# ─────────────────────────────────────────────────────────────────────────────
# Screening plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_pareto(eff: pd.Series, labels: Dict[str, str], me: float, sme: float, resp: str, path) -> None:
    order = eff.abs().sort_values()
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(order) + 1.5))
    col = [P.BLUE if eff[t] > 0 else P.ORANGE for t in order.index]
    ax.barh([labels[t] for t in order.index], order.values, color=col, height=0.7)
    ax.axvline(me, color=P.INK2, ls=(0, (4, 3)), lw=1, label=f"Lenth ME = {me:.3g}")
    ax.axvline(sme, color=P.INK, ls=(0, (1, 2)), lw=1, label=f"Lenth SME = {sme:.3g}")
    ax.set_xlabel(f"|effect| on {RESP_LABEL[resp]}")
    ax.set_title(f"Pareto chart of effects — {resp}")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=P.BLUE, label="positive effect"),
                       plt.Rectangle((0, 0), 1, 1, color=P.ORANGE, label="negative effect"),
                       plt.Line2D([], [], color=P.INK2, ls=(0, (4, 3)), label=f"Lenth ME = {me:.3g}"),
                       plt.Line2D([], [], color=P.INK, ls=(0, (1, 2)), label=f"Lenth SME = {sme:.3g}")],
              loc="lower right")
    _save(fig, path)


def plot_halfnormal(eff: pd.Series, labels: Dict[str, str], me: float, resp: str, path) -> None:
    a = eff.abs().sort_values()
    m = len(a)
    q = stats.norm.ppf(0.5 + 0.5 * (np.arange(1, m + 1) - 0.5) / m)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sig = a.values > me
    ax.scatter(q[~sig], a.values[~sig], color=P.GRAY, s=36, zorder=3, label="within Lenth ME")
    ax.scatter(q[sig], a.values[sig], color=P.BLUE, s=48, zorder=3, label="beyond Lenth ME")
    slope = np.median(a.values) / np.median(q) if np.median(q) > 0 else 0
    ax.plot([0, q.max()], [0, slope * q.max()], color=P.INK2, lw=1, ls=(0, (4, 3)), label="reference line (median)")
    for qi, (t, v), s in zip(q, a.items(), sig):
        if s or qi == q.max():
            ax.annotate(labels[t], (qi, v), textcoords="offset points", xytext=(6, 0), fontsize=8, color=P.INK)
    ax.set_xlabel("half-normal quantile")
    ax.set_ylabel("|effect|")
    ax.set_title(f"Half-normal plot of effects — {resp}")
    ax.legend(loc="upper left")
    _save(fig, path)


def plot_main_effects(df: pd.DataFrame, resp: str, path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharey=True)
    fact, cen = df[df["point"] != "center"], df[df["point"] == "center"]
    for ax, c, colr in zip(axes.flat, LETTERS, FCOL):
        f = FACTOR_BY_CODE[c]
        m, e = [], []
        for lv in (-1, 1):
            y = fact.loc[fact[c] == lv, resp]
            m.append(y.mean())
            e.append(1.96 * y.std(ddof=1) / np.sqrt(max(len(y), 1)))
        ax.errorbar([-1, 1], m, yerr=e, color=colr, lw=2, marker="o", ms=7, capsize=4)
        if len(cen):
            yc = cen[resp]
            ax.errorbar([0], [yc.mean()], yerr=[1.96 * yc.std(ddof=1) / np.sqrt(len(yc))], color=P.INK2,
                        marker="D", mfc="none", ms=7, capsize=4, ls="none")
        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels([f"{f.natural(-1):g}", f"{f.natural(0):g}", f"{f.natural(1):g}"])
        ax.set_title(f"{c}: {f.label}", fontsize=9)
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(handles=[plt.Line2D([], [], color=P.BLUE, marker="o", lw=2, label="factorial mean ± 95% CI"),
                                  plt.Line2D([], [], color=P.INK2, marker="D", mfc="none", ls="none",
                                             label="centre-point mean ± 95% CI")], loc="center")
    for ax in axes[:, 0]:
        ax.set_ylabel(RESP_LABEL[resp])
    fig.suptitle(f"Main effects — {resp}", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    _save(fig, path)


def plot_interactions(df: pd.DataFrame, resp: str, terms2: List[str], alias, path) -> None:
    fact = df[df["point"] != "center"]
    k = len(terms2)
    if k == 0:
        return
    nc = 4
    nr = int(np.ceil(k / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(15, 3.6 * nr), squeeze=False, sharey=True)
    for ax, t in zip(axes.flat, terms2):
        a, b = t[0], t[1]
        for lv, ls, colr in ((-1, (0, (4, 3)), P.GRAY), (1, "-", FCOL[LETTERS.index(b)])):
            y = [fact.loc[(fact[a] == la) & (fact[b] == lv), resp].mean() for la in (-1, 1)]
            ax.plot([-1, 1], y, ls=ls, color=colr, lw=2, marker="o", ms=6, label=f"{b} = {'+1' if lv > 0 else '−1'}")
        ax.set_xticks([-1, 1])
        ax.set_xlabel(f"{a} (coded)")
        ax.set_title(effect_label(t, alias), fontsize=9)
        ax.legend(fontsize=7, loc="best")
    for ax in list(axes.flat)[k:]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel(RESP_LABEL[resp])
    fig.suptitle(f"Two-factor interaction plots — {resp} (titles list the aliased interactions)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    _save(fig, path)


def plot_effects_by_class(df: pd.DataFrame, resp: str, path) -> pd.DataFrame:
    fact = df[df["point"] != "center"]
    groups = ([("n", v) for v in sorted(fact["n"].unique())] +
              [("congestion", v) for v in sorted(fact["congestion"].unique())] +
              [("windows", v) for v in sorted(fact["windows"].unique())])
    tab = pd.DataFrame(index=LETTERS, columns=[f"{g}={v}" for g, v in groups], dtype=float)
    for g, v in groups:
        sub = fact[fact[g] == v]
        for c in LETTERS:
            tab.loc[c, f"{g}={v}"] = sub.loc[sub[c] > 0, resp].mean() - sub.loc[sub[c] < 0, resp].mean()
    lim = float(np.nanmax(np.abs(tab.to_numpy()))) or 1.0
    fig, ax = plt.subplots(figsize=(1.0 * len(groups) + 3, 4.8))
    im = ax.imshow(tab.to_numpy(), cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(tab.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(LETTERS)))
    ax.set_yticklabels([f"{c}: {FACTOR_BY_CODE[c].label}" for c in LETTERS])
    for i in range(len(LETTERS)):
        for j in range(len(groups)):
            ax.text(j, i, f"{tab.iat[i, j]:.3g}", ha="center", va="center", fontsize=7, color=P.INK)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label=f"main effect on {resp}")
    ax.set_title(f"Main effects within instance classes — {resp}  (control × noise interaction check)")
    _save(fig, path)
    return tab


def plot_effect_vs_size(tab: pd.DataFrame, resp: str, path) -> None:
    cols = [c for c in tab.columns if c.startswith("n=")]
    ns = [int(float(c[2:])) for c in cols]
    fig, ax = plt.subplots(figsize=(8, 5))
    for c, colr in zip(LETTERS, FCOL):
        ax.plot(ns, tab.loc[c, cols].to_numpy(float), marker="o", lw=2, color=colr, label=f"{c}: {FACTOR_BY_CODE[c].label}")
    ax.axhline(0, color=P.INK2, lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(v) for v in ns])
    ax.minorticks_off()
    ax.set_xlabel("instance size n")
    ax.set_ylabel(f"main effect on {resp}")
    ax.set_title(f"Do the effects depend on instance size? — {resp}")
    ax.legend(fontsize=7, loc="best")
    _save(fig, path)


def plot_residuals(model, df: pd.DataFrame, resp: str, path) -> None:
    res, fit = model.resid, model.fittedvalues
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    (osm, osr), (sl, ic, _) = stats.probplot(res, dist="norm")
    axes[0, 0].scatter(osm, osr, s=10, color=P.BLUE)
    axes[0, 0].plot(osm, sl * osm + ic, color=P.ORANGE, lw=1.5)
    axes[0, 0].set_title("Normal Q–Q of residuals")
    axes[0, 0].set_xlabel("theoretical quantile")
    axes[0, 0].set_ylabel("residual")
    axes[0, 1].scatter(fit, res, s=10, color=P.BLUE, alpha=0.6)
    axes[0, 1].axhline(0, color=P.INK2, lw=0.8)
    axes[0, 1].set_title("Residuals vs fitted")
    axes[0, 1].set_xlabel("fitted value")
    axes[0, 1].set_ylabel("residual")
    inst = df.loc[res.index, "instance"]
    names = sorted(inst.unique())
    axes[1, 0].boxplot([res[inst == i] for i in names], showfliers=False)
    axes[1, 0].set_xticks(range(1, len(names) + 1))
    axes[1, 0].set_xticklabels(names, rotation=90, fontsize=6)
    axes[1, 0].axhline(0, color=P.INK2, lw=0.8)
    axes[1, 0].set_title("Residuals by instance")
    axes[1, 1].hist(res, bins=40, color=P.BLUE)
    axes[1, 1].set_title("Residual histogram")
    axes[1, 1].set_xlabel("residual")
    fig.suptitle(f"Residual diagnostics — {resp}", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    _save(fig, path)


def plot_config_box(df: pd.DataFrame, resp: str, path) -> None:
    cfgs = sorted(df["config"].unique())
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(cfgs) + 2), 4.5))
    bp = ax.boxplot([df.loc[df["config"] == c, resp] for c in cfgs], patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], cfgs):
        is_c = (df.loc[df["config"] == c, "point"] == "center").all()
        patch.set_facecolor(P.GRAY if is_c else "#cde2fb")
        patch.set_edgecolor(P.INK2)
    ax.set_xticks(range(1, len(cfgs) + 1))
    ax.set_xticklabels(cfgs, rotation=90)
    ax.set_ylabel(RESP_LABEL[resp])
    ax.set_title(f"Distribution per configuration — {resp}  (gray = centre points)")
    _save(fig, path)


def plot_curvature(df: pd.DataFrame, resp: str, p_ctr: float, path) -> None:
    fact, cen = df[df["point"] != "center"], df[df["point"] == "center"]
    if cen.empty:
        return
    a = fact.groupby("instance")[resp].mean()
    b = cen.groupby("instance")[resp].mean()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(a, b.loc[a.index], color=P.BLUE, s=30)
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    ax.plot([lo, hi], [lo, hi], color=P.INK2, ls=(0, (4, 3)), lw=1)
    ax.set_xlabel("mean over factorial runs")
    ax.set_ylabel("mean over centre-point runs")
    ax.set_title(f"Curvature check — {resp}  (centre-indicator p = {p_ctr:.3g})")
    _save(fig, path)


def plot_tradeoff(df: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(df["ttb_frac"], df["gap"] + 0.01, c=df["D"], cmap=P.SEQ_BLUES, s=12, alpha=0.7)
    ax.set_yscale("log")
    ax.set_xlabel("time to best (fraction of budget)")
    ax.set_ylabel("gap % + 0.01 (log scale)")
    fig.colorbar(sc, ax=ax, label="desirability D")
    ax.set_title("Quality vs time to best (every run)")
    _save(fig, path)


def plot_curves_by_config(df: pd.DataFrame, cur: pd.DataFrame, path) -> None:
    gt = gap_curves(df, cur)
    fr = np.linspace(0, 1, gt.shape[1])
    ns = sorted(df["n"].unique())
    fig, axes = plt.subplots(1, len(ns), figsize=(4.2 * len(ns), 4.2), sharey=True, squeeze=False)
    cfgs = sorted(df["config"].unique())
    for ax, nv in zip(axes[0], ns):
        for c in cfgs:
            mask = ((df["n"] == nv) & (df["config"] == c)).to_numpy()
            if mask.any():
                is_c = (df.loc[mask, "point"] == "center").all()
                ax.plot(fr, np.median(gt[mask], axis=0) + 0.01, lw=1.6 if is_c else 0.8,
                        color=P.ORANGE if is_c else P.BLUE, alpha=1 if is_c else 0.45)
        ax.set_yscale("log")
        ax.set_title(f"n = {nv}")
        ax.set_xlabel("fraction of iteration budget")
    axes[0, 0].set_ylabel("median gap % + 0.01 (log)")
    axes[0, -1].legend(handles=[plt.Line2D([], [], color=P.BLUE, label="factorial configuration"),
                                plt.Line2D([], [], color=P.ORANGE, lw=1.6, label="centre point")], loc="upper right")
    fig.suptitle("Best-so-far gap along the budget, median over instances and seeds", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, path)


def plot_response_corr(df: pd.DataFrame, path) -> None:
    keys = [k for k, _, _ in RESP] + ["ttb_s"]
    cm = df[keys].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=40, ha="right")
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(keys)
    for i in range(len(keys)):
        for j in range(len(keys)):
            ax.text(j, i, f"{cm.iat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Spearman correlation")
    ax.set_title("Correlation between responses")
    _save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
# Screening analysis
# ─────────────────────────────────────────────────────────────────────────────

def screening_analysis() -> None:
    parts = [load_phase(p) for p in SCREENING_PHASES]
    parts = [p for p in parts if p is not None]
    if not parts:
        raise SystemExit("no screening records found")
    rec = pd.concat([p[0] for p in parts], ignore_index=True)
    cur = pd.concat([p[1] for p in parts])
    df = responses(rec, cur, references(TUNE_PHASES))
    out = outdir("screening")
    pl = out / "plots"
    fact = df[df["point"] != "center"]
    terms2, alias = estimable_2fi(fact.drop_duplicates("config"))
    terms = LETTERS + terms2
    labels = {t: effect_label(t, alias) for t in terms}
    summary = {"terms": terms, "aliases": alias, "responses": {}}
    eff_table = {}
    for resp, _, better in RESP:
        eff = contrast_effects(fact, resp, terms)
        pse, me, sme = lenth(eff)
        model, aov = fit_screen_model(df, resp, terms2)
        aov.to_csv(out / f"anova_{resp}.csv")
        pvals = {t: float(aov.loc[t, "PR(>F)"]) for t in terms if t in aov.index}
        p_ctr = float(aov.loc["ctr", "PR(>F)"]) if "ctr" in aov.index else float("nan")
        eff_table[resp] = eff
        summary["responses"][resp] = dict(
            lenth_pse=pse, lenth_me=me, lenth_sme=sme, curvature_p=p_ctr, r2=model.rsquared,
            effects={labels[t]: float(eff[t]) for t in terms},
            p_values={labels[t]: pvals.get(t) for t in terms},
            significant=[labels[t] for t in terms if (pvals.get(t, 1) < ALPHA) or abs(eff[t]) > me])
        plot_pareto(eff, labels, me, sme, resp, pl / f"pareto_{resp}.png")
        plot_halfnormal(eff, labels, me, resp, pl / f"halfnormal_{resp}.png")
        plot_main_effects(df, resp, pl / f"main_effects_{resp}.png")
        plot_interactions(df, resp, terms2, alias, pl / f"interactions_{resp}.png")
        tab = plot_effects_by_class(df, resp, pl / f"effects_by_class_{resp}.png")
        tab.to_csv(out / f"effects_by_class_{resp}.csv")
        plot_effect_vs_size(tab, resp, pl / f"effect_vs_size_{resp}.png")
        plot_residuals(model, df, resp, pl / f"residuals_{resp}.png")
        plot_config_box(df, resp, pl / f"by_config_{resp}.png")
        plot_curvature(df, resp, p_ctr, pl / f"curvature_{resp}.png")
    pd.DataFrame(eff_table).rename(index=labels).to_csv(out / "effects.csv")
    plot_tradeoff(df, pl / "tradeoff_quality_vs_time.png")
    plot_curves_by_config(df, cur, pl / "convergence_by_config.png")
    plot_response_corr(df, pl / "response_correlation.png")

    # active factors: significant main effect, or member of a significant interaction, on log_gap or D
    active = set()
    for resp in ("log_gap", "D"):
        for lab in summary["responses"][resp]["significant"]:
            for word in lab.split("="):
                active.update(ch for ch in word.strip("-") if ch in LETTERS)
    sgn = eff_table["D"]
    fixed = {c: (1.0 if sgn[c] > 0 else -1.0) for c in LETTERS if c not in active}
    summary["active_factors"] = sorted(active)
    summary["fixed_levels_for_inactive"] = fixed
    (out / "screening_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    df.to_csv(out / "runs_with_responses.csv", index=False)
    print(json.dumps({"active_factors": sorted(active), "fixed_levels_for_inactive": fixed,
                      "curvature_p": {r: summary["responses"][r]["curvature_p"] for r, _, _ in RESP}}, indent=1))
    print(f"→ set run_doe.py: PHASE='ccd', CCD_FACTORS={sorted(active)}, CCD_FIXED={fixed}")
    print(f"plots and tables: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Response surface (CCD)
# ─────────────────────────────────────────────────────────────────────────────

def rsm_formula(resp: str, fs: List[str]) -> str:
    lin = " + ".join(fs)
    quad = " + ".join(f"I({f}**2)" for f in fs)
    inter = (f"({lin})**2" if len(fs) > 1 else lin)
    return f"{resp} ~ C(instance) + {inter} + {quad}"


def pure_error(df: pd.DataFrame, resp: str) -> Tuple[float, int]:
    g = df.groupby(["config", "instance"])[resp]
    ss = float(((df[resp] - g.transform("mean")) ** 2).sum())
    dfree = int((g.count() - 1).clip(lower=0).sum())
    return ss, dfree


def predict_avg(model, fs: List[str], Z: np.ndarray, insts: List[str], fixed: Dict[str, float]) -> np.ndarray:
    """Population-average prediction (mean over `insts`) at every row of Z (points × factors)."""
    Z = np.atleast_2d(Z)
    rows = pd.DataFrame(np.repeat(Z, len(insts), axis=0), columns=fs)
    for f, v in fixed.items():
        rows[f] = v
    rows["instance"] = np.tile(np.asarray(insts, dtype=object), len(Z))
    return np.asarray(model.predict(rows)).reshape(len(Z), len(insts)).mean(axis=1)


def plot_coefficients(model, resp: str, path) -> None:
    keep = [t for t in model.params.index if not t.startswith("C(instance)") and t != "Intercept"]
    ci = model.conf_int().loc[keep]
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(keep) + 1.5))
    y = np.arange(len(keep))
    sig = (ci[0] > 0) | (ci[1] < 0)
    ax.errorbar(model.params[keep], y, xerr=[model.params[keep] - ci[0], ci[1] - model.params[keep]],
                fmt="none", ecolor=P.GRAY, capsize=3)
    ax.scatter(model.params[keep], y, color=np.where(sig, P.BLUE, P.GRAY), s=40, zorder=3)
    ax.axvline(0, color=P.INK2, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(keep)
    ax.set_xlabel("coefficient (coded units) with 95% CI")
    ax.set_title(f"Second-order model coefficients — {resp}  (blue = CI excludes 0)")
    _save(fig, path)


def plot_pred_actual(model, df: pd.DataFrame, resp: str, path) -> None:
    d = df.assign(fit=model.fittedvalues).groupby("config")[[resp, "fit"]].mean()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(d["fit"], d[resp], color=P.BLUE, s=40)
    lo, hi = min(d.min()), max(d.max())
    ax.plot([lo, hi], [lo, hi], color=P.INK2, ls=(0, (4, 3)), lw=1)
    for c, r in d.iterrows():
        ax.annotate(c, (r["fit"], r[resp]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("predicted (configuration mean)")
    ax.set_ylabel("observed (configuration mean)")
    ax.set_title(f"Predicted vs observed — {resp}")
    _save(fig, path)


def _grid_pairs(fs: List[str]):
    pairs = list(combinations(fs, 2))
    nc = min(3, len(pairs))
    nr = int(np.ceil(len(pairs) / nc))
    return pairs, nr, nc


def plot_contours(fn, fs: List[str], z_opt: np.ndarray, label: str, title: str, path, better: str) -> None:
    pairs, nr, nc = _grid_pairs(fs)
    if not pairs:
        return
    g = np.linspace(-1, 1, 41)
    fig, axes = plt.subplots(nr, nc, figsize=(5.2 * nc, 4.4 * nr), squeeze=False)
    for ax, (a, b) in zip(axes.flat, pairs):
        ia, ib = fs.index(a), fs.index(b)
        GA, GB = np.meshgrid(g, g)
        pts = np.tile(z_opt, (GA.size, 1))
        pts[:, ia], pts[:, ib] = GA.ravel(), GB.ravel()
        Z = fn(pts).reshape(GA.shape)
        cmap = P.SEQ_BLUES if better == "max" else P.SEQ_BLUES.reversed()
        cs = ax.contourf(g, g, Z, levels=20, cmap=cmap)
        ax.contour(g, g, Z, levels=10, colors=P.INK2, linewidths=0.4)
        ax.plot(z_opt[ia], z_opt[ib], "*", color=P.ORANGE, ms=14, mec=P.INK, label="optimum")
        fa, fb = FACTOR_BY_CODE[a], FACTOR_BY_CODE[b]
        ax.set_xlabel(f"{a}: {fa.label} ({fa.natural(-1):g} … {fa.natural(1):g})")
        ax.set_ylabel(f"{b}: {fb.label} ({fb.natural(-1):g} … {fb.natural(1):g})")
        fig.colorbar(cs, ax=ax, label=label)
        ax.grid(False)
    for ax in list(axes.flat)[len(pairs):]:
        ax.axis("off")
    fig.suptitle(title + "  (other factors at the optimum)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    _save(fig, path)


def plot_surfaces(fn, fs: List[str], z_opt: np.ndarray, label: str, title: str, path, better: str) -> None:
    pairs, nr, nc = _grid_pairs(fs)
    if not pairs:
        return
    g = np.linspace(-1, 1, 31)
    fig = plt.figure(figsize=(5.6 * nc, 4.8 * nr))
    for k, (a, b) in enumerate(pairs):
        ax = fig.add_subplot(nr, nc, k + 1, projection="3d")
        ia, ib = fs.index(a), fs.index(b)
        A, B = np.meshgrid(g, g)
        pts = np.tile(z_opt, (A.size, 1))
        pts[:, ia], pts[:, ib] = A.ravel(), B.ravel()
        Z = fn(pts).reshape(A.shape)
        ax.plot_surface(A, B, Z, cmap=P.SEQ_BLUES if better == "max" else P.SEQ_BLUES.reversed(),
                        linewidth=0, antialiased=True, alpha=0.95)
        ax.scatter([z_opt[ia]], [z_opt[ib]], [float(fn(z_opt[None])[0])], color=P.ORANGE, s=60, depthshade=False)
        ax.set_xlabel(a)
        ax.set_ylabel(b)
        ax.set_zlabel(label, fontsize=8)
        ax.set_title(f"{a} × {b}", fontsize=9)
    fig.suptitle(title + "  (other factors at the optimum)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    _save(fig, path)


def plot_perturbation(fn, fs: List[str], z_opt: np.ndarray, label: str, title: str, path) -> None:
    g = np.linspace(-1, 1, 41)
    fig, ax = plt.subplots(figsize=(8, 5))
    for f in fs:
        i = fs.index(f)
        pts = np.tile(z_opt, (len(g), 1))
        pts[:, i] = g
        ys = fn(pts)
        ax.plot(g - z_opt[i], ys, lw=2, color=FCOL[LETTERS.index(f)], label=f"{f}: {FACTOR_BY_CODE[f].label}")
    ax.axvline(0, color=P.INK2, lw=0.8)
    ax.set_xlabel("deviation from the optimum (coded units)")
    ax.set_ylabel(label)
    ax.set_title(title)
    ax.legend(fontsize=8)
    _save(fig, path)


def ccd_analysis() -> None:
    x = load_phase("ccd")
    if x is None:
        raise SystemExit("no ccd records found")
    rec, cur = x
    df = responses(rec, cur, references(TUNE_PHASES))
    out = outdir("ccd")
    pl = out / "plots"
    fs = [c for c in LETTERS if df[c].nunique() > 1]
    fixed = {c: float(df[c].iloc[0]) for c in LETTERS if c not in fs}
    insts = sorted(df["instance"].unique())
    models, summ = {}, {"factors": fs, "fixed": fixed, "responses": {}}
    for resp, _, better in RESP:
        m = smf.ols(rsm_formula(resp, fs), data=df).fit()
        models[resp] = m
        try:
            aov = anova_lm(m, typ=2)
        except Exception:
            aov = anova_lm(m, typ=1)
        aov.to_csv(out / f"anova_{resp}.csv")
        m.summary2().tables[1].to_csv(out / f"coefficients_{resp}.csv")
        ss_pe, df_pe = pure_error(df, resp)
        ss_res, df_res = float(m.ssr), int(m.df_resid)
        if df_pe > 0 and df_res > df_pe:
            f_lof = ((ss_res - ss_pe) / (df_res - df_pe)) / (ss_pe / df_pe)
            p_lof = float(stats.f.sf(f_lof, df_res - df_pe, df_pe))
        else:
            f_lof, p_lof = float("nan"), float("nan")
        h = m.get_influence().hat_matrix_diag
        press = float(np.sum((m.resid / (1 - h)) ** 2))
        sst = float(np.sum((df[resp] - df[resp].mean()) ** 2))
        summ["responses"][resp] = dict(r2=m.rsquared, r2_adj=m.rsquared_adj, r2_pred=1 - press / sst,
                                       lack_of_fit_F=f_lof, lack_of_fit_p=p_lof)
        plot_coefficients(m, resp, pl / f"coefficients_{resp}.png")
        plot_pred_actual(m, df, resp, pl / f"predicted_vs_observed_{resp}.png")
        plot_residuals(m, df, resp, pl / f"residuals_{resp}.png")

    # desirability from the fitted log_gap and ttb_frac surfaces
    cm = df.groupby("config")[["log_gap", "ttb_frac"]].mean()
    lq, uq = cm["log_gap"].min(), cm["log_gap"].max()
    lt, ut = cm["ttb_frac"].min(), cm["ttb_frac"].max()

    def d_overall(Z, subset=insts):
        """Overall desirability at every row of Z (vectorised)."""
        yq = predict_avg(models["log_gap"], fs, Z, subset, fixed)
        yt = predict_avg(models["ttb_frac"], fs, Z, subset, fixed)
        dq = np.clip((uq - yq) / max(uq - lq, 1e-12), 0, 1)
        dt = np.clip((ut - yt) / max(ut - lt, 1e-12), 0, 1)
        return (dq ** QUALITY_WEIGHT * dt ** SPEED_WEIGHT) ** (1 / (QUALITY_WEIGHT + SPEED_WEIGHT))

    def maximise(fn) -> np.ndarray:
        rng = np.random.default_rng(0)
        best_z, best_v = None, -np.inf
        for z0 in [np.zeros(len(fs))] + [rng.uniform(-1, 1, len(fs)) for _ in range(40)]:
            r = minimize(lambda z: -float(fn(z[None])[0]), z0, method="L-BFGS-B", bounds=[(-1, 1)] * len(fs))
            if -r.fun > best_v:
                best_v, best_z = -r.fun, r.x
        return best_z

    z_opt = maximise(d_overall)
    for resp, _, better in RESP:
        fn = (lambda z, m=models[resp]: predict_avg(m, fs, z, insts, fixed))
        plot_contours(fn, fs, z_opt, resp, f"Response surface contours — {resp}", pl / f"contours_{resp}.png", better)
        plot_surfaces(fn, fs, z_opt, resp, f"Response surfaces — {resp}", pl / f"surfaces_{resp}.png", better)
        plot_perturbation(fn, fs, z_opt, RESP_LABEL[resp], f"Perturbation plot — {resp}", pl / f"perturbation_{resp}.png")
    plot_contours(d_overall, fs, z_opt, "D", "Overall desirability (quality and time to best)",
                  pl / "contours_desirability.png", "max")
    plot_surfaces(d_overall, fs, z_opt, "D", "Overall desirability", pl / "surfaces_desirability.png", "max")

    # robustness: optimum when only one instance size is considered
    by_n = {}
    for nv in sorted(df["n"].unique()):
        sub = sorted(df.loc[df["n"] == nv, "instance"].unique())
        by_n[int(nv)] = maximise(lambda z, s=sub: d_overall(z, s)).tolist()
    fig, ax = plt.subplots(figsize=(8, 5))
    ns = sorted(by_n)
    for i, f in enumerate(fs):
        ax.plot(ns, [by_n[nv][i] for nv in ns], marker="o", lw=2, color=FCOL[LETTERS.index(f)],
                label=f"{f}: {FACTOR_BY_CODE[f].label}")
        ax.axhline(z_opt[i], color=FCOL[LETTERS.index(f)], lw=0.8, ls=(0, (4, 3)))
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(v) for v in ns])
    ax.minorticks_off()
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("instance size n")
    ax.set_ylabel("optimal coded level")
    ax.set_title("Optimum per instance size (dashed: pooled optimum)")
    ax.legend(fontsize=8)
    _save(fig, pl / "optimum_by_size.png")

    # predictions at the optimum (population average and observation-level interval)
    pred = {}
    for resp, _, _ in RESP:
        rows = pd.DataFrame([{**dict(zip(fs, z_opt)), **fixed, "instance": i} for i in insts])
        sf = models[resp].get_prediction(rows).summary_frame(alpha=ALPHA)
        pred[resp] = dict(mean=float(sf["mean"].mean()), pi_low=float(sf["obs_ci_lower"].mean()),
                          pi_high=float(sf["obs_ci_upper"].mean()))
    coded = {**dict(zip(fs, map(float, z_opt))), **fixed}
    prm = decode(coded, SAParams())
    prm.to_json(RESULTS / "recommended_params.json")
    summ.update(optimum_coded=coded, optimum_desirability=float(d_overall(z_opt[None])[0]), predictions=pred,
                optimum_by_size=by_n, desirability_bounds=dict(log_gap=[lq, uq], ttb_frac=[lt, ut]))
    (out / "ccd_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    df.to_csv(out / "runs_with_responses.csv", index=False)
    print(json.dumps({"optimum_coded": coded, "desirability": float(d_overall(z_opt[None])[0]),
                      "predictions": pred}, indent=1))
    print(f"recommended parameters: {RESULTS / 'recommended_params.json'}")
    print(f"plots and tables: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Confirmation and iteration budget
# ─────────────────────────────────────────────────────────────────────────────

def confirm_analysis() -> None:
    x = load_phase("confirm")
    if x is None:
        raise SystemExit("no confirm records found")
    rec, cur = x
    df = responses(rec, cur, references(HOLD_PHASES))
    out = outdir("confirm")
    pl = out / "plots"
    sp = RESULTS / "analysis_ccd" / "ccd_summary.json"
    pred = json.loads(sp.read_text())["predictions"] if sp.exists() else {}
    rows = []
    for resp, label, _ in RESP:
        obs = df[resp]
        ci = 1.96 * obs.std(ddof=1) / np.sqrt(len(obs))
        p = pred.get(resp, {})
        rows.append(dict(response=resp, observed_mean=obs.mean(), observed_ci_low=obs.mean() - ci,
                         observed_ci_high=obs.mean() + ci, predicted_mean=p.get("mean"),
                         predicted_pi_low=p.get("pi_low"), predicted_pi_high=p.get("pi_high"),
                         inside_pi=(p.get("pi_low", -np.inf) <= obs.mean() <= p.get("pi_high", np.inf)) if p else None))
        insts = sorted(df["instance"].unique(), key=lambda s: (int(s.split("_")[0][1:]), s))
        fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(insts) + 2), 4.5))
        for k, i in enumerate(insts):
            y = df.loc[df["instance"] == i, resp]
            ax.scatter(np.full(len(y), k), y, color=P.BLUE, s=14, alpha=0.7)
        if p:
            ax.axhspan(p["pi_low"], p["pi_high"], color=P.AQUA, alpha=0.15, label="CCD prediction interval")
            ax.axhline(p["mean"], color=P.AQUA, lw=1.5, label="CCD prediction")
        ax.axhline(obs.mean(), color=P.ORANGE, lw=1.5, ls=(0, (4, 3)), label="observed mean")
        ax.set_xticks(range(len(insts)))
        ax.set_xticklabels(insts, rotation=90, fontsize=7)
        ax.set_ylabel(label)
        ax.set_title(f"Confirmation on held-out instances — {resp} (dots = seeds)")
        ax.legend(loc="upper right")
        _save(fig, pl / f"confirm_{resp}.png")
    pd.DataFrame(rows).to_csv(out / "confirmation.csv", index=False)
    plot_curves_by_config(df, cur, pl / "convergence_holdout.png")
    plot_tradeoff(df, pl / "tradeoff_quality_vs_time.png")
    df.to_csv(out / "runs_with_responses.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"plots and tables: {out}")


def budget_analysis() -> None:
    # long anchor runs at the recommended settings show where the curves flatten; confirm runs otherwise
    x = load_phase("anchor_holdout") or load_phase("confirm")
    if x is None:
        raise SystemExit("the budget is read from the anchor_holdout (or confirm) runs; none found")
    rec, cur = x
    df = responses(rec, cur, references(HOLD_PHASES))
    out = outdir("budget")
    pl = out / "plots"
    gt = gap_curves(df, cur)
    excess = gt - gt[:, -1:]
    fr = np.linspace(0, 1, gt.shape[1])
    ns = sorted(df["n"].unique())
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for nv, colr in zip(ns, FCOL):
        mask = (df["n"] == nv).to_numpy()
        med = np.median(excess[mask], axis=0)
        q10, q90 = np.quantile(excess[mask], [0.1, 0.9], axis=0)
        k = int(np.argmax(med <= BUDGET_TOL)) if np.any(med <= BUDGET_TOL) else len(fr) - 1
        budget_moves = fr[k] * float(df.loc[mask, "budget"].iloc[0])
        rows.append(dict(n=int(nv), fraction=float(fr[k]), moves_per_replica=budget_moves,
                         per_aircraft=budget_moves / nv, q90_excess_at_fraction=float(q90[k])))
        axes[0].plot(fr, med + 1e-4, color=colr, lw=2, label=f"n = {nv}")
        axes[0].fill_between(fr, q10 + 1e-4, q90 + 1e-4, color=colr, alpha=0.12)
        axes[0].axvline(fr[k], color=colr, lw=0.8, ls=(0, (4, 3)))
    axes[0].axhline(BUDGET_TOL, color=P.INK2, lw=1, ls=(0, (1, 2)), label=f"tolerance {BUDGET_TOL} points")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("fraction of the run's iteration budget")
    axes[0].set_ylabel("gap still to be gained (percentage points, log)")
    axes[0].set_title("Remaining improvement along the budget (median, 10–90% band)")
    axes[0].legend(fontsize=8)
    tab = pd.DataFrame(rows)
    b, loga = np.polyfit(np.log(tab["n"]), np.log(np.maximum(tab["moves_per_replica"], 1.0)), 1)
    a = float(np.exp(loga))
    nn = np.linspace(min(ns), max(ns), 100)
    axes[1].scatter(tab["n"], tab["moves_per_replica"], color=P.BLUE, s=50, zorder=3, label="read off the curves")
    axes[1].plot(nn, a * nn ** b, color=P.ORANGE, lw=2, label=f"rule: {a:.3g} · n^{b:.2f}")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("instance size n")
    axes[1].set_ylabel("iterations (moves per replica)")
    axes[1].set_title("Iteration budget rule")
    axes[1].legend()
    fig.tight_layout()
    _save(fig, pl / "budget_rule.png")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot([df.loc[df["n"] == nv, "ttb_frac"] for nv in ns], showfliers=True)
    ax.set_xticks(range(1, len(ns) + 1))
    ax.set_xticklabels([f"n = {v}" for v in ns])
    ax.set_ylabel("time to best (fraction of budget)")
    ax.set_title("Time to best on held-out instances")
    _save(fig, pl / "time_to_best_by_size.png")
    tab.to_csv(out / "budget_by_size.csv", index=False)
    rp = RESULTS / "recommended_params.json"
    prm = SAParams.from_json(rp) if rp.exists() else SAParams()
    prm.budget_a, prm.budget_b = a, float(b)
    prm.to_json(rp)
    print(tab.to_string(index=False))
    print(f"budget rule: iterations = {a:.4g} · n^{b:.3f}  (written to {rp})")


if __name__ == "__main__":
    {"screening": screening_analysis, "ccd": ccd_analysis, "confirm": confirm_analysis,
     "budget": budget_analysis}[ANALYSIS]()
