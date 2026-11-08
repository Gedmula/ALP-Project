"""
screening_deep.py — extended analysis of the screening phase
===============================================================================
Run from the repository root (after analyze.py has been run, or on its own):

    python sa_single/doe/screening_deep.py

analyze.py does the standard screening analysis (Pareto, half-normal, main-
effect and interaction plots, ANOVA).  This script adds what is needed to
decide between a foldover and a central composite design:

  1  effects with instance-clustered inference         (instances are the units)
  2  the 16 factorial runs read as a full 2^4 in A–D   (de-aliased contrasts)
  3  consistency of every effect across instances      (sign counts, signed-rank)
  4  control × noise interactions                      (factor × congestion, …)
  5  curvature                                         (centre points vs factorial)
  6  provisional response surfaces on A–D              (equal-curvature model)
  7  sensitivity of the optimum to the curvature split
  8  wall-clock effects and the cost of the next phase

Outputs: sa_single/doe/results/analysis_screening/deep/{tables,plots}

Why "informative" instances.  On 28 of the 48 instances every one of the 60 runs
(20 configurations × 3 seeds) ends on the same cost, so those instances have
exactly zero effect for every factor and only dilute the effect sizes by 20/48.
Effect sizes are therefore reported on the 20 instances where the gap varies,
and the all-48 values (the ones in SCREENING_REPORT.md) are kept alongside.
"""
from __future__ import annotations

import json
import sys
import warnings
from itertools import combinations
from pathlib import Path
from typing import Callable, Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import minimize

from sa_single import plots as P
from sa_single.doe import analyze as AN
from sa_single.doe import design as Dsg
from sa_single.params import FACTOR_BY_CODE

warnings.filterwarnings("ignore")

LETTERS = Dsg.LETTERS
ACTIVE = list("ABCD")
OUT = AN.RESULTS / "analysis_screening" / "deep"
TAB, PLT = OUT / "tables", OUT / "plots"
RESP3 = ["log_gap", "ttb_frac", "desir"]
RESP_NAME = {"log_gap": "log10(gap % + 0.01)", "ttb_frac": "time to best (fraction of budget)",
             "desir": "desirability", "log_wall": "ln(wall-clock seconds)", "mean_gap": "mean gap over the run (%)"}
GROUP_ORDER = ["c<1 wide", "c1.0 tight", "c1.0 wide"]
GCOL = {"c<1 wide": P.AQUA, "c1.0 tight": P.YELLOW, "c1.0 wide": P.BLUE}
FCOL = AN.FCOL


# ─────────────────────────────────────────────────────────────────────────────
# Data and the clustered regression
# ─────────────────────────────────────────────────────────────────────────────

def load() -> pd.DataFrame:
    rec, cur = AN.load_phase("screening")
    df = AN.responses(rec, cur, AN.references(["screening"]))
    df["log_wall"] = np.log(df["wall_s"])
    df["informative"] = df.groupby("instance")["gap"].transform("std") > 1e-9
    df["group"] = np.where(df["congestion"] >= 1.0, "c1.0 ", "c<1 ") + df["windows"].astype(str)
    df["short"] = df["instance"].str.replace("_tune_large", "", regex=False)
    return df


def term_columns(d: pd.DataFrame, terms: List[str]) -> pd.DataFrame:
    X = pd.DataFrame(index=d.index)
    for t in terms:
        if t == "ctr":
            X[t] = d["ctr"]
        elif t == "Q":
            X[t] = (d[ACTIVE] ** 2).sum(axis=1)
        else:
            X[t] = np.prod([d[c].to_numpy() for c in t], axis=0)
    return X


def ols(d: pd.DataFrame, y: str, X: pd.DataFrame):
    """OLS with instance fixed effects; standard errors clustered by instance, t reference df = clusters − 1."""
    dm = pd.get_dummies(d["instance"], drop_first=True).astype(float)
    XX = sm.add_constant(pd.concat([X, dm], axis=1))
    g = pd.factorize(d["instance"])[0]
    return sm.OLS(d[y], XX).fit(cov_type="cluster", cov_kwds={"groups": g}, use_t=True)


def effects(d: pd.DataFrame, y: str, terms: List[str], ctr: bool = True) -> pd.DataFrame:
    """Effect = mean(+1) − mean(−1) = 2·coefficient (coded ±1).  'ctr' is the centre-minus-factorial difference."""
    cols = term_columns(d, terms + (["ctr"] if ctr else []))
    r = ols(d, y, cols)
    rows = []
    for t in cols.columns:
        k = 1.0 if t == "ctr" else 2.0
        b, se = k * r.params[t], k * r.bse[t]
        tcrit = stats.t.ppf(0.975, r.df_resid)
        rows.append(dict(term="curvature (centre − factorial)" if t == "ctr" else t, effect=b, se=se,
                         t=b / se, p=float(r.pvalues[t]), ci_low=b - tcrit * se, ci_high=b + tcrit * se))
    return pd.DataFrame(rows)


def alias_label(term: str, cfg: pd.DataFrame) -> str:
    """Main effects and two-factor interactions of the 7 factors that share a contrast with `term`."""
    cols = Dsg.model_columns(cfg)
    v = np.prod([cfg[c].to_numpy() for c in term], axis=0)
    names = [c for c in cols.columns if c != term and
             (np.array_equal(cols[c].to_numpy(), v) or np.array_equal(cols[c].to_numpy(), -v))]
    return "=".join([term] + names)


def save(fig, name: str) -> None:
    PLT.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def star(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


# ─────────────────────────────────────────────────────────────────────────────
# 1  Effects with clustered inference
# ─────────────────────────────────────────────────────────────────────────────

def part1_effects(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, d in (("all48", df), ("informative20", df[df["informative"]])):
        for y in RESP3 + ["log_wall", "mean_gap"]:
            e = effects(d, y, LETTERS)
            e.insert(0, "response", y)
            e.insert(0, "scope", scope)
            rows.append(e)
    tab = pd.concat(rows, ignore_index=True)
    tab["pct_change_wall"] = np.where(tab["response"] == "log_wall", 100 * (np.exp(tab["effect"]) - 1), np.nan)
    tab.to_csv(TAB / "effects_clustered.csv", index=False)

    # forest plot on the informative instances
    d = tab[tab["scope"] == "informative20"]
    panels = RESP3 + ["log_wall"]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 4.6), sharey=True)
    terms = LETTERS + ["curvature (centre − factorial)"]
    ylab = [f"{c}: {FACTOR_BY_CODE[c].label}" for c in LETTERS] + ["curvature (centre − factorial)"]
    for ax, y in zip(axes, panels):
        s = d[d["response"] == y].set_index("term").loc[terms]
        pos = np.arange(len(terms))[::-1]
        sig = s["p"] < 0.05
        ax.errorbar(s["effect"], pos, xerr=[s["effect"] - s["ci_low"], s["ci_high"] - s["effect"]],
                    fmt="none", ecolor=P.GRAY, capsize=3, lw=1.2)
        ax.scatter(s["effect"], pos, color=np.where(sig, P.BLUE, P.GRAY), s=44, zorder=3)
        ax.axvline(0, color=P.INK2, lw=0.8)
        ax.set_yticks(pos)
        ax.set_yticklabels(ylab)
        ax.set_title(RESP_NAME[y], fontsize=9)
        ax.set_xlabel("effect (high − low), 95% CI")
    fig.suptitle("Effects with instance-clustered 95% CI, 20 informative instances (blue = p < 0.05)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "effects_forest.png")
    return tab


# ─────────────────────────────────────────────────────────────────────────────
# 2  Full 2^4 view of the 16 factorial runs
# ─────────────────────────────────────────────────────────────────────────────

def part2_full_factorial(df: pd.DataFrame) -> pd.DataFrame:
    cfg = df[df["point"] != "center"].drop_duplicates("config")
    terms = ["".join(t) for r in range(1, 5) for t in combinations(ACTIVE, r)]
    labels = {t: alias_label(t, cfg) for t in terms}
    rows = []
    d = df[df["informative"]]
    for y in RESP3:
        e = effects(d, y, terms)
        e = e[e["term"] != "curvature (centre − factorial)"].copy()
        e["label"] = e["term"].map(labels)
        e.insert(0, "response", y)
        rows.append(e)
        eff = e.set_index("term")["effect"]
        pse, me, sme = AN.lenth(eff)
        AN.plot_halfnormal(eff, labels, me, y, PLT / f"halfnormal_2to4_{y}.png")
        AN.plot_pareto(eff, labels, me, sme, y, PLT / f"pareto_2to4_{y}.png")
    tab = pd.concat(rows, ignore_index=True)
    tab.to_csv(TAB / "effects_2to4_dealiased.csv", index=False)
    return tab


def plot_interactions_2to4(df: pd.DataFrame, e24: pd.DataFrame) -> None:
    d = df[df["informative"]].copy()
    for y in ("log_gap", "desir"):
        d["adj"] = d[y] - d.groupby("instance")[y].transform("mean") + d[y].mean()
        fact = d[d["point"] != "center"]
        pairs = list(combinations(ACTIVE, 2))
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
        for ax, (a, b) in zip(axes.flat, pairs):
            for lv, ls, col in ((-1, (0, (4, 3)), P.GRAY), (1, "-", FCOL[LETTERS.index(b)])):
                yy = [fact.loc[(fact[a] == la) & (fact[b] == lv), "adj"].mean() for la in (-1, 1)]
                ax.plot([-1, 1], yy, ls=ls, color=col, lw=2, marker="o", ms=6,
                        label=f"{b} = {FACTOR_BY_CODE[b].natural(lv):g}")
            r = e24[(e24["response"] == y) & (e24["term"] == a + b)].iloc[0]
            ax.set_title(f"{a} × {b}   effect {r['effect']:+.3f}  (t = {r['t']:.1f}){star(r['p'])}", fontsize=9)
            ax.set_xticks([-1, 1])
            ax.set_xticklabels([f"{FACTOR_BY_CODE[a].natural(-1):g}", f"{FACTOR_BY_CODE[a].natural(1):g}"])
            ax.set_xlabel(f"{a}: {FACTOR_BY_CODE[a].label}")
            ax.legend(fontsize=7, loc="best")
        for ax in axes[:, 0]:
            ax.set_ylabel(RESP_NAME[y] + " (instance-adjusted)")
        fig.suptitle(f"Two-factor interactions among A–D, de-aliased (E, F, G inert) — {y}, 20 informative instances",
                     x=0.01, ha="left", fontweight="bold")
        fig.tight_layout()
        save(fig, f"interactions_AD_{y}.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3  Consistency across instances, 4  control × noise
# ─────────────────────────────────────────────────────────────────────────────

def part3_consistency(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["informative"] & (df["point"] != "center")]
    rows = []
    for inst, g in d.groupby("instance"):
        rows.append({"instance": g["short"].iloc[0], "n": int(g["n"].iloc[0]), "congestion": g["congestion"].iloc[0],
                     "windows": g["windows"].iloc[0], "group": g["group"].iloc[0],
                     **{c: g.loc[g[c] > 0, "log_gap"].mean() - g.loc[g[c] < 0, "log_gap"].mean() for c in LETTERS}})
    pe = pd.DataFrame(rows)
    pe["g_ord"] = pe["group"].map({g: i for i, g in enumerate(GROUP_ORDER)})
    pe = pe.sort_values(["g_ord", "n"]).drop(columns="g_ord").reset_index(drop=True)
    pe.to_csv(TAB / "instance_level_effects_log_gap.csv", index=False)

    summ = []
    for c in LETTERS:
        x = pe[c]
        summ.append(dict(factor=c, mean_effect=x.mean(), median_effect=x.median(), n_positive=int((x > 0).sum()),
                         n_negative=int((x < 0).sum()), signed_rank_p=float(stats.wilcoxon(x).pvalue)))
    summ = pd.DataFrame(summ)
    summ.to_csv(TAB / "instance_consistency_log_gap.csv", index=False)
    by_group = pe.groupby("group")[LETTERS].mean().reindex(GROUP_ORDER)
    by_group["n_instances"] = pe.groupby("group").size().reindex(GROUP_ORDER)
    by_group.to_csv(TAB / "effects_by_instance_group_log_gap.csv")

    # heatmap
    lim = float(np.abs(pe[LETTERS].to_numpy()).max())
    fig, ax = plt.subplots(figsize=(8.5, 0.34 * len(pe) + 2.6))
    im = ax.imshow(pe[LETTERS].to_numpy(), cmap=AN.DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_yticks(range(len(pe)))
    ax.set_yticklabels(pe["instance"] + "  (" + pe["group"] + ")", fontsize=8)
    ax.set_xticks(range(len(LETTERS)))
    ax.set_xticklabels([f"{c}\n{r.n_positive}+ / {r.n_negative}−\np={r.signed_rank_p:.3f}"
                        for c, r in zip(LETTERS, summ.itertuples())], fontsize=8)
    for i in range(len(pe)):
        for j, c in enumerate(LETTERS):
            ax.text(j, i, f"{pe.at[i, c]:.2f}", ha="center", va="center", fontsize=7, color=P.INK)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="effect on log10(gap % + 0.01)   (positive = high level worse)", shrink=0.7)
    ax.set_title("Is each effect the same on every instance?  (20 informative instances)")
    save(fig, "effect_consistency_by_instance.png")

    # group bars
    fig, ax = plt.subplots(figsize=(8, 4.8))
    w = 0.26
    for k, g in enumerate(GROUP_ORDER):
        ax.bar(np.arange(len(ACTIVE)) + (k - 1) * w, by_group.loc[g, ACTIVE], width=w, color=GCOL[g],
               label=f"{g}  (n = {int(by_group.loc[g, 'n_instances'])} instances)")
    ax.axhline(0, color=P.INK2, lw=0.8)
    ax.set_xticks(range(len(ACTIVE)))
    ax.set_xticklabels([f"{c}: {FACTOR_BY_CODE[c].label}" for c in ACTIVE], fontsize=8)
    ax.set_ylabel("mean effect on log10(gap % + 0.01)")
    ax.set_title("Effects by instance group: the window (D) effect is a congestion effect")
    ax.legend(fontsize=8)
    save(fig, "effects_by_instance_group.png")
    return summ


def part4_noise_interactions(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["informative"] & (df["point"] != "center")].copy()
    d["cong"] = np.where(d["congestion"] >= 1.0, 1.0, -1.0)
    d["wide"] = np.where(d["windows"] == "wide", 1.0, -1.0)
    d["large"] = np.where(d["n"] >= 300, 1.0, -1.0)
    rows = []
    for y in ("log_gap", "ttb_frac"):
        for nz, nz_name in (("cong", "congestion 1.0 vs <1"), ("wide", "wide vs tight windows"),
                            ("large", "n ≥ 300 vs < 300")):
            for c in LETTERS:
                X = pd.DataFrame({c: d[c], c + "_x_" + nz: d[c] * d[nz]})
                r = ols(d, y, X)
                b, bi = r.params[c], r.params[c + "_x_" + nz]
                rows.append(dict(response=y, noise=nz_name, factor=c, effect_at_low=2 * (b - bi),
                                 effect_at_high=2 * (b + bi), interaction_t=float(r.tvalues[c + "_x_" + nz]),
                                 interaction_p=float(r.pvalues[c + "_x_" + nz])))
    tab = pd.DataFrame(rows)
    tab.to_csv(TAB / "control_x_noise.csv", index=False)
    return tab


# ─────────────────────────────────────────────────────────────────────────────
# 5  Curvature and configuration ranking
# ─────────────────────────────────────────────────────────────────────────────

def part5_curvature_and_configs(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["informative"]]
    per = []
    for inst, g in d.groupby("instance"):
        per.append(dict(instance=g["short"].iloc[0], group=g["group"].iloc[0], n=int(g["n"].iloc[0]),
                        **{y: g.loc[g["point"] == "center", y].mean() - g.loc[g["point"] != "center", y].mean()
                           for y in RESP3}))
    per = pd.DataFrame(per).sort_values("log_gap")
    per.to_csv(TAB / "curvature_by_instance.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(range(len(per)), per["log_gap"], color=[GCOL[g] for g in per["group"]])
    ax.axhline(0, color=P.INK2, lw=0.8)
    ax.set_xticks(range(len(per)))
    ax.set_xticklabels(per["instance"], rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("centre − factorial mean, log10(gap % + 0.01)")
    ax.set_title(f"Curvature on each instance: centre better on {(per['log_gap'] < 0).sum()} of {len(per)}")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=GCOL[g], label=g) for g in GROUP_ORDER], fontsize=8)
    save(fig, "curvature_by_instance.png")

    # configuration ranking (instance-adjusted, so configurations are compared on the same instances)
    rows = []
    for cfg, g in d.groupby("config"):
        pi = g.groupby("instance")[["log_gap", "ttb_frac", "desir"]].mean()
        base = d.groupby("instance")[["log_gap", "ttb_frac", "desir"]].mean()
        dev = pi - base
        rows.append(dict(config=cfg, point=g["point"].iloc[0], **{c: float(g[c].iloc[0]) for c in LETTERS},
                         mean_gap_pct=g["gap"].mean(), log_gap=pi["log_gap"].mean(),
                         log_gap_se=dev["log_gap"].std(ddof=1) / np.sqrt(len(dev)), ttb_frac=pi["ttb_frac"].mean(),
                         desir=pi["desir"].mean(), wall_s=g["wall_s"].mean()))
    cfg = pd.DataFrame(rows).sort_values("log_gap").reset_index(drop=True)
    cfg["rank_log_gap"] = np.arange(1, len(cfg) + 1)
    cfg["rank_desir"] = cfg["desir"].rank(ascending=False).astype(int)
    cfg.to_csv(TAB / "configuration_ranking_informative.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 6.2))
    y = np.arange(len(cfg))[::-1]
    col = [P.ORANGE if p == "center" else P.BLUE for p in cfg["point"]]
    ax.errorbar(cfg["log_gap"], y, xerr=1.96 * cfg["log_gap_se"], fmt="none", ecolor=P.GRAY, capsize=2)
    ax.scatter(cfg["log_gap"], y, c=col, s=40, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.config}  A{r.A:+.0f} B{r.B:+.0f} C{r.C:+.0f} D{r.D:+.0f} E{r.E:+.0f} F{r.F:+.0f} G{r.G:+.0f}"
                        if r.point != "center" else f"{r.config}  centre point" for r in cfg.itertuples()],
                       fontsize=7, family="monospace")
    ax.set_xlabel("mean log10(gap % + 0.01) over the 20 informative instances (±1.96 se)")
    ax.set_title("Configurations ranked on quality  (orange = centre points)")
    save(fig, "configuration_ranking.png")
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# 6  Provisional surfaces on A–D, 7  where could the optimum be?
# ─────────────────────────────────────────────────────────────────────────────

PAIRS = list(combinations(range(4), 2))


class SurfaceModel:
    """y = mean intercept + Σ b_i z_i + Σ b_ij z_i z_j + Σ β_i z_i²,  β_i = 4·q·w_i.

    The screening has only one curvature degree of freedom (centre points versus the factorial average, here the
    coefficient q on Q = Σ z²), so the split of the total curvature 4q across the pure quadratic terms is an assumption
    w (default equal).  Everything about the optimum that depends on w is therefore provisional."""

    def __init__(self, d: pd.DataFrame, y: str):
        X = term_columns(d, ACTIVE + ["".join(p) for p in combinations(ACTIVE, 2)] + ["Q"])
        for c in "EFG":
            X[c] = d[c]
        r = ols(d, y, X)
        dummies = [k for k in r.params.index if k not in X.columns and k != "const"]
        self.base = float(r.params["const"] + r.params[dummies].sum() / (len(dummies) + 1))
        self.b = r.params[ACTIVE].to_numpy()
        self.bij = r.params[["".join(p) for p in combinations(ACTIVE, 2)]].to_numpy()
        self.q = float(r.params["Q"])
        self.q_se = float(r.bse["Q"])
        self.q_t = float(r.tvalues["Q"])
        self.res = r
        self.y = y

    def __call__(self, Z, w=None):
        Z = np.atleast_2d(Z)
        w = np.full(4, 0.25) if w is None else np.asarray(w, dtype=float)
        v = self.base + Z @ self.b + 4 * self.q * ((Z ** 2) @ w)
        for k, (i, j) in enumerate(PAIRS):
            v = v + self.bij[k] * Z[:, i] * Z[:, j]
        return v


def optimise(fn: Callable, better: str, bound: float, starts: int = 40) -> np.ndarray:
    sgn = 1.0 if better == "min" else -1.0
    rng = np.random.default_rng(0)
    best = None
    for z0 in [np.zeros(4)] + list(rng.uniform(-bound, bound, (starts, 4))):
        r = minimize(lambda z: sgn * float(fn(z[None])[0]), z0, method="L-BFGS-B", bounds=[(-bound, bound)] * 4)
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def part6_surfaces(df: pd.DataFrame) -> Dict[str, SurfaceModel]:
    d = df[df["informative"]]
    models = {y: SurfaceModel(d, y) for y in RESP3}
    better = {"log_gap": "min", "ttb_frac": "min", "desir": "max"}
    summ, zopt = {}, {}
    for y, m in models.items():
        zopt[y] = optimise(m, better[y], 1.0)
        summ[y] = dict(q=m.q, q_se=m.q_se, q_t=m.q_t, base=m.base, b=dict(zip(ACTIVE, m.b.round(5))),
                       bij=dict(zip(["".join(p) for p in combinations(ACTIVE, 2)], m.bij.round(5))),
                       provisional_optimum_in_box_equal_curvature=dict(zip(ACTIVE, zopt[y].round(3))),
                       predicted_at_optimum=float(m(zopt[y][None])[0]), predicted_at_centre=float(m(np.zeros(4))[0]))
        label = RESP_NAME[y] + " (informative instances)"
        AN.plot_contours(m, ACTIVE, zopt[y], label, f"Provisional response surface — {y}", PLT / f"contours_{y}.png",
                         better[y])
        AN.plot_surfaces(m, ACTIVE, zopt[y], label, f"Provisional response surface — {y}", PLT / f"surfaces_{y}.png",
                         better[y])
        AN.plot_perturbation(m, ACTIVE, zopt[y], label, f"Perturbation plot around the provisional optimum — {y}",
                             PLT / f"perturbation_{y}.png")
    (TAB / "surface_models.json").write_text(json.dumps(summ, indent=1, default=float))
    return models


def part7_sensitivity(models: Dict[str, SurfaceModel]) -> pd.DataFrame:
    scen = {"equal": [.25] * 4, "all on A": [1, 0, 0, 0], "all on B": [0, 1, 0, 0], "all on C": [0, 0, 1, 0],
            "all on D": [0, 0, 0, 1], "C and D": [0, 0, .5, .5], "A and D": [.5, 0, 0, .5]}
    better = {"log_gap": "min", "ttb_frac": "min", "desir": "max"}
    rows = []
    for y, m in models.items():
        for name, w in scen.items():
            fn = lambda Z, m=m, w=w: m(Z, w)
            zi, ze = optimise(fn, better[y], 1.0), optimise(fn, better[y], 2.0)
            rows.append(dict(response=y, curvature_on=name,
                             **{f"{c}_in_box": zi[i] for i, c in enumerate(ACTIVE)},
                             **{f"{c}_extended": ze[i] for i, c in enumerate(ACTIVE)}))
    tab = pd.DataFrame(rows)
    tab.to_csv(TAB / "optimum_sensitivity.csv", index=False)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.8), sharey=True)
    marks = {"log_gap": ("o", P.BLUE), "desir": ("s", P.ORANGE)}
    names = list(scen)
    for ax, (i, c) in zip(axes, enumerate(ACTIVE)):
        ax.axhspan(-1, 1, color=P.AQUA, alpha=0.08)
        ax.axhline(0, color=P.INK2, lw=0.6)
        for y, (mk, col) in marks.items():
            s = tab[tab["response"] == y]
            ax.scatter(np.arange(len(names)) + (0.12 if y == "desir" else -0.12), s[f"{c}_extended"].clip(-2, 2),
                       marker=mk, color=col, s=42, label=y, zorder=3)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
        ax.set_title(f"{c}: {FACTOR_BY_CODE[c].label}", fontsize=9)
        ax.set_ylim(-2.15, 2.15)
    axes[0].set_ylabel("optimal coded level (screened box = shaded, ±1)")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Where is the optimum?  It depends on which factors carry the curvature — "
                 "points at ±2 are the search limit (no interior optimum)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "optimum_sensitivity.png")
    return tab


# ─────────────────────────────────────────────────────────────────────────────
# 8  Cost of the next phase, proposed CCD
# ─────────────────────────────────────────────────────────────────────────────

def part8_cost_and_design(df: pd.DataFrame) -> pd.DataFrame:
    """Cost of the candidate next phases.  Seconds per run are taken from the screening runs on the informative
    instances: configurations with F at its cheap level (the CCD below fixes F there), or all of them for a design
    that varies F (the foldover)."""
    inf = df[df["informative"]]
    sec = {"F fixed low": inf.loc[inf["F"] < 0, "wall_s"].mean(), "F varied": inf["wall_s"].mean()}
    n_inf, n_all = inf["instance"].nunique(), df["instance"].nunique()
    sec_all = {"F fixed low": df.loc[df["F"] < 0, "wall_s"].mean(), "F varied": df["wall_s"].mean()}
    rows = []
    for name, k_cfg, wall, note in (
            ("single-factor foldover of the 2^(7-3) (16 configs)", 16, "F varied", "clears the 2FIs of one factor"),
            ("face-centred CCD on A,B,C,D; E,F,G fixed (16 + 8 + 6)", 30, "F fixed low", ""),
            ("face-centred CCD on A,B,C,D,E; F,G fixed (16 + 10 + 6)", 32, "F fixed low", "recommended"),
            ("F and G check at the CCD optimum (3 configs)", 3, "F varied", "after the CCD")):
        for seeds in (3, 2):
            rows.append(dict(design=name, configs=k_cfg, seeds=seeds, note=note, seconds_per_run=sec[wall],
                             runs_20_instances=k_cfg * n_inf * seeds,
                             hours_20_instances=k_cfg * n_inf * seeds * sec[wall] / 3600,
                             runs_48_instances=k_cfg * n_all * seeds,
                             hours_48_instances=k_cfg * n_all * seeds * sec_all[wall] / 3600))
    tab = pd.DataFrame(rows)
    tab.to_csv(TAB / "next_phase_cost.csv", index=False)
    # proposed CCD: A, B, C, E on the screened ranges, window range moved to 8 … 24 (centre 16), F low, G high
    fs = list("ABCDE")
    ccd = Dsg.ccd(fs, 6, {"F": -1.0, "G": 1.0})
    nat = pd.DataFrame({c: [FACTOR_BY_CODE[c].natural(v) for v in ccd[c]] for c in "ABCEFG"})
    nat["D"] = [round(16 + 8 * v) for v in ccd["D"]]
    nat = nat[LETTERS]
    nat.columns = [f"{c}_natural" for c in nat.columns]
    pd.concat([ccd, nat], axis=1).to_csv(TAB / "ccd_proposed_design.csv", index=False)
    return tab


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    PLT.mkdir(parents=True, exist_ok=True)
    df = load()
    inf = df[df["informative"]]
    print(f"share of informative runs with time-to-best beyond 90% of the budget: "
          f"{(inf['ttb_frac'] > 0.9).mean():.3f}; beyond 99%: {(inf['ttb_frac'] > 0.99).mean():.3f}")
    print(f"{len(df)} runs; informative instances {inf['instance'].nunique()} of {df['instance'].nunique()} "
          f"({len(inf)} runs); SA never improved on its start in {(df['ttb_iter'] == 0).sum()} runs "
          f"({(df.loc[df['ttb_iter'] == 0, 'informative']).sum()} of them on informative instances)")
    e = part1_effects(df)
    e24 = part2_full_factorial(df)
    plot_interactions_2to4(df, e24)
    part3_consistency(df)
    part4_noise_interactions(df)
    part5_curvature_and_configs(df)
    models = part6_surfaces(df)
    part7_sensitivity(models)
    part8_cost_and_design(df)
    print(f"tables: {TAB}\nplots:  {PLT}")


if __name__ == "__main__":
    main()
