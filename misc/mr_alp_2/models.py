"""
models.py — Core data structures.

Instance stores all aircraft parameters plus Phase 0 preprocessing
constants (§6) computed once at construction time.  HybridParams and
SingleSAParams carry all tunable algorithm controls.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from misc.mr_alp_2.config import EPS, MP_WEIGHTS


@dataclass
class Instance:
    """
    Preprocessed ALP instance with Phase 0 constants (§6).

    Fields set at construction time
    ────────────────────────────────
    n      : number of aircraft.
    m      : number of runways.
    r      : (n,) earliest permissible landing times.
    d      : (n,) latest permissible landing times (hard deadlines).
    delta  : (n,) target landing times δ_j.
    g      : (n,) earliness penalty rates.
    h      : (n,) tardiness penalty rates.
    s      : (n, n) wake-separation matrix; s[i,j] = required gap if i lands
             before j on the same runway.  Diagonal entries are 0.
    name   : human-readable identifier.

    Phase 0 constants (§6) — computed in __post_init__
    ───────────────────────────────────────────────────
    W_bar  : mean time-window width (1/n) Σ (d_j − r_j).
    s_bar  : mean off-diagonal separation (1/(n(n−1))) Σ_{i≠j} s[i,j].
    h_bar  : mean tardiness penalty rate (1/n) Σ h_j.
    Pen_bar: typical maximum per-aircraft penalty (1/n) Σ max(g_j,h_j)·W̄.
    T_span : planning horizon max(d) − min(r).
    p_arr  : (n,) average penalty rate (g_j + h_j) / 2.
    AF     : (n,) adjusted flexibility W_j − s̄_j (§9.1).
    CR     : (n,) criticality (g_j + h_j) / max(AF_j, ε) (§9.2).
    """
    n:     int
    m:     int
    r:     np.ndarray
    d:     np.ndarray
    delta: np.ndarray
    g:     np.ndarray
    h:     np.ndarray
    s:     np.ndarray
    name:  str = ""

    # Phase 0 scalars
    W_bar:   float       = field(init=False, repr=False)
    s_bar:   float       = field(init=False, repr=False)
    h_bar:   float       = field(init=False, repr=False)
    Pen_bar: float       = field(init=False, repr=False)
    T_span:  float       = field(init=False, repr=False)

    # Phase 0 arrays
    p_arr:   np.ndarray  = field(init=False, repr=False, default=None)
    AF:      np.ndarray  = field(init=False, repr=False, default=None)
    CR:      np.ndarray  = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        n = self.n
        off = ~np.eye(n, dtype=bool)

        # §6 scalar constants
        self.W_bar   = float(np.mean(self.d - self.r))
        sep_vals     = self.s[off]
        self.s_bar   = float(sep_vals.mean()) if sep_vals.size > 0 else 1.0
        self.h_bar   = float(np.mean(self.h))
        self.Pen_bar = float(np.mean(np.maximum(self.g, self.h)) * self.W_bar)
        self.T_span  = float(np.max(self.d) - np.min(self.r))

        # Per-aircraft derived arrays
        self.p_arr = (self.g + self.h) / 2.0

        # §9.1  Adjusted flexibility AF_j = W_j − s̄_j
        # s̄_jk = s[j,k] (identical-runway assumption).
        # s̄_j  = (1 / (2(n−1))) Σ_{k≠j} (s[j,k] + s[k,j])
        sym    = (self.s + self.s.T)                   # s[j,k] + s[k,j]
        row_sum = np.array([sym[j, off[j]].sum() for j in range(n)])
        s_bar_j = row_sum / max(2.0 * (n - 1), 1.0)
        self.AF = (self.d - self.r) - s_bar_j

        # §9.2  Criticality
        self.CR = (self.g + self.h) / np.maximum(self.AF, EPS)


@dataclass
class HybridParams:
    """
    Parameters for construction and inter-runway improvement layers.

    lambda1     : separation-burden weight in insertion cost (§8).
    lambda2     : runway-balance weight in insertion cost (§8).
    eta         : regret–criticality mix for construction priority (§9.4).
    K           : top-K aircraft for inter-runway improvement (§13).
    B_max       : maximum normal block-transfer size (§14.3).
    alpha_outer : outer SA cooling rate (§15).
    w           : move-pressure weight vector (w1..w6) (§13.6).
    """
    lambda1:     float         = 0.8
    lambda2:     float         = 0.1
    eta:         float         = 0.5
    K:           int           = 20
    B_max:       int           = 3
    alpha_outer: float         = 0.95
    w:           Tuple[float, ...] = MP_WEIGHTS


@dataclass
class SingleSAParams:
    """
    Parameters for the single-runway SA engine (§11).

    alpha   : nominal cooling rate (adapted reactively).
    N_iter  : neighbourhood moves per temperature level.
    T_min   : minimum temperature (stopping criterion).
    I_max   : maximum outer (temperature) iterations.
    M_stag  : stagnation limit — outer iterations without improvement
              before reheating is triggered.
    chi0    : target initial acceptance probability for T₀ calibration.
    """
    alpha:  float = 0.97
    N_iter: int   = 200
    T_min:  float = 1e-4
    I_max:  int   = 600
    M_stag: int   = 60
    chi0:   float = 0.50


@dataclass
class MRSAParams:
    """
    Parameters for the multi-runway dual-track SA (§14–§17).

    alpha              : proxy SA cooling rate.
    N_iter             : total proxy SA iterations.
    T_init_frac        : initial temperature = lp_init × fraction.
    M_stag             : absolute stagnation limit before reheating.
    beta_reheat        : temperature multiplier on reheat.
    B_max              : standard block-transfer cap.
    B_stag             : block cap under stagnation.
    R_pool             : candidate moves sampled per iteration.
    ejection_chain_depth : depth D for ejection chains in VND (§19/§24).
    chi0               : target initial acceptance probability.
    lp_interval_early  : LP call interval when f < 0.20.
    lp_interval_mid    : LP call interval when 0.20 ≤ f < 0.80.
    lp_interval_late   : LP call interval when f ≥ 0.80.
    """
    alpha:               float = 0.995
    N_iter:              int   = 5000
    T_init_frac:         float = 0.05
    M_stag:              int   = 400
    beta_reheat:         float = 1.5
    B_max:               int   = 3
    B_stag:              int   = 5
    R_pool:              int   = 5
    ejection_chain_depth: int  = 2
    chi0:                float = 0.30
    lp_interval_early:   int   = 20
    lp_interval_mid:     int   = 50
    lp_interval_late:    int   = 100
    

@dataclass
class HeuristicParams:
    """
    Parameters for TC-RBI / RAMP-RBI construction and the seed portfolio (§8, §25).

    mu_tc    : weight on incremental target-conflict cost ΔTC.
    mu_late  : weight on lower-bound tardiness change ΔLate.
    mu_count : weight on runway-balance count term Δcount.
    mu_sep   : weight on wake-separation burden Δsep (multiplied internally
               by h̄ to convert to cost units, §8.5).
    eta      : criticality–urgency mixing factor for RAMP-RBI screening (§10.1).
    """
    mu_tc:    float = 1.0
    mu_late:  float = 1.0
    mu_count: float = 0.5
    mu_sep:   float = 0.5
    eta:      float = 0.5