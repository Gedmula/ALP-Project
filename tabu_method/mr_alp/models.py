"""
models.py — MR-ALP Solver: Data Structures and Pre-Tuned Parameter Banks
=========================================================================
§2  Instance, HeuristicParams, MRSAParams dataclasses
§3  RBI_PARAM_BANK, SA_PARAM_BANK (pre-tuned via Optuna TPE)

Design notes
------------
* Instance.__post_init__ pre-computes derived statistics used throughout
  the solver (W_bar, s_bar, h_bar, Pen_bar, T_span, p_arr) and optionally
  uploads separation / target-time / penalty arrays to CUDA device tensors
  for GPU-accelerated TC computation.
* GPU tensors are excluded from __getstate__ / __setstate__ so that Instance
  objects can be sent to worker processes via pickle; they are re-created on
  the receiving side.
"""
from __future__ import annotations

import platform
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from mr_alp.config import (
    DEFAULT_ETA, DEFAULT_MU_TC, DEFAULT_MU_LATE, DEFAULT_MU_COUNT,
    DEFAULT_MU_SEP, USE_GPU, GPU_MIN_N,
)

# ── Optional accelerator imports ──────────────────────────────────────────
try:
    import torch as _torch
    _GPU_AVAIL   = _torch.cuda.is_available()
    _CUDA_DEVICE = _torch.device("cuda") if _GPU_AVAIL else None
except ImportError:
    _torch = None; _GPU_AVAIL = False; _CUDA_DEVICE = None


# ═══════════════════════════════════════════════════════════════════════════
#   §2  DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Instance:
    """
    Parsed and pre-processed ALP instance (0-indexed).

    Separation matrix s[i,j] is the required gap between predecessor i and
    successor j on the same runway.  Diagonal entries are zeroed after parsing.

    OR Library separation matrices violate the triangle inequality; all
    pairwise constraints must therefore be enforced explicitly in the LP and
    feasibility checks — not just consecutive-predecessor pairs.

    Derived attributes (computed in __post_init__)
    -----------------------------------------------
    W_bar   : mean time-window width  E[d_j − r_j].
    s_bar   : mean positive off-diagonal separation.
    h_bar   : mean tardiness penalty  E[h_j].
    Pen_bar : E[max(g_j, h_j)] × W_bar — runway-balance cost scale.
    T_span  : total horizon  max(d) − min(r).
    p_arr   : max(g_j, h_j) per aircraft (combined penalty rate).

    GPU tensors (_s_gpu, _delta_gpu, _p_arr_gpu) are created when
    USE_GPU=True, a CUDA device is available, and n ≥ GPU_MIN_N.
    """
    name:  str
    n:     int
    r:     np.ndarray
    delta: np.ndarray
    d:     np.ndarray
    g:     np.ndarray
    h:     np.ndarray
    s:     np.ndarray

    # Derived — set by __post_init__
    W_bar:      float      = field(init=False)
    s_bar:      float      = field(init=False)
    h_bar:      float      = field(init=False)
    Pen_bar:    float      = field(init=False)
    T_span:     float      = field(init=False)
    eps:        float      = field(init=False, default=1e-9)
    p_arr:      np.ndarray = field(init=False)
    _s_gpu:     Any        = field(init=False, default=None, repr=False)
    _delta_gpu: Any        = field(init=False, default=None, repr=False)
    _p_arr_gpu: Any        = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.W_bar  = float(np.mean(self.d - self.r))
        off  = self.s[~np.eye(self.n, dtype=bool)]
        pos  = off[off > 0]
        self.s_bar  = float(pos.mean()) if pos.size else 1.0
        self.h_bar  = float(np.mean(self.h))
        self.Pen_bar = float(np.mean(np.maximum(self.g, self.h)) * self.W_bar)
        self.T_span  = float(np.max(self.d) - np.min(self.r))
        self.eps     = 1e-9
        self.p_arr   = np.maximum(self.g, self.h)
        self._upload_gpu()

    def _upload_gpu(self) -> None:
        if USE_GPU and _GPU_AVAIL and _torch is not None and self.n >= GPU_MIN_N:
            _torch.backends.cuda.matmul.allow_tf32 = True
            kw = dict(dtype=_torch.float64, device=_CUDA_DEVICE)
            self._s_gpu     = _torch.as_tensor(self.s,     **kw)
            self._delta_gpu = _torch.as_tensor(self.delta, **kw)
            self._p_arr_gpu = _torch.as_tensor(self.p_arr, **kw)

    def __getstate__(self) -> dict:
        st = self.__dict__.copy()
        st['_s_gpu'] = st['_delta_gpu'] = st['_p_arr_gpu'] = None
        return st

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._upload_gpu()


@dataclass
class HeuristicParams:
    """
    Tunable scalar weights for the TC-RBI insertion cost function.

    eta      : screening blend weight (CR vs urgency).
    mu_tc    : weight on incremental target-time conflict ΔTC.
    mu_late  : weight on incremental tardiness lower bound ΔLate.
    mu_count : weight on runway-balance deviation Δcount.
    mu_sep   : weight on incremental separation burden ΔSep.
    """
    eta:      float = DEFAULT_ETA
    mu_tc:    float = DEFAULT_MU_TC
    mu_late:  float = DEFAULT_MU_LATE
    mu_count: float = DEFAULT_MU_COUNT
    mu_sep:   float = DEFAULT_MU_SEP

    def __str__(self) -> str:
        return (f"η={self.eta:.3f} μ_TC={self.mu_tc:.3f} "
                f"μ_late={self.mu_late:.3f} μ_count={self.mu_count:.3f} "
                f"μ_sep={self.mu_sep:.3f}")


@dataclass
class MRSAParams:
    """
    Simulated annealing control parameters for the multi-runway SA refinement.

    Tunable via Optuna TPE (optimize_sa_params in tuning module).
    chi0         : target initial acceptance probability for worsening moves.
    M_stag_frac  : stagnation threshold as fraction of N_iter.
    beta         : reheat multiplier.
    lp_gamma     : LP trigger sensitivity γ.
    chi_target   : reactive cooling target acceptance rate χ*.
    T_min_frac   : minimum temperature as fraction of initial temperature.
    B_max        : maximum block size for N4/X4 operators.
    B_stag       : block cap when stagnating.
    n_cal        : candidate moves sampled for T₀ calibration.
    alpha_step   : step size for reactive cooling alpha adjustment.
    alpha_lo     : lower bound for reactive cooling alpha.
    alpha_hi     : upper bound for reactive cooling alpha.
    max_reheats  : maximum reheats before forced termination.
    t_reheat     : reheat multiplier for temperature.
    lp_repair_interval  : iterations between LP-guided repairs (0 = disabled).
    near_zero_threshold : LP objective below this triggers TC-conflict repair.
    ejection_chain_depth: maximum ejection-chain depth (capped at 1 for m < 3).
    lambda_binding      : binding-constraint weight for LP impact scoring.
    eps_tight           : feasibility tolerance for LP-guided repair.
    max_ils_restarts    : ILS warm restarts from best_lp after MAX_REHEATS exhausted.
    """
    chi0:                float = 0.80
    M_stag_frac:         float = 0.15
    beta:                float = 1.50
    lp_gamma:            float = 0.05
    chi_target:          float = 0.20
    T_min_frac:          float = 0.01
    B_max:               int   = 3
    B_stag:              int   = 5
    n_cal:               int   = 200
    alpha_step:          float = 0.005
    alpha_lo:            float = 0.80
    alpha_hi:            float = 0.999
    max_reheats:         int   = 3
    t_reheat:            float = 2.0
    lp_repair_interval:  int   = 100
    near_zero_threshold: float = 200.0
    ejection_chain_depth: int  = 2
    lambda_binding:      float = 0.5
    eps_tight:           float = 1e-4
    max_ils_restarts:    int   = 2

    def __str__(self) -> str:
        return (f"χ₀={self.chi0:.3f}  M_stag={self.M_stag_frac:.3f}  "
                f"β={self.beta:.3f}  γ={self.lp_gamma:.4f}  χ*={self.chi_target:.3f}")


# ═══════════════════════════════════════════════════════════════════════════
#   §3  PRE-TUNED PARAMETER BANKS
# ═══════════════════════════════════════════════════════════════════════════
# Generated by Optuna TPE (see tuning.py).  When (inst.name, m) is present,
# these values are used directly; otherwise Optuna runs (if RUN_RBI_OPTUNA)
# or the dataclass defaults are used.

def _P(eta, mu_tc, mu_late, mu_count, mu_sep) -> HeuristicParams:
    return HeuristicParams(eta=eta, mu_tc=mu_tc, mu_late=mu_late,
                           mu_count=mu_count, mu_sep=mu_sep)

_DEFAULT_RBI = HeuristicParams()

RBI_PARAM_BANK: Dict[Tuple[str, int], HeuristicParams] = {
    ("airland1",  2): _P(0.571,2.324,1.594,1.422,0.376),
    ("airland1",  3): _P(0.445,4.778,0.060,0.730,0.092),
    ("airland2",  2): _P(0.724,0.724,0.681,1.042,0.188),
    ("airland2",  3): _P(0.268,4.604,1.425,1.065,0.189),
    ("airland3",  2): _P(0.684,3.613,1.552,1.698,0.334),
    ("airland3",  3): _P(0.709,4.770,0.715,2.866,0.329),
    ("airland4",  2): _P(0.511,3.612,0.246,1.818,0.357),
    ("airland4",  3): _P(0.408,1.272,0.302,2.535,0.489),
    ("airland4",  4): _P(0.443,2.683,0.492,0.557,0.186),
    ("airland5",  2): _P(0.607,0.251,0.606,0.585,0.258),
    ("airland5",  3): _P(0.225,3.786,1.156,0.648,0.193),
    ("airland5",  4): _P(0.794,2.427,0.268,0.468,0.207),
    ("airland6",  2): _P(0.524,0.676,1.895,2.555,0.262),
    ("airland6",  3): _P(0.488,4.551,1.441,1.892,0.386),
    ("airland7",  2): _P(0.229,0.441,1.142,0.715,0.370),
    ("airland8",  2): _P(0.732,4.111,1.358,2.423,0.334),
    ("airland8",  3): _P(0.490,3.755,0.210,1.108,0.256),
    ("airland9",  2): _P(0.653,3.170,1.700,0.822,0.219),
    ("airland9",  3): _P(0.372,0.568,1.416,1.895,0.001),
    ("airland9",  4): _P(0.305,4.302,1.976,2.914,0.430),
    ("airland10", 2): _P(0.424,4.730,1.509,2.373,0.044),
    ("airland10", 3): _P(0.773,2.732,0.876,0.405,0.083),
    ("airland10", 4): _P(0.794,1.716,0.667,1.703,0.433),
    ("airland10", 5): _P(0.564,3.269,0.365,2.447,0.162),
    ("airland11", 2): _P(0.530,2.408,1.362,1.895,0.002),
    ("airland11", 3): _P(0.733,2.754,1.607,0.305,0.241),
    ("airland11", 4): _P(0.314,4.472,1.543,2.700,0.146),
    ("airland11", 5): _P(0.404,0.584,1.830,1.315,0.020),
    ("airland12", 2): _P(0.536,2.500,1.882,2.811,0.186),
    ("airland12", 3): _P(0.415,4.624,0.243,2.590,0.284),
    ("airland12", 4): _P(0.254,2.792,1.160,2.117,0.127),
    ("airland12", 5): _P(0.235,1.592,1.206,2.837,0.105),
    ("airland13", 2): _P(0.486,2.788,1.551,0.862,0.214),
    ("airland13", 3): _P(0.349,3.411,0.029,0.810,0.245),
    ("airland13", 4): _P(0.339,2.534,1.394,2.935,0.435),
    ("airland13", 5): _P(0.701,1.596,0.492,0.772,0.339),
}

def _SA(chi0, M_stag_frac, beta, lp_gamma, chi_target) -> MRSAParams:
    return MRSAParams(chi0=chi0, M_stag_frac=M_stag_frac, beta=beta,
                      lp_gamma=lp_gamma, chi_target=chi_target)

# SA_PARAM_BANK is initially empty; populated by Optuna runs or user additions.
SA_PARAM_BANK: Dict[Tuple[str, int], MRSAParams] = {}