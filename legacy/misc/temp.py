"""
mr_alp.py — Multi-Runway Aircraft Landing Problem: TC-RBI + SA Refinement
===============================================================================
OVERVIEW
--------
Self-contained solver for the static Multi-Runway Aircraft Landing Problem
(MR-ALP).  The pipeline runs in two stages:

    Stage 1 — Seed portfolio construction
        Eleven construction heuristics (H1–H9 plus TC-RBI and GRASP-7) are
        evaluated in parallel.  Every seed is LP-screened via the Stage-2 LP;
        the best n_chains seeds (by LP objective, with a runway-Hamming
        diversity guard) are forwarded to Stage 2.

        Heuristics implemented:
          H1  FCFS    — arrival-order dispatching rule
          H2  EDD     — target-time round-robin
          H3  WEDD    — weighted EDD, freest-runway assignment
          H4  ATC     — apparent tardiness cost rule
          H5  ATCS    — ATC extended with separation penalty
          H6  CAF     — critical-aircraft-first insertion
          H7  MPDS    — most-penalised-displacement-first  (n ≤ MPDS_MAX_N)
          H8  WCC     — wake-compatibility chain construction
          H9a GRASP-3 — randomised greedy, RCL size 3
          H9b GRASP-7 — randomised greedy, RCL size 7
              TC-RBI  — target-conflict regret-based insertion

    Stage 2 — Exact LP timing optimisation
        Given the fixed sequences, a sparse LP with full pairwise separation
        constraints is solved via HiGHS (SciPy interface) to obtain optimal
        landing times.  This is the metric reported against benchmark optima.

    SA refinement — Parallel multi-start SA
        K controlled chains refine the selected seeds using phase-adaptive
        neighbourhood operators (N1/N2/N3b/N4 within-runway;
        X1–X4/X7/XE cross-runway; ejection chains).  Features include
        reactive cooling, LP-guided repair, elite pool, path relinking,
        and LP-VND polish.

OUTPUT FILES  (written to OUTPUT_DIR/)
---------------------------------------
    summary.csv          — one row per (instance, m): objectives, gaps, timing,
                           best seed label, number of seeds evaluated.
    schedules.csv        — best final sequences (long format).
    alternatives.csv     — elite pool alternative schedules.
    verification.txt     — feasibility audit + LP timeline + seed portfolio log.
    run_metadata.json    — run config, SA parameters, timing, seed portfolio.

    plots/
      gap/
        gap_summary.png
      convergence/
        convergence_{inst}_{m}.png
      lp_timeline/
        lp_timeline_{inst}_{m}.png
      time_to_best/
        time_to_best.png
      elite_pool/
        elite_pool_{inst}_{m}.png
      gantt/
        gantt_{inst}_{m}.png
      seeds/
        seeds_{inst}_{m}.png  — LP comparison of all evaluated seeds.

CORRECTNESS NOTE
----------------
OR Library separation matrices violate the triangle inequality.  The LP and all
feasibility checks enforce ALL ordered-pair constraints (n(n−1)/2 per runway),
not just consecutive pairs.

REFERENCES
----------
Beasley et al. (2000). Transportation Science 34(2): 180–197.
Clarke & Wright (1964). Operations Research 12(4): 568–581.
Ernst, Krishnamoorthy & Storer (1999). Networks 34: 229–241.
Feo & Resende (1995). Journal of Global Optimization 6(2): 109–133.
Lee, Bhaskaran & Pinedo (1997). IIE Transactions 29(12): 1057–1067.
Pinedo (2016). Scheduling: Theory, Algorithms, and Systems, 5th ed. Springer.
Pinol & Beasley (2006). European Journal of Operational Research 171(2): 439–462.
Vepsäläinen & Morton (1987). Management Science 33(11): 1489–1500.
Zhang et al. (2020). Transactions of Nanjing Univ. of Aeronautics and Astronautics.
"""
from __future__ import annotations

import csv, io, contextlib, json, math, platform, random, time, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

# ── Optional accelerator imports ─────────────────────────────────────────────
try:
    import numba as nb
    _NUMBA = True
except ImportError:
    nb = None; _NUMBA = False

try:
    import torch as _torch
    _GPU_AVAIL   = _torch.cuda.is_available()
    _CUDA_DEVICE = _torch.device("cuda") if _GPU_AVAIL else None
except ImportError:
    _torch = None; _GPU_AVAIL = False; _CUDA_DEVICE = None

try:
    import optuna as _optuna
    _optuna.logging.set_verbosity(_optuna.logging.WARNING)
    _OPTUNA = True
except ImportError:
    _optuna = None; _OPTUNA = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    _MPL = True
except ImportError:
    plt = mticker = None; _MPL = False

import multiprocessing as _mp
_MP_CTX = _mp.get_context(
    "spawn" if (platform.system() == "Windows" or _GPU_AVAIL) else "fork"
)


# ═════════════════════════════════════════════════════════════════════════════
#   §0  CONFIGURE HERE
# ═════════════════════════════════════════════════════════════════════════════
BATCH_MODE    = True
INSTANCE_PATH = "data/airland1.txt"
FOLDER        = "data/"

INSTANCE_RUNWAYS: Dict[str, List[int]] = {
    "airland1":  [2, 3],  "airland2":  [2, 3],  "airland3":  [2, 3],
    "airland4":  [2, 3, 4],  "airland5":  [2, 3, 4],
    "airland6":  [2, 3],  "airland7":  [2],
    "airland8":  [2, 3],  "airland9":  [2, 3, 4],
    "airland10": [2, 3, 4, 5],  "airland11": [2, 3, 4, 5],
    "airland12": [2, 3, 4, 5],  "airland13": [2, 3, 4, 5],
}

USE_ALL_SEEDS = True
N_WORKERS   = 3 if USE_ALL_SEEDS else 7
N_CHAINS    = 4
T_LIMIT     = 300.0
MAX_T_LIMIT = 1200.0

N_OPTUNA_WORKERS  = 4
RUN_RBI_OPTUNA    = True
N_RBI_TRIALS_BASE = 30
RBI_OPTUNA_SEED   = 42

RUN_SA_OPTUNA    = False
SA_N_TRIALS_BASE = 20
SA_OPTUNA_SEED   = 123
SA_N_OPTUNA_JOBS = 4
 
ELITE_POOL_MAX = 20
ELITE_MIN_DIV  = 5

USE_GPU   = True
GPU_MIN_N = 200

OUTPUT_DIR   = Path("MR_results")
SAVE_RESULTS = True
SAVE_PLOTS   = True

# ── TC-RBI defaults ───────────────────────────────────────────────────────
DEFAULT_ETA      = 0.50
DEFAULT_MU_TC    = 1.00
DEFAULT_MU_LATE  = 0.25
DEFAULT_MU_COUNT = 0.75
DEFAULT_MU_SEP   = 0.05

# ── Seed heuristic parameters ─────────────────────────────────────────────
# ATC look-ahead scaling K: large K → WSPT; small K → MS rule.
# Default 2.5 works well on OR Library instances (Pinedo 2016, §14.2).
ATC_K  = 2.5

# ATCS scaling parameters for urgency (K1) and setup (K2).
ATCS_K1 = 2.0
ATCS_K2 = 2.0

# GRASP Restricted Candidate List sizes for the two GRASP variants.
GRASP_K_VALUES: Tuple[int, int] = (3, 7)

# Maximum n for MPDS seed (O(n³/m) cost; skip for larger instances).
MPDS_MAX_N = 150
# ═════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
#   §1  BENCHMARK REFERENCE OPTIMA
# ─────────────────────────────────────────────────────────────────────────────
KNOWN_OPTIMA: Dict[str, Dict[int, float]] = {
    "airland1":  {1: 700.00,    2: 90.00,    3: 0.00},
    "airland2":  {1: 1480.00,   2: 210.00,   3: 0.00},
    "airland3":  {1: 820.00,    2: 60.00,    3: 0.00},
    "airland4":  {1: 2520.00,   2: 640.00,   3: 130.00,  4: 0.00},
    "airland5":  {1: 3100.00,   2: 650.00,   3: 170.00,  4: 0.00},
    "airland6":  {1: 24442.00,  2: 554.00,   3: 0.00},
    "airland7":  {1: 1550.00,   2: 0.00},
    "airland8":  {1: 1950.00,   2: 135.00,   3: 0.00},
    "airland9":  {1: 5611.70,   2: 444.10,   3: 75.75,   4: 0.00},
    "airland10": {1: 12821.12,  2: 1143.70,  3: 205.21,  4: 34.22,  5: 0.00},
    "airland11": {1: 12654.18,  2: 1330.91,  3: 253.07,  4: 54.53,  5: 0.00},
    "airland12": {1: 16629.10,  2: 1695.62,  3: 221.97,  4: 2.44,   5: 0.00},
    "airland13": {1: 39516.34,  2: 3943.85,  3: 673.85,  4: 89.95,  5: 0.00},
}


# ═════════════════════════════════════════════════════════════════════════════
#   §2  DATA STRUCTURES
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Instance:
    """
    Parsed and pre-processed ALP instance.

    All per-aircraft arrays are 0-indexed.  The separation matrix s has shape
    (n, n): s[i, j] is the required gap between aircraft i (predecessor) and
    j (successor) on the same runway.  The diagonal is zeroed after parsing.

    Note on OR Library separation matrices
    ----------------------------------------
    These matrices do NOT satisfy the triangle inequality.  All ordered-pair
    separation constraints must be enforced in the LP and feasibility checks.

    Derived attributes
    ------------------
    W_bar   : mean time-window width E[d_j − r_j].
    s_bar   : mean positive off-diagonal separation.
    h_bar   : mean tardiness penalty rate E[h_j].
    Pen_bar : E[max(g_j, h_j)] × W_bar — runway-balance cost scale.
    T_span  : total time horizon max(d) − min(r).
    p_arr   : max(g_j, h_j) per aircraft.
    """
    name:  str;  n: int
    r:     np.ndarray; delta: np.ndarray; d: np.ndarray
    g:     np.ndarray; h:     np.ndarray; s: np.ndarray

    W_bar:   float     = field(init=False)
    s_bar:   float     = field(init=False)
    h_bar:   float     = field(init=False)
    Pen_bar: float     = field(init=False)
    T_span:  float     = field(init=False)
    eps:     float     = field(init=False, default=1e-9)
    p_arr:   np.ndarray = field(init=False)
    _s_gpu:  object    = field(init=False, default=None, repr=False)
    _delta_gpu: object = field(init=False, default=None, repr=False)
    _p_arr_gpu: object = field(init=False, default=None, repr=False)

    def __post_init__(self):
        self.W_bar   = float(np.mean(self.d - self.r))
        off           = self.s[~np.eye(self.n, dtype=bool)]
        pos           = off[off > 0]
        self.s_bar   = float(pos.mean()) if pos.size else 1.0
        self.h_bar   = float(np.mean(self.h))
        self.Pen_bar = float(np.mean(np.maximum(self.g, self.h)) * self.W_bar)
        self.T_span  = float(np.max(self.d) - np.min(self.r))
        self.eps     = 1e-9
        self.p_arr   = np.maximum(self.g, self.h)
        if USE_GPU and _GPU_AVAIL and self.n >= GPU_MIN_N:
            _torch.backends.cuda.matmul.allow_tf32 = True
            kw = dict(dtype=_torch.float64, device=_CUDA_DEVICE)
            self._s_gpu     = _torch.as_tensor(self.s,     **kw)
            self._delta_gpu = _torch.as_tensor(self.delta, **kw)
            self._p_arr_gpu = _torch.as_tensor(self.p_arr, **kw)

    def __getstate__(self):
        st = self.__dict__.copy()
        st['_s_gpu'] = st['_delta_gpu'] = st['_p_arr_gpu'] = None
        return st

    def __setstate__(self, state):
        self.__dict__.update(state)
        if USE_GPU and _GPU_AVAIL and self.n >= GPU_MIN_N:
            kw = dict(dtype=_torch.float64, device=_CUDA_DEVICE)
            self._s_gpu     = _torch.as_tensor(self.s,     **kw)
            self._delta_gpu = _torch.as_tensor(self.delta, **kw)
            self._p_arr_gpu = _torch.as_tensor(self.p_arr, **kw)


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

    def __str__(self):
        return (f"η={self.eta:.3f} μ_TC={self.mu_tc:.3f} "
                f"μ_late={self.mu_late:.3f} μ_count={self.mu_count:.3f} "
                f"μ_sep={self.mu_sep:.3f}")


@dataclass
class MRSAParams:
    """
    Simulated annealing control parameters for the multi-runway SA refinement.

    Tunable via Optuna TPE (optimize_sa_params).
    chi0         : target initial acceptance probability for worsening moves.
    M_stag_frac  : stagnation threshold as fraction of N_iter.
    beta         : reheat multiplier label.
    lp_gamma     : LP trigger sensitivity γ.
    chi_target   : reactive cooling target acceptance rate χ*.
    T_min_frac   : minimum temperature as fraction of initial temperature.
    B_max        : maximum number of consecutive reheats.
    B_stag       : number of reheats without improvement to trigger elite pool reset.
    n_cal        : number of candidate moves to sample for LP evaluation.
    alpha_step    : step size for reactive cooling alpha adjustment.
    alpha_lo      : lower bound for reactive cooling alpha.
    alpha_hi      : upper bound for reactive cooling alpha.
    max_reheats   : maximum number of reheats before forced termination.
    t_reheat      : time limit per reheat in seconds.
    lp_repair_interval      : number of iterations between LP-guided repairs.
    near_zero_threshold     : threshold for treating LP objective improvements as zero.
    ejection_chain_depth    : maximum depth for ejection chains.
    lambda_binding          : scaling factor for binding constraint identification in LP repair.
    eps_tight               : feasibility tolerance for LP-guided repair.
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

    def __str__(self):
        return (f"χ₀={self.chi0:.3f}  M_stag={self.M_stag_frac:.3f}  "
                f"β={self.beta:.3f}  γ={self.lp_gamma:.4f}  χ*={self.chi_target:.3f}")


# ═════════════════════════════════════════════════════════════════════════════
#   §3  PRE-TUNED PARAMETER BANKS
# ═════════════════════════════════════════════════════════════════════════════

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

SA_PARAM_BANK: Dict[Tuple[str, int], MRSAParams] = {}


# ═════════════════════════════════════════════════════════════════════════════
#   §4  NUMBA JIT KERNELS
# ═════════════════════════════════════════════════════════════════════════════

if _NUMBA:
    @nb.njit(cache=True)
    def _insert_times_kernel(j, p, seq, C_prev, r, s, d):
        """
        Compute surrogate times after inserting aircraft j at position p.
        Returns (C_n, feasible).  Positions 0..p-1 are copied from C_prev.
        Feasibility is checked only for positions p..L_n.
        """
        L = len(seq); L_n = L + 1; C_n = np.empty(L_n, dtype=np.float64)
        for q in range(p): C_n[q] = C_prev[q]
        if p == 0: C_n[0] = r[j]
        else:
            val = C_n[p-1] + s[seq[p-1], j]
            C_n[p] = val if val > r[j] else r[j]
        for q in range(p+1, L_n):
            cur = seq[q-1]; prev = j if q == p+1 else seq[q-2]
            val = C_n[q-1] + s[prev, cur]
            C_n[q] = val if val > r[cur] else r[cur]
        for q in range(p, L_n):
            ac = j if q == p else seq[q-1]
            if C_n[q] > d[ac] + 1e-9: return C_n, False
        return C_n, True

    @nb.njit(cache=True)
    def _rwy_feasible_nb(seq, r, s, d):
        """Full pairwise feasibility check for one runway sequence."""
        L = len(seq)
        if L == 0: return True
        C = np.empty(L, dtype=np.float64); C[0] = r[seq[0]]
        if C[0] > d[seq[0]] + 1e-9: return False
        for q in range(1, L):
            C[q] = r[seq[q]]
            for h in range(q):
                lb = C[h] + s[seq[h], seq[q]]
                if lb > C[q]: C[q] = lb
            if C[q] > d[seq[q]] + 1e-9: return False
        return True


# ═════════════════════════════════════════════════════════════════════════════
#   §5  INSTANCE FILE PARSER  (OR Library format)
# ═════════════════════════════════════════════════════════════════════════════

def load_instance(filepath, name=None) -> Instance:
    """
    Parse an OR Library ALP instance file.

    Format: n and freeze_time on line 1; then per-aircraft rows:
    appearance_time r delta d g h s[i,0]..s[i,n-1].
    appearance_time and freeze_time are discarded.
    """
    path   = Path(filepath); name = name or path.stem.lower()
    tokens = path.read_text().split(); pos = 0

    def take_int():   nonlocal pos; v = int(tokens[pos]);   pos += 1; return v
    def take_float(): nonlocal pos; v = float(tokens[pos]); pos += 1; return v

    n = take_int(); _ = take_float()
    r = np.empty(n); delta = np.empty(n); d = np.empty(n)
    g = np.empty(n); h     = np.empty(n); s = np.empty((n, n))
    for i in range(n):
        _ = take_float(); r[i]=take_float(); delta[i]=take_float(); d[i]=take_float()
        g[i]=take_float(); h[i]=take_float()
        for j in range(n): s[i,j]=take_float()
    np.fill_diagonal(s, 0.0)
    bad=np.where(r>delta+1e-6)[0]
    if bad.size: raise ValueError(f"{name}: r>delta for aircraft {bad[:5]}")
    bad=np.where(delta>d+1e-6)[0]
    if bad.size: raise ValueError(f"{name}: delta>d for aircraft {bad[:5]}")
    if pos!=len(tokens): raise ValueError(f"{name}: token count mismatch")
    return Instance(name=name, n=n, r=r, delta=delta, d=d, g=g, h=h, s=s)


# ═════════════════════════════════════════════════════════════════════════════
#   §6  SURROGATE LANDING TIMES  (consecutive-predecessor approximation)
# ═════════════════════════════════════════════════════════════════════════════

def surrogate_times(seq: List[int], inst: Instance) -> List[float]:
    """
    Consecutive-predecessor surrogate: C[q] = max(r[seq[q]], C[q-1]+s[seq[q-1],seq[q]]).
    Not used for final objectives; used only to guide construction and SA moves.
    """
    if not seq: return []
    C = [0.0]*len(seq); C[0] = float(inst.r[seq[0]])
    for q in range(1,len(seq)):
        C[q] = max(float(inst.r[seq[q]]), C[q-1]+float(inst.s[seq[q-1],seq[q]]))
    return C


def surrogate_penalty(seq, C_hat, inst) -> float:
    """Surrogate penalty: Σ g_j·max(δ_j−C_j,0) + h_j·max(C_j−δ_j,0)."""
    if not seq: return 0.0
    sa=np.asarray(seq,dtype=np.intp); Ca=np.asarray(C_hat)
    return float((inst.g[sa]*np.maximum(inst.delta[sa]-Ca,0.)
                  + inst.h[sa]*np.maximum(Ca-inst.delta[sa],0.)).sum())


# ═════════════════════════════════════════════════════════════════════════════
#   §7  RUNWAY FEASIBILITY CHECK
# ═════════════════════════════════════════════════════════════════════════════

def _runway_feasible(seq: List[int], inst: Instance) -> bool:
    """Full pairwise feasibility check; dispatches to Numba JIT when available."""
    if not seq: return True
    if _NUMBA:
        return bool(_rwy_feasible_nb(np.asarray(seq,dtype=np.int32),inst.r,inst.s,inst.d))
    L=len(seq); C=np.empty(L); C[0]=inst.r[seq[0]]
    if C[0]>inst.d[seq[0]]+1e-9: return False
    for q in range(1,L):
        C[q]=inst.r[seq[q]]
        for h in range(q):
            lb=C[h]+inst.s[seq[h],seq[q]]
            if lb>C[q]: C[q]=lb
        if C[q]>inst.d[seq[q]]+1e-9: return False
    return True


# ═════════════════════════════════════════════════════════════════════════════
#   §8  TC-RBI PRIORITY MEASURES AND INSERTION COST COMPONENTS
# ═════════════════════════════════════════════════════════════════════════════

def compute_priorities(inst: Instance) -> Tuple[np.ndarray, np.ndarray]:
    """
    AF_j = (d_j−r_j) − mean_bilateral_separation_j
    CR_j = (g_j+h_j) / max(AF_j, ε)
    """
    s_sym=(inst.s+inst.s.T)/2.; np.fill_diagonal(s_sym,0.)
    s_b=s_sym.sum(axis=1)/max(inst.n-1,1)
    AF=(inst.d-inst.r)-s_b
    CR=(inst.g+inst.h)/np.maximum(AF,inst.eps)
    return AF, CR


def minmax_norm(arr: np.ndarray, eps: float=1e-9) -> np.ndarray:
    lo,hi=arr.min(),arr.max(); return (arr-lo)/(hi-lo+eps)


def _compute_insert_times(j, p, seq, C_hat_seq, inst):
    """Surrogate times after inserting j at position p; dispatches to Numba."""
    if _NUMBA:
        L=len(seq)
        sa=np.asarray(seq,dtype=np.int32)   if L else np.empty(0,dtype=np.int32)
        Ca=np.asarray(C_hat_seq,dtype=np.float64) if C_hat_seq else np.empty(0,dtype=np.float64)
        C_n,ok=_insert_times_kernel(j,p,sa,Ca,inst.r,inst.s,inst.d)
        return list(C_n),bool(ok)
    seq_n=seq[:p]+[j]+seq[p:]; L_n=len(seq_n); C_n=list(C_hat_seq[:p])
    if p==0: C_n.append(float(inst.r[j]))
    else:    C_n.append(max(float(inst.r[j]),C_n[p-1]+float(inst.s[seq_n[p-1],j])))
    for q in range(p+1,L_n):
        prev,cur=seq_n[q-1],seq_n[q]
        C_n.append(max(float(inst.r[cur]),C_n[q-1]+float(inst.s[prev,cur])))
    for q in range(p,L_n):
        if C_n[q]>inst.d[seq_n[q]]+1e-9: return C_n,False
    return C_n,True


def _is_feasible_anywhere(j, sequences, C_hats, inst) -> bool:
    m=len(sequences)
    for rho in range(m):
        _,ok=_compute_insert_times(j,len(sequences[rho]),sequences[rho],C_hats[rho],inst)
        if ok: return True
    for rho in range(m):
        for p in range(len(sequences[rho])):
            _,ok=_compute_insert_times(j,p,sequences[rho],C_hats[rho],inst)
            if ok: return True
    return False


def target_conflict_insert(j, p, seq, inst) -> float:
    """Incremental weighted target-time conflict from inserting j at position p."""
    pj=float(inst.p_arr[j]); cost=0.
    pred=seq[:p]
    if pred:
        pa=np.asarray(pred,dtype=np.intp)
        v=inst.s[pa,j]-(inst.delta[j]-inst.delta[pa])
        cost+=float((0.5*(inst.p_arr[pa]+pj)*np.maximum(v,0.)).sum())
    succ=seq[p:]
    if succ:
        sa=np.asarray(succ,dtype=np.intp)
        v=inst.s[j,sa]-(inst.delta[sa]-inst.delta[j])
        cost+=float((0.5*(pj+inst.p_arr[sa])*np.maximum(v,0.)).sum())
    return cost


def lower_bound_tardiness(seq, C_hat, inst) -> float:
    if not seq: return 0.
    sa=np.asarray(seq,dtype=np.intp); Ca=np.asarray(C_hat)
    return float((inst.h[sa]*np.maximum(Ca-inst.delta[sa],0.)).sum())


def count_balance_delta(rho, sequences, inst) -> float:
    m=len(sequences); n=inst.n; t=sum(len(s) for s in sequences); ol=len(sequences[rho])
    return ((ol+1-(t+1)/m)**2-(ol-t/m)**2)*float(inst.Pen_bar)/max((n/m)**2,1.)


def evaluate_insertion(j, rho, p, sequences, C_hats, B_bar, inst, params):
    """
    Composite insertion cost = μ_TC·ΔTC + μ_late·ΔLate + μ_count·Δcount + μ_sep·ΔSep.
    Returns (cost, C_n).  Returns (inf, []) if infeasible.
    """
    seq,C_hat_seq=sequences[rho],C_hats[rho]; L=len(seq)
    C_n,ok=_compute_insert_times(j,p,seq,C_hat_seq,inst)
    if not ok: return math.inf,[]
    seq_n=seq[:p]+[j]+seq[p:]
    dTC=target_conflict_insert(j,p,seq,inst)
    dLate=(lower_bound_tardiness(seq_n,C_n,inst)-lower_bound_tardiness(seq,C_hat_seq,inst))
    dCount=count_balance_delta(rho,sequences,inst)
    if L==0:        dSep_raw=0.
    elif p==0:      dSep_raw=float(inst.s[j,seq[0]])
    elif p==L:      dSep_raw=float(inst.s[seq[-1],j])
    else:
        a,b=seq[p-1],seq[p]
        dSep_raw=max(0.,float(inst.s[a,j])+float(inst.s[j,b])-float(inst.s[a,b]))
    cost=(params.mu_tc*dTC+params.mu_late*dLate
          +params.mu_count*dCount+params.mu_sep*inst.h_bar*dSep_raw)
    return cost,C_n


def _candidate_positions(j, rho, sequences, inst) -> List[int]:
    """All positions for n≤100; a centred window for larger instances."""
    seq,L=sequences[rho],len(sequences[rho])
    if inst.n<=100: return list(range(L+1))
    p0=next((p for p,u in enumerate(seq) if inst.delta[u]>=inst.delta[j]),L)
    return sorted(set(range(max(0,p0-2),min(L+1,p0+3)))|{0,L})


def _best_insertions(j, m, sequences, C_hats, B_bar, inst, params):
    """Best (runway,position) for j and regret (cost of second-best runway)."""
    per_rho=[]
    for rho in range(m):
        rc,rp,rC=math.inf,0,[]
        for p in _candidate_positions(j,rho,sequences,inst):
            c,Cn=evaluate_insertion(j,rho,p,sequences,C_hats,B_bar,inst,params)
            if c<rc: rc,rp,rC=c,p,Cn
        per_rho.append((rc,rp,rC))
    sr=sorted(range(m),key=lambda r:per_rho[r][0]); c1,p1,C1=per_rho[sr[0]]
    best1=(c1,sr[0],p1,C1)
    if m>1: c2=per_rho[sr[1]][0]
    else:
        all_c=sorted(evaluate_insertion(j,0,p,sequences,C_hats,B_bar,inst,params)[0]
                     for p in range(len(sequences[0])+1))
        c2=all_c[1] if len(all_c)>1 else math.inf
    return best1,c2


def min_violation_insert(j, sequences, inst):
    """Least-infeasible insertion for aircraft j (forced-set fallback)."""
    best_V,best_rho,best_p,best_C=math.inf,0,0,[]
    for rho,seq in enumerate(sequences):
        for p in range(len(seq)+1):
            seq_n=seq[:p]+[j]+seq[p:]; C_t=surrogate_times(seq_n,inst)
            V=sum(max(C_t[q]-inst.d[seq_n[q]],0.) for q in range(len(seq_n)))
            if V<best_V: best_V,best_rho,best_p,best_C=V,rho,p,C_t
    return best_rho,best_p,best_C


def inter_runway_repair(sequences, C_hats, inst, params, max_iterations=150):
    """Improve load balance by relocating high-TC aircraft to less-loaded runways."""
    m=len(sequences)
    if m==1: return sequences,C_hats
    sequences=[list(s) for s in sequences]; C_hats=[list(c) for c in C_hats]
    mean_load=inst.n/m
    for _ in range(max_iterations):
        rho_src=max(range(m),key=lambda r:len(sequences[r]))
        if len(sequences[rho_src])<=mean_load+1: break
        B_bar=sum(C_hats[r][-1] if C_hats[r] else 0. for r in range(m))/m
        best_tc,best_sp=-1.,-1
        for sp,j in enumerate(sequences[rho_src]):
            seq_no=sequences[rho_src][:sp]+sequences[rho_src][sp+1:]
            tc=target_conflict_insert(j,sp,seq_no,inst)
            if tc>best_tc: best_tc,best_sp=tc,sp
        if best_sp==-1: break
        j_move=sequences[rho_src][best_sp]
        best_c,best_rd,best_dp,best_Cn=math.inf,-1,-1,[]
        for rd in range(m):
            if rd==rho_src: continue
            for dp in range(len(sequences[rd])+1):
                c,Cn=evaluate_insertion(j_move,rd,dp,sequences,C_hats,B_bar,inst,params)
                if c<best_c: best_c,best_rd,best_dp,best_Cn=c,rd,dp,Cn
        if best_rd==-1 or best_c==math.inf: break
        sequences[rho_src].pop(best_sp)
        C_hats[rho_src]=surrogate_times(sequences[rho_src],inst)
        sequences[best_rd].insert(best_dp,j_move); C_hats[best_rd]=best_Cn
    return sequences,C_hats


# ═════════════════════════════════════════════════════════════════════════════
#   §9  TC-RBI CONSTRUCTION HEURISTIC
# ═════════════════════════════════════════════════════════════════════════════

def ramp_rbi(inst, m, params):
    """
    TC-RBI: Target-Conflict Regret-Based Insertion.

    Iteratively inserts aircraft by minimising a composite insertion cost
    (ΔTC + ΔLate + Δcount + ΔSep), breaking ties with regret (cost gap between
    best and second-best runway) and criticality ratio CR_j.  Aircraft with no
    feasible position are handled by minimum-violation insertion.

    Returns (sequences, C_hats).
    """
    n,eps=inst.n,inst.eps; _,CR=compute_priorities(inst)
    sequences=[[] for _ in range(m)]; C_hats=[[] for _ in range(m)]
    B=[0.]*m; B_bar=0.
    U=list(range(n)); F=set()

    def committed(rho): return C_hats[rho][-1] if C_hats[rho] else 0.
    def refresh_forced():
        for j in list(U):
            if j in F: continue
            if not _is_feasible_anywhere(j,sequences,C_hats,inst): F.add(j)
    def do_insert(j,rho,p,C_new):
        nonlocal B_bar
        sequences[rho].insert(p,j); C_hats[rho]=C_new
        B[rho]=committed(rho); B_bar=sum(B)/m; U.remove(j); F.discard(j)

    while U:
        while F&set(U):
            j_star=max([j for j in U if j in F],key=lambda j:CR[j])
            rho,p,C_new=min_violation_insert(j_star,sequences,inst)
            seq_new=sequences[rho][:p]+[j_star]+sequences[rho][p:]
            sequences[rho]=seq_new; C_hats[rho]=C_new
            B[rho]=committed(rho); B_bar=sum(B)/m
            U.remove(j_star); F.discard(j_star); refresh_forced()
        if not U: break
        U_avail=[j for j in U if j not in F]
        if not U_avail: break
        tau=min(B)
        urg=np.array([1./max(float(inst.delta[j])-tau,eps) for j in U_avail])
        cr_arr=np.array([CR[j] for j in U_avail])
        screen=(params.eta*minmax_norm(cr_arr,eps)+(1-params.eta)*minmax_norm(urg,eps))
        q_eff=(len(U_avail) if n<=100 else min(150,max(50,int(0.25*len(U_avail)))))
        top=np.argsort(screen)[::-1][:q_eff]; U_q=[U_avail[i] for i in top]
        info={}
        for j in U_q:
            (c1,rho1,p1,Cn1),c2=_best_insertions(j,m,sequences,C_hats,B_bar,inst,params)
            info[j]=(c1,rho1,p1,Cn1,c2)
        finite=[info[j][4]-info[j][0] for j in U_q if info[j][4]<math.inf]
        R_max=((max(finite)+inst.h_bar*inst.T_span) if finite else inst.h_bar*inst.T_span)
        best_c=np.array([info[j][0] for j in U_q])
        regret=np.array([(info[j][4]-info[j][0]) if info[j][4]<math.inf else R_max for j in U_q])
        cr_uq=np.array([CR[j] for j in U_q])
        score=(minmax_norm(best_c,eps)-0.20*minmax_norm(regret,eps)-0.10*minmax_norm(cr_uq,eps))
        j_star=U_q[int(np.argmin(score))]
        c_s,rho_s,p_s,Cn_s,_=info[j_star]
        if c_s==math.inf: F.add(j_star); continue
        do_insert(j_star,rho_s,p_s,Cn_s); refresh_forced()

    sequences,C_hats=inter_runway_repair(sequences,C_hats,inst,params)
    return sequences,C_hats


# ═════════════════════════════════════════════════════════════════════════════
#   §10  TC PROXY OBJECTIVE  (Optuna inner loop for n > 100)
# ═════════════════════════════════════════════════════════════════════════════

def total_target_conflict(sequences, inst) -> float:
    """Total pairwise TC = 0.5·(p_i+p_j)·max(s[i,j]−(δ_j−δ_i),0). GPU when available."""
    if USE_GPU and _GPU_AVAIL and _torch is not None and inst.n>=GPU_MIN_N:
        total=0.
        for seq in sequences:
            L=len(seq)
            if L<2: continue
            sa=np.asarray(seq,dtype=np.intp); ii,jj=np.triu_indices(L,k=1)
            i_t=_torch.from_numpy(sa[ii].astype(np.int64)).to(_CUDA_DEVICE)
            j_t=_torch.from_numpy(sa[jj].astype(np.int64)).to(_CUDA_DEVICE)
            v=inst._s_gpu[i_t,j_t]-(inst._delta_gpu[j_t]-inst._delta_gpu[i_t])
            total+=float(_torch.sum(0.5*(inst._p_arr_gpu[i_t]+inst._p_arr_gpu[j_t])
                                    *_torch.clamp(v,min=0.)))
        return total
    total=0.
    for seq in sequences:
        L=len(seq)
        if L<2: continue
        sa=np.asarray(seq,dtype=np.intp); ii,jj=np.triu_indices(L,k=1)
        v=inst.s[sa[ii],sa[jj]]-(inst.delta[sa[jj]]-inst.delta[sa[ii]])
        total+=float((0.5*(inst.p_arr[sa[ii]]+inst.p_arr[sa[jj]])*np.maximum(v,0.)).sum())
    return total


# ═════════════════════════════════════════════════════════════════════════════
#   §11  OPTUNA HYPERPARAMETER TUNING
# ═════════════════════════════════════════════════════════════════════════════

def _n_rbi_trials(n, base) -> int:
    if n<=100: return base
    if n<=250: return max(10,base//3)
    return max(5,base//6)


def _sa_n_trials(n, base) -> int:
    if n<=50:  return base
    if n<=100: return max(10,base//2)
    if n<=250: return max(6,base//4)
    return max(3,base//7)


def optimize_rbi_params(inst, m, n_trials, seed, n_jobs=1) -> HeuristicParams:
    """Tune HeuristicParams for (inst, m) using Optuna TPE.
    Objective: LP for n≤100; TC proxy otherwise."""
    if not _OPTUNA: return HeuristicParams()
    if n_trials==0: return HeuristicParams()
    use_lp=(inst.n<=100)
    def objective(trial):
        p=HeuristicParams(
            eta=trial.suggest_float('eta',0.20,0.80),
            mu_tc=trial.suggest_float('mu_tc',0.10,5.00),
            mu_late=trial.suggest_float('mu_late',0.01,2.00),
            mu_count=trial.suggest_float('mu_count',0.10,3.00),
            mu_sep=trial.suggest_float('mu_sep',0.00,0.50))
        seqs,_=ramp_rbi(inst,m,p)
        if use_lp:
            obj,_,feas,_=stage2_lp_objective(seqs,inst)
            return obj if feas else 1e12
        return total_target_conflict(seqs,inst)
    sampler=_optuna.samplers.TPESampler(seed=seed)
    study=_optuna.create_study(direction='minimize',sampler=sampler)
    study.optimize(objective,n_trials=n_trials,n_jobs=min(n_jobs,n_trials),show_progress_bar=False)
    bp=study.best_params
    return HeuristicParams(eta=bp['eta'],mu_tc=bp['mu_tc'],mu_late=bp['mu_late'],
                           mu_count=bp['mu_count'],mu_sep=bp['mu_sep'])


def optimize_sa_params(inst, m, params, n_trials, seed, n_jobs=1) -> MRSAParams:
    """Tune MRSAParams for (inst, m) using Optuna TPE (lp_repair_interval=0 for speed)."""
    if not _OPTUNA: return MRSAParams()
    if n_trials==0: return MRSAParams()
    N_tune=max(300,_n_iter(inst.n)//6)
    def objective(trial):
        p_sa=MRSAParams(
            chi0=trial.suggest_float('chi0',0.50,0.95),
            M_stag_frac=trial.suggest_float('M_stag_frac',0.05,0.30),
            beta=trial.suggest_float('beta',1.20,2.50),
            lp_gamma=trial.suggest_float('lp_gamma',0.01,0.20),
            chi_target=trial.suggest_float('chi_target',0.10,0.35),
            lp_repair_interval=0)
        seqs,_=ramp_rbi(inst,m,params)
        _,_,blp_seqs,best_lp,_,_=run_mr_sa(seqs,math.inf,inst,params,p_sa,N_tune,
                                             label="sa_tune",seed=trial.number*13+seed)
        if math.isinf(best_lp):
            lp_val,_,feas,_=stage2_lp_objective(blp_seqs or seqs,inst)
            best_lp=lp_val if feas else 1e12
        return best_lp
    sampler=_optuna.samplers.TPESampler(seed=seed)
    study=_optuna.create_study(direction='minimize',sampler=sampler)
    study.optimize(objective,n_trials=n_trials,n_jobs=min(n_jobs,n_trials),show_progress_bar=False)
    bp=study.best_params
    return MRSAParams(chi0=bp['chi0'],M_stag_frac=bp['M_stag_frac'],
                      beta=bp['beta'],lp_gamma=bp['lp_gamma'],chi_target=bp['chi_target'])


# ═════════════════════════════════════════════════════════════════════════════
#   §12  STAGE-2 LP: EXACT LANDING-TIME OPTIMISATION
# ═════════════════════════════════════════════════════════════════════════════

def stage2_lp_objective(sequences, inst, eps_tol=1e-6):
    """
    Minimise total weighted earliness/tardiness for fixed sequences.

    Decision variables: C_j ∈ [r_j,d_j], E_j≥0, T_j≥0.
    Constraints C3 cover ALL ordered pairs (i,j) with i≺j — required because
    OR Library separation matrices violate the triangle inequality.

    Returns (obj, C_lp, feasible, violations).
    """
    n=inst.n; C0,E0,T0=0,n,2*n; nv=3*n
    c_obj=np.zeros(nv); c_obj[E0:E0+n]=inst.g; c_obj[T0:T0+n]=inst.h
    sep_pairs=[(seq[a],seq[b]) for seq in sequences
               for a in range(len(seq)) for b in range(a+1,len(seq))]
    n_ineq=2*n+len(sep_pairs)
    rows=[];cols=[];vals=[];b_ub=np.empty(n_ineq);r=0
    for j in range(n):
        rows+=[r,r];cols+=[C0+j,E0+j];vals+=[-1.,-1.];b_ub[r]=-float(inst.delta[j]);r+=1
    for j in range(n):
        rows+=[r,r];cols+=[C0+j,T0+j];vals+=[1.,-1.];b_ub[r]=float(inst.delta[j]);r+=1
    for i,j in sep_pairs:
        rows+=[r,r];cols+=[C0+i,C0+j];vals+=[1.,-1.];b_ub[r]=-float(inst.s[i,j]);r+=1
    A_ub=csr_matrix((vals,(rows,cols)),shape=(n_ineq,nv))
    bounds=([(float(inst.r[j]),float(inst.d[j])) for j in range(n)]+[(0.,None)]*(2*n))
    res=linprog(c_obj,A_ub=A_ub,b_ub=b_ub,bounds=bounds,method='highs')
    if not res.success: return math.inf,None,False,[f"LP solver: {res.message}"]
    C_lp=res.x[C0:C0+n]; obj=float(res.fun); viol=[]
    for j in range(n):
        if C_lp[j]<inst.r[j]-eps_tol: viol.append(f"Aircraft {j}: C={C_lp[j]:.4f}<r={inst.r[j]:.4f}")
        if C_lp[j]>inst.d[j]+eps_tol:  viol.append(f"Aircraft {j}: C={C_lp[j]:.4f}>d={inst.d[j]:.4f}")
    for seq in sequences:
        for a in range(len(seq)):
            for b in range(a+1,len(seq)):
                i,j=seq[a],seq[b]
                if C_lp[j]-C_lp[i]<inst.s[i,j]-eps_tol:
                    viol.append(f"sep({i},{j}): {C_lp[j]-C_lp[i]:.4f}<{inst.s[i,j]:.4f}")
    return obj,C_lp,len(viol)==0,viol


# ═════════════════════════════════════════════════════════════════════════════
#   §13  EXACT FEASIBILITY VERIFICATION + EARLIEST-TIME OBJECTIVE
# ═════════════════════════════════════════════════════════════════════════════

def verify_and_exact_obj(sequences, inst, eps_tol=1e-6):
    """
    Earliest-feasible times with full pairwise separation propagation; then
    penalty objective and constraint violation check.

    Returns (feasible, violations, obj, C_exact).
    """
    n=inst.n; C_exact={}
    for rho,seq in enumerate(sequences):
        if not seq: continue
        C_r=[0.]*len(seq); C_r[0]=float(inst.r[seq[0]])
        for q in range(1,len(seq)):
            j=seq[q]; t=float(inst.r[j])
            for h in range(q): t=max(t,C_r[h]+float(inst.s[seq[h],j]))
            C_r[q]=t
        for q,j in enumerate(seq): C_exact[j]=C_r[q]
    viol=[]
    for j in range(n):
        if j not in C_exact: viol.append(f"Aircraft {j} not scheduled")
    for j,Cj in C_exact.items():
        if Cj<inst.r[j]-eps_tol: viol.append(f"Ac {j}: C={Cj:.2f}<r={inst.r[j]:.2f}")
        if Cj>inst.d[j]+eps_tol:  viol.append(f"Ac {j}: C={Cj:.2f}>d={inst.d[j]:.2f}")
    for rho,seq in enumerate(sequences):
        for qi in range(len(seq)):
            for qj in range(qi+1,len(seq)):
                i,j=seq[qi],seq[qj]; Ci=C_exact.get(i,0.); Cj=C_exact.get(j,0.)
                if Cj-Ci<inst.s[i,j]-eps_tol:
                    viol.append(f"Rwy{rho+1} sep({i},{j}): {Cj-Ci:.4f}<{inst.s[i,j]:.4f}")
    obj=sum(float(inst.g[j])*max(float(inst.delta[j])-Cj,0.)
            +float(inst.h[j])*max(Cj-float(inst.delta[j]),0.)
            for j,Cj in C_exact.items())
    return len(viol)==0,viol,obj,C_exact


# ═════════════════════════════════════════════════════════════════════════════
#   §14  SA PROXY COMPUTATION
# ═════════════════════════════════════════════════════════════════════════════

def _rwy_proxy_components(seq, inst):
    if not seq: return 0.,0.,0.
    L=len(seq); s_arr=np.asarray(seq,dtype=np.intp)
    if L>=2:
        ii,jj=np.triu_indices(L,k=1); i_ac=s_arr[ii]; j_ac=s_arr[jj]
        v=inst.s[i_ac,j_ac]-(inst.delta[j_ac]-inst.delta[i_ac])
        tc=float((0.5*(inst.p_arr[i_ac]+inst.p_arr[j_ac])*np.maximum(v,0.)).sum())
    else: tc=0.
    C_hat=np.asarray(surrogate_times(seq,inst))
    lbt=float((inst.h[s_arr]*np.maximum(C_hat-inst.delta[s_arr],0.)).sum())
    sep=float(inst.s[s_arr[:-1],s_arr[1:]].sum())*inst.h_bar if L>=2 else 0.
    return tc,lbt,sep


def _balance_term(seqs,inst)->float:
    n=inst.n; m=len(seqs)
    return (sum((len(seqs[r])-n/m)**2 for r in range(m))
            *float(inst.Pen_bar)/max((n/m)**2,1.))


def compute_proxy(seqs,tc_rwy,lbt_rwy,sep_rwy,inst,params)->float:
    return (params.mu_tc*float(tc_rwy.sum())+params.mu_late*float(lbt_rwy.sum())
            +params.mu_count*_balance_term(seqs,inst)+params.mu_sep*float(sep_rwy.sum()))


def _init_proxy_arrays(seqs,inst):
    m=len(seqs); tc_rwy=np.zeros(m); lbt_rwy=np.zeros(m); sep_rwy=np.zeros(m)
    for rho in range(m):
        tc_rwy[rho],lbt_rwy[rho],sep_rwy[rho]=_rwy_proxy_components(seqs[rho],inst)
    return tc_rwy,lbt_rwy,sep_rwy


# ═════════════════════════════════════════════════════════════════════════════
#   §15  PER-AIRCRAFT SCORING
# ═════════════════════════════════════════════════════════════════════════════

def _compute_per_aircraft_scores(seqs,inst):
    n=inst.n; pa_tc=np.zeros(n); pa_lbt=np.zeros(n)
    for seq in seqs:
        if not seq: continue
        L=len(seq); s_arr=np.asarray(seq,dtype=np.intp)
        if L>=2:
            ii,jj=np.triu_indices(L,k=1); i_ac=s_arr[ii]; j_ac=s_arr[jj]
            v=inst.s[i_ac,j_ac]-(inst.delta[j_ac]-inst.delta[i_ac])
            c=0.5*(inst.p_arr[i_ac]+inst.p_arr[j_ac])*np.maximum(v,0.)
            np.add.at(pa_tc,i_ac,c); np.add.at(pa_tc,j_ac,c)
        C_hat=np.asarray(surrogate_times(seq,inst))
        pa_lbt[s_arr]=inst.h[s_arr]*np.maximum(C_hat-inst.delta[s_arr],0.)
    return pa_tc,pa_lbt


def _lp_impact_scores(seqs,C_lp,inst,lambda_b=0.5,eps_tight=1e-4):
    """Impact_j = P_j + λ_b·binding_count_j (identifies tight-chain aircraft)."""
    n=inst.n; E=np.maximum(inst.delta-C_lp,0.); T=np.maximum(C_lp-inst.delta,0.)
    P=inst.g*E+inst.h*T; binding=np.zeros(n)
    for seq in seqs:
        L=len(seq)
        for qi in range(L):
            for qj in range(qi+1,L):
                i,j=seq[qi],seq[qj]
                if C_lp[j]-C_lp[i]-inst.s[i,j]<=eps_tight:
                    binding[i]+=1.; binding[j]+=1.
    return P+lambda_b*binding


def _pick_aircraft_targeted(seqs,inst,rng,pa_tc=None,pa_lbt=None,impact=None):
    """60% uniform, 25% top-20% by impact/pa_tc, 15% top-20% by pa_lbt."""
    m=len(seqs); flat=[(rho,pos) for rho in range(m) for pos in range(len(seqs[rho]))]
    if not flat: return 0,0
    r=rng.random(); scores=impact if impact is not None else pa_tc
    if r<0.60 or scores is None: return rng.choice(flat)
    if r<0.85:
        scored=sorted(((scores[seqs[rho][pos]],rho,pos) for rho,pos in flat),key=lambda x:-x[0])
        top=max(1,len(scored)//5); _,rho,pos=rng.choice(scored[:top]); return rho,pos
    lbt_arr=pa_lbt if pa_lbt is not None else scores
    scored=sorted(((lbt_arr[seqs[rho][pos]],rho,pos) for rho,pos in flat),key=lambda x:-x[0])
    top=max(1,len(scored)//5); _,rho,pos=rng.choice(scored[:top]); return rho,pos


# ═════════════════════════════════════════════════════════════════════════════
#   §16  SA NEIGHBOURHOOD OPERATORS
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class _MoveResult:
    seqs:     List[List[int]]
    affected: List[int]


def _op_n1_adjacent_swap(seqs,rho,p,inst):
    seq=seqs[rho]
    if p>=len(seq)-1: return None
    ns=seq[:]; ns[p],ns[p+1]=ns[p+1],ns[p]
    if not _runway_feasible(ns,inst): return None
    r=[s[:] for s in seqs]; r[rho]=ns; return _MoveResult(r,[rho])


def _op_n2_swap(seqs,rho,p,q,inst):
    seq=seqs[rho]
    if p==q or p>=len(seq) or q>=len(seq): return None
    ns=seq[:]; ns[p],ns[q]=ns[q],ns[p]
    if not _runway_feasible(ns,inst): return None
    r=[s[:] for s in seqs]; r[rho]=ns; return _MoveResult(r,[rho])


def _op_n3b_best_insertion(seqs,rho,p,inst,params):
    seq=seqs[rho]; L=len(seq)
    if L<2: return None
    ac=seq[p]; sm=seq[:p]+seq[p+1:]
    if not _runway_feasible(sm,inst): return None
    best_score,best_q=math.inf,-1
    for q in range(L):
        ns=sm[:q]+[ac]+sm[q:]
        if not _runway_feasible(ns,inst): continue
        tc,lbt,sep=_rwy_proxy_components(ns,inst)
        s=params.mu_tc*tc+params.mu_late*lbt+params.mu_sep*sep
        if s<best_score: best_score=s; best_q=q
    if best_q==-1: return None
    ns=sm[:best_q]+[ac]+sm[best_q:]
    r=[s[:] for s in seqs]; r[rho]=ns; return _MoveResult(r,[rho])


def _op_n4_block_reloc(seqs,rho,p,b,q,inst):
    seq=seqs[rho]
    if p+b>len(seq) or b<1: return None
    blk=seq[p:p+b]; rest=seq[:p]+seq[p+b:]
    ns=rest[:q%(len(rest)+1)]+blk+rest[q%(len(rest)+1):]
    if not _runway_feasible(ns,inst): return None
    r=[s[:] for s in seqs]; r[rho]=ns; return _MoveResult(r,[rho])


def _op_x1_transfer(seqs,rho_a,p,rho_b,q,inst):
    if rho_a==rho_b: return None
    sa=seqs[rho_a][:]; sb=seqs[rho_b][:]
    ac=sa.pop(p); sb.insert(min(q,len(sb)),ac)
    if not _runway_feasible(sa,inst) or not _runway_feasible(sb,inst): return None
    r=[s[:] for s in seqs]; r[rho_a]=sa; r[rho_b]=sb; return _MoveResult(r,[rho_a,rho_b])


def _op_x2_swap(seqs,rho_a,p,rho_b,q,inst):
    if rho_a==rho_b or not seqs[rho_a] or not seqs[rho_b]: return None
    if p>=len(seqs[rho_a]) or q>=len(seqs[rho_b]): return None
    sa=seqs[rho_a][:]; sb=seqs[rho_b][:]
    sa[p],sb[q]=sb[q],sa[p]
    if not _runway_feasible(sa,inst) or not _runway_feasible(sb,inst): return None
    r=[s[:] for s in seqs]; r[rho_a]=sa; r[rho_b]=sb; return _MoveResult(r,[rho_a,rho_b])


def _op_x3_best_transfer(seqs,rho_a,p,rho_b,inst,params,tc_rwy,lbt_rwy,sep_rwy):
    if rho_a==rho_b: return None
    sa=seqs[rho_a][:]; ac=sa.pop(p)
    if not _runway_feasible(sa,inst): return None
    best_delta,best_sb=math.inf,None
    n=inst.n; m=len(seqs); bs=float(inst.Pen_bar)/max((n/m)**2,1.)
    t=sum(len(s) for s in seqs); ob=(len(seqs[rho_a])-t/m)**2+(len(seqs[rho_b])-t/m)**2
    for q in range(len(seqs[rho_b])+1):
        sb=seqs[rho_b][:]; sb.insert(q,ac)
        if not _runway_feasible(sb,inst): continue
        ta,la,ea=_rwy_proxy_components(sa,inst); tb,lb,eb=_rwy_proxy_components(sb,inst)
        nb=(len(sa)-t/m)**2+(len(sb)-t/m)**2
        delta=(params.mu_tc*((ta+tb)-(tc_rwy[rho_a]+tc_rwy[rho_b]))
               +params.mu_late*((la+lb)-(lbt_rwy[rho_a]+lbt_rwy[rho_b]))
               +params.mu_count*(nb-ob)*bs
               +params.mu_sep*((ea+eb)-(sep_rwy[rho_a]+sep_rwy[rho_b])))
        if delta<best_delta: best_delta=delta; best_sb=sb
    if best_sb is None: return None
    r=[s[:] for s in seqs]; r[rho_a]=sa; r[rho_b]=best_sb; return _MoveResult(r,[rho_a,rho_b])


def _op_x4_block_transfer(seqs,rho_a,p,b,rho_b,q,inst):
    if rho_a==rho_b or p+b>len(seqs[rho_a]): return None
    blk=seqs[rho_a][p:p+b]; sa=seqs[rho_a][:p]+seqs[rho_a][p+b:]
    sb=seqs[rho_b][:]; sb[q:q]=blk
    if not _runway_feasible(sa,inst) or not _runway_feasible(sb,inst): return None
    r=[s[:] for s in seqs]; r[rho_a]=sa; r[rho_b]=sb; return _MoveResult(r,[rho_a,rho_b])


def _op_x7_tc_repair(seqs,tc_rwy,lbt_rwy,inst,params,rng,impact):
    m=len(seqs)
    cands=([(impact[seqs[rho][pos]],rho,pos) for rho in range(m) for pos in range(len(seqs[rho]))]
           if impact is not None
           else [(tc_rwy[rho]/max(len(seqs[rho]),1),rho,pos)
                 for rho in range(m) for pos in range(len(seqs[rho]))])
    if not cands: return None
    cands.sort(key=lambda x:-x[0]); _,rho_a,p=rng.choice(cands[:max(1,len(cands)//5)])
    others=[r for r in range(m) if r!=rho_a]
    if others:
        res=_op_x3_best_transfer(seqs,rho_a,p,rng.choice(others),inst,params,
                                  tc_rwy,lbt_rwy,np.zeros(m))
        if res is not None: return res
    return _op_n3b_best_insertion(seqs,rho_a,p,inst,params) if len(seqs[rho_a])>=2 else None


# ═════════════════════════════════════════════════════════════════════════════
#   §17  PHASE-DEPENDENT OPERATOR SELECTION
# ═════════════════════════════════════════════════════════════════════════════

_OPS_EARLY  = [("X1",.18),("X2",.18),("X3",.18),("X4",.10),("N2",.15),("N3b",.12),("N1",.09)]
_OPS_MID    = [("X1",.12),("X2",.12),("X3",.12),("X7",.14),("N2",.14),("N3b",.18),("N1",.10),("XE",.08)]
_OPS_LATE   = [("N1",.18),("N2",.17),("N3b",.23),("X2",.14),("X3",.10),("X7",.10),("XE",.08)]
_OPS_SINGLE = [("N1",.25),("N2",.28),("N3b",.30),("N4",.17)]


def _select_op(f,m,rng)->str:
    if m==1:     table=_OPS_SINGLE
    elif f<0.30: table=_OPS_EARLY
    elif f<0.75: table=_OPS_MID
    else:        table=_OPS_LATE
    ops,weights=zip(*table); return rng.choices(ops,weights=weights,k=1)[0]


def _apply_op(op,seqs,tc_rwy,lbt_rwy,sep_rwy,inst,params,p_sa,rng,stag,N_iter,
              pa_tc=None,pa_lbt=None,impact=None,C_lp=None):
    m=len(seqs); rho_a,pos_a=_pick_aircraft_targeted(seqs,inst,rng,pa_tc,pa_lbt,impact)
    L_a=len(seqs[rho_a])
    if op=="N1": return _op_n1_adjacent_swap(seqs,rho_a,rng.randint(0,max(L_a-2,0)),inst)
    elif op=="N2":
        if L_a<2: return None
        return _op_n2_swap(seqs,rho_a,rng.randint(0,L_a-1),rng.randint(0,L_a-1),inst)
    elif op=="N3b":
        if L_a<2: return None
        return _op_n3b_best_insertion(seqs,rho_a,rng.randint(0,L_a-1),inst,params)
    elif op=="N4":
        if L_a<2: return None
        b_cap=p_sa.B_stag if stag>=int(p_sa.M_stag_frac*N_iter) else p_sa.B_max
        b=rng.randint(1,min(b_cap,L_a))
        return _op_n4_block_reloc(seqs,rho_a,rng.randint(0,L_a-b),b,rng.randint(0,L_a-b),inst)
    elif op=="X1":
        if m<2: return None
        rho_b=rng.choice([r for r in range(m) if r!=rho_a])
        return _op_x1_transfer(seqs,rho_a,pos_a,rho_b,rng.randint(0,len(seqs[rho_b])),inst)
    elif op=="X2":
        if m<2 or not seqs[rho_a]: return None
        rho_b=rng.choice([r for r in range(m) if r!=rho_a])
        if not seqs[rho_b]: return None
        return _op_x2_swap(seqs,rho_a,pos_a,rho_b,rng.randint(0,len(seqs[rho_b])-1),inst)
    elif op=="X3":
        if m<2: return None
        rho_b=rng.choice([r for r in range(m) if r!=rho_a])
        return _op_x3_best_transfer(seqs,rho_a,pos_a,rho_b,inst,params,tc_rwy,lbt_rwy,sep_rwy)
    elif op=="X4":
        if m<2 or L_a<1: return None
        b=rng.randint(1,min(p_sa.B_max,L_a)); rho_b=rng.choice([r for r in range(m) if r!=rho_a])
        return _op_x4_block_transfer(seqs,rho_a,rng.randint(0,L_a-b),b,rho_b,
                                      rng.randint(0,len(seqs[rho_b])),inst)
    elif op=="X7": return _op_x7_tc_repair(seqs,tc_rwy,lbt_rwy,inst,params,rng,impact)
    elif op=="XE":
        if m<2: return None
        rho_b=rng.choice([r for r in range(m) if r!=rho_a])
        return _op_x3_best_transfer(seqs,rho_a,pos_a,rho_b,inst,params,tc_rwy,lbt_rwy,sep_rwy)
    return None


# ═════════════════════════════════════════════════════════════════════════════
#   §18  SA HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def _n_iter(n)->int:
    if n<=50:  return 2_000
    if n<=250: return 5_000
    return 8_000


def _R_candidates(n)->int:
    if n<=100: return 10
    if n<=250: return 20
    return 30


def _lp_repair_params(n):
    if n<=50:  return 20,20
    if n<=100: return 15,15
    if n<=250: return 10,10
    return 8,5


def _vnd_max_rounds(n)->int:
    if n<=100: return 15
    if n<=250: return 10
    return 5


def _n_full(t,N_iter)->int:
    f=t/max(N_iter,1)
    if f<=0.25: return 20
    if f<=0.75: return 50
    return 100


def _adaptive_t_limit(n,m,seed_lp,bks)->float:
    if bks is None:          return T_LIMIT
    if bks==0.:              return 60.
    if math.isinf(seed_lp):  return MAX_T_LIMIT
    gap=100.*(seed_lp-bks)/bks
    if gap<=0.:  return 60.
    if gap<=2.:  return 120.
    if gap<=5.:  return T_LIMIT
    if gap<=10.: return min(600.,MAX_T_LIMIT)
    return MAX_T_LIMIT


def _calibrate_t0(seqs,inst,params,p_sa,seed,N_iter)->float:
    rng=random.Random(seed)
    tc_rwy,lbt_rwy,sep_rwy=_init_proxy_arrays(seqs,inst)
    proxy_cur=compute_proxy(seqs,tc_rwy,lbt_rwy,sep_rwy,inst,params)
    pa_tc,pa_lbt=_compute_per_aircraft_scores(seqs,inst); deltas_pos=[]
    for _ in range(p_sa.n_cal):
        op=_select_op(0.5,len(seqs),rng)
        res=_apply_op(op,seqs,tc_rwy,lbt_rwy,sep_rwy,inst,params,p_sa,rng,0,N_iter,pa_tc,pa_lbt)
        if res is None: continue
        tc_n,lbt_n,sep_n=_init_proxy_arrays(res.seqs,inst)
        d=compute_proxy(res.seqs,tc_n,lbt_n,sep_n,inst,params)-proxy_cur
        if d>1e-9: deltas_pos.append(d)
    if not deltas_pos: return max(abs(proxy_cur)*0.01,1.)
    return max(-float(np.mean(deltas_pos))/math.log(p_sa.chi0+1e-12),1e-3)


# ═════════════════════════════════════════════════════════════════════════════
#   §19  LP-GUIDED REPAIR OPERATORS
# ═════════════════════════════════════════════════════════════════════════════

def _top_penalty_aircraft(C_lp,inst,q)->List[int]:
    E=np.maximum(inst.delta-C_lp,0.); T=np.maximum(C_lp-inst.delta,0.)
    return list(np.argsort(inst.g*E+inst.h*T)[::-1][:q])


def lp_guided_penalty_repair(seqs,C_lp,inst,params,K=15,q_lp=15):
    """Relocate each top-penalty aircraft to its globally best feasible position."""
    m=len(seqs)
    loc={seqs[rho][pos]:(rho,pos) for rho in range(m) for pos in range(len(seqs[rho]))}
    H=_top_penalty_aircraft(C_lp,inst,q_lp); candidates=[]
    for j in H:
        rho_src,pos_src=loc[j]; sm=seqs[rho_src][:pos_src]+seqs[rho_src][pos_src+1:]
        if not _runway_feasible(sm,inst): continue
        base=[s[:] for s in seqs]; base[rho_src]=sm
        for rho_dst in range(m):
            for p_dst in range(len(base[rho_dst])+1):
                cand=[s[:] for s in base]
                cand[rho_dst]=cand[rho_dst][:p_dst]+[j]+cand[rho_dst][p_dst:]
                if not _runway_feasible(cand[rho_dst],inst): continue
                tc_n,lbt_n,sep_n=_init_proxy_arrays(cand,inst)
                candidates.append((compute_proxy(cand,tc_n,lbt_n,sep_n,inst,params),cand))
        if len(candidates)>K*20:
            candidates.sort(key=lambda x:x[0]); candidates=candidates[:K*5]
    if not candidates: return None,math.inf
    candidates.sort(key=lambda x:x[0]); best_lp=math.inf; best_cand=None
    for _,cand in candidates[:K]:
        lp,_,feas,_=stage2_lp_objective(cand,inst)
        if feas and lp<best_lp: best_lp=lp; best_cand=cand
    return best_cand,best_lp


def lp_guided_pair_swap(seqs,C_lp,inst,params,q_lp=15,K=30,kappa=0.25):
    """Exchange high-penalty aircraft with target-time-compatible partners."""
    m=len(seqs); H=_top_penalty_aircraft(C_lp,inst,q_lp)
    loc={seqs[rho][pos]:(rho,pos) for rho in range(m) for pos in range(len(seqs[rho]))}
    W_bar=inst.W_bar; candidates=[]
    for i in H:
        rho_i,pos_i=loc[i]
        for rho_j in range(m):
            if rho_j==rho_i: continue
            for pos_j,j in enumerate(seqs[rho_j]):
                if abs(inst.delta[i]-inst.delta[j])>kappa*W_bar: continue
                res=_op_x2_swap(seqs,rho_i,pos_i,rho_j,pos_j,inst)
                if res is None: continue
                tc_n,lbt_n,sep_n=_init_proxy_arrays(res.seqs,inst)
                candidates.append((compute_proxy(res.seqs,tc_n,lbt_n,sep_n,inst,params),res.seqs))
    if not candidates: return None,math.inf
    candidates.sort(key=lambda x:x[0]); best_lp=math.inf; best_cand=None
    for _,cand in candidates[:K]:
        lp,_,feas,_=stage2_lp_objective(cand,inst)
        if feas and lp<best_lp: best_lp=lp; best_cand=cand
    return best_cand,best_lp


def target_conflict_repair(seqs,inst,params,K=15):
    """Deterministic repair for near-zero-objective instances."""
    m=len(seqs); conflicts=[]
    for rho,seq in enumerate(seqs):
        for qi in range(len(seq)):
            for qj in range(qi+1,len(seq)):
                i,j=seq[qi],seq[qj]
                tc=max(0.,float(inst.s[i,j])-(float(inst.delta[j])-float(inst.delta[i])))
                if tc>1e-9: conflicts.append((tc,rho,qi,i,j))
    if not conflicts: return None,math.inf
    conflicts.sort(reverse=True)
    loc={seqs[rho][pos]:(rho,pos) for rho in range(m) for pos in range(len(seqs[rho]))}
    candidates=[]
    for _,rho_c,qi,i,j in conflicts[:8]:
        for ac in [i,j]:
            rho_src,pos_src=loc[ac]; sm=seqs[rho_src][:pos_src]+seqs[rho_src][pos_src+1:]
            if not _runway_feasible(sm,inst): continue
            base=[s[:] for s in seqs]; base[rho_src]=sm
            for rho_dst in range(m):
                for p_dst in range(len(base[rho_dst])+1):
                    if rho_dst==rho_src and p_dst==pos_src: continue
                    cand=[s[:] for s in base]
                    cand[rho_dst]=cand[rho_dst][:p_dst]+[ac]+cand[rho_dst][p_dst:]
                    if not _runway_feasible(cand[rho_dst],inst): continue
                    tc_n,lbt_n,sep_n=_init_proxy_arrays(cand,inst)
                    candidates.append((compute_proxy(cand,tc_n,lbt_n,sep_n,inst,params),cand))
            if len(candidates)>K*15:
                candidates.sort(key=lambda x:x[0]); candidates=candidates[:K*4]
    if not candidates: return None,math.inf
    candidates.sort(key=lambda x:x[0]); best_lp=math.inf; best_cand=None
    for _,cand in candidates[:K]:
        lp,_,feas,_=stage2_lp_objective(cand,inst)
        if feas and lp<best_lp: best_lp=lp; best_cand=cand
    return best_cand,best_lp


def ejection_chain_transfer(seqs,C_lp,inst,params,depth=2,K=15):
    """
    Depth-D ejection chain.  Depth 1: X3 transfer.  Depth 2: j1:ρ1→ρ2 then j2:ρ2→ρ3.
    Depth capped at 1 for m < 3.
    """
    m=len(seqs)
    if m<3: depth=1
    q_lp,_=_lp_repair_params(inst.n); H=_top_penalty_aircraft(C_lp,inst,min(q_lp,6))
    loc={seqs[rho][pos]:(rho,pos) for rho in range(m) for pos in range(len(seqs[rho]))}
    candidates=[]
    for j1 in H:
        rho1,pos1=loc[j1]; sm1=seqs[rho1][:pos1]+seqs[rho1][pos1+1:]
        if not _runway_feasible(sm1,inst): continue
        for rho2 in range(m):
            if rho2==rho1: continue
            best_q2,best_seq2,best_s2=-1,None,math.inf
            for q2 in range(len(seqs[rho2])+1):
                c2=seqs[rho2][:q2]+[j1]+seqs[rho2][q2:]
                if not _runway_feasible(c2,inst): continue
                tc,lbt,sep=_rwy_proxy_components(c2,inst)
                s=params.mu_tc*tc+params.mu_late*lbt+params.mu_sep*sep
                if s<best_s2: best_s2=s; best_q2=q2; best_seq2=c2
            if best_seq2 is None: continue
            st1=[s[:] for s in seqs]; st1[rho1]=sm1; st1[rho2]=best_seq2
            if depth==1:
                tc_n,lbt_n,sep_n=_init_proxy_arrays(st1,inst)
                candidates.append((compute_proxy(st1,tc_n,lbt_n,sep_n,inst,params),[s[:] for s in st1]))
            else:
                for j2 in seqs[rho2]:
                    try: j2_pos=best_seq2.index(j2)
                    except ValueError: continue
                    sm2=best_seq2[:j2_pos]+best_seq2[j2_pos+1:]
                    if not _runway_feasible(sm2,inst): continue
                    for rho3 in range(m):
                        if rho3==rho2: continue
                        best_q3,best_seq3,best_s3=-1,None,math.inf
                        for q3 in range(len(st1[rho3])+1):
                            c3=st1[rho3][:q3]+[j2]+st1[rho3][q3:]
                            if not _runway_feasible(c3,inst): continue
                            tc,lbt,sep=_rwy_proxy_components(c3,inst)
                            s=params.mu_tc*tc+params.mu_late*lbt+params.mu_sep*sep
                            if s<best_s3: best_s3=s; best_q3=q3; best_seq3=c3
                        if best_seq3 is None: continue
                        st2=[s[:] for s in st1]; st2[rho2]=sm2; st2[rho3]=best_seq3
                        tc_n,lbt_n,sep_n=_init_proxy_arrays(st2,inst)
                        candidates.append((compute_proxy(st2,tc_n,lbt_n,sep_n,inst,params),[s[:] for s in st2]))
                    if len(candidates)>=K*20: break
                if len(candidates)>=K*20: break
            if len(candidates)>=K*20: break
        if len(candidates)>=K*20: break
    if not candidates: return None,math.inf
    candidates.sort(key=lambda x:x[0]); best_lp=math.inf; best_cand=None
    for _,cand in candidates[:K]:
        lp,_,feas,_=stage2_lp_objective(cand,inst)
        if feas and lp<best_lp: best_lp=lp; best_cand=cand
    return best_cand,best_lp


# ═════════════════════════════════════════════════════════════════════════════
#   §20  ELITE SOLUTION POOL
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class _EliteSolution:
    seqs:   List[List[int]]
    lp_obj: float
    C_lp:   Optional[np.ndarray]


class ElitePool:
    """Fixed-size pool of LP-certified schedules with runway-Hamming diversity guard."""

    def __init__(self, max_size=ELITE_POOL_MAX, min_diversity=ELITE_MIN_DIV):
        self.solutions:     List[_EliteSolution] = []
        self.max_size:      int = max_size
        self.min_diversity: int = min_diversity

    def runway_distance(self,seqs_a,seqs_b)->int:
        m=len(seqs_a)
        assign_a={seqs_a[rho][pos]:rho for rho in range(m) for pos in range(len(seqs_a[rho]))}
        return sum(1 for rho in range(len(seqs_b)) for j in seqs_b[rho] if assign_a.get(j)!=rho)

    def try_add(self,seqs,lp_obj,C_lp)->bool:
        if math.isinf(lp_obj): return False
        if not self.solutions:
            self.solutions.append(_EliteSolution([s[:] for s in seqs],lp_obj,
                                                  C_lp.copy() if C_lp is not None else None))
            return True
        diverse=all(self.runway_distance(seqs,s.seqs)>=self.min_diversity for s in self.solutions)
        worst_lp=max(s.lp_obj for s in self.solutions)
        if lp_obj<worst_lp or diverse:
            self.solutions.append(_EliteSolution([s[:] for s in seqs],lp_obj,
                                                  C_lp.copy() if C_lp is not None else None))
            if len(self.solutions)>self.max_size:
                self.solutions.sort(key=lambda s:s.lp_obj)
                self.solutions=self.solutions[:self.max_size]
            return True
        return False

    @property
    def best(self)->Optional[_EliteSolution]:
        return min(self.solutions,key=lambda s:s.lp_obj) if self.solutions else None

    def most_diverse_pair(self):
        if len(self.solutions)<2: return None,None
        best_d=-1; best_a=best_b=None
        for i in range(len(self.solutions)):
            for j in range(i+1,len(self.solutions)):
                d=self.runway_distance(self.solutions[i].seqs,self.solutions[j].seqs)
                if d>best_d: best_d=d; best_a,best_b=self.solutions[i],self.solutions[j]
        return best_a,best_b

    def best_quality_pair(self):
        if len(self.solutions)<2: return None,None
        ss=sorted(self.solutions,key=lambda s:s.lp_obj)
        for i in range(len(ss)):
            for j in range(i+1,len(ss)):
                if self.runway_distance(ss[i].seqs,ss[j].seqs)>=self.min_diversity:
                    return ss[i],ss[j]
        return self.most_diverse_pair()


# ═════════════════════════════════════════════════════════════════════════════
#   §21  PATH RELINKING
# ═════════════════════════════════════════════════════════════════════════════

def path_relink(sol_a,sol_b,inst,params,max_steps=40,eval_interval=5,K_lp=8):
    """Walk from sol_a toward sol_b by iteratively moving differing aircraft."""
    m=len(sol_a.seqs); current=[s[:] for s in sol_a.seqs]
    best_seqs=[s[:] for s in sol_a.seqs]; best_lp=sol_a.lp_obj
    assign_b={sol_b.seqs[rho][pos]:rho for rho in range(m)
              for pos in range(len(sol_b.seqs[rho]))}
    proxy_buffer=[]
    def _do_eval():
        nonlocal best_seqs,best_lp
        proxy_buffer.sort(key=lambda x:x[0])
        for _,cand in proxy_buffer[:K_lp]:
            lp,_,feas,_=stage2_lp_objective(cand,inst)
            if feas and lp<best_lp-1e-9: best_seqs=cand; best_lp=lp
        proxy_buffer.clear()
    for step in range(max_steps):
        assign_cur={current[rho][pos]:rho for rho in range(m) for pos in range(len(current[rho]))}
        differing=[(j,assign_b[j]) for j in assign_b if assign_cur.get(j)!=assign_b[j]]
        if not differing: break
        if sol_a.C_lp is not None:
            impact=_lp_impact_scores(current,sol_a.C_lp,inst)
            differing.sort(key=lambda x:-impact[x[0]])
        else:
            differing.sort(key=lambda x:-(inst.g[x[0]]+inst.h[x[0]]))
        moved=False
        for j,rho_target in differing[:5]:
            rho_cur=assign_cur.get(j)
            if rho_cur is None or rho_cur==rho_target: continue
            pos_cur=current[rho_cur].index(j)
            sm=current[rho_cur][:pos_cur]+current[rho_cur][pos_cur+1:]
            if not _runway_feasible(sm,inst): continue
            best_q,best_score=-1,math.inf
            for q in range(len(current[rho_target])+1):
                cs=current[rho_target][:q]+[j]+current[rho_target][q:]
                if not _runway_feasible(cs,inst): continue
                tc,lbt,sep=_rwy_proxy_components(cs,inst)
                s=params.mu_tc*tc+params.mu_late*lbt+params.mu_sep*sep
                if s<best_score: best_score=s; best_q=q
            if best_q==-1: continue
            current[rho_cur]=sm
            current[rho_target]=current[rho_target][:best_q]+[j]+current[rho_target][best_q:]
            moved=True; break
        if not moved: break
        tc_n,lbt_n,sep_n=_init_proxy_arrays(current,inst)
        px=compute_proxy(current,tc_n,lbt_n,sep_n,inst,params)
        proxy_buffer.append((px,[s[:] for s in current]))
        if (step+1)%eval_interval==0: _do_eval()
    if proxy_buffer: _do_eval()
    return best_seqs,best_lp


# ═════════════════════════════════════════════════════════════════════════════
#   §22  SAMPLE-AND-SELECT CANDIDATE GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def _generate_candidate_pool(f,seqs,tc_rwy,lbt_rwy,sep_rwy,inst,params,p_sa,rng,
                              stag,N_iter,R,pa_tc,pa_lbt,impact,C_lp):
    pool=[]
    for _ in range(R):
        op=_select_op(f,len(seqs),rng)
        res=_apply_op(op,seqs,tc_rwy,lbt_rwy,sep_rwy,inst,params,p_sa,rng,
                      stag,N_iter,pa_tc=pa_tc,pa_lbt=pa_lbt,impact=impact,C_lp=C_lp)
        if res is None: continue
        tc_n=tc_rwy.copy(); lbt_n=lbt_rwy.copy(); sep_n=sep_rwy.copy()
        for rho in res.affected:
            tc_n[rho],lbt_n[rho],sep_n[rho]=_rwy_proxy_components(res.seqs[rho],inst)
        pool.append((compute_proxy(res.seqs,tc_n,lbt_n,sep_n,inst,params),res,tc_n,lbt_n,sep_n))
    pool.sort(key=lambda x:x[0])
    return pool


# ═════════════════════════════════════════════════════════════════════════════
#   §23  SINGLE SA CHAIN
# ═════════════════════════════════════════════════════════════════════════════

def run_mr_sa(init_seqs,init_lp,inst,params,p_sa,N_iter,
              label="chain",seed=0,T0=None,t_deadline=None):
    """
    Run one SA chain.  Returns (best_p_seqs, best_proxy, best_lp_seqs, best_lp,
    best_C_lp, stats).  stats['lp_timeline'] logs every LP improvement as
    (wall_time_s, lp_val).  stats['t_best_lp'] is the time-to-best-LP.
    """
    CHI_TARGET=p_sa.chi_target; ALPHA_STEP=p_sa.alpha_step
    ALPHA_LO=p_sa.alpha_lo; ALPHA_HI=p_sa.alpha_hi
    MAX_REHEATS=p_sa.max_reheats; M_STAG=max(1,int(p_sa.M_stag_frac*N_iter))
    GAMMA=p_sa.lp_gamma; LP_REPAIR=p_sa.lp_repair_interval
    NZ_THRESH=p_sa.near_zero_threshold
    EC_DEPTH=min(p_sa.ejection_chain_depth,2 if len(init_seqs)<3 else p_sa.ejection_chain_depth)
    R=_R_candidates(inst.n)
    rng=random.Random(seed); m=len(init_seqs); t0=time.perf_counter()
    seqs=[s[:] for s in init_seqs]
    tc_rwy,lbt_rwy,sep_rwy=_init_proxy_arrays(seqs,inst)
    proxy=compute_proxy(seqs,tc_rwy,lbt_rwy,sep_rwy,inst,params)
    pa_tc,pa_lbt=_compute_per_aircraft_scores(seqs,inst); impact=None
    best_p_seqs=[s[:] for s in seqs]; best_proxy=proxy; t_best_proxy=0.
    best_lp_seqs=[s[:] for s in seqs]; best_lp=init_lp; best_C_lp=None; t_best_lp=0.
    lp_timeline=[(0.,init_lp)] if not math.isinf(init_lp) else []
    best_proxy_lp_checked=proxy
    T=T0 or _calibrate_t0(seqs,inst,params,p_sa,seed,N_iter)
    T_min=T*p_sa.T_min_frac; alpha=(ALPHA_HI+ALPHA_LO)/2.
    history=[]; alpha_history=[]; stag=0; n_reheats=0; n_accepted=0; n_tried=0
    q_lp,K=_lp_repair_params(inst.n)
    for t in range(1,N_iter+1):
        if t_deadline is not None and time.perf_counter()>=t_deadline: break
        f=t/N_iter
        pool=_generate_candidate_pool(f,seqs,tc_rwy,lbt_rwy,sep_rwy,inst,params,p_sa,rng,
                                       stag,N_iter,R,pa_tc,pa_lbt,impact,best_C_lp)
        if not pool: history.append(best_proxy); alpha_history.append(alpha); continue
        if rng.random()<0.80: proxy_new,res,tc_n,lbt_n,sep_n=pool[0]
        else:                 proxy_new,res,tc_n,lbt_n,sep_n=rng.choice(pool[:min(5,len(pool))])
        n_tried+=1; dlt=proxy_new-proxy
        accept=(dlt<=0 or rng.random()<math.exp(-dlt/max(T,1e-15)))
        if accept:
            seqs=res.seqs; tc_rwy=tc_n; lbt_rwy=lbt_n; sep_rwy=sep_n; proxy=proxy_new
            n_accepted+=1; stag=max(stag-1,0) if dlt<0 else stag+1
            if proxy<best_proxy-1e-9:
                best_p_seqs=[s[:] for s in seqs]; best_proxy=proxy
                t_best_proxy=time.perf_counter()-t0; stag=0
        else: stag+=1
        call_lp=(t%_n_full(t,N_iter)==0 or proxy_new<(1.-GAMMA)*best_proxy_lp_checked)
        if call_lp:
            lp_val,C_cur,lp_feas,_=stage2_lp_objective(seqs,inst)
            best_proxy_lp_checked=proxy
            if lp_feas and lp_val<best_lp-1e-9:
                best_lp_seqs=[s[:] for s in seqs]; best_lp=lp_val; best_C_lp=C_cur
                t_best_lp=time.perf_counter()-t0; stag=0
                impact=_lp_impact_scores(seqs,C_cur,inst,p_sa.lambda_binding,p_sa.eps_tight)
                lp_timeline.append((t_best_lp,lp_val))
        if LP_REPAIR>0 and t%LP_REPAIR==0 and best_C_lp is not None:
            cand,cand_lp=lp_guided_penalty_repair(best_lp_seqs,best_C_lp,inst,params,K=K,q_lp=q_lp)
            if cand is not None and cand_lp<best_lp-1e-9:
                best_lp_seqs=cand; best_lp=cand_lp
                _,best_C_lp,_,_=stage2_lp_objective(best_lp_seqs,inst)
                if best_C_lp is not None:
                    impact=_lp_impact_scores(best_lp_seqs,best_C_lp,inst,p_sa.lambda_binding,p_sa.eps_tight)
                lp_timeline.append((time.perf_counter()-t0,cand_lp)); stag=0
        if LP_REPAIR>0 and t%(LP_REPAIR*2)==0 and best_lp<NZ_THRESH:
            cand,cand_lp=target_conflict_repair(best_lp_seqs,inst,params,K=max(K//2,3))
            if cand is not None and cand_lp<best_lp-1e-9:
                best_lp_seqs=cand; best_lp=cand_lp
                _,C_new,feas_new,_=stage2_lp_objective(best_lp_seqs,inst)
                if feas_new: best_C_lp=C_new; lp_timeline.append((time.perf_counter()-t0,cand_lp))
        if LP_REPAIR>0 and t%(LP_REPAIR*3)==0 and best_C_lp is not None and m>=2:
            cand,cand_lp=ejection_chain_transfer(best_lp_seqs,best_C_lp,inst,params,
                                                  depth=EC_DEPTH,K=max(K//2,3))
            if cand is not None and cand_lp<best_lp-1e-9:
                best_lp_seqs=cand; best_lp=cand_lp
                _,C_new,feas_new,_=stage2_lp_objective(best_lp_seqs,inst)
                if feas_new:
                    best_C_lp=C_new
                    impact=_lp_impact_scores(best_lp_seqs,best_C_lp,inst,p_sa.lambda_binding,p_sa.eps_tight)
                lp_timeline.append((time.perf_counter()-t0,cand_lp)); stag=0
        if t%_n_full(t,N_iter)==0:
            chi=n_accepted/max(n_tried,1)
            alpha=(max(ALPHA_LO,alpha-ALPHA_STEP) if chi>CHI_TARGET
                   else min(ALPHA_HI,alpha+ALPHA_STEP))
            n_accepted=n_tried=0; pa_tc,pa_lbt=_compute_per_aircraft_scores(seqs,inst)
        T=max(T*alpha,T_min)
        if stag>=M_STAG:
            if n_reheats>=MAX_REHEATS: break
            T=min(T*p_sa.t_reheat,T0 or T)
            for _ in range(5):
                pres=_apply_op(rng.choice(["X4","X2"]),seqs,tc_rwy,lbt_rwy,sep_rwy,
                               inst,params,p_sa,rng,M_STAG+1,N_iter,pa_tc=pa_tc,pa_lbt=pa_lbt,impact=impact)
                if pres is not None:
                    for rho in pres.affected:
                        tc_rwy[rho],lbt_rwy[rho],sep_rwy[rho]=_rwy_proxy_components(pres.seqs[rho],inst)
                    seqs=pres.seqs; proxy=compute_proxy(seqs,tc_rwy,lbt_rwy,sep_rwy,inst,params); break
            stag=0; n_reheats+=1
        history.append(best_proxy); alpha_history.append(alpha)
    if math.isinf(best_lp):
        lp_val,C_cur,lp_feas,_=stage2_lp_objective(best_p_seqs,inst)
        if lp_feas:
            best_lp_seqs=[s[:] for s in best_p_seqs]; best_lp=lp_val; best_C_lp=C_cur
            lp_timeline.append((time.perf_counter()-t0,lp_val))
    return best_p_seqs,best_proxy,best_lp_seqs,best_lp,best_C_lp,{
        'label':label,'history':history,'alpha_history':alpha_history,
        't_best_proxy':t_best_proxy,'t_best_lp':t_best_lp,
        'wall':time.perf_counter()-t0,'lp_timeline':lp_timeline}


# ═════════════════════════════════════════════════════════════════════════════
#   §24  LP-VND POLISH
# ═════════════════════════════════════════════════════════════════════════════

def lp_vnd_polish(seqs,init_lp,C_lp,inst,params,p_sa=None,max_rounds=10,t_limit=90.):
    """
    Monotone LP-VND with four neighbourhoods (penalty repair, pair swap,
    target-conflict repair, ejection chain).  Restarts from N1 on any
    LP improvement.
    """
    p_sa=p_sa or MRSAParams()
    best_seqs=[s[:] for s in seqs]; best_lp=init_lp
    best_C=C_lp.copy() if C_lp is not None else None
    q_lp,K=_lp_repair_params(inst.n); t0=time.perf_counter(); m=len(seqs)
    ec_depth=min(p_sa.ejection_chain_depth,2 if m<3 else p_sa.ejection_chain_depth)
    for _ in range(max_rounds):
        if time.perf_counter()-t0>t_limit: break
        improved=False
        if best_C is not None:
            cand,cand_lp=lp_guided_penalty_repair(best_seqs,best_C,inst,params,K=K,q_lp=q_lp)
            if cand is not None and cand_lp<best_lp-1e-9:
                best_seqs=cand; best_lp=cand_lp; _,best_C,_,_=stage2_lp_objective(best_seqs,inst)
                improved=True; continue
        if best_C is not None:
            cand,cand_lp=lp_guided_pair_swap(best_seqs,best_C,inst,params,q_lp=q_lp,K=K)
            if cand is not None and cand_lp<best_lp-1e-9:
                best_seqs=cand; best_lp=cand_lp; _,best_C,_,_=stage2_lp_objective(best_seqs,inst)
                improved=True; continue
        if best_lp<200.:
            cand,cand_lp=target_conflict_repair(best_seqs,inst,params,K=max(K//2,3))
            if cand is not None and cand_lp<best_lp-1e-9:
                best_seqs=cand; best_lp=cand_lp
                _,C_new,feas_new,_=stage2_lp_objective(best_seqs,inst)
                if feas_new: best_C=C_new
                improved=True; continue
        if best_C is not None and m>=2:
            cand,cand_lp=ejection_chain_transfer(best_seqs,best_C,inst,params,
                                                  depth=ec_depth,K=max(K//2,3))
            if cand is not None and cand_lp<best_lp-1e-9:
                best_seqs=cand; best_lp=cand_lp; _,best_C,_,_=stage2_lp_objective(best_seqs,inst)
                improved=True; continue
        if not improved: break
    return best_seqs,best_lp


# ═════════════════════════════════════════════════════════════════════════════
#   §25  SEED CONSTRUCTION HEURISTICS  (H1–H9)
#
#   All functions return (seqs, C_hats) matching the ramp_rbi interface.
#   These are called only from _build_seed_portfolio (§25b).
# ═════════════════════════════════════════════════════════════════════════════

def seed_fcfs(inst: Instance, m: int) -> Tuple[List[List[int]], List[List[float]]]:
    """
    H1 — First-Come First-Served.

    Processes aircraft in order of earliest landing time r_j (arrival order proxy).
    Each aircraft is assigned to the runway with the smallest current committed time.
    Separation from the last aircraft on that runway is respected in the update.

    Complexity: O(n log n + n·m).
    Reference: Beasley et al. (2000); Ahmadian & Salehipour (2022).
    """
    order  = list(np.argsort(inst.r))
    seqs   = [[] for _ in range(m)]
    committed = [0.0] * m
    for j in order:
        rho  = int(np.argmin(committed))
        prev = seqs[rho][-1] if seqs[rho] else -1
        sep  = float(inst.s[prev, j]) if prev >= 0 else 0.0
        seqs[rho].append(j)
        committed[rho] = max(float(inst.r[j]), committed[rho] + sep)
    return seqs, [surrogate_times(seq, inst) for seq in seqs]


def seed_edd_balanced(inst: Instance, m: int) -> Tuple[List[List[int]], List[List[float]]]:
    """
    H2 — EDD-Balanced (Earliest Deadline / Target-time, round-robin).

    Sorts by δ_j and assigns aircraft round-robin, giving each runway a
    load-balanced target-time-ordered sequence.

    Complexity: O(n log n).
    Reference: Pinedo (2016), §3.2 (EDD optimal for 1 ‖ L_max).
    """
    order = list(np.argsort(inst.delta))
    seqs  = [[] for _ in range(m)]
    for k, j in enumerate(order):
        seqs[k % m].append(j)
    return seqs, [surrogate_times(seq, inst) for seq in seqs]


def seed_wedd(inst: Instance, m: int) -> Tuple[List[List[int]], List[List[float]]]:
    """
    H3 — Weighted EDD.

    Sorts by δ_j / (g_j + h_j): aircraft whose target time is small relative
    to their penalty rate are scheduled first.  Each aircraft is assigned to
    the runway with the smallest current committed time.

    Complexity: O(n log n + n·m).
    Reference: Ernst, Krishnamoorthy & Storer (1999).
    """
    scores    = inst.delta / np.maximum(inst.g + inst.h, inst.eps)
    order     = list(np.argsort(scores))
    seqs      = [[] for _ in range(m)]
    C_hats    = [[] for _ in range(m)]
    committed = [0.0] * m
    for j in order:
        rho = int(np.argmin(committed))
        seqs[rho].append(j)
        C_hats[rho] = surrogate_times(seqs[rho], inst)
        committed[rho] = C_hats[rho][-1] if C_hats[rho] else 0.0
    return seqs, C_hats


def seed_atc(inst: Instance, m: int,
             K: float = ATC_K) -> Tuple[List[List[int]], List[List[float]]]:
    """
    H4 — Apparent Tardiness Cost (ATC).

    When a runway opens at time t, scores each unscheduled aircraft j by:

        I_j(t) = h_j / s̄_j · exp(−max(δ_j − s̄_j − t, 0) / (K·s̄))

    where s̄_j is j's mean bilateral separation load and s̄ is the instance mean.
    Large K → WSPT rule; small K → minimum-slack rule (Pinedo 2016, §14.2).

    Complexity: O(n²·m).
    Reference: Vepsäläinen & Morton (1987); Pinedo (2016), §14.2.
    """
    s_sym  = (inst.s + inst.s.T) / 2.0; np.fill_diagonal(s_sym, 0.0)
    s_mean = s_sym.sum(axis=1) / max(inst.n - 1, 1)
    s_bar  = float(np.mean(s_mean[s_mean > 0])) if np.any(s_mean > 0) else 1.0
    Ks     = max(K * s_bar, 1e-9)
    seqs   = [[] for _ in range(m)]
    C_hats = [[] for _ in range(m)]
    committed   = [0.0] * m
    unscheduled = list(range(inst.n))
    while unscheduled:
        rho = int(np.argmin(committed)); t = committed[rho]
        best_j, best_idx = -1, -math.inf
        for j in unscheduled:
            slack = max(float(inst.delta[j]) - s_mean[j] - t, 0.0)
            idx   = (float(inst.h[j]) / max(s_mean[j], 1e-9)) * math.exp(-slack / Ks)
            if idx > best_idx: best_idx = idx; best_j = j
        seqs[rho].append(best_j)
        C_hats[rho] = surrogate_times(seqs[rho], inst)
        committed[rho] = C_hats[rho][-1] if C_hats[rho] else 0.0
        unscheduled.remove(best_j)
    return seqs, C_hats


def seed_atcs(inst: Instance, m: int,
              K1: float = ATCS_K1,
              K2: float = ATCS_K2) -> Tuple[List[List[int]], List[List[float]]]:
    """
    H5 — Apparent Tardiness Cost with Setups (ATCS).

    Extends ATC (H4) with a separation penalty for the last-landed aircraft ℓ
    on the candidate runway:

        I_j(t, ℓ) = h_j/s̄_j · exp(−max(δ_j−s̄_j−t,0)/(K₁·s̄)) · exp(−s[ℓ,j]/(K₂·s̄))

    Directly accounts for sequence-dependent separation costs.

    Complexity: O(n²·m).
    Reference: Lee, Bhaskaran & Pinedo (1997); Pinedo (2016), §14.2.
    """
    s_sym  = (inst.s + inst.s.T) / 2.0; np.fill_diagonal(s_sym, 0.0)
    s_mean = s_sym.sum(axis=1) / max(inst.n - 1, 1)
    s_bar  = float(np.mean(s_mean[s_mean > 0])) if np.any(s_mean > 0) else 1.0
    K1s    = max(K1 * s_bar, 1e-9); K2s = max(K2 * s_bar, 1e-9)
    seqs   = [[] for _ in range(m)]
    C_hats = [[] for _ in range(m)]
    committed   = [0.0] * m
    last        = [-1]  * m
    unscheduled = list(range(inst.n))
    while unscheduled:
        rho = int(np.argmin(committed)); t = committed[rho]; ell = last[rho]
        best_j, best_idx = -1, -math.inf
        for j in unscheduled:
            slack   = max(float(inst.delta[j]) - s_mean[j] - t, 0.0)
            sep_val = float(inst.s[ell, j]) if ell >= 0 else 0.0
            idx     = ((float(inst.h[j]) / max(s_mean[j], 1e-9))
                       * math.exp(-slack / K1s)
                       * math.exp(-sep_val / K2s))
            if idx > best_idx: best_idx = idx; best_j = j
        seqs[rho].append(best_j); last[rho] = best_j
        C_hats[rho] = surrogate_times(seqs[rho], inst)
        committed[rho] = C_hats[rho][-1] if C_hats[rho] else 0.0
        unscheduled.remove(best_j)
    return seqs, C_hats


def seed_caf(inst: Instance, m: int,
             params: HeuristicParams) -> Tuple[List[List[int]], List[List[float]]]:
    """
    H6 — Critical-Aircraft-First Insertion (CAF).

    Schedules aircraft in strict decreasing CR_j order, inserting each at the
    globally cheapest feasible (runway, position) using the same composite
    cost Δ(j,ρ,p) as TC-RBI.  Prevents easy aircraft from consuming positions
    needed by highly constrained ones.

    Complexity: O(n²m).
    Reference: Project framework document §9.4.
    """
    _, CR = compute_priorities(inst)
    order = list(np.argsort(CR)[::-1])
    seqs      = [[] for _ in range(m)]
    C_hats    = [[] for _ in range(m)]
    committed = [0.0] * m; B_bar = 0.0
    for j in order:
        best_cost, best_rho, best_p, best_Cn = math.inf, 0, 0, []
        for rho in range(m):
            for p in _candidate_positions(j, rho, seqs, inst):
                cost, Cn = evaluate_insertion(j, rho, p, seqs, C_hats, B_bar, inst, params)
                if cost < best_cost:
                    best_cost, best_rho, best_p, best_Cn = cost, rho, p, Cn
        if best_cost == math.inf:
            best_rho, best_p, best_Cn = min_violation_insert(j, seqs, inst)
        seqs[best_rho].insert(best_p, j); C_hats[best_rho] = best_Cn
        committed[best_rho] = C_hats[best_rho][-1] if C_hats[best_rho] else 0.0
        B_bar = sum(committed) / m
    return seqs, C_hats


def seed_mpds(inst: Instance, m: int,
              params: HeuristicParams) -> Tuple[List[List[int]], List[List[float]]]:
    """
    H7 — Most-Penalised-Displacement-First (MPDS).

    At each step scores each unscheduled aircraft j by (g_j+h_j)/min_cost_j,
    where min_cost_j is the cheapest feasible insertion across all runways.
    The highest-scoring aircraft is inserted at its best position, prioritising
    aircraft whose misplacement would be most costly.

    Only generated for n ≤ MPDS_MAX_N (default 150) due to O(n³/m) cost.
    Reference: Beasley et al. (2000) greedy construction variants.
    """
    seqs      = [[] for _ in range(m)]
    C_hats    = [[] for _ in range(m)]
    committed = [0.0] * m; B_bar = 0.0
    unscheduled = list(range(inst.n))
    while unscheduled:
        best_score                      = -math.inf
        sel_idx, sel_rho, sel_p, sel_Cn = 0, 0, 0, []
        for ji, j in enumerate(unscheduled):
            min_cost = math.inf; b_rho, b_p, b_Cn = 0, 0, []
            for rho in range(m):
                for p in _candidate_positions(j, rho, seqs, inst):
                    cost, Cn = evaluate_insertion(j, rho, p, seqs, C_hats, B_bar, inst, params)
                    if cost < min_cost:
                        min_cost, b_rho, b_p, b_Cn = cost, rho, p, Cn
            score = (float(inst.g[j]) + float(inst.h[j])) / max(min_cost, 1e-9) \
                    if min_cost < math.inf else -math.inf
            if score > best_score:
                best_score, sel_idx, sel_rho, sel_p, sel_Cn = score, ji, b_rho, b_p, b_Cn
        j = unscheduled.pop(sel_idx)
        if best_score == -math.inf:
            sel_rho, sel_p, sel_Cn = min_violation_insert(j, seqs, inst)
        seqs[sel_rho].insert(sel_p, j); C_hats[sel_rho] = sel_Cn
        committed[sel_rho] = C_hats[sel_rho][-1] if C_hats[sel_rho] else 0.0
        B_bar = sum(committed) / m
    return seqs, C_hats


def seed_wcc(inst: Instance, m: int,
             params: HeuristicParams) -> Tuple[List[List[int]], List[List[float]]]:
    """
    H8 — Wake-Compatibility Chain Construction (WCC).

    Groups aircraft with low mutual separation into chains using:

        W_ij = (s̄ − s_ij)/s̄ − |δ_j−δ_i|/T_span − max(0,r_j−δ_i)/T_span

    Chains are built greedily from the most critical aircraft (highest CR_j),
    then assigned to runways in decreasing length order, reducing wake-turbulence
    separation burden before SA begins.

    Complexity: O(n²) chain extraction + O(n²m) assignment.
    Reference: Clarke & Wright (1964) savings concept; project framework §9.3.
    """
    n = inst.n; _, CR = compute_priorities(inst)
    s_bar = inst.s_bar; T_span = max(inst.T_span, inst.eps)
    W = np.zeros((n, n))
    for i in range(n):
        savings   = (s_bar - inst.s[i]) / max(s_bar, inst.eps)
        tgt_gap   = np.abs(inst.delta - inst.delta[i]) / T_span
        idle_wait = np.maximum(inst.r - inst.delta[i], 0.0) / T_span
        W[i]      = savings - tgt_gap - idle_wait
    np.fill_diagonal(W, -np.inf)
    unassigned = set(range(n)); chains = []
    while unassigned:
        j0 = max(unassigned, key=lambda j: CR[j])
        chain = [j0]; unassigned.remove(j0)
        while True:
            tail   = chain[-1]
            best_k = max(
                (k for k in unassigned
                 if inst.delta[k] >= inst.delta[tail] and W[tail, k] > 0),
                key=lambda k: W[tail, k], default=None)
            if best_k is None: break
            chain.append(best_k); unassigned.remove(best_k)
        chains.append(chain)
    chains.sort(key=len, reverse=True)
    seqs      = [[] for _ in range(m)]
    C_hats    = [[] for _ in range(m)]
    committed = [0.0] * m; B_bar = 0.0
    for chain in chains:
        for j in chain:
            best_cost, best_rho, best_p, best_Cn = math.inf, 0, len(seqs[0]), []
            for rho in range(m):
                p = len(seqs[rho])          # append at tail — preserves chain order
                cost, Cn = evaluate_insertion(j, rho, p, seqs, C_hats, B_bar, inst, params)
                if cost < best_cost:
                    best_cost, best_rho, best_p, best_Cn = cost, rho, p, Cn
            if best_cost == math.inf:
                best_rho, best_p, best_Cn = min_violation_insert(j, seqs, inst)
            seqs[best_rho].insert(best_p, j); C_hats[best_rho] = best_Cn
            committed[best_rho] = C_hats[best_rho][-1] if C_hats[best_rho] else 0.0
            B_bar = sum(committed) / m
    return seqs, C_hats


def seed_grasp(inst: Instance, m: int, params: HeuristicParams,
               k: int = 3,
               rng_seed: int = 0) -> Tuple[List[List[int]], List[List[float]]]:
    """
    H9 — GRASP (Greedy Randomised Adaptive Search Procedure).

    At each step: score all unscheduled aircraft by CR_j × urgency_j; build a
    Restricted Candidate List (RCL) of the top k; pick one uniformly at random;
    insert at the cheapest feasible (runway, position).

    k=1 → pure greedy (equivalent to CAF without regret).
    Larger k → more diversification.  Calling with different rng_seed values
    gives structurally distinct seeds for the SA portfolio.

    Complexity: O(n²m) per seed.
    Reference: Feo & Resende (1995), Journal of Global Optimization 6(2): 109–133.
    """
    rng = random.Random(rng_seed); _, CR = compute_priorities(inst)
    seqs        = [[] for _ in range(m)]
    C_hats      = [[] for _ in range(m)]
    committed   = [0.0] * m; B_bar = 0.0
    unscheduled = list(range(inst.n))
    while unscheduled:
        tau    = min(committed)
        scores = sorted(
            ((float(CR[j]) / max(float(inst.delta[j]) - tau, inst.eps), j)
             for j in unscheduled), reverse=True)
        j = rng.choice([jj for _, jj in scores[:min(k, len(scores))]])
        best_cost, best_rho, best_p, best_Cn = math.inf, 0, 0, []
        for rho in range(m):
            for p in _candidate_positions(j, rho, seqs, inst):
                cost, Cn = evaluate_insertion(j, rho, p, seqs, C_hats, B_bar, inst, params)  # FIX 1: was missing closing )
                if cost < best_cost:                                                            # FIX 2: entire block was missing
                    best_cost, best_rho, best_p, best_Cn = cost, rho, p, Cn
        if best_cost == math.inf:
            best_rho, best_p, best_Cn = min_violation_insert(j, seqs, inst)
        seqs[best_rho].insert(best_p, j); C_hats[best_rho] = best_Cn
        committed[best_rho] = C_hats[best_rho][-1] if C_hats[best_rho] else 0.0
        B_bar = sum(committed) / m
        unscheduled.remove(j)
    return seqs, C_hats


# ═════════════════════════════════════════════════════════════════════════════
#   §25b  LP-SCREENED SEED PORTFOLIO BUILDER
# ═════════════════════════════════════════════════════════════════════════════

def _build_seed_portfolio(
    inst:     Instance,
    m:        int,
    params:   HeuristicParams,
    n_chains: int,
    seed:     int,
) -> Tuple[
    List[Tuple[str, List[List[int]]]],   # selected_starts
    List[Tuple[str, float, float, bool]],        # portfolio_info: (label, lp_val, raw_obj, selected)
    List[float],                          # selected_lps
    List[float],                          # selected_raw_objs
]:
    """
    Generate a portfolio of seed schedules from all construction heuristics,
    LP-evaluate each, and return the best n_chains seeds for SA initialisation.

    Heuristics evaluated
    --------------------
    FCFS    — arrival-order dispatching (H1)
    EDD     — target-time round-robin (H2)
    WEDD    — weighted EDD, freest-runway (H3)
    ATC     — apparent tardiness cost (H4)
    ATCS    — ATC with separation penalty (H5)
    TC-RBI  — target-conflict regret-based insertion
    CAF     — critical-aircraft-first insertion (H6)
    WCC     — wake-compatibility chain construction (H8)
    GRASP-3 — randomised greedy, RCL size 3 (H9)
    GRASP-7 — randomised greedy, RCL size 7 (H9)
    MPDS    — most-penalised-displacement-first (H7, only for n ≤ MPDS_MAX_N)

    LP screening
    ------------
    Every seed is LP-evaluated via stage2_lp_objective.  The top n_chains seeds
    by LP objective are selected with a diversity-preference pass: candidates
    whose runway-Hamming distance to every already-selected seed is ≥ ELITE_MIN_DIV
    are preferred.  When fewer than n_chains diverse seeds exist, the best
    remaining seeds are added regardless of diversity.

    Returns
    -------
    selected_starts : list of (label, seqs)         — n_chains seeds for SA chains.
    portfolio_info  : list of (label, lp_val, bool) — full portfolio for logging.
    selected_lps    : list of float                 — LP values of selected seeds.
    """
    k1, k2 = GRASP_K_VALUES

    # ── Generate all candidates ──────────────────────────────────────────────
    candidates: List[Tuple[str, List[List[int]]]] = [
        ("FCFS",     seed_fcfs(inst, m)[0]),
        ("EDD",      seed_edd_balanced(inst, m)[0]),
        ("WEDD",     seed_wedd(inst, m)[0]),
        ("ATC",      seed_atc(inst, m, K=ATC_K)[0]),
        ("ATCS",     seed_atcs(inst, m, K1=ATCS_K1, K2=ATCS_K2)[0]),
        ("TC-RBI",   ramp_rbi(inst, m, params)[0]),
        ("CAF",      seed_caf(inst, m, params)[0]),
        ("WCC",      seed_wcc(inst, m, params)[0]),
        (f"GRASP-{k1}", seed_grasp(inst, m, params, k=k1, rng_seed=seed)[0]),
        (f"GRASP-{k2}", seed_grasp(inst, m, params, k=k2, rng_seed=seed + 999)[0]),
    ]
    if inst.n <= MPDS_MAX_N:
        candidates.append(("MPDS", seed_mpds(inst, m, params)[0]))

    # ── LP-evaluate all candidates ───────────────────────────────────────────
    evaluated: List[Tuple[float, float, str, List[List[int]]]] = []
    for label, seqs in candidates:
        feas_e, _, raw_obj, _   = verify_and_exact_obj(seqs, inst)   
        raw_obj                 = raw_obj if feas_e else math.inf
        lp, _, lp_feas, _       = stage2_lp_objective(seqs, inst)        
        lp_val                  = lp if lp_feas else math.inf
        evaluated.append((lp_val, raw_obj, label, seqs))
    evaluated.sort(key=lambda x: x[0]) # sort by LP value (best first)

    # ── Diversity-aware selection ─────────────────────────────────────────────
    def _rwy_hamming(sa, sb):
        ma = len(sa)
        aa = {sa[r][p]: r for r in range(ma) for p in range(len(sa[r]))}
        return sum(1 for r in range(len(sb)) for j in sb[r] if aa.get(j) != r)

    if USE_ALL_SEEDS:
        # ── every evaluated seed becomes a SA starting point ─────────────────
        selected: List[Tuple[str, List[List[int]], float, float]] = [
            (label, seqs, lp_val, raw_obj)
            for lp_val, raw_obj, label, seqs in evaluated
        ]
    
    else:
        # ── diversity-aware top-N_CHAINS selection ────────────────────────────
        selected: List[Tuple[str, List[List[int]], float, float]] = []
        used_labels: set = set()

        # First pass: greedy by LP, prefer diverse candidates.
        for lp_val, raw_obj, label, seqs in evaluated:
            if len(selected) >= n_chains: break
            if label in used_labels: continue
            diverse = (not selected
                    or all(_rwy_hamming(seqs, s) >= ELITE_MIN_DIV
                            for _, s, _, _ in selected))
            if diverse:
                selected.append((label, seqs, lp_val, raw_obj))
                used_labels.add(label)

        # Second pass: fill remaining slots with best LP regardless of diversity.
        for lp_val, raw_obj, label, seqs in evaluated:
            if len(selected) >= n_chains: break
            if label in used_labels: continue
            selected.append((label, seqs, lp_val, raw_obj))
            used_labels.add(label)

    sel_labels        = {lbl for lbl, _, _, _ in selected}
    selected_starts   = [(lbl, s)   for lbl, s, _,  _   in selected]
    selected_lps      = [lp         for _,   _, lp, _   in selected]
    selected_raw_objs = [ro         for _,   _, _,  ro  in selected]
    portfolio_info    = [
        (label, raw_obj, lp_val, label in sel_labels)
        for lp_val, raw_obj, label, _ in evaluated
    ]

    return selected_starts, portfolio_info, selected_lps, selected_raw_objs


# ═════════════════════════════════════════════════════════════════════════════
#   §26  SPAWN-SAFE SA WORKER
# ═════════════════════════════════════════════════════════════════════════════

def _sa_worker(args: tuple) -> tuple:
    """
    Entry point for one SA chain in a worker process.

    Return tuple (11 elements):
        0  label
        1  bp_seqs      best proxy sequences
        2  b_proxy      best proxy value
        3  blp_seqs     best LP sequences
        4  b_lp         best LP value
        5  b_C_lp       LP solution vector or None
        6  history      per-iteration best_proxy list
        7  t_best_proxy wall time of proxy best
        8  t_best_lp    wall time of LP best  (time-to-best tracking)
        9  alpha_history per-iteration α list
        10 lp_timeline  list of (wall_time_s, lp_val) LP improvement events
    """
    label, init_seqs, init_lp, inst, params, p_sa, N_iter, seed, t_deadline = args
    bp_seqs, b_proxy, blp_seqs, b_lp, b_C_lp, st = run_mr_sa(
        init_seqs, init_lp, inst, params, p_sa, N_iter,
        label=label, seed=seed, t_deadline=t_deadline)
    return (label, bp_seqs, b_proxy, blp_seqs, b_lp, b_C_lp,
            st['history'], st['t_best_proxy'], st['t_best_lp'],
            st['alpha_history'], st['lp_timeline'])


# ═════════════════════════════════════════════════════════════════════════════
#   §27  PARALLEL MULTI-START SA  (main solver)
# ═════════════════════════════════════════════════════════════════════════════

def ms_mr_sa(inst, m, params, p_sa=None, n_chains=N_CHAINS,
             t_limit=T_LIMIT, seed=0):
    """
    Run K parallel SA chains; collect elite pool; apply path relinking and
    LP-VND polish; return the best overall solution.

    Seed portfolio
    --------------
    _build_seed_portfolio evaluates all construction heuristics, LP-screens them,
    and selects the best n_chains seeds with diversity preference.  The full
    portfolio (all heuristics + LP values) is stored in stats['seed_portfolio']
    for reporting and plotting.

    Time-to-best tracking
    ----------------------
    stats['t_best_lp'] is the wall-clock offset at which the best LP was found.
    stats['job_lp_timeline'] records every LP improvement event across seeds,
    SA chains, VND, and path relinking.

    Alternative schedules
    ----------------------
    The elite pool retains up to ELITE_POOL_MAX diverse LP-certified solutions
    exported to alternatives.csv.

    Returns
    -------
    best_seqs, best_lp, stats
    """
    p_sa   = p_sa or MRSAParams()
    N_iter = _n_iter(inst.n)
    t0     = time.perf_counter()
    t_dead = t0 + t_limit

    # ── Build and LP-screen seed portfolio ───────────────────────────────────
    starts, portfolio_info, seed_lps, seed_raw_objs = _build_seed_portfolio(
        inst, m, params, n_chains, seed)

    best_seed_lp  = min(seed_lps)      if seed_lps      else math.inf
    best_seed_raw = min(seed_raw_objs) if seed_raw_objs else math.inf

    print(f"  [{inst.name} m={m}] {len(portfolio_info)} seeds evaluated | "
          f"{n_chains} selected | N_iter={N_iter} | t_limit={t_limit:.0f}s")
    print(f"  SA params: {p_sa}")
    for label, raw_obj, lp_val, selected in sorted(portfolio_info, key=lambda x: x[2]):
        tag = " ← selected" if selected else ""
        raw_str = f"{raw_obj:.4f}" if not math.isinf(raw_obj) else "inf"
        lp_str  = f"{lp_val:.4f}" if not math.isinf(lp_val)  else "inf"
        print(f"    {label:<12} raw={raw_str}  LP={lp_str}{tag}")

    job_lp_timeline: List[Tuple[float, float]] = (
        [(0., best_seed_lp)] if not math.isinf(best_seed_lp) else [])

    tasks = [(lbl, s, seed_lps[i], inst, params, p_sa, N_iter, seed + i * 31, t_dead)
             for i, (lbl, s) in enumerate(starts)]

    n_sa_workers = min(len(tasks), N_WORKERS) if USE_ALL_SEEDS else min(n_chains, len(tasks))
    with ProcessPoolExecutor(max_workers=n_sa_workers,
                              mp_context=_MP_CTX) as ex:
        results = list(ex.map(_sa_worker, tasks))

    feas_rs = [r for r in results if not math.isinf(r[4])]
    if feas_rs:
        best_r    = min(feas_rs, key=lambda r: r[4])
        best_seqs = best_r[3]; best_lp = best_r[4]; best_C = best_r[5]
    else:
        warnings.warn(f"{inst.name} m={m}: no LP-feasible solution found.")
        best_r    = min(results, key=lambda r: r[2])
        best_seqs = best_r[1]; best_lp = math.inf; best_C = None

    for t_chain, lp_val in best_r[10]:
        job_lp_timeline.append((t_chain, lp_val))

    pool = ElitePool(ELITE_POOL_MAX, ELITE_MIN_DIV)
    for r in feas_rs:
        pool.try_add(r[3], r[4], r[5])

    final_lp, final_C, final_feas, final_viols = stage2_lp_objective(best_seqs, inst)
    if final_feas and final_lp < best_lp - 1e-9:
        best_lp = final_lp; best_C = final_C
        job_lp_timeline.append((time.perf_counter() - t0, final_lp))
    if final_feas and final_C is not None:
        pool.try_add(best_seqs, best_lp, final_C)

    if best_C is not None and not math.isinf(best_lp):
        vnd_lp_prev = best_lp
        best_seqs, best_lp = lp_vnd_polish(
            best_seqs, best_lp, best_C, inst, params, p_sa,
            max_rounds=_vnd_max_rounds(inst.n),
            t_limit=max(30., t_limit * 0.15))
        final_lp, final_C, final_feas, final_viols = stage2_lp_objective(best_seqs, inst)
        if final_feas and final_lp < best_lp - 1e-9: best_lp = final_lp; best_C = final_C
        if best_lp < vnd_lp_prev - 1e-9:
            job_lp_timeline.append((time.perf_counter() - t0, best_lp))
        if final_feas and final_C is not None:
            pool.try_add(best_seqs, best_lp, final_C)

    relink_improved = False
    pr_t_limit = max(20., t_limit * 0.10); pr_t0 = time.perf_counter()
    for pair_fn in [pool.best_quality_pair, pool.most_diverse_pair]:
        if time.perf_counter() - pr_t0 > pr_t_limit: break
        sol_a, sol_b = pair_fn()
        if sol_a is None: continue
        for a, b in [(sol_a, sol_b), (sol_b, sol_a)]:
            if time.perf_counter() - pr_t0 > pr_t_limit: break
            pr_seqs, pr_lp = path_relink(a, b, inst, params,
                                          max_steps=40, eval_interval=5, K_lp=8)
            if pr_lp < best_lp - 1e-9:
                best_seqs = pr_seqs; best_lp = pr_lp
                _, pr_C, pr_feas, _ = stage2_lp_objective(best_seqs, inst)
                if pr_feas:
                    best_C = pr_C
                    pool.try_add(best_seqs, best_lp, best_C)
                job_lp_timeline.append((time.perf_counter() - t0, best_lp))
                relink_improved = True

    final_lp, _, final_feas, final_viols = stage2_lp_objective(best_seqs, inst)
    if final_feas and final_lp < best_lp - 1e-9:
        best_lp = final_lp
        job_lp_timeline.append((time.perf_counter() - t0, final_lp))

    elite_solutions = [(s.lp_obj, [seq[:] for seq in s.seqs])
                       for s in sorted(pool.solutions, key=lambda s: s.lp_obj)]

    return best_seqs, best_lp, {
        'seed_lps':           seed_lps,
        'seed_raw_objs':      seed_raw_objs,      # <-- NEW
        'best_seed_raw':      best_seed_raw,       # <-- NEW
        'seed_portfolio':     portfolio_info,
        'all_results':        results,
        'wall':               time.perf_counter() - t0,
        't_best_lp':          best_r[8],
        'final_feas':         final_feas,
        'final_viols':        final_viols,
        'history':            best_r[6],
        'alpha_history':      best_r[9],
        'elite_pool_size':    len(pool.solutions),
        'relinking_improved': relink_improved,
        'elite_solutions':    elite_solutions,
        'job_lp_timeline':    job_lp_timeline,
    }


# ═════════════════════════════════════════════════════════════════════════════
#   §28  BKS-AWARE REPORTING UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def _gap_str(obj, ref, mark_new=True) -> str:
    if ref is None:  return "N/A"
    if ref == 0.:    return "0.00%" if obj < 1e-6 else "∞"
    gap = 100. * (obj - ref) / ref
    if gap < -0.001 and mark_new: return f"{gap:.2f}% ★"
    return f"{gap:.2f}%"


def _is_new_bks(obj, ref) -> bool:
    if ref is None or ref <= 0.: return False
    return obj < ref - 1e-6


def print_mr_result(inst, m, seqs, lp_obj, elapsed, seed_lps,
                    params, p_sa, stats=None):
    """Print a BKS-aware per-instance result report to stdout."""
    feas_e, viol_e, earliest_obj, _ = verify_and_exact_obj(seqs, inst)
    ref     = KNOWN_OPTIMA.get(inst.name, {}).get(m)
    new_bks = _is_new_bks(lp_obj, ref)
    sep     = "=" * 74
    print(f"\n{sep}")
    print(f"  {inst.name.upper()}  |  n={inst.n}  |  m={m} runway(s)"
          + ("  ★ NEW BKS CANDIDATE ★" if new_bks else ""))
    print(sep)
    print(f"  Runtime (SA+PR+VND total): {elapsed:.2f} s")
    print(f"  TC-RBI params            : {params}")
    print(f"  SA params                : {p_sa}")
    best_seed_raw = (stats.get('best_seed_raw', math.inf)
                     if stats else math.inf)
    best_seed_lp  = min(seed_lps) if seed_lps else math.inf
    # Three-stage improvement chain
    print(f"  Best seed raw obj        : "
          + (f"{best_seed_raw:.4f}" if not math.isinf(best_seed_raw) else "inf"))
    print(f"  Best seed LP obj         : "
          + (f"{best_seed_lp:.4f}"  if not math.isinf(best_seed_lp)  else "inf"))
    print(f"  SA+VND+PR final LP       : {lp_obj:.4f}")
    print(f"  Earliest-time objective  : {earliest_obj:.4f}")
    if ref is not None:
        label = "BKS (opt=0)" if ref == 0. else "Reference/BKS"
        print(f"  {label:<24} : {ref:.4f}")
        print(f"  Gap (seed raw → seed LP → final): "
              f"{_gap_str(best_seed_raw, ref, False)} → "
              f"{_gap_str(best_seed_lp, ref, False)} → "
              f"{_gap_str(lp_obj, ref)}")
    else:
        print(f"  Reference/BKS            : not available for m={m}")
    if stats:
        t_bl = stats.get('t_best_lp')
        print(f"  Time to best LP          : {t_bl:.2f} s"
              if isinstance(t_bl, float) else "  Time to best LP          : N/A")
        print(f"  Elite pool size          : {stats.get('elite_pool_size', 'N/A')}")
        print(f"  Path relinking improved  : {'Yes' if stats.get('relinking_improved') else 'No'}")
        portfolio = stats.get('seed_portfolio', [])
        if portfolio:
            n_sel = sum(1 for _, _, _, sel in portfolio if sel)
            print(f"  Seed portfolio           : {len(portfolio)} evaluated, {n_sel} selected")
    print(f"  Sequence feasibility     : {'PASS ✓' if feas_e else 'FAIL ✗'}")
    if not feas_e:
        for v in viol_e[:6]: print(f"    ✗ {v}")
        if len(viol_e) > 6: print(f"    ... and {len(viol_e)-6} more")
    print("  Runway load:")
    for rho, seq in enumerate(seqs):
        print(f"    Runway {rho+1}: {len(seq):4d} aircraft  "
              f"seq=[{', '.join(str(j) for j in seq[:6])}"
              f"{',...' if len(seq) > 6 else ''}]")
    print(sep)


def print_summary_table(results: List[dict]) -> None:
    """Print BKS-aware batch results table; flag new BKS candidates with ★."""
    col = ["Instance","n","m","Seed LP","Final LP","Reference",
           "Gap(seed)","Gap(SA)","BKS?","Feas","Time(s)"]
    w   = [12,5,4,12,12,12,10,10,5,6,9]
    hdr = "  " + "".join(f"{c:>{w[i]}}" for i,c in enumerate(col)); bar = "="*len(hdr)
    print(f"\n{bar}\n  MR-ALP Solver — BATCH RESULTS\n{bar}")
    print(hdr); print("-"*len(hdr))
    for r in sorted(results, key=lambda x: (x["name"], x["m"])):
        ref = r["opt"]
        row = [r["name"], r["n"], r["m"],
               f"{r['seed_lp']:.4f}" if not math.isinf(r['seed_lp']) else "inf",
               f"{r['sa_lp']:.4f}"   if not math.isinf(r['sa_lp'])   else "inf",
               f"{ref:.4f}" if ref is not None else "N/A",
               _gap_str(r['seed_lp'], ref, False), _gap_str(r['sa_lp'], ref),
               "★" if _is_new_bks(r['sa_lp'], ref) else "",
               "✓" if r["feasible"] else "✗",
               f"{r['time']:.2f}"]
        print("  " + "".join(f"{str(v):>{w[i]}}" for i,v in enumerate(row)))
    print(bar)
    pos = [r for r in results if r["opt"] is not None and r["opt"] > 0
           and not math.isinf(r["sa_lp"])]
    if pos:
        sg  = [100.*(r["seed_lp"]-r["opt"])/r["opt"] for r in pos
               if not math.isinf(r["seed_lp"])]
        ag  = [100.*(r["sa_lp"]-r["opt"])/r["opt"] for r in pos]
        fc  = sum(1 for r in results if r["feasible"])
        nb  = sum(1 for r in results if _is_new_bks(r["sa_lp"], r["opt"]))
        print(f"  Feasible         : {fc}/{len(results)}")
        print(f"  New BKS cands    : {nb}")
        if sg: print(f"  Avg seed gap     : {np.mean(sg):.2f}%  Max: {max(sg):.2f}%")
        if ag: print(f"  Avg final gap    : {np.mean(ag):.2f}%  Max: {max(ag):.2f}%")
        if sg and ag: print(f"  Avg improvement  : {np.mean(sg)-np.mean(ag):.2f}pp")
    print(bar)


# ═════════════════════════════════════════════════════════════════════════════
#   §29  RESULT PERSISTENCE
# ═════════════════════════════════════════════════════════════════════════════

def _ensure_dirs(output_dir: Path) -> None:
    """Create all plot subdirectories under output_dir."""
    for sub in ["gap", "convergence", "lp_timeline",
                "time_to_best", "elite_pool", "gantt", "seeds"]:
        (output_dir / "plots" / sub).mkdir(parents=True, exist_ok=True)


def _save_summary_csv(results, output_dir):
    """
    Write summary.csv.  Columns include best_seed_label and n_seeds_evaluated
    to capture the seed portfolio outcome for each job.
    """
    path   = output_dir / "summary.csv"
    fields = ["instance", "n", "m",
              "seed_raw_obj",           # <-- NEW: heuristic obj before LP
              "seed_lp", "sa_lp", "bks",
              "gap_seed_raw_pct",       # <-- NEW
              "gap_seed_lp_pct", "gap_sa_pct",
              "new_bks", "feasible", "time_s", "t_best_lp_s",
              "elite_pool_size", "relinking_improved",
              "best_seed_label", "n_seeds_evaluated"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in sorted(results, key=lambda x: (x["name"], x["m"])):
            ref     = r["opt"]
            raw_obj = r.get("best_seed_raw", math.inf)
            gr  = (100*(raw_obj - ref)/ref
                   if ref and ref > 0 and not math.isinf(raw_obj) else None)
            gs  = (100*(r["seed_lp"] - ref)/ref
                   if ref and ref > 0 and not math.isinf(r["seed_lp"]) else None)
            ga  = (100*(r["sa_lp"]   - ref)/ref
                   if ref and ref > 0 and not math.isinf(r["sa_lp"])   else None)
            portfolio       = r.get("seed_portfolio", [])
            n_seeds         = len(portfolio)
            sel_seeds       = [(lp, lbl) for lbl, _, lp, sel in portfolio
                               if sel and not math.isinf(lp)]
            best_seed_label = min(sel_seeds, key=lambda x: x[0])[1] if sel_seeds else ""
            w.writerow({
                "instance":         r["name"],
                "n":                r["n"],
                "m":                r["m"],
                "seed_raw_obj":     "" if math.isinf(raw_obj) else f"{raw_obj:.6f}",
                "seed_lp":          "" if math.isinf(r["seed_lp"]) else f"{r['seed_lp']:.6f}",
                "sa_lp":            "" if math.isinf(r["sa_lp"])   else f"{r['sa_lp']:.6f}",
                "bks":              "" if ref is None else ref,
                "gap_seed_raw_pct": "" if gr is None else f"{gr:.4f}",
                "gap_seed_lp_pct":  "" if gs is None else f"{gs:.4f}",
                "gap_sa_pct":       "" if ga is None else f"{ga:.4f}",
                "new_bks":          _is_new_bks(r["sa_lp"], ref),
                "feasible":         r["feasible"],
                "time_s":           f"{r['time']:.2f}",
                "t_best_lp_s":      f"{r.get('t_best_lp', 0):.2f}",
                "elite_pool_size":  r.get("elite_pool_size", ""),
                "relinking_improved": r.get("relinking_improved", False),
                "best_seed_label":  best_seed_label,
                "n_seeds_evaluated": n_seeds,
            })
    print(f"  Saved {path}")


def _save_schedules_csv(results, output_dir):
    """Write schedules.csv: best final sequences (long format, one row per aircraft)."""
    path = output_dir / "schedules.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["instance","m","rho","position","aircraft_j"])
        for r in sorted(results, key=lambda x: (x["name"], x["m"])):
            for rho, seq in enumerate(r.get("best_seqs", [])):
                for pos, j in enumerate(seq):
                    w.writerow([r["name"], r["m"], rho+1, pos+1, j])
    print(f"  Saved {path}")


def _save_alternatives_csv(results, output_dir):
    """Write alternatives.csv: elite pool alternative schedules (one row per aircraft)."""
    path = output_dir / "alternatives.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance","m","rank","lp_obj","rho","position","aircraft_j"])
        for r in sorted(results, key=lambda x: (x["name"], x["m"])):
            for rank, (lp_obj, seqs) in enumerate(r.get("elite_solutions", []), start=1):
                for rho, seq in enumerate(seqs):
                    for pos, j in enumerate(seq):
                        w.writerow([r["name"],r["m"],rank,f"{lp_obj:.6f}",rho+1,pos+1,j])
    print(f"  Saved {path}")


def _save_verification_txt(results, output_dir):
    """
    Write verification.txt: feasibility audit, LP timeline, and full seed
    portfolio log for every (instance, m) job.
    """
    path = output_dir / "verification.txt"; sep = "=" * 72
    with open(path, "w") as f:
        f.write("MR-ALP Solver — FEASIBILITY VERIFICATION REPORT\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for r in sorted(results, key=lambda x: (x["name"], x["m"])):
            ref = r["opt"]
            f.write(f"{sep}\n  {r['name'].upper()}  |  n={r['n']}  |  m={r['m']}\n{sep}\n")
            f.write(f"  SA+VND+PR LP         : {r['sa_lp']:.6f}\n")
            if ref is not None:
                f.write(f"  Reference/BKS        : {ref}\n")
                f.write(f"  Gap to BKS           : {_gap_str(r['sa_lp'], ref)}\n")
                if _is_new_bks(r["sa_lp"], ref):
                    f.write("  *** NEW BKS CANDIDATE ***\n")
            f.write(f"  Sequence feasible    : {'YES' if r['feasible'] else 'NO'}\n")
            f.write(f"  Time to best LP (s)  : {r.get('t_best_lp', 0):.2f}\n")
            f.write(f"  Total runtime (s)    : {r['time']:.2f}\n")
            viols = r.get("violations", [])
            if viols:
                f.write(f"  Violations ({len(viols)}):\n")
                for v in viols[:10]: f.write(f"    ✗ {v}\n")
                if len(viols) > 10: f.write(f"    ... and {len(viols)-10} more\n")
            # Seed portfolio log
            portfolio = r.get("seed_portfolio", [])
            if portfolio:
                f.write(f"  Seed portfolio ({len(portfolio)} heuristics evaluated):\n")
                for label, raw_obj, lp_val, selected in sorted(portfolio, key=lambda x: x[2]):
                    tag     = "  [SELECTED]" if selected else ""
                    raw_str = f"{raw_obj:.6f}" if not math.isinf(raw_obj) else "inf"
                    lp_str  = f"{lp_val:.6f}"  if not math.isinf(lp_val)  else "inf"
                    f.write(f"    {label:<12} raw={raw_str}  LP={lp_str}{tag}\n")
            # LP improvement timeline
            tl = r.get("job_lp_timeline", [])
            if tl:
                f.write("  LP improvement timeline (time_s, lp_val):\n")
                for t_s, lp_v in tl:
                    f.write(f"    t={t_s:8.2f}s  LP={lp_v:.6f}\n")
            f.write("\n")
    print(f"  Saved {path}")


def _save_run_metadata_json(results, output_dir):
    """
    Write run_metadata.json: config, SA parameters, timing, pool stats,
    and the full seed portfolio for every (instance, m) job.
    """
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
        gap = (100*(r["sa_lp"]-ref)/ref
               if ref and ref > 0 and not math.isinf(r["sa_lp"]) else None)
        p   = r.get("p_sa", MRSAParams())
        portfolio = r.get("seed_portfolio", [])
        payload["results"].append({
            "instance":           r["name"], "n": r["n"], "m": r["m"],
            "seed_lp":            None if math.isinf(r["seed_lp"]) else r["seed_lp"],
            "sa_lp":              None if math.isinf(r["sa_lp"])   else r["sa_lp"],
            "bks":                ref,
            "gap_pct":            round(gap, 4) if gap is not None else None,
            "new_bks":            _is_new_bks(r["sa_lp"], ref),
            "feasible":           r["feasible"],
            "time_s":             round(r["time"], 2),
            "t_best_lp_s":        round(r.get("t_best_lp", 0), 2),
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


def save_run_results(results, output_dir: Path) -> None:
    """Write all result files; create subdirectories if needed."""
    _ensure_dirs(output_dir)
    _save_summary_csv(results, output_dir)
    _save_schedules_csv(results, output_dir)
    _save_alternatives_csv(results, output_dir)
    _save_verification_txt(results, output_dir)
    _save_run_metadata_json(results, output_dir)


# ═════════════════════════════════════════════════════════════════════════════
#   §30  VISUALISATION
#
#   Plot subdirectory layout under OUTPUT_DIR/plots/
#   ─────────────────────────────────────────────────
#   gap/         gap_summary.png
#   convergence/ convergence_{inst}_{m}.png
#   lp_timeline/ lp_timeline_{inst}_{m}.png
#   time_to_best/time_to_best.png
#   elite_pool/  elite_pool_{inst}_{m}.png
#   gantt/       gantt_{inst}_{m}.png
#   seeds/       seeds_{inst}_{m}.png   ← seed portfolio LP comparison
# ═════════════════════════════════════════════════════════════════════════════

_PLOT_STYLE = {
    "figure.facecolor": "white", "axes.facecolor": "#f7f7f7",
    "axes.grid": True, "grid.color": "white", "grid.linewidth": 0.8,
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
}


def _plot_gap_summary(results, output_dir: Path) -> None:
    """Grouped bar chart: seed gap vs final SA+VND+PR gap."""
    if not _MPL: return
    pos_r = [r for r in results
             if r["opt"] is not None and r["opt"] > 0 and not math.isinf(r["sa_lp"])]
    if not pos_r: return
    pos_r     = sorted(pos_r, key=lambda x: (x["name"], x["m"]))
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


def _plot_convergence(result, output_dir: Path) -> None:
    """SA proxy convergence history for the best chain; y-axis normalised to start."""
    if not _MPL: return
    history = result.get("history", [])
    if not history: return
    name = result["name"]; m = result["m"]
    hist = np.asarray(history, dtype=float)
    if hist[0] != 0: hist = hist / hist[0]
    fig, ax = plt.subplots(figsize=(7, 4))
    with plt.rc_context(_PLOT_STYLE):
        ax.plot(hist, linewidth=0.8, color="#4878CF", alpha=0.9)
        ax.set_xlabel("Iteration"); ax.set_ylabel("Best proxy (relative to start)")
        ax.set_title(f"{name.upper()} m={m} — SA proxy convergence (best chain)")
        plt.tight_layout()
        out = output_dir / "plots" / "convergence" / f"convergence_{name}_{m}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {out}")


def _plot_lp_timeline(result, output_dir: Path) -> None:
    """
    Step-function LP objective vs wall time.  Vertical dashed line at
    time-to-best-LP; horizontal dashed line at BKS reference.
    """
    if not _MPL: return
    tl = result.get("job_lp_timeline", [])
    if len(tl) < 2: return
    name = result["name"]; m = result["m"]; ref = result.get("opt")
    ts  = [t for t,_ in tl] + [result["time"]]
    lps = [v for _,v in tl] + [[v for _,v in tl][-1]]
    fig, ax = plt.subplots(figsize=(7, 4))
    with plt.rc_context(_PLOT_STYLE):
        ax.step(ts, lps, where="post", linewidth=1.5, color="#D65F5F", label="LP objective")
        ax.scatter([t for t,_ in result.get("job_lp_timeline",[])],
                   [v for _,v in result.get("job_lp_timeline",[])],
                   s=30, color="#D65F5F", zorder=5)
        if ref is not None and ref > 0:
            ax.axhline(ref, color="goldenrod", linewidth=1.2, linestyle="--",
                       label=f"BKS reference ({ref:.2f})")
        t_best = result.get("t_best_lp")
        if t_best and t_best > 0:
            ax.axvline(t_best, color="steelblue", linewidth=0.9, linestyle=":",
                       label=f"t-to-best ({t_best:.1f}s)")
        ax.set_xlabel("Wall time (s)"); ax.set_ylabel("LP objective")
        ax.set_title(f"{name.upper()} m={m} — LP improvement timeline")
        ax.legend(fontsize=9); plt.tight_layout()
        out = output_dir / "plots" / "lp_timeline" / f"lp_timeline_{name}_{m}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {out}")


def _plot_time_to_best(results, output_dir: Path) -> None:
    """Scatter: time-to-best-LP vs final BKS gap; colour encodes runway count."""
    if not _MPL: return
    pos = [r for r in results
           if r["opt"] is not None and r["opt"] > 0
           and not math.isinf(r["sa_lp"]) and r.get("t_best_lp") is not None]
    if len(pos) < 2: return
    ts   = [r["t_best_lp"] for r in pos]
    gaps = [100*(r["sa_lp"]-r["opt"])/r["opt"] for r in pos]
    ms   = [r["m"] for r in pos]; m_vals = sorted(set(ms))
    cmap = plt.get_cmap("tab10")
    colours = {mv: cmap(i/max(len(m_vals)-1, 1)) for i,mv in enumerate(m_vals)}
    fig, ax = plt.subplots(figsize=(7, 5))
    with plt.rc_context(_PLOT_STYLE):
        for mv in m_vals:
            idx = [i for i,r in enumerate(pos) if r["m"] == mv]
            ax.scatter([ts[i] for i in idx], [gaps[i] for i in idx],
                       s=60, color=colours[mv], label=f"m={mv}", alpha=0.85, edgecolors="white")
        for r, t, g in zip(pos, ts, gaps):
            if abs(g) > 5:
                ax.annotate(f"{r['name']}\nm={r['m']}", (t,g), fontsize=7,
                            ha="left", va="bottom", xytext=(4,4), textcoords="offset points")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Time to best LP (s)"); ax.set_ylabel("Final gap to BKS (%)")
        ax.set_title("MR-ALP — Time to best LP vs BKS gap")
        ax.legend(title="Runways", fontsize=9); plt.tight_layout()
        out = output_dir / "plots" / "time_to_best" / "time_to_best.png"
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {out}")


def _plot_elite_pool(result, output_dir: Path) -> None:
    """Horizontal bar chart of LP objectives for the elite pool solutions."""
    if not _MPL: return
    elite = result.get("elite_solutions", [])
    if len(elite) < 2: return
    name = result["name"]; m = result["m"]; ref = result.get("opt")
    lp_vals = [lp for lp,_ in elite[:20]]; ranks = list(range(1, len(lp_vals)+1))
    fig, ax = plt.subplots(figsize=(6, max(3, len(lp_vals)*0.28)))
    with plt.rc_context(_PLOT_STYLE):
        colours = ["#D65F5F" if (ref and ref > 0 and _is_new_bks(lp, ref))
                   else "#4878CF" for lp in lp_vals]
        bars = ax.barh(ranks, lp_vals, color=colours, alpha=0.85, edgecolor="white")
        if ref is not None and ref > 0:
            ax.axvline(ref, color="goldenrod", linewidth=1.2, linestyle="--",
                       label=f"BKS ({ref:.2f})"); ax.legend(fontsize=9)
        ax.set_yticks(ranks); ax.set_yticklabels([f"Rank {r}" for r in ranks], fontsize=8)
        ax.invert_yaxis(); ax.set_xlabel("LP objective")
        ax.set_title(f"{name.upper()} m={m} — Elite pool LP distribution")
        for bar, lp in zip(bars, lp_vals):
            ax.text(bar.get_width()*1.002, bar.get_y()+bar.get_height()/2,
                    f"{lp:.2f}", va="center", fontsize=7.5)
        plt.tight_layout()
        out = output_dir / "plots" / "elite_pool" / f"elite_pool_{name}_{m}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {out}")


def _plot_gantt(result, output_dir: Path) -> None:
    """
    Gantt chart of the final LP-optimal landing schedule.

    Three layers per aircraft:
      1. Grey background span   [r_j, d_j]   — time window.
      2. Coloured landing bar   [C_j, C_j+sep_width] — blue=early, green=on-time, red=late.
      3. Black target tick at δ_j.

    Aircraft index labels shown when n ≤ 80.
    Saved to: plots/gantt/gantt_{inst}_{m}.png
    """
    if not _MPL: return
    C_lp      = result.get("C_lp")
    seqs      = result.get("best_seqs", [])
    r_arr     = result.get("r_arr")
    delta_arr = result.get("delta_arr")
    d_arr     = result.get("d_arr")
    s_mat     = result.get("s_mat")
    name = result["name"]; m = result["m"]
    if C_lp is None or not seqs or r_arr is None: return
    n_rwy    = len(seqs)
    annotate = result.get("n", 0) <= 80
    fig, ax  = plt.subplots(figsize=(14, max(3.0, n_rwy*1.8+1.2)))
    with plt.rc_context(_PLOT_STYLE):
        for rho, seq in enumerate(seqs):
            if not seq: continue
            L = len(seq)
            for qi, j in enumerate(seq):
                cj  = float(C_lp[j]); rj = float(r_arr[j])
                dj  = float(d_arr[j]); dej = float(delta_arr[j])
                ax.barh(rho, dj-rj, left=rj, height=0.65,
                        color="#cccccc", alpha=0.30, linewidth=0, zorder=1)
                min_bw = max((dj-rj)*0.04, 5.0)
                bw = max(float(s_mat[j, seq[qi+1]]), min_bw) if qi < L-1 else min_bw
                if   cj < dej - 1.0: color = "#4878CF"
                elif cj > dej + 1.0: color = "#D65F5F"
                else:                color = "#6ACC65"
                ax.barh(rho, bw, left=cj, height=0.50, color=color,
                        alpha=0.90, linewidth=0.5, edgecolor="white", zorder=3)
                ax.plot(dej, rho, marker="|", color="black",
                        markersize=9, markeredgewidth=1.5, zorder=5)
                if annotate:
                    ax.text(cj+bw*0.5, rho, str(j), ha="center", va="center",
                            fontsize=6.5, color="white", fontweight="bold", zorder=6)
        ax.set_yticks(range(n_rwy))
        ax.set_yticklabels([f"Runway {rho+1}  (n={len(seqs[rho])})" for rho in range(n_rwy)],
                           fontsize=9)
        ax.set_xlabel("Time")
        ax.set_title(f"{name.upper()}  |  m={m} runway(s) — LP-optimal landing schedule")
        ax.set_ylim(-0.7, n_rwy-0.3)
        from matplotlib.patches import Patch
        from matplotlib.lines   import Line2D
        legend_elements = [
            Patch(facecolor="#cccccc", alpha=0.50, label="Time window [r_j, d_j]"),
            Patch(facecolor="#4878CF",              label="Early  (C_j < δ_j)"),
            Patch(facecolor="#6ACC65",              label="On-time"),
            Patch(facecolor="#D65F5F",              label="Late   (C_j > δ_j)"),
            Line2D([0],[0], marker="|", color="black", linewidth=0,
                   markersize=10, markeredgewidth=1.5, label="Target δ_j"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8, framealpha=0.85)
        plt.tight_layout()
        out = output_dir / "plots" / "gantt" / f"gantt_{name}_{m}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {out}")


def _plot_seed_comparison(result, output_dir: Path) -> None:
    """
    Horizontal bar chart comparing both raw objective and LP objective across
    all evaluated seed heuristics.

    Selected seeds (forwarded to SA chains) are highlighted in dark red;
    non-selected seeds are shown in grey.  A vertical dashed line marks the
    BKS reference when available.  Bars are sorted by LP value ascending so
    the best seed appears at the top.

    Saved to: plots/seeds/seeds_{inst}_{m}.png
    """
    if not _MPL: return
    portfolio = result.get("seed_portfolio", [])
    if not portfolio: return
    name = result["name"]; m = result["m"]; ref = result.get("opt")

    # portfolio_info tuple: (label, raw_obj, lp_val, selected)
    finite = sorted(
        [(lp_val, raw_obj, label, sel)
         for label, raw_obj, lp_val, sel in portfolio
         if not math.isinf(lp_val)],
        key=lambda x: x[0]
    )
    inf_entries = [
        (label, raw_obj, sel)
        for label, raw_obj, lp_val, sel in portfolio
        if math.isinf(lp_val)
    ]
    if not finite and not inf_entries: return

    labels   = [label   for _, _, label, _   in finite] + [lbl         for lbl, _, _   in inf_entries]
    lp_vals  = [lp_val  for lp_val, _, _, _  in finite] + [None]       * len(inf_entries)
    raw_vals = [raw_obj for _, raw_obj, _, _  in finite] + [raw_obj     for _, raw_obj, _ in inf_entries]
    selected = [sel     for _, _, _, sel      in finite] + [sel         for _, _, sel   in inf_entries]

    n_items = len(labels)
    fig, axes = plt.subplots(1, 2, figsize=(14, max(3.0, n_items * 0.45 + 1.5)),
                             sharey=True)

    with plt.rc_context(_PLOT_STYLE):
        colours = ["#D65F5F" if sel else "#aaaaaa" for sel in selected]
        y_pos   = list(range(n_items))

        for ax, vals, xlabel, title_suffix in [
            (axes[0], raw_vals, "Raw objective (earliest-time)", "Raw obj"),
            (axes[1], lp_vals,  "LP objective",                  "LP obj"),
        ]:
            bar_vals = [v if v is not None else 0.0 for v in vals]
            bars = ax.barh(y_pos, bar_vals, color=colours, alpha=0.88, edgecolor="white")

            if ref is not None and ref > 0:
                ax.axvline(ref, color="goldenrod", linewidth=1.3, linestyle="--",
                           label=f"BKS ({ref:.2f})")

            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel(xlabel)
            ax.set_title(f"{name.upper()} m={m} — Seed {title_suffix}")

            for bar, v in zip(bars, vals):
                if v is not None:
                    ax.text(bar.get_width() * 1.002, bar.get_y() + bar.get_height() / 2,
                            f"{v:.2f}", va="center", fontsize=8)
                else:
                    ax.text(0.01, bar.get_y() + bar.get_height() / 2,
                            "infeasible", va="center", fontsize=8, color="#888888")

        from matplotlib.patches import Patch
        from matplotlib.lines   import Line2D
        handles = [
            Patch(facecolor="#D65F5F", alpha=0.88, label="Selected for SA"),
            Patch(facecolor="#aaaaaa", alpha=0.88, label="Not selected"),
        ]
        if ref is not None and ref > 0:
            handles.append(Line2D([0], [0], color="goldenrod", linewidth=1.3,
                                  linestyle="--", label=f"BKS ({ref:.2f})"))
        axes[1].legend(handles=handles, fontsize=8, loc="lower right")

        plt.suptitle(f"{name.upper()} m={m} — Seed portfolio comparison",
                     fontsize=11, y=1.01)
        plt.tight_layout()
        out = output_dir / "plots" / "seeds" / f"seeds_{name}_{m}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {out}")


def generate_plots(results, output_dir: Path) -> None:
    """
    Generate all plots for a completed batch run.

    Global plots (one file):
      plots/gap/gap_summary.png
      plots/time_to_best/time_to_best.png

    Per-job plots (one file per (instance, m)):
      plots/convergence/convergence_{inst}_{m}.png
      plots/lp_timeline/lp_timeline_{inst}_{m}.png
      plots/elite_pool/elite_pool_{inst}_{m}.png
      plots/gantt/gantt_{inst}_{m}.png
      plots/seeds/seeds_{inst}_{m}.png
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


# ═════════════════════════════════════════════════════════════════════════════
#   §31  JOB ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def _run_one_job(fp: str, m: int, seed: int = 0) -> dict:
    """
    Execute one (instance, runway-count) job and return a complete result dict.

    Workflow
    --------
    1. Parse instance file.
    2. Resolve TC-RBI params (RBI_PARAM_BANK → Optuna → defaults).
    3. Resolve SA params (SA_PARAM_BANK → Optuna → defaults).
    4. Evaluate base LP for adaptive time budget.
    5. Run ms_mr_sa (parallel SA + VND + path relinking).
    6. Verify feasibility; collect C_lp and instance arrays for Gantt chart.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        inst = load_instance(fp)

        # TC-RBI param resolution
        params = RBI_PARAM_BANK.get((inst.name, m))
        if params is None:
            if RUN_RBI_OPTUNA:
                n_t = _n_rbi_trials(inst.n, N_RBI_TRIALS_BASE)
                print(f"\n  [RBI Optuna] {inst.name.upper()} m={m} → {n_t} trials ...")
                t_rbi  = time.perf_counter()
                params = optimize_rbi_params(inst, m, n_t, RBI_OPTUNA_SEED,
                                              n_jobs=N_OPTUNA_WORKERS)
                print(f"  [RBI Optuna] done in {time.perf_counter()-t_rbi:.1f}s  best: {params}")
            else:
                print(f"  [WARN] ({inst.name}, m={m}) not in RBI_PARAM_BANK — using defaults.")
                params = _DEFAULT_RBI

        # SA param resolution
        p_sa_tuned = False; p_sa = SA_PARAM_BANK.get((inst.name, m))
        if p_sa is None and RUN_SA_OPTUNA:
            n_t  = _sa_n_trials(inst.n, SA_N_TRIALS_BASE)
            print(f"\n  [SA Optuna] {inst.name.upper()} m={m} → {n_t} trials ...")
            t_opt = time.perf_counter()
            p_sa  = optimize_sa_params(inst, m, params, n_trials=n_t,
                                        seed=SA_OPTUNA_SEED, n_jobs=SA_N_OPTUNA_JOBS)
            print(f"  [SA Optuna] done in {time.perf_counter()-t_opt:.1f}s  best: {p_sa}")
            p_sa_tuned = True
        if p_sa is None: p_sa = MRSAParams()

        # Adaptive time budget
        base_seqs, _ = ramp_rbi(inst, m, params)
        base_lp, _, base_feas, _ = stage2_lp_objective(base_seqs, inst)
        seed_lp_est  = base_lp if base_feas else math.inf
        bks          = KNOWN_OPTIMA.get(inst.name, {}).get(m)
        job_t_limit  = _adaptive_t_limit(inst.n, m, seed_lp_est, bks)
        print(f"  Adaptive T_LIMIT: {job_t_limit:.0f}s  "
              f"(seed_LP={seed_lp_est:.2f}  BKS={bks})")

        t0 = time.perf_counter()
        best_seqs, best_lp, stats = ms_mr_sa(
            inst, m, params, p_sa=p_sa,
            n_chains=N_CHAINS, t_limit=job_t_limit, seed=seed)
        elapsed  = time.perf_counter() - t0
        seed_lp  = min(stats['seed_lps']) if stats['seed_lps'] else math.inf
        feasible = stats['final_feas']
        feas_e, viol_e, _, _ = verify_and_exact_obj(best_seqs, inst)
        print_mr_result(inst, m, best_seqs, best_lp, elapsed,
                        stats['seed_lps'], params, p_sa, stats)

    opt = KNOWN_OPTIMA.get(inst.name, {}).get(m)
    # Retrieve final C_lp for Gantt chart (no stdout, safe outside redirect).
    _, C_lp_final, _, _ = stage2_lp_objective(best_seqs, inst)

    return dict(
        name=inst.name, n=inst.n, m=m,
        seed_lp=seed_lp, sa_lp=best_lp,
        best_seed_raw=stats.get("best_seed_raw", math.inf),
        opt=opt, feasible=feasible, time=elapsed,
        t_best_lp=stats.get("t_best_lp", 0.),
        p_sa=p_sa, p_sa_tuned=p_sa_tuned,
        best_seqs=best_seqs,
        elite_solutions=stats.get("elite_solutions", []),
        job_lp_timeline=stats.get("job_lp_timeline", []),
        elite_pool_size=stats.get("elite_pool_size", 0),
        relinking_improved=stats.get("relinking_improved", False),
        seed_portfolio=stats.get("seed_portfolio", []),
        history=stats.get("history", []),
        alpha_history=stats.get("alpha_history", []),
        violations=viol_e,
        # Arrays for Gantt chart (§30)
        C_lp=C_lp_final,
        r_arr=inst.r.copy(),
        delta_arr=inst.delta.copy(),
        d_arr=inst.d.copy(),
        s_mat=inst.s.copy(),
        output=buf.getvalue(),
    )


# ═════════════════════════════════════════════════════════════════════════════
#   §32  MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Entry point for the MR-ALP solver.

    In BATCH_MODE, discovers all airland*.txt files in FOLDER and submits one
    job per (instance, runway-count) pair to N_WORKERS workers.

    In single-file mode, runs all configured runway counts for INSTANCE_PATH.

    Writes all output files to OUTPUT_DIR when SAVE_RESULTS=True.
    Generates all plots in OUTPUT_DIR/plots/ subdirectories when SAVE_PLOTS=True.
    """
    print("="*74)
    print("  MR-ALP Solver — TC-RBI + Parallel SA + VND + Path Relinking")
    print(f"  Workers       : {N_WORKERS} processes | {N_CHAINS} chains/job")
    print(f"  T_LIMIT       : {T_LIMIT:.0f}s (adaptive, max {MAX_T_LIMIT:.0f}s)")
    print(f"  Elite pool    : max {ELITE_POOL_MAX} solutions, min diversity {ELITE_MIN_DIV}")
    print(f"  Seed heuristics: FCFS, EDD, WEDD, ATC(K={ATC_K}), "
          f"ATCS(K1={ATCS_K1},K2={ATCS_K2}), TC-RBI, CAF, WCC, "
          f"GRASP{list(GRASP_K_VALUES)}"
          + (f", MPDS(n≤{MPDS_MAX_N})" if MPDS_MAX_N > 0 else ""))
    print(f"  Output dir    : {OUTPUT_DIR}")
    print(f"  Save results  : {SAVE_RESULTS}  |  Save plots: {SAVE_PLOTS}")
    nb_str  = "Numba JIT"   if _NUMBA    else "no Numba"
    gpu_str = "PyTorch GPU" if _GPU_AVAIL else "no GPU"
    mpl_str = "matplotlib"  if _MPL      else "no matplotlib"
    rbi_opt = (f"RBI Optuna ON ({N_RBI_TRIALS_BASE} base trials)"
               if RUN_RBI_OPTUNA else "RBI Optuna OFF")
    sa_opt  = (f"SA Optuna ON ({SA_N_TRIALS_BASE} base trials)"
               if RUN_SA_OPTUNA  else "SA Optuna OFF")
    print(f"  Accel         : {nb_str}, {gpu_str}, {mpl_str}")
    print(f"  RBI tuning    : {rbi_opt}")
    print(f"  SA tuning     : {sa_opt}")
    print("="*74)

    if BATCH_MODE:
        folder = Path(FOLDER); 
        files = sorted(
            folder.glob("airland*.txt"),
            key=lambda p: int(''.join(filter(str.isdigit, p.stem)) or 0)
            )
        if not files:
            print(f"No airland*.txt files found in {folder.resolve()}"); return
        jobs = [(str(fp), m) for fp in files
                for m in INSTANCE_RUNWAYS.get(fp.stem.lower(), [1])]
        print(f"  Submitting {len(jobs)} jobs to {N_WORKERS} workers...\n")
        results = []
        with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=_MP_CTX) as ex:
            futs = {ex.submit(_run_one_job, fp, m): (fp, m) for fp, m in jobs}
            for fut in as_completed(futs):
                fp, m = futs[fut]
                try:
                    r = fut.result(); results.append(r); print(r["output"], end="")
                    bks_tag  = " ★NEW BKS★" if _is_new_bks(r["sa_lp"], r["opt"]) else ""
                    tune_tag = " [tuned]"   if r.get("p_sa_tuned") else ""
                    portfolio = r.get("seed_portfolio", [])
                    best_seed_lbl = ""
                    if portfolio:
                        sel_seeds = [(lp, lbl) for lbl, _, lp, sel in portfolio   # was: lbl, lp, sel
                                        if sel and not math.isinf(lp)]
                        if sel_seeds: 
                            best_seed_lbl = f"  best_seed={min(sel_seeds,key=lambda x:x[0])[1]}"

                    print(f"  ↳ {Path(fp).stem:<12} m={m}  "
                          f"seed={r['seed_lp']:.2f}  SA={r['sa_lp']:.2f}  "
                          f"gap={_gap_str(r['sa_lp'],r['opt'])}  "
                          f"{'✓' if r['feasible'] else '✗'}  "
                          f"t_best={r.get('t_best_lp',0):.1f}s  "
                          f"({r['time']:.1f}s){tune_tag}{bks_tag}{best_seed_lbl}")
                except Exception as exc:
                    print(f"  ERROR {Path(fp).stem} m={m}: {exc}")
        print_summary_table(results)
        if SAVE_RESULTS: save_run_results(results, OUTPUT_DIR)
        if SAVE_PLOTS:   generate_plots(results, OUTPUT_DIR)

    else:
        fp  = Path(INSTANCE_PATH); cfg = INSTANCE_RUNWAYS.get(fp.stem.lower(), [1])
        res = []
        for m in cfg:
            r = _run_one_job(str(fp), m); print(r["output"], end="")
            bks_tag = " ★NEW BKS★" if _is_new_bks(r["sa_lp"], r["opt"]) else ""
            print(f"  ↳ {fp.stem:<12} m={m}  seed={r['seed_lp']:.2f}  "
                  f"SA={r['sa_lp']:.2f}  gap={_gap_str(r['sa_lp'],r['opt'])}  "
                  f"{'✓' if r['feasible'] else '✗'}  "
                  f"t_best={r.get('t_best_lp',0):.1f}s  ({r['time']:.1f}s){bks_tag}")
            res.append(r)
        if len(cfg) > 1: print_summary_table(res)
        if SAVE_RESULTS: save_run_results(res, OUTPUT_DIR)
        if SAVE_PLOTS:   generate_plots(res, OUTPUT_DIR)


if __name__ == "__main__":
    main()

