"""
mr_sa.py — Multi-Runway Aircraft Landing Problem: SA Refinement (v3)
===============================================================================
PIPELINE
--------
    TC-RBI  →  Stage-2 LP  →  LP-guided seed repairs
            →  Sample-and-select SA  (K=4 controlled chains)
            →  LP-guided VND polish  (+ ejection chain neighbourhood)
            →  Elite pool collection
            →  Path relinking between elite solutions
            →  Final Stage-2 LP + verification
            →  Result persistence + visualisation

OUTPUT FILES  (written to OUTPUT_DIR/)
---------------------------------------
  summary.csv        — one row per (instance, m): objectives, gaps, timing.
  schedules.csv      — final landing sequences (long format, one row per aircraft).
  alternatives.csv   — elite pool alternative schedules (diverse high-quality solutions).
  verification.txt   — per-(instance, m) feasibility audit.
  run_metadata.json  — run configuration, SA parameters, wall-clock timings.
  plots/
    gap_summary.png            — grouped bar chart: seed gap vs final gap.
    convergence_{inst}_{m}.png — proxy history for the best SA chain.
    lp_timeline_{inst}_{m}.png — LP objective vs wall time for the best chain.
    time_to_best.png           — scatter: time-to-best-LP vs final BKS gap.
    elite_pool_{inst}_{m}.png  — LP distribution of elite pool solutions.

CHANGES FROM v2
---------------
1. BKS-aware reporting (§17):
       "Known optimum" renamed to "Reference/BKS".  Negative gaps (new BKS
       candidates) flagged with ★.

2. LP-slack impact scoring (§5):
       Aircraft selection combines LP penalty P_j with binding-separation
       count to identify aircraft in tight chains — the highest-leverage
       relocation targets.

3. Ejection chain operator XE (§6, §10, §13):
       Depth-2 cross-runway reassignment: j₁:ρ₁→ρ₂ then j₂:ρ₂→ρ₃.

4. Elite solution pool (§10b):
       Up to ELITE_POOL_MAX LP-certified schedules with runway-Hamming
       diversity guard.  Saved as alternative schedules in alternatives.csv.

5. Path relinking (§10c):
       Post-SA walk from one elite solution toward another, LP-evaluating
       intermediate states.

6. Adaptive time budget (§0, §20):
       Time allocation scales with the gap to BKS reference.

7. LP timeline tracking (§12, §16):
       Every LP improvement is timestamped.  The full (time, lp_obj) sequence
       is exported per job and used for lp_timeline plots.

8. Result persistence (§18):
       summary.csv, schedules.csv, alternatives.csv, verification.txt,
       run_metadata.json.

9. Visualisation (§19):
       Convergence, LP timeline, gap summary, time-to-best, elite pool.

REFERENCES
----------
Beasley, J.E., Krishnamoorthy, M., Sharaiha, Y.M., Abramson, D. (2000).
    Scheduling aircraft landings — the static case.
    Transportation Science 34(2), 180–197.
Glover, F. (1997). Tabu search and adaptive memory programming.
    In Advances in metaheuristics, optimization, and stochastic modeling.
"""
from __future__ import annotations

import csv, io, contextlib, json, math, random, time, platform, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Optional accelerators ────────────────────────────────────────────────────
try:
    import torch as _torch
    _GPU_AVAIL = _torch.cuda.is_available()
except ImportError:
    _torch = None; _GPU_AVAIL = False

try:
    import numba as nb
    _NUMBA = True
except ImportError:
    _NUMBA = False

try:
    import optuna as _optuna
    _optuna.logging.set_verbosity(_optuna.logging.WARNING)
    _OPTUNA = True
except ImportError:
    _optuna = None; _OPTUNA = False

try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend for server/HPC use
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    _MPL = True
except ImportError:
    _MPL = False

import multiprocessing as _mp
_MP_CTX = _mp.get_context("spawn" if (platform.system() == "Windows"
                                       or _GPU_AVAIL) else "fork")

from ramp_rbi import (
    Instance, HeuristicParams,
    load_instance, ramp_rbi,
    stage2_lp_objective, verify_and_exact_obj,
    surrogate_times,
    KNOWN_OPTIMA, INSTANCE_RUNWAYS,
)


# ═════════════════════════════════════════════════════════════════════════════
#   §0  CONFIGURE HERE
#
#   T_LIMIT       — Wall-clock budget per job (fallback when BKS is unavailable).
#   MAX_T_LIMIT   — Hard ceiling for adaptive time allocation.
#   ELITE_POOL_MAX — Maximum elite pool size.
#   ELITE_MIN_DIV — Minimum runway-Hamming distance for diversity admission.
#   OUTPUT_DIR    — Root directory for all saved files and plots.
#   SAVE_RESULTS  — Write CSV / JSON / TXT output files.
#   SAVE_PLOTS    — Generate and save matplotlib figures.
# ═════════════════════════════════════════════════════════════════════════════
BATCH_MODE    = True
INSTANCE_PATH = "data/airland1.txt"
FOLDER        = "data/"

N_WORKERS   = 7
N_CHAINS    = 4
T_LIMIT     = 300.0
MAX_T_LIMIT = 1200.0

RUN_SA_OPTUNA    = False
SA_N_TRIALS_BASE = 20
SA_OPTUNA_SEED   = 123
SA_N_OPTUNA_JOBS = 4

ELITE_POOL_MAX = 20
ELITE_MIN_DIV  = 5

OUTPUT_DIR   = Path("MR results")
SAVE_RESULTS = True
SAVE_PLOTS   = True


def _n_iter(n: int) -> int:
    """SA iteration budget: 2 000 / 5 000 / 8 000 for n ≤ 50 / ≤ 250 / > 250."""
    if n <= 50:  return 2_000
    if n <= 250: return 5_000
    return 8_000


def _R_candidates(n: int) -> int:
    """Sample-and-select pool size: 10 / 20 / 30 for n ≤ 100 / ≤ 250 / > 250."""
    if n <= 100: return 10
    if n <= 250: return 20
    return 30


def _lp_repair_params(n: int) -> Tuple[int, int]:
    """
    Return (q_lp, K) for LP-guided repair operators.

    q_lp — top-penalty aircraft to consider as relocation candidates.
    K    — LP evaluation budget (number of proxy-sorted candidates to LP-check).
    Both shrink with n to keep LP overhead bounded.
    """
    if n <= 50:   return 20, 20
    if n <= 100:  return 15, 15
    if n <= 250:  return 10, 10
    return 8, 5


def _vnd_max_rounds(n: int) -> int:
    """VND iteration cap: 15 / 10 / 5 for n ≤ 100 / ≤ 250 / > 250."""
    if n <= 100: return 15
    if n <= 250: return 10
    return 5


def _adaptive_t_limit(n: int, m: int, seed_lp: float, bks: Optional[float]) -> float:
    """
    Compute a per-job wall-clock budget based on the gap to the BKS reference.

    Zero-optimal instances (BKS=0) terminate as soon as F_LP=0 is confirmed
    and therefore receive only a short verification budget.  Larger positive
    gaps receive proportionally more time up to MAX_T_LIMIT.

    Gap → budget mapping
    --------------------
    BKS = 0 or seed at BKS  → 60 s
    gap ≤ 2%                 → 120 s
    gap ≤ 5%                 → T_LIMIT (300 s default)
    gap ≤ 10%                → min(600, MAX_T_LIMIT)
    gap > 10%                → MAX_T_LIMIT
    BKS unknown              → T_LIMIT

    Parameters
    ----------
    n : int            — number of aircraft (reserved for future size scaling).
    m : int            — number of runways.
    seed_lp : float    — best LP across seed sequences, before SA.
    bks : float or None — BKS reference from KNOWN_OPTIMA.

    Returns
    -------
    float  — time budget in seconds, capped at MAX_T_LIMIT.
    """
    if bks is None:         return T_LIMIT
    if bks == 0.0:          return 60.0
    if math.isinf(seed_lp): return MAX_T_LIMIT
    gap = 100.0 * (seed_lp - bks) / bks
    if gap <= 0.0:  return 60.0
    if gap <= 2.0:  return 120.0
    if gap <= 5.0:  return T_LIMIT
    if gap <= 10.0: return min(600.0, MAX_T_LIMIT)
    return MAX_T_LIMIT


# ═════════════════════════════════════════════════════════════════════════════
#   §1  PARAM BANK  (Optuna-tuned TC-RBI weights from ramp_rbi.py batch run)
# ═════════════════════════════════════════════════════════════════════════════
def _P(eta, mu_tc, mu_late, mu_count, mu_sep) -> HeuristicParams:
    """Convenience constructor for HeuristicParams with positional arguments."""
    return HeuristicParams(eta=eta, mu_tc=mu_tc,
                           mu_late=mu_late, mu_count=mu_count, mu_sep=mu_sep)

_DEFAULT = HeuristicParams()

PARAM_BANK: Dict[Tuple[str, int], HeuristicParams] = {
    ("airland1",  2): _P(0.571, 2.324, 1.594, 1.422, 0.376),
    ("airland1",  3): _P(0.445, 4.778, 0.060, 0.730, 0.092),
    ("airland2",  2): _P(0.724, 0.724, 0.681, 1.042, 0.188),
    ("airland2",  3): _P(0.268, 4.604, 1.425, 1.065, 0.189),
    ("airland3",  2): _P(0.684, 3.613, 1.552, 1.698, 0.334),
    ("airland3",  3): _P(0.709, 4.770, 0.715, 2.866, 0.329),
    ("airland4",  2): _P(0.511, 3.612, 0.246, 1.818, 0.357),
    ("airland4",  3): _P(0.408, 1.272, 0.302, 2.535, 0.489),
    ("airland4",  4): _P(0.443, 2.683, 0.492, 0.557, 0.186),
    ("airland5",  2): _P(0.607, 0.251, 0.606, 0.585, 0.258),
    ("airland5",  3): _P(0.225, 3.786, 1.156, 0.648, 0.193),
    ("airland5",  4): _P(0.794, 2.427, 0.268, 0.468, 0.207),
    ("airland6",  2): _P(0.524, 0.676, 1.895, 2.555, 0.262),
    ("airland6",  3): _P(0.488, 4.551, 1.441, 1.892, 0.386),
    ("airland7",  2): _P(0.229, 0.441, 1.142, 0.715, 0.370),
    ("airland8",  2): _P(0.732, 4.111, 1.358, 2.423, 0.334),
    ("airland8",  3): _P(0.490, 3.755, 0.210, 1.108, 0.256),
    ("airland9",  2): _P(0.653, 3.170, 1.700, 0.822, 0.219),
    ("airland9",  3): _P(0.372, 0.568, 1.416, 1.895, 0.001),
    ("airland9",  4): _P(0.305, 4.302, 1.976, 2.914, 0.430),
    ("airland10", 2): _P(0.424, 4.730, 1.509, 2.373, 0.044),
    ("airland10", 3): _P(0.773, 2.732, 0.876, 0.405, 0.083),
    ("airland10", 4): _P(0.794, 1.716, 0.667, 1.703, 0.433),
    ("airland10", 5): _P(0.564, 3.269, 0.365, 2.447, 0.162),
    ("airland11", 2): _P(0.530, 2.408, 1.362, 1.895, 0.002),
    ("airland11", 3): _P(0.733, 2.754, 1.607, 0.305, 0.241),
    ("airland11", 4): _P(0.314, 4.472, 1.543, 2.700, 0.146),
    ("airland11", 5): _P(0.404, 0.584, 1.830, 1.315, 0.020),
    ("airland12", 2): _P(0.536, 2.500, 1.882, 2.811, 0.186),
    ("airland12", 3): _P(0.415, 4.624, 0.243, 2.590, 0.284),
    ("airland12", 4): _P(0.254, 2.792, 1.160, 2.117, 0.127),
    ("airland12", 5): _P(0.235, 1.592, 1.206, 2.837, 0.105),
    ("airland13", 2): _P(0.486, 2.788, 1.551, 0.862, 0.214),
    ("airland13", 3): _P(0.349, 3.411, 0.029, 0.810, 0.245),
    ("airland13", 4): _P(0.339, 2.534, 1.394, 2.935, 0.435),
    ("airland13", 5): _P(0.701, 1.596, 0.492, 0.772, 0.339),
}


# ═════════════════════════════════════════════════════════════════════════════
#   §1b  SA PARAM BANK  (pre-tuned MRSAParams; fill from RUN_SA_OPTUNA runs)
# ═════════════════════════════════════════════════════════════════════════════
def _SA(chi0, M_stag_frac, beta, lp_gamma, chi_target) -> "MRSAParams":
    """Convenience constructor for MRSAParams with positional arguments."""
    return MRSAParams(chi0=chi0, M_stag_frac=M_stag_frac, beta=beta,
                      lp_gamma=lp_gamma, chi_target=chi_target)

SA_PARAM_BANK: Dict[Tuple[str, int], "MRSAParams"] = {}


# ═════════════════════════════════════════════════════════════════════════════
#   §2  SA PARAMETERS
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class MRSAParams:
    """
    Simulated annealing control parameters for the multi-runway SA refinement.

    Tunable via Optuna TPE (§8b)
    ----------------------------
    chi0 : float in [0.50, 0.95]
        Target initial acceptance probability for worsening moves.
        _calibrate_t0 solves: T0 = −mean(Δ+) / ln(chi0).

    M_stag_frac : float in [0.05, 0.30]
        Stagnation threshold as a fraction of N_iter:
        M_stag = int(M_stag_frac × N_iter).

    beta : float in [1.20, 2.50]
        Reheat multiplier label for Optuna (applied as t_reheat).

    lp_gamma : float in [0.01, 0.20]
        LP trigger sensitivity γ.  Fires when:
            proxy_new < (1 − γ) × best_proxy_lp_checked
        Both sides are on the same proxy scale (v2 bug fixed in v3).

    chi_target : float in [0.10, 0.35]
        Reactive cooling target acceptance rate χ*.

    Fixed structural parameters
    ---------------------------
    ejection_chain_depth : int
        Depth of the ejection chain operator (1 or 2).  Capped at 1 when
        m < 3 since a third runway is needed for depth-2.
    lp_repair_interval : int
        Iterations between periodic lp_guided_penalty_repair calls.
        Set to 0 to disable (e.g. during Optuna tuning).
    near_zero_threshold : float
        LP objective below which target_conflict_repair is triggered.
    lambda_binding : float
        Weight on binding-separation count in _lp_impact_scores:
        Impact_j = P_j + lambda_binding · binding_count_j.
    eps_tight : float
        Slack threshold for classifying a separation constraint as binding.
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

    def __str__(self) -> str:
        return (f"χ₀={self.chi0:.3f}  M_stag={self.M_stag_frac:.3f}  "
                f"β={self.beta:.3f}  γ={self.lp_gamma:.4f}  "
                f"χ*={self.chi_target:.3f}")


# ═════════════════════════════════════════════════════════════════════════════
#   §3  NUMBA JIT: full-pairwise feasibility
#
#   OR Library separation matrices violate the triangle inequality, so
#   consecutive-only checking is incorrect.  _rwy_feasible_nb propagates all
#   predecessor separations and returns False on the first d-violation.
# ═════════════════════════════════════════════════════════════════════════════
if _NUMBA:
    @nb.njit(cache=True)
    def _rwy_feasible_nb(seq: np.ndarray, r: np.ndarray,
                         s: np.ndarray, d: np.ndarray) -> bool:
        """
        Numba-compiled full pairwise feasibility check.

        C[0] = r[seq[0]];
        C[q] = max(r[seq[q]], max_{h<q}(C[h] + s[seq[h], seq[q]])).
        Returns False as soon as C[q] > d[seq[q]] + ε.
        """
        L = len(seq)
        if L == 0: return True
        C = np.empty(L, dtype=np.float64)
        C[0] = r[seq[0]]
        if C[0] > d[seq[0]] + 1e-9: return False
        for q in range(1, L):
            C[q] = r[seq[q]]
            for h in range(q):
                lb = C[h] + s[seq[h], seq[q]]
                if lb > C[q]: C[q] = lb
            if C[q] > d[seq[q]] + 1e-9: return False
        return True


def _runway_feasible(seq: List[int], inst: Instance) -> bool:
    """
    Check whether a single runway sequence is fully feasible.

    Dispatches to Numba JIT when available, otherwise uses a pure-NumPy
    fallback.  Both paths enforce all pairwise separations and time windows.

    Parameters
    ----------
    seq : list of int
    inst : Instance

    Returns
    -------
    bool
    """
    if not seq: return True
    if _NUMBA:
        return bool(_rwy_feasible_nb(
            np.asarray(seq, dtype=np.int32), inst.r, inst.s, inst.d))
    L = len(seq); C = np.empty(L)
    C[0] = inst.r[seq[0]]
    if C[0] > inst.d[seq[0]] + 1e-9: return False
    for q in range(1, L):
        C[q] = inst.r[seq[q]]
        for h in range(q):
            lb = C[h] + inst.s[seq[h], seq[q]]
            if lb > C[q]: C[q] = lb
        if C[q] > inst.d[seq[q]] + 1e-9: return False
    return True


# ═════════════════════════════════════════════════════════════════════════════
#   §4  PROXY COMPUTATION
#
#   F_hat = μ_TC·ΣTC + μ_late·ΣLBT + μ_count·Balance + μ_sep·ΣSep
#
#   Per-runway arrays tc_rwy / lbt_rwy / sep_rwy are maintained
#   incrementally: only affected runway elements are recomputed per move.
# ═════════════════════════════════════════════════════════════════════════════

def _rwy_proxy_components(
    seq: List[int], inst: Instance
) -> Tuple[float, float, float]:
    """
    Compute (TC, LBT, Sep) for a single runway.

    TC  = Σ_{i<j} 0.5·(p_i+p_j)·max(s[i,j]−(δ_j−δ_i),0)
    LBT = Σ_j h_j·max(Ĉ_j−δ_j,0)      (consecutive-predecessor surrogate)
    Sep = h_bar · Σ_{q} s[seq[q],seq[q+1]]
    """
    if not seq: return 0.0, 0.0, 0.0
    L = len(seq); s_arr = np.asarray(seq, dtype=np.intp)
    if L >= 2:
        ii, jj = np.triu_indices(L, k=1)
        i_ac = s_arr[ii]; j_ac = s_arr[jj]
        v  = inst.s[i_ac, j_ac] - (inst.delta[j_ac] - inst.delta[i_ac])
        tc = float((0.5*(inst.p_arr[i_ac]+inst.p_arr[j_ac])*np.maximum(v,0.0)).sum())
    else:
        tc = 0.0
    C_hat = np.asarray(surrogate_times(seq, inst))
    lbt   = float((inst.h[s_arr]*np.maximum(C_hat-inst.delta[s_arr],0.0)).sum())
    sep   = float(inst.s[s_arr[:-1],s_arr[1:]].sum())*inst.h_bar if L>=2 else 0.0
    return tc, lbt, sep


def _balance_term(seqs: List[List[int]], inst: Instance) -> float:
    """Scaled squared runway-load deviation: Σ(|seq_ρ|−n/m)²·Pen_bar/(n/m)²."""
    n = inst.n; m = len(seqs)
    return (sum((len(seqs[r])-n/m)**2 for r in range(m))
            * float(inst.Pen_bar) / max((n/m)**2, 1.0))


def compute_proxy(
    seqs: List[List[int]],
    tc_rwy: np.ndarray, lbt_rwy: np.ndarray, sep_rwy: np.ndarray,
    inst: Instance, params: HeuristicParams,
) -> float:
    """Assemble global F_hat from per-runway arrays and balance term."""
    return (params.mu_tc*float(tc_rwy.sum()) + params.mu_late*float(lbt_rwy.sum())
          + params.mu_count*_balance_term(seqs,inst) + params.mu_sep*float(sep_rwy.sum()))


def _init_proxy_arrays(
    seqs: List[List[int]], inst: Instance
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full recompute of tc_rwy, lbt_rwy, sep_rwy from current sequences."""
    m = len(seqs)
    tc_rwy = np.zeros(m); lbt_rwy = np.zeros(m); sep_rwy = np.zeros(m)
    for rho in range(m):
        tc_rwy[rho], lbt_rwy[rho], sep_rwy[rho] = _rwy_proxy_components(seqs[rho], inst)
    return tc_rwy, lbt_rwy, sep_rwy


# ═════════════════════════════════════════════════════════════════════════════
#   §5  PER-AIRCRAFT SCORES  (proxy TC/LBT + LP-slack impact)
#
#   v3 adds _lp_impact_scores: LP penalty P_j augmented by binding-separation
#   count.  Aircraft in tight chains propagate delays and are high-leverage
#   relocation targets beyond what pure penalty scoring captures.
# ═════════════════════════════════════════════════════════════════════════════

def _compute_per_aircraft_scores(
    seqs: List[List[int]], inst: Instance
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute proxy-based per-aircraft TC and LBT contributions.

    pa_tc[j]  — Σ pairwise TC contributions touching aircraft j.
    pa_lbt[j] — h_j · max(Ĉ_j − δ_j, 0).

    O(n²) per call; refreshed every _n_full(t, N) iterations.
    """
    n = inst.n; pa_tc = np.zeros(n); pa_lbt = np.zeros(n)
    for seq in seqs:
        if not seq: continue
        L = len(seq); s_arr = np.asarray(seq, dtype=np.intp)
        if L >= 2:
            ii, jj = np.triu_indices(L, k=1)
            i_ac = s_arr[ii]; j_ac = s_arr[jj]
            v = inst.s[i_ac, j_ac] - (inst.delta[j_ac] - inst.delta[i_ac])
            c = 0.5*(inst.p_arr[i_ac]+inst.p_arr[j_ac])*np.maximum(v, 0.0)
            np.add.at(pa_tc, i_ac, c); np.add.at(pa_tc, j_ac, c)
        C_hat = np.asarray(surrogate_times(seq, inst))
        pa_lbt[s_arr] = inst.h[s_arr]*np.maximum(C_hat-inst.delta[s_arr], 0.0)
    return pa_tc, pa_lbt


def _lp_impact_scores(
    seqs:      List[List[int]],
    C_lp:      np.ndarray,
    inst:      Instance,
    lambda_b:  float = 0.5,
    eps_tight: float = 1e-4,
) -> np.ndarray:
    """
    Compute LP-slack impact score: Impact_j = P_j + λ_b · binding_count_j.

    P_j           = g_j·max(δ_j−C_lp[j],0) + h_j·max(C_lp[j]−δ_j,0)
    binding_count_j = number of ordered pairs (i≺j) or (j≺k) on any runway
                      with separation slack C_lp[j]−C_lp[i]−s[i,j] ≤ eps_tight.

    Aircraft in tight chains propagate timing errors forward.  The composite
    score identifies both directly penalised aircraft and those that block
    others, which pure penalty scoring (v2) misses.

    Parameters
    ----------
    seqs : list of list of int
    C_lp : ndarray, shape (n,)
    inst : Instance
    lambda_b : float   — weight on binding count.
    eps_tight : float  — slack threshold for binding classification.

    Returns
    -------
    ndarray, shape (n,)  — Impact_j ≥ 0.
    """
    n = inst.n
    E = np.maximum(inst.delta - C_lp, 0.0)
    T = np.maximum(C_lp - inst.delta, 0.0)
    P = inst.g*E + inst.h*T
    binding = np.zeros(n)
    for seq in seqs:
        L = len(seq)
        for qi in range(L):
            for qj in range(qi+1, L):
                i, j  = seq[qi], seq[qj]
                slack = C_lp[j] - C_lp[i] - inst.s[i, j]
                if slack <= eps_tight:
                    binding[i] += 1.0; binding[j] += 1.0
    return P + lambda_b * binding


def _pick_aircraft_targeted(
    seqs:   List[List[int]], inst: Instance, rng: random.Random,
    pa_tc:  Optional[np.ndarray] = None,
    pa_lbt: Optional[np.ndarray] = None,
    impact: Optional[np.ndarray] = None,
) -> Tuple[int, int]:
    """
    Select (runway, position) using the best available score array.

    Priority: impact (LP-derived) > pa_tc (proxy) > uniform random.

    Distribution:
      60% — uniform random.
      25% — top-20% by impact[j] (or pa_tc[j] if no LP yet).
      15% — top-20% by pa_lbt[j].
    """
    m    = len(seqs)
    flat = [(rho, pos) for rho in range(m) for pos in range(len(seqs[rho]))]
    if not flat: return 0, 0
    r = rng.random()
    scores = impact if impact is not None else pa_tc
    if r < 0.60 or scores is None:
        return rng.choice(flat)
    if r < 0.85:
        scored = sorted(((scores[seqs[rho][pos]], rho, pos) for rho, pos in flat),
                        key=lambda x: -x[0])
        top = max(1, len(scored)//5)
        _, rho, pos = rng.choice(scored[:top]); return rho, pos
    lbt_arr = pa_lbt if pa_lbt is not None else scores
    scored  = sorted(((lbt_arr[seqs[rho][pos]], rho, pos) for rho, pos in flat),
                     key=lambda x: -x[0])
    top = max(1, len(scored)//5)
    _, rho, pos = rng.choice(scored[:top]); return rho, pos


# ═════════════════════════════════════════════════════════════════════════════
#   §6  NEIGHBOURHOOD OPERATORS
#
#   All operators share the same contract:
#     - Return None if structurally invalid or infeasible.
#     - Return _MoveResult(full_new_seqs, affected_rwy_indices).
#     - _MoveResult.affected drives incremental proxy array updates.
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class _MoveResult:
    """Neighbourhood operator result: new sequences + modified runway indices."""
    seqs:     List[List[int]]
    affected: List[int]


def _op_n1_adjacent_swap(seqs, rho, p, inst):
    """N1 — swap positions p and p+1 on runway rho."""
    seq = seqs[rho]
    if p >= len(seq)-1: return None
    ns = seq[:]; ns[p], ns[p+1] = ns[p+1], ns[p]
    if not _runway_feasible(ns, inst): return None
    r = [s[:] for s in seqs]; r[rho] = ns; return _MoveResult(r, [rho])


def _op_n2_swap(seqs, rho, p, q, inst):
    """N2 — swap any two positions p, q on runway rho."""
    seq = seqs[rho]
    if p == q or p >= len(seq) or q >= len(seq): return None
    ns = seq[:]; ns[p], ns[q] = ns[q], ns[p]
    if not _runway_feasible(ns, inst): return None
    r = [s[:] for s in seqs]; r[rho] = ns; return _MoveResult(r, [rho])


def _op_n3b_best_insertion(seqs, rho, p, inst, params):
    """
    N3b — remove aircraft at position p; reinsert at the position minimising
    μ_TC·TC + μ_late·LBT + μ_sep·Sep on the same runway.

    Replaces the random-destination N3 with an exhaustive one-step lookahead.
    """
    seq = seqs[rho]; L = len(seq)
    if L < 2: return None
    ac = seq[p]; sm = seq[:p] + seq[p+1:]
    if not _runway_feasible(sm, inst): return None
    best_score, best_q = math.inf, -1
    for q in range(L):
        ns = sm[:q] + [ac] + sm[q:]
        if not _runway_feasible(ns, inst): continue
        tc, lbt, sep = _rwy_proxy_components(ns, inst)
        s = params.mu_tc*tc + params.mu_late*lbt + params.mu_sep*sep
        if s < best_score: best_score = s; best_q = q
    if best_q == -1: return None
    ns = sm[:best_q] + [ac] + sm[best_q:]
    r = [s[:] for s in seqs]; r[rho] = ns; return _MoveResult(r, [rho])


def _op_n4_block_reloc(seqs, rho, p, b, q, inst):
    """N4 — move a contiguous block of b aircraft from position p to q in the remainder."""
    seq = seqs[rho]
    if p+b > len(seq) or b < 1: return None
    blk = seq[p:p+b]; rest = seq[:p]+seq[p+b:]
    ns = rest[:q%(len(rest)+1)] + blk + rest[q%(len(rest)+1):]
    if not _runway_feasible(ns, inst): return None
    r = [s[:] for s in seqs]; r[rho] = ns; return _MoveResult(r, [rho])


def _op_x1_transfer(seqs, rho_a, p, rho_b, q, inst):
    """X1 — transfer aircraft at p on rho_a to position q on rho_b."""
    if rho_a == rho_b: return None
    sa = seqs[rho_a][:]; sb = seqs[rho_b][:]
    ac = sa.pop(p); sb.insert(min(q, len(sb)), ac)
    if not _runway_feasible(sa, inst) or not _runway_feasible(sb, inst): return None
    r = [s[:] for s in seqs]; r[rho_a]=sa; r[rho_b]=sb; return _MoveResult(r,[rho_a,rho_b])


def _op_x2_swap(seqs, rho_a, p, rho_b, q, inst):
    """X2 — exchange aircraft at p on rho_a with aircraft at q on rho_b."""
    if rho_a==rho_b or not seqs[rho_a] or not seqs[rho_b]: return None
    if p >= len(seqs[rho_a]) or q >= len(seqs[rho_b]): return None
    sa = seqs[rho_a][:]; sb = seqs[rho_b][:]
    sa[p], sb[q] = sb[q], sa[p]
    if not _runway_feasible(sa, inst) or not _runway_feasible(sb, inst): return None
    r = [s[:] for s in seqs]; r[rho_a]=sa; r[rho_b]=sb; return _MoveResult(r,[rho_a,rho_b])


def _op_x3_best_transfer(seqs, rho_a, p, rho_b, inst, params,
                          tc_rwy, lbt_rwy, sep_rwy):
    """
    X3 — transfer aircraft from rho_a to the best position on rho_b.

    Exhaustively tests all L_b+1 insertion positions and selects the one
    minimising the incremental composite proxy cost Δ(TC+LBT+Balance+Sep).
    O(L_b) per call.
    """
    if rho_a == rho_b: return None
    sa = seqs[rho_a][:]; ac = sa.pop(p)
    if not _runway_feasible(sa, inst): return None
    best_delta, best_sb = math.inf, None
    n = inst.n; m = len(seqs)
    bs = float(inst.Pen_bar)/max((n/m)**2, 1.0)
    t  = sum(len(s) for s in seqs)
    ob = (len(seqs[rho_a])-t/m)**2 + (len(seqs[rho_b])-t/m)**2
    for q in range(len(seqs[rho_b])+1):
        sb = seqs[rho_b][:]; sb.insert(q, ac)
        if not _runway_feasible(sb, inst): continue
        ta,la,ea = _rwy_proxy_components(sa, inst)
        tb,lb,eb = _rwy_proxy_components(sb, inst)
        nb = (len(sa)-t/m)**2 + (len(sb)-t/m)**2
        delta = (params.mu_tc*((ta+tb)-(tc_rwy[rho_a]+tc_rwy[rho_b]))
               + params.mu_late*((la+lb)-(lbt_rwy[rho_a]+lbt_rwy[rho_b]))
               + params.mu_count*(nb-ob)*bs
               + params.mu_sep*((ea+eb)-(sep_rwy[rho_a]+sep_rwy[rho_b])))
        if delta < best_delta: best_delta=delta; best_sb=sb
    if best_sb is None: return None
    r = [s[:] for s in seqs]; r[rho_a]=sa; r[rho_b]=best_sb
    return _MoveResult(r, [rho_a, rho_b])


def _op_x4_block_transfer(seqs, rho_a, p, b, rho_b, q, inst):
    """X4 — transfer a contiguous block of b aircraft from rho_a to rho_b."""
    if rho_a==rho_b or p+b > len(seqs[rho_a]): return None
    blk = seqs[rho_a][p:p+b]
    sa  = seqs[rho_a][:p]+seqs[rho_a][p+b:]
    sb  = seqs[rho_b][:]; sb[q:q] = blk
    if not _runway_feasible(sa,inst) or not _runway_feasible(sb,inst): return None
    r = [s[:] for s in seqs]; r[rho_a]=sa; r[rho_b]=sb
    return _MoveResult(r, [rho_a, rho_b])


def _op_x7_tc_repair(seqs, tc_rwy, lbt_rwy, inst, params, rng, impact):
    """
    X7 — TC-targeted repair: select top-impact aircraft, attempt X3 transfer;
    fall back to N3b if X3 fails or m=1.
    """
    m = len(seqs)
    cands = ([(impact[seqs[rho][pos]], rho, pos) for rho in range(m) for pos in range(len(seqs[rho]))]
             if impact is not None
             else [(tc_rwy[rho]/max(len(seqs[rho]),1), rho, pos)
                   for rho in range(m) for pos in range(len(seqs[rho]))])
    if not cands: return None
    cands.sort(key=lambda x: -x[0])
    _, rho_a, p = rng.choice(cands[:max(1,len(cands)//5)])
    others = [r for r in range(m) if r != rho_a]
    if others:
        res = _op_x3_best_transfer(seqs, rho_a, p, rng.choice(others),
                                    inst, params, tc_rwy, lbt_rwy, np.zeros(m))
        if res is not None: return res
    return _op_n3b_best_insertion(seqs, rho_a, p, inst, params) if len(seqs[rho_a])>=2 else None


# ═════════════════════════════════════════════════════════════════════════════
#   §7  PHASE-DEPENDENT OPERATOR SELECTION
#
#   f = t/N_iter controls the diversification→intensification transition:
#     f < 0.30  (early)   — X1/X2/X3/X4 account for 56%.
#     0.30–0.75 (mid)     — balanced; adds X7 and XE.
#     f ≥ 0.75  (late)    — N1/N2/N3b account for 63%; X2/X3/X7/XE for rest.
#     m = 1               — within-runway operators only.
# ═════════════════════════════════════════════════════════════════════════════
_OPS_EARLY  = [("X1",0.18),("X2",0.18),("X3",0.18),("X4",0.10),
               ("N2",0.15),("N3b",0.12),("N1",0.09)]
_OPS_MID    = [("X1",0.12),("X2",0.12),("X3",0.12),("X7",0.14),
               ("N2",0.14),("N3b",0.18),("N1",0.10),("XE",0.08)]
_OPS_LATE   = [("N1",0.18),("N2",0.17),("N3b",0.23),
               ("X2",0.14),("X3",0.10),("X7",0.10),("XE",0.08)]
_OPS_SINGLE = [("N1",0.25),("N2",0.28),("N3b",0.30),("N4",0.17)]


def _select_op(f: float, m: int, rng: random.Random) -> str:
    """Sample an operator code from the phase-appropriate probability table."""
    if m==1:     table = _OPS_SINGLE
    elif f<0.30: table = _OPS_EARLY
    elif f<0.75: table = _OPS_MID
    else:        table = _OPS_LATE
    ops, weights = zip(*table)
    return rng.choices(ops, weights=weights, k=1)[0]


def _apply_op(op, seqs, tc_rwy, lbt_rwy, sep_rwy,
              inst, params, p_sa, rng, stag, N_iter,
              pa_tc=None, pa_lbt=None, impact=None, C_lp=None):
    """
    Dispatch to the named operator.  Uses impact scores for biased aircraft
    selection when available.  XE delegates to X3 best-transfer (the cheap
    in-loop version of the ejection chain).
    """
    m = len(seqs)
    rho_a, pos_a = _pick_aircraft_targeted(seqs, inst, rng, pa_tc, pa_lbt, impact)
    L_a = len(seqs[rho_a])
    if op=="N1":
        return _op_n1_adjacent_swap(seqs, rho_a, rng.randint(0,max(L_a-2,0)), inst)
    elif op=="N2":
        if L_a<2: return None
        return _op_n2_swap(seqs, rho_a, rng.randint(0,L_a-1), rng.randint(0,L_a-1), inst)
    elif op=="N3b":
        if L_a<2: return None
        return _op_n3b_best_insertion(seqs, rho_a, rng.randint(0,L_a-1), inst, params)
    elif op=="N4":
        if L_a<2: return None
        b_cap = p_sa.B_stag if stag >= int(p_sa.M_stag_frac*N_iter) else p_sa.B_max
        b = rng.randint(1, min(b_cap, L_a))
        return _op_n4_block_reloc(seqs, rho_a, rng.randint(0,L_a-b), b, rng.randint(0,L_a-b), inst)
    elif op=="X1":
        if m<2: return None
        rho_b = rng.choice([r for r in range(m) if r!=rho_a])
        return _op_x1_transfer(seqs, rho_a, pos_a, rho_b, rng.randint(0,len(seqs[rho_b])), inst)
    elif op=="X2":
        if m<2 or not seqs[rho_a]: return None
        rho_b = rng.choice([r for r in range(m) if r!=rho_a])
        if not seqs[rho_b]: return None
        return _op_x2_swap(seqs, rho_a, pos_a, rho_b, rng.randint(0,len(seqs[rho_b])-1), inst)
    elif op=="X3":
        if m<2: return None
        rho_b = rng.choice([r for r in range(m) if r!=rho_a])
        return _op_x3_best_transfer(seqs, rho_a, pos_a, rho_b, inst, params,
                                     tc_rwy, lbt_rwy, sep_rwy)
    elif op=="X4":
        if m<2 or L_a<1: return None
        b = rng.randint(1, min(p_sa.B_max, L_a))
        rho_b = rng.choice([r for r in range(m) if r!=rho_a])
        return _op_x4_block_transfer(seqs, rho_a, rng.randint(0,L_a-b), b, rho_b,
                                      rng.randint(0,len(seqs[rho_b])), inst)
    elif op=="X7":
        return _op_x7_tc_repair(seqs, tc_rwy, lbt_rwy, inst, params, rng, impact)
    elif op=="XE":
        if m<2: return None
        rho_b = rng.choice([r for r in range(m) if r!=rho_a])
        return _op_x3_best_transfer(seqs, rho_a, pos_a, rho_b, inst, params,
                                     tc_rwy, lbt_rwy, sep_rwy)
    return None


# ═════════════════════════════════════════════════════════════════════════════
#   §8  TEMPERATURE CALIBRATION
#   T0 = −mean(Δ+) / ln(chi0)  (Kirkpatrick 1983)
# ═════════════════════════════════════════════════════════════════════════════
def _calibrate_t0(seqs, inst, params, p_sa, seed, N_iter):
    """Estimate T0 by sampling n_cal random mid-phase moves and collecting Δ+."""
    rng = random.Random(seed)
    tc_rwy, lbt_rwy, sep_rwy = _init_proxy_arrays(seqs, inst)
    proxy_cur = compute_proxy(seqs, tc_rwy, lbt_rwy, sep_rwy, inst, params)
    pa_tc, pa_lbt = _compute_per_aircraft_scores(seqs, inst)
    deltas_pos = []
    for _ in range(p_sa.n_cal):
        op  = _select_op(0.5, len(seqs), rng)
        res = _apply_op(op, seqs, tc_rwy, lbt_rwy, sep_rwy,
                        inst, params, p_sa, rng, 0, N_iter, pa_tc, pa_lbt)
        if res is None: continue
        tc_n, lbt_n, sep_n = _init_proxy_arrays(res.seqs, inst)
        d = compute_proxy(res.seqs, tc_n, lbt_n, sep_n, inst, params) - proxy_cur
        if d > 1e-9: deltas_pos.append(d)
    if not deltas_pos: return max(abs(proxy_cur)*0.01, 1.0)
    return max(-float(np.mean(deltas_pos)) / math.log(p_sa.chi0+1e-12), 1e-3)


# ═════════════════════════════════════════════════════════════════════════════
#   §8b  SA PARAMETER OPTUNA TUNING
# ═════════════════════════════════════════════════════════════════════════════
def _sa_n_trials(n, base):
    if n<=50: return base
    if n<=100: return max(10, base//2)
    if n<=250: return max(6,  base//4)
    return max(3, base//7)


def _sa_n_iter_tune(n):
    """Reduced iteration budget for Optuna trials (~1/6 of production budget)."""
    return max(300, _n_iter(n)//6)


def optimize_sa_params(inst, m, params, n_trials, seed, n_jobs=1):
    """
    TPE search over (chi0, M_stag_frac, beta, lp_gamma, chi_target).
    lp_repair_interval is set to 0 during tuning to keep each trial fast.
    """
    if not _OPTUNA: return MRSAParams()
    if n_trials==0: return MRSAParams()
    N_tune = _sa_n_iter_tune(inst.n)
    def objective(trial):
        p_sa = MRSAParams(
            chi0        = trial.suggest_float('chi0',        0.50, 0.95),
            M_stag_frac = trial.suggest_float('M_stag_frac', 0.05, 0.30),
            beta        = trial.suggest_float('beta',        1.20, 2.50),
            lp_gamma    = trial.suggest_float('lp_gamma',   0.01, 0.20),
            chi_target  = trial.suggest_float('chi_target', 0.10, 0.35),
            lp_repair_interval = 0,
        )
        seqs, _ = ramp_rbi(inst, m, params)
        _, _, blp_seqs, best_lp, _, _ = run_mr_sa(
            seqs, math.inf, inst, params, p_sa, N_tune,
            label="sa_tune", seed=trial.number*13+seed)
        if math.isinf(best_lp):
            lp_val, _, feas, _ = stage2_lp_objective(blp_seqs or seqs, inst)
            best_lp = lp_val if feas else 1e12
        return best_lp
    sampler = _optuna.samplers.TPESampler(seed=seed)
    study   = _optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials,
                   n_jobs=min(n_jobs,n_trials), show_progress_bar=False)
    bp = study.best_params
    return MRSAParams(chi0=bp['chi0'], M_stag_frac=bp['M_stag_frac'],
                      beta=bp['beta'], lp_gamma=bp['lp_gamma'],
                      chi_target=bp['chi_target'])


# ═════════════════════════════════════════════════════════════════════════════
#   §9  ADAPTIVE LP CALL INTERVAL
#   Early (f≤0.25): every 20. Mid (f≤0.75): every 50. Late: every 100.
# ═════════════════════════════════════════════════════════════════════════════
def _n_full(t, N_iter):
    f = t/max(N_iter,1)
    if f<=0.25: return 20
    if f<=0.75: return 50
    return 100


# ═════════════════════════════════════════════════════════════════════════════
#   §10  LP-GUIDED REPAIR OPERATORS
# ═════════════════════════════════════════════════════════════════════════════

def _top_penalty_aircraft(C_lp, inst, q):
    """Return q aircraft indices sorted by descending LP penalty P_j = g_j·E_j + h_j·T_j."""
    E = np.maximum(inst.delta-C_lp, 0.0); T = np.maximum(C_lp-inst.delta, 0.0)
    return list(np.argsort(inst.g*E+inst.h*T)[::-1][:q])


def lp_guided_penalty_repair(seqs, C_lp, inst, params, K=15, q_lp=15):
    """
    Relocate each top-penalty aircraft to its globally best feasible position.

    Algorithm: enumerate all feasible (runway, position) insertions for each
    aircraft in H = top q_lp by P_j; sort by proxy; LP-evaluate the top K.

    Returns (best_cand, best_lp) or (None, inf) if no improvement found.
    """
    m = len(seqs)
    loc = {seqs[rho][pos]:(rho,pos) for rho in range(m) for pos in range(len(seqs[rho]))}
    H   = _top_penalty_aircraft(C_lp, inst, q_lp)
    candidates = []
    for j in H:
        rho_src, pos_src = loc[j]
        sm = seqs[rho_src][:pos_src]+seqs[rho_src][pos_src+1:]
        if not _runway_feasible(sm, inst): continue
        base = [s[:] for s in seqs]; base[rho_src] = sm
        for rho_dst in range(m):
            for p_dst in range(len(base[rho_dst])+1):
                cand = [s[:] for s in base]
                cand[rho_dst] = cand[rho_dst][:p_dst]+[j]+cand[rho_dst][p_dst:]
                if not _runway_feasible(cand[rho_dst], inst): continue
                tc_n,lbt_n,sep_n = _init_proxy_arrays(cand, inst)
                candidates.append((compute_proxy(cand,tc_n,lbt_n,sep_n,inst,params), cand))
        if len(candidates)>K*20:
            candidates.sort(key=lambda x:x[0]); candidates=candidates[:K*5]
    if not candidates: return None, math.inf
    candidates.sort(key=lambda x:x[0])
    best_lp=math.inf; best_cand=None
    for _,cand in candidates[:K]:
        lp,_,feas,_ = stage2_lp_objective(cand, inst)
        if feas and lp<best_lp: best_lp=lp; best_cand=cand
    return best_cand, best_lp


def lp_guided_pair_swap(seqs, C_lp, inst, params, q_lp=15, K=30, kappa=0.25):
    """
    Exchange high-penalty aircraft with target-time-compatible partners.

    Target-time filter: |δ_i − δ_j| ≤ κ·W_bar restricts swaps to aircraft
    with similar target landing times, avoiding structurally disruptive
    exchanges while permitting coordinated corrections.
    """
    m  = len(seqs)
    H  = _top_penalty_aircraft(C_lp, inst, q_lp)
    loc = {seqs[rho][pos]:(rho,pos) for rho in range(m) for pos in range(len(seqs[rho]))}
    W_bar = inst.W_bar; candidates = []
    for i in H:
        rho_i, pos_i = loc[i]
        for rho_j in range(m):
            if rho_j==rho_i: continue
            for pos_j, j in enumerate(seqs[rho_j]):
                if abs(inst.delta[i]-inst.delta[j]) > kappa*W_bar: continue
                res = _op_x2_swap(seqs, rho_i, pos_i, rho_j, pos_j, inst)
                if res is None: continue
                tc_n,lbt_n,sep_n = _init_proxy_arrays(res.seqs, inst)
                candidates.append((compute_proxy(res.seqs,tc_n,lbt_n,sep_n,inst,params), res.seqs))
    if not candidates: return None, math.inf
    candidates.sort(key=lambda x:x[0])
    best_lp=math.inf; best_cand=None
    for _,cand in candidates[:K]:
        lp,_,feas,_ = stage2_lp_objective(cand, inst)
        if feas and lp<best_lp: best_lp=lp; best_cand=cand
    return best_cand, best_lp


def target_conflict_repair(seqs, inst, params, K=15):
    """
    Deterministic repair for near-zero-objective instances.

    When F_LP is small, the schedule is close to one where C_j = δ_j for all j.
    Zero-penalty feasibility requires δ_j−δ_i ≥ s_{ij} for all i≺j.
    This operator identifies the most conflicted pairs (TC_{ij} > 0) and tries
    relocating one aircraft from each pair to reduce conflict.
    """
    m = len(seqs); conflicts = []
    for rho, seq in enumerate(seqs):
        for qi in range(len(seq)):
            for qj in range(qi+1, len(seq)):
                i,j = seq[qi], seq[qj]
                tc = max(0.0, float(inst.s[i,j])-(float(inst.delta[j])-float(inst.delta[i])))
                if tc > 1e-9: conflicts.append((tc, rho, qi, i, j))
    if not conflicts: return None, math.inf
    conflicts.sort(reverse=True)
    loc = {seqs[rho][pos]:(rho,pos) for rho in range(m) for pos in range(len(seqs[rho]))}
    candidates = []
    for _,rho_c,qi,i,j in conflicts[:8]:
        for ac in [i,j]:
            rho_src, pos_src = loc[ac]
            sm = seqs[rho_src][:pos_src]+seqs[rho_src][pos_src+1:]
            if not _runway_feasible(sm, inst): continue
            base=[s[:] for s in seqs]; base[rho_src]=sm
            for rho_dst in range(m):
                for p_dst in range(len(base[rho_dst])+1):
                    if rho_dst==rho_src and p_dst==pos_src: continue
                    cand=[s[:] for s in base]
                    cand[rho_dst]=cand[rho_dst][:p_dst]+[ac]+cand[rho_dst][p_dst:]
                    if not _runway_feasible(cand[rho_dst], inst): continue
                    tc_n,lbt_n,sep_n=_init_proxy_arrays(cand, inst)
                    candidates.append((compute_proxy(cand,tc_n,lbt_n,sep_n,inst,params), cand))
            if len(candidates)>K*15:
                candidates.sort(key=lambda x:x[0]); candidates=candidates[:K*4]
    if not candidates: return None, math.inf
    candidates.sort(key=lambda x:x[0])
    best_lp=math.inf; best_cand=None
    for _,cand in candidates[:K]:
        lp,_,feas,_ = stage2_lp_objective(cand, inst)
        if feas and lp<best_lp: best_lp=lp; best_cand=cand
    return best_cand, best_lp


def ejection_chain_transfer(
    seqs:   List[List[int]],
    C_lp:   np.ndarray,
    inst:   Instance,
    params: HeuristicParams,
    depth:  int = 2,
    K:      int = 15,
) -> Tuple[Optional[List[List[int]]], float]:
    """
    Depth-D ejection chain operator.

    Depth 1: best-position X3 transfer (j₁: ρ₁→ρ₂).
    Depth 2: j₁: ρ₁→ρ₂  (best feasible insertion on ρ₂), then
             j₂: ρ₂→ρ₃  (any original ρ₂ occupant; best insertion on ρ₃).

    This enables coordinated two-aircraft moves that no single X2/X3 can
    achieve — particularly valuable when the correct fix involves freeing a
    slot on one runway to accept a high-penalty aircraft from another.

    Depth is automatically capped at 1 when m < 3.

    Parameters
    ----------
    seqs, C_lp, inst, params : standard arguments.
    depth : int   — 1 or 2.
    K : int       — LP evaluation budget.
    """
    m = len(seqs)
    if m < 3: depth = 1
    q_lp,_ = _lp_repair_params(inst.n)
    H   = _top_penalty_aircraft(C_lp, inst, min(q_lp,6))
    loc = {seqs[rho][pos]:(rho,pos) for rho in range(m) for pos in range(len(seqs[rho]))}
    candidates = []

    for j1 in H:
        rho1, pos1 = loc[j1]
        sm1 = seqs[rho1][:pos1]+seqs[rho1][pos1+1:]
        if not _runway_feasible(sm1, inst): continue
        for rho2 in range(m):
            if rho2==rho1: continue
            best_q2, best_seq2, best_s2 = -1, None, math.inf
            for q2 in range(len(seqs[rho2])+1):
                c2 = seqs[rho2][:q2]+[j1]+seqs[rho2][q2:]
                if not _runway_feasible(c2, inst): continue
                tc,lbt,sep = _rwy_proxy_components(c2, inst)
                s = params.mu_tc*tc+params.mu_late*lbt+params.mu_sep*sep
                if s<best_s2: best_s2=s; best_q2=q2; best_seq2=c2
            if best_seq2 is None: continue
            st1 = [s[:] for s in seqs]; st1[rho1]=sm1; st1[rho2]=best_seq2
            if depth==1:
                tc_n,lbt_n,sep_n=_init_proxy_arrays(st1, inst)
                candidates.append((compute_proxy(st1,tc_n,lbt_n,sep_n,inst,params),[s[:] for s in st1]))
            else:
                for j2 in seqs[rho2]:
                    try: j2_pos = best_seq2.index(j2)
                    except ValueError: continue
                    sm2 = best_seq2[:j2_pos]+best_seq2[j2_pos+1:]
                    if not _runway_feasible(sm2, inst): continue
                    for rho3 in range(m):
                        if rho3==rho2: continue
                        best_q3, best_seq3, best_s3 = -1, None, math.inf
                        for q3 in range(len(st1[rho3])+1):
                            c3 = st1[rho3][:q3]+[j2]+st1[rho3][q3:]
                            if not _runway_feasible(c3, inst): continue
                            tc,lbt,sep=_rwy_proxy_components(c3, inst)
                            s=params.mu_tc*tc+params.mu_late*lbt+params.mu_sep*sep
                            if s<best_s3: best_s3=s; best_q3=q3; best_seq3=c3
                        if best_seq3 is None: continue
                        st2=[s[:] for s in st1]; st2[rho2]=sm2; st2[rho3]=best_seq3
                        tc_n,lbt_n,sep_n=_init_proxy_arrays(st2, inst)
                        candidates.append((compute_proxy(st2,tc_n,lbt_n,sep_n,inst,params),[s[:] for s in st2]))
                    if len(candidates)>=K*20: break
                if len(candidates)>=K*20: break
            if len(candidates)>=K*20: break
        if len(candidates)>=K*20: break

    if not candidates: return None, math.inf
    candidates.sort(key=lambda x:x[0])
    best_lp=math.inf; best_cand=None
    for _,cand in candidates[:K]:
        lp,_,feas,_=stage2_lp_objective(cand, inst)
        if feas and lp<best_lp: best_lp=lp; best_cand=cand
    return best_cand, best_lp


# ═════════════════════════════════════════════════════════════════════════════
#   §10b  ELITE SOLUTION POOL
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class _EliteSolution:
    """A single LP-certified solution stored in the elite pool."""
    seqs:   List[List[int]]
    lp_obj: float
    C_lp:   Optional[np.ndarray]


class ElitePool:
    """
    Fixed-size pool of LP-certified schedules with runway-Hamming diversity guard.

    Admission criteria for a new solution S:
      (a) lp_obj(S) < max LP in pool  — quality-based.
      (b) runway_distance(S, P) ≥ min_diversity for all P in pool
          — diversity-based.

    After admission the pool is trimmed to max_size by removing the worst
    LP solution (preserves best quality among the admitted set).

    Parameters
    ----------
    max_size : int      — pool capacity (default ELITE_POOL_MAX).
    min_diversity : int — minimum runway Hamming distance (default ELITE_MIN_DIV).
    """

    def __init__(self, max_size=ELITE_POOL_MAX, min_diversity=ELITE_MIN_DIV):
        self.solutions:    List[_EliteSolution] = []
        self.max_size:     int = max_size
        self.min_diversity: int = min_diversity

    def runway_distance(self, seqs_a, seqs_b) -> int:
        """D(A,B) = |{j : ρ_A(j) ≠ ρ_B(j)}|  (runway Hamming distance)."""
        m = len(seqs_a)
        assign_a = {seqs_a[rho][pos]: rho for rho in range(m) for pos in range(len(seqs_a[rho]))}
        return sum(1 for rho in range(len(seqs_b)) for j in seqs_b[rho]
                   if assign_a.get(j) != rho)

    def try_add(self, seqs, lp_obj, C_lp) -> bool:
        """
        Attempt to add (seqs, lp_obj, C_lp) to the pool.

        Returns True if admitted; False otherwise.
        """
        if math.isinf(lp_obj): return False
        if not self.solutions:
            self.solutions.append(_EliteSolution(
                [s[:] for s in seqs], lp_obj,
                C_lp.copy() if C_lp is not None else None))
            return True
        diverse  = all(self.runway_distance(seqs,s.seqs)>=self.min_diversity
                       for s in self.solutions)
        worst_lp = max(s.lp_obj for s in self.solutions)
        if lp_obj < worst_lp or diverse:
            self.solutions.append(_EliteSolution(
                [s[:] for s in seqs], lp_obj,
                C_lp.copy() if C_lp is not None else None))
            if len(self.solutions) > self.max_size:
                self.solutions.sort(key=lambda s: s.lp_obj)
                self.solutions = self.solutions[:self.max_size]
            return True
        return False

    @property
    def best(self) -> Optional[_EliteSolution]:
        """Solution with the lowest LP objective."""
        return min(self.solutions, key=lambda s: s.lp_obj) if self.solutions else None

    def most_diverse_pair(self):
        """Pair (A,B) with the largest runway distance.  O(|pool|²)."""
        if len(self.solutions) < 2: return None, None
        best_d = -1; best_a = best_b = None
        for i in range(len(self.solutions)):
            for j in range(i+1, len(self.solutions)):
                d = self.runway_distance(self.solutions[i].seqs, self.solutions[j].seqs)
                if d > best_d: best_d=d; best_a,best_b = self.solutions[i],self.solutions[j]
        return best_a, best_b

    def best_quality_pair(self):
        """
        Two solutions with the lowest LP objectives that also satisfy the
        diversity constraint.  Falls back to most_diverse_pair if no
        quality pair is diverse enough.
        """
        if len(self.solutions) < 2: return None, None
        ss = sorted(self.solutions, key=lambda s: s.lp_obj)
        for i in range(len(ss)):
            for j in range(i+1, len(ss)):
                if self.runway_distance(ss[i].seqs, ss[j].seqs) >= self.min_diversity:
                    return ss[i], ss[j]
        return self.most_diverse_pair()


# ═════════════════════════════════════════════════════════════════════════════
#   §10c  PATH RELINKING
# ═════════════════════════════════════════════════════════════════════════════

def path_relink(
    sol_a:         _EliteSolution,
    sol_b:         _EliteSolution,
    inst:          Instance,
    params:        HeuristicParams,
    max_steps:     int = 40,
    eval_interval: int = 5,
    K_lp:          int = 8,
) -> Tuple[List[List[int]], float]:
    """
    Walk from sol_a toward sol_b by iteratively moving differing aircraft to
    their target runway in sol_b.

    At each step, the differing aircraft with the highest LP-impact score
    (or highest penalty rate when C_lp is unavailable) is relocated to the
    best feasible insertion position on its target runway.

    LP evaluation happens every eval_interval steps using a proxy-sorted
    buffer of K_lp candidates.  The path with the best LP value found
    along the way is returned.  If no improvement is found,
    (sol_a.seqs, sol_a.lp_obj) is returned unchanged.

    Parameters
    ----------
    sol_a, sol_b : _EliteSolution — start and guide solutions.
    max_steps : int    — relinking step budget.
    eval_interval : int — steps between LP evaluations.
    K_lp : int         — LP candidates evaluated per eval point.

    Returns
    -------
    (best_seqs, best_lp)
    """
    m       = len(sol_a.seqs)
    current = [s[:] for s in sol_a.seqs]
    best_seqs = [s[:] for s in sol_a.seqs]; best_lp = sol_a.lp_obj
    assign_b = {sol_b.seqs[rho][pos]: rho
                for rho in range(m) for pos in range(len(sol_b.seqs[rho]))}
    proxy_buffer: List[Tuple[float, List[List[int]]]] = []

    def _do_eval():
        nonlocal best_seqs, best_lp
        proxy_buffer.sort(key=lambda x: x[0])
        for _, cand in proxy_buffer[:K_lp]:
            lp,_,feas,_ = stage2_lp_objective(cand, inst)
            if feas and lp < best_lp-1e-9: best_seqs=cand; best_lp=lp
        proxy_buffer.clear()

    for step in range(max_steps):
        assign_cur = {current[rho][pos]: rho
                      for rho in range(m) for pos in range(len(current[rho]))}
        differing = [(j, assign_b[j]) for j in assign_b
                     if assign_cur.get(j) != assign_b[j]]
        if not differing: break
        if sol_a.C_lp is not None:
            impact = _lp_impact_scores(current, sol_a.C_lp, inst)
            differing.sort(key=lambda x: -impact[x[0]])
        else:
            differing.sort(key=lambda x: -(inst.g[x[0]]+inst.h[x[0]]))
        moved = False
        for j, rho_target in differing[:5]:
            rho_cur = assign_cur.get(j)
            if rho_cur is None or rho_cur==rho_target: continue
            pos_cur = current[rho_cur].index(j)
            sm = current[rho_cur][:pos_cur]+current[rho_cur][pos_cur+1:]
            if not _runway_feasible(sm, inst): continue
            best_q, best_score = -1, math.inf
            for q in range(len(current[rho_target])+1):
                cs = current[rho_target][:q]+[j]+current[rho_target][q:]
                if not _runway_feasible(cs, inst): continue
                tc,lbt,sep = _rwy_proxy_components(cs, inst)
                s = params.mu_tc*tc+params.mu_late*lbt+params.mu_sep*sep
                if s<best_score: best_score=s; best_q=q
            if best_q == -1: continue
            current[rho_cur]    = sm
            current[rho_target] = current[rho_target][:best_q]+[j]+current[rho_target][best_q:]
            moved = True; break
        if not moved: break
        tc_n,lbt_n,sep_n = _init_proxy_arrays(current, inst)
        px = compute_proxy(current, tc_n, lbt_n, sep_n, inst, params)
        proxy_buffer.append((px, [s[:] for s in current]))
        if (step+1) % eval_interval == 0: _do_eval()
    if proxy_buffer: _do_eval()
    return best_seqs, best_lp


# ═════════════════════════════════════════════════════════════════════════════
#   §11  SAMPLE-AND-SELECT CANDIDATE GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def _generate_candidate_pool(f, seqs, tc_rwy, lbt_rwy, sep_rwy,
                              inst, params, p_sa, rng, stag, N_iter, R,
                              pa_tc, pa_lbt, impact, C_lp):
    """
    Draw R feasible moves, score each by proxy, return sorted ascending list.

    Each entry: (proxy_new, _MoveResult, tc_n, lbt_n, sep_n).
    The pool may be shorter than R if operators repeatedly return None.
    """
    pool = []
    for _ in range(R):
        op  = _select_op(f, len(seqs), rng)
        res = _apply_op(op, seqs, tc_rwy, lbt_rwy, sep_rwy,
                        inst, params, p_sa, rng, stag, N_iter,
                        pa_tc=pa_tc, pa_lbt=pa_lbt, impact=impact, C_lp=C_lp)
        if res is None: continue
        tc_n=tc_rwy.copy(); lbt_n=lbt_rwy.copy(); sep_n=sep_rwy.copy()
        for rho in res.affected:
            tc_n[rho],lbt_n[rho],sep_n[rho]=_rwy_proxy_components(res.seqs[rho],inst)
        pool.append((compute_proxy(res.seqs,tc_n,lbt_n,sep_n,inst,params),res,tc_n,lbt_n,sep_n))
    pool.sort(key=lambda x: x[0])
    return pool


# ═════════════════════════════════════════════════════════════════════════════
#   §12  SINGLE SA CHAIN
# ═════════════════════════════════════════════════════════════════════════════

def run_mr_sa(
    init_seqs:  List[List[int]],
    init_lp:    float,
    inst:       Instance,
    params:     HeuristicParams,
    p_sa:       MRSAParams,
    N_iter:     int,
    label:      str   = "chain",
    seed:       int   = 0,
    T0:         Optional[float] = None,
    t_deadline: Optional[float] = None,
) -> Tuple[List[List[int]], float,
           List[List[int]], float,
           Optional[np.ndarray], dict]:
    """
    Run one SA chain; return both proxy and LP incumbents plus diagnostics.

    Dual-track incumbents
    ---------------------
    Movement track  — best_p_seqs / best_proxy  (proxy objective).
    LP track        — best_lp_seqs / best_lp / best_C_lp  (Stage-2 LP).

    Key changes vs v2
    -----------------
    * LP-slack impact scores refreshed after every LP improvement and used
      by _pick_aircraft_targeted in preference to proxy TC scores.
    * XE operator available in mid and late phase tables.
    * Periodic ejection-chain repair (every 3×lp_repair_interval iterations).
    * LP timeline tracking: every LP improvement is timestamped as
      (wall_time_s, lp_val) and returned in stats['lp_timeline'].  This
      enables per-chain LP improvement plots.

    Returns
    -------
    best_p_seqs, best_proxy, best_lp_seqs, best_lp, best_C_lp, stats

    stats keys: label, history, alpha_history, t_best_proxy, t_best_lp,
                wall, lp_timeline.
    """
    CHI_TARGET  = p_sa.chi_target
    ALPHA_STEP  = p_sa.alpha_step
    ALPHA_LO    = p_sa.alpha_lo
    ALPHA_HI    = p_sa.alpha_hi
    MAX_REHEATS = p_sa.max_reheats
    M_STAG      = max(1, int(p_sa.M_stag_frac*N_iter))
    GAMMA       = p_sa.lp_gamma
    LP_REPAIR   = p_sa.lp_repair_interval
    NZ_THRESH   = p_sa.near_zero_threshold
    EC_DEPTH    = min(p_sa.ejection_chain_depth, 2 if len(init_seqs)<3 else p_sa.ejection_chain_depth)
    R           = _R_candidates(inst.n)

    rng = random.Random(seed); m = len(init_seqs); t0 = time.perf_counter()
    seqs                     = [s[:] for s in init_seqs]
    tc_rwy, lbt_rwy, sep_rwy = _init_proxy_arrays(seqs, inst)
    proxy                    = compute_proxy(seqs, tc_rwy, lbt_rwy, sep_rwy, inst, params)
    pa_tc, pa_lbt            = _compute_per_aircraft_scores(seqs, inst)
    impact: Optional[np.ndarray] = None

    best_p_seqs  = [s[:] for s in seqs]; best_proxy = proxy; t_best_proxy = 0.0
    best_lp_seqs = [s[:] for s in seqs]; best_lp    = init_lp
    best_C_lp:   Optional[np.ndarray] = None;        t_best_lp = 0.0

    # LP timeline: list of (wall_time_s, lp_val) — one entry per LP improvement.
    # Initialised with the seed LP so the plot starts from a meaningful baseline.
    lp_timeline: List[Tuple[float, float]] = [(0.0, init_lp)] if not math.isinf(init_lp) else []

    best_proxy_lp_checked = proxy    # LP trigger reference (proxy scale)

    T     = T0 or _calibrate_t0(seqs, inst, params, p_sa, seed, N_iter)
    T_min = T * p_sa.T_min_frac
    alpha = (ALPHA_HI+ALPHA_LO)/2.0

    history=[]; alpha_history=[]
    stag=0; n_reheats=0; n_accepted=0; n_tried=0
    q_lp, K = _lp_repair_params(inst.n)

    for t in range(1, N_iter+1):
        if t_deadline is not None and time.perf_counter() >= t_deadline: break

        f    = t/N_iter
        pool = _generate_candidate_pool(f, seqs, tc_rwy, lbt_rwy, sep_rwy,
                                         inst, params, p_sa, rng, stag, N_iter, R,
                                         pa_tc, pa_lbt, impact, best_C_lp)
        if not pool:
            history.append(best_proxy); alpha_history.append(alpha); continue

        # 80% exploit best; 20% explore near-best
        if rng.random() < 0.80:
            proxy_new, res, tc_n, lbt_n, sep_n = pool[0]
        else:
            proxy_new, res, tc_n, lbt_n, sep_n = rng.choice(pool[:min(5,len(pool))])

        n_tried += 1
        dlt    = proxy_new - proxy
        accept = (dlt<=0 or rng.random()<math.exp(-dlt/max(T,1e-15)))

        if accept:
            seqs=res.seqs; tc_rwy=tc_n; lbt_rwy=lbt_n; sep_rwy=sep_n; proxy=proxy_new
            n_accepted += 1
            stag = max(stag-1,0) if dlt<0 else stag+1
            if proxy < best_proxy-1e-9:
                best_p_seqs=[s[:] for s in seqs]; best_proxy=proxy
                t_best_proxy=time.perf_counter()-t0; stag=0
        else:
            stag += 1

        # ── FIXED LP trigger: proxy-to-proxy comparison ───────────────────
        call_lp = (t % _n_full(t,N_iter)==0
                   or proxy_new < (1.0-GAMMA)*best_proxy_lp_checked)
        if call_lp:
            lp_val, C_cur, lp_feas, _ = stage2_lp_objective(seqs, inst)
            best_proxy_lp_checked = proxy
            if lp_feas and lp_val < best_lp-1e-9:
                best_lp_seqs=[s[:] for s in seqs]; best_lp=lp_val
                best_C_lp=C_cur; t_best_lp=time.perf_counter()-t0; stag=0
                impact = _lp_impact_scores(seqs, C_cur, inst,
                                           p_sa.lambda_binding, p_sa.eps_tight)
                lp_timeline.append((t_best_lp, lp_val))   # timestamp LP improvement

        # ── Periodic LP-guided penalty repair ────────────────────────────
        if LP_REPAIR>0 and t%LP_REPAIR==0 and best_C_lp is not None:
            cand,cand_lp=lp_guided_penalty_repair(best_lp_seqs,best_C_lp,inst,params,K=K,q_lp=q_lp)
            if cand is not None and cand_lp<best_lp-1e-9:
                best_lp_seqs=cand; best_lp=cand_lp
                _,best_C_lp,_,_=stage2_lp_objective(best_lp_seqs,inst)
                if best_C_lp is not None:
                    impact=_lp_impact_scores(best_lp_seqs,best_C_lp,inst,p_sa.lambda_binding,p_sa.eps_tight)
                    lp_timeline.append((time.perf_counter()-t0, cand_lp))
                stag=0

        # ── Near-zero: target-conflict repair ─────────────────────────────
        if LP_REPAIR>0 and t%(LP_REPAIR*2)==0 and best_lp<NZ_THRESH:
            cand,cand_lp=target_conflict_repair(best_lp_seqs,inst,params,K=max(K//2,3))
            if cand is not None and cand_lp<best_lp-1e-9:
                best_lp_seqs=cand; best_lp=cand_lp
                _,C_new,feas_new,_=stage2_lp_objective(best_lp_seqs,inst)
                if feas_new:
                    best_C_lp=C_new
                    lp_timeline.append((time.perf_counter()-t0, cand_lp))

        # ── Periodic ejection-chain repair ────────────────────────────────
        if LP_REPAIR>0 and t%(LP_REPAIR*3)==0 and best_C_lp is not None and m>=2:
            cand,cand_lp=ejection_chain_transfer(best_lp_seqs,best_C_lp,inst,params,
                                                  depth=EC_DEPTH,K=max(K//2,3))
            if cand is not None and cand_lp<best_lp-1e-9:
                best_lp_seqs=cand; best_lp=cand_lp
                _,C_new,feas_new,_=stage2_lp_objective(best_lp_seqs,inst)
                if feas_new:
                    best_C_lp=C_new
                    impact=_lp_impact_scores(best_lp_seqs,best_C_lp,inst,p_sa.lambda_binding,p_sa.eps_tight)
                    lp_timeline.append((time.perf_counter()-t0, cand_lp))
                stag=0

        # ── Reactive cooling + score refresh ─────────────────────────────
        if t % _n_full(t,N_iter)==0:
            chi   = n_accepted/max(n_tried,1)
            alpha = (max(ALPHA_LO, alpha-ALPHA_STEP) if chi>CHI_TARGET
                     else min(ALPHA_HI, alpha+ALPHA_STEP))
            n_accepted=n_tried=0
            pa_tc, pa_lbt = _compute_per_aircraft_scores(seqs, inst)

        T = max(T*alpha, T_min)

        if stag >= M_STAG:
            if n_reheats >= MAX_REHEATS: break
            T = min(T*p_sa.t_reheat, T0 or T)
            for _ in range(5):
                pres=_apply_op(rng.choice(["X4","X2"]),seqs,tc_rwy,lbt_rwy,sep_rwy,
                               inst,params,p_sa,rng,M_STAG+1,N_iter,pa_tc=pa_tc,pa_lbt=pa_lbt,impact=impact)
                if pres is not None:
                    for rho in pres.affected:
                        tc_rwy[rho],lbt_rwy[rho],sep_rwy[rho]=_rwy_proxy_components(pres.seqs[rho],inst)
                    seqs=pres.seqs; proxy=compute_proxy(seqs,tc_rwy,lbt_rwy,sep_rwy,inst,params)
                    break
            stag=0; n_reheats+=1

        history.append(best_proxy); alpha_history.append(alpha)

    # End-of-chain LP call if LP track is still empty
    if math.isinf(best_lp):
        lp_val,C_cur,lp_feas,_=stage2_lp_objective(best_p_seqs,inst)
        if lp_feas:
            best_lp_seqs=[s[:] for s in best_p_seqs]; best_lp=lp_val; best_C_lp=C_cur
            lp_timeline.append((time.perf_counter()-t0, lp_val))

    return best_p_seqs, best_proxy, best_lp_seqs, best_lp, best_C_lp, {
        'label':         label,
        'history':       history,
        'alpha_history': alpha_history,
        't_best_proxy':  t_best_proxy,
        't_best_lp':     t_best_lp,
        'wall':          time.perf_counter()-t0,
        'lp_timeline':   lp_timeline,
    }


# ═════════════════════════════════════════════════════════════════════════════
#   §13  LP-VND POLISH
# ═════════════════════════════════════════════════════════════════════════════

def lp_vnd_polish(
    seqs:       List[List[int]],
    init_lp:    float,
    C_lp:       np.ndarray,
    inst:       Instance,
    params:     HeuristicParams,
    p_sa:       MRSAParams = None,
    max_rounds: int   = 10,
    t_limit:    float = 90.0,
) -> Tuple[List[List[int]], float]:
    """
    Monotone LP-verified VND with four neighbourhoods.

    Neighbourhood order (first-improvement restart strategy)
    --------------------------------------------------------
    1. LP-guided penalty repair
    2. LP-guided pair swap
    3. Target-conflict repair  (near-zero instances only)
    4. Ejection chain          (depth-2 where m ≥ 3)

    Restarts from neighbourhood 1 on any LP improvement.
    Terminates when no neighbourhood yields improvement, max_rounds is
    reached, or t_limit seconds elapse.
    """
    p_sa     = p_sa or MRSAParams()
    best_seqs= [s[:] for s in seqs]; best_lp=init_lp
    best_C   = C_lp.copy() if C_lp is not None else None
    q_lp, K  = _lp_repair_params(inst.n)
    t0       = time.perf_counter(); m = len(seqs)
    ec_depth = min(p_sa.ejection_chain_depth, 2 if m<3 else p_sa.ejection_chain_depth)

    for _ in range(max_rounds):
        if time.perf_counter()-t0 > t_limit: break
        improved = False
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
        if best_lp < 200.0:
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
    return best_seqs, best_lp


# ═════════════════════════════════════════════════════════════════════════════
#   §14  CONTROLLED SEED PORTFOLIO
# ═════════════════════════════════════════════════════════════════════════════

def _controlled_cross_perturb(seqs, inst, params, rng, n_moves=3):
    """Apply up to n_moves proxy-improving X3 transfers; reject worsening moves."""
    cur=[s[:] for s in seqs]; m=len(cur)
    if m<2: return cur
    tc_rwy,lbt_rwy,sep_rwy=_init_proxy_arrays(cur,inst)
    proxy_cur=compute_proxy(cur,tc_rwy,lbt_rwy,sep_rwy,inst,params)
    moves_made=0
    for _ in range(60):
        if moves_made>=n_moves: break
        rho_a=rng.randint(0,m-1)
        if not cur[rho_a]: continue
        pos_a=rng.randint(0,len(cur[rho_a])-1)
        rho_b=rng.choice([r for r in range(m) if r!=rho_a])
        res=_op_x3_best_transfer(cur,rho_a,pos_a,rho_b,inst,params,tc_rwy,lbt_rwy,sep_rwy)
        if res is None: continue
        tc_n=tc_rwy.copy(); lbt_n=lbt_rwy.copy(); sep_n=sep_rwy.copy()
        for rho in res.affected:
            tc_n[rho],lbt_n[rho],sep_n[rho]=_rwy_proxy_components(res.seqs[rho],inst)
        px_new=compute_proxy(res.seqs,tc_n,lbt_n,sep_n,inst,params)
        if px_new<proxy_cur:
            cur=res.seqs; tc_rwy=tc_n; lbt_rwy=lbt_n; sep_rwy=sep_n
            proxy_cur=px_new; moves_made+=1
    return cur


def _build_starts(inst, m, params, n_chains, seed):
    """
    Build n_chains starting sequences from controlled perturbations of TC-RBI.

    Chain 0 — canonical TC-RBI.
    Chain 1 — TC-RBI + LP-guided penalty repair on the base LP solution.
    Chain 2 — TC-RBI + up to 3 proxy-improving cross-runway transfers.
    Chain 3 — TC-RBI + target-conflict repair.

    All seeds start near TC-RBI quality, avoiding the large quality drops
    that random perturbations caused in v1.
    """
    rng=random.Random(seed); base,_=ramp_rbi(inst,m,params)
    q_lp,K=_lp_repair_params(inst.n)
    base_lp,base_C,base_feas,_=stage2_lp_objective(base,inst)
    if not base_feas: base_lp=math.inf; base_C=None

    starts=[("TC-RBI",base)]
    if n_chains>=2:
        if base_C is not None:
            cand,_=lp_guided_penalty_repair(base,base_C,inst,params,K=min(K,5),q_lp=min(q_lp,10))
            starts.append(("LP-repair", cand if cand is not None else base))
        else:
            starts.append(("TC-RBI-2",base))
    if n_chains>=3:
        starts.append(("Ctrl-X",_controlled_cross_perturb(base,inst,params,rng,3)))
    if n_chains>=4:
        if base_C is not None:
            cand4,_=target_conflict_repair(base,inst,params,K=min(K,5))
            starts.append(("TC-repair", cand4 if cand4 is not None else base))
        else:
            starts.append(("TC-RBI-4",base))
    return starts[:n_chains]


# ═════════════════════════════════════════════════════════════════════════════
#   §15  SPAWN-SAFE WORKER
# ═════════════════════════════════════════════════════════════════════════════
def _sa_worker(args: tuple) -> tuple:
    """
    Entry point for one SA chain in a worker process.

    Return tuple (11 elements):
        0  label
        1  bp_seqs         best proxy sequences
        2  b_proxy         best proxy value
        3  blp_seqs        best LP sequences
        4  b_lp            best LP value
        5  b_C_lp          LP solution vector (ndarray or None)
        6  history         per-iteration best_proxy list
        7  t_best_proxy    wall time of best proxy
        8  t_best_lp       wall time of best LP
        9  alpha_history   per-iteration α list
        10 lp_timeline     list of (wall_time_s, lp_val) LP improvement events
    """
    label,init_seqs,init_lp,inst,params,p_sa,N_iter,seed,t_deadline = args
    bp_seqs,b_proxy,blp_seqs,b_lp,b_C_lp,st = run_mr_sa(
        init_seqs,init_lp,inst,params,p_sa,N_iter,
        label=label,seed=seed,t_deadline=t_deadline)
    return (label,bp_seqs,b_proxy,blp_seqs,b_lp,b_C_lp,
            st['history'],st['t_best_proxy'],st['t_best_lp'],
            st['alpha_history'],st['lp_timeline'])


# ═════════════════════════════════════════════════════════════════════════════
#   §16  PARALLEL MULTI-START SA
# ═════════════════════════════════════════════════════════════════════════════

def ms_mr_sa(
    inst:     Instance,
    m:        int,
    params:   HeuristicParams,
    p_sa:     MRSAParams   = None,
    n_chains: int          = N_CHAINS,
    t_limit:  float        = T_LIMIT,
    seed:     int          = 0,
) -> Tuple[List[List[int]], float, dict]:
    """
    Run K parallel SA chains; collect elite pool; apply path relinking and
    LP-VND polish; return the best overall solution.

    Pipeline (v3)
    -------------
    1. Build controlled seed portfolio (_build_starts).
    2. Evaluate seed LPs; build initial job LP timeline.
    3. Run K chains concurrently via ProcessPoolExecutor.
    4. Collect LP incumbents from all chains into the elite pool.
    5. Select the chain with the best LP objective.
    6. Final LP call on winning sequences.
    7. LP-VND polish (≤ 15% of t_limit).
    8. Path relinking between best-quality pair and most-diverse pair
       (≤ 10% of t_limit, both forward and reverse directions).
    9. Final LP re-verify.

    Returns
    -------
    best_seqs : list of list of int
    best_lp : float
    stats : dict
        seed_lps, all_results, wall, t_best_lp, final_feas, final_viols,
        history, alpha_history, elite_pool_size, relinking_improved,
        elite_solutions, job_lp_timeline.
    """
    p_sa   = p_sa or MRSAParams()
    N_iter = _n_iter(inst.n)
    t0     = time.perf_counter(); t_dead = t0 + t_limit

    starts = _build_starts(inst, m, params, n_chains, seed)
    print(f"  [{inst.name} m={m}] {len(starts)} seeds | N_iter={N_iter} | t_limit={t_limit:.0f}s")
    print(f"  SA params: {p_sa}")

    seed_lps = []
    for lbl, s in starts:
        lp,_,feas,_ = stage2_lp_objective(s, inst)
        seed_lps.append(lp if feas else math.inf)
        print(f"    seed {lbl:<12} LP={lp:.4f}" if not math.isinf(lp) else f"    seed {lbl:<12} LP=inf")

    # Job-level LP timeline: starts with best seed
    best_seed_lp = min(seed_lps)
    job_lp_timeline: List[Tuple[float, float]] = [(0.0, best_seed_lp)] if not math.isinf(best_seed_lp) else []

    tasks = [(lbl,s,seed_lps[i],inst,params,p_sa,N_iter,seed+i*31,t_dead)
             for i,(lbl,s) in enumerate(starts)]

    with ProcessPoolExecutor(max_workers=min(n_chains,len(tasks)), mp_context=_MP_CTX) as ex:
        results = list(ex.map(_sa_worker, tasks))

    # r: 0=label,1=bp,2=b_proxy,3=blp,4=b_lp,5=b_C_lp,6=hist,7=tbp,8=tbl,9=ah,10=lp_timeline
    feas_rs = [r for r in results if not math.isinf(r[4])]
    if feas_rs:
        best_r    = min(feas_rs, key=lambda r: r[4])
        best_seqs = best_r[3]; best_lp = best_r[4]; best_C = best_r[5]
    else:
        warnings.warn(f"{inst.name} m={m}: no LP-feasible solution found.")
        best_r    = min(results, key=lambda r: r[2])
        best_seqs = best_r[1]; best_lp = math.inf; best_C = None

    # Absorb winning chain's LP timeline into job timeline
    for t_chain, lp_val in best_r[10]:
        job_lp_timeline.append((t_chain, lp_val))

    # Collect elite pool
    pool = ElitePool(ELITE_POOL_MAX, ELITE_MIN_DIV)
    for r in feas_rs:
        pool.try_add(r[3], r[4], r[5])

    # Final LP call on winning sequences
    final_lp,final_C,final_feas,final_viols = stage2_lp_objective(best_seqs, inst)
    if final_feas and final_lp < best_lp-1e-9:
        best_lp=final_lp; best_C=final_C
        job_lp_timeline.append((time.perf_counter()-t0, final_lp))
    if final_feas and final_C is not None:
        pool.try_add(best_seqs, best_lp, final_C)

    # LP-VND polish
    if best_C is not None and not math.isinf(best_lp):
        vnd_lp_prev = best_lp
        best_seqs, best_lp = lp_vnd_polish(
            best_seqs, best_lp, best_C, inst, params, p_sa,
            max_rounds=_vnd_max_rounds(inst.n), t_limit=max(30.0, t_limit*0.15))
        final_lp,final_C,final_feas,final_viols = stage2_lp_objective(best_seqs, inst)
        if final_feas and final_lp < best_lp-1e-9: best_lp=final_lp; best_C=final_C
        if best_lp < vnd_lp_prev-1e-9:
            job_lp_timeline.append((time.perf_counter()-t0, best_lp))
        if final_feas and final_C is not None:
            pool.try_add(best_seqs, best_lp, final_C)

    # Path relinking
    relink_improved = False
    pr_t_limit = max(20.0, t_limit*0.10); pr_t0 = time.perf_counter()
    for pair_fn in [pool.best_quality_pair, pool.most_diverse_pair]:
        if time.perf_counter()-pr_t0 > pr_t_limit: break
        sol_a, sol_b = pair_fn()
        if sol_a is None: continue
        for a, b in [(sol_a,sol_b),(sol_b,sol_a)]:
            if time.perf_counter()-pr_t0 > pr_t_limit: break
            pr_seqs, pr_lp = path_relink(a, b, inst, params, max_steps=40, eval_interval=5, K_lp=8)
            if pr_lp < best_lp-1e-9:
                best_seqs=pr_seqs; best_lp=pr_lp
                _,pr_C,pr_feas,_=stage2_lp_objective(best_seqs, inst)
                if pr_feas: best_C=pr_C; pool.try_add(best_seqs,best_lp,best_C)
                job_lp_timeline.append((time.perf_counter()-t0, best_lp))
                relink_improved=True

    # Final re-verify
    final_lp,_,final_feas,final_viols = stage2_lp_objective(best_seqs, inst)
    if final_feas and final_lp < best_lp-1e-9:
        best_lp=final_lp
        job_lp_timeline.append((time.perf_counter()-t0, final_lp))

    # Collect elite solutions for export as alternative schedules
    elite_solutions = [(s.lp_obj, [seq[:] for seq in s.seqs])
                       for s in sorted(pool.solutions, key=lambda s: s.lp_obj)]

    return best_seqs, best_lp, {
        'seed_lps':           seed_lps,
        'all_results':        results,
        'wall':               time.perf_counter()-t0,
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
#   §17  BKS-AWARE REPORTING
# ═════════════════════════════════════════════════════════════════════════════

def _gap_str(obj: float, ref: Optional[float], mark_new: bool = True) -> str:
    """
    Format the BKS gap as a percentage string.

    ref = None    → "N/A"
    ref = 0, obj ≈ 0 → "0.00%"
    ref = 0, obj > 0 → "∞"
    obj < ref     → negative gap with ★ flag (new BKS candidate)
    """
    if ref is None:  return "N/A"
    if ref == 0.0:   return "0.00%" if obj < 1e-6 else "∞"
    gap = 100.0*(obj-ref)/ref
    if gap < -0.001 and mark_new: return f"{gap:.2f}% ★"
    return f"{gap:.2f}%"


def _is_new_bks(obj: float, ref: Optional[float]) -> bool:
    """True iff obj strictly beats the BKS reference."""
    if ref is None or ref <= 0.0: return False
    return obj < ref-1e-6


def print_mr_result(inst, m, seqs, lp_obj, elapsed, seed_lps, params, p_sa, stats=None):
    """Print a BKS-aware per-instance result report to stdout."""
    feas_e, viol_e, earliest_obj, _ = verify_and_exact_obj(seqs, inst)
    ref     = KNOWN_OPTIMA.get(inst.name, {}).get(m)
    new_bks = _is_new_bks(lp_obj, ref)
    sep = "=" * 74
    print(f"\n{sep}")
    print(f"  {inst.name.upper()}  |  n={inst.n}  |  m={m} runway(s)"
          + ("  ★ NEW BKS CANDIDATE ★" if new_bks else ""))
    print(sep)
    print(f"  Runtime (SA+PR+VND total): {elapsed:.2f} s")
    print(f"  TC-RBI params            : {params}")
    print(f"  SA params                : {p_sa}")
    best_seed = min(seed_lps) if seed_lps else math.inf
    print(f"  Best seed LP             : {best_seed:.4f}")
    print(f"  SA+VND+PR final LP       : {lp_obj:.4f}")
    print(f"  Earliest-time objective  : {earliest_obj:.4f}")
    if ref is not None:
        label = "BKS (opt=0)" if ref==0.0 else "Reference/BKS"
        print(f"  {label:<24} : {ref:.4f}")
        print(f"  Gap (seed → final)       : {_gap_str(best_seed,ref)} → {_gap_str(lp_obj,ref)}")
    else:
        print(f"  Reference/BKS            : not available for m={m}")
    if stats:
        print(f"  Time to best LP          : {stats.get('t_best_lp','N/A'):.2f} s"
              if isinstance(stats.get('t_best_lp'), float) else
              f"  Time to best LP          : N/A")
        print(f"  Elite pool size          : {stats.get('elite_pool_size','N/A')}")
        print(f"  Path relinking improved  : {'Yes' if stats.get('relinking_improved') else 'No'}")
    print(f"  Sequence feasibility     : {'PASS ✓' if feas_e else 'FAIL ✗'}")
    if not feas_e:
        for v in viol_e[:6]: print(f"    ✗ {v}")
        if len(viol_e)>6: print(f"    ... and {len(viol_e)-6} more")
    print("  Runway load:")
    for rho, seq in enumerate(seqs):
        print(f"    Runway {rho+1}: {len(seq):4d} aircraft  "
              f"seq=[{', '.join(str(j) for j in seq[:6])}"
              f"{',...' if len(seq)>6 else ''}]")
    print(sep)


def print_summary_table(results: List[dict]) -> None:
    """Print BKS-aware batch results table; flag new BKS candidates with ★."""
    col = ["Instance","n","m","Seed LP","Final LP","Reference",
           "Gap(seed)","Gap(SA)","BKS?","Feas","Time(s)"]
    w   = [12,5,4,12,12,12,10,10,5,6,9]
    hdr = "  "+"".join(f"{c:>{w[i]}}" for i,c in enumerate(col))
    bar = "="*len(hdr)
    print(f"\n{bar}\n  MR-SA v3  —  BATCH RESULTS\n{bar}")
    print(hdr); print("-"*len(hdr))
    for r in sorted(results, key=lambda x: (x["name"],x["m"])):
        ref  = r["opt"]
        row  = [r["name"],r["n"],r["m"],
                f"{r['seed_lp']:.4f}" if not math.isinf(r['seed_lp']) else "inf",
                f"{r['sa_lp']:.4f}"   if not math.isinf(r['sa_lp'])   else "inf",
                f"{ref:.4f}" if ref is not None else "N/A",
                _gap_str(r['seed_lp'],ref,False), _gap_str(r['sa_lp'],ref),
                "★" if _is_new_bks(r['sa_lp'],ref) else "",
                "✓" if r["feasible"] else "✗",
                f"{r['time']:.2f}"]
        print("  "+"".join(f"{str(v):>{w[i]}}" for i,v in enumerate(row)))
    print(bar)
    pos = [r for r in results if r["opt"] is not None and r["opt"]>0
           and not math.isinf(r["sa_lp"])]
    if pos:
        sg=[100.*(r["seed_lp"]-r["opt"])/r["opt"] for r in pos if not math.isinf(r["seed_lp"])]
        ag=[100.*(r["sa_lp"]-r["opt"])/r["opt"] for r in pos]
        fc=sum(1 for r in results if r["feasible"])
        nb=sum(1 for r in results if _is_new_bks(r["sa_lp"],r["opt"]))
        print(f"  Feasible         : {fc}/{len(results)}")
        print(f"  New BKS cands    : {nb}")
        if sg: print(f"  Avg seed gap     : {np.mean(sg):.2f}%  Max: {max(sg):.2f}%")
        if ag: print(f"  Avg final gap    : {np.mean(ag):.2f}%  Max: {max(ag):.2f}%")
        if sg and ag: print(f"  Avg improvement  : {np.mean(sg)-np.mean(ag):.2f}pp")
    print(bar)


# ═════════════════════════════════════════════════════════════════════════════
#   §18  RESULT PERSISTENCE
#
#   File layout under OUTPUT_DIR/
#   ──────────────────────────────
#   summary.csv        — one row per (instance, m): all key metrics.
#   schedules.csv      — final landing sequence in long format (one row per aircraft).
#   alternatives.csv   — elite pool solutions (alternative schedule portfolio).
#   verification.txt   — per-job feasibility audit.
#   run_metadata.json  — run configuration, parameters, timing, pool stats.
# ═════════════════════════════════════════════════════════════════════════════

def _ensure_dirs(output_dir: Path) -> None:
    """Create output_dir and its plots/ subdirectory if they do not exist."""
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)


def _save_summary_csv(results: List[dict], output_dir: Path) -> None:
    """
    Write summary.csv: one row per (instance, m) run.

    Columns: instance, n, m, seed_lp, sa_lp, bks, gap_seed_pct, gap_sa_pct,
             new_bks, feasible, time_s, t_best_lp_s, elite_pool_size,
             relinking_improved.
    """
    path = output_dir / "summary.csv"
    fields = ["instance","n","m","seed_lp","sa_lp","bks",
              "gap_seed_pct","gap_sa_pct","new_bks","feasible",
              "time_s","t_best_lp_s","elite_pool_size","relinking_improved"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted(results, key=lambda x: (x["name"],x["m"])):
            ref = r["opt"]
            gs = (100*(r["seed_lp"]-ref)/ref if ref and ref>0
                  and not math.isinf(r["seed_lp"]) else None)
            ga = (100*(r["sa_lp"]-ref)/ref if ref and ref>0
                  and not math.isinf(r["sa_lp"]) else None)
            w.writerow({
                "instance":         r["name"],
                "n":                r["n"],
                "m":                r["m"],
                "seed_lp":          "" if math.isinf(r["seed_lp"]) else f"{r['seed_lp']:.6f}",
                "sa_lp":            "" if math.isinf(r["sa_lp"])   else f"{r['sa_lp']:.6f}",
                "bks":              "" if ref is None else ref,
                "gap_seed_pct":     "" if gs is None else f"{gs:.4f}",
                "gap_sa_pct":       "" if ga is None else f"{ga:.4f}",
                "new_bks":          _is_new_bks(r["sa_lp"], ref),
                "feasible":         r["feasible"],
                "time_s":           f"{r['time']:.2f}",
                "t_best_lp_s":      f"{r.get('t_best_lp',0):.2f}",
                "elite_pool_size":  r.get("elite_pool_size",""),
                "relinking_improved": r.get("relinking_improved", False),
            })
    print(f"  Saved {path}")


def _save_schedules_csv(results: List[dict], output_dir: Path) -> None:
    """
    Write schedules.csv: long-format landing sequence table.

    Columns: instance, m, rho (runway), position, aircraft_j.
    One row per aircraft in the best final solution for each (instance, m).
    """
    path = output_dir / "schedules.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance","m","rho","position","aircraft_j"])
        for r in sorted(results, key=lambda x: (x["name"],x["m"])):
            for rho, seq in enumerate(r.get("best_seqs", [])):
                for pos, j in enumerate(seq):
                    w.writerow([r["name"], r["m"], rho+1, pos+1, j])
    print(f"  Saved {path}")


def _save_alternatives_csv(results: List[dict], output_dir: Path) -> None:
    """
    Write alternatives.csv: elite pool alternative schedules.

    Columns: instance, m, rank, lp_obj, rho, position, aircraft_j.
    rank=1 is the best LP solution in the pool.  These are diverse
    high-quality solutions that could be used as alternative schedules.
    """
    path = output_dir / "alternatives.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance","m","rank","lp_obj","rho","position","aircraft_j"])
        for r in sorted(results, key=lambda x: (x["name"],x["m"])):
            for rank, (lp_obj, seqs) in enumerate(r.get("elite_solutions",[]), start=1):
                for rho, seq in enumerate(seqs):
                    for pos, j in enumerate(seq):
                        w.writerow([r["name"],r["m"],rank,f"{lp_obj:.6f}",rho+1,pos+1,j])
    print(f"  Saved {path}")


def _save_verification_txt(results: List[dict], output_dir: Path) -> None:
    """
    Write verification.txt: detailed feasibility audit for each (instance, m).

    For each job: reports LP and sequence feasibility status, the SA+VND LP
    value, BKS gap, violations (if any), runway loads, and the full LP
    improvement timeline.
    """
    path = output_dir / "verification.txt"
    sep = "=" * 72
    with open(path, "w") as f:
        f.write("MR-SA v3 — FEASIBILITY VERIFICATION REPORT\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for r in sorted(results, key=lambda x: (x["name"],x["m"])):
            ref = r["opt"]
            f.write(f"{sep}\n")
            f.write(f"  {r['name'].upper()}  |  n={r['n']}  |  m={r['m']}\n")
            f.write(f"{sep}\n")
            f.write(f"  SA+VND+PR LP         : {r['sa_lp']:.6f}\n")
            if ref is not None:
                f.write(f"  Reference/BKS        : {ref}\n")
                f.write(f"  Gap to BKS           : {_gap_str(r['sa_lp'],ref)}\n")
                if _is_new_bks(r["sa_lp"], ref):
                    f.write("  *** NEW BKS CANDIDATE ***\n")
            f.write(f"  Sequence feasible    : {'YES' if r['feasible'] else 'NO'}\n")
            f.write(f"  Time to best LP (s)  : {r.get('t_best_lp',0):.2f}\n")
            f.write(f"  Total runtime (s)    : {r['time']:.2f}\n")
            viols = r.get("violations", [])
            if viols:
                f.write(f"  Violations ({len(viols)}):\n")
                for v in viols[:10]: f.write(f"    ✗ {v}\n")
                if len(viols)>10: f.write(f"    ... and {len(viols)-10} more\n")
            # LP improvement timeline
            tl = r.get("job_lp_timeline", [])
            if tl:
                f.write("  LP improvement timeline (time_s, lp_val):\n")
                for t_s, lp_v in tl:
                    f.write(f"    t={t_s:8.2f}s  LP={lp_v:.6f}\n")
            f.write("\n")
    print(f"  Saved {path}")


def _save_run_metadata_json(
    results: List[dict], output_dir: Path
) -> None:
    """
    Write run_metadata.json: configuration, parameters, and per-job timings.

    Top-level keys: run_time, config, results.
    config: N_WORKERS, N_CHAINS, T_LIMIT, MAX_T_LIMIT, ELITE_POOL_MAX,
            ELITE_MIN_DIV, RUN_SA_OPTUNA.
    results: list of per-job dicts (name, m, sa_lp, bks, gap, feasible,
             time, t_best_lp, elite_pool_size, relinking_improved,
             sa_params, n_elite_solutions).
    """
    path = output_dir / "run_metadata.json"
    payload: Dict[str, Any] = {
        "run_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "N_WORKERS":    N_WORKERS,
            "N_CHAINS":     N_CHAINS,
            "T_LIMIT":      T_LIMIT,
            "MAX_T_LIMIT":  MAX_T_LIMIT,
            "ELITE_POOL_MAX": ELITE_POOL_MAX,
            "ELITE_MIN_DIV":  ELITE_MIN_DIV,
            "RUN_SA_OPTUNA":  RUN_SA_OPTUNA,
        },
        "results": [],
    }
    for r in sorted(results, key=lambda x: (x["name"],x["m"])):
        ref = r["opt"]
        gap = (100*(r["sa_lp"]-ref)/ref if ref and ref>0
               and not math.isinf(r["sa_lp"]) else None)
        p   = r.get("p_sa", MRSAParams())
        payload["results"].append({
            "instance":           r["name"],
            "n":                  r["n"],
            "m":                  r["m"],
            "seed_lp":            None if math.isinf(r["seed_lp"]) else r["seed_lp"],
            "sa_lp":              None if math.isinf(r["sa_lp"])   else r["sa_lp"],
            "bks":                ref,
            "gap_pct":            round(gap,4) if gap is not None else None,
            "new_bks":            _is_new_bks(r["sa_lp"],ref),
            "feasible":           r["feasible"],
            "time_s":             round(r["time"],2),
            "t_best_lp_s":        round(r.get("t_best_lp",0),2),
            "elite_pool_size":    r.get("elite_pool_size",0),
            "n_elite_solutions":  len(r.get("elite_solutions",[])),
            "relinking_improved": r.get("relinking_improved",False),
            "sa_params": {
                "chi0":       p.chi0,
                "M_stag_frac":p.M_stag_frac,
                "lp_gamma":   p.lp_gamma,
                "chi_target": p.chi_target,
                "optuna_tuned": r.get("p_sa_tuned",False),
            },
        })
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved {path}")


def save_run_results(results: List[dict], output_dir: Path) -> None:
    """
    Write all result files to output_dir.

    Creates output_dir and output_dir/plots/ if they do not exist.
    Writes: summary.csv, schedules.csv, alternatives.csv,
            verification.txt, run_metadata.json.

    Parameters
    ----------
    results : list of dict
        Per-job result dicts from _run_one_mr.
    output_dir : Path
    """
    _ensure_dirs(output_dir)
    _save_summary_csv(results, output_dir)
    _save_schedules_csv(results, output_dir)
    _save_alternatives_csv(results, output_dir)
    _save_verification_txt(results, output_dir)
    _save_run_metadata_json(results, output_dir)


# ═════════════════════════════════════════════════════════════════════════════
#   §19  VISUALISATION
#
#   All plots use a consistent style: dark-grid background, colour-blind-
#   friendly default cycle, and tight layout.  Each function saves a PNG
#   to output_dir/plots/ and returns immediately if matplotlib is unavailable.
# ═════════════════════════════════════════════════════════════════════════════

_PLOT_STYLE = {
    "figure.facecolor":  "white",
    "axes.facecolor":    "#f7f7f7",
    "axes.grid":         True,
    "grid.color":        "white",
    "grid.linewidth":    0.8,
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
}


def _plot_gap_summary(results: List[dict], output_dir: Path) -> None:
    """
    Grouped bar chart: seed gap vs final SA+VND+PR gap for every positive-BKS
    (instance, m) pair.

    Bars are grouped by job (x-axis), coloured by source (seed=steel blue,
    final=dark orange).  A ★ is drawn above any bar where the final gap is
    negative (new BKS candidate).
    """
    if not _MPL: return
    pos_results = [r for r in results
                   if r["opt"] is not None and r["opt"]>0
                   and not math.isinf(r["sa_lp"])]
    if not pos_results: return

    pos_results = sorted(pos_results, key=lambda x: (x["name"],x["m"]))
    labels   = [f"{r['name']}\nm={r['m']}" for r in pos_results]
    seed_gaps = [100*(r["seed_lp"]-r["opt"])/r["opt"]
                 if not math.isinf(r["seed_lp"]) else 0 for r in pos_results]
    sa_gaps   = [100*(r["sa_lp"]  -r["opt"])/r["opt"] for r in pos_results]

    x    = np.arange(len(labels))
    w    = 0.38
    fig, ax = plt.subplots(figsize=(max(10, len(labels)*0.55+2), 5))
    with plt.rc_context(_PLOT_STYLE):
        b1 = ax.bar(x-w/2, seed_gaps, w, label="Seed LP gap",  color="#4878CF", alpha=0.85)
        b2 = ax.bar(x+w/2, sa_gaps,   w, label="Final LP gap", color="#D65F5F", alpha=0.85)
        # Mark new BKS candidates
        for xi, r, sg in zip(x, pos_results, sa_gaps):
            if _is_new_bks(r["sa_lp"], r["opt"]):
                ax.text(xi+w/2, max(sg, 0)+0.3, "★", ha="center", va="bottom",
                        color="goldenrod", fontsize=13, fontweight="bold")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
        ax.set_ylabel("Gap to BKS (%)")
        ax.set_title("MR-SA v3 — Seed vs Final Gap to BKS Reference")
        ax.legend(loc="upper right")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
        plt.tight_layout()
        out = output_dir / "plots" / "gap_summary.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved {out}")


def _plot_convergence(result: dict, output_dir: Path) -> None:
    """
    Plot proxy convergence (best_proxy per iteration) for the best SA chain
    of one (instance, m) job.

    x-axis: iteration index; y-axis: best proxy value (normalised to its
    starting value so different instances can be compared on the same scale).
    """
    if not _MPL: return
    history = result.get("history", [])
    if not history: return
    name = result["name"]; m = result["m"]
    hist = np.asarray(history, dtype=float)
    if hist[0] != 0:
        hist = hist / hist[0]   # normalise to starting proxy = 1

    fig, ax = plt.subplots(figsize=(7, 4))
    with plt.rc_context(_PLOT_STYLE):
        ax.plot(hist, linewidth=0.8, color="#4878CF", alpha=0.9)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best proxy (relative to start)")
        ax.set_title(f"{name.upper()} m={m} — SA proxy convergence (best chain)")
        plt.tight_layout()
        out = output_dir / "plots" / f"convergence_{name}_{m}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved {out}")


def _plot_lp_timeline(result: dict, output_dir: Path) -> None:
    """
    Plot LP objective improvement vs wall time for one (instance, m) job.

    x-axis: wall time (seconds) from job start; y-axis: LP objective value.
    Each point is a timestamped LP improvement event.  A horizontal dashed
    line shows the BKS reference when available.
    Vertical annotations mark the seed LP and the time-to-best-LP.
    """
    if not _MPL: return
    tl = result.get("job_lp_timeline", [])
    if len(tl) < 2: return
    name = result["name"]; m = result["m"]
    ref  = result.get("opt")

    ts  = [t for t,_ in tl]
    lps = [v for _,v in tl]
    # Extend to end of run for step-function appearance
    ts  = ts  + [result["time"]]
    lps = lps + [lps[-1]]

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
        ax.set_xlabel("Wall time (s)")
        ax.set_ylabel("LP objective")
        ax.set_title(f"{name.upper()} m={m} — LP improvement timeline")
        ax.legend(fontsize=9)
        plt.tight_layout()
        out = output_dir / "plots" / f"lp_timeline_{name}_{m}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved {out}")


def _plot_time_to_best(results: List[dict], output_dir: Path) -> None:
    """
    Scatter plot of time-to-best-LP (seconds) vs final BKS gap (%) for all
    positive-reference (instance, m) pairs.

    Colour encodes the number of runways m.  The plot shows whether harder
    instances (larger gaps) also take longer to find their best solution, and
    whether specific runway counts are systematically harder.
    """
    if not _MPL: return
    pos = [r for r in results
           if r["opt"] is not None and r["opt"]>0
           and not math.isinf(r["sa_lp"])
           and r.get("t_best_lp") is not None]
    if len(pos) < 2: return

    ts   = [r["t_best_lp"] for r in pos]
    gaps = [100*(r["sa_lp"]-r["opt"])/r["opt"] for r in pos]
    ms   = [r["m"] for r in pos]
    m_vals = sorted(set(ms))
    cmap   = plt.get_cmap("tab10")
    colours= {mv: cmap(i/max(len(m_vals)-1,1)) for i,mv in enumerate(m_vals)}

    fig, ax = plt.subplots(figsize=(7, 5))
    with plt.rc_context(_PLOT_STYLE):
        for mv in m_vals:
            idx = [i for i,r in enumerate(pos) if r["m"]==mv]
            ax.scatter([ts[i] for i in idx], [gaps[i] for i in idx],
                       s=60, color=colours[mv], label=f"m={mv}", alpha=0.85, edgecolors="white")
        # Annotate outliers (gap > 5%)
        for r, t, g in zip(pos, ts, gaps):
            if abs(g) > 5:
                ax.annotate(f"{r['name']}\nm={r['m']}",
                            (t, g), fontsize=7, ha="left", va="bottom",
                            xytext=(4, 4), textcoords="offset points")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Time to best LP (s)")
        ax.set_ylabel("Final gap to BKS (%)")
        ax.set_title("MR-SA v3 — Time to best LP vs BKS gap")
        ax.legend(title="Runways", fontsize=9)
        plt.tight_layout()
        out = output_dir / "plots" / "time_to_best.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved {out}")


def _plot_elite_pool(result: dict, output_dir: Path) -> None:
    """
    Horizontal bar chart of LP objectives for all elite pool solutions of one
    (instance, m) job, sorted by rank (rank 1 = best).

    A vertical dashed line shows the BKS reference when available.
    The ★ next to any bar indicates a solution below the BKS reference.
    """
    if not _MPL: return
    elite = result.get("elite_solutions", [])
    if len(elite) < 2: return
    name = result["name"]; m = result["m"]
    ref  = result.get("opt")

    lp_vals = [lp for lp,_ in elite[:20]]   # cap at 20 for readability
    ranks   = list(range(1, len(lp_vals)+1))

    fig, ax = plt.subplots(figsize=(6, max(3, len(lp_vals)*0.28)))
    with plt.rc_context(_PLOT_STYLE):
        colours = ["#D65F5F" if (ref and ref>0 and _is_new_bks(lp, ref))
                   else "#4878CF" for lp in lp_vals]
        bars = ax.barh(ranks, lp_vals, color=colours, alpha=0.85, edgecolor="white")
        if ref is not None and ref > 0:
            ax.axvline(ref, color="goldenrod", linewidth=1.2, linestyle="--",
                       label=f"BKS ({ref:.2f})")
            ax.legend(fontsize=9)
        ax.set_yticks(ranks); ax.set_yticklabels([f"Rank {r}" for r in ranks], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("LP objective")
        ax.set_title(f"{name.upper()} m={m} — Elite pool LP distribution")
        # Add value labels
        for bar, lp in zip(bars, lp_vals):
            ax.text(bar.get_width()*1.002, bar.get_y()+bar.get_height()/2,
                    f"{lp:.2f}", va="center", fontsize=7.5)
        plt.tight_layout()
        out = output_dir / "plots" / f"elite_pool_{name}_{m}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved {out}")


def generate_plots(results: List[dict], output_dir: Path) -> None:
    """
    Generate all plots for a completed batch run.

    Global plots (one file):
      gap_summary.png  — seed vs final gap across all instances.
      time_to_best.png — time-to-best-LP scatter vs gap.

    Per-job plots (one file per (instance, m)):
      convergence_{inst}_{m}.png — proxy history.
      lp_timeline_{inst}_{m}.png — LP improvement vs wall time.
      elite_pool_{inst}_{m}.png  — elite pool LP distribution.

    Parameters
    ----------
    results : list of dict — per-job result dicts from _run_one_mr.
    output_dir : Path
    """
    if not _MPL:
        print("  [plots] matplotlib not available — skipping.")
        return
    _ensure_dirs(output_dir)
    _plot_gap_summary(results, output_dir)
    _plot_time_to_best(results, output_dir)
    for r in results:
        _plot_convergence(r, output_dir)
        _plot_lp_timeline(r, output_dir)
        _plot_elite_pool(r, output_dir)


# ═════════════════════════════════════════════════════════════════════════════
#   §20  ENTRY POINT HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _run_one_mr(fp: str, m: int, seed: int = 0) -> dict:
    """
    Execute one (instance, runway-count) job and return a complete result dict.

    Workflow
    --------
    1. Parse instance file.
    2. Look up TC-RBI weights from PARAM_BANK (warn and use defaults if absent).
    3. Resolve SA parameters: SA_PARAM_BANK → Optuna TPE → MRSAParams().
    4. Estimate seed LP for adaptive time budget calculation.
    5. Run ms_mr_sa (SA chains + VND + path relinking).
    6. Return result dict.

    Parameters
    ----------
    fp   : str  — path to the instance file.
    m    : int  — number of runways.
    seed : int  — base random seed (default 0).

    Returns
    -------
    dict with keys:
        name, n, m, seed_lp, sa_lp, opt, feasible, time, t_best_lp,
        p_sa, p_sa_tuned, best_seqs, elite_solutions, job_lp_timeline,
        elite_pool_size, relinking_improved, violations, output.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        inst   = load_instance(fp)
        params = PARAM_BANK.get((inst.name,m), _DEFAULT)
        if (inst.name,m) not in PARAM_BANK:
            print(f"  [WARN] ({inst.name}, m={m}) not in PARAM_BANK — using defaults.")

        p_sa_tuned = False
        p_sa = SA_PARAM_BANK.get((inst.name,m))
        if p_sa is None and RUN_SA_OPTUNA:
            n_t = _sa_n_trials(inst.n, SA_N_TRIALS_BASE)
            N_t = _sa_n_iter_tune(inst.n)
            print(f"\n  [SA Optuna] {inst.name.upper()} m={m} → "
                  f"{n_t} trials  (N_iter={N_t}, {SA_N_OPTUNA_JOBS} thread(s)) ...")
            t_opt = time.perf_counter()
            p_sa  = optimize_sa_params(inst,m,params,n_trials=n_t,
                                        seed=SA_OPTUNA_SEED,n_jobs=SA_N_OPTUNA_JOBS)
            t_opt = time.perf_counter()-t_opt
            print(f"  [SA Optuna] done in {t_opt:.1f}s  best: {p_sa}")
            p_sa_tuned = True
        if p_sa is None:
            p_sa = MRSAParams()

        # Adaptive time budget: evaluate base LP before SA
        base_seqs,_ = ramp_rbi(inst,m,params)
        base_lp,_,base_feas,_ = stage2_lp_objective(base_seqs,inst)
        seed_lp_est = base_lp if base_feas else math.inf
        bks = KNOWN_OPTIMA.get(inst.name,{}).get(m)
        job_t_limit = _adaptive_t_limit(inst.n,m,seed_lp_est,bks)
        print(f"  Adaptive T_LIMIT: {job_t_limit:.0f}s  (seed_LP={seed_lp_est:.2f}  BKS={bks})")

        t0 = time.perf_counter()
        best_seqs, best_lp, stats = ms_mr_sa(
            inst,m,params,p_sa=p_sa,n_chains=N_CHAINS,t_limit=job_t_limit,seed=seed)
        elapsed  = time.perf_counter()-t0
        seed_lp  = min(stats['seed_lps']) if stats['seed_lps'] else math.inf
        feasible = stats['final_feas']

        # Feasibility violations for verification file
        feas_e, viol_e, _, _ = verify_and_exact_obj(best_seqs, inst)

        print_mr_result(inst,m,best_seqs,best_lp,elapsed,
                        stats['seed_lps'],params,p_sa,stats)

    opt = KNOWN_OPTIMA.get(inst.name,{}).get(m)
    return dict(
        name=inst.name, n=inst.n, m=m,
        seed_lp=seed_lp, sa_lp=best_lp,
        opt=opt, feasible=feasible, time=elapsed,
        t_best_lp=stats.get("t_best_lp", 0.0),
        p_sa=p_sa, p_sa_tuned=p_sa_tuned,
        best_seqs=best_seqs,
        elite_solutions=stats.get("elite_solutions",[]),
        job_lp_timeline=stats.get("job_lp_timeline",[]),
        elite_pool_size=stats.get("elite_pool_size",0),
        relinking_improved=stats.get("relinking_improved",False),
        history=stats.get("history",[]),
        alpha_history=stats.get("alpha_history",[]),
        violations=viol_e,
        output=buf.getvalue(),
    )


def main() -> None:
    """
    Entry point for MR-SA v3.

    In BATCH_MODE, discovers all airland*.txt in FOLDER and submits one job
    per (instance, runway-count) pair to N_WORKERS workers.  Results are
    printed as each future completes; a summary table follows.  If
    SAVE_RESULTS is True, all output files are written to OUTPUT_DIR.
    If SAVE_PLOTS is True, all plots are generated in OUTPUT_DIR/plots/.

    In single-file mode, runs all configured runway counts for INSTANCE_PATH
    sequentially and applies the same save/plot logic.
    """
    print("=" * 74)
    print("  MR-SA v3  —  SA + VND + Elite Pool + Path Relinking")
    print(f"  Workers      : {N_WORKERS} processes | {N_CHAINS} chains/job")
    print(f"  T_LIMIT      : {T_LIMIT:.0f}s (adaptive, max {MAX_T_LIMIT:.0f}s)")
    print(f"  Elite pool   : max {ELITE_POOL_MAX} solutions, min diversity {ELITE_MIN_DIV}")
    print(f"  Output dir   : {OUTPUT_DIR}")
    print(f"  Save results : {SAVE_RESULTS}  |  Save plots: {SAVE_PLOTS}")
    nb_str  = "Numba JIT"   if _NUMBA    else "no Numba"
    gpu_str = "PyTorch GPU" if _GPU_AVAIL else "no GPU"
    mpl_str = "matplotlib"  if _MPL      else "no matplotlib"
    opt_str = (f"SA Optuna ON ({SA_N_TRIALS_BASE} base trials, "
               f"{SA_N_OPTUNA_JOBS} threads)" if RUN_SA_OPTUNA else "SA Optuna OFF")
    print(f"  Accel        : {nb_str}, {gpu_str}, {mpl_str}")
    print(f"  SA tuning    : {opt_str}")
    print("=" * 74)

    if BATCH_MODE:
        folder = Path(FOLDER)
        files  = sorted(folder.glob("airland*.txt"))
        if not files:
            print(f"No airland*.txt files found in {folder.resolve()}"); return
        jobs = [(str(fp),m) for fp in files
                for m in INSTANCE_RUNWAYS.get(fp.stem.lower(),[1])]
        print(f"  Submitting {len(jobs)} jobs to {N_WORKERS} workers...\n")
        results = []
        with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=_MP_CTX) as ex:
            futs = {ex.submit(_run_one_mr,fp,m):(fp,m) for fp,m in jobs}
            for fut in as_completed(futs):
                fp,m = futs[fut]
                try:
                    r = fut.result()
                    results.append(r)
                    print(r["output"], end="")
                    bks_tag  = " ★NEW BKS★" if _is_new_bks(r["sa_lp"],r["opt"]) else ""
                    tune_tag = " [tuned]"   if r.get("p_sa_tuned") else ""
                    print(f"  ↳ {Path(fp).stem:<12} m={m}  "
                          f"seed={r['seed_lp']:.2f}  SA={r['sa_lp']:.2f}  "
                          f"gap={_gap_str(r['sa_lp'],r['opt'])}  "
                          f"{'✓' if r['feasible'] else '✗'}  "
                          f"t_best={r.get('t_best_lp',0):.1f}s  "
                          f"({r['time']:.1f}s){tune_tag}{bks_tag}")
                except Exception as exc:
                    print(f"  ERROR {Path(fp).stem} m={m}: {exc}")
        print_summary_table(results)
        if SAVE_RESULTS: save_run_results(results, OUTPUT_DIR)
        if SAVE_PLOTS:   generate_plots(results, OUTPUT_DIR)

    else:
        fp  = Path(INSTANCE_PATH)
        cfg = INSTANCE_RUNWAYS.get(fp.stem.lower(),[1])
        res = []
        for m in cfg:
            r = _run_one_mr(str(fp),m)
            print(r["output"], end="")
            bks_tag = " ★NEW BKS★" if _is_new_bks(r["sa_lp"],r["opt"]) else ""
            print(f"  ↳ {fp.stem:<12} m={m}  seed={r['seed_lp']:.2f}  "
                  f"SA={r['sa_lp']:.2f}  gap={_gap_str(r['sa_lp'],r['opt'])}  "
                  f"{'✓' if r['feasible'] else '✗'}  "
                  f"t_best={r.get('t_best_lp',0):.1f}s  ({r['time']:.1f}s){bks_tag}")
            res.append(r)
        if len(cfg)>1: print_summary_table(res)
        if SAVE_RESULTS: save_run_results(res, OUTPUT_DIR)
        if SAVE_PLOTS:   generate_plots(res, OUTPUT_DIR)


if __name__ == "__main__":
    main()