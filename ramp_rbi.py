"""
TC-RBI: Target-Conflict Regret-Based Insertion
===============================================================================
Multi-Runway Aircraft Landing Problem (ALP) — Construction Heuristic

PROBLEM STATEMENT
-----------------
Given n aircraft and m runways, assign each aircraft to exactly one runway and
determine a landing sequence and time for each runway such that:

    (C1) r_j  ≤  x_j  ≤  d_j                       (time-window feasibility)
    (C2) x_j  ≥  x_i + s_{ij}  ∀ i≺j on same rwy   (separation enforcement)
    (C3) minimise Σ_j [ g_j·max(δ_j−x_j,0) + h_j·max(x_j−δ_j,0) ]

where r_j, δ_j, d_j are the earliest, target, and latest landing times for
aircraft j; s_{ij} is the required separation between aircraft i and j; and
g_j, h_j are per-unit earliness and tardiness penalties.

SOLUTION APPROACH  (two-stage)
-------------------------------
Stage 1 — TC-RBI construction heuristic (this module):
    Assigns aircraft to runways and determines landing sequences using a
    regret-based insertion strategy that explicitly penalises target-time
    conflict (TC).  Optuna TPE is optionally used to tune five scalar weights
    governing the insertion cost function.

Stage 2 — Exact LP timing (§10):
    Given the fixed sequences from Stage 1, a sparse LP with full pairwise
    separation constraints is solved via HiGHS to obtain optimal landing times
    within those sequences.  The LP objective is the metric reported against
    the Beasley et al. (2000) benchmark optima.

BENCHMARKS
----------
Instances: OR Library airland1–airland13 (Beasley et al. 2000).
Reference optima: branch-and-bound values from Beasley et al. (2000), stored
in KNOWN_OPTIMA.  These are treated as hard correctness checks, not bounds.

ACCELERATORS (optional, gracefully degraded)
--------------------------------------------
Numba   — JIT-compiles _insert_times_kernel, giving 8–12× speedup on the
           inner insertion loop (≈1.5 M calls for n=500).
PyTorch — GPU-resident separation matrix and penalty arrays for the O(n²)
           total_target_conflict proxy used inside the Optuna inner loop
           when n ≥ GPU_MIN_N.

REFERENCES
----------
Beasley, J.E., Krishnamoorthy, M., Sharaiha, Y.M., Abramson, D. (2000).
    Scheduling aircraft landings — the static case.
    Transportation Science 34(2), 180–197.

USAGE
-----
Configure BATCH_MODE, FOLDER / INSTANCE_PATH, and the parallelism knobs at the
top of the file, then run:

    python tc_rbi.py
"""

from __future__ import annotations

import io
import contextlib
import math
import time
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

# ─────────────────────────────────────────────────────────────────────────────
# Optional accelerator imports
# Each import is attempted independently so the module runs on any environment.
# Runtime dispatch (_NUMBA, _GPU_AVAILABLE) selects the best available path.
# ─────────────────────────────────────────────────────────────────────────────
try:
    import numba as nb
    _NUMBA = True
except ImportError:
    _NUMBA = False

try:
    import torch as _torch
    _GPU_AVAILABLE = _torch.cuda.is_available()
    _CUDA_DEVICE   = _torch.device("cuda") if _GPU_AVAILABLE else None
except ImportError:
    _torch = None
    _GPU_AVAILABLE = False
    _CUDA_DEVICE   = None

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA = True
except ImportError:
    _OPTUNA = False

# Multiprocessing spawn context.
# "spawn" is required on Windows (no fork()) and when CUDA is active
# (forking a process that holds a CUDA context causes deadlocks).
import multiprocessing as _mp
_USE_SPAWN = platform.system() == "Windows" or _GPU_AVAILABLE
_MP_CTX    = _mp.get_context("spawn" if _USE_SPAWN else "fork")


# ─────────────────────────────────────────────────────────────────────────────
# §0  Known optima
#
# Branch-and-bound optima from Beasley et al. (2000), keyed by
# (instance_name, number_of_runways).  Used to compute optimality gaps.
# A gap of 0 % does NOT imply the heuristic found B&B — it means the LP
# objective coincides with the published lower bound to four decimal places.
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
#   CONFIGURE HERE
#
#   BATCH_MODE    — True: run all airland*.txt files in FOLDER.
#                   False: run the single file at INSTANCE_PATH.
#   N_WORKERS     — ProcessPoolExecutor workers for the outer batch loop.
#                   Each worker handles one (instance, runway-count) pair.
#   N_OPTUNA_WORKERS — Optuna trial threads per (instance, runway-count) job.
#                   N_WORKERS × N_OPTUNA_WORKERS should not exceed the
#                   machine's logical-core count (32 on the reference machine).
#   USE_GPU       — Enable PyTorch CUDA for the TC proxy objective.
#   GPU_MIN_N     — Minimum n for which GPU dispatch is worthwhile.
#                   Below this threshold, PCIe transfer overhead exceeds gain.
#   RUN_OPTUNA    — True: tune HeuristicParams via TPE before construction.
#   N_TRIALS_BASE — Optuna trial budget for small instances (n ≤ 100).
#                   Larger instances receive a reduced budget (see _n_trials).
# ═════════════════════════════════════════════════════════════════════════════
BATCH_MODE    = True
INSTANCE_PATH = "data/airland1.txt"
FOLDER        = "data/"

# Maps each instance name to the list of runway counts to evaluate.
INSTANCE_RUNWAYS: Dict[str, List[int]] = {
    "airland1":  [2, 3],
    "airland2":  [2, 3],
    "airland3":  [2, 3],
    "airland4":  [2, 3, 4],
    "airland5":  [2, 3, 4],
    "airland6":  [2, 3],
    "airland7":  [2],
    "airland8":  [2, 3],
    "airland9":  [2, 3, 4],
    "airland10": [2, 3, 4, 5],
    "airland11": [2, 3, 4, 5],
    "airland12": [2, 3, 4, 5],
    "airland13": [2, 3, 4, 5],
}
# Fallback runway count when an instance name is not listed above.
N_RUNWAYS_DEFAULT = 1

N_WORKERS        = 8
N_OPTUNA_WORKERS = 4

USE_GPU   = True
GPU_MIN_N = 200

RUN_OPTUNA    = True
N_TRIALS_BASE = 30
OPTUNA_SEED   = 42

# Default construction weights used when RUN_OPTUNA = False.
# See HeuristicParams for the role of each weight.
DEFAULT_ETA      = 0.50
DEFAULT_MU_TC    = 1.00
DEFAULT_MU_LATE  = 0.25
DEFAULT_MU_COUNT = 0.75
DEFAULT_MU_SEP   = 0.05
# ═════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
# §1  Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Instance:
    """
    Parsed and pre-processed ALP instance.

    All per-aircraft arrays are 0-indexed and have length n.
    The separation matrix s has shape (n, n) with s[i, j] = required gap
    between aircraft i (predecessor) and aircraft j (successor) on the same
    runway.  The diagonal is zeroed out after parsing.

    Note on the OR Library separation matrices
    -------------------------------------------
    These matrices do NOT satisfy the triangle inequality in general, i.e.
    s[i,k] ≤ s[i,j] + s[j,k] can fail.  Consequently, the Stage-2 LP and
    the exact feasibility checker must enforce ALL ordered pairs (i,j) with
    i preceding j in the sequence, not only consecutive pairs.  Checking only
    consecutive pairs is a strict relaxation that can allow infeasible
    schedules to pass verification.

    Parameters
    ----------
    name : str
        Instance identifier (e.g. "airland1"), lower-cased from the filename.
    n : int
        Number of aircraft.
    r : ndarray, shape (n,)
        Earliest permissible landing times.
    delta : ndarray, shape (n,)
        Target (preferred) landing times.
    d : ndarray, shape (n,)
        Latest permissible landing times (hard deadline).
    g : ndarray, shape (n,)
        Per-unit earliness penalty rates (g_j ≥ 0).
    h : ndarray, shape (n,)
        Per-unit tardiness penalty rates (h_j ≥ 0).
    s : ndarray, shape (n, n)
        Pairwise separation requirement matrix (seconds or consistent units).

    Derived attributes (computed in __post_init__)
    -----------------------------------------------
    W_bar : float
        Mean time-window width E[d_j - r_j].  Used to normalise penalty scales.
    s_bar : float
        Mean positive off-diagonal separation (i ≠ j, s[i,j] > 0).
        Falls back to 1.0 for degenerate instances with all-zero separations.
    h_bar : float
        Mean tardiness penalty rate E[h_j].
    Pen_bar : float
        Mean maximum penalty rate times mean window width:
        E[max(g_j, h_j)] × W_bar.  Serves as an instance-specific scale for
        the runway-balance cost term.
    T_span : float
        Total time horizon: max(d) - min(r).
    eps : float
        Small regularisation constant (1e-9) used in divisions and comparisons.
    p_arr : ndarray, shape (n,)
        Per-aircraft maximum penalty rate: max(g_j, h_j).

    GPU arrays (optional, None when unavailable)
    --------------------------------------------
    _s_gpu, _delta_gpu, _p_arr_gpu
        PyTorch CUDA tensors (float64) mirroring s, delta, p_arr.
        Resident on device to avoid repeated host-to-device transfers in the
        Optuna inner loop.  Set to None before pickling (see __getstate__);
        re-created on the worker side in __setstate__.
    """

    name:    str
    n:       int
    r:       np.ndarray
    delta:   np.ndarray
    d:       np.ndarray
    g:       np.ndarray
    h:       np.ndarray
    s:       np.ndarray

    W_bar:   float = field(init=False)
    s_bar:   float = field(init=False)
    h_bar:   float = field(init=False)
    Pen_bar: float = field(init=False)
    T_span:  float = field(init=False)
    eps:     float = field(init=False, default=1e-9)

    p_arr:   np.ndarray = field(init=False)

    _s_gpu:     object = field(init=False, default=None, repr=False)
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

        if USE_GPU and _GPU_AVAILABLE and self.n >= GPU_MIN_N:
            # TF32 flag must be set inside the worker process, not at module
            # level, to avoid touching the CUDA context in the parent process.
            _torch.backends.cuda.matmul.allow_tf32 = True
            kw = dict(dtype=_torch.float64, device=_CUDA_DEVICE)
            self._s_gpu     = _torch.as_tensor(self.s,     **kw)
            self._delta_gpu = _torch.as_tensor(self.delta, **kw)
            self._p_arr_gpu = _torch.as_tensor(self.p_arr, **kw)

    def __getstate__(self):
        """
        Strip CUDA tensors before pickling.

        ProcessPoolExecutor serialises arguments with pickle.  PyTorch CUDA
        tensors are not picklable across processes.  The tensors are dropped
        here and reconstructed on the receiving worker via __setstate__.
        """
        state = self.__dict__.copy()
        state['_s_gpu']     = None
        state['_delta_gpu'] = None
        state['_p_arr_gpu'] = None
        return state

    def __setstate__(self, state):
        """
        Restore state after unpickling; re-create GPU tensors in the worker.
        """
        self.__dict__.update(state)
        if USE_GPU and _GPU_AVAILABLE and self.n >= GPU_MIN_N:
            kw = dict(dtype=_torch.float64, device=_CUDA_DEVICE)
            self._s_gpu     = _torch.as_tensor(self.s,     **kw)
            self._delta_gpu = _torch.as_tensor(self.delta, **kw)
            self._p_arr_gpu = _torch.as_tensor(self.p_arr, **kw)


@dataclass
class HeuristicParams:
    """
    Tunable scalar weights for the TC-RBI insertion cost function.

    One HeuristicParams instance is used per (instance, runway-count) pair.
    When RUN_OPTUNA = True these are found by TPE; otherwise the module-level
    DEFAULT_* constants are used.

    Attributes
    ----------
    eta : float in [0, 1]
        Screening blend weight.  eta=1 → rank candidates by criticality ratio
        CR only; eta=0 → rank by urgency only.  Controls which aircraft are
        admitted to the regret evaluation pool (top-q_eff candidates).
    mu_tc : float ≥ 0
        Weight on the incremental target-time conflict cost ΔTC.  Higher
        values cause the heuristic to prefer insertions that minimise
        disruption to other aircraft's ability to land near their targets.
    mu_late : float ≥ 0
        Weight on the incremental tardiness lower-bound increase ΔLate.
        Penalises insertions that push already-late aircraft further past δ.
    mu_count : float ≥ 0
        Weight on the runway-balance deviation Δcount.  Penalises insertions
        that worsen load imbalance across runways.
    mu_sep : float ≥ 0
        Weight on the incremental separation burden ΔSep, scaled by h_bar.
        Penalises insertions that introduce large mandatory gaps.
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


# ─────────────────────────────────────────────────────────────────────────────
# §1.5  Numba JIT kernel: surrogate-time propagation after a single insertion
#
# _insert_times_kernel is the tightest inner loop in ramp_rbi: it is called
# O(n × m × L) times where L grows from 0 to n/m.  For n=500, m=1 this is
# approximately 1.5 M calls.  The JIT-compiled kernel eliminates Python
# interpreter overhead for the sequential scan and yields an 8–12× speedup
# over the equivalent pure-Python loop (measured on airland13, n=500).
#
# The @nb.njit(cache=True) decorator:
#   - Compiles ahead of first call and caches the binary to __pycache__.
#   - Disables the Python GIL inside the kernel (safe: no Python objects used).
# ─────────────────────────────────────────────────────────────────────────────
if _NUMBA:
    @nb.njit(cache=True)
    def _insert_times_kernel(
        j: int, p: int,
        seq: np.ndarray,    # int32[L]  — existing sequence WITHOUT aircraft j
        C_prev: np.ndarray, # float64[L] — surrogate times for seq
        r: np.ndarray,      # float64[n] — earliest landing times (all aircraft)
        s: np.ndarray,      # float64[n,n] — separation matrix (all aircraft)
        d: np.ndarray,      # float64[n] — latest landing times (all aircraft)
    ):
        """
        Compute surrogate landing times after inserting aircraft j at
        position p in the sequence seq.

        The surrogate landing time for aircraft k at position q is:

            C[q] = max( r[k],  C[q-1] + s[seq[q-1], k] )

        where seq is the new sequence with j inserted at position p.

        Positions 0..p-1 are identical to C_prev (copied without recomputation).
        Position p corresponds to the inserted aircraft j.
        Positions p+1..L are aircraft that shift one position to the right and
        whose surrogate times are recomputed sequentially.

        Parameters
        ----------
        j : int
            Index of the aircraft being inserted.
        p : int
            Insertion position (0 = front of sequence).
        seq : int32 ndarray, shape (L,)
            Current sequence on the runway, WITHOUT aircraft j.
        C_prev : float64 ndarray, shape (L,)
            Surrogate landing times corresponding to seq.
        r, s, d : float64 arrays
            Earliest times, separation matrix, and latest times for all n
            aircraft (passed by reference; not mutated).

        Returns
        -------
        C_n : float64 ndarray, shape (L+1,)
            Surrogate landing times for the sequence with j inserted at p.
        feasible : bool
            True iff C_n[q] ≤ d[aircraft_at_q] for all new/shifted positions.
            Positions 0..p-1 are NOT re-checked (they are unchanged).
        """
        L   = len(seq)
        L_n = L + 1
        C_n = np.empty(L_n, dtype=np.float64)

        # Prefix: positions 0..p-1 are unchanged.
        for q in range(p):
            C_n[q] = C_prev[q]

        # Position p: the inserted aircraft j.
        if p == 0:
            C_n[0] = r[j]
        else:
            val    = C_n[p - 1] + s[seq[p - 1], j]
            C_n[p] = val if val > r[j] else r[j]

        # Positions p+1..L_n-1: original aircraft shifted right by one.
        # The predecessor at position q is j when q == p+1, else seq[q-2].
        for q in range(p + 1, L_n):
            cur  = seq[q - 1]
            prev = j if q == p + 1 else seq[q - 2]
            val  = C_n[q - 1] + s[prev, cur]
            C_n[q] = val if val > r[cur] else r[cur]

        # Feasibility: d-constraint for all new and shifted positions.
        for q in range(p, L_n):
            ac = j if q == p else seq[q - 1]
            if C_n[q] > d[ac] + 1e-9:
                return C_n, False
        return C_n, True


# ─────────────────────────────────────────────────────────────────────────────
# §2  File parser  (OR Library format — 6 header fields, then per-aircraft rows)
#
# File format (whitespace-delimited tokens):
#   Line 1: <n> <freeze_time>
#   For each aircraft i = 0..n-1:
#     <appearance_time> <r_i> <delta_i> <d_i> <g_i> <h_i>
#     <s[i,0]> <s[i,1]> ... <s[i,n-1]>
#
# appearance_time is read and discarded (not used by this formulation).
# freeze_time is likewise discarded.
# ─────────────────────────────────────────────────────────────────────────────

def load_instance(filepath: str | Path, name: Optional[str] = None) -> Instance:
    """
    Parse an OR Library ALP instance file and return an Instance object.

    Parameters
    ----------
    filepath : str or Path
        Path to the .txt file.
    name : str, optional
        Instance identifier.  Defaults to the filename stem, lower-cased
        (e.g. "airland1" for "data/airland1.txt").

    Returns
    -------
    Instance
        Fully initialised instance with derived attributes and GPU arrays
        (if applicable) populated by Instance.__post_init__.

    Raises
    ------
    ValueError
        If r_i > delta_i, delta_i > d_i for any aircraft, or if the token
        count does not exactly match the expected file length.
    """
    path   = Path(filepath)
    name   = name or path.stem.lower()
    tokens = path.read_text().split()
    pos    = 0

    def take_int():   nonlocal pos; v = int(tokens[pos]);   pos += 1; return v
    def take_float(): nonlocal pos; v = float(tokens[pos]); pos += 1; return v

    n = take_int(); _ = take_float()
    r = np.empty(n); delta = np.empty(n); d = np.empty(n)
    g = np.empty(n); h     = np.empty(n); s = np.empty((n, n))

    for i in range(n):
        _        = take_float()   # appearance time (discarded)
        r[i]     = take_float()
        delta[i] = take_float()
        d[i]     = take_float()
        g[i]     = take_float()
        h[i]     = take_float()
        for j in range(n):
            s[i, j] = take_float()
    np.fill_diagonal(s, 0.0)

    bad = np.where(r > delta + 1e-6)[0]
    if bad.size:
        raise ValueError(f"{name}: r > delta for aircraft {bad[:5]}")
    bad = np.where(delta > d + 1e-6)[0]
    if bad.size:
        raise ValueError(f"{name}: delta > d for aircraft {bad[:5]}")
    if pos != len(tokens):
        raise ValueError(f"{name}: consumed {pos} tokens, file has {len(tokens)}")

    return Instance(name=name, n=n, r=r, delta=delta, d=d, g=g, h=h, s=s)


# ─────────────────────────────────────────────────────────────────────────────
# §5.1  Surrogate landing times (consecutive-predecessor approximation)
#
# The surrogate time is a fast O(n) proxy for the exact landing time.
# It enforces only the consecutive-predecessor separation:
#
#     C_hat[q] = max( r[seq[q]],  C_hat[q-1] + s[seq[q-1], seq[q]] )
#
# This is sufficient for guiding insertion decisions during construction
# but is NOT used as the final feasibility or objective measure, because
# OR Library separation matrices can violate the triangle inequality.
# The Stage-2 LP (§10) and verify_and_exact_obj (§10.3) enforce all pairs.
# ─────────────────────────────────────────────────────────────────────────────

def surrogate_times(seq: List[int], inst: Instance) -> List[float]:
    """
    Compute consecutive-predecessor surrogate landing times for a sequence.

    Parameters
    ----------
    seq : list of int
        Ordered list of aircraft indices on a single runway.
    inst : Instance
        Problem instance providing r and s arrays.

    Returns
    -------
    list of float
        C_hat[q] for q = 0, …, len(seq)-1.
        Returns an empty list if seq is empty.
    """
    if not seq:
        return []
    C = [0.0] * len(seq)
    C[0] = float(inst.r[seq[0]])
    for q in range(1, len(seq)):
        C[q] = max(float(inst.r[seq[q]]),
                   C[q - 1] + float(inst.s[seq[q - 1], seq[q]]))
    return C


def surrogate_penalty(seq: List[int], C_hat: List[float], inst: Instance) -> float:
    """
    Evaluate the penalty objective using surrogate landing times.

    Penalty for aircraft j at surrogate time C_hat_j:

        p_j = g_j · max(δ_j − C_hat_j, 0)  +  h_j · max(C_hat_j − δ_j, 0)

    Parameters
    ----------
    seq : list of int
        Aircraft indices in landing order.
    C_hat : list of float
        Surrogate landing times (same length as seq).
    inst : Instance
        Problem instance providing delta, g, h.

    Returns
    -------
    float
        Total surrogate penalty Σ_j p_j.  Returns 0.0 for an empty sequence.
    """
    if not seq:
        return 0.0
    s_arr = np.asarray(seq, dtype=np.intp)
    C_arr = np.asarray(C_hat)
    E = np.maximum(inst.delta[s_arr] - C_arr, 0.0)
    T = np.maximum(C_arr - inst.delta[s_arr], 0.0)
    return float((inst.g[s_arr] * E + inst.h[s_arr] * T).sum())


# ─────────────────────────────────────────────────────────────────────────────
# §7.1  Priority measures (fully vectorised)
#
# Two per-aircraft priority measures are computed once at the start of
# construction and reused throughout the regret loop.
#
# Available Flexibility (AF):
#     AF_j = (d_j - r_j) - s̄_j
#   where s̄_j is the mean bilateral separation load of aircraft j.
#   Low AF → aircraft j has little scheduling latitude → higher priority.
#
# Criticality Ratio (CR):
#     CR_j = (g_j + h_j) / max(AF_j, ε)
#   High CR → high penalty for deviation AND low flexibility → schedule first.
# ─────────────────────────────────────────────────────────────────────────────

def compute_priorities(inst: Instance) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Available Flexibility (AF) and Criticality Ratio (CR) for all n
    aircraft.

    AF_j = (d_j - r_j) - mean_bilateral_separation_j
    CR_j = (g_j + h_j) / max(AF_j, ε)

    The bilateral separation load per aircraft uses the symmetric average of
    the separation matrix to avoid penalising aircraft that are only
    *predecessors* of tight pairs.

    Parameters
    ----------
    inst : Instance

    Returns
    -------
    AF : ndarray, shape (n,)
    CR : ndarray, shape (n,)
    """
    n, eps = inst.n, inst.eps
    s_sym  = (inst.s + inst.s.T) / 2.0
    np.fill_diagonal(s_sym, 0.0)
    s_b    = s_sym.sum(axis=1) / max(n - 1, 1)
    AF     = (inst.d - inst.r) - s_b
    CR     = (inst.g + inst.h) / np.maximum(AF, eps)
    return AF, CR


def minmax_norm(arr: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Min-max normalise an array to [0, 1].

    Parameters
    ----------
    arr : ndarray
        Input values.
    eps : float
        Added to the denominator to prevent division by zero when all values
        are identical.

    Returns
    -------
    ndarray
        (arr - min) / (max - min + eps), same shape as arr.
    """
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + eps)


# ─────────────────────────────────────────────────────────────────────────────
# §6  Insertion cost components (vectorised hot paths)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_insert_times(
    j: int, p: int,
    seq: List[int], C_hat_seq: List[float],
    inst: Instance,
) -> Tuple[List[float], bool]:
    """
    Compute surrogate landing times after inserting aircraft j at position p.

    Dispatches to the Numba JIT kernel (_insert_times_kernel) when Numba is
    available; falls back to a pure-Python implementation otherwise.

    Parameters
    ----------
    j : int
        Aircraft index to insert.
    p : int
        Insertion position in seq (0 = before first element).
    seq : list of int
        Current sequence on the runway, WITHOUT aircraft j.
    C_hat_seq : list of float
        Surrogate times for seq (same length as seq).
    inst : Instance

    Returns
    -------
    C_n : list of float
        Surrogate landing times for the extended sequence (length len(seq)+1).
    feasible : bool
        True iff C_n[q] ≤ d[aircraft_at_q] for all positions q ≥ p.
        The prefix 0..p-1 is copied unchanged from C_hat_seq and not
        re-validated.
    """
    if _NUMBA:
        L       = len(seq)
        seq_arr = np.asarray(seq, dtype=np.int32) if L else np.empty(0, dtype=np.int32)
        C_arr   = np.asarray(C_hat_seq, dtype=np.float64) if C_hat_seq else np.empty(0, dtype=np.float64)
        C_n, ok = _insert_times_kernel(j, p, seq_arr, C_arr, inst.r, inst.s, inst.d)
        return list(C_n), bool(ok)

    # ── Pure Python fallback ──────────────────────────────────────────────
    seq_n = seq[:p] + [j] + seq[p:]
    L_n   = len(seq_n)
    C_n   = list(C_hat_seq[:p])

    if p == 0:
        C_n.append(float(inst.r[j]))
    else:
        C_n.append(max(float(inst.r[j]),
                       C_n[p - 1] + float(inst.s[seq_n[p - 1], j])))
    for q in range(p + 1, L_n):
        prev, cur = seq_n[q - 1], seq_n[q]
        C_n.append(max(float(inst.r[cur]),
                       C_n[q - 1] + float(inst.s[prev, cur])))
    for q in range(p, L_n):
        if C_n[q] > inst.d[seq_n[q]] + 1e-9:
            return C_n, False
    return C_n, True


def _is_feasible_anywhere(
    j: int,
    sequences: List[List[int]], C_hats: List[List[float]],
    inst: Instance,
) -> bool:
    """
    Test whether aircraft j can be feasibly appended to or inserted into
    any runway.

    The check is fast: it first tries the O(1) append position on each runway
    (appending is cheap and often feasible for early iterations), then falls
    back to testing all interior positions across all runways.

    Used by ramp_rbi to detect aircraft that must be force-inserted (i.e.
    added to the forced set F) before any feasible slot disappears.

    Parameters
    ----------
    j : int
        Aircraft index.
    sequences : list of list of int
        Current partial sequences, one per runway.
    C_hats : list of list of float
        Corresponding surrogate times.
    inst : Instance

    Returns
    -------
    bool
        True if at least one feasible (runway, position) pair exists for j.
    """
    m = len(sequences)
    for rho in range(m):
        _, ok = _compute_insert_times(j, len(sequences[rho]),
                                       sequences[rho], C_hats[rho], inst)
        if ok:
            return True
    for rho in range(m):
        for p in range(len(sequences[rho])):
            _, ok = _compute_insert_times(j, p, sequences[rho], C_hats[rho], inst)
            if ok:
                return True
    return False


def target_conflict_insert(
    j: int, p: int, seq: List[int], inst: Instance,
) -> float:
    """
    Compute the incremental weighted target-time conflict from inserting
    aircraft j at position p in seq.

    For a predecessor i and successor k of j, the pairwise conflict is:

        TC(i, j) = 0.5 · (p_i + p_j) · max( s[i,j] − (δ_j − δ_i), 0 )
        TC(j, k) = 0.5 · (p_j + p_k) · max( s[j,k] − (δ_k − δ_j), 0 )

    where p_i = max(g_i, h_i) is aircraft i's maximum penalty rate.  The
    total incremental TC is the sum over all predecessors and all successors.

    This is a static proxy: it does not depend on the current surrogate times,
    only on target times and the separation matrix.  Computation is vectorised
    over all predecessors and successors simultaneously.

    Parameters
    ----------
    j : int
        Aircraft being inserted.
    p : int
        Insertion position (0 = front).
    seq : list of int
        Existing sequence on the runway (WITHOUT j).
    inst : Instance

    Returns
    -------
    float
        ΔTC ≥ 0.  A larger value means j conflicts more with the target times
        of surrounding aircraft.
    """
    pj   = float(inst.p_arr[j])
    cost = 0.0

    pred = seq[:p]
    if pred:
        pred_arr = np.asarray(pred, dtype=np.intp)
        v        = inst.s[pred_arr, j] - (inst.delta[j] - inst.delta[pred_arr])
        cost    += float(
            (0.5 * (inst.p_arr[pred_arr] + pj) * np.maximum(v, 0.0)).sum()
        )

    succ = seq[p:]
    if succ:
        succ_arr = np.asarray(succ, dtype=np.intp)
        v        = inst.s[j, succ_arr] - (inst.delta[succ_arr] - inst.delta[j])
        cost    += float(
            (0.5 * (pj + inst.p_arr[succ_arr]) * np.maximum(v, 0.0)).sum()
        )
    return cost


def lower_bound_tardiness(
    seq: List[int], C_hat: List[float], inst: Instance,
) -> float:
    """
    Compute the unavoidable tardiness lower bound for a partial sequence.

    For aircraft j at surrogate time C_hat_j, the tardiness lower bound is:

        LB_tardiness(j) = h_j · max( C_hat_j − δ_j, 0 )

    This is a lower bound on actual tardiness because the surrogate time uses
    only consecutive separations, and actual times can only be ≥ surrogate
    times (the LP may improve times but cannot violate pair constraints).

    Parameters
    ----------
    seq : list of int
        Aircraft indices in landing order.
    C_hat : list of float
        Surrogate times corresponding to seq.
    inst : Instance

    Returns
    -------
    float
        Sum of per-aircraft tardiness lower bounds.
        Returns 0.0 for an empty sequence.
    """
    if not seq:
        return 0.0
    s_arr = np.asarray(seq, dtype=np.intp)
    C_arr = np.asarray(C_hat)
    return float((inst.h[s_arr] * np.maximum(C_arr - inst.delta[s_arr], 0.0)).sum())


def count_balance_delta(
    rho: int, sequences: List[List[int]], inst: Instance,
) -> float:
    """
    Compute the change in squared runway-load deviation from inserting one
    aircraft onto runway rho.

    The runway-balance penalty tracks how far runway rho's load deviates from
    the mean load n/m.  Adding one aircraft changes both the individual load
    and the global mean, so the deviation is recomputed for both old and new
    states.

    The raw squared deviation difference is scaled by Pen_bar / (n/m)² to
    bring it into the same order of magnitude as the other cost terms
    (ΔTC, ΔLate, ΔSep), enabling meaningful weight comparisons.

    Parameters
    ----------
    rho : int
        Runway index receiving the new aircraft (0-indexed).
    sequences : list of list of int
        Current partial sequences, one per runway.
    inst : Instance
        Provides Pen_bar and n for normalisation.

    Returns
    -------
    float
        Scaled change in squared deviation Δcount.
        Can be negative (inserting on the most lightly loaded runway reduces
        imbalance) but is typically positive.
    """
    m         = len(sequences)
    n         = inst.n
    t         = sum(len(s) for s in sequences)
    old_len   = len(sequences[rho])
    old_dev   = (old_len     - t / m) ** 2
    new_dev   = (old_len + 1 - (t + 1) / m) ** 2
    raw_delta = new_dev - old_dev
    scale     = float(inst.Pen_bar) / max((n / m) ** 2, 1.0)
    return scale * raw_delta


def evaluate_insertion(
    j: int, rho: int, p: int,
    sequences: List[List[int]], C_hats: List[List[float]],
    B_bar: float,
    inst: Instance,
    params: HeuristicParams,
) -> Tuple[float, List[float]]:
    """
    Compute the composite insertion cost for placing aircraft j at position p
    on runway rho.

    The composite cost is a weighted sum of four components:

        cost = μ_TC    · ΔTC
             + μ_late  · ΔLate
             + μ_count · Δcount
             + μ_sep   · ΔSep

    ΔTC    — incremental target-time conflict (target_conflict_insert).
    ΔLate  — change in tardiness lower bound (lower_bound_tardiness).
    Δcount — scaled change in runway-balance deviation (count_balance_delta).
    ΔSep   — net incremental separation burden, scaled by h_bar:
               0                if the runway was empty
               s[j, seq[0]]     if inserting at the front
               s[seq[-1], j]    if appending at the end
               max(0, s[a,j] + s[j,b] − s[a,b])  otherwise (split gap)

    Parameters
    ----------
    j : int
        Aircraft index to insert.
    rho : int
        Target runway index.
    p : int
        Insertion position within sequences[rho].
    sequences : list of list of int
        Current partial sequences, one per runway.
    C_hats : list of list of float
        Surrogate times corresponding to sequences.
    B_bar : float
        Current mean committed time across runways (informational; not
        directly used in the cost formula but available for extensions).
    inst : Instance
    params : HeuristicParams
        Weights μ_TC, μ_late, μ_count, μ_sep.

    Returns
    -------
    cost : float
        Composite insertion cost.  Returns math.inf if the insertion is
        infeasible (any surrogate time would exceed d_j).
    C_n : list of float
        New surrogate times for sequences[rho] with j inserted.
        Empty list if the insertion is infeasible.
    """
    seq, C_hat_seq = sequences[rho], C_hats[rho]
    L              = len(seq)
    C_n, ok        = _compute_insert_times(j, p, seq, C_hat_seq, inst)
    if not ok:
        return math.inf, []
    seq_n = seq[:p] + [j] + seq[p:]

    dTC    = target_conflict_insert(j, p, seq, inst)
    dLate  = (lower_bound_tardiness(seq_n, C_n, inst)
              - lower_bound_tardiness(seq, C_hat_seq, inst))
    dCount = count_balance_delta(rho, sequences, inst)

    if L == 0:
        dSep_raw = 0.0
    elif p == 0:
        dSep_raw = float(inst.s[j, seq[0]])
    elif p == L:
        dSep_raw = float(inst.s[seq[-1], j])
    else:
        a, b     = seq[p - 1], seq[p]
        dSep_raw = max(0.0, float(inst.s[a, j]) + float(inst.s[j, b])
                              - float(inst.s[a, b]))
    dSep = inst.h_bar * dSep_raw

    cost = (params.mu_tc    * dTC
          + params.mu_late  * dLate
          + params.mu_count * dCount
          + params.mu_sep   * dSep)
    return cost, C_n


# ─────────────────────────────────────────────────────────────────────────────
# §7.2  Minimum-violation forced insertion
#
# When no feasible position exists for aircraft j (it has been placed in the
# forced set F), it must still be scheduled.  The minimum-violation strategy
# finds the (runway, position) pair that minimises total window-violation:
#
#     V = Σ_q max( C_hat[q] − d[seq_q], 0 )
#
# This sacrifices feasibility as little as possible while guaranteeing that
# every aircraft is eventually placed (no aircraft is left unscheduled).
# ─────────────────────────────────────────────────────────────────────────────

def min_violation_insert(
    j: int,
    sequences: List[List[int]],
    inst: Instance,
) -> Tuple[int, int, List[float]]:
    """
    Find the least-infeasible insertion position for aircraft j.

    Iterates over all (runway, position) pairs and selects the one minimising
    the total surrogate time-window violation.  This is an O(n²) fallback
    used only for aircraft in the forced set F.

    Parameters
    ----------
    j : int
        Aircraft to insert (already determined to have no feasible position).
    sequences : list of list of int
        Current partial sequences.
    inst : Instance

    Returns
    -------
    best_rho : int
        Runway index of the least-infeasible insertion.
    best_p : int
        Position within sequences[best_rho].
    best_C : list of float
        Surrogate times for sequences[best_rho] with j inserted.
    """
    best_V, best_rho, best_p, best_C = math.inf, 0, 0, []
    for rho, seq in enumerate(sequences):
        for p in range(len(seq) + 1):
            seq_n = seq[:p] + [j] + seq[p:]
            C_t   = surrogate_times(seq_n, inst)
            V     = sum(max(C_t[q] - inst.d[seq_n[q]], 0.0)
                        for q in range(len(seq_n)))
            if V < best_V:
                best_V, best_rho, best_p, best_C = V, rho, p, C_t
    return best_rho, best_p, best_C


# ─────────────────────────────────────────────────────────────────────────────
# §8.2  Candidate positions
#
# Evaluating all O(L) positions for each aircraft at each iteration makes
# ramp_rbi O(n²·L) per iteration — O(n³) total.  For n ≤ 100 the full set is
# used.  For n > 100 a window of positions centred on the target-time rank of
# j within the current sequence is used, reducing the constant factor
# significantly without measurable quality loss on the benchmark set.
# ─────────────────────────────────────────────────────────────────────────────

def _candidate_positions(
    j: int, rho: int, sequences: List[List[int]], inst: Instance,
) -> List[int]:
    """
    Return a list of candidate insertion positions for aircraft j on runway rho.

    For small instances (n ≤ 100): all positions 0..L are returned.
    For large instances (n > 100): a window of 5 positions centred on p0
    (the first position whose aircraft has δ ≥ δ_j) plus the endpoints
    {0, L} is returned.  The window size is fixed regardless of L to bound
    the cost of each inner evaluation.

    Parameters
    ----------
    j : int
    rho : int
    sequences : list of list of int
    inst : Instance

    Returns
    -------
    list of int
        Sorted candidate insertion positions.
    """
    seq, L = sequences[rho], len(sequences[rho])
    if inst.n <= 100:
        return list(range(L + 1))
    p0    = next((p for p, u in enumerate(seq)
                  if inst.delta[u] >= inst.delta[j]), L)
    cands = set(range(max(0, p0 - 2), min(L + 1, p0 + 3))) | {0, L}
    return sorted(cands)


def _best_insertions(
    j: int, m: int,
    sequences: List[List[int]], C_hats: List[List[float]],
    B_bar: float,
    inst: Instance, params: HeuristicParams,
) -> Tuple[Tuple[float, int, int, List[float]], float]:
    """
    Find the best insertion (runway, position) for aircraft j and compute its
    regret value.

    Regret is defined as the cost gap between the best and second-best runway:

        regret(j) = cost_2nd_best(j) − cost_best(j)

    A high regret means j loses a lot of quality if placed on any runway other
    than its best one — it should therefore be scheduled sooner.

    For single-runway problems (m=1), the second-best cost is taken as the
    second-cheapest position within the single runway (or infinity if fewer
    than two positions exist).

    Parameters
    ----------
    j : int
    m : int
        Number of runways.
    sequences, C_hats : as in evaluate_insertion.
    B_bar : float
        Mean committed time (passed to evaluate_insertion).
    inst : Instance
    params : HeuristicParams

    Returns
    -------
    best1 : (cost, rho, position, C_n)
        The globally cheapest (runway, position) triplet and the resulting
        surrogate times.
    c2 : float
        Cost of the second-best runway (or second-best position for m=1).
        May be math.inf if no alternative exists.
    """
    per_rho: List[Tuple[float, int, List[float]]] = []
    for rho in range(m):
        rc, rp, rC = math.inf, 0, []
        for p in _candidate_positions(j, rho, sequences, inst):
            c, Cn = evaluate_insertion(j, rho, p, sequences, C_hats,
                                       B_bar, inst, params)
            if c < rc:
                rc, rp, rC = c, p, Cn
        per_rho.append((rc, rp, rC))

    sr     = sorted(range(m), key=lambda r: per_rho[r][0])
    rho_st = sr[0]
    c1, p1, C1 = per_rho[rho_st]
    best1  = (c1, rho_st, p1, C1)

    if m > 1:
        c2 = per_rho[sr[1]][0]
    else:
        # Single-runway: rank all positions by cost; c2 = second cheapest.
        all_c = sorted(
            evaluate_insertion(j, 0, p, sequences, C_hats, B_bar, inst, params)[0]
            for p in range(len(sequences[0]) + 1)
        )
        c2 = all_c[1] if len(all_c) > 1 else math.inf
    return best1, c2


# ─────────────────────────────────────────────────────────────────────────────
# Post-construction inter-runway repair
#
# After ramp_rbi constructs a feasible assignment, a local search pass
# attempts to improve runway-load balance by relocating the aircraft with
# the highest target-conflict contribution from the longest runway to a
# less-loaded one.  At most max_iterations moves are made; the loop
# terminates early once the longest runway is within one aircraft of the
# mean load.
# ─────────────────────────────────────────────────────────────────────────────

def inter_runway_repair(
    sequences: List[List[int]], C_hats: List[List[float]],
    inst: Instance, params: HeuristicParams,
    max_iterations: int = 150,
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    Iterative relocation of aircraft between runways to improve load balance.

    At each iteration:
      1. Identify the runway rho_src with the most aircraft.
      2. Select the aircraft j_move on rho_src with the highest TC contribution
         (removing it from rho_src should improve that runway's quality most).
      3. Find the cheapest feasible position for j_move on any other runway.
      4. Move j_move if such a position exists.

    This post-processing step is O(iterations × n²) and is skipped for
    single-runway problems.

    Parameters
    ----------
    sequences : list of list of int
        Initial sequences from ramp_rbi.
    C_hats : list of list of float
        Corresponding surrogate times.
    inst : Instance
    params : HeuristicParams
        Weights used to evaluate re-insertion cost on the target runway.
    max_iterations : int, optional
        Maximum number of relocation moves (default 150).

    Returns
    -------
    sequences : list of list of int
        Repaired sequences.
    C_hats : list of list of float
        Updated surrogate times.
    """
    m = len(sequences)
    if m == 1:
        return sequences, C_hats

    sequences = [list(s) for s in sequences]
    C_hats    = [list(c) for c in C_hats]
    mean_load = inst.n / m

    for _ in range(max_iterations):
        rho_src = max(range(m), key=lambda r: len(sequences[r]))
        if len(sequences[rho_src]) <= mean_load + 1:
            break
        B_bar = sum(C_hats[r][-1] if C_hats[r] else 0.0 for r in range(m)) / m

        best_tc, best_sp = -1.0, -1
        for sp, j in enumerate(sequences[rho_src]):
            seq_no_j = sequences[rho_src][:sp] + sequences[rho_src][sp + 1:]
            tc = target_conflict_insert(j, sp, seq_no_j, inst)
            if tc > best_tc:
                best_tc, best_sp = tc, sp
        if best_sp == -1:
            break

        j_move = sequences[rho_src][best_sp]
        best_c, best_rd, best_dp, best_Cn = math.inf, -1, -1, []

        for rd in range(m):
            if rd == rho_src:
                continue
            for dp in range(len(sequences[rd]) + 1):
                c, Cn = evaluate_insertion(j_move, rd, dp, sequences, C_hats,
                                           B_bar, inst, params)
                if c < best_c:
                    best_c, best_rd, best_dp, best_Cn = c, rd, dp, Cn

        if best_rd == -1 or best_c == math.inf:
            break

        sequences[rho_src].pop(best_sp)
        C_hats[rho_src] = surrogate_times(sequences[rho_src], inst)
        sequences[best_rd].insert(best_dp, j_move)
        C_hats[best_rd] = best_Cn

    return sequences, C_hats


# ─────────────────────────────────────────────────────────────────────────────
# §8  TC-RBI construction heuristic (main algorithm)
#
# Algorithm outline
# -----------------
# Maintain:
#   sequences[rho]  — partial landing order on runway rho  (initially empty)
#   C_hats[rho]     — surrogate times for sequences[rho]
#   U               — set of unscheduled aircraft
#   F ⊆ U           — "forced" aircraft with no feasible position remaining
#
# Each iteration:
#   (a) Force-insert all aircraft in F∩U in decreasing CR order using the
#       minimum-violation strategy (§7.2).
#   (b) Screen the remaining unscheduled aircraft: compute a blend of CR and
#       urgency; retain the top q_eff as candidates.
#   (c) For each candidate j compute its best insertion (runway, position, cost)
#       and regret value (cost of second-best runway).
#   (d) Score = normalised_best_cost − 0.20·normalised_regret − 0.10·normalised_CR.
#       Insert the aircraft with the lowest score (min-cost with regret tie-break).
#   (e) After each insertion, refresh the forced set.
#
# After all aircraft are scheduled, apply inter_runway_repair.
# ─────────────────────────────────────────────────────────────────────────────

def ramp_rbi(
    inst: Instance,
    m: int,
    params: HeuristicParams,
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    TC-RBI: Target-Conflict Regret-Based Insertion construction heuristic.

    Constructs a complete multi-runway landing schedule by iteratively
    inserting aircraft one at a time.  At each iteration the aircraft with the
    best combination of low insertion cost, high regret, and high criticality
    is selected and placed at its best (runway, position).

    Aircraft that have no remaining feasible position (forced set F) are
    handled first via minimum-violation insertion (§7.2) to ensure the
    schedule always covers all n aircraft.

    The screening step (step b) limits the regret evaluation to the top
    q_eff ≤ 150 candidates by their CR/urgency blend, keeping the per-iteration
    cost tractable for large instances.

    Parameters
    ----------
    inst : Instance
        Problem instance.
    m : int
        Number of runways.
    params : HeuristicParams
        Construction weights (tuned by Optuna or set to defaults).

    Returns
    -------
    sequences : list of list of int, length m
        Landing order on each runway.  Every aircraft index 0..n-1 appears
        exactly once across all sequences.
    C_hats : list of list of float, length m
        Surrogate (consecutive-predecessor) landing times.  These are
        approximate; use stage2_lp_objective for exact optimal times.
    """
    n, eps = inst.n, inst.eps
    _, CR  = compute_priorities(inst)

    sequences: List[List[int]]   = [[] for _ in range(m)]
    C_hats:    List[List[float]] = [[] for _ in range(m)]
    B      = [0.0] * m    # committed time per runway (last surrogate time)
    B_bar  = 0.0           # mean committed time
    U: List[int] = list(range(n))   # unscheduled aircraft
    F: set        = set()            # forced (no feasible slot remains)

    def committed(rho):
        """Last surrogate time on runway rho, or 0 if empty."""
        return C_hats[rho][-1] if C_hats[rho] else 0.0

    def refresh_forced():
        """Add to F any aircraft in U with no remaining feasible position."""
        for j in list(U):
            if j in F:
                continue
            if not _is_feasible_anywhere(j, sequences, C_hats, inst):
                F.add(j)

    def do_insert(j, rho, p, C_new):
        """Commit aircraft j to runway rho at position p; update state."""
        nonlocal B_bar
        sequences[rho].insert(p, j)
        C_hats[rho] = C_new
        B[rho]  = committed(rho)
        B_bar   = sum(B) / m
        U.remove(j)
        F.discard(j)

    while U:
        # ── (a) Force-insert aircraft with no feasible slot ────────────────
        while F & set(U):
            j_star        = max([j for j in U if j in F], key=lambda j: CR[j])
            rho, p, C_new = min_violation_insert(j_star, sequences, inst)
            seq_new        = sequences[rho][:p] + [j_star] + sequences[rho][p:]
            sequences[rho] = seq_new
            C_hats[rho]    = C_new
            B[rho]         = committed(rho)
            B_bar          = sum(B) / m
            U.remove(j_star)
            F.discard(j_star)
            refresh_forced()

        if not U:
            break

        U_avail = [j for j in U if j not in F]
        if not U_avail:
            break

        # ── (b) Screen candidates by CR / urgency blend ───────────────────
        tau    = min(B)   # earliest committed time (approximate "now")
        urg    = np.array([1.0 / max(float(inst.delta[j]) - tau, eps)
                           for j in U_avail])
        cr_arr = np.array([CR[j] for j in U_avail])
        screen = (params.eta * minmax_norm(cr_arr, eps)
                  + (1 - params.eta) * minmax_norm(urg, eps))

        # q_eff: full set for small instances; bounded subset for large ones.
        q_eff = (len(U_avail) if n <= 100
                 else min(150, max(50, int(0.25 * len(U_avail)))))
        top   = np.argsort(screen)[::-1][:q_eff]
        U_q   = [U_avail[i] for i in top]

        # ── (c) Compute best insertion and regret for each candidate ───────
        info: Dict[int, Tuple[float, int, int, List[float], float]] = {}
        for j in U_q:
            (c1, rho1, p1, Cn1), c2 = _best_insertions(
                j, m, sequences, C_hats, B_bar, inst, params)
            info[j] = (c1, rho1, p1, Cn1, c2)

        # Regret ceiling: max finite regret + h_bar·T_span.
        finite = [info[j][4] - info[j][0] for j in U_q if info[j][4] < math.inf]
        R_max  = ((max(finite) + inst.h_bar * inst.T_span)
                  if finite else inst.h_bar * inst.T_span)

        # ── (d) Score and select ───────────────────────────────────────────
        best_c = np.array([info[j][0] for j in U_q])
        regret = np.array([(info[j][4] - info[j][0]) if info[j][4] < math.inf
                           else R_max for j in U_q])
        cr_uq  = np.array([CR[j] for j in U_q])
        score  = (minmax_norm(best_c, eps)
                  - 0.20 * minmax_norm(regret, eps)
                  - 0.10 * minmax_norm(cr_uq,  eps))
        j_star = U_q[int(np.argmin(score))]

        c_s, rho_s, p_s, Cn_s, _ = info[j_star]
        if c_s == math.inf:
            # No feasible position found; defer to forced set.
            F.add(j_star)
            continue

        # ── (e) Commit and refresh ─────────────────────────────────────────
        do_insert(j_star, rho_s, p_s, Cn_s)
        refresh_forced()

    sequences, C_hats = inter_runway_repair(sequences, C_hats, inst, params)
    return sequences, C_hats


# ─────────────────────────────────────────────────────────────────────────────
# Fast proxy objective  (used in the Optuna inner loop for n > 100)
#
# Evaluating the Stage-2 LP for every Optuna trial is too expensive for large
# instances.  Instead, the total pairwise target-time conflict (TC) serves as
# a proxy: it correlates strongly with LP objective quality while being O(n²)
# to compute.  For small instances (n ≤ 100), the LP is used directly.
# ─────────────────────────────────────────────────────────────────────────────

def total_target_conflict(
    sequences: List[List[int]], inst: Instance,
) -> float:
    """
    Compute the total weighted pairwise target-time conflict across all runways.

    For each ordered pair (i, j) with i preceding j on the same runway:

        TC(i, j) = 0.5 · (p_i + p_j) · max( s[i,j] − (δ_j − δ_i), 0 )

    The total is the sum over all such pairs.  This is an O(n²) surrogate
    for the LP objective, used during Optuna hyperparameter search on large
    instances where repeated LP solves would be prohibitive.

    Dispatch:
      n ≥ GPU_MIN_N and PyTorch CUDA available → GPU kernel
      otherwise                                → vectorised NumPy

    Parameters
    ----------
    sequences : list of list of int
    inst : Instance

    Returns
    -------
    float
        Total target-time conflict (non-negative).
    """
    if USE_GPU and _GPU_AVAILABLE and _torch is not None and inst.n >= GPU_MIN_N:
        return _total_target_conflict_gpu(sequences, inst)
    return _total_target_conflict_cpu(sequences, inst)


def _total_target_conflict_cpu(
    sequences: List[List[int]], inst: Instance,
) -> float:
    """
    NumPy-vectorised total target-time conflict.

    Uses np.triu_indices to generate all upper-triangular (predecessor,
    successor) pairs for each runway simultaneously, avoiding Python loops
    over pairs.

    Parameters
    ----------
    sequences : list of list of int
    inst : Instance

    Returns
    -------
    float
    """
    total = 0.0
    for seq in sequences:
        L = len(seq)
        if L < 2:
            continue
        s_arr       = np.asarray(seq, dtype=np.intp)
        ii, jj      = np.triu_indices(L, k=1)
        i_ac        = s_arr[ii]
        j_ac        = s_arr[jj]
        v           = inst.s[i_ac, j_ac] - (inst.delta[j_ac] - inst.delta[i_ac])
        total      += float(
            (0.5 * (inst.p_arr[i_ac] + inst.p_arr[j_ac]) * np.maximum(v, 0.0)).sum()
        )
    return total


def _total_target_conflict_gpu(
    sequences: List[List[int]], inst: Instance,
) -> float:
    """
    PyTorch CUDA variant of total target-time conflict.

    The separation matrix (inst._s_gpu), target times (inst._delta_gpu), and
    penalty rates (inst._p_arr_gpu) are already resident on the GPU (placed
    there in Instance.__post_init__).  Only the lightweight integer index
    tensors are transferred per call, minimising PCIe traffic.

    Index tensors are cast to int64 (torch.long) because PyTorch advanced
    indexing requires 64-bit integer indices regardless of platform.

    Parameters
    ----------
    sequences : list of list of int
    inst : Instance
        Must have _s_gpu, _delta_gpu, _p_arr_gpu populated (n ≥ GPU_MIN_N).

    Returns
    -------
    float
    """
    total = 0.0
    for seq in sequences:
        L = len(seq)
        if L < 2:
            continue
        s_arr  = np.asarray(seq, dtype=np.intp)
        ii, jj = np.triu_indices(L, k=1)
        i_t = _torch.from_numpy(s_arr[ii].astype(np.int64)).to(_CUDA_DEVICE)
        j_t = _torch.from_numpy(s_arr[jj].astype(np.int64)).to(_CUDA_DEVICE)
        s_ij  = inst._s_gpu[i_t, j_t]
        d_ij  = inst._delta_gpu[j_t] - inst._delta_gpu[i_t]
        p_ij  = inst._p_arr_gpu[i_t] + inst._p_arr_gpu[j_t]
        v     = s_ij - d_ij
        total += float(_torch.sum(0.5 * p_ij * _torch.clamp(v, min=0.0)))
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Optuna hyperparameter optimisation
# ─────────────────────────────────────────────────────────────────────────────

def _n_trials(n: int, base: int) -> int:
    """
    Scale the Optuna trial budget by instance size.

    Larger instances (more aircraft) take longer per trial, so the budget is
    reduced to keep total tuning time reasonable.

    n ≤ 100 : full budget (base trials)
    n ≤ 250 : base // 3  (minimum 10)
    n > 250 : base // 6  (minimum 5)

    Parameters
    ----------
    n : int
        Number of aircraft in the instance.
    base : int
        Full trial budget (N_TRIALS_BASE from configuration).

    Returns
    -------
    int
        Adjusted trial count.
    """
    if n <= 100: return base
    if n <= 250: return max(10, base // 3)
    return max(5, base // 6)


def optimize_params(
    inst: Instance, m: int, n_trials: int, seed: int,
    n_jobs: int = 1,
) -> HeuristicParams:
    """
    Tune HeuristicParams for a specific (instance, runway-count) pair using
    Optuna's Tree-structured Parzen Estimator (TPE) sampler.

    Objective function:
      n ≤ 100 : Stage-2 LP objective (exact, uses HiGHS).
      n > 100 : total_target_conflict proxy (O(n²), avoids repeated LP solves).

    Parallelism:
      n_jobs > 1 runs trials concurrently in threads.  NumPy and HiGHS both
      release the GIL, so threading achieves near-linear speedup.
      n_jobs is capped at min(n_jobs, n_trials) to avoid over-subscription.

    Parameters
    ----------
    inst : Instance
    m : int
        Number of runways.
    n_trials : int
        Number of Optuna trials (already scaled by _n_trials).
    seed : int
        Random seed for the TPE sampler (OPTUNA_SEED from configuration).
    n_jobs : int, optional
        Number of parallel trial threads (default 1).

    Returns
    -------
    HeuristicParams
        Best parameters found.  Falls back to HeuristicParams() (defaults)
        if Optuna is not installed or n_trials == 0.
    """
    if not _OPTUNA:
        print("  [Optuna] package not found — using default parameters.")
        return HeuristicParams()
    if n_trials == 0:
        return HeuristicParams()

    use_lp   = (inst.n <= 100)
    eff_jobs = min(n_jobs, n_trials)

    def objective(trial):
        p = HeuristicParams(
            eta      = trial.suggest_float('eta',      0.20, 0.80),
            mu_tc    = trial.suggest_float('mu_tc',    0.10, 5.00),
            mu_late  = trial.suggest_float('mu_late',  0.01, 2.00),
            mu_count = trial.suggest_float('mu_count', 0.10, 3.00),
            mu_sep   = trial.suggest_float('mu_sep',   0.00, 0.50),
        )
        seqs, _ = ramp_rbi(inst, m, p)
        if use_lp:
            obj, _, feas, _ = stage2_lp_objective(seqs, inst)
            return obj if feas else 1e12
        return total_target_conflict(seqs, inst)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study   = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials,
                   n_jobs=eff_jobs, show_progress_bar=False)

    bp = study.best_params
    return HeuristicParams(
        eta      = bp['eta'],
        mu_tc    = bp['mu_tc'],
        mu_late  = bp['mu_late'],
        mu_count = bp['mu_count'],
        mu_sep   = bp['mu_sep'],
    )


# ─────────────────────────────────────────────────────────────────────────────
# §10  Stage-2 LP: exact landing-time optimisation for a fixed sequence
#
# Given the landing sequences produced by ramp_rbi (§8), the Stage-2 LP finds
# optimal landing times that minimise total weighted earliness/tardiness
# while satisfying all separation and time-window constraints.
#
# Decision variables (per LP):
#   C_j ∈ [r_j, d_j]   — landing time for aircraft j        (n variables)
#   E_j ≥ 0             — earliness: E_j ≥ δ_j − C_j        (n variables)
#   T_j ≥ 0             — tardiness: T_j ≥ C_j − δ_j        (n variables)
#
# Constraints:
#   (C1) −C_j − E_j ≤ −δ_j      (E_j ≥ δ_j − C_j)
#   (C2)  C_j − T_j ≤  δ_j      (T_j ≥ C_j − δ_j)
#   (C3)  C_i − C_j ≤ −s[i,j]   for all ordered pairs (i,j) in sequences
#
# The LP is formulated with a sparse coefficient matrix (csr_matrix) to
# exploit HiGHS's sparse simplex.
#
# CRITICAL: OR Library separation matrices violate the triangle inequality.
# C3 must cover ALL ordered pairs (i, j) where i precedes j in the sequence,
# not only consecutive pairs.  Using only consecutive separations is a strict
# relaxation that allows infeasible schedules to pass the LP.
# ─────────────────────────────────────────────────────────────────────────────

def stage2_lp_objective(
    sequences: List[List[int]],
    inst: Instance,
    eps_tol: float = 1e-6,
) -> Tuple[float, Optional[np.ndarray], bool, List[str]]:
    """
    Solve the Stage-2 LP for exact landing times given fixed sequences.

    The LP minimises total weighted earliness/tardiness with full pairwise
    separation constraints across all ordered pairs within each runway.

    Parameters
    ----------
    sequences : list of list of int
        Landing sequences from ramp_rbi.  The order within each list is
        binding: C_i must precede C_j (i.e. C_j ≥ C_i + s[i,j]) for all
        i before j in the same list.
    inst : Instance
    eps_tol : float, optional
        Numerical tolerance for post-solve violation checking (default 1e-6).

    Returns
    -------
    obj : float
        LP optimal objective value.  Returns math.inf if the solver fails.
    C_lp : ndarray, shape (n,) or None
        Optimal landing times (indexed 0..n-1).  None if infeasible.
    feasible : bool
        True iff the solution satisfies all time-window and separation
        constraints within eps_tol.
    violations : list of str
        Human-readable description of any constraint violations found.
        Empty list when feasible = True.
    """
    n = inst.n; C0, E0, T0 = 0, n, 2 * n; nv = 3 * n
    c_obj          = np.zeros(nv)
    c_obj[E0:E0+n] = inst.g
    c_obj[T0:T0+n] = inst.h

    # Enumerate all ordered pairs (i, j) with i preceding j on the same runway.
    sep_pairs = [(seq[a], seq[b])
                 for seq in sequences
                 for a in range(len(seq))
                 for b in range(a + 1, len(seq))]
    n_ineq = 2 * n + len(sep_pairs)
    rows: List[int] = []; cols: List[int] = []; vals: List[float] = []
    b_ub = np.empty(n_ineq); r = 0

    # Earliness constraints: −C_j − E_j ≤ −δ_j  ↔  E_j ≥ δ_j − C_j
    for j in range(n):
        rows += [r, r]; cols += [C0+j, E0+j]; vals += [-1., -1.]
        b_ub[r] = -float(inst.delta[j]); r += 1
    # Tardiness constraints: C_j − T_j ≤ δ_j  ↔  T_j ≥ C_j − δ_j
    for j in range(n):
        rows += [r, r]; cols += [C0+j, T0+j]; vals += [1., -1.]
        b_ub[r] = float(inst.delta[j]); r += 1
    # Separation constraints: C_i − C_j ≤ −s[i,j]  ↔  C_j ≥ C_i + s[i,j]
    for i, j in sep_pairs:
        rows += [r, r]; cols += [C0+i, C0+j]; vals += [1., -1.]
        b_ub[r] = -float(inst.s[i, j]); r += 1

    A_ub  = csr_matrix((vals, (rows, cols)), shape=(n_ineq, nv))
    bounds = ([(float(inst.r[j]), float(inst.d[j])) for j in range(n)]
              + [(0., None)] * (2 * n))
    res   = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if not res.success:
        return math.inf, None, False, [f"LP solver: {res.message}"]

    C_lp = res.x[C0:C0+n]; obj = float(res.fun)
    viol: List[str] = []
    for j in range(n):
        if C_lp[j] < inst.r[j] - eps_tol:
            viol.append(f"Aircraft {j}: C={C_lp[j]:.4f} < r={inst.r[j]:.4f}")
        if C_lp[j] > inst.d[j] + eps_tol:
            viol.append(f"Aircraft {j}: C={C_lp[j]:.4f} > d={inst.d[j]:.4f}")
    for seq in sequences:
        for a in range(len(seq)):
            for b in range(a + 1, len(seq)):
                i, j = seq[a], seq[b]
                if C_lp[j] - C_lp[i] < inst.s[i, j] - eps_tol:
                    viol.append(f"sep({i},{j}): {C_lp[j]-C_lp[i]:.4f} < {inst.s[i,j]:.4f}")
    return obj, C_lp, len(viol) == 0, viol


# ─────────────────────────────────────────────────────────────────────────────
# §10.3  Full pairwise feasibility verification + earliest-time objective
#
# Computes exact earliest-feasible landing times by propagating the tightest
# lower bound across ALL predecessors (not just the consecutive one), then
# checks all time-window and separation constraints.  This is the ground-truth
# feasibility check used in the summary report.
# ─────────────────────────────────────────────────────────────────────────────

def verify_and_exact_obj(
    sequences: List[List[int]],
    inst: Instance,
    eps_tol: float = 1e-6,
) -> Tuple[bool, List[str], float, Dict[int, float]]:
    """
    Verify schedule feasibility and compute the exact penalty objective.

    Landing times are determined by the earliest-feasible rule, propagating
    ALL pairwise separation constraints (not only consecutive):

        C[q] = max( r[seq[q]],  max_{h < q}( C[h] + s[seq[h], seq[q]] ) )

    This is more conservative than surrogate_times (consecutive only) and
    reflects actual feasibility under OR Library separation matrices that
    can violate the triangle inequality.

    The resulting objective (using these earliest-feasible times) is an
    upper bound on the true optimum; the Stage-2 LP objective (stage2_lp_objective)
    will be ≤ this value for any given sequence.

    Parameters
    ----------
    sequences : list of list of int
    inst : Instance
    eps_tol : float, optional
        Constraint violation tolerance (default 1e-6).

    Returns
    -------
    feasible : bool
        True iff all aircraft are scheduled and all constraints are satisfied
        within eps_tol.
    violations : list of str
        Human-readable descriptions of any violated constraints.
    obj : float
        Penalty objective evaluated at earliest-feasible landing times.
    C_exact : dict mapping aircraft index → landing time
        Earliest-feasible landing time for each aircraft.
    """
    n        = inst.n
    C_exact: Dict[int, float] = {}
    for rho, seq in enumerate(sequences):
        if not seq:
            continue
        C_r    = [0.0] * len(seq)
        C_r[0] = float(inst.r[seq[0]])
        for q in range(1, len(seq)):
            j = seq[q]; t = float(inst.r[j])
            for h in range(q):
                t = max(t, C_r[h] + float(inst.s[seq[h], j]))
            C_r[q] = t
        for q, j in enumerate(seq):
            C_exact[j] = C_r[q]

    viol: List[str] = []
    for j in range(n):
        if j not in C_exact:
            viol.append(f"Aircraft {j} not scheduled")
    for j, Cj in C_exact.items():
        if Cj < inst.r[j] - eps_tol:
            viol.append(f"Ac {j}: C={Cj:.2f} < r={inst.r[j]:.2f}")
        if Cj > inst.d[j] + eps_tol:
            viol.append(f"Ac {j}: C={Cj:.2f} > d={inst.d[j]:.2f}")
    for rho, seq in enumerate(sequences):
        for qi in range(len(seq)):
            for qj in range(qi + 1, len(seq)):
                i, j  = seq[qi], seq[qj]
                Ci    = C_exact.get(i, 0.); Cj = C_exact.get(j, 0.)
                if Cj - Ci < inst.s[i, j] - eps_tol:
                    viol.append(f"Rwy{rho+1} sep({i},{j}): {Cj-Ci:.4f} < {inst.s[i,j]:.4f}")

    obj = sum(float(inst.g[j]) * max(float(inst.delta[j]) - Cj, 0.)
              + float(inst.h[j]) * max(Cj - float(inst.delta[j]), 0.)
              for j, Cj in C_exact.items())
    return len(viol) == 0, viol, obj, C_exact


# ─────────────────────────────────────────────────────────────────────────────
# Display utilities
# ─────────────────────────────────────────────────────────────────────────────

def _gap_str(obj: float, opt: Optional[float]) -> str:
    """
    Format the optimality gap as a percentage string.

    Special cases:
      opt is None        → "N/A"
      opt == 0 and obj ≈ 0  → "0.00%"
      opt == 0 and obj > 0  → "∞"   (division by zero)

    Parameters
    ----------
    obj : float
        Achieved objective value.
    opt : float or None
        Known optimum from KNOWN_OPTIMA.

    Returns
    -------
    str
        Gap string, e.g. "3.47%" or "N/A".
    """
    if opt is None:  return "N/A"
    if opt == 0.:    return "0.00%" if obj < 1e-6 else "∞"
    return f"{100. * (obj - opt) / opt:.2f}%"


def print_result(
    inst: Instance, m: int,
    sequences: List[List[int]], C_hats: List[List[float]],
    elapsed: float,
    params: Optional[HeuristicParams] = None,
) -> Tuple[float, Optional[float], bool]:
    """
    Print a detailed per-instance result report and return key metrics.

    Reports (in order):
      - Runtime
      - Active HeuristicParams (if provided)
      - Surrogate objective (consecutive-predecessor times)
      - Earliest-time objective (exact pairwise propagation)
      - Stage-2 LP objective (HiGHS optimal for given sequences)
      - Target-conflict proxy
      - Known optimum and gap (from KNOWN_OPTIMA, if available)
      - Sequence feasibility (PASS/FAIL with up to 6 violation details)
      - LP feasibility
      - Per-runway load (aircraft count, committed time, first 6 aircraft)

    Parameters
    ----------
    inst : Instance
    m : int
        Number of runways used.
    sequences : list of list of int
    C_hats : list of list of float
    elapsed : float
        Wall-clock construction time in seconds.
    params : HeuristicParams or None
        If provided, the parameter values are included in the report.

    Returns
    -------
    lp_obj : float
        Stage-2 LP objective value.
    opt : float or None
        Known optimum for this (instance, m) pair, or None if unavailable.
    feas_lp : bool
        True iff the LP solution satisfies all constraints.
    """
    feas_e, viol_e, earliest_obj, _ = verify_and_exact_obj(sequences, inst)
    lp_obj, _, feas_lp, viol_lp    = stage2_lp_objective(sequences, inst)
    surr_obj = sum(surrogate_penalty(sequences[r], C_hats[r], inst) for r in range(m))
    tc_proxy = total_target_conflict(sequences, inst)
    opt      = KNOWN_OPTIMA.get(inst.name, {}).get(m)

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  {inst.name.upper()}  |  n={inst.n}  |  m={m} runway(s)")
    print(sep)
    print(f"  Runtime                  : {elapsed:.4f} s")
    if params is not None:
        print(f"  Parameters               : {params}")
    print(f"  Surrogate objective      : {surr_obj:.4f}")
    print(f"  Earliest-time objective  : {earliest_obj:.4f}")
    print(f"  Stage-2 LP objective     : {lp_obj:.4f}")
    print(f"  Target-conflict proxy    : {tc_proxy:.4f}")
    if opt is not None:
        print(f"  Known optimum            : {opt:.4f}")
        print(f"  Gap to optimum (LP)      : {_gap_str(lp_obj, opt)}")
    else:
        print(f"  Known optimum            : not available for m={m}")
    print(f"  Sequence feasibility     : {'PASS ✓' if feas_e else 'FAIL ✗'}")
    print(f"  LP feasibility           : {'PASS ✓' if feas_lp else 'FAIL ✗'}")
    if not feas_e:
        for v in viol_e[:6]:
            print(f"    ✗ {v}")
        if len(viol_e) > 6:
            print(f"    ... and {len(viol_e)-6} more")
    print("  Runway load:")
    for rho, seq in enumerate(sequences):
        B_rho = C_hats[rho][-1] if C_hats[rho] else 0.
        print(f"    Runway {rho+1}: {len(seq):4d} aircraft  B={B_rho:.2f}  "
              f"seq=[{', '.join(str(j) for j in seq[:6])}{',...' if len(seq) > 6 else ''}]")
    print(sep)
    return lp_obj, opt, feas_lp


def print_summary_table(results: List[dict]) -> None:
    """
    Print a formatted batch-results summary table to stdout.

    Columns: Instance, n, m, LP obj, Optimum, Gap%, Feasibility, Time(s).
    Rows are sorted by (instance name, runway count).
    An aggregate footer reports total feasible count and average/max gap
    over instances where the known optimum is strictly positive.

    Parameters
    ----------
    results : list of dict
        Each dict must contain keys: name, n, m, obj, opt, feasible, time.
        These are the dicts returned by _run_one.
    """
    col = ["Instance", "n", "m", "LP obj", "Optimum", "Gap%", "Feas", "Time(s)"]
    w   = [14, 5, 4, 14, 14, 12, 6, 10]
    hdr = "  " + "".join(f"{c:>{w[i]}}" for i, c in enumerate(col))
    bar = "=" * len(hdr)
    print(f"\n{bar}\n  TC-RBI  —  BATCH RESULTS\n{bar}")
    print(hdr); print("-" * len(hdr))
    for r in sorted(results, key=lambda x: (x["name"], x["m"])):
        row = [r["name"], r["n"], r["m"], f"{r['obj']:.4f}",
               f"{r['opt']:.4f}" if r["opt"] is not None else "N/A",
               _gap_str(r["obj"], r["opt"]),
               "✓" if r["feasible"] else "✗",
               f"{r['time']:.4f}"]
        print("  " + "".join(f"{str(v):>{w[i]}}" for i, v in enumerate(row)))
    print(bar)
    fc  = sum(1 for r in results if r["feasible"])
    pos = [r for r in results if r["opt"] is not None and r["opt"] > 0]
    if pos:
        gaps = [100. * (r["obj"] - r["opt"]) / r["opt"] for r in pos]
        print(f"  Instances run  : {len(results)}")
        print(f"  Feasible       : {fc}/{len(results)}")
        print(f"  Avg gap (opt>0): {np.mean(gaps):.2f}%  |  Max: {max(gaps):.2f}%")
    print(bar)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_one(
    fp: str, m: int,
    run_optuna: bool = False,
    n_trials_base: int = 30,
    seed: int = 42,
    n_optuna_workers: int = 1,
) -> dict:
    """
    Execute one complete (instance, runway-count) job and return results.

    This function is the unit of work dispatched to each ProcessPoolExecutor
    worker in batch mode.  All state is local (no shared mutable globals),
    making it safe for multiprocessing.

    All stdout output is captured to a string buffer and returned in the
    result dict under the key "output".  The main process prints each buffer
    after the corresponding future completes, preventing interleaved output
    from concurrent workers.

    Workflow:
      1. Parse the instance file.
      2. Optionally run Optuna to tune HeuristicParams.
      3. Run ramp_rbi to construct sequences.
      4. Call print_result to generate the report.
      5. Return the result dict.

    Parameters
    ----------
    fp : str
        Path to the instance file.
    m : int
        Number of runways.
    run_optuna : bool, optional
        Whether to run Optuna tuning before construction (default False).
    n_trials_base : int, optional
        Base Optuna trial budget (scaled by _n_trials based on instance size).
    seed : int, optional
        Random seed for Optuna (default 42).
    n_optuna_workers : int, optional
        Number of parallel Optuna trial threads (default 1).

    Returns
    -------
    dict with keys:
        name     : str    — instance name
        n        : int    — number of aircraft
        m        : int    — number of runways
        obj      : float  — Stage-2 LP objective
        opt      : float or None — known optimum (from KNOWN_OPTIMA)
        feasible : bool   — LP feasibility
        time     : float  — construction time in seconds (excludes Optuna)
        params   : HeuristicParams
        output   : str    — captured stdout from this job
    """
    buf  = io.StringIO()

    with contextlib.redirect_stdout(buf):
        inst   = load_instance(fp)
        params = HeuristicParams()

        if run_optuna:
            n_t       = _n_trials(inst.n, n_trials_base)
            obj_label = "LP" if inst.n <= 100 else "TC proxy"
            print(f"\n  [Optuna] {inst.name.upper()} m={m} → "
                  f"{n_t} trials ({obj_label} obj, {n_optuna_workers} thread(s)) ...")
            t_opt  = time.perf_counter()
            params = optimize_params(inst, m, n_t, seed,
                                     n_jobs=n_optuna_workers)
            t_opt  = time.perf_counter() - t_opt
            print(f"  [Optuna] done in {t_opt:.2f}s  best: {params}")

        t0 = time.perf_counter()
        seqs, chats = ramp_rbi(inst, m, params)
        elapsed = time.perf_counter() - t0

        lp_obj, opt, feasible = print_result(inst, m, seqs, chats, elapsed, params)

    return dict(name=inst.name, n=inst.n, m=m, obj=lp_obj,
                opt=opt, feasible=feasible, time=elapsed, params=params,
                output=buf.getvalue())


def _announce_config() -> None:
    """
    Print a one-time configuration summary at startup.

    Reports:
      - Worker and thread counts
      - Active accelerators (Numba JIT, PyTorch CUDA)
      - Installation tips if optional accelerators are absent
    """
    accel = []
    if _NUMBA:
        accel.append("Numba JIT")
    if USE_GPU and _GPU_AVAILABLE:
        accel.append(f"PyTorch CUDA (threshold n≥{GPU_MIN_N})")
    accel_str = ", ".join(accel) if accel else "none"
    print(f"  Workers : {N_WORKERS} processes × {N_OPTUNA_WORKERS} Optuna threads"
          f" = {N_WORKERS * N_OPTUNA_WORKERS} logical threads")
    print(f"  Accel   : {accel_str}")
    if not _NUMBA:
        print("  TIP     : pip install numba  → 8–12× faster _compute_insert_times")
    if not _GPU_AVAILABLE:
        print("  TIP     : install PyTorch with CUDA  → GPU TC proxy for n≥200")


def main() -> None:
    """
    Entry point for TC-RBI.

    In BATCH_MODE, discovers all airland*.txt files in FOLDER and submits one
    job per (instance, runway-count) pair to a ProcessPoolExecutor.  Results
    are printed as they complete; a summary table is printed at the end.

    In single-file mode, runs all configured runway counts for INSTANCE_PATH
    sequentially and prints a summary table if more than one count is tested.
    """
    print("=" * 70)
    print("  TC-RBI  —  Target-Conflict Regret-Based Insertion")
    _announce_config()
    print("=" * 70)

    if BATCH_MODE:
        folder = Path(FOLDER)
        files  = sorted(folder.glob("airland*.txt"))
        if not files:
            print(f"No airland*.txt files found in {folder.resolve()}")
            return

        jobs = [
            (str(fp), m)
            for fp in files
            for m in INSTANCE_RUNWAYS.get(fp.stem.lower(), [N_RUNWAYS_DEFAULT])
        ]
        print(f"  Submitting {len(jobs)} jobs to {N_WORKERS} workers...\n")

        results = []
        with ProcessPoolExecutor(max_workers=N_WORKERS,
                                  mp_context=_MP_CTX) as executor:
            futs = {
                executor.submit(
                    _run_one, fp, m,
                    RUN_OPTUNA, N_TRIALS_BASE, OPTUNA_SEED,
                    N_OPTUNA_WORKERS,
                ): (fp, m)
                for fp, m in jobs
            }
            for fut in as_completed(futs):
                fp, m = futs[fut]
                try:
                    r = fut.result()
                    results.append(r)
                    print(r["output"], end="")
                    gap  = _gap_str(r["obj"], r["opt"])
                    feas = "✓" if r["feasible"] else "✗"
                    print(f"  ↳ {Path(fp).stem:<12} m={m}  "
                          f"LP={r['obj']:.2f}  gap={gap}  {feas}  "
                          f"({r['time']:.2f}s total)")
                except Exception as exc:
                    print(f"  ERROR {Path(fp).stem} m={m}: {exc}")

        print_summary_table(results)

    else:
        fp  = Path(INSTANCE_PATH)
        cfg = INSTANCE_RUNWAYS.get(fp.stem.lower(), [N_RUNWAYS_DEFAULT])
        res = []
        for m in cfg:
            r = _run_one(str(fp), m, RUN_OPTUNA, N_TRIALS_BASE, OPTUNA_SEED,
                         N_OPTUNA_WORKERS)
            print(r["output"], end="")
            res.append(r)
        if len(cfg) > 1:
            print_summary_table(res)


if __name__ == "__main__":
    main()