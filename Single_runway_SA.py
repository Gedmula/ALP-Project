"""
Aircraft Landing Problem
==========================================================
Target hardware : multi-core CPU
Algorithms      : MS-SA (parallel chains + ILS per chain)
                  Stage-2 LP (HiGHS via SciPy)

Dual-track feasibility (§8–10)
──────────────────────────────
SA chains are guided by evaluate_semi (consecutive-only LP, O(n) pre-filter).
Whenever a new semi-best is found the chain also calls evaluate (full pairwise
LP, O(n²) pre-filter) to update the fully-feasible incumbent.  Only fully-
feasible incumbents are reported in gap tables, written to export files, and
subjected to verify_schedule.

Worker return tuple layout (11 fields, §10):
  0  label         str
  1  pb_semi       List[int]          best semi-feasible sequence
  2  fb_semi       float              best semi-feasible objective
  3  pi_feas       Optional[List[int]] best fully-feasible sequence (None if none)
  4  fb_feas       float              best fully-feasible objective (inf if none)
  5  history       List[float]        fb_semi per outer iteration
  6  t_best_sa     float              wall time to best semi-feasible
  7  t_best_feas   float              wall time to best fully-feasible
  8  n_alt_seqs    int
  9  init_obj      float
  10 alpha_history List[float]
"""

# ═══════════════════════════════════════════════════════════════════════════
# 0.  IMPORTS & DEVICE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
import os, math, random, time, warnings, platform
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import linprog

try:
    import torch
    CUDA   = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if CUDA else "cpu")
    if CUDA: torch.backends.cuda.matmul.allow_tf32 = True
except ImportError:
    torch = None; CUDA = False; DEVICE = None

try:
    from tqdm import tqdm; _TQDM = True
except ImportError:
    _TQDM = False
    def tqdm(x, **kw): return x

warnings.filterwarnings("ignore")
N_CPU = max(os.cpu_count() - 4, 1)

import multiprocessing as _mp
_CTX = _mp.get_context("spawn" if platform.system() == "Windows" else "fork")

_MAX_ALT_SEQS = 100   # cap on distinct alternate-optimal sequences tracked per chain

# Default per-instance wall-clock budget (seconds) passed to ms_sa.
_INSTANCE_TIME_LIMIT = 3600


# ═══════════════════════════════════════════════════════════════════════════
# 1.  DATA STRUCTURES & LOADING
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ALPInstance:
    n:     int
    r:     np.ndarray
    d:     np.ndarray
    delta: np.ndarray
    g:     np.ndarray
    h:     np.ndarray
    s:     np.ndarray
    name:  str = ""

    def __post_init__(self):
        off  = ~np.eye(self.n, dtype=bool)
        vals = self.s[off & (self.s > 0)]
        self.s_bar = float(vals.mean()) if vals.size > 0 else 1.0

    def mpds_params(self) -> Tuple[float, float, float, float]:
        n, sb = self.n, self.s_bar
        d_max = np.max(self.delta); d_min = np.min(self.delta)
        C_max = max(d_min + n * sb, d_max)
        R     = (d_max - d_min) / C_max if C_max > 0 else 0.0
        K1    = 4.5 + R if R <= 0.5 else 6.0 - 2.0 * R
        tau_  = 1.0 - np.sum(self.delta) / (n * C_max)
        K2    = tau_ / (2.0 * sb) if sb > 0 else 1.0
        K3    = (np.max(self.r) - np.min(self.r)) / sb if sb > 0 else 1.0
        K4    = (np.sum(self.g) + np.sum(self.h)) / n
        return K1, K2, K3, K4


def load_orlib(path: str, name: str = "") -> ALPInstance:
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"OR Library file not found:\n  {p.resolve()}")
    tok = p.read_text().split()
    try:
        n        = int(tok[0])
        expected = 2 + n * (6 + n)
        if len(tok) < expected:
            raise ValueError(f"File has {len(tok)} tokens; need {expected}.")
        r = np.zeros(n); delta = np.zeros(n); d = np.zeros(n)
        g = np.zeros(n); h = np.zeros(n);     s = np.zeros((n, n))
        i = 2
        for j in range(n):
            i += 1
            r[j]     = float(tok[i]); i += 1
            delta[j] = float(tok[i]); i += 1
            d[j]     = float(tok[i]); i += 1
            g[j]     = float(tok[i]); i += 1
            h[j]     = float(tok[i]); i += 1
            for k in range(n):
                s[j][k] = float(tok[i]); i += 1
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Failed to parse {p.name}: {exc}") from exc
    inst = ALPInstance(n=n, r=r, d=d, delta=delta, g=g, h=h, s=s,
                       name=name or p.stem)
    edd = sorted(range(n), key=lambda j: delta[j])
    if not is_feasible(edd, inst):
        warnings.warn(f"{inst.name}: EDD sequence infeasible — check data.")
    return inst


def synthetic_instance(n: int = 15, seed: int = 0) -> ALPInstance:
    rng     = np.random.default_rng(seed)
    SEP     = np.array([[82, 69, 60], [131, 69, 60], [196, 157, 96]], dtype=float)
    max_sep = float(SEP.max())
    types   = rng.integers(0, 3, n)
    s = np.array([[SEP[types[j]][types[k]] if j != k else 0.0
                   for k in range(n)] for j in range(n)])
    interval = max_sep * 1.6; jitter = interval * 0.08
    delta = np.sort(np.arange(n) * interval + rng.uniform(-jitter, jitter, n))
    hw = max_sep * 1.5
    r  = np.maximum(delta - rng.uniform(hw * 0.6, hw, n), 0.0)
    d  = delta + rng.uniform(hw * 0.6, hw, n)
    g, h = rng.uniform(5, 30, n), rng.uniform(5, 30, n)
    inst = ALPInstance(n=n, r=r, d=d, delta=delta, g=g, h=h, s=s,
                       name=f"synthetic_n{n}")
    edd = sorted(range(n), key=lambda j: delta[j])
    assert is_feasible(edd, inst), "synthetic_instance: EDD infeasible."
    return inst


# ═══════════════════════════════════════════════════════════════════════════
# 2.  FEASIBILITY CHECK
# ═══════════════════════════════════════════════════════════════════════════

def is_feasible_old(seq: List[int], inst: ALPInstance) -> bool:
    """O(n) consecutive-separation pre-filter.  Used by evaluate_semi to
    gate SA acceptance decisions cheaply.  May admit sequences that violate
    non-consecutive pairwise separation; the LP in evaluate_semi enforces
    only the same consecutive constraints, so the two are consistent."""
    t = inst.r[seq[0]]
    if t > inst.d[seq[0]]: return False
    for l in range(1, inst.n):
        t = max(inst.r[seq[l]], t + inst.s[seq[l-1]][seq[l]])
        if t > inst.d[seq[l]]: return False
    return True


def is_feasible(seq: List[int], inst: ALPInstance) -> bool:
    """O(n²) complete pairwise-separation feasibility check.
    For each position m the earliest feasible landing time is bounded by
    the release date AND the separation from every predecessor l < m.
    Consecutive-only propagation understates this bound when the triangle
    inequality is violated (Beasley et al. 2000, p.187).  Used by evaluate
    to gate fully-feasible reporting and verification."""
    n = inst.n
    x = np.empty(n)
    x[0] = inst.r[seq[0]]
    if x[0] > inst.d[seq[0]]: return False
    for m in range(1, n):
        x[m] = inst.r[seq[m]]
        for l in range(m):
            lb = x[l] + inst.s[seq[l]][seq[m]]
            if lb > x[m]:
                x[m] = lb
        if x[m] > inst.d[seq[m]]: return False
    return True


# ═══════════════════════════════════════════════════════════════════════════
# 3.  STAGE-2 LP
# ═══════════════════════════════════════════════════════════════════════════

def _build_lp_matrices_old(seq: List[int], inst: ALPInstance):
    """Consecutive-only separation constraints (n-1 rows).
    Paired with is_feasible_old for the semi-feasible evaluation path."""
    n   = inst.n
    c   = np.concatenate([np.zeros(n), inst.g, inst.h])
    bnd = ([(inst.r[j], inst.d[j]) for j in range(n)] + [(0.0, None)] * 2 * n)
    A, b = [], []
    for l in range(n - 1):
        row = np.zeros(3 * n)
        row[seq[l]] = 1.0; row[seq[l+1]] = -1.0
        A.append(row); b.append(-inst.s[seq[l]][seq[l+1]])
    for j in range(n):
        row = np.zeros(3 * n)
        row[j] = -1.0; row[n + j] = -1.0
        A.append(row); b.append(-inst.delta[j])
    for j in range(n):
        row = np.zeros(3 * n)
        row[j] = 1.0; row[2*n + j] = -1.0
        A.append(row); b.append(inst.delta[j])
    return c, np.array(A), np.array(b), bnd


def _build_lp_matrices(seq: List[int], inst: ALPInstance):
    """Full pairwise separation constraints (n(n-1)/2 rows).
    Enforces x[seq[m]] >= x[seq[l]] + s[seq[l]][seq[m]] for ALL l < m.
    Consecutive-only is insufficient when the triangle inequality is
    violated — which occurs routinely in wake-vortex matrices
    (Beasley et al. 2000, §2.2, Eqs. 7/12)."""
    n   = inst.n
    c   = np.concatenate([np.zeros(n), inst.g, inst.h])
    bnd = ([(inst.r[j], inst.d[j]) for j in range(n)] + [(0.0, None)] * 2 * n)
    A, b = [], []
    for l in range(n - 1):
        for m in range(l + 1, n):
            row = np.zeros(3 * n)
            row[seq[l]] = 1.0; row[seq[m]] = -1.0
            A.append(row); b.append(-inst.s[seq[l]][seq[m]])
    for j in range(n):
        row = np.zeros(3 * n)
        row[j] = -1.0; row[n + j] = -1.0
        A.append(row); b.append(-inst.delta[j])
    for j in range(n):
        row = np.zeros(3 * n)
        row[j] = 1.0; row[2*n + j] = -1.0
        A.append(row); b.append(inst.delta[j])
    return c, np.array(A), np.array(b), bnd


def solve_lp(seq: List[int], inst: ALPInstance) -> float:
    """Full pairwise LP → objective scalar."""
    c, A, b, bnd = _build_lp_matrices(seq, inst)
    res = linprog(c, A_ub=A, b_ub=b, bounds=bnd,
                  method='highs', options={'disp': False, 'presolve': True})
    return float(res.fun) if res.status == 0 else float('inf')


def solve_stage2(seq: List[int],
                 inst: ALPInstance) -> Tuple[float, Optional[np.ndarray]]:
    """Full pairwise LP → (objective, landing_times) or (inf, None)."""
    c, A, b, bnd = _build_lp_matrices(seq, inst)
    res = linprog(c, A_ub=A, b_ub=b, bounds=bnd,
                  method='highs', options={'disp': False, 'presolve': True})
    if res.status != 0: return float('inf'), None
    return float(res.fun), res.x[: inst.n]


def evaluate(seq: List[int], inst: ALPInstance) -> float:
    """Fully-feasible evaluation: O(n²) pairwise check + full pairwise LP.
    Used to update the fully-feasible incumbent and for final reporting."""
    return solve_lp(seq, inst) if is_feasible(seq, inst) else float('inf')


def evaluate_semi(seq: List[int], inst: ALPInstance) -> float:
    """Semi-feasible evaluation: O(n) consecutive check + consecutive-only LP.
    Used by SA chains for acceptance decisions.  Cheaper than evaluate;
    may admit sequences with non-consecutive separation violations, but
    those will be caught when the LP is re-solved under full constraints
    at reporting time.  Returns inf for sequences infeasible under even
    the relaxed consecutive check."""
    if not is_feasible_old(seq, inst):
        return float('inf')
    c, A, b, bnd = _build_lp_matrices_old(seq, inst)
    res = linprog(c, A_ub=A, b_ub=b, bounds=bnd,
                  method='highs', options={'disp': False, 'presolve': True})
    return float(res.fun) if res.status == 0 else float('inf')


def _eval_worker(args):
    seq, inst = args
    return evaluate(seq, inst)


def batch_evaluate(seqs: List[List[int]], inst: ALPInstance,
                   n_workers: int = N_CPU) -> List[float]:
    if len(seqs) <= 4 or n_workers == 1:
        return [evaluate(s, inst) for s in seqs]
    with ProcessPoolExecutor(max_workers=min(n_workers, len(seqs)),
                             mp_context=_CTX) as ex:
        return list(ex.map(_eval_worker, [(s, inst) for s in seqs]))


# ═══════════════════════════════════════════════════════════════════════════
# 4.  MPDS PRIORITY INDEX  (Eq. 16)
# ═══════════════════════════════════════════════════════════════════════════

def mpds_idx(t: float, k: int, j: int, inst: ALPInstance,
             K1: float, K2: float, K3: float, K4: float) -> float:
    s_kj   = inst.s[k][j] if k >= 0 else 0.0
    t_land = max(inst.r[j], t + s_kj)
    slack  = max(inst.delta[j] - t_land, 0.0)
    pen    = (inst.g[j] * max(inst.delta[j] - t_land, 0.0) +
              inst.h[j] * max(t_land - inst.delta[j], 0.0))
    rwait  = max(inst.r[j] - t, 0.0)
    e      = 1e-10
    return (math.exp(-slack / (K1 + e)) * math.exp(-s_kj / (K2 * inst.s_bar + e)) *
            math.exp(-rwait / (K3 + e)) * math.exp(-pen   / (K4 + e)))


# ═══════════════════════════════════════════════════════════════════════════
# 5.  INITIAL SOLUTION GENERATORS + DOUBLE-BRIDGE
# ═══════════════════════════════════════════════════════════════════════════

def gen_erd(inst):  return sorted(range(inst.n), key=lambda j: inst.r[j])
def gen_edd(inst):  return sorted(range(inst.n), key=lambda j: inst.delta[j])

def gen_mdd(inst):
    rem, seq, t, k = list(range(inst.n)), [], 0.0, -1
    while rem:
        bst = min(rem, key=lambda j: max(inst.delta[j],
                                         t + (inst.s[k][j] if k >= 0 else 0.0)))
        seq.append(bst); t = max(inst.r[bst], t + (inst.s[k][bst] if k >= 0 else 0.0))
        k = bst; rem.remove(bst)
    return seq

def gen_atc(inst, K: float = 2.0):
    rem, seq, t, k = list(range(inst.n)), [], 0.0, -1
    while rem:
        sb  = np.mean([inst.s[k][j] for j in rem] if k >= 0 else [inst.s_bar])
        wj  = {j: (inst.g[j] + inst.h[j]) / 2 for j in rem}
        slk = {j: max(inst.delta[j] - inst.s_bar - t, 0.0) for j in rem}
        bst = max(rem, key=lambda j: wj[j] / inst.s_bar
                  * math.exp(-slk[j] / (K * sb + 1e-9)))
        seq.append(bst); t = max(inst.r[bst], t + (inst.s[k][bst] if k >= 0 else 0.0))
        k = bst; rem.remove(bst)
    return seq

def gen_mpds(inst):
    K1, K2, K3, K4 = inst.mpds_params()
    rem, seq, t, k = list(range(inst.n)), [], 0.0, -1
    while rem:
        bst = max(rem, key=lambda j: mpds_idx(t, k, j, inst, K1, K2, K3, K4))
        seq.append(bst); t = max(inst.r[bst], t + (inst.s[k][bst] if k >= 0 else 0.0))
        k = bst; rem.remove(bst)
    return seq

def _double_bridge(seq: List[int], rng: random.Random) -> List[int]:
    """
    Double-bridge (4-opt) perturbation: A|B|C|D → A|C|B|D.
    Unreachable by any 3-opt move; standard ILS escape operator for
    permutation problems (Applegate et al., 2006 §4.5).
    """
    n = len(seq)
    if n < 8:
        s = seq[:]
        i, j, k = sorted(rng.sample(range(n), 3))
        s[i], s[j], s[k] = s[k], s[i], s[j]
        return s
    a, b, c = sorted(rng.sample(range(1, n), 3))
    return seq[:a] + seq[b:c] + seq[a:b] + seq[c:]


# ═══════════════════════════════════════════════════════════════════════════
# 6.  NEIGHBOURHOOD OPERATORS
# ═══════════════════════════════════════════════════════════════════════════

def op_swap(s, i, j):
    s = s[:]; s[i], s[j] = s[j], s[i]; return s

def op_insert(s, i, j):
    s = s[:]; ac = s.pop(i); s.insert(j if j <= i else j - 1, ac); return s

def op_reverse(s, i, j):
    if i > j: i, j = j, i
    s = s[:]; s[i:j+1] = s[i:j+1][::-1]; return s

def op_or_opt_2(s, i, j):
    s = s[:]; n = len(s); i = min(i, n - 2)
    block = s[i:i+2]; del s[i:i+2]; ins = j % len(s); s[ins:ins] = block; return s

def op_or_opt_3(s, i, j):
    s = s[:]; n = len(s); i = min(i, n - 3)
    block = s[i:i+3]; del s[i:i+3]; ins = j % len(s); s[ins:ins] = block; return s

_OPS_SMALL = (op_swap, op_insert, op_reverse)
_OPS_LARGE = (op_swap, op_insert, op_reverse, op_or_opt_2, op_or_opt_3)

def neighbour(seq, rng, n):
    ops = _OPS_LARGE if n >= 50 else _OPS_SMALL
    op  = rng.choice(ops)
    i   = rng.randint(0, n - 1)
    j   = rng.randint(0, n - 1)
    while j == i: j = rng.randint(0, n - 1)
    return op(seq, i, j)


# ═══════════════════════════════════════════════════════════════════════════
# 7.  TEMPERATURE CALIBRATION  (Eq. 30)
# ═══════════════════════════════════════════════════════════════════════════

def calibrate_T0(seq, inst, n_cal=150, chi0=0.50, seed=0):
    """Calibrate initial temperature using evaluate_semi (consistent with
    the acceptance criterion used inside run_sa)."""
    rng = random.Random(seed)
    f0  = evaluate_semi(seq, inst)
    if math.isinf(f0): return 100.0
    dps = [d for nb in (neighbour(seq, rng, inst.n) for _ in range(n_cal))
           if not math.isinf(fn := evaluate_semi(nb, inst))
           for d in ([fn - f0] if fn > f0 else [])]
    return max(-np.mean(dps) / math.log(chi0 + 1e-12), 1e-3) if dps else 1.0


# ═══════════════════════════════════════════════════════════════════════════
# 8.  SINGLE SA CHAIN  (dual-track feasibility + time limit)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SAParams:
    alpha:  float = 0.99        # cooling rate
    N_iter: int   = 120         # iterations per temperature
    T_min:  float = 1e-4        # minimum temperature
    I_max:  int   = 600         # maximum number of outer iterations
    M_stag: int   = 60          # stagnation limit (outer iterations)
    chi0:   float = 0.50        # initial acceptance probability for T0 calibration


def run_sa(seq0, inst, p: SAParams, seed=0, T0=None,
           t_deadline: float = None):
    """
    Single SA chain with reactive temperature adaptation and dual-track
    feasibility tracking.

    Dual-track feasibility
    ──────────────────────
    Acceptance decisions use evaluate_semi (consecutive-only LP, O(n)
    pre-filter), which is cheaper and admits a larger neighbourhood.
    The semi-feasible incumbent (pb_semi, fb_semi) drives SA guidance:
    temperature calibration, acceptance, stagnation counter, reheating,
    and ILS perturbation seeds all reference fb_semi.

    Whenever fb_semi improves, evaluate (full pairwise LP, O(n²) pre-
    filter) is called once on the new incumbent to update the fully-
    feasible incumbent (pb_feas, fb_feas).  Only pb_feas is reported
    in gap tables and sent to verify_schedule.

    Time limit
    ──────────
    t_deadline is an absolute time.perf_counter() timestamp.  The outer
    loop exits as soon as time.perf_counter() >= t_deadline, preserving
    whatever incumbent has been found up to that point.

    Alternate-solution tracking
    ───────────────────────────
    alt_set tracks distinct sequences within _alt_tol of fb_semi.
    Reset whenever a strictly better semi-feasible solution is found.

    Returns
    -------
    (pb_semi, fb_semi, stats_dict)
    stats_dict keys:
        obj           – fb_semi  (SA guidance objective)
        obj_feas      – fb_feas  (fully-feasible objective; inf if none)
        pi_feas       – pb_feas  (best fully-feasible sequence; None if none)
        time          – total wall time
        t_best        – wall time to best semi-feasible   (t_best_sa)
        t_best_feas   – wall time to best fully-feasible
        history       – fb_semi per outer iteration
        init_obj      – objective of the starting sequence (semi)
        n_alt_seqs    – distinct sequences near fb_semi
        alpha_history – cooling rate per outer iteration
    """
    # ── Reactive-adaptation constants ────────────────────────────────
    CHI_TARGET  = 0.20
    ALPHA_STEP  = 0.005
    ALPHA_LO    = 0.80
    ALPHA_HI    = 0.999
    MAX_REHEATS = 2
    T_REHEAT    = 2.0

    rng   = random.Random(seed)
    n     = inst.n
    T     = T0 or calibrate_T0(seq0, inst, seed=seed, chi0=p.chi0)
    alpha = p.alpha

    pi = seq0[:]
    f  = evaluate_semi(pi, inst)           # semi-feasible objective of start

    # ── Semi-feasible incumbent ───────────────────────────────────────
    pb_semi, fb_semi = pi[:], f
    init_obj         = f

    # ── Fully-feasible incumbent (seed from starting sequence) ────────
    f_full_start = evaluate(pi, inst)
    pb_feas      = pi[:] if not math.isinf(f_full_start) else None
    fb_feas      = f_full_start

    _alt_tol = lambda fb_: max(0.5, abs(fb_) * 5e-5)
    alt_set       = {tuple(pi)}
    stag          = 0
    n_reheats     = 0
    history       = []
    alpha_history = []
    t_best_sa     = 0.0    # wall time to latest improvement in fb_semi
    t_best_feas   = 0.0    # wall time to latest improvement in fb_feas
    t0            = time.perf_counter()

    for outer in range(p.I_max):
        # ── Per-iteration time-limit check ────────────────────────────
        if t_deadline is not None and time.perf_counter() >= t_deadline:
            break

        improved   = False
        n_accepted = 0
        n_tried    = 0

        for _ in range(p.N_iter):
            pi2 = neighbour(pi, rng, n)
            f2  = evaluate_semi(pi2, inst)   # inf if fails old check or old LP
            if math.isinf(f2):
                continue
            n_tried += 1
            dlt = f2 - f
            if dlt <= 0 or rng.random() < math.exp(-dlt / max(T, 1e-15)):
                pi, f = pi2, f2
                n_accepted += 1

            # ── Update semi-feasible best ─────────────────────────────
            if f < fb_semi - 1e-9:
                pb_semi, fb_semi, improved = pi[:], f, True
                t_best_sa = time.perf_counter() - t0
                alt_set   = {tuple(pi)}

                # ── Check full feasibility of new semi-best ───────────
                # One full-pairwise LP call per strict semi-improvement.
                f_full = evaluate(pi, inst)
                if not math.isinf(f_full) and f_full < fb_feas - 1e-9:
                    pb_feas, fb_feas = pi[:], f_full
                    t_best_feas = time.perf_counter() - t0

            elif f <= fb_semi + _alt_tol(fb_semi) and len(alt_set) < _MAX_ALT_SEQS:
                alt_set.add(tuple(pi))

        # ── Reactive α update ─────────────────────────────────────────
        chi   = n_accepted / max(n_tried, 1)
        alpha = (max(ALPHA_LO, alpha - ALPHA_STEP)
                 if chi > CHI_TARGET
                 else min(ALPHA_HI, alpha + ALPHA_STEP))
        T    *= alpha
        stag  = 0 if improved else stag + 1
        history.append(fb_semi)
        alpha_history.append(alpha)

        if T < p.T_min:
            break

        if stag >= p.M_stag:
            if n_reheats >= MAX_REHEATS:
                break
            T         = T_REHEAT * T
            pi        = pb_semi[:]
            f         = fb_semi
            stag      = 0
            n_reheats += 1

    return pb_semi, fb_semi, {
        'obj':           fb_semi,
        'obj_feas':      fb_feas,
        'pi_feas':       pb_feas,
        'time':          time.perf_counter() - t0,
        't_best':        t_best_sa,
        't_best_feas':   t_best_feas,
        'history':       history,
        'init_obj':      init_obj,
        'n_alt_seqs':    len(alt_set),
        'alpha_history': alpha_history,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 9.  ILS WRAPPER
# ═══════════════════════════════════════════════════════════════════════════

def run_ils(seq0: List[int], inst: ALPInstance, p: SAParams,
            n_restarts: int = 3, seed: int = 0,
            t_deadline: float = None) -> Tuple[List[int], float, dict]:
    """
    Iterated Local Search: SA intensification + double-bridge perturbation.

    Both semi-feasible and fully-feasible bests are aggregated across all
    SA runs within the ILS chain.  ILS perturbation is applied to the
    semi-feasible best so the search explores broadly; the fully-feasible
    best is tracked independently for reporting.

    t_deadline is forwarded to each run_sa call and also checked between
    restarts so the ILS exits cleanly within the time budget.
    """
    rng = random.Random(seed)
    t0  = time.perf_counter()

    pi_best_semi, f_best_semi, st = run_sa(seq0, inst, p, seed=seed,
                                           t_deadline=t_deadline)
    # ── Initialise dual-track accumulators ────────────────────────────
    pi_best_feas = st.get('pi_feas')
    f_best_feas  = st.get('obj_feas', float('inf'))
    t_best_sa    = st['t_best']
    t_best_feas  = st.get('t_best_feas', 0.0)
    all_hist     = [st['history']]
    init_obj     = st['init_obj']
    total_alt    = st['n_alt_seqs']
    alpha_hist   = st.get('alpha_history', [])

    for r in range(n_restarts):
        if t_deadline is not None and time.perf_counter() >= t_deadline:
            break

        # Perturb from the semi-feasible best for maximum exploration.
        pi_pert = _double_bridge(pi_best_semi, rng)
        pi_r, f_r_semi, st_r = run_sa(pi_pert, inst, p, seed=seed + r + 1,
                                       t_deadline=t_deadline)
        all_hist.append(st_r['history'])
        total_alt += st_r['n_alt_seqs']

        # ── Update semi-feasible best ─────────────────────────────────
        if f_r_semi < f_best_semi - 1e-9:
            pi_best_semi, f_best_semi = pi_r, f_r_semi
            t_best_sa  = (time.perf_counter() - t0
                          - st_r['time'] + st_r['t_best'])
            alpha_hist = st_r.get('alpha_history', [])

        # ── Update fully-feasible best ────────────────────────────────
        f_r_feas  = st_r.get('obj_feas', float('inf'))
        pi_r_feas = st_r.get('pi_feas')
        if not math.isinf(f_r_feas) and f_r_feas < f_best_feas - 1e-9:
            f_best_feas  = f_r_feas
            pi_best_feas = pi_r_feas
            t_best_feas  = (time.perf_counter() - t0
                            - st_r['time'] + st_r.get('t_best_feas', 0.0))

    return pi_best_semi, f_best_semi, {
        'obj':           f_best_semi,
        'obj_feas':      f_best_feas,
        'pi_feas':       pi_best_feas,
        'time':          time.perf_counter() - t0,
        't_best':        t_best_sa,
        't_best_feas':   t_best_feas,
        'history':       all_hist[-1],
        'all_hist':      all_hist,
        'init_obj':      init_obj,
        'n_alt_seqs':    total_alt,
        'alpha_history': alpha_hist,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 10.  MULTI-START SA  — parallel chains
# ═══════════════════════════════════════════════════════════════════════════

def _build_starts(inst, n_starts: int = N_CPU):
    starts   = []
    bases    = {
        'MPDS':   gen_mpds(inst), 'EDD':    gen_edd(inst),
        'MDD':    gen_mdd(inst),  'ERD':    gen_erd(inst),
        'ATC_k2': gen_atc(inst, K=2.0), 'ATC_k4': gen_atc(inst, K=4.0),
    }
    for lbl, seq in bases.items():
        starts.append((lbl, seq, len(starts) * 17))
        if len(starts) >= n_starts: break
    base_seqs = list(bases.values())
    rng_db    = random.Random(42)
    for i in range(len(starts), n_starts):
        starts.append((f'DB_{i}', _double_bridge(base_seqs[i % len(base_seqs)], rng_db), i * 31))
    return starts[:n_starts]


def _sa_worker(args):
    """Spawn-safe worker for ProcessPoolExecutor.

    Args tuple layout:
        (label, seq0, inst, p, seed, n_ils, t_deadline)

    Return tuple layout (11 fields — see module docstring):
        0  label
        1  pb_semi       best semi-feasible sequence
        2  fb_semi       best semi-feasible objective
        3  pi_feas       best fully-feasible sequence (or None)
        4  fb_feas       best fully-feasible objective (inf if none)
        5  history       fb_semi per outer iter
        6  t_best_sa     wall time to best semi-feasible
        7  t_best_feas   wall time to best fully-feasible
        8  n_alt_seqs
        9  init_obj
        10 alpha_history
    """
    label, seq0, inst, p, seed, n_ils, t_deadline = args
    if n_ils > 0:
        pb, fb, st = run_ils(seq0, inst, p, n_restarts=n_ils, seed=seed,
                             t_deadline=t_deadline)
    else:
        pb, fb, st = run_sa(seq0, inst, p, seed=seed, t_deadline=t_deadline)
    return (label,
            pb,
            fb,
            st.get('pi_feas'),
            st.get('obj_feas',      float('inf')),
            st.get('history',       []),
            st.get('t_best',        0.0),
            st.get('t_best_feas',   0.0),
            st.get('n_alt_seqs',    1),
            st.get('init_obj',      fb),
            st.get('alpha_history', []))


def ms_sa(inst: ALPInstance, p: SAParams = None,
          n_workers: int = N_CPU,
          n_ils: int = 0,
          t_limit: float = _INSTANCE_TIME_LIMIT) -> Tuple[List[int], float, dict]:
    """
    Parallel multi-start SA with per-chain ILS and dual-track feasibility.

    Time budget
    ───────────
    t_limit (default 3600 s) is converted to an absolute deadline
    (time.perf_counter() + t_limit) and forwarded to every chain worker.
    Each chain's outer loop and ILS restart loop honour this deadline so
    the executor collects all futures well within the budget.

    Result selection
    ────────────────
    The reported objective and sequence are drawn from the best
    fully-feasible incumbent across all chains (fb_feas, pi_feas).
    If no chain found a fully-feasible solution the semi-feasible best
    is returned as a fallback with a warning; it will fail verification.

    Returns
    -------
    (best_pi, best_f, stats_dict)
    best_pi / best_f are the FULLY-FEASIBLE best (or semi fallback).
    stats_dict keys:
        wall_s, t_best (feas), t_best_sa (semi), history, all_histories,
        n_init_optimal, total_alt_seqs, best_f_semi
    """
    p          = p or SAParams()
    starts     = _build_starts(inst, n_starts=n_workers)
    t0         = time.perf_counter()
    t_deadline = t0 + t_limit
    tasks      = [(lbl, seq, inst, p, sd, n_ils, t_deadline)
                  for lbl, seq, sd in starts]

    print(f"  MS-SA: {len(tasks)} chains  |  {n_workers} workers  "
          f"|  ILS/chain: {n_ils}  |  t_limit: {t_limit:.0f}s")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=_CTX) as ex:
        results = list(tqdm(ex.map(_sa_worker, tasks),
                            total=len(tasks), desc="  SA chains",
                            disable=not _TQDM))

    # ── Select best fully-feasible result ──────────────────────────────
    # Tuple field reference: 4=fb_feas, 2=fb_semi, 3=pi_feas, 1=pb_semi,
    #                        5=history, 6=t_best_sa, 7=t_best_feas, 10=alpha_hist
    feas_results = [(r) for r in results if not math.isinf(r[4])]
    has_feas     = len(feas_results) > 0

    if has_feas:
        best_r    = min(feas_results, key=lambda r: r[4])
        best_pi   = best_r[3]   # pi_feas
        best_f    = best_r[4]   # fb_feas
        best_ttb  = best_r[7]   # t_best_feas
    else:
        # No fully-feasible solution found; fall back to best semi-feasible.
        warnings.warn(
            f"{inst.name}: no fully-feasible solution found in any chain; "
            "reporting semi-feasible best — expect verification failure."
        )
        best_r    = min(results, key=lambda r: r[2])
        best_pi   = best_r[1]   # pb_semi
        best_f    = best_r[2]   # fb_semi
        best_ttb  = best_r[6]   # t_best_sa

    best_lbl      = best_r[0]
    best_hist     = best_r[5]
    best_alpha_h  = best_r[10]
    best_f_semi   = min(r[2] for r in results)
    wall          = time.perf_counter() - t0

    _tol = max(0.5, abs(best_f) * 5e-5)
    n_init_optimal = sum(1 for r in results if abs(r[9] - best_f) <= _tol)
    total_alt      = sum(r[8] for r in results
                         if abs((r[4] if has_feas else r[2]) - best_f) <= _tol)

    print(f"\n  {n_init_optimal}/{len(results)} initial solutions already at best objective")
    print(f"  Total distinct near-optimal sequences found: {total_alt}")
    print(f"  MS-SA → feasible={best_f:.2f}  semi={best_f_semi:.2f}"
          f"  (best start: {best_lbl})"
          f"  [wall: {wall:.1f}s  |  TTB(feas): {best_ttb:.1f}s]")

    return best_pi, best_f, {
        'wall_s':         wall,
        't_best':         best_ttb,         # time to best FULLY-FEASIBLE
        't_best_sa':      min(r[6] for r in results),  # time to best semi
        'history':        best_hist,
        'all_histories':  results,
        'n_init_optimal': n_init_optimal,
        'total_alt_seqs': total_alt,
        'best_f_semi':    best_f_semi,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 11.  ADAPTIVE PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

def adaptive_params(n: int) -> Tuple[SAParams, int]:
    """Return (SAParams, n_ils_restarts) scaled to instance size n."""
    if n <= 20:
        sa = SAParams(alpha=0.97,  N_iter=80,  T_min=1e-4, I_max=300,  M_stag=50,  chi0=0.50)
        n_ils = 0
    elif n <= 50:
        sa = SAParams(alpha=0.98,  N_iter=150, T_min=1e-4, I_max=600,  M_stag=80,  chi0=0.50)
        n_ils = 2
    elif n <= 150:
        sa = SAParams(alpha=0.995, N_iter=300, T_min=1e-5, I_max=1200, M_stag=120, chi0=0.40)
        n_ils = 4
    else:
        sa = SAParams(alpha=0.997, N_iter=500, T_min=1e-6, I_max=2000, M_stag=200, chi0=0.35)
        n_ils = 6
    return sa, n_ils


# ═══════════════════════════════════════════════════════════════════════════
# 12.  HYPERPARAMETER TUNING
# ═══════════════════════════════════════════════════════════════════════════

def tune_sa(inst: ALPInstance, known_opt: float,
            n_trials: int = 40, n_workers: int = 8) -> SAParams:
    """Optuna TPE tuning of SAParams on a single instance."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  Optuna not installed — using defaults."); return SAParams()

    def objective(trial):
        p = SAParams(
            alpha  = trial.suggest_float("alpha",  0.90,  0.999),
            N_iter = trial.suggest_int  ("N_iter", 40,    300),
            T_min  = trial.suggest_float("T_min",  1e-6,  1e-2, log=True),
            I_max  = trial.suggest_int  ("I_max",  100,   800),
            M_stag = trial.suggest_int  ("M_stag", 20,    200),
            chi0   = trial.suggest_float("chi0",   0.25,  0.80),
        )
        _, f, _ = ms_sa(inst, p, n_workers=n_workers)
        return (f - known_opt) / max(known_opt, 1.0) * 100.0

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=3),
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)
    bp = study.best_params
    print(f"  SA best gap: {study.best_value:.2f}%  params: {bp}")
    return SAParams(alpha=bp['alpha'], N_iter=bp['N_iter'], T_min=bp['T_min'],
                    I_max=bp['I_max'], M_stag=bp['M_stag'], chi0=bp['chi0'])


# ═══════════════════════════════════════════════════════════════════════════
# 13.  PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def plot_gantt(seq: List[int], inst: ALPInstance,
               method: str = "", obj: float = None,
               save_dir: str = "plots") -> None:
    import matplotlib.patches as mpatches
    from pathlib import Path
    _, x1 = solve_stage2(seq, inst)
    if x1 is None:
        print(f"  plot_gantt: infeasible for {inst.name}, skipped.")
        return
    Path(save_dir).mkdir(exist_ok=True)
    n = inst.n
    fig, ax = plt.subplots(figsize=(11, max(6, n * 0.22)))
    ax.set_facecolor('#f9f9f9')
    for l in range(n - 1):
        j, k = seq[l], seq[l + 1]
        ax.barh(k + 1, width=inst.s[j][k], left=x1[j], height=0.52,
                color='#e8721c', alpha=0.45, zorder=0, linewidth=0)
    for j in range(n):
        y = j + 1
        ax.plot([inst.r[j], inst.d[j]], [y, y], color='#333333', linewidth=0.9, zorder=1)
        ax.plot([inst.r[j], inst.d[j]], [y, y], marker='*', markersize=6,
                color='#333333', linestyle='none', zorder=2)
        ax.plot(inst.delta[j], y, marker='s', markersize=5, color='#333333',
                markerfacecolor='none', markeredgewidth=1.0, linestyle='none', zorder=3)
        ax.plot(x1[j], y, marker='o', markersize=5, color='#1a6faf',
                markerfacecolor='none', markeredgewidth=1.2, linestyle='none', zorder=4)
    ax.set_xlabel('t / s', fontsize=11); ax.set_ylabel('Aircraft index', fontsize=11)
    ax.set_ylim(0, n + 1)
    tick_step = max(1, 5 * round(n / 50))
    ax.set_yticks(range(tick_step, n + 1, tick_step))
    ax.grid(True, linestyle=':', linewidth=0.5, color='#cccccc', alpha=0.8)
    ax.set_axisbelow(True)
    slt_lbl = f'SLT — {method}' + (f'  (obj={obj:.2f})' if obj is not None else '')
    ax.legend(handles=[
        plt.Line2D([0],[0], color='#333333', lw=0.9, marker='*', markersize=6,
                   label='Time window  [r, d]'),
        plt.Line2D([0],[0], marker='s', color='#333333', lw=0, markersize=5,
                   markerfacecolor='none', label='Target δ'),
        plt.Line2D([0],[0], marker='o', color='#1a6faf', lw=0, markersize=5,
                   markerfacecolor='none', label=slt_lbl),
        mpatches.Patch(facecolor='#e8721c', alpha=0.45,
                       label=r'Wake-vortex separation zone  $[x_j,\; x_j + s_{jk}]$'),
    ], loc='lower right', fontsize=9, framealpha=0.85, edgecolor='#aaaaaa')
    ax.set_title(inst.name, fontsize=11, pad=8)
    plt.tight_layout()
    fname = (f"{save_dir}/gantt/gantt_{inst.name}"
             f"_{method.replace('-','_').replace(' ','_')}.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {fname}")


def plot_sa_convergence(all_histories: list, inst_name: str,
                        save_dir: str = "plots") -> None:
    """Convergence plot of fb_semi per outer iteration for each chain.
    Tuple field reference: 0=label, 2=fb_semi, 4=fb_feas, 5=history."""
    from pathlib import Path
    if not all_histories: return
    Path(save_dir).mkdir(exist_ok=True)
    # Use fb_feas when available, else fb_semi, for "best objective" label.
    def _chain_obj(r):
        return r[4] if not math.isinf(r[4]) else r[2]
    finite  = [r for r in all_histories if not math.isinf(_chain_obj(r))]
    if not finite: return
    best_f  = min(_chain_obj(r) for r in finite)
    palette = plt.cm.tab20.colors
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, r in enumerate(all_histories):
        lbl, fb_chain, hist = r[0], _chain_obj(r), r[5]
        if not hist or math.isinf(fb_chain): continue
        is_best = abs(fb_chain - best_f) < 1e-4
        ax.plot(hist, linewidth=0.7 if is_best else 0.5,
                alpha=0.7 if is_best else 0.5,
                linestyle='-' if is_best else '--',
                color=palette[i % len(palette)],
                label=f'{lbl} ({fb_chain:.0f})')
    ax.set_xlabel('Outer Iteration', fontsize=10)
    ax.set_ylabel('Best Semi-Feasible Objective (log scale)', fontsize=10)
    ax.set_title(f'SA Chain Convergence — {inst_name}', fontsize=11)
    ax.set_yscale('log'); ax.legend(fontsize=7, ncol=4, loc='upper right')
    ax.grid(alpha=0.22); plt.tight_layout()
    fname = f"{save_dir}/convergence/convergence_{inst_name}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {fname}")


def plot_gap_summary(all_results: list, save_dir: str = "plots") -> None:
    from pathlib import Path
    if not all_results: return
    Path(save_dir).mkdir(exist_ok=True)

    def pct(r):
        f = r['results'].get('MS-SA', (float('inf'),))[0]
        o = r.get('opt')
        if o is None or math.isinf(f): return float('nan')
        return (f - o) / o * 100.0

    names = [r['inst'] for r in all_results]
    gaps  = [pct(r)    for r in all_results]
    clrs  = ['gold' if (not math.isnan(g) and g < -0.01) else '#1a6faf' for g in gaps]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.0), 5))
    bars = ax.bar(names, gaps, color=clrs, alpha=0.85)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    for bar, g in zip(bars, gaps):
        if not math.isnan(g) and g < -0.01:
            ax.text(bar.get_x() + bar.get_width()/2, g - 0.4, f'{g:.2f}%',
                    ha='center', fontsize=7, color='darkorange', fontweight='bold')
    ax.set_ylabel('Gap to known reference (%)', fontsize=10)
    ax.set_title('MS-SA Optimality Gap', fontsize=10)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.22); plt.tight_layout()
    fname = f"{save_dir}/gap_summary.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {fname}")


def plot_alt_solutions(all_results: list, save_dir: str = "plots") -> None:
    from pathlib import Path
    if not all_results: return
    Path(save_dir).mkdir(exist_ok=True)
    names = [r['inst']                   for r in all_results]
    alts  = [r.get('total_alt_seqs', 0)  for r in all_results]
    inits = [r.get('n_init_optimal',  0) for r in all_results]
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.0), 4))
    ax.bar(x - w/2, alts,  w, color='#1a6faf', alpha=0.85, label='Distinct alt-optimal seqs')
    ax.bar(x + w/2, inits, w, color='#c0392b', alpha=0.85, label='Heuristics at optimum (pre-SA)')
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title('Solution Degeneracy — Alternate Optimal Schedules', fontsize=10)
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.22); plt.tight_layout()
    fname = f"{save_dir}/alt_solutions.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {fname}")


def plot_alpha_trajectory(all_histories: list, inst_name: str,
                          save_dir: str = "plots") -> None:
    """Reactive cooling-rate trajectory per chain.
    Tuple field reference: 0=label, 2=fb_semi, 4=fb_feas, 10=alpha_history."""
    from pathlib import Path
    Path(save_dir).mkdir(exist_ok=True)

    def _chain_obj(r):
        return r[4] if not math.isinf(r[4]) else r[2]

    finite  = [r for r in all_histories if not math.isinf(_chain_obj(r))]
    if not finite: return
    best_f  = min(_chain_obj(r) for r in finite)
    palette = plt.cm.tab20.colors

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axhline(0.999, color='#cccccc', linewidth=0.6, linestyle='--', zorder=0)
    ax.axhline(0.80,  color='#cccccc', linewidth=0.6, linestyle='--', zorder=0)
    ax.axhline(0.20,  color='#e8721c', linewidth=0.5, linestyle=':',  zorder=0,
               label='χ* = 0.20  (acceptance target)')

    plotted = 0
    for i, r in enumerate(all_histories):
        lbl, fb, alpha_hist = r[0], _chain_obj(r), r[10]
        if not alpha_hist or math.isinf(fb): continue
        is_best = abs(fb - best_f) < 1e-4
        ax.plot(range(1, len(alpha_hist) + 1), alpha_hist,
                color=palette[i % len(palette)],
                linewidth=1.0 if is_best else 0.4,
                alpha=0.85    if is_best else 0.35,
                linestyle='-' if is_best else '--',
                label=f'{lbl}' + (' ★' if is_best else ''))
        plotted += 1

    if plotted == 0:
        plt.close(); return

    ax.set_xlabel('Outer iteration (temperature level)', fontsize=10)
    ax.set_ylabel('Cooling rate α', fontsize=10)
    ax.set_title(f'Reactive α Trajectory — {inst_name}', fontsize=11)
    ax.set_ylim(0.78, 1.002)
    ax.legend(fontsize=7, ncol=4, loc='lower right', framealpha=0.50)
    ax.grid(alpha=0.2); plt.tight_layout()
    fname = f"{save_dir}/alpha trajectory/alpha_trajectory_{inst_name}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {fname}")


def plot_seed_improvement(all_histories: list, inst_name: str,
                          known_opt: float = None,
                          save_dir: str = "plots") -> None:
    """Paired bar: heuristic seed objective vs SA final (fully-feasible when
    available, else semi-feasible) per chain.
    Tuple field reference: 0=label, 2=fb_semi, 4=fb_feas, 9=init_obj."""
    from pathlib import Path
    Path(save_dir).mkdir(exist_ok=True)

    def _chain_obj(r):
        return r[4] if not math.isinf(r[4]) else r[2]

    rows = [(r[0], r[9], _chain_obj(r)) for r in all_histories
            if not math.isinf(_chain_obj(r)) and not math.isinf(r[9])]
    if not rows: return

    rows.sort(key=lambda x: -x[1])
    labels    = [r[0] for r in rows]
    init_obj  = [r[1] for r in rows]
    final_obj = [r[2] for r in rows]
    best_f    = min(final_obj)

    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 0.9), 5))
    ax.bar(x - w/2, init_obj, w, color='#aec6e8', edgecolor='#1a6faf',
           linewidth=0.8, hatch='///', alpha=0.7, label='Heuristic seed (pre-SA)')
    colors = ['#c0392b' if abs(f - best_f) < 1e-4 else '#1a6faf' for f in final_obj]
    ax.bar(x + w/2, final_obj, w, color=colors, alpha=0.85, label='SA final objective')
    for xi, (f_i, f_f) in enumerate(zip(init_obj, final_obj)):
        delta = f_i - f_f
        if delta > max(best_f * 5e-4, 0.5):
            ax.annotate('', xy=(xi + w/2, f_f + (f_i - f_f)*0.05),
                        xytext=(xi - w/2, f_i - (f_i - f_f)*0.05),
                        arrowprops=dict(arrowstyle='->', color='#555555',
                                        lw=0.8, connectionstyle='arc3,rad=0.15'))
    if known_opt is not None:
        ax.axhline(known_opt, color='black', linewidth=0.9,
                   linestyle='--', label=f'Known optimum ({known_opt})')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Objective value', fontsize=10)
    ax.set_title(f'Seed Quality vs. SA Improvement — {inst_name}', fontsize=11)
    ax.legend(fontsize=9, loc='upper right', framealpha=0.85)
    ax.grid(axis='y', alpha=0.22); plt.tight_layout()
    fname = f"{save_dir}/seed improvement/seed_improvement_{inst_name}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {fname}")


def plot_penalty_profile(seq: List[int], inst: ALPInstance,
                         landing_times: np.ndarray,
                         method: str = "", obj: float = None,
                         save_dir: str = "plots") -> None:
    from pathlib import Path
    Path(save_dir).mkdir(exist_ok=True)
    n   = inst.n
    x   = landing_times
    pos = np.arange(1, n + 1)
    early_pen = np.array([inst.g[j] * max(inst.delta[j] - x[j], 0.0) for j in seq])
    late_pen  = np.array([inst.h[j] * max(x[j] - inst.delta[j], 0.0) for j in seq])
    deviation = np.array([x[j] - inst.delta[j] for j in seq])
    fig, ax1 = plt.subplots(figsize=(max(10, n * 0.28), 5))
    ax2 = ax1.twinx()
    ax1.bar(pos, early_pen, color='#1a6faf', alpha=0.82, label='Weighted earliness  $g_j E_j$')
    ax1.bar(pos, late_pen, bottom=early_pen, color='#c0392b', alpha=0.82,
            label='Weighted tardiness  $h_j T_j$')
    ax2.scatter(pos, deviation, marker='D', s=18, color='#555555',
                alpha=0.6, zorder=3, label='$x_j - \\delta_j$ (s)')
    ax2.axhline(0, color='#888888', linewidth=0.6, linestyle=':')
    ax2.set_ylabel('Deviation from target δ_j  (s)', fontsize=9, color='#555555')
    ax2.tick_params(axis='y', labelcolor='#555555')
    ax1.set_xlabel('Landing position in sequence', fontsize=10)
    ax1.set_ylabel('Penalty cost', fontsize=10)
    title = f'Per-Aircraft Penalty Profile — {inst.name}'
    if method: title += f'  [{method}]'
    if obj is not None: title += f'  (total = {obj:.2f})'
    ax1.set_title(title, fontsize=11)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=9, loc='upper right', framealpha=0.85)
    ax1.set_xlim(0, n + 1); ax1.grid(axis='y', alpha=0.2); ax1.set_axisbelow(True)
    # Set x-axis to show on every aircraft for n <= 20, else every 2 or 5 aircraft.
    tick_step = 1 if n <= 20 else (2 if n <= 50 else 5)
    ax1.set_xticks(pos[::tick_step])
    ax1.set_xticklabels([f'A{j}' for j in seq][::tick_step], fontsize=8)

    total_pen = early_pen + late_pen
    top_n = min(5, int((total_pen > 0).sum()))
    if top_n > 0:
        threshold = np.sort(total_pen)[-top_n]
        for l, (p_val, j) in enumerate(zip(total_pen, seq)):
            if p_val >= threshold and p_val > 0:
                ax1.text(pos[l], p_val + max(total_pen) * 0.01,
                         f'A{j}', ha='center', va='bottom', fontsize=7, color='#333333')
    plt.tight_layout()
    tag = method.replace('-', '_').replace(' ', '_')
    fname = f"{save_dir}/penalty profile/penalty_profile_{inst.name}_{tag}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {fname}")


# ═══════════════════════════════════════════════════════════════════════════
# 14.  SCHEDULE VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass, field


@dataclass
class ConstraintViolation:
    group:     str
    index:     int
    lhs:       float
    rhs:       float
    violation: float
    detail:    str


@dataclass
class VerificationReport:
    instance:       str
    sequence:       List[int]
    n_aircraft:     int
    obj_reported:   float
    obj_recomputed: float
    obj_lp_recheck: float
    landing_times:  Optional[np.ndarray]
    violations:     List[ConstraintViolation] = field(default_factory=list)
    passed:         bool  = True
    tol:            float = 1e-4

    @property
    def n_violations(self) -> int:
        return len(self.violations)

    @property
    def groups_failed(self) -> List[str]:
        return sorted({v.group for v in self.violations})

    @property
    def max_violation(self) -> float:
        return max((v.violation for v in self.violations), default=0.0)

    def summary(self) -> str:
        lines = [
            f"{'═'*68}",
            f"  Verification — {self.instance}   n={self.n_aircraft}",
            f"{'─'*68}",
            f"  Result         : {'✓ PASS' if self.passed else '✗ FAIL'}",
            f"  Obj reported   : {self.obj_reported:.6f}",
            f"  Obj recomputed : {self.obj_recomputed:.6f}  "
            f"(Δ={abs(self.obj_reported - self.obj_recomputed):.2e})",
            f"  Obj LP re-solve: {self.obj_lp_recheck:.6f}  "
            f"(Δ={abs(self.obj_reported - self.obj_lp_recheck):.2e})",
            f"  Tolerance      : {self.tol:.0e}",
            f"  Violations     : {self.n_violations}",
        ]
        if self.violations:
            lines.append(f"  Groups failed  : {', '.join(self.groups_failed)}")
            lines.append(f"  Max violation  : {self.max_violation:.4e}")
            lines.append(f"{'─'*68}")
            lines.append(f"  {'Group':<6} {'Aircraft':>8} {'LHS':>14} "
                         f"{'RHS':>14} {'Violation':>12}  Detail")
            lines.append(f"  {'─'*6} {'─'*8} {'─'*14} {'─'*14} {'─'*12}  {'─'*20}")
            for v in sorted(self.violations, key=lambda x: -x.violation):
                lines.append(
                    f"  {v.group:<6} {v.index:>8} {v.lhs:>14.4f} "
                    f"{v.rhs:>14.4f} {v.violation:>12.4e}  {v.detail}")
        lines.append(f"{'═'*68}")
        return "\n".join(lines)


def _lp_recompute(seq: List[int],
                  inst: ALPInstance) -> Tuple[float, Optional[np.ndarray]]:
    """Independent Stage-2 LP re-solve using a freshly built matrix with
    the SAME full pairwise separation constraints as solve_stage2.
    Using the same formulation is required for the C9 cross-check to be
    meaningful: an objective discrepancy would previously arise solely from
    the constraint-set mismatch (consecutive-only vs. full pairwise), not
    from any solver inconsistency.  Fixed to use n(n-1)/2 separation rows."""
    n   = inst.n
    c   = np.concatenate([np.zeros(n), inst.g, inst.h])
    bnd = [(inst.r[j], inst.d[j]) for j in range(n)] + [(0.0, None)] * 2 * n
    A, b = [], []
    # Full pairwise separation (identical to _build_lp_matrices)
    for l in range(n - 1):
        for m in range(l + 1, n):
            row = np.zeros(3 * n)
            row[seq[l]] = 1.0; row[seq[m]] = -1.0
            A.append(row); b.append(-inst.s[seq[l]][seq[m]])
    for j in range(n):
        row = np.zeros(3 * n); row[j] = -1.0; row[n + j] = -1.0
        A.append(row); b.append(-inst.delta[j])
    for j in range(n):
        row = np.zeros(3 * n); row[j] = 1.0; row[2*n + j] = -1.0
        A.append(row); b.append(inst.delta[j])
    res = linprog(c, A_ub=np.array(A), b_ub=np.array(b), bounds=bnd,
                  method='highs',
                  options={'disp': False, 'presolve': True,
                           'dual_feasibility_tolerance':   1e-9,
                           'primal_feasibility_tolerance': 1e-9})
    if res.status != 0:
        return float('inf'), None
    return float(res.fun), res.x[:n]


def verify_schedule(seq: List[int],
                    inst: ALPInstance,
                    tol: float = 1e-4) -> Tuple[bool, float, 'VerificationReport']:
    n    = inst.n
    viols: List[ConstraintViolation] = []

    def _add(group, idx, lhs, rhs, viol, detail):
        viols.append(ConstraintViolation(group, idx, lhs, rhs, viol, detail))

    if len(seq) != n:
        _add('C4', -1, len(seq), n, abs(len(seq) - n),
             f"Sequence length {len(seq)} ≠ n={n}")
    else:
        seen = set()
        for pos, j in enumerate(seq):
            if j < 0 or j >= n:
                _add('C4', j, j, n - 1, abs(j) if j < 0 else j - (n-1),
                     f"Aircraft index {j} out of range [0, {n-1}]")
            elif j in seen:
                _add('C4', j, pos, -1, 1.0,
                     f"Aircraft {j} appears more than once in sequence")
            seen.add(j)

    if any(v.group == 'C4' for v in viols):
        rep = VerificationReport(
            instance=inst.name, sequence=seq, n_aircraft=n,
            obj_reported=float('inf'), obj_recomputed=float('inf'),
            obj_lp_recheck=float('inf'), landing_times=None,
            violations=viols, passed=False, tol=tol)
        return False, float('inf'), rep

    obj_lp, x = solve_stage2(seq, inst)
    if x is None:
        _add('C9', -1, float('inf'), 0.0, float('inf'),
             "Stage-2 LP infeasible (solver status ≠ 0)")
        rep = VerificationReport(
            instance=inst.name, sequence=seq, n_aircraft=n,
            obj_reported=float('inf'), obj_recomputed=float('inf'),
            obj_lp_recheck=float('inf'), landing_times=None,
            violations=viols, passed=False, tol=tol)
        return False, float('inf'), rep

    for j in range(n):
        viol = inst.r[j] - x[j]
        if viol > tol:
            _add('C1', j, x[j], inst.r[j], viol,
                 f"x[{j}]={x[j]:.4f} < r[{j}]={inst.r[j]:.4f}")
    for j in range(n):
        viol = x[j] - inst.d[j]
        if viol > tol:
            _add('C2', j, x[j], inst.d[j], viol,
                 f"x[{j}]={x[j]:.4f} > d[{j}]={inst.d[j]:.4f}")
    for l in range(n - 1):
        for m in range(l + 1, n):
            j, k     = seq[l], seq[m]
            required = x[j] + inst.s[j][k]
            viol     = required - x[k]
            if viol > tol:
                _add('C3', k, x[k], required, viol,
                     f"A{k} at {x[k]:.4f} < A{j} ({x[j]:.4f}) "
                     f"+ sep {inst.s[j][k]:.1f} (positions {l},{m})")
    t_greedy = inst.r[seq[0]]
    for l in range(1, n):
        t_greedy = max(inst.r[seq[l]], t_greedy + inst.s[seq[l-1]][seq[l]])
        if t_greedy > inst.d[seq[l]] + tol:
            _add('C5', seq[l], t_greedy, inst.d[seq[l]],
                 t_greedy - inst.d[seq[l]],
                 f"Greedy pass: A{seq[l]} earliest={t_greedy:.4f} > d={inst.d[seq[l]]:.4f}")
    for j in range(n):
        E_j = max(inst.delta[j] - x[j], 0.0)
        if E_j < -tol:
            _add('C6', j, E_j, 0.0, -E_j, f"E[{j}]={E_j:.4f} < 0")
    for j in range(n):
        T_j = max(x[j] - inst.delta[j], 0.0)
        if T_j < -tol:
            _add('C7', j, T_j, 0.0, -T_j, f"T[{j}]={T_j:.4f} < 0")
    obj_recomputed = float(sum(
        inst.g[j] * max(inst.delta[j] - x[j], 0.0) +
        inst.h[j] * max(x[j] - inst.delta[j], 0.0)
        for j in range(n)))
    obj_delta_c8 = abs(obj_recomputed - obj_lp)
    c8_tol = max(0.5, abs(obj_lp) * 1e-4)
    if obj_delta_c8 > c8_tol:
        _add('C8', -1, obj_lp, obj_recomputed, obj_delta_c8,
             f"LP obj={obj_lp:.6f} vs Σ(gE+hT)={obj_recomputed:.6f}")
    obj_recheck, x_recheck = _lp_recompute(seq, inst)
    if math.isinf(obj_recheck):
        _add('C9', -1, float('inf'), obj_lp, float('inf'),
             "Independent LP re-solve returned infeasible")
    else:
        c9_tol = max(0.5, abs(obj_lp) * 1e-4)
        obj_delta_c9 = abs(obj_recheck - obj_lp)
        if obj_delta_c9 > c9_tol:
            _add('C9', -1, obj_recheck, obj_lp, obj_delta_c9,
                 f"Re-solve obj={obj_recheck:.6f} vs original={obj_lp:.6f}")
        if x_recheck is not None:
            for j in range(n):
                if x_recheck[j] < inst.r[j] - tol:
                    _add('C9', j, x_recheck[j], inst.r[j],
                         inst.r[j] - x_recheck[j],
                         f"Re-solve: x[{j}]={x_recheck[j]:.4f} < r={inst.r[j]:.4f}")
                if x_recheck[j] > inst.d[j] + tol:
                    _add('C9', j, x_recheck[j], inst.d[j],
                         x_recheck[j] - inst.d[j],
                         f"Re-solve: x[{j}]={x_recheck[j]:.4f} > d={inst.d[j]:.4f}")

    passed = len(viols) == 0
    report = VerificationReport(
        instance=inst.name, sequence=seq[:], n_aircraft=n,
        obj_reported=obj_lp, obj_recomputed=obj_recomputed,
        obj_lp_recheck=obj_recheck, landing_times=x,
        violations=viols, passed=passed, tol=tol)
    return passed, round(obj_recomputed, 6), report


def verify_all(seq: List[int], inst: ALPInstance,
               tol: float = 1e-4,
               verbose: bool = True,
               raise_on_fail: bool = False) -> Tuple[bool, float]:
    passed, obj, report = verify_schedule(seq, inst, tol=tol)
    if verbose:
        print(report.summary())
    if not passed and raise_on_fail:
        raise AssertionError(
            f"Schedule verification failed for {inst.name}: "
            f"{report.n_violations} violation(s) in groups "
            f"{report.groups_failed}.  Max violation = {report.max_violation:.4e}")
    return passed, obj


# ═══════════════════════════════════════════════════════════════════════════
# 14b.  RESULTS EXPORT
# ═══════════════════════════════════════════════════════════════════════════

import csv, json
from pathlib import Path
from datetime import datetime


def export_results(results_list: list, out_dir: str = "results") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    summary_path   = out / "summary.csv"
    summary_fields = [
        "instance", "n", "known_opt",
        "ms_sa_obj", "gap_pct",
        "wall_s", "ttb_s",
        "total_alt_seqs", "n_init_optimal",
        "verified",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=summary_fields)
        w.writeheader()
        for r in results_list:
            f_sa  = r["results"].get("MS-SA", (float("inf"),))[0]
            opt   = r.get("opt")
            gap   = ((f_sa - opt) / opt * 100.0
                     if opt and not math.isinf(f_sa) else float("nan"))
            wall  = r["results"].get("MS-SA", (None, 0.0))[1]
            ttb   = r.get("ttb", {}).get("MS-SA", float("nan"))
            vr    = r.get("_vreport")
            w.writerow({
                "instance":       r["inst"],
                "n":              r["n"],
                "known_opt":      "" if opt is None else f"{opt:.6f}",
                "ms_sa_obj":      "" if math.isinf(f_sa) else f"{f_sa:.6f}",
                "gap_pct":        "" if math.isnan(gap) else f"{gap:.4f}",
                "wall_s":         f"{wall:.3f}" if wall is not None else "",
                "ttb_s":          f"{ttb:.3f}"  if not math.isnan(ttb) else "",
                "total_alt_seqs": r.get("total_alt_seqs", ""),
                "n_init_optimal": r.get("n_init_optimal", ""),
                "verified":       ("PASS" if (vr and vr.passed) else
                                   "FAIL" if vr else "NOT RUN"),
            })
    print(f"  Saved: {summary_path}")

    sched_path   = out / "schedules.csv"
    sched_fields = [
        "instance", "landing_pos", "aircraft_idx",
        "r", "delta", "d", "x_scheduled",
        "earliness", "tardiness", "penalty", "g", "h",
    ]
    with open(sched_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=sched_fields)
        w.writeheader()
        for r in results_list:
            inst  = r.get("_inst")
            seq   = r.get("pi_sa")
            vr    = r.get("_vreport")
            if inst is None or seq is None: continue
            if vr is not None and vr.landing_times is not None:
                x = vr.landing_times
            else:
                _, x = solve_stage2(seq, inst)
            if x is None: continue
            for pos, j in enumerate(seq):
                E_j = max(inst.delta[j] - x[j], 0.0)
                T_j = max(x[j] - inst.delta[j], 0.0)
                pen = inst.g[j] * E_j + inst.h[j] * T_j
                w.writerow({
                    "instance":     inst.name,
                    "landing_pos":  pos + 1,
                    "aircraft_idx": j,
                    "r":            f"{inst.r[j]:.4f}",
                    "delta":        f"{inst.delta[j]:.4f}",
                    "d":            f"{inst.d[j]:.4f}",
                    "x_scheduled":  f"{x[j]:.4f}",
                    "earliness":    f"{E_j:.4f}",
                    "tardiness":    f"{T_j:.4f}",
                    "penalty":      f"{pen:.4f}",
                    "g":            f"{inst.g[j]:.4f}",
                    "h":            f"{inst.h[j]:.4f}",
                })
    print(f"  Saved: {sched_path}")

    verif_path = out / "verification.txt"
    divider    = "\n" + "─" * 70 + "\n"
    with open(verif_path, "w", encoding="utf-8") as fh:
        fh.write(f"ALP Verification Report\nGenerated : {ts}\n"
                 f"Instances : {len(results_list)}\n")
        for r in results_list:
            vr = r.get("_vreport")
            fh.write(divider)
            fh.write(vr.summary() + "\n" if vr is not None
                     else f"  {r['inst']}: no verification report available.\n")
        fh.write(divider)
    print(f"  Saved: {verif_path}")

    import sys, socket
    meta_path = out / "run_metadata.json"
    meta = {
        "timestamp": ts,
        "hostname":  socket.gethostname(),
        "python":    sys.version,
        "n_cpu":     N_CPU,
        "instances": [
            {
                "name":      r["inst"],
                "n":         r["n"],
                "known_opt": r.get("opt"),
                "obj":       (r["results"].get("MS-SA",(float("inf"),))[0]
                              if not math.isinf(
                                  r["results"].get("MS-SA",(float("inf"),))[0])
                              else None),
                "verified":  (r["_vreport"].passed if r.get("_vreport") else None),
            }
            for r in results_list
        ],
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"  Saved: {meta_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 15.  EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def _gap(f: float, opt: float = None) -> str:
    if opt is None or math.isinf(f): return "N/A"
    return f"{(f - opt) / opt * 100:+.2f}%"


def run_experiment(inst: ALPInstance,
                   known_opt: float = None,
                   sa_p: SAParams = None,
                   n_workers: int = N_CPU,
                   t_limit: float = _INSTANCE_TIME_LIMIT) -> dict:
    """
    Run MS-SA on one instance and export results.

    Gap table and verification use the FULLY-FEASIBLE objective (fb_feas)
    returned by ms_sa.  The semi-feasible best (best_f_semi) is printed
    as a diagnostic to show how much the relaxed search explored beyond
    what fully verified.

    t_limit (default 3600 s) is forwarded to ms_sa, which converts it to
    an absolute deadline shared across all parallel chains.
    """
    sa_adapt, n_ils = adaptive_params(inst.n)
    sa_p = sa_p if sa_p is not None else sa_adapt

    print(f"\n{'═'*70}")
    print(f"  Instance : {inst.name}   n={inst.n}   s̄={inst.s_bar:.0f}s")
    print(f"  CPU cores: {N_CPU}   ILS/chain: {n_ils}   t_limit: {t_limit:.0f}s")
    if known_opt: print(f"  Reference: {known_opt}")
    print(f"{'═'*70}")

    t0 = time.perf_counter()
    pi_sa, f_sa, sa_stats = ms_sa(inst, sa_p, n_workers=n_workers,
                                   n_ils=n_ils, t_limit=t_limit)
    wall          = time.perf_counter() - t0
    t_best        = sa_stats.get('t_best', wall)      # time to best FEASIBLE
    t_best_sa     = sa_stats.get('t_best_sa', wall)   # time to best SEMI
    best_f_semi   = sa_stats.get('best_f_semi', f_sa)

    print(f"\n{'─'*70}")
    print(f"  {'Method':<16} {'Obj (feas)':>12} {'Obj (semi)':>12} {'Gap':>9} "
          f"{'Wall(s)':>9} {'TTB(s)':>8} {'Alt seqs':>9}")
    print(f"  {'─'*16} {'─'*12} {'─'*12} {'─'*9} {'─'*9} {'─'*8} {'─'*9}")
    print(f"  {'MS-SA':<16} {f_sa:>12.2f} {best_f_semi:>12.2f} "
          f"{_gap(f_sa, known_opt):>9} "
          f"{wall:>9.2f} {t_best:>8.2f} "
          f"{sa_stats.get('total_alt_seqs', 0):>9}")
    print(f"{'═'*70}\n")

    # ── Verification (fully-feasible sequence only) ────────────────────
    if pi_sa is None or math.isinf(f_sa):
        print("  ✗ No fully-feasible solution found — skipping verification.")
        passed = False
        vreport = None
    else:
        passed, f_verified = verify_all(pi_sa, inst, tol=1e-4, verbose=True,
                                        raise_on_fail=False)
        _, _, vreport = verify_schedule(pi_sa, inst)
        if not passed:
            print(f"  ✗ Schedule FAILED verification — objective set to inf.")
            f_sa = float('inf')
        else:
            f_sa = f_verified
            if known_opt is not None and f_sa < known_opt - 1e-4:
                print(f"  ⚠  Verified obj {f_sa:.6f} < known reference {known_opt}.")
                print(f"     This is a genuine new best — all constraints satisfied.")

    result = {
        'inst':           inst.name,
        'n':              inst.n,
        'results':        {'MS-SA': (f_sa, wall)},
        'opt':            known_opt,
        'ttb':            {'MS-SA': t_best},
        'total_alt_seqs': sa_stats.get('total_alt_seqs', 0),
        'n_init_optimal': sa_stats.get('n_init_optimal', 0),
        'pi_sa':          pi_sa,
        '_inst':          inst,
        '_vreport':       vreport,
    }
    export_results([result], out_dir=f"results/{inst.name}")

    plot_gantt(pi_sa, inst, method='MS-SA', obj=f_sa)
    plot_sa_convergence(sa_stats.get('all_histories', []), inst.name)
    plot_alpha_trajectory(sa_stats.get('all_histories', []), inst.name)
    plot_seed_improvement(sa_stats.get('all_histories', []), inst.name, known_opt=known_opt)

    if vreport is not None and vreport.landing_times is not None:
        plot_penalty_profile(pi_sa, inst, vreport.landing_times,
                             method='MS-SA', obj=f_sa)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 16.  DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════

def diagnose_instance(inst: ALPInstance) -> None:
    print(f"\n── Instance diagnostic: {inst.name}  (n={inst.n}) ──")
    print(f"   s̄ = {inst.s_bar:.1f} s   max_sep = {inst.s[inst.s > 0].max():.0f} s")
    print(f"   delta range  : [{inst.delta.min():.0f}, {inst.delta.max():.0f}] s  "
          f"(interval ≈ {np.diff(inst.delta).mean():.0f} s)")
    print(f"   window range : r=[{inst.r.min():.0f},{inst.r.max():.0f}]  "
          f"d=[{inst.d.min():.0f},{inst.d.max():.0f}]")
    for name, seq in [("EDD", gen_edd(inst)), ("ERD", gen_erd(inst)),
                      ("MDD", gen_mdd(inst)),  ("MPDS", gen_mpds(inst))]:
        ok  = is_feasible(seq, inst)
        obj = evaluate(seq, inst) if ok else float('inf')
        print(f"   {name:<8}: {'feasible':>12}   obj = {obj:.2f}" if ok
              else f"   {name:<8}: INFEASIBLE")
    rng  = np.random.default_rng(999)
    n_ok = sum(is_feasible(rng.permutation(inst.n).tolist(), inst) for _ in range(500))
    print(f"   Random seq feasibility: {n_ok}/500 ({n_ok/5:.1f}%)\n")


# ═══════════════════════════════════════════════════════════════════════════
# 17.  MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    from pathlib import Path

    DATA_DIR = Path(__file__).parent / 'data'

    OR_DATA = {
        'airland1.txt':  700,      'airland2.txt':  1480,
        'airland3.txt':  820,      'airland4.txt':  2520,
        'airland5.txt':  3100,     'airland6.txt':  24442,
        'airland7.txt':  1550,
        'airland8.txt':  1950,
        'airland9.txt':  5611.70,  'airland10.txt': 12640.42,
        'airland11.txt': 12462.18, 'airland12.txt': 16629.10,
        'airland13.txt': 39287.52,
    }

    SA_full = None #SAParams(alpha=0.99, N_iter=250, T_min=1e-4, I_max=800, M_stag=100)

    print(f"\nSearching for OR Library files in:\n  {DATA_DIR.resolve()}\n")
    found, missing = [], []
    for fname, opt in OR_DATA.items():
        p = DATA_DIR / fname
        if p.exists():
            found.append((p, fname.replace('.txt', ''), opt)); print(f"  [FOUND]   {p.name}")
        else:
            missing.append(fname);                             print(f"  [MISSING] {p.name}")
    if missing:
        print(f"\n  {len(missing)} file(s) not found.")

    ENABLE_OPTUNA = False
    if ENABLE_OPTUNA and found:
        tune_path, tune_name, tune_opt = found[0]
        print(f"\n  [Optuna] Tuning on {tune_name}  (opt={tune_opt})")
        try:
            tune_inst = load_orlib(str(tune_path), tune_name)
            SA_full   = tune_sa(tune_inst, tune_opt, n_trials=40, n_workers=min(8, N_CPU))
            print("  Tuned parameters applied to all runs.\n")
        except Exception as exc:
            print(f"  Optuna failed ({exc}) — using defaults.\n")

    if found:
        print(f"\nRunning {len(found)} instance(s)...\n")
        all_results = []
        for path, name, opt in found:
            try:
                inst = load_orlib(str(path), name)
                diagnose_instance(inst)
                res  = run_experiment(inst, known_opt=opt, #sa_p=SA_full,
                                      n_workers=N_CPU,
                                      t_limit=_INSTANCE_TIME_LIMIT)
                all_results.append(res)
            except Exception as exc:
                print(f"  ERROR on {name}: {exc}\n")

        if len(all_results) > 1:
            print("\n" + "═"*70)
            print("  AGGREGATE RESULTS")
            print(f"  {'Instance':<14} {'n':>5} {'Gap':>9} "
                  f"{'TTB(s)':>8} {'Alt seqs':>10} {'Inits@opt':>10}")
            print("  " + "─"*60)
            for r in all_results:
                print(f"  {r['inst']:<14} {r['n']:>5} "
                      f"{_gap(r['results'].get('MS-SA',(float('inf'),))[0], r['opt']):>9} "
                      f"{r['ttb'].get('MS-SA', 0):>8.1f} "
                      f"{r.get('total_alt_seqs', 0):>10} "
                      f"{r.get('n_init_optimal', 0):>10}")
            print("═"*70)

            print("\n  Running post-benchmark full constraint audit...")
            audit_pass = 0
            for r in all_results:
                inst_r = load_orlib(
                    str(next(p for p, n_, _ in found if n_ == r['inst'])),
                    r['inst'])
                ok, _ = verify_all(r['pi_sa'], inst_r, tol=1e-4,
                                   verbose=False, raise_on_fail=False)
                status = "✓ PASS" if ok else "✗ FAIL"
                print(f"    {r['inst']:<14}: {status}")
                if ok: audit_pass += 1
            print(f"\n  Audit complete: {audit_pass}/{len(all_results)} passed.\n")

            plot_gap_summary(all_results)
            plot_alt_solutions(all_results)
            print("\n  Exporting consolidated results...")
            export_results(all_results, out_dir="results")

    else:
        print("No OR Library files found — running synthetic demo.\n")
        inst = synthetic_instance(n=20, seed=0)
        diagnose_instance(inst)
        run_experiment(inst, known_opt=None,
                       sa_p=SAParams(alpha=0.97, N_iter=100, T_min=1e-3,
                                     I_max=400, M_stag=50),
                       n_workers=N_CPU,
                       t_limit=_INSTANCE_TIME_LIMIT)