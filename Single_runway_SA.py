"""
Aircraft Landing Problem
==========================================================
Target hardware : multi-core CPU
Algorithms      : MS-SA (parallel chains + ILS per chain)
                  Stage-2 LP (HiGHS via SciPy) 
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
N_CPU = os.cpu_count() - 8 or 1

import multiprocessing as _mp
_CTX = _mp.get_context("spawn" if platform.system() == "Windows" else "fork")

_MAX_ALT_SEQS = 100   # cap on distinct alternate-optimal sequences tracked per chain


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

def is_feasible(seq: List[int], inst: ALPInstance) -> bool:
    t = inst.r[seq[0]]
    if t > inst.d[seq[0]]: return False
    for l in range(1, inst.n):
        t = max(inst.r[seq[l]], t + inst.s[seq[l-1]][seq[l]])
        if t > inst.d[seq[l]]: return False
    return True


# ═══════════════════════════════════════════════════════════════════════════
# 3.  STAGE-2 LP  (HiGHS via SciPy)
# ═══════════════════════════════════════════════════════════════════════════

def _build_lp_matrices(seq: List[int], inst: ALPInstance):
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


def solve_lp(seq: List[int], inst: ALPInstance) -> float:
    c, A, b, bnd = _build_lp_matrices(seq, inst)
    res = linprog(c, A_ub=A, b_ub=b, bounds=bnd,
                  method='highs', options={'disp': False, 'presolve': True})
    return float(res.fun) if res.status == 0 else float('inf')


def solve_stage2(seq: List[int],
                 inst: ALPInstance) -> Tuple[float, Optional[np.ndarray]]:
    """Returns (objective, landing_times) or (inf, None) if infeasible."""
    c, A, b, bnd = _build_lp_matrices(seq, inst)
    res = linprog(c, A_ub=A, b_ub=b, bounds=bnd,
                  method='highs', options={'disp': False, 'presolve': True})
    if res.status != 0: return float('inf'), None
    return float(res.fun), res.x[: inst.n]


def evaluate(seq: List[int], inst: ALPInstance) -> float:
    return solve_lp(seq, inst) if is_feasible(seq, inst) else float('inf')


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
    rng = random.Random(seed)
    f0  = evaluate(seq, inst)
    if math.isinf(f0): return 100.0
    dps = [d for nb in (neighbour(seq, rng, inst.n) for _ in range(n_cal))
           if not math.isinf(fn := evaluate(nb, inst))
           for d in ([fn - f0] if fn > f0 else [])]
    return max(-np.mean(dps) / math.log(chi0 + 1e-12), 1e-3) if dps else 1.0


# ═══════════════════════════════════════════════════════════════════════════
# 8.  SINGLE SA CHAIN  (with alternate-solution tracking)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SAParams:
    alpha:  float = 0.99        # cooling rate
    N_iter: int   = 120         # iterations per temperature
    T_min:  float = 1e-4        # minimum temperature
    I_max:  int   = 600         # maximum number of iterations
    M_stag: int   = 60          # maximum stagnation iterations
    chi0:   float = 0.50        # initial acceptance probability for worse solutions (for T0 calibration)

def run_sa_old(seq0, inst, p: SAParams, seed=0, T0=None):
    """
    Single SA chain.

    Alternate-solution tracking
    ───────────────────────────
    Every sequence π visited with f(π) within _ALT_TOL of the running best
    f_best is recorded as a distinct alternative optimal.  The set is reset
    whenever a strictly better solution is found, capped at _MAX_ALT_SEQS
    entries per chain to bound memory usage on large instances.

    Returns
    -------
    (best_seq, best_obj, stats_dict)
    stats_dict keys: obj, time, t_best, history, init_obj, n_alt_seqs
    """
    rng   = random.Random(seed)
    n     = inst.n
    T     = T0 or calibrate_T0(seq0, inst, seed=seed, chi0=p.chi0)
    pi    = seq0[:]
    f     = evaluate(pi, inst)
    pb, fb = pi[:], f
    init_obj = f                     # objective of the starting sequence

    # Tolerance for "equal to best": max of absolute 0.5 or 0.005 % of fb
    _alt_tol = lambda fb_: max(0.5, abs(fb_) * 5e-5)

    alt_set  = {tuple(pi)}           # distinct sequences achieving fb
    stag     = 0
    history  = []
    t_best   = 0.0
    t0       = time.perf_counter()

    for outer in range(p.I_max):
        improved = False
        for _ in range(p.N_iter):
            pi2 = neighbour(pi, rng, n)
            if not is_feasible(pi2, inst): continue
            f2  = evaluate(pi2, inst)
            dlt = f2 - f
            if dlt <= 0 or rng.random() < math.exp(-dlt / max(T, 1e-15)):
                pi, f = pi2, f2
            if f < fb - 1e-9:
                pb, fb, improved = pi[:], f, True
                t_best = time.perf_counter() - t0
                alt_set = {tuple(pi)}          # new best → reset alternates
            elif f <= fb + _alt_tol(fb) and len(alt_set) < _MAX_ALT_SEQS:
                alt_set.add(tuple(pi))         # another sequence at same level
        T    *= p.alpha
        stag  = 0 if improved else stag + 1
        history.append(fb)
        if T < p.T_min or stag >= p.M_stag: break

    return pb, fb, {
        'obj':         fb,
        'time':        time.perf_counter() - t0,
        't_best':      t_best,
        'history':     history,
        'init_obj':    init_obj,
        'n_alt_seqs':  len(alt_set),
    }

def run_sa(seq0, inst, p: SAParams, seed=0, T0=None):
    """
    Single SA chain with reactive temperature adaptation.

    Reactive cooling
    ────────────────
    After each temperature level the actual acceptance rate χ is compared
    to a target χ* = 0.20.  The cooling multiplier α is nudged by at most
    ±ALPHA_STEP per level so the chain self-corrects without jumps:

        α ← clip(α + sign(χ - χ*) × ALPHA_STEP,  ALPHA_LO, ALPHA_HI)

    χ > χ* → chain is too hot (accepting bad moves freely) → cool faster.
    χ < χ* → chain is freezing prematurely               → cool slower.

    Reheating on stagnation
    ───────────────────────
    When M_stag consecutive levels pass without improvement, instead of
    terminating the chain reheats:

        T ← T_reheat × T_best_level   (default T_reheat = 2.0)

    and restarts from the incumbent best.  At most MAX_REHEATS reheats are
    allowed before the chain exits.  This trades a controlled exploration
    burst for early termination.

    Alternate-solution tracking
    ───────────────────────────
    Unchanged from the original: every sequence within _alt_tol of fb is
    recorded; the set resets on a strict improvement.

    Returns
    -------
    (best_seq, best_obj, stats_dict)
    stats_dict keys: obj, time, t_best, history, init_obj, n_alt_seqs,
                     alpha_history (α value at each outer iteration)
    """
    # ── Reactive-adaptation constants ────────────────────────────────
    CHI_TARGET  = 0.20        # target acceptance rate
    ALPHA_STEP  = 0.005       # max per-level nudge to α
    ALPHA_LO    = 0.80        # hard floor on α
    ALPHA_HI    = 0.999       # hard ceiling on α
    MAX_REHEATS = 2           # maximum number of reheats before exit
    T_REHEAT    = 2.0         # reheat multiplier applied to T at stagnation

    rng   = random.Random(seed)
    n     = inst.n
    T     = T0 or calibrate_T0(seq0, inst, seed=seed, chi0=p.chi0)
    alpha = p.alpha           # mutable local copy — adapted each level

    pi    = seq0[:]
    f     = evaluate(pi, inst)
    pb, fb = pi[:], f
    init_obj = f

    _alt_tol = lambda fb_: max(0.5, abs(fb_) * 5e-5)

    alt_set      = {tuple(pi)}
    stag         = 0
    n_reheats    = 0
    history      = []
    alpha_history = []
    t_best       = 0.0
    t0           = time.perf_counter()

    for outer in range(p.I_max):
        improved   = False
        n_accepted = 0
        n_tried    = 0

        for _ in range(p.N_iter):
            pi2 = neighbour(pi, rng, n)
            if not is_feasible(pi2, inst):
                continue
            f2  = evaluate(pi2, inst)
            n_tried += 1
            dlt = f2 - f
            if dlt <= 0 or rng.random() < math.exp(-dlt / max(T, 1e-15)):
                pi, f = pi2, f2
                n_accepted += 1
            if f < fb - 1e-9:
                pb, fb, improved = pi[:], f, True
                t_best   = time.perf_counter() - t0
                alt_set  = {tuple(pi)}
            elif f <= fb + _alt_tol(fb) and len(alt_set) < _MAX_ALT_SEQS:
                alt_set.add(tuple(pi))

        # ── Reactive α update ─────────────────────────────────────────
        chi = n_accepted / max(n_tried, 1)
        if chi > CHI_TARGET:
            alpha = max(ALPHA_LO, alpha - ALPHA_STEP)   # too hot  → cool faster
        else:
            alpha = min(ALPHA_HI, alpha + ALPHA_STEP)   # too cold → cool slower

        T    *= alpha
        stag  = 0 if improved else stag + 1
        history.append(fb)
        alpha_history.append(alpha)

        if T < p.T_min:
            break

        # ── Stagnation → reheat or exit ───────────────────────────────
        if stag >= p.M_stag:
            if n_reheats >= MAX_REHEATS:
                break
            T          = T_REHEAT * T        # boost temperature
            pi         = pb[:]               # restart from incumbent
            f          = fb
            stag       = 0
            n_reheats += 1

    return pb, fb, {
        'obj':          fb,
        'time':         time.perf_counter() - t0,
        't_best':       t_best,
        'history':      history,
        'init_obj':     init_obj,
        'n_alt_seqs':   len(alt_set),
        'alpha_history': alpha_history,
    }

# ═══════════════════════════════════════════════════════════════════════════
# 9.  ILS WRAPPER
# ═══════════════════════════════════════════════════════════════════════════

def run_ils(seq0: List[int], inst: ALPInstance, p: SAParams,
            n_restarts: int = 3, seed: int = 0) -> Tuple[List[int], float, dict]:
    """
    Iterated Local Search: SA intensification + double-bridge perturbation.

    Alternate-solution counts are aggregated across all SA runs within the
    ILS chain.  init_obj reflects the objective of seq0 before any search.
    """
    rng = random.Random(seed)
    t0  = time.perf_counter()

    pi_best, f_best, st = run_sa(seq0, inst, p, seed=seed)
    all_hist    = [st['history']]
    t_best_wall = st['t_best']
    init_obj    = st['init_obj']
    total_alt   = st['n_alt_seqs']
    alpha_hist  = st.get('alpha_history', [])

    for r in range(n_restarts):
        pi_pert = _double_bridge(pi_best, rng)
        if not is_feasible(pi_pert, inst):
            pi_pert = _double_bridge(pi_best, rng)
        pi_r, f_r, st_r = run_sa(pi_pert, inst, p, seed=seed + r + 1)
        all_hist.append(st_r['history'])
        total_alt += st_r['n_alt_seqs']
        if f_r < f_best:
            pi_best, f_best = pi_r, f_r
            t_best_wall = (time.perf_counter() - t0
                           - st_r['time'] + st_r['t_best'])
            total_alt = st_r['n_alt_seqs']   # reset: new global best
            alpha_hist = st_r.get('alpha_history', [])  # reset: new global best

    return pi_best, f_best, {
        'obj':        f_best,
        'time':       time.perf_counter() - t0,
        't_best':     t_best_wall,
        'history':    all_hist[-1],
        'all_hist':   all_hist,
        'init_obj':   init_obj,
        'n_alt_seqs': total_alt,
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
    label, seq0, inst, p, seed, n_ils = args
    if n_ils > 0:
        pb, fb, st = run_ils(seq0, inst, p, n_restarts=n_ils, seed=seed)
    else:
        pb, fb, st = run_sa(seq0, inst, p, seed=seed)
    return (label, pb, fb,
            st.get('history',    []),
            st.get('t_best',     0.0),
            st.get('n_alt_seqs', 1),
            st.get('init_obj',   fb),
            st.get('alpha_history', []))


def ms_sa(inst: ALPInstance, p: SAParams = None,
          n_workers: int = N_CPU,
          n_ils: int = 0) -> Tuple[List[int], float, dict]:
    """
    Parallel multi-start SA with per-chain ILS and alternate-solution reporting.

    Pre-SA heuristic quality
    ─────────────────────────
    Each chain's init_obj (objective of its starting sequence before any SA)
    is collected and compared against the global best.  This reveals which
    dispatching rules already achieve the optimal without any search.

    Alternate-solution count
    ─────────────────────────
    n_alt_seqs is summed across all chains that independently reached the
    global best objective.  This estimates the density of the optimal (or
    near-optimal) region in sequence space.
    """
    p      = p or SAParams()
    starts = _build_starts(inst, n_starts=n_workers)
    tasks  = [(lbl, seq, inst, p, sd, n_ils) for lbl, seq, sd in starts]
    t0     = time.perf_counter()

    print(f"  MS-SA: {len(tasks)} chains  |  {n_workers} workers  "
          f"|  ILS/chain: {n_ils}")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=_CTX) as ex:
        results = list(tqdm(ex.map(_sa_worker, tasks),
                            total=len(tasks), desc="  SA chains",
                            disable=not _TQDM))

    # results: list of (label, pb, fb, hist, t_best, n_alt, init_obj, alpha_history)
    best_lbl, best_pi, best_f, best_hist, best_ttb, _, _, best_alpha_hist = min(results, key=lambda r: r[2])
    wall = time.perf_counter() - t0

    # ── Heuristic pre-SA quality report ──────────────────────────────
    _tol = max(0.5, abs(best_f) * 5e-5)
    n_init_optimal = sum(1 for r in results if abs(r[6] - best_f) <= _tol)
    total_alt = sum(r[5] for r in results if abs(r[2] - best_f) <= _tol)

    #print(f"\n  ── Heuristic quality (before SA) ──")
    #for lbl, _, f_final, _, _, n_alt, f_init in results:
    #    mark = " ✓ already optimal" if abs(f_init - best_f) <= _tol else ""
    #    impr = ((f_init - best_f) / max(best_f, 1e-9) * 100) if not math.isinf(f_init) else float('inf')
    #    print(f"    {lbl:<10} init={f_init:>10.2f} (+{impr:.1f}%)  "
    #          f"final={f_final:>10.2f}  alt_seqs={n_alt}{mark}")

    print(f"\n  {n_init_optimal}/{len(results)} initial solutions already at best objective")
    print(f"  Total distinct near-optimal sequences found: {total_alt}")
    print(f"  MS-SA → {best_f:.2f}  (best start: {best_lbl})"
          f"  [wall: {wall:.1f}s  |  TTB: {best_ttb:.1f}s]")

    return best_pi, best_f, {
        'wall_s':           wall,
        't_best':           best_ttb,
        'history':          best_hist,
        'all_histories':    results,
        'n_init_optimal':   n_init_optimal,
        'total_alt_seqs':   total_alt,
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
    """
    Zhang et al. (2020) style Gantt chart with wake-vortex separation zones.

    Visual elements
    ───────────────
    ─*────────*─  Time window  [r_j, d_j]
    □             Target landing time δ_j
    ○             Scheduled landing time x_j  (SLT, solved by Stage-2 LP)
    ▓  (orange)   Wake-vortex separation zone  [x_j, x_j + s_{j,k}]  on
                  successor aircraft k's row. The right edge is the earliest
                  feasible landing time for k given j's actual SLT.
                  A *binding* separation means k's circle sits at that edge;
                  *slack* means it lands to the right, within its window.
    |  (orange)   Separation boundary  x_j + s_{j,k}  (vertical tick)
    →  (arc)      Constraint arc j → k  (rendered only for n ≤ 20 to avoid
                  clutter; shows which predecessor imposes the blocked zone)
    """
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

    # ── Wake-vortex separation zones ──────────────────────────────────
    # For each consecutive pair j → k in the landing sequence, shade the
    # interval [x_j, x_j + s_{j,k}] on aircraft k's row.
    # Note: the y-axis is indexed by aircraft number (j+1), not sequence
    # position, so predecessor and successor may appear on non-adjacent rows.
    for l in range(n - 1):
        j, k = seq[l], seq[l + 1]
        y_j  = j + 1          # plot row of predecessor
        y_k  = k + 1          # plot row of successor
        sep  = inst.s[j][k]

        # Shaded blocked zone on successor's row
        ax.barh(y_k, width=sep, left=x1[j], height=0.52,
                color='#e8721c', alpha=0.45, zorder=0, linewidth=0)

    # ── Time windows, target times, scheduled landing times ───────────
    for j in range(n):
        y = j + 1
        ax.plot([inst.r[j], inst.d[j]], [y, y],
                color='#333333', linewidth=0.9, zorder=1)
        ax.plot([inst.r[j], inst.d[j]], [y, y],
                marker='*', markersize=6, color='#333333',
                linestyle='none', zorder=2)
        ax.plot(inst.delta[j], y,
                marker='s', markersize=5, color='#333333',
                markerfacecolor='none', markeredgewidth=1.0,
                linestyle='none', zorder=3)
        ax.plot(x1[j], y,
                marker='o', markersize=5, color='#1a6faf',
                markerfacecolor='none', markeredgewidth=1.2,
                linestyle='none', zorder=4)

    ax.set_xlabel('t / s', fontsize=11)
    ax.set_ylabel('Aircraft index', fontsize=11)
    ax.set_ylim(0, n + 1)
    tick_step = max(1, 5 * round(n / 50))
    ax.set_yticks(range(tick_step, n + 1, tick_step))
    ax.grid(True, linestyle=':', linewidth=0.5, color='#cccccc', alpha=0.8)
    ax.set_axisbelow(True)

    slt_lbl = f'SLT — {method}' + (f'  (obj={obj:.2f})' if obj is not None else '')
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color='#333333', lw=0.9, marker='*',
                       markersize=6, label='Time window  [r, d]'),
            plt.Line2D([0], [0], marker='s', color='#333333', lw=0,
                       markersize=5, markerfacecolor='none', label='Target δ'),
            plt.Line2D([0], [0], marker='o', color='#1a6faf', lw=0,
                       markersize=5, markerfacecolor='none', label=slt_lbl),
            mpatches.Patch(facecolor='#e8721c', alpha=0.45,
                           label='Wake-vortex separation zone  '
                                 r'$[x_j,\; x_j + s_{jk}]$'),
        ],
        loc='lower right', fontsize=9, framealpha=0.85, edgecolor='#aaaaaa',
    )
    ax.set_title(inst.name, fontsize=11, pad=8)
    plt.tight_layout()

    fname = (f"{save_dir}/gantt/gantt_{inst.name}"
             f"_{method.replace('-', '_').replace(' ', '_')}.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


def plot_sa_convergence(all_histories: list, inst_name: str,
                        save_dir: str = "plots") -> None:
    from pathlib import Path
    if not all_histories: return
    Path(save_dir).mkdir(exist_ok=True)
    finite  = [r for r in all_histories if not math.isinf(r[2])]
    if not finite: return
    best_f  = min(r[2] for r in finite)
    palette = plt.cm.tab20.colors
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (lbl, _, f_chain, hist, *_) in enumerate(all_histories):
        if not hist or math.isinf(f_chain): continue
        is_best = abs(f_chain - best_f) < 1e-4
        ax.plot(hist,
                linewidth=0.7 if is_best else 0.5,
                alpha=0.7     if is_best else 0.5,
                linestyle='-' if is_best else '--',
                color=palette[i % len(palette)],
                label=f'{lbl} ({f_chain:.0f})')
    ax.set_xlabel('Outer Iteration', fontsize=10)
    ax.set_ylabel('Best Objective (log scale)', fontsize=10)
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
    """
    Bar chart of distinct near-optimal sequences found per instance.
    High counts indicate a flat/degenerate objective landscape;
    low counts indicate a sharp optimum.
    """
    from pathlib import Path
    if not all_results: return
    Path(save_dir).mkdir(exist_ok=True)
    names = [r['inst']                     for r in all_results]
    alts  = [r.get('total_alt_seqs', 0)    for r in all_results]
    inits = [r.get('n_init_optimal',  0)   for r in all_results]
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
    """
    Plot the per-chain reactive cooling-rate (α) over outer iterations.

    For each SA chain, α starts at p.alpha and is nudged ±0.005 each level
    based on the observed acceptance rate vs the 20 % target.  This plot
    shows whether the adaptation mechanism is active and how different seeds
    drive different thermal histories.

    Data source
    ───────────
    all_histories : list of 8-tuples returned by ms_sa
        index 0  label (str)
        index 2  final objective (float)
        index 7  alpha_history (List[float])  ← added by reactive run_sa
    """
    from pathlib import Path
    Path(save_dir).mkdir(exist_ok=True)

    finite  = [r for r in all_histories if not math.isinf(r[2])]
    if not finite: return
    best_f  = min(r[2] for r in finite)
    palette = plt.cm.tab20.colors

    fig, ax = plt.subplots(figsize=(11, 4))

    # Reference lines
    ax.axhline(0.999, color='#cccccc', linewidth=0.6, linestyle='--', zorder=0)
    ax.axhline(0.80,  color='#cccccc', linewidth=0.6, linestyle='--', zorder=0)
    ax.axhline(0.20,  color='#e8721c', linewidth=0.5, linestyle=':',  zorder=0,
               label='χ* = 0.20  (acceptance target)')

    plotted = 0
    for i, r in enumerate(all_histories):
        lbl, _, fb, _, _, _, _, alpha_hist = r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]
        if not alpha_hist or math.isinf(fb):
            continue
        is_best = abs(fb - best_f) < 1e-4
        ax.plot(range(1, len(alpha_hist) + 1), alpha_hist,
                color    = palette[i % len(palette)],
                linewidth= 1.0 if is_best else 0.4,
                alpha    = 0.85 if is_best else 0.35,
                linestyle= '-'  if is_best else '--',
                label    = f'{lbl}' + (' ★' if is_best else ''))
        plotted += 1

    if plotted == 0:
        plt.close(); return

    ax.set_xlabel('Outer iteration (temperature level)', fontsize=10)
    ax.set_ylabel('Cooling rate α', fontsize=10)
    ax.set_title(f'Reactive α Trajectory — {inst_name}', fontsize=11)
    ax.set_ylim(0.78, 1.002)
    ax.legend(fontsize=7, ncol=4, loc='lower right', framealpha=0.50)
    ax.grid(alpha=0.2)
    plt.tight_layout()

    fname = f"{save_dir}/alpha trajectory/alpha_trajectory_{inst_name}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


def plot_seed_improvement(all_histories: list, inst_name: str,
                          known_opt: float = None,
                          save_dir: str = "plots") -> None:
    """
    Paired bar chart: initial heuristic objective vs. SA final objective per chain.

    For each chain the improvement Δ = init_obj − final_obj quantifies how
    much search effort was required beyond the dispatching rule alone.  A
    chain with Δ ≈ 0 means its heuristic seed was already near-optimal; a
    large Δ shows SA provided genuine improvement from that starting point.

    Data source
    ───────────
    all_histories : list of 8-tuples returned by ms_sa
        index 0  label (str)
        index 2  final objective (float)
        index 6  init_obj — objective before any SA (float)
    """
    from pathlib import Path
    Path(save_dir).mkdir(exist_ok=True)

    rows = [(r[0], r[6], r[2]) for r in all_histories
            if not math.isinf(r[2]) and not math.isinf(r[6])]
    if not rows: return

    # Sort by init_obj descending so the worst seed is on the left
    rows.sort(key=lambda x: -x[1])
    labels   = [r[0]  for r in rows]
    init_obj = [r[1]  for r in rows]
    final_obj= [r[2]  for r in rows]
    best_f   = min(final_obj)

    x   = np.arange(len(labels))
    w   = 0.38
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 0.9), 5))

    # Initial (open / hatched) bars
    ax.bar(x - w / 2, init_obj,  w,
           color='#aec6e8', edgecolor='#1a6faf', linewidth=0.8,
           hatch='///', alpha=0.7, label='Heuristic seed (pre-SA)')

    # Final (solid) bars
    colors = ['#c0392b' if abs(f - best_f) < 1e-4 else '#1a6faf'
              for f in final_obj]
    bars = ax.bar(x + w / 2, final_obj, w,
                  color=colors, alpha=0.85, label='SA final objective')

    # Improvement arrows on chains that actually improved
    for xi, (f_i, f_f) in enumerate(zip(init_obj, final_obj)):
        delta = f_i - f_f
        if delta > max(best_f * 5e-4, 0.5):
            ax.annotate('',
                xy    =(xi + w / 2, f_f  + (f_i - f_f) * 0.05),
                xytext=(xi - w / 2, f_i  - (f_i - f_f) * 0.05),
                arrowprops=dict(arrowstyle='->', color='#555555',
                                lw=0.8, connectionstyle='arc3,rad=0.15'))

    # Known optimum reference line
    if known_opt is not None:
        ax.axhline(known_opt, color='black', linewidth=0.9,
                   linestyle='--', label=f'Known optimum ({known_opt})')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Objective value', fontsize=10)
    ax.set_title(f'Seed Quality vs. SA Improvement — {inst_name}', fontsize=11)
    ax.legend(fontsize=9, loc='upper right', framealpha=0.85)
    ax.grid(axis='y', alpha=0.22)
    plt.tight_layout()

    fname = f"{save_dir}/seed improvement/seed_improvement_{inst_name}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


def plot_penalty_profile(seq: List[int], inst: ALPInstance,
                         landing_times: np.ndarray,
                         method: str = "", obj: float = None,
                         save_dir: str = "plots") -> None:
    """
    Stacked bar chart of per-aircraft weighted earliness and tardiness cost,
    indexed by landing position in the final sequence.

    Each bar is split into:
      g_j · E_j  (blue)  — cost for landing before target δ_j
      h_j · T_j  (red)   — cost for landing after  target δ_j

    The sum of all bars equals the total schedule objective.  Aircraft with
    zero penalty (landed exactly at δ_j) appear as empty slots, making the
    distribution of cost concentration immediately visible.

    A secondary scatter (grey diamonds) shows how far each scheduled landing
    time x_j sits from its target δ_j (signed deviation, right y-axis), so
    the cost magnitude can be interpreted alongside the timing deviation.

    Parameters
    ----------
    seq           : final landing sequence
    inst          : ALPInstance
    landing_times : array of length n, x_j values from Stage-2 LP
    method        : label string for title / filename
    obj           : total objective (for subtitle annotation)
    """
    from pathlib import Path
    Path(save_dir).mkdir(exist_ok=True)

    n   = inst.n
    x   = landing_times
    pos = np.arange(1, n + 1)   # 1-indexed landing positions

    early_pen = np.array([inst.g[j] * max(inst.delta[j] - x[j], 0.0) for j in seq])
    late_pen  = np.array([inst.h[j] * max(x[j] - inst.delta[j], 0.0) for j in seq])
    deviation = np.array([x[j] - inst.delta[j] for j in seq])   # signed, seconds

    fig, ax1 = plt.subplots(figsize=(max(10, n * 0.28), 5))
    ax2 = ax1.twinx()

    # Stacked penalty bars
    ax1.bar(pos, early_pen, color='#1a6faf', alpha=0.82,
            label='Weighted earliness  $g_j E_j$')
    ax1.bar(pos, late_pen,  bottom=early_pen, color='#c0392b', alpha=0.82,
            label='Weighted tardiness  $h_j T_j$')

    # Signed deviation scatter
    ax2.scatter(pos, deviation, marker='D', s=18, color='#555555',
                alpha=0.6, zorder=3, label='$x_j - \\delta_j$ (s)')
    ax2.axhline(0, color='#888888', linewidth=0.6, linestyle=':')
    ax2.set_ylabel('Deviation from target δ_j  (s)', fontsize=9,
                   color='#555555')
    ax2.tick_params(axis='y', labelcolor='#555555')

    ax1.set_xlabel('Landing position in sequence', fontsize=10)
    ax1.set_ylabel('Penalty cost', fontsize=10)
    title = f'Per-Aircraft Penalty Profile — {inst.name}'
    if method: title += f'  [{method}]'
    if obj is not None: title += f'  (total = {obj:.2f})'
    ax1.set_title(title, fontsize=11)

    # Combined legend from both axes
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=9, loc='upper right', framealpha=0.85)

    ax1.set_xlim(0, n + 1)
    ax1.grid(axis='y', alpha=0.2)
    ax1.set_axisbelow(True)

    # Annotate aircraft index for the top-N most costly positions
    total_pen = early_pen + late_pen
    top_n     = min(5, int((total_pen > 0).sum()))
    if top_n > 0:
        threshold = np.sort(total_pen)[-top_n]
        for l, (p_val, j) in enumerate(zip(total_pen, seq)):
            if p_val >= threshold and p_val > 0:
                ax1.text(pos[l], p_val + max(total_pen) * 0.01,
                         f'A{j}', ha='center', va='bottom',
                         fontsize=7, color='#333333')

    plt.tight_layout()
    tag = method.replace('-', '_').replace(' ', '_')
    fname = f"{save_dir}/penalty profile/penalty_profile_{inst.name}_{tag}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


# ═══════════════════════════════════════════════════════════════════════════
# 14.  SCHEDULE VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════
#
# Three functions:
#
#   verify_schedule(seq, inst)
#       → Full constraint audit. Returns (passed: bool, obj: float,
#         report: VerificationReport).  Always runs to completion even when
#         violations are found so every failure is reported, not just the
#         first one.
#
#   verify_all(seq, inst, *, verbose, raise_on_fail)
#       → Convenience wrapper.  Prints a formatted report and optionally
#         raises AssertionError on failure.  Use this in the experiment
#         runner and __main__.
#
#   _lp_recompute(seq, inst)
#       → Re-solve Stage-2 LP from scratch and return (obj, landing_times).
#         Called inside verify_schedule as an independent cross-check.
#
# Constraint groups checked
# ─────────────────────────
#   C1  Release dates      x_j ≥ r_j           (n constraints)
#   C2  Deadlines          x_j ≤ d_j           (n constraints)
#   C3  Separation         x_k ≥ x_j + s_jk    (n-1 constraints, chain)
#   C4  Permutation        seq is a valid permutation of {0..n-1}
#   C5  Feasibility        greedy forward pass matches LP landing times
#   C6  Earliness defn     E_j = max(δ_j - x_j, 0), E_j ≥ 0
#   C7  Tardiness defn     T_j = max(x_j - δ_j, 0), T_j ≥ 0
#   C8  Objective match    LP obj ≈ Σ(g_j E_j + h_j T_j)  (cross-check)
#   C9  LP re-solve        independent LP solve matches the reported obj


from dataclasses import dataclass, field


@dataclass
class ConstraintViolation:
    group:     str           # C1–C9 label
    index:     int           # aircraft index (or position) involved
    lhs:       float         # left-hand side value
    rhs:       float         # right-hand side bound
    violation: float         # amount by which lhs violates rhs (> 0 is bad)
    detail:    str           # human-readable description


@dataclass
class VerificationReport:
    instance:       str
    sequence:       List[int]
    n_aircraft:     int
    obj_reported:   float              # objective returned by the solver
    obj_recomputed: float              # Σ(g E + h T) from landing times
    obj_lp_recheck: float              # independent LP re-solve
    landing_times:  Optional[np.ndarray]
    violations:     List[ConstraintViolation] = field(default_factory=list)
    passed:         bool               = True
    tol:            float              = 1e-4

    # ── convenience properties ───────────────────────────────────────
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
                    f"{v.rhs:>14.4f} {v.violation:>12.4e}  {v.detail}"
                )
        lines.append(f"{'═'*68}")
        return "\n".join(lines)


def _lp_recompute(seq: List[int],
                  inst: ALPInstance) -> Tuple[float, Optional[np.ndarray]]:
    """
    Independent Stage-2 LP re-solve using a freshly built matrix.
    Intentionally duplicates no state from the main solver path so it
    acts as a genuine cross-check.
    """
    n   = inst.n
    c   = np.concatenate([np.zeros(n), inst.g, inst.h])
    bnd = [(inst.r[j], inst.d[j]) for j in range(n)] + [(0.0, None)] * 2 * n
    A, b = [], []
    for l in range(n - 1):
        row = np.zeros(3 * n)
        row[seq[l]] = 1.0; row[seq[l+1]] = -1.0
        A.append(row); b.append(-inst.s[seq[l]][seq[l+1]])
    for j in range(n):
        row = np.zeros(3 * n); row[j] = -1.0; row[n + j] = -1.0
        A.append(row); b.append(-inst.delta[j])
    for j in range(n):
        row = np.zeros(3 * n); row[j] = 1.0; row[2*n + j] = -1.0
        A.append(row); b.append(inst.delta[j])
    res = linprog(c, A_ub=np.array(A), b_ub=np.array(b), bounds=bnd,
                  method='highs',
                  options={'disp': False, 'presolve': True,
                           'dual_feasibility_tolerance': 1e-9,
                           'primal_feasibility_tolerance': 1e-9})
    if res.status != 0:
        return float('inf'), None
    return float(res.fun), res.x[:n]


def verify_schedule(seq: List[int],
                    inst: ALPInstance,
                    tol: float = 1e-4) -> Tuple[bool, float, 'VerificationReport']:
    """
    Full LP-constraint audit of a landing schedule.

    Parameters
    ----------
    seq  : landing sequence (permutation of aircraft indices)
    inst : ALPInstance
    tol  : absolute tolerance for all constraint comparisons

    Returns
    -------
    (passed, objective, report)

    The function never short-circuits: all nine constraint groups are
    checked regardless of earlier failures so the caller receives a
    complete picture of every violation in a single call.
    """
    n    = inst.n
    viols: List[ConstraintViolation] = []

    def _add(group, idx, lhs, rhs, viol, detail):
        viols.append(ConstraintViolation(group, idx, lhs, rhs, viol, detail))

    # ── C4: Permutation validity ──────────────────────────────────────
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

    # Abort remaining checks if the sequence is structurally invalid
    if any(v.group == 'C4' for v in viols):
        rep = VerificationReport(
            instance=inst.name, sequence=seq, n_aircraft=n,
            obj_reported=float('inf'), obj_recomputed=float('inf'),
            obj_lp_recheck=float('inf'), landing_times=None,
            violations=viols, passed=False, tol=tol)
        return False, float('inf'), rep

    # ── Solve LP to obtain landing times ─────────────────────────────
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

    # ── C1: Release dates  x_j ≥ r_j ─────────────────────────────────
    for j in range(n):
        viol = inst.r[j] - x[j]          # positive = x_j < r_j (violation)
        if viol > tol:
            _add('C1', j, x[j], inst.r[j], viol,
                 f"x[{j}]={x[j]:.4f} < r[{j}]={inst.r[j]:.4f}")

    # ── C2: Deadlines  x_j ≤ d_j ─────────────────────────────────────
    for j in range(n):
        viol = x[j] - inst.d[j]          # positive = x_j > d_j (violation)
        if viol > tol:
            _add('C2', j, x[j], inst.d[j], viol,
                 f"x[{j}]={x[j]:.4f} > d[{j}]={inst.d[j]:.4f}")

    # ── C3: Wake-vortex separations  x_k ≥ x_j + s_jk ───────────────
    for l in range(n - 1):
        j, k = seq[l], seq[l+1]
        required = x[j] + inst.s[j][k]
        viol     = required - x[k]       # positive = gap too small
        if viol > tol:
            _add('C3', k, x[k], required, viol,
                 f"A{k} at {x[k]:.4f} < A{j} ({x[j]:.4f}) + sep {inst.s[j][k]:.1f}")

    # ── C5: Greedy-feasibility consistency ───────────────────────────
    # Re-run the O(n) greedy forward pass using the LP-returned times.
    # Any discrepancy indicates the LP times violate the sequence chain.
    t_greedy = inst.r[seq[0]]
    for l in range(1, n):
        t_greedy = max(inst.r[seq[l]], t_greedy + inst.s[seq[l-1]][seq[l]])
        if t_greedy > inst.d[seq[l]] + tol:
            _add('C5', seq[l], t_greedy, inst.d[seq[l]],
                 t_greedy - inst.d[seq[l]],
                 f"Greedy pass: A{seq[l]} earliest={t_greedy:.4f} "
                 f"> d={inst.d[seq[l]]:.4f}")

    # ── C6: Earliness non-negativity and definition ───────────────────
    for j in range(n):
        E_j = max(inst.delta[j] - x[j], 0.0)
        if E_j < -tol:
            _add('C6', j, E_j, 0.0, -E_j,
                 f"E[{j}]={E_j:.4f} < 0 (definition violated)")

    # ── C7: Tardiness non-negativity and definition ───────────────────
    for j in range(n):
        T_j = max(x[j] - inst.delta[j], 0.0)
        if T_j < -tol:
            _add('C7', j, T_j, 0.0, -T_j,
                 f"T[{j}]={T_j:.4f} < 0 (definition violated)")

    # ── C8: Objective cross-check  LP obj ≈ Σ(g E + h T) ────────────
    obj_recomputed = float(sum(
        inst.g[j] * max(inst.delta[j] - x[j], 0.0) +
        inst.h[j] * max(x[j] - inst.delta[j], 0.0)
        for j in range(n)
    ))
    obj_delta_c8 = abs(obj_recomputed - obj_lp)
    # Adaptive tolerance: 0.01% of objective or absolute 0.5, whichever larger
    c8_tol = max(0.5, abs(obj_lp) * 1e-4)
    if obj_delta_c8 > c8_tol:
        _add('C8', -1, obj_lp, obj_recomputed, obj_delta_c8,
             f"LP obj={obj_lp:.6f} vs Σ(gE+hT)={obj_recomputed:.6f}")

    # ── C9: Independent LP re-solve ───────────────────────────────────
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
        # Also verify re-solved landing times satisfy all bounds
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
        instance       = inst.name,
        sequence       = seq[:],
        n_aircraft     = n,
        obj_reported   = obj_lp,
        obj_recomputed = obj_recomputed,
        obj_lp_recheck = obj_recheck,
        landing_times  = x,
        violations     = viols,
        passed         = passed,
        tol            = tol,
    )
    return passed, round(obj_recomputed, 6), report


def verify_all(seq: List[int], inst: ALPInstance,
               tol: float = 1e-4,
               verbose: bool = True,
               raise_on_fail: bool = False) -> Tuple[bool, float]:
    """
    Convenience wrapper around verify_schedule.

    Parameters
    ----------
    verbose       : print the full formatted report
    raise_on_fail : raise AssertionError if any constraint is violated

    Returns
    -------
    (passed, objective)
    """
    passed, obj, report = verify_schedule(seq, inst, tol=tol)
    if verbose:
        print(report.summary())
    if not passed and raise_on_fail:
        raise AssertionError(
            f"Schedule verification failed for {inst.name}: "
            f"{report.n_violations} violation(s) in groups "
            f"{report.groups_failed}.  Max violation = {report.max_violation:.4e}"
        )
    return passed, obj



# ═══════════════════════════════════════════════════════════════════════════
# 14b.  RESULTS EXPORT
# ═══════════════════════════════════════════════════════════════════════════

import csv, json
from pathlib import Path
from datetime import datetime


def export_results(results_list: list,
                   out_dir: str = "results") -> None:
    """
    Persist all per-instance results to disk.

    Parameters
    ----------
    results_list : list of dicts returned by run_experiment()
    out_dir      : directory to write into (created if absent)

    Files written
    ─────────────
    summary.csv
        One row per instance.  Columns:
        instance, n, known_opt, ms_sa_obj, gap_pct, wall_s, ttb_s,
        total_alt_seqs, n_init_optimal, verified

    schedules.csv
        One row per aircraft per instance.  Columns:
        instance, landing_pos, aircraft_idx,
        r (release), delta (target), d (deadline),
        x (scheduled landing), earliness, tardiness, penalty,
        g (early cost), h (late cost)

    verification.txt
        Full VerificationReport.summary() for every instance, separated
        by a divider.  Includes constraint-by-constraint violation detail
        when failures exist.

    run_metadata.json
        Run timestamp, N_CPU, hostname, Python version, total instances,
        and per-instance pass/fail status.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── 1. summary.csv ────────────────────────────────────────────────
    summary_path = out / "summary.csv"
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

    # ── 2. schedules.csv ──────────────────────────────────────────────
    sched_path = out / "schedules.csv"
    sched_fields = [
        "instance", "landing_pos", "aircraft_idx",
        "r", "delta", "d",
        "x_scheduled",
        "earliness", "tardiness", "penalty",
        "g", "h",
    ]
    with open(sched_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=sched_fields)
        w.writeheader()
        for r in results_list:
            inst  = r.get("_inst")
            seq   = r.get("pi_sa")
            vr    = r.get("_vreport")
            if inst is None or seq is None:
                continue
            # Use landing times from the verification report if available;
            # otherwise re-solve to get them.
            if vr is not None and vr.landing_times is not None:
                x = vr.landing_times
            else:
                _, x = solve_stage2(seq, inst)
            if x is None:
                continue
            for pos, j in enumerate(seq):
                E_j = max(inst.delta[j] - x[j], 0.0)
                T_j = max(x[j] - inst.delta[j], 0.0)
                pen = inst.g[j] * E_j + inst.h[j] * T_j
                w.writerow({
                    "instance":    inst.name,
                    "landing_pos": pos + 1,        # 1-indexed position in sequence
                    "aircraft_idx": j,
                    "r":           f"{inst.r[j]:.4f}",
                    "delta":       f"{inst.delta[j]:.4f}",
                    "d":           f"{inst.d[j]:.4f}",
                    "x_scheduled": f"{x[j]:.4f}",
                    "earliness":   f"{E_j:.4f}",
                    "tardiness":   f"{T_j:.4f}",
                    "penalty":     f"{pen:.4f}",
                    "g":           f"{inst.g[j]:.4f}",
                    "h":           f"{inst.h[j]:.4f}",
                })
    print(f"  Saved: {sched_path}")

    # ── 3. verification.txt ───────────────────────────────────────────
    verif_path = out / "verification.txt"
    divider = "\n" + "─" * 70 + "\n"
    with open(verif_path, "w", encoding="utf-8") as fh:
        fh.write(f"ALP Verification Report\n"
                 f"Generated : {ts}\n"
                 f"Instances : {len(results_list)}\n")
        for r in results_list:
            vr = r.get("_vreport")
            fh.write(divider)
            if vr is not None:
                fh.write(vr.summary() + "\n")
            else:
                fh.write(f"  {r['inst']}: no verification report available.\n")
        fh.write(divider)
    print(f"  Saved: {verif_path}")

    # ── 4. run_metadata.json ──────────────────────────────────────────
    import sys, socket
    meta_path = out / "run_metadata.json"
    meta = {
        "timestamp":   ts,
        "hostname":    socket.gethostname(),
        "python":      sys.version,
        "n_cpu":       N_CPU,
        "instances":   [
            {
                "name":      r["inst"],
                "n":         r["n"],
                "known_opt": r.get("opt"),
                "obj":       (r["results"].get("MS-SA", (float("inf"),))[0]
                              if not math.isinf(
                                  r["results"].get("MS-SA", (float("inf"),))[0])
                              else None),
                "verified":  (r["_vreport"].passed
                              if r.get("_vreport") else None),
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
                   n_workers: int = N_CPU) -> dict:

    sa_adapt, n_ils = adaptive_params(inst.n)
    sa_p = sa_p or sa_adapt

    print(f"\n{'═'*70}")
    print(f"  Instance : {inst.name}   n={inst.n}   s̄={inst.s_bar:.0f}s")
    print(f"  CPU cores: {N_CPU}   ILS/chain: {n_ils}")
    if known_opt: print(f"  Reference: {known_opt}")
    print(f"{'═'*70}")

    t0 = time.perf_counter()
    pi_sa, f_sa, sa_stats = ms_sa(inst, sa_p, n_workers=n_workers, n_ils=n_ils)
    wall   = time.perf_counter() - t0
    t_best = sa_stats.get('t_best', wall)

    print(f"\n{'─'*70}")
    print(f"  {'Method':<16} {'Objective':>12} {'Gap':>9} "
          f"{'Wall(s)':>9} {'TTB(s)':>8} {'Alt seqs':>9}")
    print(f"  {'─'*16} {'─'*12} {'─'*9} {'─'*9} {'─'*8} {'─'*9}")
    print(f"  {'MS-SA':<16} {f_sa:>12.2f} {_gap(f_sa, known_opt):>9} "
          f"{wall:>9.2f} {t_best:>8.2f} "
          f"{sa_stats.get('total_alt_seqs', 0):>9}")
    print(f"{'═'*70}\n")

    # Verification
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

    # Build and immediately export the single-instance result
    result = {
        'inst':           inst.name,
        'n':              inst.n,
        'results':        {'MS-SA': (f_sa, wall)},
        'opt':            known_opt,
        'ttb':            {'MS-SA': t_best},
        'total_alt_seqs': sa_stats.get('total_alt_seqs', 0),
        'n_init_optimal': sa_stats.get('n_init_optimal', 0),
        'pi_sa':          pi_sa,
        '_inst':          inst,       # kept for export; not printed in table
        '_vreport':       vreport,    # full VerificationReport object
    }
    export_results([result], out_dir=f"results/{inst.name}")   # per-instance incremental save

    # Plots
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
        'airland7.txt':  1550,     'airland8.txt':  1950,
        'airland9.txt':  5611.70,  'airland10.txt': 12640.42,
        'airland11.txt': 12462.18, 'airland12.txt': 16629.10,
        'airland13.txt': 39287.52,
    }

    SA_full = SAParams(alpha=0.99,  # cooling rate
                    N_iter=250,     # total iterations (including all ILS chains)
                    T_min=1e-4,     # minimum temperature
                    I_max=800,     # max iterations without improvement before termination
                    M_stag=100      # stagnation threshold for adaptive parameter tuning
                )

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

    # ── Pre-benchmark Optuna tuning (optional) ────────────────────────
    ENABLE_OPTUNA = False

    if ENABLE_OPTUNA and found:
        tune_path, tune_name, tune_opt = found[0]   # airland1: fastest
        print(f"\n  [Optuna] Tuning on {tune_name}  (opt={tune_opt})")
        try:
            tune_inst = load_orlib(str(tune_path), tune_name)
            SA_full   = tune_sa(tune_inst, tune_opt,
                                n_trials=40, n_workers=min(8, N_CPU))
            print("  Tuned parameters applied to all runs.\n")
        except Exception as exc:
            print(f"  Optuna failed ({exc}) — using defaults.\n")

    # ── Benchmark loop ────────────────────────────────────────────────
    if found:
        print(f"\nRunning {len(found)} instance(s)...\n")
        all_results = []
        for path, name, opt in found:
            try:
                inst = load_orlib(str(path), name)
                diagnose_instance(inst)
                res  = run_experiment(inst, known_opt=opt,
                                      sa_p=SA_full, n_workers=N_CPU)
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
            
            # ── Post-benchmark full audit ─────────────────────────────────
            print("\n  Running post-benchmark full constraint audit...")
            audit_pass = 0
            for r in all_results:
                inst_r = load_orlib(
                    str(next(p for p, n_, _ in found if n_ == r['inst'])),
                    r['inst']
                )
                ok, _ = verify_all(r['pi_sa'], inst_r,
                                    tol=1e-4, verbose=False, raise_on_fail=False)
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
                       n_workers=N_CPU)
