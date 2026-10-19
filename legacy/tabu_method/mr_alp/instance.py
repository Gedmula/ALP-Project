"""
instance.py — MR-ALP Solver: Instance Parser, Numba Kernels, Feasibility, Surrogate
====================================================================================
§4  Numba JIT kernels (_insert_times_kernel, _rwy_feasible_nb)
§5  OR Library instance file parser (load_instance)
§6  Surrogate landing times and surrogate penalty
§7  Full pairwise runway feasibility check (_runway_feasible)

Notes
-----
Surrogate times use the consecutive-predecessor approximation and are used
only as a fast guide for construction and SA move evaluation — never for
final objectives or feasibility certification.

All feasibility-certified computations (LP, verification, pairwise checks)
must use all O(n²) ordered-pair constraints because OR Library separation
matrices violate the triangle inequality.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from mr_alp.models import Instance

# ── Optional Numba import ─────────────────────────────────────────────────
try:
    import numba as nb
    _NUMBA = True
except ImportError:
    nb = None
    _NUMBA = False


# ═══════════════════════════════════════════════════════════════════════════
#   §4  NUMBA JIT KERNELS
# ═══════════════════════════════════════════════════════════════════════════

if _NUMBA:
    @nb.njit(cache=True)
    def _insert_times_kernel(j, p, seq, C_prev, r, s, d):
        """
        Surrogate landing times after inserting aircraft j at position p.

        Positions 0..p-1 are copied from C_prev; positions p..L are
        recomputed.  Returns (C_new, feasible).  Feasibility is checked
        for positions p..L only (earlier positions cannot be violated by
        inserting j).
        """
        L    = len(seq)
        L_n  = L + 1
        C_n  = np.empty(L_n, dtype=np.float64)
        for q in range(p):
            C_n[q] = C_prev[q]
        if p == 0:
            C_n[0] = r[j]
        else:
            v = C_n[p - 1] + s[seq[p - 1], j]
            C_n[p] = v if v > r[j] else r[j]
        for q in range(p + 1, L_n):
            cur  = seq[q - 1]
            prev = j if q == p + 1 else seq[q - 2]
            v    = C_n[q - 1] + s[prev, cur]
            C_n[q] = v if v > r[cur] else r[cur]
        for q in range(p, L_n):
            ac = j if q == p else seq[q - 1]
            if C_n[q] > d[ac] + 1e-9:
                return C_n, False
        return C_n, True

    @nb.njit(cache=True)
    def _rwy_feasible_nb(seq, r, s, d):
        """
        Full O(L²) pairwise feasibility check for one runway sequence.

        Computes earliest feasible times respecting all ordered-pair
        constraints, then checks that no aircraft exceeds its deadline d_j.
        """
        L = len(seq)
        if L == 0:
            return True
        C = np.empty(L, dtype=np.float64)
        C[0] = r[seq[0]]
        if C[0] > d[seq[0]] + 1e-9:
            return False
        for q in range(1, L):
            C[q] = r[seq[q]]
            for h in range(q):
                lb = C[h] + s[seq[h], seq[q]]
                if lb > C[q]:
                    C[q] = lb
            if C[q] > d[seq[q]] + 1e-9:
                return False
        return True

else:
    # Pure-Python fallbacks with identical semantics, so the package imports
    # and runs on machines without Numba (previously construction.py did an
    # unconditional `from mr_alp.instance import _insert_times_kernel`, which
    # raised ImportError when Numba was absent and broke the whole package).
    def _insert_times_kernel(j, p, seq, C_prev, r, s, d):
        """Surrogate landing times after inserting aircraft j at position p."""
        L   = len(seq)
        L_n = L + 1
        C_n = np.empty(L_n, dtype=np.float64)
        for q in range(p):
            C_n[q] = C_prev[q]
        if p == 0:
            C_n[0] = r[j]
        else:
            v = C_n[p - 1] + s[seq[p - 1], j]
            C_n[p] = v if v > r[j] else r[j]
        for q in range(p + 1, L_n):
            cur  = seq[q - 1]
            prev = j if q == p + 1 else seq[q - 2]
            v    = C_n[q - 1] + s[prev, cur]
            C_n[q] = v if v > r[cur] else r[cur]
        for q in range(p, L_n):
            ac = j if q == p else seq[q - 1]
            if C_n[q] > d[ac] + 1e-9:
                return C_n, False
        return C_n, True

    def _rwy_feasible_nb(seq, r, s, d):
        """Full O(L²) pairwise feasibility check for one runway sequence."""
        L = len(seq)
        if L == 0:
            return True
        C = np.empty(L, dtype=np.float64)
        C[0] = r[seq[0]]
        if C[0] > d[seq[0]] + 1e-9:
            return False
        for q in range(1, L):
            C[q] = r[seq[q]]
            for h in range(q):
                lb = C[h] + s[seq[h], seq[q]]
                if lb > C[q]:
                    C[q] = lb
            if C[q] > d[seq[q]] + 1e-9:
                return False
        return True


# ═══════════════════════════════════════════════════════════════════════════
#   §5  OR LIBRARY INSTANCE FILE PARSER
# ═══════════════════════════════════════════════════════════════════════════

def load_instance(filepath: str, name: Optional[str] = None) -> Instance:
    """
    Parse an OR Library ALP instance file.

    File format (Beasley et al. 2000)
    ----------------------------------
    Line 1 : n  freeze_time   (freeze_time discarded)
    Per aircraft (one entry per line):
        appearance_time  r  delta  d  g  h  s[i,0] … s[i,n-1]
    appearance_time is discarded.  Diagonal entries of s are set to 0.

    Parameters
    ----------
    filepath : path to the .txt instance file.
    name     : instance name override; defaults to the file stem.

    Returns
    -------
    Instance
        Fully initialised and validated Instance dataclass.

    Raises
    ------
    ValueError
        If window ordering constraints r ≤ δ ≤ d are violated, or the
        token count does not match the expected format.
    """
    path   = Path(filepath)
    name   = name or path.stem.lower()
    tokens = path.read_text().split()
    pos    = 0

    def take_int()   -> int:   nonlocal pos; v = int(tokens[pos]);   pos += 1; return v
    def take_float() -> float: nonlocal pos; v = float(tokens[pos]); pos += 1; return v

    n  = take_int(); _ = take_float()               # n, freeze_time
    r  = np.empty(n); delta = np.empty(n); d = np.empty(n)
    g  = np.empty(n); h     = np.empty(n); s = np.empty((n, n))

    for i in range(n):
        _ = take_float()                             # appearance_time (discarded)
        r[i] = take_float(); delta[i] = take_float(); d[i] = take_float()
        g[i] = take_float(); h[i]     = take_float()
        for j in range(n):
            s[i, j] = take_float()

    np.fill_diagonal(s, 0.0)

    bad = np.where(r > delta + 1e-6)[0]
    if bad.size:
        raise ValueError(f"{name}: r > delta for aircraft {bad[:5].tolist()}")
    bad = np.where(delta > d + 1e-6)[0]
    if bad.size:
        raise ValueError(f"{name}: delta > d for aircraft {bad[:5].tolist()}")
    if pos != len(tokens):
        raise ValueError(
            f"{name}: token count mismatch — expected {pos}, found {len(tokens)}")

    return Instance(name=name, n=n, r=r, delta=delta, d=d, g=g, h=h, s=s)


# ═══════════════════════════════════════════════════════════════════════════
#   §6  SURROGATE LANDING TIMES  (consecutive-predecessor approximation)
# ═══════════════════════════════════════════════════════════════════════════

def surrogate_times(seq: List[int], inst: Instance) -> List[float]:
    """
    Compute surrogate landing times for a single runway sequence.

    C[0] = r[seq[0]]
    C[q] = max(r[seq[q]], C[q-1] + s[seq[q-1], seq[q]])  for q ≥ 1

    This consecutive-predecessor approximation is used only for construction
    and SA move guidance — never for certified objectives.

    Complexity: O(L).
    """
    if not seq:
        return []
    C    = [0.0] * len(seq)
    C[0] = float(inst.r[seq[0]])
    for q in range(1, len(seq)):
        C[q] = max(float(inst.r[seq[q]]),
                   C[q - 1] + float(inst.s[seq[q - 1], seq[q]]))
    return C


def surrogate_penalty(seq: List[int], C_hat: List[float],
                      inst: Instance) -> float:
    """
    Weighted earliness / tardiness under surrogate times.

    Σ_j  g_j · max(δ_j − C_j, 0) + h_j · max(C_j − δ_j, 0)
    """
    if not seq:
        return 0.0
    sa = np.asarray(seq,   dtype=np.intp)
    Ca = np.asarray(C_hat)
    return float(
        (inst.g[sa] * np.maximum(inst.delta[sa] - Ca, 0.0)
         + inst.h[sa] * np.maximum(Ca - inst.delta[sa], 0.0)).sum()
    )


# ═══════════════════════════════════════════════════════════════════════════
#   §7  FULL PAIRWISE RUNWAY FEASIBILITY CHECK
# ═══════════════════════════════════════════════════════════════════════════

def runway_feasible(seq: List[int], inst: Instance) -> bool:
    """
    Return True iff seq is feasible under all ordered-pair separation
    constraints and deadline d_j.

    Uses the Numba JIT kernel when available; falls back to pure Python
    with identical semantics.

    Complexity: O(L²) — required because s violates the triangle inequality.
    """
    if not seq:
        return True
    if _NUMBA:
        return bool(_rwy_feasible_nb(
            np.asarray(seq, dtype=np.int32), inst.r, inst.s, inst.d))

    L = len(seq)
    C = np.empty(L)
    C[0] = inst.r[seq[0]]
    if C[0] > inst.d[seq[0]] + 1e-9:
        return False
    for q in range(1, L):
        C[q] = inst.r[seq[q]]
        for h in range(q):
            lb = C[h] + inst.s[seq[h], seq[q]]
            if lb > C[q]:
                C[q] = lb
        if C[q] > inst.d[seq[q]] + 1e-9:
            return False
    return True