"""
mutation.py — ruin-and-recreate along congestion
===============================================================================
Remove a contiguous stretch of `size` aircraft and re-insert them with TC-RBI's
randomised regret rule (const_heuristics.ramp_rbi), keeping the rest of the
schedule.  The stretch is drawn with probability proportional to the total
shadow price of the separations it spans, so mutations land where congestion
generates cost; uncongested stretches keep a small floor probability.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from const_heuristics.alp_core import INF, Problem, cet
from const_heuristics.ramp_rbi import TCRBIParams, _construct
from sa_single.kernels import shadow_prices

_TCRBI = TCRBIParams()


def ruin_recreate(P: Problem, seq: List[int], size: int,
                  rng: np.random.Generator) -> Optional[Tuple[List[int], float]]:
    """Mutated copy of `seq` and its exact cost, or None if re-insertion failed."""
    L = len(seq)
    size = max(2, min(size, L - 1))
    arr = np.asarray(seq, dtype=np.int64)
    x = np.empty(L)
    cet(arr, L, *P.A, x)
    lam = np.empty(max(L - 1, 1))
    shadow_prices(arr, L, x, P.r, P.d, P.dl, P.g, P.h, P.s, lam)
    # weight of a stretch [a, a+size) = total price of the separations inside it
    csum = np.concatenate([[0.0], np.cumsum(lam[:L - 1])])
    starts = np.arange(L - size + 1)
    w = csum[np.minimum(starts + size - 1, L - 1)] - csum[starts]
    w = w + 1e-3 * (w.max() if w.max() > 0 else 1.0)
    a = int(rng.choice(starts, p=w / w.sum()))
    removed = seq[a:a + size]
    rest = seq[:a] + seq[a + size:]
    c_rest = P.exact_cost(rest)
    if c_rest == INF:
        return None
    seqs = np.zeros((1, P.n), np.int64)
    seqs[0, :len(rest)] = rest
    res = _construct(P, 1, _TCRBI, rng, start=(seqs, np.array([len(rest)], np.int64), np.array([c_rest])),
                     pending=removed)
    if res is None:
        return None
    s_new, l_new, c_new = res
    return s_new[0, :l_new[0]].tolist(), float(c_new[0])
