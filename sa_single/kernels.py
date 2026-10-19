"""
kernels.py — Numba kernels of the single-runway SA
===============================================================================
shadow prices   λ_k = rate at which the optimal cost falls when the separation
                between positions k and k+1 shrinks (exact one-sided derivative of
                the consecutive timing LP; checked against finite differences)
moves           insert / swap / block(2–3), span 1..window, P(span t) ∝ 1/t
                guided:  around separation k drawn with P ∝ λ_k
                uniform: around a uniformly drawn position
run_batch       Metropolis moves at a fixed temperature with exact costs (PAVA);
                moves whose PAVA optimum violates a non-consecutive separation
                are rejected here and stored for an LP re-check
"""
from __future__ import annotations

import numpy as np
from numba import njit

from const_heuristics.alp_core import INF, cet


@njit(cache=True)
def seed_rng(seed):
    np.random.seed(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Shadow prices of the consecutive separations
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def shadow_prices(seq, L, x, r, d, dl, g, h, s, lam):
    """
    λ_k for k = 0..L−2.  Shrinking separation k lets either a run of aircraft
    ending at k slide later or a run starting at k+1 slide earlier, inside the
    tight block; the best such run gives the rate.  Runs cannot contain an
    aircraft pinned at its window bound in the direction of the slide.
    """
    for k in range(L - 1):
        lam[k] = 0.0
    best = -INF
    for k in range(L - 1):                                   # runs ending at k, sliding later
        if k == 0 or x[k] - x[k - 1] - s[seq[k - 1], seq[k]] >= 1e-9:
            best = -INF
        j = seq[k]
        if x[k] >= d[j] - 1e-9:
            best = -INF
        else:
            gk = -h[j] if x[k] - dl[j] >= -1e-9 else g[j]
            best = gk + (best if best > 0.0 else 0.0)
        if x[k + 1] - x[k] - s[j, seq[k + 1]] < 1e-9 and best > 0.0:
            lam[k] = best
    best = -INF
    for k in range(L - 2, -1, -1):                           # runs starting at k+1, sliding earlier
        if k == L - 2 or x[k + 2] - x[k + 1] - s[seq[k + 1], seq[k + 2]] >= 1e-9:
            best = -INF
        j = seq[k + 1]
        if x[k + 1] <= r[j] + 1e-9:
            best = -INF
        else:
            gk = h[j] if x[k + 1] - dl[j] > 1e-9 else -g[j]
            best = gk + (best if best > 0.0 else 0.0)
        if x[k + 1] - x[k] - s[seq[k], j] < 1e-9 and best > lam[k]:
            lam[k] = best


@njit(cache=True)
def prices(seq, L, x, r, d, dl, g, h, s, lam, clam):
    """Shadow prices and their running sum (for sampling ∝ λ); returns the total."""
    shadow_prices(seq, L, x, r, d, dl, g, h, s, lam)
    tot = 0.0
    for k in range(L - 1):
        tot += lam[k]
        clam[k] = tot
    return tot


# ─────────────────────────────────────────────────────────────────────────────
# Moves
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _insert(seq, L, a, b, out):
    """Element at position a ends at position b."""
    if a < b:
        for k in range(a):
            out[k] = seq[k]
        for k in range(a, b):
            out[k] = seq[k + 1]
        out[b] = seq[a]
        for k in range(b + 1, L):
            out[k] = seq[k]
    else:
        for k in range(b):
            out[k] = seq[k]
        out[b] = seq[a]
        for k in range(b + 1, a + 1):
            out[k] = seq[k - 1]
        for k in range(a + 1, L):
            out[k] = seq[k]


@njit(cache=True)
def _swap(seq, L, a, b, out):
    for k in range(L):
        out[k] = seq[k]
    out[a] = seq[b]
    out[b] = seq[a]


@njit(cache=True)
def _block(seq, L, a, bl, b, out):
    """Block seq[a:a+bl] moved so that it starts at position b of the result (0 ≤ b ≤ L − bl)."""
    i_rest = 0
    for k in range(L):
        if b <= k < b + bl:
            out[k] = seq[a + k - b]
        else:
            out[k] = seq[i_rest] if i_rest < a else seq[i_rest + bl]
            i_rest += 1


@njit(cache=True)
def _span(dcdf, w):
    """Span t in 1..w with P(t) ∝ 1/t (dcdf = normalised cumulative weights)."""
    u = np.random.random()
    for t in range(w):
        if u <= dcdf[t]:
            return t + 1
    return w


@njit(cache=True)
def _pick(clam, m, v):
    """Smallest k with clam[k] > v."""
    lo = 0
    hi = m - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if clam[mid] > v:
            hi = mid
        else:
            lo = mid + 1
    return lo


@njit(cache=True)
def propose(seq, L, x, dl, clam, total, w, guide, p_ins, p_swap, dcdf, out):
    """
    Write a neighbour of seq into out; False if the drawn move is not valid.
    Positions i < j bound the move (j − i = span).  `earlier` = the move brings
    the aircraft at j forward (else the aircraft at i backward).
    """
    if L < 2:
        return False
    t = _span(dcdf, w)
    if total > 0.0 and np.random.random() < guide:
        k = _pick(clam, L - 1, np.random.random() * total)       # separation between k and k+1
        u = np.random.randint(0, t)
        i = k - u
        j = k + 1 + (t - 1 - u)
        if i < 0:
            j -= i
            i = 0
        if j > L - 1:
            i -= j - (L - 1)
            j = L - 1
        if i < 0:
            i = 0
        if i >= j:
            return False
        late_j = x[j] > dl[seq[j]] + 1e-9
        early_i = x[i] < dl[seq[i]] - 1e-9
        if late_j and not early_i:
            earlier = True
        elif early_i and not late_j:
            earlier = False
        else:
            earlier = np.random.random() < 0.5
    else:
        p = np.random.randint(0, L)
        if np.random.random() < 0.5:
            i = max(0, p - t)
            j = p
            earlier = True
        else:
            i = p
            j = min(L - 1, p + t)
            earlier = False
        if i >= j:
            return False
    mv = np.random.random()
    if mv < p_ins:
        if earlier:
            _insert(seq, L, j, i, out)
        else:
            _insert(seq, L, i, j, out)
    elif mv < p_ins + p_swap:
        _swap(seq, L, i, j, out)
    else:
        bl = 2 + np.random.randint(0, 2)
        if earlier:
            a = j
            b = i
            if a + bl > L:
                return False
        else:
            a = i - bl + 1
            b = j - bl + 1
            if a < 0:
                return False
        if b == a or b < 0 or b > L - bl:
            return False
        _block(seq, L, a, bl, b, out)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Metropolis batch and temperature calibration
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def run_batch(seq, x, lam, clam, best_seq, cost, best_cost, T, n_moves, w, guide, p_ins, p_swap,
              dcdf, r, d, dl, g, h, s, s_max, buf, xbuf, flag_seqs, flag_lb):
    """
    n_moves Metropolis steps at temperature T.  seq/x/lam/clam/best_seq are
    updated in place.  Returns (cost, best_cost, accepted, priced, it_best,
    n_flagged); it_best is the step of the last best improvement (−1 if none).
    """
    L = seq.shape[0]
    total = clam[L - 2] if L > 1 else 0.0
    n_acc = 0
    n_priced = 0
    it_best = -1
    n_flag = 0
    max_flag = flag_lb.shape[0]
    for it in range(n_moves):
        if not propose(seq, L, x, dl, clam, total, w, guide, p_ins, p_swap, dcdf, buf):
            continue
        c, ok = cet(buf, L, r, d, dl, g, h, s, s_max, xbuf)
        if c == INF:
            continue
        if not ok:
            if c < cost - 1e-9 and n_flag < max_flag:
                for k in range(L):
                    flag_seqs[n_flag, k] = buf[k]
                flag_lb[n_flag] = c
                n_flag += 1
            continue
        n_priced += 1
        dlt = c - cost
        if dlt <= 0.0 or np.random.random() < np.exp(-dlt / T):
            for k in range(L):
                seq[k] = buf[k]
                x[k] = xbuf[k]
            cost = c
            n_acc += 1
            total = prices(seq, L, x, r, d, dl, g, h, s, lam, clam)
            if cost < best_cost - 1e-9:
                best_cost = cost
                it_best = it
                for k in range(L):
                    best_seq[k] = seq[k]
    return cost, best_cost, n_acc, n_priced, it_best, n_flag


@njit(cache=True)
def sample_uphill(seq, x, clam, cost, n_samples, w, guide, p_ins, p_swap, dcdf,
                  r, d, dl, g, h, s, s_max, buf, xbuf, out):
    """Cost increases of up to n_samples feasible uphill neighbours of seq; returns how many."""
    L = seq.shape[0]
    total = clam[L - 2] if L > 1 else 0.0
    m = 0
    for _ in range(20 * n_samples):
        if m >= n_samples:
            break
        if not propose(seq, L, x, dl, clam, total, w, guide, p_ins, p_swap, dcdf, buf):
            continue
        c, ok = cet(buf, L, r, d, dl, g, h, s, s_max, xbuf)
        if c == INF or not ok:
            continue
        if c > cost + 1e-9:
            out[m] = c - cost
            m += 1
    return m


def span_cdf(w: int) -> np.ndarray:
    """Normalised cumulative weights of P(t) ∝ 1/t, t = 1..w."""
    p = 1.0 / np.arange(1, w + 1)
    return np.cumsum(p / p.sum())
