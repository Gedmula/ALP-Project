"""
instance.py — OR-Library instance loading and synthetic instance generation.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from misc.mr_alp_2.models import Instance


def load_orlib(path: str, m: int = 1, name: str = "") -> Instance:
    """
    Load an OR-Library ALP instance file.

    File format (Beasley et al. 2000):
        Token 0   : n (number of aircraft)
        Token 1   : freeze_time (ignored)
        Per aircraft j = 0..n-1, 6 + n tokens:
            appearance_time  r[j]  delta[j]  d[j]  g[j]  h[j]
            s[j][0] ... s[j][n-1]

    Parameters
    ----------
    path : path to the .txt file.
    m    : number of runways (not encoded in the file; caller supplies it).
    name : instance identifier; defaults to the file stem.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"OR-Library file not found: {p.resolve()}")
    tok = p.read_text().split()
    try:
        n        = int(tok[0])
        expected = 2 + n * (6 + n)
        if len(tok) < expected:
            raise ValueError(
                f"{p.name}: {len(tok)} tokens found, {expected} expected.")
        r     = np.zeros(n); delta = np.zeros(n); d = np.zeros(n)
        g     = np.zeros(n); h     = np.zeros(n); s = np.zeros((n, n))
        idx   = 2
        for j in range(n):
            idx += 1                           # skip appearance_time
            r[j]     = float(tok[idx]); idx += 1
            delta[j] = float(tok[idx]); idx += 1
            d[j]     = float(tok[idx]); idx += 1
            g[j]     = float(tok[idx]); idx += 1
            h[j]     = float(tok[idx]); idx += 1
            for k in range(n):
                s[j, k] = float(tok[idx]); idx += 1
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Failed to parse {p.name}: {exc}") from exc

    return Instance(n=n, m=m, r=r, d=d, delta=delta, g=g, h=h, s=s,
                    name=name or p.stem)


def synthetic_instance(n: int = 15, m: int = 2, seed: int = 0) -> Instance:
    """
    Generate a synthetic ALP instance using ICAO-like wake-vortex separations.

    Aircraft types are drawn uniformly from {Heavy, Medium, Light}.
    Separation matrix follows the 3×3 ICAO matrix (Beasley et al. 2000, §2).
    Target times are spaced to allow feasible schedules.
    """
    rng   = np.random.default_rng(seed)
    SEP   = np.array([[82, 69, 60],
                      [131, 69, 60],
                      [196, 157, 96]], dtype=float)
    types = rng.integers(0, 3, n)
    s     = np.array([[SEP[types[j], types[k]] if j != k else 0.0
                       for k in range(n)] for j in range(n)])
    interval = float(SEP.max()) * 1.6
    jitter   = interval * 0.08
    delta    = np.sort(np.arange(n) * interval
                       + rng.uniform(-jitter, jitter, n))
    hw = float(SEP.max()) * 1.5
    r  = np.maximum(delta - rng.uniform(hw * 0.6, hw, n), 0.0)
    d  = delta + rng.uniform(hw * 0.6, hw, n)
    g, h = rng.uniform(5, 30, n), rng.uniform(5, 30, n)
    return Instance(n=n, m=m, r=r, d=d, delta=delta, g=g, h=h, s=s,
                    name=f"synthetic_n{n}_m{m}")