"""
generator.py — synthetic single-runway instances for the design of experiments
===============================================================================
Structure follows the OR Library instances (Beasley et al. 2000) without
reusing them:

  wake classes      light / medium / heavy with shares 0.3 / 0.5 / 0.2; class
                    separation matrix (row = leader, seconds)
                        light  [ 82,  69,  60]
                        medium [131,  69,  60]
                        heavy  [196, 157,  96]
                    times a per-pair factor U(0.95, 1.05) (aircraft-specific rows),
                    then min-plus closure so s_ik ≤ s_ij + s_jk for every triple
                    (12 of the 13 OR Library instances satisfy the triangle
                    inequality)
  congestion ρ      mean separation s̄ / mean gap between consecutive target times;
                    target gaps ~ Gamma(shape 2) with mean s̄ / ρ
  windows           "wide"  (like airland9–13):  δ − r = U(0.5, 1.5)·4 s̄,
                                                 d − δ = U(0.5, 1.5)·17 s̄
                    "tight" (like airland6–7):   δ − r = U(0.5, 1.5)·1 s̄,
                                                 d − δ = U(0.5, 1.5)·1.6 s̄
                    then d is extended where needed so that the target-order
                    sequence is feasible (every generated instance is feasible)
  penalties         g, h ~ U(1, 3)

Instances are written in OR Library format so alp_core.load_instance reads them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from const_heuristics.alp_core import Instance

CLASS_SHARE = np.array([0.3, 0.5, 0.2])
CLASS_SEP = np.array([[82.0, 69.0, 60.0],
                      [131.0, 69.0, 60.0],
                      [196.0, 157.0, 96.0]])


def metric_closure(s: np.ndarray) -> np.ndarray:
    """Smallest matrix ≤ s satisfying s_ik ≤ s_ij + s_jk (min-plus closure, diagonal 0)."""
    s = s.copy()
    np.fill_diagonal(s, 0.0)
    for k in range(s.shape[0]):
        s = np.minimum(s, s[:, k:k + 1] + s[k:k + 1, :])
    return s


def triangle_violation_rate(s: np.ndarray, samples: int = 200_000, seed: int = 0) -> float:
    """Share of random distinct triples (i, j, k) with s_ik > s_ij + s_jk."""
    rng = np.random.default_rng(seed)
    n = s.shape[0]
    i, j, k = (rng.integers(0, n, samples) for _ in range(3))
    keep = (i != j) & (j != k) & (i != k)
    i, j, k = i[keep], j[keep], k[keep]
    return float(np.mean(s[i, k] > s[i, j] + s[j, k] + 1e-9))


WINDOWS = {"wide": (4.0, 17.0), "tight": (1.0, 1.6)}     # median (δ − r, d − δ) in units of s̄


def generate(n: int, congestion: float, windows: str, seed: int, name: str = "") -> Instance:
    if windows not in WINDOWS:
        raise ValueError(f"windows must be one of {sorted(WINDOWS)}, not {windows!r}")
    rng = np.random.default_rng(seed)
    cls = rng.choice(3, size=n, p=CLASS_SHARE)
    s = metric_closure(np.rint(CLASS_SEP[cls][:, cls] * rng.uniform(0.95, 1.05, (n, n))))
    np.fill_diagonal(s, 0.0)
    s_bar = float(s[~np.eye(n, dtype=bool)].mean())

    gaps = rng.gamma(2.0, s_bar / congestion / 2.0, size=n)
    delta = np.rint(rng.uniform(100.0, 300.0) + np.cumsum(gaps))
    early, late = WINDOWS[windows]
    r = np.maximum(0.0, np.rint(delta - rng.uniform(0.5, 1.5, n) * early * s_bar))
    d = np.rint(delta + rng.uniform(0.5, 1.5, n) * late * s_bar)

    # make the target-order sequence feasible (earliest times over all predecessor pairs)
    order = np.argsort(delta, kind="stable")
    x = np.empty(n)
    for q, j in enumerate(order):
        t = r[j]
        if q:
            t = max(t, float(np.max(x[:q] + s[order[:q], j])))
        x[q] = t
        if t > d[j]:
            d[j] = np.rint(t + rng.uniform(0.0, 2.0) * s_bar)
    g = np.round(rng.uniform(1.0, 3.0, n), 2)
    h = np.round(rng.uniform(1.0, 3.0, n), 2)
    return Instance(name or f"gen_n{n}_c{congestion:g}_{windows}_{seed}", n, r, delta, d, g, h, s)


def write_orlib(inst: Instance, path) -> None:
    """OR Library format: `n freeze`, then per aircraft `appearance r delta d g h` and its separation row."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{inst.n} 0"]
    for j in range(inst.n):
        lines.append(f"{max(0.0, inst.r[j] - 60):g} {inst.r[j]:g} {inst.delta[j]:g} {inst.d[j]:g} "
                     f"{inst.g[j]:g} {inst.h[j]:g}")
        row = inst.s[j]
        for a in range(0, inst.n, 20):
            lines.append(" ".join(f"{v:g}" for v in row[a:a + 20]))
    path.write_text("\n".join(lines) + "\n")
