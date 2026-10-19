"""
design.py — experimental designs in coded units (−1, 0, +1)
===============================================================================
screening()   2^(7−3) fractional factorial, resolution IV, generators
              E = ABC, F = BCD, G = ACD (defining relation I = ABCE = BCDF = ACDG
              = ADEF = BDEG = ABFG = CEFG), plus centre points.
              Main effects are clear of two-factor interactions; two-factor
              interactions are aliased in chains (alias_chains()).
foldover()    single-factor foldover: the same 16 runs with one factor's sign
              reversed.  Combined with the original fraction it separates every
              two-factor interaction involving that factor from its chain.
ccd()         face-centred central composite design on k factors: 2^k corners
              (2^(k−1), resolution V, when k ≥ 5), 2k axial points at ±1 and
              centre points.
"""
from __future__ import annotations

from itertools import combinations, product
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

LETTERS = list("ABCDEFG")
GENERATORS = {"E": "ABC", "F": "BCD", "G": "ACD"}


def _base(k: int) -> np.ndarray:
    """Full 2^k factorial in standard (Yates) order, first factor changing fastest."""
    rows = list(product([-1, 1], repeat=k))
    return np.array([r[::-1] for r in rows], dtype=float)


def screening(n_center: int = 4) -> pd.DataFrame:
    base = _base(4)
    df = pd.DataFrame(base, columns=list("ABCD"))
    for f, word in GENERATORS.items():
        df[f] = np.prod([df[c] for c in word], axis=0)
    df["point"] = "factorial"
    centre = pd.DataFrame([{c: 0.0 for c in LETTERS} | {"point": "center"} for _ in range(n_center)])
    df = pd.concat([df, centre], ignore_index=True)
    df.insert(0, "config", [f"S{i + 1:02d}" for i in range(len(df))])
    return df


def foldover(factor: str) -> pd.DataFrame:
    """Factorial runs of the screening fraction with `factor` reversed (no new centre points)."""
    df = screening(0)
    df[factor] = -df[factor]
    df["point"] = f"foldover_{factor}"
    df["config"] = [f"F{factor}{i + 1:02d}" for i in range(len(df))]
    return df


def ccd(factors: List[str], n_center: int = 6, fixed: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """Face-centred CCD on `factors`; the other factors are held at `fixed` (coded, default 0)."""
    k = len(factors)
    fixed = fixed or {}
    if k >= 5:
        base = _base(k - 1)
        corners = np.column_stack([base, np.prod(base, axis=1)])      # last = product of the others (res V)
    else:
        corners = _base(k)
    rows = [dict(zip(factors, r), point="factorial") for r in corners]
    for f in factors:
        for v in (-1.0, 1.0):
            rows.append({g: (v if g == f else 0.0) for g in factors} | {"point": "axial"})
    rows += [{g: 0.0 for g in factors} | {"point": "center"} for _ in range(n_center)]
    df = pd.DataFrame(rows)
    for c in LETTERS:
        if c not in df:
            df[c] = float(fixed.get(c, 0.0))
    df = df[LETTERS + ["point"]]
    df.insert(0, "config", [f"R{i + 1:02d}" for i in range(len(df))])
    return df


def model_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Main-effect and two-factor-interaction columns (products of coded levels)."""
    out = df[LETTERS].copy()
    for a, b in combinations(LETTERS, 2):
        out[a + b] = df[a] * df[b]
    return out


def alias_chains(df: Optional[pd.DataFrame] = None) -> List[List[str]]:
    """Groups of two-factor interactions with identical contrasts over the factorial runs."""
    df = screening(0) if df is None else df[df["point"] != "center"]
    cols = model_columns(df)
    chains: Dict[tuple, List[str]] = {}
    for a, b in combinations(LETTERS, 2):
        key = tuple(cols[a + b].astype(int))
        neg = tuple(-v for v in key)
        if neg in chains:
            chains[neg].append("-" + a + b)
        else:
            chains.setdefault(key, []).append(a + b)
    for c in LETTERS:                       # check: no main effect shares a contrast with a 2FI
        key = tuple(cols[c].astype(int))
        assert key not in chains and tuple(-v for v in key) not in chains, f"{c} aliased with a 2FI"
    return list(chains.values())


def resolution_check() -> str:
    """Human-readable summary of the screening fraction."""
    chains = alias_chains()
    lines = ["2^(7-3) resolution IV, generators " + ", ".join(f"{k} = {v}" for k, v in GENERATORS.items()),
             "main effects: clear of two-factor interactions",
             f"two-factor interaction alias chains ({len(chains)}):"]
    lines += ["   " + " = ".join(ch) for ch in chains]
    return "\n".join(lines)


if __name__ == "__main__":
    print(screening().to_string(index=False))
    print()
    print(resolution_check())
