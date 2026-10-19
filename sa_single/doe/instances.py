"""
instances.py — generated instance sets for the DOE
===============================================================================
Two disjoint sets with the same 24 classes (4 sizes × 3 congestion levels × 2
window types), one instance per class each:

    tune     used by the screening and response-surface phases
    holdout  used only by the confirmation phase

Files are written once (OR Library format) to sa_single/doe/instances/<set>/
with a manifest.csv of the class of every instance.  Run from the repository
root:  python sa_single/doe/instances.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from const_heuristics.alp_core import Instance, load_instance
from sa_single.generator import generate, triangle_violation_rate, write_orlib

SIZES = [100, 200, 300, 400]
CONGESTION = [0.6, 0.8, 1.0]
WINDOW_TYPES = ["wide", "tight"]
SET_SEED = {"tune": 1000, "holdout": 2000}
INST_DIR = ROOT / "sa_single" / "doe" / "instances"


def instance_name(n: int, rho: float, windows: str, set_name: str) -> str:
    return f"g{n}_c{int(round(rho * 10))}_{windows[0]}_{set_name}"


def build(set_name: str) -> pd.DataFrame:
    rows = []
    k = 0
    for n in SIZES:
        for rho in CONGESTION:
            for win in WINDOW_TYPES:
                name = instance_name(n, rho, win, set_name)
                path = INST_DIR / set_name / f"{name}.txt"
                inst = generate(n, rho, win, SET_SEED[set_name] + k, name)
                write_orlib(inst, path)
                rows.append(dict(instance=name, set=set_name, n=n, congestion=rho, windows=win,
                                 seed=SET_SEED[set_name] + k, file=str(path.relative_to(ROOT)),
                                 triangle_violation_rate=triangle_violation_rate(inst.s),
                                 mean_window=float(np.mean(inst.d - inst.r))))
                k += 1
    df = pd.DataFrame(rows)
    df.to_csv(INST_DIR / set_name / "manifest.csv", index=False)
    return df


def manifest(set_name: str) -> pd.DataFrame:
    """Manifest of a set, generating the files if they do not exist yet."""
    path = INST_DIR / set_name / "manifest.csv"
    return pd.read_csv(path) if path.exists() else build(set_name)


def load(row) -> Instance:
    return load_instance(ROOT / row["file"], row["instance"])


if __name__ == "__main__":
    for s in SET_SEED:
        df = build(s)
        print(df[["instance", "n", "congestion", "windows", "triangle_violation_rate", "mean_window"]].to_string(index=False))
