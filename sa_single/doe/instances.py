"""
instances.py — generated instance sets for the DOE
===============================================================================
The *_large sets (tune_large, holdout_large: 8 sizes 50 … 500 × 3 congestion × 2 window types = 48 instances) are
read from their committed manifests; the two sets below can be regenerated.

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
from typing import List

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


DOE_DIR = INST_DIR.parent
RESULTS_DIR = DOE_DIR / "results"


def informative_instances(phase: str = "screening", tol: float = 1e-9) -> List[str]:
    """Instances on which the runs of `phase` did not all end on the same cost.

    On the others every configuration reaches the same best value, so they carry no information about the factors
    (28 of the 48 tune_large instances in the screening).  Read from the committed records, so the list is fixed."""
    rec = pd.read_csv(RESULTS_DIR / phase / "records.csv", usecols=["instance", "best_cost"])
    g = rec.groupby("instance")["best_cost"]
    spread = (g.max() - g.min()) / np.maximum(g.mean().abs(), 1.0)
    return sorted(spread[spread > tol].index)


def manifest(set_name: str, subset=None) -> pd.DataFrame:
    """Manifest of a set (tune, holdout, tune_large, holdout_large).

    subset: None for every instance, "informative" for the instances that discriminated in the screening, or an
    explicit list of instance names."""
    path = INST_DIR / set_name / "manifest.csv"
    if path.exists():
        df = pd.read_csv(path)
    elif set_name in SET_SEED:
        df = build(set_name)
    else:
        raise FileNotFoundError(f"no manifest for instance set {set_name!r}: {path}")
    if subset is not None:
        names = informative_instances() if subset == "informative" else list(subset)
        missing = sorted(set(names) - set(df["instance"]))
        if missing:
            raise ValueError(f"subset instances not in set {set_name!r}: {missing}")
        df = df[df["instance"].isin(names)].reset_index(drop=True)
    return df


def _resolve(file: str) -> Path:
    """Manifest paths are relative to the repository root (tune, holdout) or to sa_single/doe (the *_large sets)."""
    for base in (ROOT, DOE_DIR):
        if (base / file).exists():
            return base / file
    raise FileNotFoundError(file)


def load(row) -> Instance:
    return load_instance(_resolve(row["file"]), row["instance"])


if __name__ == "__main__":
    for s in SET_SEED:
        df = build(s)
        print(df[["instance", "n", "congestion", "windows", "triangle_violation_rate", "mean_window"]].to_string(index=False))
