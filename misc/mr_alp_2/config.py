"""
config.py — Global constants and parameter defaults.

All tunable scalars are defined here to avoid hard-coded magic numbers
across modules.  Units follow the OR-Library convention (seconds, penalty
rate per second deviation from target).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple, List

# ── Parallelism ────────────────────────────────────────────────────────────
N_WORKERS: int = max(os.cpu_count() - 1, 1)

# ── Numerical floor ────────────────────────────────────────────────────────
EPS: float = 1e-9                  # §6: ε used throughout to guard divisions

# ── Insertion-cost weights (§8) ────────────────────────────────────────────
DEFAULT_LAMBDA1: float = 0.8       # separation burden weight
DEFAULT_LAMBDA2: float = 0.1       # runway-balance weight

# ── Construction priority mixing factor (§9.4) ────────────────────────────
DEFAULT_ETA: float = 0.5           # regret vs criticality blend

# ── Inter-runway improvement (§12) ────────────────────────────────────────
DEFAULT_K:     int   = 20          # top-K aircraft by move-pressure (§13)
DEFAULT_B_MAX: int   = 3           # normal block-transfer size cap (§14.3)

# ── Seed portfolio (§25 / §25b) ────────────────────────────────────────────
ATC_K:          float         = 2.0        # ATC lookahead scale factor
ATCS_K1:        float         = 2.0        # ATCS tardiness scale
ATCS_K2:        float         = 0.5        # ATCS separation scale
GRASP_K_VALUES: Tuple[int, int] = (3, 5)  # GRASP RCL sizes evaluated
MPDS_MAX_N:     int           = 150        # max n for O(n³) MPDS seed
ELITE_MIN_DIV:  int           = 2          # min runway-assignment Hamming diversity
USE_ALL_SEEDS:  bool          = False      # True → forward every feasible seed
N_CHAINS:       int           = 6          # number of seeds forwarded to SA

# ── Outer SA (§15) ────────────────────────────────────────────────────────
DEFAULT_ALPHA_OUTER:   float = 0.95
T_OUTER_INIT_FRAC:     float = 0.05  # T_outer_0 = best_lp * fraction

# ── Single-runway SA defaults (§11) ───────────────────────────────────────
DEFAULT_ALPHA_INNER: float = 0.97
DEFAULT_N_ITER:      int   = 2000
DEFAULT_T_MIN:       float = 1e-4
DEFAULT_I_MAX:       int   = 600
DEFAULT_M_STAG:      int   = 60
DEFAULT_CHI0:        float = 0.50

# ── Move-pressure weights w1..w6 (§13.6) ──────────────────────────────────
# w1=P_j, w2=DR_j, w3=SB_j, w4=CR_j, w5=G^move_j, w6=LB_rho
MP_WEIGHTS: Tuple[float, ...] = (0.25, 0.25, 0.15, 0.15, 0.15, 0.05)

# ── Output ─────────────────────────────────────────────────────────────────
OUTPUT_DIR:   Path = Path("hybrid_results")
SAVE_RESULTS: bool = True
SAVE_PLOTS:   bool = True

# ── Known reference optima ─────────────────────────────────────────────────
# Single-runway values: B&B-certified optima (Beasley et al. 2000).
# Multi-runway values: Zhang et al. (2020) heuristic BKS.
KNOWN_OPTIMA: Dict[str, Dict[int, float]] = {
    "airland1":  {1: 700.00,    2: 90.00,    3: 0.00},
    "airland2":  {1: 1480.00,   2: 210.00,   3: 0.00},
    "airland3":  {1: 820.00,    2: 60.00,    3: 0.00},
    "airland4":  {1: 2520.00,   2: 640.00,   3: 130.00,  4: 0.00},
    "airland5":  {1: 3100.00,   2: 650.00,   3: 170.00,  4: 0.00},
    "airland6":  {1: 24442.00,  2: 554.00,   3: 0.00},
    "airland7":  {1: 1550.00,   2: 0.00},
    "airland8":  {1: 1950.00,   2: 135.00,   3: 0.00},
    "airland9":  {1: 5611.70,   2: 444.10,   3: 75.75,   4: 0.00},
    "airland10": {1: 12821.12,  2: 1143.70,  3: 205.21,  4: 34.22,  5: 0.00},
    "airland11": {1: 12654.18,  2: 1330.91,  3: 253.07,  4: 54.53,  5: 0.00},
    "airland12": {1: 16629.10,  2: 1695.62,  3: 221.97,  4: 2.44,   5: 0.00},
    "airland13": {1: 39516.34,  2: 3943.85,  3: 673.85,  4: 89.95,  5: 0.00},
}

# ── Default runway counts per OR-Library instance ──────────────────────────
INSTANCE_RUNWAYS: Dict[str, List[int]] = {
    "airland1":  [2, 3],
    "airland2":  [2, 3],
    "airland3":  [2, 3],
    "airland4":  [2, 3, 4],
    "airland5":  [2, 3, 4],
    "airland6":  [2, 3],
    "airland7":  [2],
    "airland8":  [2, 3],
    "airland9":  [2, 3, 4],
    "airland10": [2, 3, 4, 5],
    "airland11": [2, 3, 4, 5],
    "airland12": [2, 3, 4, 5],
    "airland13": [2, 3, 4, 5],
}

# ── Multi-runway SA (§14–§17) ──────────────────────────────────────────────
MR_SA_N_ITER:     int   = 5000     # proxy SA iterations per call
MR_SA_ALPHA:      float = 0.995    # cooling rate
MR_SA_BETA:       float = 1.5      # reheat multiplier
MR_SA_M_STAG:     int   = 400      # stagnation iteration limit
MR_SA_B_MAX:      int   = 3        # standard block cap
MR_SA_B_STAG:     int   = 5        # stagnation block cap
MR_SA_R_POOL:     int   = 5        # candidate moves sampled per iteration
MR_SA_CHI0:       float = 0.30     # target acceptance prob for T calibration

# ── Elite pool (§20) ───────────────────────────────────────────────────────
ELITE_POOL_MAX:   int   = 8        # maximum pool size

# ── Adaptive move-pressure weights (§16.6) ─────────────────────────────────
ADAPT_W_XI:       float = 0.10     # learning rate ξ
ADAPT_W_BETA:     float = 1.5      # greediness β