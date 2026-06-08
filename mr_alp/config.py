"""
config.py — MR-ALP Solver: Runtime Configuration and Benchmark Reference Data
==============================================================================
All runtime-tunable constants live here.  No other module in the package
modifies these values at runtime.  Edit §0 to change solver behaviour;
edit §1 only when benchmark data is updated.

§0  Run-time settings
§1  Benchmark reference optima  (KNOWN_OPTIMA)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

# ═══════════════════════════════════════════════════════════════════════════
#   §0  CONFIGURE HERE
# ═══════════════════════════════════════════════════════════════════════════

BATCH_MODE    = True
INSTANCE_PATH = "data/airland1.txt"
FOLDER        = "data/"

# Runway configurations to evaluate for each OR Library instance.
INSTANCE_RUNWAYS: Dict[str, List[int]] = {
    "airland1":  [2, 3],        "airland2":  [2, 3],
    "airland3":  [2, 3],        "airland4":  [2, 3, 4],
    "airland5":  [2, 3, 4],     "airland6":  [2, 3],
    "airland7":  [2],           "airland8":  [2, 3],
    "airland9":  [2, 3, 4],     "airland10": [2, 3, 4, 5],
    "airland11": [2, 3, 4, 5],  "airland12": [2, 3, 4, 5],
    "airland13": [2, 3, 4, 5],
}

# ── Parallelism & chain counts ─────────────────────────────────────────────
# USE_ALL_SEEDS: every evaluated seed heuristic becomes an SA starting point.
# When True, N_WORKERS limits concurrent SA processes; N_CHAINS is overridden.
# When False, only the top N_CHAINS seeds (by LP value, with diversity guard)
# are forwarded to SA, and N_WORKERS SA chains run in parallel.
USE_ALL_SEEDS = False
N_WORKERS     = 3 if USE_ALL_SEEDS else 7
N_CHAINS      = 4          # SA starting points when USE_ALL_SEEDS=False

# ── Time budgets ──────────────────────────────────────────────────────────
T_LIMIT     = 600.0        # default adaptive SA+VND+PR budget (seconds)
MAX_T_LIMIT = 1200.0       # hard ceiling for large / high-gap instances

# ── RBI Optuna tuning ─────────────────────────────────────────────────────
N_OPTUNA_WORKERS  = 4
RUN_RBI_OPTUNA    = True   # run Optuna when (inst,m) is absent from RBI_PARAM_BANK
N_RBI_TRIALS_BASE = 30
RBI_OPTUNA_SEED   = 42

# ── SA Optuna tuning ──────────────────────────────────────────────────────
RUN_SA_OPTUNA    = True    # run Optuna when (inst,m) is absent from SA_PARAM_BANK
SA_N_TRIALS_BASE = 20
SA_OPTUNA_SEED   = 123
SA_N_OPTUNA_JOBS = 4

# ── Elite pool ────────────────────────────────────────────────────────────
ELITE_POOL_MAX = 20        # maximum retained elite solutions
ELITE_MIN_DIV  = 5         # minimum runway-Hamming distance for diversity admission

# ── GPU dispatch ──────────────────────────────────────────────────────────
USE_GPU   = True
GPU_MIN_N = 200            # minimum n to activate GPU-accelerated TC computation

# ── Output ────────────────────────────────────────────────────────────────
OUTPUT_DIR   = Path("MR_results")
SAVE_RESULTS = True
SAVE_PLOTS   = True

# ── TC-RBI insertion-cost weight defaults ─────────────────────────────────
DEFAULT_ETA      = 0.50
DEFAULT_MU_TC    = 1.00
DEFAULT_MU_LATE  = 0.25
DEFAULT_MU_COUNT = 0.75
DEFAULT_MU_SEP   = 0.05

# ── Seed heuristic scalars ────────────────────────────────────────────────
# ATC look-ahead scaling K: large → WSPT; small → minimum-slack (Pinedo §14.2)
ATC_K   = 2.5
ATCS_K1 = 2.0   # ATC urgency scaling for ATCS (H5)
ATCS_K2 = 2.0   # ATC separation scaling for ATCS (H5)

# GRASP Restricted Candidate List sizes for the two GRASP seed variants (H9)
GRASP_K_VALUES: Tuple[int, int] = (3, 7)

# MPDS (H7) is O(n³/m); skip for instances above this threshold
MPDS_MAX_N = 150


# ═══════════════════════════════════════════════════════════════════════════
#   §1  BENCHMARK REFERENCE OPTIMA
# ═══════════════════════════════════════════════════════════════════════════
#
# Single-runway values (m=1) are B&B-certified optima from Beasley et al.
# (2000) and serve as hard correctness targets.  Multi-runway values are
# Zhang et al. (2020) heuristic BKS; negative gaps are valid and flagged.
#
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