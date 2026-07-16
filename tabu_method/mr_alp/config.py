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

import os
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
USE_ALL_SEEDS = os.environ.get("ALP_USE_ALL_SEEDS", "0").strip().lower() in {
    "1", "true", "yes", "on",
}
N_WORKERS     = int(os.environ.get(
    "ALP_N_WORKERS", "3" if USE_ALL_SEEDS else "7"))
N_CHAINS      = int(os.environ.get("ALP_N_CHAINS", "4"))

# ── Time budgets ──────────────────────────────────────────────────────────
T_LIMIT     = 600.0        # default adaptive SA+VND+PR budget (seconds)
MAX_T_LIMIT = 1200.0       # hard ceiling for large / high-gap instances

# ── SA phasing ────────────────────────────────────────────────────────────
# Wall-clock phasing sets search phase f = max(iteration, wall-clock)
# progress.  On large deadline-bound instances this pulls chains into the
# expensive mid/late operator tables while iterations are still slow,
# which starves the search (observed: airland13 m=2 completed 431/8000
# iterations and regressed 3.31% → 7.88%).  Keep False for comparable
# experiments until per-move delta evaluation lands.
WALL_CLOCK_PHASING       = False
WALL_CLOCK_PHASING_MAX_N = 250   # when True, applies only to n <= this

# ── Delta evaluation for insertion-scan operators (N3b, X3, X7, XE) ───────
# Staged position search (see mr_alp/delta.py): O(L) scoring of all
# insertion positions, exact check on the top DELTA_FINALISTS_K only.
# Profiled motivation: X3+N3b consumed 91% of candidate time on
# airland13 m=2 (42.6 / 25.7 ms per candidate).  Set False to A/B
# against the legacy full scan — results CAN differ (lbt is scored on
# finalists only), so compare via the 3-seed protocol, not single runs.
DELTA_EVAL        = os.environ.get("ALP_DELTA_EVAL", "1").strip().lower() not in {
    "0", "false", "no", "off",
}
DELTA_FINALISTS_K = 5

# Diversify SA control parameters across parallel chains.  The seed portfolio
# already gives different starting schedules; this gives the chains different
# search personalities at no extra wall-clock budget.
SA_CHAIN_DIVERSITY = os.environ.get("ALP_SA_CHAIN_DIVERSITY", "1").strip().lower() not in {
    "0", "false", "no", "off",
}

# Short-term memory for SA moves.  This is a Tabu Search idea: allow bad
# moves through SA, but avoid immediately undoing cross-runway assignments.
SA_TABU_ENABLED = os.environ.get("ALP_SA_TABU", "1").strip().lower() not in {
    "0", "false", "no", "off",
}
SA_TABU_TENURE = int(os.environ.get("ALP_SA_TABU_TENURE", "80"))
SA_TABU_MODE = os.environ.get(
    "ALP_SA_TABU_MODE", "fixed_attribute").strip().lower()
if SA_TABU_MODE not in {"fixed_attribute", "iteration", "reactive"}:
    raise ValueError(
        "ALP_SA_TABU_MODE must be fixed_attribute, iteration, or reactive")
SA_TABU_ASPIRATION = os.environ.get(
    "ALP_SA_TABU_ASPIRATION", "proxy").strip().lower()
if SA_TABU_ASPIRATION not in {"proxy", "lp", "hybrid"}:
    raise ValueError(
        "ALP_SA_TABU_ASPIRATION must be proxy, lp, or hybrid")
SA_TABU_FALLBACK = os.environ.get(
    "ALP_SA_TABU_FALLBACK", "unfiltered").strip().lower()
if SA_TABU_FALLBACK not in {"unfiltered", "least_recent"}:
    raise ValueError(
        "ALP_SA_TABU_FALLBACK must be unfiltered or least_recent")
SA_TABU_REACTIVE_MIN = int(os.environ.get(
    "ALP_SA_TABU_REACTIVE_MIN", str(max(1, SA_TABU_TENURE // 2))))
SA_TABU_REACTIVE_MAX = int(os.environ.get(
    "ALP_SA_TABU_REACTIVE_MAX", str(max(SA_TABU_TENURE, SA_TABU_TENURE * 2))))

LP_IMPACT_INIT = os.environ.get("ALP_LP_IMPACT_INIT", "1").strip().lower() not in {
    "0", "false", "no", "off",
}
BOTTLENECK_LNS = os.environ.get("ALP_BOTTLENECK_LNS", "1").strip().lower() not in {
    "0", "false", "no", "off",
}
ASSIGNMENT_REPAIR = os.environ.get("ALP_ASSIGNMENT_REPAIR", "1").strip().lower() not in {
    "0", "false", "no", "off",
}
SET_PARTITION_RECOMBINE = os.environ.get("ALP_SET_PARTITION_RECOMBINE", "1").strip().lower() not in {
    "0", "false", "no", "off",
}

# ── RBI Optuna tuning ─────────────────────────────────────────────────────
N_OPTUNA_WORKERS  = 4
RUN_RBI_OPTUNA    = True   # run Optuna when (inst,m) is absent from RBI_PARAM_BANK
N_RBI_TRIALS_BASE = 30
RBI_OPTUNA_SEED   = 42

# ── SA Optuna tuning ──────────────────────────────────────────────────────
# NOTE: SA_PARAM_BANK is empty, so RUN_SA_OPTUNA=True re-tunes SA params on
# every run via parallel TPE (n_jobs=4), which is NOT deterministic — and in
# an env without optuna it silently falls back to MRSAParams() defaults.
# The 2026-06-11 hard-case rerun used defaults while the baseline used tuned
# params, contaminating the comparison.  Keep False so all experiment arms
# share identical (default) SA params; re-enable only with a populated bank.
RUN_SA_OPTUNA    = False   # run Optuna when (inst,m) is absent from SA_PARAM_BANK
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
