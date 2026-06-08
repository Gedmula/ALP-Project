# Multi-Runway Aircraft Landing Problem — MR-ALP Solver

Multi-runway scheduling under wake-vortex separation constraints, formulated as an extension of the single-runway ALP to $m$ parallel runways, and solved via a two-stage decomposition: parallel multi-start SA with elite-pool management and LP-VND post-processing, followed by exact timing optimisation via a HiGHS-backed LP.

For the single-runway version, see [README.md](README.md).

---

## Table of Contents

1. [Problem Formulation](#1-problem-formulation)
2. [Repository Layout](#2-repository-layout)
3. [Dependencies](#3-dependencies)
4. [Data](#4-data)
5. [Architecture Overview](#5-architecture-overview)
6. [Module Reference](#6-module-reference)
7. [Algorithms](#7-algorithms)
8. [Configuration Reference](#8-configuration-reference)
9. [Running the Pipeline](#9-running-the-pipeline)
10. [Outputs](#10-outputs)
11. [References](#11-references)

---

## 1. Problem Formulation

The multi-runway ALP generalises the single-runway scheduling problem to $m$ parallel landing runways. Given a set $J = \{1, \ldots, n\}$ of aircraft and $m$ runways, the problem is to assign each aircraft to exactly one runway and determine a landing sequence and landing time on that runway such that:

| Symbol | Meaning |
|---|---|
| $r_j$ | Release (earliest landing) time |
| $\delta_j$ | Target (preferred) landing time |
| $d_j$ | Deadline (latest landing) time |
| $s_{jk}$ | Minimum separation required when $j$ lands before $k$ **on the same runway** |
| $g_j$ | Cost per unit of earliness |
| $h_j$ | Cost per unit of tardiness |

Let $\boldsymbol{\pi} = (\pi_1, \ldots, \pi_m)$ denote the $m$ per-runway landing sequences and let $\rho(j)$ denote the runway assigned to aircraft $j$. The optimisation problem is:

$$x_j \in [r_j,\; d_j] \qquad \forall j \in J$$

$$x_k \geq x_j + s_{jk} \qquad \text{if } j \prec k \text{ on the same runway}$$

$$\min \sum_{j \in J} \bigl(g_j \cdot \max(\delta_j - x_j,\; 0) + h_j \cdot \max(x_j - \delta_j,\; 0)\bigr)$$

Separation is enforced only within a runway; inter-runway coupling arises solely through the assignment constraint (each aircraft lands on exactly one runway). Adding runways relaxes sequence pressure and allows near-zero or zero penalties once enough runways are available.

The pipeline exploits the two-stage structure: once the per-runway sequences $\boldsymbol{\pi}$ are fixed, all binary assignment decisions are determined and the timing subproblem reduces to a single joint LP solved by HiGHS.

---

## 2. Repository Layout

```
project_root/
├── mr_sa_alp.py             # Entry point — discovers instances, runs jobs, writes outputs
├── ramp_rbi.py              # Standalone TC-RBI construction heuristic demo
├── mr_alp/                  # Solver package
│   ├── __init__.py          # Public API re-exports
│   ├── config.py            # Runtime settings and benchmark reference optima (§0–§1)
│   ├── models.py            # Instance, HeuristicParams, MRSAParams; pre-tuned param banks
│   ├── instance.py          # OR Library parser, Numba kernels, feasibility checks
│   ├── lp.py                # Stage-2 LP (HiGHS) and exact verification
│   ├── construction.py      # TC-RBI heuristic (§8–§9) and seed portfolio H1–H9 (§25)
│   ├── operators.py         # SA neighbourhood operators N1–N4, X1–X4, X7, XE (§16–§17)
│   ├── proxy.py             # Surrogate proxy objective and per-aircraft scoring (§10, §14–§15)
│   ├── sa.py                # SA chain, Optuna tuning, spawn-safe worker (§11, §18, §23, §26)
│   ├── repair.py            # LP-guided repair, VND polish, elite pool, path relinking (§19–§24)
│   ├── solver.py            # Parallel multi-start SA — ms_mr_sa (§27)
│   └── output.py            # Reporting, persistence, and visualisation (§28–§30)
├── data/
│   └── airland{1-13}.txt    # OR Library benchmark instances
├── MR_results/              # Auto-generated outputs (written by output.py)
│   ├── summary.csv
│   ├── schedules.csv
│   ├── alternatives.csv
│   ├── verification.txt
│   ├── run_metadata.json
│   └── plots/
│       ├── gap/
│       ├── convergence/
│       ├── lp_timeline/
│       ├── time_to_best/
│       ├── elite_pool/
│       ├── gantt/
│       └── seeds/
└── Single_runway_SA.py      # Single-runway solver (see README.md)
```

---

## 3. Dependencies

Python 3.9 or later is required.

**Required:**

```
numpy
scipy          # linprog / HiGHS interface
matplotlib
```

**Optional but recommended:**

```
tqdm           # progress bars
optuna         # Optuna TPE for RBI and SA hyperparameter tuning
numba          # JIT-compiled feasibility and insertion kernels (8–12× speedup on inner loop)
torch          # GPU-accelerated TC computation (CPU fallback is automatic)
```

Install everything at once:

```bash
pip install numpy scipy matplotlib tqdm optuna numba torch
```

The LP solver is HiGHS, accessed through `scipy.optimize.linprog` with `method='highs'`. No separate HiGHS installation is required when using SciPy ≥ 1.9.

---

## 4. Data

The solver uses the same OR Library ALP instances as the single-runway version (airland1–airland13). See [README.md §4](README.md#4-data) for download instructions and file format. Data files are placed in `./data/` relative to `mr_sa_alp.py`.

**Multi-runway benchmark reference values (`KNOWN_OPTIMA` in `config.py`):**

Multi-runway values below are Zhang et al. (2020) heuristic BKS — not certified optima. Negative gaps (solver beats BKS) are valid and flagged with `★` in console output. $m=1$ values are B&B-certified optima from Beasley et al. (2000).

| Instance | $m=1$ | $m=2$ | $m=3$ | $m=4$ | $m=5$ |
|---|---|---|---|---|---|
| airland1 | 700.00 | 90.00 | 0.00 | — | — |
| airland2 | 1480.00 | 210.00 | 0.00 | — | — |
| airland3 | 820.00 | 60.00 | 0.00 | — | — |
| airland4 | 2520.00 | 640.00 | 130.00 | 0.00 | — |
| airland5 | 3100.00 | 650.00 | 170.00 | 0.00 | — |
| airland6 | 24442.00 | 554.00 | 0.00 | — | — |
| airland7 | 1550.00 | 0.00 | — | — | — |
| airland8 | 1950.00 | 135.00 | 0.00 | — | — |
| airland9 | 5611.70 | 444.10 | 75.75 | 0.00 | — |
| airland10 | 12821.12 | 1143.70 | 205.21 | 34.22 | 0.00 |
| airland11 | 12654.18 | 1330.91 | 253.07 | 54.53 | 0.00 |
| airland12 | 16629.10 | 1695.62 | 221.97 | 2.44 | 0.00 |
| airland13 | 39516.34 | 3943.85 | 673.85 | 89.95 | 0.00 |

> **Note on zero entries.** Zero entries indicate that enough runways are available for every aircraft to land without competing for slots — the LP trivially achieves zero penalty by spreading the fleet across runways.

---

## 5. Architecture Overview

```
mr_sa_alp.py  (§31–§32)
│
├── _run_one_job (§31)
│     Load instance → resolve params → seed portfolio → ms_mr_sa → verify → export
│
└── main (§32)
      Discover instances × runway counts → submit jobs → flush outputs in arrival order

mr_alp package
│
├── Instance                  Data container; pre-computes W_bar, s_bar, Pen_bar, T_span
├── load_instance             OR Library parser + token-count validation
│
├── Stage-2 LP
│     stage2_lp_objective     Full pairwise LP via HiGHS (joint across all runways)
│     verify_and_exact_obj    Earliest-time propagation + constraint audit
│
├── Construction
│     ramp_rbi                TC-RBI: regret-based insertion with five Optuna-tuned weights
│     _build_seed_portfolio   LP-evaluated portfolio (H1–H9) with per-seed wall-time tracking
│
├── Proxy objective
│     total_target_conflict   O(n²) pairwise TC sum; GPU-accelerated for n ≥ GPU_MIN_N
│     compute_proxy           Composite proxy: TC + LBT + SEP terms
│
├── Neighbourhood operators (§16–§17)
│     Within-runway:  N1 (adjacent swap), N2 (arbitrary swap),
│                     N3b (best re-insertion), N4 (block relocation)
│     Cross-runway:   X1 (transfer), X2 (cross-swap), X3 (best transfer),
│                     X4 (block transfer), X7 (TC-guided repair), XE (ejection alias)
│
├── SA (§23)
│     run_mr_sa               Single SA chain with reactive cooling and LP-timeline tracking
│     _sa_worker              Spawn-safe ProcessPoolExecutor entry point
│
├── Repair & post-processing (§19–§24)
│     lp_guided_penalty_repair   Globally relocate highest-penalty aircraft
│     lp_guided_pair_swap        Cross-runway swap of near-δ-time aircraft
│     target_conflict_repair     Deterministic repair for near-zero objectives
│     ejection_chain_transfer    Depth-D ejection chain from high-impact aircraft
│     lns_remove_reinsert        LNS: simultaneous removal and joint reinsertion
│     ElitePool                  Fixed-size pool with runway-Hamming diversity guard
│     path_relink                Walk between elite pairs, LP-evaluating at step intervals
│     lp_vnd_polish              Monotone LP-VND cycling through all five repair operators
│
└── ms_mr_sa (§27)            Parallel K-chain SA + elite pool + VND + path relinking
```

---

## 6. Module Reference

### `mr_alp/config.py`

All runtime-tunable constants. Edit `§0` to change solver behaviour; `§1` (`KNOWN_OPTIMA`) is read-only benchmark data.

Key constants:

| Constant | Default | Effect |
|---|---|---|
| `INSTANCE_RUNWAYS` | per-instance dict | Runway counts $m$ evaluated for each instance |
| `BATCH_MODE` | `True` | Process all instances in `FOLDER`; single-file mode when `False` |
| `N_WORKERS` | 7 | Concurrent SA processes |
| `N_CHAINS` | 4 | SA starting points per job (when `USE_ALL_SEEDS=False`) |
| `USE_ALL_SEEDS` | `False` | When `True`, all H1–H9 seeds become SA start points |
| `T_LIMIT` | 600 s | Default SA+VND+PR time budget per job |
| `MAX_T_LIMIT` | 1200 s | Hard ceiling for large or high-gap instances |
| `ELITE_POOL_MAX` | 20 | Maximum retained elite solutions |
| `ELITE_MIN_DIV` | 5 | Minimum runway-Hamming distance for elite pool admission |
| `RUN_RBI_OPTUNA` | `True` | Run Optuna when (inst, m) absent from `RBI_PARAM_BANK` |
| `RUN_SA_OPTUNA` | `True` | Run Optuna when (inst, m) absent from `SA_PARAM_BANK` |
| `USE_GPU` | `True` | GPU-accelerated TC computation (CPU fallback automatic) |
| `GPU_MIN_N` | 200 | Minimum $n$ for GPU dispatch |
| `OUTPUT_DIR` | `MR_results/` | Output directory |

### `mr_alp/models.py`

#### `Instance`

Parsed and pre-processed ALP instance (0-indexed). Fields mirror the problem inputs (`n`, `r`, `delta`, `d`, `g`, `h`, `s`). Derived statistics computed in `__post_init__`:

| Attribute | Meaning |
|---|---|
| `W_bar` | Mean time-window width $E[d_j - r_j]$ |
| `s_bar` | Mean positive off-diagonal separation |
| `h_bar` | Mean tardiness penalty $E[h_j]$ |
| `Pen_bar` | $E[\max(g_j, h_j)] \times W_\text{bar}$ — runway-balance cost scale |
| `T_span` | Total horizon $\max(d) - \min(r)$ |
| `p_arr` | $\max(g_j, h_j)$ per aircraft (combined penalty rate) |

GPU tensors (`_s_gpu`, `_delta_gpu`, `_p_arr_gpu`) are excluded from `__getstate__` so that `Instance` objects are safely serialised across `ProcessPoolExecutor` workers.

#### `HeuristicParams`

Five Optuna-tuned scalar weights governing the TC-RBI insertion cost function:

| Field | Default | Role |
|---|---|---|
| `eta` | 0.50 | Screening blend weight (CR vs urgency) |
| `mu_tc` | 1.00 | Weight on incremental target-time conflict ΔTC |
| `mu_late` | 0.25 | Weight on incremental tardiness lower bound ΔLate |
| `mu_count` | 0.75 | Weight on runway-balance deviation Δcount |
| `mu_sep` | 0.05 | Weight on incremental separation burden ΔSep |

Pre-tuned values for all 34 (instance, $m$) pairs are stored in `RBI_PARAM_BANK`.

#### `MRSAParams`

SA control parameters for the multi-runway chain. Key fields:

| Field | Default | Meaning |
|---|---|---|
| `chi0` | 0.80 | Target initial acceptance probability for T₀ calibration |
| `M_stag_frac` | 0.15 | Stagnation threshold as fraction of `N_iter` |
| `beta` | 1.50 | Reheat temperature multiplier |
| `lp_gamma` | 0.05 | LP trigger sensitivity γ |
| `chi_target` | 0.20 | Reactive cooling target acceptance rate χ* |
| `T_min_frac` | 0.01 | Minimum temperature as fraction of initial temperature |
| `B_max` | 3 | Maximum block size for N4/X4 operators |
| `max_reheats` | 3 | Maximum reheats before forced termination |
| `lp_repair_interval` | 100 | Iterations between LP-guided repairs (0 = disabled) |
| `max_ils_restarts` | 2 | ILS warm-restarts from `best_lp` after reheats exhausted |

### `mr_alp/instance.py`

#### `load_instance(path)`

Parses an OR Library text file and returns an `Instance`. Validates the token count against the declared $n$ and zeroes diagonal entries of the separation matrix after loading.

#### `runway_feasible(seq, inst)`

Returns `True` iff the sequence satisfies all pairwise separation constraints and time-window bounds. Uses the Numba JIT kernel `_rwy_feasible_nb` when Numba is available; falls back to pure NumPy otherwise.

#### `surrogate_times(seq, inst)`

Fast consecutive-predecessor surrogate landing times. Used exclusively inside the SA inner loop for proxy computation — never for LP evaluation or verification — because OR Library separation matrices violate the triangle inequality and consecutive propagation is not sufficient for feasibility certification.

### `mr_alp/lp.py`

#### `stage2_lp_objective(sequences, inst)`

Solves the joint Stage-2 LP for fixed per-runway landing sequences. The LP has $3n$ variables ($x_j$, $E_j$, $T_j$) and enforces **all ordered pairs** $(i, j)$ where $i$ precedes $j$ on the same runway:

$$\sum_{\rho=1}^{m} \frac{|\pi_\rho|(|\pi_\rho|-1)}{2}$$

separation constraints — not just $n - m$ consecutive ones. This is required because OR Library separation matrices violate the triangle inequality. Returns `(obj, C_lp, feasible, violations)`.

#### `verify_and_exact_obj(sequences, inst)`

Earliest-time propagation through all per-runway sequences followed by an explicit constraint audit. Returns `(feasible, violations, earliest_obj, C_earliest)`. Used in the verification step of `_run_one_job`.

### `mr_alp/construction.py`

#### `ramp_rbi(inst, m, params)`

TC-RBI (Target-Conflict Regret-Based Insertion) construction heuristic. Inserts aircraft one at a time in decreasing Critical Ratio order, placing each aircraft at the (runway, position) pair that minimises a five-term weighted insertion cost. Returns `(sequences, surrogate_times_list)`.

#### `_build_seed_portfolio(inst, m, params, seed)`

Constructs and LP-evaluates all nine seed heuristics (H1–H9). Returns the best seed LP value, the corresponding sequences, and a `portfolio_timing` dict with per-seed construction and LP evaluation wall times. Timing data is consumed by `ms_mr_sa` to form the unified `job_lp_timeline`.

**Seed heuristics:**

| ID | Rule |
|---|---|
| H1 | FCFS — First-Come First-Served (by release date $r_j$) |
| H2 | EDD — Earliest Due Date (target date $\delta_j$) |
| H3 | WEDD — Weighted EDD (by $h_j / g_j$, broken by $\delta_j$) |
| H4 | ATC — Apparent Tardiness Cost ($K =$ `ATC_K`) |
| H5 | ATCS — ATC with Separation ($K_1 =$ `ATCS_K1`, $K_2 =$ `ATCS_K2`) |
| H6 | CAF — Cost-Adjusted FCFS |
| H7 | MPDS — Multi-Priority Dispatching Sequence (Zhang et al., 2020, Eq. 16) |
| H8 | WCC — Weighted Critical Conflict |
| H9 | GRASP — Randomised Greedy with RCL sizes `GRASP_K_VALUES` (two variants) |

All heuristics produce a priority ordering that is then assigned to runways via the TC-RBI insertion step.

### `mr_alp/operators.py`

Within-runway and cross-runway SA neighbourhood operators. Phase-dependent selection tables (`_OPS_EARLY`, `_OPS_MID`, `_OPS_LATE`, `_OPS_SINGLE`) vary operator weights by iteration fraction $f = t / N_\text{iter}$:

| Phase | $f$ range | Emphasis |
|---|---|---|
| EARLY | $f < 0.30$ | Cross-runway diversification (X1–X4) |
| MID | $0.30 \leq f < 0.75$ | Balanced; X7 (TC-guided repair) activated |
| LATE | $f \geq 0.75$ | Within-runway intensification (N3b, N2) |
| SINGLE | $m = 1$ | Within-runway operators only (N1–N3b) |

**Operator summary:**

| ID | Type | Description |
|---|---|---|
| N1 | Within | Adjacent swap — swap positions $p$ and $p+1$ on runway $\rho$ |
| N2 | Within | Arbitrary swap — swap any two positions on runway $\rho$ |
| N3b | Within | Best re-insertion — remove at $p$, reinsert at cheapest feasible position on same runway |
| N4 | Within | Block relocation — move a consecutive block to a new position on the same runway |
| X1 | Cross | Transfer — move one aircraft to any position on a different runway |
| X2 | Cross | Cross-runway swap — exchange one aircraft from each of two runways |
| X3 | Cross | Best transfer — move one aircraft to its globally best position on any runway |
| X4 | Cross | Block transfer — move a consecutive block to a different runway |
| X7 | Cross | TC-guided repair — target X3 at the highest-impact aircraft |
| XE | Cross | Ejection alias — X3 used in the late-phase table |

### `mr_alp/proxy.py`

#### `compute_proxy(seqs, tc_arr, lbt_arr, sep_arr, inst, params)`

Composite surrogate objective used for SA Metropolis acceptance decisions:

$$\hat{F} = \mu_\text{TC} \cdot \text{TC} + \mu_\text{late} \cdot \text{LBT} + \mu_\text{sep} \cdot \text{Sep}$$

The proxy is never compared against LP values or BKS references. It is a relative guide for the SA acceptance step only.

#### `total_target_conflict(sequences, inst)`

Pairwise target-time conflict sum across all runways:

$$\text{TC} = \sum_{\rho} \sum_{i \prec j \text{ on } \rho} \tfrac{1}{2}(p_i + p_j) \cdot \max\!\bigl(s_{ij} - (\delta_j - \delta_i),\; 0\bigr)$$

GPU-dispatched when `USE_GPU=True` and $n \geq$ `GPU_MIN_N`.

### `mr_alp/sa.py`

#### `run_mr_sa(inst, m, params, p_sa, seed, t_limit, elite_pool)`

Single SA chain with reactive cooling, LP-guided repair triggers, and LP-timeline tracking. Returns `(best_p_seqs, best_proxy, best_lp_seqs, best_lp, best_C_lp, stats)`. The `stats` dict includes `lp_timeline` — a list of `(chain_relative_seconds, lp_val)` pairs recording each LP improvement event. Chain-relative times are converted to job-relative times by `ms_mr_sa`.

#### `optimize_rbi_params(inst, m, n_trials, seed, n_jobs)`

Optuna TPE study over the five `HeuristicParams` fields. For $n > 100$, the proxy objective (total target conflict) is used per trial to avoid expensive LP evaluations. Returns the best `HeuristicParams`.

#### `optimize_sa_params(inst, m, n_trials, seed, n_jobs)`

Optuna TPE study over five `MRSAParams` fields (`chi0`, `M_stag_frac`, `beta`, `lp_gamma`, `chi_target`). Returns the best `MRSAParams`.

### `mr_alp/repair.py`

Post-SA repair operators applied by `lp_vnd_polish` and `path_relink`:

| Function | Description |
|---|---|
| `lp_guided_penalty_repair` | Globally relocate the $q$ highest-penalty aircraft; LP-evaluate the top $K$ proxy-ranked candidates |
| `lp_guided_pair_swap` | Cross-runway swap of aircraft pairs with similar target times |
| `target_conflict_repair` | Deterministic repair for near-zero objectives: minimise residual TC |
| `ejection_chain_transfer` | Depth-$D$ ejection chain starting from the highest-impact aircraft |
| `lns_remove_reinsert` | LNS: remove the top-$k$ aircraft simultaneously, reinsert jointly via TC-RBI |
| `ElitePool` | Fixed-size pool; admits a solution only if runway-Hamming distance from every existing member is ≥ `ELITE_MIN_DIV` |
| `path_relink` | Walk from one elite solution toward another by transferring per-runway differences; LP-evaluate at regular step intervals |
| `lp_vnd_polish` | Monotone VND: cycle through all five repair operators until no improvement or `_vnd_max_rounds(n)` rounds exceeded |

### `mr_alp/solver.py`

#### `ms_mr_sa(inst, m, params, p_sa, n_chains, t_limit, seed)`

Top-level solver. Workflow:

1. **Seed portfolio** (`_build_seed_portfolio`): construct and LP-evaluate all H1–H9 heuristics; capture per-seed timing for the unified `job_lp_timeline`.
2. **SA dispatch**: submit `n_chains` tasks in parallel via `ProcessPoolExecutor` (spawn on Windows/GPU, fork on Linux). Convert chain-relative LP timestamps to job-relative by adding `t_sa_dispatch_offset`.
3. **Elite pool**: collect all chains into `ElitePool` with runway-Hamming diversity guard.
4. **LP-VND polish**: refine the best solution with `lp_vnd_polish`.
5. **Path relinking**: apply `path_relink` between diverse elite pairs; add improving solutions to the pool.
6. **Final LP check**: re-verify and record the reported objective.

Returns `(best_sequences, best_lp, stats_dict)`. The `stats_dict` includes `job_lp_timeline`, `total_t_best` (earliest job-relative time the final best was first achieved), and per-chain summaries.

### `mr_alp/output.py`

#### `save_run_results(results, out_dir)`

Writes four files to `out_dir` (and per-instance subdirectories for incremental saves):

- `summary.csv` — one row per (instance, $m$): objective, gap, wall time, timing breakdown, verification status.
- `schedules.csv` — one row per aircraft: runway assignment, landing position, $r_j$, $\delta_j$, $d_j$, $x_j$, earliness, tardiness, per-aircraft penalty.
- `alternatives.csv` — near-optimal sequence variants found across chains.
- `verification.txt` — per-instance constraint audit reports.
- `run_metadata.json` — hostname, Python version, CPU count, full solver configuration snapshot, per-job pass/fail.

#### `generate_plots(results, out_dir)`

Seven plot families written to `out_dir/plots/`:

| Directory | Content |
|---|---|
| `gap/` | `gap_summary.png` — bar chart of % gap to BKS, coloured by new-BKS / matched / missed |
| `convergence/` | Per-(inst,$m$) SA chain LP objective vs wall time |
| `lp_timeline/` | Unified job LP improvement timeline across seeds, SA chains, VND, and path relinking |
| `time_to_best/` | Scatter: time-to-best vs instance size, annotated by improvement phase |
| `elite_pool/` | Elite pool runway-Hamming diversity matrix per job |
| `gantt/` | Per-(inst,$m$) Gantt chart with time windows $[r_j, d_j]$, target times $\delta_j$, and scheduled landings $x_j$ |
| `seeds/` | Per-(inst,$m$) three-stage improvement: raw seed proxy → best seed LP → SA+VND+PR final |

---

## 7. Algorithms

### 7.1 TC-RBI Construction Heuristic

TC-RBI (Target-Conflict Regret-Based Insertion) constructs an initial $m$-runway assignment and sequence by inserting aircraft in decreasing order of Critical Ratio:

$$\text{CR}_j = \frac{g_j + h_j}{\max(d_j - r_j - \bar{s}_j,\; \varepsilon)}$$

where $\bar{s}_j$ is the mean bilateral separation of aircraft $j$ with all others. For each aircraft, all (runway, position) combinations are enumerated. The five-term insertion cost is:

$$\text{cost}(j, \rho, p) = \mu_\text{TC} \cdot \Delta\text{TC} + \mu_\text{late} \cdot \Delta\text{LBT} + \mu_\text{count} \cdot \Delta\text{count} + \mu_\text{sep} \cdot \Delta\text{Sep}$$

The aircraft is placed at the minimum-cost feasible (runway, position) pair; ties are broken by runway balance. The five $\mu$ weights are resolved from `RBI_PARAM_BANK` (34 pre-tuned entries covering all OR Library instances and configured runway counts), falling back to Optuna TPE tuning on-the-fly for uncovered configurations.

### 7.2 Stage-2 LP

Once per-runway sequences $\boldsymbol{\pi}$ are fixed, the joint Stage-2 LP determines optimal landing times:

**Variables:** $x_j \in [r_j, d_j]$, $E_j \geq 0$, $T_j \geq 0$.

**Constraints:**
- $C_j - E_j \geq \delta_j$ and $C_j + T_j \geq \delta_j$ (earliness/tardiness linearisation).
- $x_j - x_i \geq s_{ij}$ for all ordered pairs $i \prec j$ on the same runway — all $\sum_\rho \binom{|\pi_\rho|}{2}$ pairs, not just consecutive ones. This is required because OR Library separation matrices violate the triangle inequality.

The LP has $3n$ variables and is solved by HiGHS via `scipy.optimize.linprog`.

### 7.3 Proxy Objective

The proxy $\hat{F}$ is used exclusively for SA Metropolis acceptance decisions. It combines:
- **TC** (total target-conflict): aggregate time-window pressure created by separation requirements relative to target-time differences across all runways.
- **LBT** (lower-bound tardiness): surrogate tardiness computed from `surrogate_times` (consecutive-predecessor approximation).
- **Sep** (separation burden): aggregate separation load of the current assignment.

The proxy cannot be compared against LP values or BKS references; it is a relative guide only.

### 7.4 SA Chain (`run_mr_sa`)

Each SA chain applies the phase-dependent operator table to generate a candidate pool of $R$ neighbours per step, selects the best by proxy, and accepts with Metropolis probability $\exp(-\Delta\hat{F} / T)$.

**Reactive cooling.** After every temperature level, the observed acceptance rate $\chi$ is computed and the cooling rate adjusted:

$$\alpha \;\leftarrow\; \operatorname{clip}\!\left(\alpha + \operatorname{sign}(\chi - \chi^*) \times \alpha_\text{step},\;\; \alpha_\text{lo},\;\; \alpha_\text{hi}\right)$$

**LP trigger.** An LP call is made when the proxy improves by more than $\gamma \cdot \hat{F}_\text{best}$ (governed by `lp_gamma`). This avoids LP evaluations on every move while still tracking objective progress via the `lp_timeline`.

**Stagnation and reheating.** When `M_stag_frac × N_iter` consecutive non-improving steps occur, the chain reheats by $\beta \times T$ and resets to `best_lp_seqs`. Up to `max_reheats` reheats are permitted; after exhaustion, up to `max_ils_restarts` ILS warm-restarts from the current LP-best are attempted.

**Candidate pool size $R$** scales with instance size:

| $n$ | $R$ |
|---|---|
| $\leq 100$ | 10 |
| $\leq 250$ | 20 |
| $> 250$ | 30 |

### 7.5 Elite Pool and Path Relinking

After all SA chains complete, every chain's best LP solution is offered to an `ElitePool` (capacity `ELITE_POOL_MAX`). A solution is admitted only if its runway-Hamming distance from every existing pool member is ≥ `ELITE_MIN_DIV`, enforcing structural diversity. Path relinking then walks from one elite solution toward another by sequentially applying the per-runway sequence differences, evaluating the LP at regular step intervals and retaining improving intermediate solutions.

### 7.6 LP-VND Polish

`lp_vnd_polish` applies the five repair operators (`lp_guided_penalty_repair`, `lp_guided_pair_swap`, `target_conflict_repair`, `ejection_chain_transfer`, `lns_remove_reinsert`) in round-robin order, accepting only LP improvements, until no operator improves the solution or the round cap is exhausted:

$$\text{max rounds} = \begin{cases} 15 & n \leq 100 \\ 10 & n \leq 250 \\ 5 & \text{otherwise} \end{cases}$$

### 7.7 Job Timeline and Time-to-Best

All LP improvement events — from seed constructions, SA chains, VND polish, and path relinking — are recorded on a single unified `job_lp_timeline` with job-relative timestamps (wall seconds from the start of `ms_mr_sa`). `total_t_best` is the earliest job-relative time at which the final best LP value was first achieved, written to `summary.csv` as `t_best_lp_s`.

### 7.8 Optuna Parameter Tuning

When (inst, $m$) is absent from the pre-tuned banks, two separate Optuna TPE studies tune:

1. **RBI params** (`optimize_rbi_params`): 5 `HeuristicParams` weights; $n_\text{trials}$ scales with $n$. Objective: LP value of the TC-RBI construction (proxy TC for $n > 100$).
2. **SA params** (`optimize_sa_params`): 5 `MRSAParams` fields; objective: best LP value after a short SA run.

Pre-tuned values cover all 34 (inst, $m$) combinations in `INSTANCE_RUNWAYS`. Set `RUN_RBI_OPTUNA = False` and `RUN_SA_OPTUNA = False` to skip tuning entirely and use defaults for any uncovered configuration.

---

## 8. Configuration Reference

Edit `mr_alp/config.py §0` before running.

**Runway configurations per instance:**

```python
INSTANCE_RUNWAYS: Dict[str, List[int]] = {
    "airland1":  [2, 3],        "airland2":  [2, 3],
    "airland9":  [2, 3, 4],     "airland10": [2, 3, 4, 5],
    ...
}
```

Add or remove entries to evaluate different runway counts.

**Parallelism:**

```python
USE_ALL_SEEDS = False   # True: all H1–H9 seeds → SA starting points
N_WORKERS     = 7       # concurrent SA processes
N_CHAINS      = 4       # SA starting points when USE_ALL_SEEDS=False
```

**Time budgets:**

```python
T_LIMIT     = 600.0     # default per-job budget (seconds)
MAX_T_LIMIT = 1200.0    # ceiling for large/high-gap instances
```

**Optuna:**

```python
RUN_RBI_OPTUNA    = True
N_RBI_TRIALS_BASE = 30   # base trial count; scaled by n
RUN_SA_OPTUNA     = True
SA_N_TRIALS_BASE  = 20
```

Set both `RUN_*_OPTUNA` to `False` to skip Optuna entirely and use `RBI_PARAM_BANK` / `SA_PARAM_BANK` defaults.

**GPU:**

```python
USE_GPU   = True    # automatic CPU fallback if no CUDA device
GPU_MIN_N = 200     # activate GPU only for n ≥ GPU_MIN_N
```

---

## 9. Running the Pipeline

**Full benchmark (all instances in `./data/`):**

```bash
python mr_sa_alp.py
```

Set `BATCH_MODE = True` and `FOLDER = "data/"` in `mr_alp/config.py`. The solver auto-discovers all `airland*.txt` files, runs all configured (instance, $m$) jobs, and writes outputs to `MR_results/`.

**Single instance:**

Set `BATCH_MODE = False` and `INSTANCE_PATH = "data/airland8.txt"` in `config.py`, then:

```bash
python mr_sa_alp.py
```

Or programmatically:

```python
from mr_alp import load_instance, ms_mr_sa
from mr_alp.models import HeuristicParams, RBI_PARAM_BANK

inst   = load_instance("data/airland8.txt")
m      = 2
params = RBI_PARAM_BANK.get(("airland8", m), HeuristicParams())
seqs, obj, stats = ms_mr_sa(inst, m, params)
print(f"LP objective (m={m}): {obj:.2f}")
```

**Standalone TC-RBI (construction only):**

```bash
python ramp_rbi.py
```

Runs TC-RBI alone without SA refinement, printing the construction LP value and gap for all configured instances. Useful for evaluating construction quality in isolation.

**Disabling Optuna at runtime:**

```python
# mr_alp/config.py
RUN_RBI_OPTUNA = False
RUN_SA_OPTUNA  = False
```

All 34 (inst, $m$) pairs in `INSTANCE_RUNWAYS` have pre-tuned entries in `RBI_PARAM_BANK`; uncovered configurations fall back to `HeuristicParams()` defaults.

**Worker count.** `N_WORKERS` in `config.py` controls concurrent SA processes. For single-process debugging, set `n_chains=1` when calling `ms_mr_sa` directly.

---

## 10. Outputs

### `MR_results/summary.csv`

One row per (instance, $m$):

| Column | Description |
|---|---|
| `instance` | OR Library instance name |
| `m` | Runway count |
| `lp_obj` | Best LP objective |
| `bks` | BKS reference value |
| `gap_pct` | % gap to BKS (`★` suffix if new BKS) |
| `wall_s` | Total wall time (seconds) |
| `t_seed_construct_s` | Wall time for all seed constructions |
| `t_seed_lp_eval_s` | Wall time for all seed LP evaluations |
| `t_best_seed_lp_s` | Job-relative time when the best seed LP was achieved |
| `t_sa_start_s` | Job-relative time when SA chains were dispatched |
| `t_best_lp_s` | Job-relative total time-to-best |
| `verified` | `PASS` / `FAIL` / `INFEASIBLE` |

### `MR_results/schedules.csv`

One row per aircraft per (instance, $m$): runway assignment, landing position, $r_j$, $\delta_j$, $d_j$, $x_j$, earliness, tardiness, per-aircraft penalty, $g_j$, $h_j$.

### `MR_results/alternatives.csv`

Near-optimal sequence variants found across chains, with LP objectives and runway-Hamming distances from the reported best solution.

### `MR_results/verification.txt`

Constraint audit reports for every (instance, $m$) job. Each report lists pass/fail per constraint group with per-violation detail.

### `MR_results/run_metadata.json`

Timestamp, hostname, Python version, CPU count, full solver configuration snapshot, and per-job pass/fail status.

---

## 11. References

- Beasley, J. E., Krishnamoorthy, M., Sharaiha, Y. M., & Abramson, D. (2000). Scheduling aircraft landings — the static case. *Transportation Science*, 34(2), 180–197.
- Zhang, J., Zhao, P., Yang, C., & Hu, R. (2020). A new meta-heuristic approach for the aircraft landing problem. *Transactions of Nanjing University of Aeronautics and Astronautics*, 37(2), 197–208.
- Pinedo, M. L. (2016). *Scheduling: Theory, Algorithms, and Systems* (5th ed.). Springer.
- Glover, F. (1996). Tabu search and adaptive memory programming — Advances, applications and challenges. In *Interfaces in Computer Science and Operations Research*. Kluwer.
- Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by simulated annealing. *Science*, 220(4598), 671–680.
