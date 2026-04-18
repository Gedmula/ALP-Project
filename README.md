# Aircraft Landing Problem — Solver Pipeline

Single-runway scheduling under wake-vortex separation constraints, formulated as 

$$1 | r_j, s_{jk}, δ̄_j | Σ(g_j E_j + h_j T_j)$$ 

and solved via a two-stage decomposition: sequence optimisation by parallel multi-start SA with ILS, followed by exact timing optimisation via a linear program.

---

## Table of Contents

1. [Problem Formulation](#1-problem-formulation)
2. [Repository Layout](#2-repository-layout)
3. [Dependencies](#3-dependencies)
4. [Data](#4-data)
5. [Architecture Overview](#5-architecture-overview)
6. [Module Reference](#6-module-reference)
7. [Algorithms](#7-algorithms)
8. [SAParams Reference](#8-saparams-reference)
9. [Running the Pipeline](#9-running-the-pipeline)
10. [Outputs](#10-outputs)
11. [Verification System](#11-verification-system)
12. [Design Decisions and Known Trade-offs](#12-design-decisions-and-known-trade-offs)
13. [References](#13-references)

---

## 1. Problem Formulation

Given a set $J = \{1, \ldots, n\}$ of aircraft, each characterised by:

| Symbol | Meaning |
|---|---|
| $r_j$ | Release (earliest landing) time |
| $\delta_j$ | Target (preferred) landing time |
| $d_j$ | Deadline (latest landing) time |
| $s_{jk}$ | Minimum separation time required between $j$ landing before $k$ |
| $g_j$ | Cost per unit of earliness (landing before $\delta_j$) |
| $h_j$ | Cost per unit of tardiness (landing after $\delta_j$) |

The objective is to find a landing sequence $\pi$ and scheduled landing times $x_j$ such that:

$$x_j \in [r_j,\; d_j] \qquad \forall j \in J$$

$$x_{\pi(l+1)} \geq x_{\pi(l)} + s_{\pi(l),\,\pi(l+1)} \qquad l = 1, \ldots, n-1$$

$$\min \sum_{j \in J} \bigl(g_j \cdot \max(\delta_j - x_j,\; 0) + h_j \cdot \max(x_j - \delta_j,\; 0)\bigr)$$

The MILP is NP-hard in general. The pipeline exploits the two-stage structure: once a sequence $\pi$ is fixed, all binary sequencing variables are determined and the timing subproblem reduces to an LP solved by HiGHS.

---

## 2. Repository Layout

```
project_root/
├── Single_runway_SA.py      # Main solver — MS-SA + ILS + Stage-2 LP
├── doe_alp.py               # Design of Experiments module (imports from above)
├── data/
│   ├── airland1.txt         # OR Library benchmark instances
│   ├── airland2.txt
│   └── ...                  # airland1–airland13
├── plots/
│   ├── gantt/               # Gantt charts
│   ├── convergence/         # SA chain convergence curves
│   ├── alpha trajectory/    # Reactive cooling-rate trajectories
│   ├── seed improvement/    # Heuristic seed vs SA final objective
│   ├── penalty profile/     # Per-aircraft penalty profiles
│   ├── gap_summary.png      # Aggregate gap bar chart
│   └── alt_solutions.png    # Alternate-optimal sequence counts
├── results/                 # Auto-generated CSV/JSON/TXT exports
│   ├── summary.csv
│   ├── schedules.csv
│   ├── verification.txt
│   └── run_metadata.json
└── doe_results/             # DOE outputs (written by doe_alp.py)
    ├── exp1_heuristic/
    ├── exp2_ils_depth/
    └── exp3_parameter/
```

OR Library files are **not** included in the repository. See [Section 4](#4-data) for download instructions.

---

## 3. Dependencies

Python 3.9 or later is required. All core dependencies are available via pip.

**Required:**

```
numpy
scipy          # linprog / HiGHS interface
matplotlib
```

**Optional but recommended:**

```
tqdm           # progress bars for SA chains
optuna         # hyperparameter tuning (disable with ENABLE_OPTUNA = False)
torch          # GPU-accelerated components (CPU fallback is automatic)
```

Install everything at once:

```bash
pip install numpy scipy matplotlib tqdm optuna torch
```

The LP solver is HiGHS, accessed through `scipy.optimize.linprog` with `method='highs'`. No separate HiGHS installation is required when using SciPy ≥ 1.9.

---

## 4. Data

OR Library ALP instances were originally published by Beasley et al. (2000). They are freely available at:

```
http://people.brunel.ac.uk/~mastjjb/jeb/orlib/airlandinfo.html
```

Download `airland1.txt` through `airland13.txt` and place them in `./data/` relative to `Single_runway_SA.py`. The loader detects whichever files are present and skips missing ones. The actual instance size $n$ for each file is printed by `diagnose_instance` at runtime; the values below are taken from Beasley et al. (2000).

**Known optimal values** — CPLEX reference from Zhang et al. (2020); airland8 certified by Beasley et al. (2000) branch-and-bound:

| Instance | Known optimum | Source |
|---|---|---|
| airland1 | 700 | Zhang et al. (2020) |
| airland2 | 1480 | Zhang et al. (2020) |
| airland3 | 820 | Zhang et al. (2020) |
| airland4 | 2520 | Zhang et al. (2020) |
| airland5 | 3100 | Zhang et al. (2020) |
| airland6 | 24442 | Zhang et al. (2020) |
| airland7 | 1550 | Zhang et al. (2020) |
| airland8 | 1950 | Beasley et al. (2000) — certified global optimum |
| airland9 | 5611.70 | Zhang et al. (2020) |
| airland10 | 12640.42 | Zhang et al. (2020) |
| airland11 | 12462.18 | Zhang et al. (2020) |
| airland12 | 16629.10 | Zhang et al. (2020) |
| airland13 | 39287.52 | Zhang et al. (2020) |

> **Note on airland8.** The value 1950 is a certified global optimum established by branch-and-bound in Beasley et al. (2000), not a heuristic bound. Any solver result below 1950 indicates a constraint-enforcement error, not a new best.

**File format.** Each file begins with $n$ and a freeze-time token (unused), followed by $n$ blocks of the form `appear_j r_j δ_j d_j g_j h_j s[j][0] … s[j][n−1]`. The loader validates the total token count before parsing.

---

## 5. Architecture Overview

```
Single_runway_SA.py
│
├── ALPInstance                  Data container + MPDS parameter computation
├── load_orlib / synthetic_instance    Instance I/O
│
├── Dual-track feasibility
│     is_feasible_old            O(n)  consecutive-only check  (SA guidance)
│     is_feasible                O(n²) full pairwise check     (reporting + verification)
│
├── Stage-2 LP
│     _build_lp_matrices_old     Consecutive-only LP (n−1 separation rows) — semi path
│     _build_lp_matrices         Full pairwise LP (n(n−1)/2 rows)          — feas path
│     solve_stage2 / solve_lp    Full pairwise LP via HiGHS
│     _lp_recompute              Independent full pairwise re-solve for C9 verification
│
├── Evaluation
│     evaluate_semi              O(n) check + consecutive LP   (SA acceptance decisions)
│     evaluate                   O(n²) check + full pairwise LP (incumbent reporting)
│     batch_evaluate             Parallel evaluate across sequence list
│
├── Initial solution generators
│     gen_erd, gen_edd, gen_mdd, gen_atc, gen_mpds
│
├── Neighbourhood operators
│     op_swap, op_insert, op_reverse, op_or_opt_2, op_or_opt_3
│     _double_bridge              4-opt ILS perturbation (A|B|C|D → A|C|B|D)
│
├── calibrate_T0                 Empirical T_0 calibration (Eq. 30)
├── run_sa                       Single SA chain with reactive α adaptation
├── run_ils                      ILS wrapper (SA + double-bridge perturbation)
├── ms_sa                        Parallel multi-start SA (ProcessPoolExecutor)
│
├── adaptive_params              Size-scaled default SAParams
├── tune_sa                      Optuna TPE hyperparameter search
│
├── verify_schedule              Nine-group constraint audit (C1–C9)
├── verify_all                   Convenience wrapper + formatted report
│
├── export_results               summary.csv, schedules.csv, verification.txt, metadata.json
├── run_experiment               End-to-end benchmark runner for one instance
│
└── Visualisation
      plot_gantt
      plot_sa_convergence
      plot_alpha_trajectory
      plot_seed_improvement
      plot_penalty_profile
      plot_gap_summary
      plot_alt_solutions
```

---

## 6. Module Reference

### `ALPInstance`

Dataclass holding all instance data. Constructed by `load_orlib` or `synthetic_instance`. The field $\bar{s}$ (`s_bar`) — the mean off-diagonal separation, computed in `__post_init__` — is used throughout as a normalisation constant in MPDS priority index computation and ATC lookahead scaling. Do not modify `s_bar` manually.

### `load_orlib(path, name="")`

Parses an OR Library text file and returns an `ALPInstance`. Raises `FileNotFoundError` if the path does not exist and `ValueError` if the token count is inconsistent with the declared $n$. Issues a warning if the EDD sequence is infeasible on the loaded instance.

### `synthetic_instance(n, seed)`

Generates a random feasible instance of size $n$ using ICAO wake-vortex separation categories. The separation matrix is constructed from three aircraft-type classes with realistic Category-A/B/C values. Useful for quick tests without OR Library files.

### `is_feasible_old(seq, inst)`

$O(n)$ consecutive-separation pre-filter. Propagates the earliest feasible landing time forward through the sequence, checking only adjacent pairs. Used exclusively by `evaluate_semi` to gate SA acceptance decisions cheaply. May admit sequences that violate non-adjacent separation constraints when the wake-vortex matrix does not satisfy the triangle inequality.

### `is_feasible(seq, inst)`

$O(n^2)$ full pairwise feasibility check. For each position $m$ in the sequence, the earliest feasible landing time accounts for the release date **and** the separation required from every predecessor $l < m$. This is the correct check for OR Library instances, where the wake-vortex separation matrix routinely violates the triangle inequality (Beasley et al., 2000, §2.2). Used by `evaluate` for incumbent reporting, gap tables, and verification.

### `evaluate_semi(seq, inst)`

Semi-feasible evaluation path: gates with `is_feasible_old`, then solves the consecutive-only LP (`_build_lp_matrices_old`). Returns $\infty$ if the sequence fails the relaxed check. Used inside SA inner loops for acceptance decisions; accepts a larger effective neighbourhood than `evaluate` but may overestimate feasibility.

### `evaluate(seq, inst)`

Fully-feasible evaluation path: gates with `is_feasible`, then solves the full pairwise LP (`_build_lp_matrices`). Returns $\infty$ if either check fails. Used to update the fully-feasible incumbent whenever the semi-feasible incumbent improves, and for all gap-table reporting and verification.

### `solve_stage2(seq, inst)`

Solves the full pairwise Stage-2 LP for the given sequence. Returns `(objective, landing_times)` or $(\infty, \texttt{None})$ if HiGHS reports infeasibility or numerical failure.

### `run_sa(seq0, inst, p, seed, T0, t_deadline)`

Single SA chain with dual-track feasibility and reactive temperature adaptation. `t_deadline` is an absolute `time.perf_counter()` timestamp; the outer loop exits cleanly when this is reached. Returns `(pb_semi, fb_semi, stats_dict)`. `stats_dict` contains `obj` (semi), `obj_feas` (fully-feasible), `pi_feas`, `time`, `t_best`, `t_best_feas`, `history`, `init_obj`, `n_alt_seqs`, and `alpha_history`. See Section 7.2 for the dual-track feasibility contract.

### `run_ils(seq0, inst, p, n_restarts, seed, t_deadline)`

ILS wrapper around `run_sa`. After each SA run, applies `_double_bridge` to the semi-feasible incumbent to escape the current local-optimum basin, then re-runs SA from the perturbed sequence. Both semi-feasible and fully-feasible bests are tracked independently across all restarts. `t_deadline` is forwarded to every inner `run_sa` call and also checked between restarts.

### `ms_sa(inst, p, n_workers, n_ils, t_limit)`

Launches one ILS/SA chain per worker in parallel. Starting sequences are drawn from the six named heuristics plus double-bridge variants if `n_workers` exceeds the number of named seeds. The reported solution is the best **fully-feasible** result across all chains; if no chain finds a fully-feasible solution, the semi-feasible best is returned as a fallback with a warning. `t_limit` (default 3600 s) is converted to an absolute deadline shared by all chains.

### `adaptive_params(n)`

Returns `(SAParams, n_ils_restarts)` scaled to instance size. Small instances ($n \leq 20$) use tight parameters with no ILS; large instances ($n > 150$) use slow cooling and up to six ILS restarts per chain.

### `tune_sa(inst, known_opt, n_trials, n_workers)`

Optuna TPE study over the six `SAParams` fields using gap to known optimum as the objective. Disabled by default (`ENABLE_OPTUNA = False` in `__main__`).

### `verify_schedule(seq, inst, tol)`

Full nine-group constraint audit (C1–C9). Never short-circuits — all groups are checked regardless of earlier failures. The C9 re-solve uses `_lp_recompute`, which builds an independent full-pairwise LP matrix with tighter solver tolerances ($10^{-9}$), making it a genuine cross-check rather than a cache hit. Returns `(passed, objective, VerificationReport)`.

### `export_results(results_list, out_dir)`

Writes `summary.csv`, `schedules.csv`, `verification.txt`, and `run_metadata.json` to `out_dir`. Called per-instance inside `run_experiment` for incremental saves.

### `run_experiment(inst, known_opt, sa_p, n_workers, t_limit)`

End-to-end runner for a single instance: runs MS-SA, verifies the fully-feasible result, exports outputs, and generates all visualisation plots. If verification fails, the objective is set to $\infty$ in all exports. Returns a result dict suitable for aggregate reporting.

### `diagnose_instance(inst)`

Prints instance statistics ($n$, separation range, window range), evaluates all four deterministic heuristics, and estimates the random-sequence feasibility rate from 500 trials. Useful for characterising instance difficulty before a full run.

---

## 7. Algorithms

### 7.1 Dual-Track Feasibility

The solver maintains two parallel feasibility tracks that serve distinct purposes.

**Semi-feasible track** (SA guidance): Uses `is_feasible_old` ($O(n)$ consecutive propagation) and the consecutive-only LP (`_build_lp_matrices_old`, $n-1$ separation rows). This is the path taken on every SA acceptance decision. It is fast, admits a broader effective neighbourhood, and is consistent with the LP formulation used for those decisions. Semi-feasible incumbents drive temperature calibration, acceptance, stagnation counters, reheating, and ILS perturbation seeds.

**Fully-feasible track** (reporting): Uses `is_feasible` ($O(n^2)$ pairwise propagation) and the full pairwise LP (`_build_lp_matrices`, $\frac{n(n-1)}{2}$ separation rows). This is called once per strict improvement in the semi-feasible incumbent. Only fully-feasible incumbents are reported in gap tables, written to export files, and subjected to `verify_schedule`. If a chain produces no fully-feasible solution, its result is excluded from gap reporting and a warning is issued.

**Why the distinction is necessary.** OR Library wake-vortex separation matrices routinely violate the triangle inequality: for some triples $(i, j, k)$ it holds that $s_{ik} > s_{ij} + s_{jk}$, meaning a non-consecutive pair can impose a binding separation that consecutive propagation would miss. Enforcing only consecutive constraints in both the feasibility check and the LP causes silently infeasible schedules to pass all verification checks, as was the root cause of the airland8 objective-1860 anomaly discovered during development.

### 7.2 Stage-2 LP

Once a landing sequence $\pi$ is fixed, the Stage-2 LP determines optimal landing times and computes the weighted earliness-tardiness cost. The LP used in `_build_lp_matrices` (fully-feasible path) enforces separation between **all ordered pairs** in the sequence:

$$x_{\pi(m)} \geq x_{\pi(l)} + s_{\pi(l),\,\pi(m)} \qquad \forall\; l < m$$

This yields $\frac{n(n-1)}{2}$ separation constraints. The consecutive-only formulation in Zhang et al. (2020), Eq. 3 — which reduces this to $n-1$ constraints — is valid only when the separation matrix satisfies the triangle inequality globally, an assumption that empirically fails for all OR Library instances.

The LP has $3n$ variables $(x_j, E_j, T_j)$ and is solved by HiGHS via `scipy.optimize.linprog`.

### 7.3 Reactive SA (`run_sa`)

The SA chain adapts its cooling rate after every temperature level based on the observed acceptance rate $\chi$, targeting $\chi^* = 0.20$:

$$\alpha \;\leftarrow\; \operatorname{clip}\!\left(\alpha + \operatorname{sign}(\chi - \chi^*) \times 0.005,\;\; 0.80,\;\; 0.999\right)$$

`p.alpha` sets the initial cooling rate; the chain drifts away from it organically. When `M_stag` consecutive levels yield no improvement, the chain reheats:

$$T \leftarrow 2.0 \times T, \qquad \pi_\text{curr} \leftarrow \pi_\text{best\_semi}$$

Up to two reheats are permitted per chain. After the second, the chain exits. A hard per-instance wall-clock limit (`t_deadline`) is also checked at the top of every outer iteration and between ILS restarts.

### 7.4 Multi-Start Strategy (`ms_sa`)

Six deterministic dispatching rules generate distinct initial sequences:

| Rule | Ordering principle |
|---|---|
| ERD | Earliest release date (arrival order) |
| EDD | Earliest target time |
| MDD | Modified Due Date — minimises $\max(\delta_j,\; t + s_{kj})$ |
| ATC ($K=2$) | Apparent Tardiness Cost with lookahead factor $K=2$ |
| ATC ($K=4$) | ATC with wider lookahead |
| MPDS | Multi-Priority Dispatching Sequence (Zhang et al., 2020, Eq. 16) |

If `n_workers` exceeds six, additional starts are generated by applying `_double_bridge` to the named-seed sequences. All chains run in parallel via `ProcessPoolExecutor` using fork-based multiprocessing on Linux/macOS and spawn on Windows.

### 7.5 Iterated Local Search (`run_ils`)

Each SA chain optionally runs an ILS wrapper controlled by the `n_ils` parameter from `adaptive_params`. After an initial SA run, the double-bridge (4-opt) operator perturbs the semi-feasible incumbent:

$$A \mid B \mid C \mid D \;\longrightarrow\; A \mid C \mid B \mid D$$

This move is unreachable by any 3-opt sequence, ensuring the perturbation genuinely escapes the current local-optimum basin (Applegate et al., 2006, §4.5). For instances with $n > 30$, random sequences are nearly always infeasible; double-bridge perturbation from an incumbent is the only practical mechanism for chain restart.

### 7.6 Initial Solution Generators

All six rules greedily select aircraft one at a time, tracking the current time $t$ and the last-scheduled aircraft $k$. The MPDS priority index (Zhang et al., 2020, Eq. 16) balances four exponential terms:

$$I_\text{MPDS}(t,k)_j = e^{-\text{slack}/K_1} \cdot e^{-s_{kj}/(K_2\bar{s})} \cdot e^{-r\text{wait}/K_3} \cdot e^{-\text{pen}/K_4}$$

where the four terms correspond to minimum-slack priority, shortest separation preference, earliest release date priority, and minimum-penalty priority incorporating both $g_j$ and $h_j$. The scaling parameters $K_1$–$K_4$ are computed analytically from instance statistics per Zhang et al. (2020), Eqs. 19–22.

---

## 8. SAParams Reference

`SAParams` is a dataclass with the following fields. `run_sa` adapts `alpha` reactively during execution; all other fields are fixed for the duration of the chain.

| Field | Default | Meaning |
|---|---|---|
| `alpha` | 0.99 | Initial geometric cooling rate $\alpha \in (0, 1)$ |
| `N_iter` | 120 | Neighbour evaluations per temperature level |
| `T_min` | 1e-4 | Temperature floor; chain exits when $T < T_\text{min}$ |
| `I_max` | 600 | Hard cap on outer (temperature-level) iterations |
| `M_stag` | 60 | Stagnation threshold: trigger reheat after this many non-improving levels |
| `chi0` | 0.50 | Target initial acceptance probability used by `calibrate_T0` |

`adaptive_params(n)` returns recommended values per instance-size tier:

| $n$ | `alpha` | `N_iter` | `T_min` | `I_max` | `M_stag` | `n_ils` |
|---|---|---|---|---|---|---|
| $\leq 20$ | 0.97 | 80 | $10^{-4}$ | 300 | 50 | 0 |
| $\leq 50$ | 0.98 | 150 | $10^{-4}$ | 600 | 80 | 2 |
| $\leq 150$ | 0.995 | 300 | $10^{-5}$ | 1200 | 120 | 4 |
| $> 150$ | 0.997 | 500 | $10^{-6}$ | 2000 | 200 | 6 |

---

## 9. Running the Pipeline

**Full benchmark (all OR Library instances found in `./data/`):**

```bash
python Single_runway_SA.py
```

The script auto-discovers all `airland*.txt` files, runs `diagnose_instance` followed by `run_experiment` on each, prints per-instance and aggregate gap tables, and exports consolidated results. Each instance is allocated a 3600-second wall-clock budget.

**Single instance:**

```python
from Single_runway_SA import load_orlib, run_experiment

inst = load_orlib("data/airland2.txt", "airland2")
result = run_experiment(inst, known_opt=1480)
```

**Synthetic demo (no data files required):**

If no OR Library files are found, the pipeline falls back to a 20-aircraft synthetic instance automatically.

**Optuna hyperparameter tuning:**

Set `ENABLE_OPTUNA = True` in `__main__`. Tuning runs on the first discovered instance for `n_trials=40` trials. The tuned parameters replace `SA_full` for all subsequent runs.

**Custom parameters:**

```python
from Single_runway_SA import SAParams, ms_sa, load_orlib

inst = load_orlib("data/airland8.txt", "airland8")
p = SAParams(alpha=0.98, N_iter=200, T_min=1e-4, I_max=1000, M_stag=80)
best_seq, best_obj, stats = ms_sa(inst, p, n_workers=16, n_ils=3)
```

**Worker count.** `N_CPU` defaults to `max(os.cpu_count() − 4, 1)`. Override by passing `n_workers` explicitly to `ms_sa` or `run_experiment`.

**Design of Experiments:**

```bash
python doe_alp.py            # all three experiments
python doe_alp.py --exp 1    # heuristic seeding study only
python doe_alp.py --exp 2 3  # ILS depth + parameter factorial
```

`doe_alp.py` imports directly from `Single_runway_SA.py` and writes all outputs to `./doe_results/`. See the module docstring in `doe_alp.py` for full experiment design details.

---

## 10. Outputs

### `plots/`

| File | Content |
|---|---|
| `gantt/gantt_{instance}_{method}.png` | Gantt chart with time windows $[r_j, d_j]$, target times $\delta_j$ (□), scheduled landing times $x_j$ (○), and wake-vortex separation zones (orange shading) |
| `convergence/convergence_{instance}.png` | Best semi-feasible objective history per SA chain on log scale |
| `alpha trajectory/alpha_trajectory_{instance}.png` | Reactive cooling-rate $\alpha$ per outer iteration per chain |
| `seed improvement/seed_improvement_{instance}.png` | Paired bars: heuristic seed objective vs SA final objective per chain |
| `penalty profile/penalty_profile_{instance}_{method}.png` | Per-aircraft $g_j E_j$ and $h_j T_j$ bars with deviation scatter |
| `gap_summary.png` | Bar chart of percentage gap to known optimum across all instances |
| `alt_solutions.png` | Distinct near-optimal sequence counts and heuristic seeds at optimum, per instance |

### `results/`

| File | Content |
|---|---|
| `summary.csv` | One row per instance: fully-feasible objective, gap, wall time, time-to-best, alternate sequence counts, verification status |
| `schedules.csv` | One row per aircraft per instance: landing position, $r_j$, $\delta_j$, $d_j$, $x_j$, earliness, tardiness, per-aircraft penalty, $g_j$, $h_j$ |
| `verification.txt` | Full `VerificationReport.summary()` for every instance, with per-constraint violation detail |
| `run_metadata.json` | Timestamp, hostname, Python version, CPU count, per-instance pass/fail |

Per-instance subdirectories (`results/{instance_name}/`) are written incrementally during the benchmark loop so partial results are preserved if a run is interrupted.

---

## 11. Verification System

Every solution is audited against nine constraint groups before the objective is accepted. Verification is applied to the **fully-feasible** sequence only; semi-feasible results are never reported as verified.

| Group | Constraint | Rows |
|---|---|---|
| C1 | $x_j \geq r_j$ | $n$ |
| C2 | $x_j \leq d_j$ | $n$ |
| C3 | $x_{\pi(m)} \geq x_{\pi(l)} + s_{\pi(l),\pi(m)} \;\; \forall\, l < m$ | $\frac{n(n-1)}{2}$ |
| C4 | Permutation validity: distinct indices in $[0, n-1]$ | $n$ |
| C5 | Greedy-pass consistency: $O(n)$ forward pass feasibility | $n-1$ |
| C6 | $E_j \geq 0$ | $n$ |
| C7 | $T_j \geq 0$ | $n$ |
| C8 | $\text{LP obj} \approx \sum_j (g_j E_j + h_j T_j)$ | 1 |
| C9 | Independent LP re-solve via `_lp_recompute` (tolerances $10^{-9}$, fresh matrix) | 1 |

The audit never short-circuits — all groups are evaluated regardless of earlier failures. C3 enforces all $\frac{n(n-1)}{2}$ pairwise separation constraints, not just the $n-1$ consecutive ones. This is required because OR Library wake-vortex matrices violate the triangle inequality: a gap that is satisfied for consecutive pairs may still be violated between non-adjacent aircraft in the schedule.

If verification fails, `run_experiment` sets the reported objective to $\infty$ and flags the instance as `FAIL` in all exports. A verified objective strictly below the known reference is reported as a genuine new best with all constraint checks confirmed.

---

## 12. Design Decisions and Known Trade-offs

**Full pairwise vs. consecutive-only separation constraints.** The Stage-2 LP in Zhang et al. (2020) (Eq. 3) enforces only $n-1$ consecutive separation constraints, which is correct when the separation matrix satisfies the triangle inequality globally. OR Library wake-vortex matrices violate this assumption routinely. Both the LP (`_build_lp_matrices`) and the feasibility check (`is_feasible`) therefore enforce all $\frac{n(n-1)}{2}$ pairwise constraints. The consecutive-only versions (`_build_lp_matrices_old`, `is_feasible_old`) are retained solely for the semi-feasible SA guidance path, where speed is more important than completeness.

**Dual-track feasibility serves distinct purposes.** Using the full $O(n^2)$ check on every SA acceptance decision would increase per-move cost by approximately $n$-fold and reduce the effective neighbourhood explored within a fixed time budget. Using the relaxed $O(n)$ check for gap reporting would silently accept infeasible solutions. The dual-track architecture resolves this: cheap checks guide search; expensive checks govern reporting. Conflating the two produces silently wrong results.

**Known optima never bias computation.** Reference values from Zhang et al. (2020) and Beasley et al. (2000) appear only in final gap-percentage reporting and Optuna's tuning objective. They are never used as bounds, warm starts, or termination conditions inside any search routine.

**Sequence-based (permutation) formulation vs. time-indexed LP.** A column-and-row generation approach using a time-indexed LP relaxation was explored and rejected. The LP relaxation gap is structurally too wide for effective branch-and-price on the ALP; permutation-based search with per-sequence LP evaluation consistently dominates.

**ILS perturbation is necessary for large instances.** For $n > 30$, random sequences are nearly always infeasible. Double-bridge perturbation from an incumbent is the only practical mechanism for genuine restart, as it guarantees a structural departure from the current basin that is unreachable by any 3-opt move.

**Alternate-solution tracking.** The `_MAX_ALT_SEQS = 100` cap bounds memory usage on large instances. The count of distinct near-optimal sequences is a landscape-degeneracy metric: high counts indicate a flat objective surface; low counts indicate a sharp, isolated optimum.

**ProcessPoolExecutor with fork/spawn context.** Fork is used on Linux/macOS for low overhead; spawn is used on Windows for correctness. The context is set at import time via `_CTX`. To debug in single-process mode, set `n_workers=1` when calling `ms_sa` or `run_experiment`.

---

## 13. References

- Beasley, J. E., Krishnamoorthy, M., Sharaiha, Y. M., & Abramson, D. (2000). Scheduling aircraft landings — the static case. *Transportation Science*, 34(2), 180–197.
- Zhang, J., Zhao, P., Yang, C., & Hu, R. (2020). A new meta-heuristic approach for the aircraft landing problem. *Transactions of Nanjing University of Aeronautics and Astronautics*, 37(2), 197–208.
- Pinedo, M. L. (2016). *Scheduling: Theory, Algorithms, and Systems* (5th ed.). Springer.
- Vepsalainen, A. P. J., & Morton, T. E. (1987). Priority rules for job shops with weighted tardiness costs. *Management Science*, 33(8), 1035–1047.
- Baker, K. R., & Bertrand, J. W. M. (1982). A dynamic priority rule for scheduling against due-dates. *Journal of Operations Management*, 3(1), 37–42.
- Applegate, D. L., Bixby, R. E., Chvátal, V., & Cook, W. J. (2006). *The Traveling Salesman Problem: A Computational Study*. Princeton University Press.
- Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by simulated annealing. *Science*, 220(4598), 671–680.
