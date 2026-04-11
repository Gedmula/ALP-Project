# Aircraft Landing Problem — Solver Pipeline

Single-runway scheduling under wake-vortex separation constraints, formulated as **$1 \mid r_j, s_{jk}, \bar{\delta}_j \mid \sum (g_j E_j + h_j T_j)$** and solved via a two-stage decomposition: sequence optimisation by metaheuristic search, followed by exact timing optimisation via a linear program.

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

- $x_j \in [r_j, d_j]$ for all $j$ (window constraints)
- $x_{\pi(l+1)} \ge x_{\pi(l)} + s_{\pi(l), \pi(l+1)}$ for all consecutive pairs in $\pi$ (separation constraints)
- $\sum_j \left(g_j \max(\delta_j - x_j, 0) + h_j \max(x_j - \delta_j, 0)\right)$ is minimised

The MILP is NP-hard in general. The pipeline exploits the two-stage structure: once a sequence $\pi$ is fixed, all binary sequencing variables are determined and the timing subproblem reduces to an LP with $3n$ variables and $O(n)$ constraints (Zhang et al., 2020).

---

## 2. Repository Layout

```text
project_root/
├── alp_pipeline.py          # Main solver — MS-SA + Stage-2 LP
├── alp_benders.py           # Benders decomposition module
├── data/
│   ├── airland1.txt         # OR Library benchmark instances
│   ├── airland2.txt
│   └── ...                  # airland1–airland13
├── plots/                   # Auto-generated figures (Gantt, convergence, gaps)
├── results/                 # Auto-generated CSV/JSON/TXT exports
│   ├── summary.csv
│   ├── schedules.csv
│   ├── verification.txt
│   └── run_metadata.json
└── README.md
```

OR Library files are **not** included in the repository. See [Section 4](#4-data) for download instructions.

---

## 3. Dependencies

Python 3.9 or later is required. All core dependencies are available via pip.

**Required:**

```text
numpy
scipy          # linprog / HiGHS interface
matplotlib
```

**Optional but recommended:**

```text
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

```text
http://people.brunel.ac.uk/~mastjjb/jeb/orlib/airlandinfo.html
```

Download the files `airland1.txt` through `airland13.txt` and place them in `./data/` relative to `alp_pipeline.py`. The loader will automatically detect whichever files are present and skip missing ones.

**Known optimal values** used for gap reporting (CPLEX reference from Zhang et al., 2020):

| Instance | n | Known optimum |
|---|---|---|
| airland1 | 10 | 700 |
| airland2 | 15 | 1480 |
| airland3 | 20 | 820 |
| airland4 | 10 | 2520 |
| airland5 | 10 | 3100 |
| airland6 | 30 | 24442 |
| airland7 | 44 | 1550 |
| airland8 | 50 | 1950 |
| airland9 | 100 | 5611.70 |
| airland10 | 150 | 12640.42 |
| airland11 | 200 | 12462.18 |
| airland12 | 300 | 16629.10 |
| airland13 | 500 | 39287.52 |

**File format.** Each file begins with `n` and a freeze-time token (unused), followed by $n$ blocks of the form `appear_j r_j δ_j d_j g_j h_j s[j][0] … s[j][n−1]`. The separation row for aircraft $j$ immediately follows its six scalar parameters. The loader validates the total token count before parsing.

---

## 5. Architecture Overview

```text
alp_pipeline.py
│
├── ALPInstance                  Data container + MPDS parameter computation
├── load_orlib / synthetic_instance    Instance I/O
│
├── is_feasible                  O(n) greedy forward-pass feasibility check
├── solve_stage2 / solve_lp      Stage-2 LP via HiGHS (SciPy interface)
├── evaluate / batch_evaluate    Feasibility gate + LP objective
│
├── Initial solution generators
│     gen_erd, gen_edd, gen_mdd, gen_atc, gen_mpds
│
├── Neighbourhood operators
│     op_swap, op_insert, op_reverse, op_or_opt_2, op_or_opt_3
│
├── calibrate_T0                 Empirical T_0 calibration (Eq. 30)
├── run_sa                       Single SA chain with reactive adaptation
├── run_ils                      ILS wrapper (SA + double-bridge perturbation)
├── ms_sa                        Parallel multi-start SA (ProcessPoolExecutor)
│
├── adaptive_params              Size-scaled default SAParams
├── tune_sa                      Optuna TPE hyperparameter search
│
├── verify_schedule              Nine-group constraint audit
├── verify_all                   Convenience wrapper + formatted report
│
├── export_results               summary.csv, schedules.csv, verification.txt, metadata.json
├── run_experiment               End-to-end benchmark runner for one instance
│
└── Visualisation
      plot_gantt, plot_sa_convergence, plot_gap_summary, plot_alt_solutions
```

---

## 6. Module Reference

### `ALPInstance`

Dataclass holding all instance data. Constructed by `load_orlib` or `synthetic_instance`. The field `$s_{\mathrm{bar}}$` (mean off-diagonal separation, computed in `__post_init__`) is used throughout as a normalisation constant. Do not modify `$s_{\mathrm{bar}}$` manually; it is derived from `s` automatically.

### `load_orlib(path, name="")`

Parses an OR Library text file and returns an `ALPInstance`. Raises `FileNotFoundError` if the path does not exist and `ValueError` if the token count is inconsistent with the declared $n$.

### `synthetic_instance(n, seed)`

Generates a random feasible instance of size n using ICAO wake-vortex separation categories. Useful for quick tests without OR Library files.

### `is_feasible(seq, inst)`

Runs the $O(n)$ greedy forward pass. Returns `True` if and only if every aircraft in `seq` can land within its window given the separation requirements of its predecessor. This check gates every LP call; infeasible sequences are rejected without invoking HiGHS.

### `solve_stage2(seq, inst)`

Solves the Stage-2 LP for the given sequence. Returns `(objective, landing_times)` or `(inf, None)` if HiGHS reports infeasibility or numerical failure.

### `evaluate(seq, inst)`

Calls `is_feasible` first; invokes `solve_lp` only if feasible. Returns `inf` for infeasible sequences. This is the objective function called inside the SA inner loop.

### `run_sa(seq0, inst, p, seed, T0)`

Single SA chain with reactive temperature adaptation. See [Section 7.2](#72-reactive-sa-run_sa) for full details. Returns `(best_seq, best_obj, stats_dict)`. The `stats_dict` contains `obj`, `time`, `t_best`, `history`, `init_obj`, `n_alt_seqs`, and `alpha_history`.

### `run_ils(seq0, inst, p, n_restarts, seed)`

Wraps `run_sa` with an Iterated Local Search loop. After each SA run, applies `_double_bridge` to the incumbent best, then re-runs SA from the perturbed sequence. The global best across all restarts is returned.

### `ms_sa(inst, p, n_workers, n_ils)`

Launches one SA/ILS chain per worker in parallel using `ProcessPoolExecutor`. Starting sequences are drawn from the six named heuristics (MPDS, EDD, MDD, ERD, ATC_k2, ATC_k4) plus double-bridge variants if more workers are available than named seeds. Returns the best result across all chains together with aggregate statistics.

### `adaptive_params(n)`

Returns a `(SAParams, n_ils_restarts)` tuple scaled to instance size. Small instances ($n \le 20$) use tight parameters with no ILS; large instances ($n > 150$) use slow cooling and up to six ILS restarts per chain.

### `tune_sa(inst, known_opt, n_trials, n_workers)`

Optuna TPE study over the six `SAParams` fields. Disabled by default (`ENABLE_OPTUNA = False` in `__main__`). If enabled, tunes on the first found instance and applies the result to all subsequent runs.

### `verify_schedule(seq, inst, tol)`

Full constraint audit covering nine groups (C1–C9): release dates, deadlines, separation chain, permutation validity, greedy-pass consistency, earliness/tardiness non-negativity, objective cross-check, and an independent LP re-solve. Never short-circuits — all groups are checked regardless of earlier failures. Returns `(passed, objective, VerificationReport)`.

### `export_results(results_list, out_dir)`

Writes `summary.csv`, `schedules.csv`, `verification.txt`, and `run_metadata.json` to `out_dir`. Also called per-instance inside `run_experiment` for incremental saves (in case of mid-run failure).

### `run_experiment(inst, known_opt, sa_p, n_workers)`

End-to-end runner for a single instance: runs MS-SA, verifies the result, exports outputs, and generates Gantt and convergence plots. Returns a result dict suitable for aggregate reporting.

---

## 7. Algorithms

### 7.1 Stage-2 LP Decomposition

The key structural insight exploited throughout is that once the landing sequence $\pi$ is fixed, the binary sequencing variables $q_{jk}$ are fully determined. The MILP separation constraints reduce to a linear chain:


$$
x_{\pi(l+1)} \ge x_{\pi(l)} + s_{\pi(l), \pi(l+1)}, \qquad l = 1, \ldots, n-1
$$


The resulting LP has $3n$ variables $(x_j, E_j, T_j)$ and $O(n)$ constraints, compared to $O(n^2)$ in the full MILP. HiGHS solves instances with $n \le 50$ in sub-millisecond time, making it practical to call the LP at every SA move.

### 7.2 Reactive SA (`run_sa`)

The SA chain adapts its own cooling rate after every temperature level based on the observed acceptance rate $\chi$:

**Acceptance rate target:** $\chi^* = 0.20$. If $\chi > \chi^*$, the chain is too hot (accepting near-random moves); $\alpha$ is nudged downward. If $\chi < \chi^*$, the chain is freezing; $\alpha$ is nudged upward.

$$
\alpha \leftarrow \operatorname{clip}\!\left(\alpha + \operatorname{sign}(\chi - \chi^*) \times 0.005,\; 0.80,\; 0.999\right)
$$

The nudge magnitude (0.005 per level) ensures smooth adaptation without instability. `p.alpha` in `SAParams` sets the initial cooling rate; the chain drifts away from it organically.

**Stagnation reheating:** When `M_stag` consecutive levels yield no improvement, instead of terminating, the chain reheats:

$$
T \leftarrow 2.0\,T
$$

$$
\pi \leftarrow \pi_{\mathrm{best}}
$$

Up to three reheats are permitted (`MAX_REHEATS = 3`). After the third, the chain exits. This replaces the hard stop of the original design with a controlled exploration burst from the known incumbent.

### 7.3 Multi-Start Strategy (`ms_sa`)

The primary methodological contribution relative to Zhang et al. (2020) is the diversified multi-start approach. Six deterministic dispatching rules generate distinct initial sequences:

| Rule | Ordering principle |
|---|---|
| ERD | Earliest release date (arrival order) |
| EDD | Earliest target time |
| MDD | Modified Due Date — minimises $\max(\delta_j, \text{earliest feasible time})$ |
| ATC (K=2) | Apparent Tardiness Cost with lookahead factor $K=2$ |
| ATC (K=4) | ATC with wider lookahead |
| MPDS | Multi-Priority Dispatching Sequence (Zhang et al., 2020, Eq. 16) |

If more workers are available than named seeds, additional starts are generated by applying `_double_bridge` perturbation to the named seeds. All chains run in parallel via `ProcessPoolExecutor` using fork-based multiprocessing on Linux/macOS and spawn on Windows.

### 7.4 Iterated Local Search (`run_ils`)

Each SA chain can optionally run an ILS wrapper controlled by the `n_ils` parameter returned by `adaptive_params`. After an initial SA run, the double-bridge (4-opt) operator perturbs the incumbent:

```text
A | B | C | D  →  A | C | B | D
```

This move is unreachable by any 3-opt sequence, ensuring the perturbation genuinely escapes the current local-optimum basin (Applegate et al., 2006). The perturbed sequence seeds a new SA run; if it improves the global best, the alternate-solution counter resets.

### 7.5 Initial Solution Generators

All six rules produce landing sequences by greedily selecting aircraft one at a time from the remaining unscheduled set. The current time $t$ and the last-scheduled aircraft $k$ are tracked throughout to compute accurate separation requirements for each candidate.

**MPDS priority index** (Zhang et al., 2020, Eq. 16):

$$
I_{\mathrm{MPDS}}(j)
=
\exp\!\left(-\frac{\mathrm{slack}}{K_1}\right)
\exp\!\left(-\frac{s_{kj}}{K_2 \cdot \bar{s}}\right)
\exp\!\left(-\frac{r_{\mathrm{wait}}}{K_3}\right)
\exp\!\left(-\frac{\mathrm{penalty}}{K_4}\right)
$$

where $K_1$–$K_4$ are instance-level scaling parameters derived from the distribution of target times, separation values, and cost weights.

---

## 8. SAParams Reference

`SAParams` is a dataclass with the following fields. All values are initial settings; `run_sa` may adapt `alpha` reactively during execution.

| Field | Default | Meaning |
|---|---|---|
| `alpha` | 0.99 | Initial geometric cooling rate $\alpha \in (0, 1)$ |
| `N_iter` | 120 | Number of neighbour evaluations per temperature level |
| `$T_{\min}$` | 1e-4 | Temperature floor; chain exits when $T < T_{\min}$ |
| `$I_{\max}$` | 600 | Hard cap on the number of outer (temperature-level) iterations |
| `$M_{\mathrm{stag}}$` | 60 | Stagnation threshold: trigger reheat after this many non-improving levels |
| `chi0` | 0.50 | Target initial acceptance probability used by `calibrate_T0` |

`adaptive_params(n)` returns recommended values for each instance size tier. The `SA_full` override in `__main__` is appropriate for full benchmark runs.

---

## 9. Running the Pipeline

**Full benchmark (all OR Library instances found in `./data/`):**

```bash
python alp_pipeline.py
```

The script auto-discovers all `airland*.txt` files present in `./data/`, runs `diagnose_instance` followed by `run_experiment` on each, prints per-instance and aggregate tables, then exports consolidated results.

**Single instance:**

```python
from alp_pipeline import load_orlib, run_experiment, SAParams

inst = load_orlib("data/airland2.txt", "airland2")
result = run_experiment(inst, known_opt=1480)
```

**Synthetic demo (no data files required):**

If no OR Library files are found, the pipeline automatically falls back to a 20-aircraft synthetic instance.

**Optuna hyperparameter tuning:**

Set `ENABLE_OPTUNA = True` in `__main__`. Tuning runs on the first discovered instance (fastest) for `n_trials=40` trials before the benchmark loop. The tuned parameters replace `SA_full` for all subsequent runs.

**Custom parameters:**

```python
from alp_pipeline import SAParams, ms_sa, load_orlib

inst = load_orlib("data/airland8.txt", "airland8")
p = SAParams(alpha=0.98, N_iter=200, T_min=1e-4, I_max=1000, M_stag=80)
best_seq, best_obj, stats = ms_sa(inst, p, n_workers=16, n_ils=3)
```

**Worker count.** `N_CPU` defaults to `os.cpu_count() − 8`. Override by passing `n_workers` explicitly. Each worker runs one SA/ILS chain; the number of chains equals `n_workers`.

---

## 10. Outputs

All outputs are written relative to the working directory.

### `plots/`

| File | Content |
|---|---|
| `gantt_{instance}_{method}.png` | Gantt chart with time windows [r_j, d_j], target times δ_j (□), scheduled landing times x_j (○), and wake-vortex separation zones (orange shading) |
| `convergence_{instance}.png` | Best-objective history per SA chain on a log scale; the chain achieving the global best is drawn with higher opacity |
| `gap_summary.png` | Bar chart of percentage gap to known optimum across all instances |
| `alt_solutions.png` | Counts of distinct near-optimal sequences found and heuristic seeds already at optimum, per instance |

### `results/`

| File | Content |
|---|---|
| `summary.csv` | One row per instance: objective, gap, wall time, time-to-best, alternate sequence counts, verification status |
| `schedules.csv` | One row per aircraft per instance: landing position, release time, target time, deadline, scheduled time, earliness, tardiness, per-aircraft penalty |
| `verification.txt` | Full `VerificationReport.summary()` for every instance, with per-constraint violation detail when failures exist |
| `run_metadata.json` | Timestamp, hostname, Python version, CPU count, per-instance pass/fail |

Per-instance subdirectories (`results/{instance_name}/`) are written incrementally during the benchmark loop, so partial results are preserved if the run is interrupted.

---

## 11. Verification System

Every solution is audited against nine constraint groups before the objective value is accepted.

| Group | Constraint checked | Count |
|---|---|---|
| C1 | Release dates: $x_j \ge r_j$ | n |
| C2 | Deadlines: $x_j \le d_j$ | n |
| C3 | Separation chain: $x_{\pi(l+1)} \ge x_{\pi(l)} + s_{\pi(l),\pi(l+1)}$ | n−1 |
| C4 | Permutation validity: distinct indices in [0, n−1] | n |
| C5 | Greedy-pass consistency: O(n) forward pass matches LP times | n−1 |
| C6 | Earliness non-negativity: $E_j \ge 0$ | n |
| C7 | Tardiness non-negativity: $T_j \ge 0$ | n |
| C8 | Objective cross-check: $\mathrm{LP\ obj} \approx \sum (g_j E_j + h_j T_j)$ | 1 |
| C9 | Independent LP re-solve: fresh HiGHS call from scratch | 1 |

The audit never short-circuits — all groups are evaluated regardless of earlier failures. The C9 re-solve uses separately constructed LP matrices and tighter solver tolerances (1e-9) than the main solve path, making it a genuine independent cross-check rather than a cache hit.

If verification fails, `run_experiment` sets the reported objective to `inf` and flags the result as `FAIL` in all exports. A verified objective that is strictly below the known reference is reported as a potential new best and all constraints are confirmed to be satisfied.

---

## 12. Design Decisions and Known Trade-offs

**Sequence-based (permutation) formulation vs. time-indexed LP.** An earlier column-and-row generation approach using a time-indexed LP relaxation was explored and explicitly rejected. The LP relaxation gap in time-indexed formulations is structurally too wide for the ALP to support effective branch-and-price; permutation-based search with per-sequence LP evaluation consistently dominates. See `alp_cg_solver.py` (archived) for reference.

**Known optima never bias computation.** Reference values from Zhang et al. (2020) appear only in final gap reporting and Optuna's tuning objective. They are never used as bounds, warm starts, or termination conditions inside any search routine.

**Tight big-M values in Benders.** The Benders decomposition module (`alp_benders.py`) uses per-pair big-M values $M_{jk} = d_{\pi(l+1)} - r_{\pi(l)}$ derived from actual time windows, rather than a single global constant. A global $M$ renders all optimality cuts trivially satisfiable and prevents convergence.

**ProcessPoolExecutor with fork/spawn context.** Fork is used on Linux/macOS for low overhead; spawn is used on Windows for correctness. The context is set at import time via `_CTX`. If you encounter issues with multiprocessing on specific environments, set `N_CPU = 1` to disable parallelism and debug in single-process mode.

**Alternate-solution tracking.** The `_MAX_ALT_SEQS = 100` cap bounds memory usage on large instances. The count of distinct near-optimal sequences is reported as a landscape-degeneracy metric: high counts indicate a flat objective surface; low counts indicate a sharp, isolated optimum.

---

## 13. References

- Beasley, J. E., Krishnamoorthy, M., Sharaiha, Y. M., & Abramson, D. (2000). Scheduling aircraft landings — the static case. *Transportation Science*, 34(2), 180–197.
- Zhang, R., Liang, P., & Chen, X. (2020). Multi-priority dispatching for the aircraft landing problem. *Computers & Operations Research*, 123, 105017.
- Pinedo, M. L. (2016). *Scheduling: Theory, Algorithms, and Systems* (5th ed.). Springer.
- Vepsalainen, A. P. J., & Morton, T. E. (1987). Priority rules for job shops with weighted tardiness costs. *Management Science*, 33(8), 1035–1047.
- Baker, K. R., & Bertrand, J. W. M. (1982). A dynamic priority rule for scheduling against due-dates. *Journal of Operations Management*, 3(1), 37–42.
- Applegate, D. L., Bixby, R. E., Chvátal, V., & Cook, W. J. (2006). *The Traveling Salesman Problem: A Computational Study*. Princeton University Press.