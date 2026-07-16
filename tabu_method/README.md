# MR-SA-Tabu Run

This folder contains the Tabu experiment for the multi-runway aircraft landing
solver.

The code is still the original multi-chain simulated annealing solver. The
change is a short memory rule added inside the SA loop. When an accepted move
sends an aircraft from one runway to another, the solver records the reverse
move for a short period. That stops the search from quickly undoing the same
assignment and cycling around the same local solution.

This is not a separate Tabu Search model. It is simulated annealing with a
Tabu filter on recent cross-runway transfers.

## What Was Run

The final check reran the full Tabu study:

```text
4 hard cases x 3 seeds x 9 settings = 108 runs
```

Cases:

```text
airland11 m=2
airland12 m=2
airland13 m=2
airland13 m=3
```

Seeds:

```text
0, 1, 2
```

Settings:

```text
tabu_off
fixed_t20
fixed_t40
fixed_t80
fixed_t160
iteration_t80
fixed_t80_lp_aspiration
fixed_t80_least_recent
reactive_combined
```

All 108 rerun schedules were feasible.

## Main Result

The rerun confirmed the first controlled study. The best mean settings did not
change.

| Case | Best mean setting | Mean gap | Best gap |
|---|---:|---:|---:|
| airland11 m=2 | fixed tenure 80 | 1.8651% | 1.7146% |
| airland12 m=2 | fixed tenure 20 | 2.6549% | 0.1121% |
| airland13 m=2 | fixed tenure 40 | 3.8922% | 2.2032% |
| airland13 m=3 | fixed tenure 40 | 4.9680% | 3.4978% |

Best single runs:

| Case | Setting | Seed | Gap |
|---|---:|---:|---:|
| airland11 m=2 | fixed tenure 80 | 0 | 1.7146% |
| airland12 m=2 | fixed tenure 20 | 1 | 0.1121% |
| airland13 m=2 | fixed tenure 40 | 1 | 2.2032% |
| airland13 m=3 | fixed tenure 80 | 2 | 2.4056% |

Only two row-level gaps changed from the first saved study. Both were
non-winning airland13 m=2 rows, so the recommendation stayed the same.

## Files To Read

```text
results/full_tabu_rerun_20260715/ALP_Tabu_Full_Run_Results_2026-07-15.pdf
results/full_tabu_rerun_20260715/study_summary.csv
results/full_tabu_rerun_20260715/study_rows.csv
```

The PDF is the main result file. The two CSV files are included so another
person can check the numbers directly.

## How To Run One Case

Install requirements:

```powershell
python -m pip install -r requirements.txt
```

Run the best airland12 setting:

```powershell
python run_tabu_case.py airland12 2 --seed 1 --tabu on --tenure 20
```

Run the matched no-Tabu comparison:

```powershell
python run_tabu_case.py airland12 2 --seed 1 --tabu off --tenure 20
```

Run the full matrix again:

```powershell
python run_tabu_recommendation_study.py `
  --cases airland11:2 airland12:2 airland13:2 airland13:3 `
  --seeds 0 1 2 `
  --output results\new_tabu_rerun
```

The airland13 cases are slow. A full rerun can take many hours.

## Current Recommendation

Use fixed Tabu tenure as the current baseline:

```text
airland11 m=2 -> tenure 80
airland12 m=2 -> tenure 20
airland13 m=2 -> tenure 40
airland13 m=3 -> tenure 40 by mean gap
```

For airland13 m=3, tenure 80 still produced the best single run.

The adaptive variants did not win the mean gap on any case.
