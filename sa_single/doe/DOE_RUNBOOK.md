# DOE runbook: how to set up and run each stage

Written for: whoever runs the SA design of experiments on the cluster machine, from the screening onward.

The screening is done (`results/screening`) and analysed
(`results/analysis_screening/SCREENING_ANALYSIS.md`). The next stage is a
face-centred central composite design (CCD). Every stage is one named run
configuration in [`stages.py`](stages.py); [`run_doe.py`](run_doe.py) runs it,
[`analyze.py`](analyze.py) analyses it. Neither script takes command-line arguments: set the
CONFIGURE HERE block at the top of the script (`STAGE`, `N_SEEDS`, `REPLICAS`, `DRY_RUN` in
`run_doe.py`; `ANALYSIS` in `analyze.py`) and run it from the repository root with
`python sa_single/doe/run_doe.py` or `python sa_single/doe/analyze.py`.

## 1. The sequence

Run everything from the repository root. In the table, "Run" is the value to give `STAGE` in
`run_doe.py` and "Then analyse" the value to give `ANALYSIS` in `analyze.py`. With `DRY_RUN = True`,
`run_doe.py` prints the design, the instances, the run count and a time estimate and runs nothing.

| # | Stage | `STAGE =` | Then `ANALYSIS =` | Runs | Est. time | Results folder |
|---|---|---|---|---|---|---|
| 0 | screening (done) | `"screening"` | `"screening"`, plus `screening_deep.py` | 2880 | 41 h (done) | `results/screening` |
| 1 | **CCD** on A–E | `"ccd"` | `"ccd"` (provisional) | 1920 | ~32 h | `results/ccd` |
| 2 | anchors, tune set | `"anchor_tune"` | `"ccd"` again (final) | 40 | ~7 h | `results/anchor_tune` |
| 3 | F / G check | `"check"` | `"check"` | 240 | ~6 h | `results/check` |
| 4 | confirmation, holdout | `"confirm"` | `"confirm"` | 144 | ~1.5 h | `results/confirm` |
| 5 | anchors, holdout | `"anchor_holdout"` | `"budget"` | 96 | ~10 h | `results/anchor_holdout` |
| – | foldover (optional) | `"foldover"` | `"screening"` | 960 | ~27 h | `results/foldover` |

Why this order: stage 2 runs at the settings that stage 1's analysis recommends
(`results/recommended_params.json`), and the analysis is then repeated so that
every gap is measured against the stronger reference values. Stages 3–5 read
that file too and refuse to start without it. The foldover is not part of the
plan (see `SCREENING_ANALYSIS.md` section 8); the stage exists in case the
CCD leaves the AF = BG = DE chain unexplained.

Time estimates come from the screening's seconds per run at the same F level
(mutation size is what makes runs slow; for the holdout instances, the screening mean for the same size), so they hold only for this machine class. About 56 h in total for stages 1–5 at 3 / 2 seeds, plus the optional foldover.

## 2. Setup (once)

1. **Environment: Linux only for running.** The starting-schedule constructors use `fork` multiprocessing
   (`const_heuristics/alp_core.py`), so `run_doe.py` cannot run on native Windows (it fails with
   `cannot find context for 'fork'`); use WSL2 or Linux. A dry run (`DRY_RUN = True`), `analyze.py` and `screening_deep.py` run anywhere.
   Use the environment of the screening, so that results are
   comparable across stages: Ubuntu 24.04 under WSL2, Python 3.12.3,
   numpy 1.26.4, numba 0.60.0, scipy 1.11.4, pandas 2.1.4, highspy 1.15.1,
   12 logical processors (10 replicas per run). Newer numba/numpy versions run but are not what
   the screening used. Analysis additionally needs matplotlib and statsmodels (versions were not recorded).
2. **Code.** Pull the branch that contains `stages.py`, the phase-aware
   `params.py` (`FACTOR_SETS`) and `instances.py` (large sets, informative subset).
3. **Plan check.** With `STAGE = "ccd"` and `DRY_RUN = True`, `python sa_single/doe/run_doe.py` must show 32
   configurations × 20 instances × 3 seeds = 1920 runs, window levels 8 / 16 / 24,
   F = 4, G = 1000.
4. **Machine.** Nothing else running (each run uses every replica process and
   timing is a response), sleep and updates off, laptop on power.
5. **First run builds starting schedules** for each instance (all cores, cached in
   `sa_single/doe/cache/<instance>.json`, seed 0). The first CCD runs on each
   instance are therefore slower.

## 3. Stage by stage

### Stage 1: CCD (`ccd`)

- **What:** 16 half-fraction corners (E = ABCD, resolution V), 10 axial and 6 centre configurations on
  A hot acceptance 0.30–0.80, B cold acceptance 0.001–0.05, C exchange interval 10–200 moves/aircraft,
  **D window 8 / 16 / 24** (screening range 4–16 moved up, because the optimum sat at or beyond its top),
  E guided share 0.50–0.95. F = 4 and G = 1000 moves/aircraft are held fixed. 20 instances, 3 seeds.
- **Run:** `STAGE = "ccd"`, `DRY_RUN = False`, then `python sa_single/doe/run_doe.py` (resumable: run it again unchanged).
  Run order is randomised, so an interrupted stage holds a random subset, not a partial design.
- **Instances:** the 20 `tune_large` instances on which the screening runs did not all end on the same cost
  (`instances.informative_instances()`, read from `results/screening/records.csv`). The other 28 cannot
  discriminate configurations. They are checked again, all of them, in the confirmation stage.
- **Analyse:** `ANALYSIS = "ccd"` in `analyze.py`, then `python sa_single/doe/analyze.py` (`PYTHONIOENCODING=utf-8` on Windows consoles).
  Writes `results/analysis_ccd/` and **`results/recommended_params.json`**.
- **Read before moving on** (`results/analysis_ccd/ccd_summary.json` and the tables):
  - fit: `r2_pred` should be clearly positive and lack of fit (`lack_of_fit_p`) not significant for log gap;
  - **optimum on the box edge:** any factor with |coded level| ≥ 0.9 in `optimum_coded`, above all D at +1
    (window 24), means the optimum is outside the box. Do not proceed to confirmation: shift or
    extend the range (section 5) and run a second CCD;
  - `optimum_by_class.csv` / `optimum_by_class.png`: the screening found the window effect to be
    a congestion effect (−0.96 on congested wide instances, −0.09 below c = 1.0) and B and E to reverse with window type.
    If the class optima differ materially from the pooled one, a single setting costs quality on one class.
  - `control_x_noise.csv`: factor × class interactions on the CCD data (instance-clustered).
  - `plots/contours_*`, `surfaces_*`, `perturbation_*`, `optimum_by_size.png`.

### Stage 2: anchors on the tune set (`anchor_tune`)

- **What:** the recommended settings with ten times the budget, 2 seeds, on the 20 instances. Gaps are measured
  against the best cost found for each instance in any tune-set phase, so these long runs sharpen
  the reference that the near-zero gaps of the best configurations are compared with.
- **Run:** `STAGE = "anchor_tune"` in `run_doe.py`, then run `analyze.py` again with `ANALYSIS = "ccd"`.
  The recommendation can move slightly, and it is the final one that the later stages use.

### Stage 3: check of F and G (`check`)

- **What:** four configurations at the final optimum: the optimum itself (`X01`), F raised from 4 to the centre 10 (`X02`)
  and to 16 (`X03`), and G lowered from 1000 to 100 (`X04`). 3 seeds, 20 instances. The screening could not separate
  AF, BG and DE; this tests the question that matters: does moving F or G pay at the optimum?
- **Run / analyse:** `STAGE = "check"` in `run_doe.py`, then `ANALYSIS = "check"` in `analyze.py`.
  Output `results/analysis_check/check_vs_optimum.csv`, paired by instance.
- **Decision:** adopt a variant only if log gap improves (negative, p < 0.05) at a log wall-time you accept
  (F at 16 roughly doubled wall time in the screening). Otherwise the CCD optimum stands. If a variant is adopted,
  put the changed level into `ccd_fixed` of the `ccd` stage, re-fit locally; do not edit `recommended_params.json` by hand.
- Do not run the `"ccd"` analysis again after this stage: check runs enter the tune-set reference, which would change the CCD gaps.

### Stage 4: confirmation (`confirm`)

- **What:** the recommended settings on the 48 unseen `holdout_large` instances (the flat ones included), 3 seeds.
- **Run / analyse:** `STAGE = "confirm"` in `run_doe.py`, then `ANALYSIS = "confirm"` in `analyze.py`.
  Compares observed means with the CCD prediction intervals (`results/analysis_confirm/confirmation.csv`).
  Check that the 28 instances that were flat in the screening stay flat and that every run's audit is PASS.

### Stage 5: holdout anchors and the iteration budget (`anchor_holdout`)

- **Run / analyse:** `STAGE = "anchor_holdout"` in `run_doe.py`, then `ANALYSIS = "budget"` in `analyze.py`.
  The best-so-far curves of the 10× runs give the budget rule iterations = a·n^b, written into
  `recommended_params.json`. In the screening, 14.6% of runs on the informative instances found
  their best after 90% of the 5000 moves/aircraft budget, so the rule is expected to ask for more on the hardest sizes.

### Stage 0: screening (done)

Reproduce the analysis: `ANALYSIS = "screening"` in `analyze.py` (standard plots),
`python sa_single/doe/screening_deep.py` (clustered effects, de-aliasing, surfaces, cost tables).

### Optional: foldover (`foldover`)

Single-factor foldover on E (16 configurations, 20 instances, 3 seeds). Run only if a decision needs AF, BG and DE separated before the CCD:
`STAGE = "foldover"` in `run_doe.py`, then `ANALYSIS = "screening"` in `analyze.py` (the
screening analysis pools `screening` and `foldover` records automatically; `screening_deep.py` does not).

## 4. Changing a run configuration

Everything lives in the `STAGES` dictionary of [`stages.py`](stages.py):

| To change | Edit |
|---|---|
| number of seeds | `n_seeds` in the stage, or `N_SEEDS` in `run_doe.py` (seeds may be added on resume; run seeds are 1000 × seed + configuration index) |
| CCD factors, fixed levels, centre points | `ccd_factors`, `ccd_fixed` (coded levels), `n_center` of the `ccd` stage. F = 10 instead of 4: `ccd_fixed={"F": 0.0, "G": 1.0}` |
| instances | `instance_set` and `subset` (`None` = all 48, `"informative"` = 20) |
| check variants | `check_moves` (coded levels moved away from the optimum) |
| anchor budget | `anchor_mult` |
| factor ranges | `FACTORS_CCD` in `sa_single/params.py` (section 5) |

A stage's settings are saved in `results/<out>/stage.json`; resuming with different settings is
refused (the number of seeds may grow). Use a new results folder if you really mean a different experiment.

## 5. If the optimum is on the edge of the box

The coding of the response-surface stages is `FACTORS_CCD` (`params.py`), the screening's is `FACTORS`.
Data coded one way must be decoded the same way, so **never edit a range once its records exist.** Instead:
add a new entry to `FACTOR_SETS` (for example `"ccd2"`), add a stage with that `coding` and a new `out`, and
add the coding to the `CODING` line of `analyze.py`. Do not pool records of different codings in one model; the CCD analysis reads one folder only.

## 6. What the stage scripts write

| File | Content |
|---|---|
| `results/<out>/stage.json` | the run configuration used |
| `design.csv` | configurations, coded and natural levels |
| `instances.csv` | instances of the stage |
| `records.csv` | one row per run: best cost, start cost, time to best, wall time, audit |
| `curves.csv` | global best at 101 fractions of the budget |
| `results/recommended_params.json` | written by the `"ccd"` analysis, read by stages 2–5 |
| `results/analysis_<stage>/` | tables and plots of the analysis |

Commit `records.csv`, `curves.csv`, `design.csv`, `stage.json`, `instances.csv` of every finished stage and the analysis folders, as for the screening.

## 7. If something goes wrong

- **Interrupted:** re-issue the same command; finished run IDs are skipped.
- **"holds a different run configuration":** `stages.py` differs from the saved `stage.json`; undo the edit or use a new folder.
- **"needs recommended_params.json":** run the `"ccd"` analysis first.
- **A run with audit FAIL:** do not analyse; report it. The screening had none in 2880 runs.
- **Timings 2× the estimate:** check for a background load; the estimates are the screening's seconds per run on the
  same instances.
- **Different results on another machine:** the screening's seed 0 reproduced on a second machine in 959 of 960 runs; one differing run
  was a better value found. Use the pinned environment to keep it that way.

## 8. What changed in the code for this setup

- `params.py`: `FACTORS_CCD` (window 8–24) and `FACTOR_SETS`; `decode` and `encode` take a `coding`.
- `instances.py`: manifests of the `*_large` sets, path resolution for them, `informative_instances()` and `subset`.
- `stages.py` (new): the run configuration of every stage.
- `run_doe.py`: stage-driven (`STAGE`, `DRY_RUN` in its configure block), `stage.json` guard, `check` phase, no silent fallback to default settings for `check` and `confirm`.
- `analyze.py`: coding by stage; `ccd` adds the optimum per instance class and factor × class interactions; new `check` analysis; the desirability/`D` column and `C(instance)` fixes.
