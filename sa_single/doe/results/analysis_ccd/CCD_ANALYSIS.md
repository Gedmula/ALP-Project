# CCD analysis: findings and the decision before stage 2

Analysis of `results/ccd`: face-centred CCD on A–E (16 half-fraction corners, 10 axial, 6 centre = 32
configurations), F = 4 and G = 1000 moves/aircraft fixed, window 8 / 16 / 24, the 20 informative
`tune_large` instances, 3 seeds, 1920 runs. Tables and plots are written by `analyze.py`
(`ANALYSIS = "ccd"`). Where this document gives an **instance-clustered** test or a contrast between
settings, it comes from re-fitting the same quadratic model on `runs_with_responses.csv` with standard
errors clustered by instance (t on 19 df), as in the screening analysis; `analyze.py` itself reports only the
naive per-run tests. Log gap is log10(gap % + 0.01), negative is better. Coded levels: −1 / 0 / +1.

## 1. Bottom line

| Question | Answer |
|---|---|
| Is the data sound? | Yes. 1920/1920 runs, every audit PASS, 3 seeds in all 640 configuration × instance cells. |
| What matters for quality? | **B cold acceptance** (low is better, clustered t 3.8) and **C exchange interval** (low is better, t 4.1). Both also make the run reach its best sooner (t 3.0 and 3.6). |
| What does not matter? | **A hot acceptance** (t 1.7 on quality, none on speed; it only costs +6% wall time across its range) and, pooled over instances, **D window** (t −0.9). E is minor (high E is worse, +0.10 at E = 0.95). |
| Is there a real optimum region? | Yes: **cold acceptance ≲ 0.0025, exchange interval ≈ 10–25 moves/aircraft, window 16–24, guide 0.50–0.72, any A**. Inside it the configurations cannot be told apart. |
| Is the recommended point special? | **No.** It is one point of a flat top (section 4). It beats the centre (−0.155 log gap, t −2.8; −0.11 of the budget to the best, t 3.7); it is not distinguishable from the best observed corner R09 (−0.018, t −0.5). |
| How much does tuning buy? | Little against the centre, a lot against the bad corners. Mean gap: centre 0.14%, best corner 0.11%, B and C both high 0.44–0.51%. The gain is concentrated on about five hard instances (section 7). |
| Runbook gates | Fit: **pass on r²_pred, fail on lack of fit** (section 6). Box edge: **tripped** for C (−0.93) and borderline for B (−0.84) and D (class optima at +1) (section 5). |
| Next stage | Run **stage 2 (`anchor_tune`) as planned**, after the three things in section 9. **Hold stage 3–5 until the refit after stage 2 has been read** against the edge rule (section 10). |

## 2. The data

- 14.45 h for the stage against 32 h estimated, because the run used **30 replicas per run** (the machine has 32 logical
  CPUs; `replicas: null` means CPUs − 2). The screening used 10. See section 9.
- 51% of all runs end at the reference (gap 0), and 75% of the runs of the best configuration R09. The reference is the best cost
  any tune-set run has found, so a run cannot do better than 0 and the top configurations are compressed against that floor.
  Anchors (stage 2) are what lifts it.
- Five of the 20 instances are almost flat in this design (share of runs at gap 0: g100_c10_w 100%, g200_c8_w 100%, g100_c8_w 99%,
  g250_c10_t 96%, g300_c6_w 96%). They sat in the "informative" set because some screening run differed, but here they say nothing.
  The signal comes from about 12–15 instances.
- Noise scale: the six centre configurations are replicates, and their mean log gap spans −1.557 to −1.475 (sd 0.038), their mean time to best
  has sd 0.03 of the budget. Differences between configuration means of about 0.1 or less are therefore not real, which covers
  the first ten of the 32 configurations.

## 3. Effects (instance-clustered)

Quadratic model with instance intercepts. Coefficients per coded unit; naive t (what `analyze.py` prints) in brackets.

| Term | log gap | clustered t | time to best (fraction of budget) | clustered t | log wall time | clustered t |
|---|---|---|---|---|---|---|
| A hot acceptance | +0.018 | 1.7 (1.5) | −0.009 | −1.9 | **+0.029** | 6.9 |
| B cold acceptance | **+0.186** | **3.8** (15.9) | **+0.056** | **3.0** | +0.004 | 3.9 |
| C exchange interval | **+0.116** | **4.1** (9.9) | **+0.047** | **3.6** | **−0.069** | −18.8 |
| D window | −0.034 | −0.9 (−2.9) | −0.016 | −1.9 | **−0.056** | −24.7 |
| E guide | +0.027 | 1.3 (2.3) | +0.012 | 1.3 | −0.008 | −2.1 |
| B² | **+0.160** | **4.1** | +0.014 | 0.5 | −0.001 | −0.6 |
| C² | +0.072 | 1.9 | −0.004 | −0.2 | **+0.022** | 7.0 |
| D² | −0.044 | −1.7 | +0.010 | 0.5 | +0.011 | 5.6 |
| E² | +0.052 | 2.3 | +0.003 | 0.2 | −0.002 | −0.6 |

No two-factor interaction reaches p < 0.05 on log gap once clustered (B×E −0.030, t −1.9; C×E +0.013, t 2.0; B×D +0.023, t 1.6). On time to best,
B×C is −0.018 (t −2.9) and A×D −0.017 (t −2.1). The naive tests overstate everything by treating 1920 runs as independent (B: 15.9 against 3.8).
Use the clustered column; the same applies to the p-values in `anova_*.csv`.

What each factor does:

- **B (cold acceptance).** Convex in quality: the model minimum is at coded −0.58 (B ≈ 0.0022), and B = 0.001 (−1) is 0.03 worse in log gap than
  that, a difference too small to detect. Going up is what hurts: B = 0.0071 (centre) is +0.059 (t 1.3) and +0.05 of the budget (t 2.3) against the
  recommended 0.0014, B = 0.05 about +0.35 over the centre. Speed is linear in B, so speed alone keeps pushing it to the lower edge, but B = 0.001 is already
  near greedy at the cold end. Worse on 17 of 20 instances.
- **C (exchange interval).** Same shape: quality minimum at coded −0.64 to −0.81 (about 13–17 moves/aircraft), linear for speed so lower is faster.
  Moving C from the recommended −0.93 to −0.5 (about 24) changes log gap by −0.001: the whole range 10–25 is equal on quality. Lower C costs wall time:
  10 against 200 moves/aircraft is +15%. Worse when high on 18 of 20 instances.
- **D (window).** The screening's largest effect (4 → 16, −0.50) is gone over 8 → 24 (−0.07 pooled, not significant): the benefit saturates somewhere between 8
  and 16. The pooled null hides a split by congestion (section 7): large gains on congested wide instances, a loss on the lighter ones.
  D is never costly: a larger window is 11% faster in wall time from 8 to 24, so window 24 is a free choice unless an instance says otherwise.
- **E (guide share).** High E is worse (0.95 against the recommended 0.64: +0.096, t 2.6); 0.50 to 0.72 is equal. Its sign flips with congestion
  (E × congestion t 4.3 in `control_x_noise.csv`).
- **A (hot acceptance).** No effect on quality or speed anywhere in 0.30–0.80, but higher A costs wall time (+6%). If anything use the low half.

## 4. The recommended settings and how far to trust them

`results/recommended_params.json`: hot 0.550, cold 0.00138, exchange 11.2 moves/aircraft, window 22, guide 0.639 (coded A −0.00, B −0.84, C −0.93, D +0.78, E −0.38).

The optimiser maximises a clipped desirability, and the result is on its plateau:

- Overall desirability at the optimum is exactly **1.000**, and the predicted log gap (−1.6581) equals the best observed configuration mean
  (R09, −1.6578), the lower bound used to scale the desirability. Beyond that value the objective cannot improve, so the optimiser stops
  as soon as it is on the plateau.
- **A = −0.0012** is its start value (zero), not a finding: A has no gradient there.
- The model's own minimum of log gap alone is at A −1, B −0.67, C −0.64, D +1, E −0.35 and predicts 0.065 lower than the recommendation (about 15% in gap + 0.01),
  mostly from A and D, neither of which is significant.

Contrasts between settings (clustered, same model; negative log gap is better than the recommendation):

| Against the recommendation | Δ log gap (t) | Δ time to best (t) | Δ wall time |
|---|---|---|---|
| centre (all coded 0) | +0.155 (**2.8**) | +0.110 (**3.7**) | −4% |
| R09 corner (A −1, B −1, C −1, D +1, E −1) | −0.018 (−0.5) | +0.029 (1.0) | −3% |
| D = +1 (window 24) | −0.029 (−1.7) | +0.001 (0.1) | −1% |
| A = −1 (0.30) | −0.027 (−1.0) | +0.046 (2.3) | −3% |
| C = −0.5 (24 /aircraft) | −0.001 (−0.1) | +0.028 (1.8) | −4% |
| B = −0.5 (0.0022) | −0.003 (−0.2) | +0.019 (1.4) | 0% |
| B = 0 (0.0071) | +0.059 (1.3) | +0.053 (**2.3**) | +1% |
| E = 0 (0.725) | +0.007 (1.0) | +0.007 (0.9) | 0% |
| E = +1 (0.95) | +0.096 (**2.6**) | +0.029 (1.3) | −1.5% |

Read: leaving the plateau on B or E (up) or to the centre costs something real; moving about on the plateau costs or gains nothing detectable.
**Use the recommendation as a point inside the region of section 1, not as a located optimum.** The predicted log gap of −1.66 and the intervals in
`ccd_summary.json` (−2.42 to −0.90) are population averages at the floor, too wide to confirm anything; do not use them as the stage-4 pass criterion.

## 5. Optimum on the edge of the box (runbook gate)

| Factor | Pooled | By size (n ≥ 200) | By class | Reading |
|---|---|---|---|---|
| B | −0.84 | −0.77 to −1 | −0.27 tight, −0.97 wide | Quality interior (≈ −0.6), speed at the edge. Tight-window instances want a higher B. |
| C | **−0.93** | −1 for all five sizes n ≥ 200 | −1 tight, −0.92 wide | Quality interior (≈ −0.7), speed at the edge (C below 10 is untested). **Trips the ≥ 0.9 rule.** |
| D | +0.78 | +1 for n ≥ 300 | +1 for both c = 1.0 classes | Not bracketed from above for congested instances. |
| E | −0.38 | −0.44 to −0.80 | −0.80 tight, −0.95 wide | Interior pooled; low end of the range. |
| A | −0.00 | +0.2 to +1 | +1 tight, +0.05 wide | Free (null effect), ignore. |

- The classes `c<1 wide` and the sizes 100 and 150 return the **centre** (all zeros). That is the optimiser's start point on a surface with
  no information (most of those instances are flat), not evidence that the centre is best there. Ignore those rows.
- Quality is bracketed on B and C (convex with an interior minimum); the edge is a speed effect. A second CCD is therefore not
  needed to find the quality optimum, but it would be needed to learn whether C < 10 or D > 24 is faster or better on the congested instances.
- The tight-window instances behave differently: on g500_c10_t and g300_c10_t the corner R09 (0.92% and 0.77% mean gap) is **worse** than the centre
  (0.79% and 0.44%). These two carry the largest unexplained gap at the best settings (section 7).

## 6. Model fit

- r² 0.76, r²_pred 0.75 for log gap, but most of that is the instance intercepts (823 of 1162 total sum of squares). The factor terms explain 18% of the variation left after the instance effect (61 of 339).
- **Lack of fit is significant for every response** (log gap F 3.9, p ≈ 4e-92; the runbook asks for it not to be). The pure error is seed-to-seed
  variation within a configuration × instance cell, and the model has no factor × instance terms. Instances respond very differently
  (section 7: the B effect runs from 0 to +1.3, the D effect from −1.24 to +0.56), so the quadratic surface is a population average and no instance's own
  response. It is adequate for choosing a pooled setting and wrong for predicting any one instance.
- The response is zero-inflated (51% of runs at the floor of −2), so classical residuals fail (`plots/residuals_*.png`). The clustered tests and the per-instance
  signs are the safeguards, as in the screening.

## 7. Instance heterogeneity

Per-instance linear effects over the full range of each factor (log gap; `+` = worse when high). Count of the 20 instances with a positive effect:
**B 17**, **C 18**, A 12, E 12, **D 8**.

| Group | What it shows |
|---|---|
| D by congestion | Large window helps the congested wide instances: g200_c10_w −1.24, g300_c10_w −0.47, g500_c10_w −0.26, g400_c10_w −0.22. It hurts some c = 0.8 ones: g500_c8_w +0.56, g300_c8_w +0.26. Control × noise D × congestion: t −2.7, p 0.013. |
| B by window type | The B penalty is large on wide-window instances (g300_c8_w +1.27, g400_c10_w +1.18, g400_c8_w +1.07, g500_c8_w +0.73, g150_c10_w +0.72) and small on tight ones (B × wide t 2.8, p 0.012). Same direction as in the screening. |
| E by congestion | E × congestion t 4.3 (p 0.0004): high E helps the lighter instances and hurts c = 1.0 (as E × window type in the screening). |
| Where the gap is | Mean gap over all 32 configurations is 1.46% on g300_c10_w, 0.89% g500_c10_t, 0.61% g300_c10_t, 0.36% g500_c10_w, 0.43% g200_c10_w; every other instance is below 0.2%. These five (all c = 1.0) are where tuning matters. |
| Remaining gap at the best settings | The best configuration still has mean gap 0.52% on g500_c10_t, 0.29% g300_c10_w, 0.12% g300_c10_t. Large instances also use the budget: mean time to best is 0.50 of it at n = 300, 0.60 at n = 400, 0.63 at n = 500, and 16–24% of runs there find their best after 90%. This is a budget question (stage 5) and a reference question (stage 2), not something a better setting of A–E is likely to remove. |

## 8. What the CCD changes in the screening's conclusions

| Screening (A–D, 20 instances, 10 replicas) | CCD |
|---|---|
| **A**: +0.22 clustered (t 5.5), the first-ranked factor | Not reproduced: +0.04 over the same range (t 1.7). Within the screening it was +0.30 at window 4 and +0.15 at window 16. |
| **B**: +0.11 (t 1.4), "reverses with window type" | The strongest term: +0.37 over the range, convex, t 3.8. The window-type split persists. |
| **C**: +0.25 (t 5.2) | Reproduced: +0.23 (t 4.1), now with an interior quality minimum. |
| **D**: −0.50 (t −3.2), "the largest effect, optimum beyond 16" | Saturates: −0.07 over 8 → 24 (t −0.9). The beyond-16 gain is small on average and real only on congested instances. |
| **E**: inert on quality, flips with window type | Minor; flips with congestion. |
| **Curvature** is strong, optimum interior | Confirmed, on B and C. |
| **AF = BG = DE chain open** | E is in the model now, and D × E is nil (t −0.2). The chain remains a question only about A × F and B × G, which the F/G check (stage 3) tests. |

A, B and D changed their role. The data cannot say why, because several things changed at once: replicas 10 → 30, window range 4–16 → 8–24, E varied, the
reference now includes the CCD's own best runs, and software versions (section 9). Of these, the replica count is the most plausible cause for A and B (the
temperature ladder between hot and cold is spaced differently with 30 replicas than with 10), but that is a guess, not a result.

## 9. Before stage 2: things to settle

1. **Replicas.** The tuned values are conditional on 30 replicas. `recommended_params.json` has `replicas: null`, which resolves to CPU count − 2 at run time, so stages 2–5 on this machine will
   also use 30 and are comparable with the CCD. On a 12-thread machine they would use 10 and would not be. Decide whether the final parameters are meant for 30 replicas, and
   if so put `"replicas": 30` in the file. The runbook's time estimates (screening speed at 10 replicas) are about 2.2× too high for this machine: expect stage 2 at roughly 3 h, not 7.
2. **Software environment.** The runbook pins Python 3.12.3, numpy 1.26.4, numba 0.60.0, scipy 1.11.4, pandas 2.1.4 so that stages stay comparable with the screening.
   `.venv` was created on 2026-11-08 at 17:47 with unpinned latest versions (Python 3.10.12, numpy 2.2.6, numba 0.68.1, scipy 1.15.3, pandas 2.3.3, statsmodels 0.15.0)
   and the CCD started at 18:06; it was most likely run in it, though nothing in the output records the environment. Whether this changed any result is not known. A cheap check: re-run a handful of seed-0 CCD runs in the pinned
   environment and compare best costs (the screening reproduced on a second machine in 959 of 960 runs). Either way, run stage 2 in the same environment as the CCD, and note it.
3. **Keep the first recommendation.** The stage-2 refit overwrites `recommended_params.json`. Copy it now (for example to `recommended_params_ccd1.json`) so the CCD-only optimum survives for comparison.
4. **Commit** `ccd/records.csv`, `curves.csv`, `analysis_ccd/` and `recommended_params.json`; they are modified or untracked. `run_doe.py` (`STAGE`) and `analyze.py` (`ANALYSIS`)
   are also edited; `run_doe.py` is already set to `anchor_tune`.

## 10. Decision

- **Run stage 2 (`anchor_tune`).** It runs the recommended settings at ten times the budget, two seeds, on the 20 instances: about 40 runs, roughly 3 h here. It does not depend on where exactly the optimum is
  (any plateau point does the job) and it is what lifts the reference off the floor that compresses the best configurations (section 2).
- **Then refit (`ANALYSIS = "ccd"`) and read three things before stage 3:** (a) does the quality optimum of C and B stay interior, or does the unfloored log gap now pull C below 10 / B below 0.001;
  (b) does D at +1 become significant on the congested instances; (c) does the plateau shrink, so that one corner can now be told from the rest.
- **A second CCD (`ccd2`, new coding, `FACTOR_SETS`, runbook section 5) is warranted only if (a) or (b) answers yes.** The natural extension is C below 10 and D above 24 on the congested instances, restricted to the five hard c = 1.0 instances plus a few
  of the others, which is far cheaper than a second full CCD. Do not edit `FACTORS_CCD`.
- **Do not run `"ccd"` analysis after stage 3** (check runs enter the reference); stages 3–5 stand as in the runbook.
- **Unchanged from the screening's plan:** F stays at 4 and G at 1000 until the check stage; this CCD says nothing about them.

## 11. Files

`results/analysis_ccd/`: `ccd_summary.json` (fit, optimum, predictions), `anova_*.csv` and `coefficients_*.csv` (naive tests; use section 3 for inference), `control_x_noise.csv` (instance-clustered),
`optimum_by_class.csv`, `runs_with_responses.csv` (the data behind section 3 to 7), `plots/` (contours, surfaces, perturbation, coefficients, residuals, `optimum_by_size.png`, `optimum_by_class.png`).
