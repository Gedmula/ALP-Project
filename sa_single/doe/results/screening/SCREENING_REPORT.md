# Screening phase: results at 3 seeds

Ibrahim Oyeyinka, 2026-11-02.

The screening phase of the SA design of experiments (design.py:
2^(7-3) resolution IV plus 4 centre points, 20 configurations) run on
the large tuning set (instances/tune_large: 48 instances, sizes 50 to
500, congestion 0.6, 0.8, 1.0, wide and tight windows) at the full
3 seeds of the design: 20 x 48 x 3 = 2880 runs. Protocol per
run_doe.py: run seed = 1000*s + config index, budget 5000 moves per
aircraft per replica, 10 replicas, randomised run order
(ORDER_SEED 2026).

## Start selection

These runs use the reconstructed start selection: pools of 8 from
each of the four constructors (TC-RBI, ETRI, SBC, RGI) plus the
target-order sequence when feasible, deduplicated, sorted by exact
cost, and the best 5 plus the worst 5 taken as the replica starts
(sa_single/search.py, build_starts). This replaces the earlier
best-10 selection, which drew all 10 starts from the cheap end of the
pool and gave the replicas little diversity.

## Execution record

- Environment: Ubuntu 24.04 under WSL2, Python 3.12.3, pinned
  requirements (numpy 1.26.4, numba 0.60.0, scipy 1.11.4,
  pandas 2.1.4, highspy 1.15.1), 12 logical processors.
- Before the campaign, a reference run reproduced the timing pilot
  exactly (g50_c8_w, centre point, seed 0: start best 3374.99,
  SA best 3297.48, audit PASS).
- The campaign was interrupted once and resumed from the records
  checkpoint (377 runs done); no run was repeated or lost, and no
  run errored.
- Outcome: 2880 of 2880 runs recorded, 2880 unique run IDs, every
  run audit PASS, every run ending on the iteration budget. Total
  recorded solve time 41.0 h (mean 51.3 s per run).
- Cross-check: seed 0 of this campaign was also run on a second
  machine under an identical protocol. Start costs agree on all
  960 runs and best costs on 959; the single differing run
  (S10 on g500_c10_w) found a better value here (83583.84 against
  83629.61, both audited feasible), consistent with a rare
  timing-dependent exchange in the parallel tempering on the
  largest instance class.

## Findings

Main effects from the records (blocked by instance and seed;
responses as defined in analyze.py). On log gap, negative is better:

- D, move window: effect -0.206 (t -4.9). The larger window (16)
  is better.
- C, exchange interval: +0.103 (t +5.7). The smaller interval (10,
  exchange often) is better.
- A, hot acceptance: +0.093 (t +5.9). The lower value (0.30) is
  better.
- B, cold acceptance: small and mixed on final quality, a clear
  cost on speed (time-to-best fraction t +6.0). The lower value
  (0.001) is favoured on desirability.
- E, guide: near-inert. F, mutation size, and G, stagnation: no
  meaningful effect at 3 seeds.
- Curvature is present: centre points beat the factorial average
  on log gap (-0.124, se 0.028) and desirability (+0.043,
  se 0.008).
- Best single configuration: S09 (A low, B low, C low, D high),
  mean gap 0.108% over its 144 runs. The four centre
  configurations rank 3, 4, 6 and 9 of 20.
- In 1262 of 2880 runs (43.8%) the SA never improved on the best
  construction start, so on those instance-configuration pairs the
  budget bought nothing over the starts.

## Next steps

Foldover or a face-centred CCD on the active factors A, B, C, D,
then confirmation of the recommended parameters on
instances/holdout_large. Open item: analyze.py as committed raises
in current patsy, because its model formula term C(instance)
collides with the factor column C; the effects above were computed
with the equivalent blocked calculation.
