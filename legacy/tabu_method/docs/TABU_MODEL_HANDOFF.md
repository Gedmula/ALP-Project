# Tabu Memory Model for the MR-ALP Solver

## Purpose

This change adds short-term tabu memory to the existing multi-chain simulated
annealing (SA) solver. It is not a separate Tabu Search algorithm. SA still
generates, ranks, and probabilistically accepts candidate schedules; the tabu
component prevents recently accepted cross-runway assignments from being
immediately reversed.

The experimental objective is to reduce cycling while preserving SA's ability
to accept worsening moves.

## Solution Representation

A solution is a list of runway sequences:

```text
S = [S_0, S_1, ..., S_(m-1)]
```

where `S_r` is the ordered list of aircraft assigned to runway `r`.

Let `rho_S(i)` denote the runway assigned to aircraft `i` in solution `S`.
For a transition from current solution `S` to candidate `S'`, the set of
cross-runway assignment moves is

```text
M(S, S') = {(i, rho_S(i), rho_S'(i)) : rho_S(i) != rho_S'(i)}.
```

Changes in landing order within the same runway are not stored in tabu memory.

## Tabu Attribute

Each tabu attribute is a directed aircraft transfer:

```text
(aircraft, source_runway, destination_runway)
```

If an accepted move sends aircraft `i` from runway `a` to runway `b`, the
reverse transfer `(i, b, a)` is inserted into tabu memory. This prevents the
search from immediately sending the aircraft back to its previous runway.

For an accepted transition containing several cross-runway transfers, one
reverse attribute is stored for every transferred aircraft.

## Candidate Admissibility

A candidate `S'` is tabu if any assignment move in `M(S, S')` appears in the
current tabu set:

```text
tabu(S, S') = any(move in TabuSet for move in M(S, S')).
```

The candidate is admissible when either:

1. none of its cross-runway assignment moves is tabu; or
2. the aspiration rule is satisfied.

The aspiration rule permits a tabu candidate when its proxy objective improves
the best proxy objective found by the current SA chain:

```text
Proxy(S') < BestProxy - 1e-9.
```

This aspiration test uses the fast proxy objective, not the exact LP objective.

If filtering would remove every candidate in the generated pool, the current
implementation keeps the unfiltered pool. This is a deadlock escape: tabu
status becomes advisory when no admissible candidate was generated.

## Tenure

The default capacity is:

```text
SA_TABU_TENURE = 80
```

Tabu memory is implemented with:

- a FIFO deque for expiration order;
- a set for constant-time membership checks.

When the deque exceeds the configured capacity, the oldest attribute is
removed. Duplicate attributes can occur in the deque; an attribute is removed
from the set only when no duplicate remains in the deque.

Important interpretation: the current tenure is measured in stored transfer
attributes, not SA iterations. A move transferring four aircraft consumes four
of the 80 entries and therefore shortens the effective memory horizon.

## Interaction with Simulated Annealing

At every SA iteration:

1. Generate `R` candidate schedules using the existing neighborhood operators.
2. Evaluate each candidate with the proxy objective.
3. Sort the candidate pool by proxy objective.
4. Remove tabu candidates unless aspiration applies.
5. Select the best remaining candidate with probability 0.80; otherwise select
   randomly from the best five remaining candidates.
6. Apply the existing SA acceptance rule:

```text
delta = Proxy(S') - Proxy(S)
accept if delta <= 0
otherwise accept with probability exp(-delta / T)
```

7. If accepted, add the reverse cross-runway transfers to tabu memory.

Tabu filtering happens before SA acceptance. Therefore, SA can still accept a
worsening admissible move, but normally cannot accept a tabu move.

## Pseudocode

```text
TabuQueue <- empty FIFO queue
TabuSet   <- empty set

for each SA iteration:
    Pool <- GenerateAndEvaluateCandidates(CurrentSolution)

    Allowed <- empty list
    for Candidate in Pool:
        Moves <- CrossRunwayMoves(CurrentSolution, Candidate)
        IsTabu <- any(Move in TabuSet for Move in Moves)

        if not IsTabu or Proxy(Candidate) < BestProxy - epsilon:
            add Candidate to Allowed

    if Allowed is not empty:
        Pool <- Allowed

    Candidate <- SelectBySACompetition(Pool)

    if SimulatedAnnealingAccepts(Candidate):
        Moves <- CrossRunwayMoves(CurrentSolution, Candidate)
        CurrentSolution <- Candidate

        for (Aircraft, OldRunway, NewRunway) in Moves:
            Reverse <- (Aircraft, NewRunway, OldRunway)
            push Reverse onto TabuQueue
            add Reverse to TabuSet

            while size(TabuQueue) > TabuTenure:
                Expired <- pop oldest from TabuQueue
                if Expired no longer occurs in TabuQueue:
                    remove Expired from TabuSet
```

## Configuration

The feature can be controlled without modifying source code:

```powershell
$env:ALP_SA_TABU = "1"          # enable
$env:ALP_SA_TABU_TENURE = "80" # memory capacity
```

Disable it for an A/B run with:

```powershell
$env:ALP_SA_TABU = "0"
```

The defaults are defined in `mr_alp/config.py`. The implementation is in
`mr_alp/sa.py`, mainly `_assignment_moves()` and `run_mr_sa()`.

## Current Evidence

The same current solver configuration was tested with tabu off and on for all
seven cases having a positive gap in the cloned baseline results.

| Instance | Baseline gap | Current, tabu off | Current, tabu on | Effect of tabu |
|---|---:|---:|---:|---:|
| airland9, m=2 | 1.9860% | 2.0896% | 2.0896% | tie |
| airland10, m=2 | 1.0667% | 1.8012% | 1.1970% | -0.6042 pp |
| airland11, m=2 | 1.9047% | 2.2834% | 1.7146% | -0.5688 pp |
| airland12, m=2 | 5.1916% | 6.5705% | 3.3410% | -3.2295 pp |
| airland12, m=3 | 3.1175% | 2.8878% | 2.8878% | tie |
| airland13, m=2 | 3.3117% | 4.8904% | 3.8627% | -1.0277 pp |
| airland13, m=3 | 4.9047% | 5.4775% | 5.7401% | +0.2626 pp |

Across these runs, the mean gap changed from 3.7143% without tabu to 2.9761%
with tabu. Tabu improved four cases, tied two, and worsened one. These are
single-seed comparisons, so they support continued testing but are not yet a
statistical conclusion.

## Review Questions and Proposed Next Tests

1. Should tenure represent accepted SA iterations instead of stored aircraft
   transfers? Iteration-based expiration would give tenure a stable meaning
   across single-aircraft and block moves.
2. Should the aspiration rule use the best exact LP objective when an LP value
   is available, rather than only the proxy objective?
3. Should the all-candidates-tabu fallback select the least-recent tabu move,
   instead of restoring the whole unfiltered pool?
4. Should tenure be reactive, increasing during repeated cycling and decreasing
   during stagnation or intensification?
5. Run at least three matched seeds for tenure values `{20, 40, 80, 160}` on
   airland11 m=2, airland12 m=2, airland13 m=2, and airland13 m=3. Report mean,
   median, best gap, runtime, and completed iterations.

The immediate recommendation is to retain tabu as an experimental feature, but
not claim that `tenure=80` is universally best. The strongest next improvement
is iteration-based or reactive tenure followed by the matched-seed experiment.

## Follow-up Results: Full 108-Run Matrix

The proposed variants were implemented behind environment flags:

```powershell
$env:ALP_SA_TABU_MODE = "fixed_attribute" # or iteration, reactive
$env:ALP_SA_TABU_ASPIRATION = "proxy"     # or lp, hybrid
$env:ALP_SA_TABU_FALLBACK = "unfiltered"  # or least_recent
```

The follow-up evidence changes the recommendation:

| Case | Control | Strongest tested setting | Result |
|---|---:|---:|---|
| airland11, m=2 | tabu off mean 2.4209% | fixed tenure 80 mean 1.8651% | retain fixed 80 |
| airland12, m=2 | tabu off mean 4.8449% | fixed tenure 20 mean 2.6549% | test fixed 20 further |
| airland13, m=2 | tabu off mean 4.1971% | fixed tenure 40 mean 3.8922% | retain fixed 40 |
| airland13, m=3 | tabu off mean 6.3708% | fixed tenure 40 mean 4.9680% | retain fixed 40 by mean |

All four cases use seeds 0, 1, and 2. Every saved schedule passed the exact LP
and independent sequence verifier.

Iteration-based tenure, LP-aware aspiration, least-recent fallback, and the
combined reactive variant did not win the mean gap on any case. The current
recommendation is therefore case-specific fixed tenure: fixed 80 for airland11,
fixed 20 for airland12, and fixed 40 for the airland13 mean-gap baseline.
