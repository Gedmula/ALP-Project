"""
stages.py — the run configuration of every DOE stage
===============================================================================
One entry per stage; run_doe.py runs the stage named in its STAGE setting
(DRY_RUN = True there prints the design, instances, run count and time estimate
and runs nothing):

    python sa_single/doe/run_doe.py

Fields
    phase          design family: screening | foldover | ccd | check | anchor | confirm
    out            results folder (sa_single/doe/results/<out>); analyze.py reads these names
    coding         factor coding of the design (sa_single.params.FACTOR_SETS): "screening" or "ccd"
    instance_set   tune_large | holdout_large   (48 instances each, sizes 50 … 500)
    subset         None = every instance of the set, "informative" = the 20 tune_large instances on which the
                   screening runs did not all end on the same cost (instances.informative_instances)
    n_seeds        independent runs per (configuration, instance)
    ccd_factors    factors varied in the CCD (the rest sit at ccd_fixed)
    ccd_fixed      coded levels of the factors held fixed in the CCD
    n_center       centre-point configurations of the CCD
    fold_factor    factor whose sign is reversed in the foldover
    check_moves    check phase: configurations = the recommended optimum plus one configuration per entry,
                   each entry giving the coded levels to move (coding "ccd")
    anchor_mult    budget multiplier of the anchor runs (budget_per_aircraft x anchor_mult)
    needs_recommended   the stage reads results/recommended_params.json (written by the "ccd" analysis)

Everything else (budget of 5000 moves per aircraft per replica, replicas = logical CPUs - 2, run seed
1000 x seed + configuration index, run order seeded with ORDER_SEED) is the protocol of the screening and is kept.
"""
from __future__ import annotations

ORDER_SEED = 2026

STAGES = {
    # ── done ─────────────────────────────────────────────────────────────────
    "screening": dict(
        phase="screening", out="screening", coding="screening", instance_set="tune_large", subset=None, n_seeds=3,
        note="2^(7-3) resolution IV + 4 centre points. Completed: results/screening (2880 runs)."),

    # ── optional: only if the chain AF = BG = DE has to be separated before the CCD ──────────────────────────────
    "foldover": dict(
        phase="foldover", out="foldover", coding="screening", instance_set="tune_large", subset="informative",
        n_seeds=3, fold_factor="E",
        note="Single-factor foldover of the screening fraction on E (separates DE from AF = BG). Not recommended "
             "as the next step: see SCREENING_ANALYSIS.md section 8."),

    # ── next stage ───────────────────────────────────────────────────────────
    "ccd": dict(
        phase="ccd", out="ccd", coding="ccd", instance_set="tune_large", subset="informative", n_seeds=3,
        ccd_factors=["A", "B", "C", "D", "E"], ccd_fixed={"F": -1.0, "G": 1.0}, n_center=6,
        note="Face-centred CCD on A-E (16 half-fraction corners, E = ABCD, + 10 axial + 6 centre = 32 configurations) "
             "with F = 4 and G = 1000 moves/aircraft fixed; window coded 8 / 16 / 24. 20 informative instances."),

    # ── reference values: long runs at the current recommended settings ──────────────────────────────────────
    "anchor_tune": dict(
        phase="anchor", out="anchor_tune", coding="ccd", instance_set="tune_large", subset="informative", n_seeds=2,
        anchor_mult=10,
        note="10x budget at results/recommended_params.json (defaults if absent) on the 20 informative instances; "
             "strengthens the reference cost that every gap is measured against."),

    # ── after the CCD analysis ───────────────────────────────────────────────
    "check": dict(
        phase="check", out="check", coding="ccd", instance_set="tune_large", subset="informative", n_seeds=3,
        check_moves=[{"F": 0.0}, {"F": 1.0}, {"G": -1.0}], needs_recommended=True,
        note="Is the F and G choice neutral at the optimum? Optimum, then F -> 10, F -> 16, G -> 100 one at a time."),

    "confirm": dict(
        phase="confirm", out="confirm", coding="ccd", instance_set="holdout_large", subset=None, n_seeds=3,
        needs_recommended=True,
        note="Recommended settings on the 48 unseen holdout_large instances (all of them, flat ones included)."),

    "anchor_holdout": dict(
        phase="anchor", out="anchor_holdout", coding="ccd", instance_set="holdout_large", subset=None, n_seeds=2,
        anchor_mult=10, needs_recommended=True,
        note="10x budget at the recommended settings on holdout_large: reference values for the confirmation and "
             "the data for the iteration-budget rule (the budget analysis)."),
}

ORDER = ["screening", "ccd", "anchor_tune", "check", "confirm", "anchor_holdout"]   # the planned sequence
