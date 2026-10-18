"""
mr_alp — Hybrid Decomposition Solver for the Multi-Runway Aircraft Landing Problem.

Phase structure
───────────────
0  Instance preprocessing       models.py          Instance.__post_init__
1  Runway-assignment portfolio   construction.py    build_assignment_portfolio
                                                    build_seed_portfolio
2  Per-runway initial sequences  sequencing.py      generate_runway_sequences
3  Single-runway SA              sa_single.py       run_single_runway_sa
4  Stage-2 LP evaluation         lp.py              stage2_lp
5  Inter-runway improvement      inter_runway.py    inter_runway_improve
                                 move_pressure.py   compute_move_pressure
                                 operators.py       apply_op
                                 repair.py          lp_vnd_polish
                                 proxy.py           compute_proxy
6  Final verification            verify.py          verify_schedule

Entry point:  python -m mr_alp.run
"""
from misc.mr_alp_2.config       import KNOWN_OPTIMA, INSTANCE_RUNWAYS
from misc.mr_alp_2.models       import Instance, HybridParams, SingleSAParams, HeuristicParams, MRSAParams
from misc.mr_alp_2.instance     import load_orlib, synthetic_instance
from misc.mr_alp_2.surrogate    import surrogate_times, surrogate_penalty, insertion_cost
from misc.mr_alp_2.priority     import compute_priority_scores
from misc.mr_alp_2.construction import (
    build_assignment_portfolio,
    build_seed_portfolio,
    compute_priorities,
    ramp_rbi,
    seed_fcfs,
    seed_edd_balanced,
    seed_wedd,
    seed_atc,
    seed_atcs,
    seed_caf,
    seed_mpds,
    seed_wcc,
    seed_grasp,
)
from misc.mr_alp_2.sequencing   import generate_runway_sequences
from misc.mr_alp_2.sa_single    import run_single_runway_sa
from misc.mr_alp_2.lp           import stage2_lp, stage2_lp_objective, lp_recheck
from misc.mr_alp_2.proxy        import (
    runway_feasible,
    _rwy_proxy_components,
    init_proxy_arrays,
    compute_proxy,
    lp_impact_scores,
)
from misc.mr_alp_2.operators    import (
    MoveResult,
    select_op,
    apply_op,
    generate_candidate_pool,
    op_n1_adjacent_swap,
    op_n2_swap,
    op_n3b_best_insertion,
    op_n4_block_reloc,
    op_x1_transfer,
    op_x2_swap,
    op_x3_best_transfer,
    op_x4_block_transfer,
    op_x7_tc_repair,
    pick_aircraft_targeted,
)
from misc.mr_alp_2.repair       import (
    ElitePool,
    lp_guided_penalty_repair,
    lp_guided_pair_swap,
    target_conflict_repair,
    ejection_chain_transfer,
    lns_remove_reinsert,
    lp_vnd_polish,
    path_relink,
)
from misc.mr_alp_2.move_pressure import (
    compute_move_pressure,
    select_candidate_aircraft,
    AdaptiveWeightUpdater,
)
from misc.mr_alp_2.inter_runway import inter_runway_improve
from misc.mr_alp_2.solver       import solve
from misc.mr_alp_2.verify       import verify_schedule, VerificationReport
from misc.mr_alp_2.output       import (
    print_result,
    print_runway_assignment,
    print_summary_table,
    save_results,
    plot_gantt,
    plot_penalty_profile,
)

__all__ = [
    # ── Config ────────────────────────────────────────────────────────────
    "KNOWN_OPTIMA",
    "INSTANCE_RUNWAYS",

    # ── Data structures ───────────────────────────────────────────────────
    "Instance",
    "HybridParams",
    "SingleSAParams",
    "HeuristicParams",
    "MRSAParams",

    # ── Instance I/O ──────────────────────────────────────────────────────
    "load_orlib",
    "synthetic_instance",

    # ── Surrogate evaluation ──────────────────────────────────────────────
    "surrogate_times",
    "surrogate_penalty",
    "insertion_cost",

    # ── Priority measures ─────────────────────────────────────────────────
    "compute_priority_scores",
    "compute_priorities",

    # ── Construction — Tier A (assignment portfolio) ───────────────────────
    "build_assignment_portfolio",

    # ── Construction — Tier B (LP-screened seed portfolio) ────────────────
    "build_seed_portfolio",
    "ramp_rbi",
    "seed_fcfs",
    "seed_edd_balanced",
    "seed_wedd",
    "seed_atc",
    "seed_atcs",
    "seed_caf",
    "seed_mpds",
    "seed_wcc",
    "seed_grasp",

    # ── Sequencing ────────────────────────────────────────────────────────
    "generate_runway_sequences",

    # ── Single-runway SA ──────────────────────────────────────────────────
    "run_single_runway_sa",

    # ── LP evaluation ─────────────────────────────────────────────────────
    "stage2_lp",
    "stage2_lp_objective",
    "lp_recheck",

    # ── Proxy objective ───────────────────────────────────────────────────
    "runway_feasible",
    "_rwy_proxy_components",
    "init_proxy_arrays",
    "compute_proxy",
    "lp_impact_scores",

    # ── SA operators ──────────────────────────────────────────────────────
    "MoveResult",
    "select_op",
    "apply_op",
    "generate_candidate_pool",
    "op_n1_adjacent_swap",
    "op_n2_swap",
    "op_n3b_best_insertion",
    "op_n4_block_reloc",
    "op_x1_transfer",
    "op_x2_swap",
    "op_x3_best_transfer",
    "op_x4_block_transfer",
    "op_x7_tc_repair",
    "pick_aircraft_targeted",

    # ── LP-guided repair and VND ──────────────────────────────────────────
    "ElitePool",
    "lp_guided_penalty_repair",
    "lp_guided_pair_swap",
    "target_conflict_repair",
    "ejection_chain_transfer",
    "lns_remove_reinsert",
    "lp_vnd_polish",
    "path_relink",

    # ── Move pressure ─────────────────────────────────────────────────────
    "compute_move_pressure",
    "select_candidate_aircraft",
    "AdaptiveWeightUpdater",

    # ── Inter-runway improvement ──────────────────────────────────────────
    "inter_runway_improve",

    # ── Top-level solver ──────────────────────────────────────────────────
    "solve",

    # ── Verification ─────────────────────────────────────────────────────
    "verify_schedule",
    "VerificationReport",

    # ── Output ────────────────────────────────────────────────────────────
    "print_result",
    "print_runway_assignment",
    "print_summary_table",
    "save_results",
    "plot_gantt",
    "plot_penalty_profile",
]