

"""
mr_alp — Multi-Runway Aircraft Landing Problem solver package.

    Public API re-exports for interactive / notebook use.
 """
from mr_alp.config        import KNOWN_OPTIMA, INSTANCE_RUNWAYS
from mr_alp.models        import Instance, HeuristicParams, MRSAParams
from mr_alp.instance      import load_instance, surrogate_times, runway_feasible
from mr_alp.lp            import stage2_lp_objective, verify_and_exact_obj
from mr_alp.construction  import ramp_rbi, _build_seed_portfolio
from mr_alp.proxy         import total_target_conflict, compute_proxy
from mr_alp.solver        import ms_mr_sa
from mr_alp.output        import (
    print_mr_result, print_summary_table,
    save_run_results, generate_plots,
)

__all__ = [
    "KNOWN_OPTIMA", "INSTANCE_RUNWAYS",
    "Instance", "HeuristicParams", "MRSAParams",
    "load_instance", "surrogate_times", "runway_feasible",
    "stage2_lp_objective", "verify_and_exact_obj",
    "ramp_rbi", "_build_seed_portfolio",
    "total_target_conflict", "compute_proxy",
    "ms_mr_sa",
    "print_mr_result", "print_summary_table",
    "save_run_results", "generate_plots",
]