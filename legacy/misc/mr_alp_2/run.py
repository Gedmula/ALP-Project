"""
run.py — Experiment driver for the hybrid MR-ALP solver.

Each OR-Library instance is solved once per entry in its INSTANCE_RUNWAYS
list, so airland1 produces three result rows (m=1, m=2, m=3) and airland13
produces five.  After each (instance, m) solve, the gap to the known
reference objective is computed immediately and printed.  All gaps are
stored in the result dict and written to summary.csv.

Usage
─────
    python -m mr_alp.run
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

from misc.mr_alp_2.config import KNOWN_OPTIMA, INSTANCE_RUNWAYS, OUTPUT_DIR
from misc.mr_alp_2.models import HybridParams
from misc.mr_alp_2.instance import load_orlib, synthetic_instance
from misc.mr_alp_2.solver import solve
from misc.mr_alp_2.output import (
    compute_gap,
    print_gap_report,
    print_summary_table,
    save_results,
    plot_gantt,
    plot_penalty_profile,
)


DATA_DIR = Path(__file__).parent.parent / "data"

OR_FILES: Dict[str, str] = {
    "airland1":  "airland1.txt",
    "airland2":  "airland2.txt",
    "airland3":  "airland3.txt",
    "airland4":  "airland4.txt",
    "airland5":  "airland5.txt",
    "airland6":  "airland6.txt",
    "airland7":  "airland7.txt",
    "airland8":  "airland8.txt",
    "airland9":  "airland9.txt",
    "airland10": "airland10.txt",
    "airland11": "airland11.txt",
    "airland12": "airland12.txt",
    "airland13": "airland13.txt",
}


def run_all(
    t_limit:   float = 3600.0,
    n_outer:   int   = 200,
    save_dir:  str   = str(OUTPUT_DIR / "plots"),
    out_dir:   str   = str(OUTPUT_DIR),
    seed:      int   = 0,
    n_workers: int   = None,
) -> List[dict]:
    """
    Run the hybrid solver on every (instance, runway-count) combination.

    After each solve the gap to the known reference objective is computed,
    printed immediately, and stored in the result dict under 'gap_pct' and
    'gap_abs'.  The final summary table and CSV include all gap columns.

    Parameters
    ----------
    t_limit   : wall-clock budget per (instance, m) run in seconds.
    n_outer   : maximum Phase 5 inter-runway improvement iterations.
    save_dir  : directory for plot files.
    out_dir   : directory for CSV/JSON/text export files.
    seed      : master RNG seed forwarded to solve().
    n_workers : worker process count; None defers to config.N_WORKERS.

    Returns
    -------
    List of result dicts, one per (instance, m) pair attempted.
    """
    from misc.mr_alp_2.config import N_WORKERS
    if n_workers is None:
        n_workers = N_WORKERS

    params  = HybridParams()
    results = []

    print(f"\nSearching for OR-Library files in: {DATA_DIR.resolve()}\n")
    found, missing = [], []
    for name, fname in OR_FILES.items():
        p = DATA_DIR / fname
        if p.exists():
            found.append((name, p))
            print(f"  [FOUND]   {fname}")
        else:
            missing.append(fname)
            print(f"  [MISSING] {fname}")

    if missing:
        print(f"\n  {len(missing)} file(s) not found — skipped.\n")

    for name, path in found:
        runway_configs: List[int] = INSTANCE_RUNWAYS.get(name, [1])

        for m in runway_configs:
            opt       = KNOWN_OPTIMA.get(name, {}).get(m)
            run_label = f"{name}_m{m}"

            print(f"\n{'═'*64}")
            print(f"  {name}   m={m}   "
                  f"known_opt={'N/A' if opt is None else f'{opt:.4f}'}")
            print(f"{'═'*64}")

            try:
                inst = load_orlib(str(path), m=m, name=run_label)
                res  = solve(
                    inst,
                    params=params,
                    n_outer_iters=n_outer,
                    seed=seed,
                    t_limit=t_limit,
                    n_workers=n_workers,
                    verbose=True,
                )

                # ── Compute and store gap immediately after solve ─────────
                lp      = res.get('lp_obj', math.inf)
                gap_pct = compute_gap(lp, opt)
                gap_abs = ((lp - opt)
                           if opt is not None and not math.isinf(lp)
                           else float('nan'))

                res['name']      = run_label
                res['n']         = inst.n
                res['m']         = m
                res['known_opt'] = opt
                res['gap_pct']   = gap_pct
                res['gap_abs']   = gap_abs

                # ── Print gap report immediately ───────────────────────────
                print_gap_report(res, inst=inst)

                # ── Plots (gap included in title) ─────────────────────────
                if res['seqs'] is not None and res['x'] is not None:
                    plot_gantt(
                        res['seqs'], inst,
                        x=res['x'],
                        obj=lp,
                        known_opt=opt,
                        save_dir=save_dir,
                    )
                    plot_penalty_profile(
                        res['seqs'], inst,
                        res['x'],
                        obj=lp,
                        known_opt=opt,
                        save_dir=save_dir,
                    )

                results.append(res)

            except Exception as exc:
                print(f"  ERROR on {run_label}: {exc}\n")

    if not found:
        print("No OR-Library files found — running synthetic demo.\n")
        inst = synthetic_instance(n=15, m=2, seed=0)
        res  = solve(
            inst,
            params=params,
            n_outer_iters=50,
            seed=0,
            t_limit=120.0,
            n_workers=n_workers,
            verbose=True,
        )
        lp = res.get('lp_obj', math.inf)
        res.update(
            name=inst.name, n=inst.n, m=inst.m,
            known_opt=None,
            gap_pct=float('nan'),
            gap_abs=float('nan'),
        )
        print_gap_report(res)
        if res['seqs'] is not None and res['x'] is not None:
            plot_gantt(res['seqs'], inst, x=res['x'],
                       obj=lp, save_dir=save_dir)
            plot_penalty_profile(res['seqs'], inst, res['x'],
                                 obj=lp, save_dir=save_dir)
        results.append(res)

    if results:
        print_summary_table(results)
        save_results(results, out_dir=out_dir)

    return results


if __name__ == "__main__":
    run_all(
        t_limit=3600.0,
        n_outer=200,
        save_dir=str(OUTPUT_DIR / "plots"),
        out_dir=str(OUTPUT_DIR),
    )