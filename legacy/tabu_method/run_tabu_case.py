"""Run one reproducible MR-SA-Tabu experiment from the handoff folder."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "instance",
        help="Instance name such as airland12, or a path to an instance file.",
    )
    parser.add_argument("runways", type=int, help="Number of runways.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--tabu", choices=("on", "off"), default="on", help="Tabu memory.",
    )
    parser.add_argument(
        "--tenure", type=int, default=80, help="Tabu attribute capacity.",
    )
    parser.add_argument(
        "--mode",
        choices=("fixed_attribute", "iteration", "reactive"),
        default="fixed_attribute",
        help="How tabu tenure is measured and adjusted.",
    )
    parser.add_argument(
        "--aspiration", choices=("proxy", "lp", "hybrid"), default="proxy",
        help="Rule that can admit a tabu candidate.",
    )
    parser.add_argument(
        "--fallback", choices=("unfiltered", "least_recent"),
        default="unfiltered",
        help="Action when every generated candidate is tabu.",
    )
    parser.add_argument(
        "--chains", type=int, default=6, help="Number of SA chains.",
    )
    parser.add_argument(
        "--workers", type=int, default=7, help="Maximum worker processes.",
    )
    return parser.parse_args()


def _instance_path(value: str) -> Path:
    supplied = Path(value)
    if supplied.exists():
        return supplied.resolve()
    name = supplied.stem
    if not name.lower().startswith("airland"):
        raise FileNotFoundError(f"Unknown instance: {value}")
    candidate = Path(__file__).resolve().parent / "data" / f"{name}.txt"
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = _parse_args()
    if args.tenure <= 0:
        raise ValueError("--tenure must be positive")
    if args.runways <= 0 or args.chains <= 0 or args.workers <= 0:
        raise ValueError("runways, chains, and workers must be positive")

    # Set experiment controls before importing mr_alp.config.
    os.environ["ALP_SA_TABU"] = "1" if args.tabu == "on" else "0"
    os.environ["ALP_SA_TABU_TENURE"] = str(args.tenure)
    os.environ["ALP_SA_TABU_MODE"] = args.mode
    os.environ["ALP_SA_TABU_ASPIRATION"] = args.aspiration
    os.environ["ALP_SA_TABU_FALLBACK"] = args.fallback
    os.environ["ALP_N_CHAINS"] = str(args.chains)
    os.environ["ALP_N_WORKERS"] = str(args.workers)
    os.environ["ALP_USE_ALL_SEEDS"] = "0"
    os.environ["ALP_DELTA_EVAL"] = "1"
    os.environ["ALP_SA_CHAIN_DIVERSITY"] = "1"
    os.environ["ALP_LP_IMPACT_INIT"] = "1"
    os.environ["ALP_BOTTLENECK_LNS"] = "1"
    os.environ["ALP_ASSIGNMENT_REPAIR"] = "1"
    os.environ["ALP_SET_PARTITION_RECOMBINE"] = "1"

    from mr_sa_alp import _run_one_job

    path = _instance_path(args.instance)
    result = _run_one_job(str(path), args.runways, seed=args.seed)
    print(result["output"], end="")
    print("\nEXPERIMENT RESULT")
    print(f"instance       : {path.stem}")
    print(f"runways        : {args.runways}")
    print(f"seed           : {args.seed}")
    print(f"tabu           : {args.tabu}")
    print(f"tenure         : {args.tenure}")
    print(f"mode           : {args.mode}")
    print(f"aspiration     : {args.aspiration}")
    print(f"fallback       : {args.fallback}")
    print(f"objective      : {result['sa_lp']:.10f}")
    print(f"BKS            : {result['opt']}")
    print(f"feasible       : {result['feasible']}")
    print(f"runtime_s      : {result['time']:.3f}")
    print(f"iterations     : {result.get('iterations', 'see run metadata')}")


if __name__ == "__main__":
    main()
