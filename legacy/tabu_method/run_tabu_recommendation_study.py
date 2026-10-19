"""Run a resumable matched-seed study of the tabu recommendations."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_CASES = (("airland11", 2), ("airland12", 2))

ARMS = (
    {
        "name": "tabu_off",
        "tabu": "0",
        "tenure": 80,
        "mode": "fixed_attribute",
        "aspiration": "proxy",
        "fallback": "unfiltered",
    },
    {
        "name": "fixed_t20",
        "tabu": "1",
        "tenure": 20,
        "mode": "fixed_attribute",
        "aspiration": "proxy",
        "fallback": "unfiltered",
    },
    {
        "name": "fixed_t40",
        "tabu": "1",
        "tenure": 40,
        "mode": "fixed_attribute",
        "aspiration": "proxy",
        "fallback": "unfiltered",
    },
    {
        "name": "fixed_t80",
        "tabu": "1",
        "tenure": 80,
        "mode": "fixed_attribute",
        "aspiration": "proxy",
        "fallback": "unfiltered",
    },
    {
        "name": "fixed_t160",
        "tabu": "1",
        "tenure": 160,
        "mode": "fixed_attribute",
        "aspiration": "proxy",
        "fallback": "unfiltered",
    },
    {
        "name": "iteration_t80",
        "tabu": "1",
        "tenure": 80,
        "mode": "iteration",
        "aspiration": "proxy",
        "fallback": "unfiltered",
    },
    {
        "name": "fixed_t80_lp_aspiration",
        "tabu": "1",
        "tenure": 80,
        "mode": "fixed_attribute",
        "aspiration": "hybrid",
        "fallback": "unfiltered",
    },
    {
        "name": "fixed_t80_least_recent",
        "tabu": "1",
        "tenure": 80,
        "mode": "fixed_attribute",
        "aspiration": "proxy",
        "fallback": "least_recent",
    },
    {
        "name": "reactive_combined",
        "tabu": "1",
        "tenure": 80,
        "mode": "reactive",
        "aspiration": "hybrid",
        "fallback": "least_recent",
    },
)

BASE_ENV = {
    "ALP_N_CHAINS": "6",
    "ALP_N_WORKERS": "6",
    "ALP_USE_ALL_SEEDS": "0",
    "ALP_DELTA_EVAL": "1",
    "ALP_SA_CHAIN_DIVERSITY": "1",
    "ALP_LP_IMPACT_INIT": "1",
    "ALP_BOTTLENECK_LNS": "1",
    "ALP_ASSIGNMENT_REPAIR": "1",
    "ALP_SET_PARTITION_RECOMBINE": "1",
    "PYTHONUTF8": "1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[f"{name}:{m}" for name, m in DEFAULT_CASES],
        help="Case specifications such as airland11:2.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=[arm["name"] for arm in ARMS],
        default=[arm["name"] for arm in ARMS],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "tabu_recommendation_study",
    )
    return parser.parse_args()


def parse_case(value: str) -> tuple[str, int]:
    name, runway_text = value.rsplit(":", 1)
    return name, int(runway_text)


def run_one(
    out_root: Path,
    arm: dict,
    instance: str,
    runways: int,
    seed: int,
) -> dict:
    run_dir = out_root / arm["name"] / f"{instance}_m{runways}_seed{seed}"
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        return payload["results"][0]

    run_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(BASE_ENV)
    env.update({
        "ALP_SA_TABU": arm["tabu"],
        "ALP_SA_TABU_TENURE": str(arm["tenure"]),
        "ALP_SA_TABU_MODE": arm["mode"],
        "ALP_SA_TABU_ASPIRATION": arm["aspiration"],
        "ALP_SA_TABU_FALLBACK": arm["fallback"],
        "ALP_RUN_OUT": str(run_dir),
        "ALP_STUDY_INSTANCE": instance,
        "ALP_STUDY_M": str(runways),
        "ALP_STUDY_SEED": str(seed),
    })
    code = (
        "import os; "
        "from pathlib import Path; "
        "from mr_sa_alp import _run_one_job; "
        "from mr_alp.output import save_run_results; "
        "p=Path('data')/(os.environ['ALP_STUDY_INSTANCE']+'.txt'); "
        "r=_run_one_job(str(p), int(os.environ['ALP_STUDY_M']), "
        "seed=int(os.environ['ALP_STUDY_SEED'])); "
        "save_run_results([r], Path(os.environ['ALP_RUN_OUT']))"
    )
    with (run_dir / "stdout.log").open(
        "w", encoding="utf-8", errors="replace"
    ) as log:
        proc = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{arm['name']} {instance} m={runways} seed={seed} failed; "
            f"see {run_dir / 'stdout.log'}")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    return payload["results"][0]


def write_rows(out_root: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with (out_root / "study_rows.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out_root / "study_rows.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")


def load_existing_rows(out_root: Path) -> list[dict]:
    path = out_root / "study_rows.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_summary(out_root: Path, rows: list[dict]) -> None:
    groups: dict[tuple[str, int, str], list[dict]] = {}
    for row in rows:
        groups.setdefault(
            (row["instance"], int(row["m"]), row["arm"]), []
        ).append(row)

    summary = []
    for (instance, runways, arm), group in sorted(groups.items()):
        gaps = [float(row["gap_pct"]) for row in group]
        runtimes = [float(row["runtime_s"]) for row in group]
        summary.append({
            "instance": instance,
            "m": runways,
            "arm": arm,
            "n_seeds": len(group),
            "mean_gap_pct": round(statistics.mean(gaps), 6),
            "median_gap_pct": round(statistics.median(gaps), 6),
            "best_gap_pct": round(min(gaps), 6),
            "stdev_gap_pct": (
                round(statistics.stdev(gaps), 6) if len(gaps) > 1 else 0.0
            ),
            "mean_runtime_s": round(statistics.mean(runtimes), 3),
            "all_feasible": all(row["feasible"] for row in group),
        })
    if summary:
        with (out_root / "study_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        (out_root / "study_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    cases = [parse_case(value) for value in args.cases]
    selected_arms = [arm for arm in ARMS if arm["name"] in args.arms]
    out_root = args.output.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    rows = load_existing_rows(out_root)
    existing_keys = {
        (row["instance"], int(row["m"]), int(row["seed"]), row["arm"])
        for row in rows
    }

    total = len(cases) * len(args.seeds) * len(selected_arms)
    completed = 0
    for instance, runways in cases:
        for seed in args.seeds:
            for arm in selected_arms:
                completed += 1
                print(
                    f"[{completed}/{total}] {instance} m={runways} "
                    f"seed={seed} arm={arm['name']}",
                    flush=True,
                )
                started = time.perf_counter()
                result = run_one(
                    out_root, arm, instance, runways, seed)
                tabu = result.get("tabu", {}).get("best_chain", {})
                row = {
                    "instance": instance,
                    "m": runways,
                    "seed": seed,
                    "arm": arm["name"],
                    "tenure": arm["tenure"],
                    "mode": arm["mode"],
                    "aspiration": arm["aspiration"],
                    "fallback": arm["fallback"],
                    "objective": result["sa_lp"],
                    "bks": result["bks"],
                    "gap_pct": result["gap_pct"],
                    "feasible": result["feasible"],
                    "runtime_s": result["timing"]["total_s"],
                    "best_iter_done": result["iterations"]["best_chain_done"],
                    "best_iter_budget": result["iterations"]["best_chain_budget"],
                    "blocked_candidates": tabu.get("blocked_candidates", 0),
                    "cycles_detected": tabu.get("cycles_detected", 0),
                    "lp_aspirations": tabu.get("lp_aspirations", 0),
                    "fallback_uses": tabu.get("fallback_uses", 0),
                    "tenure_final": tabu.get("tenure_final", arm["tenure"]),
                }
                key = (instance, runways, seed, arm["name"])
                if key in existing_keys:
                    rows = [
                        old for old in rows
                        if (
                            old["instance"],
                            int(old["m"]),
                            int(old["seed"]),
                            old["arm"],
                        ) != key
                    ]
                rows.append(row)
                existing_keys.add(key)
                write_rows(out_root, rows)
                write_summary(out_root, rows)
                print(
                    f"  gap={row['gap_pct']:.4f}% "
                    f"runtime={row['runtime_s']:.1f}s "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )


if __name__ == "__main__":
    main()
