#!/usr/bin/env bash
# Exact SA on the Pinol-Beasley non-linear objective, instances without proven optima.
set -euo pipefail
cd "$(dirname "$0")"
OUT=${OUT:-nonlinear_bench}
SEEDS=${SEEDS:-"0 1 2 3 4"}
COMMON="--objective nonlinear --chi0 0.05 --restart-frac 0.5 --workers 4 --out $OUT"

SMALL="airland8:1 airland8:2"
MID="airland9:1 airland9:2 airland9:3 airland9:4 airland10:1 airland10:2 airland10:3 airland10:4
     airland11:1 airland11:2 airland11:3 airland11:4 airland11:5
     airland12:1 airland12:2 airland12:3 airland12:4 airland12:5"
LARGE="airland13:1 airland13:2 airland13:3 airland13:4 airland13:5"

for s in $SEEDS; do
  python -u run_exact_sa.py $SMALL --t 60  --cycle 20  --seed $((s * 1000)) $COMMON
  python -u run_exact_sa.py $MID   --t 300 --cycle 75  --seed $((s * 1000)) $COMMON
  python -u run_exact_sa.py $LARGE --t 600 --cycle 150 --seed $((s * 1000)) $COMMON
done
