#!/usr/bin/env bash
# Multi-seed benchmark of the exact-objective SA on every multi-runway job with
# a nonzero best known value. Cold starts only (construction seeds, no warm start).
set -euo pipefail
cd "$(dirname "$0")"
OUT=${OUT:-exact_sa_bench}
SEEDS=${SEEDS:-"0 1 2 3 4"}
COMMON="--chi0 0.05 --restart-frac 0.5 --window 8 --workers 4 --out $OUT"

SMALL="airland1:2 airland2:2 airland3:2 airland4:2 airland4:3 airland5:2 airland5:3 airland6:2 airland8:2"
MID="airland9:2 airland9:3 airland10:2 airland10:3 airland10:4 airland11:2 airland11:3 airland11:4 airland12:2 airland12:3 airland12:4"
LARGE="airland13:2 airland13:3 airland13:4"

for s in $SEEDS; do
  python -u run_exact_sa.py $SMALL --t 60  --cycle 20  --seed $((s * 1000)) $COMMON
  python -u run_exact_sa.py $MID   --t 300 --cycle 75  --seed $((s * 1000)) $COMMON
  python -u run_exact_sa.py $LARGE --t 600 --cycle 150 --seed $((s * 1000)) $COMMON
done
