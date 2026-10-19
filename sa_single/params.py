"""
params.py — SA parameters and the design-of-experiments factor definitions
===============================================================================
The seven DOE factors (A–G) are the only tuned settings.  Everything else is
a constant of the method.  Coded levels −1 / 0 / +1 map to natural values on
a linear or logarithmic scale; the centre point (all zeros) is the default
used until the DOE recommends settings.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Dict, Optional


@dataclass
class SAParams:
    # ── DOE factors (defaults = coded centre point) ──────────────────────────
    hot_accept:              float = 0.55       # A: P(accept a median uphill move) at the hottest replica
    cold_accept:             float = 0.007071   # B: P(accept a 5th-percentile uphill move) at the coldest replica
    exchange_per_aircraft:   float = 44.72      # C: moves per replica between exchange rounds, per aircraft
    window:                  int   = 10         # D: maximum positions an aircraft moves
    guide:                   float = 0.725      # E: share of shadow-price-guided moves
    mutation_size:           int   = 10         # F: aircraft removed by ruin-and-recreate
    stagnation_per_aircraft: float = 316.2      # G: moves per aircraft without global improvement before reseeding

    # ── budget ───────────────────────────────────────────────────────────────
    budget_per_aircraft:     float = 5000.0     # iterations (moves per replica) per aircraft (DOE budget)
    budget_a:          Optional[float] = None   # production rule from the DOE: iterations = a · n^b
    budget_b:                float = 1.0
    time_limit:              float = 3600.0     # hard wall-clock cap, seconds

    def iteration_budget(self, n: int) -> int:
        """Moves per replica for an instance of n aircraft."""
        if self.budget_a is not None:
            return int(round(self.budget_a * n ** self.budget_b))
        return int(round(self.budget_per_aircraft * n))

    # ── constants of the method (not tuned) ──────────────────────────────────
    replicas:          Optional[int] = None     # None = logical CPUs − 2
    p_insert:          float = 0.45             # move mix: insert / swap / block(2–3) = 0.45 / 0.35 / 0.20
    p_swap:            float = 0.35
    hot_mutation_prob: float = 0.05             # per round, ruin-and-recreate the hottest replica
    archive_size:      int   = 10               # elite schedules kept for reseeding
    lp_checks:         int   = 16               # flagged moves re-priced by LP per batch (coldest replica)
    calib_samples:     int   = 2000             # uphill moves sampled to calibrate the temperature ladder
    hot_quantile:      float = 0.50             # uphill-move quantile that A refers to (typical move)
    cold_quantile:     float = 0.05             # uphill-move quantile that B refers to (small refining move)

    def to_json(self, path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def from_json(cls, path) -> "SAParams":
        d = json.loads(Path(path).read_text())
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass(frozen=True)
class Factor:
    code: str          # letter used in the design (A–G)
    name: str          # SAParams field
    low: float         # natural value at coded −1
    high: float        # natural value at coded +1
    scale: str         # "lin" or "log"
    integer: bool
    label: str         # axis label for plots

    def natural(self, coded: float):
        if self.scale == "log":
            lo, hi = math.log10(self.low), math.log10(self.high)
            v = 10 ** (lo + (coded + 1) / 2 * (hi - lo))
        else:
            v = self.low + (coded + 1) / 2 * (self.high - self.low)
        return int(round(v)) if self.integer else float(v)

    def coded(self, natural: float) -> float:
        if self.scale == "log":
            lo, hi = math.log10(self.low), math.log10(self.high)
            return 2 * (math.log10(natural) - lo) / (hi - lo) - 1
        return 2 * (natural - self.low) / (self.high - self.low) - 1


FACTORS = [
    Factor("A", "hot_accept",              0.30,  0.80,  "lin", False, "hot-end acceptance"),
    Factor("B", "cold_accept",             0.001, 0.05,  "log", False, "cold-end acceptance"),
    Factor("C", "exchange_per_aircraft",   10.0,  200.0, "log", False, "exchange interval (moves/aircraft)"),
    Factor("D", "window",                  4,     16,    "lin", True,  "move window (positions)"),
    Factor("E", "guide",                   0.50,  0.95,  "lin", False, "guided-move share"),
    Factor("F", "mutation_size",           4,     16,    "lin", True,  "mutation size (aircraft)"),
    Factor("G", "stagnation_per_aircraft", 100.0, 1000.0, "log", False, "stagnation threshold (moves/aircraft)"),
]
FACTOR_BY_CODE: Dict[str, Factor] = {f.code: f for f in FACTORS}


def decode(coded: Dict[str, float], base: Optional[SAParams] = None) -> SAParams:
    """SAParams with each coded factor (letter → level) set; unspecified factors keep `base`."""
    p = base or SAParams()
    return replace(p, **{FACTOR_BY_CODE[c].name: FACTOR_BY_CODE[c].natural(v) for c, v in coded.items()})


def encode(p: SAParams) -> Dict[str, float]:
    """Coded levels of the seven factors in `p`."""
    return {f.code: f.coded(getattr(p, f.name)) for f in FACTORS}
