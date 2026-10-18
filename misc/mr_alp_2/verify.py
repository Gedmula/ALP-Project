"""
verify.py — Phase 6: Final schedule verification (§18).

Constraint groups enforced
──────────────────────────
C1 : r_j ≤ C_j                 (earliest landing time)
C2 : C_j ≤ d_j                 (hard deadline)
C3 : C_j ≥ C_i + s[i,j]       (pairwise same-runway separation, all pairs)
C4 : partition and index        (each aircraft appears exactly once)
C5 : greedy consecutive pass    (independent O(n) feasibility check)
C6 : E_j ≥ 0                   (non-negative earliness)
C7 : T_j ≥ 0                   (non-negative tardiness)
C8 : Σ(g·E + h·T) ≈ LP obj     (objective consistency)
C9 : independent LP re-solve   (cross-check via lp_recheck)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from misc.mr_alp_2.models import Instance
from misc.mr_alp_2.lp import stage2_lp, lp_recheck


# ─────────────────────────────────────────────────────────────────────────────
#   Violation records and report
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConstraintViolation:
    group:     str
    index:     int
    lhs:       float
    rhs:       float
    violation: float
    detail:    str


@dataclass
class VerificationReport:
    instance:       str
    seqs:           List[List[int]]
    n_aircraft:     int
    obj_reported:   float
    obj_recomputed: float
    obj_recheck:    float
    landing_times:  Optional[np.ndarray]
    violations:     List[ConstraintViolation] = field(default_factory=list)
    passed:         bool  = True
    tol:            float = 1e-4

    @property
    def n_violations(self) -> int:
        return len(self.violations)

    @property
    def groups_failed(self) -> List[str]:
        return sorted({v.group for v in self.violations})

    @property
    def max_violation(self) -> float:
        return max((v.violation for v in self.violations), default=0.0)

    def summary(self) -> str:
        lines = [
            "═" * 66,
            f"  Verification — {self.instance}   n={self.n_aircraft}",
            "─" * 66,
            f"  Result         : {'✓ PASS' if self.passed else '✗ FAIL'}",
            f"  Obj reported   : {self.obj_reported:.6f}",
            f"  Obj recomputed : {self.obj_recomputed:.6f}  "
            f"(Δ={abs(self.obj_reported - self.obj_recomputed):.2e})",
            f"  Obj LP recheck : {self.obj_recheck:.6f}  "
            f"(Δ={abs(self.obj_reported - self.obj_recheck):.2e})",
            f"  Violations     : {self.n_violations}",
        ]
        if self.violations:
            lines += [
                f"  Groups failed  : {', '.join(self.groups_failed)}",
                f"  Max violation  : {self.max_violation:.4e}",
                "─" * 66,
            ]
            for v in sorted(self.violations, key=lambda x: -x.violation):
                lines.append(
                    f"  {v.group:<4} idx={v.index:>4}  "
                    f"lhs={v.lhs:>12.4f}  rhs={v.rhs:>12.4f}  "
                    f"viol={v.violation:.4e}  {v.detail}")
        lines.append("═" * 66)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#   Main verification function
# ─────────────────────────────────────────────────────────────────────────────

def verify_schedule(
    seqs:         List[List[int]],
    x:            Optional[np.ndarray],
    obj_reported: float,
    inst:         Instance,
    tol:          float = 1e-4,
) -> Tuple[bool, VerificationReport]:
    """
    Verify a complete multi-runway schedule against all constraint groups.

    If x is None the Stage-2 LP is re-solved here to obtain landing times
    before running the constraint checks.

    Parameters
    ----------
    seqs         : multi-runway sequences from the solver.
    x            : LP landing times (None triggers an internal LP solve).
    obj_reported : LP objective value as reported by the solver.
    inst         : instance data.
    tol          : absolute tolerance for numerical comparisons.

    Returns
    -------
    (passed, VerificationReport)
    """
    n     = inst.n
    viols: List[ConstraintViolation] = []

    def _add(group: str, idx: int, lhs: float, rhs: float,
             viol: float, detail: str) -> None:
        viols.append(ConstraintViolation(group, idx, lhs, rhs, viol, detail))

    # ── C4: Partition condition ───────────────────────────────────────────
    all_j = [j for seq in seqs for j in seq]
    if sorted(all_j) != list(range(n)):
        _add('C4', -1, float(len(all_j)), float(n),
             float(abs(len(all_j) - n)),
             f"Indices do not form a valid partition of [0, {n - 1}].")
    else:
        seen: set = set()
        for j in all_j:
            if j in seen:
                _add('C4', j, 1.0, 0.0, 1.0,
                     f"Aircraft {j} appears more than once.")
            seen.add(j)

    if any(v.group == 'C4' for v in viols):
        return False, VerificationReport(
            instance=inst.name, seqs=seqs, n_aircraft=n,
            obj_reported=obj_reported, obj_recomputed=math.inf,
            obj_recheck=math.inf, landing_times=None,
            violations=viols, passed=False, tol=tol)

    # ── Obtain landing times ───────────────────────────────────────────────
    if x is None:
        lp_obj, x = stage2_lp(seqs, inst)
        if x is None:
            _add('C9', -1, math.inf, 0.0, math.inf,
                 "Stage-2 LP infeasible during verification.")
            return False, VerificationReport(
                instance=inst.name, seqs=seqs, n_aircraft=n,
                obj_reported=obj_reported, obj_recomputed=math.inf,
                obj_recheck=math.inf, landing_times=None,
                violations=viols, passed=False, tol=tol)
    else:
        lp_obj = obj_reported

    # ── C1: Earliest landing time ─────────────────────────────────────────
    for j in range(n):
        viol = float(inst.r[j]) - x[j]
        if viol > tol:
            _add('C1', j, x[j], float(inst.r[j]), viol,
                 f"x[{j}]={x[j]:.4f} < r={inst.r[j]:.4f}")

    # ── C2: Hard deadline ─────────────────────────────────────────────────
    for j in range(n):
        viol = x[j] - float(inst.d[j])
        if viol > tol:
            _add('C2', j, x[j], float(inst.d[j]), viol,
                 f"x[{j}]={x[j]:.4f} > d={inst.d[j]:.4f}")

    # ── C3: Full pairwise same-runway separation ───────────────────────────
    for seqr in seqs:
        L = len(seqr)
        for li in range(L - 1):
            for mi in range(li + 1, L):
                i, j = seqr[li], seqr[mi]
                req  = x[i] + float(inst.s[i, j])
                viol = req - x[j]
                if viol > tol:
                    _add('C3', j, x[j], req, viol,
                         f"A{j}@{x[j]:.2f} < A{i}@{x[i]:.2f} "
                         f"+ sep{inst.s[i, j]:.1f}")

    # ── C5: Independent greedy consecutive pass ───────────────────────────
    for rho, seqr in enumerate(seqs):
        if not seqr:
            continue
        t_g = float(inst.r[seqr[0]])
        for li in range(1, len(seqr)):
            t_g = max(float(inst.r[seqr[li]]),
                      t_g + float(inst.s[seqr[li - 1], seqr[li]]))
            if t_g > float(inst.d[seqr[li]]) + tol:
                _add('C5', seqr[li], t_g, float(inst.d[seqr[li]]),
                     t_g - float(inst.d[seqr[li]]),
                     f"Runway {rho}: A{seqr[li]} "
                     f"earliest={t_g:.4f} > d={inst.d[seqr[li]]:.4f}")

    # ── C6, C7: Non-negativity of E_j and T_j ────────────────────────────
    for j in range(n):
        E_j = max(float(inst.delta[j]) - x[j], 0.0)
        T_j = max(x[j] - float(inst.delta[j]), 0.0)
        if E_j < -tol:
            _add('C6', j, E_j, 0.0, -E_j, f"E[{j}]={E_j:.4f} < 0")
        if T_j < -tol:
            _add('C7', j, T_j, 0.0, -T_j, f"T[{j}]={T_j:.4f} < 0")

    # ── C8: Objective consistency ─────────────────────────────────────────
    obj_rc = float(sum(
        float(inst.g[j]) * max(float(inst.delta[j]) - x[j], 0.0)
        + float(inst.h[j]) * max(x[j] - float(inst.delta[j]), 0.0)
        for j in range(n)))
    c8_tol = max(0.5, abs(lp_obj) * 1e-4)
    if abs(obj_rc - lp_obj) > c8_tol:
        _add('C8', -1, lp_obj, obj_rc, abs(obj_rc - lp_obj),
             f"LP obj={lp_obj:.6f} vs Σ(gE+hT)={obj_rc:.6f}")

    # ── C9: Independent LP re-solve ───────────────────────────────────────
    obj_rck, _ = lp_recheck(seqs, inst)
    if math.isinf(obj_rck):
        _add('C9', -1, math.inf, lp_obj, math.inf,
             "Independent LP re-solve returned infeasible.")
    else:
        c9_tol = max(0.5, abs(lp_obj) * 1e-4)
        if abs(obj_rck - lp_obj) > c9_tol:
            _add('C9', -1, obj_rck, lp_obj, abs(obj_rck - lp_obj),
                 f"Re-solve={obj_rck:.6f} vs original={lp_obj:.6f}")

    passed = len(viols) == 0
    return passed, VerificationReport(
        instance=inst.name, seqs=seqs, n_aircraft=n,
        obj_reported=lp_obj, obj_recomputed=obj_rc,
        obj_recheck=obj_rck, landing_times=x,
        violations=viols, passed=passed, tol=tol)