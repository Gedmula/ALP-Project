"""
move_pressure.py — Phase 5: Move-pressure scoring for aircraft selection (§13).

MP_j = w1·P̃_j + w2·DR̃_j + w3·SB̃_j + w4·CR̃_j + w5·G̃^move_j + w6·LB̃_ρ

Components
──────────
P_j      : individual LP penalty (§13.1).
DR_j     : downstream relief — surrogate penalty reduction on ρ if j is
           removed (§13.2).
SB_j     : wake-separation burden from current neighbours (§13.3).
CR_j     : criticality (g_j + h_j) / max(AF_j, ε), precomputed in
           Instance.__post_init__ (§13.4).
G^move_j : expected gain from transferring j to its best alternative
           runway (§13.5).
LB_ρ     : runway-overload pressure max(0, B_ρ − B̄) (§13.5).

All six components are min-max normalised to [0, 1] before weighting so
that no single component dominates regardless of its absolute scale.
"""
from __future__ import annotations

from typing import List, Tuple
import math
import numpy as np

from misc.mr_alp_2.models import Instance, HybridParams
from misc.mr_alp_2.surrogate import surrogate_penalty, committed_time, best_insertion
from misc.mr_alp_2.config import EPS


# ─────────────────────────────────────────────────────────────────────────────
#   Component calculations
# ─────────────────────────────────────────────────────────────────────────────

def _individual_penalty(j: int, x: np.ndarray, inst: Instance) -> float:
    """
    P_j: actual LP penalty of aircraft j using LP landing times x (§13.1).
    """
    return (float(inst.g[j]) * max(float(inst.delta[j]) - x[j], 0.0)
            + float(inst.h[j]) * max(x[j] - float(inst.delta[j]), 0.0))


def _downstream_relief(
    j:    int,
    rho:  int,
    seqs: List[List[int]],
    inst: Instance,
) -> float:
    """
    DR_j = F_ρ − F_ρ^{−j} (§13.2).

    Surrogate penalty of runway ρ before removing j minus the surrogate
    penalty after.  A large positive value means j is causing significant
    downstream congestion on its current runway.
    """
    seq   = seqs[rho]
    F_rho = surrogate_penalty(seq, inst)
    F_out = surrogate_penalty([k for k in seq if k != j], inst)
    return F_rho - F_out


def _separation_burden(
    j:    int,
    rho:  int,
    seqs: List[List[int]],
    inst: Instance,
) -> float:
    """
    SB_j: wake-separation burden from current neighbours (§13.3).

    Interior position (a, j, b):  max(0, s[a,j] + s[j,b] − s[a,b])
    First position    (j, b)   :  s[j, b]
    Last position     (a, j)   :  s[a, j]
    Singleton runway            :  0
    """
    seq   = seqs[rho]
    pos   = seq.index(j)
    has_a = pos > 0
    has_b = pos < len(seq) - 1
    a     = seq[pos - 1] if has_a else -1
    b     = seq[pos + 1] if has_b else -1
    if has_a and has_b:
        return max(0.0,
                   float(inst.s[a, j])
                   + float(inst.s[j, b])
                   - float(inst.s[a, b]))
    elif has_b:
        return float(inst.s[j, b])
    elif has_a:
        return float(inst.s[a, j])
    return 0.0


def _best_alt_gain(
    j:      int,
    rho:    int,
    seqs:   List[List[int]],
    inst:   Instance,
    params: HybridParams,
) -> float:
    """
    G^move_j = RemoveGain_j − BestAlt_j (§13.5).

    Returns 0 when no alternative runway exists (m=1 or all insertions
    are surrogate-infeasible) so the component contributes a neutral
    value rather than propagating -inf into the pressure score.
    """
    seq         = seqs[rho]
    F_rho       = surrogate_penalty(seq, inst)
    F_out       = surrogate_penalty([k for k in seq if k != j], inst)
    remove_gain = F_rho - F_out
    best_alt, _, _ = best_insertion(j, seqs, inst, params, exclude_rho=rho)
    if math.isinf(best_alt):
        return 0.0
    return remove_gain - best_alt


def _normalise(arr: np.ndarray) -> np.ndarray:
    """
    Min-max normalise arr to [0, 1].

    lo and hi are computed from finite entries only so that a single
    ±inf or nan value does not collapse the scale or produce nan outputs.
    Non-finite entries are mapped to 0 after normalisation.
    """
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=float)
    lo  = float(finite.min())
    hi  = float(finite.max())
    out = (arr - lo) / max(hi - lo, EPS)
    return np.where(np.isfinite(out), out, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
#   Aggregate move-pressure score
# ─────────────────────────────────────────────────────────────────────────────

def compute_move_pressure(
    seqs:   List[List[int]],
    x:      np.ndarray,
    inst:   Instance,
    params: HybridParams,
) -> np.ndarray:
    """
    Compute MP_j for every aircraft j ∈ J.

    Parameters
    ----------
    seqs   : current multi-runway sequences.
    x      : LP landing times from the most recent Stage-2 LP solve.
    inst   : instance data.
    params : HybridParams carrying the weight vector w = (w1, …, w6).

    Returns
    -------
    MP : (n,) array of move-pressure scores, higher means more pressure.
    """
    n = inst.n
    m = len(seqs)

    # Aircraft → runway lookup
    rho_map = np.full(n, -1, dtype=int)
    for rho in range(m):
        for j in seqs[rho]:
            rho_map[j] = rho

    # Per-runway overload pressure LB_ρ = max(0, B_ρ − B̄)
    B      = np.array([committed_time(seqs[rho], inst) for rho in range(m)])
    B_bar  = float(B.mean())
    LB_rho = np.maximum(B - B_bar, 0.0)

    P_raw  = np.zeros(n)
    DR_raw = np.zeros(n)
    SB_raw = np.zeros(n)
    G_raw  = np.zeros(n)
    LB_raw = np.zeros(n)

    for j in range(n):
        rho        = int(rho_map[j])
        P_raw[j]   = _individual_penalty(j, x, inst)
        DR_raw[j]  = _downstream_relief(j, rho, seqs, inst)
        SB_raw[j]  = _separation_burden(j, rho, seqs, inst)
        G_raw[j]   = _best_alt_gain(j, rho, seqs, inst, params)
        LB_raw[j]  = float(LB_rho[rho])

    w  = params.w
    MP = (w[0] * _normalise(P_raw)
          + w[1] * _normalise(DR_raw)
          + w[2] * _normalise(SB_raw)
          + w[3] * _normalise(inst.CR)
          + w[4] * _normalise(G_raw)
          + w[5] * _normalise(LB_raw))
    return MP


def select_candidate_aircraft(
    seqs:   List[List[int]],
    x:      np.ndarray,
    inst:   Instance,
    params: HybridParams,
) -> List[int]:
    """
    Return the top-K aircraft indices ranked by MP score (§13.6).

    K = min(params.K, n) so the list is always bounded even for small
    instances.
    """
    MP    = compute_move_pressure(seqs, x, inst, params)
    K     = min(params.K, inst.n)
    return list(np.argsort(-MP)[:K])

# ─────────────────────────────────────────────────────────────────────────────
#   §16.6  Adaptive move-pressure weight updater
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveWeightUpdater:
    """
    Online exponential-smoothing tracker for the six move-pressure components.

    At each LP improvement event, the component that was the primary driver
    for the selected aircraft is rewarded proportionally to the improvement
    magnitude.  All utilities decay at rate ξ when no improvement is made,
    preserving exploration.

    The resulting weight vector w satisfies w_q ≥ 0 and Σw_q = 1.

    Parameters
    ----------
    xi   : learning rate ξ ∈ (0, 1).  Lower values → slower adaptation.
    beta : greediness β ≥ 1.  Higher values concentrate weight on the
           historically best-performing component.
    """

    def __init__(self, xi: float = 0.10, beta: float = 1.5) -> None:
        self.xi   = xi
        self.beta = beta
        # Initialise to uniform utility so all components are equally likely
        # at the start before any evidence has been accumulated.
        self.U: np.ndarray = np.ones(6) / 6.0

    def reward(self, component_idx: int, improvement: float) -> None:
        """
        Reward one component after an LP-improving move.

        The reward is the LP improvement magnitude (always ≥ 0).
        All other components receive zero reward, but their utility
        is partially preserved by the (1 − ξ) decay.
        """
        R         = np.zeros(6)
        R[component_idx] = max(float(improvement), 0.0)
        self.U    = (1.0 - self.xi) * self.U + self.xi * R

    def decay(self) -> None:
        """
        Apply exploration decay when no LP improvement was found.

        Prevents any single component from monopolising the weight
        vector after a long non-improving streak.
        """
        self.U = (1.0 - self.xi) * self.U

    def weights(self) -> Tuple[float, ...]:
        """
        Return the current weight vector as a tuple.

        Computed via softmax-like power normalisation:
          w_q = (U_q + ε)^β / Σ_r (U_r + ε)^β
        This guarantees w_q > 0 and Σw_q = 1 regardless of U values.
        """
        powered = np.maximum(self.U, EPS) ** self.beta
        w       = powered / powered.sum()
        return tuple(float(x) for x in w)