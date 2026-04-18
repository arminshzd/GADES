"""
GADES SurfAnalysis — Step 2: Feature construction.

Turns the per-snapshot scalars produced by Step 1 into a standardised
feature matrix suitable for clustering (Step 3) and CV learning (Step 5).

For each snapshot i the raw feature vector is::

    φ_i = [log|λ_1|, …, log|λ_K|,        (K entries — log-compressed eigenvalues)
           sign(λ_1), …, sign(λ_K),        (K entries — basin / saddle flags)
           |F|, f_∥, U, tr(H), logdet(H_+)] (5 scalar invariants)

giving  Φ ∈ ℝ^{N × (2K+5)}.

After construction, each column is Z-scored across the N snapshots so
that every feature contributes equally to distance-based clustering.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Sequence

import numpy as np

from .preprocessing import Step1Result


# ── result container ──────────────────────────────────────────────────────────

class Step2Result(NamedTuple):
    """
    Feature matrix and standardisation statistics from Step 2.

    Attributes:
        Phi           (N, 2K+5) Z-scored feature matrix (use for clustering).
        Phi_raw       (N, 2K+5) un-standardised feature matrix.
        feature_names List of column labels (length 2K+5).
        mean          (2K+5,)  per-column mean used for standardisation.
        std           (2K+5,)  per-column std  used for standardisation
                               (columns with zero std are left as zero).
        K             int      number of eigenvalue modes used.
    """
    Phi:           np.ndarray
    Phi_raw:       np.ndarray
    feature_names: list
    mean:          np.ndarray
    std:           np.ndarray
    K:             int


# ── feature construction ──────────────────────────────────────────────────────

def _log_abs(x: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    """log|x|, clipped to avoid log(0)."""
    return np.log(np.abs(x) + eps)


def build_features(result: Step1Result, K: Optional[int] = None) -> tuple:
    """
    Construct the raw (un-standardised) feature matrix from a Step1Result.

    Args:
        result: Output of :func:`preprocessing.run`.
        K:      Number of eigenvalue modes to include.  Defaults to all
                non-rigid modes stored in ``result.eigenvalues``.

    Returns:
        Phi_raw       (N, 2K+5) float array.
        feature_names List[str] of column labels.
    """
    evals = result.eigenvalues            # (N, n_nontrivial)
    N, n_nontrivial = evals.shape

    if K is None:
        K = n_nontrivial
    elif K > n_nontrivial:
        raise ValueError(
            f"K={K} exceeds the number of non-rigid eigenvalue modes "
            f"({n_nontrivial}) available in the Step1Result."
        )

    evals_K = evals[:, :K]               # (N, K) — softest K modes

    # ── block 1: log|λ_k| ────────────────────────────────────────────────────
    log_evals = _log_abs(evals_K)         # (N, K)

    # ── block 2: sign(λ_k) ───────────────────────────────────────────────────
    sign_evals = np.sign(evals_K)         # (N, K)

    # ── block 3: five scalar invariants ──────────────────────────────────────
    F_norm     = result.F_norm[:, None]                       # (N, 1)
    f_parallel = result.f_parallel[:, None]                   # (N, 1)
    U          = result.U[:, None]                            # (N, 1)

    # tr(H) = sum of all non-rigid eigenvalues
    tr_H = result.eigenvalues.sum(axis=1, keepdims=True)      # (N, 1)

    # logdet(H_+) = Σ_{k: λ_k > 0} log(λ_k)
    pos_mask = result.eigenvalues > 0                         # (N, n_nontrivial)
    log_evals_all = _log_abs(result.eigenvalues)              # (N, n_nontrivial)
    logdet_Hplus = (log_evals_all * pos_mask).sum(axis=1, keepdims=True)  # (N, 1)

    Phi_raw = np.hstack([log_evals, sign_evals,
                         F_norm, f_parallel, U, tr_H, logdet_Hplus])

    # ── feature names ─────────────────────────────────────────────────────────
    names = (
        [f"log|λ_{k+1}|" for k in range(K)]
        + [f"sign(λ_{k+1})" for k in range(K)]
        + ["|F|", "f_∥", "U", "tr(H)", "logdet(H_+)"]
    )

    return Phi_raw, names


def standardise(
    Phi_raw: np.ndarray,
) -> tuple:
    """
    Z-score each column of *Phi_raw* across snapshots.

    Columns whose standard deviation is zero (constant features) are set
    to zero rather than producing NaN.

    Returns:
        Phi:  (N, d) standardised matrix.
        mean: (d,)  per-column mean.
        std:  (d,)  per-column standard deviation.
    """
    mean = Phi_raw.mean(axis=0)
    std  = Phi_raw.std(axis=0)
    safe_std = np.where(std == 0, 1.0, std)      # avoid division by zero
    Phi = (Phi_raw - mean) / safe_std
    Phi[:, std == 0] = 0.0                        # constant columns → 0
    return Phi, mean, std


# ── public entry point ────────────────────────────────────────────────────────

def run(result: Step1Result, K: Optional[int] = None) -> Step2Result:
    """
    Execute Step 2 of the GADES CV workflow.

    Builds the feature matrix from a :class:`~preprocessing.Step1Result`
    and Z-scores it for use in clustering.

    Args:
        result: Output of :func:`preprocessing.run` (Step 1).
        K:      Number of eigenvalue modes to include in the feature vector.
                Defaults to all non-rigid modes in *result*.

    Returns:
        :class:`Step2Result` with the standardised matrix, raw matrix,
        column names, and standardisation statistics.
    """
    Phi_raw, feature_names = build_features(result, K=K)
    Phi, mean, std         = standardise(Phi_raw)

    return Step2Result(
        Phi           = Phi,
        Phi_raw       = Phi_raw,
        feature_names = feature_names,
        mean          = mean,
        std           = std,
        K             = (Phi_raw.shape[1] - 5) // 2,   # recover K from 2K+5
    )
