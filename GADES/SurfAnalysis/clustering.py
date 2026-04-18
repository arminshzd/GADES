"""
GADES SurfAnalysis — Step 3: Clustering into landscape regions.

Partitions snapshots into geometrically coherent clusters (basins, ridges,
shoulders) using HDBSCAN on the standardised feature matrix Φ from Step 2,
then verifies that the dominant soft eigenvector is internally consistent
within each cluster (mode-coherence check).

Clusters that fail the coherence test are subdivided by the sign of the
softest eigenvalue λ₁ — a natural separator between basin (λ₁ > 0) and
saddle-like (λ₁ < 0) snapshots.

Cluster labels in the result:
    ≥ 0   normal cluster id
     -1   noise / outlier (HDBSCAN convention)

Clusters with ⟨m⟩ ≥ 1 are saddle candidates; clusters with ⟨m⟩ = 0 and
small ⟨|F|⟩ are basin candidates.
"""

from __future__ import annotations

import warnings
from typing import NamedTuple, Optional

import numpy as np

from .preprocessing import Step1Result
from .features import Step2Result


# ── HDBSCAN import (sklearn ≥ 1.3 preferred, fall back to hdbscan package) ───

try:
    from sklearn.cluster import HDBSCAN as _SklearnHDBSCAN

    def _run_hdbscan(X: np.ndarray, min_cluster_size: int, min_samples: int) -> np.ndarray:
        return _SklearnHDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        ).fit_predict(X)

except ImportError:                          # sklearn < 1.3
    try:
        import hdbscan as _hdbscan_pkg

        def _run_hdbscan(X: np.ndarray, min_cluster_size: int, min_samples: int) -> np.ndarray:
            return _hdbscan_pkg.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
            ).fit_predict(X)

    except ImportError:
        _run_hdbscan = None  # type: ignore[assignment]


# ── result container ──────────────────────────────────────────────────────────

class Step3Result(NamedTuple):
    """
    Clustering results from Step 3.

    Attributes:
        labels               (N,)              Cluster label per snapshot; -1 = noise.
        cluster_ids          (n_clusters,)      Unique cluster ids (all ≥ 0).
        cluster_sizes        (n_clusters,)      Number of snapshots per cluster.
        cluster_mean_U       (n_clusters,)      Mean potential energy ⟨U⟩.
        cluster_mean_lambda1 (n_clusters,)      Mean softest eigenvalue ⟨λ₁⟩.
        cluster_mean_morse   (n_clusters,)      Mean Morse index ⟨m⟩.
        cluster_v_bar        (n_clusters, dof)  Normalised mean soft eigenvector v̄₁.
        cluster_coherence    (n_clusters,)      Mode-coherence score s_C ∈ [0, 1].
        n_noise              int                Number of snapshots labelled -1.
    """
    labels:               np.ndarray
    cluster_ids:          np.ndarray
    cluster_sizes:        np.ndarray
    cluster_mean_U:       np.ndarray
    cluster_mean_lambda1: np.ndarray
    cluster_mean_morse:   np.ndarray
    cluster_v_bar:        np.ndarray
    cluster_coherence:    np.ndarray
    n_noise:              int


# ── eigenvector utilities ─────────────────────────────────────────────────────

def sign_align(vecs: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Flip the sign of each row in *vecs* so that ``v · ref ≥ 0``.

    Args:
        vecs: (n, d) array of eigenvectors.
        ref:  (d,)  reference direction.

    Returns:
        Sign-aligned copy of *vecs*, shape (n, d).
    """
    dots  = vecs @ ref                            # (n,)
    signs = np.where(dots >= 0, 1.0, -1.0)
    return vecs * signs[:, None]


def mode_coherence(vecs: np.ndarray) -> tuple:
    """
    Compute the mean eigenvector and alignment score for a set of eigenvectors.

    Sign-aligns all rows to the first row, then averages and normalises.

    Args:
        vecs: (n, d) eigenvectors (unit vectors from ``np.linalg.eigh``).

    Returns:
        v_bar: (d,) normalised mean eigenvector; zero vector when n == 0 or
               all vectors cancel.
        s_C:   float — mean |v · v̄| over the cluster, in [0, 1].
               Returns 0.0 when n == 0 or v_bar is zero.
    """
    if len(vecs) == 0:
        d = vecs.shape[1] if vecs.ndim == 2 else 0
        return np.zeros(d), 0.0

    ref     = vecs[0] / (np.linalg.norm(vecs[0]) + 1e-30)
    aligned = sign_align(vecs, ref)

    v_bar = aligned.mean(axis=0)
    norm  = np.linalg.norm(v_bar)
    if norm == 0.0:
        return v_bar, 0.0

    v_bar = v_bar / norm
    s_C   = float(np.abs(aligned @ v_bar).mean())
    return v_bar, s_C


# ── per-cluster statistics ────────────────────────────────────────────────────

def _compute_cluster_stats(
    labels: np.ndarray,
    step1:  Step1Result,
) -> tuple:
    """Compute per-cluster summary statistics (noise label -1 excluded)."""
    unique = np.unique(labels)
    unique = unique[unique >= 0]
    n_c    = len(unique)
    dof    = step1.eigenvectors.shape[2]

    cluster_ids      = unique
    cluster_sizes    = np.empty(n_c, dtype=np.int64)
    cluster_mean_U   = np.empty(n_c)
    cluster_mean_l1  = np.empty(n_c)
    cluster_mean_m   = np.empty(n_c)
    cluster_v_bar    = np.empty((n_c, dof))
    cluster_coherence = np.empty(n_c)

    for j, c in enumerate(unique):
        mask = labels == c
        cluster_sizes[j]   = int(mask.sum())
        cluster_mean_U[j]  = step1.U[mask].mean()
        cluster_mean_l1[j] = step1.eigenvalues[mask, 0].mean()
        cluster_mean_m[j]  = step1.morse_index[mask].mean()

        v1_c              = step1.eigenvectors[mask, 0, :]   # (|C|, dof)
        v_bar, s_C        = mode_coherence(v1_c)
        cluster_v_bar[j]   = v_bar
        cluster_coherence[j] = s_C

    return (cluster_ids, cluster_sizes, cluster_mean_U,
            cluster_mean_l1, cluster_mean_m, cluster_v_bar, cluster_coherence)


# ── subdivision ───────────────────────────────────────────────────────────────

def _subdivide_incoherent(
    labels:            np.ndarray,
    step1:             Step1Result,
    cluster_ids:       np.ndarray,
    cluster_coherence: np.ndarray,
    threshold:         float,
) -> np.ndarray:
    """
    Split incoherent clusters (s_C < *threshold*) by the sign of λ₁.

    Clusters that cannot be split (all λ₁ same sign) emit a RuntimeWarning
    and are left intact.

    Returns a new labels array; original label values for incoherent clusters
    are replaced by two new unique labels.
    """
    new_labels = labels.copy()
    next_label = int(labels.max()) + 1

    for j, c in enumerate(cluster_ids):
        if cluster_coherence[j] >= threshold:
            continue

        mask    = labels == c
        lambda1 = step1.eigenvalues[mask, 0]
        pos_sub = lambda1 >= 0

        if pos_sub.all() or (~pos_sub).all():
            warnings.warn(
                f"Cluster {c} has low coherence ({cluster_coherence[j]:.2f}) "
                "but all λ₁ values share the same sign; cannot subdivide.",
                RuntimeWarning,
                stacklevel=3,
            )
            continue

        idx = np.where(mask)[0]
        new_labels[idx[pos_sub]]  = next_label
        new_labels[idx[~pos_sub]] = next_label + 1
        next_label += 2

    return new_labels


# ── public entry point ────────────────────────────────────────────────────────

def _pca_reduce(Phi: np.ndarray, n_components: int) -> np.ndarray:
    """Project Phi onto its top-*n_components* principal components."""
    n_components = min(n_components, Phi.shape[1], Phi.shape[0] - 1)
    Phi_c = Phi - Phi.mean(axis=0)
    _, _, Vt = np.linalg.svd(Phi_c, full_matrices=False)
    return Phi_c @ Vt[:n_components].T


def run(
    step2:               Step2Result,
    step1:               Step1Result,
    min_cluster_size:    int            = 20,
    min_samples:         int            = 5,
    coherence_threshold: float          = 0.8,
    n_pca_components:    Optional[int]  = None,
) -> Step3Result:
    """
    Execute Step 3 of the GADES CV workflow.

    Clusters snapshots by their standardised feature vectors (HDBSCAN), then
    checks mode coherence within each cluster and subdivides incoherent ones
    by the sign of λ₁.

    Args:
        step2:               Output of :func:`features.run` (Step 2).
        step1:               Output of :func:`preprocessing.run` (Step 1).
        min_cluster_size:    HDBSCAN ``min_cluster_size`` (≈ 20 for N ≈ 1000).
        min_samples:         HDBSCAN ``min_samples`` (≈ 5).
        coherence_threshold: Clusters with s_C below this are subdivided
                             (default 0.8).
        n_pca_components:    If set, reduce Φ to this many PCA components
                             before clustering.  Recommended when the feature
                             dimension (2K+5) is large relative to N: HDBSCAN's
                             density estimates degrade in high dimensions and
                             label all points as noise.  ``None`` clusters on
                             the full Φ (default).

    Returns:
        :class:`Step3Result` with cluster labels and per-cluster summary.

    Raises:
        ImportError: If neither scikit-learn (≥ 1.3) nor hdbscan is installed.
    """
    if _run_hdbscan is None:
        raise ImportError(
            "Neither scikit-learn (≥ 1.3) nor the hdbscan package is available. "
            "Install one: `pip install scikit-learn` or `pip install hdbscan`."
        )

    # ── optional dimensionality reduction ─────────────────────────────────────
    if n_pca_components is not None:
        Phi_cluster = _pca_reduce(step2.Phi, n_pca_components)
    else:
        Phi_cluster = step2.Phi

    # ── primary clustering ────────────────────────────────────────────────────
    labels = _run_hdbscan(Phi_cluster, min_cluster_size, min_samples).astype(np.int64)

    # ── mode-coherence check and subdivision ──────────────────────────────────
    if (labels >= 0).any():
        stats = _compute_cluster_stats(labels, step1)
        cluster_ids, _, _, _, _, _, cluster_coherence = stats

        labels = _subdivide_incoherent(
            labels, step1, cluster_ids, cluster_coherence, coherence_threshold
        )

    # ── final per-cluster statistics ──────────────────────────────────────────
    if (labels >= 0).any():
        (cluster_ids, cluster_sizes, cluster_mean_U,
         cluster_mean_l1, cluster_mean_m, cluster_v_bar,
         cluster_coherence) = _compute_cluster_stats(labels, step1)
    else:
        dof               = step1.eigenvectors.shape[2]
        cluster_ids       = np.empty(0, dtype=np.int64)
        cluster_sizes     = np.empty(0, dtype=np.int64)
        cluster_mean_U    = np.empty(0)
        cluster_mean_l1   = np.empty(0)
        cluster_mean_m    = np.empty(0)
        cluster_v_bar     = np.empty((0, dof))
        cluster_coherence = np.empty(0)

    return Step3Result(
        labels               = labels,
        cluster_ids          = cluster_ids,
        cluster_sizes        = cluster_sizes,
        cluster_mean_U       = cluster_mean_U,
        cluster_mean_lambda1 = cluster_mean_l1,
        cluster_mean_morse   = cluster_mean_m,
        cluster_v_bar        = cluster_v_bar,
        cluster_coherence    = cluster_coherence,
        n_noise              = int((labels == -1).sum()),
    )
