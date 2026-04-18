"""
Tests for GADES.SurfAnalysis.clustering (Step 3).
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose

from GADES.SurfAnalysis.clustering import (
    sign_align,
    mode_coherence,
    Step3Result,
    _compute_cluster_stats,
    _subdivide_incoherent,
    run,
)
from GADES.SurfAnalysis.preprocessing import Step1Result
from GADES.SurfAnalysis.features import Step2Result


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_step1(
    N: int,
    dof: int = 9,
    n_nontrivial: int = 3,
    eigvec_dir=None,
    seed: int = 0,
) -> Step1Result:
    """Synthetic Step1Result.  eigvec_dir controls the direction of v₁."""
    rng = np.random.default_rng(seed)
    evals = rng.standard_normal((N, n_nontrivial))

    K = 1  # only softest mode stored in eigenvectors
    eigvecs = np.zeros((N, K, dof))
    if eigvec_dir is None:
        eigvec_dir = np.zeros(dof)
        eigvec_dir[0] = 1.0
    eigvec_dir = np.asarray(eigvec_dir, float)
    eigvec_dir /= np.linalg.norm(eigvec_dir)
    noise = rng.standard_normal((N, dof)) * 0.05
    vecs = eigvec_dir + noise
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    eigvecs[:, 0, :] = vecs

    return Step1Result(
        steps        = np.arange(N) * 200,
        U            = rng.standard_normal(N),
        A            = rng.standard_normal(N),
        F_norm       = np.abs(rng.standard_normal(N)),
        f_parallel   = rng.standard_normal(N),
        eigenvalues  = evals,
        eigenvectors = eigvecs,
        morse_index  = (evals[:, 0] < 0).astype(np.int64),
        spectral_gap = rng.standard_normal(N),
        n_rigid      = dof - n_nontrivial,
    )


def _make_step2(Phi: np.ndarray) -> Step2Result:
    """Wrap a feature matrix in a Step2Result."""
    d = Phi.shape[1]
    return Step2Result(
        Phi           = Phi,
        Phi_raw       = Phi.copy(),
        feature_names = [f"f{k}" for k in range(d)],
        mean          = np.zeros(d),
        std           = np.ones(d),
        K             = 1,
    )


def _make_clustered(n_per_cluster: int = 60, n_clusters: int = 3, seed: int = 0):
    """
    Three well-separated Gaussian blobs in 2-D feature space, each with
    coherent eigenvectors.  Returns (step1, step2).
    """
    rng = np.random.default_rng(seed)
    N = n_per_cluster * n_clusters

    centers = np.array([[0.0, 0.0], [20.0, 0.0], [0.0, 20.0]])[:n_clusters]
    Phi = np.vstack([
        rng.standard_normal((n_per_cluster, 2)) * 0.3 + c
        for c in centers
    ])

    dof = 9
    ref_dirs = rng.standard_normal((n_clusters, dof))
    ref_dirs /= np.linalg.norm(ref_dirs, axis=1, keepdims=True)

    eigvecs = np.zeros((N, 1, dof))
    for k in range(n_clusters):
        sl = slice(k * n_per_cluster, (k + 1) * n_per_cluster)
        v = ref_dirs[k] + rng.standard_normal((n_per_cluster, dof)) * 0.05
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        eigvecs[sl, 0, :] = v

    evals = rng.standard_normal((N, 3))
    step1 = Step1Result(
        steps        = np.arange(N) * 200,
        U            = rng.standard_normal(N),
        A            = rng.standard_normal(N),
        F_norm       = np.abs(rng.standard_normal(N)),
        f_parallel   = rng.standard_normal(N),
        eigenvalues  = evals,
        eigenvectors = eigvecs,
        morse_index  = (evals[:, 0] < 0).astype(np.int64),
        spectral_gap = rng.standard_normal(N),
        n_rigid      = 6,
    )
    step2 = _make_step2(Phi)
    return step1, step2


# ── sign_align ────────────────────────────────────────────────────────────────

class TestSignAlign:
    def test_positive_dot_unchanged(self):
        ref  = np.array([1.0, 0.0, 0.0])
        vecs = np.array([[0.9, 0.1, 0.0], [0.5, 0.5, 0.0]])
        out  = sign_align(vecs, ref)
        assert_allclose(out, vecs)

    def test_negative_dot_flipped(self):
        ref  = np.array([1.0, 0.0, 0.0])
        vecs = np.array([[-0.9, 0.1, 0.0]])
        out  = sign_align(vecs, ref)
        assert_allclose(out, -vecs)

    def test_mixed_signs(self):
        ref  = np.array([1.0, 0.0])
        vecs = np.array([[0.8, 0.0], [-0.8, 0.0]])
        out  = sign_align(vecs, ref)
        assert out[0, 0] > 0
        assert out[1, 0] > 0

    def test_zero_dot_treated_as_nonnegative(self):
        ref  = np.array([1.0, 0.0])
        vecs = np.array([[0.0, 1.0]])   # dot = 0
        out  = sign_align(vecs, ref)
        assert_allclose(out, vecs)      # should not flip

    def test_output_shape_preserved(self):
        ref  = np.ones(5) / np.sqrt(5)
        vecs = np.random.default_rng(0).standard_normal((10, 5))
        out  = sign_align(vecs, ref)
        assert out.shape == vecs.shape


# ── mode_coherence ────────────────────────────────────────────────────────────

class TestModeCoherence:
    def test_identical_vectors_score_one(self):
        v    = np.array([1.0, 0.0, 0.0])
        vecs = np.tile(v, (20, 1))
        _, s_C = mode_coherence(vecs)
        assert_allclose(s_C, 1.0, atol=1e-12)

    def test_opposite_signs_score_one_after_alignment(self):
        """Half the vectors are flipped; after sign-alignment they should align."""
        rng  = np.random.default_rng(1)
        v    = np.array([1.0, 0.0, 0.0])
        vecs = np.tile(v, (10, 1))
        vecs[5:] *= -1.0
        _, s_C = mode_coherence(vecs)
        assert_allclose(s_C, 1.0, atol=1e-12)

    def test_nearly_aligned_score_near_one(self):
        rng  = np.random.default_rng(2)
        v    = np.zeros(9); v[0] = 1.0
        noise = rng.standard_normal((50, 9)) * 0.02
        vecs  = v + noise
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        _, s_C = mode_coherence(vecs)
        assert s_C > 0.95

    def test_empty_input_returns_zero_score(self):
        vecs = np.empty((0, 5))
        v_bar, s_C = mode_coherence(vecs)
        assert s_C == 0.0
        assert v_bar.shape == (5,)

    def test_v_bar_is_unit_vector(self):
        rng  = np.random.default_rng(3)
        vecs = rng.standard_normal((10, 6))
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        v_bar, _ = mode_coherence(vecs)
        if np.linalg.norm(v_bar) > 0:
            assert_allclose(np.linalg.norm(v_bar), 1.0, atol=1e-12)


# ── _compute_cluster_stats ────────────────────────────────────────────────────

class TestComputeClusterStats:
    def setup_method(self):
        rng = np.random.default_rng(7)
        N = 30
        self.step1 = _make_step1(N, dof=9, n_nontrivial=3, seed=7)
        self.labels = np.array([0] * 15 + [1] * 15, dtype=np.int64)

    def test_cluster_ids_correct(self):
        ids, *_ = _compute_cluster_stats(self.labels, self.step1)
        assert list(ids) == [0, 1]

    def test_sizes_sum_to_N_minus_noise(self):
        _, sizes, *_ = _compute_cluster_stats(self.labels, self.step1)
        assert sizes.sum() == len(self.labels)

    def test_mean_U_correct(self):
        _, _, mean_U, *_ = _compute_cluster_stats(self.labels, self.step1)
        expected0 = self.step1.U[:15].mean()
        assert_allclose(mean_U[0], expected0, atol=1e-12)

    def test_v_bar_shape(self):
        *_, v_bar, _ = _compute_cluster_stats(self.labels, self.step1)
        assert v_bar.shape == (2, 9)

    def test_coherence_shape(self):
        *_, coherence = _compute_cluster_stats(self.labels, self.step1)
        assert coherence.shape == (2,)

    def test_noise_excluded_from_stats(self):
        labels_with_noise = self.labels.copy()
        labels_with_noise[:3] = -1
        ids, sizes, *_ = _compute_cluster_stats(labels_with_noise, self.step1)
        assert -1 not in ids
        assert sizes.sum() == (labels_with_noise >= 0).sum()


# ── _subdivide_incoherent ─────────────────────────────────────────────────────

class TestSubdivideIncoherent:
    def _make_incoherent_step1(self, N=40, dof=9, seed=0):
        """Step1 where cluster 0 has random (incoherent) eigenvectors."""
        rng = np.random.default_rng(seed)
        evals = rng.standard_normal((N, 3))
        eigvecs = np.zeros((N, 1, dof))
        vecs = rng.standard_normal((N, dof))
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        eigvecs[:, 0, :] = vecs
        # Make eigenvalues: first half positive, second half negative (for λ₁)
        evals[:N // 2, 0] = np.abs(evals[:N // 2, 0])
        evals[N // 2:, 0] = -np.abs(evals[N // 2:, 0])
        return Step1Result(
            steps        = np.arange(N) * 200,
            U            = rng.standard_normal(N),
            A            = rng.standard_normal(N),
            F_norm       = np.abs(rng.standard_normal(N)),
            f_parallel   = rng.standard_normal(N),
            eigenvalues  = evals,
            eigenvectors = eigvecs,
            morse_index  = (evals[:, 0] < 0).astype(np.int64),
            spectral_gap = rng.standard_normal(N),
            n_rigid      = dof - 3,
        )

    def test_incoherent_cluster_split_into_two(self):
        N = 40
        step1  = self._make_incoherent_step1(N)
        labels = np.zeros(N, dtype=np.int64)
        cluster_ids       = np.array([0], dtype=np.int64)
        cluster_coherence = np.array([0.3])   # below 0.8 threshold

        new_labels = _subdivide_incoherent(
            labels, step1, cluster_ids, cluster_coherence, threshold=0.8
        )
        unique = np.unique(new_labels)
        # Original label 0 should be gone, replaced by two new labels
        assert 0 not in unique
        assert len(unique) == 2

    def test_coherent_cluster_unchanged(self):
        N = 20
        step1  = _make_step1(N)
        labels = np.zeros(N, dtype=np.int64)
        cluster_ids       = np.array([0], dtype=np.int64)
        cluster_coherence = np.array([0.95])   # above threshold

        new_labels = _subdivide_incoherent(
            labels, step1, cluster_ids, cluster_coherence, threshold=0.8
        )
        assert_allclose(new_labels, labels)

    def test_same_sign_lambda1_emits_warning(self):
        """Cluster with low coherence but all λ₁ > 0 cannot be split."""
        N = 20
        rng = np.random.default_rng(5)
        step1 = _make_step1(N, seed=5)
        # Force all λ₁ positive
        evals = step1.eigenvalues.copy()
        evals[:, 0] = np.abs(evals[:, 0])
        step1 = step1._replace(eigenvalues=evals)

        labels            = np.zeros(N, dtype=np.int64)
        cluster_ids       = np.array([0], dtype=np.int64)
        cluster_coherence = np.array([0.1])   # definitely incoherent

        with pytest.warns(RuntimeWarning, match="same sign"):
            new_labels = _subdivide_incoherent(
                labels, step1, cluster_ids, cluster_coherence, threshold=0.8
            )
        # Should be unchanged since split is impossible
        assert_allclose(new_labels, labels)


# ── run ───────────────────────────────────────────────────────────────────────

class TestStep3Run:
    def test_returns_step3result(self):
        step1, step2 = _make_clustered()
        out = run(step2, step1, min_cluster_size=10, min_samples=3)
        assert isinstance(out, Step3Result)

    def test_labels_length(self):
        step1, step2 = _make_clustered(n_per_cluster=60)
        out = run(step2, step1, min_cluster_size=10, min_samples=3)
        assert len(out.labels) == len(step1.U)

    def test_n_noise_consistent_with_labels(self):
        step1, step2 = _make_clustered()
        out = run(step2, step1, min_cluster_size=10, min_samples=3)
        assert out.n_noise == int((out.labels == -1).sum())

    def test_cluster_ids_nonnegative(self):
        step1, step2 = _make_clustered()
        out = run(step2, step1, min_cluster_size=10, min_samples=3)
        assert (out.cluster_ids >= 0).all()

    def test_cluster_summary_arrays_same_length(self):
        step1, step2 = _make_clustered()
        out = run(step2, step1, min_cluster_size=10, min_samples=3)
        n_c = len(out.cluster_ids)
        assert len(out.cluster_sizes)        == n_c
        assert len(out.cluster_mean_U)       == n_c
        assert len(out.cluster_mean_lambda1) == n_c
        assert len(out.cluster_mean_morse)   == n_c
        assert len(out.cluster_coherence)    == n_c
        assert out.cluster_v_bar.shape       == (n_c, 9)

    def test_cluster_sizes_sum_to_non_noise(self):
        step1, step2 = _make_clustered()
        out = run(step2, step1, min_cluster_size=10, min_samples=3)
        assert out.cluster_sizes.sum() == (out.labels >= 0).sum()

    def test_all_noise_case(self):
        """N < min_cluster_size → HDBSCAN labels everything as noise."""
        step1 = _make_step1(N=3, seed=9)
        Phi   = np.random.default_rng(9).standard_normal((3, 2))
        step2 = _make_step2(Phi)
        # min_samples must be ≤ n_samples; min_cluster_size > N forces all-noise
        out   = run(step2, step1, min_cluster_size=50, min_samples=2)
        assert len(out.cluster_ids) == 0
        assert out.n_noise == 3

    def test_finds_expected_number_of_clusters(self):
        """Well-separated blobs should yield exactly 3 clusters (no noise)."""
        step1, step2 = _make_clustered(n_per_cluster=80, n_clusters=3, seed=42)
        out = run(step2, step1, min_cluster_size=10, min_samples=3)
        assert len(out.cluster_ids) == 3
        assert out.n_noise == 0

    def test_coherence_scores_in_unit_interval(self):
        step1, step2 = _make_clustered()
        out = run(step2, step1, min_cluster_size=10, min_samples=3)
        assert np.all(out.cluster_coherence >= 0.0)
        assert np.all(out.cluster_coherence <= 1.0)

    def test_coherent_clusters_above_threshold(self):
        """Clusters made from a single direction should be coherent."""
        step1, step2 = _make_clustered(seed=0)
        out = run(step2, step1, min_cluster_size=10, min_samples=3,
                  coherence_threshold=0.8)
        if len(out.cluster_ids) > 0:
            assert np.all(out.cluster_coherence >= 0.8)
