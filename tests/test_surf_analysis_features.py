"""
Tests for GADES.SurfAnalysis.features (Step 2).
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose

from GADES.SurfAnalysis.features import build_features, standardise, run, Step2Result
from GADES.SurfAnalysis.preprocessing import Step1Result


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_result(N=5, n_nontrivial=4, seed=0) -> Step1Result:
    """Synthetic Step1Result with controllable shape."""
    rng = np.random.default_rng(seed)
    evals = rng.standard_normal((N, n_nontrivial))
    evecs = np.zeros((N, n_nontrivial, 3 * 3))   # dof=9 (M=3 atoms)
    return Step1Result(
        steps        = np.arange(N) * 200,
        U            = rng.standard_normal(N),
        A            = rng.standard_normal(N),
        F_norm       = np.abs(rng.standard_normal(N)),
        f_parallel   = rng.standard_normal(N),
        eigenvalues  = evals,
        eigenvectors = evecs,
        morse_index  = (evals < 0).sum(axis=1).astype(np.int64),
        spectral_gap = rng.standard_normal(N),
        n_rigid      = 5,
    )


# ── build_features ────────────────────────────────────────────────────────────

class TestBuildFeatures:
    def test_output_shape_default_K(self):
        result = _make_result(N=10, n_nontrivial=6)
        Phi_raw, names = build_features(result)
        assert Phi_raw.shape == (10, 2 * 6 + 5)

    def test_output_shape_custom_K(self):
        result = _make_result(N=10, n_nontrivial=6)
        K = 3
        Phi_raw, names = build_features(result, K=K)
        assert Phi_raw.shape == (10, 2 * K + 5)

    def test_feature_names_length_matches_columns(self):
        result = _make_result(N=5, n_nontrivial=4)
        Phi_raw, names = build_features(result)
        assert len(names) == Phi_raw.shape[1]

    def test_feature_names_contain_expected_entries(self):
        result = _make_result(N=5, n_nontrivial=3)
        _, names = build_features(result, K=3)
        assert "log|λ_1|" in names
        assert "sign(λ_1)" in names
        assert "|F|" in names
        assert "f_∥" in names
        assert "U" in names
        assert "tr(H)" in names
        assert "logdet(H_+)" in names

    def test_K_exceeds_n_nontrivial_raises(self):
        result = _make_result(N=5, n_nontrivial=3)
        with pytest.raises(ValueError, match="K=10"):
            build_features(result, K=10)

    def test_log_abs_eigenvalue_block(self):
        """log|λ| columns should equal log(|eigenvalue|)."""
        result = _make_result(N=4, n_nontrivial=2, seed=1)
        Phi_raw, _ = build_features(result, K=2)
        expected = np.log(np.abs(result.eigenvalues[:, :2]) + 1e-30)
        assert_allclose(Phi_raw[:, :2], expected, atol=1e-12)

    def test_sign_block_values_in_minus1_0_1(self):
        result = _make_result(N=6, n_nontrivial=3)
        Phi_raw, _ = build_features(result, K=3)
        sign_block = Phi_raw[:, 3:6]
        assert set(sign_block.flatten()).issubset({-1.0, 0.0, 1.0})

    def test_F_norm_column_matches_result(self):
        result = _make_result(N=5, n_nontrivial=2)
        Phi_raw, names = build_features(result, K=2)
        col = names.index("|F|")
        assert_allclose(Phi_raw[:, col], result.F_norm)

    def test_U_column_matches_result(self):
        result = _make_result(N=5, n_nontrivial=2)
        Phi_raw, names = build_features(result, K=2)
        col = names.index("U")
        assert_allclose(Phi_raw[:, col], result.U)

    def test_tr_H_is_sum_of_eigenvalues(self):
        result = _make_result(N=5, n_nontrivial=4)
        Phi_raw, names = build_features(result)
        col = names.index("tr(H)")
        expected = result.eigenvalues.sum(axis=1)
        assert_allclose(Phi_raw[:, col], expected, atol=1e-12)

    def test_logdet_Hplus_uses_only_positive_eigenvalues(self):
        """logdet(H_+) = Σ_{k: λ_k > 0} log(λ_k); negatives contribute 0."""
        result = _make_result(N=3, n_nontrivial=4, seed=7)
        Phi_raw, names = build_features(result)
        col = names.index("logdet(H_+)")
        for i in range(3):
            pos = result.eigenvalues[i][result.eigenvalues[i] > 0]
            expected = np.log(pos + 1e-30).sum() if pos.size > 0 else 0.0
            assert_allclose(Phi_raw[i, col], expected, atol=1e-10)

    def test_no_nans_in_output(self):
        result = _make_result(N=8, n_nontrivial=5)
        Phi_raw, _ = build_features(result)
        assert np.all(np.isfinite(Phi_raw))


# ── standardise ───────────────────────────────────────────────────────────────

class TestStandardise:
    def test_output_shape_preserved(self):
        X = np.random.default_rng(0).standard_normal((10, 7))
        Phi, mean, std = standardise(X)
        assert Phi.shape == X.shape

    def test_mean_near_zero_after_standardise(self):
        X = np.random.default_rng(1).standard_normal((50, 6)) * 5 + 3
        Phi, _, _ = standardise(X)
        assert_allclose(Phi.mean(axis=0), np.zeros(6), atol=1e-12)

    def test_std_near_one_after_standardise(self):
        X = np.random.default_rng(2).standard_normal((50, 6)) * 5 + 3
        Phi, _, _ = standardise(X)
        assert_allclose(Phi.std(axis=0), np.ones(6), atol=1e-12)

    def test_constant_column_becomes_zero(self):
        X = np.random.default_rng(3).standard_normal((10, 4))
        X[:, 2] = 7.0   # constant column
        Phi, _, std = standardise(X)
        assert_allclose(Phi[:, 2], np.zeros(10), atol=1e-12)
        assert std[2] == 0.0

    def test_returned_mean_and_std_are_correct(self):
        X = np.random.default_rng(4).standard_normal((20, 3))
        _, mean, std = standardise(X)
        assert_allclose(mean, X.mean(axis=0), atol=1e-12)
        assert_allclose(std,  X.std(axis=0),  atol=1e-12)

    def test_no_nans_even_with_constant_column(self):
        X = np.ones((5, 3))
        Phi, _, _ = standardise(X)
        assert np.all(np.isfinite(Phi))


# ── run ───────────────────────────────────────────────────────────────────────

class TestStep2Run:
    def test_returns_step2result(self):
        result = _make_result(N=10, n_nontrivial=4)
        out = run(result)
        assert isinstance(out, Step2Result)

    def test_Phi_shape(self):
        result = _make_result(N=10, n_nontrivial=4)
        out = run(result)
        assert out.Phi.shape == (10, 2 * 4 + 5)

    def test_Phi_raw_shape_matches_Phi(self):
        result = _make_result(N=10, n_nontrivial=4)
        out = run(result)
        assert out.Phi.shape == out.Phi_raw.shape

    def test_K_recorded_correctly(self):
        result = _make_result(N=10, n_nontrivial=6)
        out = run(result, K=4)
        assert out.K == 4

    def test_Phi_columns_standardised(self):
        result = _make_result(N=50, n_nontrivial=4)
        out = run(result)
        nonconstant = out.std > 0
        assert_allclose(out.Phi[:, nonconstant].mean(axis=0),
                        np.zeros(nonconstant.sum()), atol=1e-10)
        assert_allclose(out.Phi[:, nonconstant].std(axis=0),
                        np.ones(nonconstant.sum()),  atol=1e-10)

    def test_feature_names_count(self):
        K = 3
        result = _make_result(N=5, n_nontrivial=3)
        out = run(result, K=K)
        assert len(out.feature_names) == 2 * K + 5

    def test_mean_std_shapes(self):
        result = _make_result(N=8, n_nontrivial=5)
        out = run(result)
        assert out.mean.shape == (out.Phi.shape[1],)
        assert out.std.shape  == (out.Phi.shape[1],)

    def test_Phi_raw_recoverable_from_Phi(self):
        """Phi_raw == Phi * std + mean (where std > 0)."""
        result = _make_result(N=20, n_nontrivial=4)
        out = run(result)
        nonconstant = out.std > 0
        recovered = out.Phi[:, nonconstant] * out.std[nonconstant] + out.mean[nonconstant]
        assert_allclose(recovered, out.Phi_raw[:, nonconstant], atol=1e-10)
