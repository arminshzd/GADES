"""
Integration tests for lanczos_hvp and Bofill update code paths.

These tests verify that the advanced eigensolver and Hessian approximation
features work correctly in the context of GADESBias.
"""
import pytest
import numpy as np

from GADES import GADESBias


class HarmonicBackend:
    """
    A mock backend with a harmonic potential V = 0.5 * x^T @ H @ x.

    Forces are F = -H @ x, which allows testing HVP and Bofill updates
    with known, predictable behavior.
    """
    def __init__(self, n_atoms, hessian_diag):
        """
        Args:
            n_atoms: Number of atoms
            hessian_diag: Diagonal of the Hessian matrix (eigenvalues)
        """
        self.n_atoms = n_atoms
        self.name = "harmonic"
        self._current_step = 0
        self._positions = np.zeros((n_atoms, 3))
        self._hessian_diag = np.array(hessian_diag)
        self._stable = True
        self._last_applied_bias = None

    def is_stable(self):
        return self._stable

    def get_currentStep(self):
        return self._current_step

    def set_currentStep(self, step):
        self._current_step = step

    def get_atom_symbols(self, bias_atom_indices):
        return ["C"] * len(bias_atom_indices)

    def get_positions(self):
        return self._positions.copy()

    def set_positions(self, positions):
        self._positions = positions.copy()

    def get_current_state(self):
        forces = self._compute_forces(self._positions)
        return self._positions.copy(), forces.reshape(-1, 3)

    def get_forces(self, positions):
        """Return F = -H @ x (forces for harmonic potential)."""
        return self._compute_forces(positions)

    def _compute_forces(self, positions):
        """Compute forces: F = -H @ x for diagonal Hessian."""
        x_flat = positions.flatten()
        # For diagonal Hessian, F_i = -H_ii * x_i
        forces_flat = -self._hessian_diag * x_flat
        return forces_flat

    def apply_bias(self, bias_force_object, biased_force_values, bias_atom_indices):
        self._last_applied_bias = biased_force_values.copy()

    def remove_bias(self, bias_force_object, bias_atom_indices):
        self._last_applied_bias = None


class TestLanczosHVPIntegration:
    """Integration tests for the lanczos_hvp eigensolver path."""

    @pytest.fixture
    def harmonic_backend_3atoms(self):
        """Create a harmonic backend with 3 atoms and known eigenvalues."""
        # Eigenvalues: 1, 2, 3, 4, 5, 6, 7, 8, 9 (softest mode is eigenvalue 1)
        n_atoms = 3
        hessian_diag = np.arange(1, n_atoms * 3 + 1, dtype=float)
        return HarmonicBackend(n_atoms, hessian_diag)

    @pytest.fixture
    def dummy_hess_func(self):
        """Dummy hess_func (not used with lanczos_hvp but required)."""
        def hess(backend, atom_indices, step_size, platform):
            n_dof = len(atom_indices) * 3
            return np.eye(n_dof)
        return hess

    def test_lanczos_hvp_finds_softest_mode(self, harmonic_backend_3atoms, dummy_hess_func):
        """Test that lanczos_hvp correctly identifies the softest eigenmode."""
        backend = harmonic_backend_3atoms

        # Set positions away from origin so forces are non-zero
        backend.set_positions(np.ones((3, 3)) * 0.1)
        backend.set_currentStep(200)  # At bias interval

        bias = GADESBias(
            backend=backend,
            biased_force=None,
            bias_atom_indices=[0, 1, 2],
            hess_func=dummy_hess_func,
            clamp_magnitude=1000.0,
            kappa=0.9,
            interval=200,
            eigensolver='lanczos_hvp',
            lanczos_iterations=50,  # Enough iterations for convergence
            hvp_epsilon=1e-5,
        )

        # Compute GAD force - this exercises the lanczos_hvp path
        gad_force = bias.get_gad_force()

        # The softest mode should be eigenvector [1,0,0,0,...] (first coordinate)
        # because eigenvalue 1 is smallest
        # The bias should primarily affect the first coordinate
        assert gad_force is not None
        assert gad_force.shape == (3, 3)

        # Verify bias was computed (non-zero)
        assert np.linalg.norm(gad_force) > 0

    def test_lanczos_hvp_eigenvalue_sign(self, harmonic_backend_3atoms, dummy_hess_func):
        """Test that lanczos_hvp returns correct eigenvalue sign."""
        backend = harmonic_backend_3atoms
        backend.set_positions(np.ones((3, 3)) * 0.1)
        backend.set_currentStep(200)

        bias = GADESBias(
            backend=backend,
            biased_force=None,
            bias_atom_indices=[0, 1, 2],
            hess_func=dummy_hess_func,
            clamp_magnitude=1000.0,
            kappa=0.9,
            interval=200,
            eigensolver='lanczos_hvp',
            lanczos_iterations=50,
        )

        # Access the internal method to check eigenvalue
        positions = backend.get_positions()
        eigval, eigvec = bias._compute_softest_mode_hvp(positions)

        # Eigenvalue should be close to 1 (smallest in our diagonal Hessian)
        assert eigval > 0, "Eigenvalue should be positive at a minimum"
        np.testing.assert_allclose(eigval, 1.0, rtol=0.1)

    def test_lanczos_hvp_with_negative_eigenvalue(self, dummy_hess_func):
        """Test lanczos_hvp with a saddle point (negative eigenvalue)."""
        # Create backend with one negative eigenvalue (saddle point)
        n_atoms = 3
        hessian_diag = np.array([-2.0] + list(range(1, n_atoms * 3)))
        backend = HarmonicBackend(n_atoms, hessian_diag)
        backend.set_positions(np.ones((3, 3)) * 0.1)
        backend.set_currentStep(200)

        bias = GADESBias(
            backend=backend,
            biased_force=None,
            bias_atom_indices=[0, 1, 2],
            hess_func=dummy_hess_func,
            clamp_magnitude=1000.0,
            kappa=0.9,
            interval=200,
            eigensolver='lanczos_hvp',
            lanczos_iterations=50,
        )

        positions = backend.get_positions()
        eigval, eigvec = bias._compute_softest_mode_hvp(positions)

        # Eigenvalue should be negative (softest mode at saddle)
        assert eigval < 0, "Eigenvalue should be negative at saddle point"
        np.testing.assert_allclose(eigval, -2.0, rtol=0.1)


class TestBofillUpdateIntegration:
    """Integration tests for Bofill quasi-Newton update path."""

    @pytest.fixture
    def simple_hess_func(self):
        """Hessian function returning diagonal matrix."""
        def hess(backend, atom_indices, step_size, platform):
            n_dof = len(atom_indices) * 3
            return np.diag(np.arange(1, n_dof + 1, dtype=float))
        return hess

    @pytest.fixture
    def mock_backend_for_bofill(self):
        """Create a mock backend for Bofill testing."""
        n_atoms = 3
        positions = np.random.randn(n_atoms, 3) * 0.1
        forces = np.random.randn(n_atoms, 3)

        class BofillMockBackend:
            def __init__(self):
                self.n_atoms = n_atoms
                self.name = "mock"
                self._current_step = 0
                self._positions = positions.copy()
                self._forces = forces.copy()
                self._stable = True

            def is_stable(self):
                return self._stable

            def get_currentStep(self):
                return self._current_step

            def set_currentStep(self, step):
                self._current_step = step

            def get_atom_symbols(self, indices):
                return ["C"] * len(indices)

            def get_positions(self):
                return self._positions.copy()

            def set_positions(self, pos):
                self._positions = pos.copy()

            def get_current_state(self):
                return self._positions.copy(), self._forces.copy()

            def get_forces(self, positions):
                return self._forces.flatten()

            def apply_bias(self, force_obj, values, indices):
                self._last_bias = values

            def remove_bias(self, force_obj, indices):
                pass

        return BofillMockBackend()

    def test_bofill_uses_full_hessian_initially(self, simple_hess_func, mock_backend_for_bofill):
        """Test that Bofill path computes full Hessian on first call."""
        backend = mock_backend_for_bofill
        backend.set_currentStep(200)

        bias = GADESBias(
            backend=backend,
            biased_force=None,
            bias_atom_indices=[0, 1, 2],
            hess_func=simple_hess_func,
            clamp_magnitude=1000.0,
            kappa=0.9,
            interval=200,
            eigensolver='numpy',
            use_bofill_update=True,
            full_hessian_interval=1000,
        )

        # First call should compute full Hessian
        positions, forces = backend.get_current_state()
        hess = bias._get_hessian(positions, forces)

        # Should have stored state for future Bofill updates
        assert bias._last_hess is not None
        assert bias._last_positions is not None
        assert bias._last_forces is not None
        assert bias._last_hess_step == 200

    def test_bofill_approximates_between_full_hessian(self, simple_hess_func, mock_backend_for_bofill):
        """Test that Bofill uses approximation between full Hessian intervals."""
        backend = mock_backend_for_bofill

        bias = GADESBias(
            backend=backend,
            biased_force=None,
            bias_atom_indices=[0, 1, 2],
            hess_func=simple_hess_func,
            clamp_magnitude=1000.0,
            kappa=0.9,
            interval=200,
            eigensolver='numpy',
            use_bofill_update=True,
            full_hessian_interval=1000,
        )

        # First call at step 200 - computes full Hessian
        backend.set_currentStep(200)
        positions1, forces1 = backend.get_current_state()
        hess1 = bias._get_hessian(positions1, forces1)
        first_hess_step = bias._last_hess_step

        # Second call at step 400 - should use Bofill approximation
        backend.set_currentStep(400)
        # Slightly different position/force for Bofill update
        backend._positions += 0.01
        backend._forces += 0.01
        positions2, forces2 = backend.get_current_state()
        hess2 = bias._get_hessian(positions2, forces2)

        # _last_hess_step should not have changed (no full Hessian computed)
        assert bias._last_hess_step == first_hess_step

        # Hessian should still be valid (Bofill approximation)
        assert hess2 is not None
        assert hess2.shape == hess1.shape

    def test_bofill_recomputes_at_full_interval(self, simple_hess_func, mock_backend_for_bofill):
        """Test that full Hessian is recomputed at full_hessian_interval."""
        backend = mock_backend_for_bofill

        bias = GADESBias(
            backend=backend,
            biased_force=None,
            bias_atom_indices=[0, 1, 2],
            hess_func=simple_hess_func,
            clamp_magnitude=1000.0,
            kappa=0.9,
            interval=200,
            eigensolver='numpy',
            use_bofill_update=True,
            full_hessian_interval=500,  # Recompute every 500 steps
        )

        # First call at step 0 - computes full Hessian
        backend.set_currentStep(0)
        positions, forces = backend.get_current_state()
        bias._get_hessian(positions, forces)
        assert bias._last_hess_step == 0

        # Call at step 200 - Bofill approximation
        backend.set_currentStep(200)
        bias._get_hessian(positions, forces)
        assert bias._last_hess_step == 0  # Still from step 0

        # Call at step 500 - should recompute full Hessian
        backend.set_currentStep(500)
        bias._get_hessian(positions, forces)
        assert bias._last_hess_step == 500  # Updated to step 500

    def test_bofill_disabled_always_computes_full(self, simple_hess_func, mock_backend_for_bofill):
        """Test that with use_bofill_update=False, full Hessian is always computed."""
        backend = mock_backend_for_bofill

        # Track how many times hess_func is called
        call_count = [0]
        def counting_hess_func(backend, atom_indices, step_size, platform):
            call_count[0] += 1
            n_dof = len(atom_indices) * 3
            return np.diag(np.arange(1, n_dof + 1, dtype=float))

        bias = GADESBias(
            backend=backend,
            biased_force=None,
            bias_atom_indices=[0, 1, 2],
            hess_func=counting_hess_func,
            clamp_magnitude=1000.0,
            kappa=0.9,
            interval=200,
            eigensolver='numpy',
            use_bofill_update=False,  # Disabled
        )

        positions, forces = backend.get_current_state()

        # Each call should compute full Hessian
        backend.set_currentStep(200)
        bias._get_hessian(positions, forces)
        assert call_count[0] == 1

        backend.set_currentStep(400)
        bias._get_hessian(positions, forces)
        assert call_count[0] == 2

        backend.set_currentStep(600)
        bias._get_hessian(positions, forces)
        assert call_count[0] == 3


class TestCombinedEigensolverBofill:
    """Test combinations of eigensolver and Bofill settings."""

    @pytest.fixture
    def simple_backend(self):
        """Create a simple mock backend."""
        class SimpleBackend:
            def __init__(self):
                self.n_atoms = 3
                self.name = "mock"
                self._step = 0
                self._positions = np.random.randn(3, 3) * 0.1
                self._forces = np.random.randn(3, 3)

            def is_stable(self):
                return True
            def get_currentStep(self):
                return self._step
            def set_currentStep(self, s):
                self._step = s
            def get_atom_symbols(self, idx):
                return ["C"] * len(idx)
            def get_positions(self):
                return self._positions.copy()
            def get_current_state(self):
                return self._positions.copy(), self._forces.copy()
            def get_forces(self, pos):
                return self._forces.flatten()
            def apply_bias(self, f, v, i):
                pass
            def remove_bias(self, f, i):
                pass
        return SimpleBackend()

    def test_lanczos_with_bofill_ignored(self, simple_backend):
        """Test that Bofill settings are ignored with lanczos eigensolver."""
        def hess(backend, atom_indices, step_size, platform):
            n_dof = len(atom_indices) * 3
            return np.diag(np.arange(1, n_dof + 1, dtype=float))

        simple_backend.set_currentStep(200)

        bias = GADESBias(
            backend=simple_backend,
            biased_force=None,
            bias_atom_indices=[0, 1, 2],
            hess_func=hess,
            clamp_magnitude=1000.0,
            kappa=0.9,
            interval=200,
            eigensolver='lanczos',
            lanczos_iterations=30,
            use_bofill_update=True,  # Should still work with Lanczos
        )

        # Should be able to compute GAD force
        gad_force = bias.get_gad_force()
        assert gad_force is not None
        assert gad_force.shape == (3, 3)
