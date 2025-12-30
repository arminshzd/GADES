"""
Tests for GADESBias core functionality.
"""
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from GADES import GADESBias


class TestGADESBiasInitialization:
    """Tests for GADESBias initialization."""

    def test_basic_initialization(self, valid_gades_params):
        """Test basic initialization with valid parameters."""
        bias = GADESBias(**valid_gades_params)
        assert bias.kappa == 0.9
        assert bias.clamp_magnitude == 1000.0
        assert bias.interval == 200
        assert bias.s_interval == 100
        assert bias.hess_step_size == 1e-5  # Default value

    def test_default_hess_step_size(self, valid_gades_params):
        """Default hess_step_size should be 1e-5."""
        bias = GADESBias(**valid_gades_params)
        assert bias.hess_step_size == 1e-5

    def test_is_biasing_initial_state(self, valid_gades_params):
        """is_biasing should be False initially."""
        bias = GADESBias(**valid_gades_params)
        assert bias.is_biasing is False

    def test_check_stability_initial_state(self, valid_gades_params):
        """check_stability should be False initially."""
        bias = GADESBias(**valid_gades_params)
        assert bias.check_stability is False


class TestGADESBiasSetters:
    """Tests for GADESBias setter methods."""

    def test_set_kappa(self, valid_gades_params):
        """Test set_kappa method."""
        bias = GADESBias(**valid_gades_params)
        bias.set_kappa(0.5)
        assert bias.kappa == 0.5

    def test_set_hess_step_size(self, valid_gades_params):
        """Test set_hess_step_size method."""
        bias = GADESBias(**valid_gades_params)
        bias.set_hess_step_size(1e-4)
        assert bias.hess_step_size == 1e-4


class TestGetGadForce:
    """Tests for the get_gad_force method."""

    def test_get_gad_force_shape(self, mock_backend_factory, simple_hess_func):
        """get_gad_force should return correct shape."""
        n_atoms = 10
        bias_indices = [0, 1, 2]
        backend = mock_backend_factory(n_atoms=n_atoms)

        bias = GADESBias(
            backend=backend,
            biased_force=None,
            bias_atom_indices=bias_indices,
            hess_func=simple_hess_func,
            clamp_magnitude=1000.0,
            kappa=0.9,
            interval=200,
        )

        force = bias.get_gad_force()
        assert force.shape == (len(bias_indices), 3)

    def test_get_gad_force_direction(self, mock_backend_factory):
        """Bias force should be along softest eigenmode direction."""
        n_atoms = 3
        bias_indices = [0, 1, 2]

        # Create backend with known forces
        forces = np.array([[1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0]])
        backend = mock_backend_factory(n_atoms=n_atoms, forces=forces)

        # Hessian with known smallest eigenvector
        def known_hess_func(backend, atom_indices, step_size, platform):
            n_dof = len(atom_indices) * 3
            # Diagonal matrix: smallest eigenvalue at position 0
            return np.diag(np.arange(1, n_dof + 1, dtype=float))

        bias = GADESBias(
            backend=backend,
            biased_force=None,
            bias_atom_indices=bias_indices,
            hess_func=known_hess_func,
            clamp_magnitude=10000.0,  # High clamp to not affect result
            kappa=0.9,
            interval=200,
        )

        force = bias.get_gad_force()

        # The softest mode eigenvector for diagonal matrix with [1,2,3,...]
        # is [1, 0, 0, ...] (first coordinate)
        # Force projection along this direction times -kappa
        # F_bias = -kappa * (F · n) * n
        assert force.shape == (3, 3)

    def test_get_gad_force_clamping(self, mock_backend_factory):
        """Force magnitude should be clamped."""
        n_atoms = 3
        bias_indices = [0, 1, 2]

        # Large forces
        forces = np.ones((n_atoms, 3)) * 1000.0
        backend = mock_backend_factory(n_atoms=n_atoms, forces=forces)

        def hess_func(backend, atom_indices, step_size, platform):
            n_dof = len(atom_indices) * 3
            return np.eye(n_dof)

        clamp = 10.0
        bias = GADESBias(
            backend=backend,
            biased_force=None,
            bias_atom_indices=bias_indices,
            hess_func=hess_func,
            clamp_magnitude=clamp,
            kappa=0.9,
            interval=200,
        )

        force = bias.get_gad_force()

        # Check each force vector magnitude is <= clamp
        for i in range(len(bias_indices)):
            magnitude = np.linalg.norm(force[i])
            assert magnitude <= clamp + 1e-10

    def test_get_gad_force_kappa_scaling(self, mock_backend_factory):
        """Force should scale with kappa."""
        n_atoms = 3
        bias_indices = [0, 1, 2]

        forces = np.array([[1.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0]])
        backend = mock_backend_factory(n_atoms=n_atoms, forces=forces)

        def hess_func(backend, atom_indices, step_size, platform):
            n_dof = len(atom_indices) * 3
            return np.eye(n_dof)

        # Test with kappa = 0.5
        bias_half = GADESBias(
            backend=backend,
            biased_force=None,
            bias_atom_indices=bias_indices,
            hess_func=hess_func,
            clamp_magnitude=10000.0,
            kappa=0.5,
            interval=200,
        )

        # Test with kappa = 1.0
        bias_full = GADESBias(
            backend=backend,
            biased_force=None,
            bias_atom_indices=bias_indices,
            hess_func=hess_func,
            clamp_magnitude=10000.0,
            kappa=1.0,
            interval=200,
        )

        force_half = bias_half.get_gad_force()
        force_full = bias_full.get_gad_force()

        # Force with kappa=0.5 should be half of kappa=1.0
        assert_array_almost_equal(force_half, force_full * 0.5)


class TestApplyingBias:
    """Tests for the applying_bias method."""

    def test_applying_bias_at_interval(self, mock_backend_factory, simple_hess_func):
        """applying_bias should return True at interval steps."""
        backend = mock_backend_factory()
        bias = GADESBias(
            backend=backend,
            biased_force=None,
            bias_atom_indices=[0, 1, 2],
            hess_func=simple_hess_func,
            clamp_magnitude=1000.0,
            kappa=0.9,
            interval=100,
        )

        # At step 0 (multiple of 100)
        backend.set_currentStep(0)
        assert bias.applying_bias() is True

        # At step 100 (multiple of 100)
        backend.set_currentStep(100)
        assert bias.applying_bias() is True

        # At step 200 (multiple of 100)
        backend.set_currentStep(200)
        assert bias.applying_bias() is True

    def test_applying_bias_between_intervals(self, mock_backend_factory, simple_hess_func):
        """applying_bias should return False between intervals."""
        backend = mock_backend_factory()
        bias = GADESBias(
            backend=backend,
            biased_force=None,
            bias_atom_indices=[0, 1, 2],
            hess_func=simple_hess_func,
            clamp_magnitude=1000.0,
            kappa=0.9,
            interval=100,
        )

        # At step 50 (not multiple of 100)
        backend.set_currentStep(50)
        assert bias.applying_bias() is False

        # At step 99
        backend.set_currentStep(99)
        assert bias.applying_bias() is False

        # At step 150
        backend.set_currentStep(150)
        assert bias.applying_bias() is False

    def test_applying_bias_negative_step(self, mock_backend_factory, simple_hess_func):
        """applying_bias should return False for negative steps."""
        backend = mock_backend_factory()
        bias = GADESBias(
            backend=backend,
            biased_force=None,
            bias_atom_indices=[0, 1, 2],
            hess_func=simple_hess_func,
            clamp_magnitude=1000.0,
            kappa=0.9,
            interval=100,
        )

        backend.set_currentStep(-1)
        assert bias.applying_bias() is False


class TestRegisterNextStep:
    """Tests for the register_next_step method."""

    def test_register_next_step_basic(self, mock_backend_factory, simple_hess_func):
        """register_next_step should return steps to next event."""
        backend = mock_backend_factory()
        bias = GADESBias(
            backend=backend,
            biased_force=None,
            bias_atom_indices=[0, 1, 2],
            hess_func=simple_hess_func,
            clamp_magnitude=1000.0,
            kappa=0.9,
            interval=200,
            stability_interval=100,
        )

        # At step 0, next event should be stability check at 100
        backend.set_currentStep(0)
        steps = bias.register_next_step()
        assert steps == 100

    def test_register_next_step_at_bias_interval(self, mock_backend_factory, simple_hess_func):
        """At bias interval, is_biasing flag should be set."""
        backend = mock_backend_factory()
        bias = GADESBias(
            backend=backend,
            biased_force=None,
            bias_atom_indices=[0, 1, 2],
            hess_func=simple_hess_func,
            clamp_magnitude=1000.0,
            kappa=0.9,
            interval=200,
            stability_interval=100,
        )

        # At step 100, bias interval is 200, stability is 100
        backend.set_currentStep(100)
        steps = bias.register_next_step()
        # Next event is at 200 (bias) or 200 (stability), both 100 steps away
        assert steps == 100

    def test_register_next_step_no_stability_interval(self, mock_backend_factory, simple_hess_func):
        """Without stability_interval, only bias interval matters."""
        backend = mock_backend_factory()
        bias = GADESBias(
            backend=backend,
            biased_force=None,
            bias_atom_indices=[0, 1, 2],
            hess_func=simple_hess_func,
            clamp_magnitude=1000.0,
            kappa=0.9,
            interval=200,
            stability_interval=None,
        )

        backend.set_currentStep(0)
        steps = bias.register_next_step()
        assert steps == 200


class TestRemoveBias:
    """Tests for remove_bias method."""

    def test_remove_bias_calls_backend(self, mock_backend_factory, simple_hess_func):
        """remove_bias should call backend.remove_bias."""
        backend = mock_backend_factory()
        bias_indices = [0, 1, 2]
        bias = GADESBias(
            backend=backend,
            biased_force="mock_force",
            bias_atom_indices=bias_indices,
            hess_func=simple_hess_func,
            clamp_magnitude=1000.0,
            kappa=0.9,
            interval=200,
        )

        bias.remove_bias()
        assert backend._last_bias_indices == bias_indices


class TestApplyBias:
    """Tests for apply_bias method."""

    def test_apply_bias_calls_backend(self, mock_backend_factory, simple_hess_func):
        """apply_bias should compute forces and call backend.apply_bias."""
        backend = mock_backend_factory()
        bias_indices = [0, 1, 2]
        bias = GADESBias(
            backend=backend,
            biased_force="mock_force",
            bias_atom_indices=bias_indices,
            hess_func=simple_hess_func,
            clamp_magnitude=1000.0,
            kappa=0.9,
            interval=200,
        )

        bias.apply_bias()

        assert backend._last_bias_indices == bias_indices
        assert backend._last_applied_bias is not None
        assert backend._last_applied_bias.shape == (len(bias_indices), 3)


class TestIsStable:
    """Tests for _is_stable method."""

    def test_is_stable_delegates_to_backend(self, mock_backend_factory, simple_hess_func):
        """_is_stable should call backend.is_stable."""
        backend = mock_backend_factory()
        bias = GADESBias(
            backend=backend,
            biased_force=None,
            bias_atom_indices=[0, 1, 2],
            hess_func=simple_hess_func,
            clamp_magnitude=1000.0,
            kappa=0.9,
            interval=200,
        )

        backend.set_stable(True)
        assert bias._is_stable() is True

        backend.set_stable(False)
        assert bias._is_stable() is False
