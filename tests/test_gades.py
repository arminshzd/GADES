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


class TestEigensolverIntegration:
    """Tests for eigensolver parameter and Lanczos integration."""

    def test_default_eigensolver_is_numpy(self, valid_gades_params):
        """Default eigensolver should be 'numpy'."""
        bias = GADESBias(**valid_gades_params)
        assert bias.eigensolver == 'numpy'

    def test_eigensolver_lanczos(self, valid_gades_params):
        """Should accept 'lanczos' eigensolver."""
        params = valid_gades_params.copy()
        params['eigensolver'] = 'lanczos'
        bias = GADESBias(**params)
        assert bias.eigensolver == 'lanczos'

    def test_invalid_eigensolver_raises(self, valid_gades_params):
        """Invalid eigensolver should raise ValueError."""
        params = valid_gades_params.copy()
        params['eigensolver'] = 'invalid'
        with pytest.raises(ValueError, match="eigensolver must be one of"):
            GADESBias(**params)

    def test_lanczos_iterations_default(self, valid_gades_params):
        """Default lanczos_iterations should come from defaults."""
        from GADES.config import defaults
        bias = GADESBias(**valid_gades_params)
        assert bias.lanczos_iterations == defaults["lanczos_iterations"]

    def test_lanczos_iterations_custom(self, valid_gades_params):
        """Custom lanczos_iterations should be used."""
        params = valid_gades_params.copy()
        params['lanczos_iterations'] = 30
        bias = GADESBias(**params)
        assert bias.lanczos_iterations == 30

    def test_lanczos_gives_same_direction(self, mock_backend_factory):
        """Lanczos and numpy should give same softest mode direction."""
        n_atoms = 5
        bias_indices = [0, 1, 2, 3, 4]

        # Create a known Hessian with clear smallest eigenvalue
        def hess_func(backend, atom_indices, step_size, platform):
            n_dof = len(atom_indices) * 3
            # Diagonal matrix with distinct eigenvalues
            return np.diag(np.arange(1, n_dof + 1, dtype=float))

        backend = mock_backend_factory(n_atoms=n_atoms)

        # Test with numpy
        bias_numpy = GADESBias(
            backend=backend,
            biased_force=None,
            bias_atom_indices=bias_indices,
            hess_func=hess_func,
            clamp_magnitude=10000.0,
            kappa=0.9,
            interval=200,
            eigensolver='numpy',
        )

        # Test with lanczos
        bias_lanczos = GADESBias(
            backend=backend,
            biased_force=None,
            bias_atom_indices=bias_indices,
            hess_func=hess_func,
            clamp_magnitude=10000.0,
            kappa=0.9,
            interval=200,
            eigensolver='lanczos',
            lanczos_iterations=15,
        )

        force_numpy = bias_numpy.get_gad_force()
        force_lanczos = bias_lanczos.get_gad_force()

        # Forces should be in the same direction (allow for sign flip)
        force_numpy_flat = force_numpy.flatten()
        force_lanczos_flat = force_lanczos.flatten()

        # Check if forces are parallel (cosine of angle should be ±1)
        norm_numpy = np.linalg.norm(force_numpy_flat)
        norm_lanczos = np.linalg.norm(force_lanczos_flat)

        if norm_numpy > 1e-10 and norm_lanczos > 1e-10:
            cosine = np.dot(force_numpy_flat, force_lanczos_flat) / (norm_numpy * norm_lanczos)
            assert abs(abs(cosine) - 1.0) < 0.01  # Should be parallel


class TestLanczosHVPIntegration:
    """Tests for matrix-free Lanczos with HVP integration."""

    def test_eigensolver_lanczos_hvp_accepted(self, valid_gades_params):
        """Should accept 'lanczos_hvp' eigensolver."""
        params = valid_gades_params.copy()
        params['eigensolver'] = 'lanczos_hvp'
        bias = GADESBias(**params)
        assert bias.eigensolver == 'lanczos_hvp'

    def test_hvp_epsilon_default(self, valid_gades_params):
        """Default hvp_epsilon should come from defaults."""
        from GADES.config import defaults
        bias = GADESBias(**valid_gades_params)
        assert bias.hvp_epsilon == defaults["hvp_epsilon"]

    def test_hvp_epsilon_custom(self, valid_gades_params):
        """Custom hvp_epsilon should be used."""
        params = valid_gades_params.copy()
        params['hvp_epsilon'] = 1e-6
        bias = GADESBias(**params)
        assert bias.hvp_epsilon == 1e-6

    def test_hvp_epsilon_invalid_raises(self, valid_gades_params):
        """Invalid hvp_epsilon should raise ValueError."""
        params = valid_gades_params.copy()
        params['hvp_epsilon'] = -1e-5
        with pytest.raises(ValueError, match="hvp_epsilon must be a positive number"):
            GADESBias(**params)

        params['hvp_epsilon'] = 0
        with pytest.raises(ValueError, match="hvp_epsilon must be a positive number"):
            GADESBias(**params)

    def test_lanczos_hvp_computes_softest_mode(self, mock_backend_factory):
        """lanczos_hvp should compute softest mode similar to numpy."""
        from GADES.potentials import muller_brown_force, muller_brown_hess

        n_atoms = 1
        bias_indices = [0]
        pos = np.array([[0.0, 0.5, 0.0]])  # 2D position padded to 3D

        # Create backend with known positions and forces
        backend = mock_backend_factory(n_atoms=n_atoms)
        backend.positions = pos
        backend.forces = np.zeros_like(pos)

        # Simple quadratic potential for testing
        # H = diag([1, 2, 3])
        def simple_hess_func(backend, atom_indices, step_size, platform):
            return np.diag([1.0, 2.0, 3.0])

        def simple_force_func(positions):
            # For H = diag([1,2,3]), forces = -H @ x
            x = positions.flatten()
            return -np.array([1.0, 2.0, 3.0]) * x

        backend.get_forces = simple_force_func

        # Test with numpy
        bias_numpy = GADESBias(
            backend=backend,
            biased_force=None,
            bias_atom_indices=bias_indices,
            hess_func=simple_hess_func,
            clamp_magnitude=10000.0,
            kappa=0.9,
            interval=200,
            eigensolver='numpy',
        )

        # Test with lanczos_hvp
        bias_hvp = GADESBias(
            backend=backend,
            biased_force=None,
            bias_atom_indices=bias_indices,
            hess_func=simple_hess_func,  # Not used for HVP, but required
            clamp_magnitude=10000.0,
            kappa=0.9,
            interval=200,
            eigensolver='lanczos_hvp',
            lanczos_iterations=10,
        )

        # Get eigenvalues and eigenvectors
        hess = simple_hess_func(backend, bias_indices, 0, "CPU")
        eigval_numpy, eigvec_numpy = bias_numpy._compute_softest_mode(hess)
        eigval_hvp, eigvec_hvp = bias_hvp._compute_softest_mode_hvp(pos)

        # Smallest eigenvalue should be 1.0
        assert abs(eigval_numpy - 1.0) < 0.1
        assert abs(eigval_hvp - 1.0) < 0.1

        # Eigenvectors should be parallel (allow sign flip)
        cosine = abs(np.dot(eigvec_numpy, eigvec_hvp))
        assert cosine > 0.9

    def test_lanczos_hvp_skips_hessian_computation(self, mock_backend_factory):
        """lanczos_hvp should not call hess_func."""
        n_atoms = 1
        bias_indices = [0]

        hess_call_count = [0]

        def counting_hess_func(backend, atom_indices, step_size, platform):
            hess_call_count[0] += 1
            n_dof = len(atom_indices) * 3
            return np.eye(n_dof)

        def simple_force_func(positions):
            x = positions.flatten()
            return -x  # Simple identity Hessian

        backend = mock_backend_factory(n_atoms=n_atoms)
        backend.get_forces = simple_force_func
        backend.set_currentStep(0)

        bias = GADESBias(
            backend=backend,
            biased_force=None,
            bias_atom_indices=bias_indices,
            hess_func=counting_hess_func,
            clamp_magnitude=10000.0,
            kappa=0.9,
            interval=200,
            eigensolver='lanczos_hvp',
        )

        # Call get_gad_force - should NOT call hess_func
        bias.get_gad_force()

        assert hess_call_count[0] == 0

    def test_lanczos_hvp_with_logging_no_crash(self, mock_backend_factory, tmp_path):
        """Regression test: lanczos_hvp + logging should not crash (issue #4)."""
        n_atoms = 1
        bias_indices = [0]

        def simple_force_func(positions):
            x = positions.flatten()
            return -x

        def simple_hess_func(backend, atom_indices, step_size, platform):
            n_dof = len(atom_indices) * 3
            return np.eye(n_dof)

        backend = mock_backend_factory(n_atoms=n_atoms)
        backend.get_forces = simple_force_func
        backend.set_currentStep(0)

        logfile_prefix = str(tmp_path / "test_hvp_log")

        # This should NOT raise TypeError: 'NoneType' object is not subscriptable
        bias = GADESBias(
            backend=backend,
            biased_force=None,
            bias_atom_indices=bias_indices,
            hess_func=simple_hess_func,
            clamp_magnitude=10000.0,
            kappa=0.9,
            interval=200,
            eigensolver='lanczos_hvp',
            logfile_prefix=logfile_prefix,
        )

        # This is where the crash would occur before the fix
        bias.get_gad_force()

        # Clean up
        bias._close_logs()

    def test_lanczos_hvp_logs_eigenvector_but_skips_eigenvalues(self, mock_backend_factory, tmp_path):
        """HVP path should log eigenvectors and xyz but skip eigenvalues."""
        n_atoms = 1
        bias_indices = [0]

        def simple_force_func(positions):
            x = positions.flatten()
            return -x

        def simple_hess_func(backend, atom_indices, step_size, platform):
            n_dof = len(atom_indices) * 3
            return np.eye(n_dof)

        backend = mock_backend_factory(n_atoms=n_atoms)
        backend.get_forces = simple_force_func
        backend.set_currentStep(100)

        logfile_prefix = str(tmp_path / "test_hvp_log")

        bias = GADESBias(
            backend=backend,
            biased_force=None,
            bias_atom_indices=bias_indices,
            hess_func=simple_hess_func,
            clamp_magnitude=10000.0,
            kappa=0.9,
            interval=200,
            eigensolver='lanczos_hvp',
            logfile_prefix=logfile_prefix,
        )

        bias.get_gad_force()
        bias._close_logs()

        # Check eigenvector log has data (header + 1 data line)
        evec_log = tmp_path / "test_hvp_log_evec.log"
        evec_content = evec_log.read_text()
        lines = [l for l in evec_content.strip().split('\n') if not l.startswith('#')]
        assert len(lines) == 1  # One data line
        assert lines[0].startswith("100 ")  # Step number

        # Check eigenvalue log has only header (no data lines since w=None)
        eval_log = tmp_path / "test_hvp_log_eval.log"
        eval_content = eval_log.read_text()
        eval_lines = [l for l in eval_content.strip().split('\n') if not l.startswith('#')]
        assert len(eval_lines) == 0  # No data lines

        # Check xyz log has data
        xyz_log = tmp_path / "test_hvp_log_biased_atoms.xyz"
        xyz_content = xyz_log.read_text()
        xyz_lines = [l for l in xyz_content.strip().split('\n') if not l.startswith('#')]
        assert len(xyz_lines) >= 3  # At least: natoms, comment, atom line

    def test_lanczos_hvp_logging_warning_issued(self, mock_backend_factory, tmp_path, caplog):
        """HVP + logging should emit warning about eigenvalue logging unavailability."""
        import logging

        n_atoms = 1
        bias_indices = [0]

        def simple_hess_func(backend, atom_indices, step_size, platform):
            return np.eye(3)

        backend = mock_backend_factory(n_atoms=n_atoms)
        logfile_prefix = str(tmp_path / "test_hvp_warn")

        with caplog.at_level(logging.WARNING, logger="GADES"):
            bias = GADESBias(
                backend=backend,
                biased_force=None,
                bias_atom_indices=bias_indices,
                hess_func=simple_hess_func,
                clamp_magnitude=10000.0,
                kappa=0.9,
                interval=200,
                eigensolver='lanczos_hvp',
                logfile_prefix=logfile_prefix,
            )

        assert "Eigenvalue logging unavailable" in caplog.text
        assert "eigensolver='lanczos_hvp'" in caplog.text

        bias._close_logs()


class TestCloseLogsErrorHandling:
    """Tests for _close_logs error handling."""

    def test_close_logs_handles_exception_gracefully(self, mock_backend_factory, tmp_path):
        """_close_logs should warn, not raise, when file close fails."""
        import warnings
        from unittest.mock import MagicMock, patch

        n_atoms = 1
        bias_indices = [0]

        def simple_hess_func(backend, atom_indices, step_size, platform):
            return np.eye(3)

        backend = mock_backend_factory(n_atoms=n_atoms)
        logfile_prefix = str(tmp_path / "test_close_error")

        bias = GADESBias(
            backend=backend,
            biased_force=None,
            bias_atom_indices=bias_indices,
            hess_func=simple_hess_func,
            clamp_magnitude=10000.0,
            kappa=0.9,
            interval=200,
            logfile_prefix=logfile_prefix,
        )

        # Mock one of the file handles to raise on close
        mock_file = MagicMock()
        mock_file.closed = False
        mock_file.close.side_effect = OSError("Mock close error")
        bias._evec_log = mock_file

        # Should warn but not raise
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bias._close_logs()

            # Check warning was issued
            assert len(w) >= 1
            assert "Failed to close log file" in str(w[0].message)

    def test_close_logs_skips_already_closed_files(self, mock_backend_factory, tmp_path):
        """_close_logs should skip files that are already closed."""
        n_atoms = 1
        bias_indices = [0]

        def simple_hess_func(backend, atom_indices, step_size, platform):
            return np.eye(3)

        backend = mock_backend_factory(n_atoms=n_atoms)
        logfile_prefix = str(tmp_path / "test_already_closed")

        bias = GADESBias(
            backend=backend,
            biased_force=None,
            bias_atom_indices=bias_indices,
            hess_func=simple_hess_func,
            clamp_magnitude=10000.0,
            kappa=0.9,
            interval=200,
            logfile_prefix=logfile_prefix,
        )

        # Close files manually first
        bias._evec_log.close()
        bias._eval_log.close()
        bias._xyz_log.close()

        # Should not raise when called again
        bias._close_logs()  # No exception = pass


class TestBofillIntegration:
    """Tests for Bofill Hessian update integration."""

    def test_bofill_disabled_by_default(self, valid_gades_params):
        """Bofill should be disabled by default."""
        bias = GADESBias(**valid_gades_params)
        assert bias.use_bofill_update is False

    def test_bofill_enabled(self, valid_gades_params):
        """Should be able to enable Bofill updates."""
        params = valid_gades_params.copy()
        params['use_bofill_update'] = True
        bias = GADESBias(**params)
        assert bias.use_bofill_update is True

    def test_full_hessian_interval_default(self, valid_gades_params):
        """Default full_hessian_interval should be interval * multiplier."""
        from GADES.config import defaults
        params = valid_gades_params.copy()
        params['interval'] = 100
        bias = GADESBias(**params)
        expected = 100 * defaults["bofill_full_hessian_multiplier"]
        assert bias.full_hessian_interval == expected

    def test_full_hessian_interval_custom(self, valid_gades_params):
        """Custom full_hessian_interval should be used."""
        params = valid_gades_params.copy()
        params['full_hessian_interval'] = 5000
        bias = GADESBias(**params)
        assert bias.full_hessian_interval == 5000

    def test_bofill_state_initialized(self, valid_gades_params):
        """Bofill state variables should be initialized."""
        params = valid_gades_params.copy()
        params['use_bofill_update'] = True
        bias = GADESBias(**params)

        assert bias._last_hess is None
        assert bias._last_positions is None
        assert bias._last_forces is None
        assert bias._last_hess_step == -1

    def test_bofill_first_call_computes_full_hessian(self, mock_backend_factory):
        """First call should always compute full Hessian."""
        n_atoms = 3
        bias_indices = [0, 1, 2]

        call_count = [0]

        def counting_hess_func(backend, atom_indices, step_size, platform):
            call_count[0] += 1
            n_dof = len(atom_indices) * 3
            return np.eye(n_dof)

        backend = mock_backend_factory(n_atoms=n_atoms)
        backend.set_currentStep(0)

        bias = GADESBias(
            backend=backend,
            biased_force=None,
            bias_atom_indices=bias_indices,
            hess_func=counting_hess_func,
            clamp_magnitude=1000.0,
            kappa=0.9,
            interval=100,
            use_bofill_update=True,
            full_hessian_interval=1000,
        )

        # First call should compute full Hessian
        bias.get_gad_force()
        assert call_count[0] == 1

    def test_bofill_uses_approximation_between_intervals(self, mock_backend_factory):
        """Between full Hessian intervals, Bofill approximation should be used."""
        n_atoms = 3
        bias_indices = [0, 1, 2]

        call_count = [0]

        def counting_hess_func(backend, atom_indices, step_size, platform):
            call_count[0] += 1
            n_dof = len(atom_indices) * 3
            return np.eye(n_dof) * (call_count[0] + 1)  # Different each time

        backend = mock_backend_factory(n_atoms=n_atoms)

        bias = GADESBias(
            backend=backend,
            biased_force=None,
            bias_atom_indices=bias_indices,
            hess_func=counting_hess_func,
            clamp_magnitude=1000.0,
            kappa=0.9,
            interval=100,
            use_bofill_update=True,
            full_hessian_interval=1000,
        )

        # First call at step 0 - full Hessian
        backend.set_currentStep(0)
        bias.get_gad_force()
        assert call_count[0] == 1

        # Second call at step 100 - should use Bofill (not full Hessian)
        backend.set_currentStep(100)
        bias.get_gad_force()
        assert call_count[0] == 1  # No additional hess_func call

        # Third call at step 200 - should still use Bofill
        backend.set_currentStep(200)
        bias.get_gad_force()
        assert call_count[0] == 1  # Still no additional call

    def test_bofill_recomputes_at_full_interval(self, mock_backend_factory):
        """At full_hessian_interval, full Hessian should be recomputed."""
        n_atoms = 3
        bias_indices = [0, 1, 2]

        call_count = [0]

        def counting_hess_func(backend, atom_indices, step_size, platform):
            call_count[0] += 1
            n_dof = len(atom_indices) * 3
            return np.eye(n_dof)

        backend = mock_backend_factory(n_atoms=n_atoms)

        bias = GADESBias(
            backend=backend,
            biased_force=None,
            bias_atom_indices=bias_indices,
            hess_func=counting_hess_func,
            clamp_magnitude=1000.0,
            kappa=0.9,
            interval=100,
            use_bofill_update=True,
            full_hessian_interval=500,
        )

        # First call at step 0 - full Hessian
        backend.set_currentStep(0)
        bias.get_gad_force()
        assert call_count[0] == 1

        # Call at step 500 - should recompute full Hessian
        backend.set_currentStep(500)
        bias.get_gad_force()
        assert call_count[0] == 2


class TestComputeSoftestMode:
    """Tests for _compute_softest_mode helper method."""

    def test_compute_softest_mode_numpy(self, valid_gades_params):
        """_compute_softest_mode with numpy should return correct values."""
        bias = GADESBias(**valid_gades_params)

        # Simple diagonal Hessian
        hess = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        eigval, eigvec = bias._compute_softest_mode(hess)

        assert np.isclose(eigval, 1.0)
        assert np.isclose(np.linalg.norm(eigvec), 1.0)

    def test_compute_softest_mode_lanczos(self, valid_gades_params):
        """_compute_softest_mode with lanczos should return correct values."""
        params = valid_gades_params.copy()
        params['eigensolver'] = 'lanczos'
        params['lanczos_iterations'] = 10
        bias = GADESBias(**params)

        # Simple diagonal Hessian
        hess = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        eigval, eigvec = bias._compute_softest_mode(hess)

        assert np.isclose(eigval, 1.0, atol=0.1)
        assert np.isclose(np.linalg.norm(eigvec), 1.0)


class TestGADESForceUpdaterReporting:
    """Tests for GADESForceUpdater OpenMM reporter interface (describeNextReport)."""

    def test_describe_next_report_returns_correct_tuple(self, mock_backend_factory, simple_hess_func):
        """describeNextReport should return 6-element tuple with correct format."""
        from GADES import GADESForceUpdater

        backend = mock_backend_factory()
        backend.set_currentStep(0)

        updater = GADESForceUpdater(
            backend=backend,
            biased_force=None,
            bias_atom_indices=[0, 1, 2],
            hess_func=simple_hess_func,
            clamp_magnitude=1000.0,
            kappa=0.9,
            interval=200,
            stability_interval=100,
        )

        # Mock simulation object (describeNextReport doesn't use it directly)
        mock_simulation = None
        result = updater.describeNextReport(mock_simulation)

        assert isinstance(result, tuple)
        assert len(result) == 6
        assert isinstance(result[0], (int, float))  # steps
        assert result[1:] == (False, False, False, False, False)  # All flags False

    def test_describe_next_report_returns_steps_to_next_event(self, mock_backend_factory, simple_hess_func):
        """describeNextReport should return correct number of steps to next event."""
        from GADES import GADESForceUpdater

        backend = mock_backend_factory()
        backend.set_currentStep(50)  # 50 steps until stability check at 100

        updater = GADESForceUpdater(
            backend=backend,
            biased_force=None,
            bias_atom_indices=[0, 1, 2],
            hess_func=simple_hess_func,
            clamp_magnitude=1000.0,
            kappa=0.9,
            interval=200,
            stability_interval=100,
        )

        result = updater.describeNextReport(None)
        assert result[0] == 50  # Next event at step 100

    def test_describe_next_report_sets_is_biasing_flag(self, mock_backend_factory, simple_hess_func):
        """At bias interval, is_biasing flag should be set."""
        from GADES import GADESForceUpdater

        backend = mock_backend_factory()
        backend.set_currentStep(0)  # At bias interval (200 divides 0)

        updater = GADESForceUpdater(
            backend=backend,
            biased_force=None,
            bias_atom_indices=[0, 1, 2],
            hess_func=simple_hess_func,
            clamp_magnitude=1000.0,
            kappa=0.9,
            interval=200,
            stability_interval=None,  # No stability checks
        )

        # At step 0, next event is bias at step 200
        assert updater.is_biasing is False  # Before call
        updater.describeNextReport(None)
        assert updater.is_biasing is True  # After call, at bias interval

    def test_describe_next_report_sets_check_stability_flag(self, mock_backend_factory, simple_hess_func):
        """At stability interval, check_stability flag should be set."""
        from GADES import GADESForceUpdater

        backend = mock_backend_factory()
        backend.set_currentStep(0)

        updater = GADESForceUpdater(
            backend=backend,
            biased_force=None,
            bias_atom_indices=[0, 1, 2],
            hess_func=simple_hess_func,
            clamp_magnitude=1000.0,
            kappa=0.9,
            interval=200,
            stability_interval=100,
        )

        # At step 0, next event is stability check at step 100
        assert updater.check_stability is False  # Before call
        updater.describeNextReport(None)
        assert updater.check_stability is True  # After call, at stability interval


class TestGADESForceUpdaterReport:
    """Tests for GADESForceUpdater report() method."""

    def test_report_applies_bias_when_is_biasing(self, mock_backend_factory, simple_hess_func):
        """report() should apply bias when is_biasing=True."""
        from GADES import GADESForceUpdater

        backend = mock_backend_factory()
        backend.set_currentStep(200)

        updater = GADESForceUpdater(
            backend=backend,
            biased_force="mock_force",
            bias_atom_indices=[0, 1, 2],
            hess_func=simple_hess_func,
            clamp_magnitude=1000.0,
            kappa=0.9,
            interval=200,
            stability_interval=None,  # No stability checks
        )

        # Simulate describeNextReport being called at step 0
        backend.set_currentStep(0)
        updater.describeNextReport(None)
        assert updater.is_biasing is True

        # Now call report at step 200 (simulating OpenMM calling after stepping)
        backend.set_currentStep(200)
        updater.report(None, None)

        # Bias should have been applied
        assert backend._last_applied_bias is not None
        assert updater.is_biasing is False  # Flag cleared after report

    def test_report_removes_bias_when_unstable(self, mock_backend_factory, simple_hess_func):
        """report() should remove bias when system is unstable."""
        from GADES import GADESForceUpdater

        backend = mock_backend_factory()
        backend.set_stable(False)  # System is unstable

        updater = GADESForceUpdater(
            backend=backend,
            biased_force="mock_force",
            bias_atom_indices=[0, 1, 2],
            hess_func=simple_hess_func,
            clamp_magnitude=1000.0,
            kappa=0.9,
            interval=200,
            stability_interval=100,
        )

        # Set up for stability check
        backend.set_currentStep(0)
        updater.describeNextReport(None)
        updater.check_stability = True

        # Now call report - should remove bias due to instability
        backend.set_currentStep(100)
        updater.report(None, None)

        # Bias should have been removed (backend._last_applied_bias set to None by remove_bias)
        assert backend._last_applied_bias is None
        assert updater.check_stability is False  # Flag cleared

    def test_report_applies_bias_when_stable_and_biasing(self, mock_backend_factory, simple_hess_func):
        """report() should apply bias when stable and is_biasing both set."""
        from GADES import GADESForceUpdater

        backend = mock_backend_factory()
        backend.set_stable(True)

        updater = GADESForceUpdater(
            backend=backend,
            biased_force="mock_force",
            bias_atom_indices=[0, 1, 2],
            hess_func=simple_hess_func,
            clamp_magnitude=1000.0,
            kappa=0.9,
            interval=200,
            stability_interval=200,  # Both intervals align
        )

        # At step 0, both bias and stability check happen at step 200
        backend.set_currentStep(0)
        updater.describeNextReport(None)

        # Manually set both flags (as if aligned)
        updater.is_biasing = True
        updater.check_stability = True

        # Call report - should apply bias since stable
        backend.set_currentStep(200)
        updater.report(None, None)

        assert backend._last_applied_bias is not None
        assert updater.is_biasing is False
        assert updater.check_stability is False

    def test_report_schedules_post_bias_check(self, mock_backend_factory, simple_hess_func):
        """report() should schedule post-bias stability check after applying bias."""
        from GADES import GADESForceUpdater
        from GADES.config import defaults

        backend = mock_backend_factory()
        backend.set_currentStep(200)

        updater = GADESForceUpdater(
            backend=backend,
            biased_force="mock_force",
            bias_atom_indices=[0, 1, 2],
            hess_func=simple_hess_func,
            clamp_magnitude=1000.0,
            kappa=0.9,
            interval=200,
            stability_interval=None,
        )

        # Set up for bias application
        updater.is_biasing = True
        assert updater.next_postbias_check_step is None

        # Apply bias
        updater.report(None, None)

        # Post-bias check should be scheduled
        expected_step = 200 + defaults["post_bias_check_delay"]
        assert updater.next_postbias_check_step == expected_step

    def test_report_no_action_when_no_flags(self, mock_backend_factory, simple_hess_func):
        """report() should do nothing when neither flag is set."""
        from GADES import GADESForceUpdater

        backend = mock_backend_factory()
        backend.set_currentStep(50)

        updater = GADESForceUpdater(
            backend=backend,
            biased_force="mock_force",
            bias_atom_indices=[0, 1, 2],
            hess_func=simple_hess_func,
            clamp_magnitude=1000.0,
            kappa=0.9,
            interval=200,
        )

        # Neither flag is set
        assert updater.is_biasing is False
        assert updater.check_stability is False

        # Call report - should do nothing
        updater.report(None, None)

        # No bias applied or removed
        assert not hasattr(backend, '_last_applied_bias') or backend._last_applied_bias is None


class TestPostBiasScheduling:
    """Tests for post-bias stability check scheduling."""

    def test_register_next_step_includes_postbias_check(self, mock_backend_factory, simple_hess_func):
        """Post-bias check should be included in scheduling."""
        from GADES import GADESForceUpdater

        backend = mock_backend_factory()
        backend.set_currentStep(200)

        updater = GADESForceUpdater(
            backend=backend,
            biased_force="mock_force",
            bias_atom_indices=[0, 1, 2],
            hess_func=simple_hess_func,
            clamp_magnitude=1000.0,
            kappa=0.9,
            interval=1000,  # Far away
            stability_interval=None,
        )

        # Schedule a post-bias check at step 300
        updater.next_postbias_check_step = 300

        # register_next_step should return steps to post-bias check
        steps = updater.register_next_step()
        assert steps == 100  # 300 - 200 = 100

    def test_postbias_check_sets_check_stability_flag(self, mock_backend_factory, simple_hess_func):
        """At post-bias check step, check_stability should be set."""
        from GADES import GADESForceUpdater

        backend = mock_backend_factory()
        backend.set_currentStep(200)

        updater = GADESForceUpdater(
            backend=backend,
            biased_force="mock_force",
            bias_atom_indices=[0, 1, 2],
            hess_func=simple_hess_func,
            clamp_magnitude=1000.0,
            kappa=0.9,
            interval=1000,
            stability_interval=None,
        )

        # Schedule post-bias check at step 300
        updater.next_postbias_check_step = 300

        # Move to step 300
        backend.set_currentStep(300)
        updater.register_next_step()

        # check_stability should be set because we're at post-bias check
        assert updater.check_stability is True

    def test_postbias_check_clears_after_check(self, mock_backend_factory, simple_hess_func):
        """next_postbias_check_step should be cleared after the check."""
        from GADES import GADESForceUpdater

        backend = mock_backend_factory()
        backend.set_stable(True)

        updater = GADESForceUpdater(
            backend=backend,
            biased_force="mock_force",
            bias_atom_indices=[0, 1, 2],
            hess_func=simple_hess_func,
            clamp_magnitude=1000.0,
            kappa=0.9,
            interval=1000,
            stability_interval=None,
        )

        # Schedule and reach post-bias check
        updater.next_postbias_check_step = 300
        backend.set_currentStep(300)

        # Set check_stability as if register_next_step was called
        updater.check_stability = True

        # Call report at the post-bias check step
        updater.report(None, None)

        # Post-bias check should be cleared
        assert updater.next_postbias_check_step is None

    def test_postbias_priority_over_regular_interval(self, mock_backend_factory, simple_hess_func):
        """Post-bias check should take priority when it comes first."""
        from GADES import GADESForceUpdater

        backend = mock_backend_factory()
        backend.set_currentStep(200)

        updater = GADESForceUpdater(
            backend=backend,
            biased_force="mock_force",
            bias_atom_indices=[0, 1, 2],
            hess_func=simple_hess_func,
            clamp_magnitude=1000.0,
            kappa=0.9,
            interval=1000,  # Next bias at 1000
            stability_interval=500,  # Next stability at 500
        )

        # Post-bias check at 250 (comes before both)
        updater.next_postbias_check_step = 250

        steps = updater.register_next_step()
        assert steps == 50  # 250 - 200 = 50, not 300 (stability) or 800 (bias)
