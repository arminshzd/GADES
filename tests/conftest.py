"""
Pytest fixtures for GADES test suite.
"""
import pytest
import numpy as np


class MockBackend:
    """
    A mock backend for testing GADESBias without requiring OpenMM or ASE.
    """
    def __init__(self, n_atoms=10, positions=None, forces=None):
        self.n_atoms = n_atoms
        self.name = "mock"
        self._current_step = 0
        self._positions = positions if positions is not None else np.random.randn(n_atoms, 3)
        self._forces = forces if forces is not None else np.random.randn(n_atoms, 3)
        self._stable = True

    def is_stable(self):
        return self._stable

    def set_stable(self, stable):
        self._stable = stable

    def get_currentStep(self):
        return self._current_step

    def set_currentStep(self, step):
        self._current_step = step

    def get_atom_symbols(self, bias_atom_indices):
        return ["C"] * len(bias_atom_indices)

    def get_current_state(self):
        return self._positions.copy(), self._forces.copy()

    def get_forces(self, positions):
        # Return negative flattened forces (matching real backend behavior)
        return -self._forces.flatten()

    def apply_bias(self, bias_force_object, biased_force_values, bias_atom_indices):
        # Store for verification in tests
        self._last_applied_bias = biased_force_values
        self._last_bias_indices = bias_atom_indices

    def remove_bias(self, bias_force_object, bias_atom_indices):
        self._last_applied_bias = None
        self._last_bias_indices = bias_atom_indices


@pytest.fixture
def mock_backend():
    """Create a mock backend with default settings."""
    return MockBackend(n_atoms=10)


@pytest.fixture
def mock_backend_factory():
    """Factory fixture to create mock backends with custom settings."""
    def _create(n_atoms=10, positions=None, forces=None):
        return MockBackend(n_atoms=n_atoms, positions=positions, forces=forces)
    return _create


def simple_hessian_func(backend, atom_indices, step_size, platform):
    """
    A simple mock Hessian function that returns a known Hessian matrix.
    Returns identity matrix scaled, which has eigenvalue 1 with known eigenvectors.
    """
    n_dof = len(atom_indices) * 3
    # Return a diagonal matrix with distinct eigenvalues for predictable testing
    # Smallest eigenvalue will be at index 0
    eigenvalues = np.arange(1, n_dof + 1, dtype=float)
    return np.diag(eigenvalues)


def negative_eigenvalue_hessian_func(backend, atom_indices, step_size, platform):
    """
    A Hessian function that returns a matrix with a negative eigenvalue.
    This simulates being at a saddle point or transition state.
    """
    n_dof = len(atom_indices) * 3
    # First eigenvalue is negative (softest mode)
    eigenvalues = np.concatenate([[-1.0], np.arange(1, n_dof, dtype=float)])
    return np.diag(eigenvalues)


@pytest.fixture
def simple_hess_func():
    """Return a simple Hessian function for testing."""
    return simple_hessian_func


@pytest.fixture
def negative_eigenvalue_hess_func():
    """Return a Hessian function with negative eigenvalue."""
    return negative_eigenvalue_hessian_func


@pytest.fixture
def sample_bias_atom_indices():
    """Return sample atom indices for biasing."""
    return [0, 1, 2]


@pytest.fixture
def valid_gades_params(simple_hess_func, sample_bias_atom_indices):
    """Return valid parameters for GADESBias initialization."""
    return {
        'backend': None,  # Can be set to mock_backend in tests
        'biased_force': None,
        'bias_atom_indices': sample_bias_atom_indices,
        'hess_func': simple_hess_func,
        'clamp_magnitude': 1000.0,
        'kappa': 0.9,
        'interval': 200,
        'stability_interval': 100,
        'logfile_prefix': None,
    }
