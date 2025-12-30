"""
Tests for Backend implementations.
"""
import pytest
import warnings
import numpy as np

# Try to import backend module - may fail if ASE not installed
try:
    from GADES.backend import Backend, ASEBackend, GADESCalculator
    HAS_BACKEND = True
except ImportError:
    HAS_BACKEND = False
    Backend = None
    ASEBackend = None
    GADESCalculator = None

pytestmark = pytest.mark.skipif(not HAS_BACKEND, reason="ASE not installed")


class TestBackendInterface:
    """Tests for the Backend base class interface."""

    def test_backend_default_is_stable(self):
        """Base Backend.is_stable should return True."""
        backend = Backend()
        assert backend.is_stable() is True

    def test_backend_default_get_current_step(self):
        """Base Backend.get_currentStep should return 0."""
        backend = Backend()
        assert backend.get_currentStep() == 0

    def test_backend_has_required_methods(self):
        """Backend should have all required interface methods."""
        backend = Backend()
        assert hasattr(backend, 'is_stable')
        assert hasattr(backend, 'get_currentStep')
        assert hasattr(backend, 'get_atom_symbols')
        assert hasattr(backend, 'get_current_state')
        assert hasattr(backend, 'apply_bias')
        assert hasattr(backend, 'remove_bias')


class MockAtoms:
    """Mock ASE Atoms object for testing."""

    def __init__(self, n_atoms=10, temperature=300.0):
        self._n_atoms = n_atoms
        self._temperature = temperature
        self._positions = np.random.randn(n_atoms, 3)
        self._kinetic_energy = 0.5 * n_atoms * temperature * 8.617e-5  # Approximate
        self.calc = None

    def __iter__(self):
        """Allow iteration over atoms."""
        class MockAtom:
            def __init__(self, symbol, index):
                self.symbol = symbol
                self.index = index
        return iter([MockAtom("Ar", i) for i in range(self._n_atoms)])

    def __len__(self):
        return self._n_atoms

    def get_positions(self):
        return self._positions.copy()

    def set_positions(self, positions):
        self._positions = positions.copy()

    def get_temperature(self):
        return self._temperature

    def set_temperature(self, temp):
        self._temperature = temp

    def get_kinetic_energy(self):
        return self._kinetic_energy


class MockBaseCalculator:
    """Mock ASE Calculator for testing."""

    def __init__(self, n_atoms=10):
        self.atoms = None
        self.results = {
            'forces': np.random.randn(n_atoms, 3),
            'energy': -100.0,
        }

    def calculate(self, atoms=None, properties=None, system_changes=None):
        pass


class MockIntegrator:
    """Mock integrator for testing."""

    def __init__(self, nsteps=0, temp=None, temperature=None):
        self.nsteps = nsteps
        # Langevin style
        if temp is not None:
            self.temp = temp
        # Berendsen style
        if temperature is not None:
            self.temperature = temperature


class MockGADESCalculator:
    """Mock GADESCalculator for testing ASEBackend."""

    def __init__(self, base_calc):
        self.base_calc = base_calc


class TestASEBackendInitialization:
    """Tests for ASEBackend initialization."""

    def test_basic_initialization(self):
        """Test basic ASEBackend initialization."""
        atoms = MockAtoms()
        base_calc = MockBaseCalculator()
        calc = MockGADESCalculator(base_calc)

        backend = ASEBackend(calc, atoms)

        assert backend.name == "ase"
        assert backend.atoms is atoms
        assert backend.base_calc is base_calc
        assert backend.integrator is None
        assert backend.current_step == -1

    def test_initialization_with_target_temperature(self):
        """Test initialization with explicit target_temperature."""
        atoms = MockAtoms()
        base_calc = MockBaseCalculator()
        calc = MockGADESCalculator(base_calc)

        backend = ASEBackend(calc, atoms, target_temperature=350.0)

        assert backend.target_temperature == 350.0

    def test_initialization_sets_atoms_calc(self):
        """Initialization should set atoms.calc to calculator."""
        atoms = MockAtoms()
        base_calc = MockBaseCalculator()
        calc = MockGADESCalculator(base_calc)

        backend = ASEBackend(calc, atoms)

        assert atoms.calc is calc


class TestASEBackendGetTargetTemperature:
    """Tests for ASEBackend._get_target_temperature method."""

    def test_explicit_target_temperature(self):
        """Explicit target_temperature should be returned."""
        atoms = MockAtoms()
        calc = MockGADESCalculator(MockBaseCalculator())
        backend = ASEBackend(calc, atoms, target_temperature=350.0)

        assert backend._get_target_temperature() == 350.0

    def test_langevin_integrator_temp(self):
        """Should read temp from Langevin-style integrator."""
        atoms = MockAtoms()
        calc = MockGADESCalculator(MockBaseCalculator())
        backend = ASEBackend(calc, atoms)

        # Set integrator with 'temp' attribute (Langevin style)
        backend.integrator = MockIntegrator(temp=400.0)

        assert backend._get_target_temperature() == 400.0

    def test_berendsen_integrator_temperature(self):
        """Should read temperature from Berendsen-style integrator."""
        atoms = MockAtoms()
        calc = MockGADESCalculator(MockBaseCalculator())
        backend = ASEBackend(calc, atoms)

        # Set integrator with 'temperature' attribute (Berendsen style)
        backend.integrator = MockIntegrator(temperature=450.0)

        assert backend._get_target_temperature() == 450.0

    def test_explicit_overrides_integrator(self):
        """Explicit target_temperature should override integrator."""
        atoms = MockAtoms()
        calc = MockGADESCalculator(MockBaseCalculator())
        backend = ASEBackend(calc, atoms, target_temperature=350.0)
        backend.integrator = MockIntegrator(temp=400.0)

        # Explicit should take precedence
        assert backend._get_target_temperature() == 350.0

    def test_no_target_available(self):
        """Should return None when no target temperature available."""
        atoms = MockAtoms()
        calc = MockGADESCalculator(MockBaseCalculator())
        backend = ASEBackend(calc, atoms)

        # No integrator set
        assert backend._get_target_temperature() is None

    def test_nve_integrator_no_temp(self):
        """NVE integrator without temp should return None."""
        atoms = MockAtoms()
        calc = MockGADESCalculator(MockBaseCalculator())
        backend = ASEBackend(calc, atoms)

        # NVE integrator has no temp attribute
        backend.integrator = MockIntegrator()  # No temp or temperature

        assert backend._get_target_temperature() is None


class TestASEBackendIsStable:
    """Tests for ASEBackend.is_stable method."""

    def test_stable_when_temp_close(self):
        """Should return True when temperature is close to target."""
        atoms = MockAtoms(temperature=300.0)
        calc = MockGADESCalculator(MockBaseCalculator())
        backend = ASEBackend(calc, atoms, target_temperature=300.0)

        assert backend.is_stable() is True

    def test_stable_within_threshold(self):
        """Should return True when within 50K threshold."""
        atoms = MockAtoms(temperature=340.0)
        calc = MockGADESCalculator(MockBaseCalculator())
        backend = ASEBackend(calc, atoms, target_temperature=300.0)

        # 340 - 300 = 40K < 50K threshold
        assert backend.is_stable() is True

    def test_unstable_above_threshold(self):
        """Should return False when temperature exceeds threshold."""
        atoms = MockAtoms(temperature=360.0)
        calc = MockGADESCalculator(MockBaseCalculator())
        backend = ASEBackend(calc, atoms, target_temperature=300.0)

        # 360 - 300 = 60K > 50K threshold
        assert backend.is_stable() is False

    def test_unstable_below_threshold(self):
        """Should return False when temperature is too low."""
        atoms = MockAtoms(temperature=240.0)
        calc = MockGADESCalculator(MockBaseCalculator())
        backend = ASEBackend(calc, atoms, target_temperature=300.0)

        # 300 - 240 = 60K > 50K threshold
        assert backend.is_stable() is False

    def test_exactly_at_threshold(self):
        """At exactly 50K difference, should be stable (<=)."""
        atoms = MockAtoms(temperature=350.0)
        calc = MockGADESCalculator(MockBaseCalculator())
        backend = ASEBackend(calc, atoms, target_temperature=300.0)

        # 350 - 300 = 50K, exactly at threshold
        assert backend.is_stable() is True

    def test_no_target_returns_true_with_warning(self):
        """Without target temperature, should warn and return True."""
        atoms = MockAtoms(temperature=500.0)  # Very hot
        calc = MockGADESCalculator(MockBaseCalculator())
        backend = ASEBackend(calc, atoms)  # No target_temperature

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = backend.is_stable()

            assert result is True
            assert len(w) == 1
            assert "target temperature" in str(w[0].message).lower()

    def test_warning_only_issued_once(self):
        """Warning should only be issued once."""
        atoms = MockAtoms()
        calc = MockGADESCalculator(MockBaseCalculator())
        backend = ASEBackend(calc, atoms)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Call multiple times
            backend.is_stable()
            backend.is_stable()
            backend.is_stable()

            # Only one warning
            stability_warnings = [x for x in w if "target temperature" in str(x.message).lower()]
            assert len(stability_warnings) == 1


class TestASEBackendGetCurrentStep:
    """Tests for ASEBackend.get_currentStep method."""

    def test_no_integrator_returns_minus_one(self):
        """Without integrator, should return -1."""
        atoms = MockAtoms()
        calc = MockGADESCalculator(MockBaseCalculator())
        backend = ASEBackend(calc, atoms)

        assert backend.get_currentStep() == -1

    def test_with_integrator_returns_nsteps(self):
        """With integrator, should return integrator.nsteps."""
        atoms = MockAtoms()
        calc = MockGADESCalculator(MockBaseCalculator())
        backend = ASEBackend(calc, atoms)
        backend.integrator = MockIntegrator(nsteps=500)

        assert backend.get_currentStep() == 500


class TestASEBackendGetAtomSymbols:
    """Tests for ASEBackend.get_atom_symbols method."""

    def test_get_atom_symbols(self):
        """Should return correct symbols for given indices."""
        atoms = MockAtoms(n_atoms=5)
        calc = MockGADESCalculator(MockBaseCalculator(n_atoms=5))
        backend = ASEBackend(calc, atoms)

        symbols = backend.get_atom_symbols([0, 2, 4])
        assert symbols == ["Ar", "Ar", "Ar"]
        assert len(symbols) == 3
