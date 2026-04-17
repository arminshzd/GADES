"""
Tests for Backend implementations.
"""
import pytest
import warnings
import numpy as np

# Try to import backend module - may fail if ASE not installed
try:
    from GADES.backend import Backend, ASEBackend, GADESCalculator
    from ase import units as ase_units
    HAS_BACKEND = True
except ImportError:
    HAS_BACKEND = False
    Backend = None
    ASEBackend = None
    GADESCalculator = None
    ase_units = None

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
        """Should read temp from Langevin-style integrator and convert from eV to Kelvin."""
        atoms = MockAtoms()
        calc = MockGADESCalculator(MockBaseCalculator())
        backend = ASEBackend(calc, atoms)

        # Real ASE Langevin stores self.temp = units.kB * temperature_K (in eV)
        backend.integrator = MockIntegrator(temp=400.0 * ase_units.kB)

        assert backend._get_target_temperature() == pytest.approx(400.0)

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
        backend.integrator = MockIntegrator(temp=400.0 * ase_units.kB)

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


class TestASEBackendWithGades:
    """Tests for ASEBackend.with_gades factory method."""

    def test_with_gades_basic(self):
        """Factory method should create fully wired ASEBackend."""
        atoms = MockAtoms(n_atoms=5)
        base_calc = MockBaseCalculator(n_atoms=5)

        def mock_hess_func(backend, pos, indices, step_size, platform):
            n = len(indices) * 3
            return np.eye(n)

        backend = ASEBackend.with_gades(
            atoms=atoms,
            base_calc=base_calc,
            bias_atom_indices=[0, 1, 2],
            hess_func=mock_hess_func,
            clamp_magnitude=1000,
            kappa=0.9,
            interval=100,
        )

        assert backend.name == "ase"
        assert backend.atoms is atoms
        assert backend.base_calc is base_calc
        assert backend.gades_bias is not None

    def test_with_gades_wires_backend_reference(self):
        """Factory method should wire backend reference in GADESBias."""
        atoms = MockAtoms(n_atoms=5)
        base_calc = MockBaseCalculator(n_atoms=5)

        def mock_hess_func(backend, pos, indices, step_size, platform):
            n = len(indices) * 3
            return np.eye(n)

        backend = ASEBackend.with_gades(
            atoms=atoms,
            base_calc=base_calc,
            bias_atom_indices=[0, 1, 2],
            hess_func=mock_hess_func,
            clamp_magnitude=1000,
            kappa=0.9,
            interval=100,
        )

        # GADESBias should have backend reference
        assert backend.gades_bias.backend is backend

    def test_with_gades_sets_atoms_calc(self):
        """Factory method should set atoms.calc to GADESCalculator."""
        atoms = MockAtoms(n_atoms=5)
        base_calc = MockBaseCalculator(n_atoms=5)

        def mock_hess_func(backend, pos, indices, step_size, platform):
            n = len(indices) * 3
            return np.eye(n)

        backend = ASEBackend.with_gades(
            atoms=atoms,
            base_calc=base_calc,
            bias_atom_indices=[0, 1, 2],
            hess_func=mock_hess_func,
            clamp_magnitude=1000,
            kappa=0.9,
            interval=100,
        )

        # atoms.calc should be set to a GADESCalculator
        assert atoms.calc is not None
        assert hasattr(atoms.calc, 'force_updater')
        assert atoms.calc.force_updater is backend.gades_bias

    def test_with_gades_optional_parameters(self):
        """Factory method should pass optional parameters to GADESBias."""
        atoms = MockAtoms(n_atoms=5)
        base_calc = MockBaseCalculator(n_atoms=5)

        def mock_hess_func(backend, pos, indices, step_size, platform):
            n = len(indices) * 3
            return np.eye(n)

        backend = ASEBackend.with_gades(
            atoms=atoms,
            base_calc=base_calc,
            bias_atom_indices=[0, 1, 2],
            hess_func=mock_hess_func,
            clamp_magnitude=1000,
            kappa=0.9,
            interval=100,
            stability_interval=500,
            eigensolver="lanczos",
            lanczos_iterations=20,
            use_bofill_update=True,
            full_hessian_interval=50,
            target_temperature=350.0,
        )

        assert backend.gades_bias.eigensolver == "lanczos"
        assert backend.gades_bias.lanczos_iterations == 20
        assert backend.gades_bias.use_bofill_update is True
        assert backend.gades_bias.full_hessian_interval == 50
        assert backend.target_temperature == 350.0

    def test_with_gades_gades_bias_none_for_regular_init(self):
        """Regular __init__ should leave gades_bias as None."""
        atoms = MockAtoms(n_atoms=5)
        base_calc = MockBaseCalculator(n_atoms=5)
        calc = MockGADESCalculator(base_calc)

        backend = ASEBackend(calc, atoms)

        assert backend.gades_bias is None

    def test_with_gades_invalid_index_raises(self):
        """with_gades should raise ValueError for out-of-bounds bias_atom_indices."""
        atoms = MockAtoms(n_atoms=10)
        base_calc = MockBaseCalculator(n_atoms=10)

        def mock_hess_func(backend, atom_indices, step_size, platform):
            n = len(atom_indices) * 3
            return np.eye(n)

        with pytest.raises(ValueError) as exc_info:
            ASEBackend.with_gades(
                atoms=atoms,
                base_calc=base_calc,
                bias_atom_indices=[0, 1, 100],  # Index 100 is out of bounds
                hess_func=mock_hess_func,
                clamp_magnitude=1000,
                kappa=0.9,
                interval=100,
            )

        assert "100" in str(exc_info.value)
        assert "10 atoms" in str(exc_info.value)

    def test_with_gades_valid_max_index_succeeds(self):
        """with_gades should accept index == n_atoms - 1 (last valid index)."""
        atoms = MockAtoms(n_atoms=10)
        base_calc = MockBaseCalculator(n_atoms=10)

        def mock_hess_func(backend, atom_indices, step_size, platform):
            n = len(atom_indices) * 3
            return np.eye(n)

        # Index 9 is valid for 10 atoms (0-9)
        backend = ASEBackend.with_gades(
            atoms=atoms,
            base_calc=base_calc,
            bias_atom_indices=[0, 5, 9],
            hess_func=mock_hess_func,
            clamp_magnitude=1000,
            kappa=0.9,
            interval=100,
        )

        assert backend.gades_bias is not None
        assert list(backend.gades_bias.bias_atom_indices) == [0, 5, 9]


class TestGADESCalculatorPartialBiasing:
    """Tests for GADESCalculator with partial atom biasing (A10 regression tests)."""

    def test_partial_biasing_force_shape(self):
        """
        GADESCalculator.calculate() should handle partial atom biasing correctly.

        This is a regression test for A10: when N_bias < N_atoms, the bias
        should be applied only to biased atoms without shape mismatch errors.
        """
        n_atoms = 10
        bias_indices = [0, 1, 2]  # Only 3 atoms biased out of 10

        atoms = MockAtoms(n_atoms=n_atoms)
        base_calc = MockBaseCalculator(n_atoms=n_atoms)

        # Use a simple Hessian function that returns identity
        def mock_hess_func(backend, atom_indices, step_size, platform):
            n = len(atom_indices) * 3
            return np.eye(n)

        # Create real GADESCalculator via ASEBackend.with_gades
        backend = ASEBackend.with_gades(
            atoms=atoms,
            base_calc=base_calc,
            bias_atom_indices=bias_indices,
            hess_func=mock_hess_func,
            clamp_magnitude=1000,
            kappa=0.9,
            interval=100,
        )

        # Attach integrator and set step to trigger bias application
        integrator = MockIntegrator(nsteps=100)  # Step 100 = multiple of interval
        backend.integrator = integrator

        # Verify applying_bias returns True
        assert backend.gades_bias.applying_bias() is True

        # Call calculate - this would raise ValueError before A10 fix
        gades_calc = atoms.calc
        gades_calc.calculate(atoms=atoms, properties=('forces',))

        # Verify output forces have correct shape (N_atoms, 3)
        result_forces = gades_calc.results['forces']
        assert result_forces.shape == (n_atoms, 3), \
            f"Expected shape ({n_atoms}, 3), got {result_forces.shape}"

    def test_partial_biasing_only_affects_biased_atoms(self):
        """
        Bias should only be applied to atoms in bias_atom_indices.

        Non-biased atoms should retain their original forces from base calculator.
        """
        n_atoms = 10
        bias_indices = [0, 1, 2]  # Only first 3 atoms biased
        non_bias_indices = [3, 4, 5, 6, 7, 8, 9]

        atoms = MockAtoms(n_atoms=n_atoms)

        # Create base calculator with known forces
        base_calc = MockBaseCalculator(n_atoms=n_atoms)
        original_forces = np.ones((n_atoms, 3)) * 5.0  # All forces = 5.0
        base_calc.results['forces'] = original_forces.copy()

        def mock_hess_func(backend, atom_indices, step_size, platform):
            n = len(atom_indices) * 3
            return np.eye(n)

        backend = ASEBackend.with_gades(
            atoms=atoms,
            base_calc=base_calc,
            bias_atom_indices=bias_indices,
            hess_func=mock_hess_func,
            clamp_magnitude=1000,
            kappa=0.9,
            interval=100,
        )

        # Trigger bias application
        integrator = MockIntegrator(nsteps=100)
        backend.integrator = integrator

        gades_calc = atoms.calc
        gades_calc.calculate(atoms=atoms, properties=('forces',))

        result_forces = gades_calc.results['forces']

        # Non-biased atoms should have unchanged forces
        np.testing.assert_array_equal(
            result_forces[non_bias_indices, :],
            original_forces[non_bias_indices, :],
            err_msg="Non-biased atoms should have unchanged forces"
        )

    def test_no_bias_when_not_applying(self):
        """
        When applying_bias() returns False, forces should be unchanged.
        """
        n_atoms = 10
        bias_indices = [0, 1, 2]

        atoms = MockAtoms(n_atoms=n_atoms)
        base_calc = MockBaseCalculator(n_atoms=n_atoms)
        original_forces = np.ones((n_atoms, 3)) * 7.0
        base_calc.results['forces'] = original_forces.copy()

        def mock_hess_func(backend, atom_indices, step_size, platform):
            n = len(atom_indices) * 3
            return np.eye(n)

        backend = ASEBackend.with_gades(
            atoms=atoms,
            base_calc=base_calc,
            bias_atom_indices=bias_indices,
            hess_func=mock_hess_func,
            clamp_magnitude=1000,
            kappa=0.9,
            interval=100,
        )

        # Step 50 is NOT a multiple of interval=100, so no bias
        integrator = MockIntegrator(nsteps=50)
        backend.integrator = integrator

        assert backend.gades_bias.applying_bias() is False

        gades_calc = atoms.calc
        gades_calc.calculate(atoms=atoms, properties=('forces',))

        result_forces = gades_calc.results['forces']

        # All forces should be unchanged
        np.testing.assert_array_equal(
            result_forces,
            original_forces,
            err_msg="Forces should be unchanged when not applying bias"
        )


class TestGADESCalculatorPersistentBias:
    """Tests for GADESCalculator persistent bias behavior (F1)."""

    def test_bias_stored_after_applying(self):
        """Bias should be stored after applying_bias() returns True."""
        n_atoms = 10
        bias_indices = [0, 1, 2]

        atoms = MockAtoms(n_atoms=n_atoms)
        base_calc = MockBaseCalculator(n_atoms=n_atoms)
        base_calc.results['forces'] = np.ones((n_atoms, 3))

        def mock_hess_func(backend, atom_indices, step_size, platform):
            n = len(atom_indices) * 3
            return np.eye(n)

        backend = ASEBackend.with_gades(
            atoms=atoms,
            base_calc=base_calc,
            bias_atom_indices=bias_indices,
            hess_func=mock_hess_func,
            clamp_magnitude=1000,
            kappa=0.9,
            interval=100,
        )

        gades_calc = atoms.calc

        # Initially no stored bias
        assert gades_calc._stored_bias is None
        assert gades_calc._bias_active is False

        # Step 100 triggers bias computation
        backend.integrator = MockIntegrator(nsteps=100)
        gades_calc.calculate(atoms=atoms, properties=('forces',))

        # Bias should now be stored
        assert gades_calc._stored_bias is not None
        assert gades_calc._bias_active is True
        assert gades_calc._stored_bias.shape == (n_atoms, 3)

    def test_stored_bias_applied_at_non_update_steps(self):
        """Stored bias should be applied at steps between updates."""
        n_atoms = 10
        bias_indices = [0, 1, 2]

        atoms = MockAtoms(n_atoms=n_atoms)
        base_calc = MockBaseCalculator(n_atoms=n_atoms)
        original_forces = np.ones((n_atoms, 3))
        base_calc.results['forces'] = original_forces.copy()

        def mock_hess_func(backend, atom_indices, step_size, platform):
            n = len(atom_indices) * 3
            return np.eye(n)

        backend = ASEBackend.with_gades(
            atoms=atoms,
            base_calc=base_calc,
            bias_atom_indices=bias_indices,
            hess_func=mock_hess_func,
            clamp_magnitude=1000,
            kappa=0.9,
            interval=100,
        )

        gades_calc = atoms.calc

        # Step 100: compute and store bias
        backend.integrator = MockIntegrator(nsteps=100)
        gades_calc.calculate(atoms=atoms, properties=('forces',))
        stored_bias = gades_calc._stored_bias.copy()

        # Step 101: NOT an update step, but stored bias should still be applied
        backend.integrator = MockIntegrator(nsteps=101)
        base_calc.results['forces'] = original_forces.copy()
        gades_calc.calculate(atoms=atoms, properties=('forces',))

        # Forces should include the stored bias
        expected = original_forces + stored_bias
        np.testing.assert_array_almost_equal(
            gades_calc.results['forces'],
            expected,
            err_msg="Stored bias should be applied at non-update steps"
        )

    def test_bias_persists_across_multiple_steps(self):
        """Same bias should be applied across multiple steps until next update."""
        n_atoms = 10
        bias_indices = [0, 1, 2]

        atoms = MockAtoms(n_atoms=n_atoms)
        base_calc = MockBaseCalculator(n_atoms=n_atoms)
        original_forces = np.ones((n_atoms, 3))

        def mock_hess_func(backend, atom_indices, step_size, platform):
            n = len(atom_indices) * 3
            return np.eye(n)

        backend = ASEBackend.with_gades(
            atoms=atoms,
            base_calc=base_calc,
            bias_atom_indices=bias_indices,
            hess_func=mock_hess_func,
            clamp_magnitude=1000,
            kappa=0.9,
            interval=100,
        )

        gades_calc = atoms.calc

        # Step 100: compute bias
        backend.integrator = MockIntegrator(nsteps=100)
        base_calc.results['forces'] = original_forces.copy()
        gades_calc.calculate(atoms=atoms, properties=('forces',))
        stored_bias = gades_calc._stored_bias.copy()

        # Steps 101-105: same bias should be applied
        for step in [101, 102, 103, 104, 105]:
            backend.integrator = MockIntegrator(nsteps=step)
            base_calc.results['forces'] = original_forces.copy()
            gades_calc.calculate(atoms=atoms, properties=('forces',))

            expected = original_forces + stored_bias
            np.testing.assert_array_almost_equal(
                gades_calc.results['forces'],
                expected,
                err_msg=f"Stored bias should persist at step {step}"
            )

    def test_stability_failure_clears_bias(self):
        """Stability failure should clear stored bias."""
        n_atoms = 10
        bias_indices = [0, 1, 2]

        atoms = MockAtoms(n_atoms=n_atoms, temperature=300.0)
        base_calc = MockBaseCalculator(n_atoms=n_atoms)
        base_calc.results['forces'] = np.ones((n_atoms, 3))

        def mock_hess_func(backend, atom_indices, step_size, platform):
            n = len(atom_indices) * 3
            return np.eye(n)

        backend = ASEBackend.with_gades(
            atoms=atoms,
            base_calc=base_calc,
            bias_atom_indices=bias_indices,
            hess_func=mock_hess_func,
            clamp_magnitude=1000,
            kappa=0.9,
            interval=100,
            stability_interval=500,
            target_temperature=300.0,
        )

        gades_calc = atoms.calc

        # Step 100: compute bias
        backend.integrator = MockIntegrator(nsteps=100)
        gades_calc.calculate(atoms=atoms, properties=('forces',))
        assert gades_calc._bias_active is True

        # Step 500: stability check - simulate unstable by setting high temperature
        backend.integrator = MockIntegrator(nsteps=500)
        # Make is_stable return False by setting temperature far from target (300K)
        atoms._temperature = 400.0  # 100K deviation > 50K threshold

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gades_calc.calculate(atoms=atoms, properties=('forces',))

        # Bias should be cleared
        assert gades_calc._stored_bias is None
        assert gades_calc._bias_active is False

    def test_bias_restored_after_instability(self):
        """Bias should be restored at next update interval after instability."""
        n_atoms = 10
        bias_indices = [0, 1, 2]

        atoms = MockAtoms(n_atoms=n_atoms, temperature=300.0)
        base_calc = MockBaseCalculator(n_atoms=n_atoms)
        original_forces = np.ones((n_atoms, 3))
        base_calc.results['forces'] = original_forces.copy()

        def mock_hess_func(backend, atom_indices, step_size, platform):
            n = len(atom_indices) * 3
            return np.eye(n)

        backend = ASEBackend.with_gades(
            atoms=atoms,
            base_calc=base_calc,
            bias_atom_indices=bias_indices,
            hess_func=mock_hess_func,
            clamp_magnitude=1000,
            kappa=0.9,
            interval=100,
            stability_interval=500,
            target_temperature=300.0,
        )

        gades_calc = atoms.calc

        # Step 100: compute bias
        backend.integrator = MockIntegrator(nsteps=100)
        gades_calc.calculate(atoms=atoms, properties=('forces',))
        assert gades_calc._bias_active is True

        # Step 500: trigger instability
        backend.integrator = MockIntegrator(nsteps=500)
        atoms._temperature = 400.0  # 100K deviation > 50K threshold

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gades_calc.calculate(atoms=atoms, properties=('forces',))

        assert gades_calc._bias_active is False

        # Step 600: next update interval - restore temperature and compute bias
        backend.integrator = MockIntegrator(nsteps=600)
        atoms._temperature = 300.0  # Back to normal
        base_calc.results['forces'] = original_forces.copy()
        gades_calc.calculate(atoms=atoms, properties=('forces',))

        # Bias should be restored
        assert gades_calc._bias_active is True
        assert gades_calc._stored_bias is not None


class TestGADESCalculatorPostBiasChecks:
    """Tests for ASE post-bias stability checks (G2 fix)."""

    def test_post_bias_check_scheduled_after_applying_bias(self):
        """Post-bias check should be scheduled after applying bias."""
        from GADES.config import defaults

        n_atoms = 10
        bias_indices = [0, 1, 2]

        atoms = MockAtoms(n_atoms=n_atoms)
        base_calc = MockBaseCalculator(n_atoms=n_atoms)
        base_calc.results['forces'] = np.ones((n_atoms, 3))

        def mock_hess_func(backend, atom_indices, step_size, platform):
            n = len(atom_indices) * 3
            return np.eye(n)

        backend = ASEBackend.with_gades(
            atoms=atoms,
            base_calc=base_calc,
            bias_atom_indices=bias_indices,
            hess_func=mock_hess_func,
            clamp_magnitude=1000,
            kappa=0.9,
            interval=100,
            stability_interval=None,  # No regular checks - only post-bias
        )

        gades_calc = atoms.calc
        gades_bias = backend.gades_bias

        # Initially no post-bias check scheduled
        assert gades_bias.next_postbias_check_step is None

        # Step 100: apply bias
        backend.integrator = MockIntegrator(nsteps=100)
        gades_calc.calculate(atoms=atoms, properties=('forces',))

        # Post-bias check should be scheduled
        expected_check_step = 100 + defaults["post_bias_check_delay"]
        assert gades_bias.next_postbias_check_step == expected_check_step

    def test_should_check_stability_true_at_postbias_step(self):
        """should_check_stability() should return True at post-bias check step."""
        from GADES.config import defaults

        n_atoms = 10
        bias_indices = [0, 1, 2]

        atoms = MockAtoms(n_atoms=n_atoms)
        base_calc = MockBaseCalculator(n_atoms=n_atoms)
        base_calc.results['forces'] = np.ones((n_atoms, 3))

        def mock_hess_func(backend, atom_indices, step_size, platform):
            n = len(atom_indices) * 3
            return np.eye(n)

        backend = ASEBackend.with_gades(
            atoms=atoms,
            base_calc=base_calc,
            bias_atom_indices=bias_indices,
            hess_func=mock_hess_func,
            clamp_magnitude=1000,
            kappa=0.9,
            interval=100,
            stability_interval=None,  # No regular checks
        )

        gades_bias = backend.gades_bias
        post_bias_delay = defaults["post_bias_check_delay"]

        # Step 100: apply bias, schedules post-bias check
        backend.integrator = MockIntegrator(nsteps=100)
        atoms.calc.calculate(atoms=atoms, properties=('forces',))

        # Step 101-199: should NOT check stability (before post-bias step)
        backend.integrator = MockIntegrator(nsteps=101)
        assert gades_bias.should_check_stability() is False

        backend.integrator = MockIntegrator(nsteps=100 + post_bias_delay - 1)
        assert gades_bias.should_check_stability() is False

        # Step at post-bias check: SHOULD check stability
        backend.integrator = MockIntegrator(nsteps=100 + post_bias_delay)
        assert gades_bias.should_check_stability() is True

    def test_post_bias_check_cleared_after_firing(self):
        """Post-bias check should be cleared after it fires."""
        from GADES.config import defaults

        n_atoms = 10
        bias_indices = [0, 1, 2]

        atoms = MockAtoms(n_atoms=n_atoms, temperature=300.0)
        base_calc = MockBaseCalculator(n_atoms=n_atoms)
        base_calc.results['forces'] = np.ones((n_atoms, 3))

        def mock_hess_func(backend, atom_indices, step_size, platform):
            n = len(atom_indices) * 3
            return np.eye(n)

        # Use interval=150 so post-bias check (step 250) doesn't coincide with
        # a bias update step (multiples of 150: 150, 300, 450, ...)
        backend = ASEBackend.with_gades(
            atoms=atoms,
            base_calc=base_calc,
            bias_atom_indices=bias_indices,
            hess_func=mock_hess_func,
            clamp_magnitude=1000,
            kappa=0.9,
            interval=150,
            stability_interval=None,
            target_temperature=300.0,
        )

        gades_calc = atoms.calc
        gades_bias = backend.gades_bias
        post_bias_delay = defaults["post_bias_check_delay"]  # 100

        # Step 150: apply bias (first update step for interval=150)
        backend.integrator = MockIntegrator(nsteps=150)
        gades_calc.calculate(atoms=atoms, properties=('forces',))

        # Post-bias check scheduled at step 250 (150 + 100)
        assert gades_bias.next_postbias_check_step == 150 + post_bias_delay

        # Step 250: post-bias check fires (not a bias update step)
        backend.integrator = MockIntegrator(nsteps=150 + post_bias_delay)
        gades_calc.calculate(atoms=atoms, properties=('forces',))

        # Post-bias check should be cleared (and no new one scheduled since not update step)
        assert gades_bias.next_postbias_check_step is None

    def test_stability_interval_none_still_gets_post_bias_checks(self):
        """With stability_interval=None, post-bias checks should still occur."""
        from GADES.config import defaults

        n_atoms = 10
        bias_indices = [0, 1, 2]

        atoms = MockAtoms(n_atoms=n_atoms, temperature=300.0)
        base_calc = MockBaseCalculator(n_atoms=n_atoms)
        base_calc.results['forces'] = np.ones((n_atoms, 3))

        def mock_hess_func(backend, atom_indices, step_size, platform):
            n = len(atom_indices) * 3
            return np.eye(n)

        backend = ASEBackend.with_gades(
            atoms=atoms,
            base_calc=base_calc,
            bias_atom_indices=bias_indices,
            hess_func=mock_hess_func,
            clamp_magnitude=1000,
            kappa=0.9,
            interval=100,
            stability_interval=None,  # Key: no regular interval checks
            target_temperature=300.0,
        )

        gades_calc = atoms.calc
        gades_bias = backend.gades_bias
        post_bias_delay = defaults["post_bias_check_delay"]

        # Step 100: apply bias
        backend.integrator = MockIntegrator(nsteps=100)
        gades_calc.calculate(atoms=atoms, properties=('forces',))
        assert gades_bias.next_postbias_check_step == 100 + post_bias_delay

        # Before post-bias check step, stability should not be checked
        backend.integrator = MockIntegrator(nsteps=150)
        assert gades_bias.should_check_stability() is False

        # At post-bias check step, stability SHOULD be checked
        backend.integrator = MockIntegrator(nsteps=100 + post_bias_delay)
        assert gades_bias.should_check_stability() is True

    def test_post_bias_check_detects_instability(self):
        """Post-bias check should detect instability and remove bias."""
        from GADES.config import defaults

        n_atoms = 10
        bias_indices = [0, 1, 2]

        atoms = MockAtoms(n_atoms=n_atoms, temperature=300.0)
        base_calc = MockBaseCalculator(n_atoms=n_atoms)
        base_calc.results['forces'] = np.ones((n_atoms, 3))

        def mock_hess_func(backend, atom_indices, step_size, platform):
            n = len(atom_indices) * 3
            return np.eye(n)

        backend = ASEBackend.with_gades(
            atoms=atoms,
            base_calc=base_calc,
            bias_atom_indices=bias_indices,
            hess_func=mock_hess_func,
            clamp_magnitude=1000,
            kappa=0.9,
            interval=100,
            stability_interval=None,  # Only post-bias checks
            target_temperature=300.0,
        )

        gades_calc = atoms.calc
        post_bias_delay = defaults["post_bias_check_delay"]

        # Step 100: apply bias
        backend.integrator = MockIntegrator(nsteps=100)
        gades_calc.calculate(atoms=atoms, properties=('forces',))
        assert gades_calc._bias_active is True

        # Simulate instability: temperature spikes after bias applied
        atoms._temperature = 400.0  # 100K deviation > 50K threshold

        # Step at post-bias check: should detect instability
        backend.integrator = MockIntegrator(nsteps=100 + post_bias_delay)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gades_calc.calculate(atoms=atoms, properties=('forces',))

        # Bias should be cleared due to instability
        assert gades_calc._bias_active is False
        assert gades_calc._stored_bias is None


# ============================================================================
# OpenMMBackend Tests
# ============================================================================

# Try to import OpenMM-specific classes
try:
    from GADES.backend import OpenMMBackend, _OPENMM_AVAILABLE
    HAS_OPENMM = _OPENMM_AVAILABLE
except ImportError:
    HAS_OPENMM = False
    OpenMMBackend = None


class MockOpenMMQuantity:
    """Mock OpenMM Quantity for unit handling."""

    def __init__(self, value):
        self._value = value

    def value_in_unit(self, unit):
        return self._value


class MockOpenMMState:
    """Mock OpenMM State object."""

    def __init__(self, kinetic_energy=1000.0, positions=None, forces=None):
        self._kinetic_energy = kinetic_energy
        self._positions = positions if positions is not None else np.zeros((10, 3))
        self._forces = forces if forces is not None else np.zeros((10, 3))

    def getKineticEnergy(self):
        return MockOpenMMQuantity(self._kinetic_energy)

    def getPositions(self, asNumpy=False):
        if asNumpy:
            return MockOpenMMQuantity(self._positions)
        return self._positions

    def getForces(self, asNumpy=False):
        if asNumpy:
            return MockOpenMMQuantity(self._forces)
        return self._forces


class MockOpenMMContext:
    """Mock OpenMM Context object."""

    def __init__(self, kinetic_energy=1000.0, positions=None, forces=None):
        self._kinetic_energy = kinetic_energy
        self._positions = positions if positions is not None else np.zeros((10, 3))
        self._forces = forces if forces is not None else np.zeros((10, 3))

    def getState(self, getEnergy=False, getPositions=False, getForces=False, groups=None):
        return MockOpenMMState(
            kinetic_energy=self._kinetic_energy,
            positions=self._positions,
            forces=self._forces
        )

    def setPositions(self, positions):
        if hasattr(positions, '_value'):
            self._positions = positions._value
        else:
            self._positions = positions


class MockOpenMMIntegrator:
    """Mock OpenMM Integrator."""

    def __init__(self, temperature=None):
        self._temperature = temperature

    def getTemperature(self):
        if self._temperature is None:
            raise AttributeError("This integrator has no temperature")
        return MockOpenMMQuantity(self._temperature)


class MockOpenMMElement:
    """Mock OpenMM Element."""

    def __init__(self, symbol):
        self.symbol = symbol


class MockOpenMMAtom:
    """Mock OpenMM Atom."""

    def __init__(self, index, symbol="C"):
        self.index = index
        self.element = MockOpenMMElement(symbol)


class MockOpenMMTopology:
    """Mock OpenMM Topology."""

    def __init__(self, n_atoms=10, symbols=None):
        self._n_atoms = n_atoms
        self._symbols = symbols if symbols else ["C"] * n_atoms

    def atoms(self):
        return [MockOpenMMAtom(i, self._symbols[i]) for i in range(self._n_atoms)]


class MockOpenMMForce:
    """Mock OpenMM Force."""
    pass


class MockOpenMMSystem:
    """Mock OpenMM System."""

    def __init__(self, n_particles=10, n_constraints=0, masses=None, has_cm_remover=False):
        self._n_particles = n_particles
        self._n_constraints = n_constraints
        self._masses = masses if masses else [1.0] * n_particles
        self._has_cm_remover = has_cm_remover

    def getNumParticles(self):
        return self._n_particles

    def getNumConstraints(self):
        return self._n_constraints

    def getParticleMass(self, i):
        return MockOpenMMQuantity(self._masses[i])

    def getConstraintParameters(self, i):
        # Return (p1, p2, distance)
        return (0, 1, 1.0)

    def getNumForces(self):
        return 1 if self._has_cm_remover else 0

    def getForce(self, i):
        if self._has_cm_remover:
            # Return something that would match CMMotionRemover type check
            return MockOpenMMForce()
        return MockOpenMMForce()


class MockOpenMMSimulation:
    """Mock OpenMM Simulation object."""

    def __init__(
        self,
        n_atoms=10,
        current_step=0,
        kinetic_energy=1000.0,
        integrator_temp=None,
        n_constraints=0,
        has_cm_remover=False,
        symbols=None,
    ):
        self.system = MockOpenMMSystem(
            n_particles=n_atoms,
            n_constraints=n_constraints,
            has_cm_remover=has_cm_remover
        )
        self.context = MockOpenMMContext(kinetic_energy=kinetic_energy)
        self.integrator = MockOpenMMIntegrator(temperature=integrator_temp)
        self.topology = MockOpenMMTopology(n_atoms=n_atoms, symbols=symbols)
        self.currentStep = current_step


@pytest.mark.skipif(not HAS_OPENMM, reason="OpenMM not installed")
class TestOpenMMBackendInitialization:
    """Tests for OpenMMBackend initialization."""

    def test_basic_initialization(self):
        """Test basic OpenMMBackend initialization."""
        simulation = MockOpenMMSimulation()
        backend = OpenMMBackend(simulation)

        assert backend.name == "openmm"
        assert backend.simulation is simulation
        assert backend.system is simulation.system
        assert backend.target_temperature is None
        assert backend._stability_warning_issued is False

    def test_initialization_with_target_temperature(self):
        """Test initialization with explicit target_temperature."""
        simulation = MockOpenMMSimulation()
        backend = OpenMMBackend(simulation, target_temperature=350.0)

        assert backend.target_temperature == 350.0


@pytest.mark.skipif(not HAS_OPENMM, reason="OpenMM not installed")
class TestOpenMMBackendGetTargetTemperature:
    """Tests for OpenMMBackend._get_target_temperature method."""

    def test_explicit_target_temperature(self):
        """Explicit target_temperature should be returned."""
        simulation = MockOpenMMSimulation()
        backend = OpenMMBackend(simulation, target_temperature=350.0)

        assert backend._get_target_temperature() == 350.0

    def test_langevin_integrator_temperature(self):
        """Should read temperature from LangevinIntegrator."""
        simulation = MockOpenMMSimulation(integrator_temp=400.0)
        backend = OpenMMBackend(simulation)

        assert backend._get_target_temperature() == 400.0

    def test_explicit_overrides_integrator(self):
        """Explicit target_temperature should override integrator."""
        simulation = MockOpenMMSimulation(integrator_temp=400.0)
        backend = OpenMMBackend(simulation, target_temperature=350.0)

        assert backend._get_target_temperature() == 350.0

    def test_no_target_available(self):
        """Should return None when no target temperature available."""
        simulation = MockOpenMMSimulation(integrator_temp=None)
        backend = OpenMMBackend(simulation)

        assert backend._get_target_temperature() is None


@pytest.mark.skipif(not HAS_OPENMM, reason="OpenMM not installed")
class TestOpenMMBackendIsStable:
    """Tests for OpenMMBackend.is_stable method."""

    def test_no_target_returns_true_with_warning(self):
        """Without target temperature, should warn and return True."""
        simulation = MockOpenMMSimulation(integrator_temp=None)
        backend = OpenMMBackend(simulation)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = backend.is_stable()

            assert result is True
            assert len(w) == 1
            assert "target temperature" in str(w[0].message).lower()

    def test_warning_only_issued_once(self):
        """Warning should only be issued once."""
        simulation = MockOpenMMSimulation(integrator_temp=None)
        backend = OpenMMBackend(simulation)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            backend.is_stable()
            backend.is_stable()
            backend.is_stable()

            stability_warnings = [x for x in w if "target temperature" in str(x.message).lower()]
            assert len(stability_warnings) == 1

    def test_zero_dof_returns_true_with_warning(self):
        """With zero DOF (e.g., all virtual sites), should warn and return True."""
        # Create simulation with zero-mass particles (virtual sites)
        simulation = MockOpenMMSimulation(integrator_temp=300.0)
        simulation.system._masses = [0.0] * 10  # All zero mass = 0 DOF

        backend = OpenMMBackend(simulation)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = backend.is_stable()

            assert result is True
            assert len(w) == 1
            assert "dof" in str(w[0].message).lower()

    def test_zero_dof_warning_only_issued_once(self):
        """Zero DOF warning should only be issued once."""
        simulation = MockOpenMMSimulation(integrator_temp=300.0)
        simulation.system._masses = [0.0] * 10

        backend = OpenMMBackend(simulation)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            backend.is_stable()
            backend.is_stable()
            backend.is_stable()

            dof_warnings = [x for x in w if "dof" in str(x.message).lower()]
            assert len(dof_warnings) == 1


@pytest.mark.skipif(not HAS_OPENMM, reason="OpenMM not installed")
class TestOpenMMBackendGetCurrentStep:
    """Tests for OpenMMBackend.get_currentStep method."""

    def test_returns_simulation_current_step(self):
        """Should return simulation.currentStep."""
        simulation = MockOpenMMSimulation(current_step=500)
        backend = OpenMMBackend(simulation)

        assert backend.get_currentStep() == 500

    def test_returns_zero_initially(self):
        """Should return 0 for new simulation."""
        simulation = MockOpenMMSimulation(current_step=0)
        backend = OpenMMBackend(simulation)

        assert backend.get_currentStep() == 0


@pytest.mark.skipif(not HAS_OPENMM, reason="OpenMM not installed")
class TestOpenMMBackendGetAtomSymbols:
    """Tests for OpenMMBackend.get_atom_symbols method."""

    def test_get_atom_symbols(self):
        """Should return correct symbols for given indices."""
        symbols = ["C", "N", "O", "H", "H"]
        simulation = MockOpenMMSimulation(n_atoms=5, symbols=symbols)
        backend = OpenMMBackend(simulation)

        result = backend.get_atom_symbols([0, 2, 4])
        assert result == ["C", "O", "H"]

    def test_get_all_atom_symbols(self):
        """Should return all symbols when given all indices."""
        symbols = ["C", "N", "O"]
        simulation = MockOpenMMSimulation(n_atoms=3, symbols=symbols)
        backend = OpenMMBackend(simulation)

        result = backend.get_atom_symbols([0, 1, 2])
        assert result == ["C", "N", "O"]


@pytest.mark.skipif(not HAS_OPENMM, reason="OpenMM not installed")
class TestOpenMMBackendGetPositions:
    """Tests for OpenMMBackend.get_positions method."""

    def test_get_positions_returns_array(self):
        """Should return positions as numpy array."""
        simulation = MockOpenMMSimulation(n_atoms=5)
        # Set known positions
        positions = np.array([[1.0, 2.0, 3.0]] * 5)
        simulation.context._positions = positions

        backend = OpenMMBackend(simulation)
        result = backend.get_positions()

        assert isinstance(result, np.ndarray)
        assert result.shape == (5, 3)


@pytest.mark.skipif(not HAS_OPENMM, reason="OpenMM not installed")
class TestOpenMMBackendGetAtoms:
    """Tests for OpenMMBackend.get_atoms method."""

    def test_get_atoms_returns_iterator(self):
        """Should return atom iterator from topology."""
        simulation = MockOpenMMSimulation(n_atoms=5)
        backend = OpenMMBackend(simulation)

        atoms = list(backend.get_atoms())
        assert len(atoms) == 5
