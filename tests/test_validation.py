"""
Tests for input validation in GADES.
"""
import logging
import pytest
import warnings
import numpy as np

from GADES import GADESBias, createGADESBiasForce


class TestCreateGADESBiasForceValidation:
    """Tests for createGADESBiasForce input validation."""

    def test_valid_n_particles(self):
        """Valid n_particles should create force without error."""
        force = createGADESBiasForce(10)
        assert force is not None

    def test_zero_particles(self):
        """Zero particles should be valid (edge case)."""
        force = createGADESBiasForce(0)
        assert force is not None

    def test_negative_n_particles(self):
        """Negative n_particles should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative integer"):
            createGADESBiasForce(-1)

    def test_float_n_particles(self):
        """Float n_particles should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative integer"):
            createGADESBiasForce(10.5)

    def test_string_n_particles(self):
        """String n_particles should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative integer"):
            createGADESBiasForce("10")

    def test_none_n_particles(self):
        """None n_particles should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative integer"):
            createGADESBiasForce(None)

    def test_numpy_int_n_particles(self):
        """NumPy integer should be accepted."""
        force = createGADESBiasForce(np.int64(10))
        assert force is not None


class TestGADESBiasValidation:
    """Tests for GADESBias.__init__ input validation."""

    @pytest.fixture
    def valid_hess_func(self):
        """A valid callable for hess_func."""
        def hess(backend, atom_indices, step_size, platform):
            n_dof = len(atom_indices) * 3
            return np.eye(n_dof)
        return hess

    # --- hess_func validation ---

    def test_valid_hess_func(self, valid_hess_func):
        """Valid callable hess_func should work."""
        bias = GADESBias(
            backend=None,
            biased_force=None,
            bias_atom_indices=[0, 1, 2],
            hess_func=valid_hess_func,
            clamp_magnitude=1000.0,
            kappa=0.9,
            interval=200,
        )
        assert bias.hess_func is valid_hess_func

    def test_non_callable_hess_func(self):
        """Non-callable hess_func should raise TypeError."""
        with pytest.raises(TypeError, match="hess_func must be callable"):
            GADESBias(
                backend=None,
                biased_force=None,
                bias_atom_indices=[0, 1, 2],
                hess_func="not_callable",
                clamp_magnitude=1000.0,
                kappa=0.9,
                interval=200,
            )

    def test_none_hess_func(self):
        """None hess_func should raise TypeError."""
        with pytest.raises(TypeError, match="hess_func must be callable"):
            GADESBias(
                backend=None,
                biased_force=None,
                bias_atom_indices=[0, 1, 2],
                hess_func=None,
                clamp_magnitude=1000.0,
                kappa=0.9,
                interval=200,
            )

    # --- bias_atom_indices validation ---

    def test_valid_bias_atom_indices_list(self, valid_hess_func):
        """List of integers should work."""
        bias = GADESBias(
            backend=None,
            biased_force=None,
            bias_atom_indices=[0, 1, 2],
            hess_func=valid_hess_func,
            clamp_magnitude=1000.0,
            kappa=0.9,
            interval=200,
        )
        assert list(bias.bias_atom_indices) == [0, 1, 2]

    def test_valid_bias_atom_indices_numpy(self, valid_hess_func):
        """NumPy array should work."""
        indices = np.array([0, 1, 2])
        bias = GADESBias(
            backend=None,
            biased_force=None,
            bias_atom_indices=indices,
            hess_func=valid_hess_func,
            clamp_magnitude=1000.0,
            kappa=0.9,
            interval=200,
        )
        assert list(bias.bias_atom_indices) == [0, 1, 2]

    def test_valid_bias_atom_indices_tuple(self, valid_hess_func):
        """Tuple should work."""
        bias = GADESBias(
            backend=None,
            biased_force=None,
            bias_atom_indices=(0, 1, 2),
            hess_func=valid_hess_func,
            clamp_magnitude=1000.0,
            kappa=0.9,
            interval=200,
        )
        assert list(bias.bias_atom_indices) == [0, 1, 2]

    def test_empty_bias_atom_indices(self, valid_hess_func):
        """Empty bias_atom_indices should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            GADESBias(
                backend=None,
                biased_force=None,
                bias_atom_indices=[],
                hess_func=valid_hess_func,
                clamp_magnitude=1000.0,
                kappa=0.9,
                interval=200,
            )

    def test_non_sequence_bias_atom_indices(self, valid_hess_func):
        """Non-sequence should raise TypeError."""
        with pytest.raises(TypeError, match="must be a sequence"):
            GADESBias(
                backend=None,
                biased_force=None,
                bias_atom_indices=42,
                hess_func=valid_hess_func,
                clamp_magnitude=1000.0,
                kappa=0.9,
                interval=200,
            )

    def test_negative_bias_atom_indices(self, valid_hess_func):
        """Negative indices should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative integers"):
            GADESBias(
                backend=None,
                biased_force=None,
                bias_atom_indices=[0, -1, 2],
                hess_func=valid_hess_func,
                clamp_magnitude=1000.0,
                kappa=0.9,
                interval=200,
            )

    def test_float_bias_atom_indices(self, valid_hess_func):
        """Float indices should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative integers"):
            GADESBias(
                backend=None,
                biased_force=None,
                bias_atom_indices=[0, 1.5, 2],
                hess_func=valid_hess_func,
                clamp_magnitude=1000.0,
                kappa=0.9,
                interval=200,
            )

    def test_out_of_bounds_bias_atom_indices(self, valid_hess_func, mock_backend):
        """Indices exceeding system size should raise ValueError."""
        # mock_backend has 10 atoms (indices 0-9 valid)
        with pytest.raises(ValueError, match="index 100.*only has 10 atoms"):
            GADESBias(
                backend=mock_backend,
                biased_force=None,
                bias_atom_indices=[0, 1, 100],  # 100 is out of bounds
                hess_func=valid_hess_func,
                clamp_magnitude=1000.0,
                kappa=0.9,
                interval=200,
            )

    def test_valid_max_index_bias_atom_indices(self, valid_hess_func, mock_backend):
        """Index at system boundary (n_atoms - 1) should work."""
        # mock_backend has 10 atoms (indices 0-9 valid)
        bias = GADESBias(
            backend=mock_backend,
            biased_force=None,
            bias_atom_indices=[0, 5, 9],  # 9 is the last valid index
            hess_func=valid_hess_func,
            clamp_magnitude=1000.0,
            kappa=0.9,
            interval=200,
        )
        assert list(bias.bias_atom_indices) == [0, 5, 9]

    # --- clamp_magnitude validation ---

    def test_valid_clamp_magnitude(self, valid_hess_func):
        """Positive clamp_magnitude should work."""
        bias = GADESBias(
            backend=None,
            biased_force=None,
            bias_atom_indices=[0, 1, 2],
            hess_func=valid_hess_func,
            clamp_magnitude=1000.0,
            kappa=0.9,
            interval=200,
        )
        assert bias.clamp_magnitude == 1000.0

    def test_zero_clamp_magnitude(self, valid_hess_func):
        """Zero clamp_magnitude should raise ValueError."""
        with pytest.raises(ValueError, match="positive number"):
            GADESBias(
                backend=None,
                biased_force=None,
                bias_atom_indices=[0, 1, 2],
                hess_func=valid_hess_func,
                clamp_magnitude=0.0,
                kappa=0.9,
                interval=200,
            )

    def test_negative_clamp_magnitude(self, valid_hess_func):
        """Negative clamp_magnitude should raise ValueError."""
        with pytest.raises(ValueError, match="positive number"):
            GADESBias(
                backend=None,
                biased_force=None,
                bias_atom_indices=[0, 1, 2],
                hess_func=valid_hess_func,
                clamp_magnitude=-100.0,
                kappa=0.9,
                interval=200,
            )

    def test_string_clamp_magnitude(self, valid_hess_func):
        """String clamp_magnitude should raise ValueError."""
        with pytest.raises(ValueError, match="positive number"):
            GADESBias(
                backend=None,
                biased_force=None,
                bias_atom_indices=[0, 1, 2],
                hess_func=valid_hess_func,
                clamp_magnitude="1000",
                kappa=0.9,
                interval=200,
            )

    # --- interval validation ---

    def test_valid_interval(self, valid_hess_func):
        """Positive integer interval should work."""
        bias = GADESBias(
            backend=None,
            biased_force=None,
            bias_atom_indices=[0, 1, 2],
            hess_func=valid_hess_func,
            clamp_magnitude=1000.0,
            kappa=0.9,
            interval=200,
        )
        assert bias.interval == 200

    def test_zero_interval(self, valid_hess_func):
        """Zero interval should raise ValueError."""
        with pytest.raises(ValueError, match="positive integer"):
            GADESBias(
                backend=None,
                biased_force=None,
                bias_atom_indices=[0, 1, 2],
                hess_func=valid_hess_func,
                clamp_magnitude=1000.0,
                kappa=0.9,
                interval=0,
            )

    def test_negative_interval(self, valid_hess_func):
        """Negative interval should raise ValueError."""
        with pytest.raises(ValueError, match="positive integer"):
            GADESBias(
                backend=None,
                biased_force=None,
                bias_atom_indices=[0, 1, 2],
                hess_func=valid_hess_func,
                clamp_magnitude=1000.0,
                kappa=0.9,
                interval=-100,
            )

    def test_float_interval(self, valid_hess_func):
        """Float interval should raise ValueError."""
        with pytest.raises(ValueError, match="positive integer"):
            GADESBias(
                backend=None,
                biased_force=None,
                bias_atom_indices=[0, 1, 2],
                hess_func=valid_hess_func,
                clamp_magnitude=1000.0,
                kappa=0.9,
                interval=200.5,
            )

    def test_small_interval_warning(self, valid_hess_func, caplog):
        """Interval < 100 should log warning and be overridden to 110."""
        with caplog.at_level(logging.WARNING):
            bias = GADESBias(
                backend=None,
                biased_force=None,
                bias_atom_indices=[0, 1, 2],
                hess_func=valid_hess_func,
                clamp_magnitude=1000.0,
                kappa=0.9,
                interval=50,
            )
        assert bias.interval == 110
        assert "larger than 100 steps" in caplog.text

    # --- stability_interval validation ---

    def test_valid_stability_interval(self, valid_hess_func):
        """Positive integer stability_interval should work."""
        bias = GADESBias(
            backend=None,
            biased_force=None,
            bias_atom_indices=[0, 1, 2],
            hess_func=valid_hess_func,
            clamp_magnitude=1000.0,
            kappa=0.9,
            interval=200,
            stability_interval=100,
        )
        assert bias.s_interval == 100

    def test_none_stability_interval(self, valid_hess_func):
        """None stability_interval should work (optional)."""
        bias = GADESBias(
            backend=None,
            biased_force=None,
            bias_atom_indices=[0, 1, 2],
            hess_func=valid_hess_func,
            clamp_magnitude=1000.0,
            kappa=0.9,
            interval=200,
            stability_interval=None,
        )
        assert bias.s_interval is None

    def test_zero_stability_interval(self, valid_hess_func):
        """Zero stability_interval should raise ValueError."""
        with pytest.raises(ValueError, match="positive integer"):
            GADESBias(
                backend=None,
                biased_force=None,
                bias_atom_indices=[0, 1, 2],
                hess_func=valid_hess_func,
                clamp_magnitude=1000.0,
                kappa=0.9,
                interval=200,
                stability_interval=0,
            )

    def test_negative_stability_interval(self, valid_hess_func):
        """Negative stability_interval should raise ValueError."""
        with pytest.raises(ValueError, match="positive integer"):
            GADESBias(
                backend=None,
                biased_force=None,
                bias_atom_indices=[0, 1, 2],
                hess_func=valid_hess_func,
                clamp_magnitude=1000.0,
                kappa=0.9,
                interval=200,
                stability_interval=-50,
            )

    # --- kappa validation (warning, not error) ---

    def test_valid_kappa_in_range(self, valid_hess_func):
        """Kappa in (0, 1] should work without warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bias = GADESBias(
                backend=None,
                biased_force=None,
                bias_atom_indices=[0, 1, 2],
                hess_func=valid_hess_func,
                clamp_magnitude=1000.0,
                kappa=0.9,
                interval=200,
            )
            # No UserWarning about kappa should be raised
            kappa_warnings = [x for x in w if "kappa" in str(x.message)]
            assert len(kappa_warnings) == 0
        assert bias.kappa == 0.9

    def test_kappa_exactly_one(self, valid_hess_func):
        """Kappa = 1.0 should work without warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bias = GADESBias(
                backend=None,
                biased_force=None,
                bias_atom_indices=[0, 1, 2],
                hess_func=valid_hess_func,
                clamp_magnitude=1000.0,
                kappa=1.0,
                interval=200,
            )
            kappa_warnings = [x for x in w if "kappa" in str(x.message)]
            assert len(kappa_warnings) == 0

    def test_kappa_greater_than_one_warning(self, valid_hess_func):
        """Kappa > 1 should issue warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bias = GADESBias(
                backend=None,
                biased_force=None,
                bias_atom_indices=[0, 1, 2],
                hess_func=valid_hess_func,
                clamp_magnitude=1000.0,
                kappa=1.5,
                interval=200,
            )
            kappa_warnings = [x for x in w if "kappa" in str(x.message).lower()]
            assert len(kappa_warnings) == 1
            assert "outside the recommended range" in str(kappa_warnings[0].message)

    def test_kappa_zero_warning(self, valid_hess_func):
        """Kappa = 0 should issue warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bias = GADESBias(
                backend=None,
                biased_force=None,
                bias_atom_indices=[0, 1, 2],
                hess_func=valid_hess_func,
                clamp_magnitude=1000.0,
                kappa=0.0,
                interval=200,
            )
            kappa_warnings = [x for x in w if "kappa" in str(x.message).lower()]
            assert len(kappa_warnings) == 1

    def test_kappa_negative_warning(self, valid_hess_func):
        """Negative kappa should issue warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bias = GADESBias(
                backend=None,
                biased_force=None,
                bias_atom_indices=[0, 1, 2],
                hess_func=valid_hess_func,
                clamp_magnitude=1000.0,
                kappa=-0.5,
                interval=200,
            )
            kappa_warnings = [x for x in w if "kappa" in str(x.message).lower()]
            assert len(kappa_warnings) == 1


class TestSetHessStepSizeValidation:
    """Tests for set_hess_step_size validation."""

    @pytest.fixture
    def gades_bias(self):
        """Create a GADESBias instance for testing."""
        def hess(backend, atom_indices, step_size, platform):
            return np.eye(len(atom_indices) * 3)
        return GADESBias(
            backend=None,
            biased_force=None,
            bias_atom_indices=[0, 1, 2],
            hess_func=hess,
            clamp_magnitude=1000.0,
            kappa=0.9,
            interval=200,
        )

    def test_valid_delta(self, gades_bias):
        """Valid positive delta should work."""
        gades_bias.set_hess_step_size(1e-4)
        assert gades_bias.hess_step_size == 1e-4

    def test_zero_delta(self, gades_bias):
        """Zero delta should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            gades_bias.set_hess_step_size(0.0)

    def test_negative_delta(self, gades_bias):
        """Negative delta should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            gades_bias.set_hess_step_size(-1e-4)

    def test_string_delta(self, gades_bias):
        """String delta should raise TypeError."""
        with pytest.raises(TypeError, match="must be a number"):
            gades_bias.set_hess_step_size("1e-4")

    def test_none_delta(self, gades_bias):
        """None delta should raise TypeError."""
        with pytest.raises(TypeError, match="must be a number"):
            gades_bias.set_hess_step_size(None)

    def test_numpy_float_delta(self, gades_bias):
        """NumPy float delta should work."""
        gades_bias.set_hess_step_size(np.float64(1e-4))
        assert gades_bias.hess_step_size == pytest.approx(1e-4)
