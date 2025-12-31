"""
GADES configuration defaults.

This module contains default values for GADES parameters that can be
modified at runtime by users.

Example usage:
    >>> from GADES import defaults
    >>> print(defaults)
    {'stability_threshold_temp_diff': 50, ...}

    >>> # Modify a default
    >>> defaults["stability_threshold_temp_diff"] = 75

    >>> # Or update multiple values
    >>> defaults.update({
    ...     "stability_threshold_temp_diff": 75,
    ...     "post_bias_check_delay": 200,
    ... })
"""

defaults = {
    # Temperature deviation threshold (K) for stability checking.
    # If |T_current - T_target| > this value, system is considered unstable.
    "stability_threshold_temp_diff": 50,

    # Number of steps after a bias update to perform a stability check.
    # This ensures the system remains stable after applying new bias forces.
    "post_bias_check_delay": 100,

    # Minimum allowed bias update interval (steps).
    # If user specifies interval < (this - 10), it will be overridden to this value.
    "min_bias_update_interval": 110,

    # OpenMM force group for GADES bias forces.
    # Used to separate GADES forces from other forces in the simulation.
    "gades_force_group": 1,

    # Number of Lanczos iterations for eigenvalue computation.
    # More iterations give better accuracy but increase computation time.
    # For most systems, 20 iterations is sufficient.
    "lanczos_iterations": 20,

    # Multiplier for full Hessian recalculation when using Bofill updates.
    # Full Hessian is computed every (interval * bofill_full_hessian_multiplier) steps.
    # For example, if interval=1000 and multiplier=10, full Hessian computed every 10000 steps.
    "bofill_full_hessian_multiplier": 10,

    # Finite difference step size for Hessian-vector product (HVP) computation.
    # Used when eigensolver='lanczos_hvp'. Smaller values give more accuracy
    # but may suffer from numerical noise. Default balances accuracy and stability.
    "hvp_epsilon": 1e-5,
}
