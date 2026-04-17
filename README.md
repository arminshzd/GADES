![GADES](/docs/docs/imgs/GADES_logo_wbg.png)

# GADES

Gentlest Ascent Dynamics for Enhanced Sampling.
GADES is an enhanced sampling method based on Gentlest Ascent Dynamics (GAD) for exploring molecular configuration space without prior knowledge of reaction coordinates.

## Installation

to install, clone the repository

``` bash
git clone https://github.com/arminshzd/GADES.git
```

create a conda environment (`python >= 3.10`)

``` bash
conda create -n GADES python=3.10
conda activate GADES
```

and install from inside the `GADES` directory using `pip`

``` bash
pip install -e .
```

## Usage

GADES supports [OpenMM](https://openmm.org/) and [ASE](https://ase-lib.org/). Find the API documentation and working examples on the [documentations](https://arminshzd.github.io/GADES/) website and the example scripts under `examples`.

To use GADES with OpenMM:

1) Import the hessian calculation method and the force biasing method `createGADESBiasForce`:

    ``` python
    from GADES.utils import compute_hessian_force_fd_richardson as hessian
    from GADES import createGADESBiasForce, GADESForceUpdater
    from GADES.backend import OpenMMBackend
    ```

2) Create a `GAD_force` object and add it to your system

    ``` python
    GAD_force = createGADESBiasForce(system.getNumParticles())
    system.addForce(GAD_force)
    ```

3) Create the backend and append `GADESForceUpdater` to your simulation reporters

    ``` python
    backend = OpenMMBackend(simulation)
    simulation.reporters.append(
        GADESForceUpdater(
            backend=backend,
            biased_force=GAD_force,
            bias_atom_indices=biasing_atom_ids,
            hess_func=hessian,
            clamp_magnitude=CLAMP_MAGNITUDE,
            kappa=KAPPA,
            interval=BIAS_UPDATE_FREQ,
            stability_interval=STABILITY_CHECK_FREQ,
            logfile_prefix=LOG_PREFIX
            )
        )
    ```

The example script `examples/BluePrint/sys_example.py` and `examples/BluePrint/sys_example2.py` demonstrate how to use GADES with the OpenMM backend.

To use GADES with ASE:

1) Import the required modules:

    ``` python
    from GADES.utils import compute_hessian_force_fd_richardson as hessian
    from GADES.backend import ASEBackend
    ```

2) Create your atoms and base calculator:

    ``` python
    atoms = bulk('Ar', 'fcc', a=5.26).repeat((2, 2, 2))
    lammps_calc = LAMMPS(**parameters)
    ```

3) Use the `ASEBackend.with_gades()` factory method to set up everything:

    ``` python
    biasing_atom_ids = [atom.index for atom in atoms if atom.symbol == 'Ar']

    backend = ASEBackend.with_gades(
        atoms=atoms,
        base_calc=lammps_calc,
        bias_atom_indices=biasing_atom_ids,
        hess_func=hessian,
        clamp_magnitude=CLAMP_MAGNITUDE,
        kappa=KAPPA,
        interval=BIAS_UPDATE_FREQ,
        stability_interval=STABILITY_CHECK_FREQ,
        logfile_prefix=LOG_PREFIX,
        target_temperature=300.0,  # Kelvin — always set this explicitly
    )
    ```

    > **Important:** Always pass `target_temperature` (in Kelvin) to `ASEBackend`.
    > Without it, GADES tries to infer the temperature from the integrator object,
    > but ASE integrators store temperature under different attribute names and in
    > different units depending on the integrator type. This auto-detection can
    > silently fail or return the wrong value for integrators such as
    > `NoseHooverChain`. Setting `target_temperature` explicitly avoids all of this.

4) Create a time integrator and attach it to the backend:

    ``` python
    dyn = VelocityVerlet(atoms, timestep_fs * units.fs)
    backend.integrator = dyn
    ```

The example script `examples/ase/mdrun.py` demonstrates how to use GADES with the ASE backend.

### Large Systems (1000+ atoms)

For large molecular systems where the full Hessian doesn't fit in memory, use the matrix-free Lanczos eigensolver:

``` python
backend = ASEBackend.with_gades(
    atoms=atoms,
    base_calc=calculator,
    bias_atom_indices=list(range(len(atoms))),
    hess_func=hessian,
    clamp_magnitude=1000,
    kappa=0.9,
    interval=1000,
    eigensolver='lanczos_hvp',      # Matrix-free eigensolver (O(N) memory)
    lanczos_iterations=50,           # More iterations for eigenvector accuracy
    hvp_epsilon=1e-5,                # Finite difference step size
)
```

See the [Large Systems Guide](https://arminshzd.github.io/GADES/guides/large_systems/) for details on eigensolver options and accuracy considerations.

## Implementation notes

The `GADESBias` class provides the generic implementation of the GADES bias applied to any MD engine, or backend, such as OpenMM and ASE.

The `GADESForceUpdater` class is derived from `GADESBias` and provides the functions (`describeNextReport()` and `report()`)required by the OpenMM Reporter API.

The parameters of the `GADESBias` constructor are:

**Core parameters:**

* `backend`: The simulation engine to be driven by, or coupled with, `GADESBias`
* `biased_force`: The bias force associated with GADES
* `bias_atom_indices`: Indices of the atoms to be biased. This option allows using block Hessians for the GADES calculations instead of the full Hessian. It also allows for finer control of which part of the system you want to bias.
* `hess_func`: The method for Hessian estimation. There are two methods available in `GADES.utils` module, both of which use finite difference to estimate the Hessian from forces. We suggest using `compute_hessian_force_fd_richardson` which uses Richardson extrapolation for a more accurate estimation and less sensitivity to step size.
* `clamp_magnitude`: Maximum magnitude of the bias force applied to the system. Same units as your MD engine.
* `kappa`: κ value in the GADES force formulation. We suggest setting it to 0.9 for the most effective exploration and control the biasing effects through `clamp_magnitude` instead.
* `interval`: Intervals at which the bias magnitude and direction gets updated. This also controls the frequency of Hessian calculation and diagonalization.
* `stability_interval`: Intervals at which stability checks are performed to ensure system stability. We suggest setting this to `interval`//2 or smaller.
* `logfile_prefix`: Prefix for the log files created by GADES. `None` would skip logging the most negative eigenvalue, the corresponding eigenvector, and the current location on the potential energy surface at each update interval.

**Eigensolver options (for large systems):**

* `eigensolver`: Method for finding the softest mode. Options:
  - `'numpy'` (default): Full eigendecomposition via `np.linalg.eigh()`. O(N²) memory, O(N³) time.
  - `'lanczos'`: Matrix-based Lanczos iteration. O(N²) memory, O(k·N²) time.
  - `'lanczos_hvp'`: Matrix-free Lanczos with Hessian-vector products. O(N) memory, suitable for large systems.
* `lanczos_iterations`: Number of Lanczos iterations (default: 20). Use 50-100 for production runs to ensure eigenvector accuracy.
* `hvp_epsilon`: Finite difference step size for HVP computation (default: 1e-5).

**Bofill Hessian approximation:**

* `use_bofill_update`: Enable Bofill quasi-Newton Hessian updates between full computations (default: False). When enabled, consider setting `interval` to 1-10 steps for frequent bias updates.
* `full_hessian_interval`: Steps between full Hessian recomputations when using Bofill (default: 10 × interval).


## Development

To set up a development environment:

``` bash
git clone https://github.com/arminshzd/GADES.git
cd GADES
pip install -e ".[dev]"
```

To run the test suite:

``` bash
pytest
```

To run tests with coverage:

``` bash
pytest --cov=GADES --cov-report=term-missing
```

### Disclaimer

This repo is still under active development.
