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

1) Import the hessian calculation method and the Force biasing method `getGADESBiasForce`:

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

3) Append `GADESForceUpdater` to your simulation reporters

    ``` python
    simulation.reporters.append(
        GADESForceUpdater(
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

1) Import the hessian calculation method and the Force biasing method `getGADESBiasForce`:

    ``` python
    from GADES.utils import compute_hessian_force_fd_richardson as hessian
    from GADES import GADESBias
    from GADES.backend import ASEBackend, GADESCalculator
    ```

2) Create a base calculator, e.g. a LAMMPS calculator

    ``` python
    lammps_calc = LAMMPS(**parameters)
    ```

3) Create a `GADESBias` object

    ``` python
    biasing_atom_ids = np.array([atom.index for atom in atoms if (atom.symbol == 'Ar')])

    force_bias = GADESBias(backend=None,
                       biased_force=None, # maybe can be acquired internally
                       bias_atom_indices=biasing_atom_ids,
                       hess_func=hessian,
                       clamp_magnitude=CLAMP_MAGNITUDE,
                       kappa=KAPPA, 
                       interval=BIAS_UPDATE_FREQ, 
                       stability_interval=STABILITY_CHECK_FREQ, 
                       logfile_prefix=LOG_PREFIX
                       )
    ```
4) Create a `GADESCalculator` object that adds GADES forces to those from the LAMMPS calculator:

    ``` python
    gades_calc = GADESCalculator(lammps_calc, force_bias)
    ```

5) Create an ASE `Atoms` object and an `ASEBackend` object. Inside the ASEBackend constructor, we use `gades_calc` for `atoms.calc`:

    ``` python
    atoms = bulk('Ar', 'fcc', a=5.26).repeat((2, 2, 2))
    ...
    backend = ASEBackend(gades_calc, atoms)
    ```

6) Set the backend for the force bias, because the `GADESBias` object needs to query the backend for atom forces, atom symbols and so on:

    ``` python
    force_bias.backend = backend
    ```

7) Create a time integrator object and let the backend aware of the integrator for step tracking:

    ``` python
    dyn = VelocityVerlet(atoms, timestep_fs * units.fs)
    backend.integrator = dyn
    ```

The example script `examples/ase/mdrun.py` demonstrates how to use GADES with the ASE backend.

## Implemention notes

The `GADESBias` class provides the generic implementation of the GADES bias applied to any MD engine, or backend, such as OpenMM and ASE.

The `GADESForceUpdater` class is derived from `GADESBias` and provides the functions (`describeNextReport()` and `report()`)required by the OpenMM Reporter API.

The parameters of the `GADESBias` constructor are:

* `backend`: The simulation engine to be driven by, or coupled with, `GADESBias`
* `biased_force`: The bias force associated with GADES
* `bias_atom_indices`: Indecies of the atoms to be biased. This option allows using block Hessians for the GADES calculations instead of the full Hessian. It also allow for a finer control of the which part of the system you want to bias.
* `hess_func`: The method for Hessian estimation. There are two methods available in `gades.util` module, both of which use finite difference to estimate the Hessian from OpenMM forces. We suggest using `compute_hessian_force_fd_richardson` which uses Richardson extrapolation for a more accurate estimation and less sensitivity to step size.
* `clamp_magnitude`: Maximum magnitude of the bias force applied to the system. Same units as OpenMM.
* `kappa`: $\kappa$ value in the GADES force formulation. We suggest setting it to 0.9 for the most effective exploration and control the biasing effects through `clamp_magnitude` instead.
* `interval`: Intervals at which the bias magnitude and direction gets updated. This also controls the frequency of Hessian calculation and diagonalization.
* `stability_interval`: Intervals at which stability checks are performed to ensure system stability. We suggest a setting this to `interval`//2 or smaller.
* `logfile_prefix`: Prefix for the log files created by GADES. `None` would skip logging the most negative eigenvalue, the corresponding eigenvector, and the current location on the potential energy surface at each update interval.


### Disclaimer

This repo is still under active development.
