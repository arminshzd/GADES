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

GADES is prepared as a drop-in plugin to the OpenMM molecular dynamics package. Find the API documentation and working examples on the [documentations](https://arminshzd.github.io/GADES/) website.

To use GADES:

1) Import the hessian calculation method and the Force biasing method `getGADESBiasForce`:

    ``` python
    from gades.utils import compute_hessian_force_fd_richardson as hessian
    from gades.gades import getGADESBiasForce
    ```

2) Create a `GAD_force` object and add it to your system

    ``` python
    GAD_force = getGADESBiasForce(system.getNumParticles())
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

The parameters of `GADESForceUpdater` are:

* `biased_force`: The bias force associated with GADES
* `bias_atom_indices`: Indecies of the atoms to be biased. This option allows using block Hessians for the GADES calculations instead of the full Hessian. It also allow for a finer control of the which part of the system you want to bias.
* `hess_func`: The method for Hessian estimation. There are two methods available in `gades.util` module, both of which use finite difference to estimate the Hessian from OpenMM forces. We suggest using `compute_hessian_force_fd_richardson` which uses Richardson extrapolation for a more accurate estimation and less sensitivity to step size.
* `clamp_magnitude`: Maximum magnitude of the bias force applied to the system. Same units as OpenMM.
* `kappa`: $\kappa$ value in the GADES force formulation. We suggest setting it to 0.9 for the most effective exploration and control the biasing effects through `clamp_magnitude` instead.
* `interval`: Intervals at which the bias magnitude and direction gets updated. This also controls the frequency of Hessian calculation and diagonalization.
* `stability_interval`: Intervals at which stability checks are performed to ensure system stability. We suggest a setting this to `interval`//2 or smaller.
* `logfile_prefix`: Prefix for the log files created by GADES. `None` would skip logging the most negative eigenvalue, the corresponding eigenvector, and the current location on the potential energy surface at each update interval.

There is an example script available in `examples/BluePrint/sys_example.py`.

### Disclaimer

This repo is still under active development.
