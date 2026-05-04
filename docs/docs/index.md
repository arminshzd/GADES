![GADES](imgs/GADES_logo_b.png)

# GADES: Gentlest Ascent Dynamics for Enhanced Sampling

GADES is an enhanced sampling method based on Gentlest Ascent Dynamics (GAD) for exploring molecular configuration space without prior knowledge of reaction coordinates.

## Installation

to install, clone the repository

``` bash
git clone https://github.com/arminshzd/GADES.git
```

create a conda environment `(python >= 3.10)`

```bash
conda create -n GADES python=3.10 
conda activate GADES
```

and install from inside the `GADES` directory using `pip`

``` bash
pip install -e .
```

## Development

To set up a development environment with test dependencies:

```bash
pip install -e ".[dev]"
```

To run the test suite:

```bash
pytest
```

To run tests with coverage:

```bash
pytest --cov=GADES --cov-report=term-missing
```

!!! note "Under active development"
    GADES is still under active development. APIs may change between releases.
