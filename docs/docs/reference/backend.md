# Backend API Reference

The backend module provides interfaces for integrating GADES with different molecular dynamics engines.

## Overview

GADES supports multiple simulation backends:

- **OpenMMBackend**: For OpenMM-based simulations
- **ASEBackend**: For ASE (Atomic Simulation Environment) based simulations

## ASE Backend

The ASE backend allows GADES to work with any calculator supported by ASE, including LAMMPS, VASP, Quantum ESPRESSO, and many others.

### Quick Start with `with_gades` Factory Method

The recommended way to create an ASE backend with GADES bias is using the `with_gades` factory method:

```python
from GADES.backend import ASEBackend
from GADES.utils import compute_hessian_force_fd_richardson as hessian

backend = ASEBackend.with_gades(
    atoms=atoms,
    base_calc=lammps_calc,
    bias_atom_indices=biasing_atom_ids,
    hess_func=hessian,
    clamp_magnitude=1000,
    kappa=0.9,
    interval=100,
    stability_interval=1000,
)

# Attach integrator for step tracking
backend.integrator = dyn

# Access GADESBias if needed
print(backend.gades_bias.kappa)
```

### Understanding the Architecture

The ASE integration involves three interconnected components:

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────┐
│   GADESBias     │────▶│  GADESCalculator │────▶│  ASEBackend │
│                 │◀────│                  │◀────│             │
│ Computes bias   │     │ Wraps base calc  │     │ Manages     │
│ forces along    │     │ and adds GADES   │     │ atoms and   │
│ softest mode    │     │ bias to forces   │     │ state       │
└─────────────────┘     └──────────────────┘     └─────────────┘
        │                                               │
        └───────────────────────────────────────────────┘
                    Circular reference
```

This creates a **circular dependency** at initialization time:

1. `GADESBias` needs a `Backend` to query forces and positions
2. `GADESCalculator` needs a `GADESBias` to compute bias forces
3. `ASEBackend` needs a `GADESCalculator` to wrap

The `with_gades` factory method solves this by handling the wiring internally.

## API Reference

::: GADES.backend
    options:
      members:
        - Backend
        - OpenMMBackend
        - ASEBackend
        - GADESCalculator
