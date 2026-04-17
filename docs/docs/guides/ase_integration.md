# ASE Integration Guide

This guide explains how to use GADES with the Atomic Simulation Environment (ASE) and covers the architecture decisions behind the integration.

## Quick Start

The simplest way to use GADES with ASE is via the `with_gades` factory method:

```python
from ase import Atoms
from ase.calculators.lammpsrun import LAMMPS
from ase.md.verlet import VelocityVerlet

from GADES.backend import ASEBackend
from GADES.utils import compute_hessian_force_fd_richardson as hessian

# Set up your atoms and calculator
atoms = bulk('Ar', 'fcc', a=5.26).repeat((2, 2, 2))
lammps_calc = LAMMPS(**parameters)

# Atoms to bias (e.g., all Ar atoms)
biasing_atom_ids = [atom.index for atom in atoms if atom.symbol == 'Ar']

# Create backend with GADES bias in one step
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

# Set up MD and attach integrator
dyn = VelocityVerlet(atoms, 5 * units.fs)
backend.integrator = dyn

# Run simulation
dyn.run(1000)
```

## The Circular Dependency Problem

### Why Does It Exist?

The GADES-ASE integration involves three tightly coupled components that have mutual dependencies:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Dependency Chain                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  GADESBias ──needs──▶ Backend (to get forces/positions)         │
│      │                    ▲                                     │
│      │                    │                                     │
│      ▼                    │                                     │
│  GADESCalculator ◀──creates── ASEBackend                        │
│      │                                                          │
│      │                                                          │
│      ▼                                                          │
│  (wraps base calculator and calls GADESBias.get_gad_force())    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**The circular dependency arises because:**

1. **GADESBias** needs a reference to `Backend` to:
   - Query current atomic positions via `backend.get_current_state()`
   - Compute forces at perturbed positions via `backend.get_forces()`
   - Get atom symbols for logging via `backend.get_atom_symbols()`
   - Check simulation stability via `backend.is_stable()`

2. **GADESCalculator** needs a reference to `GADESBias` to:
   - Call `gades_bias.get_gad_force()` during force computation
   - Check if bias should be applied via `gades_bias.applying_bias()`

3. **ASEBackend** needs a reference to `GADESCalculator` to:
   - Access the base calculator for unbiased forces
   - Manage the atoms-calculator relationship

This creates a chicken-and-egg problem: you can't fully initialize any component without the others already existing.

### The Old 4-Step Workaround

Before the `with_gades` factory method, users had to manually wire the components with a post-initialization patch:

```python
from GADES import GADESBias
from GADES.backend import ASEBackend, GADESCalculator
from GADES.utils import compute_hessian_force_fd_richardson as hessian

# Step 1: Create GADESBias with backend=None (incomplete initialization!)
force_bias = GADESBias(
    backend=None,                    # ⚠️ Cannot provide backend yet
    biased_force=None,
    bias_atom_indices=biasing_atom_ids,
    hess_func=hessian,
    clamp_magnitude=1000,
    kappa=0.9,
    interval=100,
    stability_interval=1000,
)

# Step 2: Create GADESCalculator with the GADESBias
gades_calc = GADESCalculator(lammps_calc, force_bias)

# Step 3: Create ASEBackend with the GADESCalculator
backend = ASEBackend(gades_calc, atoms)

# Step 4: Patch the backend reference back into GADESBias
force_bias.backend = backend         # ⚠️ Manual post-initialization patching!
```

**Problems with this approach:**

- **Error-prone**: Forgetting step 4 leads to `AttributeError` or `NoneType` errors at runtime
- **Non-obvious**: New users don't expect to need post-initialization patching
- **Violates encapsulation**: Internal wiring details are exposed to users
- **Hard to document**: The pattern is unusual and confusing

### The Solution: Factory Method Pattern

The `with_gades` factory method encapsulates all the wiring logic:

```python
from GADES.backend import ASEBackend
from GADES.utils import compute_hessian_force_fd_richardson as hessian

# Single step: All wiring handled internally
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

# Access GADESBias if needed
print(f"Kappa: {backend.gades_bias.kappa}")
```

**How it works internally:**

```python
@classmethod
def with_gades(cls, atoms, base_calc, bias_atom_indices, ...):
    # Import here to avoid circular import at module load time
    from .gades import GADESBias

    # Step 1: Create GADESBias with backend=None
    gades_bias = GADESBias(backend=None, biased_force=None, ...)

    # Step 2: Create GADESCalculator
    gades_calc = GADESCalculator(base_calc, gades_bias)

    # Step 3: Create ASEBackend
    backend = cls(gades_calc, atoms, ...)

    # Step 4: Wire up the circular reference
    gades_bias.backend = backend

    # Step 5: Store reference for user access
    backend.gades_bias = gades_bias

    return backend
```

**Benefits:**

- **Single entry point**: One method call creates a fully configured system
- **Encapsulated complexity**: Users don't need to understand the internal wiring
- **Type-safe**: Proper type hints without circular import issues (uses `TYPE_CHECKING`)
- **Accessible**: The `GADESBias` instance is available via `backend.gades_bias` if needed

## Avoiding Circular Imports

The implementation uses Python's `TYPE_CHECKING` constant to provide type hints without runtime circular imports:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .gades import GADESBias  # Only imported during type checking

class ASEBackend:
    gades_bias: Optional["GADESBias"]  # Forward reference as string

    @classmethod
    def with_gades(cls, ...):
        # Import at runtime inside the method
        from .gades import GADESBias
        ...
```

This pattern:

1. **At type-checking time** (e.g., mypy, IDE): The import happens, enabling autocomplete and type validation
2. **At runtime**: The import is deferred to when `with_gades` is actually called, avoiding circular import errors

## Advanced Usage

### Accessing GADESBias Parameters

After creating the backend, you can access and modify GADESBias parameters:

```python
backend = ASEBackend.with_gades(...)

# Read parameters
print(f"Current kappa: {backend.gades_bias.kappa}")
print(f"Eigensolver: {backend.gades_bias.eigensolver}")

# Modify parameters
backend.gades_bias.set_kappa(0.8)
backend.gades_bias.set_hess_step_size(0.001)
```

### Using Optional Features

The factory method supports all GADESBias options:

```python
backend = ASEBackend.with_gades(
    atoms=atoms,
    base_calc=base_calc,
    bias_atom_indices=indices,
    hess_func=hessian,
    clamp_magnitude=1000,
    kappa=0.9,
    interval=100,
    # Optional parameters:
    stability_interval=500,           # Check stability every 500 steps
    logfile_prefix="simulation",      # Write logs to simulation_*.log
    eigensolver="lanczos",            # Use Lanczos instead of full eigendecomposition
    lanczos_iterations=20,            # Number of Lanczos iterations
    use_bofill_update=True,           # Use Bofill Hessian approximation
    full_hessian_interval=50,         # Recompute full Hessian every 50 bias updates
    target_temperature=300.0,         # For stability checking
)
```

### Manual Initialization (Legacy)

The manual 4-step pattern is still supported for advanced use cases where you need more control:

```python
from GADES import GADESBias
from GADES.backend import ASEBackend, GADESCalculator

# Create components manually
force_bias = GADESBias(backend=None, biased_force=None, ...)
gades_calc = GADESCalculator(base_calc, force_bias)
backend = ASEBackend(gades_calc, atoms)
force_bias.backend = backend  # Don't forget this!
```

!!! warning "Remember to wire the backend"
    If using manual initialization, you **must** set `force_bias.backend = backend`
    after creating the ASEBackend, or GADES will fail at runtime.

## Comparison with OpenMM Backend

The OpenMM backend doesn't have the same circular dependency issue because OpenMM uses a different force application mechanism:

| Aspect | ASE Backend | OpenMM Backend |
|--------|-------------|----------------|
| Force application | Via GADESCalculator wrapper | Via CustomExternalForce |
| Bias storage | In calculator results | In OpenMM context |
| Circular dependency | Yes (solved by factory) | No |
| Recommended pattern | `ASEBackend.with_gades()` | Direct `OpenMMBackend()` |

## Summary

- Use `ASEBackend.with_gades()` for the simplest and safest initialization
- The factory method handles all internal wiring automatically
- Access `backend.gades_bias` to read or modify GADES parameters
- The circular dependency is a consequence of tight integration between force computation and bias application
