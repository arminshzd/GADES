# Guides

In-depth guides for using GADES with different simulation backends and advanced features.

## Backend Integration

- [OpenMM Integration](openmm_integration.md) - Using GADES with OpenMM
- [ASE Integration](ase_integration.md) - Using GADES with the Atomic Simulation Environment

## Performance & Scaling

- [Scaling to Large Systems](large_systems.md) - Matrix-free Lanczos for 1000+ atom systems

## Topics Covered

### OpenMM Integration Guide

The OpenMM integration guide covers:

- Quick start with `GADESForceUpdater` as an OpenMM reporter
- How `createGADESBiasForce` works and why force groups matter
- Setup constraints (force must be added before `Simulation`)
- Advanced options: eigensolver, Bofill update, stability checking
- Comparison with the ASE backend

### ASE Integration Guide

The ASE integration guide covers:

- Quick start with the `with_gades` factory method
- Understanding the circular dependency architecture
- Comparison of initialization patterns (factory vs. manual)
- Advanced usage and optional features
- Comparison with OpenMM backend

### Large Systems Guide

The large systems guide covers:

- Understanding the scaling challenge (O(N²) memory, O(N³) time)
- Matrix-free solution with Hessian-vector products
- Eigensolver options comparison (`numpy`, `lanczos`, `lanczos_hvp`)
- Parameter tuning for large systems
- Benchmarking and troubleshooting
