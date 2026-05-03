# Examples

The examples below progress from a simple 2D analytical system to a production protein simulation, covering both the OpenMM and ASE backends.

| Example | System | Backend | Highlights |
|---|---|---|---|
| [Müller-Brown](MB_pot.md) | 2D analytical potential | Pure NumPy / PyTorch | Full workflow: GADES → minima detection → path metadynamics → FES |
| [AIB9](AIB9.md) | Aib₉ peptide in explicit solvent | OpenMM + GROMACS/PLUMED | GADES → SNRV CV learning → metadynamics → FES reweighting |
| [Argon Crystal](ase.md) | Ar FCC supercell | ASE + LAMMPS | `ASEBackend.with_gades()` factory pattern |
| [OpenMM Templates](protein.md) | Placeholder PDB | OpenMM | Reference templates showing correct GADES setup structure |
