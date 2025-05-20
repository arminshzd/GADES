# initialize.py — create solvated ADP system
from openmm.app import *
from openmm import unit
from sys import stdout

proj_path = '/home/siddarthachar/box-uchi/UChicago-notes/GAD/'
openmm_app_path = '/home/siddarthachar/miniconda3/envs/dmff/lib/python3.9/site-packages/openmm/app/data/'

pdb = PDBFile(f"{proj_path}/adp/alanine-dipeptide.pdb")
modeller = Modeller(pdb.topology, pdb.positions)
forcefield = ForceField(f"{openmm_app_path}/amber14/protein.ff14SB.xml", f"{openmm_app_path}/amber14/tip3p.xml")
modeller.addSolvent(forcefield, model="tip3p", padding=0.8 * unit.nanometer)

with open("init.pdb", "w") as outfile:
    PDBFile.writeFile(modeller.topology, modeller.positions, outfile)

print("Solvated system written to init.pdb")
