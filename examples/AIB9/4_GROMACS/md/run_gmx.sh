## Needs a plumed installation with the PYTORCH_MODEL module enabled.
## Also needs the .ptc file from `export_srv.ipynb` to be in the same directory.
## See `plumed.dat`.

gmx grompp -f md.mdp -c ../npt/npt.gro -r ../npt/npt.gro -p ../../topol.top -o md.tpr
gmx mdrun -deffnm md -plumed plumed.dat
