gmx grompp -f npt.mdp -c ../nvt/nvt.gro -r ../nvt/nvt.gro -t ../nvt/nvt.cpt -p ../../topol.top -o npt.tpr
gmx mdrun -deffnm npt