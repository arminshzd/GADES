gmx grompp -f nvt.mdp -c ../em/em.gro -r ../em/em.gro -p ../../topol.top -o nvt.tpr
gmx mdrun -deffnm nvt