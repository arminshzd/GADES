gmx grompp -f em.mdp -c ../../aib9.gro -p ../../topol.top -o em.tpr -maxwarn 1
gmx mdrun -deffnm em