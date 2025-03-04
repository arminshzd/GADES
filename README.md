# GADES
Gentlest Ascent Dynamics for Enhanced Sampling

## TODO list:
- [x] Applying GAD to the Muller-Brown system with BAOAB integrator
* Explored the effect of $\gamma$ on the discovered MEP between the two states
* Explored the application of the inverse iteration method for finding the dominant eigenvalue/vector of the Hessian.
* Explored the accuracy of numerical approximation for calculating the Hessian from forces.
- [ ] Setting up the AMBER forcefield with and automatic diffrentiation backend
* Set up an ADP system using [DMFF](https://github.com/deepmodeling/DMFF/tree/master) and OpenMM
* Calculate the full Hessian using AD and running a GAD-biased trajectory to explore the minima
* Compare the full Hessian vs. backbone-only Hessian performance for GADES
- [ ] Replace the full Hessian calculation with partial Hessian calculations using Lanczos and JVP
* Use [Lanczos method](https://en.wikipedia.org/wiki/Lanczos_algorithm) for finding the dominant eigenvalue/vector together with [JVP](https://iclr-blogposts.github.io/2024/blog/bench-hvp/) to circumvent full Hessian calculations
* Compare efficiency and accuracy with the full Hessian results
- [ ] Run an unbiased and a GADES biased trajectory using OpenMM to probe exploration (directly manipulating the forces; similar to the Muller-Brown case)
- [ ] Apply GADES to a larger system
- [ ] Create path collective variables from GADES and run OpenMM+PLUMED simulations
       
