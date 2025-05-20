Replace the ForceReporter with a loop to print eigne vectors and the simulation seems to crash. Can run as `python production.py`. Makes use of equilibrated `pdb` structures.

```
# Main simulation loop
n_steps = int(1e8)
gad_update_interval = 1000

n_vec_list = []

start = time.time()
for step in range(0, n_steps, gad_update_interval):    
    if step > 0:
        print(f"[step {step}] Updating GAD forces...")
        n_vec = update_biased_forces(simulation, biased_force, adp_atom_indices)
        n_vec_np = np.array(n_vec)  
        n_vec_list.append([step] + list(n_vec_np))  # save step + vector - first column is step
        np.savetxt('n_vec.dat',np.array(n_vec_list)) # first column in the step
    simulation.step(gad_update_interval)
end = time.time()

print("[production] GAD-biased simulation complete. Time elapsed (s):", end - start)
np.savetxt('time_gad_solvated.dat', np.array([end-start]))
```
