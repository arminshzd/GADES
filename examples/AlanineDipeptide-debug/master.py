# master.py — run full GAD simulation pipeline step-by-step
import subprocess

# print("=== Step 1: Initialization (solvation) ===")
# subprocess.run(["python", "initialize.py"])

# print("=== Step 2: Energy Minimization ===")
# subprocess.run(["python", "minimize.py"])

# print("=== Step 3: Equilibration ===")
# subprocess.run(["python", "equilibrate.py"])

print("=== Step 4: Production with GAD Forces ===")
subprocess.run(["python", "production.py"])

print("=== GAD Simulation Complete ===")
