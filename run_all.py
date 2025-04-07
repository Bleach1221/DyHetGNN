import os
import subprocess

tasks = [
    ("core", "python train.py"),
    ("baselines", "python baseline_runner.py"),
    ("ablation", "python ablation_runner.py")
]

print(" Starting Full Experiment Pipeline...\n")

for name, cmd in tasks:
    print(f"\n Running: {name.upper()}...")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f" Failed at {name}, exit code {result.returncode}")
        break

print("\n All experiments completed.")
