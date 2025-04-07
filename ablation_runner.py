
import subprocess
import sys

ablation_variants = ["without_mp", "without_dyn", "without_dhr", "without_mes"]

for ablation in ablation_variants:
    print(f"\n=== Running ablation: {ablation} ===")
    subprocess.run([sys.executable, "ablation.py", "--ablation", ablation])
