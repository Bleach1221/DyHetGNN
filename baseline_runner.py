
from baselines import train_baseline

if __name__ == "__main__":
    models = ["lstm", "gcn", "hgt", "tgcn", "dhgnn"]
    for model in models:
        print(f"\n========== Running: {model.upper()} ==========")
        train_baseline(model)
