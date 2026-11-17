"""PW3 - Session 1: explore distributions and normality of the heart dataset."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

FOLDER = Path(__file__).resolve().parent
file = FOLDER / "heart.csv"
if not file.exists():
    raise FileNotFoundError("Put heart.csv in the same PW3 folder as session1.py")

# Task 1: load, inspect, and isolate the four continuous variables
heart = pd.read_csv(file)
columns = ["age", "chol", "trestbps", "thalach"]
missing = [name for name in columns + ["target"] if name not in heart.columns]
if missing:
    raise ValueError(f"Missing required columns in heart.csv: {missing}")

print("=== TASK 1: LOAD DATA ===")
print("Dataset shape (rows, columns):", heart.shape)
print("\nFirst five rows:\n", heart.head().to_string(index=False))
print("\nSummary statistics:\n", heart.describe().to_string())
print("\nMissing values:\n", heart[columns + ["target"]].isna().sum().to_string())

continuous = heart[columns].apply(pd.to_numeric, errors="coerce")

# Task 2: histograms of all four continuous variables
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, name in zip(axes.flat, columns):
    values = continuous[name].dropna()
    ax.hist(values, bins=18, edgecolor="black")
    ax.set_title(f"Distribution of {name}")
    ax.set_xlabel(name)
    ax.set_ylabel("Number of patients")
    ax.grid(alpha=0.2)
fig.suptitle("PW3 Session 1: Continuous Variable Distributions")
fig.tight_layout()
fig.savefig(FOLDER / "heart_histograms.png", dpi=160)
plt.close(fig)
print("\nCreated: heart_histograms.png")

# Task 3: Q-Q plots and Shapiro-Wilk tests
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
print("\n=== TASK 3: NORMALITY CHECK ===")
for ax, name in zip(axes.flat, columns):
    values = continuous[name].dropna().to_numpy()
    stats.probplot(values, dist="norm", plot=ax)
    ax.set_title(f"Normal Q-Q plot: {name}")
    if len(values) >= 3:
        statistic, p = stats.shapiro(values)
        print(f"{name:10s} n={len(values):3d} | Shapiro p={p:.5g} | ", end="")
        print("reject normality at 5%" if p < 0.05 else "normality NOT rejected at 5%")
    else:
        print(f"{name}: too few values to run Shapiro-Wilk")
fig.suptitle("PW3 Session 1: Normality Q-Q Plots")
fig.tight_layout()
fig.savefig(FOLDER / "heart_qq_plots.png", dpi=160)
plt.close(fig)
print("Created: heart_qq_plots.png")
print("\nImportant: p >= 0.05 does not prove that data are normal.")
print("Check histogram shape and Q-Q plots as well; write your own conclusion in README.")
