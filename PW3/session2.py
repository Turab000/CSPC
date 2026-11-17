"""PW3 - Session 2: group tests, correlations, confounding, and bonus entropy."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

FOLDER = Path(__file__).resolve().parent


def load_csv(name, required):
    path = FOLDER / name
    if not path.exists():
        raise FileNotFoundError(f"Put {name} in the same PW3 folder as session2.py")
    data = pd.read_csv(path)
    absent = [column for column in required if column not in data.columns]
    if absent:
        raise ValueError(f"{name} is missing columns: {absent}")
    for column in required:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.dropna(subset=required)


def correlation(data, x, y):
    """Pearson r, p-value (linear association)."""
    if len(data) < 3 or data[x].nunique() < 2 or data[y].nunique() < 2:
        return float("nan"), float("nan")
    result = stats.pearsonr(data[x], data[y])
    return float(result.statistic), float(result.pvalue)


# Task 4: compare heart rates in patients with target 1 vs 0
heart = load_csv("heart.csv", ["age", "thalach", "target"])
sick = heart.loc[heart["target"] == 1, "thalach"].to_numpy()
healthy = heart.loc[heart["target"] == 0, "thalach"].to_numpy()
if min(len(sick), len(healthy)) < 3:
    raise ValueError("Need at least 3 patients in each target group")

print("=== TASK 4: HEART RATE, DISEASE vs HEALTHY ===")
for label, group in [("Disease (target=1)", sick), ("Healthy (target=0)", healthy)]:
    mean = np.mean(group)
    sd = np.std(group, ddof=1)
    sem = stats.sem(group)
    ci_half = stats.t.ppf(0.975, df=len(group)-1) * sem
    shapiro_p = stats.shapiro(group).pvalue if len(group) <= 5000 else stats.shapiro(group[:5000]).pvalue
    print(f"{label}: n={len(group)}, mean={mean:.3f}, SD={sd:.3f}, "
          f"95% CI=[{mean-ci_half:.3f}, {mean+ci_half:.3f}], Shapiro p={shapiro_p:.4g}")

p_sick = stats.shapiro(sick).pvalue if len(sick) <= 5000 else stats.shapiro(sick[:5000]).pvalue
p_healthy = stats.shapiro(healthy).pvalue if len(healthy) <= 5000 else stats.shapiro(healthy[:5000]).pvalue
if p_sick >= 0.05 and p_healthy >= 0.05:
    result = stats.ttest_ind(sick, healthy, equal_var=False)
    print(f"Group comparison: Welch t-test; p={result.pvalue:.6g}")
else:
    result = stats.mannwhitneyu(sick, healthy, alternative="two-sided")
    print(f"Group comparison: Mann-Whitney U test; p={result.pvalue:.6g}")
    print("Mann-Whitney tests rank/distribution differences, not exactly mean equality.")
print("p<0.05 suggests evidence of a difference, but does not measure effect size.")

labels = ["Healthy (0)", "Disease (1)"]
groups = [healthy, sick]
means = [np.mean(g) for g in groups]
ci_errors = [stats.t.ppf(0.975, len(g)-1) * stats.sem(g) for g in groups]
fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(labels, means, yerr=ci_errors, fmt="o", capsize=7)
ax.set_ylabel("Mean thalach (max heart rate)")
ax.set_title("Thalach by Target: Mean and 95% Confidence Interval")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FOLDER / "heart_group_comparison.png", dpi=160)
plt.close(fig)
print("Created: heart_group_comparison.png")

# Task 5: age-thalach relationship
print("\n=== TASK 5: AGE and THALACH ===")
r, p = correlation(heart, "age", "thalach")
spearman = stats.spearmanr(heart["age"], heart["thalach"])
print(f"Pearson r={r:.4f}, p={p:.4g}")
print(f"Spearman rho={spearman.statistic:.4f}, p={spearman.pvalue:.4g}")
print("Interpret the sign (positive/negative) and size; correlation is not causation.")
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(heart["age"], heart["thalach"], alpha=0.55)
ax.set_xlabel("Age (years)")
ax.set_ylabel("Thalach (maximum heart rate)")
ax.set_title("Relationship Between Age and Thalach")
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(FOLDER / "heart_age_thalach.png", dpi=160)
plt.close(fig)
print("Created: heart_age_thalach.png")

# Task 6: naive association of chemicals with malignancy
cols = ["benzene", "cadmium", "pollution_index", "age", "malignancy"]
chem = load_csv("chemicals_cancer.csv", cols)
print("\n=== TASK 6: NAIVE CHEMICAL ANALYSIS ===")
naive = {}
for chemical in ["benzene", "cadmium"]:
    r, p = correlation(chem, chemical, "malignancy")
    naive[chemical] = r
    print(f"{chemical}: Pearson r={r:.4f}, p={p:.4g} (n={len(chem)})")
stronger = max(naive, key=lambda name: abs(naive[name]))
print(f"Stronger NAIVE linear association: {stronger} (by |Pearson r|)")
print("This does NOT establish a causal relationship.")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, chemical in zip(axes, ["benzene", "cadmium"]):
    ax.scatter(chem[chemical], chem["malignancy"], alpha=0.4)
    ax.set_xlabel(chemical)
    ax.set_ylabel("Malignancy score")
    ax.set_title(f"Naive association: {chemical}")
    ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(FOLDER / "chemicals_naive.png", dpi=160)
plt.close(fig)
print("Created: chemicals_naive.png")

# Task 7: repeat analysis within pollution strata
print("\n=== TASK 7: ADJUST FOR POLLUTION BY STRATIFYING ===")
strata = {
    "Low pollution (<40)": chem[chem["pollution_index"] < 40],
    "High pollution (>=60)": chem[chem["pollution_index"] >= 60],
}
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for row, (label, part) in enumerate(strata.items()):
    print(f"\n{label}, n={len(part)}")
    for col, chemical in enumerate(["benzene", "cadmium"]):
        r, p = correlation(part, chemical, "malignancy")
        print(f"  {chemical}: Pearson r={r:.4f}, p={p:.4g}")
        ax = axes[row, col]
        ax.scatter(part[chemical], part["malignancy"], alpha=0.5)
        ax.set_xlabel(chemical)
        ax.set_ylabel("Malignancy score")
        ax.set_title(f"{label}: {chemical} (r={r:.2f})")
        ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(FOLDER / "chemicals_stratified.png", dpi=160)
plt.close(fig)
print("\nCreated: chemicals_stratified.png")
print("Compare naive correlations with correlations within each pollution stratum.")
print("If a chemical's naive link becomes weak, pollution may be a confounder.")
print("Observational associations alone cannot prove which chemical causes cancer.")

# Optional extra check: remove linear effects of pollution_index and age from both variables.
def residualize(y, control_matrix):
    matrix = np.column_stack([np.ones(len(control_matrix)), control_matrix])
    coefficients = np.linalg.lstsq(matrix, y, rcond=None)[0]
    return y - matrix @ coefficients

controls = chem[["pollution_index", "age"]].to_numpy()
y_residual = residualize(chem["malignancy"].to_numpy(), controls)
print("\nOptional: partial Pearson r controlling pollution_index and age")
for chemical in ["benzene", "cadmium"]:
    x_residual = residualize(chem[chemical].to_numpy(), controls)
    if np.std(x_residual) == 0 or np.std(y_residual) == 0:
        print(chemical, ": not defined (zero residual variance)")
    else:
        r = stats.pearsonr(x_residual, y_residual).statistic
        print(f"  {chemical}: partial r={r:.4f}")

# Bonus: quantify imbalance/uncertainty of the binary target
print("\n=== BONUS: CATEGORICAL BALANCE ===")
proportions = heart["target"].value_counts(normalize=True).sort_index()
entropy_bits = -sum(p * np.log2(p) for p in proportions if p > 0)
print("Target proportions:")
print(proportions.to_string())
print(f"Shannon entropy = {entropy_bits:.4f} bits")
print("For a binary target, H=1 bit is balanced; H near 0 is highly imbalanced.")
