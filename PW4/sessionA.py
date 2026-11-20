import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.inspection import permutation_importance
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
os.makedirs("results", exist_ok=True)


def rmse(a, b):
    return np.sqrt(mean_squared_error(a, b))


def adj_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


# ================= A2: clean and look =================
df = pd.read_csv("data/concrete.csv")
print("== A2 ==")
print("rows before:", len(df), "| duplicates:", df.duplicated().sum(),
      "| missing:", int(df.isna().sum().sum()))
df = df.drop_duplicates().reset_index(drop=True)
print("rows after:", len(df))
print(df.describe().round(2))
X = df.drop(columns="strength")
y = df["strength"]

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].scatter(X["cement"], y, s=8)
ax[0].set_xlabel("cement (kg/m3)")
ax[1].scatter(X["age"], y, s=8)
ax[1].set_xlabel("age (days)")
for a in ax:
    a.set_ylabel("strength (MPa)")
fig.tight_layout()
fig.savefig("results/A2_scatter.png", dpi=150)
plt.close(fig)

# ================= A3: split =================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\n== A3 == train:", len(X_train), "test:", len(X_test))

# ================= A4: linear regression =================
lin = LinearRegression().fit(X_train, y_train)
print("\n== A4 linear regression ==")
print("intercept (MPa): %.4f" % lin.intercept_)
print(pd.Series(lin.coef_, index=X.columns).round(5))
ptr, pte = lin.predict(X_train), lin.predict(X_test)
r2tr = r2_score(y_train, ptr)
A4 = pd.DataFrame([{
    "train_RMSE": rmse(y_train, ptr), "test_RMSE": rmse(y_test, pte),
    "test_MAE": mean_absolute_error(y_test, pte), "train_R2": r2tr,
    "train_adjR2": adj_r2(r2tr, len(X_train), X_train.shape[1]),
    "test_R2": r2_score(y_test, pte)}])
print(A4.round(3).to_string(index=False))
A4.round(4).to_csv("results/A4_linear.csv", index=False)

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(X_test["age"], y_test - pte, s=10)
ax.axhline(0, color="k", lw=1)
ax.set_xscale("log")
ax.set_xlabel("age (days, log scale)")
ax.set_ylabel("residual = true - predicted (MPa)")
fig.tight_layout()
fig.savefig("results/A4_residuals.png", dpi=150)
plt.close(fig)

# ================= A5: polynomial regression =================
rows = []
for d in [1, 2, 3, 4]:
    poly = make_pipeline(PolynomialFeatures(degree=d, include_bias=False),
                         StandardScaler(), LinearRegression())
    poly.fit(X_train, y_train)
    p = poly[0].n_output_features_
    ptr, pte = poly.predict(X_train), poly.predict(X_test)
    r2tr = r2_score(y_train, ptr)
    rows.append({"degree": d, "p": p, "train_RMSE": rmse(y_train, ptr),
                 "test_RMSE": rmse(y_test, pte), "train_R2": r2tr,
                 "train_adjR2": adj_r2(r2tr, len(X_train), p),
                 "test_R2": r2_score(y_test, pte)})
A5 = pd.DataFrame(rows)
print("\n== A5 polynomial regression (n_train = %d) ==" % len(X_train))
print(A5.round(3).to_string(index=False))
A5.round(4).to_csv("results/A5_polynomial.csv", index=False)

# ================= A6: classification =================
y_class = (df["strength"] >= 40).astype(int)
print("\n== A6 == class proportions:\n", y_class.value_counts(normalize=True).round(3))
Xc_train, Xc_test, c_train, c_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class)

# ---- KNN sweep ----
Ks = [1, 3, 5, 7, 15, 25, 51, 101]
rows = []
for K in Ks:
    m = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=K)).fit(Xc_train, c_train)
    rows.append({"K": K, "train_acc": m.score(Xc_train, c_train), "test_acc": m.score(Xc_test, c_test)})
knn_sweep = pd.DataFrame(rows)
print("\nKNN sweep (scaled):\n", knn_sweep.round(4).to_string(index=False))
knn_sweep.round(4).to_csv("results/A6_knn_sweep.csv", index=False)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(knn_sweep["K"], knn_sweep["train_acc"], "o-", label="train")
ax.plot(knn_sweep["K"], knn_sweep["test_acc"], "s-", label="test")
ax.set_xscale("log")
ax.set_xlabel("K")
ax.set_ylabel("accuracy")
ax.legend()
fig.tight_layout()
fig.savefig("results/A6_knn_sweep.png", dpi=150)
plt.close(fig)

K = int(knn_sweep.loc[knn_sweep["test_acc"].idxmax(), "K"])
print("chosen K (best test accuracy):", K)
knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=K)).fit(Xc_train, c_train)
knn_raw = KNeighborsClassifier(n_neighbors=K).fit(Xc_train, c_train)
print("test accuracy at K=%d: with scaler %.4f | without scaler %.4f"
      % (K, knn.score(Xc_test, c_test), knn_raw.score(Xc_test, c_test)))

accs = []
for rs in range(5):
    a_tr, a_te, b_tr, b_te = train_test_split(X, y_class, test_size=0.2, random_state=rs, stratify=y_class)
    m = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=K)).fit(a_tr, b_tr)
    accs.append(m.score(a_te, b_te))
print("five splits, test accuracy:", np.round(accs, 4), "| mean %.4f | std %.4f | range %.4f"
      % (np.mean(accs), np.std(accs, ddof=1), max(accs) - min(accs)))

# ---- Tree sweep ----
depths = [1, 2, 3, 4, 5, 6, 8, None]
rows = []
for d in depths:
    t = DecisionTreeClassifier(criterion="entropy", max_depth=d, random_state=0).fit(Xc_train, c_train)
    rows.append({"max_depth": "None" if d is None else d, "leaves": t.get_n_leaves(),
                 "actual_depth": t.get_depth(), "train_acc": t.score(Xc_train, c_train),
                 "test_acc": t.score(Xc_test, c_test)})
tree_sweep = pd.DataFrame(rows)
tree_sweep["gap"] = tree_sweep["train_acc"] - tree_sweep["test_acc"]
print("\nTree sweep:\n", tree_sweep.round(4).to_string(index=False))
tree_sweep.round(4).to_csv("results/A6_tree_sweep.csv", index=False)
fig, ax = plt.subplots(figsize=(6, 4))
xs = [d if d is not None else int(tree_sweep["actual_depth"].iloc[-1]) for d in depths]
ax.plot(xs, tree_sweep["train_acc"], "o-", label="train")
ax.plot(xs, tree_sweep["test_acc"], "s-", label="test")
ax.set_xlabel("max_depth (None plotted at its actual depth)")
ax.set_ylabel("accuracy")
ax.legend()
fig.tight_layout()
fig.savefig("results/A6_tree_sweep.png", dpi=150)
plt.close(fig)

# chosen depth: among trees whose train-test gap is still small (<= 0.06), the one with the best test accuracy
cand = tree_sweep[tree_sweep["gap"] <= 0.06]
depth_chosen = cand.loc[cand["test_acc"].idxmax(), "max_depth"]
depth_chosen = None if depth_chosen == "None" else int(depth_chosen)
print("chosen depth:", depth_chosen)
tree = DecisionTreeClassifier(criterion="entropy", max_depth=depth_chosen, random_state=0).fit(Xc_train, c_train)
print("\nTop of chosen tree:")
print(export_text(tree, feature_names=list(X.columns), max_depth=2))

# ================= A7: metrics =================
def metrics_row(name, y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {"model": name, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred), "f1": f1_score(y_true, y_pred)}


proba = knn.predict_proba(Xc_test)[:, 1]
rows = []
for t in [0.3, 0.5, 0.7, 0.9]:
    r = metrics_row("KNN K=%d" % K, c_test, (proba >= t).astype(int))
    r["threshold"] = t
    rows.append(r)
thr = pd.DataFrame(rows)[["threshold", "TP", "FP", "FN", "TN", "precision", "recall"]]
print("\n== A7 threshold table (KNN) ==\n", thr.round(3).to_string(index=False))
thr.round(4).to_csv("results/A7_thresholds.csv", index=False)

ok = thr[(thr["precision"] >= 0.95) & (thr["recall"] > 0)]
if len(ok):
    t_chosen = float(ok["threshold"].iloc[0])
    print("chosen threshold (lowest with precision >= 0.95 and recall > 0):", t_chosen)
else:  # fall back: best precision among thresholds that still find some positives
    t_chosen = float(thr[thr["recall"] > 0].sort_values(["precision", "recall"], ascending=False)["threshold"].iloc[0])
    print("chosen threshold (best precision with recall > 0):", t_chosen)

final = pd.DataFrame([
    metrics_row("KNN K=%d (t=0.5)" % K, c_test, knn.predict(Xc_test)),
    metrics_row("Tree depth=%s (t=0.5)" % depth_chosen, c_test, tree.predict(Xc_test)),
    metrics_row("KNN K=%d (t=%.1f)" % (K, t_chosen), c_test, (proba >= t_chosen).astype(int)),
])
print("\nFinal metrics:\n", final.round(3).to_string(index=False))
final.round(4).to_csv("results/A_metrics.csv", index=False)

# ================= A8: importance =================
reg = make_pipeline(StandardScaler(), LinearRegression()).fit(X_train, y_train)
imp = pd.DataFrame({
    "raw_coef": lin.coef_,
    "std_coef": reg[-1].coef_,
    "tree_importance": tree.feature_importances_,
}, index=X.columns)
pi_knn = permutation_importance(knn, Xc_test, c_test, scoring="f1", n_repeats=20, random_state=0)
pi_tree = permutation_importance(tree, Xc_test, c_test, scoring="f1", n_repeats=20, random_state=0)
imp["perm_knn"] = pi_knn.importances_mean
imp["perm_tree"] = pi_tree.importances_mean
print("\n== A8 importance ==\n", imp.round(4))
imp.round(5).to_csv("results/A8_importance.csv")
print("\nrank by |raw_coef|:", list(imp["raw_coef"].abs().sort_values(ascending=False).index))
print("rank by |std_coef|:", list(imp["std_coef"].abs().sort_values(ascending=False).index))
print("rank by tree importance:", list(imp["tree_importance"].sort_values(ascending=False).index))
print("rank by perm KNN:", list(imp["perm_knn"].sort_values(ascending=False).index))
print("rank by perm tree:", list(imp["perm_tree"].sort_values(ascending=False).index))

fig, ax = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
order = X.columns
ax[0].barh(order, imp.loc[order, "std_coef"]); ax[0].set_title("standardised coefficient (MPa / std)")
ax[1].barh(order, imp.loc[order, "tree_importance"]); ax[1].set_title("tree importance")
ax[2].barh(order, imp.loc[order, "perm_knn"], xerr=pi_knn.importances_std[[list(X.columns).index(c) for c in order]])
ax[2].set_title("permutation importance (KNN, F1)")
fig.tight_layout()
fig.savefig("results/A8_importance.png", dpi=150)
plt.close(fig)

# ================= Bonus: physics-chosen features =================
def feats(frame, use_log_age, use_wb):
    f = frame.copy()
    if use_log_age:
        f["age"] = np.log(f["age"])
        f = f.rename(columns={"age": "log_age"})
    if use_wb:
        f["w_b"] = f["water"] / (f["cement"] + f["slag"] + f["ash"])
        f = f.drop(columns="water")
    return f


rows = []
for name, la, wb in [("plain", False, False), ("log(age)", True, False), ("log(age) + w/b", True, True)]:
    m = LinearRegression().fit(feats(X_train, la, wb), y_train)
    p_te = m.predict(feats(X_test, la, wb))
    rows.append({"model": name, "p": feats(X_train, la, wb).shape[1],
                 "test_RMSE": rmse(y_test, p_te), "test_R2": r2_score(y_test, p_te)})
bonus = pd.DataFrame(rows)
print("\n== Bonus ==\n", bonus.round(3).to_string(index=False))
bonus.round(4).to_csv("results/A_bonus.csv", index=False)