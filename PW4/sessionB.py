import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier

os.makedirs("results", exist_ok=True)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

# ---------------- B1 ----------------
wine = load_wine(as_frame=True)
X = wine.data
cultivar = wine.target          # hidden until B6
print(wine.DESCR[:1500])
d = X.describe().T[["mean", "std", "min", "max"]]
d["range"] = d["max"] - d["min"]
print("\n== B1 describe (sorted by std) ==")
print(d.sort_values("std", ascending=False).round(3))

# ---------------- B2: PCA on raw data ----------------
pca_raw = PCA().fit(X)
print("\n== B2 raw PCA ==")
print("explained variance ratio:", pca_raw.explained_variance_ratio_.round(4))
load_raw = pd.Series(pca_raw.components_[0], index=X.columns)
print("PC1 loadings (raw), sorted by |value|:")
print(load_raw.reindex(load_raw.abs().sort_values(ascending=False).index).round(4))

# ---------------- B3: scale, PCA ----------------
scaler = StandardScaler()
Z = scaler.fit_transform(X)
print("\nscaled mean:", Z.mean(axis=0).round(6)[:3], "... std:", Z.std(axis=0).round(6)[:3], "...")
pca = PCA().fit(Z)
evr = pca.explained_variance_ratio_
cum = np.cumsum(evr)
tab = pd.DataFrame({"PC": [f"PC{i+1}" for i in range(len(evr))], "ratio": evr, "cumulative": cum})
print("\n== B3 scaled PCA ==")
print(tab.round(4).to_string(index=False))
print("components for 90%:", int(np.argmax(cum >= 0.9) + 1), "| PC1+PC2:", round(cum[1], 4))

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(range(1, 14), evr, label="per component")
ax.plot(range(1, 14), cum, "o-", color="C1", label="cumulative")
ax.axhline(0.9, color="k", ls="--", lw=1)
ax.set_xlabel("component"); ax.set_ylabel("explained variance ratio"); ax.legend()
fig.tight_layout(); fig.savefig("results/B3_scree.png", dpi=150); plt.close(fig)

load = pd.DataFrame(pca.components_[:2].T, index=X.columns, columns=["PC1", "PC2"])
print("\nPC1 loadings sorted:\n", load["PC1"].sort_values().round(3))
print("\nPC2 loadings sorted:\n", load["PC2"].sort_values().round(3))
load.round(4).to_csv("results/B3_loadings.csv")

S = pca.transform(Z)
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(S[:, 0], S[:, 1], s=14, color="gray")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
fig.tight_layout(); fig.savefig("results/B3_pc1pc2.png", dpi=150); plt.close(fig)

# ---------------- B4: K-Means ----------------
print("\n== B4a: k=3, n_init=1, seeds 0..7 ==")
for rs in range(8):
    k1 = KMeans(n_clusters=3, init="k-means++", n_init=1, max_iter=300, random_state=rs).fit(Z)
    print(rs, round(k1.inertia_, 3))
ine = []
for k in range(1, 11):
    ine.append(KMeans(n_clusters=k, n_init=10, random_state=0).fit(Z).inertia_)
print("\n== B4b inertia vs k ==")
for k, v in zip(range(1, 11), ine):
    print(k, round(v, 2))
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(range(1, 11), ine, "o-"); ax.set_xlabel("k"); ax.set_ylabel("inertia")
fig.tight_layout(); fig.savefig("results/B4_elbow.png", dpi=150); plt.close(fig)

# ---------------- B5 ----------------
km = KMeans(n_clusters=3, init="k-means++", n_init=10, max_iter=300, random_state=0).fit(Z)
print("\n== B5 ==")
print("cluster sizes:", np.bincount(km.labels_))
centres = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=X.columns)
print(centres.round(2).T)
centres.round(3).to_csv("results/B5_centres.csv")
fig, ax = plt.subplots(figsize=(5.5, 5))
sc = ax.scatter(S[:, 0], S[:, 1], c=km.labels_, s=14, cmap="viridis")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
fig.tight_layout(); fig.savefig("results/B5_clusters.png", dpi=150); plt.close(fig)

# ---------------- B6 ----------------
def agreement(labels):
    t = pd.crosstab(cultivar, labels)
    return t, t.max(axis=0).sum() / len(cultivar)

res = {}
t1, res["scaled data"] = agreement(KMeans(3, n_init=10, random_state=0).fit(Z).labels_)
t2, res["raw data"] = agreement(KMeans(3, n_init=10, random_state=0).fit(X).labels_)
t3, res["first 2 PCs (scaled)"] = agreement(KMeans(3, n_init=10, random_state=0).fit(S[:, :2]).labels_)
print("\n== B6 ==")
for name, t in [("scaled", t1), ("raw", t2), ("PC1-PC2", t3)]:
    print(f"\ncrosstab ({name}):\n{t}")
agr = pd.DataFrame({"version": list(res), "agreement": [round(v, 4) for v in res.values()]})
print("\n", agr.to_string(index=False))
agr.to_csv("results/B_agreement.csv", index=False)

# ---------------- Bonus ----------------
Xtr, Xte, ytr, yte = train_test_split(X, cultivar, test_size=0.25, random_state=42, stratify=cultivar)
m1 = make_pipeline(StandardScaler(), KNeighborsClassifier(5)).fit(Xtr, ytr)
m2 = make_pipeline(StandardScaler(), PCA(n_components=2), KNeighborsClassifier(5)).fit(Xtr, ytr)
print("\n== Bonus ==")
print("KNN scaled, 13 features:", round(m1.score(Xte, yte), 4))
print("KNN scaled + PCA(2):    ", round(m2.score(Xte, yte), 4))