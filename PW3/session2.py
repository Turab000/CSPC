sick = heart[heart["target"] == 1]["thalach"]

healthy = heart[heart["target"] == 0]["thalach"]

from scipy.stats import pearsonr

r, p = pearsonr(heart["age"], heart["thalach"])

from scipy.stats import pearsonr

r1, p1 = pearsonr(
    chemicals["benzene"],
    chemicals["malignancy"]
)

r2, p2 = pearsonr(
    chemicals["cadmium"],
    chemicals["malignancy"]
)

low = chemicals[
    chemicals["pollution_index"] < 40
]

high = chemicals[
    chemicals["pollution_index"] >= 60
]

heart["target"].value_counts(normalize=True)