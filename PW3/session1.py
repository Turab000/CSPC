import pandas as pd

heart = pd.read_csv("heart.csv")

print(heart.shape)

print(heart.head())

print(heart.describe())

ax.hist(values, bins=18)

from scipy import stats

statistic, p = stats.shapiro(values)