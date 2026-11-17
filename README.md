# CSPC - Computer Science for Physics and Chemistry

My coursework repository for the Computer Science for Physics and Chemistry course.

Each practical work is organized under its corresponding `PW<n>/Lab <X>/` folder.

---# CSPC - Computer Science for Physics and Chemistry

My coursework repository. Each practical is under PW<n>/Lab <X>/.

## Setup

Create the environment for a given lab:

    conda env create -f "PW1/Lab A/environment.yml"
    conda activate cspc

---

## PW1 - Lab A: Reproducible Foundations

**What I built:**
- I created the CSPC repository with the PW1/Lab A structure and a reproducible Conda environment using Python 3.11, NumPy and pytest.
- I used Git and GitHub for version control, practised branching and merging, added the radioactive decay simulation, automated tests and a speed comparison script.

**Speed comparison (loop vs NumPy):**
- loop : 2.499994 s
- numpy : 0.000432 s
- speed-up: 5780.69 x faster

**Tests:** all passing? yes
- 4 tests passed successfully.

**Conclusion:**
- I learned how to create and manage a reproducible Python project using Conda, Git and GitHub.
- I learned how to use Git branches and pytest for automated testing, and how to compare a stochastic radioactive decay simulation with the theoretical exponential decay model.
- The NumPy implementation was significantly faster than the pure-Python loop implementation.

## PW1 - Lab B: Data, Plotting and Automation

**What I built:**
- I created the PW1/Lab B structure and added the observed decay dataset and plotting script.
- I used NumPy to read the `decay_observed.csv` file and extracted the time and observed count values.
- I calculated the analytical exponential decay curve using the model \(N(t)=N_0e^{-\lambda t}\) with \(\lambda = 0.3\).
- I used Matplotlib to create a 1x2 figure comparing the observed data with the analytical decay curve.
- I created a Snakemake pipeline to automate the generation of `figure.png`.

**Observed data vs analytical model:**
- The observed count decreases with time and follows a clear exponential decay trend.
- The observed data and the analytical curve show very similar behaviour on the same scale.
- The analytical model provides a good representation of the observed decay data.

**Snakemake:**
- Input: `decay_observed.csv`
- Output: `figure.png`
- Command used: `python plot.py`
- The pipeline rebuilds `figure.png` when the input changes or when the output file is missing.
- If nothing has changed, Snakemake detects that the output is already up to date and does not rerun the task.

**Conclusion:**
- I learned how to read experimental data from a CSV file using NumPy.
- I learned how to compare observed data with an analytical mathematical model using Matplotlib.
- I learned how to automate a simple data-processing workflow using Snakemake.
- The observed data showed good agreement with the expected exponential decay law.

## PW2 — Lab A: Motion from Tracking Data

The position data from `freefall.csv` was used to calculate velocity and acceleration using `np.gradient()`.

### Results
- **Mean acceleration: -8.5797 m/s²**
- **Acceleration standard deviation: 28.7161 m/s²**
- **Theoretical acceleration: -9.81 m/s²**
- **Sampling interval: 0.1 s**

The acceleration is very noisy because numerical differentiation amplifies small measurement errors. Since acceleration is calculated after differentiating twice, the noise becomes much larger.

The noisy acceleration was integrated back to velocity and position using `cumulative_trapezoid()`. The recovered position stayed close to the original data, showing that **integration suppresses noise while differentiation amplifies it**.

The final position, velocity and acceleration plots are saved in **`motion.png`**.


# PW2 — Lab B: Optimization in Chemistry

## Part 2: Three optimization methods

We compared gradient descent, Newton's method,
and SLSQP.

For f(x) = (x-3)^2 + 1, all three methods
found the minimum at x = 3.

For g(x) = x^4 - 3x^2 + x + 5, the results
depend on the starting point and the method.

Newton's method finds stationary points,
which are not necessarily minima. Starting
at x = 0, it converges to a local maximum.

Gradient descent and SLSQP search for minima.

## Part 3: Reaction kinetics

We fitted the model C(t) = C0 * exp(-k*t)
to the experimental data.

Fitted rate constant: k = 0.2618

The fitted curve follows the measured data.

Plot: kinetics.png

## Part 4: Chemical equilibrium

Reaction: H2 + I2 <-> 2HI
Equilibrium constant: K = 15.6

Newton's method and SLSQP give approximately
the same equilibrium extent:

x = 0.66385

Equilibrium amounts:
H2 = 0.33615 mol
I2 = 0.33615 mol
HI = 1.32770 mol

Plot: equilibrium.png

## Part 5: Titration

We calculated the slope of the pH curve
using np.gradient().

The maximum slope occurs at:

Equivalence point = 50.0 mL

Plot: titration.png

# PW3 — Data, Distributions, Testing, and Causation
Course: Computer Science for Physics and Chemistry
Practical work: PW3 (two sessions)
Tools: Python, pandas, NumPy, SciPy, Matplotlib
Aim
The aim of this practical work is to explore and describe real data, examine whether variables follow a normal distribution, compare two groups, measure correlations, and investigate how a confounding variable can make an association misleading. The analysis uses the Cleveland heart disease dataset and a synthetic chemical-exposure dataset.

Data sources: The heart disease data are the Cleveland dataset supplied for the lab (based on the UCI Heart Disease dataset; also distributed through Kaggle). chemicals_cancer.csv is the synthetic dataset supplied for PW3 on Moodle. All numerical results below were obtained from the session1.py and session2.py terminal outputs.
Session 1 — Characterise the data
Task 1 — Load and inspect the heart disease dataset
I loaded heart.csv with pandas.read_csv(), inspected its first five rows, and calculated summary statistics.
- Dataset shape: 303 rows × 14 columns.
- Columns: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target.
- Continuous variables examined: age, chol, trestbps, and thalach.
- Missing values: 0 in each of the four analysed continuous columns and in target.
Descriptive statistics
Statistic	Age (age)	Cholesterol (chol)	Resting BP (trestbps)	Maximum heart rate (thalach)
Count	303	303	303	303
Mean	54.438944	246.693069	131.689769	149.607261
Standard deviation	9.038662	51.776918	17.599748	22.875003
Minimum	29	126	94	71
25th percentile	48	211	120	133.5
Median	56	241	130	153
75th percentile	61	275	140	166
Maximum	77	564	200	202


The patients' average age is about 54.44 years. The mean cholesterol level is 246.69, the mean resting blood pressure is 131.69, and the mean maximum heart rate is 149.61. Cholesterol has a particularly high maximum value (564) compared with its median (241), suggesting the presence of high-end observations.

For the binary outcome target, 164 patients (54.13%) have target = 0, and 139 patients (45.87%) have target = 1.
Task 2 — Describe the distributions
I generated four histograms using the continuous variables age, chol, trestbps, and thalach.

Figure: heart_histograms.png

The histograms show the spread of each variable. The numerical summary suggests that cholesterol has some unusually high observations. Shape judgments should be checked against the histograms themselves; a numerical summary alone does not establish whether a distribution is symmetric, skewed, or multimodal.
Task 3 — Check whether the variables are normally distributed
I examined Q–Q plots and used the Shapiro–Wilk test, with a significance threshold of 0.05. The null hypothesis is that the data come from a normal distribution. A p-value below 0.05 gives evidence against normality.
Variable	Shapiro–Wilk p-value	Decision at 5%
age	0.0060686	Reject normality
chol	5.9115 × 10⁻⁹	Reject normality
trestbps	1.8021 × 10⁻⁶	Reject normality
thalach	6.9965 × 10⁻⁵	Reject normality


Conclusion: All four variables have p < 0.05, so none passes the Shapiro–Wilk normality test for the full sample. This result guided the choice of statistical tests in Session 2.

Figure: heart_qq_plots.png

Session 2 — Draw conclusions
Task 4 — Do heart disease patients have a different maximum heart rate?
I compared thalach (maximum heart rate) between patients with heart disease (target = 1) and those without (target = 0). For each group I calculated the mean, standard deviation, and 95% confidence interval of the mean.
Group	Sample size	Mean thalach	Standard deviation	95% confidence interval for mean	Shapiro p-value
Disease (target = 1)	139	139.259	22.593	[135.470, 143.048]	0.3523
Healthy (target = 0)	164	158.378	19.199	[155.418, 161.338]	0.0005433


The healthy group has an average maximum heart rate approximately 19.12 beats/min higher than the disease group (158.378 − 139.259).

The disease group's Shapiro test did not reject normality, but the healthy group's test did (p = 0.0005433). Therefore, I used the Mann–Whitney U test to compare the groups without requiring both distributions to be normal.
- Test: Mann–Whitney U
- p-value: 1.86101 × 10⁻¹³
- Decision: p < 0.05, indicating a statistically significant difference between the groups' distributions.
Conclusion: In this dataset, patients with heart disease tend to have lower maximum heart rates than healthy patients. The test supports a difference between the groups, but does not establish that heart disease itself causes the lower heart rate. The Mann–Whitney U test concerns rank/distribution differences rather than testing equality of the means directly.

Figure: heart_group_comparison.png

Task 5 — Do age and maximum heart rate move together?
I compared age with thalach using a scatter plot and two correlation measures.
Correlation method	Coefficient	p-value
Pearson	−0.3938	1.109 × 10⁻¹²
Spearman	−0.3916	1.511 × 10⁻¹²


Both coefficients are negative and of moderate size. Their small p-values show evidence of a relationship.

Conclusion: In this dataset, older patients tend to have a lower maximum heart rate. This is a moderate negative association, not evidence that age alone is the cause of the difference.

Figure: heart_age_thalach.png

Dataset 2 — Chemical-exposure mystery (synthetic data)
The chemicals_cancer.csv dataset has 1,000 observations and includes benzene exposure, cadmium exposure, pollution_index, age, and a malignancy score.
Task 6 — Naive analysis (before adjustment)
I first calculated Pearson correlations between each chemical and malignancy without accounting for pollution or age.
Chemical	Pearson r	p-value	Sample size
Benzene	0.3847	1.276 × 10⁻³⁶	1,000
Cadmium	0.8800	Displayed as 0 by the software (extremely small)	1,000


Both chemicals have a positive correlation with malignancy. Cadmium appears much more strongly associated with malignancy in the naive analysis because 0.8800 is much larger than 0.3847.

Naive conclusion: Based only on these two unadjusted correlations, cadmium looks like the stronger candidate. However, correlation is not proof of causation, and pollution may influence both exposure and malignancy.

Figure: chemicals_naive.png

Task 7 — Account for pollution (confounding)
I repeated the correlation analysis within two pollution categories:
- Low pollution: pollution_index < 40 (n = 402)
- High pollution: pollution_index >= 60 (n = 398)
The remaining 200 observations are outside these two categories and are not part of this stratified comparison.
Correlations within each pollution group
Pollution group	Chemical	Pearson r	p-value
Low (n = 402)	Benzene	0.6110	1.640 × 10⁻⁴²
Low (n = 402)	Cadmium	0.5256	6.227 × 10⁻³⁰
High (n = 398)	Benzene	0.6003	2.521 × 10⁻⁴⁰
High (n = 398)	Cadmium	0.5139	3.303 × 10⁻²⁸


Within both pollution groups, benzene has the stronger observed correlation with malignancy: 0.6110 vs 0.5256 in the low-pollution group, and 0.6003 vs 0.5139 in the high-pollution group.
Additional check: partial correlations controlling for pollution index and age
The script also calculated partial Pearson correlations after accounting for pollution_index and age:
Chemical	Partial Pearson r
Benzene	0.7595
Cadmium	0.0069


Interpretation: After adjusting for pollution index and age, the benzene–malignancy association remains strong, while the cadmium–malignancy association becomes approximately zero. This changes the picture obtained from the naive correlations. The stronger initial cadmium correlation was likely driven largely by confounding. In contrast, benzene is the chemical whose association persists after adjustment.

The broad low/high pollution groups do not completely control for different pollution levels within each group, which helps explain why the stratified cadmium correlations remain positive even though the more detailed partial correlation is near zero.

Conclusion: In this synthetic exercise, benzene is the stronger candidate for the chemical related to malignancy after controlling for pollution and age. Nonetheless, these observational correlations cannot, by themselves, prove that benzene causes malignancy.

Figure: chemicals_stratified.png

Bonus — How unpredictable is a category?
I examined the proportions of the binary target variable in the heart disease dataset.
Category	Count	Proportion	Percentage
target = 0	164	0.541254	54.13%
target = 1	139	0.458746	45.87%


The categories are reasonably balanced. To quantify uncertainty, I calculated Shannon entropy:
\[
H=-\sum_i p_i\log_2 p_i
\]
Result: H = 0.9951 bits.

For a binary variable, the maximum possible entropy is 1 bit, obtained when both categories have equal probability. The result is very close to 1, so target is nearly maximally uncertain when only its overall class frequency is known. This does not mean individual patients cannot be predicted using other information.
Overall conclusion
This practical work showed why statistical interpretation requires more than a single summary number.
1. The Cleveland dataset contains 303 patients, and the four studied continuous variables all failed the Shapiro–Wilk normality check at the 5% level.
2. The heart-rate distributions differ significantly between disease and healthy groups (Mann–Whitney p = 1.86101 × 10⁻¹³), with a lower observed mean in the disease group.
3. Age and maximum heart rate have a moderate negative correlation (Pearson r = −0.3938).
4. In the synthetic chemical dataset, cadmium initially appears more strongly associated with malignancy (r = 0.8800) than benzene (r = 0.3847).
5. After controlling for pollution and age, the results change: the partial correlation is 0.7595 for benzene but only 0.0069 for cadmium. This demonstrates why potential confounding variables must be considered before drawing causal conclusions.
Main lesson: A strong correlation can be misleading. Statistical significance does not automatically establish causation.
Reproducibility and submission
Project structure:
PW3/
├── heart.csv
├── chemicals_cancer.csv
├── session1.py
├── session2.py
├── heart_histograms.png
├── heart_qq_plots.png
├── heart_group_comparison.png
├── heart_age_thalach.png
├── chemicals_naive.png
├── chemicals_stratified.png
└── README.md
To reproduce the analysis from the PW3 directory:
conda activate cspc
python session1.py
python session2.py
The two scripts print the numerical results and generate the figures listed above. The resulting files can then be committed and pushed to the CSPC repository.