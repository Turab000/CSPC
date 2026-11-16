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

