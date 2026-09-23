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