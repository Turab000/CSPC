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

