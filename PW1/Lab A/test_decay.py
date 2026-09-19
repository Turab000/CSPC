"""
Tests for the decay simulation.

One complete test is given as a model. Add the two tests described in the
lab handout (a negative-rate test, and a test against the analytical law).
Run with:  pytest -v
"""
import numpy as np
import pytest

from decay import simulate


# Müəllimin verdiyi hazır test
def test_initial_value():
    result = simulate(1000, 0.4)
    assert result[0] == 1000


# Sənin əlavə etdiyin test 2
def test_negative_rate():
    with pytest.raises(ValueError):
        simulate(1000, -0.4)


def test_average_matches_theory():
    N0 = 1000
    rate = 0.4
    dt = 0.05
    t = 3.0

    steps = int(t / dt)
    results = []

    for seed in range(200):
        simulation = simulate(
            N0,
            rate,
            dt=dt,
            steps=steps,
            seed=seed
        )
        results.append(simulation[-1])

    average = np.mean(results)

    expected = N0 * np.exp(-rate * t)

    assert average == pytest.approx(expected, rel=0.05)

import numpy as np
import pytest
from decay import simulate, simulate_loop


def test_starts_at_N0():
    # at time zero, no atoms have decayed yet
    assert simulate(1000, 0.4)[0] == 1000


# TODO 1: test_rejects_negative_rate
#   Check that calling simulate(...) with a negative lam raises a ValueError.
#   Which pytest tool checks that an error is raised?


# TODO 2: test_matches_law
#   Check that the simulation's AVERAGE over many seeds is close to the
#   physical law  N0 * exp(-lam * t).
#   Which pytest tool compares floating-point values with a tolerance?
