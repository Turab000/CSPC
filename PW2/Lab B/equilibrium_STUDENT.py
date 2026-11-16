"""
PW2 Lab B Part 4 -- chemical equilibrium via the equilibrium constant K.

Reaction  H2 + I2 <=> 2 HI, starting from 1 mol H2 and 1 mol I2.
As the reaction proceeds by an extent x:  H2 = 1-x,  I2 = 1-x,  HI = 2x.
At equilibrium the composition satisfies the equilibrium constant
        K = [HI]^2 / ([H2][I2]) = (2x)^2 / ((1-x)(1-x)).
Given K, find the extent x. Solve it TWO ways and compare.
Run:  python equilibrium.py
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton, minimize

K = 15.6
a = b = 1.0

# TODO 1: Equilibrium imbalance
def k_imbalance(x):
    return (2*x)**2 / ((a-x)*(b-x)) - K

# TODO 2: Newton's method
x_newton = newton(k_imbalance, x0=0.5)
print("Newton x =", x_newton)

# TODO 3: SLSQP
def error(x):
    return k_imbalance(x[0])**2

result = minimize(
    error,
    x0=[0.5],
    method="SLSQP",
    bounds=[(0, 0.999)]
)

x_slsqp = result.x[0]
print("SLSQP x =", x_slsqp)

# TODO 4: Equilibrium amounts
x_eq = x_newton

H2 = a - x_eq
I2 = b - x_eq
HI = 2*x_eq

print("H2 =", H2)
print("I2 =", I2)
print("HI =", HI)

# Plot amounts against reaction extent
x_values = np.linspace(0, 1, 200)

plt.plot(x_values, a-x_values, label="H2")
plt.plot(x_values, b-x_values, label="I2")
plt.plot(x_values, 2*x_values, label="HI")

plt.axvline(x_eq, color="black",
            linestyle="--",
            label="Equilibrium")

plt.xlabel("Reaction extent x")
plt.ylabel("Amount (mol)")
plt.title("Chemical Equilibrium")
plt.legend()
plt.grid()
plt.savefig("equilibrium.png", dpi=150)
plt.show()
