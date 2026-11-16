"""
PW2 Lab B Part 2 -- three routes to a minimum.

Compare gradient descent, Newton, and SLSQP on two functions:
  2A: f(x) = (x-3)**2 + 1          (easy, one minimum at x=3)
  2B: g(x) = x**4 - 3*x**2 + x + 5 (harder, several stationary points)
Run:  python warmup.py
"""
import numpy as np
from scipy.optimize import newton, minimize

# ---------- 2A: easy convex function ----------
def f(x):   return (x-3)**2 + 1
def df(x):  return 2*(x-3)
def d2f(x): return 2.0

# TODO 2A: minimise f three ways from x0=0 and print each result:
#   (1) gradient descent by hand (loop x = x - lr*df(x) until the step is tiny)
#   (2) scipy.optimize.newton(df, x0, fprime=d2f)
#   (3) scipy.optimize.minimize(f, x0, method="SLSQP")

# ---------- 2B: harder landscape ----------
def g(x):   return x**4 - 3*x**2 + x + 5
def dg(x):  return 4*x**3 - 6*x + 1
def d2g(x): return 12*x**2 - 6

# TODO 2B: run the same three methods on g, from x0=0 AND from x0=2.
#   For Newton (which solves dg(x)=0), also check the sign of d2g at the answer:
#   d2g > 0 means a minimum, d2g < 0 means a maximum.
#   In your README note: do the methods agree? did Newton find a minimum or
#   another stationary point? how did the starting point change the result?


# Method 1: Gradient descent
x = 0.0
lr = 0.1

for i in range(1000):
    x_new = x - lr * df(x)
    if abs(x_new - x) < 1e-8:
        break
    x = x_new

print("2A Gradient descent:", x)

# Method 2: Newton
result_newton = newton(df, x0=0, fprime=d2f)
print("2A Newton:", result_newton)

# Method 3: SLSQP
result_slsqp = minimize(
    f, x0=[0], method="SLSQP"
)
print("2A SLSQP:", result_slsqp.x[0])


for x0 in [0, 2]:
    print("\nStarting point:", x0)

    # Gradient descent
    x = float(x0)
    lr = 0.01

    for i in range(10000):
        x_new = x - lr * dg(x)
        if abs(x_new - x) < 1e-8:
            break
        x = x_new

    print("Gradient descent:", x)

    # Newton
    xn = newton(dg, x0=x0, fprime=d2g)
    print("Newton:", xn)
    print("Second derivative:", d2g(xn))

    # SLSQP
    result = minimize(g, [x0], method="SLSQP")
    print("SLSQP:", result.x[0])