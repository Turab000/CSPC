import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# TODO 1: Read experimental data
data = np.genfromtxt(
    "kinetics.csv",
    delimiter=",",
    names=True
)

t = data["time"]
C = data["concentration"]
C0 = C[0]

# TODO 2: Define total error
def total_error(k):
    k = float(np.atleast_1d(k)[0])
    C_model = C0 * np.exp(-k * t)
    return np.sum((C - C_model)**2)

# TODO 3: Find the best k
result = minimize(
    total_error,
    x0=[0.5],
    method="SLSQP",
    bounds=[(0, 5)]
)

k_fit = result.x[0]
print("Fitted k =", k_fit)

# TODO 4: Plot data and fitted curve
t_smooth = np.linspace(t.min(), t.max(), 300)
C_fit = C0 * np.exp(-k_fit * t_smooth)

plt.scatter(t, C, label="Measured data")
plt.plot(t_smooth, C_fit, label="Fitted curve")
plt.xlabel("Time")
plt.ylabel("Concentration")
plt.title("Reaction Kinetics")
plt.legend()
plt.grid()
plt.savefig("kinetics.png", dpi=150)
plt.show()

