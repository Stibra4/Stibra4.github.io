import math
import numpy as np
import matplotlib.pyplot as plt

# PHYSICAL PARAMETERS
K = 0.1     # Diffusion coefficient
L = 1.0     # Domain size
Time = 20.  # Integration time

V = 1
lamda = 1

# NUMERICAL PARAMETERS
NX = 2  # Number of grid points
NT = 10000   # Number of time steps max
ifre = 1000000  # Plot every ifre time iterations
eps = 0.001     # Relative convergence ratio
niter_refinement = 1      # niter different calculations with variable mesh size

error_L2 = []  # Store L2 errors
error_H1 = []  # Store H1 errors

for iter in range(niter_refinement):
    NX = NX + 3

    dx = L / (NX - 1)                 # Grid step (space)
    dt = dx ** 2 / (V * dx + K + dx ** 2)   # Grid step (time)

    # Initialisation
    x = np.linspace(0.0, 1.0, NX)
    T = np.zeros((NX))
    F = np.zeros((NX))
    rest = []
    RHS = np.zeros((NX))

    Tex = np.zeros((NX))
    Texx = np.zeros((NX))
    for j in range(1, NX - 1):
        Tex[j] = np.sin(2 * j * math.pi / NX)
    for j in range(1, NX - 1):
        Texx[j] = (Tex[j + 1] - Tex[j - 1]) / (2 * dx)
        Txx = (Tex[j + 1] - 2 * Tex[j] + Tex[j - 1]) / (dx ** 2)
        F[j] = V * Texx[j] - K * Txx + lamda * Tex[j]

    dt = dx ** 2 / (V * dx + 2 * K + abs(np.max(F)) * dx ** 2)

    # Main loop en temps
    n = 0
    res = 1
    res0 = 1

    while n < NT and res / res0 > eps:
        n += 1
        res = 0
        for j in range(1, NX - 1):
            xnu = K + 0.5 * dx * abs(V)
            Tx = (T[j + 1] - T[j - 1]) / (2 * dx)
            Txx = (T[j - 1] - 2 * T[j] + T[j + 1]) / (dx ** 2)
            RHS[j] = dt * (-V * Tx + xnu * Txx - lamda * T[j] + F[j])
            res += abs(RHS[j])

        for j in range(1, NX - 1):
            T[j] += RHS[j]
            RHS[j] = 0

        if n == 1:
            res0 = res
        rest.append(res)

        err_L2 = np.sqrt(np.dot(T - Tex, T - Tex))
        error_L2.append(err_L2)

        err_H1 = 0
        for j in range(1, NX - 1):
            err_H1 += (Texx[j] - (T[j + 1] - T[j - 1]) / (2 * dx)) ** 2
        error_H1.append(np.sqrt(err_H1))


plt.figure(1)
plt.subplot(121)
plt.plot(error_L2, label='L2 Error')
plt.xlabel('Iteration')
plt.ylabel('L2 Error')
plt.legend()

plt.subplot(122)
plt.plot(error_H1, label='H1 Error')
plt.xlabel('Iteration')
plt.ylabel('H1 Error')
plt.legend()

plt.show()
