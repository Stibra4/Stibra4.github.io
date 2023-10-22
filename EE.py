import numpy as np
from scipy.integrate import quad

def f(x):
    return x**2

def riemann(f, a, b, n):
    dx = (b - a) / n
    integral = 0
    for i in range(n):
        x_i = a + i * dx
        integral += f(x_i) * dx
    return integral

def lebesgue(f, a, b, n):
    x = np.linspace(a, b, n)
    dx = (b - a) / n
    integral = np.sum(f(x) * dx)
    return integral

a = 0
b = 1
n = 1000

riemann_result = riemann(f, a, b, n)
lebesgue_result = lebesgue(f, a, b,n)  
exact,_ = quad(f,a,b)

print("Intégrale de Riemann:", riemann_result)
print("Intégrale de Lebesgue:", lebesgue_result)
print("Intégrale de Lebesgue:", exact)
