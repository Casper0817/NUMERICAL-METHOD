import numpy as np

# 定義積分函數
def f(x):
    return np.exp(x) * np.sin(4 * x)

a, b = 1, 2
h = 0.1
n = int((b - a) / h)

x = np.linspace(a, b, n + 1)
y = f(x)

# (a) Composite Trapezoidal Rule
trap = h / 2 * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])

# (b) Composite Simpson's Rule
if n % 2 != 0:
    raise ValueError("Simpson's rule requires even n")
simpson = h / 3 * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1])

# (c) Composite Midpoint Rule
midpoints = np.linspace(a + h/2, b - h/2, n)
mid = h * np.sum(f(midpoints))

print(f"(a) Composite Trapezoidal Rule: {trap:.6f}")
print(f"(b) Composite Simpson's Rule: {simpson:.6f}")
print(f"(c) Composite Midpoint Rule: {mid:.6f}")
