import numpy as np
from scipy.integrate import dblquad
from scipy.special import roots_legendre

# ========================================
# 定義欲積分的函數 f(x, y)
# ========================================
def f(x, y):
    return 2 * y * np.sin(x) + np.cos(x)**2

# ========================================
# 計算精確值 (使用 SciPy 的數值積分)
# ========================================
def exact_integral():
    result, _ = dblquad(lambda y, x: f(x, y), 0, np.pi / 4, lambda x: np.sin(x), lambda x: np.cos(x))
    return result

# ========================================
# Simpson's Rule (n = m = 4)
# ========================================
def simpsons_rule_2d():
    n = m = 4
    a, b = 0, np.pi / 4
    hx = (b - a) / n
    result = 0

    for i in range(n + 1):
        x = a + i * hx
        wx = 1
        if i == 0 or i == n:
            wx = 1
        elif i % 2 == 1:
            wx = 4
        else:
            wx = 2

        ymin = np.sin(x)
        ymax = np.cos(x)
        hy = (ymax - ymin) / m

        inner = 0
        for j in range(m + 1):
            y = ymin + j * hy
            wy = 1
            if j == 0 or j == m:
                wy = 1
            elif j % 2 == 1:
                wy = 4
            else:
                wy = 2

            inner += wy * f(x, y)

        result += wx * hy / 3 * inner

    return hx / 3 * result

# ========================================
# Gaussian Quadrature (n = m = 3)
# ========================================
def gaussian_quadrature_2d():
    n = m = 3
    a, b = 0, np.pi / 4
    nodes_x, weights_x = roots_legendre(n)
    nodes_y, weights_y = roots_legendre(m)

    total = 0
    for i in range(n):
        xi = 0.5 * (nodes_x[i] + 1) * (b - a) + a
        wi = weights_x[i]
        ymin = np.sin(xi)
        ymax = np.cos(xi)

        inner = 0
        for j in range(m):
            yj = 0.5 * (nodes_y[j] + 1) * (ymax - ymin) + ymin
            wj = weights_y[j]
            inner += wj * f(xi, yj)

        total += wi * (ymax - ymin) / 2 * inner

    return total * (b - a) / 2

# ========================================
# 主程式執行
# ========================================
if __name__ == "__main__":
    simpson_result = simpsons_rule_2d()
    gaussian_result = gaussian_quadrature_2d()
    exact_result = exact_integral()

    print(f"Simpson's Rule Result (n = m = 4): {simpson_result:.6f}")
    print(f"Gaussian Quadrature Result (n = m = 3): {gaussian_result:.6f}")
    print(f"Exact Integral Value: {exact_result:.6f}")
    print(f"Error (Simpson): {abs(simpson_result - exact_result):.2e}")
    print(f"Error (Gaussian): {abs(gaussian_result - exact_result):.2e}")
