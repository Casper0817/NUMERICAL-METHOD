import numpy as np
from scipy.integrate import quad

# 被積分的函數 f(x) = x^2 * ln(x)
def f(x):
    return x**2 * np.log(x)

# 自訂高斯求積法節點與權重的轉換（從 [-1, 1] 映射到 [a, b]）
def gaussian_quadrature(a, b, n):
    nodes, weights = np.polynomial.legendre.leggauss(n)
    mapped_nodes = 0.5 * (nodes + 1) * (b - a) + a
    mapped_weights = 0.5 * (b - a) * weights
    return mapped_nodes, mapped_weights

# 使用高斯求積法計算近似積分
def compute_integral(a, b, n):
    nodes, weights = gaussian_quadrature(a, b, n)
    return np.sum(weights * f(nodes))

# 積分區間
a = 1
b = 1.5

# 分別使用 n=3 和 n=4 計算
integral_n3 = compute_integral(a, b, 3)
integral_n4 = compute_integral(a, b, 4)

# 計算精確值（使用 scipy 的 quad 方法）
true_integral, _ = quad(f, a, b)

# 顯示結果
print(f"高斯求積法 n=3 的結果: {integral_n3:.8f}")
print(f"高斯求積法 n=4 的結果: {integral_n4:.8f}")
print(f"精確值:              {true_integral:.8f}")
print(f"n=3 的誤差:          {abs(true_integral - integral_n3):.8f}")
print(f"n=4 的誤差:          {abs(true_integral - integral_n4):.8f}")
