import numpy as np
import math
from scipy.interpolate import lagrange

# 給定的數據點 (x_i, f(x_i))
x_values = np.array([0.698, 0.733, 0.768, 0.803])
y_values = np.array([0.7661, 0.7432, 0.7193, 0.6946])

# 目標 x 值
x_target = 0.750

# 使用 SciPy 計算拉格朗日插值多項式
poly1 = lagrange(x_values[:2], y_values[:2])  # 一次插值
poly2 = lagrange(x_values[:3], y_values[:3])  # 二次插值
poly3 = lagrange(x_values[:4], y_values[:4])  # 三次插值
poly4 = lagrange(x_values, y_values)  # 四次插值

# 計算插值結果
P1 = poly1(x_target)
P2 = poly2(x_target)
P3 = poly3(x_target)
P4 = poly4(x_target)

# 實際值
true_value = 0.7317
error1 = abs(true_value - P1)
error2 = abs(true_value - P2)
error3 = abs(true_value - P3)
error4 = abs(true_value - P4)

# 計算誤差界 (error bound)
def lagrange_error(M, x_values, x_target, n):
    """ 計算拉格朗日誤差界 """
    prod = np.prod([abs(x_target - xi) for xi in x_values[:n+1]])  # 計算乘積項
    return (M / math.factorial(n+1)) * prod

# 設定 M 值（\cos(x) 或 \sin(x) 的最大值在 [0.698, 0.803] 區間）
M1 = 1  # 二階導數最大值（cos 或 sin 最大為 1）
M2 = 1  # 三階導數最大值
M3 = 1  # 四階導數最大值
M4 = 1  # 五階導數最大值

# 計算誤差界
error_bound1 = lagrange_error(M1, x_values[:2], x_target, 1)
error_bound2 = lagrange_error(M2, x_values[:3], x_target, 2)
error_bound3 = lagrange_error(M3, x_values[:4], x_target, 3)
error_bound4 = lagrange_error(M4, x_values, x_target, 4)

# 顯示結果
print(f"一次插值: P1(0.750) = {P1:.6f}, 誤差 = {error1:.6e}, 誤差界 = {error_bound1:.6e}")
print(f"二次插值: P2(0.750) = {P2:.6f}, 誤差 = {error2:.6e}, 誤差界 = {error_bound2:.6e}")
print(f"三次插值: P3(0.750) = {P3:.6f}, 誤差 = {error3:.6e}, 誤差界 = {error_bound3:.6e}")
print(f"四次插值: P4(0.750) = {P4:.6f}, 誤差 = {error4:.6e}, 誤差界 = {error_bound4:.6e}")
