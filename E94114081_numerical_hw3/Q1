import numpy as np
from scipy.interpolate import lagrange

# 給定的數據點 (x_i, f(x_i))
x_values = np.array([0.698, 0.733, 0.768, 0.803])
y_values = np.array([0.7661, 0.7432, 0.7193, 0.6946])

# 要近似的 x 值
x_target = 0.750

# 使用 SciPy 計算拉格朗日插值多項式
poly1 = lagrange(x_values[:2], y_values[:2])  # 一次插值
poly2 = lagrange(x_values[:3], y_values[:3])  # 二次插值
poly3 = lagrange(x_values[:4], y_values[:4])  # 三次插值

# 計算插值結果
P1 = poly1(x_target)
P2 = poly2(x_target)
P3 = poly3(x_target)

# 計算實際值與誤差
true_value = 0.7317
error1 = abs(true_value - P1)
error2 = abs(true_value - P2)
error3 = abs(true_value - P3)

# 顯示結果
print(f"一次插值: P1(0.750) = {P1}, 誤差 = {error1}")
print(f"二次插值: P2(0.750) = {P2}, 誤差 = {error2}")
print(f"三次插值: P3(0.750) = {P3}, 誤差 = {error3}")
