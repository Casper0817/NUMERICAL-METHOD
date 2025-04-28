import numpy as np
import math

# 定義微分方程 y' = 1 + (y/t) + (y/t)^2
def f(t, y):
    return 1 + (y / t) + (y / t)**2

# 定義二階導數 f''(t, y)，給 Taylor's method 使用
def f2(t, y):
    dy_dt = f(t, y)
    df_dt = -(y / t**2) - 2 * (y**2) / (t**3)  # 對 t 的偏微分
    df_dy = (1/t) + (2*y)/(t**2)               # 對 y 的偏微分
    return df_dt + df_dy * dy_dt

# 精確解
def exact_solution(t):
    return t * math.tan(math.log(t))

# 初始條件
h = 0.1
t_start = 1.0
t_end = 2.0
n_steps = int((t_end - t_start) / h)

# 初始化儲存
t_vals = [t_start]
euler_vals = [0]      # y(1) = 0
taylor_vals = [0]
exact_vals = [exact_solution(t_start)]

# 初始值
t = t_start
y_euler = 0
y_taylor = 0

# 進行迴圈計算
for _ in range(n_steps):
    # Euler method
    y_euler += h * f(t, y_euler)
    # Taylor method
    y_taylor += h * f(t, y_taylor) + (h**2 / 2) * f2(t, y_taylor)

    t += h
    t_vals.append(t)
    euler_vals.append(y_euler)
    taylor_vals.append(y_taylor)
    exact_vals.append(exact_solution(t))

# ----------------
# 輸出表格
# ----------------
print(f"{'t':>6} {'Euler Approx':>15} {'Taylor Approx':>15} {'Exact Value':>15} {'Euler Error':>15} {'Taylor Error':>15}")
print("-" * 81)

for i in range(len(t_vals)):
    t = t_vals[i]
    euler = euler_vals[i]
    taylor = taylor_vals[i]
    exact = exact_vals[i]
    euler_error = abs(euler - exact)
    taylor_error = abs(taylor - exact)

    print(f"{t:6.2f} {euler:15.6f} {taylor:15.6f} {exact:15.6f} {euler_error:15.6e} {taylor_error:15.6e}")

