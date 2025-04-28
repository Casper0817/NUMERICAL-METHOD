import numpy as np
import math

# 定義系統的微分方程
def f1(t, u1, u2):
    return 9*u1 + 24*u2 + 5*np.cos(t) - (1/3)*np.sin(t)

def f2(t, u1, u2):
    return -24*u1 - 52*u2 - 9*np.cos(t) + (1/3)*np.sin(t)

# 精確解
def exact_u1(t):
    return 2*np.exp(-3*t) - np.exp(-39*t) + (1/3)*np.cos(t)

def exact_u2(t):
    return -np.exp(-3*t) + 2*np.exp(-39*t) - (1/3)*np.cos(t)

# Runge-Kutta 4th-order Method
def runge_kutta_system(h, t_end):
    t_vals = [0]
    u1_vals = [4/3]
    u2_vals = [2/3]

    t = 0
    u1 = 4/3
    u2 = 2/3

    while t < t_end - 1e-10:   # 加個小容差，避免浮點數誤差
        if t + h > t_end:
            h = t_end - t

        k1_1 = h * f1(t, u1, u2)
        k1_2 = h * f2(t, u1, u2)

        k2_1 = h * f1(t + h/2, u1 + k1_1/2, u2 + k1_2/2)
        k2_2 = h * f2(t + h/2, u1 + k1_1/2, u2 + k1_2/2)

        k3_1 = h * f1(t + h/2, u1 + k2_1/2, u2 + k2_2/2)
        k3_2 = h * f2(t + h/2, u1 + k2_1/2, u2 + k2_2/2)

        k4_1 = h * f1(t + h, u1 + k3_1, u2 + k3_2)
        k4_2 = h * f2(t + h, u1 + k3_1, u2 + k3_2)

        u1 += (k1_1 + 2*k2_1 + 2*k3_1 + k4_1) / 6
        u2 += (k1_2 + 2*k2_2 + 2*k3_2 + k4_2) / 6

        t += h
        t_vals.append(t)
        u1_vals.append(u1)
        u2_vals.append(u2)

    return np.array(t_vals), np.array(u1_vals), np.array(u2_vals)

# 印出結果表格
def print_results(t_vals, u1_vals, u2_vals, h):
    print(f"\n結果（h = {h}）:")
    print(f"{'t':>5} {'u1 (RK4)':>12} {'u1 (exact)':>12} {'u1 error':>12} {'u2 (RK4)':>12} {'u2 (exact)':>12} {'u2 error':>12}")
    print("-" * 81)
    for i in range(len(t_vals)):
        t = t_vals[i]
        u1_rk = u1_vals[i]
        u2_rk = u2_vals[i]
        u1_exact = exact_u1(t)
        u2_exact = exact_u2(t)
        err1 = abs(u1_rk - u1_exact)
        err2 = abs(u2_rk - u2_exact)
        print(f"{t:5.2f} {u1_rk:12.6f} {u1_exact:12.6f} {err1:12.2e} {u2_rk:12.6f} {u2_exact:12.6f} {err2:12.2e}")

# 主程式
if __name__ == "__main__":
    t_end = 1.0

    # h = 0.1
    t_01, u1_01, u2_01 = runge_kutta_system(h=0.1, t_end=t_end)
    print_results(t_01, u1_01, u2_01, h=0.1)

    # h = 0.05
    t_005, u1_005, u2_005 = runge_kutta_system(h=0.05, t_end=t_end)
    print_results(t_005, u1_005, u2_005, h=0.05)
