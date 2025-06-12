#!/usr/bin/env python3
import numpy as np

# -------------------------------------------------
# 網格與初條件
dx = dt = 0.1
r  = dt / dx           # = 1
x  = np.arange(0, 1+dx, dx)   # 0,0.1,…,1   共 11 點
Nt = 11                # 計算到 t = 1.0（同樣 0.1 間距）

# 邊界值（固定隨時間不變）
pL, pR = 1.0, 2.0

# 初始位形 p(x,0) 與初速度 p_t(x,0)
p0 = np.cos(2*np.pi*x)
v0 = 2*np.pi*np.sin(2*np.pi*x)

# 內部節點索引（排除兩端邊界）
inner = slice(1, -1)

# -------------------------------------------------
# 建立時間層陣列
p_nm1 = p0.copy()                 # n-1 (t=0)
p_nm1[0], p_nm1[-1] = pL, pR      # 套邊界

# 第一層 n=1
p_n = p0.copy()
p_n[inner] = (p0[inner]
              + dt * v0[inner]
              + 0.5 * r**2 * (p0[2:] - 2*p0[1:-1] + p0[:-2]))
p_n[0], p_n[-1] = pL, pR

# -------------------------------------------------
# 時間向前推進
history = [p_nm1.copy(), p_n.copy()]   # 存 t=0, t=0.1

for n in range(2, Nt+1):
    p_np1 = np.zeros_like(p_n)
    # 差分公式 (r=1 ⇒ 2(1-r^2)=0)
    p_np1[inner] = (p_n[2:] + p_n[:-2] - p_nm1[inner])
    # 邊界
    p_np1[0], p_np1[-1] = pL, pR

    history.append(p_np1.copy())
    # 交換指標
    p_nm1, p_n = p_n, p_np1

# -------------------------------------------------
# 輸出結果：表格 (t=0~1, Δt=0.1)
print("     x  ", end="")
for n in range(Nt+1):
    print(f" t={n*dt:3.1f}", end="")
print()

for i, xi in enumerate(x):
    print(f"x={xi:4.1f}", end="")
    for n in range(Nt+1):
        print(f"{history[n][i]:9.4f}", end="")
    print()
