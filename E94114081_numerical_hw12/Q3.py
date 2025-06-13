
import numpy as np

# ---------- 網格設定 ----------
r_in, r_out = 0.5, 1.0
theta_max   = np.pi / 3

Nr, Nt = 41, 61              # 節點數 (可自行調高以增精度)
dr     = (r_out - r_in) / (Nr - 1)
dθ     = theta_max / (Nt - 1)

r  = np.linspace(r_in,  r_out, Nr)
θ  = np.linspace(0.0, theta_max, Nt)

# ---------- 溫度陣列 ----------
T = np.zeros((Nt, Nr))

# ---------- 套用邊界條件 ----------
T[:, 0]   = 50.0           # r = 0.5
T[:, -1]  = 100.0          # r = 1
T[0, :]   = 0.0            # θ = 0
T[-1, :]  = 0.0            # θ = π/3

# 四個角點取兩邊平均，緩和衝突
T[0,  0]  = (50.0 + 0.0) / 2
T[0, -1]  = (100.0 + 0.0) / 2
T[-1, 0]  = (50.0 + 0.0) / 2
T[-1,-1]  = (100.0 + 0.0) / 2

# ---------- Gauss-Seidel 疊代 ----------
max_iter = 25_000
tol      = 1e-6            # 最大單點改變量 < tol 視為收斂

for it in range(max_iter):
    delta = 0.0
    for j in range(1, Nt - 1):        # θ 方向
        for i in range(1, Nr - 1):    # r 方向
            ri  = r[i]

            # 方便閱讀：鄰近節點
            Trp, Trm = T[j, i+1], T[j, i-1]
            Ttp, Ttm = T[j+1, i], T[j-1, i]

            # 五點差分離散式
            denom = 2/dr**2 + 2/(ri**2 * dθ**2)
            numer = (Trp + Trm) / dr**2 \
                  + (Ttp + Ttm) / (ri**2 * dθ**2) \
                  + (Trp - Trm) / (2 * ri * dr)

            T_new  = numer / denom
            delta  = max(delta, abs(T_new - T[j, i]))
            T[j, i] = T_new

    if delta < tol:
        print(f"Converged in {it} iterations (max Δ = {delta:.2e})")
        break
else:
    print("Warning: not converged within max_iter")

# ---------- 列印 θ=π/6 (中線) 上的溫度 ----------
mid_j = Nt // 2            # θ ≈ π/6
print("\\nθ = π/6 (約 30°) 截面溫度：")
for i, ri in enumerate(r):
    print(f"r = {ri:.2f}   T ≈ {T[mid_j, i]:7.2f}")
