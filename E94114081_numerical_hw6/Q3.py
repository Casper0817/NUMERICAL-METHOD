import numpy as np

# 三對角矩陣系統
a = np.array([0, -1, -1, -1], dtype=float)  # sub-diagonal
b = np.array([3, 3, 3, 3], dtype=float)     # diagonal
c = np.array([-1, -1, -1, 0], dtype=float)  # super-diagonal
d = np.array([2, 3, 4, 1], dtype=float)     # right-hand side

n = len(b)

# Crout 分解：L[i] 對角線, M[i] 下對角比例, U[i] 上對角元素（實際只存 u1, u2, u3）
L = np.zeros(n)
U = np.zeros(n)
M = np.zeros(n)

# 初始值
L[0] = b[0]
U[0] = c[0] / L[0]

# 前向遞推
for i in range(1, n):
    M[i] = a[i] / L[i-1]
    L[i] = b[i] - M[i] * c[i-1]
    if i < n-1:
        U[i] = c[i] / L[i]

# 前向替代 Ly = d
y = np.zeros(n)
y[0] = d[0] / L[0]
for i in range(1, n):
    y[i] = (d[i] - M[i] * y[i-1]) / L[i]

# 後向替代 Ux = y
x = np.zeros(n)
x[-1] = y[-1]
for i in range(n-2, -1, -1):
    x[i] = y[i] - U[i] * x[i+1]

# 輸出結果
for i in range(n):
    print(f"x{i+1} = {x[i]:.6f}")
