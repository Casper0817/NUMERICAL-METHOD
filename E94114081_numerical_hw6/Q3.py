def crout_tridiagonal(a, b, c, d):
    """
    解三對角矩陣 Ax = d，其中：
    a: 對角線元素（長度 n）
    b: 上對角線元素（長度 n-1）
    c: 下對角線元素（長度 n）
    d: 右邊常數向量（長度 n）
    回傳：解向量 x
    """
    n = len(d)
    l = [0.0] * n
    u = [0.0] * (n - 1)
    y = [0.0] * n
    x = [0.0] * n

    # Crout 分解
    l[0] = a[0]
    u[0] = b[0] / l[0]
    for i in range(1, n - 1):
        l[i] = a[i] - c[i] * u[i - 1]
        u[i] = b[i] / l[i]
    l[n - 1] = a[n - 1] - c[n - 1] * u[n - 2]

    # 前代 Ly = d
    y[0] = d[0] / l[0]
    for i in range(1, n):
        y[i] = (d[i] - c[i] * y[i - 1]) / l[i]

    # 回代 Ux = y
    x[n - 1] = y[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = y[i] - u[i] * x[i + 1]

    return x

# 三對角系統的對應係數
a = [3, 3, 3, 3]          # 對角線元素
b = [-1, -1, -1]          # 上對角線元素
c = [0, -1, -1, -1]       # 下對角線元素（第0項不使用）
d = [2, 3, 4, 1]          # 右邊常數項

# 求解
solution = crout_tridiagonal(a, b, c, d)

# 輸出結果
print("Crout 分解法解得：")
for i, val in enumerate(solution):
    print(f"x{i+1} = {val:.6f}")
