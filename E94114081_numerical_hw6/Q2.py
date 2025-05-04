import numpy as np

# 定義矩陣 A
A = np.array([
    [4, 1, -1, 0],
    [1, 3, -1, 0],
    [-1, -1, 6, 2],
    [0, 0, 2, 5]
], dtype=float)

# 求反矩陣
A_inv = np.linalg.inv(A)

# 顯示結果
np.set_printoptions(precision=4, suppress=True)
print("A 的反矩陣 A⁻¹ 為：")
print(A_inv)
