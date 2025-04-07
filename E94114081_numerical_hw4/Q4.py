import numpy as np
from scipy.integrate import simpson

# (a) 變數變換後的函數: t^(-7/4) * sin(1/t)
def f_a(t):
    return t**(-7/4) * np.sin(1/t)

# (b) 變數變換後的函數: t^2 * sin(1/t)
def f_b(t):
    result = np.zeros_like(t)
    non_zero = t != 0
    result[non_zero] = t[non_zero]**2 * np.sin(1 / t[non_zero])
    result[~non_zero] = 0  # 處理 t = 0
    return result

# 積分區間與節點設定
n = 1000  # 增加分點以提高精度

# (a) 積分區間 [1, 20] 近似代替 [1, ∞]
t1 = np.linspace(1, 100, 5000)
y1 = f_a(t1)
I1 = simpson(y1, t1)

# (b) 積分區間 [1e-4, 1]，避免 t = 0 的奇異點
t2 = np.linspace(1e-4, 1, n+1)
y2 = f_b(t2)
I2 = simpson(y2, t2)

# 顯示結果
print(f"(a) 變數變換後近似值：{I1:.5f}")
print(f"(b) 變數變換後近似值：{I2:.5f}")

