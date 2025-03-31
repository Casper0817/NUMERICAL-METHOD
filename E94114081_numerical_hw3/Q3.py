import numpy as np
from scipy.interpolate import CubicHermiteSpline

# 給定的數據點
T = np.array([0, 3, 5, 8, 13])   # 時間 (秒)
D = np.array([0, 200, 375, 620, 990])  # 距離 (英尺)
V = np.array([75, 77, 80, 74, 72])  # 速度 (英尺/秒)

# 使用 Hermite 插值構建位置函數
hermite_interp = CubicHermiteSpline(T, D, V)

# (a) 預測 t = 10 的位置和速度
t_target = 10
D_10 = hermite_interp(t_target)
V_10 = hermite_interp.derivative()(t_target)

# (b) 確認車輛是否超過 55 mi/h (1 mi = 5280 ft, 55 mi/h = 55 * 5280 / 3600 ft/s)
speed_limit = 55 * 5280 / 3600  # 80.67 ft/s

# 在範圍內細分時間點以獲取更精確的速度曲線
t_fine = np.linspace(min(T), max(T), 1000)
V_fine = hermite_interp.derivative()(t_fine)

# 檢查是否超速
exceed_indices = np.where(V_fine > speed_limit)[0]
first_time_exceed = t_fine[exceed_indices[0]] if exceed_indices.size > 0 else None

# (c) 預測最大速度
max_speed = np.max(V_fine)

# 印出結果
print(f"(a) 當 t = 10 秒時, 位置 D(10) ≈ {D_10:.2f} 英尺, 速度 V(10) ≈ {V_10:.2f} 英尺/秒")
print(f"(b) 車輛是否超速? {'是' if first_time_exceed else '否'}, 最早超速時間: {first_time_exceed:.2f} 秒" if first_time_exceed else "車輛沒有超速")
print(f"(c) 預測最大速度: {max_speed:.2f} 英尺/秒")
