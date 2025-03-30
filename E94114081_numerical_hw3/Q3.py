import numpy as np
import scipy.interpolate as spi

# 給定的數據點
T = np.array([0, 3, 5, 8, 13])   # 時間 (秒)
D = np.array([0, 200, 375, 620, 990])  # 距離 (英尺)
V = np.array([75, 77, 80, 74, 72])  # 速度 (英尺/秒)

# 使用 Hermite 插值構建位置函數
hermite_interp = spi.BarycentricInterpolator(T, D)
hermite_derivative = spi.BarycentricInterpolator(T, V)  # 速度即為位置導數

# (a) 預測 t = 10 的位置和速度
t_target = 10
D_10 = hermite_interp(t_target)
V_10 = hermite_derivative(t_target)

# (b) 確認車輛是否超過 55 mi/h (1 mi = 5280 ft, 55 mi/h = 55 * 5280 / 3600 ft/s)
speed_limit = 55 * 5280 / 3600  # 80.67 ft/s

# 檢查速度是否超過 speed_limit
exceeds_speed = np.any(V > speed_limit)
first_time_exceed = None

if exceeds_speed:
    for i, v in enumerate(V):
        if v > speed_limit:
            first_time_exceed = T[i]
            break

# (c) 預測最大速度
max_speed = np.max(V)

print(f"(a) 當 t = 10 秒時, 位置 D(10) ≈ {D_10:.2f} 英尺, 速度 V(10) ≈ {V_10:.2f} 英尺/秒")
print(f"(b) 車輛是否超速? {exceeds_speed}, 最早超速時間: {first_time_exceed}")
print(f"(c) 預測最大速度: {max_speed} 英尺/秒")
