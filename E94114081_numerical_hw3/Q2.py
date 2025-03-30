
import numpy as np
from scipy.interpolate import lagrange
from scipy.optimize import fsolve


x_values = np.array([0.3, 0.4, 0.5, 0.6])  
y_values = np.exp(-x_values)  


inv_poly = lagrange(y_values, x_values)

# 解 x = e^(-x) -> 找 y = x 的數值解
def equation(x):
    return inv_poly(x) - x

# 使用 fsolve 求解，初始猜測值為 0.5
x_solution = fsolve(equation, 0.5)

# 輸出結果
x_solution[0]
