import numpy as np

# -------------------------------------------------
#  原始資料
# -------------------------------------------------
x = np.array([4.0, 4.2, 4.5, 4.7, 5.1, 5.5, 5.9, 6.3])
y = np.array([102.6, 113.2, 130.1, 142.1, 167.5, 195.1, 224.9, 256.8])
n = len(x)

# -------------------------------------------------
# (a) 二次多項式  y ≈ a0 + a1 x + a2 x²
# -------------------------------------------------
A_quad = np.vstack([np.ones_like(x), x, x**2]).T
a0, a1, a2 = np.linalg.lstsq(A_quad, y, rcond=None)[0]
y_quad = A_quad @ np.array([a0, a1, a2])
sse_quad = np.sum((y - y_quad)**2)
rmse_quad = np.sqrt(sse_quad/n)

# -------------------------------------------------
# (b) 指數型  y ≈ b e^{a x}     →  ln y = ln b + a x
# -------------------------------------------------
lny = np.log(y)
A_exp = np.vstack([np.ones_like(x), x]).T
lnb_exp, a_exp = np.linalg.lstsq(A_exp, lny, rcond=None)[0]
b_exp = np.exp(lnb_exp)
y_exp = b_exp * np.exp(a_exp * x)
sse_exp  = np.sum((y - y_exp)**2)
rmse_exp = np.sqrt(sse_exp/n)

# -------------------------------------------------
# (c) 乘冪型  y ≈ b x^{a}      →  ln y = ln b + a ln x
# -------------------------------------------------
lnx = np.log(x)
A_pow = np.vstack([np.ones_like(lnx), lnx]).T
lnb_pow, a_pow = np.linalg.lstsq(A_pow, lny, rcond=None)[0]
b_pow = np.exp(lnb_pow)
y_pow = b_pow * x**a_pow
sse_pow  = np.sum((y - y_pow)**2)
rmse_pow = np.sqrt(sse_pow/n)

# -------------------------------------------------
#  印出結果
# -------------------------------------------------
print("=== Least-Squares Approximations ===\n")

print("(a) Quadratic  y ≈ a0 + a1 x + a2 x²")
print(f"    a0 = {a0:.8f}\n    a1 = {a1:.8f}\n    a2 = {a2:.8f}")
print(f"    SSE = {sse_quad:.5e} ,  RMSE = {rmse_quad:.5e}\n")

print("(b) Exponential  y ≈ b e^ax")
print(f"    b  = {b_exp:.8f}\n    a  = {a_exp:.8f}")
print(f"    SSE = {sse_exp:.5e} ,  RMSE = {rmse_exp:.5e}\n")

print("(c) Power  y ≈ b x^a")
print(f"    b  = {b_pow:.8f}\n    a  = {a_pow:.8f}")
print(f"    SSE = {sse_pow:.5e} ,  RMSE = {rmse_pow:.5e}")
