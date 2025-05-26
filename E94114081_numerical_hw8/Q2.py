import sympy as sp
import numpy as np

# ------------------------------------------------------------
#  1. 目標函數與區間
# ------------------------------------------------------------
x    = sp.symbols('x')
f    = sp.Rational(1,2)*sp.cos(x) + sp.Rational(1,4)*sp.sin(2*x)
a, b = -1, 1                     # 區間

# ------------------------------------------------------------
#  2. 使用 Legendre 正交基 (P0,P1,P2) 求最小平方係數
# ------------------------------------------------------------
P0, P1, P2 = sp.legendre(0, x), sp.legendre(1, x), sp.legendre(2, x)

def inner(g, h):
    """內積 ⟨g,h⟩ = ∫_{-1}^{1} g·h dx"""
    return sp.integrate(g*h, (x, a, b))

c0 = inner(f, P0) / inner(P0, P0)
c1 = inner(f, P1) / inner(P1, P1)
c2 = inner(f, P2) / inner(P2, P2)

# 將 p2 由 Legendre 形式 → 標準 1,x,x² 係數
p_leg  = c0*P0 + c1*P1 + c2*P2
p_poly = sp.expand(p_leg)                            # a0 + a1 x + a2 x²
a0 = sp.N(p_poly.coeff(x, 0), 12)
a1 = sp.N(p_poly.coeff(x, 1), 12)
a2 = sp.N(p_poly.coeff(x, 2), 12)

# ------------------------------------------------------------
#  3. 估算均方根誤差 (RMSE) 供參考
# ------------------------------------------------------------
f_num = sp.lambdify(x, f, "numpy")
p_num = sp.lambdify(x, a0 + a1*x + a2*x**2, "numpy")
xs    = np.linspace(a, b, 10_001)
rmse  = np.sqrt(np.mean((f_num(xs) - p_num(xs))**2))

# ------------------------------------------------------------
#  4. 顯示結果
# ------------------------------------------------------------
print("最小平方二次多項式  p₂(x) = a0 + a1 x + a2 x²")
print(f"a0 = {a0:.10f}")
print(f"a1 = {a1:.10f}")
print(f"a2 = {a2:.10f}")
print(f"\nRMSE ≈ {rmse:.5e}")
