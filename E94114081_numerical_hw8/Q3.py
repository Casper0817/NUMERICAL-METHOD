import numpy as np, math, textwrap

m = 16
n = 4
x = np.arange(m) / m
f = x**2 * np.sin(x)

a0 = np.sum(f) / m
ak = np.array([2*np.sum(f * np.cos(2*np.pi*k*x)) / m for k in range(1, n+1)])
bk = np.array([2*np.sum(f * np.sin(2*np.pi*k*x)) / m for k in range(1, n+1)])

def S4(z):
    z = np.asarray(z)
    val = a0 * np.ones_like(z)
    for k in range(1, n+1):
        val += ak[k-1]*np.cos(2*np.pi*k*z) + bk[k-1]*np.sin(2*np.pi*k*z)
    return val

integral_S4 = a0
integral_exact = math.cos(1) + 2*math.sin(1) - 2
abs_diff = abs(integral_S4 - integral_exact)
E_S4 = np.sum((f - S4(x))**2)

print("Discrete least‑squares trigonometric polynomial S4 (coefficients):")
print(f"  a0  = {a0:.10f}")
for k in range(1, n+1):
    print(f"  a{k:>1} = {ak[k-1]: .10f},   b{k:>1} = {bk[k-1]: .10f}")

expr_lines = [f"{a0:.10f}"]
for k in range(1, n+1):
    expr_lines.append(f"{ak[k-1]:+.10f} * cos(2π·{k}·x)")
    expr_lines.append(f"{bk[k-1]:+.10f} * sin(2π·{k}·x)")
poly_expr = " +\n    ".join(expr_lines)

print("\nS4(x) =")
print(textwrap.indent(poly_expr, "    "))

print(f"\n(b) ∫₀¹ S₄(x) dx = {integral_S4:.10f}")
print(f"(c) ∫₀¹ x² sin x dx = {integral_exact:.10f}")
print(f"    Absolute difference = {abs_diff:.2e}")

print(f"\n(d) Discrete least‑squares error  E(S4) = {E_S4:.10e}")
