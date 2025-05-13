import numpy as np

# --- Problem data -----------------------------------------------------------
A = np.array([
    [ 4, -1,  0, -1,  0,  0],
    [-1,  4, -1,  0, -1,  0],
    [ 0, -1,  4,  0,  1, -1],   # 注意第 3 列 a35 = +1
    [-1,  0,  0,  4, -1, -1],
    [ 0, -1,  0, -1,  4, -1],
    [ 0,  0, -1,  0, -1,  4]
], dtype=float)

b = np.array([0, -1, 9, 4, 8, 6], dtype=float)

# --- Parameters -------------------------------------------------------------
tol       = 1.0e-10
max_iter  = 1000
omega     = 1.25                       # SOR relaxation factor
x0        = np.zeros_like(b)           # initial guess

# --- Helper functions -------------------------------------------------------
def jacobi(A, b, x0):
    D_inv = 1.0 / np.diag(A)
    R     = A - np.diagflat(np.diag(A))
    x     = x0.copy()
    for k in range(1, max_iter+1):
        x_new = D_inv * (b - R @ x)
        if np.linalg.norm(x_new - x, np.inf) < tol:
            return x_new, k
        x = x_new
    return x, max_iter

def gauss_seidel(A, b, x0):
    x = x0.copy()
    n = len(b)
    for k in range(1, max_iter+1):
        x_old = x.copy()
        for i in range(n):
            s = A[i, :i] @ x[:i] + A[i, i+1:] @ x[i+1:]
            x[i] = (b[i] - s) / A[i, i]
        if np.linalg.norm(x - x_old, np.inf) < tol:
            return x, k
    return x, max_iter

def sor(A, b, x0, w):
    x = x0.copy()
    n = len(b)
    for k in range(1, max_iter+1):
        x_old = x.copy()
        for i in range(n):
            s = A[i, :i] @ x[:i] + A[i, i+1:] @ x[i+1:]
            x[i] = x[i] + w * ((b[i] - s) / A[i, i] - x[i])
        if np.linalg.norm(x - x_old, np.inf) < tol:
            return x, k
    return x, max_iter

def conjugate_gradient(A, b, x0):
    x  = x0.copy()
    r  = b - A @ x
    p  = r.copy()
    rs = r @ r
    for k in range(1, max_iter+1):
        Ap = A @ p
        alpha = rs / (p @ Ap)
        x += alpha * p
        r -= alpha * Ap
        rs_new = r @ r
        if np.sqrt(rs_new) < tol:
            return x, k
        p  = r + (rs_new / rs) * p
        rs = rs_new
    return x, max_iter

# --- Solve ------------------------------------------------------------------
sol_jacobi, it_jacobi   = jacobi(A, b, x0)
sol_gs, it_gs           = gauss_seidel(A, b, x0)
sol_sor, it_sor         = sor(A, b, x0, omega)
sol_cg, it_cg           = conjugate_gradient(A, b, x0)

# --- Exact (reference) solution --------------------------------------------
x_exact = np.linalg.solve(A, b)

# --- Display ----------------------------------------------------------------
def pretty(name, sol, it):
    err = np.linalg.norm(sol - x_exact, np.inf)
    vec = np.array2string(sol, precision=8, floatmode='fixed')
    print(f"{name:<18s} iter={it:3d}  max|err|={err:.2e}  {vec}")

print(" Iterative solutions")
pretty("Jacobi",             sol_jacobi, it_jacobi)
pretty("Gauss-Seidel",       sol_gs,      it_gs)
pretty("SOR  (ω=1.25)",      sol_sor,     it_sor)
pretty("Conjugate Gradient", sol_cg,      it_cg)
print("\nExact solution      ", np.array2string(x_exact, precision=8, floatmode='fixed'))
