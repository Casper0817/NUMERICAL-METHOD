import numpy as np

# 系統矩陣 A 和常數項 b
A = np.array([
    [4, -1, 0, -1, 0, 0],
    [-1, 4, -1, 0, -1, 0],
    [0, -1, 4, 0, 0, -1],
    [-1, 0, 0, 4, -1, -1],
    [0, -1, 0, -1, 4, -1],
    [0, 0, -1, -1, -1, 4]
], dtype=float)

b = np.array([0, -1, 9, 4, 8, 6], dtype=float)

# 初始猜測
x0 = np.zeros_like(b)
tol = 1e-6
max_iter = 1000

# Jacobi Method
def jacobi(A, b, x0, tol, max_iter):
    x = x0.copy()
    D = np.diag(A)
    R = A - np.diagflat(D)
    for k in range(max_iter):
        x_new = (b - np.dot(R, x)) / D
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, k+1
        x = x_new
    return x, max_iter

# Gauss-Seidel Method
def gauss_seidel(A, b, x0, tol, max_iter):
    x = x0.copy()
    n = len(b)
    for k in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            sigma = np.dot(A[i,:i], x_new[:i]) + np.dot(A[i,i+1:], x[i+1:])
            x_new[i] = (b[i] - sigma) / A[i,i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, k+1
        x = x_new
    return x, max_iter

# SOR Method (ω = 1.25 是常見選擇)
def sor(A, b, x0, omega, tol, max_iter):
    x = x0.copy()
    n = len(b)
    for k in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            sigma = np.dot(A[i,:i], x_new[:i]) + np.dot(A[i,i+1:], x[i+1:])
            x_new[i] = x[i] + omega * ((b[i] - sigma) / A[i,i] - x[i])
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, k+1
        x = x_new
    return x, max_iter

# Conjugate Gradient Method
def conjugate_gradient(A, b, x0, tol, max_iter):
    x = x0.copy()
    r = b - A @ x
    p = r.copy()
    rs_old = np.dot(r, r)
    for k in range(max_iter):
        Ap = A @ p
        alpha = rs_old / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rs_new = np.dot(r, r)
        if np.sqrt(rs_new) < tol:
            return x, k+1
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x, max_iter

# 執行四種方法
jacobi_sol, jacobi_iter = jacobi(A, b, x0, tol, max_iter)
gs_sol, gs_iter = gauss_seidel(A, b, x0, tol, max_iter)
sor_sol, sor_iter = sor(A, b, x0, omega=1.25, tol=tol, max_iter=max_iter)
cg_sol, cg_iter = conjugate_gradient(A, b, x0, tol, max_iter)

# 顯示結果
def show_result(method, x, iters):
    print(f"{method} ({iters} iterations):")
    for i, val in enumerate(x):
        print(f"x{i+1} = {val:.6f}")
    print("-" * 40)

show_result("Jacobi", jacobi_sol, jacobi_iter)
show_result("Gauss-Seidel", gs_sol, gs_iter)
show_result("SOR", sor_sol, sor_iter)
show_result("Conjugate Gradient", cg_sol, cg_iter)
