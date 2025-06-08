import numpy as np
import math

# ---------------- parameters & grid ---------------------------
pi = math.pi
h  = 0.1 * pi          # grid spacing Δx = Δy
Nx = int(round(pi   / h))   # 10 intervals  → 11 nodes in x
Ny = int(round(pi/2 / h))   #  5 intervals  →  6 nodes in y

x_vals = np.linspace(0, pi,   Nx + 1)   # 0 … π
y_vals = np.linspace(0, pi/2, Ny + 1)   # 0 … π/2

# ---------------- boundary function ---------------------------
def boundary_u(i, j):
    """Boundary value at node indices (i,j)."""
    x = x_vals[i]
    y = y_vals[j]
    if i == 0:          # x = 0
        return math.cos(y)
    if i == Nx:         # x = π
        return -math.cos(y)
    if j == 0:          # y = 0
        return math.cos(x)
    if j == Ny:         # y = π/2
        return 0.0
    raise ValueError("Called boundary_u on interior node!")

# ---------------- helper for linear index ---------------------
def idx(i, j):
    """Map interior grid point (i=1..Nx-1, j=1..Ny-1) to 1-D index 0..n-1."""
    return (j - 1) * (Nx - 1) + (i - 1)

n_int = (Nx - 1) * (Ny - 1)        # total interior unknowns
A = np.zeros((n_int, n_int))
b = np.zeros(n_int)
inv_h2 = 1.0 / h**2

# ---------------- assemble A x = b ----------------------------
for j in range(1, Ny):          # interior y  (1 ~ Ny-1)
    for i in range(1, Nx):      # interior x  (1 ~ Nx-1)
        row = idx(i, j)

        # central coefficient
        A[row, row] = -4.0 * inv_h2

        # east  (i+1, j)
        if i + 1 <= Nx - 1:
            A[row, idx(i + 1, j)] = inv_h2
        else:   # neighbor is boundary at x = π
            b[row] -= inv_h2 * boundary_u(Nx, j)

        # west  (i-1, j)
        if i - 1 >= 1:
            A[row, idx(i - 1, j)] = inv_h2
        else:   # boundary at x = 0
            b[row] -= inv_h2 * boundary_u(0, j)

        # north (i, j+1)
        if j + 1 <= Ny - 1:
            A[row, idx(i, j + 1)] = inv_h2
        else:   # boundary at y = π/2
            b[row] -= inv_h2 * boundary_u(i, Ny)

        # south (i, j-1)
        if j - 1 >= 1:
            A[row, idx(i, j - 1)] = inv_h2
        else:   # boundary at y = 0
            b[row] -= inv_h2 * boundary_u(i, 0)

        # RHS f = x*y  evaluated at node (i,j)
        b[row] += x_vals[i] * y_vals[j]

# ---------------- solve linear system -------------------------
u_int = np.linalg.solve(A, b)

# ---------------- reconstruct full grid -----------------------
u = np.zeros((Ny + 1, Nx + 1))

# boundary nodes
for j in range(Ny + 1):
    u[j, 0]  = boundary_u(0, j)    # x = 0
    u[j, Nx] = boundary_u(Nx, j)   # x = π
for i in range(Nx + 1):
    u[0,  i] = boundary_u(i, 0)    # y = 0
    u[Ny, i] = boundary_u(i, Ny)   # y = π/2

# interior nodes
for j in range(1, Ny):
    for i in range(1, Nx):
        u[j, i] = u_int[idx(i, j)]

# ---------------- pretty print table --------------------------
print(f"\nFinite-difference solution  (h = k = {h:.3f})\n")
header = ["y\\x"] + [f"{x/pi:4.2f}π" for x in x_vals]   # x/π for compactness
print("  ".join(f"{h:>10}" for h in header))
for j, y in enumerate(y_vals):
    row = [f"{y/pi:4.2f}π"] + [f"{u[j,i]:10.6f}" for i in range(Nx+1)]
    print("  ".join(row))
