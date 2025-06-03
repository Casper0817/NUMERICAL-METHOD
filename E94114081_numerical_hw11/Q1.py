import numpy as np
from scipy.integrate import quad

# -------------------------------------------------------------------
# Common data
h   = 0.1                          # mesh step for tabulated output
xs  = np.arange(0.0, 1.0 + h, h)   # 0,0.1,...,1
# -------------------------------------------------------------------
# (a) Shooting -------------------------------------------------------
def ode_f(x, y):
    """Return derivatives [y', y''] for state y=[y, y']"""
    y1, y2 = y
    dy1 = y2
    dy2 = -(x + 1) * y2 + 2 * y1 + (1 - x**2) * np.exp(-x)
    return np.array([dy1, dy2])

def rk4_step(x, y, h):
    k1 = ode_f(x, y)
    k2 = ode_f(x + 0.5*h, y + 0.5*h*k1)
    k3 = ode_f(x + 0.5*h, y + 0.5*h*k2)
    k4 = ode_f(x + h,     y + h*k3)
    return y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

def integrate_given_slope(s, xgrid):
    y = np.array([1.0, s])          # y(0)=1 , y'(0)=s
    sol = [y[0]]
    for k in range(1, len(xgrid)):
        h = xgrid[k] - xgrid[k-1]
        y = rk4_step(xgrid[k-1], y, h)
        sol.append(y[0])
    return np.array(sol)

def shooting_method(xgrid, tol=1e-10, max_iter=25):
    s0, s1 = 0.0, 5.0                       # initial slope guesses
    y0 = integrate_given_slope(s0, xgrid)
    y1 = integrate_given_slope(s1, xgrid)
    f0 = y0[-1] - 2.0
    f1 = y1[-1] - 2.0
    for _ in range(max_iter):
        if abs(f1 - f0) < 1e-14:            # avoid divide‑by‑zero
            break
        s  = s1 - f1 * (s1 - s0) / (f1 - f0)   # secant update
        ys = integrate_given_slope(s, xgrid)
        f  = ys[-1] - 2.0
        if abs(f) < tol:
            return ys
        # shift
        s0, f0, s1, f1, y0, y1 = s1, f1, s, f, y1, ys
    return ys                                # last iterate anyway
y_shoot = shooting_method(xs)

# -------------------------------------------------------------------
# (b) Finite‑Difference (second‑order central) -----------------------
def finite_difference_solution(h):
    x = np.arange(0.0, 1.0 + h, h)
    n = len(x)
    A = np.zeros((n, n))
    b = np.zeros(n)

    # Boundary rows
    A[0,0]   = 1.0
    b[0]     = 1.0
    A[-1,-1] = 1.0
    b[-1]    = 2.0

    for i in range(1, n-1):
        xi = x[i]
        p  =  (xi + 1)           # coefficient of y'
        q  = -2.0
        r  =  (1 - xi**2) * np.exp(-xi)

        A[i, i-1] = 1.0 - (h/2)*p
        A[i, i]   = -2.0 + h**2 * q
        A[i, i+1] = 1.0 + (h/2)*p
        b[i]      = h**2 * r

    return np.linalg.solve(A, b)

y_fd = finite_difference_solution(h)

# -------------------------------------------------------------------
# (c) Galerkin Variation (sine basis) -------------------------------
def variation_solution(N=3):
    """Return y values at xs using N sine basis functions"""
    # basis functions and derivatives
    def phi(k, x):  # k>=1
        return np.sin(k*np.pi*x)
    def dphi(k,x):
        return k*np.pi*np.cos(k*np.pi*x)
    def ddphi(k,x):
        return -(k*np.pi)**2 * np.sin(k*np.pi*x)

    # Assemble matrix A and RHS b  (A_ij = <L phi_j, phi_i>)
    A = np.zeros((N, N))
    b = np.zeros(N)

    # helper: operator L acting on function g
    def L_on_phi(k, x):
        return ddphi(k,x) + (x+1)*dphi(k,x) - 2*phi(k,x)

    # constant part (from 1 + x) moved to RHS:
    def rhs_integrand(x):
        return (1 - x**2)*np.exp(-x) + 1 + x     # note + sign (see analysis)

    for i in range(1, N+1):
        # compute RHS i
        b[i-1], _ = quad(lambda t: rhs_integrand(t)*phi(i, t), 0, 1)
        for j in range(1, N+1):
            A[i-1, j-1], _ = quad(
                lambda t: L_on_phi(j, t) * phi(i, t),
                0, 1
            )

    c = np.linalg.solve(A, b)   # coefficients

    # evaluate y on xs
    y = 1 + xs                  # baseline 1 + x
    for k in range(1, N+1):
        y += c[k-1] * phi(k, xs)
    return y

y_var = variation_solution(N=3)

# -------------------------------------------------------------------
# tabulate to console
if __name__ == "__main__":
    print("\nSolved BVP (h = 0.1)\n")
    print("   x      Shooting       FD         Variation")
    for xi, ys, yf, yv in zip(xs, y_shoot, y_fd, y_var):
        print(f" {xi:4.2f}   {ys:10.6f}  {yf:10.6f}  {yv:10.6f}")
