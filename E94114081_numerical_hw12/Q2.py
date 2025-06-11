
#!/usr/bin/env python3
import numpy as np

# Physical & grid parameters
K = 0.1
alpha = 4*K        # 0.4
dr = 0.1
r0, rN = 0.5, 1.0
M = int(round((rN-r0)/dr))       # 5 -> 6 nodes
r = np.linspace(r0, rN, M+1)

# Time params
dt_exp = 0.01      # forward Euler
dt_imp = 0.5       # backward & CN
t_max  = 10.0

inv_dr2 = 1.0/dr**2

def initial_T():
    return 200.0 * (r - 0.5)

def apply_bc(T, t_now):
    T[-1] = 100.0 + 40.0*t_now
    T[0]  = T[1] / (1.0 + 3.0*dr)

def laplacian(T, i):
    return ((T[i+1]-2*T[i]+T[i-1])*inv_dr2
            + (T[i+1]-T[i-1])/(2*r[i]*dr))

def run_forward():
    N_exp = int(round(t_max/dt_exp))
    T = initial_T()
    t_now = 0.0
    for _ in range(N_exp):
        Tn = T.copy()
        for i in range(1, M):
            Tn[i] = T[i] + dt_exp*alpha*laplacian(T, i)
        t_now += dt_exp
        apply_bc(Tn, t_now)
        T = Tn
    return T

def coeff_triplet(ri):
    A = alpha*dt_imp*(inv_dr2 - 1/(2*ri*dr))
    B = alpha*dt_imp*(-2*inv_dr2)
    C = alpha*dt_imp*(inv_dr2 + 1/(2*ri*dr))
    return A,B,C

def build_tridiag(theta):
    n = M-1
    a = np.zeros(n); b = np.zeros(n); c = np.zeros(n)
    for j in range(1, M):
        A,B,C = coeff_triplet(r[j])
        a[j-1] = -theta*A
        b[j-1] = 1 - theta*B
        c[j-1] = -theta*C
    return a,b,c

def thomas(a,b,c,d):
    n=len(b)
    for i in range(1,n):
        m = a[i-1]/b[i-1]
        b[i] -= m*c[i-1]
        d[i] -= m*d[i-1]
    x = np.zeros_like(d)
    x[-1] = d[-1]/b[-1]
    for i in range(n-2,-1,-1):
        x[i] = (d[i]-c[i]*x[i+1])/b[i]
    return x

def run_implicit(theta):
    a,b,c = build_tridiag(theta)
    N_imp = int(round(t_max/dt_imp))
    T = initial_T()
    t_now = 0.0
    for _ in range(N_imp):
        rhs = T[1:-1].copy()
        if theta < 1.0:
            for j in range(1, M):
                A,B,C = coeff_triplet(r[j])
                rhs[j-1] += (1-theta)*(A*T[j-1] + B*T[j] + C*T[j+1])
        rhs[0]  -= a[0]*T[0]
        rhs[-1] -= c[-1]*T[-1]
        T_int = thomas(a.copy(), b.copy(), c.copy(), rhs)
        T[1:-1] = T_int
        t_now += dt_imp
        apply_bc(T, t_now)
    return T

if __name__ == "__main__":
    Tf = run_forward()
    Tb = run_implicit(theta=1.0)
    Tc = run_implicit(theta=0.5)

    print("\nTemperature profile at t = 10 s")
    print("r    Forward(dt=0.01)  Backward(dt=0.5)  Crank-Nicolson")
    for i,ri in enumerate(r):
        print(f"{ri:.2f}  {Tf[i]:16.2f}  {Tb[i]:16.2f}  {Tc[i]:16.2f}")
