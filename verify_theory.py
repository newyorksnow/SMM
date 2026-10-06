"""
Numerical verification of the claims used in the revised Section 3.1 of the paper.

Model:  k_i' = b_i exp(-l_i t) * w,  k_j' = b_j exp(-l_j t) * w,  w = a (|k_i-k_j| + 1)
D = k_j - k_i,   Phi(t) = a [ (b_j/l_j)(1-e^{-l_j t}) - (b_i/l_i)(1-e^{-l_i t}) ]
z = sgn(D) ln(1+|D|)   ->   claim: z(t) = z0 + Phi(t)  and  D = sgn(z)(e^{|z|}-1)
"""
import numpy as np
from scipy.integrate import solve_ivp

rng = np.random.default_rng(0)

def Phi(t, bi, li, bj, lj, a):
    return a*((bj/lj)*(1-np.exp(-lj*t)) - (bi/li)*(1-np.exp(-li*t)))

def z_of_D(D):  return np.sign(D)*np.log1p(np.abs(D))
def D_of_z(z):  return np.sign(z)*np.expm1(np.abs(z))

def simulate(bi, li, bj, lj, a, D0, T, n=4001, ki0=0.0):
    f = lambda t, y: [bi*np.exp(-li*t)*a*(abs(y[0]-y[1])+1),
                      bj*np.exp(-lj*t)*a*(abs(y[0]-y[1])+1)]
    ts = np.linspace(0, T, n)
    sol = solve_ivp(f, (0, T), [ki0, ki0+D0], t_eval=ts, rtol=1e-12, atol=1e-13, method="DOP853")
    return ts, sol.y[0], sol.y[1]

def closed_form_D(ts, bi, li, bj, lj, a, D0):
    return D_of_z(z_of_D(D0) + Phi(ts, bi, li, bj, lj, a))

# ---------------------------------------------------------------- 1. z-formula vs ODE, any sign of D0
# (checked in z-space and as a relative error: D itself can be astronomically large because D ~ e^{Phi})
worst_z = 0.0; worst_rel = 0.0
for _ in range(300):
    bi, bj = rng.uniform(0.05, 2, 2); li, lj = rng.uniform(0.3, 2, 2); a = rng.uniform(0.2, 2)
    D0 = rng.uniform(-2, 2)
    ts, ki, kj = simulate(bi, li, bj, lj, a, D0, T=25)
    Dsim = kj - ki
    zc = z_of_D(D0) + Phi(ts, bi, li, bj, lj, a)
    worst_z = max(worst_z, np.max(np.abs(z_of_D(Dsim) - zc)))
    Dc = D_of_z(zc)
    worst_rel = max(worst_rel, np.max(np.abs(Dsim - Dc)/(1 + np.abs(Dc))))
print(f"[1] z-formula vs ODE, 300 random sets, D0 in (-2,2): max|z_ode-(z0+Phi)| = {worst_z:.1e}; max rel. error in D = {worst_rel:.1e}")

# ---------------------------------------------------------------- 2. the counterexample to 'Phi_inf<=0 and D0>D0* => order preserved'
bi, li, bj, lj, a, D0 = 1.0, 1.0, 0.5, 0.6, 1.0, 0.2
Pinf = a*(bj/lj - bi/li); D0star = np.exp(-Pinf) - 1
ts, ki, kj = simulate(bi, li, bj, lj, a, D0, T=60, n=60001)
D = kj-ki
print(f"[2] counterexample: Phi_inf={Pinf:.4f}  D0*={D0star:.4f}  D0={D0}  (Phi_inf<=0: {Pinf<=0}, D0>D0*: {D0>D0star})")
print(f"    Condition (M) [b_j<=b_i and l_j>=l_i] holds? {bj<=bi and lj>=li}")
imin = D.argmin()
print(f"    min D(t) = {D.min():.4f} at t={ts[imin]:.3f};  D_inf = {D[-1]:.4f};  closed form (D0+1)e^Pinf-1 = {(D0+1)*np.exp(Pinf)-1:.4f}")
tmin = np.log(bi/bj)/(li-lj)          # where Phi' = 0
print(f"    argmin of Phi at t={tmin:.4f}, Phi_min={Phi(tmin,bi,li,bj,lj,a):.4f}, -ln(1+D0)={-np.log(1+D0):.4f}")
t_cross = ts[np.where(D < 0)[0][[0, -1]]]
print(f"    D<0 on t in [{t_cross[0]:.3f}, {t_cross[1]:.3f}]")

# ---------------------------------------------------------------- 3. Theorem 1 under (M)+(T): D>0 for all t, nonincreasing, limit (6)
bad = 0; N = 0
for _ in range(2000):
    bi = rng.uniform(0.05, 2); bj = rng.uniform(0.02, bi)              # b_j <= b_i
    li = rng.uniform(0.05, 2); lj = rng.uniform(li, li+2)               # l_j >= l_i
    a = rng.uniform(0.2, 3)
    Pinf = a*(bj/lj - bi/li); D0star = np.exp(-Pinf)-1
    if D0star > 1e6: continue          # beyond double-precision resolution of D_inf
    D0 = D0star + rng.uniform(1e-3, 3)
    ts = np.linspace(0, 400, 40001)
    Dc = closed_form_D(ts, bi, li, bj, lj, a, D0)
    N += 1
    ok = (Dc > 0).all() and (np.diff(Dc) <= 1e-12).all() and abs(Dc[-1] - ((D0+1)*np.exp(Pinf)-1)) < 1e-3*(1+abs(Dc[-1]))
    bad += (not ok)
print(f"[3] Theorem 1 under (M)+(T): {N-bad}/{N} random cases pass (D>0 for all t, nonincreasing, limit = Eq.(6))")

# ---------------------------------------------------------------- 4. Proposition 3 under (M), 0<D0<D0*
bad = 0; N = 0; dinf_err = 0
for _ in range(2000):
    bi = rng.uniform(0.05, 2); bj = rng.uniform(0.02, bi)
    li = rng.uniform(0.05, 2); lj = rng.uniform(li, li+2)
    a = rng.uniform(0.2, 3)
    Pinf = a*(bj/lj - bi/li)
    if Pinf > -1e-3: continue
    D0star = np.exp(-Pinf)-1
    D0 = rng.uniform(1e-3, 0.999)*D0star
    ts = np.linspace(0, 600, 120001)
    Dc = closed_form_D(ts, bi, li, bj, lj, a, D0)
    neg = Dc < 0
    # exactly one sign change, stays negative afterwards, D_inf = 1 - e^{-Pinf}/(1+D0) < 0
    sc = np.count_nonzero(np.diff(neg.astype(int)) != 0)
    Dinf = 1 - np.exp(-Pinf)/(1+D0)
    N += 1
    ok = (sc == 1) and neg[-1] and Dinf < 0 and (np.diff(Dc) <= 1e-12).all()
    dinf_err = max(dinf_err, abs(Dc[-1]-Dinf))
    bad += (not ok)
print(f"[4] Prop. 3 under (M), 0<D0<D0*: {N-bad}/{N} pass (single zero crossing, stays reversed, D_inf<0); max |D(T)-D_inf| = {dinf_err:.1e}")

# ---------------------------------------------------------------- 4b. D0 == D0*  -> equalisation, NOT reversal
bi, li, bj, lj, a = 1.0, 0.5, 0.4, 1.0, 1.0
Pinf = a*(bj/lj - bi/li); D0 = np.exp(-Pinf)-1
ts = np.linspace(0, 25, 2501); Dc = closed_form_D(ts, bi, li, bj, lj, a, D0)
print(f"[4b] D0=D0*={D0:.4f}: min D(t) over [0,25] = {Dc.min():.2e}  (>0 for all finite t: {(Dc>0).all()}, strictly decreasing: {(np.diff(Dc)<0).all()}),  D(25)={Dc[-1]:.2e} -> 0")

# ---------------------------------------------------------------- 5. Order reversal formula D_inf for D0<D0* (user's formula) incl. negative D0 (M-case)
bi, li, bj, lj, a = 1.0, 0.5, 0.4, 1.0, 1.0
Pinf = a*(bj/lj - bi/li)
for D0 in (0.1, 0.0, -0.3):
    ts, ki, kj = simulate(bi, li, bj, lj, a, D0, T=80)
    Dsim = (kj-ki)[-1]
    if D0 >= 0: Dform = 1 - np.exp(-Pinf)/(1+D0)
    else:       Dform = 1 - (1-D0)*np.exp(-Pinf)
    print(f"[5] D0={D0:+.1f}: ODE D(80)={Dsim:+.6f}   formula={Dform:+.6f}")

# ---------------------------------------------------------------- 6. Global bound on |D| and on the plateau of k_i (Prop. 2 proof)
worst_ratio = 0
for _ in range(300):
    bi, bj = rng.uniform(0.05, 2, 2); li, lj = rng.uniform(0.05, 2, 2); a = rng.uniform(0.2, 3)
    D0 = rng.uniform(-2, 2); ki0 = rng.uniform(0, 1)
    ts, ki, kj = simulate(bi, li, bj, lj, a, D0, T=80, ki0=ki0)
    z0 = abs(z_of_D(D0)); S = a*(bi/li + bj/lj)
    Dbar = np.exp(z0 + S) - 1                       # bound on |D|
    C = a*(Dbar + 1)
    assert np.max(np.abs(kj-ki)) <= Dbar + 1e-9
    plateau_bound_i = ki0 + C*bi/li
    plateau_bound_j = (ki0 + D0) + C*bj/lj
    assert ki[-1] <= plateau_bound_i + 1e-9 and kj[-1] <= plateau_bound_j + 1e-9
print("[6] |D| <= exp(|z0| + a(b_i/l_i + b_j/l_j)) - 1 and k_i(inf) <= k_i(0)+C b_i/l_i  hold in 300/300 random runs")

# ---------------------------------------------------------------- 7. equal-gain remark: Phi_inf=0, D_inf = D0, but D(t) not constant when l_i != l_j
bi, li, lj = 1.0, 1.0, 2.0; bj = lj*bi/li; a = 1.0; D0 = 0.5
ts = np.linspace(0, 40, 4001); Dc = closed_form_D(ts, bi, li, bj, lj, a, D0)
print(f"[7] equal gains (b/l equal, l_i!=l_j): D0={D0}, D(40)={Dc[-1]:.6f}, max D={Dc.max():.4f}, min D={Dc.min():.4f}  (non-constant)")
