"""All experiments for the report. Run: python experiments.py

Saves arrays into results/ that make_figures.py turns into the report figures.

Protocols
  E1  clean recovery showcase: dt=0.001, T=25, finite differences, lam=0.05
  E2  noise sweep at dt=0.01: FD vs Savitzky-Golay, 20 noise draws per level
  E3a step size sweep, clean and mildly noisy (FD only)
  E3b trajectory length sweep at 1% noise (Savitzky-Golay)
  E4  threshold sweep on linear2d and lorenz, clean and 1% noise
  E5  library degree sweep on lorenz at several noise levels

The threshold rule everywhere except E1/E4: lam = half the smallest true
coefficient of the system, so the threshold is never the thing that fails.
Noise draws are paired across methods and sweep values (same 20 noisy
trajectories reused), which makes comparisons cleaner.
"""

import os

import numpy as np
from scipy.integrate import solve_ivp

from systems import SYSTEMS, simulate, add_noise
from sindy import (build_library, poly_terms, stlsq, deriv_fd, deriv_savgol,
                   trim, true_coef_matrix, coef_error, support_stats)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
NAMES = ["linear2d", "lotka", "lorenz"]
DEG = 3
TRIALS = 50
SG_WINDOW, SG_POLY = 21, 3
EDGE = 11  # samples trimmed at each end (> half the Savitzky-Golay window)


def lam_for(name):
    sys_ = SYSTEMS[name]
    smallest = min(abs(v) for coefs in sys_["true"].values() for v in coefs.values())
    return 0.5 * smallest


def fit(X, dt, lam, method="fd", degree=DEG):
    dX = deriv_fd(X, dt) if method == "fd" else deriv_savgol(X, dt, SG_WINDOW, SG_POLY)
    X_, dX_ = trim([X, dX], EDGE)
    Theta, terms = build_library(X_, degree)
    return stlsq(Theta, dX_, lam), terms


def rhs_from(Xi, degree):
    """Turn recovered coefficients back into a callable right-hand side."""
    def f(t, s):
        row, _ = build_library(np.asarray(s)[None, :], degree)
        return (row @ Xi).ravel()
    return f


def resimulate(Xi, degree, x0, t):
    sol = solve_ivp(rhs_from(Xi, degree), (t[0], t[-1]), x0, t_eval=t,
                    method="RK45", rtol=1e-10, atol=1e-10)
    return sol.y.T


def e1_clean():
    print("E1: clean recovery")
    out = {}
    for name in NAMES:
        sys_ = SYSTEMS[name]
        dt = 0.001
        t, X = simulate(name, T=25.0, dt=dt)
        Xi, terms = fit(X, dt, lam=0.05)
        Xi_true = true_coef_matrix(sys_, terms)
        Xr = resimulate(Xi, DEG, sys_["x0"], t)
        out[f"{name}_t"] = t
        out[f"{name}_X"] = X
        out[f"{name}_Xr"] = Xr
        out[f"{name}_Xi"] = Xi
        out[f"{name}_Xi_true"] = Xi_true
        out[f"{name}_err"] = coef_error(Xi, Xi_true)
        print(f"  {name}: err={out[f'{name}_err']:.2e} "
              f"exact={support_stats(Xi, Xi_true)['exact']}")
    np.savez(os.path.join(OUT, "E1.npz"), **out)


def e2_noise():
    print("E2: noise sweep")
    levels = np.logspace(-4, -1, 10)
    dt = 0.01
    rng = np.random.default_rng(0)
    med_err = np.zeros((len(NAMES), 2, len(levels)))
    success = np.zeros_like(med_err)
    for si, name in enumerate(NAMES):
        lam = lam_for(name)
        t, X = simulate(name, T=25.0, dt=dt)
        Xi_true = true_coef_matrix(SYSTEMS[name], poly_terms(SYSTEMS[name]["n"], DEG))
        noisy = [add_noise(X, 1.0, rng) for _ in range(TRIALS)]  # unit draws, scaled below
        base = [(n - X) for n in noisy]
        for li, level in enumerate(levels):
            for mi, method in enumerate(["fd", "savgol"]):
                errs, hits = [], 0
                for tr in range(TRIALS):
                    Xn = X + level * base[tr]
                    Xi, _ = fit(Xn, dt, lam, method)
                    errs.append(coef_error(Xi, Xi_true))
                    hits += support_stats(Xi, Xi_true)["exact"]
                med_err[si, mi, li] = np.median(errs)
                success[si, mi, li] = hits / TRIALS
        print(f"  {name} done (lam={lam:.3g})")
    np.savez(os.path.join(OUT, "E2.npz"), levels=levels, med_err=med_err,
             success=success, systems=np.array(NAMES), methods=np.array(["fd", "savgol"]))


def e3a_stepsize():
    print("E3a: step size sweep")
    dts = np.array([0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1])
    noise = 0.001
    rng = np.random.default_rng(1)
    err_clean = np.zeros((len(NAMES), len(dts)))
    err_noisy = np.zeros_like(err_clean)
    for si, name in enumerate(NAMES):
        lam = lam_for(name)
        Xi_true = true_coef_matrix(SYSTEMS[name], poly_terms(SYSTEMS[name]["n"], DEG))
        for di, dt in enumerate(dts):
            t, X = simulate(name, T=25.0, dt=dt)
            Xi, _ = fit(X, dt, lam, "fd")
            err_clean[si, di] = coef_error(Xi, Xi_true)
            errs = []
            for tr in range(TRIALS):
                Xn = add_noise(X, noise, rng)
                Xi, _ = fit(Xn, dt, lam, "fd")
                errs.append(coef_error(Xi, Xi_true))
            err_noisy[si, di] = np.median(errs)
        print(f"  {name} done")
    np.savez(os.path.join(OUT, "E3a.npz"), dts=dts, err_clean=err_clean,
             err_noisy=err_noisy, noise=noise, systems=np.array(NAMES))


def e3b_length():
    print("E3b: trajectory length sweep")
    Ts = np.array([2.0, 5.0, 10.0, 25.0, 50.0])
    noise, dt = 0.01, 0.01
    rng = np.random.default_rng(2)
    med_err = np.zeros((len(NAMES), len(Ts)))
    success = np.zeros_like(med_err)
    for si, name in enumerate(NAMES):
        lam = lam_for(name)
        Xi_true = true_coef_matrix(SYSTEMS[name], poly_terms(SYSTEMS[name]["n"], DEG))
        for ti, T in enumerate(Ts):
            t, X = simulate(name, T=T, dt=dt)
            errs, hits = [], 0
            for tr in range(TRIALS):
                Xn = add_noise(X, noise, rng)
                Xi, _ = fit(Xn, dt, lam, "savgol")
                errs.append(coef_error(Xi, Xi_true))
                hits += support_stats(Xi, Xi_true)["exact"]
            med_err[si, ti] = np.median(errs)
            success[si, ti] = hits / TRIALS
        print(f"  {name} done")
    np.savez(os.path.join(OUT, "E3b.npz"), Ts=Ts, med_err=med_err,
             success=success, noise=noise, systems=np.array(NAMES))


def e4_threshold():
    print("E4: threshold sweep")
    lams = np.logspace(-3, 0.7, 30)
    noise, dt = 0.01, 0.01
    rng = np.random.default_rng(3)
    names = ["linear2d", "lorenz"]
    n_terms = np.zeros((len(names), 2, len(lams)))     # clean, noisy
    err = np.zeros_like(n_terms)
    success = np.zeros_like(n_terms)
    for si, name in enumerate(names):
        t, X = simulate(name, T=25.0, dt=dt)
        Xi_true = true_coef_matrix(SYSTEMS[name], poly_terms(SYSTEMS[name]["n"], DEG))
        noisy = [add_noise(X, noise, rng) for _ in range(TRIALS)]
        for li, lam in enumerate(lams):
            Xi, _ = fit(X, dt, lam, "fd")
            n_terms[si, 0, li] = np.count_nonzero(Xi)
            err[si, 0, li] = coef_error(Xi, Xi_true)
            success[si, 0, li] = support_stats(Xi, Xi_true)["exact"]
            cnt, hits, errs = [], 0, []
            for tr in range(TRIALS):
                Xi, _ = fit(noisy[tr], dt, lam, "savgol")
                cnt.append(np.count_nonzero(Xi))
                errs.append(coef_error(Xi, Xi_true))
                hits += support_stats(Xi, Xi_true)["exact"]
            n_terms[si, 1, li] = np.mean(cnt)
            err[si, 1, li] = np.median(errs)
            success[si, 1, li] = hits / TRIALS
        print(f"  {name} done")
    np.savez(os.path.join(OUT, "E4.npz"), lams=lams, n_terms=n_terms, err=err,
             success=success, noise=noise, systems=np.array(names),
             true_active=np.array([sum(len(c) for c in SYSTEMS[n]["true"].values())
                                   for n in names]))


def e5_library():
    print("E5: library size sweep (lorenz)")
    degrees = [2, 3, 4, 5]
    levels = [0.0, 0.001, 0.01]
    dt = 0.01
    rng = np.random.default_rng(4)
    lam = lam_for("lorenz")
    t, X = simulate("lorenz", T=25.0, dt=dt)
    noisy = {lv: [add_noise(X, lv, rng) for _ in range(TRIALS)] for lv in levels if lv > 0}
    med_err = np.zeros((len(levels), len(degrees)))
    success = np.zeros_like(med_err)
    conds = np.zeros(len(degrees))
    n_lib = np.zeros(len(degrees), dtype=int)
    for di, deg in enumerate(degrees):
        terms = poly_terms(3, deg)
        n_lib[di] = len(terms)
        Xi_true = true_coef_matrix(SYSTEMS["lorenz"], terms)
        Theta, _ = build_library(trim([X], EDGE)[0], deg)
        conds[di] = np.linalg.cond(Theta)
        for li, lv in enumerate(levels):
            if lv == 0:
                Xi, _ = fit(X, dt, lam, "fd", degree=deg)
                med_err[li, di] = coef_error(Xi, Xi_true)
                success[li, di] = support_stats(Xi, Xi_true)["exact"]
            else:
                errs, hits = [], 0
                for tr in range(TRIALS):
                    Xi, _ = fit(noisy[lv][tr], dt, lam, "savgol", degree=deg)
                    errs.append(coef_error(Xi, Xi_true))
                    hits += support_stats(Xi, Xi_true)["exact"]
                med_err[li, di] = np.median(errs)
                success[li, di] = hits / TRIALS
        print(f"  degree {deg}: {len(terms)} terms, cond(Theta)={conds[di]:.2e}")
    np.savez(os.path.join(OUT, "E5.npz"), degrees=np.array(degrees),
             levels=np.array(levels), med_err=med_err, success=success,
             conds=conds, n_lib=n_lib)


def e6_orbit():
    """Lotka-Volterra identifiability: one closed orbit vs two.

    On a single orbit the library columns {1, y} come close to spanning the
    true dx/dt, so at low noise the solver sometimes swaps the true terms for
    that on-orbit surrogate. A second orbit breaks the degeneracy. At higher
    noise the failures come from derivative noise instead and stay.
    """
    print("E6: single-orbit identifiability (lotka)")
    dt = 0.01
    rng = np.random.default_rng(5)
    t, X = simulate("lotka", T=25.0, dt=dt)
    t2, X2 = simulate("lotka", T=25.0, dt=dt, x0=[5.0, 1.0])
    lam = lam_for("lotka")
    Xi_true = true_coef_matrix(SYSTEMS["lotka"], poly_terms(2, DEG))

    # how well (1, y) explains the true dx/dt along the sampled orbit
    x, y = X[:, 0], X[:, 1]
    f = x - 0.5 * x * y
    A = np.column_stack([np.ones_like(y), y])
    proj = A @ np.linalg.lstsq(A, f, rcond=None)[0]
    r2 = 1 - np.sum((f - proj) ** 2) / np.sum((f - f.mean()) ** 2)

    levels = [1e-4, 1e-2]
    trials = 100
    fails = np.zeros((len(levels), 2))
    for li, level in enumerate(levels):
        for tr in range(trials):
            Xn, Xn2 = add_noise(X, level, rng), add_noise(X2, level, rng)
            dX, dX2 = deriv_fd(Xn, dt), deriv_fd(Xn2, dt)
            A1, b1 = trim([Xn, dX], EDGE)
            A2, b2 = trim([Xn2, dX2], EDGE)
            Theta, _ = build_library(A1, DEG)
            Xi = stlsq(Theta, b1, lam)
            fails[li, 0] += not support_stats(Xi, Xi_true)["exact"]
            Theta2, _ = build_library(np.vstack([A1, A2]), DEG)
            Xi2 = stlsq(Theta2, np.vstack([b1, b2]), lam)
            fails[li, 1] += not support_stats(Xi2, Xi_true)["exact"]
        print(f"  sigma={level:g}: one orbit {int(fails[li, 0])}/{trials} fail, "
              f"two orbits {int(fails[li, 1])}/{trials} fail")
    np.savez(os.path.join(OUT, "E6.npz"), levels=np.array(levels),
             fails=fails, trials=trials, r2=r2)
    print(f"  R^2 of dx/dt on span(1, y) along one orbit: {r2:.3f}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    e1_clean()
    e2_noise()
    e3a_stepsize()
    e3b_length()
    e4_threshold()
    e5_library()
    e6_orbit()
    print("all experiments done ->", OUT)
