"""Cross-check my implementation against PySINDy (optional, needs pysindy).

Same data, same library degree, same threshold. PySINDy's STLSQ adds a small
ridge penalty by default, so alpha=0 to match my plain least squares.
"""

import numpy as np
import pysindy as ps

from systems import SYSTEMS, simulate, add_noise
from sindy import build_library, stlsq, deriv_fd, trim, true_coef_matrix, coef_error


def compare(name, X, dt, lam, degree=3):
    dX = deriv_fd(X, dt)
    X_, dX_ = trim([X, dX], 11)
    Theta, terms = build_library(X_, degree)
    Xi_mine = stlsq(Theta, dX_, lam)

    model = ps.SINDy(
        optimizer=ps.STLSQ(threshold=lam, alpha=0.0),
        feature_library=ps.PolynomialLibrary(degree=degree),
        differentiation_method=ps.FiniteDifference(),
    )
    model.fit(X_, t=dt)
    Xi_ps = model.coefficients().T

    diff = np.abs(Xi_mine - Xi_ps).max()
    same_support = (Xi_mine != 0).astype(int).tolist() == (Xi_ps != 0).astype(int).tolist()
    print(f"{name}: max |coef difference| = {diff:.2e}, same support = {same_support}")
    return diff


if __name__ == "__main__":
    t, X = simulate("lorenz", T=25.0, dt=0.001)
    compare("lorenz clean (dt=0.001)", X, 0.001, lam=0.05)

    rng = np.random.default_rng(0)
    t, X = simulate("lotka", T=25.0, dt=0.01)
    compare("lotka 1% noise (dt=0.01)", add_noise(X, 0.01, rng), 0.01, lam=0.125)
