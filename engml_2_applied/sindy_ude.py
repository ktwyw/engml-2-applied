"""Sparse regression of rate laws (thresholded least squares, with derivatives or running integrals), the steady state of a stirred tank with a supplied rate, and its sensitivities."""

import numpy as np


def sindy_stlsq(Theta: np.ndarray, dX_dt: np.ndarray, threshold: float = 0.05, max_iter: int = 20) -> np.ndarray:
    """
    Sequentially Thresholded Least Squares (STLSQ) algorithm of Brunton, Proctor & Kutz (2016)
    for discovering sparse governing ODEs / chemical rate laws dX/dt = Theta(X) * Xi.
    """
    Lib = np.asarray(Theta, dtype=float)
    target = np.asarray(dX_dt, dtype=float)
    if target.ndim == 1:
        target = target[:, None]

    n_states = target.shape[1]
    Xi = np.linalg.lstsq(Lib, target, rcond=None)[0]

    for _ in range(max_iter):
        small_mask = np.abs(Xi) < threshold
        Xi[small_mask] = 0.0
        for s_idx in range(n_states):
            active = ~small_mask[:, s_idx]
            if np.any(active):
                Xi[active, s_idx] = np.linalg.lstsq(Lib[:, active], target[:, s_idx], rcond=None)[0]

    return Xi.ravel() if n_states == 1 else Xi


def integral_sindy_stlsq(
    t: np.ndarray, X: np.ndarray, Theta: np.ndarray, threshold: float = 0.05, max_iter: int = 20
) -> np.ndarray:
    """
    Noise-robust Integral SINDy formulation:
      X(t_k) - X(t_0) = ( int_{t_0}^{t_k} Theta(X(s)) ds ) * Xi
    Avoids numerical differentiation of noisy concentration trajectories by applying cumulative trapezoidal
    quadrature to the candidate monomial library Theta(X).
    """
    t_arr = np.asarray(t, dtype=float).ravel()
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr[:, None]
    Lib = np.asarray(Theta, dtype=float)
    dt = np.diff(t_arr)[:, None]
    avg_lib = 0.5 * (Lib[:-1, :] + Lib[1:, :])
    Int_Theta = np.vstack([np.zeros((1, Lib.shape[1])), np.cumsum(dt * avg_lib, axis=0)])
    delta_X = X_arr - X_arr[0:1, :]
    return sindy_stlsq(Int_Theta[1:], delta_X[1:], threshold=threshold, max_iter=max_iter)


def hybrid_cstr_steady_state_residual(
    C: np.ndarray | float,
    C_in: float,
    tau: float,
    rate_fn,
) -> np.ndarray | float:
    """
    Exact steady-state CSTR species conservation residual:
      R(C) = (C_in - C) / tau - r_hat(C) = 0
    pairing a macroscopic dilution shell with a learned constitutive rate closure r_hat(C).
    """
    if tau <= 0.0:
        raise ValueError("Residence time tau must be strictly positive.")
    c_arr = np.asarray(C, dtype=float)
    r_val = np.asarray(rate_fn(np.atleast_1d(c_arr)), dtype=float)
    res = (float(C_in) - c_arr) / float(tau) - (r_val[0] if np.ndim(C) == 0 else r_val.reshape(c_arr.shape))
    return float(res) if np.ndim(C) == 0 else res


def solve_hybrid_cstr_steady_state(
    C_in: float,
    tau: float,
    rate_fn,
    tol: float = 1e-12,
    max_iter: int = 80,
) -> dict:
    """
    Bracketed bisection solver for the hybrid grey-box CSTR balance (C_in - C)/tau - r_hat(C) = 0
    on the physical stoichiometric interval C in [0, C_in].
    """
    if tau <= 0.0:
        raise ValueError("Residence time tau must be strictly positive.")
    low, high = 0.0, float(C_in)
    it = 0
    for it in range(1, max_iter + 1):
        mid = 0.5 * (low + high)
        res_mid = float(hybrid_cstr_steady_state_residual(mid, C_in, tau, rate_fn))
        if abs(res_mid) <= tol or 0.5 * (high - low) <= tol:
            low = high = mid
            break
        if res_mid > 0.0:
            low = mid
        else:
            high = mid
    c_star = 0.5 * (low + high)
    r_star = float(np.asarray(rate_fn(np.array([c_star])), dtype=float)[0])
    residual = float(hybrid_cstr_steady_state_residual(c_star, C_in, tau, rate_fn))
    conversion = (float(C_in) - c_star) / float(C_in) if C_in > 0.0 else 0.0
    return {"C_out": c_star, "conversion": conversion, "rate": r_star, "residual": residual, "iterations": it}


def implicit_cstr_sensitivity(C_star: float, C_in: float, tau: float, dr_dC: float) -> dict:
    """
    Implicit Function Theorem (IFT) sensitivities of steady-state CSTR outlet concentration C^*
    satisfying g(C^*, C_in, tau) = (C_in - C^*)/tau - r(C^*) = 0:
      dC^*/dC_in = 1 / (1 + tau * dr/dC(C^*))
      dC^*/dtau  = -(C_in - C^*) / (tau * (1 + tau * dr/dC(C^*)))
    """
    denom = 1.0 + float(tau) * float(dr_dC)
    dC_dCin = 1.0 / denom
    dC_dtau = -(float(C_in) - float(C_star)) / (float(tau) * denom)
    return {"dC_dCin": dC_dCin, "dC_dtau": dC_dtau}
