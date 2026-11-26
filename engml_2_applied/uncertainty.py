"""Split conformal prediction intervals and the Mahalanobis distance of a query from a training set."""

import numpy as np


def split_conformal_interval(
    y_calib: np.ndarray, y_pred_calib: np.ndarray, y_pred_test: np.ndarray, alpha: float = 0.10
) -> dict:
    """
    Distribution-free Split Conformal Prediction (Vovk et al.; Lei et al.):
    Given calibration residuals r_i = |y_i - y_hat_i|, the finite-sample quantile level is
      q_hat = quantile(r, ceil((n + 1) * (1 - alpha)) / n),
    guaranteeing P(Y_test in [y_hat_test - q_hat, y_hat_test + q_hat]) >= 1 - alpha for exchangeable data.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("Mis-coverage level alpha must lie in (0, 1).")
    yc = np.asarray(y_calib, dtype=float).ravel()
    ypc = np.asarray(y_pred_calib, dtype=float).ravel()
    ypt = np.asarray(y_pred_test, dtype=float).ravel()
    n = len(yc)
    residuals = np.sort(np.abs(yc - ypc))
    k_idx = min(n - 1, max(0, int(np.ceil((n + 1) * (1.0 - alpha))) - 1))
    q_hat = float(residuals[k_idx])
    return {
        "q_hat": q_hat,
        "lower": ypt - q_hat,
        "upper": ypt + q_hat,
        "alpha": alpha,
    }


def mahalanobis_ood_score(X_train: np.ndarray, X_query: np.ndarray, reg: float = 1e-8) -> np.ndarray:
    """Compute squared Mahalanobis distance (x - mu)^T Sigma^(-1) (x - mu) for out-of-distribution detection."""
    Xt = np.atleast_2d(np.asarray(X_train, dtype=float))
    Xq = np.atleast_2d(np.asarray(X_query, dtype=float))
    mu = np.mean(Xt, axis=0)
    cov = np.cov(Xt, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    cov_reg = cov + reg * np.eye(cov.shape[0])
    diff = Xq - mu[None, :]
    inv_cov_diff = np.linalg.solve(cov_reg, diff.T).T
    return np.sum(diff * inv_cov_diff, axis=1)
