"""Linear models: ridge, lasso and elastic net, collinearity diagnostics, logistic regression, least squares with equality constraints, data reconciliation, monotone regression, I-splines, and Huber and quantile regression."""

import numpy as np


def ridge_regression(X: np.ndarray, y: np.ndarray, lam: float = 0.0, penalize_intercept: bool = False) -> np.ndarray:
    """Solve (X^T X + lam * I_reg) w = X^T y stably."""
    if lam < 0.0:
        raise ValueError("Regularization parameter lam must be non-negative.")
    X_mat = np.asarray(X, dtype=float)
    y_vec = np.asarray(y, dtype=float)
    d = X_mat.shape[1]
    reg = lam * np.eye(d)
    if not penalize_intercept and d > 0:
        reg[0, 0] = 0.0
    return np.linalg.solve(X_mat.T @ X_mat + reg, X_mat.T @ y_vec)


def soft_threshold(z: np.ndarray | float, gamma: float) -> np.ndarray | float:
    """Exact proximal L1 soft-thresholding operator S(z, gamma) = sign(z) * max(|z| - gamma, 0)."""
    if gamma < 0.0:
        raise ValueError("Threshold gamma must be non-negative.")
    z_arr = np.asarray(z, dtype=float)
    res = np.sign(z_arr) * np.maximum(0.0, np.abs(z_arr) - gamma)
    return float(res) if np.ndim(z) == 0 else res


def lasso_coordinate_descent(
    X: np.ndarray, y: np.ndarray, alpha: float = 0.01, max_iter: int = 500, tol: float = 1e-9
) -> np.ndarray:
    """
    From-scratch Coordinate Descent solver for Lasso:
      min_w (1 / (2N)) ||X w - y||_2^2 + alpha * ||w_{1:}||_1
    using the exact soft-thresholding operator S(z, gamma) = sign(z) * max(|z| - gamma, 0).
    """
    if alpha < 0.0:
        raise ValueError("Lasso penalty alpha must be non-negative.")
    X_mat = np.asarray(X, dtype=float)
    y_vec = np.asarray(y, dtype=float).ravel()
    N, d = X_mat.shape
    w = np.zeros(d, dtype=float)
    col_norms_sq = np.sum(X_mat ** 2, axis=0) / N

    for _ in range(max_iter):
        w_old = w.copy()
        for j in range(d):
            if col_norms_sq[j] < 1e-15:
                continue
            r_j = y_vec - (X_mat @ w - X_mat[:, j] * w[j])
            rho_j = float(np.dot(X_mat[:, j], r_j)) / N
            if j == 0:
                w[j] = rho_j / col_norms_sq[j]
            else:
                w[j] = float(soft_threshold(rho_j, alpha)) / col_norms_sq[j]
        if np.max(np.abs(w - w_old)) < tol:
            break
    return w


def elastic_net_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.01,
    l1_ratio: float = 0.5,
    max_iter: int = 500,
    tol: float = 1e-9,
) -> np.ndarray:
    """
    Coordinate Descent solver for Elastic Net (Zou & Hastie, 2005):
      min_w (1 / (2N)) ||X w - y||_2^2 + alpha * l1_ratio * ||w_{1:}||_1 + (alpha * (1 - l1_ratio) / 2) * ||w_{1:}||_2^2
    Combines L1 sparsity with L2 grouping of correlated physical descriptors.
    """
    if alpha < 0.0 or not (0.0 <= l1_ratio <= 1.0):
        raise ValueError("Require alpha >= 0 and l1_ratio in [0, 1].")
    X_mat = np.asarray(X, dtype=float)
    y_vec = np.asarray(y, dtype=float).ravel()
    N, d = X_mat.shape
    w = np.zeros(d, dtype=float)
    col_norms_sq = np.sum(X_mat ** 2, axis=0) / N
    l1_pen = alpha * l1_ratio
    l2_pen = alpha * (1.0 - l1_ratio)

    for _ in range(max_iter):
        w_old = w.copy()
        for j in range(d):
            if col_norms_sq[j] < 1e-15:
                continue
            r_j = y_vec - (X_mat @ w - X_mat[:, j] * w[j])
            rho_j = float(np.dot(X_mat[:, j], r_j)) / N
            if j == 0:
                w[j] = rho_j / col_norms_sq[j]
            else:
                w[j] = float(soft_threshold(rho_j, l1_pen)) / (col_norms_sq[j] + l2_pen)
        if np.max(np.abs(w - w_old)) < tol:
            break
    return w


def polynomial_vandermonde(x: np.ndarray, degree: int) -> np.ndarray:
    """Construct the polynomial Vandermonde design matrix [1, x, x^2, ..., x^degree]."""
    if degree < 0:
        raise ValueError("Polynomial degree must be >= 0.")
    xv = np.asarray(x, dtype=float).ravel()
    return np.column_stack([xv ** k for k in range(degree + 1)])


def variance_inflation_factors(X_features: np.ndarray) -> np.ndarray:
    """
    Compute the Variance Inflation Factor VIF_j = (R_corr^(-1))_{jj} = 1 / (1 - R_j^2)
    for each standardized feature column j in X_features (excluding any constant intercept column).
    """
    X_mat = np.asarray(X_features, dtype=float)
    X_c = X_mat - np.mean(X_mat, axis=0)
    std = np.std(X_c, axis=0)
    X_std = X_c / np.where(std < 1e-15, 1.0, std)
    corr = (X_std.T @ X_std) / max(1, X_mat.shape[0])
    inv_corr = np.linalg.pinv(corr)
    return np.diag(inv_corr)


def ridge_gcv_score(X: np.ndarray, y: np.ndarray, lam: float, penalize_intercept: bool = False) -> dict:
    """
    Evaluate Ridge regression and its closed-form Generalized Cross-Validation (GCV) score (Golub, Heath & Wahba, 1979):
      GCV(lam) = (1 / N) * ||y - y_hat||_2^2 / (1 - df_eff(lam) / N)^2
    where df_eff(lam) = Tr(H_lam) = Tr(X (X^T X + lam * I_reg)^(-1) X^T) is the effective degrees of freedom.
    """
    X_mat = np.asarray(X, dtype=float)
    y_vec = np.asarray(y, dtype=float).ravel()
    N, d = X_mat.shape
    reg = lam * np.eye(d)
    if not penalize_intercept and d > 0:
        reg[0, 0] = 0.0
    A_inv = np.linalg.inv(X_mat.T @ X_mat + reg)
    w = A_inv @ (X_mat.T @ y_vec)
    y_hat = X_mat @ w
    df_eff = float(np.trace(X_mat @ A_inv @ X_mat.T))
    mse = float(np.mean((y_vec - y_hat) ** 2))
    denom = max(1e-12, (1.0 - df_eff / N) ** 2)
    return {"w": w, "df_eff": df_eff, "mse": mse, "gcv": mse / denom}


def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable logistic sigmoid function sigma(z) = 1 / (1 + exp(-z))."""
    z_arr = np.clip(np.asarray(z, dtype=float), -500.0, 500.0)
    res = 1.0 / (1.0 + np.exp(-z_arr))
    return float(res) if np.ndim(z) == 0 else res


def logistic_regression_newton(
    X: np.ndarray, y: np.ndarray, lam: float = 1e-4, max_iter: int = 50, tol: float = 1e-8
) -> dict:
    """
    Iteratively Reweighted Least Squares (IRLS / Newton-Raphson) solver for binary Logistic Regression
    classifying thermodynamic phase or safety regimes y in {0, 1}:
      min_w  -sum_i [y_i ln p_i + (1 - y_i) ln(1 - p_i)] + (lam / 2) ||w_{1:}||_2^2
    """
    X_mat = np.asarray(X, dtype=float)
    y_vec = np.asarray(y, dtype=float).ravel()
    N, d = X_mat.shape
    w = np.zeros(d, dtype=float)
    reg = lam * np.eye(d)
    if d > 0:
        reg[0, 0] = 0.0

    for _ in range(max_iter):
        p = np.asarray(sigmoid(X_mat @ w), dtype=float)
        grad = X_mat.T @ (p - y_vec) + reg @ w
        weights_r = np.maximum(p * (1.0 - p), 1e-8)
        H = (X_mat.T * weights_r) @ X_mat + reg
        step = np.linalg.solve(H, grad)
        w -= step
        if np.max(np.abs(step)) < tol:
            break

    probs = np.asarray(sigmoid(X_mat @ w), dtype=float)
    preds = (probs >= 0.5).astype(int)
    acc = float(np.mean(preds == y_vec.astype(int)))
    return {"w": w, "probabilities": probs, "predictions": preds, "accuracy": acc}


def equality_constrained_lstsq(
    X: np.ndarray, y: np.ndarray, C: np.ndarray, d: np.ndarray, lam: float = 1e-10
) -> dict:
    """
    Physics-constrained linear regression solving:
      min_w  (1/2) ||X w - y||_2^2 + (lam/2) ||w||_2^2   subject to   C w = d
    via the exact Karush-Kuhn-Tucker (KKT) saddle-point block matrix:
      [ X^T X + lam*I   C^T ] [ w  ] = [ X^T y ]
      [       C          0  ] [ mu ]   [   d   ]
    Guarantees exact conservation (atom balance, mass balance, enthalpy balance, or Gibbs-Duhem constraint).
    """
    X_mat = np.asarray(X, dtype=float)
    y_vec = np.asarray(y, dtype=float).ravel()
    C_mat = np.atleast_2d(np.asarray(C, dtype=float))
    d_vec = np.atleast_1d(np.asarray(d, dtype=float)).ravel()

    n_feat = X_mat.shape[1]
    n_cons = C_mat.shape[0]
    if C_mat.shape[1] != n_feat or len(d_vec) != n_cons:
        raise ValueError("Constraint dimensions C and d must match feature dimension of X.")

    H = X_mat.T @ X_mat + lam * np.eye(n_feat)
    KKT = np.zeros((n_feat + n_cons, n_feat + n_cons), dtype=float)
    KKT[:n_feat, :n_feat] = H
    KKT[:n_feat, n_feat:] = C_mat.T
    KKT[n_feat:, :n_feat] = C_mat

    rhs = np.concatenate([X_mat.T @ y_vec, d_vec])
    sol = np.linalg.solve(KKT, rhs)
    w = sol[:n_feat]
    mu = sol[n_feat:]
    cons_residual = float(np.max(np.abs(C_mat @ w - d_vec)))
    return {"w": w, "lagrange_multipliers": mu, "constraint_residual": cons_residual}


def pava_monotone_regression(y: np.ndarray, increasing: bool = True) -> np.ndarray:
    """
    Exact Pool-Adjacent-Violators Algorithm (PAVA) for L2 isotonic regression:
      min_f ||f - y||_2^2  subject to  f_1 <= f_2 <= ... <= f_n (or decreasing when increasing=False).
    Used to enforce thermodynamic mechanical stability (dP/dV <= 0) on noisy isotherm measurements.
    """
    vals = np.asarray(y, dtype=float).ravel()
    if not increasing:
        return -pava_monotone_regression(-vals, increasing=True)

    blocks: list[list[float]] = []
    for v in vals:
        blocks.append([float(v), 1.0])
        while len(blocks) >= 2 and (blocks[-2][0] / blocks[-2][1]) > (blocks[-1][0] / blocks[-1][1]):
            last = blocks.pop()
            blocks[-1][0] += last[0]
            blocks[-1][1] += last[1]

    out = np.empty_like(vals)
    idx = 0
    for total, count in blocks:
        cnt = int(round(count))
        out[idx : idx + cnt] = total / count
        idx += cnt
    return out


def crowe_data_reconciliation(y_meas: np.ndarray, sigma_meas: np.ndarray, A: np.ndarray, b: np.ndarray) -> dict:
    """
    Crowe steady-state weighted process data reconciliation solving:
      min_y (y - y_meas)^T Sigma^(-1) (y - y_meas)   subject to   A y = b
    with closed-form oblique projection:
      y_hat = y_meas - Sigma A^T (A Sigma A^T)^(-1) (A y_meas - b)
    and global chi-squared gross-error detection statistic:
      chi2_stat = r^T (A Sigma A^T)^(-1) r,   where r = A y_meas - b.
    """
    ym = np.asarray(y_meas, dtype=float).ravel()
    sig = np.asarray(sigma_meas, dtype=float).ravel()
    A_mat = np.atleast_2d(np.asarray(A, dtype=float))
    b_vec = np.atleast_1d(np.asarray(b, dtype=float)).ravel()

    var_vec = sig ** 2
    Sigma = np.diag(var_vec)
    r = A_mat @ ym - b_vec
    S_r = A_mat @ Sigma @ A_mat.T
    S_r_inv_r = np.linalg.solve(S_r, r)

    y_rec = ym - Sigma @ (A_mat.T @ S_r_inv_r)
    chi2_stat = float(np.dot(r, S_r_inv_r))
    # Standardized nodal test statistics z_j = r_j / sqrt((S_r)_jj)
    nodal_z = r / np.sqrt(np.maximum(np.diag(S_r), 1e-15))
    # Measurement test statistics m_i = |y_rec_i - ym_i| / sqrt(Var(y_rec_i - ym_i))
    adj_cov_diag = np.diag(Sigma @ A_mat.T @ np.linalg.solve(S_r, A_mat @ Sigma))
    meas_z = np.abs(y_rec - ym) / np.sqrt(np.maximum(adj_cov_diag, 1e-15))
    return {
        "y_reconciled": y_rec,
        "balance_residual_before": r,
        "balance_residual_after": A_mat @ y_rec - b_vec,
        "chi2_stat": chi2_stat,
        "dof": int(A_mat.shape[0]),
        "nodal_z": nodal_z,
        "measurement_z": meas_z,
    }


def ispline_basis_1d(x: np.ndarray, knots: np.ndarray, width: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Smooth non-negative bell-shaped M-spline basis M_m(x) >= 0 and cumulative monotone I-spline basis
    I_m(x) = int_{-inf}^x M_m(u) du in [0, 1] using exact Gaussian-error-function quadrature.
    Any linear combination f(x) = beta_0 + sum_{m=1}^M beta_m I_m(x) with beta_m >= 0 guarantees
    f'(x) = sum_{m=1}^M beta_m M_m(x) >= 0 everywhere.
    """
    import math

    xv = np.asarray(x, dtype=float).ravel()
    kv = np.asarray(knots, dtype=float).ravel()
    if width is None:
        width = float((kv[-1] - kv[0]) / max(1, len(kv) - 1)) * 0.75
    w = max(float(width), 1e-12)
    z = (xv[:, None] - kv[None, :]) / w
    M_mat = np.exp(-0.5 * (z ** 2)) / (np.sqrt(2.0 * np.pi) * w)
    vec_erf = np.vectorize(math.erf, otypes=[float])
    I_mat = 0.5 * (1.0 + vec_erf(z / np.sqrt(2.0)))
    return I_mat, M_mat


def fit_monotone_ispline(
    x: np.ndarray,
    y: np.ndarray,
    knots: np.ndarray,
    increasing: bool = True,
    lam: float = 1e-6,
    max_iter: int = 800,
) -> dict:
    """
    Fit a smooth monotone curve f(x) = beta_0 + sgn * sum_{m=1}^M beta_m I_m(x) with beta_m >= 0
    via projected coordinate descent on the non-negative orthant.
    """
    xv = np.asarray(x, dtype=float).ravel()
    yv = np.asarray(y, dtype=float).ravel()
    sgn = 1.0 if increasing else -1.0
    I_mat, M_mat = ispline_basis_1d(xv, knots)
    Phi = np.column_stack([np.ones_like(xv), sgn * I_mat])
    n, p = Phi.shape
    beta = np.zeros(p, dtype=float)
    beta[0] = float(np.mean(yv))
    col_sq = np.sum(Phi ** 2, axis=0) + np.concatenate([[0.0], np.full(p - 1, lam * n)])
    residual = yv - Phi @ beta

    for _ in range(max_iter):
        b_old = beta.copy()
        for j in range(p):
            old_bj = beta[j]
            rho_j = float(np.dot(Phi[:, j], residual)) + float(np.sum(Phi[:, j] ** 2)) * old_bj
            val = rho_j / col_sq[j]
            new_bj = val if j == 0 else max(0.0, val)
            if new_bj != old_bj:
                residual -= Phi[:, j] * (new_bj - old_bj)
                beta[j] = new_bj
        if np.max(np.abs(beta - b_old)) < 1e-10:
            break

    y_pred = Phi @ beta
    dy_dx = sgn * (M_mat @ beta[1:])
    return {"beta": beta, "y_pred": y_pred, "dy_dx": dy_dx, "knots": np.asarray(knots, dtype=float)}


def huber_irls_regression(
    X: np.ndarray, y: np.ndarray, delta: float = 1.35, max_iter: int = 100, tol: float = 1e-9
) -> dict:
    """
    Outlier-robust Huber M-estimation via Iteratively Reweighted Least Squares (IRLS)
    with adaptive weighting w_i = min(1, delta / |r_i|).
    """
    X_mat = np.asarray(X, dtype=float)
    y_vec = np.asarray(y, dtype=float).ravel()
    w = np.linalg.lstsq(X_mat, y_vec, rcond=None)[0]
    sample_weights = np.ones_like(y_vec)

    for _ in range(max_iter):
        w_old = w.copy()
        res = np.abs(y_vec - X_mat @ w)
        sample_weights = np.where(res <= delta, 1.0, delta / np.maximum(res, 1e-15))
        W_sqrt = np.sqrt(sample_weights)[:, None]
        w = np.linalg.lstsq(X_mat * W_sqrt, y_vec * W_sqrt.ravel(), rcond=None)[0]
        if np.max(np.abs(w - w_old)) < tol:
            break
    return {"w": w, "sample_weights": sample_weights, "residuals": y_vec - X_mat @ w}


def quantile_irls_regression(
    X: np.ndarray, y: np.ndarray, tau: float = 0.50, eps: float = 1e-4, max_iter: int = 150, tol: float = 1e-8
) -> np.ndarray:
    """
    Pinball check-loss quantile regression min_w sum_i rho_tau(y_i - x_i^T w)
    via majorization-minimization IRLS with asymmetric weights.
    """
    if not (0.0 < tau < 1.0):
        raise ValueError("Quantile level tau must lie in (0, 1).")
    X_mat = np.asarray(X, dtype=float)
    y_vec = np.asarray(y, dtype=float).ravel()
    w = np.linalg.lstsq(X_mat, y_vec, rcond=None)[0]

    for _ in range(max_iter):
        w_old = w.copy()
        r = y_vec - X_mat @ w
        asym = np.where(r >= 0.0, tau, 1.0 - tau)
        weights = asym / np.maximum(np.abs(r), eps)
        W_sqrt = np.sqrt(weights)[:, None]
        w = np.linalg.lstsq(X_mat * W_sqrt, y_vec * W_sqrt.ravel(), rcond=None)[0]
        if np.max(np.abs(w - w_old)) < tol:
            break
    return w



