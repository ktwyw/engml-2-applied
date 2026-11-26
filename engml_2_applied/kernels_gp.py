"""Gaussian process regression: stationary kernels and their spectral densities, Cholesky regression with the log marginal likelihood, derivative observations and a two-level (multi-fidelity) model."""

import numpy as np


def rbf_kernel(X1: np.ndarray, X2: np.ndarray, length_scale: float = 1.0, sigma_f: float = 1.0) -> np.ndarray:
    """Squared-exponential (RBF) covariance matrix K_ij = sigma_f^2 * exp(-||x_i - x_j||^2 / (2 * l^2))."""
    if length_scale <= 0.0 or sigma_f <= 0.0:
        raise ValueError("length_scale and sigma_f must be positive.")
    A = np.atleast_2d(np.asarray(X1, dtype=float))
    B = np.atleast_2d(np.asarray(X2, dtype=float))
    sq_dist = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)
    return (sigma_f ** 2) * np.exp(-0.5 * sq_dist / (length_scale ** 2))


def matern32_kernel(X1: np.ndarray, X2: np.ndarray, length_scale: float = 1.0, sigma_f: float = 1.0) -> np.ndarray:
    """Matern (nu = 3/2) once-mean-square-differentiable covariance kernel."""
    if length_scale <= 0.0 or sigma_f <= 0.0:
        raise ValueError("length_scale and sigma_f must be positive.")
    A = np.atleast_2d(np.asarray(X1, dtype=float))
    B = np.atleast_2d(np.asarray(X2, dtype=float))
    dist = np.sqrt(np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1))
    s = np.sqrt(3.0) * dist / length_scale
    return (sigma_f ** 2) * (1.0 + s) * np.exp(-s)


def matern52_kernel(X1: np.ndarray, X2: np.ndarray, length_scale: float = 1.0, sigma_f: float = 1.0) -> np.ndarray:
    """Matern (nu = 5/2) twice-differentiable covariance kernel for physical surrogate surfaces."""
    if length_scale <= 0.0 or sigma_f <= 0.0:
        raise ValueError("length_scale and sigma_f must be positive.")
    A = np.atleast_2d(np.asarray(X1, dtype=float))
    B = np.atleast_2d(np.asarray(X2, dtype=float))
    dist = np.sqrt(np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1))
    s = np.sqrt(5.0) * dist / length_scale
    return (sigma_f ** 2) * (1.0 + s + (s ** 2) / 3.0) * np.exp(-s)


def ard_rbf_kernel(
    X1: np.ndarray, X2: np.ndarray, length_scales: np.ndarray, sigma_f: float = 1.0
) -> np.ndarray:
    """Anisotropic Automatic Relevance Determination (ARD) RBF covariance kernel across D dimensions."""
    ls = np.asarray(length_scales, dtype=float).ravel()
    if np.any(ls <= 0.0) or sigma_f <= 0.0:
        raise ValueError("All length_scales and sigma_f must be positive.")
    A = np.atleast_2d(np.asarray(X1, dtype=float)) / ls[None, :]
    B = np.atleast_2d(np.asarray(X2, dtype=float)) / ls[None, :]
    sq_dist = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)
    return (sigma_f ** 2) * np.exp(-0.5 * sq_dist)


def bochner_spectral_density_1d(omega: np.ndarray, length_scale: float = 1.0, kernel: str = "rbf") -> np.ndarray:
    """Analytical 1-D Bochner power spectral density S(omega) = int k(r) e^{-i omega r} dr."""
    w = np.asarray(omega, dtype=float)
    ell = float(length_scale)
    if kernel == "rbf":
        return np.sqrt(2.0 * np.pi) * ell * np.exp(-0.5 * (ell * w) ** 2)
    if kernel == "matern32":
        lam = np.sqrt(3.0) / ell
        return (4.0 * (lam ** 3)) / ((lam ** 2 + w ** 2) ** 2)
    if kernel == "matern52":
        lam = np.sqrt(5.0) / ell
        return (16.0 * (lam ** 5) / 3.0) / ((lam ** 2 + w ** 2) ** 3)
    raise ValueError(f"Unsupported kernel '{kernel}'.")


class GaussianProcessRegressor:
    """Exact Cholesky Gaussian Process Regressor with predictive mean, variance, and log marginal likelihood."""

    def __init__(
        self, length_scale: float = 1.0, sigma_f: float = 1.0, sigma_n: float = 1e-4, kernel: str = "rbf"
    ) -> None:
        self.length_scale = length_scale
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        self.kernel = kernel
        self.X_train_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None
        self.L_: np.ndarray | None = None
        self.alpha_: np.ndarray | None = None
        self.log_marginal_likelihood_: float = 0.0

    def _eval_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        if self.kernel == "rbf":
            return rbf_kernel(X1, X2, self.length_scale, self.sigma_f)
        if self.kernel == "matern32":
            return matern32_kernel(X1, X2, self.length_scale, self.sigma_f)
        if self.kernel == "matern52":
            return matern52_kernel(X1, X2, self.length_scale, self.sigma_f)
        raise ValueError(f"Unsupported kernel '{self.kernel}'.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcessRegressor":
        """Fit the model to the training data and return it."""
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
        y_arr = np.asarray(y, dtype=float).ravel()

        self.X_train_ = X_arr
        self.y_train_ = y_arr
        n = len(y_arr)
        K = self._eval_kernel(X_arr, X_arr) + (self.sigma_n ** 2) * np.eye(n)
        self.L_ = np.linalg.cholesky(K)
        z = np.linalg.solve(self.L_, y_arr)
        self.alpha_ = np.linalg.solve(self.L_.T, z)
        log_det = 2.0 * float(np.sum(np.log(np.diag(self.L_))))
        quad_term = float(np.dot(y_arr, self.alpha_))
        self.log_marginal_likelihood_ = float(-0.5 * quad_term - 0.5 * log_det - 0.5 * n * np.log(2.0 * np.pi))
        return self

    def predict(self, X_star: np.ndarray, return_std: bool = True) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Predict at the given inputs."""
        if self.X_train_ is None or self.L_ is None or self.alpha_ is None:
            raise RuntimeError("GP must be fitted before calling predict.")
        Xs = np.asarray(X_star, dtype=float)
        if Xs.ndim == 1:
            Xs = Xs[:, None]
        K_s = self._eval_kernel(self.X_train_, Xs)
        mu = K_s.T @ self.alpha_
        if not return_std:
            return mu
        v = np.linalg.solve(self.L_, K_s)
        k_ss_diag = (self.sigma_f ** 2) * np.ones(Xs.shape[0])
        var = np.maximum(0.0, k_ss_diag - np.sum(v ** 2, axis=0))
        return mu, np.sqrt(var)


def gp_1d_with_derivative_observations(
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_deriv: np.ndarray,
    dy_deriv: np.ndarray,
    x_star: np.ndarray,
    length_scale: float = 1.0,
    sigma_f: float = 1.0,
    sigma_n: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Physics-informed 1-D Gaussian Process that jointly assimilates function values y(x) and exact derivative
    observations dy/dx(x_d) (e.g., thermodynamic slope dG/dP = V or boundary condition du/dx = 0):
      k(x, x') = sigma_f^2 * exp(-(x - x')^2 / (2 l^2))
      Cov(y(x), y'(x')) = dk/dx' = ((x - x') / l^2) * k(x, x')
      Cov(y'(x), y'(x')) = d^2k/(dx dx') = (1/l^2 - (x - x')^2 / l^4) * k(x, x')
    """
    xv = np.asarray(x_val, dtype=float).ravel()
    yv = np.asarray(y_val, dtype=float).ravel()
    xd = np.asarray(x_deriv, dtype=float).ravel()
    dyd = np.asarray(dy_deriv, dtype=float).ravel()
    xs = np.asarray(x_star, dtype=float).ravel()
    l2 = length_scale ** 2
    l4 = l2 ** 2

    def k_ff(a, b):
        d = a[:, None] - b[None, :]
        return (sigma_f ** 2) * np.exp(-0.5 * (d ** 2) / l2)

    def k_fd(a, b):
        d = a[:, None] - b[None, :]
        return (d / l2) * k_ff(a, b)

    def k_dd(a, b):
        d = a[:, None] - b[None, :]
        return (1.0 / l2 - (d ** 2) / l4) * k_ff(a, b)

    K_vv = k_ff(xv, xv)
    K_vd = k_fd(xv, xd)
    K_dd = k_dd(xd, xd)
    K_joint = np.block([[K_vv, K_vd], [K_vd.T, K_dd]]) + (sigma_n ** 2) * np.eye(len(xv) + len(xd))
    y_joint = np.concatenate([yv, dyd])

    L = np.linalg.cholesky(K_joint)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_joint))

    K_s_joint = np.vstack([k_ff(xv, xs), k_fd(xs, xd).T])
    mu_star = K_s_joint.T @ alpha
    v = np.linalg.solve(L, K_s_joint)
    var_star = np.maximum(0.0, (sigma_f ** 2) - np.sum(v ** 2, axis=0))
    return mu_star, np.sqrt(var_star)


def multifidelity_cokriging_1d(
    x_low: np.ndarray,
    y_low: np.ndarray,
    x_high: np.ndarray,
    y_high_at_xhigh: np.ndarray,
    y_low_at_xhigh: np.ndarray,
    x_star: np.ndarray,
    length_scale_low: float = 0.8,
    length_scale_delta: float = 1.2,
    sigma_n: float = 1e-5,
) -> dict:
    """
    Kennedy-O'Hagan autoregressive multi-fidelity co-Kriging:
      f_H(x) = rho * f_L(x) + delta(x)
    where f_L(x) is a dense low-fidelity physical/DFT baseline and delta(x) is a sparse discrepancy GP.
    """
    xl = np.asarray(x_low, dtype=float).ravel()
    yl = np.asarray(y_low, dtype=float).ravel()
    xh = np.asarray(x_high, dtype=float).ravel()
    yh = np.asarray(y_high_at_xhigh, dtype=float).ravel()
    yl_h = np.asarray(y_low_at_xhigh, dtype=float).ravel()
    xs = np.asarray(x_star, dtype=float).ravel()

    gp_low = GaussianProcessRegressor(length_scale=length_scale_low, sigma_f=1.0, sigma_n=sigma_n, kernel="matern52")
    gp_low.fit(xl, yl)
    mu_low_star, std_low_star = gp_low.predict(xs)

    rho = float(np.dot(yl_h, yh) / max(float(np.dot(yl_h, yl_h)), 1e-15))
    delta_obs = yh - rho * yl_h
    gp_delta = GaussianProcessRegressor(
        length_scale=length_scale_delta, sigma_f=1.0, sigma_n=sigma_n, kernel="matern52"
    )
    gp_delta.fit(xh, delta_obs)
    mu_delta_star, std_delta_star = gp_delta.predict(xs)

    mu_high_star = rho * mu_low_star + mu_delta_star
    std_high_star = np.sqrt((rho ** 2) * (std_low_star ** 2) + (std_delta_star ** 2))
    return {"mu": mu_high_star, "std": std_high_star, "rho": rho, "mu_low": mu_low_star, "mu_delta": mu_delta_star}
