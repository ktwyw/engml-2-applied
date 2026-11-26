"""Dimensional homogeneity and the Buckingham Pi theorem, a corresponding-states vapour-pressure correlation, robust summary statistics, scalers and data splits."""

import numpy as np

from . import constants
from .descriptors import StandardScaler


def ideal_gas_molar_volume(T_K: np.ndarray | float, P_Pa: np.ndarray | float) -> np.ndarray | float:
    """Compute ideal-gas molar volume V_m = R * T / P [m^3/mol] in strict SI base units."""
    t_arr = np.asarray(T_K, dtype=float)
    p_arr = np.asarray(P_Pa, dtype=float)
    if np.any(t_arr <= 0.0) or np.any(p_arr <= 0.0):
        raise ValueError("Absolute temperature T_K and pressure P_Pa must be strictly positive.")
    vm = constants.R * t_arr / p_arr
    return float(vm) if (np.ndim(T_K) == 0 and np.ndim(P_Pa) == 0) else vm


def all_pairs_squared_distances(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """Compute all-pairs squared Euclidean distance matrix D_ij = ||x1_i - x2_j||_2^2 via 3-D array broadcasting."""
    a = np.atleast_2d(np.asarray(X1, dtype=float))
    b = np.atleast_2d(np.asarray(X2, dtype=float))
    diff = a[:, None, :] - b[None, :, :]
    return np.sum(diff ** 2, axis=-1)


def robust_eda_summary(y: np.ndarray, z_threshold: float = 3.5) -> dict:
    """
    Compute classical (mean, sample std) and outlier-resistant (median, IQR, MAD, modified Z-score) statistics.
    Modified Z-score (Iglewicz & Hoaglin, 1993): Z_mod = 0.6745 * |y - median(y)| / MAD.
    """
    arr = np.asarray(y, dtype=float).ravel()
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    med_val = float(np.median(arr))
    q25, q75 = np.percentile(arr, [25.0, 75.0])
    iqr_val = float(q75 - q25)
    mad_val = float(np.median(np.abs(arr - med_val)))
    safe_mad = max(mad_val, 1e-15)
    mod_z = 0.6745 * np.abs(arr - med_val) / safe_mad
    outliers = np.where(mod_z > z_threshold)[0]
    return {
        "mean": mean_val,
        "std": std_val,
        "median": med_val,
        "iqr": iqr_val,
        "mad": mad_val,
        "modified_z": mod_z,
        "outlier_indices": outliers,
    }


def verify_dimensional_homogeneity(D_matrix: np.ndarray, exponents: np.ndarray, tol: float = 1e-12) -> dict:
    """
    Verify whether a power-law product Pi = prod_j q_j^(alpha_j) is dimensionless: D @ alpha == 0.
    Rows of D_matrix correspond to fundamental SI dimensions (e.g., [M, L, T, Theta, N]),
    and columns correspond to physical variables q_j.
    """
    d_mat = np.asarray(D_matrix, dtype=float)
    alpha = np.asarray(exponents, dtype=float).ravel()
    residual_vec = d_mat @ alpha
    max_res = float(np.max(np.abs(residual_vec)))
    return {
        "is_dimensionless": bool(max_res <= tol),
        "residual_vector": residual_vec,
        "max_residual": max_res,
    }


def buckingham_pi_nullspace(D_matrix: np.ndarray, tol: float = 1e-10) -> dict:
    """
    Analyze the dimensional matrix D (n_dims x n_vars) via Singular Value Decomposition:
      n_pi = n_vars - rank(D)
    and return an orthonormal basis for the dimensionless Pi nullspace {alpha : D @ alpha = 0}.
    """
    d_mat = np.asarray(D_matrix, dtype=float)
    _, s_vals, vt = np.linalg.svd(d_mat, full_matrices=True)
    rank = int(np.sum(s_vals > tol))
    n_vars = d_mat.shape[1]
    n_pi = n_vars - rank
    null_basis = vt[rank:, :].T  # shape (n_vars, n_pi)
    max_res = float(np.max(np.abs(d_mat @ null_basis))) if n_pi > 0 else 0.0
    return {
        "n_vars": n_vars,
        "rank": rank,
        "n_pi": n_pi,
        "nullspace_basis": null_basis,
        "max_residual": max_res,
    }


def pitzer_vapor_pressure_reduced(T_r: np.ndarray | float, omega: np.ndarray | float) -> np.ndarray | float:
    """
    Lee-Kesler / Pitzer corresponding-states reduced vapor pressure correlation:
      ln(P_r^sat) = f^(0)(T_r) + omega * f^(1)(T_r)
    where:
      f^(0)(T_r) = 5.92714 - 6.09648 / T_r - 1.28862 * ln(T_r) + 0.169347 * T_r^6
      f^(1)(T_r) = 15.2518 - 15.6875 / T_r - 13.4721 * ln(T_r) + 0.43577 * T_r^6
    At the critical point T_r = 1.0, f^(0)(1) = 0 and f^(1)(1) = 0, so P_r^sat(1) = 1.0 identically.
    """
    tr = np.asarray(T_r, dtype=float)
    w = np.asarray(omega, dtype=float)
    if np.any(tr <= 0.0):
        raise ValueError("Reduced temperature T_r must be strictly positive.")
    f0 = 5.92714 - 6.09648 / tr - 1.28862 * np.log(tr) + 0.169347 * (tr ** 6)
    f1 = 15.2518 - 15.6875 / tr - 13.4721 * np.log(tr) + 0.43577 * (tr ** 6)
    pr = np.exp(f0 + w * f1)
    return float(pr) if (np.ndim(T_r) == 0 and np.ndim(omega) == 0) else pr


class MinMaxScaler:
    """Reversible [0, 1] Min-Max feature scaler with zero-variance column protection."""

    def __init__(self) -> None:
        self.min_: np.ndarray | None = None
        self.range_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        """Store the minimum and the range of each column; a constant column gets the range 1."""
        arr = np.asarray(X, dtype=float)
        self.min_ = np.min(arr, axis=0)
        span = np.max(arr, axis=0) - self.min_
        self.range_ = np.where(span < 1e-15, 1.0, span)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Map each column to [0, 1] with the stored minimum and range."""
        if self.min_ is None or self.range_ is None:
            raise RuntimeError("MinMaxScaler must be fitted first.")
        return (np.asarray(X, dtype=float) - self.min_) / self.range_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit to X and return the scaled X."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """Return scaled values to the original units."""
        if self.min_ is None or self.range_ is None:
            raise RuntimeError("MinMaxScaler must be fitted first.")
        return np.asarray(X_scaled, dtype=float) * self.range_ + self.min_


def leak_free_train_test_scale(X_train: np.ndarray, X_test: np.ndarray) -> dict:
    """
    Demonstrate leak-free preprocessing: fit StandardScaler strictly on X_train, then transform both X_train and X_test.
    """
    scaler = StandardScaler().fit(X_train)
    return {
        "scaler": scaler,
        "X_train_scaled": scaler.transform(X_train),
        "X_test_scaled": scaler.transform(X_test),
    }


def group_holdout_split(groups: np.ndarray, holdout_groups: list | np.ndarray) -> dict:
    """Partition sample indices into train and test sets by chemical family / homologous group label."""
    g_arr = np.asarray(groups)
    hold_mask = np.isin(g_arr, np.asarray(holdout_groups))
    return {
        "train_idx": np.where(~hold_mask)[0],
        "test_idx": np.where(hold_mask)[0],
    }
