"""Preprocessing of spectra, principal components with Hotelling's T^2 and the residual Q, regression on principal components, and partial least squares (NIPALS, one response)."""

import numpy as np


def pca_svd(X: np.ndarray, n_components: int = 2) -> dict:
    """
    Center X and compute exact Principal Component Analysis via Singular Value Decomposition X_c = U S V^T.
    Also computes Hotelling's T^2 and Squared Prediction Error (Q / SPE) fault-detection metrics for each sample.
    """
    X_mat = np.asarray(X, dtype=float)
    N, d = X_mat.shape
    if not (1 <= n_components <= min(N, d)):
        raise ValueError("Invalid n_components for PCA.")
    mean = np.mean(X_mat, axis=0)
    X_c = X_mat - mean
    U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    eigvals = (S ** 2) / max(1, N - 1)
    total_var = float(np.sum(eigvals))
    explained_ratio = eigvals[:n_components] / total_var if total_var > 0 else np.zeros(n_components)

    P = Vt[:n_components, :].T  # loading matrix (d, k)
    scores = X_c @ P            # T = X_c P (N, k)
    X_hat = scores @ P.T
    residuals = X_c - X_hat

    # Hotelling's T^2_i = sum_{a=1}^k t_{i,a}^2 / lambda_a
    t2 = np.sum((scores ** 2) / np.maximum(eigvals[:n_components], 1e-15), axis=1)
    # Q (SPE) statistic = ||x_c,i - x_hat,i||_2^2
    q_spe = np.sum(residuals ** 2, axis=1)
    return {
        "mean": mean,
        "loadings": P,
        "scores": scores,
        "eigenvalues": eigvals[:n_components],
        "explained_variance_ratio": explained_ratio,
        "hotelling_t2": t2,
        "q_spe": q_spe,
        "reconstructed": X_hat + mean,
    }


def pls1_nipals(X: np.ndarray, y: np.ndarray, n_components: int = 2) -> dict:
    """
    From-scratch NIPALS Partial Least Squares (PLS1) regression for collinear spectroscopic data (X -> y).
    """
    X_mat = np.asarray(X, dtype=float)
    y_vec = np.asarray(y, dtype=float).ravel()
    x_mean = np.mean(X_mat, axis=0)
    y_mean = float(np.mean(y_vec))

    E = X_mat - x_mean
    f = y_vec - y_mean
    N, d = X_mat.shape

    W = np.zeros((d, n_components), dtype=float)
    P = np.zeros((d, n_components), dtype=float)
    T = np.zeros((N, n_components), dtype=float)
    q = np.zeros(n_components, dtype=float)

    for a in range(n_components):
        w_a = E.T @ f
        w_norm = np.linalg.norm(w_a)
        if w_norm < 1e-15:
            break
        w_a /= w_norm
        t_a = E @ w_a
        t_sq = float(np.dot(t_a, t_a))
        p_a = (E.T @ t_a) / t_sq
        q_a = float(np.dot(f, t_a)) / t_sq

        E = E - np.outer(t_a, p_a)
        f = f - q_a * t_a

        W[:, a] = w_a
        P[:, a] = p_a
        T[:, a] = t_a
        q[a] = q_a

    # Regression coefficient vector in original centered space: beta = W (P^T W)^(-1) q
    R_mat = W @ np.linalg.inv(P.T @ W)
    beta = R_mat @ q
    intercept = y_mean - float(np.dot(x_mean, beta))
    y_pred = X_mat @ beta + intercept
    return {"coef": beta, "intercept": intercept, "scores": T, "y_pred": y_pred}


def snv_preprocess(X: np.ndarray) -> np.ndarray:
    """
    Standard Normal Variate (SNV) spectral scatter correction (Barnes et al., 1989):
    row-centers each spectrum and divides by its row standard deviation to remove path-length
    and baseline offset variations.
    """
    X_mat = np.atleast_2d(np.asarray(X, dtype=float))
    means = np.mean(X_mat, axis=1, keepdims=True)
    stds = np.std(X_mat, axis=1, keepdims=True)
    return (X_mat - means) / np.where(stds < 1e-15, 1.0, stds)


def spectral_first_derivative(X: np.ndarray, d_lambda: float = 1.0) -> np.ndarray:
    """Central finite-difference first derivative along wavelength axis to eliminate constant and linear baselines."""
    X_mat = np.atleast_2d(np.asarray(X, dtype=float))
    return np.gradient(X_mat, float(d_lambda), axis=1)


def pcr_regression(X: np.ndarray, y: np.ndarray, n_components: int = 2) -> dict:
    """
    Principal Component Regression (PCR): project X onto top-k SVD principal components T_k = X_c P_k,
    regress centered target y_c onto T_k, and map regression weights back to original spectral coordinates.
    """
    X_mat = np.asarray(X, dtype=float)
    y_vec = np.asarray(y, dtype=float).ravel()
    pca = pca_svd(X_mat, n_components=n_components)
    T = pca["scores"]
    P = pca["loadings"]
    y_mean = float(np.mean(y_vec))
    y_c = y_vec - y_mean
    gamma = np.linalg.lstsq(T, y_c, rcond=None)[0]
    beta = P @ gamma
    intercept = y_mean - float(np.dot(pca["mean"], beta))
    return {
        "coef": beta,
        "intercept": intercept,
        "y_pred": X_mat @ beta + intercept,
        "scores": T,
        "explained_variance_ratio": pca["explained_variance_ratio"],
    }

