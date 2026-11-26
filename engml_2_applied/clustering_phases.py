"""Clustering: K-means with k-means++ seeding, the mean silhouette, spectral clustering with the normalized Laplacian, and Gaussian mixtures by the EM algorithm."""

import numpy as np


def kmeans_fit(X: np.ndarray, k: int = 3, max_iter: int = 100, seed: int = 42) -> dict:
    """K-Means++ initialization followed by Lloyd's expectation-maximization updates."""
    X_mat = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    N, d = X_mat.shape
    if not (1 <= k <= N):
        raise ValueError("Require 1 <= k <= N.")

    centers = np.zeros((k, d), dtype=float)
    centers[0] = X_mat[rng.integers(0, N)]
    for c_idx in range(1, k):
        dist_sq = np.min(np.sum((X_mat[:, None, :] - centers[None, :c_idx, :]) ** 2, axis=-1), axis=1)
        probs = dist_sq / max(1e-15, float(np.sum(dist_sq)))
        centers[c_idx] = X_mat[rng.choice(N, p=probs)]

    labels = np.zeros(N, dtype=int)
    for _ in range(max_iter):
        dists = np.sum((X_mat[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                centers[j] = np.mean(X_mat[mask], axis=0)

    inertia = float(np.sum((X_mat - centers[labels]) ** 2))
    return {"centers": centers, "labels": labels, "inertia": inertia}


def spectral_clustering(X: np.ndarray, k: int = 2, sigma: float = 1.0, seed: int = 42) -> dict:
    """
    Spectral clustering with the symmetric normalized Laplacian I - D^(-1/2) W D^(-1/2) of the graph of
    Gaussian similarities W: K-means on the row-normalized leading eigenvectors.
    """
    X_mat = np.asarray(X, dtype=float)
    sq_dist = np.sum((X_mat[:, None, :] - X_mat[None, :, :]) ** 2, axis=-1)
    W = np.exp(-sq_dist / (2.0 * (sigma ** 2)))
    np.fill_diagonal(W, 0.0)
    deg = np.sum(W, axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
    L_sym = np.eye(len(X_mat)) - d_inv_sqrt @ W @ d_inv_sqrt

    eigvals, eigvecs = np.linalg.eigh(L_sym)
    # Deterministic sign convention for eigenvectors:
    U = eigvecs[:, :k].copy()
    for j in range(k):
        idx_max = int(np.argmax(np.abs(U[:, j])))
        if U[idx_max, j] < 0.0:
            U[:, j] = -U[:, j]
    row_norms = np.linalg.norm(U, axis=1, keepdims=True)
    Y = U / np.maximum(row_norms, 1e-12)
    best_km = None
    for s_try in range(seed, seed + 25):
        km_try = kmeans_fit(Y, k=k, seed=s_try)
        if best_km is None or km_try["inertia"] < best_km["inertia"]:
            best_km = km_try
    assert best_km is not None
    return {"labels": best_km["labels"], "embedding": Y, "laplacian_eigenvalues": eigvals[:k + 2]}


def silhouette_score_mean(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Mean Rousseeuw Silhouette coefficient s = mean_i (b_i - a_i) / max(a_i, b_i) in [-1, 1]
    for model-free cluster count selection.
    """
    X_mat = np.asarray(X, dtype=float)
    lbl = np.asarray(labels, dtype=int).ravel()
    uniq = np.unique(lbl)
    if len(uniq) < 2 or len(uniq) >= len(X_mat):
        return 0.0
    dists = np.sqrt(np.sum((X_mat[:, None, :] - X_mat[None, :, :]) ** 2, axis=-1))
    s_vals = np.zeros(len(X_mat), dtype=float)
    for i in range(len(X_mat)):
        own = lbl == lbl[i]
        own[i] = False
        a_i = float(np.mean(dists[i, own])) if np.any(own) else 0.0
        b_i = min(float(np.mean(dists[i, lbl == c])) for c in uniq if c != lbl[i])
        denom = max(a_i, b_i)
        s_vals[i] = (b_i - a_i) / denom if denom > 1e-15 else 0.0
    return float(np.mean(s_vals))


def gmm_em_fit(X: np.ndarray, k: int = 2, max_iter: int = 80, tol: float = 1e-6, seed: int = 42) -> dict:
    """
    Diagonal-covariance Gaussian Mixture Model (GMM) fitted via Expectation-Maximization (EM)
    returning weights, means, variances, responsibilities and the log-likelihood.
    """
    X_mat = np.asarray(X, dtype=float)
    N, d = X_mat.shape
    km = kmeans_fit(X_mat, k=k, seed=seed)
    means = km["centers"].copy()
    variances = np.zeros((k, d), dtype=float)
    weights = np.zeros(k, dtype=float)
    for j in range(k):
        mask = km["labels"] == j
        weights[j] = max(float(np.mean(mask)), 1.0 / (2.0 * k))
        variances[j] = np.var(X_mat[mask], axis=0) + 1e-3 if np.any(mask) else np.var(X_mat, axis=0) + 1e-3
    weights /= np.sum(weights)
    resp = np.zeros((N, k), dtype=float)
    prev_ll = -np.inf

    for _ in range(max_iter):
        log_prob = np.zeros((N, k), dtype=float)
        for j in range(k):
            var_j = np.maximum(variances[j], 1e-9)
            diff_sq = ((X_mat - means[j]) ** 2) / var_j
            log_det = float(np.sum(np.log(2.0 * np.pi * var_j)))
            log_prob[:, j] = np.log(max(weights[j], 1e-15)) - 0.5 * (log_det + np.sum(diff_sq, axis=1))
        max_lp = np.max(log_prob, axis=1, keepdims=True)
        exp_lp = np.exp(log_prob - max_lp)
        sum_exp = np.sum(exp_lp, axis=1, keepdims=True)
        resp = exp_lp / np.maximum(sum_exp, 1e-15)
        ll = float(np.sum(max_lp.ravel() + np.log(np.maximum(sum_exp.ravel(), 1e-15))))
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

        nk = np.sum(resp, axis=0) + 1e-12
        weights = nk / N
        for j in range(k):
            means[j] = (resp[:, j] @ X_mat) / nk[j]
            variances[j] = (resp[:, j] @ ((X_mat - means[j]) ** 2)) / nk[j] + 1e-6

    return {
        "means": means,
        "variances": variances,
        "weights": weights,
        "responsibilities": resp,
        "labels": np.argmax(resp, axis=1),
        "log_likelihood": prev_ll,
    }

