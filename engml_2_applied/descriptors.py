"""Topological descriptors of molecular graphs (Wiener and Zagreb indices, Laplacian spectrum) and a standard scaler."""

import numpy as np


def path_graph_adjacency(n_nodes: int) -> np.ndarray:
    """Construct the adjacency matrix A of a linear chain graph with n_nodes heavy atoms (e.g., n-alkane)."""
    if n_nodes < 1:
        raise ValueError("n_nodes must be >= 1.")
    A = np.zeros((n_nodes, n_nodes), dtype=float)
    for i in range(n_nodes - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0
    return A


def star_graph_adjacency(n_nodes: int) -> np.ndarray:
    """Construct adjacency matrix A of a star-branched graph (1 central atom bonded to n_nodes - 1 leaves)."""
    if n_nodes < 1:
        raise ValueError("n_nodes must be >= 1.")
    A = np.zeros((n_nodes, n_nodes), dtype=float)
    for i in range(1, n_nodes):
        A[0, i] = 1.0
        A[i, 0] = 1.0
    return A


def shortest_path_distances(A: np.ndarray) -> np.ndarray:
    """All-pairs shortest-path topological distance matrix D via the Floyd-Warshall algorithm."""
    adj = np.asarray(A, dtype=float)
    n = adj.shape[0]
    D = np.full((n, n), np.inf, dtype=float)
    np.fill_diagonal(D, 0.0)
    D[adj > 0.0] = 1.0
    for k in range(n):
        D = np.minimum(D, D[:, [k]] + D[[k], :])
    return D


def graph_topological_descriptors(A: np.ndarray) -> dict:
    """
    Compute exact topological graph invariants from heavy-atom adjacency matrix A:
      - Wiener index W = (1/2) sum_{i,j} D_ij
      - First Zagreb index M1 = sum_i deg(i)^2
      - Second Zagreb index M2 = sum_{(i,j) in E} deg(i)*deg(j)
      - Randic connectivity index chi = sum_{(i,j) in E} 1 / sqrt(deg(i)*deg(j))
      - Graph Laplacian spectral gap (Fiedler eigenvalue lambda_2 of L = D_deg - A)
    """
    adj = np.asarray(A, dtype=float)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1] or not np.allclose(adj, adj.T):
        raise ValueError("Adjacency matrix A must be symmetric square.")
    n = adj.shape[0]
    deg = np.sum(adj > 0.0, axis=1).astype(float)
    D_sp = shortest_path_distances(adj)
    wiener = 0.5 * float(np.sum(D_sp))
    m1 = float(np.sum(deg ** 2))

    m2 = 0.0
    randic = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0.0:
                m2 += deg[i] * deg[j]
                if deg[i] > 0 and deg[j] > 0:
                    randic += 1.0 / np.sqrt(deg[i] * deg[j])

    L = np.diag(deg) - adj
    eigs = np.linalg.eigvalsh(L)
    fiedler = float(eigs[1]) if n >= 2 else 0.0
    return {
        "n_atoms": n,
        "wiener_index": wiener,
        "zagreb_m1": m1,
        "zagreb_m2": float(m2),
        "randic_index": float(randic),
        "fiedler_value": fiedler,
        "laplacian_eigenvalues": eigs,
    }


class StandardScaler:
    """Zero-mean, unit-variance feature standardizer with exact inverse transform."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """Store the mean and standard deviation of each column; a constant column gets the scale 1."""
        arr = np.asarray(X, dtype=float)
        self.mean_ = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        self.scale_ = np.where(std < 1e-15, 1.0, std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Subtract the stored means and divide by the stored scales."""
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler must be fitted first.")
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit to X and return the standardized X."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """Return standardized values to the original units."""
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler must be fitted first.")
        return np.asarray(X_scaled, dtype=float) * self.scale_ + self.mean_
