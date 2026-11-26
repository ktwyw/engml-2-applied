"""Regression trees, random forests with permutation importance, and gradient boosting with the squared and the Huber loss."""

import numpy as np


class DecisionTreeRegressor:
    """Recursive variance-reduction CART regression tree in pure NumPy."""

    def __init__(self, max_depth: int = 3, min_samples_split: int = 4) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree_: dict | None = None

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> dict:
        n, d = X.shape
        mean_val = float(np.mean(y))
        if depth >= self.max_depth or n < self.min_samples_split or np.std(y) < 1e-12:
            return {"leaf": True, "value": mean_val}

        best_feat = -1
        best_thresh = 0.0
        best_sse = float(np.sum((y - mean_val) ** 2))

        for j in range(d):
            vals = np.unique(X[:, j])
            if len(vals) <= 1:
                continue
            thresholds = 0.5 * (vals[:-1] + vals[1:])
            for th in thresholds:
                left = X[:, j] <= th
                right = ~left
                if np.sum(left) < 1 or np.sum(right) < 1:
                    continue
                y_l, y_r = y[left], y[right]
                sse = float(np.sum((y_l - np.mean(y_l)) ** 2) + np.sum((y_r - np.mean(y_r)) ** 2))
                if sse < best_sse - 1e-12:
                    best_sse = sse
                    best_feat = j
                    best_thresh = float(th)

        if best_feat == -1:
            return {"leaf": True, "value": mean_val}

        left_mask = X[:, best_feat] <= best_thresh
        return {
            "leaf": False,
            "feature": best_feat,
            "threshold": best_thresh,
            "left": self._build(X[left_mask], y[left_mask], depth + 1),
            "right": self._build(X[~left_mask], y[~left_mask], depth + 1),
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeRegressor":
        """Fit the model to the training data and return it."""
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
        self.tree_ = self._build(X_arr, np.asarray(y, dtype=float).ravel(), 0)
        return self

    def _pred_one(self, node: dict, x_row: np.ndarray) -> float:
        if node["leaf"]:
            return node["value"]
        if x_row[node["feature"]] <= node["threshold"]:
            return self._pred_one(node["left"], x_row)
        return self._pred_one(node["right"], x_row)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict at the given inputs."""
        if self.tree_ is None:
            raise RuntimeError("DecisionTreeRegressor must be fitted first.")
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
        return np.array([self._pred_one(self.tree_, row) for row in X_arr], dtype=float)


class GradientBoostingRegressor:
    """L2 Functional Gradient Boosted Regression Trees from scratch."""

    def __init__(self, n_estimators: int = 25, learning_rate: float = 0.15, max_depth: int = 2) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.f0_: float = 0.0
        self.trees_: list[DecisionTreeRegressor] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingRegressor":
        """Fit the model to the training data and return it."""
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
        y_arr = np.asarray(y, dtype=float).ravel()
        self.f0_ = float(np.mean(y_arr))
        pred = np.full_like(y_arr, self.f0_)
        self.trees_ = []
        for _ in range(self.n_estimators):
            residual = y_arr - pred
            tree = DecisionTreeRegressor(max_depth=self.max_depth).fit(X_arr, residual)
            pred += self.learning_rate * tree.predict(X_arr)
            self.trees_.append(tree)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict at the given inputs."""
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
        pred = np.full(X_arr.shape[0], self.f0_, dtype=float)
        for tree in self.trees_:
            pred += self.learning_rate * tree.predict(X_arr)
        return pred


def prefix_sum_best_split_1d(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Exact O(N log N) sorting + O(1) prefix-sum scan for optimal 1-D CART variance-reduction split:
      Gain(s) = S_L^2 / n_L + S_R^2 / n_R - S_total^2 / N.
    """
    xv = np.asarray(x, dtype=float).ravel()
    yv = np.asarray(y, dtype=float).ravel()
    order = np.argsort(xv)
    xs = xv[order]
    ys = yv[order]
    n = len(ys)
    prefix_s = np.cumsum(ys)
    s_tot = float(prefix_s[-1])
    sst = float(np.sum((ys - np.mean(ys)) ** 2))

    best_gain = -1.0
    best_th = float(xs[0])
    for i in range(1, n):
        if xs[i] == xs[i - 1]:
            continue
        s_l = float(prefix_s[i - 1])
        s_r = s_tot - s_l
        gain = (s_l ** 2) / i + (s_r ** 2) / (n - i) - (s_tot ** 2) / n
        if gain > best_gain:
            best_gain = gain
            best_th = 0.5 * float(xs[i - 1] + xs[i])
    return {"threshold": best_th, "sse_reduction": max(0.0, best_gain), "sse_after": max(0.0, sst - best_gain)}


class RandomForestRegressor:
    """Bootstrap-Aggregated (Bagged) ensemble of CART DecisionTreeRegressors with permutation importance."""

    def __init__(self, n_estimators: int = 20, max_depth: int = 4, min_samples_split: int = 3, seed: int = 42) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.seed = seed
        self.trees_: list[DecisionTreeRegressor] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestRegressor":
        """Fit the model to the training data and return it."""
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
        y_arr = np.asarray(y, dtype=float).ravel()
        n = len(y_arr)
        rng = np.random.default_rng(self.seed)
        self.trees_ = []
        for _ in range(self.n_estimators):
            boot_idx = rng.integers(0, n, size=n)
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth, min_samples_split=self.min_samples_split
            ).fit(X_arr[boot_idx], y_arr[boot_idx])
            self.trees_.append(tree)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict at the given inputs."""
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
        preds = np.column_stack([t.predict(X_arr) for t in self.trees_])
        return np.mean(preds, axis=1)


def permutation_feature_importance(model, X: np.ndarray, y: np.ndarray, seed: int = 42) -> np.ndarray:
    """Compute Breiman permutation feature importance as increase in MSE when column j is randomly shuffled."""
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).ravel()
    base_mse = float(np.mean((model.predict(X_arr) - y_arr) ** 2))
    rng = np.random.default_rng(seed)
    importances = np.zeros(X_arr.shape[1], dtype=float)
    for j in range(X_arr.shape[1]):
        X_perm = X_arr.copy()
        X_perm[:, j] = rng.permutation(X_perm[:, j])
        perm_mse = float(np.mean((model.predict(X_perm) - y_arr) ** 2))
        importances[j] = max(0.0, perm_mse - base_mse)
    return importances


class HuberGradientBoostingRegressor:
    """Outlier-robust Functional Gradient Boosting using clipped Huber pseudo-residuals."""

    def __init__(
        self,
        n_estimators: int = 25,
        learning_rate: float = 0.15,
        max_depth: int = 2,
        delta: float = 1.0,
        min_samples_split: int = 6,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.delta = delta
        self.min_samples_split = min_samples_split
        self.f0_: float = 0.0
        self.trees_: list[DecisionTreeRegressor] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HuberGradientBoostingRegressor":
        """Fit the model to the training data and return it."""
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
        y_arr = np.asarray(y, dtype=float).ravel()
        self.f0_ = float(np.median(y_arr))
        pred = np.full_like(y_arr, self.f0_)
        self.trees_ = []
        for _ in range(self.n_estimators):
            raw_res = y_arr - pred
            pseudo_res = np.clip(raw_res, -self.delta, self.delta)
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth, min_samples_split=self.min_samples_split
            ).fit(X_arr, pseudo_res)
            pred += self.learning_rate * tree.predict(X_arr)
            self.trees_.append(tree)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict at the given inputs."""
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
        pred = np.full(X_arr.shape[0], self.f0_, dtype=float)
        for tree in self.trees_:
            pred += self.learning_rate * tree.predict(X_arr)
        return pred

