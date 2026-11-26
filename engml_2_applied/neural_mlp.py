"""A network with one hidden layer of tanh units, with closed-form derivatives with respect to its input up to the third, and a least-squares fit of its output layer."""

import numpy as np


class DifferentiableMLP1D:
    """
    A 1-hidden-layer tanh neural network u_theta(x) = sum_j v_j * tanh(w_j * x + b_j) + c
    with exact analytical derivatives:
      - du/dx(x)  = sum_j v_j * w_j * sech^2(z_j)
      - d^2u/dx^2 = sum_j -2 * v_j * w_j^2 * tanh(z_j) * sech^2(z_j)
      - d^3u/dx^3 = sum_j -2 * v_j * w_j^3 * sech^2(z_j) * (1 - 3 * tanh^2(z_j))
    and convex Extreme Learning Machine (ELM) ridge solver for output weights (v, c).
    """

    def __init__(self, n_hidden: int = 16, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.n_hidden = n_hidden
        self.w = rng.normal(0.0, 1.0, size=n_hidden)
        self.b = rng.uniform(-1.0, 1.0, size=n_hidden)
        self.v = rng.normal(0.0, 0.5, size=n_hidden)
        self.c = 0.0

    def forward_and_derivatives(self, x: np.ndarray) -> dict:
        """Evaluate u(x), du/dx(x), d^2u/dx^2(x), and d^3u/dx^3(x) analytically at 1-D input array x."""
        xv = np.asarray(x, dtype=float).ravel()
        Z = xv[:, None] * self.w[None, :] + self.b[None, :]  # (N, H)
        th = np.tanh(Z)
        sech2 = 1.0 - (th ** 2)
        d_sech2 = -2.0 * th * sech2
        d2_sech2 = -2.0 * sech2 * (1.0 - 3.0 * (th ** 2))

        u = th @ self.v + self.c
        du_dx = (sech2 * self.w[None, :]) @ self.v
        d2u_dx2 = (d_sech2 * (self.w[None, :] ** 2)) @ self.v
        d3u_dx3 = (d2_sech2 * (self.w[None, :] ** 3)) @ self.v
        return {"u": u, "du_dx": du_dx, "d2u_dx2": d2u_dx2, "d3u_dx3": d3u_dx3, "th": th, "sech2": sech2}

    def fit_elm(self, x: np.ndarray, y: np.ndarray, lam: float = 1e-10) -> "DifferentiableMLP1D":
        """Fit linear output weights v and bias c via regularized Extreme Learning Machine (ELM) least squares."""
        xv = np.asarray(x, dtype=float).ravel()
        yv = np.asarray(y, dtype=float).ravel()
        fwd = self.forward_and_derivatives(xv)
        Phi = np.column_stack([fwd["th"], np.ones_like(xv)])
        m = Phi.shape[1]
        reg = lam * np.eye(m)
        reg[-1, -1] = 0.0
        theta = np.linalg.solve(Phi.T @ Phi + reg, Phi.T @ yv)
        self.v = theta[:-1]
        self.c = float(theta[-1])
        return self
