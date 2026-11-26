"""Collocation solvers for linear second-order boundary-value problems on smooth basis functions, and a search for the Thiele modulus from interior values."""

import numpy as np


def solve_linear_bvp_pinn(
    x_colloc: np.ndarray,
    centers: np.ndarray,
    sigma: float,
    q_coeff_fn,
    rhs_fn,
    x_bc_left: float,
    bc_left_type: str,
    bc_left_val: float,
    x_bc_right: float,
    bc_right_type: str,
    bc_right_val: float,
    bc_weight: float = 1000.0,
) -> dict:
    """
    Physics-Informed Radial Basis Function (RBF) Collocation Network solving the 2nd-order linear BVP:
      u''(x) + q(x) * u(x) = f(x)   on [x_L, x_R]
    with trial network u_hat(x) = sum_{j=1}^M w_j * phi_j(x), where phi_j(x) = exp(-(x - c_j)^2 / (2 * sigma^2)).
    Exact analytical derivatives of the basis functions are used to assemble the physics residual matrix at
    interior collocation points and boundary conditions at the endpoints.
    """
    xc = np.asarray(x_colloc, dtype=float).ravel()
    c = np.asarray(centers, dtype=float).ravel()
    s2 = sigma ** 2
    s4 = s2 ** 2

    def basis_eval(x_pts: np.ndarray):
        d = x_pts[:, None] - c[None, :]
        phi = np.exp(-0.5 * (d ** 2) / s2)
        dphi = -(d / s2) * phi
        d2phi = ((d ** 2) / s4 - 1.0 / s2) * phi
        return phi, dphi, d2phi

    phi_c, _, d2phi_c = basis_eval(xc)
    q_vals = np.asarray(q_coeff_fn(xc), dtype=float).ravel()
    f_vals = np.asarray(rhs_fn(xc), dtype=float).ravel()

    A_pde = d2phi_c + q_vals[:, None] * phi_c

    phi_b, dphi_b, _ = basis_eval(np.array([x_bc_left, x_bc_right]))
    row_left = phi_b[0] if bc_left_type == "dirichlet" else dphi_b[0]
    row_right = phi_b[1] if bc_right_type == "dirichlet" else dphi_b[1]

    A_total = np.vstack([A_pde, bc_weight * row_left, bc_weight * row_right])
    b_total = np.concatenate([f_vals, [bc_weight * bc_left_val, bc_weight * bc_right_val]])

    # Least squares by SVD.  The Gaussian basis is very ill-conditioned, and the normal equations
    # would square its condition number and make the result depend on the linear-algebra library.
    weights = np.linalg.lstsq(A_total, b_total, rcond=None)[0]

    def predict_fn(x_eval: np.ndarray) -> dict:
        xe = np.asarray(x_eval, dtype=float).ravel()
        p, dp, d2p = basis_eval(xe)
        return {"u": p @ weights, "du_dx": dp @ weights, "d2u_dx2": d2p @ weights}

    res_col = predict_fn(xc)
    pde_residual = float(np.sqrt(np.mean((res_col["d2u_dx2"] + q_vals * res_col["u"] - f_vals) ** 2)))
    return {"weights": weights, "predict": predict_fn, "pde_rms_residual": pde_residual}


def solve_hard_bc_thiele_pinn(x_colloc: np.ndarray, phi: float, n_modes: int = 8) -> dict:
    """
    Hard-boundary-condition Lagaris trial ansatz for porous catalyst reaction-diffusion:
      u''(x) - phi^2 u(x) = 0 on [0, 1], with exact u'(0) = 0 and u(1) = 1.
    Uses even polynomial basis psi_k(x) = x^{2k} - 1 so u_hat(x) = 1 + sum_{k=1}^M w_k (x^{2k} - 1)
    identically satisfies u_hat'(0) = 0 and u_hat(1) = 1 for every weight vector w.
    """
    xc = np.asarray(x_colloc, dtype=float).ravel()
    ks = np.arange(1, n_modes + 1, dtype=float)
    psi = (xc[:, None] ** (2.0 * ks[None, :])) - 1.0
    d2psi = (2.0 * ks[None, :]) * (2.0 * ks[None, :] - 1.0) * (xc[:, None] ** (2.0 * ks[None, :] - 2.0))
    A = d2psi - (phi ** 2) * psi
    b = np.full_like(xc, phi ** 2)
    w = np.linalg.lstsq(A, b, rcond=None)[0]

    def predict_fn(x_eval: np.ndarray) -> dict:
        xe = np.asarray(x_eval, dtype=float).ravel()
        p = (xe[:, None] ** (2.0 * ks[None, :])) - 1.0
        dp = (2.0 * ks[None, :]) * (xe[:, None] ** (2.0 * ks[None, :] - 1.0))
        d2p = (2.0 * ks[None, :]) * (2.0 * ks[None, :] - 1.0) * (xe[:, None] ** (2.0 * ks[None, :] - 2.0))
        return {"u": 1.0 + p @ w, "du_dx": dp @ w, "d2u_dx2": d2p @ w}

    out_c = predict_fn(xc)
    eta = float(predict_fn(np.array([1.0]))["du_dx"][0]) / (phi ** 2) if phi > 0.0 else 1.0
    rms_res = float(np.sqrt(np.mean((out_c["d2u_dx2"] - (phi ** 2) * out_c["u"]) ** 2)))
    return {"weights": w, "predict": predict_fn, "effectiveness_factor": eta, "pde_rms_residual": rms_res}


def inverse_pinn_thiele_modulus(
    x_sensors: np.ndarray,
    u_sensors: np.ndarray,
    x_colloc: np.ndarray,
    centers: np.ndarray,
    sigma: float = 0.22,
    phi_grid: np.ndarray | None = None,
) -> dict:
    """
    Variable-Projection Inverse PINN that infers the unknown Thiele modulus phi > 0 from noisy internal
    concentration sensors u_sensors at x_sensors while enforcing u''(x) - phi^2 u(x) = 0, u'(0)=0, u(1)=1.
    """
    xs = np.asarray(x_sensors, dtype=float).ravel()
    us = np.asarray(u_sensors, dtype=float).ravel()
    if phi_grid is None:
        phi_grid = np.linspace(0.2, 5.0, 97)

    best_phi = float(phi_grid[0])
    best_loss = float("inf")
    best_sol = None
    losses = []
    for phi_cand in phi_grid:
        sol = solve_linear_bvp_pinn(
            x_colloc,
            centers,
            sigma=sigma,
            q_coeff_fn=lambda x, p=float(phi_cand): np.full_like(x, -(p ** 2)),
            rhs_fn=lambda x: np.zeros_like(x),
            x_bc_left=0.0,
            bc_left_type="neumann",
            bc_left_val=0.0,
            x_bc_right=1.0,
            bc_right_type="dirichlet",
            bc_right_val=1.0,
        )
        u_hat = sol["predict"](xs)["u"]
        loss = float(np.mean((u_hat - us) ** 2))
        losses.append(loss)
        if loss < best_loss:
            best_loss = loss
            best_phi = float(phi_cand)
            best_sol = sol

    return {"phi_est": best_phi, "sensor_mse": best_loss, "pinn_solution": best_sol, "losses": np.asarray(losses)}
