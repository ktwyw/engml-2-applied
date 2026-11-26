"""Validation of engml_2_applied against exact mathematical identities, physical conservation laws, and reference benchmarks.

Run:  python scripts/validate.py      (writes docs/validation.md)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engml_2_applied import (
    bo_active,
    clustering_phases,
    constants,
    descriptors,
    dim_reduction,
    kernels_gp,
    linear_physics,
    neural_mlp,
    pinn,
    sindy_ude,
    trees_ensembles,
    uncertainty,
)

RESULTS: list[tuple] = []


def check(
    section: str, name: str, reference: str, value: float, expected: float, rtol: float = 0.0, atol: float = 0.0
) -> None:
    ok = bool(np.isclose(value, expected, rtol=rtol, atol=atol))
    RESULTS.append((section, name, reference, float(value), float(expected), ok))
    if not ok:
        print(f"FAIL: {section} | {name}: computed {value!r}, expected {expected!r}")


# ------------------------------------------------------------------ 1. Molecular graph descriptors & standardization
S = "Molecular graph descriptors, the table of compounds and standardization"
# Exact Wiener index of an N-atom linear path graph (n-alkane) is W(N) = N * (N^2 - 1) / 6:
for n_c in range(2, 11):
    A_path = descriptors.path_graph_adjacency(n_c)
    desc = descriptors.graph_topological_descriptors(A_path)
    w_exact = n_c * ((n_c ** 2) - 1.0) / 6.0
    m1_exact = 2.0 * 1.0 + max(0, n_c - 2) * 4.0  # 2 terminal deg-1 atoms, (N-2) internal deg-2 atoms
    check(
        S,
        f'linear n-alkane C{n_c} Wiener index W = N(N^2-1)/6',
        'exact graph theorem',
        desc['wiener_index'],
        w_exact,
        rtol=1e-14,
    )
    check(
        S,
        f'linear n-alkane C{n_c} First Zagreb index M1 = 4N - 6',
        'exact graph theorem',
        desc['zagreb_m1'],
        m1_exact,
        rtol=1e-14,
    )
    # Smallest Laplacian eigenvalue of any connected graph is identically 0:
    check(
        S,
        f'linear n-alkane C{n_c} Laplacian smallest eigenvalue lambda_1 = 0',
        'spectral graph theory',
        float(desc['laplacian_eigenvalues'][0]),
        0.0,
        atol=1e-13,
    )

# Star graph (e.g., neopentane C5 carbon skeleton: 1 central quaternary C bonded to 4 methyl C):
for n_star in [4, 5, 6]:
    A_star = descriptors.star_graph_adjacency(n_star)
    d_star = descriptors.graph_topological_descriptors(A_star)
    # 1 pair at distance 1 for each of (n-1) leaves + C(n-1, 2) leaf-leaf pairs at distance 2:
    w_star_exact = (n_star - 1.0) * 1.0 + (n_star - 1.0) * (n_star - 2.0) * 1.0
    m1_star_exact = ((n_star - 1.0) ** 2) + (n_star - 1.0) * 1.0
    check(
        S,
        f'star-branched graph (N={n_star}) Wiener index W = (N-1)^2',
        'exact graph theorem',
        d_star['wiener_index'],
        w_star_exact,
        rtol=1e-14,
    )
    check(
        S,
        f'star-branched graph (N={n_star}) First Zagreb index M1 = N(N-1)',
        'exact graph theorem',
        d_star['zagreb_m1'],
        m1_star_exact,
        rtol=1e-14,
    )

# Branched isomer has strictly smaller Wiener index (more compact) than linear isomer of same N:
check(
    S,
    "neopentane (star C5) Wiener index (16) < n-pentane (path C5) Wiener index (20)",
    "isomer branching rule",
    float(
        descriptors.graph_topological_descriptors(descriptors.star_graph_adjacency(5))["wiener_index"]
        < descriptors.graph_topological_descriptors(descriptors.path_graph_adjacency(5))["wiener_index"]
    ),
    1.0,
)

rng = np.random.default_rng(2026)
X_raw = rng.normal(loc=[350.0, 2.5, -10.0], scale=[45.0, 0.4, 3.0], size=(60, 3))
scaler = descriptors.StandardScaler()
X_std = scaler.fit_transform(X_raw)
check(
    S,
    'StandardScaler transformed column means equal 0',
    'identity',
    float(np.max(np.abs(np.mean(X_std, axis=0)))),
    0.0,
    atol=1e-14,
)
check(
    S,
    'StandardScaler transformed column stds equal 1',
    'identity',
    float(np.max(np.abs(np.std(X_std, axis=0) - 1.0))),
    0.0,
    atol=1e-14,
)
check(
    S,
    'StandardScaler inverse_transform round-trip error',
    'identity',
    float(np.max(np.abs(scaler.inverse_transform(X_std) - X_raw))),
    0.0,
    atol=1e-12,
)


# ------------------------------------------------------------------ 2. Conservation-constrained & monotone regression
S = "Regularized regression, equality constraints and monotone regression"
X_reg = np.column_stack([np.ones(50), rng.uniform(-2.0, 2.0, size=(50, 4))])
w_true = np.array([3.0, 1.5, -0.8, 0.5, 1.2])
y_reg = X_reg @ w_true
w_ols = linear_physics.ridge_regression(X_reg, y_reg, lam=0.0)
check(
    S,
    'unregularized Ridge (lam=0) recovers exact noiseless parameters',
    'exact OLS',
    float(np.max(np.abs(w_ols - w_true))),
    0.0,
    atol=1e-12,
)

for lam_v in [0.1, 1.0, 10.0, 100.0]:
    w_r1 = linear_physics.ridge_regression(X_reg, y_reg, lam=lam_v, penalize_intercept=True)
    w_r2 = linear_physics.ridge_regression(X_reg, y_reg, lam=lam_v * 2.0, penalize_intercept=True)
    check(
        S,
        f'Ridge L2 norm decreases monotonically with penalty (lam={lam_v} -> {2 * lam_v})',
        'convexity',
        float(np.linalg.norm(w_r2) < np.linalg.norm(w_r1)),
        1.0,
    )

# Orthogonal design Lasso check against exact soft-thresholding formula:
X_orth = np.eye(5) * np.sqrt(5.0)
y_orth = np.array([2.0, 1.0, -0.5, 0.05, -0.02]) * np.sqrt(5.0)
w_lasso = linear_physics.lasso_coordinate_descent(X_orth, y_orth, alpha=0.10, tol=1e-12)
check(S, "Lasso on orthogonal design: unpenalized intercept w_0", "exact", w_lasso[0], 2.0, rtol=1e-10)
check(
    S,
    'Lasso on orthogonal design: soft-thresholded positive weight w_1 = 1.0 - 0.10',
    'exact soft-threshold',
    w_lasso[1],
    0.9,
    rtol=1e-08,
)
check(
    S,
    'Lasso on orthogonal design: soft-thresholded negative weight w_2 = -0.5 + 0.10',
    'exact soft-threshold',
    w_lasso[2],
    -0.4,
    rtol=1e-08,
)
check(
    S,
    'Lasso on orthogonal design: small weights |rho_j| < alpha are set to exact 0',
    'sparsity',
    abs(w_lasso[3]) + abs(w_lasso[4]),
    0.0,
    atol=1e-14,
)

# KKT equality-constrained regression C w = d (e.g., mass/atom conservation):
C_cons = np.array([[0.0, 1.0, 1.0, 1.0, 1.0], [0.0, 1.0, -1.0, 0.0, 0.0]])
d_cons = np.array([2.0, 0.0])
y_noisy = y_reg + rng.normal(0.0, 0.5, size=50)
kkt_res = linear_physics.equality_constrained_lstsq(X_reg, y_noisy, C_cons, d_cons, lam=1e-8)
check(
    S,
    'KKT equality-constrained regression satisfies C w = d to machine precision',
    'KKT saddle-point',
    kkt_res['constraint_residual'],
    0.0,
    atol=1e-13,
)
check(
    S,
    'KKT constraint #1: sum of stoichiometric coefficients equals 2.0',
    'mass balance',
    float(np.sum(kkt_res['w'][1:])),
    2.0,
    rtol=1e-13,
)
check(
    S,
    'KKT constraint #2: w_1 == w_2 symmetry holds exactly',
    'symmetry balance',
    float(kkt_res['w'][1] - kkt_res['w'][2]),
    0.0,
    atol=1e-13,
)

# PAVA monotone regression (mechanical stability dP/dV <= 0):
V_grid = np.linspace(1.0, 5.0, 40)
P_noisy = 10.0 / V_grid + rng.normal(0.0, 0.4, size=40)
P_mono = linear_physics.pava_monotone_regression(P_noisy, increasing=False)
check(
    S,
    'PAVA decreasing isotherm satisfies dP/dV <= 0 everywhere',
    'thermodynamic stability',
    float(np.max(np.diff(P_mono))),
    0.0,
    atol=1e-14,
)
check(
    S,
    'PAVA preserves sample mean sum(f_i) == sum(y_i)',
    'PAVA projection property',
    float(np.mean(P_mono)),
    float(np.mean(P_noisy)),
    rtol=1e-13,
)
check(
    S,
    'PAVA on already-monotone array is an exact identity',
    'idempotence',
    float(np.max(np.abs(linear_physics.pava_monotone_regression(V_grid, True) - V_grid))),
    0.0,
    atol=1e-14,
)


# ------------------------------------------------------------------ 3. Gaussian Processes, Derivative GPs & Bayesian
S = "Gaussian processes, derivative observations and Bayesian optimization"
x_gp = np.linspace(0.0, 3.0, 12)
y_gp = np.sin(x_gp)
for kern_name in ("rbf", "matern52"):
    gp = kernels_gp.GaussianProcessRegressor(length_scale=1.0, sigma_f=1.0, sigma_n=1e-06, kernel=kern_name).fit(
        x_gp,
        y_gp,
    )
    mu_train, std_train = gp.predict(x_gp)
    check(
        S,
        f'GP ({kern_name}) interpolates training targets as sigma_n -> 0',
        'exact kriging',
        float(np.max(np.abs(mu_train - y_gp))),
        0.0,
        atol=0.0001,
    )
    check(
        S,
        f'GP ({kern_name}) predictive uncertainty vanishes at training points',
        'exact kriging',
        float(np.max(std_train)),
        0.0,
        atol=0.001,
    )
    _, std_far = gp.predict(np.array([100.0]))
    check(
        S,
        f'GP ({kern_name}) far-field predictive std approaches prior sigma_f = 1.0',
        'prior reversion',
        float(std_far[0]),
        1.0,
        rtol=1e-06,
    )

# Derivative-informed GP test on f(x) = cos(x), f'(x) = -sin(x):
x_v = np.array([0.0, 3.0])
x_d = np.array([0.0, 1.0, 2.0, 3.0])
x_test_gp = np.linspace(0.0, 3.0, 25)
gp_vo = kernels_gp.GaussianProcessRegressor(length_scale=1.2, sigma_f=1.0, sigma_n=1e-5).fit(x_v, np.cos(x_v))
mu_val_only, _ = gp_vo.predict(x_test_gp)
(mu_with_deriv, _) = kernels_gp.gp_1d_with_derivative_observations(
    x_v,
    np.cos(x_v),
    x_d,
    -np.sin(x_d),
    x_test_gp,
    length_scale=1.2,
    sigma_f=1.0,
    sigma_n=1e-05,
)
err_val_only = float(np.max(np.abs(mu_val_only - np.cos(x_test_gp))))
err_with_deriv = float(np.max(np.abs(mu_with_deriv - np.cos(x_test_gp))))
check(
    S,
    'Derivative-informed GP accuracy on cos(x) with 4 slope observations',
    'physics-guided GP',
    err_with_deriv,
    0.0,
    atol=0.005,
)
check(
    S,
    'Adding derivative observations reduces GP max error by > 10x',
    'information gain',
    float(err_with_deriv < 0.1 * err_val_only),
    1.0,
)

# Bayesian Optimization Expected Improvement checks:
check(
    S,
    'Expected Improvement vanishes when sigma = 0 and mu <= y_best',
    'analytical limit',
    float(bo_active.expected_improvement(np.array([1.0]), np.array([0.0]), y_best=1.5, xi=0.0)[0]),
    0.0,
    atol=1e-14,
)
check(
    S,
    'Expected Improvement equals (mu - y_best) when sigma -> 0 and mu > y_best',
    'analytical limit',
    float(bo_active.expected_improvement(np.array([2.5]), np.array([1e-14]), y_best=1.5, xi=0.0)[0]),
    1.0,
    rtol=1e-12,
)
check(
    S,
    'Expected Improvement increases monotonically with predictive uncertainty sigma',
    'exploration property',
    float(
        bo_active.expected_improvement(np.array([1.5]), np.array([0.8]), 1.5, 0.0)[0]
        > bo_active.expected_improvement(np.array([1.5]), np.array([0.2]), 1.5, 0.0)[0]
    ),
    1.0,
)


# ------------------------------------------------------------------ 4. Chemometrics (PCA, Hotelling T^2, PLS) & Unsup
S = "Principal components, partial least squares and clustering"
pca_full = dim_reduction.pca_svd(X_raw, n_components=3)
check(
    S,
    'PCA explained variance ratios sum to 1.0 when n_components = d',
    'Parseval identity',
    float(np.sum(pca_full['explained_variance_ratio'])),
    1.0,
    rtol=1e-14,
)
check(
    S,
    'PCA loading vectors are orthonormal P^T P = I',
    'orthogonality',
    float(np.max(np.abs(pca_full['loadings'].T @ pca_full['loadings'] - np.eye(3)))),
    0.0,
    atol=1e-13,
)
check(
    S,
    'PCA score columns are mutually orthogonal',
    'orthogonality',
    float(np.abs(np.dot(pca_full['scores'][:, 0], pca_full['scores'][:, 1]))),
    0.0,
    atol=1e-10,
)
check(
    S,
    'PCA full-rank reconstruction residual Q_SPE = 0',
    'exact reconstruction',
    float(np.max(pca_full['q_spe'])),
    0.0,
    atol=1e-24,
)
check(
    S,
    'Hotelling T^2 mean across N samples equals k * (N - 1) / N',
    'exact statistical identity',
    float(np.mean(pca_full['hotelling_t2'])),
    3.0 * 59.0 / 60.0,
    rtol=1e-13,
)

pls_full = dim_reduction.pls1_nipals(X_raw, X_raw @ np.array([1.2, -0.7, 0.4]) + 5.0, n_components=3)
check(
    S,
    'PLS1 with n_components = d recovers exact linear regression coefficients',
    'NIPALS equivalence',
    float(np.max(np.abs(pls_full['coef'] - np.array([1.2, -0.7, 0.4])))),
    0.0,
    atol=1e-11,
)

X_blobs = np.vstack([rng.normal(-5.0, 0.2, size=(20, 2)), rng.normal(5.0, 0.2, size=(20, 2))])
km_res = clustering_phases.kmeans_fit(X_blobs, k=2, seed=42)
check(
    S,
    'K-Means++ cleanly separates two well-separated thermodynamic phase clusters',
    'clustering purity',
    float(len(np.unique(km_res['labels'][:20])) == 1 and km_res['labels'][0] != km_res['labels'][20]),
    1.0,
)
sp_res = clustering_phases.spectral_clustering(X_blobs, k=2, sigma=2.0, seed=42)
check(
    S,
    'Spectral clustering cleanly separates two phase clusters',
    'Laplacian clustering',
    float(len(np.unique(sp_res['labels'][:20])) == 1 and sp_res['labels'][0] != sp_res['labels'][20]),
    1.0,
)


# ------------------------------------------------------------------ 5. Trees, Ensembles, Neural Autodiff, PINNs, SIND
S = "Trees, a differentiable network, collocation, sparse regression and conformal intervals"
x_tr = np.linspace(0.0, 4.0, 25)
y_tr = np.sin(x_tr)
gb1 = trees_ensembles.GradientBoostingRegressor(n_estimators=5, learning_rate=0.2, max_depth=2).fit(x_tr, y_tr)
gb2 = trees_ensembles.GradientBoostingRegressor(n_estimators=30, learning_rate=0.2, max_depth=2).fit(x_tr, y_tr)
check(
    S,
    'Gradient boosting training MSE decreases monotonically with n_estimators',
    'boosting convergence',
    float(np.mean((gb2.predict(x_tr) - y_tr) ** 2) < np.mean((gb1.predict(x_tr) - y_tr) ** 2)),
    1.0,
)

mlp = neural_mlp.DifferentiableMLP1D(n_hidden=12, seed=7)
x_probe = np.linspace(-1.5, 1.5, 15)
eps_fd = 1e-4
out_0 = mlp.forward_and_derivatives(x_probe)
out_p = mlp.forward_and_derivatives(x_probe + eps_fd)
out_m = mlp.forward_and_derivatives(x_probe - eps_fd)
du_fd = (out_p["u"] - out_m["u"]) / (2.0 * eps_fd)
d2u_fd = (out_p["u"] - 2.0 * out_0["u"] + out_m["u"]) / (eps_fd ** 2)
check(
    S,
    'MLP analytical 1st derivative du/dx matches central finite difference',
    'exact autodiff',
    float(np.max(np.abs(out_0['du_dx'] - du_fd))),
    0.0,
    atol=1e-07,
)
check(
    S,
    'MLP analytical 2nd derivative d^2u/dx^2 matches central finite difference',
    'exact autodiff',
    float(np.max(np.abs(out_0['d2u_dx2'] - d2u_fd))),
    0.0,
    atol=1e-06,
)

# PINN collocation verification on reaction-diffusion BVP: u''(x) - phi^2 * u(x) = 0, u'(0)=0, u(1)=1 -> u(x) = cosh(p
for phi_pinn in [1.0, 2.0, 3.0]:
    x_col = np.linspace(0.0, 1.0, 41)
    c_rbf = np.linspace(-0.2, 1.2, 25)
    pinn_sol = pinn.solve_linear_bvp_pinn(
        x_col, c_rbf, sigma=0.22,
        q_coeff_fn=lambda x, p=phi_pinn: np.full_like(x, -(p ** 2)),
        rhs_fn=lambda x: np.zeros_like(x),
        x_bc_left=0.0, bc_left_type="neumann", bc_left_val=0.0,
        x_bc_right=1.0, bc_right_type="dirichlet", bc_right_val=1.0,
    )
    u_pred = pinn_sol["predict"](x_col)["u"]
    u_exact = np.cosh(phi_pinn * x_col) / np.cosh(phi_pinn)
    check(
        S,
        f'PINN collocation BVP max error on catalyst slab (phi={phi_pinn})',
        'exact cosh(phi x)/cosh(phi)',
        float(np.max(np.abs(u_pred - u_exact))),
        0.0,
        atol=2e-04,
    )

# SINDy sparse reaction rate law discovery: dC/dt = -k1 * C - k2 * C^2 with candidate library [1, C, C^2, C^3, C^4]
C_traj = np.linspace(0.1, 2.0, 60)
k1_true, k2_true = 0.45, 1.20
dC_dt = -k1_true * C_traj - k2_true * (C_traj ** 2)
Theta_lib = np.column_stack([np.ones_like(C_traj), C_traj, C_traj ** 2, C_traj ** 3, C_traj ** 4])
xi_sindy = sindy_ude.sindy_stlsq(Theta_lib, dC_dt, threshold=0.05)
check(
    S,
    'SINDy discovers exact 1st-order rate constant k1 = 0.45',
    'sparse identification',
    float(xi_sindy[1]),
    -k1_true,
    rtol=1e-12,
)
check(
    S,
    'SINDy discovers exact 2nd-order rate constant k2 = 1.20',
    'sparse identification',
    float(xi_sindy[2]),
    -k2_true,
    rtol=1e-12,
)
check(
    S,
    'SINDy zeroes out spurious library terms (1, C^3, C^4)',
    'parsimony',
    float(abs(xi_sindy[0]) + abs(xi_sindy[3]) + abs(xi_sindy[4])),
    0.0,
    atol=1e-14,
)

# Conformal prediction & Mahalanobis OOD checks:
y_cal = rng.normal(0.0, 1.0, size=200)
y_test_cal = rng.normal(0.0, 1.0, size=500)
cp_int = uncertainty.split_conformal_interval(y_cal, np.zeros_like(y_cal), np.zeros_like(y_test_cal), alpha=0.10)
empirical_cov = float(np.mean((y_test_cal >= cp_int["lower"]) & (y_test_cal <= cp_int["upper"])))
check(
    S,
    ('Split conformal 90% prediction interval achieves >= 85% empirical test ' 'coverage'),
    'Vovk conformal guarantee',
    float(empirical_cov >= 0.85),
    1.0,
)

ood_scores = uncertainty.mahalanobis_ood_score(
    X_raw,
    np.vstack([np.mean(X_raw, axis=0), np.mean(X_raw, axis=0) + 10.0 * np.std(X_raw, axis=0)]),
)
check(S, "Mahalanobis OOD score at training centroid equals 0", "exact", float(ood_scores[0]), 0.0, atol=1e-12)
check(
    S,
    'Mahalanobis OOD score flags 10-sigma extrapolated point (> 90)',
    'extrapolation guardrail',
    float(ood_scores[1] > 90.0),
    1.0,
)

# Pure-component QSPR boiling-point, critical-temperature & topological index checks across MOLECULES_DB:
alkane_names = [
    "methane", "ethane", "propane", "n-butane", "n-pentane",
    "n-hexane", "n-heptane", "n-octane", "n-nonane", "n-decane",
]
for i_alk in range(len(alkane_names) - 1):
    m_curr = constants.MOLECULES_DB[alkane_names[i_alk]]
    m_next = constants.MOLECULES_DB[alkane_names[i_alk + 1]]
    check(
        'Molecular graph descriptors, the table of compounds and standardization',
        f'homologous alkane Tb monotonicity: {alkane_names[i_alk]} < {alkane_names[i_alk + 1]}',
        'NIST WebBook',
        float(m_curr['Tb'] < m_next['Tb']),
        1.0,
    )
    check(
        'Molecular graph descriptors, the table of compounds and standardization',
        f'homologous alkane Tc monotonicity: {alkane_names[i_alk]} < {alkane_names[i_alk + 1]}',
        'NIST WebBook',
        float(m_curr['Tc'] < m_next['Tc']),
        1.0,
    )
    check(
        'Molecular graph descriptors, the table of compounds and standardization',
        f'homologous alkane omega monotonicity: {alkane_names[i_alk]} < {alkane_names[i_alk + 1]}',
        'Pitzer acentric factor',
        float(m_curr['omega'] < m_next['omega']),
        1.0,
    )

for n_nodes_test in range(3, 21):
    A_p = descriptors.path_graph_adjacency(n_nodes_test)
    d_p = descriptors.graph_topological_descriptors(A_p)
    # Randic index of a linear chain of N >= 3 vertices has 2 terminal edges (1/sqrt(1*2)) and (N-3) internal edges (1
    r_exact = np.sqrt(2.0) + 0.5 * (n_nodes_test - 3.0)
    check(
        'Molecular graph descriptors, the table of compounds and standardization',
        f'Randic connectivity index of C{n_nodes_test} path graph = sqrt(2) + (N-3)/2',
        'Randic (1975)',
        d_p['randic_index'],
        r_exact,
        rtol=1e-14,
    )
    m2_exact = 2.0 * (1.0 * 2.0) + max(0, n_nodes_test - 3) * (2.0 * 2.0)
    check(
        'Molecular graph descriptors, the table of compounds and standardization',
        f'Second Zagreb index M2 of C{n_nodes_test} path graph = 4(N-2)',
        'Gutman & Trinajstic (1972)',
        d_p['zagreb_m2'],
        m2_exact,
        rtol=1e-14,
    )


# ------------------------------------------------------------------ 6. Applied Volume II extensions (Crowe, I-splines
S = "Data reconciliation, I-splines, robust regression and preprocessing of spectra"
rec_crowe = linear_physics.crowe_data_reconciliation(
    np.array([102.0, 60.0, 39.0]), np.array([1.0, 1.0, 2.0]), np.array([[1.0, -1.0, -1.0]]), np.array([0.0])
)
check(
    S,
    'Crowe data reconciliation satisfies A y_rec = b exactly',
    'Crowe (1983)',
    float(np.max(np.abs(rec_crowe['balance_residual_after']))),
    0.0,
    atol=1e-14,
)
check(
    S,
    ('Crowe data reconciliation 3-stream splitter global chi^2 = r^2 / ' 'sum(sigma_i^2)'),
    'exact chi^2',
    rec_crowe['chi2_stat'],
    1.5,
    rtol=1e-14,
)

x_isp = np.linspace(0.0, 5.0, 30)
I_b, M_b = linear_physics.ispline_basis_1d(x_isp, np.linspace(0.0, 5.0, 6))
check(
    S,
    'M-spline basis functions are strictly non-negative everywhere',
    'Ramsay (1988)',
    float(np.min(M_b) >= 0.0),
    1.0,
)
isp_res = linear_physics.fit_monotone_ispline(x_isp, np.sqrt(x_isp), np.linspace(0.0, 5.0, 6), increasing=True)
check(
    S,
    'Monotone I-spline fit preserves non-negative first derivative dy/dx >= 0',
    'monotone spline',
    float(np.min(isp_res['dy_dx']) >= -1e-15),
    1.0,
)

X_hub = np.column_stack([np.ones(30), np.linspace(-1.0, 1.0, 30)])
y_hub = X_hub @ np.array([4.0, 2.0])
y_hub[15] += 50.0
w_hub = linear_physics.huber_irls_regression(X_hub, y_hub, delta=0.1)["w"]
w_q50 = linear_physics.quantile_irls_regression(X_hub, y_hub, tau=0.50)
check(S, "Huber IRLS recovers true slope despite +50 outlier spike", "Huber (1964)", float(w_hub[1]), 2.0, atol=1e-2)
check(
    S,
    'Median quantile regression (tau=0.50) recovers true slope despite +50 spike',
    'Koenker & Bassett (1978)',
    float(w_q50[1]),
    2.0,
    atol=0.01,
)

X_snv_test = dim_reduction.snv_preprocess(np.vstack([2.5 * np.sin(x_isp) + 3.0, 0.4 * np.sin(x_isp) - 1.2]))
check(
    S,
    ('SNV preprocessing is invariant to multiplicative path-length and ' 'additive offset'),
    'Barnes et al. (1989)',
    float(np.max(np.abs(X_snv_test[0] - X_snv_test[1]))),
    0.0,
    atol=1e-13,
)
pcr_full = dim_reduction.pcr_regression(X_raw, X_raw @ np.array([1.2, -0.7, 0.4]) + 5.0, n_components=3)
check(
    S,
    'Full-rank PCR (k=d) recovers exact linear regression coefficients',
    'PCR equivalence',
    float(np.max(np.abs(pcr_full['coef'] - np.array([1.2, -0.7, 0.4])))),
    0.0,
    atol=1e-11,
)

gmm_res = clustering_phases.gmm_em_fit(X_blobs, k=2, seed=42)
check(
    S,
    'GMM EM posterior responsibilities sum to 1.0 across phases',
    'partition of unity',
    float(np.max(np.abs(np.sum(gmm_res['responsibilities'], axis=1) - 1.0))),
    0.0,
    atol=1e-14,
)
sil_blobs = clustering_phases.silhouette_score_mean(X_blobs, km_res["labels"])
check(
    S,
    'Silhouette score of well-separated phase clusters exceeds 0.90',
    'Rousseeuw (1987)',
    float(sil_blobs > 0.9),
    1.0,
)

ps_split = trees_ensembles.prefix_sum_best_split_1d(np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.0, 0.0, 10.0, 10.0]))
check(
    S,
    ('O(1) prefix-sum CART split locates exact step midpoint 2.5 with zero ' 'post-split SSE'),
    'Breiman CART',
    ps_split['threshold'],
    2.5,
    rtol=1e-14,
)
rf_fit = trees_ensembles.RandomForestRegressor(n_estimators=15, max_depth=3, seed=42).fit(x_tr, y_tr)
check(
    S,
    'Random Forest ensemble achieves low training MSE on sin(x)',
    'Breiman (2001)',
    float(np.mean((rf_fit.predict(x_tr) - y_tr) ** 2) < 0.03),
    1.0,
)


# ------------------------------------------------------------------ Write report
passed = sum(r[5] for r in RESULTS)
lines = [
    "# Validation report",
    "",
    "Generated by `python scripts/validate.py`. The same checks are run by `pytest` (`tests/test_validation.py`).",
    "",
    f"**{passed} / {len(RESULTS)} checks pass.**",
    "",
]
sections = list(dict.fromkeys(result[0] for result in RESULTS))      # each section once, in order of first appearance
for section in sections:
    lines += ["", f"## {section}", "", "| Check | Reference | engml_2_applied | Expected | Result |", "|---|---|---|---|:-:|"]
    for sec, name, ref, val, exp, ok in RESULTS:
        if sec != section:
            continue
        name_md = name.replace("|", "\\|")
        ref_md = ref.replace("|", "\\|")
        lines.append(f"| {name_md} | {ref_md} | {val:.10g} | {exp:.10g} | {'pass' if ok else '**FAIL**'} |")

report_text = "\n".join(lines) + "\n"
if __name__ == "__main__":      # the tests run this file with another name and do not rewrite the report
    (ROOT / "docs" / "validation.md").write_text(report_text, encoding="utf-8")
    print(f"{passed}/{len(RESULTS)} checks pass")
    if passed != len(RESULTS):
        sys.exit(1)
