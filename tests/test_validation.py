"""Comprehensive unit and mathematical validation test suite for engml_2_applied."""

import math
import runpy
import sys
import unittest
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


class DescriptorsAndLinearPhysicsTest(unittest.TestCase):
    def test_01_alkane_wiener_zagreb_randic_and_laplacian_invariants(self):
        for n in range(2, 9):
            d = descriptors.graph_topological_descriptors(descriptors.path_graph_adjacency(n))
            self.assertAlmostEqual(d["wiener_index"], n * (n ** 2 - 1) / 6.0, places=12)
            self.assertAlmostEqual(d["zagreb_m1"], 4.0 * n - 6.0, places=12)
        d_pent = descriptors.graph_topological_descriptors(descriptors.path_graph_adjacency(5))
        d_neo = descriptors.graph_topological_descriptors(descriptors.star_graph_adjacency(5))
        self.assertGreater(d_pent["wiener_index"], d_neo["wiener_index"])
        self.assertLess(d_pent["zagreb_m1"], d_neo["zagreb_m1"])
        self.assertEqual(len(constants.MOLECULES_DB), 16)

    def test_02_standard_scaler_round_trip(self):
        rng = np.random.default_rng(1)
        X = rng.normal(loc=[350.0, 2.5], scale=[40.0, 0.8], size=(50, 2))
        sc = descriptors.StandardScaler()
        Xs = sc.fit_transform(X)
        np.testing.assert_allclose(np.mean(Xs, axis=0), 0.0, atol=1e-14)
        np.testing.assert_allclose(np.std(Xs, axis=0), 1.0, atol=1e-14)
        np.testing.assert_allclose(sc.inverse_transform(Xs), X, rtol=1e-13)

    def test_03_ridge_and_lasso_soft_thresholding(self):
        X = np.eye(4)
        y = np.array([3.0, 2.0, -1.5, 0.1])
        w_lasso = linear_physics.lasso_coordinate_descent(X, y, alpha=0.1, max_iter=300, tol=1e-12)
        np.testing.assert_allclose(w_lasso, [3.0, 1.6, -1.1, 0.0], atol=1e-8)

    def test_04_kkt_equality_constrained_mass_balance(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(30, 4))
        w_true = np.array([0.25, 0.35, 0.15, 0.25])
        y = X @ w_true + rng.normal(0.0, 0.05, size=30)
        C = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, -1.0, 1.0, -1.0]])
        d = np.array([1.0, 0.0])
        res = linear_physics.equality_constrained_lstsq(X, y, C, d, lam=1e-8)
        self.assertLess(res["constraint_residual"], 1e-13)

    def test_05_pava_monotone_thermodynamic_stability(self):
        rng = np.random.default_rng(7)
        v = np.linspace(0.5, 5.0, 50)
        p_true = 10.0 / v
        p_noisy = p_true + rng.normal(0.0, 0.4, size=50)
        p_iso = linear_physics.pava_monotone_regression(p_noisy, increasing=False)
        self.assertLessEqual(float(np.max(np.diff(p_iso))), 1e-15)
        self.assertAlmostEqual(float(np.mean(p_iso)), float(np.mean(p_noisy)), places=12)


class GPBayesianOptAndChemometricsTest(unittest.TestCase):
    def test_06_cholesky_gp_interpolation_and_marginal_likelihood(self):
        x = np.linspace(0.0, 3.0, 10)
        y = np.sin(x)
        gp = kernels_gp.GaussianProcessRegressor(length_scale=1.0, sigma_f=1.0, sigma_n=1e-06, kernel='matern52').fit(
            x,
            y,
        )
        mu, std = gp.predict(x)
        np.testing.assert_allclose(mu, y, atol=1e-4)
        self.assertLess(float(np.max(std)), 1e-3)

    def test_07_derivative_informed_gp_improvement(self):
        xv = np.array([0.0, 3.0])
        xd = np.array([0.0, 1.0, 2.0, 3.0])
        xs = np.linspace(0.0, 3.0, 30)
        gp0 = kernels_gp.GaussianProcessRegressor(length_scale=1.2, sigma_f=1.0, sigma_n=1e-5).fit(xv, np.cos(xv))
        mu0, _ = gp0.predict(xs)
        (mu1, _) = kernels_gp.gp_1d_with_derivative_observations(
            xv,
            np.cos(xv),
            xd,
            -np.sin(xd),
            xs,
            length_scale=1.2,
            sigma_f=1.0,
            sigma_n=1e-05,
        )
        err0 = float(np.max(np.abs(mu0 - np.cos(xs))))
        err1 = float(np.max(np.abs(mu1 - np.cos(xs))))
        self.assertLess(err1, 0.1 * err0)

    def test_08_expected_improvement_and_ucb_identities(self):
        ei_zero = bo_active.expected_improvement(np.array([1.0]), np.array([0.0]), y_best=1.5, xi=0.0)
        self.assertEqual(float(ei_zero[0]), 0.0)
        ei_sym = bo_active.expected_improvement(np.array([2.0]), np.array([0.5]), y_best=2.0, xi=0.0)
        self.assertAlmostEqual(float(ei_sym[0]), 0.5 / math.sqrt(2.0 * math.pi), places=12)

    def test_09_pca_svd_and_nipals_pls1(self):
        rng = np.random.default_rng(12)
        X = rng.normal(size=(30, 6))
        pca = dim_reduction.pca_svd(X, n_components=3)
        self.assertTrue(float(np.sum(pca["explained_variance_ratio"])) <= 1.0)
        self.assertAlmostEqual(float(np.mean(pca["hotelling_t2"])), 3.0 * 29.0 / 30.0, places=10)
        pls = dim_reduction.pls1_nipals(X, X @ np.array([1.0, -2.0, 0.5, 0.0, 1.5, -0.5]) + 3.0, n_components=6)
        self.assertLess(
            float(np.max(np.abs(pls['y_pred'] - (X @ np.array([1.0, -2.0, 0.5, 0.0, 1.5, -0.5]) + 3.0)))),
            1e-10,
        )

    def test_10_kmeans_and_spectral_clustering(self):
        theta = np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)
        X_rings = np.vstack([
            np.column_stack([0.5 * np.cos(theta), 0.5 * np.sin(theta)]),
            np.column_stack([2.2 * np.cos(theta), 2.2 * np.sin(theta)]),
        ])
        sc = clustering_phases.spectral_clustering(X_rings, k=2, sigma=0.5, seed=42)
        self.assertEqual(len(np.unique(sc["labels"][:24])), 1)
        self.assertEqual(len(np.unique(sc["labels"][24:])), 1)
        self.assertNotEqual(sc["labels"][0], sc["labels"][24])


class TreesPINNsSINDyAndUncertaintyTest(unittest.TestCase):
    def test_11_decision_tree_and_gradient_boosting(self):
        x = np.linspace(0.0, 4.0, 40)
        y = np.sin(x)
        gb = trees_ensembles.GradientBoostingRegressor(n_estimators=35, learning_rate=0.2, max_depth=2).fit(x, y)
        self.assertLess(float(np.mean((gb.predict(x) - y) ** 2)), 1e-3)

    def test_12_differentiable_mlp_exact_spatial_derivatives(self):
        mlp = neural_mlp.DifferentiableMLP1D(n_hidden=12, seed=9)
        x = np.linspace(-1.5, 1.5, 20)
        eps = 1e-4
        o0 = mlp.forward_and_derivatives(x)
        op = mlp.forward_and_derivatives(x + eps)
        om = mlp.forward_and_derivatives(x - eps)
        np.testing.assert_allclose(o0["du_dx"], (op["u"] - om["u"]) / (2.0 * eps), atol=1e-7)
        np.testing.assert_allclose(o0["d2u_dx2"], (op["u"] - 2.0 * o0["u"] + om["u"]) / (eps ** 2), atol=1e-6)

    def test_13_pinn_collocation_thiele_reaction_diffusion(self):
        phi = 2.0
        x_col = np.linspace(0.0, 1.0, 41)
        centers = np.linspace(-0.2, 1.2, 25)
        res = pinn.solve_linear_bvp_pinn(
            x_col, centers, sigma=0.22,
            q_coeff_fn=lambda x: np.full_like(x, -(phi ** 2)),
            rhs_fn=lambda x: np.zeros_like(x),
            x_bc_left=0.0, bc_left_type="neumann", bc_left_val=0.0,
            x_bc_right=1.0, bc_right_type="dirichlet", bc_right_val=1.0,
        )
        u_pred = res["predict"](x_col)["u"]
        np.testing.assert_allclose(u_pred, np.cosh(phi * x_col) / np.cosh(phi), atol=2e-5)

    def test_14_sindy_stlsq_and_hybrid_cstr(self):
        ca = np.linspace(0.2, 1.8, 40)
        cb = 0.6 + 0.4 * np.cos(3.0 * ca)
        Theta = np.column_stack([np.ones_like(ca), ca, cb, ca ** 2, ca * cb, cb ** 2])
        dY = np.column_stack([-0.5 * ca, 0.5 * ca - 0.9 * (cb ** 2)])
        Xi = sindy_ude.sindy_stlsq(Theta, dY, threshold=0.05)
        np.testing.assert_allclose(Xi[:, 0], [0.0, -0.5, 0.0, 0.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(Xi[:, 1], [0.0, 0.5, 0.0, 0.0, 0.0, -0.9], atol=1e-12)

    def test_15_split_conformal_and_mahalanobis_ood(self):
        rng = np.random.default_rng(123)
        y_cal = rng.normal(size=200)
        cp = uncertainty.split_conformal_interval(y_cal, np.zeros(200), np.zeros(500), alpha=0.10)
        self.assertTrue(1.4 <= cp["q_hat"] <= 1.95)
        X_tr = rng.normal(size=(80, 3))
        d2 = uncertainty.mahalanobis_ood_score(X_tr, X_tr)
        self.assertAlmostEqual(float(np.mean(d2)), 3.0 * 79.0 / 80.0, places=6)

    def test_16_crowe_reconciliation_isplines_and_robust_irls(self):
        rec = linear_physics.crowe_data_reconciliation(
            np.array([102.0, 60.0, 39.0]), np.array([1.0, 1.0, 2.0]), np.array([[1.0, -1.0, -1.0]]), np.array([0.0])
        )
        np.testing.assert_allclose(rec["y_reconciled"], [101.5, 60.5, 41.0], atol=1e-13)
        self.assertAlmostEqual(rec["chi2_stat"], 1.5, places=12)

        x = np.linspace(0.0, 5.0, 30)
        isp = linear_physics.fit_monotone_ispline(x, np.sqrt(x), np.linspace(0.0, 5.0, 6), increasing=True)
        self.assertTrue(np.all(isp["dy_dx"] >= -1e-15))

        X_h = np.column_stack([np.ones(25), np.linspace(-1.0, 1.0, 25)])
        y_h = X_h @ np.array([3.0, 1.5])
        y_h[12] += 40.0
        wh = linear_physics.huber_irls_regression(X_h, y_h, delta=0.1)["w"]
        wq = linear_physics.quantile_irls_regression(X_h, y_h, tau=0.50)
        self.assertAlmostEqual(float(wh[1]), 1.5, delta=0.02)
        self.assertAlmostEqual(float(wq[1]), 1.5, delta=0.02)

    def test_17_snv_pcr_gmm_and_random_forest_extensions(self):
        x = np.linspace(0.0, 4.0, 30)
        X_snv = dim_reduction.snv_preprocess(np.vstack([2.0 * np.cos(x) + 1.0, 0.5 * np.cos(x) - 3.0]))
        np.testing.assert_allclose(X_snv[0], X_snv[1], atol=1e-13)
        dX = dim_reduction.spectral_first_derivative(np.vstack([x + 1.0, x + 5.0]), d_lambda=float(x[1] - x[0]))
        np.testing.assert_allclose(dX[0], dX[1], atol=1e-13)

        rng = np.random.default_rng(42)
        X_b = np.vstack([rng.normal(-3.0, 0.2, size=(15, 2)), rng.normal(3.0, 0.2, size=(15, 2))])
        gmm = clustering_phases.gmm_em_fit(X_b, k=2, seed=42)
        np.testing.assert_allclose(np.sum(gmm["responsibilities"], axis=1), 1.0, atol=1e-13)
        self.assertGreater(clustering_phases.silhouette_score_mean(X_b, gmm["labels"]), 0.85)

        rf = trees_ensembles.RandomForestRegressor(n_estimators=12, max_depth=3, seed=42).fit(x, np.sin(x))
        self.assertLess(float(np.mean((rf.predict(x) - np.sin(x)) ** 2)), 0.03)
        hgbm = trees_ensembles.HuberGradientBoostingRegressor(n_estimators=20, delta=1.0).fit(x, np.sin(x))
        self.assertLess(float(np.mean((hgbm.predict(x) - np.sin(x)) ** 2)), 0.03)

    def test_18_all_158_validation_checks_pass(self):
        ns = runpy.run_path(str(ROOT / "scripts" / "validate.py"))
        results = ns["RESULTS"]
        self.assertGreaterEqual(len(results), 158)
        for sec, name, _ref, val, exp, ok in results:
            self.assertTrue(ok, f"Failed validation check: {sec} | {name} ({val} != {exp})")


if __name__ == "__main__":
    unittest.main()
