"""Acquisition functions for Bayesian optimization: probability of improvement, expected improvement, upper confidence bound, constrained expected improvement, and the straddle score for level sets."""

import math

import numpy as np


def _std_normal_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * (z ** 2)) / np.sqrt(2.0 * np.pi)


def _std_normal_cdf(z: np.ndarray) -> np.ndarray:
    vec_erf = np.vectorize(math.erf, otypes=[float])
    return 0.5 * (1.0 + vec_erf(z / np.sqrt(2.0)))


def probability_of_improvement(mu: np.ndarray, sigma: np.ndarray, y_best: float, xi: float = 0.01) -> np.ndarray:
    """Analytical Probability of Improvement PI(x) = Phi((mu(x) - y_best - xi) / sigma(x))."""
    m = np.asarray(mu, dtype=float)
    s = np.asarray(sigma, dtype=float)
    imp = m - y_best - xi
    safe_s = np.where(s < 1e-12, 1.0, s)
    Z = imp / safe_s
    pi = _std_normal_cdf(Z)
    return np.where(s < 1e-12, np.where(imp > 0.0, 1.0, 0.0), pi)


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, y_best: float, xi: float = 0.01) -> np.ndarray:
    """
    Analytical Expected Improvement (EI) for maximization:
      EI(x) = (mu - y_best - xi) * Phi(Z) + sigma * phi(Z),   Z = (mu - y_best - xi) / sigma.
    """
    m = np.asarray(mu, dtype=float)
    s = np.asarray(sigma, dtype=float)
    imp = m - y_best - xi
    safe_s = np.where(s < 1e-12, 1.0, s)
    Z = imp / safe_s
    ei = imp * _std_normal_cdf(Z) + safe_s * _std_normal_pdf(Z)
    return np.where(s < 1e-12, np.maximum(0.0, imp), ei)


def ucb_acquisition(mu: np.ndarray, sigma: np.ndarray, kappa: float = 2.0) -> np.ndarray:
    """Upper Confidence Bound acquisition function UCB(x) = mu(x) + kappa * sigma(x)."""
    return np.asarray(mu, dtype=float) + kappa * np.asarray(sigma, dtype=float)


def straddle_level_set_acquisition(
    mu: np.ndarray, sigma: np.ndarray, threshold: float = 0.0, beta: float = 1.96
) -> np.ndarray:
    """
    Bryan's (2005) Straddle heuristic for active phase-boundary / level-set discovery f(x) = threshold:
      a_straddle(x) = beta * sigma(x) - |mu(x) - threshold|.
    High values occur where predictive uncertainty is large AND the predicted mean sits near the phase boundary.
    """
    m = np.asarray(mu, dtype=float)
    s = np.asarray(sigma, dtype=float)
    return beta * s - np.abs(m - float(threshold))


def constrained_expected_improvement(
    mu_obj: np.ndarray,
    sigma_obj: np.ndarray,
    y_best: float,
    mu_con: np.ndarray,
    sigma_con: np.ndarray,
    con_upper: float = 0.0,
    xi: float = 0.01,
) -> np.ndarray:
    """
    Gardner et al. (2014) Constrained Expected Improvement:
      cEI(x) = EI(x) * P(c(x) <= con_upper) = EI(x) * Phi((con_upper - mu_con(x)) / sigma_con(x)).
    """
    ei = expected_improvement(mu_obj, sigma_obj, y_best=y_best, xi=xi)
    mc = np.asarray(mu_con, dtype=float)
    sc = np.asarray(sigma_con, dtype=float)
    safe_sc = np.where(sc < 1e-12, 1.0, sc)
    z_feas = (float(con_upper) - mc) / safe_sc
    p_feas = np.where(sc < 1e-12, np.where(mc <= con_upper, 1.0, 0.0), _std_normal_cdf(z_feas))
    return ei * p_feas
