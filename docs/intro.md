# Machine learning and physics-informed modelling for chemical and materials engineers. Volume II: Applied and physics-constrained modelling

*Models that respect balances, monotonicity and the structure of spectra, and that say how far
they can be trusted, with every method written out in NumPy and checked against an exact result.*

A model fitted to process or materials data is asked to do more than reproduce the data. Its
predicted yields should close the atom balances. A fitted isotherm should not let the pressure
rise with the volume. A calibration of a spectrometer should survive a change of path length in
the probe. A prediction should come with an interval, and with a warning when the input is unlike
anything in the training set. This volume teaches the methods that provide these properties, in
six parts: conservation laws as exact constraints; monotone, convex and robust fits; the
chemometric methods for spectra and correlated sensors; clustering; trees and their ensembles;
and prediction intervals, with a capstone that assembles a soft sensor from the parts.

Each lecture is an executed Jupyter notebook of about one hour. It states the principle and its
equations, works through four short demonstrations, and then repeats by hand the calculation of
the function of the package that it has used, to show that there is nothing hidden in it. The
notebooks share a small Python package, `engml_2_applied`, written to be read as much as to be
used; its core needs NumPy alone. Each lecture has two numbered figures and ends with three
exercises: one that verifies a result, one that looks for a way in which the method fails, and one
that asks the reader to write a function. A worked solution to every exercise is given in the
*Solutions* part of the book.

All data sets in this volume are synthetic: they are generated in the notebooks from stated
formulas, seeds and assumed constants, so that the answer is known and a method can be judged
against it. The results show how the methods behave and are not estimates of the accuracy to be
expected on a plant.

The level is that of the third or fourth year of an engineering degree or the first year of
graduate study. The lectures assume linear algebra up to the singular value decomposition, least
squares, material balances, some thermodynamics, and Python with NumPy, all of which are covered
in [Volume I](https://github.com/ktwyw/engml-1-foundations). Symbols are collected in
[Notation and conventions](notation.md), a plan for a semester in the [syllabus](syllabus.md), the
checks of the package in the [validation report](validation.md), and books and articles for
further study in [Suggested references](references.md).

## Part I. Conservation as a constraint

| Lecture | Topics |
|---|---|
| [0. Atom, mass and charge balances as linear constraints](../notebooks/part1_conservation/00_elemental_mass_and_charge_conservation_matrices.ipynb) | the matrix of atom counts and charges; total mass as a dependent row; the null space and extents of reaction; projection of a noisy analysis onto the balances |
| [1. Least squares with exact constraints: the KKT system](../notebooks/part1_conservation/01_kkt_equality_constrained_regression.ipynb) | the KKT system; the correction that turns the unconstrained solution into the constrained one; the covariance of the constrained estimate; penalties and conditioning |
| [2. Data reconciliation and the detection of gross errors](../notebooks/part1_conservation/02_process_data_reconciliation_and_gross_error_detection.ipynb) | weighted reconciliation; a splitter by hand; a reactor, separator and recycle; the global $\chi^2$ test and the measurement test |

## Part II. Monotone, convex and robust fits

| Lecture | Topics |
|---|---|
| [3. Monotone fits and the pool-adjacent-violators algorithm](../notebooks/part2_shape/03_thermodynamic_stability_and_pava_isotonic_regression.ipynb) | convexity of the Helmholtz energy and a monotone pressure; the pool-adjacent-violators algorithm by hand and with a stack; comparison with polynomials; the van der Waals loop |
| [4. Smooth monotone fits with I-splines](../notebooks/part2_shape/04_monotone_splines_and_convex_free_energy_potentials.ipynb) | M-splines and I-splines; projected coordinate descent with non-negative weights; value and slope against the staircase and a polynomial; a convex free energy |
| [5. Robust regression: the Huber loss and quantile regression](../notebooks/part2_shape/05_outlier_robust_regression_huber_irls_and_quantile_loss.ipynb) | the Huber loss and the weights of IRLS; median and quantile regression with the pinball loss; the choice of the threshold |

## Part III. Chemometrics and process monitoring

| Lecture | Topics |
|---|---|
| [6. Spectra, the Beer–Lambert law and preprocessing](../notebooks/part3_chemometrics/06_beer_lambert_spectroscopy_and_spectral_preprocessing.ipynb) | the Beer–Lambert law for a mixture; the standard normal variate; the first derivative and its noise; the effect on a PLS calibration |
| [7. Principal component analysis and regression on components](../notebooks/part3_chemometrics/07_svd_principal_component_analysis_and_pcr.ipynb) | principal components by SVD; orthogonality of loadings and scores; the residual of a truncated model; regression on components and the case in which it fails |
| [8. Partial least squares for soft sensors](../notebooks/part3_chemometrics/08_partial_least_squares_nipals_for_soft_sensors.ipynb) | the NIPALS algorithm; orthogonal scores; PLS against minimum-norm least squares with more channels than samples; the number of latent variables |
| [9. Process monitoring with Hotelling's T² and the Q residual](../notebooks/part3_chemometrics/09_multivariate_process_monitoring_hotelling_t2_and_q.ipynb) | the model plane and the residual; Hotelling's $T^2$ and $Q$; a throughput surge against a drifting sensor; contributions to $Q$ |

## Part IV. Clustering and phases

| Lecture | Topics |
|---|---|
| [10. Clustering with K-means, and where it fails](../notebooks/part4_clustering/10_unsupervised_phase_clustering_kmeans_and_voronoi.ipynb) | k-means++ seeding and the two alternating steps; the silhouette; concentric rings, where the method fails |
| [11. Spectral clustering with the normalized Laplacian](../notebooks/part4_clustering/11_normalized_laplacian_spectral_clustering_manifolds.ipynb) | the graph of Gaussian similarities; the normalized Laplacian; clustering its eigenvectors; the bandwidth and the gap between eigenvalues |
| [12. Gaussian mixtures and the Gibbs phase rule](../notebooks/part4_clustering/12_gaussian_mixture_models_and_gibbs_phase_rule.ipynb) | the EM algorithm; responsibilities; their entropy at boundaries; the Bayesian information criterion and the Gibbs phase rule |

## Part V. Trees and ensembles

| Lecture | Topics |
|---|---|
| [13. Regression trees and splitting by running sums](../notebooks/part5_ensembles/13_decision_trees_cart_and_prefix_sum_splitting.ipynb) | the best split from running sums; trees of increasing depth; conservation of the mean; constant predictions outside the training range |
| [14. Bagging, random forests and permutation importance](../notebooks/part5_ensembles/14_random_forests_bagging_and_feature_importance.ipynb) | bootstrap samples and averaging; the variance of an average of correlated estimates; permutation importance; the number of trees |
| [15. Gradient boosting and the Huber loss](../notebooks/part5_ensembles/15_functional_gradient_boosting_and_huber_clipping.ipynb) | fitting residuals in stages; the learning rate; clipped residuals of the Huber loss; data with gross outliers |

## Part VI. Prediction intervals and a capstone

| Lecture | Topics |
|---|---|
| [16. Conformal prediction intervals and out-of-distribution alarms](../notebooks/part6_guardrails/16_split_conformal_prediction_and_mahalanobis_ood.ipynb) | split conformal intervals and their finite-sample quantile; coverage under skewed noise; the Mahalanobis distance; a query inside every single range and outside the data |
| [17. Capstone: a soft sensor with a carbon balance, intervals and an alarm](../notebooks/part6_guardrails/17_capstone_industrial_soft_sensor_and_reconciliation.ipynb) | SNV, three PLS models, projection onto the carbon balance, weighted reconciliation, conformal intervals, and a limit on $Q$ that withholds a prediction |

## Where to start

| If you are… | Read |
|---|---|
| working with plant data and balances | Lectures 0 to 2, then 9 |
| fitting isotherms, rheology or calibration curves | Lectures 3 to 5 |
| calibrating a spectrometer | Lectures 6 to 8, then 17 |
| looking for groups in unlabelled data | Lectures 10 to 12 |
| using tree ensembles | Lectures 13 to 15, then 16 |
| short of time | Lectures 1, 3, 8 and 16 |

## The three volumes

This is the second of three volumes. [Volume I](https://github.com/ktwyw/engml-1-foundations)
covers scientific Python, dimensional analysis, corresponding states, molecular graph descriptors
and regularized regression. [Volume III](https://github.com/ktwyw/engml-3-frontier) covers Gaussian processes, Bayesian
optimization, physics-informed collocation and the identification of rate laws. Each volume is
a separate repository with its own package (`engml_1_foundations`, `engml_2_applied`,
`engml_3_frontier`).

## Companion books

- [engml-1-foundations](https://github.com/ktwyw/engml-1-foundations) and [engml-3-frontier](https://github.com/ktwyw/engml-3-frontier): Volumes I and III of this series.
- [engtransport](https://github.com/ktwyw/engtransport): heat and mass transfer through worked problems.
- [onsager-variational-methods](https://github.com/ktwyw/onsager-variational-methods): variational methods in soft matter.
- [error-analysis](https://github.com/ktwyw/error-analysis): evaluating, propagating and reporting uncertainty.
- [from-formulas-to-programs](https://github.com/ktwyw/from-formulas-to-programs): a first course in programming for engineers.

## Author and licence

Yanwei Wang, Department of Chemical and Materials Engineering, School of Engineering, Nazarbayev
University, Astana, Kazakhstan (yanwei.wang@nu.edu.kz, ORCID
[0000-0002-8488-9833](https://orcid.org/0000-0002-8488-9833)). The code is under the MIT licence;
the text, figures and notebook outputs are under CC BY 4.0. The source is at
[github.com/ktwyw/engml-2-applied](https://github.com/ktwyw/engml-2-applied).
