# Changelog

## 1.0.0 (2026-11-26)

First release. An open book of 18 executed lectures in six parts, each with four
demonstrations, two figures, three exercises and executed worked solutions, and the
`engml_2_applied` package underneath.

### The book

- **Part I. Conservation as a constraint** (Lectures 0 to 2): atom, mass and charge
  balances as a linear system and the reactions as its null space; least squares with
  exact equality constraints; data reconciliation and the detection of gross errors.
- **Part II. Monotone, convex and robust fits** (Lectures 3 to 5): the
  pool-adjacent-violators algorithm, with the van der Waals loop and Maxwell's equal-area
  rule as a check; I-splines; Huber and quantile regression.
- **Part III. Chemometrics and process monitoring** (Lectures 6 to 9): the Beer–Lambert law,
  the standard normal variate and derivatives; principal components and regression on them;
  partial least squares; Hotelling's $T^2$ and the residual $Q$.
- **Part IV. Clustering and phases** (Lectures 10 to 12): K-means and the silhouette;
  spectral clustering; Gaussian mixtures, the entropy of their responsibilities and the
  Gibbs phase rule.
- **Part V. Trees and ensembles** (Lectures 13 to 15): regression trees; bagging, random
  forests and permutation importance; gradient boosting with the squared and the Huber loss.
- **Part VI. Prediction intervals and a capstone** (Lectures 16 and 17): split conformal
  prediction and the Mahalanobis distance; a soft sensor on synthetic spectra whose
  predictions satisfy a carbon balance, carry an interval and are withheld when the spectrum
  is disturbed.
- Every lecture compares the library with a calculation by hand and, where one exists, with an
  exact result. The statements in the text are those of the stored outputs: every number quoted
  in a lecture can be found in the output above it.
- All data sets are synthetic and labelled so.
- A book site (Jupyter Book 2) with an introduction, a 14-week syllabus, notation, the
  validation report and suggested references.

### The package

- `engml_2_applied` 1.0.0 (distribution name `engml-2-applied`): `linear_physics`,
  `dim_reduction`, `clustering_phases`, `trees_ensembles`, `uncertainty` and `plotting` are
  used in this volume. The modules of Volume I (`constants`, `foundations`, `descriptors`) and
  of Volume III (`kernels_gp`, `bo_active`, `neural_mlp`, `pinn`, `sindy_ude`) are included
  with their tests; each volume has its own package (`engml_1_foundations`,
  `engml_3_frontier`).
- 158 checks of the package against exact identities, limits and reference values
  (`scripts/validate.py`, reported in `docs/validation.md`), run by `pytest` together with
  checks that all 36 notebooks are stored executed and free of errors.
