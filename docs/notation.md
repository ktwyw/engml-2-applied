# Notation and conventions

## Conventions

- **Units.** Quantities are in SI units unless a symbol says otherwise. Flows of the reconciliation
  examples are in kg/h, amounts of species in mol, pressures of the isotherms in MPa or bar, and
  absorbances in absorbance units (AU). The van der Waals isotherms of Lecture 3 are in reduced
  variables.
- **Lectures** are numbered from 0 to 17. Equations are numbered within a lecture, (1.1), (1.2), …,
  and figures by lecture, Figure 6.1, Figure 6.2.
- **Vectors and matrices.** Bold lower-case letters are column vectors, bold capitals are matrices.
  Rows of a data matrix are samples (spectra, sets of sensor readings) and columns are variables
  (channels, sensors).
- **Indices in code** start at 0, as in Python. In the text, sensors and streams are numbered from 1.
- **Synthetic data** are generated in the notebooks with a fixed seed and are labelled as synthetic.
  Assumed parameters are labelled as assumed.
- **Errors.** "Error against the true curve" means the root-mean-square difference from the function
  that generated the synthetic data, which is known because the data are synthetic.

## Symbols

| Symbol | Meaning | Unit |
|---|---|---|
| $\mathbf{C}$, $\mathbf{d}$ | matrix of a set of linear balances (atoms or charge per species) and its right-hand side | – |
| $\mathbf{w}$ (Lectures 0, 1) | amounts or yields of the species | mol |
| $\boldsymbol{\nu}$, $\boldsymbol{\xi}$ | stoichiometric matrix (a basis of the null space of $\mathbf{C}$); extents of reaction | –, mol |
| $\boldsymbol{\mu}$ | Lagrange multipliers of the constraints | – |
| $\mu_{\mathrm{pen}}$ | weight of a penalty used in place of a constraint | – |
| $\kappa$ | condition number, $\sigma_{\max}/\sigma_{\min}$ | – |
| $\mathbf{A}$, $\mathbf{b}$ (Lecture 2) | incidence matrix of the unit balances, $\mathbf{A}\mathbf{y} = \mathbf{b}$ | – |
| $\mathbf{y}$, $\boldsymbol{\Sigma}$ (Lecture 2) | measured flows and the covariance of their errors | kg/h, (kg/h)² |
| $\chi^2$, $z_i$ | global statistic of the adjustment; standardized adjustment of measurement $i$ | – |
| $P$, $V_m$, $T$ | pressure, molar volume, temperature | MPa, L/mol, K |
| $P_r$, $V_r$, $T_r$ | pressure, volume and temperature divided by their critical values | – |
| $A_m$ | molar Helmholtz energy; $P = -(\partial A_m/\partial V_m)_T$ | J/mol |
| $M_m(x)$, $I_m(x)$ | M-spline and I-spline basis functions; $I_m' = M_m \ge 0$ | – |
| $\boldsymbol{\beta}$ | weights of a spline fit (Lecture 4) or regression vector of a calibration (Lectures 7, 8) | – |
| $r_i$ | residual of sample $i$ | units of $y$ |
| $\delta$ | threshold of the Huber loss | units of $y$ |
| $\tau$ | quantile level of the pinball loss | – |
| MAD | median absolute deviation | – |
| $A_i(\lambda_j)$ | absorbance of spectrum $i$ at wavelength $\lambda_j$ | AU |
| $b_i$, $\alpha_i$ | path-length factor and baseline offset of spectrum $i$ (Lecture 6) | –, AU |
| $c_{i,s}$, $\varepsilon_s(\lambda)$ | concentration of species $s$ in sample $i$; its absorptivity | mol/L, AU L/mol |
| $\mathbf{X}_c$ | data matrix with the mean of each column subtracted | – |
| $\mathbf{T}$, $\mathbf{P}$ | scores and loadings of a principal component or PLS model | – |
| $\mathbf{W}$, $\mathbf{q}$ (Lecture 8) | weights and response loadings of PLS | – |
| $\lambda_a$ | eigenvalue of the covariance matrix: variance of component $a$ | – |
| $k$ | number of components or latent variables kept | – |
| $T^2$ | Hotelling's statistic, $\sum_a t_a^2/\lambda_a$ | – |
| $Q$ | squared norm of the residual of a principal component model (also called SPE) | – |
| $K$ | number of clusters or mixture components | – |
| $J$ | within-cluster sum of squares of K-means | – |
| $\bar{s}$ | mean silhouette | – |
| $\mathbf{W}$, $\mathbf{D}$ (Lecture 11) | matrix of similarities and the diagonal matrix of its row sums | – |
| $\mathbf{L}_{\mathrm{sym}}$ | normalized Laplacian, $\mathbf{I} - \mathbf{D}^{-1/2}\mathbf{W}\mathbf{D}^{-1/2}$ | – |
| $\sigma$ (Lecture 11) | bandwidth of the Gaussian similarity | units of $x$ |
| $\gamma_{i,k}$ | responsibility of component $k$ for point $i$ | – |
| $H$ | entropy of the responsibilities, $-\sum_k \gamma_k \ln \gamma_k$ | nats |
| BIC | Bayesian information criterion | – |
| $F = C - P + 2$ | Gibbs phase rule: degrees of freedom, components, phases | – |
| $B$, $\rho$ | number of trees of a forest; correlation between their predictions | – |
| $M$, $\eta$ | number of stages and learning rate of gradient boosting | – |
| HV | Vickers hardness | – |
| $\alpha$ (Lecture 16) | allowed fraction of misses; the target coverage is $1 - \alpha$ | – |
| $\hat{q}$ | half-width of a split conformal interval | units of $y$ |
| $D_M^2$ | squared Mahalanobis distance from the training data | – |
| RMSE, MSE | root-mean-square error, mean squared error | – |

Several letters have more than one meaning ($\mathbf{W}$, $\mathbf{P}$, $\mathbf{D}$, $\sigma$,
$\alpha$, $\boldsymbol{\beta}$); each lecture defines its symbols where they first appear.
