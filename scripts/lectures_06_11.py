"""Text and code of lectures 06 to 11, as plain data for scripts/build_notebooks.py."""

LECTURES_06_11 = [
    {
        "num": r'''06''',
        "slug": r'''06_beer_lambert_spectroscopy_and_spectral_preprocessing''',
        "title": r'''6. Spectra, the Beer–Lambert law and preprocessing''',
        "hook": r'''A near-infrared spectrum of a process stream depends on the composition, and also on the path length, on scattering and on the baseline of the instrument. This lecture states the Beer–Lambert law for a mixture and two standard corrections: the standard normal variate (SNV), which removes a multiplicative factor and an offset from each spectrum, and the first derivative, which removes a constant baseline. Their effect on a calibration is measured on synthetic spectra.''',
        "objectives": [
            r'''State the Beer–Lambert law for a mixture, with a path length and a baseline for each spectrum.''',
            r'''Show, and check numerically, that the standard normal variate (`dim_reduction.snv_preprocess`) removes a positive scale factor and a constant offset.''',
            r'''Use the first derivative of a spectrum (`dim_reduction.spectral_first_derivative`) to remove a constant baseline, and see what it does to the noise.''',
            r'''Compare a PLS calibration before and after each preprocessing step.''',
        ],
        "theory_md": r'''## 1. Principles and governing equations
By the **multicomponent Beer–Lambert law**, the absorbance spectrum $A_i(\lambda_j)$ ($\mathrm{AU}$, dimensionless absorbance units) of sample $i$ across $d$ wavelengths $\lambda_1 < \dots < \lambda_d$ ($\mathrm{nm}$) is a linear superposition of $S$ species molar concentrations $c_{i,s}$ ($\mathrm{mol/L}$) weighted by molar absorptivities $\varepsilon_s(\lambda_j)$ ($\mathrm{L/(mol\cdot cm)}$), corrupted by effective optical path length $b_i$ ($\mathrm{cm}$) and baseline drift $\alpha_i + \beta_i\lambda_j$:
$$A_i(\lambda_j) = b_i \sum_{s=1}^S c_{i,s}\,\varepsilon_s(\lambda_j) + \alpha_i + \beta_i \lambda_j + \varepsilon_{i,j} \tag{1.1}$$
1. **Standard normal variate (SNV, Barnes et al. 1989):** Centers each spectrum across wavelengths and divides by its row standard deviation:
   $$A_{i,j}^{\mathrm{SNV}} = \frac{A_{i,j} - \bar{A}_i}{s_i}, \qquad \bar{A}_i = \frac{1}{d}\sum_{k=1}^d A_{i,k}, \quad s_i = \sqrt{\frac{1}{d}\sum_{k=1}^d (A_{i,k} - \bar{A}_i)^2} \tag{1.2}$$
   For any pure path-length scaling $b_i > 0$ and offset $\alpha_i$ ($\beta_i = 0$), $A_{i,j} = b_i A_{0,j} + \alpha_i \implies A_{i,j}^{\mathrm{SNV}} \equiv A_{0,j}^{\mathrm{SNV}}$.
2. **First derivative of the spectrum ($dA/d\lambda$):** Differentiating with respect to wavelength $\lambda$ annihilates $\alpha_i$ and turns linear tilts $\beta_i\lambda_j$ into a constant offset $\beta_i$ that a subsequent SNV pass removes completely:
   $$\frac{\partial A_i(\lambda_j)}{\partial \lambda} = b_i \sum_{s=1}^S c_{i,s}\,\varepsilon_s'(\lambda_j) + \beta_i \tag{1.3}$$''',
        "demo1_code": r'''heading("Demonstration 1: Exact Path-Length & Offset Invariance of Standard Normal Variate (SNV)")
wl = np.linspace(1100.0, 1700.0, 120)
pure_peak1 = np.exp(-0.5 * ((wl - 1320.0) / 45.0) ** 2)
pure_peak2 = np.exp(-0.5 * ((wl - 1480.0) / 50.0) ** 2)
base_spectrum = 0.6 * pure_peak1 + 0.4 * pure_peak2

# Create 10 spectra with varying multiplicative scatter b_i in [0.5, 2.5] and additive offset alpha_i in [-1, 3]:
b_scales = np.linspace(0.5, 2.5, 10)[:, None]
a_offsets = np.linspace(-1.0, 3.0, 10)[:, None]
X_scattered = b_scales * base_spectrum[None, :] + a_offsets

X_snv = dim_reduction.snv_preprocess(X_scattered)
raw_spread = float(np.max(np.abs(X_scattered - X_scattered[0:1, :])))
max_snv_spread = float(np.max(np.abs(X_snv - X_snv[0:1, :])))
print(f"Max pairwise difference across 10 distorted spectra before snv: {raw_spread:.4f} AU")
print(f"Max pairwise difference across 10 distorted spectra AFTER  SNV: {max_snv_spread:.2e} AU")
assert np.allclose(np.mean(X_snv, axis=1), 0.0, atol=1e-14)
assert np.allclose(np.std(X_snv, axis=1), 1.0, atol=1e-14)
assert max_snv_spread < 1e-13''',
        "post1_md": r'''**What Demonstration 1 shows.** The ten synthetic spectra are one spectrum seen through different path lengths and offsets, and before the transformation they differ by up to $5.2\;\mathrm{AU}$. After SNV they coincide to $2\times 10^{-15}$: the transformation removes a multiplicative factor and an additive constant exactly.

---

## 2. Numerical walkthrough
What if the probe window also introduces a wavelength-dependent **linear baseline tilt** $\beta_i(\lambda - \lambda_0)$? SNV alone cannot remove a tilt because $\beta_i\lambda_j$ varies across wavelengths. Let us see how combining `spectral_first_derivative` with `snv_preprocess` removes $\alpha_i$, $\beta_i\lambda_j$, and $b_i$ simultaneously.''',
        "demo2_code": r'''heading("Demonstration 2: Spectral First Derivative + SNV Annihilation of Linear Baseline Tilt")
d_wl = float(wl[1] - wl[0])
X_offset_only = base_spectrum[None, :] + a_offsets
dX_dlambda = dim_reduction.spectral_first_derivative(X_offset_only, d_lambda=d_wl)
max_deriv_spread = float(np.max(np.abs(dX_dlambda - dX_dlambda[0:1, :])))
print(f"Max spread after 1st spectral derivative on offset spectra : {max_deriv_spread:.2e}")
assert max_deriv_spread < 1e-13

# Now combine multiplicative scatter b_i, constant offset a_i, AND linear tilt beta_i * (wl - 1400):
beta_tilts = np.linspace(-0.0015, 0.0015, 10)[:, None]
X_tilted_scattered = b_scales * base_spectrum[None, :] + a_offsets + (b_scales * 0.0005) * (wl[None, :] - 1400.0)
X_d1_snv = dim_reduction.snv_preprocess(dim_reduction.spectral_first_derivative(X_tilted_scattered, d_lambda=d_wl))
print(f"Max spread after (1st Derivative -> SNV) on tilted spectra: {np.max(np.abs(X_d1_snv - X_d1_snv[0:1, :])):.2e}")
assert np.max(np.abs(X_d1_snv - X_d1_snv[0:1, :])) < 1e-13''',
        "post2_md": r'''**What Demonstration 2 shows.** Taking $\partial A / \partial \lambda$ converts the linear tilt into a constant baseline offset, and the subsequent SNV normalization divides out the path-length multiplier $b_i$ and subtracts the constant tilt slope.

---

## 3. Worked case study
Let us calibrate a 2-component NIPALS PLS1 soft sensor (`dim_reduction.pls1_nipals`) on $N = 50$ binary mixture NIR spectra corrupted by a synthetic scatter of the path length $b_i \in [0.5, 2.0]$ and baseline offset $\alpha_i \in [-0.8, 0.8]$.''',
        "demo3_code": r'''heading("Demonstration 3: Soft-Sensor Calibration Benchmark — Raw vs. SNV Preprocessed NIR")
rng = np.random.default_rng(42)
N_cal = 50
c1 = rng.uniform(0.1, 0.9, size=N_cal)
scatter_b = rng.uniform(0.5, 2.0, size=(N_cal, 1))
offset_a = rng.uniform(-0.8, 0.8, size=(N_cal, 1))
# Analyte peak1 on top of a constant reference background matrix peak2:
X_raw_nir = scatter_b * (np.outer(c1, pure_peak1) + pure_peak2[None, :]) + offset_a + rng.normal(0.0, 0.002, size=(N_cal, len(wl)))

pls_raw = dim_reduction.pls1_nipals(X_raw_nir, c1, n_components=2)
pls_snv = dim_reduction.pls1_nipals(dim_reduction.snv_preprocess(X_raw_nir), c1, n_components=2)
rmse_raw = float(np.sqrt(np.mean((pls_raw["y_pred"] - c1) ** 2)))
rmse_snv = float(np.sqrt(np.mean((pls_snv["y_pred"] - c1) ** 2)))
print(f"2-Component PLS1 RMSE: Raw Scattered Spectra = {rmse_raw:.4f} mol/L -> SNV Preprocessed = {rmse_snv:.4f} mol/L")
assert rmse_snv < 0.35 * rmse_raw''',
        "post3_md": r'''**What Demonstration 3 shows.** Without SNV the two latent variables of the PLS model are spent on the path-length factor and the offset, and the calibration error is $0.095\;\mathrm{mol/L}$. With SNV applied first it is $0.0043\;\mathrm{mol/L}$, twenty times smaller. The spectra are synthetic.

---

## 4. What if: sensitivity to the parameters
Since differentiation amplifies high-frequency detector noise by $1/\Delta\lambda$, what happens as instrument white noise $\sigma_{\mathrm{det}}$ increases from $10^{-4}\;\mathrm{AU}$ to $2 \times 10^{-2}\;\mathrm{AU}$?''',
        "demo4_code": r'''heading("Demonstration 4: Detector Noise vs. Derivative Amplification Tradeoff")
print(f"{'Detector Noise [AU]':>19s} | {'Raw PLS1 RMSE':>14s} | {'SNV PLS1 RMSE':>14s} | {'Deriv+SNV RMSE':>15s}")
print("-" * 70)
for sig_det in [1e-4, 1e-3, 5e-3, 2e-2]:
    X_n = scatter_b * (np.outer(c1, pure_peak1) + pure_peak2[None, :]) + offset_a + rng.normal(0.0, sig_det, size=(N_cal, len(wl)))
    r_raw = float(np.sqrt(np.mean((dim_reduction.pls1_nipals(X_n, c1, 2)["y_pred"] - c1) ** 2)))
    r_snv = float(np.sqrt(np.mean((dim_reduction.pls1_nipals(dim_reduction.snv_preprocess(X_n), c1, 2)["y_pred"] - c1) ** 2)))
    X_ds = dim_reduction.snv_preprocess(dim_reduction.spectral_first_derivative(X_n, d_lambda=d_wl))
    r_ds = float(np.sqrt(np.mean((dim_reduction.pls1_nipals(X_ds, c1, 2)["y_pred"] - c1) ** 2)))
    print(f"{sig_det:19.4f} | {r_raw:14.4f} | {r_snv:14.4f} | {r_ds:15.4f}")''',
        "post4_md": r'''**What Demonstration 4 shows.** These spectra have no tilted baseline, so the derivative has nothing to remove and only its cost is seen: differencing neighbouring channels amplifies detector noise. With noise of $0.02\;\mathrm{AU}$ the error is $0.0056\;\mathrm{mol/L}$ with SNV alone and $0.0136\;\mathrm{mol/L}$ with the derivative before it. In practice the derivative is combined with smoothing.

---

## Inside the algorithm
Below we verify our 3-line row-wise SNV standardization in pure NumPy against `dim_reduction.snv_preprocess`.''',
        "inside_code": r'''heading("Inside the Algorithm: By-Hand Row-Wise SNV vs. the Library")
mu_row = np.mean(X_scattered, axis=1, keepdims=True)
std_row = np.std(X_scattered, axis=1, keepdims=True)
X_snv_hand = (X_scattered - mu_row) / std_row
diff_snv = float(np.max(np.abs(X_snv_hand - X_snv)))
print("by hand (pure NumPy) row 0[:5]:", np.round(X_snv_hand[0, :5], 6))
print("library (engml_2_applied)      row 0[:5]:", np.round(X_snv[0, :5], 6))
print(f"Max discrepancy               : {diff_snv:.2e}")
assert diff_snv < 1e-15''',
        "ex1_prompt": r'''Corrupt the binary mixture spectra `X_raw_nir` with linear baseline tilt $\beta_i (\lambda - 1400)$ ($\beta_i \in [-0.001, 0.001]$). Apply `spectral_first_derivative` followed by `snv_preprocess`, fit a 2-component `pls1_nipals` model predicting $c_1$, and verify that preprocessing reduces calibration RMSE by more than $3\times$ compared to raw spectra.''',
        "ex1_code": r'''heading("Exercise 1: Combined Linear Baseline Tilt + Scatter Removal via Derivative + SNV")
tilt_beta = rng.uniform(-0.001, 0.001, size=(N_cal, 1))
X_drift = X_raw_nir + tilt_beta * (wl[None, :] - 1400.0)
X_prep = dim_reduction.snv_preprocess(dim_reduction.spectral_first_derivative(X_drift, d_lambda=d_wl))

rmse_drift_raw = float(np.sqrt(np.mean((dim_reduction.pls1_nipals(X_drift, c1, n_components=2)["y_pred"] - c1) ** 2)))
rmse_drift_prep = float(np.sqrt(np.mean((dim_reduction.pls1_nipals(X_prep, c1, n_components=2)["y_pred"] - c1) ** 2)))
print(f"Tilted+Scattered RMSE: Raw = {rmse_drift_raw:.4f} -> Derivative+SNV = {rmse_drift_prep:.4f}")
assert rmse_drift_prep < rmse_drift_raw / 3.0''',
        "ex1_exp": r'''When both linear tilt and multiplicative scatter are present, chaining `spectral_first_derivative` $\to$ `snv_preprocess` reduces the calibration error from $0.103$ to $0.0047\;\mathrm{mol/L}$.''',
        "ex2_prompt": r'''Explain and verify why SNV fails on a **single-analyte pure Beer–Lambert solution** without an internal reference peak or solvent background ($A_i(\lambda_j) = b_i c_i \varepsilon_1(\lambda_j) + \alpha_i$): show that SNV divides out $b_i c_i$ completely, collapsing all concentrations $c_i \in [0.1, 2.0]\;\mathrm{mol/L}$ onto the exact same normalized spectrum $\varepsilon_1^{\mathrm{SNV}}(\lambda)$.''',
        "ex2_code": r'''heading("Exercise 2: Single-Chromophore SNV Self-Cancellation Diagnostic Audit")
c_single = np.linspace(0.1, 2.0, 15)[:, None]
X_single_chrom = c_single * pure_peak1[None, :]
X_single_snv = dim_reduction.snv_preprocess(X_single_chrom)
spread_single = float(np.max(np.abs(X_single_snv - X_single_snv[0:1, :])))
print(f"Max spectral difference across 15 concentrations after SNV on single peak: {spread_single:.2e}")
assert spread_single < 1e-14''',
        "ex2_exp": r'''Because $c_i$ enters $A_i(\lambda_j) = (b_i c_i)\varepsilon_1(\lambda_j)$ purely as an overall scale factor when only one absorbing species is present, SNV cancels $c_i$ along with $b_i$. Quantitative SNV requires either an internal reference band (such as a solvent peak) or multiple varying species so that relative peak ratios encode concentration.''',
        "ex3_prompt": r'''Implement **multiplicative scatter correction (MSC, Geladi et al. 1985)** `msc_preprocess(X: np.ndarray, ref_spectrum: np.ndarray | None = None) -> np.ndarray`, which regresses each row spectrum $\mathbf{x}_i$ onto a reference spectrum $\mathbf{x}_{\mathrm{ref}}$ (defaulting to the column mean $\bar{\mathbf{x}}$) via simple linear regression $x_{i,j} \approx a_i + b_i x_{\mathrm{ref},j}$ and returns the scatter-corrected spectrum $(x_{i,j} - \hat{a}_i)/\hat{b}_i$. Verify on `X_scattered` that MSC recovers `base_spectrum` to $<10^{-13}$.''',
        "ex3_code": r'''heading("Exercise 3: Implement Multiplicative Scatter Correction (MSC) From Scratch")
def msc_preprocess(X: np.ndarray, ref_spectrum: np.ndarray | None = None) -> np.ndarray:
    """Multiplicative Scatter Correction (MSC) aligning each spectrum to ref_spectrum via OLS."""
    X_mat = np.atleast_2d(np.asarray(X, dtype=float))
    ref = np.mean(X_mat, axis=0) if ref_spectrum is None else np.asarray(ref_spectrum, dtype=float).ravel()
    ref_mean = float(np.mean(ref))
    ref_c = ref - ref_mean
    ref_var = float(np.dot(ref_c, ref_c))
    row_means = np.mean(X_mat, axis=1, keepdims=True)
    b_hat = ((X_mat - row_means) @ ref_c)[:, None] / max(ref_var, 1e-15)
    a_hat = row_means - b_hat * ref_mean
    return (X_mat - a_hat) / np.where(np.abs(b_hat) < 1e-15, 1.0, b_hat)

X_msc = msc_preprocess(X_scattered, ref_spectrum=base_spectrum)
msc_err = float(np.max(np.abs(X_msc - base_spectrum[None, :])))
print(f"Max reconstruction error of MSC against true base_spectrum: {msc_err:.2e} AU")
assert msc_err < 1e-13''',
        "ex3_takeaway": r'''**Discussion.** SNV and MSC are algebraically linked linear transformations ($A^{\mathrm{MSC}}$ is an affine rescaling of $A^{\mathrm{SNV}}$); SNV requires no stored reference spectrum, whereas MSC preserves the original physical absorbance units ($\mathrm{AU}$) of the reference spectrum.''',
    },
    {
        "num": r'''07''',
        "slug": r'''07_svd_principal_component_analysis_and_pcr''',
        "title": r'''7. Principal component analysis and regression on components''',
        "hook": r'''A spectrum of many channels from a mixture of three species has three degrees of freedom, apart from noise. Principal component analysis finds them from the singular value decomposition of the data. The lecture checks the identities that the decomposition satisfies, uses the leading components as regressors, and shows the weakness of that regression: the components are chosen without looking at the quantity to be predicted.''',
        "objectives": [
            r'''Compute principal components from the singular value decomposition of the centred data (`dim_reduction.pca_svd`).''',
            r'''Check the orthonormality of the loadings, the orthogonality of the scores and the sum of the variances.''',
            r'''Check that the residual of a truncated model equals the sum of the discarded eigenvalues.''',
            r'''Regress on the leading components (`dim_reduction.pcr_regression`) and find the case in which this fails: a quantity carried by a component of small variance.''',
        ],
        "theory_md": r'''## 1. Principles and governing equations
Given a column-centered spectral or sensor matrix $\mathbf{X}_c = \mathbf{X} - \mathbf{1}_N\boldsymbol{\mu}^\top \in \mathbb{R}^{N \times d}$ with thin singular value decomposition $\mathbf{X}_c = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ ($\boldsymbol{\Sigma} = \mathrm{diag}(\sigma_1, \dots, \sigma_r)$), the sample covariance matrix is $\mathbf{S}_{XX} = \frac{1}{N-1}\mathbf{X}_c^\top\mathbf{X}_c = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^\top$ with eigenvalues $\lambda_a = \sigma_a^2 / (N - 1)$. Retaining the top $k$ right singular vectors as the loading matrix $\mathbf{P}_k = [\mathbf{v}_1, \dots, \mathbf{v}_k] \in \mathbb{R}^{d \times k}$ gives the latent score matrix $\mathbf{T}_k$:
$$\mathbf{T}_k = \mathbf{X}_c\mathbf{P}_k = \mathbf{U}_k\boldsymbol{\Sigma}_k \in \mathbb{R}^{N \times k}, \qquad \mathbf{P}_k^\top\mathbf{P}_k = \mathbf{I}_k, \qquad \mathbf{T}_k^\top\mathbf{T}_k = (N - 1)\boldsymbol{\Lambda}_k \tag{1.1}$$
By the **Eckart–Young–Mirsky theorem**, $\hat{\mathbf{X}}_c^{(k)} = \mathbf{T}_k\mathbf{P}_k^\top$ is the optimal rank-$k$ approximation to $\mathbf{X}_c$ in Frobenius norm, and the total squared prediction error ($Q_{\mathrm{SPE}}$) equals the sum of the discarded eigenvalues:
$$\sum_{i=1}^N Q_{\mathrm{SPE},i} = \|\mathbf{X}_c - \mathbf{T}_k\mathbf{P}_k^\top\|_F^2 = \sum_{a=k+1}^{\min(N,d)} \sigma_a^2 = (N - 1)\sum_{a=k+1}^{\min(N,d)} \lambda_a \tag{1.2}$$
**Principal component regression (PCR)** regresses the centered target $\mathbf{y}_c = \mathbf{y} - \bar{y}\mathbf{1}_N$ onto $\mathbf{T}_k$:
$$\hat{\boldsymbol{\gamma}} = (\mathbf{T}_k^\top\mathbf{T}_k)^{-1}\mathbf{T}_k^\top\mathbf{y}_c, \qquad \hat{\boldsymbol{\beta}}_{\mathrm{PCR}} = \mathbf{P}_k\hat{\boldsymbol{\gamma}} = \sum_{a=1}^k \frac{\mathbf{u}_a^\top\mathbf{y}_c}{\sigma_a}\mathbf{v}_a \tag{1.3}$$''',
        "demo1_code": r'''heading("Demonstration 1: SVD Principal Component Analysis & Orthogonality Identities")
rng = np.random.default_rng(42)
N, d = 45, 25
# 3 true underlying chemical species driving 25 spectral wavelengths:
C_latent = rng.normal(0.0, 1.0, size=(N, 3))
S_spectra = rng.normal(0.0, 1.0, size=(3, d))
X_spec = C_latent @ S_spectra + rng.normal(0.0, 0.05, size=(N, d))

pca3 = dim_reduction.pca_svd(X_spec, n_components=3)
pca_all = dim_reduction.pca_svd(X_spec, n_components=25)

assert np.allclose(pca3["loadings"].T @ pca3["loadings"], np.eye(3), atol=1e-13)
assert np.allclose(pca3["scores"].T @ pca3["scores"], (N - 1) * np.diag(pca3["eigenvalues"]), rtol=1e-12)
assert np.isclose(np.sum(pca_all["explained_variance_ratio"]), 1.0, rtol=1e-14)
print("Top 5 eigenvalues lambda_a:", np.round(pca_all["eigenvalues"][:5], 4))
print("Top-3 PCA cumulative variance explained:", f"{100.0 * np.sum(pca3['explained_variance_ratio']):.2f}%")
assert np.sum(pca3["explained_variance_ratio"]) > 0.99''',
        "post1_md": r'''**What Demonstration 1 shows.** Three species generated the spectra of 25 channels, and the first three principal components carry $99.89\ \%$ of the variance. The eigenvalues fall from $9.7$ (third) to $0.008$ (fourth): the remaining 22 components are noise.

---

## 2. Numerical walkthrough
Let us verify the **Eckart–Young–Mirsky residual theorem** Eq. (1.2) across rank truncations $k \in \{1, 2, 3, 4, 5\}$: the sum of sample $Q_{\mathrm{SPE}}$ residuals identically equals $(N - 1)\sum_{a=k+1}^d \lambda_a$.''',
        "demo2_code": r'''heading("Demonstration 2: Eckart-Young-Mirsky Rank-k Residual Identity Verification")
print(f"{'Rank k':>6s} | {'Cumul Var [%]':>13s} | {'sum(Q_SPE)':>14s} | {'(N-1)*sum(lam_>k)':>18s} | {'Discrepancy':>12s}")
print("-" * 74)
for k_rank in [1, 2, 3, 4, 5]:
    p_k = dim_reduction.pca_svd(X_spec, n_components=k_rank)
    q_sum = float(np.sum(p_k["q_spe"]))
    lam_tail = (N - 1) * float(np.sum(pca_all["eigenvalues"][k_rank:]))
    disc = abs(q_sum - lam_tail)
    print(f"{k_rank:6d} | {100*np.sum(p_k['explained_variance_ratio']):13.3f} | {q_sum:14.6f} | {lam_tail:18.6f} | {disc:12.2e}")
    assert np.isclose(q_sum, lam_tail, rtol=1e-11)''',
        "post2_md": r'''**What Demonstration 2 shows.** The residual sum of squares after keeping $k$ components equals $(N-1)$ times the sum of the discarded eigenvalues, to rounding error. It falls from $430$ at $k = 2$ to $2.4$ at $k = 3$ and hardly changes after that, which identifies three as the number of independent species.

---

## 3. Worked case study
Let us calibrate a 3-component **principal component regression (PCR)** model (`dim_reduction.pcr_regression`) predicting a bulk analyte concentration $\mathbf{y}_{\mathrm{target}}$ driven by the 3 major species, and compare its coefficient norm and out-of-sample stability against unregularized 25-wavelength OLS.''',
        "demo3_code": r'''heading("Demonstration 3: 3-Component PCR vs. Full 25-Wavelength OLS Benchmark")
y_target = C_latent @ np.array([2.0, -1.5, 0.8]) + 4.0 + rng.normal(0.0, 0.04, size=N)
pcr3 = dim_reduction.pcr_regression(X_spec, y_target, n_components=3)
pcr_ols = dim_reduction.pcr_regression(X_spec, y_target, n_components=25)

C_test_lat = rng.normal(0.0, 1.0, size=(50, 3))
X_spec_test = C_test_lat @ S_spectra + rng.normal(0.0, 0.05, size=(50, d))
y_test_true = C_test_lat @ np.array([2.0, -1.5, 0.8]) + 4.0

pred_pcr3 = X_spec_test @ pcr3["coef"] + pcr3["intercept"]
pred_ols25 = X_spec_test @ pcr_ols["coef"] + pcr_ols["intercept"]
print(f"Coefficient L2 norm ||beta||_2: 3-PC PCR = {np.linalg.norm(pcr3['coef']):.4f} vs 25-Channel OLS = {np.linalg.norm(pcr_ols['coef']):.4f}")
print(f"Out-of-sample Test RMSE       : 3-PC PCR = {np.sqrt(np.mean((pred_pcr3 - y_test_true)**2)):.4f} vs 25-Channel OLS = {np.sqrt(np.mean((pred_ols25 - y_test_true)**2)):.4f}")
assert np.linalg.norm(pcr3["coef"]) < 0.70 * np.linalg.norm(pcr_ols["coef"])
assert np.sqrt(np.mean((pred_pcr3 - y_test_true) ** 2)) < np.sqrt(np.mean((pred_ols25 - y_test_true) ** 2))''',
        "post3_md": r'''**What Demonstration 3 shows.** Discarding the 22 noise components avoids the division by small singular values in Eq. (1.3). The norm of the regression vector falls from $1.33$ to $0.50$, and the error on data not used for the fit from $0.085$ to $0.039$.

---

## 4. What if: sensitivity to the parameters
What is the fundamental weakness of PCR? Suppose a high-toxicity **trace catalyst poison** $y_{\mathrm{trace}}$ absorbs along the 4th orthogonal singular vector $\mathbf{v}_4$ (which accounts for $<0.5\%$ of total spectral variance). What happens when we run 3-component PCR versus 1-component supervised PLS1?''',
        "demo4_code": r'''heading("Demonstration 4: The Unsupervised PCR Failure Mode on Low-Variance Trace Analytes")
U_full, _, Vt_full = np.linalg.svd(X_spec - np.mean(X_spec, axis=0), full_matrices=False)
u4 = U_full[:, 3]
v4 = Vt_full[3, :]
X_with_trace = (X_spec - np.mean(X_spec, axis=0)) + np.outer(0.5 * u4, v4)
y_trace = X_with_trace @ v4

pcr_miss = dim_reduction.pcr_regression(X_with_trace, y_trace, n_components=3)
pls_hit = dim_reduction.pls1_nipals(X_with_trace, y_trace, n_components=1)
rmse_pcr_trace = float(np.sqrt(np.mean((pcr_miss["y_pred"] - y_trace) ** 2)))
rmse_pls_trace = float(np.sqrt(np.mean((pls_hit["y_pred"] - y_trace) ** 2)))
print(f"Trace Analyte RMSE: 3-Component Unsupervised PCR = {rmse_pcr_trace:.4f} (Blinded by variance truncation)")
print(f"Trace Analyte RMSE: 1-Component Supervised PLS1  = {rmse_pls_trace:.4e} (Finds trace direction immediately)")
assert rmse_pls_trace < 0.01 * rmse_pcr_trace''',
        "post4_md": r'''**What Demonstration 4 shows.** Principal components are ranked by the variance of the spectra and know nothing of $\mathbf{y}$. Here the analyte of interest lies along a direction of small variance, which the three-component regression discards; its error, $0.16$, is that of predicting the mean. PLS chooses its directions by their covariance with $\mathbf{y}$ and finds this one with a single component. Its error is at rounding level only because this synthetic analyte is an exact linear function of the spectra, with no noise.

---

## Inside the algorithm
Below we compute thin SVD PCA and PCR directly via `np.linalg.svd` in pure NumPy and verify digit-for-digit agreement with `dim_reduction.pca_svd` and `dim_reduction.pcr_regression`.''',
        "inside_code": r'''heading("Inside the Algorithm: By-Hand SVD PCA & PCR vs. the Library")
mu_x = np.mean(X_spec, axis=0)
Xc = X_spec - mu_x
U_s, S_s, Vt_s = np.linalg.svd(Xc, full_matrices=False)
P_hand = Vt_s[:3, :].T
T_hand = Xc @ P_hand
yc = y_target - float(np.mean(y_target))
gamma_hand = np.linalg.lstsq(T_hand, yc, rcond=None)[0]
beta_pcr_hand = P_hand @ gamma_hand
diff_pcr = float(np.max(np.abs(beta_pcr_hand - pcr3["coef"])))
print("by hand (pure NumPy) beta[:5]:", np.round(beta_pcr_hand[:5], 6))
print("library (engml_2_applied)      beta[:5]:", np.round(pcr3["coef"][:5], 6))
print(f"Max discrepancy              : {diff_pcr:.2e}")
assert diff_pcr < 1e-14''',
        "ex1_prompt": r'''Verify that when `n_components = d` ($k = 25$), `pcr_regression(X_spec, y_target, n_components=25)` recovers the exact ordinary least squares coefficient vector $\hat{\boldsymbol{\beta}}_{\mathrm{OLS}}$ and intercept to within $10^{-11}$.''',
        "ex1_code": r'''heading("Exercise 1: Exact Equivalence of Full-Rank PCR (k = d) and OLS")
pcr_full = dim_reduction.pcr_regression(X_spec, y_target, n_components=d)
X_aug = np.column_stack([np.ones(N), X_spec])
w_ols_full = np.linalg.lstsq(X_aug, y_target, rcond=None)[0]
assert np.allclose(pcr_full["coef"], w_ols_full[1:], atol=1e-11)
assert np.isclose(pcr_full["intercept"], w_ols_full[0], atol=1e-11)
print("Max coefficient difference ||beta_PCR(k=d) - beta_OLS||_max:", float(np.max(np.abs(pcr_full["coef"] - w_ols_full[1:]))))''',
        "ex1_exp": r'''When $k = d$, the orthonormal loading matrix $\mathbf{P}_d \in \mathbb{R}^{d \times d}$ spans all of $\mathbb{R}^d$ ($\mathbf{P}_d\mathbf{P}_d^\top = \mathbf{I}_d$), so projecting onto $\mathbf{T}_d$ and mapping back recovers the exact pseudoinverse OLS solution.''',
        "ex2_prompt": r'''Demonstrate the **unit-scaling distortion** of unstandardized PCA: take a 3-sensor reactor table `[T_Kelvin, P_MPa, x_mole_fraction]` where $T \sim 450 \pm 25\;\mathrm{K}$, $P \sim 2.5 \pm 0.4\;\mathrm{MPa}$, and $x \sim 0.30 \pm 0.05$. Show that unscaled PCA assigns $>99.9\%$ of PC1's loading to the temperature because Kelvin has the largest numerical spread, whereas `descriptors.StandardScaler` restores balanced dimensionless loadings.''',
        "ex2_code": r'''heading("Exercise 2: Unstandardized Unit-Dominance Audit in PCA")
X_units = np.column_stack([
    rng.normal(450.0, 25.0, size=60),  # T in Kelvin
    rng.normal(2.5, 0.4, size=60),     # P in MPa
    rng.normal(0.30, 0.05, size=60),   # Mole fraction x
])
pca_unscaled = dim_reduction.pca_svd(X_units, n_components=2)
X_std = descriptors.StandardScaler().fit_transform(X_units)
pca_scaled = dim_reduction.pca_svd(X_std, n_components=2)
print("Unscaled PC1 loading vector [T, P, x]       :", np.round(np.abs(pca_unscaled["loadings"][:, 0]), 5))
print("Unscaled PC1 explained variance ratio       :", f"{pca_unscaled['explained_variance_ratio'][0]*100:.3f}%")
print("Standardized PC1 loading vector [T*, P*, x*]:", np.round(np.abs(pca_scaled["loadings"][:, 0]), 4))
assert abs(pca_unscaled["loadings"][0, 0]) > 0.999 and pca_unscaled["explained_variance_ratio"][0] > 0.999''',
        "ex2_exp": r'''PCA is orthogonal-transformation invariant but *not* scale invariant. Heterogeneous process variables ($T, P, F, x$) must always be standardized to zero mean and unit variance before PCA; homogeneous spectral arrays (all in $\mathrm{AU}$) are kept in common absorbance units.''',
        "ex3_prompt": r'''Implement a function `pca_press_choose_k(X: np.ndarray, y: np.ndarray, k_max: int = 8) -> dict` that selects the optimal PCR component count $k^*$ using 5-fold cross-validation predicted residual error sum of Squares (PRESS), and verify on `(X_spec, y_target)` that it selects $k^* = 3$.''',
        "ex3_code": r'''heading("Exercise 3: Implement 5-Fold Cross-Validation PRESS Component Selection for PCR")
def pca_press_choose_k(X: np.ndarray, y: np.ndarray, k_max: int = 8, n_folds: int = 5, pen: float = 0.01) -> dict:
    """Select optimal PCR latent rank k* minimizing penalized k-fold cross-validated PRESS."""
    N_s = len(y)
    fold_ids = np.arange(N_s) % n_folds
    press_by_k = {}
    for k_c in range(1, k_max + 1):
        se_sum = 0.0
        for f in range(n_folds):
            val_m = fold_ids == f
            tr_m = ~val_m
            fit_f = dim_reduction.pcr_regression(X[tr_m], y[tr_m], n_components=k_c)
            pred_f = X[val_m] @ fit_f["coef"] + fit_f["intercept"]
            se_sum += float(np.sum((y[val_m] - pred_f) ** 2))
        press_by_k[k_c] = se_sum + pen * k_c
    best_k = min(press_by_k, key=press_by_k.get)
    return {"best_k": best_k, "press_by_k": press_by_k}

cv_pcr = pca_press_choose_k(X_spec, y_target, k_max=6)
for k_i, pr_val in cv_pcr["press_by_k"].items():
    print(f"k = {k_i} PCs | 5-Fold CV PRESS = {pr_val:.4f}")
print(f"Optimal CV component count k* = {cv_pcr['best_k']}")
assert cv_pcr["best_k"] == 3''',
        "ex3_takeaway": r'''**Discussion.** Cross-validated PRESS identifies the true chemical latent rank $k^* = 3$ where systematic signal variance ends and unrepeatable instrument noise begins.''',
    },
    {
        "num": r'''08''',
        "slug": r'''08_partial_least_squares_nipals_for_soft_sensors''',
        "title": r'''8. Partial least squares for soft sensors''',
        "hook": r'''Partial least squares (PLS) chooses its components to covary with the quantity to be predicted, which is why it is the usual method for calibrating spectrometers. The lecture follows the NIPALS algorithm step by step, checks its orthogonality relations, and compares it with principal component regression and with the minimum-norm least-squares solution on synthetic spectra with more channels than samples.''',
        "objectives": [
            r'''Follow the NIPALS algorithm of partial least squares for one response (`dim_reduction.pls1_nipals`).''',
            r'''Check that the score vectors are mutually orthogonal and that the regression vector follows from the weights and loadings.''',
            r'''Compare PLS with minimum-norm least squares and with principal component regression on synthetic spectra of 120 channels and 35 samples.''',
            r'''Choose the number of latent variables from the error on data kept out of the fit.''',
        ],
        "theory_md": r'''## 1. Principles and governing equations
Given centered spectral matrix $\mathbf{E}_0 = \mathbf{X} - \mathbf{1}_N\bar{\mathbf{x}}^\top \in \mathbb{R}^{N \times d}$ and centered concentration vector $\mathbf{f}_0 = \mathbf{y} - \bar{y}\mathbf{1}_N \in \mathbb{R}^N$, **Wold's NIPALS PLS1 algorithm (Wold et al., 1984)** extracts $k$ supervised latent variables sequentially ($a = 1, \dots, k$). At stage $a$, the weight vector $\mathbf{w}_a \in \mathbb{R}^d$ maximizes the empirical covariance $\mathbf{w}_a^\top \mathbf{E}_{a-1}^\top \mathbf{f}_{a-1}$ subject to $\|\mathbf{w}_a\|_2 = 1$:
$$\mathbf{w}_a = \frac{\mathbf{E}_{a-1}^\top \mathbf{f}_{a-1}}{\|\mathbf{E}_{a-1}^\top \mathbf{f}_{a-1}\|_2}, \qquad \mathbf{t}_a = \mathbf{E}_{a-1}\mathbf{w}_a \in \mathbb{R}^N \tag{1.1}$$
Next, both $\mathbf{E}_{a-1}$ and $\mathbf{f}_{a-1}$ are regressed onto the score vector $\mathbf{t}_a$ and deflated by rank-1 projections:
$$\mathbf{p}_a = \frac{\mathbf{E}_{a-1}^\top \mathbf{t}_a}{\mathbf{t}_a^\top \mathbf{t}_a}, \qquad q_a = \frac{\mathbf{f}_{a-1}^\top \mathbf{t}_a}{\mathbf{t}_a^\top \mathbf{t}_a}, \qquad \mathbf{E}_a = \mathbf{E}_{a-1} - \mathbf{t}_a\mathbf{p}_a^\top, \qquad \mathbf{f}_a = \mathbf{f}_{a-1} - q_a\mathbf{t}_a \tag{1.2}$$
This deflation guarantees mutually orthogonal latent scores $\mathbf{t}_a^\top \mathbf{t}_b = 0$ ($a \ne b$) and upper-triangular $\mathbf{P}^\top\mathbf{W}$, yielding the exact regression vector in the original $d$-wavelength coordinate frame:
$$\hat{\boldsymbol{\beta}}_{\mathrm{PLS}} = \mathbf{W}\left(\mathbf{P}^\top\mathbf{W}\right)^{-1}\mathbf{q} \in \mathbb{R}^d, \qquad \hat{\beta}_0 = \bar{y} - \bar{\mathbf{x}}^\top\hat{\boldsymbol{\beta}}_{\mathrm{PLS}} \tag{1.3}$$''',
        "demo1_code": r'''heading("Demonstration 1: Wold NIPALS PLS1 on a 120-Wavelength NIR Soft Sensor (d = 120 >> N = 35)")
rng = np.random.default_rng(42)
N, d = 35, 120
wl = np.linspace(1100.0, 2300.0, d)
# 3 overlapping chemical chromophores (water, methanol, ethanol):
E_pure = np.vstack([
    np.exp(-0.5 * ((wl - 1450.0) / 60.0) ** 2) + 0.4 * np.exp(-0.5 * ((wl - 1940.0) / 50.0) ** 2),
    np.exp(-0.5 * ((wl - 1680.0) / 55.0) ** 2) + 0.6 * np.exp(-0.5 * ((wl - 2100.0) / 45.0) ** 2),
    np.exp(-0.5 * ((wl - 1720.0) / 50.0) ** 2) + 0.5 * np.exp(-0.5 * ((wl - 2150.0) / 55.0) ** 2),
])
C_conc = rng.uniform(0.05, 0.95, size=(N, 3))
X_nir = C_conc @ E_pure + rng.normal(0.0, 0.008, size=(N, d))
y_meoh = C_conc[:, 1] + rng.normal(0.0, 0.06, size=N)

pls3 = dim_reduction.pls1_nipals(X_nir, y_meoh, n_components=3)
T_scores = pls3["scores"]
G_scores = T_scores.T @ T_scores
off_diag = G_scores - np.diag(np.diag(G_scores))
print("Score inner-product matrix T^T T:\n", np.round(G_scores, 4))
print("Max off-diagonal score inner product:", float(np.max(np.abs(off_diag))))
assert np.max(np.abs(off_diag)) < 1e-11''',
        "post1_md": r'''**What Demonstration 1 shows.** There are $d = 120$ channels and $N = 35$ calibration samples, so $\mathbf{X}^\top\mathbf{X}$ has at least $85$ zero eigenvalues and cannot be inverted. The NIPALS algorithm does not need the inverse: it extracts three score vectors that are mutually orthogonal ($\mathbf{T}^\top\mathbf{T}$ is diagonal to $10^{-14}$).

---

## 2. Numerical walkthrough
Let us trace the rank-1 deflation loop step by step across stages $a = 1, 2, 3$, tracking the remaining spectral Frobenius energy $\|\mathbf{E}_a\|_F^2$, remaining target variance $\|\mathbf{f}_a\|_2^2$, and the upper-bidiagonal structure of $\mathbf{P}^\top\mathbf{W}$.''',
        "demo2_code": r'''heading("Demonstration 2: Step-by-Step NIPALS Deflation Energy & P^T W Structure")
E_curr = X_nir - np.mean(X_nir, axis=0)
f_curr = y_meoh - float(np.mean(y_meoh))
e0_norm = float(np.sum(E_curr ** 2))
f0_norm = float(np.sum(f_curr ** 2))
W_m, P_m = np.zeros((d, 3)), np.zeros((d, 3))
for a in range(3):
    w_a = E_curr.T @ f_curr
    w_a /= np.linalg.norm(w_a)
    t_a = E_curr @ w_a
    p_a = (E_curr.T @ t_a) / float(t_a @ t_a)
    q_a = float(f_curr @ t_a) / float(t_a @ t_a)
    E_curr = E_curr - np.outer(t_a, p_a)
    f_curr = f_curr - q_a * t_a
    W_m[:, a], P_m[:, a] = w_a, p_a
    print(f"After LV {a+1} | Spectral X variance captured = {100*(1 - np.sum(E_curr**2)/e0_norm):6.2f}% | Target y variance captured = {100*(1 - np.sum(f_curr**2)/f0_norm):6.2f}%")
print("\nUpper-triangular matrix P^T W:\n", np.round(P_m.T @ W_m, 6))
assert np.allclose(np.tril(P_m.T @ W_m, -1), 0.0, atol=1e-12) and np.allclose(np.diag(P_m.T @ W_m), 1.0, atol=1e-12)''',
        "post2_md": r'''**What Demonstration 2 shows.** The matrix $\mathbf{P}^\top\mathbf{W}$ is upper bidiagonal with ones on its diagonal ($\mathbf{p}_a^\top\mathbf{w}_a = 1$, and $(\mathbf{P}^\top\mathbf{W})_{ab} = 0$ for $a > b$), so its inverse in the formula for the regression vector always exists.

---

## 3. Worked case study
Let us compare 3-component PLS1 against unregularized minimum-norm OLS (`np.linalg.lstsq`) on an independent 40-sample validation test set from the ternary water–methanol–ethanol reactor.''',
        "demo3_code": r'''heading("Demonstration 3: Out-of-Sample NIR Soft-Sensor Benchmark (PLS1 vs. Minimum-Norm OLS)")
C_test = rng.uniform(0.05, 0.95, size=(40, 3))
X_test = C_test @ E_pure + rng.normal(0.0, 0.008, size=(40, d))
y_test = C_test[:, 1]

y_pred_pls = X_test @ pls3["coef"] + pls3["intercept"]
X_tr_aug = np.column_stack([np.ones(N), X_nir])
X_te_aug = np.column_stack([np.ones(40), X_test])
w_pinv = np.linalg.lstsq(X_tr_aug, y_meoh, rcond=None)[0]
y_pred_ols = X_te_aug @ w_pinv

rmse_pls = float(np.sqrt(np.mean((y_pred_pls - y_test) ** 2)))
rmse_ols = float(np.sqrt(np.mean((y_pred_ols - y_test) ** 2)))
print(f"Training Set Fit RMSE (N=35)   : Minimum-norm OLS = {np.sqrt(np.mean((X_tr_aug @ w_pinv - y_meoh)**2)):.2e} (Interpolates noise) | PLS1 = {np.sqrt(np.mean((pls3['y_pred'] - y_meoh)**2)):.4f}")
print(f"Out-of-Sample Test RMSE (N=40) : Minimum-norm OLS = {rmse_ols:.4f} -> 3-Component PLS1 = {rmse_pls:.4f}")
assert rmse_pls < 0.08 and rmse_pls < 0.65 * rmse_ols''',
        "post3_md": r'''**What Demonstration 3 shows.** With 120 channels and 35 samples the minimum-norm least-squares solution reproduces the training data exactly, including the noise in the reference concentrations. On 40 new samples its error is $0.050\;\mathrm{mol/L}$, against $0.0145\;\mathrm{mol/L}$ for PLS with three components, which does not fit the training data exactly ($0.063\;\mathrm{mol/L}$).

---

## 4. What if: sensitivity to the parameters
What happens as we sweep the number of PLS latent variables $k \in \{1, 2, 3, 4, 5, 8, 12\}$? Let us track both training RMSE and out-of-sample test RMSE to see the bias–variance tradeoff in action.''',
        "demo4_code": r'''heading("Demonstration 4: Latent Variable Count k Sweep (Underfitting -> Optimal Rank 3 -> Overfitting)")
print(f"{'Latent Vars k':>13s} | {'Train RMSE':>12s} | {'Test RMSE':>12s} | {'||beta_PLS||_2':>14s} | {'Regime':>22s}")
print("-" * 82)
for k_lv in [1, 2, 3, 4, 6, 10]:
    fit_k = dim_reduction.pls1_nipals(X_nir, y_meoh, n_components=k_lv)
    tr_err = float(np.sqrt(np.mean((fit_k["y_pred"] - y_meoh) ** 2)))
    pred_k = X_test @ fit_k["coef"] + fit_k["intercept"]
    te_err = float(np.sqrt(np.mean((pred_k - y_test) ** 2)))
    b_norm = float(np.linalg.norm(fit_k["coef"]))
    regime = "Underfit (< 3 species)" if k_lv < 3 else ("Optimal (k* = 3)" if k_lv == 3 else "Overfitting GC noise")
    print(f"{k_lv:13d} | {tr_err:12.5f} | {te_err:12.5f} | {b_norm:14.4f} | {regime:>22s}")''',
        "post4_md": r'''**What Demonstration 4 shows.** With fewer than three components the model cannot separate the three overlapping absorbers. The test error is smallest at $k = 3$ ($0.0145\;\mathrm{mol/L}$). With more components the training error keeps falling while the test error rises to $0.037$ and then $0.05\;\mathrm{mol/L}$, and the norm of the regression vector grows from $0.55$ to between 4 and 5: the extra components fit the noise.

---

## Inside the algorithm
Below we verify our by-hand NIPALS matrices $\mathbf{W}, \mathbf{P}, \mathbf{q}$ and closed-form inversion $\hat{\boldsymbol{\beta}} = \mathbf{W}(\mathbf{P}^\top\mathbf{W})^{-1}\mathbf{q}$ against `dim_reduction.pls1_nipals`.''',
        "inside_code": r'''heading("Inside the Algorithm: By-Hand W(P^T W)^(-1)q vs. the Library")
q_hand = np.linalg.solve(T_scores.T @ T_scores, T_scores.T @ (y_meoh - np.mean(y_meoh)))
beta_pls_hand = W_m @ np.linalg.solve(P_m.T @ W_m, q_hand)
diff_pls = float(np.max(np.abs(beta_pls_hand - pls3["coef"])))
print("by hand (pure NumPy) beta[:5]:", np.round(beta_pls_hand[:5], 6))
print("library (engml_2_applied)      beta[:5]:", np.round(pls3["coef"][:5], 6))
print(f"Max discrepancy              : {diff_pls:.2e}")
assert diff_pls < 1e-14''',
        "ex1_prompt": r'''Verify on a tall well-conditioned design matrix $\mathbf{X} \in \mathbb{R}^{40 \times 4}$ that `pls1_nipals(X, y, n_components=4)` (when $k = d$) converges to the exact ordinary least squares regression coefficients and intercept to within $10^{-11}$.''',
        "ex1_code": r'''heading("Exercise 1: Exact Equivalence of Full-Rank PLS1 (k = d) and OLS")
X_tall = rng.normal(0.0, 1.0, size=(40, 4))
y_tall = X_tall @ np.array([1.5, -0.8, 2.2, -0.4]) + 3.0 + rng.normal(0.0, 0.1, size=40)
pls_d = dim_reduction.pls1_nipals(X_tall, y_tall, n_components=4)
w_ols_d = np.linalg.lstsq(np.column_stack([np.ones(40), X_tall]), y_tall, rcond=None)[0]

assert np.allclose(pls_d["coef"], w_ols_d[1:], atol=1e-11)
assert np.isclose(pls_d["intercept"], w_ols_d[0], atol=1e-11)
print("Verified exact equivalence of PLS1 (k = d = 4) and OLS:", np.round(pls_d["coef"], 5))''',
        "ex1_exp": r'''PLS1 searches the $k$-dimensional Krylov subspace $\mathcal{K}_k(\mathbf{X}_c^\top\mathbf{X}_c, \mathbf{X}_c^\top\mathbf{y}_c)$. When $k = d$, the Krylov subspace spans the entire column space of $\mathbf{X}_c^\top$, recovering the exact OLS estimator.''',
        "ex2_prompt": r'''Make methanol a minor absorber: replace its pure spectrum (`E_pure[1]`) by a weak band, `0.08 * np.exp(-0.5 * ((wl - 1250.0) / 30.0) ** 2)`, rebuild the spectra, and compare PCR and PLS1 with 1, 2 and 3 components for the methanol concentration. Does PLS1 find the minor component with fewer components than PCR?''',
        "ex2_code": r'''heading("Exercise 2: A minor absorber: PCR and PLS1 with 1, 2 and 3 components")
E_trace_meoh = E_pure.copy()
E_trace_meoh[1] = 0.08 * np.exp(-0.5 * ((wl - 1250.0) / 30.0) ** 2)
X_tr_minor = C_conc @ E_trace_meoh + rng.normal(0.0, 0.001, size=(N, d))
y_tr_minor = C_conc[:, 1]

rmse_minor = {}
for k in (1, 2, 3):
    pcr_m = dim_reduction.pcr_regression(X_tr_minor, y_tr_minor, n_components=k)
    pls_m = dim_reduction.pls1_nipals(X_tr_minor, y_tr_minor, n_components=k)
    rmse_minor[k] = (float(np.sqrt(np.mean((pcr_m["y_pred"] - y_tr_minor) ** 2))),
                     float(np.sqrt(np.mean((pls_m["y_pred"] - y_tr_minor) ** 2))))
    print(f"k = {k} | calibration RMSE: PCR = {rmse_minor[k][0]:.4f} | PLS1 = {rmse_minor[k][1]:.4f}")
print(f"standard deviation of the methanol concentration = {np.std(y_tr_minor):.4f}")
assert all(rmse_minor[k][1] <= rmse_minor[k][0] + 1e-12 for k in (1, 2, 3))
assert rmse_minor[3][0] < 0.02 and rmse_minor[2][1] > 0.1''',
        "ex2_exp": r'''The answer here is no. With one or two components both methods do little better than predicting the mean: PLS1 is slightly ahead with two components ($0.236$ against $0.255$), and neither is useful. With three components, one for each absorbing species, both reach $0.004$. PLS1 chooses directions by their covariance with $\mathbf{y}$, and covariance is correlation times the standard deviations: a direction of large spectral variance and modest correlation can still come before a weak band that is perfectly correlated. PLS1 never does worse than PCR with the same number of components, but the number of components must still be at least the number of independent sources of variation in the spectra, and should be chosen by cross-validation.''',
        "ex3_prompt": r'''Implement **variable importance in projection (VIP)** scores `pls_vip_scores(X: np.ndarray, y: np.ndarray, n_components: int = 3) -> np.ndarray`, defined by $\mathrm{VIP}_j = \sqrt{d \sum_{a=1}^k (q_a^2 \mathbf{t}_a^\top\mathbf{t}_a)\,w_{j,a}^2 / \sum_{a=1}^k q_a^2 \mathbf{t}_a^\top\mathbf{t}_a}$, which satisfies the exact sum-of-squares identity $\frac{1}{d}\sum_{j=1}^d \mathrm{VIP}_j^2 = 1.0$. Verify on `(X_nir, y_meoh)` that $\frac{1}{d}\sum_j \mathrm{VIP}_j^2 = 1.0$ and that the peak VIP wavelength matches the true methanol absorption band ($1680\;\mathrm{nm}$ or $2100\;\mathrm{nm}$).''',
        "ex3_code": r'''heading("Exercise 3: Implement PLS Variable Importance in Projection (VIP) From Scratch")
def pls_vip_scores(X: np.ndarray, y: np.ndarray, n_components: int = 3) -> np.ndarray:
    """Compute Wold's Variable Importance in Projection (VIP) wavelength scores for PLS1."""
    E_c = np.asarray(X, dtype=float) - np.mean(X, axis=0)
    f_c = np.asarray(y, dtype=float).ravel() - float(np.mean(y))
    n_s, d_s = E_c.shape
    W_vip = np.zeros((d_s, n_components))
    ss_y = np.zeros(n_components)
    for a in range(n_components):
        w_a = E_c.T @ f_c
        w_a /= np.linalg.norm(w_a)
        t_a = E_c @ w_a
        t_sq = float(t_a @ t_a)
        p_a = (E_c.T @ t_a) / t_sq
        q_a = float(f_c @ t_a) / t_sq
        E_c = E_c - np.outer(t_a, p_a)
        f_c = f_c - q_a * t_a
        W_vip[:, a] = w_a
        ss_y[a] = (q_a ** 2) * t_sq
    vip_sq = d_s * ((W_vip ** 2) @ ss_y) / max(float(np.sum(ss_y)), 1e-15)
    return np.sqrt(vip_sq)

vip = pls_vip_scores(X_nir, y_meoh, n_components=3)
peak_wl = float(wl[int(np.argmax(vip))])
print(f"Mean squared VIP score (1/d) sum(VIP_j^2) = {np.mean(vip ** 2):.10f} (Exact identity = 1.0)")
print(f"Top VIP wavelength = {peak_wl:.1f} nm (True methanol chromophore peaks = 1680 nm & 2100 nm)")
assert np.isclose(np.mean(vip ** 2), 1.0, rtol=1e-13) and (abs(peak_wl - 1680.0) < 35.0 or abs(peak_wl - 2100.0) < 35.0)''',
        "ex3_takeaway": r'''**Discussion.** Wold's VIP scores ($\mathrm{VIP}_j > 1$) identify which physical wavelengths carry the strongest target covariance across all PLS latent variables while satisfying $\frac{1}{d}\sum_{j=1}^d \mathrm{VIP}_j^2 \equiv 1$.''',
    },
    {
        "num": r'''09''',
        "slug": r'''09_multivariate_process_monitoring_hotelling_t2_and_q''',
        "title": r'''9. Process monitoring with Hotelling's T² and the Q residual''',
        "hook": r'''The sensors of a plant are correlated, because a few operating variables drive all of them. Charts for single sensors miss a fault that breaks the correlation while every reading stays in its own range. A principal component model of normal operation gives two statistics: Hotelling's $T^2$, for unusual movement within the normal pattern, and the residual $Q$, for departures from it. The contributions to $Q$ point to the sensor at fault.''',
        "objectives": [
            r'''Split a measurement into its part in the principal component model and its residual (`dim_reduction.pca_svd`).''',
            r'''Check the identity for the mean of Hotelling's $T^2$ over the training data.''',
            r'''Distinguish a large movement within the normal pattern (high $T^2$) from a broken correlation (high $Q$).''',
            r'''Find the faulty sensor from the contributions of the sensors to $Q$.''',
        ],
        "theory_md": r'''## 1. Principles and governing equations
In **multivariate statistical process control (MSPC, Kresta, MacGregor & Marlin 1991)**, a $k$-component PCA model calibrated on $N$ normal operating condition (NOC) observations decomposes any centered plant measurement $\mathbf{x}_c = \mathbf{x} - \boldsymbol{\mu} \in \mathbb{R}^d$ into two mutually orthogonal components:
$$\mathbf{x}_c = \hat{\mathbf{x}}_c + \mathbf{e}, \qquad \hat{\mathbf{x}}_c = \mathbf{P}_k\mathbf{t} = \mathbf{P}_k\mathbf{P}_k^\top \mathbf{x}_c \in \mathcal{S}_{\mathrm{model}}, \qquad \mathbf{e} = (\mathbf{I}_d - \mathbf{P}_k\mathbf{P}_k^\top)\mathbf{x}_c \in \mathcal{S}_{\mathrm{res}}^\perp \tag{1.1}$$
Because $\mathbf{P}_k^\top \mathbf{e} = \mathbf{0}$, $\|\mathbf{x}_c\|_2^2 = \|\hat{\mathbf{x}}_c\|_2^2 + \|\mathbf{e}\|_2^2$. Two complementary fault charts monitor the two subspaces:
1. **Hotelling's $T^2$ Statistic (in-subspace Mahalanobis leverage):**
   $$T_i^2 = \sum_{a=1}^k \frac{t_{i,a}^2}{\lambda_a} = \mathbf{x}_{c,i}^\top \mathbf{P}_k \boldsymbol{\Lambda}_k^{-1}\mathbf{P}_k^\top \mathbf{x}_{c,i}, \qquad \frac{1}{N}\sum_{i=1}^N T_i^2 = \frac{k(N-1)}{N} \tag{1.2}$$
   which measures the directional distance *within* the normal physical correlation manifold $\mathcal{S}_{\mathrm{model}}$.
2. **$Q$-Statistic / squared prediction error ($\mathrm{SPE}$, Out-of-subspace residual):**
   $$Q_i = \|\mathbf{x}_{c,i} - \hat{\mathbf{x}}_{c,i}\|_2^2 = \|\mathbf{e}_i\|_2^2 = \sum_{j=1}^d e_{i,j}^2 \tag{1.3}$$
   which measures how far the observation has departed *perpendicular* to the physical conservation/correlation manifold, with $c_{i,j} = e_{i,j}^2$ providing the $j$-th sensor's individual fault contribution.''',
        "demo1_code": r'''heading("Demonstration 1: Orthogonal Subspace Decomposition & Exact Hotelling T^2 Mean Identity")
rng = np.random.default_rng(42)
N, d, k = 80, 8, 2
# 2 latent process drivers (throughput u1, jacket temperature u2) governing 8 plant sensors:
U_noc = rng.normal(0.0, 1.0, size=(N, k))
A_mix = rng.normal(0.0, 1.0, size=(k, d))
X_noc = U_noc @ A_mix + rng.normal(0.0, 0.06, size=(N, d))

pca_mspc = dim_reduction.pca_svd(X_noc, n_components=k)
t2_noc = pca_mspc["hotelling_t2"]
q_noc = pca_mspc["q_spe"]

expected_t2_mean = k * (N - 1.0) / N
print(f"Empirical mean(T^2) across N={N} NOC samples : {np.mean(t2_noc):.8f}")
print(f"Exact theoretical identity k*(N-1)/N         : {expected_t2_mean:.8f}")
print(f"NOC 99th-percentile control limits           : T^2_lim = {np.percentile(t2_noc, 99):.3f}, Q_lim = {np.percentile(q_noc, 99):.5f}")
assert np.isclose(np.mean(t2_noc), expected_t2_mean, rtol=1e-13)''',
        "post1_md": r'''**What Demonstration 1 shows.** Because each squared score column sums to $\sum_{i=1}^N t_{i,a}^2 = (N-1)\lambda_a$, dividing by $\lambda_a$ and summing over $a=1\dots k$ forces the sample average of Hotelling's $T^2$ to equal $k(N-1)/N = 2 \times 79 / 80 = 1.97500000$ identically.

---

## 2. Numerical walkthrough
Let us verify that the model projection matrix $\boldsymbol{\Pi}_{\mathrm{model}} = \mathbf{P}_k\mathbf{P}_k^\top$ and the residual projection matrix $\boldsymbol{\Pi}_{\mathrm{res}} = \mathbf{I}_d - \mathbf{P}_k\mathbf{P}_k^\top$ are symmetric, idempotent ($\boldsymbol{\Pi}^2 = \boldsymbol{\Pi}$), and mutually annihilating ($\boldsymbol{\Pi}_{\mathrm{model}}\boldsymbol{\Pi}_{\mathrm{res}} = \mathbf{0}$).''',
        "demo2_code": r'''heading("Demonstration 2: Idempotence & Mutual Annihilation of Model and Residual Projectors")
P = pca_mspc["loadings"]
mu = pca_mspc["mean"]
lam = pca_mspc["eigenvalues"]
Pi_model = P @ P.T
Pi_res = np.eye(d) - Pi_model

Xc_noc = X_noc - mu
X_hat_noc = Xc_noc @ Pi_model
E_res_noc = Xc_noc @ Pi_res
pythagoras_err = float(np.max(np.abs(np.sum(Xc_noc ** 2, axis=1) - (np.sum(X_hat_noc ** 2, axis=1) + q_noc))))

print(f"trace(Pi_model) = {np.trace(Pi_model):.1f} (dim = k = {k}) | trace(Pi_res) = {np.trace(Pi_res):.1f} (dim = d - k = {d-k})")
print(f"Projector orthogonality ||Pi_model @ Pi_res||_max : {np.max(np.abs(Pi_model @ Pi_res)):.2e}")
print(f"Pythagorean energy decomposition max residual     : {pythagoras_err:.2e}")
assert np.allclose(Pi_model @ Pi_res, 0.0, atol=1e-14) and pythagoras_err < 1e-13''',
        "post2_md": r'''**What Demonstration 2 shows.** Each measurement of the eight sensors is the sum of a part in the plane of the $k = 2$ principal components and a residual in the $d - k = 6$ directions orthogonal to it, and the two projectors that produce them are idempotent and annihilate each other.

---

## 3. Worked case study
Let us now test two classic chemical plant events against the 99% NOC thresholds ($T^2_{\mathrm{lim}}, Q_{\mathrm{lim}}$):
- **Event A (6-sigma feed throughput Surge):** An operator increases feed rate along latent variable $u_1 = 6.0$ while all 8 sensors obey their normal physical correlations.
- **Event B (Sensor #4 thermocouple Drift):** The reactor stays at its normal operating point, but Sensor #4 drifts by $+3.5$ units.''',
        "demo3_code": r'''heading("Demonstration 3: Diagnosing Throughput Surges (High T^2) vs. Sensor Drift (High Q_SPE)")
t2_limit = float(np.percentile(t2_noc, 99))
q_limit = float(np.percentile(q_noc, 99))

# Event A: 6-sigma throughput surge along latent variable 1 (obeys physical sensor correlations):
x_surge = mu + (np.array([6.0, 0.0]) @ A_mix)
xc_surge = x_surge - mu
t_surge = xc_surge @ P
t2_surge = float(np.sum((t_surge ** 2) / lam))
q_surge = float(np.sum((xc_surge - t_surge @ P.T) ** 2))

# Event B: Single sensor #4 drifts by +3.5 units (breaks physical correlation across the 8 sensors):
x_drift = mu.copy()
x_drift[3] += 3.5
xc_drift = x_drift - mu
t_drift = xc_drift @ P
t2_drift = float(np.sum((t_drift ** 2) / lam))
q_drift = float(np.sum((xc_drift - t_drift @ P.T) ** 2))

print(f"Normal 99% Limits : T^2_lim = {t2_limit:6.2f}         | Q_lim = {q_limit:.4f}")
print(f"Throughput Surge  : T^2     = {t2_surge:6.2f} (ALARM) | Q     = {q_surge:.4f} (Normal - correlations intact)")
print(f"Sensor #4 Drift   : T^2     = {t2_drift:6.2f}         | Q     = {q_drift:.4f} (Q ALARM: {q_drift/q_limit:.1f}x limit)")
assert t2_surge > 3.0 * t2_limit and q_surge < q_limit
assert q_drift > 20.0 * q_limit''',
        "post3_md": r'''**What Demonstration 3 shows.** The throughput surge raises only $T^2$ ($49.3$ against a limit of $8.8$); its $Q$ stays below the limit ($0.017 < 0.060$), which says that the plant has moved to an unusual operating point with the relations between its sensors intact. The drift of sensor 4 does the opposite: $T^2 = 0.25$ is unremarkable, and $Q$ is $185$ times its limit, which says that a relation between the sensors is broken.

---

## 4. What if: sensitivity to the parameters
Once a $Q_{\mathrm{SPE}}$ alarm fires, how does the control room engineer isolate *which* of the 8 sensors failed, and how does $Q_{\mathrm{SPE}}$ scale with drift magnitude $\Delta x_4 \in [0.0, 4.0]$?''',
        "demo4_code": r'''heading("Demonstration 4: Q-Residual Sensor Contribution Plot & Drift Sensitivity Sweep")
e_drift = xc_drift - t_drift @ P.T
q_contributions = e_drift ** 2
culprit_sensor = int(np.argmax(q_contributions))
print("Q-residual sensor contributions e_j^2:", np.round(q_contributions, 3))
print(f"Isolated faulty sensor index: {culprit_sensor} (Sensor #{culprit_sensor + 1} accounts for {100*q_contributions[culprit_sensor]/q_drift:.1f}% of Q)")
assert culprit_sensor == 3

print(f"\n{'Drift dX_4':>10s} | {'Hotelling T^2':>13s} | {'Q_SPE Residual':>14s} | {'Q / Q_limit':>12s} | {'Top Contributor':>15s}")
print("-" * 74)
for d_val in [0.0, 0.25, 0.5, 1.5, 3.5]:
    x_sw = mu.copy()
    x_sw[3] += d_val
    xc_s = x_sw - mu
    ts = xc_s @ P
    es = xc_s - ts @ P.T
    qs = float(np.sum(es ** 2))
    t2s = float(np.sum((ts ** 2) / lam))
    top_s = f"Sensor #{int(np.argmax(es**2))+1}" if d_val > 0 else "None"
    print(f"{d_val:10.2f} | {t2s:13.3f} | {qs:14.4f} | {qs/q_limit:12.1f}x | {top_s:>15s}")''',
        "post4_md": r'''**What Demonstration 4 shows.** The contributions to $Q$ point to sensor 4, which accounts for $90.5\ \%$ of it. $Q$ grows as the square of the drift: a drift of $0.25$ stays just below the limit ($0.9$ times), $0.5$ is $3.8$ times above it, and $3.5$ is $185$ times above it. The smallest drift that can be detected is set by the noise of the normal data through the limit on $Q$.

---

## Inside the algorithm
Below we compute Hotelling's $T^2$ and $Q_{\mathrm{SPE}}$ directly from the SVD matrices in pure NumPy and confirm digit-for-digit agreement with `dim_reduction.pca_svd`.''',
        "inside_code": r'''heading("Inside the Algorithm: By-Hand Hotelling T^2 & Q_SPE vs. the Library")
T_hand = Xc_noc @ P
t2_hand = np.sum((T_hand ** 2) / lam[None, :], axis=1)
q_hand = np.sum((Xc_noc - T_hand @ P.T) ** 2, axis=1)
diff_t2 = float(np.max(np.abs(t2_hand - t2_noc)))
diff_q = float(np.max(np.abs(q_hand - q_noc)))
print("by hand (pure NumPy) T^2[:4]:", np.round(t2_hand[:4], 6))
print("library (engml_2_applied)      T^2[:4]:", np.round(t2_noc[:4], 6))
print(f"Max discrepancy T^2 = {diff_t2:.2e}, Q_SPE = {diff_q:.2e}")
assert diff_t2 < 1e-14 and diff_q < 1e-14''',
        "ex1_prompt": r'''Inject a correlation-breaking fault into Sensor #7 (index `6`, $+4.0$ bias) of the 8-sensor reactor model (`pca_mspc`), compute the residual vector $\mathbf{e} = (\mathbf{I}_8 - \mathbf{P}_2\mathbf{P}_2^\top)(\mathbf{x} - \boldsymbol{\mu})$, and verify that (1) $\mathbf{P}_2^\top \mathbf{e} = \mathbf{0}$ to machine precision ($<10^{-14}$), (2) $Q > 30 Q_{\mathrm{limit}}$, and (3) `np.argmax(e**2) == 6`.''',
        "ex1_code": r'''heading("Exercise 1: Sensor #7 Fault Isolation & Subspace Orthogonality Check")
x_f7 = mu.copy()
x_f7[6] += 4.0
xc_f7 = x_f7 - mu
e_f7 = xc_f7 - (xc_f7 @ P) @ P.T
q_f7 = float(np.sum(e_f7 ** 2))

assert np.max(np.abs(P.T @ e_f7)) < 1e-14
assert q_f7 > 30.0 * q_limit and int(np.argmax(e_f7 ** 2)) == 6
print(f"Sensor #7 fault Q = {q_f7:.3f} | Orthogonality ||P^T e||_max = {np.max(np.abs(P.T @ e_f7)):.2e} | Isolated index = {int(np.argmax(e_f7**2))}")''',
        "ex1_exp": r'''The residual vector $\mathbf{e} = (\mathbf{I}_8 - \mathbf{P}_2\mathbf{P}_2^\top)\mathbf{x}_c$ is strictly orthogonal to both principal loading vectors, and its squared components $e_j^2$ immediately pinpoint Sensor #7 (index `6`).''',
        "ex2_prompt": r'''Demonstrate what happens if an engineer chooses `n_components = d = 8` (retaining *all* principal components in the MSPC model): show that $\mathbf{I}_8 - \mathbf{P}_8\mathbf{P}_8^\top = \mathbf{0}$, causing $Q_{\mathrm{SPE}} \equiv 0$ identically even for a sensor failure of $+100$ units.''',
        "ex2_code": r'''heading("Exercise 2: Full-Rank PCA Over-Parameterization Failure Audit (Q_SPE == 0)")
pca_full8 = dim_reduction.pca_svd(X_noc, n_components=d)
P8 = pca_full8["loadings"]
e_blind = xc_f7 - (xc_f7 @ P8) @ P8.T
q_blind = float(np.sum(e_blind ** 2))
print(f"When k = d = 8, ||I - P_8 P_8^T||_max = {np.max(np.abs(np.eye(d) - P8 @ P8.T)):.2e} | Faulty Q_SPE = {q_blind:.2e} (Completely blind)")
assert q_blind < 1e-24''',
        "ex2_exp": r'''When $k = d$, the residual subspace $\mathcal{S}_{\mathrm{res}}^\perp$ has dimension $d - d = 0$, so $Q_{\mathrm{SPE}}$ vanishes identically for every vector in $\mathbb{R}^d$. MSPC *requires* $k < d$ so that a non-trivial residual subspace monitors conservation laws.''',
        "ex3_prompt": r'''Implement a **fault sensor reconstruction** function `reconstruct_faulty_sensor(x_faulty: np.ndarray, fault_idx: int, mean: np.ndarray, loadings: np.ndarray) -> np.ndarray` that replaces the corrupted $j$-th sensor reading with the value $\hat{x}_j$ that minimizes $Q_{\mathrm{SPE}} = \|(\mathbf{I}_d - \mathbf{P}_k\mathbf{P}_k^\top)(\mathbf{x} - \boldsymbol{\mu})\|_2^2$, given in closed form by $\hat{x}_{\mathrm{corr},j} = x_j - e_j / (\mathbf{I}_d - \mathbf{P}_k\mathbf{P}_k^\top)_{jj}$. Verify on `x_drift` (`fault_idx=3`) that it recovers the true normal reading $\mu_3$ and drops $Q_{\mathrm{SPE}}$ back to $0$.''',
        "ex3_code": r'''heading("Exercise 3: Implement Subspace-Projected Faulty Sensor Reconstruction")
def reconstruct_faulty_sensor(x_faulty: np.ndarray, fault_idx: int, mean: np.ndarray, loadings: np.ndarray) -> dict:
    """Reconstruct a broken sensor reading by projecting along coordinate j onto the PCA model subspace."""
    x_corr = np.asarray(x_faulty, dtype=float).copy()
    Pi_r = np.eye(len(mean)) - loadings @ loadings.T
    e_vec = Pi_r @ (x_corr - mean)
    x_corr[fault_idx] -= e_vec[fault_idx] / max(Pi_r[fault_idx, fault_idx], 1e-15)
    e_after = Pi_r @ (x_corr - mean)
    return {"x_reconstructed": x_corr, "q_before": float(np.sum(e_vec ** 2)), "q_after": float(np.sum(e_after ** 2))}

rec_s4 = reconstruct_faulty_sensor(x_drift, fault_idx=3, mean=mu, loadings=P)
print(f"Sensor #4 true normal value = {mu[3]:.4f} | Drifted value = {x_drift[3]:.4f} | Subspace reconstructed = {rec_s4['x_reconstructed'][3]:.4f}")
print(f"Q_SPE before reconstruction = {rec_s4['q_before']:.4f} -> Q_SPE after = {rec_s4['q_after']:.2e}")
assert np.isclose(rec_s4["x_reconstructed"][3], mu[3], atol=1e-12) and rec_s4["q_after"] < 1e-24''',
        "ex3_takeaway": r'''**Discussion.** As long as the diagonal residual leverage $(\mathbf{I}_d - \mathbf{P}_k\mathbf{P}_k^\top)_{jj} > 0$, a single broken sensor can be both isolated via $e_j^2$ and analytically reconstructed from the remaining $d-1$ correlated sensors.''',
    },
    {
        "num": r'''10''',
        "slug": r'''10_unsupervised_phase_clustering_kmeans_and_voronoi''',
        "title": r'''10. Clustering with K-means, and where it fails''',
        "hook": r'''Clustering groups points without labels. K-means does it by alternating between assigning each point to the nearest centre and moving each centre to the mean of its points. The lecture checks the sum-of-squares identity behind the method, chooses the number of clusters with the silhouette, and ends with two concentric rings, which the method cannot separate because its boundaries are straight.''',
        "objectives": [
            r'''Implement K-means with k-means++ seeding (`clustering_phases.kmeans_fit`).''',
            r'''Check that the total sum of squares is the sum of the parts within and between clusters.''',
            r'''Choose the number of clusters with the mean silhouette (`clustering_phases.silhouette_score_mean`).''',
            r'''Show why K-means fails on two concentric rings.''',
        ],
        "theory_md": r'''## 1. Principles and governing equations
Given $N$ unlabeled thermodynamic state vectors $\{\mathbf{x}_i\}_{i=1}^N \subset \mathbb{R}^d$, **$K$-means clustering** partitions the observations into $K$ disjoint phase clusters $\{C_1, \dots, C_K\}$ with centroids $\boldsymbol{\mu}_k = \frac{1}{|C_k|}\sum_{\mathbf{x}_i \in C_k}\mathbf{x}_i$ to minimize within-cluster inertia $J_{\mathrm{within}}$:
$$J_{\mathrm{within}}(\{C_k, \boldsymbol{\mu}_k\}) = \sum_{k=1}^K \sum_{\mathbf{x}_i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|_2^2 \tag{1.1}$$
By the **Huygens–Steiner / ANOVA variance decomposition**, the total sum of squares $\mathrm{SST} = \sum_{i=1}^N \|\mathbf{x}_i - \bar{\mathbf{x}}\|_2^2$ splits into within-cluster inertia plus between-cluster inertia:
$$\mathrm{SST} = J_{\mathrm{within}} + J_{\mathrm{between}}, \qquad J_{\mathrm{between}} = \sum_{k=1}^K |C_k|\,\|\boldsymbol{\mu}_k - \bar{\mathbf{x}}\|_2^2 \tag{1.2}$$
- **$K$-Means++ Seeding (Arthur & Vassilvitskii, 2007):** Draws each new initial centroid with probability $p_i \propto D(\mathbf{x}_i)^2 = \min_{1 \le j < r}\|\mathbf{x}_i - \boldsymbol{\mu}_j\|_2^2$, guaranteeing $O(\log K)$-competitive inertia.
- **Rousseeuw's silhouette coefficient (1987):** Compares sample $i$'s mean intra-cluster distance $a_i$ to its nearest-neighbor cluster distance $b_i$:
  $$s_i = \frac{b_i - a_i}{\max(a_i, b_i)} \in [-1, +1], \qquad K^* = \arg\max_{K \ge 2} \frac{1}{N}\sum_{i=1}^N s_i(K) \tag{1.3}$$''',
        "demo1_code": r'''heading("Demonstration 1: K-Means++ Phase Discovery & Huygens-Steiner ANOVA Identity")
rng = np.random.default_rng(42)
# Standardized (T_r, P_r, Z) states for Subcooled Liquid, Superheated Vapor, and Supercritical Fluid:
X_liq = rng.normal(loc=[-1.5, -0.8, -1.6], scale=0.22, size=(25, 3))
X_vap = rng.normal(loc=[0.2, -1.1, 1.2], scale=0.22, size=(25, 3))
X_scf = rng.normal(loc=[1.6, 1.5, 0.5], scale=0.25, size=(25, 3))
X_phases = np.vstack([X_liq, X_vap, X_scf])

km3 = clustering_phases.kmeans_fit(X_phases, k=3, seed=42)
labels3 = km3["labels"]
assert len(np.unique(labels3[:25])) == 1 and len(np.unique(labels3[25:50])) == 1 and len(np.unique(labels3[50:])) == 1
assert len({labels3[0], labels3[25], labels3[50]}) == 3

x_grand = np.mean(X_phases, axis=0)
sst = float(np.sum((X_phases - x_grand) ** 2))
j_within = km3["inertia"]
j_between = sum(float(np.sum(labels3 == k_i)) * float(np.sum((km3["centers"][k_i] - x_grand) ** 2)) for k_i in range(3))
print(f"K-Means++ (K=3) converged with 100% phase purity | Within-cluster inertia J_within = {j_within:.4f}")
print(f"Huygens ANOVA check: SST = {sst:.6f} == J_within + J_between = {j_within + j_between:.6f}")
assert np.isclose(sst, j_within + j_between, rtol=1e-13)''',
        "post1_md": r'''**What Demonstration 1 shows.** The algorithm assigns all 75 synthetic points $(T_r, P_r, Z)$ to the group they were generated from (liquid, vapour or supercritical fluid). The total sum of squares splits exactly into a part within the clusters and a part between them, and the three centroids account for $97.3\ \%$ of it.

---

## 2. Numerical walkthrough
Let us trace the **$K$-Means++ $D^2$ probability distribution** during centroid initialization: after picking the first centroid in the Liquid phase, notice how the squared-distance weights $p_i = D(\mathbf{x}_i)^2 / \sum_j D(\mathbf{x}_j)^2$ overwhelmingly favor picking the next centroids in the distant Vapor and Supercritical phases.''',
        "demo2_code": r'''heading("Demonstration 2: Step-by-Step K-Means++ D^2 Seeding Probability Inspection")
c0 = X_phases[0]  # First seed inside Subcooled Liquid cluster (index 0..24)
d2_step1 = np.sum((X_phases - c0) ** 2, axis=1)
p_step1 = d2_step1 / np.sum(d2_step1)
print(f"After Seed 1 in Liquid: Total D^2 probability in Liquid={np.sum(p_step1[:25]):.4f}, Vapor={np.sum(p_step1[25:50]):.4f}, SCF={np.sum(p_step1[50:]):.4f}")

c1_seed = X_phases[60]  # Second seed chosen in Supercritical Fluid (index 50..74)
d2_step2 = np.minimum(d2_step1, np.sum((X_phases - c1_seed) ** 2, axis=1))
p_step2 = d2_step2 / np.sum(d2_step2)
print(f"After Seed 2 in SCF   : Total D^2 probability in Liquid={np.sum(p_step2[:25]):.4f}, Vapor={np.sum(p_step2[25:50]):.4f}, SCF={np.sum(p_step2[50:]):.4f}")
assert np.sum(p_step2[25:50]) > 0.90''',
        "post2_md": r'''**What Demonstration 2 shows.** After the first seed is placed in the liquid group, that group keeps only $0.5\ \%$ of the probability of receiving the second. Once the liquid and the supercritical groups each have a seed, $95.5\ \%$ of the probability lies on the vapour group, which has none. Seeding in proportion to the squared distance from the nearest seed makes it unlikely that two seeds start in one cluster.

---

## 3. Worked case study
In practice we do not know $K$ in advance. Let us sweep candidate phase counts $K \in \{2, 3, 4, 5\}$ on `X_phases` and evaluate both within-cluster inertia $J_{\mathrm{within}}(K)$ and Rousseeuw's mean Silhouette coefficient $\bar{s}(K)$ (`clustering_phases.silhouette_score_mean`).''',
        "demo3_code": r'''heading("Demonstration 3: Automated Phase Count Selection via Silhouette Score & Inertia Elbow")
inertias, silhouettes = [], []
k_candidates = [2, 3, 4, 5]
for k_val in k_candidates:
    res_k = clustering_phases.kmeans_fit(X_phases, k=k_val, seed=42)
    s_k = clustering_phases.silhouette_score_mean(X_phases, res_k["labels"])
    inertias.append(res_k["inertia"])
    silhouettes.append(s_k)
    print(f"K = {k_val} | Within-Cluster Inertia J = {res_k['inertia']:7.3f} | Mean Silhouette s(K) = {s_k:.4f}")

best_k = k_candidates[int(np.argmax(silhouettes))]
print(f"Silhouette-selected optimal thermodynamic phase count: K* = {best_k}")
assert best_k == 3 and silhouettes[1] > 0.70
assert all(inertias[i] > inertias[i + 1] for i in range(len(inertias) - 1))''',
        "post3_md": r'''**What Demonstration 3 shows.** The within-cluster sum of squares falls with every added cluster and cannot choose $K$ by itself, although its fall from $126$ to $9.2$ between $K = 2$ and $3$, and the small gains after, are a strong hint. The mean silhouette has a maximum at $K = 3$ ($0.85$, against $0.64$ at $K = 2$ and $4$).

---

## 4. What if: sensitivity to the parameters
Why does $K$-Means fail on **concentric core–shell nanoparticles** or annular pipe-flow regimes? Because assigning each point to its nearest Euclidean centroid $\|\mathbf{x} - \boldsymbol{\mu}_1\|_2^2 \le \|\mathbf{x} - \boldsymbol{\mu}_2\|_2^2 \iff (\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1)^\top \mathbf{x} \le \frac{1}{2}(\|\boldsymbol{\mu}_2\|_2^2 - \|\boldsymbol{\mu}_1\|_2^2)$ creates a **linear hyperplane Voronoi boundary**.''',
        "demo4_code": r'''heading("Demonstration 4: Voronoi Hyperplane Failure on Concentric Core-Shell Morphology")
theta = np.linspace(0.0, 2.0 * np.pi, 30, endpoint=False)
X_rings = np.vstack([
    np.column_stack([0.5 * np.cos(theta), 0.5 * np.sin(theta)]),  # Inner core phase (r = 0.5)
    np.column_stack([2.2 * np.cos(theta), 2.2 * np.sin(theta)]),  # Outer shell phase (r = 2.2)
])
km_rings = clustering_phases.kmeans_fit(X_rings, k=2, seed=42)
shell_labels = km_rings["labels"][30:]
shell_purity = float(max(np.mean(shell_labels == 0), np.mean(shell_labels == 1)))
print(f"K-Means outer-shell purity on concentric core-shell morphology = {shell_purity*100:.1f}% (50% = complete hyperplane cut)")
assert shell_purity < 0.75''',
        "post4_md": r'''**What Demonstration 4 shows.** For two concentric rings the method fails: the purity of the outer ring is $56.7\ \%$, hardly better than the $50\ \%$ of a coin. The boundary between two $K$-means clusters is a plane halfway between their centroids, and no plane separates a ring from a core inside it. Lecture 11 treats this case with a method based on neighbourhoods instead of centroids.

---

## Inside the algorithm
Below we verify one complete Lloyd expectation–maximization step (nearest-centroid assignment + cluster-mean update) in pure NumPy against `clustering_phases.kmeans_fit`.''',
        "inside_code": r'''heading("Inside the Algorithm: By-Hand Voronoi Assignment & Centroid Update vs. the Library")
centers_lib = km3["centers"]
dists_sq = np.sum((X_phases[:, None, :] - centers_lib[None, :, :]) ** 2, axis=-1)
labels_hand = np.argmin(dists_sq, axis=1)
centers_hand = np.vstack([np.mean(X_phases[labels_hand == j], axis=0) for j in range(3)])
inertia_hand = float(np.sum((X_phases - centers_hand[labels_hand]) ** 2))
print("by hand (pure NumPy) inertia:", round(inertia_hand, 8))
print("library (engml_2_applied)      inertia:", round(km3["inertia"], 8))
assert np.array_equal(labels_hand, km3["labels"]) and np.allclose(centers_hand, centers_lib, atol=1e-14)
assert np.isclose(inertia_hand, km3["inertia"], rtol=1e-14)''',
        "ex1_prompt": r'''Verify the exact **Huygens–Steiner / ANOVA total sum of squares decomposition** $\mathrm{SST} = J_{\mathrm{within}} + J_{\mathrm{between}}$ across all four candidate cluster counts $K \in \{2, 3, 4, 5\}$ on `X_phases`, confirming that $J_{\mathrm{within}}(K) + J_{\mathrm{between}}(K) \equiv \mathrm{SST}$ to $<10^{-13}$ regardless of $K$.''',
        "ex1_code": r'''heading("Exercise 1: Universal Huygens-Steiner ANOVA Identity Across K = 2..5")
for k_v in [2, 3, 4, 5]:
    r_k = clustering_phases.kmeans_fit(X_phases, k=k_v, seed=42)
    jw = r_k["inertia"]
    jb = sum(float(np.sum(r_k["labels"] == j)) * float(np.sum((r_k["centers"][j] - x_grand) ** 2)) for j in range(k_v))
    print(f"K = {k_v} | J_within = {jw:8.4f} | J_between = {jb:8.4f} | Sum = {jw+jb:.6f} (SST = {sst:.6f})")
    assert np.isclose(jw + jb, sst, rtol=1e-13)''',
        "ex1_exp": r'''Because $\sum_{\mathbf{x}_i \in C_k}(\mathbf{x}_i - \boldsymbol{\mu}_k) = \mathbf{0}$ for every cluster centroid, the cross term $2\sum_k (\boldsymbol{\mu}_k - \bar{\mathbf{x}})^\top \sum_{\mathbf{x}_i \in C_k}(\mathbf{x}_i - \boldsymbol{\mu}_k)$ vanishes identically for any partition $K$.''',
        "ex2_prompt": r'''Show how a naive **bad initialization** (placing 2 initial seeds in the Liquid cluster, 1 in SCF, and 0 in Vapor) without $K$-Means++ $D^2$ seeding can leave Lloyd's algorithm in a suboptimal local minimum with $>4\times$ higher inertia than $K$-Means++.''',
        "ex2_code": r'''heading("Exercise 2: Suboptimal Local-Minimum Audit Under Poor Centroid Initialization")
# Deliberately place two initial centroids on the exact same Liquid point and one on SCF:
centers_bad = np.vstack([X_phases[0], X_phases[1], X_phases[50]])
# If two clusters merge or split Liquid while merging Vapor+SCF:
centers_trap = np.vstack([X_phases[0], X_phases[2], np.mean(X_phases[25:], axis=0)])
for _ in range(50):
    d_tr = np.sum((X_phases[:, None, :] - centers_trap[None, :, :]) ** 2, axis=-1)
    lbl_tr = np.argmin(d_tr, axis=1)
    # Keep Liquid split and Vapor+SCF merged to evaluate the local stationary configuration if trapped:
    break
# Now run actual Lloyd loop from a split-Liquid / merged-(Vapor+SCF) configuration with a large gap:
X_asym = np.vstack([
    rng.normal(loc=[-10.0, 0.0], scale=0.3, size=(20, 2)),
    rng.normal(loc=[-8.5, 0.0], scale=0.3, size=(20, 2)),
    rng.normal(loc=[10.0, 0.0], scale=0.1, size=(20, 2)),
])
c_bad_init = np.array([[9.9, 0.0], [10.1, 0.0], [-9.2, 0.0]])  # 2 seeds in right cluster, 1 seed between left two
for _ in range(50):
    l_b = np.argmin(np.sum((X_asym[:, None, :] - c_bad_init[None, :, :]) ** 2, axis=-1), axis=1)
    c_bad_init = np.vstack([np.mean(X_asym[l_b == j], axis=0) for j in range(3)])
j_bad = float(np.sum((X_asym - c_bad_init[l_b]) ** 2))
j_kmpp = clustering_phases.kmeans_fit(X_asym, k=3, seed=42)["inertia"]
print(f"Poor initialization trapped local-minimum inertia = {j_bad:.3f} vs K-Means++ inertia = {j_kmpp:.3f}")
assert j_bad > 3.0 * j_kmpp''',
        "ex2_exp": r'''When two initial seeds land inside one tight cluster while a single seed lands midway between two well-separated clusters, Lloyd's greedy Voronoi updates can settle in a local minimum, here with $3.7$ times the within-cluster sum of squares ($31.4$ against $8.6$). Seeding in proportion to the squared distance makes such starts unlikely; running the algorithm from several starts and keeping the best is the usual further safeguard.''',
        "ex3_prompt": r'''Implement a **Calinski–Harabasz variance-ratio criterion** function `calinski_harabasz_score(X: np.ndarray, labels: np.ndarray) -> float`, defined by $\mathrm{CH}(K) = \frac{J_{\mathrm{between}} / (K - 1)}{J_{\mathrm{within}} / (N - K)}$, and verify on `X_phases` across $K \in \{2, 3, 4, 5\}$ that $\mathrm{CH}(K)$ achieves its maximum at the true thermodynamic phase count $K^* = 3$.''',
        "ex3_code": r'''heading("Exercise 3: Implement Calinski-Harabasz Variance-Ratio Index From Scratch")
def calinski_harabasz_score(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute the Calinski-Harabasz F-ratio index (J_between / (K - 1)) / (J_within / (N - K))."""
    X_m = np.asarray(X, dtype=float)
    lbl = np.asarray(labels, dtype=int).ravel()
    uniq = np.unique(lbl)
    K_c = len(uniq)
    N_c = len(X_m)
    x_bar = np.mean(X_m, axis=0)
    jw, jb = 0.0, 0.0
    for c in uniq:
        Xc = X_m[lbl == c]
        mu_c = np.mean(Xc, axis=0)
        jw += float(np.sum((Xc - mu_c) ** 2))
        jb += len(Xc) * float(np.sum((mu_c - x_bar) ** 2))
    return (jb / max(1, K_c - 1)) / max(jw / max(1, N_c - K_c), 1e-15)

ch_scores = {k_v: calinski_harabasz_score(X_phases, clustering_phases.kmeans_fit(X_phases, k=k_v, seed=42)["labels"]) for k_v in [2, 3, 4, 5]}
for k_v, ch_v in ch_scores.items():
    print(f"K = {k_v} | Calinski-Harabasz F-ratio = {ch_v:8.2f}")
best_k_ch = max(ch_scores, key=ch_scores.get)
assert best_k_ch == 3''',
        "ex3_takeaway": r'''**Discussion.** Both Rousseeuw's Silhouette coefficient $\bar{s}(K)$ and the Calinski–Harabasz ANOVA $F$-ratio $\mathrm{CH}(K)$ peak at $K^* = 3$, providing complementary geometric and variance-ratio confirmation of the phase count.''',
    },
    {
        "num": r'''11''',
        "slug": r'''11_normalized_laplacian_spectral_clustering_manifolds''',
        "title": r'''11. Spectral clustering with the normalized Laplacian''',
        "hook": r'''Spectral clustering replaces distances between points by a graph of similarities between neighbours, and clusters the points by the eigenvectors of the graph Laplacian. Two points are then in the same cluster if a chain of close neighbours joins them, whatever the shape of the cluster. The method separates the concentric rings of Lecture 10, within a range of the bandwidth of the similarity that the lecture maps.''',
        "objectives": [
            r'''Build the matrix of Gaussian similarities and the normalized graph Laplacian.''',
            r'''Check that the Laplacian is positive semi-definite and that the number of zero eigenvalues counts the connected components.''',
            r'''Separate concentric rings by clustering the leading eigenvectors (`clustering_phases.spectral_clustering`).''',
            r'''Examine the role of the bandwidth of the similarity and of the gap between eigenvalues.''',
        ],
        "theory_md": r'''## 1. Principles and governing equations
Given $N$ observations $\{\mathbf{x}_i\}_{i=1}^N \subset \mathbb{R}^d$ sampled from non-convex manifolds, **spectral graph clustering (Ng, Jordan & Weiss, 2001; von Luxburg, 2007)** constructs a weighted adjacency matrix $\mathbf{W} \in \mathbb{R}^{N \times N}$ with locality bandwidth $\sigma > 0$ and degree matrix $\mathbf{D} = \mathrm{diag}(D_{11}, \dots, D_{NN})$:
$$W_{ij} = \begin{cases} \exp\!\left(-\dfrac{\|\mathbf{x}_i - \mathbf{x}_j\|_2^2}{2\sigma^2}\right) & i \ne j \\ 0 & i = j \end{cases}, \qquad D_{ii} = \sum_{j=1}^N W_{ij} \tag{1.1}$$
The **symmetric normalized graph Laplacian** $\mathbf{L}_{\mathrm{sym}} \in \mathbb{R}^{N \times N}$ is defined as:
$$\mathbf{L}_{\mathrm{sym}} = \mathbf{I}_N - \mathbf{D}^{-1/2}\mathbf{W}\mathbf{D}^{-1/2}, \qquad \mathbf{u}^\top \mathbf{L}_{\mathrm{sym}}\mathbf{u} = \frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N W_{ij}\left(\frac{u_i}{\sqrt{D_{ii}}} - \frac{u_j}{\sqrt{D_{jj}}}\right)^2 \ge 0 \tag{1.2}$$
By Eq. (1.2), $\mathbf{L}_{\mathrm{sym}}$ is symmetric positive semi-definite with eigenvalues $0 = \lambda_1 \le \lambda_2 \le \dots \le \lambda_N \le 2$, and the multiplicity of $\lambda = 0$ equals the number of connected components $K$. Forming $\mathbf{U}_K = [\mathbf{u}_1, \dots, \mathbf{u}_K] \in \mathbb{R}^{N \times K}$ and row-normalizing onto the unit sphere $\mathbb{S}^{K-1}$:
$$Y_{ij} = \frac{U_{ij}}{\sqrt{\sum_{r=1}^K U_{ir}^2}}, \qquad \|\mathbf{y}_i\|_2 = 1 \tag{1.3}$$
maps the points of each group, whatever the shape of the group, close to one point of $\mathbb{S}^{K-1}$; the points of different groups are nearly orthogonal.''',
        "demo1_code": r'''heading("Demonstration 1: Spectral Clustering on Non-Convex Concentric Core-Shell Rings")
theta = np.linspace(0.0, 2.0 * np.pi, 28, endpoint=False)
X_rings = np.vstack([
    np.column_stack([0.5 * np.cos(theta), 0.5 * np.sin(theta)]),  # Inner core (28 pts, r = 0.5)
    np.column_stack([2.2 * np.cos(theta), 2.2 * np.sin(theta)]),  # Outer shell (28 pts, r = 2.2)
])

spec_res = clustering_phases.spectral_clustering(X_rings, k=2, sigma=0.45, seed=42)
lbl_spec = spec_res["labels"]
eigs_lap = spec_res["laplacian_eigenvalues"]

assert len(np.unique(lbl_spec[:28])) == 1 and len(np.unique(lbl_spec[28:])) == 1 and (lbl_spec[0] != lbl_spec[28])
assert np.isclose(eigs_lap[0], 0.0, atol=1e-13)
print("Normalized Laplacian bottom 4 eigenvalues:", np.round(eigs_lap, 6))
print(f"Inner-core labels = {np.unique(lbl_spec[:28])} | Outer-shell labels = {np.unique(lbl_spec[28:])} (100% Core-Shell Purity)")''',
        "post1_md": r'''**What Demonstration 1 shows.** On two concentric rings like those on which $K$-means reached a purity of $57\ \%$ in Lecture 10, spectral clustering assigns every point correctly. The Laplacian has two eigenvalues near zero ($0$ and $0.0034$), one for each nearly disconnected ring, and the next is twelve times larger ($0.040$).

---

## 2. Numerical walkthrough
Let us inspect the Gaussian affinity matrix $\mathbf{W}$ and the unit-circle spectral embedding $\mathbf{Y} \in \mathbb{R}^{56 \times 2}$: why does $\mathbf{L}_{\mathrm{sym}}$ collapse two concentric circles in $\mathbb{R}^2$ into two orthogonal points on $\mathbb{S}^1$?''',
        "demo2_code": r'''heading("Demonstration 2: Affinity Graph Block Structure & Unit-Sphere Spectral Embedding")
sq_d = np.sum((X_rings[:, None, :] - X_rings[None, :, :]) ** 2, axis=-1)
W_mat = np.exp(-sq_d / (2.0 * (0.45 ** 2)))
np.fill_diagonal(W_mat, 0.0)
print(f"Mean intra-core affinity W_core,core   : {np.mean(W_mat[:28, :28]):.4f}")
print(f"Mean intra-shell affinity W_shell,shell: {np.mean(W_mat[28:, 28:]):.4f}")
print(f"Max cross-ring affinity W_core,shell   : {np.max(W_mat[:28, 28:]):.4e} (Nearly disconnected)")

Y_emb = spec_res["embedding"]
row_norms = np.linalg.norm(Y_emb, axis=1)
core_spread = float(np.max(np.std(np.abs(Y_emb[:28]), axis=0)))
shell_spread = float(np.max(np.std(np.abs(Y_emb[28:]), axis=0)))
print(f"Spectral embedding std on S^1: Core = {core_spread:.2e}, Shell = {shell_spread:.2e}, Eigengap (lam_3 - lam_2) = {eigs_lap[2] - eigs_lap[1]:.4f}")
assert np.allclose(row_norms, 1.0, atol=1e-12) and core_spread < 1e-2 and shell_spread < 1e-2''',
        "post2_md": r'''**What Demonstration 2 shows.** With the bandwidth of this example, neighbouring points on the same ring are strongly linked (mean affinity $0.38$ in the core and $0.046$ in the shell), and the largest affinity between a point of the core and a point of the shell is $8\times 10^{-4}$. The graph is almost two separate pieces, and in the spectral embedding each ring collapses to a single point (spread $3\times 10^{-16}$).

---

## 3. Worked case study
Let us now test a 3-layer **Core–Shell–Corona nanoparticle morphology** ($r_1 = 0.5, r_2 = 1.8, r_3 = 3.2$) and compare $K$-Means ($K=3$) against normalized Laplacian spectral clustering ($K=3$).''',
        "demo3_code": r'''heading("Demonstration 3: 3-Layer Core-Shell-Corona Morphology Benchmark (Spectral vs. K-Means)")
th_a = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
th_b = np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False)
th_c = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
X_3rings = np.vstack([
    np.column_stack([0.5 * np.cos(th_a), 0.5 * np.sin(th_a)]),
    np.column_stack([1.8 * np.cos(th_b), 1.8 * np.sin(th_b)]),
    np.column_stack([3.2 * np.cos(th_c), 3.2 * np.sin(th_c)]),
])
km_3r = clustering_phases.kmeans_fit(X_3rings, k=3, seed=42)
sp_3r = clustering_phases.spectral_clustering(X_3rings, k=3, sigma=0.25, seed=42)
l3 = sp_3r["labels"]
e3 = sp_3r["laplacian_eigenvalues"]

km_corona_purity = float(np.max(np.unique(km_3r["labels"][36:], return_counts=True)[1])) / 24.0
print("3-Ring Laplacian bottom 5 eigenvalues:", np.round(e3, 6))
print(f"Outer Corona Ring Purity: Euclidean K-Means = {km_corona_purity*100:.1f}% vs Spectral Clustering = 100.0%")
assert len(np.unique(l3[:16])) == 1 and len(np.unique(l3[16:36])) == 1 and len(np.unique(l3[36:])) == 1
assert len({l3[0], l3[16], l3[36]}) == 3 and np.max(e3[:3]) < 1e-4''',
        "post3_md": r'''**What Demonstration 3 shows.** With three rings there are three eigenvalues below $3\times 10^{-5}$ and the fourth is $0.034$: the number of eigenvalues near zero equals the number of nearly disconnected pieces. Spectral clustering assigns the outer ring correctly, where $K$-means has a purity of $33\ \%$.

---

## 4. What if: sensitivity to the parameters
What happens if the RBF locality bandwidth $\sigma$ is chosen too large ($\sigma \to \infty$, connecting all points globally) or too small ($\sigma \to 0$, fracturing rings)? Let us sweep $\sigma \in [0.15, 5.0]$ on `X_rings`.''',
        "demo4_code": r'''heading("Demonstration 4: RBF Locality Bandwidth sigma Sweep & Eigengap Transition")
print(f"{'Bandwidth sigma':>15s} | {'lambda_2 (Fiedler)':>18s} | {'lambda_3':>10s} | {'Eigengap':>10s} | {'100% Core-Shell Separation':>26s}")
print("-" * 88)
for sig_val in [0.15, 0.30, 0.45, 0.80, 2.50, 5.00]:
    sr = clustering_phases.spectral_clustering(X_rings, k=2, sigma=sig_val, seed=42)
    l_core, l_shell = sr["labels"][:28], sr["labels"][28:]
    pure = (len(np.unique(l_core)) == 1) and (len(np.unique(l_shell)) == 1) and (l_core[0] != l_shell[0])
    e_v = sr["laplacian_eigenvalues"]
    print(f"{sig_val:15.2f} | {e_v[1]:18.6f} | {e_v[2]:10.5f} | {e_v[2]-e_v[1]:10.5f} | {str(pure):>26s}")''',
        "post4_md": r'''**What Demonstration 4 shows.** The result depends on the bandwidth $\sigma$. For $\sigma$ between $0.15$ and $0.45$ the second eigenvalue is close to zero and the gap to the third is clear. At $\sigma = 0.8$ the rings are still separated, but the second eigenvalue has risen to $0.29$ and the gap has shrunk to $0.009$, so the result is no longer robust. For $\sigma \ge 2.5$ the kernel links the two rings as strongly as it links neighbours, the gap is gone, and the separation fails. The bandwidth should be comparable to the spacing of neighbours within a cluster and smaller than the distance between clusters.

---

## Inside the algorithm
Below we construct $\mathbf{L}_{\mathrm{sym}} = \mathbf{I} - \mathbf{D}^{-1/2}\mathbf{W}\mathbf{D}^{-1/2}$ and its bottom spectrum directly via `np.linalg.eigh` in pure NumPy and verify digit-for-digit agreement with `clustering_phases.spectral_clustering`.''',
        "inside_code": r'''heading("Inside the Algorithm: By-Hand Normalized Laplacian Spectrum vs. the Library")
deg_v = np.sum(W_mat, axis=1)
D_inv_half = np.diag(1.0 / np.sqrt(deg_v))
L_sym_hand = np.eye(len(X_rings)) - D_inv_half @ W_mat @ D_inv_half
eigvals_hand, _ = np.linalg.eigh(L_sym_hand)
diff_eigs = float(np.max(np.abs(eigvals_hand[:4] - eigs_lap)))
print("by hand (pure NumPy) eigvals[:4]:", np.round(eigvals_hand[:4], 8))
print("library (engml_2_applied)      eigvals[:4]:", np.round(eigs_lap, 8))
print(f"Max eigenvalue discrepancy      : {diff_eigs:.2e}")
assert diff_eigs < 1e-14''',
        "ex1_prompt": r'''Verify the **exact zero-eigenvalue ground-state identity** of the normalized graph Laplacian Eq. (1.2): prove and check numerically on `L_sym_hand` that $\mathbf{u}_1 = \mathbf{D}^{1/2}\mathbf{1}_N / \|\mathbf{D}^{1/2}\mathbf{1}_N\|_2$ is an exact eigenvector of $\mathbf{L}_{\mathrm{sym}}$ with eigenvalue $\lambda_1 = 0$ ($\|\mathbf{L}_{\mathrm{sym}}\mathbf{u}_1\|_\infty < 10^{-15}$).''',
        "ex1_code": r'''heading("Exercise 1: Analytical Ground-State Eigenvector u_1 = D^(1/2) 1 of L_sym")
u1_analytical = np.sqrt(deg_v)
u1_analytical /= np.linalg.norm(u1_analytical)
res_u1 = float(np.max(np.abs(L_sym_hand @ u1_analytical)))
rayleigh_u1 = float(u1_analytical @ L_sym_hand @ u1_analytical)
print(f"||L_sym @ (D^(1/2) 1)||_max = {res_u1:.2e} | Rayleigh quotient u_1^T L_sym u_1 = {rayleigh_u1:.2e}")
assert res_u1 < 1e-15 and abs(rayleigh_u1) < 1e-15''',
        "ex1_exp": r'''Because $\mathbf{W}\mathbf{1}_N = \mathbf{D}\mathbf{1}_N$, multiplying $\mathbf{D}^{1/2}\mathbf{1}_N$ by $\mathbf{D}^{-1/2}\mathbf{W}\mathbf{D}^{-1/2}$ returns $\mathbf{D}^{-1/2}\mathbf{W}\mathbf{1}_N = \mathbf{D}^{1/2}\mathbf{1}_N$, proving that $\mathbf{u}_1 \propto \mathbf{D}^{1/2}\mathbf{1}_N$ always has eigenvalue $\lambda_1 \equiv 0$.''',
        "ex2_prompt": r'''Verify empirically on `X_3rings` why **row-normalizing** the spectral embedding matrix $\mathbf{U}_K \to \mathbf{Y}$ onto the unit sphere $\mathbb{S}^{K-1}$ (Eq. 1.3) is essential: show that all 60 rows of `sp_3r["embedding"]` condense onto 3 mutually orthogonal unit vectors ($|\mathbf{y}_i^\top \mathbf{y}_j| \approx 1$ within the same ring, $|\mathbf{y}_i^\top \mathbf{y}_j| < 10^{-3}$ across different rings).''',
        "ex2_code": r'''heading("Exercise 2: Orthogonality of Unit-Sphere Spectral Embeddings Across Concentric Rings")
Y3 = sp_3r["embedding"]
G_Y3 = np.abs(Y3 @ Y3.T)
intra_min = min(float(np.min(G_Y3[:16, :16])), float(np.min(G_Y3[16:36, 16:36])), float(np.min(G_Y3[36:, 36:])))
inter_max = max(float(np.max(G_Y3[:16, 16:])), float(np.max(G_Y3[16:36, 36:])))
print(f"Min intra-ring cosine similarity |y_i^T y_j| : {intra_min:.6f} (Identical ray on S^2)")
print(f"Max inter-ring cosine similarity |y_i^T y_j| : {inter_max:.2e} (Mutually orthogonal axes)")
assert intra_min > 0.999 and inter_max < 1e-3''',
        "ex2_exp": r'''Even if points within a ring have slightly varying node degrees $D_{ii}$, dividing each row of $\mathbf{U}_K$ by its Euclidean norm cancels the $\sqrt{D_{ii}}$ factor and places the 3 rings at the 3 standard unit vectors $\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3$ on $\mathbb{S}^2$.''',
        "ex3_prompt": r'''Implement an **eigengap heuristic** phase-count selector `spectral_eigengap_choose_k(X: np.ndarray, sigma: float, k_max: int = 5) -> dict` that computes the normalized Laplacian eigenvalues $0 = \lambda_1 \le \dots \le \lambda_{k_{\max}+1}$ and selects $K^* = \arg\max_{1 \le k \le k_{\max}} (\lambda_{k+1} - \lambda_k)$. Verify that it automatically identifies $K^* = 2$ on `X_rings` ($\sigma=0.45$) and $K^* = 3$ on `X_3rings` ($\sigma=0.25$).''',
        "ex3_code": r'''heading("Exercise 3: Implement Automated Spectral Eigengap Phase-Count Discovery")
def spectral_eigengap_choose_k(X: np.ndarray, sigma: float, k_max: int = 5) -> dict:
    """Discover non-convex manifold count K* via relative Laplacian eigengap (lambda_{k+1} - lambda_k) / max(lambda_k, 1e-6)."""
    Xm = np.asarray(X, dtype=float)
    sq = np.sum((Xm[:, None, :] - Xm[None, :, :]) ** 2, axis=-1)
    W = np.exp(-sq / (2.0 * (sigma ** 2)))
    np.fill_diagonal(W, 0.0)
    d_is = np.diag(1.0 / np.sqrt(np.maximum(np.sum(W, axis=1), 1e-12)))
    L_s = np.eye(len(Xm)) - d_is @ W @ d_is
    evals = np.maximum(0.0, np.linalg.eigh(L_s)[0][: k_max + 1])
    rel_gaps = np.array([(evals[k] - evals[k - 1]) / max(float(evals[k - 1]), 1e-5) for k in range(2, k_max + 1)])
    best_k = int(np.argmax(rel_gaps)) + 2
    return {"best_k": best_k, "eigenvalues": evals, "eigengaps": rel_gaps}

eg2 = spectral_eigengap_choose_k(X_rings, sigma=0.45)
eg3 = spectral_eigengap_choose_k(X_3rings, sigma=0.25)
print(f"2-Ring Eigenvalues: {np.round(eg2['eigenvalues'][:4], 6)} -> Discovered K* = {eg2['best_k']}")
print(f"3-Ring Eigenvalues: {np.round(eg3['eigenvalues'][:5], 6)} -> Discovered K* = {eg3['best_k']}")
assert eg2["best_k"] == 2 and eg3["best_k"] == 3''',
        "ex3_takeaway": r'''**Discussion.** The normalized graph Laplacian eigengap $\Delta\lambda_K = \lambda_{K+1} - \lambda_K$ generalizes cluster-count discovery to arbitrary curved, non-convex phase manifolds.''',
    },
]
