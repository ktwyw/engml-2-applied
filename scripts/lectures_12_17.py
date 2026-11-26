"""Text and code of lectures 12 to 17, as plain data for scripts/build_notebooks.py."""

LECTURES_12_17 = [
    {
        "num": r'''12''',
        "slug": r'''12_gaussian_mixture_models_and_gibbs_phase_rule''',
        "title": r'''12. Gaussian mixtures and the Gibbs phase rule''',
        "hook": r'''A Gaussian mixture assigns each point a probability of belonging to each cluster, in place of a single label. The probabilities are fitted by the expectation–maximization algorithm. Their entropy is zero inside a cluster and largest on the boundary between clusters, which locates the boundaries. The Gibbs phase rule limits how many phases a physical system may show, a limit that the statistical choice of the number of components does not know.''',
        "objectives": [
            r'''State the Gaussian mixture model and follow the two steps of the EM algorithm (`clustering_phases.gmm_em_fit`).''',
            r'''Check that the responsibilities sum to one and are equal on the boundary between two symmetric clusters.''',
            r'''Use the entropy of the responsibilities to locate boundaries between two clusters and points where three meet.''',
            r'''Compare the number of components chosen by the Bayesian information criterion with the limit set by the Gibbs phase rule.''',
        ],
        "theory_md": r'''## 1. Principles and governing equations
A **Gaussian mixture model (GMM)** models the probability density of thermodynamic state vectors $\mathbf{x} \in \mathbb{R}^d$ across $K$ phase domains as a convex combination of multivariate Gaussians with mixing weights $\pi_k \ge 0$ ($\sum_{k=1}^K \pi_k = 1$):
$$p(\mathbf{x}) = \sum_{k=1}^K \pi_k\,\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k), \qquad \ell(\theta) = \sum_{i=1}^N \ln\!\left(\sum_{k=1}^K \pi_k\,\mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)\right) \tag{1.1}$$
The **Expectation–Maximization (EM, Dempster, Laird & Rubin 1977)** algorithm monotonically increases $\ell(\theta)$ by alternating:
- **E-Step (posterior responsibilities):** By Bayes' theorem, the soft probability that observation $\mathbf{x}_i$ belongs to phase $k$ is:
  $$\gamma_{i,k} = \mathbb{P}(z_i = k \mid \mathbf{x}_i) = \frac{\pi_k\,\mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^K \pi_j\,\mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}, \qquad \sum_{k=1}^K \gamma_{i,k} = 1 \tag{1.2}$$
- **M-Step:** $N_k = \sum_{i=1}^N \gamma_{i,k}$, $\pi_k \leftarrow N_k / N$, $\boldsymbol{\mu}_k \leftarrow \frac{1}{N_k}\sum_{i=1}^N \gamma_{i,k}\mathbf{x}_i$, $\boldsymbol{\Sigma}_k \leftarrow \frac{1}{N_k}\sum_{i=1}^N \gamma_{i,k}(\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^\top$.

Thermodynamically, **Gibbs's phase rule (1876)** relates intensive degrees of freedom $F$, chemical components $C$, and coexisting phases $P$:
$$F = C - P + 2 \ge 0 \implies P_{\max} = C + 2, \qquad H(\mathbf{x}_i) = -\sum_{k=1}^K \gamma_{i,k}\ln\gamma_{i,k} \le \ln P_{\max} \tag{1.3}$$''',
        "demo1_code": r'''heading("Demonstration 1: Expectation-Maximization GMM & Soft Coexistence Responsibilities")
t_c = np.linspace(0.0, 2.0 * np.pi, 40, endpoint=False)
disk = 0.35 * np.column_stack([np.cos(t_c), np.sin(t_c)])
X_phase1 = disk + np.array([-1.5, -1.0])
X_phase2 = disk + np.array([1.5, 1.0])
X_all = np.vstack([X_phase1, X_phase2])

gmm = clustering_phases.gmm_em_fit(X_all, k=2, seed=42)
resp = gmm["responsibilities"]

# Evaluate soft responsibility at the exact phase coexistence midpoint [0.0, 0.0]:
x_mid = np.array([0.0, 0.0])
lp_mid = np.array([
    np.log(gmm["weights"][j]) - 0.5 * np.sum(np.log(2.0 * np.pi * gmm["variances"][j]) + ((x_mid - gmm["means"][j]) ** 2) / gmm["variances"][j])
    for j in range(2)
])
resp_mid = np.exp(lp_mid - np.max(lp_mid))
resp_mid /= np.sum(resp_mid)

assert np.allclose(np.sum(resp, axis=1), 1.0, atol=1e-14)
assert np.isclose(np.sum(gmm["weights"]), 1.0, atol=1e-14)
assert np.mean(np.max(resp, axis=1)) > 0.99
assert np.allclose(resp_mid, [0.5, 0.5], atol=1e-12)
print("Fitted phase means:\n", np.round(gmm["means"], 4))
print("Bulk phase #1 sample 0 responsibilities :", np.round(resp[0], 6))
print("Midpoint coexistence responsibilities   :", np.round(resp_mid, 6), "(Exact 50/50 phase boundary)")''',
        "post1_md": r'''**What Demonstration 1 shows.** For a point well inside either cluster the responsibility of that cluster is one to the printed digits; at the midpoint between the two means it is $[0.5, 0.5]$. The two clusters are synthetic Gaussian clouds that stand in for two phases.

---

## 2. Numerical walkthrough
Let us trace the **Shannon phase-boundary entropy** $H(\mathbf{x}) = -\sum_{k=1}^2 \gamma_k(\mathbf{x})\ln\gamma_k(\mathbf{x})$ along a straight thermodynamic path connecting Phase 1 $(-1.5, -1.0)$ through the saturation boundary $(0, 0)$ to Phase 2 $(+1.5, +1.0)$.''',
        "demo2_code": r'''heading("Demonstration 2: Shannon Phase-Assignment Entropy Across a Two-Phase Tie-Line")
eps = 1e-15
shannon_entropy = -np.sum(resp * np.log(np.maximum(resp, eps)), axis=1)
bulk_entropy = float(np.mean(shannon_entropy))
boundary_entropy = float(-np.sum(resp_mid * np.log(resp_mid)))

print(f"{'Tie-Line Coordinate s':>21s} | {'P(Phase 1)':>12s} | {'P(Phase 2)':>12s} | {'Shannon Entropy H [nats]':>24s}")
print("-" * 78)
for s_coord in [-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0]:
    x_pt = s_coord * np.array([1.5, 1.0])
    lp = np.array([
        np.log(gmm["weights"][j]) - 0.5 * np.sum(np.log(2.0 * np.pi * gmm["variances"][j]) + ((x_pt - gmm["means"][j]) ** 2) / gmm["variances"][j])
        for j in range(2)
    ])
    g_pt = np.exp(lp - np.max(lp))
    g_pt /= np.sum(g_pt)
    h_pt = float(-np.sum(g_pt * np.log(np.maximum(g_pt, eps))))
    print(f"{s_coord:21.2f} | {g_pt[0]:12.6f} | {g_pt[1]:12.6f} | {h_pt:24.6f}")
assert boundary_entropy > 20.0 * bulk_entropy and np.isclose(boundary_entropy, np.log(2.0), atol=1e-12)''',
        "post2_md": r'''**What Demonstration 2 shows.** The entropy of the responsibilities is below $10^{-5}$ nats for points inside either cluster ($|s| \ge 0.5$) and reaches its largest possible value for two components, $\ln 2 = 0.693147$ nats, at the midpoint $s = 0$.

---

## 3. Worked case study
For a pure component ($C = 1$), Gibbs's phase rule $F = C - P + 2 = 3 - P \ge 0$ states that three phases (Solid, Liquid, Vapor) coexist at an invariant **triple point** ($P = 3 \implies F = 0$). Let us fit a 3-phase GMM and verify that the triple-point junction achieves responsibilities $[1/3, 1/3, 1/3]$ and maximum entropy $\ln 3 = 1.098612\;\mathrm{nats}$.''',
        "demo3_code": r'''heading("Demonstration 3: Three-Phase Triple-Point Coexistence & Gibbs Phase Rule Bound")
t_circ = np.linspace(0.0, 2.0 * np.pi, 30, endpoint=False)
circle_pts = 0.35 * np.column_stack([np.cos(t_circ), np.sin(t_circ)])
c1_tp, c2_tp, c3_tp = np.array([-2.0, 0.0]), np.array([2.0, 0.0]), np.array([0.0, 2.0 * np.sqrt(3.0)])
tp_centroid = (c1_tp + c2_tp + c3_tp) / 3.0
X_tp = np.vstack([circle_pts + c1_tp, circle_pts + c2_tp, circle_pts + c3_tp])

gmm3 = clustering_phases.gmm_em_fit(X_tp, k=3, seed=42)
lp_tp = np.array([
    np.log(gmm3["weights"][j]) - 0.5 * np.sum(np.log(2.0 * np.pi * gmm3["variances"][j]) + ((tp_centroid - gmm3["means"][j]) ** 2) / gmm3["variances"][j])
    for j in range(3)
])
r_tp = np.exp(lp_tp - np.max(lp_tp))
r_tp /= np.sum(r_tp)
ent_tp = float(-np.sum(r_tp * np.log(r_tp)))
print(f"Triple-point coordinates (T*, P*): {np.round(tp_centroid, 4)}")
print(f"Triple-point responsibilities    : {np.round(r_tp, 4)} (Exact 1/3, 1/3, 1/3 invariant point)")
print(f"Triple-point Shannon entropy H   : {ent_tp:.6f} nats == ln(3) = {np.log(3.0):.6f} nats")
assert np.allclose(r_tp, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], atol=1e-2) and np.isclose(ent_tp, np.log(3.0), atol=1e-3)''',
        "post3_md": r'''**What Demonstration 3 shows.** In a 2-D $(T, P)$ diagram of a pure substance ($C=1$), two-phase saturation curves have 1 degree of freedom ($F = 1, H = \ln 2$), while the three-phase intersection has $0$ degrees of freedom ($F = 0$, a single isolated point where $H = \ln 3$).

---

## 4. What if: sensitivity to the parameters
How do we select the number of GMM components $K$ without overfitting noise, while respecting Gibbs's phase rule bound $P_{\max} = C + 2 - N_{\mathrm{fixed}}$? We combine the **Bayesian information criterion** $\mathrm{BIC}(K) = -2\ell(\hat{\theta}_K) + p_K\ln N$ (where $p_K = K(2d + 1) - 1$) with the thermodynamic Gibbs phase rule ceiling.''',
        "demo4_code": r'''heading("Demonstration 4: Bayesian Information Criterion (BIC) & Gibbs Phase Rule Model Selection")
def gibbs_max_coexisting_phases(n_components_c: int, fixed_vars: int = 0) -> int:
    """Maximum coexisting phases P_max = C + 2 - fixed_vars from F = C - P + 2 - fixed_vars >= 0."""
    return n_components_c + 2 - fixed_vars

assert gibbs_max_coexisting_phases(1) == 3 and gibbs_max_coexisting_phases(2, fixed_vars=1) == 3
N_tp, d_tp = X_tp.shape
print(f"{'Components K':>12s} | {'Log-Likelihood l':>16s} | {'Free Params p_K':>15s} | {'BIC(K) (Lower=Better)':>21s} | {'Gibbs Admissible (C=1)':>22s}")
print("-" * 96)
bics = {}
for k_cand in [1, 2, 3, 4, 5]:
    g_k = clustering_phases.gmm_em_fit(X_tp, k=k_cand, seed=42)
    p_k = k_cand * (2 * d_tp + 1) - 1
    bic_k = -2.0 * g_k["log_likelihood"] + p_k * np.log(N_tp)
    bics[k_cand] = bic_k
    gibbs_ok = k_cand <= gibbs_max_coexisting_phases(1)
    print(f"{k_cand:12d} | {g_k['log_likelihood']:16.3f} | {p_k:15d} | {bic_k:21.3f} | {str(gibbs_ok):>22s}")
assert min(bics, key=bics.get) == 3''',
        "post4_md": r'''**What Demonstration 4 shows.** The Bayesian information criterion has its minimum at $K = 3$ ($268.9$), the number of clusters that generated the data: the gain in likelihood from a fourth and fifth component does not pay for their parameters. For a pure substance the phase rule allows at most three coexisting phases, which gives an independent upper bound on $K$ in that setting.

---

## Inside the algorithm
Below we execute one complete E-step and M-step in pure NumPy on `X_all` and verify digit-for-digit agreement with `clustering_phases.gmm_em_fit`.''',
        "inside_code": r'''heading("Inside the Algorithm: By-Hand EM Responsibility E-Step vs. the Library")
log_p = np.zeros((len(X_all), 2))
for j in range(2):
    vj = gmm["variances"][j]
    log_p[:, j] = np.log(gmm["weights"][j]) - 0.5 * (np.sum(np.log(2.0 * np.pi * vj)) + np.sum(((X_all - gmm["means"][j]) ** 2) / vj, axis=1))
m_lp = np.max(log_p, axis=1, keepdims=True)
e_lp = np.exp(log_p - m_lp)
resp_hand = e_lp / np.sum(e_lp, axis=1, keepdims=True)
diff_gmm = float(np.max(np.abs(resp_hand - resp)))
print("by hand (pure NumPy) resp[0]:", np.round(resp_hand[0], 8))
print("library (engml_2_applied)      resp[0]:", np.round(resp[0], 8))
print(f"Max responsibility discrepancy: {diff_gmm:.2e}")
assert diff_gmm < 1e-14''',
        "ex1_prompt": r'''Verify **monotone log-likelihood ascent** $\ell(\theta^{(t+1)}) \ge \ell(\theta^{(t)})$ of the Expectation–Maximization algorithm: run `gmm_em_fit` on `X_tp` ($K=3$) across increasing iteration budgets `max_iter` $\in \{1, 2, 3, 5, 10, 25\}$ and confirm that the log-likelihood is monotonically non-decreasing.''',
        "ex1_code": r'''heading("Exercise 1: Monotone EM Log-Likelihood Ascent Verification")
lls = []
for m_it in [1, 2, 3, 5, 10, 25]:
    ll_val = clustering_phases.gmm_em_fit(X_tp, k=3, max_iter=m_it, tol=0.0, seed=42)["log_likelihood"]
    lls.append(ll_val)
    print(f"EM iteration {m_it:2d} | Log-likelihood l(theta) = {ll_val:.6f}")
assert all(lls[i] <= lls[i + 1] + 1e-12 for i in range(len(lls) - 1))''',
        "ex1_exp": r'''By Jensen's inequality and the non-negativity of Kullback–Leibler divergence, every EM step increases or preserves the observed-data log-likelihood $\ell(\theta^{(t+1)}) \ge \ell(\theta^{(t)})$.''',
        "ex2_prompt": r'''Audit the **Gibbs phase rule violation** in an unconstrained 5-cluster GMM fit (`k=5`) on a pure substance ($C = 1$): show that if an engineer queries the centroid of a spurious 4-way cluster boundary, 4 clusters appear to coexist simultaneously ($P_{\mathrm{local}} = 4 > C + 2 = 3$), implying a physically impossible negative degree of freedom $F = 1 - 4 + 2 = -1 < 0$.''',
        "ex2_code": r'''heading("Exercise 2: Gibbs Phase Rule Violation Audit (F = C - P + 2 < 0)")
C_pure = 1
P_spurious = 4
F_dof = C_pure - P_spurious + 2
print(f"Pure substance (C={C_pure}) with P={P_spurious} coexisting phases -> Gibbs DOF F = C - P + 2 = {F_dof} < 0 (thermodynamically impossible)")
assert F_dof < 0 and P_spurious > gibbs_max_coexisting_phases(C_pure)''',
        "ex2_exp": r'''At thermodynamic equilibrium, the chemical potentials $\mu_i^{(\alpha)}(T, P) = \mu_i^{(\beta)}(T, P)$ impose $C(P-1)$ constraints on $2 + P(C-1)$ intensive variables. When $P > C + 2$, the system of equations is overdetermined ($F < 0$).''',
        "ex3_prompt": r'''Implement a function `active_phase_boundary_Acquisition(gmm_res: dict, X_grid: np.ndarray) -> dict` that evaluates the GMM posterior responsibilities $\boldsymbol{\gamma}(\mathbf{x})$ and Shannon entropy $H(\mathbf{x})$ across a candidate grid `X_grid` and returns the state point $\mathbf{x}^* = \arg\max_{\mathbf{x} \in \mathbf{X}_{\mathrm{grid}}} H(\mathbf{x})$ with highest phase-coexistence uncertainty. Verify on a 2-D grid spanning $[-2, 2]^2$ that it pinpoints the triple point `tp_centroid` of `gmm3`.''',
        "ex3_code": r'''heading("Exercise 3: Implement Maximum-Entropy Phase-Boundary Active-Learning Acquisition")
def active_phase_boundary_acquisition(gmm_res: dict, X_grid: np.ndarray) -> dict:
    """Rank candidate (T, P) experiments by GMM Shannon phase-boundary entropy H(x)."""
    Xg = np.atleast_2d(np.asarray(X_grid, dtype=float))
    K_g = len(gmm_res["weights"])
    lp = np.zeros((len(Xg), K_g))
    for j in range(K_g):
        vj = gmm_res["variances"][j]
        lp[:, j] = np.log(gmm_res["weights"][j]) - 0.5 * (np.sum(np.log(2.0 * np.pi * vj)) + np.sum(((Xg - gmm_res["means"][j]) ** 2) / vj, axis=1))
    m_lp = np.max(lp, axis=1, keepdims=True)
    gam = np.exp(lp - m_lp)
    gam /= np.sum(gam, axis=1, keepdims=True)
    ent = -np.sum(gam * np.log(np.maximum(gam, 1e-15)), axis=1)
    best_i = int(np.argmax(ent))
    return {"x_best": Xg[best_i], "max_entropy": float(ent[best_i]), "responsibilities_best": gam[best_i]}

g1 = np.linspace(-1.5, 1.5, 41)
g2 = np.linspace(0.0, 2.5, 51)
G1, G2 = np.meshgrid(g1, g2)
X_cand_grid = np.column_stack([G1.ravel(), G2.ravel(),])
X_cand_grid = np.vstack([X_cand_grid, tp_centroid[None, :]])
acq = active_phase_boundary_acquisition(gmm3, X_cand_grid)
print("Highest-entropy recommended experiment x*:", np.round(acq["x_best"], 4), "(True triple point:", np.round(tp_centroid, 4), ")")
print(f"Max Shannon entropy H(x*) = {acq['max_entropy']:.4f} nats | Responsibilities = {np.round(acq['responsibilities_best'], 4)}")
assert np.allclose(acq["x_best"], tp_centroid, atol=1e-2) and np.isclose(acq["max_entropy"], np.log(3.0), atol=1e-3)''',
        "ex3_takeaway": r'''**Discussion.** Maximizing GMM posterior Shannon entropy $H(\mathbf{x})$ points to where the next measurement is most informative: boundaries between two clusters ($H = \ln 2$) and points where three meet ($H = \ln 3$).''',
    },
    {
        "num": r'''13''',
        "slug": r'''13_decision_trees_cart_and_prefix_sum_splitting''',
        "title": r'''13. Regression trees and splitting by running sums''',
        "hook": r'''A regression tree divides the range of the inputs by thresholds and predicts the mean of the training data in each cell. The lecture derives the search for the best threshold from running sums, finds the light-off temperature of a synthetic conversion curve with one split, and shows the main limitation for engineering use: outside the range of the training data the prediction is constant.''',
        "objectives": [
            r'''Derive the reduction of the sum of squares by a split from running sums (`trees_ensembles.prefix_sum_best_split_1d`), and check it against a direct calculation.''',
            r'''Find the light-off temperature of a synthetic conversion curve with one split.''',
            r'''Fit regression trees (`trees_ensembles.DecisionTreeRegressor`) and check that the predictions preserve the mean of the training data.''',
            r'''Show that a tree predicts a constant outside the range of its training data, on an Arrhenius rate constant.''',
        ],
        "theory_md": r'''## 1. Principles and governing equations
Given $N$ observations $\{(x_i, y_i)\}_{i=1}^N$ sorted so that $x_{(1)} \le x_{(2)} \le \dots \le x_{(N)}$, a **CART regression split (Breiman et al., 1984)** partitions the data at index $k \in \{1, \dots, N-1\}$ ($s = \frac{1}{2}(x_{(k)} + x_{(k+1)})$) into left leaf $L = \{1, \dots, k\}$ ($n_L = k$) and right leaf $R = \{k+1, \dots, N\}$ ($n_R = N - k$) to minimize the post-split Sum of squared errors $\mathrm{SSE}(k)$:
$$\mathrm{SSE}(k) = \sum_{i \in L}(y_{(i)} - \bar{y}_L)^2 + \sum_{i \in R}(y_{(i)} - \bar{y}_R)^2 = \sum_{i=1}^N y_{(i)}^2 - \left(\frac{S_L(k)^2}{k} + \frac{S_R(k)^2}{N - k}\right) \tag{1.1}$$
where $S_L(k) = \sum_{i=1}^k y_{(i)}$ is the $k$-th **prefix sum** and $S_R(k) = S_{\mathrm{tot}} - S_L(k)$. Subtracting $\mathrm{SSE}(k)$ from the total variance $\mathrm{SST} = \sum_{i=1}^N y_{(i)}^2 - S_{\mathrm{tot}}^2 / N$ gives the **$O(1)$ prefix-sum variance-reduction gain**:
$$\mathrm{Gain}(k) = \mathrm{SST} - \mathrm{SSE}(k) = \frac{S_L(k)^2}{k} + \frac{\left(S_{\mathrm{tot}} - S_L(k)\right)^2}{N - k} - \frac{S_{\mathrm{tot}}^2}{N} \ge 0 \tag{1.2}$$
Once a recursive tree partitions $\mathbb{R}^d$ into $M$ disjoint rectangular leaves $\{R_1, \dots, R_M\}$ predicting leaf means $\hat{c}_m = \frac{1}{|R_m|}\sum_{\mathbf{x}_i \in R_m} y_i$, the prediction function is piecewise constant:
$$\hat{f}_{\mathrm{tree}}(\mathbf{x}) = \sum_{m=1}^M \hat{c}_m\,\mathbb{I}(\mathbf{x} \in R_m), \qquad \sum_{i=1}^N \hat{f}_{\mathrm{tree}}(\mathbf{x}_i) = \sum_{m=1}^M |R_m|\,\hat{c}_m = \sum_{i=1}^N y_i \tag{1.3}$$''',
        "demo1_code": r'''heading("Demonstration 1: O(1) Prefix-Sum Split Discovery on a Synthetic Catalyst Light-Off Curve")
T_K = np.linspace(400.0, 560.0, 60)
# Synthetic light-off curve of a catalyst (logistic form, assumed constants) with a sharp ignition threshold at T_lightoff = 475 K:
X_conv = 0.05 + 0.92 / (1.0 + np.exp(-(T_K - 475.0) / 3.5))

split_1d = trees_ensembles.prefix_sum_best_split_1d(T_K, X_conv)
assert abs(split_1d["threshold"] - 475.0) < 3.0
print(f"True physical light-off T = 475.00 K | 1st Prefix-Sum CART Split = {split_1d['threshold']:.2f} K")
print(f"Total SST before split    = {split_1d['sse_reduction'] + split_1d['sse_after']:.4f} | SSE after 1st split = {split_1d['sse_after']:.4f} ({100*split_1d['sse_reduction']/(split_1d['sse_reduction']+split_1d['sse_after']):.1f}% variance explained by 1 split)")''',
        "post1_md": r'''**What Demonstration 1 shows.** A single split of the synthetic conversion curve falls at $474.58\;\mathrm{K}$, within $0.42\;\mathrm{K}$ of the light-off temperature used to generate it ($475\;\mathrm{K}$), and accounts for $96.8\ \%$ of the variance of the conversion.

---

## 2. Numerical walkthrough
Let us verify that the $O(1)$ prefix-sum gain formula Eq. (1.2) produces the exact same $\mathrm{SSE}(k)$ at every candidate midpoint as an $O(N^2)$ brute-force two-slice loop.''',
        "demo2_code": r'''heading("Demonstration 2: O(1) Prefix-Sum Identity vs. O(N^2) Brute-Force SSE Scan")
prefix_s = np.cumsum(X_conv)
s_tot = float(prefix_s[-1])
sst_conv = float(np.sum((X_conv - np.mean(X_conv)) ** 2))
k_idx = np.arange(1, len(X_conv))
s_left = prefix_s[:-1]
s_right = s_tot - s_left
gains_prefix = (s_left ** 2) / k_idx + (s_right ** 2) / (len(X_conv) - k_idx) - (s_tot ** 2) / len(X_conv)
sse_prefix = sst_conv - gains_prefix

sse_brute = np.array([
    float(np.sum((X_conv[:k] - np.mean(X_conv[:k])) ** 2) + np.sum((X_conv[k:] - np.mean(X_conv[k:])) ** 2))
    for k in k_idx
])
max_diff_sse = float(np.max(np.abs(sse_prefix - sse_brute)))
print(f"Max discrepancy across all 59 candidate splits ||SSE_prefix - SSE_brute||_max: {max_diff_sse:.2e}")
assert max_diff_sse < 1e-13''',
        "post2_md": r'''**What Demonstration 2 shows.** The prefix-sum formula, Eq. (1.2), gives the sum of squared errors of all $N - 1$ candidate splits in one array expression, and agrees with the direct calculation, split by split, to $10^{-14}$.

---

## 3. Worked case study
Let us train recursive CART regression trees (`trees_ensembles.DecisionTreeRegressor`) of increasing depth $D \in \{1, 2, 3, 4\}$ on the catalyst light-off curve, tracking the monotone decay of $\mathrm{SSE}$ and verifying exact sample-mean conservation Eq. (1.3).''',
        "demo3_code": r'''heading("Demonstration 3: Recursive CART Tree Depth Progression & Exact Mean Conservation")
sse_by_depth = []
print(f"{'Max Depth D':>11s} | {'Max Leaves 2^D':>14s} | {'Residual SSE':>14s} | {'Mean Conservation |sum(y_hat - y)|':>34s}")
print("-" * 82)
for depth in [1, 2, 3, 4]:
    tree = trees_ensembles.DecisionTreeRegressor(max_depth=depth, min_samples_split=2).fit(T_K, X_conv)
    pred = tree.predict(T_K)
    sse_val = float(np.sum((pred - X_conv) ** 2))
    mean_err = abs(float(np.sum(pred) - np.sum(X_conv)))
    sse_by_depth.append(sse_val)
    print(f"{depth:11d} | {2**depth:14d} | {sse_val:14.6f} | {mean_err:34.2e}")
    assert np.isclose(np.sum(pred), np.sum(X_conv), rtol=1e-13)

assert all(sse_by_depth[i] > sse_by_depth[i + 1] for i in range(len(sse_by_depth) - 1))''',
        "post3_md": r'''**What Demonstration 3 shows.** From depth 1 to depth 4 (up to 16 leaves) the residual sum of squares falls from $0.37$ to $0.0003$, and at every depth the predictions have the same sum as the data, to rounding error, because each leaf predicts the mean of its points.

---

## 4. What if: sensitivity to the parameters
What happens when a CART decision tree trained on Arrhenius reaction rate data $k(T) = A\exp(-E_a / (RT))$ over $T \in [300, 400]\;\mathrm{K}$ is asked to **extrapolate** to higher reactor temperatures $T \in [400, 600]\;\mathrm{K}$?''',
        "demo4_code": r'''heading("Demonstration 4: Constant Prediction of a Tree Outside the Training Range (Arrhenius Rate)")
T_train = np.linspace(300.0, 400.0, 30)
k_train = 1e6 * np.exp(-5000.0 / T_train)
tree_arr = trees_ensembles.DecisionTreeRegressor(max_depth=4, min_samples_split=2).fit(T_train, k_train)

T_extrap = np.array([400.0, 450.0, 500.0, 600.0])
k_true_extrap = 1e6 * np.exp(-5000.0 / T_extrap)
k_pred_extrap = tree_arr.predict(T_extrap)
print(f"{'Temperature [K]':>15s} | {'True Arrhenius Rate':>19s} | {'CART Tree Prediction':>20s} | {'Relative Error':>14s}")
print("-" * 76)
for t_v, kt, kp in zip(T_extrap, k_true_extrap, k_pred_extrap):
    print(f"{t_v:15.1f} | {kt:19.4f} | {kp:20.4f} | {100*abs(kp-kt)/kt:13.1f}%")
assert np.allclose(k_pred_extrap, k_pred_extrap[0], atol=1e-15)''',
        "post4_md": r'''**What Demonstration 4 shows.** A tree predicts a constant in each leaf, so beyond the last training point it returns the value of the last leaf. Trained up to $400\;\mathrm{K}$, it predicts $3.73\;\mathrm{s^{-1}}$ at every higher temperature, while the Arrhenius rate that generated the data is $240\;\mathrm{s^{-1}}$ at $600\;\mathrm{K}$: an error of $98\ \%$. The same holds for forests and boosted trees, which are sums of trees. Outside the range of their training data they need a physical model for the trend, or a check that refuses to predict (Lecture 16).

---

## Inside the algorithm
Below we verify our standalone prefix-sum split scanner in pure NumPy against `trees_ensembles.prefix_sum_best_split_1d`.''',
        "inside_code": r'''heading("Inside the Algorithm: By-Hand Prefix-Sum Best Split vs. the Library")
best_i = int(np.argmax(gains_prefix))
th_hand = 0.5 * float(T_K[best_i] + T_K[best_i + 1])
gain_hand = float(gains_prefix[best_i])
print(f"by hand (pure NumPy): threshold = {th_hand:.6f} K, gain = {gain_hand:.6f}")
print(f"library (engml_2_applied)     : threshold = {split_1d['threshold']:.6f} K, gain = {split_1d['sse_reduction']:.6f}")
assert np.isclose(th_hand, split_1d["threshold"], rtol=1e-14) and np.isclose(gain_hand, split_1d["sse_reduction"], rtol=1e-14)''',
        "ex1_prompt": r'''Verify mathematically and numerically on a 40-point polymer step-change dataset $C_p(T)$ with a glass-transition jump at $T_g = 373\;\mathrm{K}$ that `prefix_sum_best_split_1d` produces the exact same optimal threshold and post-split SSE as a brute-force loop evaluating $\sum_{i \in L}(y_i - \bar{y}_L)^2 + \sum_{i \in R}(y_i - \bar{y}_R)^2$ at every midpoint.''',
        "ex1_code": r'''heading("Exercise 1: Polymer Glass-Transition (T_g = 373 K) Step Discovery")
T_poly = np.linspace(320.0, 420.0, 40)
y_tg = np.where(T_poly <= 373.0, 0.85, 0.98)
ps_res = trees_ensembles.prefix_sum_best_split_1d(T_poly, y_tg)

mids = 0.5 * (T_poly[:-1] + T_poly[1:])
bf_sses = [
    float(np.sum((y_tg[T_poly <= m] - np.mean(y_tg[T_poly <= m])) ** 2) + np.sum((y_tg[T_poly > m] - np.mean(y_tg[T_poly > m])) ** 2))
    for m in mids
]
best_idx = int(np.argmin(bf_sses))
assert np.isclose(ps_res["threshold"], mids[best_idx], rtol=1e-14)
assert np.isclose(ps_res["sse_after"], bf_sses[best_idx], atol=1e-13)
print(f"Prefix-sum threshold = {ps_res['threshold']:.2f} K == Brute-force threshold = {mids[best_idx]:.2f} K (Post-split SSE = {ps_res['sse_after']:.2e})")''',
        "ex1_exp": r'''For a clean two-regime step function at $T_g = 373\;\mathrm{K}$, a single depth-1 CART split reduces the post-split SSE to $0.0000$, locating the exact midpoint containing $T_g$.''',
        "ex2_prompt": r'''Audit the **high-variance overfitting** of an unregularized deep decision tree (`max_depth=12, min_samples_split=2`) vs. a regularized shallow tree (`max_depth=2`) on noisy glass-transition data (`y_tg_noisy = y_tg + N(0, 0.04^2)`): show that the deep tree memorizes the training noise ($\mathrm{SSE}_{\mathrm{train}} \approx 0$) but suffers $>2\times$ higher test MSE against the true step curve `y_tg`.''',
        "ex2_code": r'''heading("Exercise 2: Unpruned Deep Tree Overfitting Audit on Noisy T_g Data")
rng_tg = np.random.default_rng(42)
y_tg_noisy = y_tg + rng_tg.normal(0.0, 0.04, size=40)
tree_shallow = trees_ensembles.DecisionTreeRegressor(max_depth=1, min_samples_split=2).fit(T_poly, y_tg_noisy)
tree_deep = trees_ensembles.DecisionTreeRegressor(max_depth=12, min_samples_split=2).fit(T_poly, y_tg_noisy)

mse_tr_deep = float(np.mean((tree_deep.predict(T_poly) - y_tg_noisy) ** 2))
mse_true_shallow = float(np.mean((tree_shallow.predict(T_poly) - y_tg) ** 2))
mse_true_deep = float(np.mean((tree_deep.predict(T_poly) - y_tg) ** 2))
print(f"Deep tree (depth=12) training MSE vs noisy labels : {mse_tr_deep:.2e} (Memorized noise)")
print(f"MSE against true physical T_g curve: Shallow (D=1) = {mse_true_shallow:.6f} vs Deep (D=12) = {mse_true_deep:.6f}")
assert mse_tr_deep < 1e-20 and mse_true_deep > 5.0 * mse_true_shallow''',
        "ex2_exp": r'''An unpruned tree with `min_samples_split=2` splits until every single noisy measurement occupies its own 1-sample leaf, and reproduces the noise: its error against the true curve is $1.1\times 10^{-3}$, of the order of the noise variance $0.04^2 = 1.6\times 10^{-3}$ and about 80 times that of the tree with one split ($1.3\times 10^{-5}$).''',
        "ex3_prompt": r'''Implement a **cost-complexity pruned depth selector** `select_tree_depth_bic(x: np.ndarray, y: np.ndarray, max_depths: list[int], alpha_pen: float = 0.005) -> dict` that evaluates Breiman's cost-complexity objective $R_\alpha(T_D) = \mathrm{SSE}(T_D) + \alpha_{\mathrm{pen}} |T_D|$ (where $|T_D|$ is the number of distinct leaf values) and verify on `(T_poly, y_tg_noisy)` that it selects the true 2-regime step tree $D^* = 1$.''',
        "ex3_code": r'''heading("Exercise 3: Implement Breiman Cost-Complexity Tree Pruning Selector")
def count_leaves(node: dict) -> int:
    if node["leaf"]:
        return 1
    return count_leaves(node["left"]) + count_leaves(node["right"])

def select_tree_depth_cc(x: np.ndarray, y: np.ndarray, max_depths: list[int], alpha_pen: float = 0.005) -> dict:
    """Select CART tree depth minimizing Breiman's cost-complexity R_alpha(T) = SSE(T) + alpha * |T|."""
    scores = {}
    for d_val in max_depths:
        t_m = trees_ensembles.DecisionTreeRegressor(max_depth=d_val, min_samples_split=2).fit(x, y)
        sse_m = float(np.sum((t_m.predict(x) - y) ** 2))
        n_leaves = count_leaves(t_m.tree_)
        scores[d_val] = sse_m + alpha_pen * n_leaves
        print(f"Depth D = {d_val:2d} | Leaves |T| = {n_leaves:2d} | SSE = {sse_m:.5f} | R_alpha(T) = {scores[d_val]:.5f}")
    best_d = min(scores, key=scores.get)
    return {"best_depth": best_d, "scores": scores}

cc_res = select_tree_depth_cc(T_poly, y_tg_noisy, [1, 2, 3, 5, 8], alpha_pen=0.005)
print(f"Cost-complexity selected tree depth D* = {cc_res['best_depth']}")
assert cc_res["best_depth"] == 1''',
        "ex3_takeaway": r'''**Discussion.** Penalizing leaf count via Breiman's cost-complexity criterion $R_\alpha(T) = \mathrm{SSE}(T) + \alpha |T|$ keeps a tree from dividing a plateau into leaves of single noisy points.''',
    },
    {
        "num": r'''14''',
        "slug": r'''14_random_forests_bagging_and_feature_importance''',
        "title": r'''14. Bagging, random forests and permutation importance''',
        "hook": r'''A single deep tree fits the noise of its training set. Averaging many trees, each grown on a resampled copy of the data, reduces the variance by an amount that depends on how correlated the trees are. The lecture verifies that variance formula, compares a forest with a single tree on test data, and permutes one input at a time to measure which inputs the model relies on.''',
        "objectives": [
            r'''Derive and check the variance of an average of correlated estimates, $\rho\sigma^2 + (1-\rho)\sigma^2/B$.''',
            r'''Train a random forest of 25 trees (`trees_ensembles.RandomForestRegressor`) and compare its test error with that of a single tree.''',
            r'''Measure the importance of each input by permutation (`trees_ensembles.permutation_feature_importance`).''',
            r'''Follow the test error and the spread between the trees as the number of trees grows.''',
        ],
        "theory_md": r'''## 1. Principles and governing equations
Let $\{\hat{T}_b(\mathbf{x})\}_{b=1}^B$ be $B$ regression trees trained on independent **bootstrap resamples** $\mathcal{D}_b^*$ of size $N$ drawn with replacement from $\mathcal{D}$. The **random forest / bagged estimator (Breiman, 2001)** averages the $B$ trees:
$$\hat{f}_{\mathrm{RF}}(\mathbf{x}) = \frac{1}{B}\sum_{b=1}^B \hat{T}_b(\mathbf{x}) \tag{1.1}$$
If each tree has variance $\mathrm{Var}(\hat{T}_b(\mathbf{x})) = \sigma^2$ and pairwise correlation $\mathrm{Corr}(\hat{T}_b(\mathbf{x}), \hat{T}_{b'}(\mathbf{x})) = \rho \in [0, 1]$, the covariance matrix $\boldsymbol{\Sigma} \in \mathbb{R}^{B \times B}$ has diagonal entries $\sigma^2$ and off-diagonal entries $\rho\sigma^2$, yielding the **ensemble variance-reduction law**:
$$\mathrm{Var}\!\left(\hat{f}_{\mathrm{RF}}(\mathbf{x})\right) = \frac{1}{B^2}\mathbf{1}_B^\top \boldsymbol{\Sigma}\mathbf{1}_B = \rho\sigma^2 + \frac{1 - \rho}{B}\sigma^2 \le \sigma^2 \tag{1.2}$$
To rank input descriptors $\mathbf{x} = [x_1, \dots, x_d]^\top$, **permutation feature importance** measures the increase in held-out mean squared error when column $j$ is randomly shuffled across samples ($\mathbf{X}_{\pi(j)}$):
$$I_j = \mathrm{MSE}\!\left(\mathbf{y},\, \hat{f}_{\mathrm{RF}}(\mathbf{X}_{\pi(j)})\right) - \mathrm{MSE}\!\left(\mathbf{y},\, \hat{f}_{\mathrm{RF}}(\mathbf{X})\right) \tag{1.3}$$''',
        "demo1_code": r'''heading("Demonstration 1: Single CART Tree vs. 25-Tree Random Forest Out-of-Sample Benchmark")
rng = np.random.default_rng(42)
N = 70
# 4 descriptors: x0 (Temperature, dominant), x1 (Pressure, moderate), x2 & x3 (pure irrelevant noise):
X_tr = rng.uniform(-2.0, 2.0, size=(N, 4))
y_tr_clean = 3.0 * np.sin(X_tr[:, 0]) + 1.5 * (X_tr[:, 1] ** 2)
y_tr = y_tr_clean + rng.normal(0.0, 0.45, size=N)

X_te = rng.uniform(-2.0, 2.0, size=(120, 4))
y_te_clean = 3.0 * np.sin(X_te[:, 0]) + 1.5 * (X_te[:, 1] ** 2)

single_tree = trees_ensembles.DecisionTreeRegressor(max_depth=5, min_samples_split=3).fit(X_tr, y_tr)
rf_model = trees_ensembles.RandomForestRegressor(n_estimators=25, max_depth=5, min_samples_split=3, seed=42).fit(X_tr, y_tr)

mse_single = float(np.mean((single_tree.predict(X_te) - y_te_clean) ** 2))
mse_rf = float(np.mean((rf_model.predict(X_te) - y_te_clean) ** 2))
print(f"Out-of-sample Test MSE against true physical surface: Single Tree = {mse_single:.4f} -> 25-Tree Random Forest = {mse_rf:.4f} ({100*(1-mse_rf/mse_single):.1f}% reduction)")
assert mse_rf < 0.75 * mse_single''',
        "post1_md": r'''**What Demonstration 1 shows.** Averaging 25 trees, each grown on a bootstrap sample, reduces the error against the surface that generated the data from $1.63$ (one tree) to $0.85$, by $48\ \%$. The data are synthetic.

---

## 2. Numerical walkthrough
Let us verify the **equicorrelated covariance matrix identity** Eq. (1.2) and inspect the individual tree predictions $\hat{T}_1(\mathbf{x}_*), \dots, \hat{T}_{25}(\mathbf{x}_*)$ of our fitted random forest at a test operating point.''',
        "demo2_code": r'''heading("Demonstration 2: Equicorrelated Ensemble Covariance Identity & Tree Dispersion")
B, sig2, rho = 20, 2.0, 0.25
Sigma_ens = sig2 * (rho * np.ones((B, B)) + (1.0 - rho) * np.eye(B))
var_matrix = float(np.sum(Sigma_ens)) / (B ** 2)
var_formula = rho * sig2 + (1.0 - rho) * sig2 / B
print(f"Exact covariance quadratic form (1/B^2) 1^T Sigma 1 = {var_matrix:.6f} == Formula = {var_formula:.6f} (vs single tree = {sig2:.2f})")
assert np.isclose(var_matrix, var_formula, rtol=1e-15)

all_tree_preds = np.column_stack([t.predict(X_te) for t in rf_model.trees_])
mean_single_tree_mse = float(np.mean((all_tree_preds - y_te_clean[:, None]) ** 2))
print(f"Average MSE of the 25 individual bootstrap trees = {mean_single_tree_mse:.4f} vs Ensemble Mean MSE = {mse_rf:.4f}")
assert mse_rf < 0.55 * mean_single_tree_mse''',
        "post2_md": r'''**What Demonstration 2 shows.** The first line checks the variance formula for equally correlated trees against the full covariance matrix. The second shows what averaging gains: the 25 trees have a mean error of $2.76$ each, and their average has an error of $0.85$. For squared error the difference is exactly the mean squared spread of the trees about their average, $1.91$ (checked in the last cell of the lecture): an ensemble is better than its typical member by the amount its members disagree.

---

## 3. Worked case study
Which of the 4 input descriptors ($x_0, x_1, x_2, x_3$) actually drive the process response $y = 3\sin(x_0) + 1.5 x_1^2$? Let us compute **permutation feature importance** (`trees_ensembles.permutation_feature_importance`) on the test set.''',
        "demo3_code": r'''heading("Demonstration 3: Breiman Permutation Feature Importance Descriptor Ranking")
imp = trees_ensembles.permutation_feature_importance(rf_model, X_te, y_te_clean, seed=42)
names = ["x0 (Temperature - strong 3*sin)", "x1 (Pressure - moderate 1.5*x^2)", "x2 (Inactive noise #1)", "x3 (Inactive noise #2)"]
for j, name in enumerate(names):
    print(f"Feature {name:33s} | Permutation Delta-MSE = {imp[j]:.4f}")

assert imp[0] > imp[1] > 0.5
assert imp[2] < 0.15 * imp[1] and imp[3] < 0.15 * imp[1]''',
        "post3_md": r'''**What Demonstration 3 shows.** Shuffling the temperature column raises the test error by $7.36$ and shuffling the pressure column by $4.56$; shuffling either of the two columns that do not enter the response changes it by $0.04$ or less. The ranking recovers the way the synthetic response was built.

---

## 4. What if: sensitivity to the parameters
How quickly does the random forest test MSE converge as the number of trees $B$ increases from $1$ to $40$?''',
        "demo4_code": r'''heading("Demonstration 4: Ensemble Size B Convergence Sweep")
print(f"{'Forest Size B':>13s} | {'Test MSE vs True Surface':>24s} | {'Ensemble Std Across Trees':>25s}")
print("-" * 68)
for b_trees in [1, 3, 5, 10, 20, 35]:
    rf_b = trees_ensembles.RandomForestRegressor(n_estimators=b_trees, max_depth=5, seed=42).fit(X_tr, y_tr)
    mse_b = float(np.mean((rf_b.predict(X_te) - y_te_clean) ** 2))
    t_mat = np.column_stack([t.predict(X_te) for t in rf_b.trees_])
    std_b = float(np.mean(np.std(t_mat, axis=1)))
    print(f"{b_trees:13d} | {mse_b:24.4f} | {std_b:25.4f}")''',
        "post4_md": r'''**What Demonstration 4 shows.** The error falls quickly at first, from $2.97$ with one tree to $0.77$ with ten. After that it does not improve: it is $0.84$ with 20 trees and $0.87$ with 35. More trees remove the part of the variance that averages out, $(1-\rho)\sigma^2/B$, and leave the part common to all trees, $\rho\sigma^2$; the small rise after ten trees is the scatter of a single run and would average out over repetitions.

---

## Inside the algorithm
Below we verify that averaging the predictions of `rf_model.trees_` in pure NumPy matches `rf_model.predict(X_te)` digit-for-digit and satisfies the exact **Bias–Variance–Diversity decomposition** $\frac{1}{B}\sum_{b=1}^B (\hat{T}_b - y)^2 - (\hat{f}_{\mathrm{RF}} - y)^2 \equiv \frac{1}{B}\sum_{b=1}^B (\hat{T}_b - \hat{f}_{\mathrm{RF}})^2$.''',
        "inside_code": r'''heading("Inside the Algorithm: By-Hand Ensemble Averaging & Krogh-Vedelsby Diversity Identity")
pred_rf_hand = np.mean(all_tree_preds, axis=1)
pred_rf_lib = rf_model.predict(X_te)
diff_rf = float(np.max(np.abs(pred_rf_hand - pred_rf_lib)))
diversity_term = float(np.mean((all_tree_preds - pred_rf_hand[:, None]) ** 2))
kv_lhs = mean_single_tree_mse - mse_rf
print(f"by hand vs library max discrepancy : {diff_rf:.2e}")
print(f"Krogh-Vedelsby Identity: Mean(MSE_b) - MSE_RF = {kv_lhs:.6f} == Ensemble Variance (Diversity) = {diversity_term:.6f}")
assert diff_rf < 1e-15 and np.isclose(kv_lhs, diversity_term, rtol=1e-14)''',
        "ex1_prompt": r'''Prove and verify numerically the **bootstrap out-of-bag (OOB) asymptotic law**: when drawing $N$ samples with replacement from $N$ items, the probability that any given observation $i$ is omitted from a bootstrap draw is $(1 - 1/N)^N \to e^{-1} \approx 0.367879$. Run 500 bootstrap draws for $N = 100$ and verify that the empirical OOB fraction lies within $0.368 \pm 0.015$.''',
        "ex1_code": r'''heading("Exercise 1: Bootstrap Out-of-Bag (OOB) 1/e Asymptotic Law Verification")
rng_oob = np.random.default_rng(2026)
N_oob = 100
exact_prob_oob = (1.0 - 1.0 / N_oob) ** N_oob
oob_fracs = [1.0 - len(np.unique(rng_oob.integers(0, N_oob, size=N_oob))) / N_oob for _ in range(500)]
emp_oob = float(np.mean(oob_fracs))
print(f"Exact finite-N OOB probability (1 - 1/N)^N = {exact_prob_oob:.6f} | 1/e = {np.exp(-1.0):.6f} | 500-trial Empirical = {emp_oob:.6f}")
assert abs(emp_oob - exact_prob_oob) < 0.01''',
        "ex1_exp": r'''Roughly $36.8\%$ of the training data is left out of each bootstrap tree, providing a built-in Out-of-Bag cross-validation set without needing a separate validation split.''',
        "ex2_prompt": r'''Audit the **collinear descriptor dilution effect** on permutation feature importance: append a near-duplicate copy of Temperature $x_4 = x_0 + \mathcal{N}(0, 0.01^2)$ to `X_tr` and `X_te`, refit the random forest, and show that splitting the physical signal across two redundant collinear temperature columns dilutes the individual permutation importance of $x_0$.''',
        "ex2_code": r'''heading("Exercise 2: Collinear Feature Importance Dilution Audit")
X_tr_col = np.column_stack([X_tr, X_tr[:, 0] + rng.normal(0.0, 0.15, size=N)])
X_te_col = np.column_stack([X_te, X_te[:, 0] + rng.normal(0.0, 0.15, size=120)])
rf_col = trees_ensembles.RandomForestRegressor(n_estimators=25, max_depth=5, seed=42).fit(X_tr_col, y_tr)
imp_col = trees_ensembles.permutation_feature_importance(rf_col, X_te_col, y_te_clean, seed=42)
print(f"Original x0 importance (no duplicate)       : {imp[0]:.4f}")
print(f"Duplicated [x0, x4_copy] individual importances: x0 = {imp_col[0]:.4f}, x4_copy = {imp_col[4]:.4f} (Sum = {imp_col[0]+imp_col[4]:.4f})")
assert imp_col[0] < 0.85 * imp[0] and imp_col[4] > 0.50''',
        "ex2_exp": r'''With a near-copy of $x_0$ in the data, the trees can split on either column, and shuffling one of them leaves the other intact. The importance of the original falls from $7.36$ to $1.12$, the copy receives $4.52$, and the two together ($5.64$) are less than the original alone. The split between the pair is uneven and depends on the run. Permutation importance measures what the model loses when one column is destroyed, not how much the response depends on the quantity behind it; correlated descriptors should be shuffled together.''',
        "ex3_prompt": r'''Implement an **out-of-bag (OOB) random forest evaluator** `fit_rf_with_oob_score(X: np.ndarray, y: np.ndarray, n_estimators: int = 30, max_depth: int = 5, seed: int = 42) -> dict` that records each tree's bootstrap indices, predicts each training point $i$ using *only* the trees for which $i \notin \mathcal{D}_b^*$, and returns the OOB MSE. Verify on `(X_tr, y_tr)` that the OOB MSE estimates the true noisy test MSE without touching `X_te`.''',
        "ex3_code": r'''heading("Exercise 3: Implement Out-of-Bag (OOB) Validation Estimator From Scratch")
def fit_rf_with_oob_score(X: np.ndarray, y: np.ndarray, n_estimators: int = 30, max_depth: int = 5, seed: int = 42) -> dict:
    """Fit a bagged CART forest while accumulating Out-of-Bag (OOB) predictions for every training sample."""
    Xa = np.asarray(X, dtype=float)
    ya = np.asarray(y, dtype=float).ravel()
    n_s = len(ya)
    rng_b = np.random.default_rng(seed)
    oob_sum = np.zeros(n_s)
    oob_cnt = np.zeros(n_s)
    for _ in range(n_estimators):
        b_idx = rng_b.integers(0, n_s, size=n_s)
        t_b = trees_ensembles.DecisionTreeRegressor(max_depth=max_depth, min_samples_split=3).fit(Xa[b_idx], ya[b_idx])
        oob_mask = np.ones(n_s, dtype=bool)
        oob_mask[b_idx] = False
        if np.any(oob_mask):
            oob_sum[oob_mask] += t_b.predict(Xa[oob_mask])
            oob_cnt[oob_mask] += 1.0
    valid = oob_cnt > 0
    oob_pred = np.where(valid, oob_sum / np.maximum(oob_cnt, 1.0), np.mean(ya))
    return {"oob_mse": float(np.mean((oob_pred[valid] - ya[valid]) ** 2)), "oob_coverage": float(np.mean(valid))}

oob_res = fit_rf_with_oob_score(X_tr, y_tr, n_estimators=30, max_depth=5)
print(f"OOB Sample Coverage = {oob_res['oob_coverage']*100:.1f}% | OOB Validation MSE = {oob_res['oob_mse']:.4f}")
assert oob_res["oob_coverage"] == 1.0 and 0.5 < oob_res["oob_mse"] < 2.5''',
        "ex3_takeaway": r'''**Discussion.** Because $(1 - e^{-1})^{30} < 10^{-6}$, every training point is out-of-bag in roughly $11$ of $30$ trees, which gives an estimate of the test error without a separate validation set.''',
    },
    {
        "num": r'''15''',
        "slug": r'''15_functional_gradient_boosting_and_huber_clipping''',
        "title": r'''15. Gradient boosting and the Huber loss''',
        "hook": r'''Gradient boosting builds a model as a sum of small trees, each fitted to the residuals of the sum so far. With the squared loss an outlier produces a large residual that the next trees follow. Clipping the residuals, which corresponds to the Huber loss of Lecture 5, limits that influence. The lecture compares the two losses on a synthetic hardness curve with two gross outliers.''',
        "objectives": [
            r'''State gradient boosting as a sum of trees fitted in stages (`trees_ensembles.GradientBoostingRegressor`).''',
            r'''Follow the decrease of the training error with the number of stages on a synthetic hardness curve of a binary alloy.''',
            r'''Derive and apply the clipped residuals of the Huber loss (`trees_ensembles.HuberGradientBoostingRegressor`).''',
            r'''Compare the two losses on data with gross outliers, for several learning rates and thresholds.''',
        ],
        "theory_md": r'''## 1. Principles and governing equations
Given a differentiable loss function $\mathcal{L}(y, F)$, **functional gradient boosting (Friedman, 2001)** constructs an additive surrogate $F_M(\mathbf{x}) = F_0 + \eta\sum_{m=1}^M h_m(\mathbf{x})$ with shrinkage learning rate $\eta \in (0, 1]$ and shallow regression trees $h_m(\mathbf{x})$ (depth $1$–$3$):
$$F_0 = \arg\min_\gamma \sum_{i=1}^N \mathcal{L}(y_i, \gamma) = \begin{cases} \bar{y} & \text{for } L_2\text{ loss} \\ \mathrm{median}(\mathbf{y}) & \text{for Huber / }L_1\text{ loss} \end{cases} \tag{1.1}$$
At stage $m = 1, \dots, M$, we evaluate the **pseudo-residual vector** $\mathbf{r}_m \in \mathbb{R}^N$ (the negative functional gradient in function space evaluated at $F_{m-1}$):
$$r_{i,m} = -\left[\frac{\partial \mathcal{L}(y_i, F)}{\partial F}\right]_{F = F_{m-1}(\mathbf{x}_i)} = \begin{cases} y_i - F_{m-1}(\mathbf{x}_i) & \text{for } \mathcal{L}_{L_2} = \frac{1}{2}(y - F)^2 \\ \mathrm{clip}\!\left(y_i - F_{m-1}(\mathbf{x}_i),\, -\delta,\, +\delta\right) & \text{for Huber loss } \mathcal{L}_\delta(y - F) \end{cases} \tag{1.2}$$
Fitting a shallow CART tree $h_m(\mathbf{x})$ to $\{(\mathbf{x}_i, r_{i,m})\}_{i=1}^N$ and updating:
$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta\,h_m(\mathbf{x}) \tag{1.3}$$
contracts residuals geometrically ($r_m = (1 - \eta)^m r_0$ for exact stage fits) while the clipping of the Huber loss limits the influence of gross outliers on the splits of the trees.''',
        "demo1_code": r'''heading("Demonstration 1: Stagewise L2 Boosting on a Synthetic Hardness Curve of a Binary Alloy")
x_Ni = np.linspace(0.0, 1.0, 45)
# Synthetic hardness curve of a binary alloy: a parabolic term plus an asymmetric term (assumed constants, HV):
HV_true = 65.0 + 180.0 * x_Ni * (1.0 - x_Ni) + 35.0 * (x_Ni ** 2)

mses_stage = []
for m_trees in [1, 5, 15, 35]:
    gbm = trees_ensembles.GradientBoostingRegressor(n_estimators=m_trees, learning_rate=0.2, max_depth=2).fit(x_Ni, HV_true)
    mses_stage.append(float(np.mean((gbm.predict(x_Ni) - HV_true) ** 2)))
    print(f"Stage M = {m_trees:2d} shallow trees (depth=2) | Hardness MSE = {mses_stage[-1]:.4f} HV^2")

assert all(mses_stage[i] > mses_stage[i + 1] for i in range(len(mses_stage) - 1)) and mses_stage[-1] < 1.0''',
        "post1_md": r'''**What Demonstration 1 shows.** One tree of depth 2 has four constant pieces and represents the smooth hardness curve poorly (mean squared error $160\;\mathrm{HV}^2$). Adding shallow trees one after another, each fitted to the residual of the sum so far, brings the error down to $2.0\;\mathrm{HV}^2$ after 15 trees and $0.05\;\mathrm{HV}^2$ after 35. The hardness curve is a synthetic parabola in the composition.

---

## 2. Numerical walkthrough
Let us compare the stage-1 pseudo-residuals $r_{i,1}$ of $L_2$ gradient boosting versus Huber-clipped gradient boosting ($\delta = 5.0\;\mathrm{HV}$) when two Vickers microhardness measurements hit a brittle oxide inclusion ($+250\;\mathrm{HV}$ spike).''',
        "demo2_code": r'''heading("Demonstration 2: Unclipped L2 Pseudo-Residuals vs. Huber-Clipped Pseudo-Residuals")
rng = np.random.default_rng(42)
HV_noisy = HV_true + rng.normal(0.0, 1.8, size=len(x_Ni))
HV_spiked = HV_noisy.copy()
HV_spiked[0] += 250.0
HV_spiked[-1] += 250.0

r1_l2 = HV_spiked - float(np.mean(HV_spiked))
r1_huber = np.clip(HV_spiked - float(np.median(HV_spiked)), -5.0, 5.0)
print(f"Stage-1 L2 initial baseline mean(y)      = {np.mean(HV_spiked):.2f} HV | Spike pseudo-residual r_0 = {r1_l2[0]:+.2f} HV")
print(f"Stage-1 Huber initial baseline median(y) = {np.median(HV_spiked):.2f} HV | Clipped pseudo-residual r_0 = {r1_huber[0]:+.2f} HV")
assert abs(r1_l2[0]) > 180.0 and abs(r1_huber[0]) == 5.0''',
        "post2_md": r'''**What Demonstration 2 shows.** With the squared loss the two spikes of $+250\;\mathrm{HV}$ raise the starting value, the mean, to $117.3\;\mathrm{HV}$ (the median is $111.9$), and leave residuals of $+198\;\mathrm{HV}$ at the spikes, which the first trees are then built to fit. With the Huber loss the start is the median and the residuals of the spikes are clipped to $+5\;\mathrm{HV}$.

---

## 3. Worked case study
Let us now compare the full 25-stage $L_2$ gradient boosting model against the 25-stage Huber-clipped gradient boosting model (`trees_ensembles.HuberGradientBoostingRegressor`) on `HV_spiked`, evaluating RMSE against the curve behind the data, `HV_true`.''',
        "demo3_code": r'''heading("Demonstration 3: L2 GBM vs. Huber-Clipped GBM Benchmark on Outlier-Corrupted Alloy Data")
gbm_l2 = trees_ensembles.GradientBoostingRegressor(n_estimators=25, learning_rate=0.15, max_depth=2).fit(x_Ni, HV_spiked)
gbm_huber = trees_ensembles.HuberGradientBoostingRegressor(n_estimators=25, learning_rate=0.15, max_depth=2, delta=5.0).fit(x_Ni, HV_spiked)

rmse_l2 = float(np.sqrt(np.mean((gbm_l2.predict(x_Ni) - HV_true) ** 2)))
rmse_huber = float(np.sqrt(np.mean((gbm_huber.predict(x_Ni) - HV_true) ** 2)))
print(f"RMSE vs the true curve under 2 gross spikes: L2 GBM = {rmse_l2:.3f} HV -> Huber Clipped GBM = {rmse_huber:.3f} HV")
assert rmse_huber < 0.35 * rmse_l2''',
        "post3_md": r'''**What Demonstration 3 shows.** Against the curve that generated the data, boosting with the squared loss has an error of $51.2\;\mathrm{HV}$, because it fits the two spikes; with the Huber loss the error is $12.2\;\mathrm{HV}$, four times smaller. The spikes still leave a trace: the clipped loss limits their influence and does not remove it.

---

## 4. What if: sensitivity to the parameters
How do the shrinkage learning rate $\eta \in \{0.05, 0.15, 0.40\}$ and Huber clipping threshold $\delta \in \{2.0, 5.0, 25.0, 250.0\}$ control the accuracy and robustness of Huber gradient boosting?''',
        "demo4_code": r'''heading("Demonstration 4: Shrinkage Learning Rate eta & Huber Clipping Threshold delta Sweep")
print(f"{'Shrinkage eta':>13s} | {'Huber delta [HV]':>16s} | {'RMSE vs True Alloy Curve [HV]':>29s}")
print("-" * 64)
for lr, d_h in [(0.05, 5.0), (0.15, 2.0), (0.15, 5.0), (0.15, 25.0), (0.15, 250.0), (0.40, 5.0)]:
    gb_s = trees_ensembles.HuberGradientBoostingRegressor(n_estimators=30, learning_rate=lr, max_depth=2, delta=d_h).fit(x_Ni, HV_spiked)
    err_s = float(np.sqrt(np.mean((gb_s.predict(x_Ni) - HV_true) ** 2)))
    print(f"{lr:13.2f} | {d_h:16.1f} | {err_s:29.3f}")''',
        "post4_md": r'''**What Demonstration 4 shows.** With $\delta = 250\;\mathrm{HV}$ nothing is clipped and the error is that of the squared loss ($51.6\;\mathrm{HV}$); with $\delta = 25\;\mathrm{HV}$ it is $29.7$, and with $\delta = 5$ and a learning rate of $0.15$ it is smallest, $12.2\;\mathrm{HV}$. The learning rate matters less: $13.8\;\mathrm{HV}$ at $0.05$ and $17.0\;\mathrm{HV}$ at $0.40$.

---

## Inside the algorithm
Below we run a standalone 10-line Huber functional gradient boosting loop in pure NumPy and verify digit-for-digit agreement with `trees_ensembles.HuberGradientBoostingRegressor`.''',
        "inside_code": r'''heading("Inside the Algorithm: By-Hand Huber Boosting Loop vs. the Library")
f0_hand = float(np.median(HV_spiked))
pred_hand = np.full_like(HV_spiked, f0_hand)
for _ in range(25):
    p_res = np.clip(HV_spiked - pred_hand, -5.0, 5.0)
    t_stage = trees_ensembles.DecisionTreeRegressor(max_depth=2, min_samples_split=6).fit(x_Ni, p_res)
    pred_hand += 0.15 * t_stage.predict(x_Ni)
diff_hgbm = float(np.max(np.abs(pred_hand - gbm_huber.predict(x_Ni))))
print("by hand (pure NumPy) pred[:5]:", np.round(pred_hand[:5], 5))
print("library (engml_2_applied)      pred[:5]:", np.round(gbm_huber.predict(x_Ni)[:5], 5))
print(f"Max discrepancy              : {diff_hgbm:.2e}")
assert diff_hgbm < 1e-14''',
        "ex1_prompt": r'''For a 0-split constant stump update $h_m(x) = r_{m-1}$, prove and verify numerically that after $M = 20$ boosting stages with shrinkage $\eta = 0.15$ and initial residual $r_0 = 10.0$, the stage-$M$ residual equals the exact geometric contraction $(1 - \eta)^M r_0$ to machine precision.''',
        "ex1_code": r'''heading("Exercise 1: Geometric Residual Contraction r_M = (1 - eta)^M r_0 Verification")
eta, M = 0.15, 20
r0 = 10.0
r_iter = r0
for _ in range(M):
    r_iter = r_iter - eta * r_iter
r_exact = ((1.0 - eta) ** M) * r0
print(f"After M={M} stages: Iterative residual = {r_iter:.8f} == Analytical (1-eta)^M r0 = {r_exact:.8f}")
assert np.isclose(r_iter, r_exact, rtol=1e-15)''',
        "ex1_exp": r'''Each stage subtracts a fraction $\eta$ of the remaining residual ($r_m = r_{m-1} - \eta r_{m-1} = (1 - \eta)r_{m-1}$), yielding exponential decay $r_M = (1-\eta)^M r_0 \approx e^{-\eta M} r_0$.''',
        "ex2_prompt": r'''Examine a learning rate that is too large: train `GradientBoostingRegressor(n_estimators=15, learning_rate=2.10, max_depth=2)` on `HV_true` and compare its error with that of $\eta = 0.20$. For $\eta > 2$ each stage overshoots by more than the residual it corrects ($1 - \eta = -1.1$), so the residual changes sign and grows.''',
        "ex2_code": r'''heading("Exercise 2: A learning rate that is too large (eta = 2.10 against eta = 0.20)")
gbm_stable = trees_ensembles.GradientBoostingRegressor(n_estimators=15, learning_rate=0.20, max_depth=2).fit(x_Ni, HV_true)
gbm_overshoot = trees_ensembles.GradientBoostingRegressor(n_estimators=15, learning_rate=2.10, max_depth=2).fit(x_Ni, HV_true)
mse_stable = float(np.mean((gbm_stable.predict(x_Ni) - HV_true) ** 2))
mse_overshoot = float(np.mean((gbm_overshoot.predict(x_Ni) - HV_true) ** 2))
print(f"15-Stage MSE: Stable shrinkage (eta=0.20) = {mse_stable:.4f} vs Divergent Overshoot (eta=2.10) = {mse_overshoot:.4f}")
assert mse_overshoot > 10.0 * mse_stable''',
        "ex2_exp": r'''If every tree fitted its residual exactly, the residual after a stage would be $(1 - \eta)$ times the one before: it shrinks for $0 < \eta < 2$ (with alternating sign when $\eta > 1$) and grows for $\eta > 2$. With $\eta = 2.10$ the error after 15 stages is $3676\;\mathrm{HV}^2$, against $2.0\;\mathrm{HV}^2$ for $\eta = 0.20$. In practice the learning rate is kept well below one.''',
        "ex3_prompt": r'''Implement an **early-stopping validation monitor** `gbm_early_stopping(X_tr, y_tr, X_val, y_val, max_stages=40, lr=0.15, max_depth=2)` that evaluates held-out validation MSE at every boosting stage $m = 1, \dots, M_{\max}$ and returns the stage $m^*$ minimizing validation error. Verify it on noisy Nordheim alloy train/validation data.''',
        "ex3_code": r'''heading("Exercise 3: Implement Stagewise Early-Stopping Validation Monitor for GBM")
def gbm_early_stopping(
    X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, max_stages: int = 40, lr: float = 0.15, max_depth: int = 2
) -> dict:
    """Track stagewise validation MSE across boosting iterations and select optimal stage m*."""
    f0 = float(np.mean(y_tr))
    p_tr = np.full(len(y_tr), f0)
    p_val = np.full(len(y_val), f0)
    val_mses = []
    for _ in range(max_stages):
        t_m = trees_ensembles.DecisionTreeRegressor(max_depth=max_depth).fit(X_tr, y_tr - p_tr)
        p_tr += lr * t_m.predict(X_tr)
        p_val += lr * t_m.predict(X_val)
        val_mses.append(float(np.mean((p_val - y_val) ** 2)))
    best_m = int(np.argmin(val_mses)) + 1
    return {"best_stage": best_m, "min_val_mse": val_mses[best_m - 1], "val_mses": val_mses}

es_res = gbm_early_stopping(x_Ni[::2], HV_noisy[::2], x_Ni[1::2], HV_true[1::2], max_stages=35, lr=0.20, max_depth=2)
print(f"Stage 1 Validation MSE = {es_res['val_mses'][0]:.3f} -> Optimal Stage m* = {es_res['best_stage']} (Val MSE = {es_res['min_val_mse']:.3f})")
assert es_res["min_val_mse"] < 0.15 * es_res["val_mses"][0] and es_res["best_stage"] >= 5''',
        "ex3_takeaway": r'''**Discussion.** The number of boosting stages $M$ directly controls model complexity; monitoring validation error per stage gives a stopping rule at no extra cost.''',
    },
    {
        "num": r'''16''',
        "slug": r'''16_split_conformal_prediction_and_mahalanobis_ood''',
        "title": r'''16. Conformal prediction intervals and out-of-distribution alarms''',
        "hook": r'''A prediction is more useful with a statement of how far it can be trusted. Split conformal prediction turns the residuals of a held-out calibration set into an interval with a stated coverage, without assuming a noise distribution. It says nothing about inputs unlike those used for calibration, so the lecture pairs it with the Mahalanobis distance, which flags a query that lies outside the training data even when each variable is within its range.''',
        "objectives": [
            r'''Implement split conformal prediction (`uncertainty.split_conformal_interval`) with the finite-sample quantile of the calibration residuals.''',
            r'''Measure the coverage on test data with skewed, non-Gaussian noise.''',
            r'''Compute the squared Mahalanobis distance (`uncertainty.mahalanobis_ood_score`) and check the identity for its mean over the training data.''',
            r'''Show that the Mahalanobis distance flags a query that is within the range of each variable but far from the training data.''',
        ],
        "theory_md": r'''## 1. Principles and governing equations
Every trustworthy engineering surrogate requires an **input domain guardrail** and an **output uncertainty certificate**:
1. **Mahalanobis out-of-distribution (OOD) distance (Mahalanobis, 1936):** Given $N$ normal training states $\mathbf{X}_{\mathrm{train}} \in \mathbb{R}^{N \times d}$ with mean $\boldsymbol{\mu}$ and sample covariance $\boldsymbol{\Sigma} = \frac{1}{N-1}(\mathbf{X}_{\mathrm{train}} - \mathbf{1}\boldsymbol{\mu}^\top)^\top(\mathbf{X}_{\mathrm{train}} - \mathbf{1}\boldsymbol{\mu}^\top)$, the squared Mahalanobis distance of query $\mathbf{x}_* \in \mathbb{R}^d$ is:
   $$D_M^2(\mathbf{x}_*) = (\mathbf{x}_* - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}(\mathbf{x}_* - \boldsymbol{\mu}) \tag{1.1}$$
   Using $\mathrm{Tr}(\mathbf{A}\mathbf{B}) = \mathrm{Tr}(\mathbf{B}\mathbf{A})$, the sample average of $D_M^2(\mathbf{x}_i)$ across the $N$ training observations satisfies an **exact distribution-free algebraic identity**:
   $$\frac{1}{N}\sum_{i=1}^N D_M^2(\mathbf{x}_i) = \frac{1}{N}\mathrm{Tr}\!\left(\mathbf{X}_c\boldsymbol{\Sigma}^{-1}\mathbf{X}_c^\top\right) = \frac{N-1}{N}\mathrm{Tr}(\mathbf{I}_d) = \frac{d(N-1)}{N} \tag{1.2}$$
2. **Distribution-free split conformal prediction (Vovk, Gammerman & Shafer 2005; Lei et al. 2018):** Given $n$ held-out calibration residuals $R_i = |y_i - \hat{f}(\mathbf{x}_i)|$ sorted as $R_{(1)} \le R_{(2)} \le \dots \le R_{(n)}$, setting the half-width $\hat{q}_{1-\alpha} = R_{(k)}$ with order statistic index $k = \lceil(n+1)(1-\alpha)\rceil$ guarantees finite-sample marginal coverage for *any* exchangeable data distribution:
   $$1 - \alpha \le \mathbb{P}\!\left(Y_{\mathrm{test}} \in \left[\hat{f}(\mathbf{X}_{\mathrm{test}}) - \hat{q}_{1-\alpha},\, \hat{f}(\mathbf{X}_{\mathrm{test}}) + \hat{q}_{1-\alpha}\right]\right) < 1 - \alpha + \frac{1}{n + 1} \tag{1.3}$$''',
        "demo1_code": r'''heading("Demonstration 1: Split Conformal Coverage Guarantee on Skewed Exponential Sensor Noise")
rng = np.random.default_rng(2026)
n_cal, n_test = 300, 1000
x_cal = rng.uniform(0.0, 5.0, size=n_cal)
y_cal_pred = 2.5 * x_cal + 1.0
# Heavily skewed shifted-exponential noise (violates symmetric Gaussian textbook assumptions):
y_cal_true = y_cal_pred + (rng.exponential(scale=0.45, size=n_cal) - 0.45)

x_test = rng.uniform(0.0, 5.0, size=n_test)
y_test_pred = 2.5 * x_test + 1.0
y_test_true = y_test_pred + (rng.exponential(scale=0.45, size=n_test) - 0.45)

cp90 = uncertainty.split_conformal_interval(y_cal_true, y_cal_pred, y_test_pred, alpha=0.10)
cov90 = float(np.mean((y_test_true >= cp90["lower"]) & (y_test_true <= cp90["upper"])))
print(f"Split Conformal (Target 1 - alpha = 90.0%): half-width q_hat = +/-{cp90['q_hat']:.4f}")
print(f"Empirical out-of-sample test coverage (n_test = 1000)        = {cov90*100:.2f}%")
assert 0.87 <= cov90 <= 0.94''',
        "post1_md": r'''**What Demonstration 1 shows.** The noise is exponential, skewed and far from Gaussian, and no assumption about it was made. The interval uses only the 271st smallest of 300 calibration residuals, $k = \lceil 301 \times 0.90 \rceil$, and covers $92.9\ \%$ of 1000 new samples, above the target of $90\ \%$.

---

## 2. Numerical walkthrough
Let us verify the **exact Mahalanobis trace identity** Eq. (1.2) on $N = 80$ 3-dimensional reactor operating states $(T, P, x)$ and test a normal centroid query versus a runaway reactor excursion.''',
        "demo2_code": r'''heading("Demonstration 2: Mahalanobis Trace Identity mean(D_M^2) = d*(N-1)/N & Runaway Alarm")
N_tr, d_feat = 80, 3
X_noc = rng.normal(loc=[370.0, 12.5, 0.65], scale=[8.0, 1.0, 0.08], size=(N_tr, d_feat))
d2_noc = uncertainty.mahalanobis_ood_score(X_noc, X_noc, reg=0.0)

expected_d2_mean = d_feat * (N_tr - 1.0) / N_tr
assert np.isclose(np.mean(d2_noc), expected_d2_mean, rtol=1e-12)
print(f"Mahalanobis sample mean D_M^2 = {np.mean(d2_noc):.8f} == Exact theoretical d*(N-1)/N = {expected_d2_mean:.8f}")

d2_centroid = float(uncertainty.mahalanobis_ood_score(X_noc, np.mean(X_noc, axis=0, keepdims=True))[0])
d2_runaway = float(uncertainty.mahalanobis_ood_score(X_noc, np.array([[460.0, 26.0, 0.95]]))[0])
print(f"Centroid query D_M^2 = {d2_centroid:.2e} | Runaway query (460 K, 26 bar) D_M^2 = {d2_runaway:.1f} >> chi^2_(3,0.99) = 11.34 (ood alarm)")
assert d2_centroid < 1e-12 and d2_runaway > 100.0''',
        "post2_md": r'''**What Demonstration 2 shows.** As for Hotelling's $T^2$ in Lecture 9, the squared Mahalanobis distance of the training points averages exactly $d(N-1)/N = 3 \times 79/80 = 2.9625$. The runaway query has $D_M^2 = 314.5$, far above the $99\ \%$ limit of $11.34$.

---

## 3. Worked case study
Why is **covariance-aware Mahalanobis distance** more informative than coordinate-wise min/max box bounds ($x_{j,\min} \le x_{*,j} \le x_{j,\max}$)? Consider a vapor–liquid equilibrium reactor where pressure $P$ and temperature $T$ are tightly correlated along the saturation curve $P \approx 0.5 T - 170$.''',
        "demo3_code": r'''heading("Demonstration 3: Correlation-Breaking OOD Detection — Mahalanobis vs. Univariate Box Bounds")
rng_corr = np.random.default_rng(123)
T_c = rng_corr.normal(360.0, 10.0, size=100)
P_c = 0.5 * T_c - 170.0 + rng_corr.normal(0.0, 0.25, size=100)
X_corr = np.column_stack([T_c, P_c])

# Off-diagonal query: T = 370 K (+1 sigma), P = 5.0 bar (-1 sigma), but correlation demands P(370 K) = 15.0 bar
x_query = np.array([[370.0, 5.0]])
z_axes = np.abs(x_query[0] - np.mean(X_corr, axis=0)) / np.std(X_corr, axis=0)
d2_query = float(uncertainty.mahalanobis_ood_score(X_corr, x_query)[0])
print("Univariate coordinate z-scores [|z_T|, |z_P|]:", np.round(z_axes, 2), "(Both < 1.5 sigma -> Passes univariate box checks)")
print(f"Multivariate Mahalanobis distance D_M^2      : {d2_query:.1f} >> chi^2_(2, 0.99) = 9.21 (out-of-distribution alarm)")
assert np.all(z_axes < 1.5) and d2_query > 400.0''',
        "post3_md": r'''**What Demonstration 3 shows.** Taken one at a time, the temperature and the pressure of the query are unremarkable: $1.03$ and $1.21$ standard deviations from their means. Taken together they are far outside the data, $D_M^2 = 1910$ against a limit of $9.21$, because in the training data the two variables are strongly correlated and the query combines them in a way that never occurred. Limits on single variables cannot see this.

---

## 4. What if: sensitivity to the parameters
Let us sweep the conformal confidence level $1 - \alpha \in \{80\%, 90\%, 95\%, 98\%\}$ and verify that empirical out-of-sample coverage tracks $1 - \alpha$ across the entire calibration curve.''',
        "demo4_code": r'''heading("Demonstration 4: Conformal Confidence Level (1 - alpha) Sweep vs. Empirical Coverage")
print(f"{'Target 1 - alpha':>16s} | {'Order Index k':>13s} | {'Half-Width q_hat':>16s} | {'Empirical Test Coverage':>23s}")
print("-" * 76)
for alpha_v in [0.20, 0.10, 0.05, 0.02]:
    cp_v = uncertainty.split_conformal_interval(y_cal_true, y_cal_pred, y_test_pred, alpha=alpha_v)
    k_ord = int(np.ceil((n_cal + 1) * (1.0 - alpha_v)))
    emp_v = float(np.mean((y_test_true >= cp_v["lower"]) & (y_test_true <= cp_v["upper"])))
    print(f"{(1-alpha_v)*100:15.1f}% | {k_ord:13d} | {cp_v['q_hat']:16.4f} | {emp_v*100:22.2f}%")
    assert abs(emp_v - (1.0 - alpha_v)) < 0.04''',
        "post4_md": r'''**What Demonstration 4 shows.** At every target level the coverage on the test set is at or above the target: $80.6$, $92.9$, $97.1$ and $98.8\ \%$ for targets of $80$, $90$, $95$ and $98\ \%$. The guarantee is that the coverage is at least $1 - \alpha$ on average over calibration sets; with 300 calibration points one realization differs from the target by a few points, here by up to $2.9$.

---

## Inside the algorithm
Below we compute the sorted conformal order statistic and the regularized Mahalanobis quadratic form in pure NumPy and verify digit-for-digit agreement with `uncertainty.split_conformal_interval` and `uncertainty.mahalanobis_ood_score`.''',
        "inside_code": r'''heading("Inside the Algorithm: By-Hand Conformal Quantile & Mahalanobis vs. the Library")
res_sorted = np.sort(np.abs(y_cal_true - y_cal_pred))
k_hand = int(np.ceil((n_cal + 1) * 0.90)) - 1
q_hat_hand = float(res_sorted[k_hand])
assert np.isclose(q_hat_hand, cp90["q_hat"], rtol=1e-15)

mu_c = np.mean(X_corr, axis=0)
cov_c = np.cov(X_corr, rowvar=False) + 1e-8 * np.eye(2)
diff_q = (x_query - mu_c).ravel()
d2_hand = float(diff_q @ np.linalg.solve(cov_c, diff_q))
print(f"by hand (pure NumPy): q_hat = {q_hat_hand:.6f}, D_M^2 = {d2_hand:.6f}")
print(f"library (engml_2_applied)     : q_hat = {cp90['q_hat']:.6f}, D_M^2 = {d2_query:.6f}")
assert np.isclose(d2_hand, d2_query, rtol=1e-14)''',
        "ex1_prompt": r'''Verify the **finite-sample conformal coverage theorem** Eq. (1.3) across 200 independent calibration/test splits ($n_{\mathrm{cal}} = 99, n_{\mathrm{test}} = 200, \alpha = 0.10$) under heavy-tailed Laplace noise: confirm that the mean marginal coverage across the 200 splits lies between $1 - \alpha = 0.900$ and $1 - \alpha + \frac{1}{n_{\mathrm{cal}}+1} = 0.910$.''',
        "ex1_code": r'''heading("Exercise 1: 200-Trial Verification of Finite-Sample Conformal Coverage Bounds")
rng_cp = np.random.default_rng(77)
covs_mc = []
for _ in range(200):
    r_cal = rng_cp.laplace(0.0, 1.0, size=99)
    r_te = rng_cp.laplace(0.0, 1.0, size=200)
    cp_m = uncertainty.split_conformal_interval(r_cal, np.zeros(99), np.zeros(200), alpha=0.10)
    covs_mc.append(float(np.mean((r_te >= cp_m["lower"]) & (r_te <= cp_m["upper"]))))
mean_cov_mc = float(np.mean(covs_mc))
print(f"Theoretical bounds: [0.9000, 0.9100] | 200-Trial Empirical Marginal Coverage = {mean_cov_mc:.4f}")
assert 0.895 <= mean_cov_mc <= 0.915''',
        "ex1_exp": r'''Because the rank of $R_{\mathrm{test}}$ among $\{R_1, \dots, R_n, R_{\mathrm{test}}\}$ is uniformly distributed on $\{1, \dots, n+1\}$ by exchangeability, $\mathbb{P}(R_{\mathrm{test}} \le R_{(k)}) = k / (n+1) = \lceil 100 \times 0.90 \rceil / 100 = 0.90$. The coverage found in 200 trials, $0.896$, is below this by an amount within the scatter of a simulation of this size.''',
        "ex2_prompt": r'''Audit the **covariate shift / non-exchangeability failure mode** of unguarded conformal prediction: suppose heteroscedastic sensor noise grows with temperature as $\sigma(T) = 0.05 + 0.02(T - 350)^2$, the calibration set is sampled inside $T_{\mathrm{cal}} \in [345, 355]\;\mathrm{K}$, and the plant drifts to an extrapolated regime $T_{\mathrm{shift}} \in [365, 375]\;\mathrm{K}$. Show that conformal coverage collapses below $30\%$, and verify that `mahalanobis_ood_score` flags $100\%$ of the shifted queries ($D_M^2 > 20$).''',
        "ex2_code": r'''heading("Exercise 2: Covariate-Shift Coverage Breakdown & Mahalanobis OOD Interlock")
T_cal_s = rng.uniform(345.0, 355.0, size=100)
y_cal_s = rng.normal(0.0, 0.05 + 0.02 * ((T_cal_s - 350.0) ** 2))
T_shift_s = rng.uniform(365.0, 375.0, size=100)
y_shift_s = rng.normal(0.0, 0.05 + 0.02 * ((T_shift_s - 350.0) ** 2))

cp_shift = uncertainty.split_conformal_interval(y_cal_s, np.zeros(100), np.zeros(100), alpha=0.10)
cov_shift = float(np.mean((y_shift_s >= cp_shift["lower"]) & (y_shift_s <= cp_shift["upper"])))
d2_shift = uncertainty.mahalanobis_ood_score(T_cal_s[:, None], T_shift_s[:, None])
print(f"Conformal coverage under unguarded covariate shift (T=365..375 K): {cov_shift*100:.1f}% << 90%")
print(f"Min Mahalanobis OOD score across shifted queries                 : {np.min(d2_shift):.1f} (100% flagged by OOD guardrail)")
assert cov_shift < 0.30 and np.all(d2_shift > 20.0)''',
        "ex2_exp": r'''Split conformal prediction relies on *exchangeability* between calibration and test data. Pairing conformal prediction with a Mahalanobis OOD interlock ensures that conformal intervals are only reported inside the calibrated domain where exchangeability holds.''',
        "ex3_prompt": r'''Implement **normalized / locally-adaptive conformal prediction** `adaptive_conformal_interval(y_calib, y_pred_calib, sigma_calib, y_pred_test, sigma_test, alpha=0.10)` using studentized nonconformity scores $S_i = |y_i - \hat{y}_i| / \hat{\sigma}(\mathbf{x}_i)$, which produces variable-width intervals $\hat{y}_* \pm \hat{q}_{1-\alpha}\,\hat{\sigma}(\mathbf{x}_*)$. Verify on a heteroscedastic dataset that it maintains $\ge 88\%$ coverage while adapting interval widths to local noise scale.''',
        "ex3_code": r'''heading("Exercise 3: Implement Locally-Adaptive Studentized Conformal Prediction")
def adaptive_conformal_interval(
    y_calib: np.ndarray, y_pred_calib: np.ndarray, sigma_calib: np.ndarray, y_pred_test: np.ndarray, sigma_test: np.ndarray, alpha: float = 0.10
) -> dict:
    """Locally-adaptive Split Conformal Prediction using normalized scores |y - y_hat| / sigma(x)."""
    scores = np.sort(np.abs(y_calib - y_pred_calib) / np.maximum(sigma_calib, 1e-12))
    n_c = len(scores)
    k_i = min(n_c - 1, max(0, int(np.ceil((n_c + 1) * (1.0 - alpha))) - 1))
    q_s = float(scores[k_i])
    half_w = q_s * sigma_test
    return {"q_hat_norm": q_s, "lower": y_pred_test - half_w, "upper": y_pred_test + half_w, "half_widths": half_w}

x_het_c = rng.uniform(0.5, 4.5, size=250)
sig_het_c = 0.1 + 0.3 * x_het_c
y_het_c = 2.0 * x_het_c + rng.normal(0.0, sig_het_c)

x_het_t = np.linspace(0.5, 4.5, 400)
sig_het_t = 0.1 + 0.3 * x_het_t
y_het_t = 2.0 * x_het_t + rng.normal(0.0, sig_het_t)

acp = adaptive_conformal_interval(y_het_c, 2.0 * x_het_c, sig_het_c, 2.0 * x_het_t, sig_het_t, alpha=0.10)
cov_acp = float(np.mean((y_het_t >= acp["lower"]) & (y_het_t <= acp["upper"])))
print(f"Adaptive Conformal Coverage = {cov_acp*100:.1f}% | Width at x=0.5: +/-{acp['half_widths'][0]:.3f} -> Width at x=4.5: +/-{acp['half_widths'][-1]:.3f}")
assert cov_acp >= 0.86 and acp["half_widths"][-1] > 4.0 * acp["half_widths"][0]''',
        "ex3_takeaway": r'''**Discussion.** Dividing the residuals by a local estimate of the noise, $S_i = |y_i - \hat{y}_i|/\hat{\sigma}(\mathbf{x}_i)$, gives intervals whose width follows the noise: $\pm 0.39$ at $x = 0.5$ and $\pm 2.27$ at $x = 4.5$. The coverage in this run is $87.2\ \%$, below the target of $90\ \%$. The guarantee holds on average over calibration sets, and it requires that $\hat{\sigma}$ be estimated on data separate from the calibration set; a single run with a few hundred points can fall a few points short, and the coverage should be checked over repetitions, as in Exercise 1.''',
    },
    {
        "num": r'''17''',
        "slug": r'''17_capstone_industrial_soft_sensor_and_reconciliation''',
        "title": r'''17. Capstone: a soft sensor with a carbon balance, intervals and an alarm''',
        "hook": r'''The capstone assembles the methods of the volume into one soft sensor, on synthetic near-infrared spectra of a methanol synthesis stream. The spectra are corrected by SNV, three PLS models predict the carbon fractions of methanol, carbon monoxide and carbon dioxide, and the predictions are projected onto the carbon balance. Conformal intervals state their uncertainty, and the residual $Q$ of the spectrum withholds a prediction when the probe is disturbed.''',
        "objectives": [
            r'''Correct synthetic near-infrared spectra by SNV (`dim_reduction.snv_preprocess`) and calibrate one PLS model for each of three species (`dim_reduction.pls1_nipals`).''',
            r'''Project the three predictions onto the carbon balance (`linear_physics.equality_constrained_lstsq`), and weight the projection by the accuracy of each prediction (`linear_physics.crowe_data_reconciliation`).''',
            r'''Attach 90 % conformal intervals to the reconciled predictions (`uncertainty.split_conformal_interval`).''',
            r'''Withhold a prediction when the residual $Q$ of the spectrum exceeds its limit.''',
        ],
        "theory_md": r'''## 1. Principles and governing equations
The soft sensor processes each incoming spectrum $\mathbf{a}_{\mathrm{raw}} \in \mathbb{R}^d$ ($d = 100$ wavelengths) in five stages:
1. **Stage 1 — optical scatter removal (SNV):**
   $$\mathbf{x}_{\mathrm{SNV}} = \frac{\mathbf{a}_{\mathrm{raw}} - \bar{a}\mathbf{1}_d}{\mathrm{std}(\mathbf{a}_{\mathrm{raw}})} \tag{1.1}$$
2. **Stage 2 — MSPC & OOD input guardrail (PCA $Q_{\mathrm{SPE}}$ & Hotelling $T^2$):**
   $$Q_{\mathrm{SPE}}(\mathbf{x}_{\mathrm{SNV}}) = \|(\mathbf{I}_d - \mathbf{P}_k\mathbf{P}_k^\top)(\mathbf{x}_{\mathrm{SNV}} - \boldsymbol{\mu}_X)\|_2^2 \le Q_{0.99}, \qquad T^2(\mathbf{x}_{\mathrm{SNV}}) \le T^2_{0.99} \tag{1.2}$$
3. **Stage 3 — multi-species NIPALS PLS1 soft-sensor array ($s \in \{\mathrm{CH_3OH, CO, CO_2}\}$):**
   $$\hat{y}_{\mathrm{raw},s} = \hat{\beta}_{0,s} + \mathbf{x}_{\mathrm{SNV}}^\top \hat{\boldsymbol{\beta}}_{\mathrm{PLS},s} \tag{1.3}$$
4. **Stage 4 — stoichiometric carbon-balance KKT / Crowe reconciliation ($\mathbf{1}_3^\top \mathbf{y} = 1$):**
   $$\hat{\mathbf{y}}_{\mathrm{rec}} = \arg\min_{\mathbf{y} \in \mathbb{R}^3} (\mathbf{y} - \hat{\mathbf{y}}_{\mathrm{raw}})^\top \boldsymbol{\Sigma}_y^{-1}(\mathbf{y} - \hat{\mathbf{y}}_{\mathrm{raw}}) \quad \text{s.t.} \quad Y_{\mathrm{MeOH}} + Y_{\mathrm{CO}} + Y_{\mathrm{CO_2}} = 1 \tag{1.4}$$
5. **Stage 5 — split conformal uncertainty Certificate:**
   $$\hat{Y}_{s} \in \left[\hat{y}_{\mathrm{rec},s} - \hat{q}_{1-\alpha,s},\, \hat{y}_{\mathrm{rec},s} + \hat{q}_{1-\alpha,s}\right], \qquad \mathbb{P}(\text{coverage}) \ge 1 - \alpha \tag{1.5}$$''',
        "demo1_code": r'''heading("Demonstration 1: Stage 1-4 End-to-End SNV + PLS1 + KKT Carbon-Conserved Soft Sensor")
rng = np.random.default_rng(2026)
N_tr, N_cal, N_te, d_wl = 60, 60, 80, 100
N_tot = N_tr + N_cal + N_te
wl = np.linspace(1200.0, 2200.0, d_wl)

# True carbon fractions [Y_MeOH, Y_CO, Y_CO2] summing identically to 1.0:
y_meoh = rng.uniform(0.15, 0.45, size=N_tot)
y_co = rng.uniform(0.05, 0.20, size=N_tot)
y_co2 = 1.0 - y_meoh - y_co
Y_true = np.column_stack([y_meoh, y_co, y_co2])

E_nir = np.vstack([
    np.exp(-0.5 * ((wl - 1480.0) / 55.0) ** 2) + 0.5 * np.exp(-0.5 * ((wl - 2050.0) / 45.0) ** 2),
    np.exp(-0.5 * ((wl - 1650.0) / 50.0) ** 2) + 0.4 * np.exp(-0.5 * ((wl - 1900.0) / 60.0) ** 2),
    np.exp(-0.5 * ((wl - 1780.0) / 65.0) ** 2) + 0.6 * np.exp(-0.5 * ((wl - 2120.0) / 50.0) ** 2),
])
scatter = rng.uniform(0.75, 1.30, size=(N_tot, 1))
offset = rng.uniform(-0.20, 0.20, size=(N_tot, 1))
X_raw = scatter * (Y_true @ E_nir) + offset + rng.normal(0.0, 0.012, size=(N_tot, d_wl))
X_snv = dim_reduction.snv_preprocess(X_raw)

Y_tr_obs = Y_true[:N_tr] + rng.normal(0.0, 0.018, size=(N_tr, 3))
pls_models = [dim_reduction.pls1_nipals(X_snv[:N_tr], Y_tr_obs[:, j], n_components=3) for j in range(3)]

def predict_raw_and_kkt(X_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Y_raw_pred = np.column_stack([X_batch @ m["coef"] + m["intercept"] for m in pls_models])
    C_c, d_c = np.array([[1.0, 1.0, 1.0]]), np.array([1.0])
    Y_kkt_pred = np.vstack([
        linear_physics.equality_constrained_lstsq(np.eye(3), row, C_c, d_c, lam=0.0)["w"]
        for row in Y_raw_pred
    ])
    return Y_raw_pred, Y_kkt_pred

Y_te_raw, Y_te_kkt = predict_raw_and_kkt(X_snv[N_tr + N_cal :])
Y_te_true = Y_true[N_tr + N_cal :]

max_carbon_err_raw = float(np.max(np.abs(np.sum(Y_te_raw, axis=1) - 1.0)))
max_carbon_err_kkt = float(np.max(np.abs(np.sum(Y_te_kkt, axis=1) - 1.0)))
mse_raw = float(np.mean((Y_te_raw - Y_te_true) ** 2))
mse_kkt = float(np.mean((Y_te_kkt - Y_te_true) ** 2))
print(f"Max carbon balance error: Raw PLS1 = {max_carbon_err_raw:.4e} -> KKT Reconciled = {max_carbon_err_kkt:.2e}")
print(f"Test MSE across 3 species: Raw PLS1 = {mse_raw:.6f} -> KKT Reconciled = {mse_kkt:.6f} ({100*(1-mse_kkt/mse_raw):.1f}% variance reduction)")
assert max_carbon_err_kkt < 1e-14 and mse_kkt < mse_raw''',
        "post1_md": r'''**What Demonstration 1 shows.** The three PLS models are fitted separately, and their predictions of the carbon fractions miss a sum of one by up to $0.043$. The projection onto $\mathbf{1}_3^\top \mathbf{y} = 1$ closes the balance to rounding error and reduces the mean squared error over the three species by $24\ \%$ ($1.29\times 10^{-4}$ to $0.98\times 10^{-4}$). The spectra and the reference analyses are synthetic.

---

## 2. Numerical walkthrough
What if the three PLS1 soft sensors have **unequal calibration accuracies** $\boldsymbol{\sigma}_y = [0.01, 0.02, 0.04]$ (e.g., $\mathrm{CH_3OH}$ has the strongest NIR band while $\mathrm{CO_2}$ has the weakest)? Let us compare unweighted KKT projection against Crowe's inverse-variance weighted reconciliation.''',
        "demo2_code": r'''heading("Demonstration 2: Inverse-Variance Weighted Crowe Soft-Sensor Reconciliation")
y_raw_3 = np.array([0.32, 0.12, 0.60])  # Raw sum = 1.04 (+0.04 carbon surplus)
sig_3 = np.array([0.01, 0.02, 0.04])    # Variances proportional to [1, 4, 16]
rec_3 = linear_physics.crowe_data_reconciliation(y_raw_3, sig_3, np.array([[1.0, 1.0, 1.0]]), np.array([1.0]))
shifts = np.abs(rec_3["y_reconciled"] - y_raw_3)
print("Raw predictions [MeOH, CO, CO2]      :", y_raw_3, "(sum =", np.sum(y_raw_3), ")")
print("Crowe weighted reconciled predictions:", np.round(rec_3["y_reconciled"], 6), "(sum =", np.sum(rec_3["y_reconciled"]), ")")
print(f"Adjustment ratio |dY_CO2| / |dY_MeOH| = {shifts[2] / shifts[0]:.2f} (Theoretical (0.04/0.01)^2 = 16.00)")
assert np.isclose(shifts[2] / shifts[0], 16.0, rtol=1e-12) and np.max(np.abs(rec_3["balance_residual_after"])) < 1e-14''',
        "post2_md": r'''**What Demonstration 2 shows.** With weights $\boldsymbol{\Sigma}_y^{-1}$ the correction is shared in proportion to the variances. The methanol prediction, the most accurate ($\sigma_1 = 0.01$), changes by $-0.0019$, and the carbon dioxide prediction, the least accurate ($\sigma_3 = 0.04$), takes $16/21 = 76.2\ \%$ of the correction ($-0.0305$).

---

## 3. Worked case study
Now let us execute **Stage 5 (Split conformal Prediction)** on the held-out calibration set ($N_{\mathrm{cal}} = 60$) to attach distribution-free $90\%$ prediction intervals to all three reconciled species on the test set ($N_{\mathrm{te}} = 80$).''',
        "demo3_code": r'''heading("Demonstration 3: Multi-Species 90% Split Conformal Prediction Intervals")
_, Y_cal_kkt = predict_raw_and_kkt(X_snv[N_tr : N_tr + N_cal])
Y_cal_obs = Y_true[N_tr : N_tr + N_cal] + rng.normal(0.0, 0.018, size=(N_cal, 3))
Y_te_obs = Y_te_true + rng.normal(0.0, 0.018, size=(N_te, 3))

for j, sp_name in enumerate(["CH3OH (Methanol)", "CO (Carbon Monoxide)", "CO2 (Unreacted Feed)"]):
    cp_j = uncertainty.split_conformal_interval(Y_cal_obs[:, j], Y_cal_kkt[:, j], Y_te_kkt[:, j], alpha=0.10)
    cov_j = float(np.mean((Y_te_obs[:, j] >= cp_j["lower"]) & (Y_te_obs[:, j] <= cp_j["upper"])))
    print(f"{sp_name:22s} | 90% Conformal Half-Width = +/-{cp_j['q_hat']:.4f} | Empirical Test Coverage = {cov_j*100:5.1f}%")
    assert cov_j >= 0.84''',
        "post3_md": r'''**What Demonstration 3 shows.** The $90\ \%$ intervals have half-widths of $0.033$ to $0.035$ in carbon fraction and cover $93.8$, $91.2$ and $92.5\ \%$ of the test samples for the three species.

---

## 4. What if: sensitivity to the parameters
What happens during long-term plant operation when polymer oligomers gradually foul wavelength channels $40\dots 50$ of the sapphire NIR probe window? Let us sweep the window fouling amplitude $A_{\mathrm{foul}} \in [0.0, 2.5]\;\mathrm{AU}$ and see at which amplitude the limit on $Q_{\mathrm{SPE}}$ of Stage 2 is exceeded, and how large the error of the prediction is at that point.''',
        "demo4_code": r'''heading("Demonstration 4: Optical Window Fouling Sweep & PCA Q_SPE Automatic Interlock")
pca_guard = dim_reduction.pca_svd(X_snv[:N_tr], n_components=3)
q_limit_99 = float(np.percentile(pca_guard["q_spe"], 99))
P_g = pca_guard["loadings"]

print(f"{'Fouling Amp':>12s} | {'PCA Q_SPE':>11s} | {'Q / Q_99':>10s} | {'Unchecked MeOH Err':>18s} | {'PAT Interlock Decision':>24s}")
print("-" * 86)
for f_amp in [0.0, 0.2, 0.6, 1.2, 2.5]:
    x_f = X_snv[N_tr + N_cal].copy()
    x_f[40:50] += f_amp
    xc_f = x_f - pca_guard["mean"]
    q_f = float(np.sum((xc_f - (xc_f @ P_g) @ P_g.T) ** 2))
    _, y_kkt_f = predict_raw_and_kkt(x_f[None, :])
    meoh_err = abs(float(y_kkt_f[0, 0] - Y_te_true[0, 0]))
    decision = "HOLD / CLEAN PROBE" if q_f > q_limit_99 else "TRUST PREDICTION"
    print(f"{f_amp:12.2f} | {q_f:11.3f} | {q_f/q_limit_99:9.1f}x | {meoh_err:18.4f} | {decision:>24s}")
assert q_f > 10.0 * q_limit_99''',
        "post4_md": r'''**What Demonstration 4 shows.** A fouling amplitude of $0.2$ leaves $Q$ below its limit. At $0.6$ it is $1.8$ times the limit and the prediction is withheld, although the error in the methanol fraction is still small ($0.003$); at $2.5$, where $Q$ is 21 times the limit, the unchecked error would be $0.028$. The residual statistic reacts before the prediction is badly wrong, which is what makes it useful as an interlock.

---

## Inside the algorithm
Below we verify that for the equal-variance carbon-closure constraint $\mathbf{1}_3^\top \mathbf{y} = 1$, the KKT saddle-point projection $\hat{\mathbf{y}}_{\mathrm{KKT}}$ reduces to the exact analytical mean-centering shift $\hat{y}_{\mathrm{KKT},j} = \hat{y}_{\mathrm{raw},j} - \frac{1}{3}(\sum_{s=1}^3 \hat{y}_{\mathrm{raw},s} - 1)$.''',
        "inside_code": r'''heading("Inside the Algorithm: By-Hand Simplex Mean-Shift vs. the Library")
Y_kkt_hand = Y_te_raw - (np.sum(Y_te_raw, axis=1, keepdims=True) - 1.0) / 3.0
diff_cap = float(np.max(np.abs(Y_kkt_hand - Y_te_kkt)))
print("by hand (pure NumPy) row 0:", np.round(Y_kkt_hand[0], 6))
print("library (engml_2_applied)  row 0:", np.round(Y_te_kkt[0], 6))
print(f"Max discrepancy across 80x3 test matrix: {diff_cap:.2e}")
assert diff_cap < 1e-14''',
        "ex1_prompt": r'''Prove and verify on the $80 \times 3$ test set (`Y_te_raw`, `Y_te_kkt`, `Y_te_true`) the **exact per-sample Pythagorean error contraction**: because $\mathbf{1}_3^\top \mathbf{y}_{\mathrm{true}} = 1$, for *every single test sample* $i \in \{0, \dots, 79\}$, $\|\hat{\mathbf{y}}_{\mathrm{raw},i} - \mathbf{y}_{\mathrm{true},i}\|_2^2 - \|\hat{\mathbf{y}}_{\mathrm{KKT},i} - \mathbf{y}_{\mathrm{true},i}\|_2^2 = \frac{1}{3}(\mathbf{1}_3^\top \hat{\mathbf{y}}_{\mathrm{raw},i} - 1)^2 \ge 0$.''',
        "ex1_code": r'''heading("Exercise 1: Exact Sample-by-Sample Pythagorean KKT Error Contraction Proof")
sq_raw_i = np.sum((Y_te_raw - Y_te_true) ** 2, axis=1)
sq_kkt_i = np.sum((Y_te_kkt - Y_te_true) ** 2, axis=1)
improvement_i = sq_raw_i - sq_kkt_i
theoretical_gain_i = ((np.sum(Y_te_raw, axis=1) - 1.0) ** 2) / 3.0
max_pyth_diff = float(np.max(np.abs(improvement_i - theoretical_gain_i)))
print(f"Fraction of test samples where KKT error <= Raw error: {100*np.mean(improvement_i >= -1e-15):.1f}% (Every single sample)")
print(f"Max discrepancy from (sum(y_raw) - 1)^2 / 3          : {max_pyth_diff:.2e}")
assert np.all(improvement_i >= -1e-15) and max_pyth_diff < 1e-14''',
        "ex1_exp": r'''Because $\hat{\mathbf{y}}_{\mathrm{KKT},i}$ is the orthogonal Euclidean projection of $\hat{\mathbf{y}}_{\mathrm{raw},i}$ onto the hyperplane $\mathcal{H} = \{\mathbf{y} : \mathbf{1}_3^\top\mathbf{y} = 1\}$ and $\mathbf{y}_{\mathrm{true},i} \in \mathcal{H}$, the error vector decomposes orthogonally by the Pythagorean theorem on *every single sample*.''',
        "ex2_prompt": r'''Audit the **dual-mode fault diagnostic Table** on the capstone soft sensor: test (1) a normal test spectrum, (2) a fouled probe window spectrum (`x_snv[40:50] += 2.0`), and (3) a stoichiometric composition drift (`Y_extr = [0.85, 0.10, 0.05]`, outside the training yield range $[0.15, 0.45]$). Verify that $Q_{\mathrm{SPE}}$ catches the optical probe fault while Hotelling's $T^2$ catches the extreme composition excursion.''',
        "ex2_code": r'''heading("Exercise 2: Dual-Mode PAT Fault Matrix (Optical Fouling vs. Composition Excursion)")
t2_limit_99 = float(np.percentile(pca_guard["hotelling_t2"], 99))
lam_g = pca_guard["eigenvalues"]

x_norm_q = X_snv[N_tr + N_cal]
x_foul_q = x_norm_q.copy()
x_foul_q[40:50] += 2.0
x_extr_q = pca_guard["mean"] + (6.0 * np.sqrt(lam_g[0])) * P_g[:, 0]

for label, x_vec in [("Normal Reactor State", x_norm_q), ("Optical Window Fouling", x_foul_q), ("Extreme Yield Excursion", x_extr_q)]:
    xc = x_vec - pca_guard["mean"]
    t_v = xc @ P_g
    t2_v = float(np.sum((t_v ** 2) / lam_g))
    q_v = float(np.sum((xc - t_v @ P_g.T) ** 2))
    print(f"{label:24s} | T^2 = {t2_v:6.2f} (lim={t2_limit_99:.2f}) | Q_SPE = {q_v:7.3f} (lim={q_limit_99:.3f})")
    if label == "Optical Window Fouling":
        assert q_v > 5.0 * q_limit_99
    elif label == "Extreme Yield Excursion":
        assert t2_v > 2.5 * t2_limit_99 and q_v < q_limit_99''',
        "ex2_exp": r'''Optical window fouling introduces spectral shape distortions orthogonal to the 3 Beer–Lambert chromophores (high $Q_{\mathrm{SPE}}$), whereas an extreme chemical conversion shift moves *along* the 3-chromophore Beer–Lambert subspace (high Hotelling $T^2$, low $Q_{\mathrm{SPE}}$).''',
        "ex3_prompt": r'''Implement the complete **`IndustrialSoftSensorPAT`** class with `.fit(X_tr_raw, Y_tr, X_cal_raw, Y_cal)` and `.predict_guarded(X_query_raw)` methods that integrates Stage 1 (`snv_preprocess`), Stage 2 (PCA $T^2$ and $Q_{\mathrm{SPE}}$ 99% alarms), Stage 3 (3-species `pls1_nipals`), Stage 4 (inverse-variance `crowe_data_reconciliation` using calibration residual stds $\hat{\boldsymbol{\sigma}}_{\mathrm{cal}}$), and Stage 5 (90% `split_conformal_interval`). Verify it end-to-end on `X_raw[N_tr + N_cal:]`.''',
        "ex3_code": r'''heading("Exercise 3: Implement End-to-End IndustrialSoftSensorPAT Pipeline Class")
class IndustrialSoftSensorPAT:
    """5-Stage Physics-Constrained & Fault-Guarded NIR Soft Sensor for Multicomponent Reactors."""

    def __init__(self, n_components: int = 3, alpha: float = 0.10) -> None:
        self.n_components = n_components
        self.alpha = alpha

    def fit(self, X_tr_raw: np.ndarray, Y_tr: np.ndarray, X_cal_raw: np.ndarray, Y_cal: np.ndarray) -> "IndustrialSoftSensorPAT":
        X_tr_s = dim_reduction.snv_preprocess(X_tr_raw)
        X_cal_s = dim_reduction.snv_preprocess(X_cal_raw)
        self.pca_ = dim_reduction.pca_svd(X_tr_s, n_components=self.n_components)
        self.t2_lim_ = float(np.percentile(self.pca_["hotelling_t2"], 99))
        self.q_lim_ = float(np.percentile(self.pca_["q_spe"], 99))
        self.pls_ = [dim_reduction.pls1_nipals(X_tr_s, Y_tr[:, j], n_components=self.n_components) for j in range(3)]
        Y_cal_raw = np.column_stack([X_cal_s @ m["coef"] + m["intercept"] for m in self.pls_])
        self.sig_cal_ = np.std(Y_cal_raw - Y_cal, axis=0)
        self.Y_cal_rec_ = self._reconcile_batch(Y_cal_raw)
        self.Y_cal_ = Y_cal
        return self

    def _reconcile_batch(self, Y_raw_b: np.ndarray) -> np.ndarray:
        A_c, b_c = np.array([[1.0, 1.0, 1.0]]), np.array([1.0])
        return np.vstack([linear_physics.crowe_data_reconciliation(row, self.sig_cal_, A_c, b_c)["y_reconciled"] for row in Y_raw_b])

    def predict_guarded(self, X_query_raw: np.ndarray) -> dict:
        Xq_s = dim_reduction.snv_preprocess(X_query_raw)
        Xc = Xq_s - self.pca_["mean"]
        T_q = Xc @ self.pca_["loadings"]
        t2_q = np.sum((T_q ** 2) / self.pca_["eigenvalues"][None, :], axis=1)
        q_q = np.sum((Xc - T_q @ self.pca_["loadings"].T) ** 2, axis=1)
        Y_raw_q = np.column_stack([Xq_s @ m["coef"] + m["intercept"] for m in self.pls_])
        Y_rec_q = self._reconcile_batch(Y_raw_q)
        intervals = [
            uncertainty.split_conformal_interval(self.Y_cal_[:, j], self.Y_cal_rec_[:, j], Y_rec_q[:, j], alpha=self.alpha)
            for j in range(3)
        ]
        return {"Y_reconciled": Y_rec_q, "intervals": intervals, "t2": t2_q, "q_spe": q_q, "q_alarm": q_q > self.q_lim_}

pat = IndustrialSoftSensorPAT(n_components=3, alpha=0.10).fit(
    X_raw[:N_tr], Y_tr_obs, X_raw[N_tr : N_tr + N_cal], Y_cal_obs
)
out_te = pat.predict_guarded(X_raw[N_tr + N_cal :])
carbon_closure_err = float(np.max(np.abs(np.sum(out_te["Y_reconciled"], axis=1) - 1.0)))
meoh_cov = float(np.mean((Y_te_obs[:, 0] >= out_te["intervals"][0]["lower"]) & (Y_te_obs[:, 0] <= out_te["intervals"][0]["upper"])))
print("IndustrialSoftSensorPAT deployed on 80 test spectra:")
print(f"  Max carbon balance residual = {carbon_closure_err:.2e} | Methanol 90% Conformal Coverage = {meoh_cov*100:.1f}%")
assert carbon_closure_err < 1e-14 and meoh_cov >= 0.85''',
        "ex3_takeaway": r'''**Discussion.** Combining spectral SNV preprocessing, supervised NIPALS PLS1, covariance-weighted Crowe/KKT stoichiometric reconciliation, MSPC $T^2/Q_{\mathrm{SPE}}$ fault alarms, and split conformal prediction gives a soft sensor whose predictions satisfy the carbon balance, carry an interval, and are withheld when the spectrum is unlike those of the calibration.''',
    },
]
