"""Text and code of lectures 00 to 05, as plain data for scripts/build_notebooks.py."""

LECTURES_00_05 = [
    {
        "num": r'''00''',
        "slug": r'''00_elemental_mass_and_charge_conservation_matrices''',
        "title": r'''0. Atom, mass and charge balances as linear constraints''',
        "hook": r'''Flow meters and analysers are noisy, so a set of measured flows around a reactor rarely satisfies the atom balances exactly, and a model fitted to such data inherits the error. This lecture writes the balances of the elements, of total mass and of charge as one linear system, $\mathbf{C}\mathbf{w} = \mathbf{d}$. The null space of $\mathbf{C}$ then gives the independent reactions, and every composition that satisfies the balances is the feed plus a combination of them.''',
        "objectives": [
            r'''Assemble the matrix of atom counts and charges for a gas-phase synthesis and for an aqueous electrolyte.''',
            r'''Show, and check numerically, that the balance of total mass follows from the balances of the elements.''',
            r'''Compute the null space of the balance matrix by SVD and write every admissible outlet composition in terms of extents of reaction.''',
            r'''Measure by how much noisy measurements violate the balances of atoms and mass.''',
        ],
        "theory_md": r'''## 1. Principles and governing equations
In any non-nuclear chemical process involving $S$ chemical species and $E$ conserved elements (e.g., $\mathrm{C, H, O, N}$) plus ionic charge $z_j$, every physically admissible molar flow or yield vector $\mathbf{w} \in \mathbb{R}^S$ ($\mathrm{mol/s}$) must satisfy the exact linear conservation system:
$$\mathbf{C}\mathbf{w} = \mathbf{d}, \qquad \mathbf{C} \in \mathbb{R}^{E \times S}, \quad \mathbf{d} = \mathbf{C}\mathbf{w}_{\mathrm{feed}} \in \mathbb{R}^E \tag{1.1}$$
where $C_{e,j}$ is the number of atoms of element $e$ (or ionic charge $z_j$) in one molecule of species $j$, and $d_e$ is the conserved elemental feed rate ($\mathrm{mol/s}$). Because the molar mass vector $\mathbf{m} \in \mathbb{R}^S$ ($\mathrm{g/mol}$) is a linear combination of the rows of $\mathbf{C}$ weighted by the atomic weights $\mathbf{a} \in \mathbb{R}^E$ ($\mathbf{m}^\top = \mathbf{a}^\top \mathbf{C}$), total mass closure is automatically guaranteed whenever Eq. (1.1) holds:
$$\mathbf{m}^\top \mathbf{w} = \mathbf{a}^\top (\mathbf{C}\mathbf{w}) = \mathbf{a}^\top \mathbf{d} = m_{\mathrm{feed}} \tag{1.2}$$
By the **rank-nullity theorem**, the set of all conserved reactor states forms an affine manifold of dimension $R = S - \mathrm{rank}(\mathbf{C})$ parameterized by the independent reaction extent vector $\boldsymbol{\xi} \in \mathbb{R}^R$:
$$\mathbf{w}(\boldsymbol{\xi}) = \mathbf{w}_{\mathrm{feed}} + \boldsymbol{\nu}^\top \boldsymbol{\xi}, \qquad \mathbf{C}\boldsymbol{\nu}^\top = \mathbf{0}_{E \times R} \tag{1.3}$$
where the $R$ rows of the stoichiometric matrix $\boldsymbol{\nu} \in \mathbb{R}^{R \times S}$ span the right nullspace $\mathcal{N}(\mathbf{C})$.''',
        "demo1_code": r'''heading("Demonstration 1: Atomic Formula Matrix & Mass-Balance Linear Dependence")
# Species order: [CH3OH, CO, H2O, CO2, H2] for methanol synthesis (Feed: 1 mol CO2 + 3 mol H2)
species = ["CH3OH", "CO", "H2O", "CO2", "H2"]
C_atom = np.array([
    [1.0, 1.0, 0.0, 1.0, 0.0],  # Carbon (C) atoms per molecule
    [4.0, 0.0, 2.0, 0.0, 2.0],  # Hydrogen (H) atoms per molecule
    [1.0, 1.0, 1.0, 2.0, 0.0],  # Oxygen (O) atoms per molecule
])
atomic_masses = np.array([12.011, 1.008, 15.999])  # [C, H, O] in g/mol
molar_masses = atomic_masses @ C_atom
w_feed = np.array([0.0, 0.0, 0.0, 1.0, 3.0])
d_atom = C_atom @ w_feed
feed_mass = float(molar_masses @ w_feed)

print(f"{'Species':>8s} | {'C':>3s} {'H':>3s} {'O':>3s} | {'Molar Mass [g/mol]':>18s} | {'Feed [mol]':>10s}")
print("-" * 54)
for sp, col, mw, wf in zip(species, C_atom.T, molar_masses, w_feed):
    print(f"{sp:>8s} | {int(col[0]):3d} {int(col[1]):3d} {int(col[2]):3d} | {mw:18.3f} | {wf:10.3f}")

C_augmented_mass = np.vstack([C_atom, molar_masses])
assert np.linalg.matrix_rank(C_atom) == 3
assert np.linalg.matrix_rank(C_augmented_mass) == 3, "Mass row m^T = a^T C must not increase rank!"
print(f"\nElemental feed d = {d_atom} mol | Total feed mass = {feed_mass:.3f} g | rank(C) = {np.linalg.matrix_rank(C_atom)}")''',
        "post1_md": r'''**What Demonstration 1 shows.** Every species molar mass is an exact linear combination of the $\mathrm{C, H, O}$ rows ($\mathbf{m}^\top = \mathbf{a}^\top \mathbf{C}$), so appending a total mass-balance row leaves $\mathrm{rank}(\mathbf{C}) = 3$ unchanged.

---

## 2. Numerical walkthrough
To construct the stoichiometric reaction subspace for the $S = 5$ species and $\mathrm{rank}(\mathbf{C}) = 3$ elemental invariants, we note that $R = S - \mathrm{rank}(\mathbf{C}) = 5 - 3 = 2$ independent reactions exist:
1. **Methanol synthesis ($\xi_1$):** $\mathrm{CO_2 + 3\,H_2 \rightleftharpoons CH_3OH + H_2O}$, $\boldsymbol{\nu}_1 = [+1, 0, +1, -1, -3]$.
2. **Reverse water–gas shift ($\xi_2$):** $\mathrm{CO_2 + H_2 \rightleftharpoons CO + H_2O}$, $\boldsymbol{\nu}_2 = [0, +1, +1, -1, -1]$.

We verify below that $\mathbf{C}\boldsymbol{\nu}^\top = \mathbf{0}$ and that the right singular vectors of $\mathbf{C}$ corresponding to zero singular values span the exact same 2-D reaction subspace.''',
        "demo2_code": r'''heading("Demonstration 2: Stoichiometric Reaction Matrix & SVD Nullspace Projection")
nu = np.array([
    [1.0, 0.0, 1.0, -1.0, -3.0],  # Rxn 1: CO2 + 3 H2 <-> CH3OH + H2O
    [0.0, 1.0, 1.0, -1.0, -1.0],  # Rxn 2: CO2 + H2   <-> CO    + H2O
])
U_c, s_c, Vt_c = np.linalg.svd(C_atom, full_matrices=True)
N_svd = Vt_c[3:, :].T  # Orthonormal basis for the 2-D nullspace N(C)
P_null = N_svd @ N_svd.T

# Project physical reaction vectors nu onto the SVD nullspace basis:
nu_projected = (P_null @ nu.T).T
print("Singular values of C_atom:", np.round(s_c, 6))
print("Stoichiometric nullspace residual ||C_atom @ nu^T||_max:", float(np.max(np.abs(C_atom @ nu.T))))
print("Subspace projection discrepancy ||nu - P_null nu||_max  :", float(np.max(np.abs(nu - nu_projected))))
assert np.allclose(C_atom @ nu.T, 0.0, atol=1e-15)
assert np.allclose(nu, nu_projected, atol=1e-14)''',
        "post2_md": r'''**What Demonstration 2 shows.** Both chemical reactions lie identically in the 2-D nullspace $\mathcal{N}(\mathbf{C})$ spanned by the trailing singular vectors of $\mathbf{C}_{\mathrm{atom}}$.

---

## 3. Worked case study
Let us compare true stoichiometric reactor effluents $\mathbf{w}(\boldsymbol{\xi}) = \mathbf{w}_{\mathrm{feed}} + \boldsymbol{\nu}^\top \boldsymbol{\xi}$ against raw, unconstrained gas-chromatograph (GC) measurements $\mathbf{w}_{\mathrm{GC}} = \mathbf{w}_{\mathrm{true}} + \boldsymbol{\varepsilon}$ at three operating points.''',
        "demo3_code": r'''heading("Demonstration 3: Stoichiometric Reactor Benchmark vs. Raw Unconstrained GC Sensor")
xi_grid = np.array([[0.24, 0.08], [0.35, 0.12], [0.10, 0.20]])
for idx, xi in enumerate(xi_grid, start=1):
    w_state = w_feed + nu.T @ xi
    atom_res = float(np.max(np.abs(C_atom @ w_state - d_atom)))
    mass_now = float(molar_masses @ w_state)
    print(f"Operating Point {idx} (xi={xi}) | Effluent w={np.round(w_state, 3)} | Mass={mass_now:.3f} g | Atom err={atom_res:.1e}")
    assert np.allclose(C_atom @ w_state, d_atom, atol=1e-14)
    assert np.isclose(mass_now, feed_mass, rtol=1e-14)

rng = np.random.default_rng(42)
w_true = w_feed + nu.T @ np.array([0.24, 0.08])
w_noisy_gc = w_true + rng.normal(0.0, 0.05, size=5)
atom_violation = C_atom @ w_noisy_gc - d_atom
mass_violation = float(molar_masses @ w_noisy_gc) - feed_mass
print(f"\nNoisy raw GC elemental imbalance [dC, dH, dO]: {np.round(atom_violation, 4)} mol")
print(f"Noisy raw GC phantom mass created            : {mass_violation:+.4f} g (out of {feed_mass:.3f} g)")
assert abs(mass_violation) > 0.1''',
        "post3_md": r'''**What Demonstration 3 shows.** The three operating points computed from reaction extents conserve every element to rounding error. A synthetic analysis with Gaussian noise of $0.05\;\mathrm{mol}$ on each species does not: it is out of balance by $0.01$, $-0.06$ and $0.09\;\mathrm{mol}$ in carbon, hydrogen and oxygen, and reports $1.58\;\mathrm{g}$ more mass than the $50.057\;\mathrm{g}$ that were fed.

---

## 4. What if: sensitivity to the parameters
What happens to the worst-case phantom mass violation $|\mathbf{m}^\top \mathbf{w}_{\mathrm{GC}} - m_{\mathrm{feed}}|$ as GC analyzer noise $\sigma_{\mathrm{GC}}$ increases from $0.001\;\mathrm{mol}$ to $0.10\;\mathrm{mol}$? And how does orthogonal projection onto the affine conservation plane remove this error?''',
        "demo4_code": r'''heading("Demonstration 4: Analyzer Noise Sweep & Affine Conservation Projection")
C_pinv = C_atom.T @ np.linalg.inv(C_atom @ C_atom.T)
unit_noise = rng.normal(0.0, 1.0, size=5)
print(f"{'Noise sigma [mol]':>18s} | {'Raw Atom Err [mol]':>18s} | {'Raw Mass Err [g]':>16s} | {'Projected Atom Err':>18s}")
print("-" * 78)
for sig in [0.001, 0.01, 0.05, 0.10]:
    w_raw = w_true + sig * unit_noise
    w_proj = w_raw - C_pinv @ (C_atom @ w_raw - d_atom)
    raw_a = float(np.max(np.abs(C_atom @ w_raw - d_atom)))
    raw_m = abs(float(molar_masses @ w_raw) - feed_mass)
    prj_a = float(np.max(np.abs(C_atom @ w_proj - d_atom)))
    print(f"{sig:18.3f} | {raw_a:18.5f} | {raw_m:16.5f} | {prj_a:18.2e}")
    assert prj_a < 1e-14''',
        "post4_md": r'''**What Demonstration 4 shows.** The violations of the element and mass balances by the raw analysis grow in proportion to the noise. The projection onto $\mathbf{C}\mathbf{w} = \mathbf{d}$ restores the balances to rounding error at every noise level. The projection makes the analysis consistent; it does not remove the noise in the directions that conserve atoms, which are the reaction extents.

---

## Inside the algorithm
Below we verify side-by-side that a pure-NumPy minimum-norm affine projection $\hat{\mathbf{w}} = \mathbf{w}_{\mathrm{raw}} - \mathbf{C}^\top(\mathbf{C}\mathbf{C}^\top)^{-1}(\mathbf{C}\mathbf{w}_{\mathrm{raw}} - \mathbf{d})$ agrees digit-for-digit with `linear_physics.equality_constrained_lstsq(np.eye(5), w_raw, C_atom, d_atom, lam=0.0)`.''',
        "inside_code": r'''heading("Inside the Algorithm: By-Hand Affine Projection vs. the Library")
w_by_hand = w_noisy_gc - C_atom.T @ np.linalg.solve(C_atom @ C_atom.T, C_atom @ w_noisy_gc - d_atom)
w_library = linear_physics.equality_constrained_lstsq(np.eye(5), w_noisy_gc, C_atom, d_atom, lam=0.0)["w"]
diff_norm = float(np.max(np.abs(w_by_hand - w_library)))
print("by hand (pure NumPy):", np.round(w_by_hand, 6))
print("library (engml_2_applied)     :", np.round(w_library, 6))
print(f"Max digit-for-digit discrepancy: {diff_norm:.2e}")
assert diff_norm < 1e-13''',
        "ex1_prompt": r'''Construct the $4 \times 6$ elemental + charge conservation matrix $\mathbf{C}_{\mathrm{aq}}$ (rows: $\mathrm{C, H, O}$, and ionic charge $z$) for the 6-species aqueous carbonate system $[\mathrm{CO_2(aq)}, \mathrm{H_2O}, \mathrm{H^+}, \mathrm{OH^-}, \mathrm{HCO_3^-}, \mathrm{CO_3^{2-}}]$. Verify that all three equilibrium reactions ($\mathrm{CO_2 + H_2O \rightleftharpoons H^+ + HCO_3^-}$, $\mathrm{HCO_3^- \rightleftharpoons H^+ + CO_3^{2-}}$, and $\mathrm{H_2O \rightleftharpoons H^+ + OH^-}$) lie identically in $\mathcal{N}(\mathbf{C}_{\mathrm{aq}})$ and prove that $\mathrm{rank}(\mathbf{C}_{\mathrm{aq}}) = 3$ due to the oxidation-state identity $z = n_{\mathrm{H}} - 2n_{\mathrm{C}} - n_{\mathrm{O}}$.''',
        "ex1_code": r'''heading("Exercise 1: Aqueous Carbonate Elemental & Electroneutrality Matrix")
# Species: [CO2, H2O, H+, OH-, HCO3-, CO3^2-]
C_aq = np.array([
    [1.0, 0.0, 0.0, 0.0, 1.0, 1.0],    # C
    [0.0, 2.0, 1.0, 1.0, 1.0, 0.0],    # H
    [2.0, 1.0, 0.0, 1.0, 3.0, 3.0],    # O
    [0.0, 0.0, 1.0, -1.0, -1.0, -2.0], # Ionic charge z = H - 2*C - O
])
nu_aq = np.array([
    [-1.0, -1.0, 1.0, 0.0,  1.0, 0.0], # CO2 + H2O <-> H+ + HCO3-
    [ 0.0,  0.0, 1.0, 0.0, -1.0, 1.0], # HCO3- <-> H+ + CO3^2-
    [ 0.0, -1.0, 1.0, 1.0,  0.0, 0.0], # H2O <-> H+ + OH-
])
rank_C = int(np.linalg.matrix_rank(C_aq))
null_res = float(np.max(np.abs(C_aq @ nu_aq.T)))
ox_check = float(np.max(np.abs(C_aq[3] - (C_aq[1] + 4.0 * C_aq[0] - 2.0 * C_aq[2]))))
print(f"Aqueous carbonate conservation matrix rank = {rank_C} (6 species - 3 invariants = 3 reactions)")
print(f"Stoichiometric nullspace residual ||C_aq @ nu_aq^T||_max = {null_res:.2e}")
print(f"Oxidation-state charge identity residual |z - (H + 4C - 2O)| = {ox_check:.2e}")
assert rank_C == 3 and null_res < 1e-15 and ox_check < 1e-15''',
        "ex1_exp": r'''Because carbon, hydrogen, and oxygen preserve fixed formal oxidation states ($+4, +1, -2$) across all six carbonate species, the charge of every species equals $n_{\mathrm{H}} + 4n_{\mathrm{C}} - 2n_{\mathrm{O}}$, so the charge row is a combination of the three element rows, giving $\mathrm{rank}(\mathbf{C}_{\mathrm{aq}}) = 3$ and $R = 6 - 3 = 3$ independent reactions.''',
        "ex2_prompt": r'''Suppose an uncalibrated pH/ion-selective electrode array reports the perturbed aqueous concentrations $\mathbf{w}_{\mathrm{bad}} = [0.010, 55.50, 10^{-4}, 10^{-10}, 0.0020, 0.0005]^\top\;\mathrm{mol/L}$. Compute the unphysical net ionic charge imbalance $\Delta z = \mathbf{c}_z^\top \mathbf{w}_{\mathrm{bad}}$ ($\mathrm{mol\;charge/L}$) and reconcile the vector onto the exact electroneutrality and elemental conservation manifold.''',
        "ex2_code": r'''heading("Exercise 2: Electroneutrality Violation Audit & Conservation Projection")
w_aq_feed = np.array([0.012, 55.50, 0.0, 0.0, 0.0, 0.0])
d_aq = C_aq[:3] @ w_aq_feed
w_bad = np.array([0.010, 55.50, 1e-4, 1e-10, 0.0020, 0.0005])
charge_imbalance_before = float(C_aq[3] @ w_bad)
w_aq_rec = linear_physics.equality_constrained_lstsq(np.eye(6), w_bad, C_aq[:3], d_aq, lam=0.0)["w"]
charge_imbalance_after = float(C_aq[3] @ w_aq_rec)
print(f"Net ionic charge before reconciliation: {charge_imbalance_before:+.6f} mol/L (Violates electroneutrality)")
print(f"Net ionic charge after reconciliation : {charge_imbalance_after:+.2e} mol/L")
assert abs(charge_imbalance_before) > 1e-3 and abs(charge_imbalance_after) < 1e-14''',
        "ex2_exp": r'''Enforcing the three independent elemental balances $\mathrm{C, H, O}$ automatically enforces exact aqueous electroneutrality $\mathbf{c}_z^\top \hat{\mathbf{w}} = 0$ to machine precision.''',
        "ex3_prompt": r'''Implement a reusable function `stoichiometric_nullspace_basis(C: np.ndarray, tol: float = 1e-10) -> np.ndarray` using `np.linalg.svd` that returns an orthonormal basis matrix $\mathbf{N} \in \mathbb{R}^{S \times R}$ for $\mathcal{N}(\mathbf{C})$, and verify on both `C_atom` ($R=2$) and `C_aq` ($R=3$) that $\mathbf{C}\mathbf{N} = \mathbf{0}$ and $\mathbf{N}^\top\mathbf{N} = \mathbf{I}_R$.''',
        "ex3_code": r'''heading("Exercise 3: Implement stoichiometric_nullspace_basis From Scratch")
def stoichiometric_nullspace_basis(C: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Return an orthonormal basis N (S x R) for the right nullspace of conservation matrix C."""
    C_mat = np.atleast_2d(np.asarray(C, dtype=float))
    _, s_vals, Vt = np.linalg.svd(C_mat, full_matrices=True)
    rank = int(np.sum(s_vals > tol))
    return Vt[rank:, :].T

N_gas = stoichiometric_nullspace_basis(C_atom)
N_aqu = stoichiometric_nullspace_basis(C_aq)
print(f"Methanol gas system nullspace shape: {N_gas.shape} | ||C N||_max = {np.max(np.abs(C_atom @ N_gas)):.2e}")
print(f"Aqueous carbonate nullspace shape  : {N_aqu.shape} | ||C N||_max = {np.max(np.abs(C_aq @ N_aqu)):.2e}")
assert N_gas.shape == (5, 2) and np.allclose(C_atom @ N_gas, 0.0, atol=1e-14) and np.allclose(N_gas.T @ N_gas, np.eye(2), atol=1e-14)
assert N_aqu.shape == (6, 3) and np.allclose(C_aq @ N_aqu, 0.0, atol=1e-14) and np.allclose(N_aqu.T @ N_aqu, np.eye(3), atol=1e-14)''',
        "ex3_takeaway": r'''**Discussion.** The SVD right singular vectors corresponding to zero singular values give an orthonormal basis of the space of reactions $\mathcal{N}(\mathbf{C})$ of any chemical or electrochemical system.''',
    },
    {
        "num": r'''01''',
        "slug": r'''01_kkt_equality_constrained_regression''',
        "title": r'''1. Least squares with exact constraints: the KKT system''',
        "hook": r'''A least-squares fit of yields or flows does not satisfy a balance unless it is told to. This lecture adds linear equality constraints to least squares and solves the resulting Karush–Kuhn–Tucker (KKT) system. The constrained solution is the unconstrained one plus a correction, and the balance holds to rounding error. The alternative of a large penalty is shown to make the linear system ill-conditioned.''',
        "objectives": [
            r'''Set up and solve the KKT system of least squares with linear equality constraints (`linear_physics.equality_constrained_lstsq`).''',
            r'''Derive and check the formula that turns the unconstrained solution into the constrained one.''',
            r'''Compare the variances of the constrained and unconstrained estimates, by the formula and by repeated synthetic experiments.''',
            r'''Compare the conditioning of the KKT system with that of a least-squares problem with a large penalty.''',
        ],
        "theory_md": r'''## 1. Principles and governing equations
To minimize regularized empirical squared error $\frac{1}{2}\|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2 + \frac{\lambda}{2}\|\mathbf{w}\|_2^2$ subject to $m$ exact linear physical invariants $\mathbf{C}\mathbf{w} = \mathbf{d}$ ($\mathbf{C} \in \mathbb{R}^{m \times d}$ of full row rank $m$), we form the Lagrangian $\mathcal{L}(\mathbf{w}, \boldsymbol{\mu}) = \frac{1}{2}\|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2 + \frac{\lambda}{2}\|\mathbf{w}\|_2^2 + \boldsymbol{\mu}^\top(\mathbf{C}\mathbf{w} - \mathbf{d})$ with dual multipliers $\boldsymbol{\mu} \in \mathbb{R}^m$. Setting $\nabla_{\mathbf{w},\boldsymbol{\mu}}\mathcal{L} = \mathbf{0}$ yields the **KKT saddle-point system**:
$$\begin{pmatrix} \mathbf{G} & \mathbf{C}^\top \\ \mathbf{C} & \mathbf{0}_{m \times m} \end{pmatrix} \begin{pmatrix} \hat{\mathbf{w}}_{\mathrm{KKT}} \\ \hat{\boldsymbol{\mu}} \end{pmatrix} = \begin{pmatrix} \mathbf{X}^\top\mathbf{y} \\ \mathbf{d} \end{pmatrix}, \qquad \mathbf{G} = \mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I}_d \tag{1.1}$$
Eliminating $\hat{\boldsymbol{\mu}}$ via the symmetric positive-definite Schur complement $\mathbf{S} = \mathbf{C}\mathbf{G}^{-1}\mathbf{C}^\top \in \mathbb{R}^{m \times m}$ reveals that $\hat{\mathbf{w}}_{\mathrm{KKT}}$ is an exact $\mathbf{G}^{-1}$-oblique projection of the unconstrained estimator $\hat{\mathbf{w}}_{\mathrm{OLS}} = \mathbf{G}^{-1}\mathbf{X}^\top\mathbf{y}$:
$$\hat{\mathbf{w}}_{\mathrm{KKT}} = \hat{\mathbf{w}}_{\mathrm{OLS}} - \mathbf{G}^{-1}\mathbf{C}^\top\left(\mathbf{C}\mathbf{G}^{-1}\mathbf{C}^\top\right)^{-1}\left(\mathbf{C}\hat{\mathbf{w}}_{\mathrm{OLS}} - \mathbf{d}\right) \tag{1.2}$$
Under homoscedastic sensor noise $\mathrm{Cov}(\mathbf{y}) = \sigma^2\mathbf{I}_N$ (as $\lambda \to 0^+$), the estimator covariance matrix shrinks by a positive semi-definite matrix:
$$\mathrm{Cov}(\hat{\mathbf{w}}_{\mathrm{KKT}}) = \sigma^2\left[\mathbf{G}^{-1} - \mathbf{G}^{-1}\mathbf{C}^\top\mathbf{S}^{-1}\mathbf{C}\mathbf{G}^{-1}\right] \preceq \mathrm{Cov}(\hat{\mathbf{w}}_{\mathrm{OLS}}) \tag{1.3}$$''',
        "demo1_code": r'''heading("Demonstration 1: Unconstrained OLS vs. Exact KKT Conservation on a Methanol Reactor")
rng = np.random.default_rng(42)
N, d = 35, 5
X_scan = np.eye(d)[np.arange(N) % d] + rng.normal(0.0, 0.15, size=(N, d))
# True methanol reactor yields conserving 1 mol C, 6 mol H, 2 mol O:
w_true = np.array([0.24, 0.08, 0.32, 0.68, 2.20])
C_atom = np.array([
    [1.0, 1.0, 0.0, 1.0, 0.0],
    [4.0, 0.0, 2.0, 0.0, 2.0],
    [1.0, 1.0, 1.0, 2.0, 0.0],
])
d_atom = np.array([1.0, 6.0, 2.0])
y_obs = X_scan @ w_true + rng.normal(0.0, 0.14, size=N)

lam = 1e-8
w_ols = linear_physics.ridge_regression(X_scan, y_obs, lam=lam, penalize_intercept=True)
kkt_res = linear_physics.equality_constrained_lstsq(X_scan, y_obs, C_atom, d_atom, lam=lam)
w_kkt = kkt_res["w"]

print(f"{'Species':>8s} | {'True Yield':>10s} | {'Unconstrained OLS':>17s} | {'KKT Constrained':>15s}")
print("-" * 58)
for sp, wt, wo, wk in zip(["CH3OH", "CO", "H2O", "CO2", "H2"], w_true, w_ols, w_kkt):
    print(f"{sp:>8s} | {wt:10.4f} | {wo:17.4f} | {wk:15.4f}")
print("OLS elemental violation ||C w_OLS - d||_max :", float(np.max(np.abs(C_atom @ w_ols - d_atom))))
print("KKT elemental residual  ||C w_KKT - d||_max :", kkt_res["constraint_residual"])
assert kkt_res["constraint_residual"] < 1e-13''',
        "post1_md": r'''**What Demonstration 1 shows.** Unconstrained least squares violates the element balances by $0.14\;\mathrm{mol}$. The constrained fit closes all three to $10^{-15}\;\mathrm{mol}$, and moves four of the five yields closer to the true ones; the hydrogen yield moves away ($2.163$ against $2.185$, true $2.200$). The constraint reduces the error of the vector as a whole (Demonstration 3), not of every component.

---

## 2. Numerical walkthrough
Let us explicitly construct the Gram matrix $\mathbf{G} = \mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I}_5$, the $3 \times 3$ Schur complement $\mathbf{S} = \mathbf{C}\mathbf{G}^{-1}\mathbf{C}^\top$, and the dual Lagrange multipliers $\hat{\boldsymbol{\mu}} = \mathbf{S}^{-1}(\mathbf{C}\hat{\mathbf{w}}_{\mathrm{OLS}} - \mathbf{d})$, confirming Eq. (1.2).''',
        "demo2_code": r'''heading("Demonstration 2: Schur-Complement Oblique Projection & Dual Multipliers")
G = X_scan.T @ X_scan + lam * np.eye(d)
G_inv_CT = np.linalg.solve(G, C_atom.T)
schur = C_atom @ G_inv_CT
mu_schur = np.linalg.solve(schur, C_atom @ w_ols - d_atom)
w_schur = w_ols - G_inv_CT @ mu_schur

print("Schur complement matrix S = C G^(-1) C^T:\n", np.round(schur, 5))
print("Lagrange multipliers mu (KKT block)     :", np.round(kkt_res["lagrange_multipliers"], 6))
print("Lagrange multipliers mu (Schur formula) :", np.round(mu_schur, 6))
assert np.allclose(w_kkt, w_schur, atol=1e-13)
assert np.allclose(kkt_res["lagrange_multipliers"], mu_schur, atol=1e-13)''',
        "post2_md": r'''**What Demonstration 2 shows.** The Lagrange multipliers $\hat{\boldsymbol{\mu}}$ represent the shadow prices (sensitivity of the residual sum of squares $\partial \mathcal{L}^*/\partial d_e$) of the carbon, hydrogen, and oxygen conservation constraints.

---

## 3. Worked case study
Next we verify the **variance reduction theorem** Eq. (1.3): projecting out noise components orthogonal to the stoichiometric nullspace $\mathcal{N}(\mathbf{C})$ reduces the trace of the covariance of the estimate. For orthonormal regressors the reduction would be $m/d = 3/5$, the number of constraints divided by the number of unknowns; here it is close to that.''',
        "demo3_code": r'''heading("Demonstration 3: Analytical Covariance Trace & Parameter Error Reduction")
err_ols = float(np.linalg.norm(w_ols - w_true))
err_kkt = float(np.linalg.norm(w_kkt - w_true))
G_inv = np.linalg.inv(G)
P_cov_kkt = G_inv - G_inv_CT @ np.linalg.solve(schur, G_inv_CT.T)
trace_ols = float(np.trace(G_inv))
trace_kkt = float(np.trace(P_cov_kkt))

print(f"Empirical L2 parameter error: OLS = {err_ols:.4f} -> KKT = {err_kkt:.4f}")
print(f"Normalized variance trace   : Tr(G^-1) = {trace_ols:.4f} -> Tr(Cov_KKT) = {trace_kkt:.4f} ({100*(1-trace_kkt/trace_ols):.1f}% reduction)")
assert err_kkt < err_ols and trace_kkt < 0.55 * trace_ols''',
        "post3_md": r'''**What Demonstration 3 shows.** Three element balances among five species leave $d - m = 2$ free directions, the reaction extents. The trace of the covariance of the estimate falls by $62\ \%$ ($0.681$ to $0.259$ in units of the noise variance), and in this realization the error of the estimate falls from $0.113$ to $0.049$.

---

## 4. What if: sensitivity to the parameters
Why not minimize a soft quadratic penalty $\|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2 + \mu_{\mathrm{pen}}\|\mathbf{C}\mathbf{w} - \mathbf{d}\|_2^2$? Let us sweep $\mu_{\mathrm{pen}} \in [10^0, 10^{12}]$ and track the Hessian condition number $\kappa(\mathbf{G} + \mu_{\mathrm{pen}}\mathbf{C}^\top\mathbf{C})$ versus the conservation violation.''',
        "demo4_code": r'''heading("Demonstration 4: The Soft-Penalty Ill-Conditioning vs. Exact KKT")
print(f"{'Penalty mu_pen':>14s} | {'cond(H_soft)':>14s} | {'Balance Violation':>18s} | {'||w_soft - w_KKT||':>18s}")
print("-" * 72)
for mu_pen in [1e0, 1e2, 1e6, 1e10, 1e12]:
    H_soft = G + mu_pen * (C_atom.T @ C_atom)
    w_soft = np.linalg.solve(H_soft, X_scan.T @ y_obs + mu_pen * (C_atom.T @ d_atom))
    cond_soft = float(np.linalg.cond(H_soft))
    res_soft = float(np.max(np.abs(C_atom @ w_soft - d_atom)))
    diff_kkt = float(np.linalg.norm(w_soft - w_kkt))
    print(f"{mu_pen:14.1e} | {cond_soft:14.2e} | {res_soft:18.2e} | {diff_kkt:18.2e}")
assert kkt_res["constraint_residual"] < 1e-13''',
        "post4_md": r'''**What Demonstration 4 shows.** With a soft penalty the violation of the balance falls as $1/\mu_{\mathrm{pen}}$ and the condition number of the matrix grows as $\mu_{\mathrm{pen}}$: a residual of $10^{-12}$ needs $\mu_{\mathrm{pen}} = 10^{12}$ and a condition number of $4\times 10^{12}$. The solution then loses accuracy: its distance from the constrained solution, which had fallen below $10^{-6}$ at $\mu_{\mathrm{pen}} = 10^{10}$, is back to about $2\times 10^{-4}$. In this regime the distance is set by rounding errors, and its digits depend on the version of the linear-algebra library. The $8\times 8$ KKT system has a condition number of $153$ and satisfies the balances to rounding error.

---

## Inside the algorithm
Below we assemble the $8 \times 8$ KKT saddle-point block matrix directly in pure NumPy and verify digit-for-digit agreement with `linear_physics.equality_constrained_lstsq`.''',
        "inside_code": r'''heading("Inside the Algorithm: By-Hand KKT Block Matrix vs. the Library")
KKT_block = np.block([
    [G, C_atom.T],
    [C_atom, np.zeros((3, 3))],
])
rhs_block = np.concatenate([X_scan.T @ y_obs, d_atom])
sol_by_hand = np.linalg.solve(KKT_block, rhs_block)
w_by_hand = sol_by_hand[:d]
diff_kkt_lib = float(np.max(np.abs(w_by_hand - kkt_res["w"])))
print("by hand (pure NumPy):", np.round(w_by_hand, 6))
print("library (engml_2_applied)     :", np.round(kkt_res["w"], 6))
print(f"Max discrepancy     : {diff_kkt_lib:.2e} | cond(KKT_block) = {np.linalg.cond(KKT_block):.2f}")
assert diff_kkt_lib < 1e-14''',
        "ex1_prompt": r'''Run a 200-trial Monte Carlo simulation on the 5-species methanol synthesis reactor (`C_atom`, `d_atom`, `w_true`) with sensor noise $\sigma = 0.14$. Verify empirically that the average squared parameter error $\mathbb{E}[\|\hat{\mathbf{w}}_{\mathrm{KKT}} - \mathbf{w}_{\mathrm{true}}\|_2^2]$ is strictly less than $55\%$ of $\mathbb{E}[\|\hat{\mathbf{w}}_{\mathrm{OLS}} - \mathbf{w}_{\mathrm{true}}\|_2^2]$ and that every single KKT trial satisfies $\|\mathbf{C}\hat{\mathbf{w}}_{\mathrm{KKT}} - \mathbf{d}\|_\infty < 10^{-13}$.''',
        "ex1_code": r'''heading("Exercise 1: 200-Trial Monte Carlo Verification of KKT Variance Reduction")
rng_mc = np.random.default_rng(2026)
sq_err_ols, sq_err_kkt, max_res_all = [], [], 0.0
for _ in range(200):
    y_mc = X_scan @ w_true + rng_mc.normal(0.0, 0.14, size=N)
    wo = linear_physics.ridge_regression(X_scan, y_mc, lam=1e-8, penalize_intercept=True)
    rk = linear_physics.equality_constrained_lstsq(X_scan, y_mc, C_atom, d_atom, lam=1e-8)
    sq_err_ols.append(float(np.sum((wo - w_true) ** 2)))
    sq_err_kkt.append(float(np.sum((rk["w"] - w_true) ** 2)))
    max_res_all = max(max_res_all, rk["constraint_residual"])

ratio = float(np.mean(sq_err_kkt) / np.mean(sq_err_ols))
print(f"200-trial MC MSE ratio (KKT / OLS) = {ratio:.4f} | Worst-case conservation residual = {max_res_all:.2e}")
assert ratio < 0.55 and max_res_all < 1e-13''',
        "ex1_exp": r'''Across 200 independent noise realizations, KKT projection eliminates 3 of the 5 noise degrees of freedom, and the mean squared error of the estimate is $36\ \%$ of that of unconstrained least squares.''',
        "ex2_prompt": r'''Diagnose what happens when a user accidentally passes a rank-deficient constraint matrix $\mathbf{C}_{\mathrm{dup}}$ containing a redundant total-mass row ($\mathbf{m}^\top = \mathbf{a}^\top \mathbf{C}_{\mathrm{atom}}$) to the unregularized KKT block matrix. Show that the $9 \times 9$ KKT matrix becomes singular ($\det = 0$), and demonstrate how SVD row-pruning restores full row rank $m = 3$.''',
        "ex2_code": r'''heading("Exercise 2: Rank-Deficient Constraint Audit & SVD Row Pruning")
m_row = np.array([12.011, 1.008, 15.999]) @ C_atom
C_dup = np.vstack([C_atom, m_row])
KKT_singular = np.block([[G, C_dup.T], [C_dup, np.zeros((4, 4))]])
s_vals_kkt = np.linalg.svd(KKT_singular, compute_uv=False)
print(f"Smallest singular value of redundant 9x9 KKT matrix: {s_vals_kkt[-1]:.2e} (SINGULAR)")
assert s_vals_kkt[-1] < 1e-12''',
        "ex2_exp": r'''Because the fourth row of $\mathbf{C}_{\mathrm{dup}}$ is a linear combination of the first three, the $9 \times 9$ KKT matrix has a 1-D nullspace in its multiplier block. Independent conservation laws must always be pruned to full row rank before assembling the KKT system.''',
        "ex3_prompt": r'''Implement a function `nullspace_reparameterized_lstsq(X, y, C, d, w_feed)` that solves $\min_{\boldsymbol{\xi}} \|\mathbf{X}(\mathbf{w}_{\mathrm{feed}} + \mathbf{N}\boldsymbol{\xi}) - \mathbf{y}\|_2^2$ directly in the reduced $R$-dimensional stoichiometric reaction-extent coordinates $\boldsymbol{\xi} \in \mathbb{R}^R$, and verify that it matches $\hat{\mathbf{w}}_{\mathrm{KKT}}$ to $<10^{-11}$.''',
        "ex3_code": r'''heading("Exercise 3: Nullspace-Reparameterized Reaction-Extent Least Squares")
def nullspace_reparameterized_lstsq(X: np.ndarray, y: np.ndarray, C: np.ndarray, w_feed: np.ndarray) -> np.ndarray:
    """Solve equality-constrained least squares by reparameterizing w = w_feed + N xi in N(C)."""
    _, s_v, Vt = np.linalg.svd(C, full_matrices=True)
    N_basis = Vt[int(np.sum(s_v > 1e-10)):, :].T
    X_xi = X @ N_basis
    y_shifted = y - X @ w_feed
    xi_hat = np.linalg.lstsq(X_xi, y_shifted, rcond=None)[0]
    return w_feed + N_basis @ xi_hat

w_null_est = nullspace_reparameterized_lstsq(X_scan, y_obs, C_atom, np.array([0.0, 0.0, 0.0, 1.0, 3.0]))
w_kkt_zero = linear_physics.equality_constrained_lstsq(X_scan, y_obs, C_atom, d_atom, lam=0.0)["w"]
diff_null = float(np.max(np.abs(w_null_est - w_kkt_zero)))
print("Nullspace extent estimator :", np.round(w_null_est, 6))
print("KKT saddle-point estimator :", np.round(w_kkt_zero, 6))
print(f"Max discrepancy            : {diff_null:.2e}")
assert diff_null < 1e-11''',
        "ex3_takeaway": r'''**Discussion.** Nullspace reparameterization ($\mathbf{w} = \mathbf{w}_{\mathrm{feed}} + \mathbf{N}\boldsymbol{\xi}$) and KKT saddle-point projection produce the exact same constrained optimum from complementary primal and dual perspectives.''',
    },
    {
        "num": r'''02''',
        "slug": r'''02_process_data_reconciliation_and_gross_error_detection''',
        "title": r'''2. Data reconciliation and the detection of gross errors''',
        "hook": r'''The measured flows of a plant do not close its mass balances. Data reconciliation finds the smallest adjustment, weighted by the accuracy of each instrument, that makes them close. The size of the adjustment is also a test: if it is larger than the noise can explain, an instrument is biased, and the lecture shows how to look for it.''',
        "objectives": [
            r'''State weighted data reconciliation as a constrained least-squares problem (`linear_physics.crowe_data_reconciliation`).''',
            r'''Check on a splitter with three streams that the adjustments are proportional to the variances of the instruments.''',
            r'''Reconcile the five streams of a reactor, separator and recycle.''',
            r'''Test for a biased flow meter with the chi-squared statistic of the adjustment and with the standardized adjustments of the single measurements.''',
        ],
        "theory_md": r'''## 1. Principles and governing equations
Given $d$ noisy plant flowrate measurements $\mathbf{y}_{\mathrm{meas}} \in \mathbb{R}^d$ ($\mathrm{kg/h}$) with diagonal sensor covariance matrix $\boldsymbol{\Sigma} = \mathrm{diag}(\sigma_1^2, \dots, \sigma_d^2)$ and $m$ nodal steady-state conservation balances $\mathbf{A}\mathbf{y} = \mathbf{b}$ ($\mathbf{A} \in \mathbb{R}^{m \times d}$), **Crowe's weighted data reconciliation (1983)** solves the maximum-likelihood problem:
$$\min_{\mathbf{y} \in \mathbb{R}^d} (\mathbf{y} - \mathbf{y}_{\mathrm{meas}})^\top \boldsymbol{\Sigma}^{-1}(\mathbf{y} - \mathbf{y}_{\mathrm{meas}}) \quad \text{subject to} \quad \mathbf{A}\mathbf{y} = \mathbf{b} \tag{1.1}$$
Letting $\mathbf{r} = \mathbf{A}\mathbf{y}_{\mathrm{meas}} - \mathbf{b} \in \mathbb{R}^m$ denote the raw nodal balance residuals with covariance $\mathbf{S}_r = \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^\top$, the closed-form reconciled flow vector is:
$$\hat{\mathbf{y}}_{\mathrm{rec}} = \mathbf{y}_{\mathrm{meas}} - \boldsymbol{\Sigma}\mathbf{A}^\top\left(\mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^\top\right)^{-1}\mathbf{r} \tag{1.2}$$
Under the null hypothesis $H_0$ of zero-mean Gaussian noise ($\mathbf{y}_{\mathrm{meas}} \sim \mathcal{N}(\mathbf{y}_{\mathrm{true}}, \boldsymbol{\Sigma})$), the global gross-error statistic and individual standardized measurement test statistics follow exact distributions:
$$\chi^2_{\mathrm{stat}} = \mathbf{r}^\top \mathbf{S}_r^{-1}\mathbf{r} \sim \chi^2_m, \qquad z_{\mathrm{meas},i} = \frac{|\hat{y}_{\mathrm{rec},i} - y_{\mathrm{meas},i}|}{\sqrt{(\boldsymbol{\Sigma}\mathbf{A}^\top\mathbf{S}_r^{-1}\mathbf{A}\boldsymbol{\Sigma})_{ii}}} \sim |\mathcal{N}(0,1)| \tag{1.3}$$''',
        "demo1_code": r'''heading("Demonstration 1: 3-Stream Splitter Pen-and-Paper Micro-Example")
# Splitter node: Feed F1 splits into Product F2 and Recycle F3 -> F1 - F2 - F3 = 0
A_split = np.array([[1.0, -1.0, -1.0]])
b_split = np.array([0.0])
y_meas_split = np.array([102.0, 60.0, 39.0])  # Raw imbalance r = 102 - 60 - 39 = +3.0 kg/h
sigma_split = np.array([1.0, 1.0, 2.0])       # Variances = [1, 1, 4], sum = 6.0

rec_split = linear_physics.crowe_data_reconciliation(y_meas_split, sigma_split, A_split, b_split)
# Exact hand calculation: adjustments proportional to variance [-1/6 * 3, +1/6 * 3, +4/6 * 3] = [-0.5, +0.5, +2.0]
assert np.allclose(rec_split["y_reconciled"], [101.5, 60.5, 41.0], atol=1e-14)
assert np.isclose(rec_split["chi2_stat"], (3.0 ** 2) / 6.0, rtol=1e-14)
print("Measured flows   [kg/h]:", y_meas_split, f"| Raw imbalance = {rec_split['balance_residual_before'][0]:+.2f} kg/h")
print("Reconciled flows [kg/h]:", rec_split["y_reconciled"], f"| New imbalance = {rec_split['balance_residual_after'][0]:+.1e} kg/h")
print(f"Adjustments      [kg/h]: {rec_split['y_reconciled'] - y_meas_split} (Proportional to variances [1, 1, 4])")
print(f"Global chi^2 statistic : {rec_split['chi2_stat']:.4f} (df = {rec_split['dof']})")''',
        "post1_md": r'''**What Demonstration 1 shows.** Stream 3 has twice the standard deviation ($\sigma_3 = 2.0\;\mathrm{kg/h}$, variance $4.0$) of Streams 1 and 2 ($\sigma_1=\sigma_2=1.0\;\mathrm{kg/h}$, variance $1.0$), so Crowe projection assigns $4/6 = 66.7\%$ of the $3.0\;\mathrm{kg/h}$ correction ($+2.0\;\mathrm{kg/h}$) to Stream 3.

---

## 2. Numerical walkthrough
Now consider a 5-stream, 3-node **Reactor–Separator–Recycle network**:
- **Streams:** $y_1$ (fresh feed, Coriolis $\sigma_1=0.5$), $y_2$ (reactor inlet, $\sigma_2=1.2$), $y_3$ (reactor outlet, $\sigma_3=1.2$), $y_4$ (overhead product, Coriolis $\sigma_4=0.5$), $y_5$ (recycle loop, Orifice $\sigma_5=2.0$).
- **Balances at the nodes ($\mathbf{A}_{\mathrm{net}}\mathbf{y} = \mathbf{0}$):** Mixer ($y_1 + y_5 - y_2 = 0$), Reactor ($y_2 - y_3 = 0$), Separator ($y_3 - y_4 - y_5 = 0$).''',
        "demo2_code": r'''heading("Demonstration 2: 5-Stream Reactor-Separator-Recycle Incidence & Residual Covariance")
A_net = np.array([
    [1.0, -1.0,  0.0,  0.0,  1.0],  # Node 1: Mixer
    [0.0,  1.0, -1.0,  0.0,  0.0],  # Node 2: Reactor
    [0.0,  0.0,  1.0, -1.0, -1.0],  # Node 3: Separator
])
b_net = np.zeros(3)
y_true_net = np.array([100.0, 140.0, 140.0, 100.0, 40.0])
sigma_net = np.array([0.5, 1.2, 1.2, 0.5, 2.0])

rng = np.random.default_rng(101)
y_normal = y_true_net + rng.normal(0.0, sigma_net)
rec_normal = linear_physics.crowe_data_reconciliation(y_normal, sigma_net, A_net, b_net)
print("Raw nodal balance residuals [Mixer, Reactor, Separator]:", np.round(rec_normal["balance_residual_before"], 4))
print("Post-reconciliation nodal residuals                    :", np.round(rec_normal["balance_residual_after"], 15))
print(f"Normal operation chi^2 = {rec_normal['chi2_stat']:.3f} < chi^2_(3, 0.95) = 7.815 (No gross error alarm)")
assert np.max(np.abs(rec_normal["balance_residual_after"])) < 1e-13 and rec_normal["chi2_stat"] < 7.815''',
        "post2_md": r'''**What Demonstration 2 shows.** With random noise only, the reconciled flows close the three unit balances to $10^{-14}\;\mathrm{kg/h}$, and the test statistic, $\chi^2 = 4.67$, is below the $95\ \%$ threshold for three degrees of freedom, $7.815$. The flows and their noise are synthetic.

---

## 3. Worked case study
What if Stream 3 (reactor outlet flowmeter) suffers a $+12.0\;\mathrm{kg/h}$ gross bias due to calibration drift? Let us inspect how a naive reconciliation without gross-error detection smears the $+12.0\;\mathrm{kg/h}$ bias across healthy meters, and how `measurement_z` pinpoints Stream 3.''',
        "demo3_code": r'''heading("Demonstration 3: Gross-Error Injection & Faulty Sensor Isolation via Measurement z-Test")
y_faulty = y_normal.copy()
y_faulty[2] += 12.0  # Inject +12 kg/h gross bias into Stream 3 (Reactor Outlet flowmeter)
rec_faulty = linear_physics.crowe_data_reconciliation(y_faulty, sigma_net, A_net, b_net)
fault_sensor_idx = int(np.argmax(rec_faulty["measurement_z"]))

print(f"Faulty operation chi^2 = {rec_faulty['chi2_stat']:.2f} >> 7.815 (gross error alarm triggered)")
print(f"{'Stream':>8s} | {'True':>7s} | {'Measured':>9s} | {'Reconciled':>10s} | {'Meas z-score':>12s}")
print("-" * 58)
for i in range(5):
    flag = " <-- largest!" if i == fault_sensor_idx else ""
    print(f"Stream {i+1:1d} | {y_true_net[i]:7.2f} | {y_faulty[i]:9.2f} | {rec_faulty['y_reconciled'][i]:10.2f} | {rec_faulty['measurement_z'][i]:12.2f}{flag}")
assert rec_faulty["chi2_stat"] > 30.0 and fault_sensor_idx == 2''',
        "post3_md": r'''**What Demonstration 3 shows.** With a bias of $12\;\mathrm{kg/h}$ added to the meter of Stream 3, the global statistic is $\chi^2 = 87.9$, far above $7.815$. The measurement test is largest for Stream 3 ($z = 9.28$), which identifies the faulty meter. It is also large for Stream 2 ($z = 7.40$): the two streams enter the same balance, so a gross error in one is partly attributed to the other. The usual procedure is to remove the meter with the largest statistic, reconcile again, and repeat (Exercise 3).

---

## 4. What if: sensitivity to the parameters
Let us sweep the magnitude of the Stream 3 bias $\Delta y_3 \in [0, 12]\;\mathrm{kg/h}$ to map out the exact operating threshold where $\chi^2_{\mathrm{stat}}$ crosses the $95\%$ ($\chi^2_{3,0.95}=7.815$) and $99\%$ ($\chi^2_{3,0.99}=11.345$) detection limits.''',
        "demo4_code": r'''heading("Demonstration 4: Gross-Error Magnitude Sweep & Chi-Squared Power Curve")
print(f"{'Bias dY_3 [kg/h]':>16s} | {'Global chi^2':>12s} | {'Max z_meas':>10s} | {'Isolated Stream':>15s} | {'Alarm Status':>14s}")
print("-" * 76)
for bias in [0.0, 2.0, 4.5, 7.0, 12.0]:
    y_sw = y_normal.copy()
    y_sw[2] += bias
    r_sw = linear_physics.crowe_data_reconciliation(y_sw, sigma_net, A_net, b_net)
    idx_max = int(np.argmax(r_sw["measurement_z"])) + 1
    status = "ALARM (>99%)" if r_sw["chi2_stat"] > 11.345 else ("WARN (>95%)" if r_sw["chi2_stat"] > 7.815 else "Normal")
    print(f"{bias:16.1f} | {r_sw['chi2_stat']:12.2f} | {np.max(r_sw['measurement_z']):10.2f} | {'Stream ' + str(idx_max):>15s} | {status:>14s}")''',
        "post4_md": r'''**What Demonstration 4 shows.** A bias of $2\;\mathrm{kg/h}$ in Stream 3 already takes the global statistic past the $95\ \%$ threshold ($10.6 > 7.815$), and from $4.5\;\mathrm{kg/h}$ it is past the $99\ \%$ threshold as well. In every case with a bias, the largest measurement statistic is that of Stream 3.

---

## Inside the algorithm
Below we verify a pure-NumPy covariance-weighted projection $\hat{\mathbf{y}} = \mathbf{y}_{\mathrm{meas}} - \boldsymbol{\Sigma}\mathbf{A}^\top(\mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^\top)^{-1}(\mathbf{A}\mathbf{y}_{\mathrm{meas}} - \mathbf{b})$ against `linear_physics.crowe_data_reconciliation`.''',
        "inside_code": r'''heading("Inside the Algorithm: By-Hand Crowe Projection vs. the Library")
Sigma_mat = np.diag(sigma_net ** 2)
r_vec = A_net @ y_normal - b_net
S_r_mat = A_net @ Sigma_mat @ A_net.T
y_rec_hand = y_normal - Sigma_mat @ A_net.T @ np.linalg.solve(S_r_mat, r_vec)
diff_crowe = float(np.max(np.abs(y_rec_hand - rec_normal["y_reconciled"])))
print("by hand (pure NumPy):", np.round(y_rec_hand, 6))
print("library (engml_2_applied)     :", np.round(rec_normal["y_reconciled"], 6))
print(f"Max discrepancy     : {diff_crowe:.2e}")
assert diff_crowe < 1e-13''',
        "ex1_prompt": r'''In the 5-stream reactor-separator network (`A_net`, `b_net`, `sigma_net`), inject a $-10.0\;\mathrm{kg/h}$ gross bias into the recycle orifice meter (Stream 5, index `4`) on top of `y_true_net`. Run `crowe_data_reconciliation`, verify that $\chi^2_{\mathrm{stat}} > 15.0$, and confirm that `np.argmax(rec['measurement_z']) == 4` isolates Stream 5.''',
        "ex1_code": r'''heading("Exercise 1: Recycle Orifice Meter Gross-Error Isolation")
y_bias5 = y_true_net.copy()
y_bias5[4] -= 10.0
rec_bias5 = linear_physics.crowe_data_reconciliation(y_bias5, sigma_net, A_net, b_net)
isolated_idx = int(np.argmax(rec_bias5["measurement_z"]))
print(f"Recycle fault chi^2 = {rec_bias5['chi2_stat']:.2f} | Isolated stream index = {isolated_idx} (z = {rec_bias5['measurement_z'][isolated_idx]:.2f})")
assert rec_bias5["chi2_stat"] > 15.0 and isolated_idx == 4
assert np.max(np.abs(rec_bias5["balance_residual_after"])) < 1e-13''',
        "ex1_exp": r'''Even though the recycle orifice meter has the largest uncertainty ($\sigma_5 = 2.0\;\mathrm{kg/h}$), a $-10.0\;\mathrm{kg/h}$ ($5\sigma_5$) bias violates the Mixer and Separator balances simultaneously, which gives $\chi^2 = 20.6$ and the largest measurement statistic at index `4` ($z = 4.54$).''',
        "ex2_prompt": r'''Quantify the smearing damage caused by reconciling a flowsheet *without* removing a faulty sensor first: compare the mean absolute error (MAE) across the 4 healthy streams (`[0, 1, 3, 4]`) when `y_faulty` (with the $+12\;\mathrm{kg/h}$ bias on Stream 3) is reconciled directly, versus replacing the faulty Stream 3 uncertainty with a huge dummy standard deviation $\sigma_3 = 1000\;\mathrm{kg/h}$ (effectively unweighting the broken meter).''',
        "ex2_code": r'''heading("Exercise 2: Smearing Damage Audit & Faulty-Sensor Downweighting")
healthy_mask = [0, 1, 3, 4]
mae_smeared = float(np.mean(np.abs(rec_faulty["y_reconciled"][healthy_mask] - y_true_net[healthy_mask])))
sigma_cleaned = sigma_net.copy()
sigma_cleaned[2] = 1000.0  # Unweight isolated faulty sensor #3
rec_cleaned = linear_physics.crowe_data_reconciliation(y_faulty, sigma_cleaned, A_net, b_net)
mae_cleaned = float(np.mean(np.abs(rec_cleaned["y_reconciled"][healthy_mask] - y_true_net[healthy_mask])))
print(f"Healthy-stream MAE with unremoved fault (smearing) : {mae_smeared:.3f} kg/h")
print(f"Healthy-stream MAE after downweighting Stream 3    : {mae_cleaned:.3f} kg/h")
print(f"Reconstructed flow for broken Stream 3             : {rec_cleaned['y_reconciled'][2]:.2f} kg/h (True = 140.00)")
assert mae_cleaned < 0.55 * mae_smeared and rec_cleaned["chi2_stat"] < 2.0''',
        "ex2_exp": r'''Reconciling with the faulty meter left in spreads its error over the others: their mean absolute error is $2.18\;\mathrm{kg/h}$. With the faulty meter given a very large variance, the mean absolute error of the healthy meters is $1.00\;\mathrm{kg/h}$, and the balances reconstruct the flow of Stream 3 as $138.0\;\mathrm{kg/h}$ (true value $140$).''',
        "ex3_prompt": r'''Implement an automated gross-error elimination function `iterative_crowe_reconcile(y_meas, sigma_meas, A, b, chi2_threshold=7.815)` that checks $\chi^2_{\mathrm{stat}} > \chi^2_{\mathrm{threshold}}$, identifies the sensor with maximum `measurement_z`, inflates its standard deviation by $10^4\times$ to remove its corrupting leverage, and re-runs Crowe reconciliation. Verify it on `y_faulty`.''',
        "ex3_code": r'''heading("Exercise 3: Implement Automated Gross-Error Elimination & Reconciliation")
def iterative_crowe_reconcile(
    y_meas: np.ndarray, sigma_meas: np.ndarray, A: np.ndarray, b: np.ndarray, chi2_threshold: float = 7.815
) -> dict:
    """Detect gross error via chi^2 test, unweight the highest-z sensor, and reconcile."""
    sig_curr = np.asarray(sigma_meas, dtype=float).copy()
    rec = linear_physics.crowe_data_reconciliation(y_meas, sig_curr, A, b)
    flagged = []
    if rec["chi2_stat"] > chi2_threshold:
        bad_idx = int(np.argmax(rec["measurement_z"]))
        flagged.append(bad_idx)
        sig_curr[bad_idx] *= 1e4
        rec = linear_physics.crowe_data_reconciliation(y_meas, sig_curr, A, b)
    rec["flagged_sensors"] = flagged
    return rec

auto_res = iterative_crowe_reconcile(y_faulty, sigma_net, A_net, b_net)
print("Flagged sensor indices:", auto_res["flagged_sensors"])
print("Cleaned reconciled flows [kg/h]:", np.round(auto_res["y_reconciled"], 2))
print(f"Post-cleaning chi^2 = {auto_res['chi2_stat']:.4f} | Max balance residual = {np.max(np.abs(auto_res['balance_residual_after'])):.2e}")
assert auto_res["flagged_sensors"] == [2] and auto_res["chi2_stat"] < 2.0''',
        "ex3_takeaway": r'''**Discussion.** Reconciliation followed by the $\chi^2$ test and the removal of the measurement with the largest standardized adjustment keeps one biased instrument from spreading its error over the other streams.''',
    },
    {
        "num": r'''03''',
        "slug": r'''03_thermodynamic_stability_and_pava_isotonic_regression''',
        "title": r'''3. Monotone fits and the pool-adjacent-violators algorithm''',
        "hook": r'''The pressure of a stable fluid cannot rise with its volume at constant temperature, and a curve fitted to noisy data does not know this. The pool-adjacent-violators algorithm (PAVA) finds the monotone sequence closest to the data in one pass. Applied to a van der Waals isotherm below the critical temperature, it reproduces Maxwell's equal-area construction.''',
        "objectives": [
            r'''Relate the convexity of the Helmholtz energy in the volume to a pressure that does not rise with the volume.''',
            r'''Follow the pool-adjacent-violators algorithm (`linear_physics.pava_monotone_regression`) on four points by hand.''',
            r'''Compare the monotone fit with unconstrained polynomials on a noisy synthetic isotherm.''',
            r'''Apply the algorithm to a van der Waals isotherm below the critical temperature and compare the plateau with Maxwell's equal-area rule.''',
        ],
        "theory_md": r'''## 1. Principles and governing equations
For a single-phase pure fluid at constant temperature $T$, the differential of molar Helmholtz free energy is $dA_m = -S_m\,dT - P\,dV_m$, so $P = -\left(\frac{\partial A_m}{\partial V_m}\right)_T$. **Second-law mechanical stability** requires $A_m(V_m)$ to be locally convex ($\partial^2 A_m / \partial V_m^2 \ge 0$), which demands non-negative isothermal compressibility $\kappa_T > 0$ and a monotonically non-increasing pressure isotherm:
$$\left(\frac{\partial P}{\partial V_m}\right)_T = -\left(\frac{\partial^2 A_m}{\partial V_m^2}\right)_T \le 0, \qquad \kappa_T = -\frac{1}{V_m}\left(\frac{\partial V_m}{\partial P}\right)_T \ge 0 \tag{1.1}$$
Given $N$ noisy measurements $\mathbf{y} \in \mathbb{R}^N$ ordered along $V_{m,1} < V_{m,2} < \dots < V_{m,N}$, **isotonic regression** computes the orthogonal $L_2$ projection onto the closed convex monotone cone $\mathcal{C}_{\downarrow} = \{\mathbf{f} \in \mathbb{R}^N : f_1 \ge f_2 \ge \dots \ge f_N\}$:
$$\hat{\mathbf{f}}_{\mathrm{PAVA}} = \Pi_{\mathcal{C}_{\downarrow}}(\mathbf{y}) = \arg\min_{\mathbf{f} \in \mathcal{C}_{\downarrow}} \frac{1}{2}\|\mathbf{f} - \mathbf{y}\|_2^2 \tag{1.2}$$
By Moreau's convex cone projection theorem, the **pool-adjacent-violators algorithm (PAVA, ayer et al. 1955)** finds the exact global minimum in $O(N)$ time and satisfies two fundamental identities for any true monotone curve $\mathbf{f}_{\mathrm{true}} \in \mathcal{C}_{\downarrow}$:
$$\sum_{i=1}^N \hat{f}_{\mathrm{PAVA},i} = \sum_{i=1}^N y_i \quad (\text{mean conservation}), \qquad \|\hat{\mathbf{f}}_{\mathrm{PAVA}} - \mathbf{f}_{\mathrm{true}}\|_2 \le \|\mathbf{y} - \mathbf{f}_{\mathrm{true}}\|_2 \quad (L_2\text{ Contraction}) \tag{1.3}$$''',
        "demo1_code": r'''heading("Demonstration 1: 4-Point Pen-and-Paper PAVA Walkthrough and a noisy synthetic isotherm")
y_micro = np.array([1.0, 3.0, 2.0, 4.5])
f_micro = linear_physics.pava_monotone_regression(y_micro, increasing=True)
print("4-Point Micro-Example Input y :", y_micro)
print("4-Point PAVA Increasing Output:", f_micro, "(Adjacent violators [3.0, 2.0] pooled to mean 2.5)")
assert np.allclose(f_micro, [1.0, 2.5, 2.5, 4.5], atol=1e-15)

# A synthetic isotherm of cubic-equation-of-state form with assumed constants (V_m in L/mol, P in MPa):
rng = np.random.default_rng(42)
Vm = np.linspace(0.12, 0.80, 40)
P_true = 33.2 / (Vm - 0.065) - 9.4 / (Vm ** 2 + 0.13 * Vm)
P_noisy = P_true + rng.normal(0.0, 2.2, size=len(Vm))
P_pava = linear_physics.pava_monotone_regression(P_noisy, increasing=False)

assert float(np.max(np.diff(P_pava))) <= 1e-15, "PAVA isotherm must satisfy dP/dVm <= 0 everywhere!"
assert np.isclose(np.mean(P_pava), np.mean(P_noisy), rtol=1e-14), "PAVA must conserve the sample mean!"
print(f"Noisy isotherm stability violations (dP > 0): {int(np.sum(np.diff(P_noisy) > 0))} intervals")
print(f"PAVA isotherm stability violations  (dP > 0): {int(np.sum(np.diff(P_pava) > 0))} intervals")''',
        "post1_md": r'''**What Demonstration 1 shows.** Where noise makes the pressure rise with the volume ($P_{i+1} > P_i$ at $V_{m,i+1} > V_{m,i}$), the algorithm replaces the offending neighbours by their mean, and repeats until no violation is left. The synthetic isotherm has 9 such intervals before and none after, and the mean of the data is unchanged.

---

## 2. Numerical walkthrough
Let us inspect the block-pooling structure of PAVA on the first 8 molar-volume points of the isotherm, showing each pooled plateau's block size $n_b$, block sum $S_b$, and constant plateau pressure $\bar{P}_b = S_b / n_b$.''',
        "demo2_code": r'''heading("Demonstration 2: Step-by-Step Stack Block-Pooling Inspection")
y_sub8 = P_noisy[:8]
blocks = []
for idx, val in enumerate(y_sub8):
    blocks.append([float(val), 1])
    while len(blocks) >= 2 and (blocks[-2][0] / blocks[-2][1]) < (blocks[-1][0] / blocks[-1][1]):  # decreasing check
        last = blocks.pop()
        blocks[-1][0] += last[0]
        blocks[-1][1] += last[1]
    plateaus = [round(b[0] / b[1], 2) for b in blocks]
    counts = [b[1] for b in blocks]
    print(f"Step {idx+1} (insert P={val:6.2f} MPa) -> Active blocks (mean, count): {list(zip(plateaus, counts))}")''',
        "post2_md": r'''**What Demonstration 2 shows.** Each point enters the stack once, and each merge removes one block, so the inner `while` loop runs at most $N - 1$ times over the whole data set: the cost is proportional to $N$.

---

## 3. Worked case study
Let us compare our non-parametric $O(N)$ PAVA projection against an unconstrained degree-7 polynomial fit `np.polyfit(Vm, P_noisy, deg=7)` on the 40 points of the isotherm.''',
        "demo3_code": r'''heading("Demonstration 3: PAVA vs. Unconstrained Degree-7 Polynomial Benchmark")
poly_coef = np.polyfit(Vm, P_noisy, deg=7)
P_poly7 = np.polyval(poly_coef, Vm)
poly_violations = int(np.sum(np.diff(P_poly7) > 0))

rmse_noisy = float(np.sqrt(np.mean((P_noisy - P_true) ** 2)))
rmse_poly7 = float(np.sqrt(np.mean((P_poly7 - P_true) ** 2)))
rmse_pava = float(np.sqrt(np.mean((P_pava - P_true) ** 2)))
print(f"RMSE vs True Isotherm: Noisy = {rmse_noisy:.3f} MPa | Poly-7 = {rmse_poly7:.3f} MPa ({poly_violations} violations) | PAVA = {rmse_pava:.3f} MPa (0 violations)")
assert rmse_pava < rmse_noisy and poly_violations > 0''',
        "post3_md": r'''**What Demonstration 3 shows.** Against the isotherm that generated the data, the error of the noisy points is $1.80\;\mathrm{MPa}$. The monotone fit reduces it to $1.47\;\mathrm{MPa}$, by $18\ \%$, with no violation. The unconstrained polynomial of degree 7 is worse than the raw data ($2.57\;\mathrm{MPa}$) and still has two intervals where the pressure rises with the volume.

---

## 4. What if: sensitivity to the parameters
What happens when the algorithm is applied to an isotherm of the van der Waals equation below the critical temperature, which contains a loop ($(\partial P / \partial V_m)_T > 0$)?''',
        "demo4_code": r'''heading("Demonstration 4: The van der Waals loop and the equal-area plateau")
# Reduced van der Waals isotherm at T_r = 0.9:  P_r = 8 T_r / (3 V_r - 1) - 3 / V_r^2
T_r = 0.9
Vm_sub = np.linspace(0.45, 5.0, 1821)                      # reduced volume, uniform grid
P_vdw = 8.0 * T_r / (3.0 * Vm_sub - 1.0) - 3.0 / Vm_sub ** 2
P_maxwell_pava = linear_physics.pava_monotone_regression(P_vdw, increasing=False)
n_spinodal = int(np.sum(np.diff(P_vdw) > 0))
n_after = int(np.sum(np.diff(P_maxwell_pava) > 0))
plateau_vals, plateau_counts = np.unique(np.round(P_maxwell_pava, 9), return_counts=True)
p_sat_est = plateau_vals[int(np.argmax(plateau_counts))]

# Independent check: Maxwell's equal-area rule, solved by bisection on the saturation pressure.
def equal_area_mismatch(p_sat):
    """Integral of (P - p_sat) dV between the outer roots of P(V) = p_sat; zero at saturation."""
    roots = np.roots([3.0 * p_sat, -(p_sat + 8.0 * T_r), 9.0, -3.0])       # cubic in V_r
    v_l, v_g = np.min(roots.real), np.max(roots.real)
    antiderivative = lambda v: (8.0 * T_r / 3.0) * np.log(3.0 * v - 1.0) + 3.0 / v
    return antiderivative(v_g) - antiderivative(v_l) - p_sat * (v_g - v_l), v_l, v_g

p_lo, p_hi = 0.45, 0.72                                     # between the two spinodal pressures
for _ in range(60):
    p_mid = 0.5 * (p_lo + p_hi)
    if equal_area_mismatch(p_mid)[0] > 0.0:
        p_lo = p_mid
    else:
        p_hi = p_mid
p_sat_maxwell, v_liq, v_gas = p_mid, *equal_area_mismatch(p_mid)[1:]
on_plateau = np.isclose(P_maxwell_pava, p_sat_est, atol=1e-8)
print(f"Intervals where the pressure rises with the volume: {n_spinodal} -> after PAVA: {n_after}")
print(f"Plateau of the monotone fit    : P_r = {p_sat_est:.5f}, from V_r = {Vm_sub[on_plateau][0]:.3f} to {Vm_sub[on_plateau][-1]:.3f} ({np.max(plateau_counts)} grid points)")
print(f"Maxwell's equal-area rule      : P_r = {p_sat_maxwell:.5f}, from V_r = {v_liq:.3f} to {v_gas:.3f}")
assert n_spinodal > 10 and n_after == 0
assert np.isclose(np.mean(P_maxwell_pava), np.mean(P_vdw), rtol=1e-12)
assert abs(p_sat_est - p_sat_maxwell) < 2e-3''',
        "post4_md": r'''**What Demonstration 4 shows.** Below the critical temperature the van der Waals isotherm has a loop in which the pressure rises with the volume. The monotone fit replaces the loop and parts of its two flanks by one plateau at $P_r = 0.64700$, between $V_r = 0.605$ and $2.348$. Maxwell's equal-area rule, solved independently by bisection, gives $P_r = 0.64700$ between $V_r = 0.603$ and $2.349$. The agreement is not a coincidence: on a uniform grid in volume the plateau is the mean of the pressures it replaces, so the areas between the isotherm and the plateau cancel, and the plateau ends where the isotherm returns to the plateau pressure. These are the two conditions of the equal-area rule, and the small differences in the end points are the spacing of the grid.

---

## Inside the algorithm
Below we run a standalone 12-line stack-based PAVA implementation in pure NumPy and verify digit-for-digit agreement with `linear_physics.pava_monotone_regression`.''',
        "inside_code": r'''heading("Inside the Algorithm: By-Hand Stack PAVA vs. the Library")
def pava_increasing_by_hand(y_arr: np.ndarray) -> np.ndarray:
    st: list[list[float]] = []
    for v in y_arr:
        st.append([float(v), 1.0])
        while len(st) >= 2 and st[-2][0] / st[-2][1] > st[-1][0] / st[-1][1]:
            s_r, c_r = st.pop()
            st[-1][0] += s_r
            st[-1][1] += c_r
    return np.concatenate([np.full(int(c), s / c) for s, c in st])

p_hand = -pava_increasing_by_hand(-P_noisy)
diff_pava = float(np.max(np.abs(p_hand - P_pava)))
print("by hand (pure NumPy) first 6 pts:", np.round(p_hand[:6], 5))
print("library (engml_2_applied)      first 6 pts:", np.round(P_pava[:6], 5))
print(f"Max digit-for-digit discrepancy : {diff_pava:.2e}")
assert diff_pava < 1e-15''',
        "ex1_prompt": r'''Generate $N = 50$ noisy Langmuir $\mathrm{CO_2}$ adsorption uptake measurements $q(P) = q_{\max} K P / (1 + K P)$ of Langmuir form with assumed constants ($q_{\max} = 5.2\;\mathrm{mol/kg}, K = 1.4\;\mathrm{bar}^{-1}$) across $P \in [0.1, 10.0]\;\mathrm{bar}$ with Gaussian noise $\sigma = 0.25\;\mathrm{mol/kg}$. Apply `pava_monotone_regression(.., increasing=True)` and verify that (1) $\min(\Delta \hat{q}) \ge 0$, (2) $\mathrm{mean}(\hat{q}) = \mathrm{mean}(y)$, and (3) $\|\hat{\mathbf{q}}_{\mathrm{PAVA}} - \mathbf{q}_{\mathrm{true}}\|_2 < \|\mathbf{y} - \mathbf{q}_{\mathrm{true}}\|_2$.''',
        "ex1_code": r'''heading("Exercise 1: PAVA Projection of a Synthetic CO2 Adsorption Isotherm")
rng_ex = np.random.default_rng(77)
P_ads = np.linspace(0.1, 10.0, 50)
q_true = 5.2 * 1.4 * P_ads / (1.0 + 1.4 * P_ads)
q_noisy = q_true + rng_ex.normal(0.0, 0.25, size=50)
q_pava = linear_physics.pava_monotone_regression(q_noisy, increasing=True)

assert float(np.min(np.diff(q_pava))) >= -1e-15
assert np.isclose(np.mean(q_pava), np.mean(q_noisy), rtol=1e-14)
assert np.linalg.norm(q_pava - q_true) < np.linalg.norm(q_noisy - q_true)
print(f"Synthetic adsorption isotherm, L2 error: Raw = {np.linalg.norm(q_noisy - q_true):.3f} -> PAVA = {np.linalg.norm(q_pava - q_true):.3f} mol/kg")''',
        "ex1_exp": r'''Thermodynamic adsorption stability requires $(\partial q / \partial P)_T \ge 0$. PAVA projects the noisy uptake curve onto the monotone increasing cone and reduces the $L_2$ error from $1.87$ to $0.82\;\mathrm{mol/kg}$.''',
        "ex2_prompt": r'''Show why Newton's method for the inverse problem $P(V_m) = P_{\mathrm{target}}$ fails on a staircase: for the pooled van der Waals isotherm of Demonstration 4 (`P_maxwell_pava` on the grid `Vm_sub`), compute the finite-difference slope $\Delta \hat{P}/\Delta V_m$ on every interval and report the fraction of intervals where it is zero.''',
        "ex2_code": r'''heading("Exercise 2: Staircase Zero-Derivative Audit on PAVA Plateaus")
dP_pava = np.diff(P_maxwell_pava) / np.diff(Vm_sub)
zero_slope_frac = float(np.mean(np.abs(dP_pava) < 1e-14))
print(f"Fraction of V_m intervals with zero slope (dP/dVm == 0) on PAVA staircase: {zero_slope_frac*100:.1f}%")
assert zero_slope_frac > 0.30''',
        "ex2_exp": r'''The grid `Vm_sub` covers the liquid, the two-phase region and the vapour. On the part pooled into the plateau the slope is exactly zero, and the printed fraction is the share of the grid that this part takes. Newton's step, $\Delta V_m = -(P - P_{\mathrm{target}})/(dP/dV_m)$, divides by the slope and is undefined there. A monotone fit that is to be inverted by a gradient method needs a non-zero, continuous derivative, which is the subject of Lecture 4.''',
        "ex3_prompt": r'''Implement a **weighted PAVA** function `weighted_pava_increasing(y: np.ndarray, weights: np.ndarray) -> np.ndarray` that minimizes $\sum_{i=1}^N w_i (f_i - y_i)^2$ subject to $f_1 \le \dots \le f_N$ by accumulating `[weighted_sum, total_weight]` in each stack block. Verify that it conserves the weighted mean $\sum_i w_i \hat{f}_i = \sum_i w_i y_i$ and matches `pava_monotone_regression` when $w_i \equiv 1$.''',
        "ex3_code": r'''heading("Exercise 3: Implement Inverse-Variance Weighted PAVA From Scratch")
def weighted_pava_increasing(y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted isotonic regression minimizing sum_i w_i (f_i - y_i)^2 subject to f_1 <= ... <= f_N."""
    yv = np.asarray(y, dtype=float).ravel()
    wv = np.asarray(weights, dtype=float).ravel()
    blocks: list[list] = []
    for val, wt in zip(yv, wv):
        blocks.append([float(wt * val), float(wt), 1])
        while len(blocks) >= 2 and (blocks[-2][0] / blocks[-2][1]) > (blocks[-1][0] / blocks[-1][1]):
            sw_r, w_r, c_r = blocks.pop()
            blocks[-1][0] += sw_r
            blocks[-1][1] += w_r
            blocks[-1][2] += c_r
    return np.concatenate([np.full(c, sw / w) for sw, w, c in blocks])

w_weights = np.linspace(1.0, 5.0, 50)
q_wpava = weighted_pava_increasing(q_noisy, w_weights)
q_unif = weighted_pava_increasing(q_noisy, np.ones(50))
assert np.all(np.diff(q_wpava) >= -1e-15)
assert np.isclose(np.sum(w_weights * q_wpava), np.sum(w_weights * q_noisy), rtol=1e-14)
assert np.allclose(q_unif, q_pava, atol=1e-14)
print("Weighted PAVA weighted-mean residual:", abs(float(np.sum(w_weights * (q_wpava - q_noisy)))))''',
        "ex3_takeaway": r'''**Discussion.** Replacing uniform counts with inverse-variance weights $w_i = \sigma_i^{-2}$ generalizes PAVA to heteroscedastic thermodynamic datasets while preserving $O(N)$ complexity and exact weighted-mean conservation.''',
    },
    {
        "num": r'''04''',
        "slug": r'''04_monotone_splines_and_convex_free_energy_potentials''',
        "title": r'''4. Smooth monotone fits with I-splines''',
        "hook": r'''The monotone fit of Lecture 3 is a staircase, with zero slope on each step and no derivative at the jumps. A smooth monotone fit is obtained from I-splines, which are integrals of non-negative basis functions, with coefficients constrained to be non-negative. The lecture builds the basis, fits it by projected coordinate descent, and compares it with the staircase and with an unconstrained fit, including the cases where it does worse.''',
        "objectives": [
            r'''Construct the non-negative M-spline basis and the monotone I-spline basis (`linear_physics.ispline_basis_1d`).''',
            r'''Check that the derivative of each I-spline is the corresponding M-spline, by finite differences.''',
            r'''Fit smooth monotone curves by projected coordinate descent with non-negative coefficients (`linear_physics.fit_monotone_ispline`).''',
            r'''Build a convex Helmholtz energy from a monotone pressure curve.''',
        ],
        "theory_md": r'''## 1. Principles and governing equations
To obtain a smooth $C^\infty$ monotone surrogate $\hat{f}(x)$, **Ramsay's monotone spline framework (1988)** begins with $M$ non-negative localized bell-shaped **M-spline** basis functions $M_m(x) \ge 0$ centered at knots $\kappa_1 < \dots < \kappa_M$ with width $w$:
$$M_m(x) = \frac{1}{\sqrt{2\pi}\,w}\exp\!\left(-\frac{(x - \kappa_m)^2}{2w^2}\right) > 0, \qquad \int_{-\infty}^{\infty} M_m(u)\,du = 1 \tag{1.1}$$
Integrating each non-negative M-spline yields the monotonically increasing **I-spline** basis $I_m(x) \in (0, 1)$:
$$I_m(x) = \int_{-\infty}^{x} M_m(u)\,du = \frac{1}{2}\left[1 + \mathrm{erf}\!\left(\frac{x - \kappa_m}{\sqrt{2}\,w}\right)\right], \qquad \frac{dI_m(x)}{dx} = M_m(x) > 0 \tag{1.2}$$
Expanding our surrogate with a sign indicator $s \in \{+1, -1\}$ and **non-negative coefficients** $\beta_m \ge 0$ ($m=1,\dots,M$):
$$\hat{f}(x) = \beta_0 + s\sum_{m=1}^M \beta_m I_m(x), \qquad \beta_m \ge 0 \implies s\,\hat{f}'(x) = \sum_{m=1}^M \beta_m M_m(x) \ge 0 \quad \forall x \in \mathbb{R} \tag{1.3}$$
We solve $\min_{\beta_0 \in \mathbb{R},\,\boldsymbol{\beta}_{1:M} \ge \mathbf{0}} \frac{1}{2}\|\boldsymbol{\Phi}\boldsymbol{\beta} - \mathbf{y}\|_2^2 + \frac{\lambda N}{2}\|\boldsymbol{\beta}_{1:M}\|_2^2$ via projected coordinate descent onto the non-negative orthant $\beta_m \leftarrow \max(0, \tilde{\beta}_m)$.''',
        "demo1_code": r'''heading("Demonstration 1: Non-Negative M-Spline & Cumulative I-Spline Basis Verification")
x_grid = np.linspace(0.0, 10.0, 101)
knots = np.linspace(0.0, 10.0, 8)
I_mat, M_mat = linear_physics.ispline_basis_1d(x_grid, knots)

eps = 1e-5
I_plus, _ = linear_physics.ispline_basis_1d(x_grid + eps, knots)
I_minus, _ = linear_physics.ispline_basis_1d(x_grid - eps, knots)
dI_fd = (I_plus - I_minus) / (2.0 * eps)

print(f"Basis matrix shapes: I_mat = {I_mat.shape}, M_mat = {M_mat.shape}")
print(f"M-spline range     : [{np.min(M_mat):.4e}, {np.max(M_mat):.4f}] (Strictly non-negative)")
print(f"I-spline range     : [{np.min(I_mat):.4f}, {np.max(I_mat):.4f}] (Strictly bounded in [0, 1])")
print(f"Max derivative check ||dI/dx - M||_max: {np.max(np.abs(dI_fd - M_mat)):.2e}")
assert np.all(M_mat >= 0.0) and np.all((I_mat >= 0.0) & (I_mat <= 1.0))
assert np.allclose(dI_fd, M_mat, atol=1e-7)''',
        "post1_md": r'''**What Demonstration 1 shows.** Every column of $\mathbf{M}$ is non-negative and is the derivative of the corresponding column of $\mathbf{I}$ (to $2\times 10^{-11}$ by finite differences). A combination of I-splines with non-negative coefficients therefore has a non-negative, continuous derivative everywhere.

---

## 2. Numerical walkthrough
Let us fit a smooth monotone increasing I-spline surrogate to $N = 50$ noisy synthetic $\mathrm{CO_2}$ adsorption uptake points $q(P)$ and inspect the non-negative coefficient vector $\hat{\boldsymbol{\beta}}_{1:M} \ge \mathbf{0}$ found by projected coordinate descent.''',
        "demo2_code": r'''heading("Demonstration 2: Projected Coordinate Descent & Non-Negative Coefficient Spectrum")
rng = np.random.default_rng(42)
P_bar = np.linspace(0.1, 10.0, 50)
q_true = 5.2 * 1.4 * P_bar / (1.0 + 1.4 * P_bar)
q_obs = q_true + rng.normal(0.0, 0.18, size=50)

knots_ads = np.linspace(0.1, 10.0, 9)
isp_fit = linear_physics.fit_monotone_ispline(P_bar, q_obs, knots=knots_ads, increasing=True)
print("Intercept beta_0 :", round(float(isp_fit["beta"][0]), 4))
print("Non-negative I-spline weights beta_{1..9}:", np.round(isp_fit["beta"][1:], 4))
print(f"Min analytical slope dq/dP across [0.1, 10] bar: {np.min(isp_fit['dy_dx']):.4e} >= 0")
assert np.all(isp_fit["beta"][1:] >= 0.0) and np.all(isp_fit["dy_dx"] >= -1e-15)''',
        "post2_md": r'''**What Demonstration 2 shows.** Projected coordinate descent leaves four of the nine weights exactly at zero, on the boundary of the constraint, and the fitted isotherm has a slope $dq/dP$ of at least $0.028$ over the whole range from $0.1$ to $10\;\mathrm{bar}$.

---

## 3. Worked case study
Let us benchmark our smooth monotone I-spline against staircase PAVA and an unconstrained degree-8 polynomial on both **value accuracy** ($\mathrm{RMSE}(q)$) and **derivative accuracy** ($\mathrm{RMSE}(dq/dP)$).''',
        "demo3_code": r'''heading("Demonstration 3: Value & Derivative Benchmark — I-Spline vs. PAVA vs. Polynomial")
dq_dP_true = 5.2 * 1.4 / ((1.0 + 1.4 * P_bar) ** 2)
q_pava_ads = linear_physics.pava_monotone_regression(q_obs, increasing=True)
dq_dP_pava = np.gradient(q_pava_ads, P_bar)

poly8 = np.polyfit(P_bar, q_obs, deg=8)
q_poly8 = np.polyval(poly8, P_bar)
dq_dP_poly8 = np.polyval(np.polyder(poly8), P_bar)

for name, q_m, dq_m in [("Unconstrained Poly-8", q_poly8, dq_dP_poly8), ("Staircase PAVA", q_pava_ads, dq_dP_pava), ("Smooth I-Spline", isp_fit["y_pred"], isp_fit["dy_dx"])]:
    v_rmse = float(np.sqrt(np.mean((q_m - q_true) ** 2)))
    d_rmse = float(np.sqrt(np.mean((dq_m - dq_dP_true) ** 2)))
    n_viol = int(np.sum(dq_m < -1e-12))
    print(f"{name:22s} | Value RMSE = {v_rmse:.4f} | Slope RMSE = {d_rmse:.4f} | Slope Violations = {n_viol}")
assert np.sqrt(np.mean((isp_fit["y_pred"] - q_true) ** 2)) < 0.12
assert int(np.sum(isp_fit["dy_dx"] < -1e-12)) == 0 and int(np.sum(dq_dP_poly8 < -1e-12)) > 0''',
        "post3_md": r'''**What Demonstration 3 shows.** The comparison does not favour the constrained fits on accuracy. The unconstrained polynomial of degree 8 has the smallest error in the value ($0.047$) and in the slope ($0.23$), but its slope has the wrong sign in 5 places. The monotone fits have no violation and larger errors: $0.083$ and $0.43$ for the staircase, $0.100$ and $0.56$ for the I-spline with this set of knots. What the I-spline offers over the staircase is a continuous derivative, which a process simulator needs; what it costs here is resolution, and Demonstration 4 shows that more knots recover it.

---

## 4. What if: sensitivity to the parameters
How does the number of knots $M \in \{4, 8, 14, 22\}$ affect the smooth I-spline fit, and how does integrating a decreasing pressure I-spline $\hat{P}(V_m)$ produce a guaranteed **convex Helmholtz free-energy potential** $\hat{A}_m(V_m)$ with $\partial^2 \hat{A}_m / \partial V_m^2 = -\partial \hat{P}/\partial V_m \ge 0$?''',
        "demo4_code": r'''heading("Demonstration 4: Knot Density Sweep & Convex Helmholtz Free-Energy Curvature")
for n_k in [4, 8, 14, 22]:
    fit_k = linear_physics.fit_monotone_ispline(P_bar, q_obs, knots=np.linspace(0.1, 10.0, n_k), increasing=True)
    v_err = float(np.sqrt(np.mean((fit_k["y_pred"] - q_true) ** 2)))
    print(f"Knots M = {n_k:2d} | Active positive weights = {int(np.sum(fit_k['beta'][1:] > 1e-5)):2d} | Value RMSE = {v_err:.4f} | Min slope = {np.min(fit_k['dy_dx']):.2e}")

Vm = np.linspace(0.15, 0.80, 45)
P_iso_noisy = 25.0 / Vm + rng.normal(0.0, 1.5, size=45)
isp_p = linear_physics.fit_monotone_ispline(Vm, P_iso_noisy, knots=np.linspace(0.15, 0.80, 8), increasing=False)
d2A_dVm2 = -isp_p["dy_dx"]
print(f"\nMin Helmholtz free-energy curvature d^2 A_m / dV_m^2 = {np.min(d2A_dVm2):.4e} >= 0 (Strictly convex)")
assert np.all(d2A_dVm2 >= -1e-15)''',
        "post4_md": r'''**What Demonstration 4 shows.** With more knots the error falls, from $0.25$ with 4 knots to $0.061$ with 22, and the slope stays non-negative in every case: the constraint $\beta_m \ge 0$ prevents the oscillations that an unconstrained fit of the same flexibility would show. The second derivative of the Helmholtz energy of the last example is positive everywhere.

---

## Inside the algorithm
Below we run a standalone projected coordinate descent loop on the non-negative orthant in pure NumPy and verify digit-for-digit agreement with `linear_physics.fit_monotone_ispline`.''',
        "inside_code": r'''heading("Inside the Algorithm: By-Hand Projected Coordinate Descent vs. the Library")
I_b, M_b = linear_physics.ispline_basis_1d(P_bar, knots_ads)
Phi = np.column_stack([np.ones_like(P_bar), I_b])
n_s, p_s = Phi.shape
beta_hand = np.zeros(p_s)
beta_hand[0] = float(np.mean(q_obs))
col_sq = np.sum(Phi ** 2, axis=0) + np.concatenate([[0.0], np.full(p_s - 1, 1e-6 * n_s)])
res_vec = q_obs - Phi @ beta_hand
for _ in range(800):
    b_prev = beta_hand.copy()
    for j in range(p_s):
        old_v = beta_hand[j]
        rho = float(np.dot(Phi[:, j], res_vec)) + float(np.sum(Phi[:, j] ** 2)) * old_v
        new_v = (rho / col_sq[j]) if j == 0 else max(0.0, rho / col_sq[j])
        if new_v != old_v:
            res_vec -= Phi[:, j] * (new_v - old_v)
            beta_hand[j] = new_v
    if np.max(np.abs(beta_hand - b_prev)) < 1e-10:
        break
diff_isp = float(np.max(np.abs(beta_hand - isp_fit["beta"])))
print("by hand (pure NumPy) beta[:5]:", np.round(beta_hand[:5], 6))
print("library (engml_2_applied)      beta[:5]:", np.round(isp_fit["beta"][:5], 6))
print(f"Max digit-for-digit discrepancy: {diff_isp:.2e}")
assert diff_isp < 1e-13''',
        "ex1_prompt": r'''Fit a decreasing smooth monotone I-spline (`increasing=False`) to noisy shear-thinning polymer viscosity measurements $\log_{10}\eta(\dot{\gamma})$ across $\log_{10}\dot{\gamma} \in [-1, 3]$ generated from a Carreau–Yasuda fluid $\eta(\dot{\gamma}) = \eta_0 (1 + (\lambda\dot{\gamma})^2)^{(n-1)/2}$ ($\eta_0 = 1000\;\mathrm{Pa\cdot s}, \lambda = 1.0\;\mathrm{s}, n = 0.35$) with noise $\sigma = 0.08$. Verify that the fitted slope $d(\log_{10}\hat{\eta})/d(\log_{10}\dot{\gamma}) \le 0$ at every point and that the test RMSE against the true viscosity curve is below $0.05$.''',
        "ex1_code": r'''heading("Exercise 1: Carreau-Yasuda Shear-Thinning Polymer Viscosity I-Spline")
rng_rheo = np.random.default_rng(19)
log_gdot = np.linspace(-1.0, 3.0, 45)
gdot = 10.0 ** log_gdot
log_eta_true = np.log10(1000.0 * ((1.0 + gdot ** 2) ** ((0.35 - 1.0) / 2.0)))
log_eta_noisy = log_eta_true + rng_rheo.normal(0.0, 0.08, size=45)

rheo_isp = linear_physics.fit_monotone_ispline(log_gdot, log_eta_noisy, knots=np.linspace(-1.0, 3.0, 8), increasing=False)
rmse_rheo = float(np.sqrt(np.mean((rheo_isp["y_pred"] - log_eta_true) ** 2)))
print(f"Shear-thinning I-spline RMSE = {rmse_rheo:.4f} | Max slope = {np.max(rheo_isp['dy_dx']):.4e} <= 0")
assert np.all(rheo_isp["dy_dx"] <= 1e-15) and rmse_rheo < 0.05''',
        "ex1_exp": r'''Non-negative I-spline weights with `increasing=False` guarantee $d(\log_{10}\eta)/d(\log_{10}\dot{\gamma}) \le 0$ across both the Newtonian plateau and the power-law shear-thinning regime.''',
        "ex2_prompt": r'''Compare Newton–Raphson inversion $\hat{q}(P^*) = q_{\mathrm{target}} = 3.0\;\mathrm{mol/kg}$ on the smooth I-spline surrogate (`isp_fit`) versus the staircase PAVA curve. Show that Newton–Raphson on the smooth I-spline converges to $|\hat{q}(P^*) - 3.0| < 10^{-12}$ in $\le 6$ iterations.''',
        "ex2_code": r'''heading("Exercise 2: Newton-Raphson Process Inversion on Smooth I-Spline Surrogate")
q_target = 3.0
P_curr = 2.0
b_isp = isp_fit["beta"]
for it in range(1, 8):
    I_c, M_c = linear_physics.ispline_basis_1d(np.array([P_curr]), knots_ads)
    q_val = float(b_isp[0] + I_c[0] @ b_isp[1:])
    dq_val = float(M_c[0] @ b_isp[1:])
    res = q_val - q_target
    P_curr -= res / dq_val
    print(f"Newton iter {it} | P = {P_curr:.8f} bar | q(P) - q_target = {res:+.2e} | slope dq/dP = {dq_val:.4f}")
    if abs(res) < 1e-12:
        break
assert abs(res) < 1e-12 and it <= 6''',
        "ex2_exp": r'''Because the I-spline derivative $dq/dP = \sum_m \beta_m M_m(P) > 0$ is smooth and strictly positive, Newton's method converges in six iterations, with the residual falling from $10^{-3}$ to $10^{-6}$ to $10^{-12}$ in the last three.''',
        "ex3_prompt": r'''Implement a **C-spline** (convex spline) basis generator `convex_cspline_basis_1d(x, knots)` by integrating the monotone I-spline basis $C_m(x) = \int_{x_0}^x I_m(u)\,du$ via cumulative trapezoidal quadrature so that $C_m''(x) = M_m(x) \ge 0$, and verify that any non-negative combination $\hat{A}(x) = \sum_{m=1}^M \beta_m C_m(x)$ ($\beta_m > 0$) has strictly positive second derivative $\hat{A}''(x) > 0$ everywhere.''',
        "ex3_code": r'''heading("Exercise 3: Implement Convex C-Spline Free-Energy Basis From Scratch")
def convex_cspline_basis_1d(x: np.ndarray, knots: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return convex C-spline basis C_mat, its 1st derivative I_mat, and 2nd derivative M_mat >= 0."""
    xv = np.asarray(x, dtype=float).ravel()
    I_m, M_m = linear_physics.ispline_basis_1d(xv, knots)
    dx = np.diff(xv)[:, None]
    trapz_inc = 0.5 * (I_m[:-1, :] + I_m[1:, :]) * dx
    C_m = np.vstack([np.zeros((1, len(knots))), np.cumsum(trapz_inc, axis=0)])
    return C_m, I_m, M_m

C_b, I_b_c, M_b_c = convex_cspline_basis_1d(Vm, np.linspace(0.15, 0.80, 8))
weights_pos = np.ones(8)
A_potential = C_b @ weights_pos
d2A_num = np.gradient(np.gradient(A_potential, Vm), Vm)
print(f"Min analytical curvature sum(beta_m M_m): {np.min(M_b_c @ weights_pos):.4e} > 0")
print(f"Min numerical 2nd derivative d^2A/dVm^2 : {np.min(d2A_num):.4e} > 0")
assert np.all((M_b_c @ weights_pos) > 0.0) and np.all(d2A_num > 0.0)''',
        "ex3_takeaway": r'''**Discussion.** Integrating M-splines once yields monotone I-splines ($I_m' = M_m \ge 0$); integrating twice yields convex C-splines ($C_m'' = M_m \ge 0$) suitable for parameterizing convex thermodynamic free energies $A_m(V_m)$ and $G_m(x)$.''',
    },
    {
        "num": r'''05''',
        "slug": r'''05_outlier_robust_regression_huber_irls_and_quantile_loss''',
        "title": r'''5. Robust regression: the Huber loss and quantile regression''',
        "hook": r'''Least squares penalizes a residual by its square, so a few bad readings can move the whole fit. The Huber loss is quadratic for small residuals and linear for large ones, and it is minimized by iteratively reweighted least squares. The same reweighting with an asymmetric loss fits quantiles in place of the mean, which gives a band around the fit without assuming a noise distribution.''',
        "objectives": [
            r'''State the Huber loss and derive the weights of iteratively reweighted least squares (`linear_physics.huber_irls_regression`).''',
            r'''Compare the Huber fit with least squares on synthetic reactor data with spikes.''',
            r'''Fit the 10 %, 50 % and 90 % quantiles with the pinball loss (`linear_physics.quantile_irls_regression`).''',
            r'''Examine how the fit depends on the threshold of the Huber loss and on the fraction of outliers.''',
        ],
        "theory_md": r'''## 1. Principles and governing equations
When sensor errors follow a heavy-tailed or spike-contaminated distribution $(1-\epsilon)\mathcal{N}(0,\sigma^2) + \epsilon\,\Delta_{\mathrm{spike}}$, the maximum-likelihood estimator replaces quadratic loss $\frac{1}{2}r^2$ with a linear-tail loss function:
1. **Huber loss & adaptive IRLS (Huber, 1964):** Quadratic for small Gaussian inliers ($|r| \le \delta$) and linear for gross outliers ($|r| > \delta$):
   $$\mathcal{L}_\delta(r) = \begin{cases} \frac{1}{2}r^2 & |r| \le \delta \\ \delta\left(|r| - \frac{1}{2}\delta\right) & |r| > \delta \end{cases}, \qquad \psi_\delta(r) = \mathcal{L}_\delta'(r) = \mathrm{clip}(r, -\delta, +\delta) \tag{1.1}$$
   Writing the stationarity condition $\sum_{i=1}^N \psi_\delta(r_i)\mathbf{x}_i = \mathbf{0}$ as a weighted normal equation $\mathbf{X}^\top \mathbf{W}^{(t)}\mathbf{X}\mathbf{w}^{(t+1)} = \mathbf{X}^\top \mathbf{W}^{(t)}\mathbf{y}$ yields the **iteratively reweighted least squares (IRLS)** weight update:
   $$w_i^{(t)} = \frac{\psi_\delta(r_i^{(t)})}{r_i^{(t)}} = \min\!\left(1,\, \frac{\delta}{|r_i^{(t)}|}\right) \in (0, 1] \tag{1.2}$$
2. **Pinball check loss (quantile regression, Koenker & Bassett 1978):** For target percentile $\tau \in (0, 1)$, minimizing the asymmetric check loss estimates the conditional $100\tau$-th quantile:
   $$\rho_\tau(r) = \max(\tau r,\, (\tau - 1)r) = \begin{cases} \tau |r| & r \ge 0 \\ (1-\tau)|r| & r < 0 \end{cases}, \qquad w_{\tau,i}^{(t)} = \frac{\tau\,\mathbb{I}(r_i \ge 0) + (1-\tau)\,\mathbb{I}(r_i < 0)}{\max(|r_i^{(t)}|, \varepsilon)} \tag{1.3}$$''',
        "demo1_code": r'''heading("Demonstration 1: Huber Loss vs. Quadratic L2 Loss & IRLS Weighting")
rng = np.random.default_rng(42)
N = 45
T_norm = np.linspace(-2.0, 2.0, N)
X_des = np.column_stack([np.ones(N), T_norm])
w_true = np.array([12.0, 3.5])
y_clean = X_des @ w_true + rng.normal(0.0, 0.25, size=N)

# Corrupt 3 sensor readings with spikes of +25.0 (~100 sigma):
y_spiked = y_clean.copy()
spike_idx = [5, 22, 38]
y_spiked[spike_idx] += 25.0

w_ols = linear_physics.ridge_regression(X_des, y_spiked, lam=0.0)
huber_res = linear_physics.huber_irls_regression(X_des, y_spiked, delta=1.0)
w_huber = huber_res["w"]

print("True parameters   [intercept, slope]:", w_true)
print("Corrupted OLS     [intercept, slope]:", np.round(w_ols, 4), f"(L2 error = {np.linalg.norm(w_ols - w_true):.4f})")
print("Robust Huber IRLS [intercept, slope]:", np.round(w_huber, 4), f"(L2 error = {np.linalg.norm(w_huber - w_true):.4f})")
assert np.linalg.norm(w_huber - w_true) < 0.15 * np.linalg.norm(w_ols - w_true)''',
        "post1_md": r'''**What Demonstration 1 shows.** Three spikes of $+25$ raise the least-squares intercept by $1.68$. The Huber fit returns `[12.092, 3.517]`, an error of $0.093$ against $1.681$, which is $94\ \%$ smaller. The data are synthetic.

---

## 2. Numerical walkthrough
Let us inspect the converged IRLS sample weights $w_i = \min(1, \delta/|r_i|)$ assigned to the 3 corrupted spike observations versus the 42 normal inlier observations.''',
        "demo2_code": r'''heading("Demonstration 2: Inspecting Converged IRLS Sample Weights on Inliers vs. Spikes")
weights_conv = huber_res["sample_weights"]
res_conv = huber_res["residuals"]
inlier_mask = np.ones(N, dtype=bool)
inlier_mask[spike_idx] = False

print(f"Mean IRLS weight across 42 normal inliers: {np.mean(weights_conv[inlier_mask]):.4f}")
for idx in spike_idx:
    print(f"Spike sample #{idx:2d} | Residual r_i = {res_conv[idx]:+7.3f} | Converged IRLS weight w_i = {weights_conv[idx]:.4f}")
assert np.all(weights_conv[spike_idx] < 0.08) and np.mean(weights_conv[inlier_mask]) > 0.98''',
        "post2_md": r'''**What Demonstration 2 shows.** Every normal inlier has $|r_i| \le \delta = 1.0$ and receives weight $w_i = 1.0000$, while the three $+25$ spikes receive weight $w_i \approx 1.0 / 25.0 = 0.040$, capping their influence to $\pm\delta$.

---

## 3. Worked case study
Now let us use **pinball quantile regression** (`linear_physics.quantile_irls_regression`) at $\tau \in \{0.10, 0.50, 0.90\}$ to construct a robust median calibration line ($\tau=0.50$) and an $80\%$ process operating envelope ($[\hat{y}_{0.10}, \hat{y}_{0.90}]$) directly on the spike-corrupted dataset.''',
        "demo3_code": r'''heading("Demonstration 3: Pinball Quantile Envelopes (10th, 50th Median, 90th Percentiles)")
w_q10 = linear_physics.quantile_irls_regression(X_des, y_spiked, tau=0.10)
w_q50 = linear_physics.quantile_irls_regression(X_des, y_spiked, tau=0.50)
w_q90 = linear_physics.quantile_irls_regression(X_des, y_spiked, tau=0.90)

y_q10, y_q50, y_q90 = X_des @ w_q10, X_des @ w_q50, X_des @ w_q90
print("10th Percentile weights (tau=0.10):", np.round(w_q10, 4))
print("50th Median L1 weights  (tau=0.50):", np.round(w_q50, 4), f"(L2 error vs true = {np.linalg.norm(w_q50 - w_true):.4f})")
print("90th Percentile weights (tau=0.90):", np.round(w_q90, 4))
assert np.all(y_q10 < y_q50) and np.all(y_q50 < y_q90)
assert np.linalg.norm(w_q50 - w_true) < 0.20 * np.linalg.norm(w_ols - w_true)''',
        "post3_md": r'''**What Demonstration 3 shows.** The median regression ($\tau = 0.5$) depends on the signs of the residuals and not on their sizes, so an outlier in $y$, however large, counts as one point above the line. Its error against the true parameters is $0.044$. The fits for $\tau = 0.1$ and $0.9$ bracket it and give a band for the data.

---

## 4. What if: sensitivity to the parameters
How sensitive is Huber IRLS to the transition threshold $\delta$? Let us sweep $\delta \in [0.1, 50.0]$ and watch the smooth transition from $L_1$-like median behavior ($\delta \to 0$) to non-robust OLS behavior ($\delta \to \infty$).''',
        "demo4_code": r'''heading("Demonstration 4: Huber Threshold delta Sweep (From L1 Limit to OLS Limit)")
print(f"{'Huber delta':>12s} | {'Intercept w_0':>13s} | {'Slope w_1':>10s} | {'Param L2 Error':>14s} | {'Spike Weight':>12s}")
print("-" * 70)
for d_val in [0.1, 0.3, 1.0, 5.0, 50.0]:
    res_d = linear_physics.huber_irls_regression(X_des, y_spiked, delta=d_val)
    wh = res_d["w"]
    print(f"{d_val:12.1f} | {wh[0]:13.4f} | {wh[1]:10.4f} | {np.linalg.norm(wh - w_true):14.4f} | {res_d['sample_weights'][spike_idx[0]]:12.4f}")''',
        "post4_md": r'''**What Demonstration 4 shows.** For thresholds $\delta$ from $0.1$ to $1$ the error stays below $0.1$ and the spikes receive weights of $0.04$ or less. At $\delta = 5$ the spikes keep a weight of $0.21$ and the error rises to $0.38$; at $\delta = 50$ every weight is one and the fit is least squares. A common choice is $\delta = 1.345\,\hat{\sigma}$ with $\hat{\sigma} = 1.4826\,\mathrm{MAD}$ of the residuals, which loses about $5\ \%$ in efficiency when the noise is Gaussian.

---

## Inside the algorithm
Below we run a standalone 10-line Huber IRLS solver in pure NumPy and confirm digit-for-digit agreement with `linear_physics.huber_irls_regression`.''',
        "inside_code": r'''heading("Inside the Algorithm: By-Hand Huber IRLS Loop vs. the Library")
w_hand = np.linalg.lstsq(X_des, y_spiked, rcond=None)[0]
for _ in range(100):
    w_prev = w_hand.copy()
    r_abs = np.abs(y_spiked - X_des @ w_hand)
    sw = np.where(r_abs <= 1.0, 1.0, 1.0 / np.maximum(r_abs, 1e-15))
    W_half = np.sqrt(sw)[:, None]
    w_hand = np.linalg.lstsq(X_des * W_half, y_spiked * W_half.ravel(), rcond=None)[0]
    if np.max(np.abs(w_hand - w_prev)) < 1e-9:
        break
diff_huber = float(np.max(np.abs(w_hand - w_huber)))
print("by hand (pure NumPy):", np.round(w_hand, 6))
print("library (engml_2_applied)     :", np.round(w_huber, 6))
print(f"Max discrepancy     : {diff_huber:.2e}")
assert diff_huber < 1e-14''',
        "ex1_prompt": r'''Verify on `y_spiked` that the empirical fraction of observations falling below the fitted $\tau = 0.50$ median regression line (`w_q50`) lies within $0.50 \pm 0.08$, and that the $\tau = 0.50$ median slope is within $0.15$ of the true slope $3.5$ despite the three $+25.0$ gross outliers.''',
        "ex1_code": r'''heading("Exercise 1: Quantile Split Balance & Median Slope Verification")
frac_below_50 = float(np.mean(y_spiked <= (X_des @ w_q50)))
slope_err_50 = abs(float(w_q50[1]) - float(w_true[1]))
print(f"Empirical fraction <= median line = {frac_below_50:.3f} | Median slope error = {slope_err_50:.4f}")
assert abs(frac_below_50 - 0.50) <= 0.08 and slope_err_50 < 0.15''',
        "ex1_exp": r'''By first-order optimality of the Pinball check loss $\rho_\tau(r)$, an exact $\tau$-quantile regression line partitions the dataset so that approximately $100\tau\%$ of the observations lie at or below the fitted line.''',
        "ex2_prompt": r'''Demonstrate the **high-leverage X-outlier vulnerability** of standard M-estimators: corrupt both the input $x_{44} = 2.0 \to 25.0$ and target $y_{44} \to -100.0$ (a simultaneous input-output leverage point) and compare `huber_irls_regression` with and without a $3\sigma_x$ input leverage pre-filter.''',
        "ex2_code": r'''heading("Exercise 2: High-Leverage X-Outlier Breakdown Audit & Leverage Guardrail")
X_lev = X_des.copy()
y_lev = y_spiked.copy()
X_lev[-2:, 1] = 250.0
y_lev[-2:] = -200.0
w_huber_unfiltered = linear_physics.huber_irls_regression(X_lev, y_lev, delta=1.0)["w"]

# Guardrail: Median + MAD input domain screen |x - med(x)| <= 3 * 1.4826 * MAD(x):
x_col = X_lev[:, 1]
mad_x = 1.4826 * float(np.median(np.abs(x_col - np.median(x_col))))
keep = np.abs(x_col - np.median(x_col)) <= 3.0 * mad_x
w_huber_guarded = linear_physics.huber_irls_regression(X_lev[keep], y_lev[keep], delta=1.0)["w"]
print("Unfiltered Huber on X-leverage outlier:", np.round(w_huber_unfiltered, 3))
print("MAD-Guarded Huber on X-leverage data  :", np.round(w_huber_guarded, 3), "(True = [12.0, 3.5])")
assert np.linalg.norm(w_huber_guarded - w_true) < 0.25 and np.linalg.norm(w_huber_unfiltered - w_true) > 1.5''',
        "ex2_exp": r'''While Huber and $L_1$ loss bound the influence of vertical $y$-outliers, their influence function $\psi_\delta(r)\mathbf{x}_i$ is still proportional to $\mathbf{x}_i$. Pairing Huber IRLS with an input domain guardrail (MAD or Mahalanobis OOD screen) protects against both $x$- and $y$-outliers.''',
        "ex3_prompt": r'''Implement a function `auto_mad_huber_regression(X: np.ndarray, y: np.ndarray, c: float = 1.345) -> dict` that computes an initial $\tau=0.50$ median regression fit, estimates the robust noise scale $\hat{\sigma}_{\mathrm{MAD}} = 1.4826\,\mathrm{median}(|\mathbf{r}_0 - \mathrm{median}(\mathbf{r}_0)|)$, and runs `huber_irls_regression` with the self-tuning threshold $\delta = c\,\hat{\sigma}_{\mathrm{MAD}}$. Verify it on `y_spiked`.''',
        "ex3_code": r'''heading("Exercise 3: Implement Self-Tuning MAD-Scaled Huber Regression")
def auto_mad_huber_regression(X: np.ndarray, y: np.ndarray, c: float = 1.345) -> dict:
    """Self-tuning Huber regression using median-regression MAD scale estimation."""
    w_init = linear_physics.quantile_irls_regression(X, y, tau=0.50)
    r0 = y - X @ w_init
    sigma_mad = 1.4826 * float(np.median(np.abs(r0 - np.median(r0))))
    delta_auto = max(1e-6, c * sigma_mad)
    res = linear_physics.huber_irls_regression(X, y, delta=delta_auto)
    res["sigma_mad"] = sigma_mad
    res["delta_auto"] = delta_auto
    return res

auto_h = auto_mad_huber_regression(X_des, y_spiked)
print(f"Estimated robust scale sigma_MAD = {auto_h['sigma_mad']:.4f} (True sigma = 0.2500)")
print(f"Self-tuned Huber delta           = {auto_h['delta_auto']:.4f} | Fitted w = {np.round(auto_h['w'], 4)}")
assert abs(auto_h["sigma_mad"] - 0.25) < 0.08 and np.linalg.norm(auto_h["w"] - w_true) < 0.15''',
        "ex3_takeaway": r'''**Discussion.** Scaling Huber's threshold via the median absolute deviation ($\delta = 1.345 \times 1.4826\,\mathrm{MAD}$) makes the fit independent of the units of the response and leaves no threshold to choose by hand. Here the estimated scale is $0.224$ (true value $0.25$).''',
    },
]
