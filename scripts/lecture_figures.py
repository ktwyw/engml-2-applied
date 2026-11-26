"""Figure cells of the 18 lectures.

``FIGURES[num]`` is a list of ``(after, code, caption)``: the code cell is placed after the
demonstration named by ``after`` (``"demo1"`` ... ``"demo4"``) and uses the variables that
the demonstrations have defined; the caption is written below the figure.
"""

FIGURES = {}

FIGURES["00"] = [
    ("demo3", r'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
species = [r"CH$_3$OH", "CO", r"H$_2$O", r"CO$_2$", r"H$_2$"]
xi1_line = np.linspace(0.0, 0.5, 51)
W_line = w_feed[None, :] + xi1_line[:, None] * nu[0][None, :] + 0.08 * nu[1][None, :]
for j, name in enumerate(species):
    ax1.plot(xi1_line, W_line[:, j], label=name)
ax1.set_xlabel(r"extent of the methanol reaction, $\xi_1$ (mol)"); ax1.set_ylabel("amount in the effluent (mol)")
ax1.legend(ncol=2, fontsize=9, loc="upper right"); panel_label(ax1, "(a)", x=0.03, y=0.55)
ax2.bar(["C", "H", "O"], atom_violation, color=["0.3", "C0", "C3"], width=0.5)
ax2.axhline(0.0, color="k", lw=0.8); ax2.tick_params(axis="x", which="minor", bottom=False, top=False)
ax2.set_ylabel("element imbalance of the noisy analysis (mol)"); panel_label(ax2, "(b)")
plt.show()
''',
     "(a) The effluent of the reactor as the extent of the methanol reaction varies, with the extent of the "
     "second reaction fixed at 0.08 mol: every composition on these lines conserves carbon, hydrogen and oxygen. "
     "(b) A synthetic analysis with noise of 0.05 mol on each species does not lie on them: the imbalance in "
     "each element."),
    ("demo4", r'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
sig_line = np.logspace(-3, -1, 21); raw_a, raw_m, prj_a = [], [], []
for sig in sig_line:
    w_raw = w_true + sig * unit_noise
    w_proj = w_raw - C_pinv @ (C_atom @ w_raw - d_atom)
    raw_a.append(np.max(np.abs(C_atom @ w_raw - d_atom))); raw_m.append(abs(molar_masses @ w_raw - feed_mass))
    prj_a.append(max(np.max(np.abs(C_atom @ w_proj - d_atom)), 1e-17))
ax1.loglog(sig_line, raw_a, label="element balance, raw (mol)"); ax1.loglog(sig_line, raw_m, "--", label="mass balance, raw (g)")
ax1.loglog(sig_line, prj_a, ":", color="k", label="element balance, projected (mol)")
ax1.set_xlabel("noise of the analysis (mol)"); ax1.set_ylabel("violation of the balance")
ax1.legend(fontsize=8.5, loc="center right"); panel_label(ax1, "(a)")
pos = np.arange(5); width = 0.27
ax2.bar(pos - width, w_raw - w_true, width, color="C3", label="raw analysis")
ax2.bar(pos, w_proj - w_true, width, color="C0", label="after projection")
ax2.axhline(0.0, color="k", lw=0.8); ax2.set_xticks(pos - width / 2)
ax2.set_xticklabels([r"CH$_3$OH", "CO", r"H$_2$O", r"CO$_2$", r"H$_2$"], fontsize=9); ax2.tick_params(axis="x", which="minor", bottom=False, top=False)
ax2.set_ylabel("error in the amount (mol), noise 0.1 mol"); ax2.legend(fontsize=9); panel_label(ax2, "(b)", x=0.45)
plt.show()
''',
     "(a) The violation of the balances by the raw analysis grows in proportion to its noise; after the "
     "projection it is at rounding level. (b) The error in each species before and after the projection, at the "
     "largest noise. The projection removes the part of the error that breaks the balances and leaves the part "
     "that looks like a change of reaction extent."),
]

FIGURES["01"] = [
    ("demo1", r'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
species = [r"CH$_3$OH", "CO", r"H$_2$O", r"CO$_2$", r"H$_2$"]
pos = np.arange(5); width = 0.38
ax1.bar(pos - width / 2, w_ols - w_true, width, color="C3", label="least squares")
ax1.bar(pos + width / 2, w_kkt - w_true, width, color="C0", label="with element balances")
ax1.axhline(0.0, color="k", lw=0.8); ax1.set_xticks(pos); ax1.set_xticklabels(species, fontsize=9)
ax1.tick_params(axis="x", which="minor", bottom=False, top=False)
ax1.set_ylabel("estimated minus true yield (mol)"); ax1.legend(fontsize=9, loc="lower left"); panel_label(ax1, "(a)", x=0.45)
elements = ["C", "H", "O"]; pos3 = np.arange(3)
ax2.bar(pos3 - width / 2, np.abs(C_atom @ w_ols - d_atom), width, color="C3", label="least squares")
ax2.bar(pos3 + width / 2, np.maximum(np.abs(C_atom @ w_kkt - d_atom), 1e-17), width, color="C0", label="with element balances")
ax2.set_yscale("log"); ax2.set_ylim(1e-17, 1.0); ax2.set_xticks(pos3); ax2.set_xticklabels(elements)
ax2.tick_params(axis="x", which="minor", bottom=False, top=False)
ax2.set_ylabel("violation of the element balance (mol)"); ax2.legend(fontsize=9, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2); ax2.set_title("(b)", loc="left", fontweight="bold")
plt.show()
''',
     "Estimates of five yields from synthetic data, without and with the three element balances as exact "
     "constraints. (a) The error in each yield. (b) The violation of each element balance, on a logarithmic "
     "scale: about 0.1 mol without the constraints and at rounding level with them."),
    ("demo4", r'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
mu_line = np.logspace(0, 14, 43); cond_s, res_s, diff_s = [], [], []
for mu_pen in mu_line:
    H_soft = G + mu_pen * (C_atom.T @ C_atom)
    w_soft = np.linalg.solve(H_soft, X_scan.T @ y_obs + mu_pen * (C_atom.T @ d_atom))
    cond_s.append(np.linalg.cond(H_soft)); res_s.append(np.max(np.abs(C_atom @ w_soft - d_atom))); diff_s.append(np.linalg.norm(w_soft - w_kkt))
ax1.loglog(mu_line, res_s, label="violation of the balances (mol)")
ax1.loglog(mu_line, diff_s, "--", label="distance from the constrained solution")
ax1.set_xlabel(r"penalty weight $\mu_{\mathrm{pen}}$"); ax1.set_ylabel("magnitude"); ax1.legend(fontsize=8.5, loc="lower left")
panel_label(ax1, "(a)", x=0.9)
ax2.loglog(mu_line, cond_s, label="penalty method")
ax2.axhline(np.linalg.cond(np.block([[G, C_atom.T], [C_atom, np.zeros((3, 3))]])), color="C3", ls="--", label="KKT system")
ax2.axhline(1.0 / np.finfo(float).eps, color="0.5", ls=":", lw=1.0)
ax2.set_xlabel(r"penalty weight $\mu_{\mathrm{pen}}$"); ax2.set_ylabel("condition number"); ax2.legend(loc="center right"); panel_label(ax2, "(b)")
plt.show()
''',
     "A soft penalty in place of the exact constraint. (a) The violation of the balances falls as the penalty "
     "weight grows, but the solution first approaches the constrained one and then moves away again, as "
     "rounding errors take over. (b) The reason: the condition number of the penalized matrix grows in "
     "proportion to the weight and approaches the limit of double precision (dotted), while that of the KKT "
     "system (dashed) does not depend on any weight."),
]

FIGURES["02"] = [
    ("demo2", r'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
pos = np.arange(5); width = 0.38
ax1.bar(pos - width / 2, y_normal - y_true_net, width, yerr=sigma_net, color="C3", ecolor="0.3", label="measured (bar: one standard deviation)")
ax1.bar(pos + width / 2, rec_normal["y_reconciled"] - y_true_net, width, color="C0", label="reconciled")
ax1.axhline(0.0, color="k", lw=0.8); ax1.set_xticks(pos); ax1.set_xticklabels([f"stream {i}" for i in range(1, 6)], fontsize=9)
ax1.tick_params(axis="x", which="minor", bottom=False, top=False)
ax1.set_ylabel("flow minus true flow (kg/h)"); ax1.legend(fontsize=8.5, loc="lower left"); panel_label(ax1, "(a)")
nodes = ["mixer", "reactor", "separator"]; pos3 = np.arange(3)
ax2.bar(pos3 - width / 2, np.abs(A_net @ y_normal), width, color="C3", label="measured")
ax2.bar(pos3 + width / 2, np.maximum(np.abs(A_net @ rec_normal["y_reconciled"]), 1e-16), width, color="C0", label="reconciled")
ax2.set_yscale("log"); ax2.set_ylim(1e-16, 30.0); ax2.set_xticks(pos3); ax2.set_xticklabels(nodes)
ax2.tick_params(axis="x", which="minor", bottom=False, top=False)
ax2.set_ylabel("imbalance of the unit (kg/h)"); ax2.legend(fontsize=9, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2); ax2.set_title("(b)", loc="left", fontweight="bold")
plt.show()
''',
     "Reconciliation of five measured flows around a mixer, a reactor and a separator (synthetic data). (a) The "
     "error of each flow before and after; the meters with the larger standard deviations are adjusted more. "
     "(b) The imbalance of each unit, on a logarithmic scale: of the order of 1 kg/h in the measurements, and "
     "at rounding level after reconciliation."),
    ("demo4", r'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
ax1.bar(np.arange(1, 6) - 0.19, rec_normal["measurement_z"], 0.38, color="C0", label="noise only")
ax1.bar(np.arange(1, 6) + 0.19, rec_faulty["measurement_z"], 0.38, color="C3", label="stream 3 biased by 12 kg/h")
ax1.axhline(1.96, color="k", ls=":", lw=1.0)
ax1.set_xlabel("stream"); ax1.set_ylabel("measurement test statistic $|z|$"); ax1.tick_params(axis="x", which="minor", bottom=False, top=False)
ax1.legend(fontsize=9, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2); ax1.set_title("(a)", loc="left", fontweight="bold")
bias_line = np.linspace(0.0, 12.0, 49); chi_line, z3_line, z2_line = [], [], []
for bias in bias_line:
    y_sw = y_normal.copy(); y_sw[2] += bias
    r_sw = linear_physics.crowe_data_reconciliation(y_sw, sigma_net, A_net, b_net)
    chi_line.append(r_sw["chi2_stat"]); z3_line.append(r_sw["measurement_z"][2]); z2_line.append(r_sw["measurement_z"][1])
ax2.semilogy(bias_line, chi_line, label=r"global $\chi^2$")
ax2.semilogy(bias_line, z3_line, "--", label="$|z|$ of stream 3"); ax2.semilogy(bias_line, z2_line, "-.", label="$|z|$ of stream 2")
ax2.axhline(7.815, color="0.5", ls=":", lw=1.0); ax2.axhline(11.345, color="0.5", ls=":", lw=1.0)
ax2.set_xlabel("bias added to stream 3 (kg/h)"); ax2.set_ylabel("test statistic"); ax2.legend(fontsize=9, loc="lower right"); panel_label(ax2, "(b)")
plt.show()
''',
     "Detection of a gross error. (a) The measurement test for each stream, with random noise only and with a "
     "bias on stream 3; the dotted line is the 95 % limit of a standard normal variable. The bias also raises "
     "the statistic of stream 2, which shares a balance with stream 3. (b) The global statistic and the two "
     "measurement statistics against the size of the bias; the dotted lines are the 95 % and 99 % limits of "
     "the $\\chi^2$ distribution with three degrees of freedom."),
]

FIGURES["03"] = [
    ("demo3", r'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
ax1.plot(Vm, P_noisy, "o", ms=3.5, color="0.5", label="noisy data")
ax1.plot(Vm, P_true, "-", color="k", lw=1.0, label="isotherm behind the data")
ax1.step(Vm, P_pava, where="mid", color="C0", label="monotone fit (PAVA)")
ax1.set_yscale("log"); ax1.set_xlabel(r"molar volume $V_m$ (L/mol)"); ax1.set_ylabel("pressure (MPa)")
ax1.legend(fontsize=9); panel_label(ax1, "(a)", x=0.03, y=0.12)
ax2.plot(Vm, P_noisy - P_true, "o", ms=3.5, color="0.5", label="noisy data")
ax2.step(Vm, P_pava - P_true, where="mid", color="C0", label="monotone fit")
ax2.plot(Vm, P_poly7 - P_true, "--", color="C3", label="polynomial of degree 7")
ax2.axhline(0.0, color="k", lw=0.8); ax2.set_ylim(-8.0, 8.0)
ax2.set_xlabel(r"molar volume $V_m$ (L/mol)"); ax2.set_ylabel("pressure minus true pressure (MPa)")
ax2.legend(fontsize=9, loc="lower right"); ax2.set_title("(b)", loc="left", fontweight="bold")
plt.show()
''',
     "A synthetic isotherm with noise. (a) The data, the isotherm that generated them and the monotone fit, "
     "which is a staircase: wherever the data rise, neighbours are pooled into a step. (b) The deviations from "
     "the true isotherm. The polynomial of degree 7 cannot follow the steep rise at small volumes and "
     "oscillates (part of its curve is outside the range shown)."),
    ("demo4", r"""
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
ax1.plot(Vm_sub, P_vdw, "-", color="C3", label="van der Waals isotherm, $T_r$ = 0.9")
ax1.plot(Vm_sub, P_maxwell_pava, "-", color="C0", label="monotone fit")
ax1.plot([v_liq, v_gas], [p_sat_maxwell, p_sat_maxwell], "k|", ms=12, mew=1.5, label="equal-area rule: ends of the tie line")
ax1.set_xlim(0.4, 3.5); ax1.set_ylim(0.2, 1.1)
ax1.set_xlabel(r"reduced volume $V_r$"); ax1.set_ylabel(r"reduced pressure $P_r$")
ax1.legend(fontsize=8.5, loc="upper right"); panel_label(ax1, "(a)", x=0.03, y=0.12)
ax2.plot(Vm_sub[on_plateau], P_vdw[on_plateau] - p_sat_est, "-", color="C3")
ax2.fill_between(Vm_sub[on_plateau], P_vdw[on_plateau] - p_sat_est, 0.0, where=P_vdw[on_plateau] > p_sat_est, color="C3", alpha=0.3)
ax2.fill_between(Vm_sub[on_plateau], P_vdw[on_plateau] - p_sat_est, 0.0, where=P_vdw[on_plateau] < p_sat_est, color="C0", alpha=0.3)
ax2.axhline(0.0, color="k", lw=0.8)
ax2.set_xlabel(r"reduced volume $V_r$"); ax2.set_ylabel("isotherm minus plateau pressure"); panel_label(ax2, "(b)", x=0.9)
dV = Vm_sub[1] - Vm_sub[0]
print(f"area above the plateau = {np.sum(np.clip(P_vdw[on_plateau] - p_sat_est, 0, None)) * dV:.5f}; area below = {np.sum(np.clip(p_sat_est - P_vdw[on_plateau], 0, None)) * dV:.5f}")
plt.show()
""",
     "(a) The reduced van der Waals isotherm at $T_r = 0.9$ and its monotone fit, which replaces the loop by a "
     "plateau. The two marks are the ends of the tie line from Maxwell's equal-area rule, computed "
     "independently. (b) The isotherm minus the plateau pressure over the pooled points: the two shaded areas "
     "are equal."),
]

FIGURES["04"] = [
    ("demo1", r'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
for m in range(M_mat.shape[1]):
    ax1.plot(x_grid, M_mat[:, m]); ax2.plot(x_grid, I_mat[:, m])
ax1.set_xlabel("$x$"); ax1.set_ylabel("M-spline basis function $M_m(x)$"); ax1.set_title("(a)", loc="left", fontweight="bold")
ax2.set_xlabel("$x$"); ax2.set_ylabel("I-spline basis function $I_m(x)$"); ax2.set_title("(b)", loc="left", fontweight="bold")
plt.show()
''',
     "The two bases of this lecture, for eight knots. (a) The M-splines: non-negative bumps. (b) The I-splines, "
     "their integrals: each rises from 0 to 1 and never falls, so a sum of them with non-negative weights "
     "cannot fall either."),
    ("demo3", r'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
ax1.plot(P_bar, q_obs, "o", ms=3, color="0.6", label="noisy data")
ax1.plot(P_bar, q_true, "-", color="k", lw=1.0, label="isotherm behind the data")
ax1.plot(P_bar, q_poly8, "--", color="C3", label="polynomial of degree 8")
ax1.step(P_bar, q_pava_ads, where="mid", color="C2", lw=1.0, label="staircase (PAVA)")
ax1.plot(P_bar, isp_fit["y_pred"], "-", color="C0", label="I-spline")
ax1.set_xlabel("pressure (bar)"); ax1.set_ylabel("amount adsorbed $q$ (mol/kg)"); ax1.legend(fontsize=8.5, loc="lower right"); panel_label(ax1, "(a)")
ax2.plot(P_bar, dq_dP_true, "-", color="k", lw=1.0, label="true slope")
ax2.plot(P_bar, dq_dP_poly8, "--", color="C3", label="polynomial of degree 8")
ax2.plot(P_bar, isp_fit["dy_dx"], "-", color="C0", label="I-spline")
ax2.axhline(0.0, color="0.5", lw=0.8); ax2.set_ylim(-0.6, 3.0)
ax2.set_xlabel("pressure (bar)"); ax2.set_ylabel("slope $dq/dP$ (mol kg$^{-1}$ bar$^{-1}$)"); ax2.legend(fontsize=9); panel_label(ax2, "(b)", x=0.45)
plt.show()
''',
     "Three fits to a synthetic adsorption isotherm. (a) The fitted curves. (b) Their slopes (the slope of the "
     "staircase is zero or undefined and is not drawn). The polynomial follows the true slope most closely but "
     "its slope becomes negative near the upper end; the slope of the I-spline is never negative, and with "
     "nine knots it is too coarse to follow the steep rise at low pressure."),
]

FIGURES["05"] = [
    ("demo1", r'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
for ax in (ax1, ax2):
    ax.plot(T_norm, y_spiked, "o", ms=3.5, color="0.5", label="data")
    ax.plot(T_norm, X_des @ w_true, "-", color="k", lw=1.0, label="true line")
    ax.plot(T_norm, X_des @ w_ols, "--", color="C3", label="least squares")
    ax.plot(T_norm, X_des @ w_huber, "-", color="C0", label="Huber")
    ax.set_xlabel("scaled temperature")
ax1.set_ylabel("response"); ax1.legend(fontsize=9, loc="center left"); panel_label(ax1, "(a)")
ax2.set_ylim(3.0, 21.0); ax2.set_ylabel("response (spikes outside the range)"); panel_label(ax2, "(b)")
plt.show()
''',
     "Synthetic calibration data with three spikes of +25. (a) All the data. (b) The same without the spikes in "
     "view: the least-squares line is lifted by the spikes over its whole length, and the Huber line lies on "
     "the true one."),
    ("demo4", r'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
ax1.plot(T_norm, y_spiked, "o", ms=3.5, color="0.5", label="data")
ax1.plot(T_norm, y_q50, "-", color="C0", label=r"median, $\tau$ = 0.5")
ax1.plot(T_norm, y_q10, "--", color="C2", label=r"$\tau$ = 0.1 and 0.9"); ax1.plot(T_norm, y_q90, "--", color="C2")
ax1.set_ylim(3.0, 21.0); ax1.set_xlabel("scaled temperature"); ax1.set_ylabel("response (spikes outside the range)")
ax1.legend(fontsize=9, loc="lower right"); panel_label(ax1, "(a)")
d_line = np.logspace(-1, 1.7, 28); err_line, wt_line = [], []
for d_val in d_line:
    res_d = linear_physics.huber_irls_regression(X_des, y_spiked, delta=float(d_val))
    err_line.append(np.linalg.norm(res_d["w"] - w_true)); wt_line.append(np.mean(res_d["sample_weights"][spike_idx]))
ax2.loglog(d_line, err_line, "o-", ms=3.5, label="error of the two parameters")
ax2.loglog(d_line, wt_line, "s--", ms=3.5, label="weight given to the spikes")
ax2.set_xlabel(r"Huber threshold $\delta$"); ax2.set_ylabel("magnitude"); ax2.legend(fontsize=9, loc="lower right"); panel_label(ax2, "(b)")
plt.show()
''',
     "(a) Quantile regression on the same data: the median line and the lines for the 10th and 90th "
     "percentiles. (b) The Huber fit against its threshold: a small threshold gives the spikes a small weight "
     "and a small error, and a threshold larger than the spikes gives them the weight one and returns least "
     "squares."),
]

FIGURES["06"] = [
    ("demo1", r'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
for i in range(X_scattered.shape[0]):
    ax1.plot(wl, X_scattered[i], color=plt.cm.viridis(i / 9.0), lw=1.2)
    ax2.plot(wl, X_snv[i], color=plt.cm.viridis(i / 9.0), lw=1.2)
ax1.set_xlabel("wavelength (nm)"); ax1.set_ylabel("absorbance (AU)"); ax1.set_title("(a) ten spectra as recorded", loc="left", fontsize=10.5)
ax2.set_xlabel("wavelength (nm)"); ax2.set_ylabel("absorbance after SNV"); ax2.set_title("(b) the same after SNV", loc="left", fontsize=10.5)
plt.show()
''',
     "(a) One synthetic spectrum recorded through ten different path lengths and offsets. (b) After the standard "
     "normal variate transformation, which subtracts the mean of each spectrum and divides by its standard "
     "deviation, the ten curves coincide."),
    ("demo4", r'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
ax1.plot([0.0, 1.0], [0.0, 1.0], "k-", lw=0.9)
ax1.plot(c1, pls_raw["y_pred"], "s", ms=4, mfc="none", color="C3", label=f"raw spectra, RMSE {rmse_raw:.3f}")
ax1.plot(c1, pls_snv["y_pred"], "o", ms=4, color="C0", label=f"after SNV, RMSE {rmse_snv:.4f}")
ax1.set_xlabel("reference concentration (mol/L)"); ax1.set_ylabel("predicted concentration (mol/L)")
ax1.legend(fontsize=9, loc="lower right"); panel_label(ax1, "(a)")
rng_fig = np.random.default_rng(6); sig_line = np.logspace(-4, -1.5, 11); r_snv_line, r_ds_line = [], []
for sig_det in sig_line:
    X_n = scatter_b * (np.outer(c1, pure_peak1) + pure_peak2[None, :]) + offset_a + rng_fig.normal(0.0, sig_det, size=(N_cal, len(wl)))
    r_snv_line.append(np.sqrt(np.mean((dim_reduction.pls1_nipals(dim_reduction.snv_preprocess(X_n), c1, 2)["y_pred"] - c1) ** 2)))
    X_ds = dim_reduction.snv_preprocess(dim_reduction.spectral_first_derivative(X_n, d_lambda=d_wl))
    r_ds_line.append(np.sqrt(np.mean((dim_reduction.pls1_nipals(X_ds, c1, 2)["y_pred"] - c1) ** 2)))
ax2.loglog(sig_line, r_snv_line, "o-", ms=3.5, label="SNV")
ax2.loglog(sig_line, r_ds_line, "s--", ms=3.5, label="first derivative, then SNV")
ax2.set_xlabel("detector noise (AU)"); ax2.set_ylabel("calibration RMSE (mol/L)"); ax2.legend(fontsize=9, loc="upper left"); panel_label(ax2, "(b)", x=0.9, y=0.12)
plt.show()
''',
     "(a) Calibration of a two-component PLS model on synthetic spectra with random path lengths and offsets, "
     "without and with SNV. (b) The calibration error against the detector noise, for SNV alone and for a "
     "first derivative followed by SNV: with no baseline tilt to remove, the derivative only amplifies the "
     "noise."),
]

FIGURES["07"] = [
    ("demo1", r'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
s_all = np.linalg.svd(X_spec - np.mean(X_spec, axis=0), compute_uv=False)
eig_all = s_all**2 / (N - 1)
ax1.semilogy(np.arange(1, len(eig_all) + 1), eig_all, "o-", ms=4)
ax1.set_xlabel("principal component"); ax1.set_ylabel("eigenvalue (variance)"); panel_label(ax1, "(a)", x=0.9)
ax2.plot(np.arange(1, 9), 100.0 * np.cumsum(eig_all)[:8] / np.sum(eig_all), "o-", ms=4)
ax2.set_xlabel("number of components kept"); ax2.set_ylabel("variance explained (%)"); ax2.set_ylim(40.0, 102.0); panel_label(ax2, "(b)", x=0.9, y=0.12)
plt.show()
''',
     "Principal components of synthetic spectra of 25 channels generated by three species. (a) The eigenvalues "
     "of the covariance matrix: three stand three orders of magnitude above the rest, which are noise. "
     "(b) The share of the variance explained by the first components."),
    ("demo3", r'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
ax1.plot(np.arange(1, d + 1), pcr_ols["coef"], "s--", ms=3.5, color="C3", label="all 25 components (least squares)")
ax1.plot(np.arange(1, d + 1), pcr3["coef"], "o-", ms=3.5, color="C0", label="3 components")
ax1.axhline(0.0, color="k", lw=0.8)
ax1.set_xlabel("spectral channel"); ax1.set_ylabel("regression coefficient"); ax1.legend(fontsize=9, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2); ax1.set_title("(a)", loc="left", fontweight="bold")
lims = [float(np.min(y_test_true)) - 0.5, float(np.max(y_test_true)) + 0.5]
ax2.plot(lims, lims, "k-", lw=0.9)
ax2.plot(y_test_true, pred_ols25, "s", ms=4, mfc="none", color="C3", label="all 25 components")
ax2.plot(y_test_true, pred_pcr3, "o", ms=3.5, color="C0", label="3 components")
ax2.set_xlabel("true value, test set"); ax2.set_ylabel("predicted value"); ax2.legend(fontsize=9, loc="lower right"); panel_label(ax2, "(b)")
plt.show()
''',
     "Regression on three principal components and on all 25. (a) The regression coefficients: with all "
     "components they are larger and irregular, because directions that carry only noise are divided by small "
     "singular values. (b) Predictions for 50 new samples."),
]

FIGURES["08"] = [
    ("demo1", r'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
for j, name in enumerate(["water", "methanol", "ethanol"]):
    ax1.plot(wl, E_pure[j], label=name)
ax1.plot(wl, X_nir[0], ":", color="k", lw=1.0, label="one mixture")
ax1.set_xlabel("wavelength (nm)"); ax1.set_ylabel("absorbance (AU)"); ax1.legend(fontsize=9); ax1.set_title("(a)", loc="left", fontweight="bold")
ax2.plot(wl, pls3["coef"], "-", color="C0", label="regression vector, 3 components")
ax2.plot(wl, E_pure[1] * np.max(np.abs(pls3["coef"])), ":", color="C1", label="methanol spectrum (scaled)")
ax2.axhline(0.0, color="k", lw=0.8)
ax2.set_xlabel("wavelength (nm)"); ax2.set_ylabel("coefficient (mol/L per AU)"); ax2.legend(fontsize=9, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2); ax2.set_title("(b)", loc="left", fontweight="bold")
plt.show()
''',
     "(a) The synthetic pure spectra of three overlapping absorbers and one of the 35 mixture spectra. (b) The "
     "regression vector of the PLS model for methanol. It is positive on the side of each methanol band away "
     "from ethanol and negative on the ethanol side, which is how a linear model subtracts an interfering "
     "species whose bands overlap."),
    ("demo4", r'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
k_all = np.arange(1, 13); tr_all, te_all = [], []
for k_lv in k_all:
    fit_k = dim_reduction.pls1_nipals(X_nir, y_meoh, n_components=int(k_lv))
    tr_all.append(np.sqrt(np.mean((fit_k["y_pred"] - y_meoh) ** 2)))
    te_all.append(np.sqrt(np.mean((X_test @ fit_k["coef"] + fit_k["intercept"] - y_test) ** 2)))
ax1.semilogy(k_all, tr_all, "o-", ms=4, label="training set (noisy reference values)")
ax1.semilogy(k_all, te_all, "s-", ms=4, label="test set (true concentrations)")
ax1.set_xlabel("number of latent variables"); ax1.set_ylabel("RMSE (mol/L)"); ax1.legend(fontsize=9, loc="lower left"); panel_label(ax1, "(a)", x=0.9)
ax2.plot([0.0, 1.0], [0.0, 1.0], "k-", lw=0.9)
ax2.plot(y_test, y_pred_ols, "s", ms=4, mfc="none", color="C3", label=f"minimum-norm least squares, RMSE {rmse_ols:.3f}")
ax2.plot(y_test, y_pred_pls, "o", ms=3.5, color="C0", label=f"PLS, 3 latent variables, RMSE {rmse_pls:.3f}")
ax2.set_xlabel("true methanol concentration (mol/L)"); ax2.set_ylabel("predicted (mol/L)"); ax2.legend(fontsize=8.5, loc="lower right"); panel_label(ax2, "(b)")
plt.show()
''',
     "(a) The error of the PLS model against the number of latent variables. On the training set it keeps "
     "falling; on the test set it is smallest at three, the number of absorbing species. (b) Predictions for "
     "40 new samples from the three-variable model and from the least-squares solution that fits the 35 "
     "training samples exactly."),
]

FIGURES["09"] = [
    ("demo3", r'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
T_noc = (X_noc - mu) @ P
ax1.plot(T_noc[:, 0], T_noc[:, 1], "o", ms=3.5, color="0.5", label="normal operation")
ax1.plot(t_surge[0], t_surge[1], "*", ms=13, color="C3", label="throughput surge")
ax1.plot(t_drift[0], t_drift[1], "D", ms=7, color="C0", label="drift of sensor 4")
ax1.set_xlabel("score on component 1"); ax1.set_ylabel("score on component 2"); ax1.legend(fontsize=9, loc="lower left"); ax1.set_title("(a)", loc="left", fontweight="bold")
ax2.loglog(t2_noc, q_noc, "o", ms=3.5, color="0.5", label="normal operation")
ax2.loglog(t2_surge, q_surge, "*", ms=13, color="C3", label="throughput surge")
ax2.loglog(t2_drift, q_drift, "D", ms=7, color="C0", label="drift of sensor 4")
ax2.axvline(t2_limit, color="k", ls=":", lw=1.0); ax2.axhline(q_limit, color="k", ls=":", lw=1.0)
ax2.set_xlabel(r"Hotelling $T^2$"); ax2.set_ylabel(r"residual $Q$"); ax2.legend(fontsize=9, loc="center left"); ax2.set_title("(b)", loc="left", fontweight="bold")
plt.show()
''',
     "Two kinds of abnormal event in a plant with eight correlated sensors (synthetic data). (a) In the plane "
     "of the two principal components the surge lies far from the normal points, and the drifting sensor is "
     "indistinguishable from them. (b) The two monitoring statistics, with their 99 % limits (dotted): the "
     "surge exceeds the limit of $T^2$ only, and the drift the limit of $Q$ only."),
    ("demo4", r'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
ax1.bar(np.arange(1, d + 1), q_contributions, color=["C3" if j == culprit_sensor else "0.6" for j in range(d)])
ax1.set_xlabel("sensor"); ax1.set_ylabel(r"contribution to $Q$, $e_j^2$"); ax1.tick_params(axis="x", which="minor", bottom=False, top=False)
panel_label(ax1, "(a)")
drift_line = np.logspace(-1.3, 0.6, 30); q_line = []
for d_val in drift_line:
    xc_s = np.zeros(d); xc_s[3] = d_val
    q_line.append(np.sum((xc_s - (xc_s @ P) @ P.T) ** 2))
ax2.loglog(drift_line, q_line, "-", label=r"$Q$ for a drift of sensor 4")
ax2.axhline(q_limit, color="k", ls=":", lw=1.0, label="99 % limit")
ax2.set_xlabel("drift of sensor 4"); ax2.set_ylabel(r"residual $Q$"); ax2.legend(fontsize=9, loc="lower right"); panel_label(ax2, "(b)")
print(f"smallest drift of sensor 4 that reaches the limit: {np.interp(np.log(q_limit), np.log(q_line), drift_line):.2f}")
plt.show()
''',
     "(a) The contributions of the eight sensors to $Q$ for a drift of 3.5 in sensor 4. (b) $Q$ against the "
     "size of the drift: it grows as the square of the drift and crosses the limit at the value printed above."),
]

FIGURES["10"] = [
    ("demo1", r'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
for ax, (i, j), names in [(ax1, (0, 1), ("scaled $T_r$", "scaled $P_r$")), (ax2, (0, 2), ("scaled $T_r$", "scaled $Z$"))]:
    for k_i in range(3):
        sel = labels3 == k_i
        ax.plot(X_phases[sel, i], X_phases[sel, j], "os^"[k_i], ms=4.5, color=f"C{k_i}", label=f"cluster {k_i + 1}")
    ax.plot(km3["centers"][:, i], km3["centers"][:, j], "k+", ms=13, mew=2, label="centroids")
    ax.set_xlabel(names[0]); ax.set_ylabel(names[1])
ax1.legend(fontsize=9, loc="upper left"); panel_label(ax1, "(a)", x=0.9, y=0.12); panel_label(ax2, "(b)", x=0.9, y=0.12)
plt.show()
''',
     "The 75 synthetic state points in two projections of the three scaled variables, coloured by the cluster "
     "that the algorithm assigned, with the three centroids."),
    ("demo4", r'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
ax1.plot(k_candidates, silhouettes, "o-", color="C0")
ax1.set_xlabel("number of clusters $K$"); ax1.set_ylabel("mean silhouette", color="C0"); ax1.set_xticks(k_candidates)
ax1.tick_params(axis="x", which="minor", bottom=False, top=False); ax1.tick_params(axis="y", colors="C0")
ax1b = ax1.twinx(); ax1b.semilogy(k_candidates, inertias, "s--", color="C3")
ax1b.set_ylabel("within-cluster sum of squares", color="C3"); ax1b.tick_params(axis="y", colors="C3")
ax1.set_title("(a)", loc="left", fontweight="bold")
lab_r = km_rings["labels"]
ax2.plot(X_rings[lab_r == 0, 0], X_rings[lab_r == 0, 1], "o", color="C0", label="cluster 1")
ax2.plot(X_rings[lab_r == 1, 0], X_rings[lab_r == 1, 1], "s", color="C3", label="cluster 2")
ax2.plot(km_rings["centers"][:, 0], km_rings["centers"][:, 1], "k+", ms=13, mew=2)
ax2.set_aspect("equal"); ax2.set_xlabel("$x_1$"); ax2.set_ylabel("$x_2$"); ax2.legend(fontsize=9, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2); ax2.set_title("(b)", loc="left", fontweight="bold")
plt.show()
''',
     "(a) Choosing the number of clusters: the mean silhouette (left axis) has a maximum at three, and the "
     "within-cluster sum of squares (right axis, logarithmic) stops falling quickly after three. (b) Two "
     "concentric rings, where the method fails: the boundary between the two clusters is a straight line, so "
     "the inner ring and most of the outer ring fall into one cluster."),
]

FIGURES["11"] = [
    ("demo3", r'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
ax1.plot(X_rings[lbl_spec == lbl_spec[0], 0], X_rings[lbl_spec == lbl_spec[0], 1], "o", color="C0", label="cluster 1")
ax1.plot(X_rings[lbl_spec != lbl_spec[0], 0], X_rings[lbl_spec != lbl_spec[0], 1], "s", color="C3", label="cluster 2")
ax1.set_aspect("equal"); ax1.set_xlabel("$x_1$"); ax1.set_ylabel("$x_2$"); ax1.legend(fontsize=9, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3); ax1.set_title("(a)", loc="left", fontweight="bold")
ax2.plot(np.arange(1, len(eigs_lap) + 1), np.maximum(eigs_lap, 0.0), "o-", ms=5, label="two rings")
ax2.plot(np.arange(1, len(e3) + 1), np.maximum(e3, 0.0), "s--", ms=5, label="three rings")
ax2.set_xticks(np.arange(1, len(e3) + 1)); ax2.tick_params(axis="x", which="minor", bottom=False, top=False); ax2.legend(fontsize=9, loc="upper left")
ax2.set_xlabel("index of the eigenvalue"); ax2.set_ylabel("eigenvalue of the normalized Laplacian"); ax2.set_title("(b)", loc="left", fontweight="bold")
plt.show()
''',
     "(a) Two concentric rings, as in Lecture 10, with the clusters found by spectral clustering. (b) The smallest "
     "eigenvalues of the normalized Laplacian for two rings and for the three rings of Demonstration 3: the "
     "number of eigenvalues close to zero equals the number of rings, and a gap separates them from the next."),
    ("demo4", r'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
for k_i, (marker, color) in enumerate(zip("os^", ["C0", "C3", "C2"])):
    sel = l3 == k_i
    ax1.plot(X_3rings[sel, 0], X_3rings[sel, 1], marker, ms=4.5, color=color, label=f"cluster {k_i + 1}")
ax1.set_aspect("equal"); ax1.set_xlabel("$x_1$"); ax1.set_ylabel("$x_2$"); ax1.legend(fontsize=9, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3); ax1.set_title("(a)", loc="left", fontweight="bold")
sig_line = np.logspace(-0.9, 0.7, 25); lam2, lam3 = [], []
for sig_val in sig_line:
    e_v = clustering_phases.spectral_clustering(X_rings, k=2, sigma=float(sig_val), seed=42)["laplacian_eigenvalues"]
    lam2.append(max(e_v[1], 1e-12)); lam3.append(e_v[2])
ax2.loglog(sig_line, lam2, "o-", ms=3.5, label=r"second eigenvalue $\lambda_2$")
ax2.loglog(sig_line, lam3, "s--", ms=3.5, label=r"third eigenvalue $\lambda_3$")
ax2.xaxis.set_minor_formatter(plt.NullFormatter()); ax2.set_xlim(0.1, 10.0)
ax2.set_xlabel(r"bandwidth $\sigma$ of the similarity"); ax2.set_ylabel("eigenvalue"); ax2.legend(fontsize=9, loc="lower right"); ax2.set_title("(b)", loc="left", fontweight="bold")
plt.show()
''',
     "(a) Three concentric rings and the clusters found by spectral clustering. (b) The second and third "
     "eigenvalues for the two rings against the bandwidth of the similarity: the method works while the second "
     "is far below the third. The two merge when the bandwidth exceeds about 0.7, less than half the distance "
     "of 1.7 between the rings."),
]

FIGURES["12"] = [
    ("demo2", r'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
def _resp(points, model):
    lp = np.array([np.log(model["weights"][j]) - 0.5 * np.sum(np.log(2.0 * np.pi * model["variances"][j])
                   + (points - model["means"][j]) ** 2 / model["variances"][j], axis=1) for j in range(len(model["weights"]))]).T
    g = np.exp(lp - lp.max(axis=1, keepdims=True))
    return g / g.sum(axis=1, keepdims=True)
gx, gy = np.meshgrid(np.linspace(-2.6, 2.6, 121), np.linspace(-2.0, 2.0, 101))
g_grid = _resp(np.column_stack([gx.ravel(), gy.ravel()]), gmm)
h_grid = -np.sum(g_grid * np.log(np.maximum(g_grid, 1e-300)), axis=1).reshape(gx.shape)
im = ax1.pcolormesh(gx, gy, h_grid, cmap="Greys", shading="auto", vmin=0.0, vmax=np.log(2.0))
ax1.plot(X_phase1[:, 0], X_phase1[:, 1], "o", ms=3.5, color="C0"); ax1.plot(X_phase2[:, 0], X_phase2[:, 1], "s", ms=3.5, color="C3")
fig.colorbar(im, ax=ax1, label="entropy of the responsibilities")
ax1.set_xlabel("$x_1$"); ax1.set_ylabel("$x_2$"); ax1.set_title("(a)", loc="left", fontweight="bold")
s_line = np.linspace(-1.0, 1.0, 401); g_line = _resp(np.outer(s_line, [1.5, 1.0]), gmm)
ax2.plot(s_line, g_line[:, 0], "-", label="responsibility of component 1")
ax2.plot(s_line, g_line[:, 1], "--", label="responsibility of component 2")
ax2.plot(s_line, -np.sum(g_line * np.log(np.maximum(g_line, 1e-300)), axis=1), ":", color="k", label="entropy")
ax2.set_xlabel("position $s$ on the line between the two centres"); ax2.set_ylabel("value")
ax2.set_ylim(-0.05, 1.3); ax2.legend(fontsize=8.5, loc="upper center", ncol=1); ax2.set_title("(b)", loc="left", fontweight="bold")
plt.show()
''',
     "(a) Two clusters of synthetic points and the entropy of the responsibilities of the fitted two-component "
     "mixture: it is zero inside each cluster and reaches $\\ln 2$ on the line that separates them. (b) The "
     "two responsibilities and the entropy along the straight line from one centre ($s=-1$) to the other "
     "($s=1$)."),
    ("demo4", r'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
gx, gy = np.meshgrid(np.linspace(-3.2, 3.2, 129), np.linspace(-1.2, 4.6, 117))
g3 = _resp(np.column_stack([gx.ravel(), gy.ravel()]), gmm3)
h3 = -np.sum(g3 * np.log(np.maximum(g3, 1e-300)), axis=1).reshape(gx.shape)
im = ax1.pcolormesh(gx, gy, h3, cmap="Greys", shading="auto", vmin=0.0, vmax=np.log(3.0))
ax1.plot(X_tp[:, 0], X_tp[:, 1], "o", ms=3, color="C0"); ax1.plot(tp_centroid[0], tp_centroid[1], "*", ms=13, color="C3")
fig.colorbar(im, ax=ax1, label="entropy of the responsibilities")
ax1.set_aspect("equal"); ax1.set_xlabel("$x_1$"); ax1.set_ylabel("$x_2$"); ax1.set_title("(a)", loc="left", fontweight="bold")
k_list = sorted(bics); ax2.plot(k_list, [bics[k_] for k_ in k_list], "o-")
ax2.set_xticks(k_list); ax2.tick_params(axis="x", which="minor", bottom=False, top=False)
ax2.set_xlabel("number of components $K$"); ax2.set_ylabel("Bayesian information criterion"); ax2.set_title("(b)", loc="left", fontweight="bold")
plt.show()
''',
     "(a) Three clusters at the corners of an equilateral triangle. The entropy of the responsibilities is "
     "$\\ln 2$ on the three lines that separate pairs of clusters and $\\ln 3$ where they meet (star). (b) The "
     "Bayesian information criterion of mixtures with one to five components fitted to the same points; the "
     "smallest value marks the number of components that the criterion selects."),
]

FIGURES["13"] = [
    ("demo2", r'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
T_mid = 0.5 * (T_K[:-1] + T_K[1:])
ax1.semilogy(T_mid, sse_prefix, "-", label="sum of squares after the split")
ax1.axvline(split_1d["threshold"], color="k", ls=":", lw=1.0, label="best threshold")
ax1.set_xlabel("threshold temperature (K)"); ax1.set_ylabel("sum of squared errors"); ax1.legend(fontsize=9, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2); ax1.set_title("(a)", loc="left", fontweight="bold")
ax2.plot(T_K, X_conv, "o", ms=3.5, color="0.5", label="conversion (synthetic)")
for depth, style in [(1, "--"), (3, "-")]:
    tree_f = trees_ensembles.DecisionTreeRegressor(max_depth=depth, min_samples_split=2).fit(T_K, X_conv)
    ax2.step(T_K, tree_f.predict(T_K), style, where="mid", label=f"tree of depth {depth}")
ax2.set_xlabel("temperature (K)"); ax2.set_ylabel("conversion"); ax2.legend(fontsize=9, loc="upper left"); panel_label(ax2, "(b)", x=0.9, y=0.12)
plt.show()
''',
     "(a) The sum of squared errors left after one split of the synthetic light-off curve, against the "
     "position of the threshold; the minimum is at the ignition temperature. (b) The curve and the piecewise "
     "constant fits of trees of depth 1 and 3."),
    ("demo4", r'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
depths = [1, 2, 3, 4]
ax1.semilogy(depths, sse_by_depth, "o-"); ax1.set_xticks(depths); ax1.tick_params(axis="x", which="minor", bottom=False, top=False)
ax1.set_xlabel("depth of the tree"); ax1.set_ylabel("sum of squared errors on the training points"); panel_label(ax1, "(a)", x=0.9)
T_line = np.linspace(300.0, 600.0, 301)
ax2.semilogy(T_line, 1e6 * np.exp(-5000.0 / T_line), "k-", lw=1.0, label="Arrhenius law (assumed constants)")
ax2.semilogy(T_train, k_train, "o", ms=3.5, color="0.5", label="training points")
ax2.semilogy(T_line, tree_arr.predict(T_line), "-", color="C3", label="tree of depth 4")
ax2.axvline(400.0, color="k", ls=":", lw=1.0)
ax2.set_xlabel("temperature (K)"); ax2.set_ylabel("rate constant (1/s)"); ax2.legend(fontsize=9, loc="lower right"); panel_label(ax2, "(b)")
plt.show()
''',
     "(a) The training error of the tree against its depth. (b) A tree trained on an Arrhenius rate constant "
     "between 300 and 400 K (left of the dotted line) and evaluated up to 600 K: beyond the last training "
     "point it returns the value of its last leaf."),
]

FIGURES["14"] = [
    ("demo2", r'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
lims = [float(np.min(y_te_clean)) - 0.5, float(np.max(y_te_clean)) + 0.5]
ax1.plot(lims, lims, "k-", lw=0.9)
ax1.plot(y_te_clean, single_tree.predict(X_te), "s", ms=3.5, mfc="none", color="C3", label=f"one tree, MSE {mse_single:.2f}")
ax1.plot(y_te_clean, rf_model.predict(X_te), "o", ms=3.5, color="C0", label=f"forest of 25 trees, MSE {mse_rf:.2f}")
ax1.set_xlabel("true value, test set"); ax1.set_ylabel("predicted value"); ax1.legend(fontsize=9, loc="lower right"); panel_label(ax1, "(a)")
B_line = np.arange(1, 101)
for rho_v, style in [(0.0, ":"), (0.25, "-"), (0.6, "--")]:
    ax2.loglog(B_line, rho_v * sig2 + (1.0 - rho_v) * sig2 / B_line, style, label=rf"$\rho$ = {rho_v:g}")
ax2.set_xlabel("number of trees $B$"); ax2.set_ylabel("variance of the average"); ax2.legend(fontsize=9, loc="lower left"); panel_label(ax2, "(b)", x=0.9)
plt.show()
''',
     "(a) Predictions of one tree and of a forest of 25 trees on 120 test points of a synthetic function of "
     "four variables. (b) The variance of the average of $B$ estimates of variance 2 with pairwise correlation "
     "$\\rho$: it falls as $1/B$ only for uncorrelated estimates and otherwise levels off at $\\rho\\sigma^2$."),
    ("demo4", r'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
ax1.bar(np.arange(4), np.asarray(imp), color=["C0", "C0", "0.6", "0.6"])
ax1.set_xticks(np.arange(4)); ax1.set_xticklabels(["$x_0$", "$x_1$", "$x_2$", "$x_3$"]); ax1.tick_params(axis="x", which="minor", bottom=False, top=False)
ax1.set_ylabel("increase of the test MSE"); panel_label(ax1, "(a)", x=0.9)
b_list = [1, 2, 3, 5, 8, 12, 20, 35, 50]; mse_list = []
for b_trees in b_list:
    rf_b = trees_ensembles.RandomForestRegressor(n_estimators=b_trees, max_depth=5, seed=42).fit(X_tr, y_tr)
    mse_list.append(np.mean((rf_b.predict(X_te) - y_te_clean) ** 2))
ax2.semilogx(b_list, mse_list, "o-"); ax2.set_xlabel("number of trees $B$"); ax2.set_ylabel("test MSE"); panel_label(ax2, "(b)", x=0.9)
plt.show()
''',
     "(a) Permutation importance of the four variables: the function depends on $x_0$ and $x_1$ only, and the "
     "other two are noise. (b) The test error of the forest against the number of trees."),
]

FIGURES["15"] = [
    ("demo1", r'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
ax1.plot(x_Ni, HV_true, "k-", lw=1.0, label="hardness (synthetic)")
for m_trees, style in [(1, ":"), (5, "--"), (35, "-")]:
    gbm_f = trees_ensembles.GradientBoostingRegressor(n_estimators=m_trees, learning_rate=0.2, max_depth=2).fit(x_Ni, HV_true)
    ax1.step(x_Ni, gbm_f.predict(x_Ni), style, where="mid", label=f"{m_trees} tree" + ("s" if m_trees > 1 else ""))
ax1.set_xlabel("mole fraction of nickel"); ax1.set_ylabel("hardness (HV)"); ax1.legend(fontsize=9, loc="lower center", ncol=2); panel_label(ax1, "(a)")
m_list = np.arange(1, 61); mse_m = []
for m_trees in m_list:
    gbm_f = trees_ensembles.GradientBoostingRegressor(n_estimators=int(m_trees), learning_rate=0.2, max_depth=2).fit(x_Ni, HV_true)
    mse_m.append(np.mean((gbm_f.predict(x_Ni) - HV_true) ** 2))
ax2.semilogy(m_list, mse_m, "-"); ax2.set_xlabel("number of trees"); ax2.set_ylabel("training MSE (HV$^2$)"); panel_label(ax2, "(b)", x=0.9)
plt.show()
''',
     "(a) Gradient boosting of trees of depth 2 on a synthetic hardness curve of a binary alloy, after 1, 5 "
     "and 35 trees with a learning rate of 0.2. (b) The training error against the number of trees."),
    ("demo3", r'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
ax1.plot(x_Ni, HV_spiked, "o", ms=3.5, color="0.5", label="data with two outliers")
ax1.plot(x_Ni, HV_true, "k-", lw=1.0, label="hardness behind the data")
ax1.plot(x_Ni, gbm_l2.predict(x_Ni), "--", color="C3", label="squared loss")
ax1.plot(x_Ni, gbm_huber.predict(x_Ni), "-", color="C0", label="Huber loss")
ax1.set_xlabel("mole fraction of nickel"); ax1.set_ylabel("hardness (HV)"); ax1.legend(fontsize=9, loc="upper center"); ax1.set_title("(a)", loc="left", fontweight="bold")
ax2.plot(x_Ni, gbm_l2.predict(x_Ni) - HV_true, "--", color="C3", label=f"squared loss, RMSE {rmse_l2:.1f} HV")
ax2.plot(x_Ni, gbm_huber.predict(x_Ni) - HV_true, "-", color="C0", label=f"Huber loss, RMSE {rmse_huber:.1f} HV")
ax2.axhline(0.0, color="k", lw=0.8)
ax2.set_xlabel("mole fraction of nickel"); ax2.set_ylabel("fit minus true hardness (HV)"); ax2.legend(fontsize=9, loc="upper center"); ax2.set_title("(b)", loc="left", fontweight="bold")
plt.show()
''',
     "(a) The hardness data with an outlier of +250 HV at each end, and the fits of 25 boosted trees with the "
     "squared loss and with the Huber loss. (b) The errors of the two fits with respect to the curve behind "
     "the data."),
]

FIGURES["16"] = [
    ("demo1", r'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
res_cal = y_cal_true - y_cal_pred
ax1.hist(res_cal, bins=30, color="0.7", edgecolor="0.4")
half_width = float(np.mean(cp90["upper"] - cp90["lower"])) / 2.0
ax1.axvline(-half_width, color="C3", ls="--"); ax1.axvline(half_width, color="C3", ls="--", label="conformal interval, 90 %")
ax1.set_xlabel("residual on the calibration set"); ax1.set_ylabel("count"); ax1.legend(fontsize=9, loc="upper right"); panel_label(ax1, "(a)", x=0.9, y=0.55)
alpha_line = np.linspace(0.02, 0.30, 15); cov_line = []
for alpha_v in alpha_line:
    cp_v = uncertainty.split_conformal_interval(y_cal_true, y_cal_pred, y_test_pred, alpha=float(alpha_v))
    cov_line.append(np.mean((y_test_true >= cp_v["lower"]) & (y_test_true <= cp_v["upper"])))
ax2.plot(100 * (1 - alpha_line), 100 * (1 - alpha_line), "k-", lw=0.9, label="target")
ax2.plot(100 * (1 - alpha_line), 100 * np.array(cov_line), "o", ms=4.5, label="measured on 1000 test points")
ax2.set_xlabel("target coverage (%)"); ax2.set_ylabel("measured coverage (%)"); ax2.legend(fontsize=9, loc="lower right"); panel_label(ax2, "(b)")
plt.show()
''',
     "(a) The residuals of the calibration set, which follow a skewed (shifted exponential) distribution, and "
     "the symmetric 90 % interval of split conformal prediction. (b) The coverage measured on the test set "
     "against the target coverage."),
    ("demo3", r'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
ax1.plot(X_corr[:, 0], X_corr[:, 1], "o", ms=3.5, color="0.5", label="training points")
ax1.plot(x_query[0, 0], x_query[0, 1], "*", ms=13, color="C3", label="query")
mu_c = np.mean(X_corr, axis=0); sd_c = np.std(X_corr, axis=0)
ax1.plot(mu_c[0] + sd_c[0] * np.array([-3, 3, 3, -3, -3]), mu_c[1] + sd_c[1] * np.array([-3, -3, 3, 3, -3]), "k:", lw=1.0, label="within 3 standard deviations on each axis")
ax1.set_xlabel("temperature (K)"); ax1.set_ylabel("pressure (bar)"); ax1.legend(fontsize=8.5, loc="lower right", bbox_to_anchor=(1.0, 1.0)); ax1.set_title("(a)", loc="left", fontweight="bold")
d2_train = uncertainty.mahalanobis_ood_score(X_corr, X_corr)
ax2.hist(d2_train, bins=np.logspace(-2.5, 1.5, 25), color="0.7", edgecolor="0.4", label="training points")
ax2.axvline(d2_query, color="C3", ls="--", label="query")
ax2.set_xscale("log"); ax2.set_xlabel("squared Mahalanobis distance"); ax2.set_ylabel("count"); ax2.legend(fontsize=9, loc="upper left"); ax2.set_title("(b)", loc="left", fontweight="bold")
print(f"query: {z_axes[0]:.2f} and {z_axes[1]:.2f} standard deviations on the two axes; squared Mahalanobis distance {d2_query:.0f}; largest among the training points {np.max(d2_train):.1f}")
plt.show()
''',
     "(a) Synthetic temperatures and pressures that lie close to a line, and a query that is within the range "
     "of each variable separately but far from the line. (b) The squared Mahalanobis distances of the training "
     "points and of the query."),
]

FIGURES["17"] = [
    ("demo1", r'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
ax1.plot(np.arange(N_te), np.sum(Y_te_raw, axis=1) - 1.0, "s", ms=3.5, mfc="none", color="C3", label="PLS predictions")
ax1.plot(np.arange(N_te), np.sum(Y_te_kkt, axis=1) - 1.0, "o", ms=3, color="C0", label="after the projection")
ax1.set_xlabel("test sample"); ax1.set_ylabel("sum of the three fractions minus 1"); ax1.legend(fontsize=9, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2); ax1.set_title("(a)", loc="left", fontweight="bold")
ax2.plot([0.1, 0.5], [0.1, 0.5], "k-", lw=0.9)
ax2.plot(Y_te_true[:, 0], Y_te_raw[:, 0], "s", ms=3.5, mfc="none", color="C3", label="PLS prediction")
ax2.plot(Y_te_true[:, 0], Y_te_kkt[:, 0], "o", ms=3, color="C0", label="after the projection")
ax2.set_xlabel("true methanol fraction"); ax2.set_ylabel("predicted methanol fraction"); ax2.legend(fontsize=9, loc="lower right"); ax2.set_title("(b)", loc="left", fontweight="bold")
plt.show()
''',
     "(a) The error of the carbon balance of the three predicted fractions on the 80 test samples (synthetic "
     "spectra), before and after the projection onto the balance. (b) The predicted methanol fraction against "
     "the true one."),
    ("demo4", r'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
cp_m = uncertainty.split_conformal_interval(Y_cal_obs[:, 0], Y_cal_kkt[:, 0], Y_te_kkt[:, 0], alpha=0.10)
order = np.argsort(Y_te_kkt[:, 0]); inside = (Y_te_obs[:, 0] >= cp_m["lower"]) & (Y_te_obs[:, 0] <= cp_m["upper"])
ax1.fill_between(np.arange(N_te), cp_m["lower"][order], cp_m["upper"][order], color="C0", alpha=0.25, label="90 % interval")
ax1.plot(np.arange(N_te)[inside[order]], Y_te_obs[order, 0][inside[order]], "o", ms=3, color="0.4", label="reference value inside")
ax1.plot(np.arange(N_te)[~inside[order]], Y_te_obs[order, 0][~inside[order]], "x", ms=6, color="C3", label="outside")
ax1.set_xlabel("test sample, sorted by prediction"); ax1.set_ylabel("methanol fraction"); ax1.legend(fontsize=9, loc="upper left"); panel_label(ax1, "(a)", x=0.9, y=0.12)
amp_line = np.logspace(-2, 0.5, 30); q_line, err_line = [], []
for f_amp in amp_line:
    x_f = X_snv[N_tr + N_cal].copy(); x_f[40:50] += f_amp
    xc_f = x_f - pca_guard["mean"]
    q_line.append(np.sum((xc_f - (xc_f @ P_g) @ P_g.T) ** 2))
    err_line.append(abs(predict_raw_and_kkt(x_f[None, :])[1][0, 0] - Y_te_true[0, 0]))
ax2.loglog(amp_line, q_line, "-", label=r"residual $Q$ of the spectrum")
ax2.loglog(amp_line, err_line, "--", label="error of the methanol fraction")
ax2.axhline(q_limit_99, color="k", ls=":", lw=1.0, label=r"99 % limit of $Q$")
ax2.set_xlabel("size of the disturbance in ten channels (AU)"); ax2.set_ylabel("value"); ax2.legend(fontsize=9, loc="upper left"); panel_label(ax2, "(b)", x=0.9, y=0.12)
print(f"reference values inside the interval: {int(np.sum(inside))} of {N_te}")
plt.show()
''',
     "(a) The 90 % conformal interval of the methanol fraction on the test samples, with the noisy reference "
     "values. (b) A disturbance added to ten channels of one spectrum: the residual $Q$ and the error of the "
     "predicted methanol fraction against the size of the disturbance, with the limit above which the "
     "prediction is withheld."),
]
