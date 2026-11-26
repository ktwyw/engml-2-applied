"""Animation at the top of README.md: monotone regression of van der Waals isotherms.

Left: the reduced van der Waals isotherm at one temperature below the critical one, and its
monotone fit by the pool-adjacent-violators algorithm, which replaces the loop by a plateau.
Right: the ends of the plateaus, collected as the temperature rises, on the coexistence curve
computed independently from Maxwell's equal-area rule.

Run from the repository root:  python scripts/make_readme_animation.py
Output: docs/images/maxwell_plateau.gif
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, root)

from engml_2_applied import linear_physics  # noqa: E402
from engml_2_applied.plotting import use_style  # noqa: E402


def isotherm(v, t_r):
    """Reduced van der Waals pressure."""
    return 8.0 * t_r / (3.0 * v - 1.0) - 3.0 / v**2


def equal_area(t_r):
    """Saturation pressure and the two volumes from Maxwell's equal-area rule, by bisection."""
    def mismatch(p):
        roots = np.roots([3.0 * p, -(p + 8.0 * t_r), 9.0, -3.0]).real
        v_l, v_g = roots.min(), roots.max()
        anti = lambda v: (8.0 * t_r / 3.0) * np.log(3.0 * v - 1.0) + 3.0 / v
        return anti(v_g) - anti(v_l) - p * (v_g - v_l), v_l, v_g
    # the two spinodal pressures bracket the saturation pressure
    v_s = np.roots([4.0 * t_r, -9.0, 6.0, -1.0]).real          # dP/dV = 0
    v_s = np.sort(v_s[v_s > 1.0 / 3.0 + 1e-9])
    p_lo, p_hi = max(isotherm(v_s[0], t_r), 1e-6), isotherm(v_s[-1], t_r)
    for _ in range(80):
        p_mid = 0.5 * (p_lo + p_hi)
        if mismatch(p_mid)[0] > 0.0:
            p_lo = p_mid
        else:
            p_hi = p_mid
    return (p_mid,) + mismatch(p_mid)[1:]


use_style()
volume = np.linspace(0.42, 6.0, 2233)                      # uniform grid in the reduced volume
temperatures = np.linspace(0.85, 0.995, 30)
frames = []
for t_r in temperatures:
    pressure = isotherm(volume, t_r)
    fit = linear_physics.pava_monotone_regression(pressure, increasing=False)
    values, counts = np.unique(np.round(fit, 9), return_counts=True)
    plateau = values[np.argmax(counts)]
    on = np.isclose(fit, plateau, atol=1e-8)
    frames.append((t_r, pressure, fit, plateau, volume[on][0], volume[on][-1]))

t_line = np.linspace(0.84, 0.9995, 120)
exact = np.array([equal_area(t) for t in t_line])
worst = max(abs(f[3] - equal_area(f[0])[0]) for f in frames)
print(f"largest difference between the plateau and the equal-area pressure over {len(frames)} temperatures: {worst:.1e}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
for ax in (ax1, ax2):
    ax.set_xlim(0.4, 4.0); ax.set_ylim(0.3, 1.15)
    ax.set_xlabel(r"reduced volume $V_r$"); ax.set_ylabel(r"reduced pressure $P_r$")
ax2.set_xscale("log"); ax2.set_xlim(0.45, 4.0)
ax2.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g")); ax2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax2.set_xticks([0.5, 1.0, 2.0, 4.0])
ax1.set_title("van der Waals isotherm and its monotone fit", fontsize=10.5)
ax2.set_title("ends of the plateaus, and the equal-area rule", fontsize=10.5)
ax2.plot(np.r_[exact[:, 1], 1.0, exact[::-1, 2]], np.r_[exact[:, 0], 1.0, exact[::-1, 0]], "k-", lw=1.0, label="Maxwell's equal-area rule")
line_p, = ax1.plot([], [], "-", color="C3", label="isotherm")
line_f, = ax1.plot([], [], "-", color="C0", lw=2.0, label="monotone fit (PAVA)")
text = ax1.text(0.96, 0.62, "", transform=ax1.transAxes, ha="right", fontsize=10)
dots, = ax2.plot([], [], "o", color="C0", ms=4, label="ends of the plateau of the fit")
tie, = ax2.plot([], [], "-", color="C0", lw=1.5)
ax1.legend(fontsize=9, loc="upper right"); ax2.legend(fontsize=9, loc="lower left")
n_hold = 10


def update(frame):
    k = min(frame, len(frames) - 1)
    t_r, pressure, fit, plateau, v_left, v_right = frames[k]
    line_p.set_data(volume, pressure); line_f.set_data(volume, fit)
    text.set_text(rf"$T_r$ = {t_r:.3f}")
    dots.set_data([f[4] for f in frames[:k + 1]] + [f[5] for f in frames[:k + 1]], [f[3] for f in frames[:k + 1]] * 2)
    tie.set_data([v_left, v_right], [plateau, plateau])
    return []


anim = FuncAnimation(fig, update, frames=len(frames) + n_hold, interval=160)
out = os.path.join(root, "docs", "images", "maxwell_plateau.gif")
anim.save(out, writer=PillowWriter(fps=6), dpi=88)
print("wrote", os.path.normpath(out))
