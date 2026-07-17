"""Build the report figures from results/. Run after experiments.py.

Writes vector PDFs into figures/ (used by the report) and PNG previews.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")

# palette: colorblind-safe order, checked against a light surface
BLUE, GREEN, MAGENTA = "#2a78d6", "#008300", "#e87ba4"
SEQ = ["#86b6ef", "#2a78d6", "#104281"]         # light -> dark, one hue
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "xtick.color": AXIS, "ytick.color": AXIS,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.labelcolor": INK, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 150,
})

SYS_LABEL = {"linear2d": "damped oscillator", "lotka": "Lotka-Volterra",
             "lorenz": "Lorenz"}
SYS_COLOR = {"linear2d": BLUE, "lotka": GREEN, "lorenz": MAGENTA}


def grid(ax, axis="y"):
    ax.grid(True, axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", dpi=180)
    plt.close(fig)
    print("wrote", name)


def fig1_clean():
    d = np.load(os.path.join(RES, "E1.npz"), allow_pickle=True)
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.2))

    ax = axes[0]
    X, Xr = d["linear2d_X"], d["linear2d_Xr"]
    ax.plot(X[:, 0], X[:, 1], color=BLUE, lw=1.4, label="true")
    ax.plot(Xr[:, 0], Xr[:, 1], color=GREEN, lw=1.1, ls=(0, (4, 3)), label="recovered")
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_title("damped oscillator", color=INK2)
    ax.legend(loc="upper left", handlelength=1.6, borderaxespad=0.2)

    ax = axes[1]
    X, Xr = d["lotka_X"], d["lotka_Xr"]
    ax.plot(X[:, 0], X[:, 1], color=BLUE, lw=1.4)
    ax.plot(Xr[:, 0], Xr[:, 1], color=GREEN, lw=1.1, ls=(0, (4, 3)))
    ax.set_xlabel("$x$ (prey)"); ax.set_ylabel("$y$ (predator)")
    ax.set_title("Lotka-Volterra", color=INK2)

    ax = axes[2]
    X, Xr = d["lorenz_X"], d["lorenz_Xr"]
    ax.plot(X[:, 0], X[:, 2], color=BLUE, lw=0.4, alpha=0.9)
    ax.plot(Xr[:, 0], Xr[:, 2], color=GREEN, lw=0.4, alpha=0.55)
    ax.set_xlabel("$x$"); ax.set_ylabel("$z$")
    ax.set_title("Lorenz", color=INK2)

    for ax in axes:
        grid(ax, "both")
    fig.tight_layout(w_pad=1.4)
    save(fig, "fig1_clean")


def fig2_noise():
    d = np.load(os.path.join(RES, "E2.npz"), allow_pickle=True)
    levels, err, suc = d["levels"], d["med_err"], d["success"]
    names = list(d["systems"])
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 3.6), sharex=True)
    for si, name in enumerate(names):
        ax = axes[0, si]
        ax.loglog(levels, err[si, 0], color=BLUE, lw=1.6, marker="o", ms=3)
        ax.loglog(levels, err[si, 1], color=GREEN, lw=1.6, marker="s", ms=3)
        ax.set_title(SYS_LABEL[str(name)], color=INK2)
        grid(ax)
        ax = axes[1, si]
        ax.semilogx(levels, suc[si, 0], color=BLUE, lw=1.6, marker="o", ms=3,
                    label="finite differences")
        ax.semilogx(levels, suc[si, 1], color=GREEN, lw=1.6, marker="s", ms=3,
                    label="Savitzky-Golay")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("noise level $\\sigma$")
        grid(ax)
    axes[0, 0].set_ylabel("median coef. error")
    axes[1, 0].set_ylabel("exact-recovery rate")
    axes[1, 0].legend(loc="lower left", handlelength=1.6)
    fig.tight_layout(w_pad=1.2, h_pad=0.8)
    save(fig, "fig2_noise")


def fig3_sampling():
    a = np.load(os.path.join(RES, "E3a.npz"), allow_pickle=True)
    b = np.load(os.path.join(RES, "E3b.npz"), allow_pickle=True)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.3))

    ax = axes[0]
    dts = a["dts"]
    si = list(a["systems"]).index("lorenz")
    ax.loglog(dts, a["err_clean"][si], color=BLUE, lw=1.6, marker="o", ms=3,
              label="clean")
    ax.loglog(dts, a["err_noisy"][si], color=GREEN, lw=1.6, marker="s", ms=3,
              label="0.1% noise")
    ref = a["err_clean"][si][2] * (dts / dts[2]) ** 2
    ax.loglog(dts, ref, color=MUTED, lw=0.9, ls=":", label="$\\propto \\Delta t^2$")
    ax.set_xlabel("step size $\\Delta t$")
    ax.set_ylabel("median coef. error")
    ax.set_title("Lorenz, finite differences", color=INK2)
    ax.legend(handlelength=1.6)
    grid(ax)

    ax = axes[1]
    Ts = b["Ts"]
    for name in ["linear2d", "lotka", "lorenz"]:
        si = list(b["systems"]).index(name)
        ax.plot(Ts, b["success"][si], color=SYS_COLOR[name], lw=1.6,
                marker="o", ms=3, label=SYS_LABEL[name])
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("trajectory length $T$")
    ax.set_ylabel("exact-recovery rate")
    ax.set_title("1% noise, Savitzky-Golay", color=INK2)
    ax.legend(handlelength=1.6)
    grid(ax)
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig3_sampling")


def fig4_threshold():
    d = np.load(os.path.join(RES, "E4.npz"), allow_pickle=True)
    lams, n_terms, true_active = d["lams"], d["n_terms"], d["true_active"]
    names = list(d["systems"])
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.3))
    for si, name in enumerate(names):
        ax = axes[si]
        ax.axhline(true_active[si], color=MUTED, lw=0.9, ls=":")
        ax.text(lams[0] * 1.15, true_active[si] + 0.4, "true count",
                color=MUTED, fontsize=7.5)
        ax.semilogx(lams, n_terms[si, 0], color=BLUE, lw=1.6, label="clean")
        ax.semilogx(lams, n_terms[si, 1], color=GREEN, lw=1.6, label="1% noise")
        ax.set_xlabel("threshold $\\lambda$")
        ax.set_title(SYS_LABEL[str(name)], color=INK2)
        grid(ax)
    axes[0].set_ylabel("active terms")
    axes[0].legend(handlelength=1.6)
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig4_threshold")


def fig5_library():
    d = np.load(os.path.join(RES, "E5.npz"), allow_pickle=True)
    degrees, levels = d["degrees"], d["levels"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.3))

    ax = axes[0]
    for li, lv in enumerate(levels):
        label = "clean" if lv == 0 else f"{100 * lv:g}% noise"
        ax.plot(degrees, d["success"][li], color=SEQ[li], lw=1.6, marker="o",
                ms=3.5, label=label)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(degrees)
    ax.set_xlabel("library max degree")
    ax.set_ylabel("exact-recovery rate")
    ax.legend(handlelength=1.6, loc="center left")
    grid(ax)

    ax = axes[1]
    ax.semilogy(degrees, d["conds"], color=BLUE, lw=1.6, marker="o", ms=3.5)
    ax.set_xticks(degrees)
    ax.set_xticklabels([f"{deg}\n({n} terms)" for deg, n in zip(degrees, d["n_lib"])])
    ax.set_xlabel("library max degree")
    ax.set_ylabel("cond$(\\Theta)$")
    grid(ax)
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig5_library")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    fig1_clean()
    fig2_noise()
    fig3_sampling()
    fig4_threshold()
    fig5_library()
    print("all figures ->", FIG)
