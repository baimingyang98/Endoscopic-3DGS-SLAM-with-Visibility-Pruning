"""
Method figure for the ICRA paper: "TVS-Guided Soft Pruning for Endoscopic
Gaussian SLAM".

Three panels (one row, sized for a full-width figure* / \\includegraphics[width=\\textwidth]):
  (a) TVS soft-decay gate            d_i = sigma((log TVS - log tau_sig) / T)
  (b) reversible opacity over time   alpha decays when unobserved, recovers when re-seen
  (c) spatial floater mask, side view  flag if |z_i - D| > gamma * D, then alpha <- eta * alpha

All constants are taken from config.py so the figure stays consistent with the
reported runs. Output: pictures/tvs_mechanism.png and .pdf (300 dpi).

Run:  python make_tvs_figure.py
Deps: numpy, matplotlib
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, FancyArrowPatch

# ----------------------------------------------------------------------------
# Hyper-parameters (must match config.py)
# ----------------------------------------------------------------------------
TAU_SIG = 0.02       # tvs_tau_sig : significance midpoint of the gate
T = 1.0              # tvs_temperature
BETA = 0.1           # tvs_beta : opacity confidence exponent
LAMBDA = 0.18        # tvs_ema_lambda : EMA coefficient for the visibility freq.
ALPHA_FLOOR = 0.01   # tvs_opacity_floor
ALPHA_CLEAN = 0.011  # tvs_cleanup_threshold : final cleanup deletes at or below
GAMMA = 0.5          # distance_gamma : relative depth tolerance of the mask
ETA_SPATIAL = 0.9    # eta_spatial : mask attenuation factor

# ----------------------------------------------------------------------------
# Palette (kept readable in print; matches the proposed figure mock)
# ----------------------------------------------------------------------------
GREEN = "#1D9E75"
BLUE = "#2E6DB4"
RED = "#C0392B"
GRAY = "#6B6B6B"
BAND = "#CFE9DE"

# IEEE figures: Times-like serif + math, and TrueType (Type 42) embedding so the
# PDF passes IEEE PDF eXpress font checks. STIX ships with matplotlib, so it works
# even where Times New Roman is not installed (e.g. Colab).
plt.rcParams.update({
    "font.size": 8,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "axes.linewidth": 0.6,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "lines.linewidth": 1.6,
})


def gate(tvs):
    """Log-space sigmoid decay gate (eq. 5)."""
    return 1.0 / (1.0 + np.exp(-(np.log(tvs + 1e-12) - np.log(TAU_SIG)) / T))


# ----------------------------------------------------------------------------
# Panel (a): the soft-decay gate
# ----------------------------------------------------------------------------
SCENES = ["cecum_t1_b", "cecum_t2_b", "cecum_t3_a",
          "sigmoid_t1_a", "sigmoid_t2_a", "sigmoid_t3_a",
          "trans_t1_b", "trans_t2_c", "trans_t4_a", "trans_t4_b"]


def pooled_opacity(group_dir):
    """Final opacities of every Gaussian in every scene of a group."""
    out = []
    for s in SCENES:
        logit = np.load(os.path.join(group_dir, s, "params.npz"))["logit_opacities"]
        out.append(1.0 / (1.0 + np.exp(-logit.ravel())))
    return np.concatenate(out)


def panel_hist(ax, baseline_dir, tvs_dir, cache):
    """(a) Final opacity distribution, baseline against TVS before cleanup.

    The TVS group must be a run *without* the final cleanup: once the cleanup
    has run, the population this panel exists to show has already been deleted
    from params.npz, and the histogram would be empty below the threshold.
    """
    if cache and os.path.exists(cache):
        z = np.load(cache)
        hb, ht, edges, fb, ft = z["hb"], z["ht"], z["edges"], z["fb"], z["ft"]
    else:
        b, t = pooled_opacity(baseline_dir), pooled_opacity(tvs_dir)
        edges = np.logspace(-3, 0, 40)
        hb = np.histogram(np.clip(b, 1e-3, 1), bins=edges)[0] / len(b) * 100
        ht = np.histogram(np.clip(t, 1e-3, 1), bins=edges)[0] / len(t) * 100
        fb, ft = (b <= ALPHA_CLEAN).mean() * 100, (t <= ALPHA_CLEAN).mean() * 100
        if cache:
            np.savez(cache, hb=hb, ht=ht, edges=edges, fb=fb, ft=ft)

    mid = np.sqrt(edges[:-1] * edges[1:])
    ax.axvspan(1e-3, ALPHA_CLEAN, color=RED, alpha=0.07)
    ax.step(mid, np.maximum(hb, 1e-3), where="mid", color=GRAY)
    ax.step(mid, np.maximum(ht, 1e-3), where="mid", color=GREEN)
    ax.axvline(ALPHA_CLEAN, color=GRAY, ls="--", lw=0.8)

    # labelled on the curves rather than in a legend: the panel is 2.2 in wide
    # and a legend box lands on top of one series or the other
    # everything sits above the curves: the band's lower half is crossed by the
    # baseline's near-vertical drop, which was overprinting the caption text
    ax.text(1.6e-3, 6, "TVS", fontsize=7, color=GREEN, ha="center")
    ax.text(0.42, 9, "baseline", fontsize=7, color=GRAY, ha="center")
    # the band caption owns the top strip; the threshold label sits mid-height
    # to its right, so the two never share a line
    ax.text(1.15e-3, 105, "removed by cleanup", fontsize=6.5, color=RED,
            va="center")
    ax.annotate(r"$\alpha_{\mathrm{clean}}$", xy=(ALPHA_CLEAN, 22),
                xytext=(3, 0), textcoords="offset points", fontsize=7)
    ax.text(0.95, 0.075, "%.0f%% vs %.0f%% below" % (ft, fb),
            fontsize=6.5, ha="right", color="black")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e-3, 1.0); ax.set_ylim(0.05, 150)
    ax.set_xlabel(r"final opacity $\alpha_i$ (log)")
    ax.set_ylabel("% of Gaussians (log)")
    ax.set_title("(a) opacity distribution")


# ----------------------------------------------------------------------------
# Panel (b): reversible opacity, simulated faithfully from the update rule
# ----------------------------------------------------------------------------
def panel_opacity(ax):
    n = 195
    # seen -> unobserved -> re-observed -> unobserved again to the end of the
    # stream, where the cleanup deletes what never came back
    v = np.ones(n)
    v[40:95] = 0.0
    v[130:] = 0.0
    alpha, f = 0.80, 1.0          # start from a matured, visible Gaussian
    A = np.empty(n)
    for t in range(n):
        f = (1 - LAMBDA) * f + LAMBDA * v[t]      # EMA visibility frequency (eq. 3)
        if v[t] > 0.5:
            # re-observed: mapping gradients pull opacity back up; gate ~ 1
            alpha = min(alpha + 0.06 * (0.80 - alpha), 0.80)
        else:
            tvs = f * (alpha ** BETA)             # eq. 4
            alpha = max(alpha * gate(tvs), ALPHA_FLOOR)   # eq. 5-6
        A[t] = alpha
    frames = np.arange(n)

    ax.axvspan(0, 40, color=GREEN, alpha=0.06)
    ax.axvspan(40, 95, color=RED, alpha=0.06)
    ax.axvspan(95, 130, color=GREEN, alpha=0.06)
    ax.axvspan(130, n, color=RED, alpha=0.06)
    ax.plot(frames, A, color=BLUE)
    ax.axhline(ALPHA_FLOOR, color=GRAY, ls="--", lw=0.8)
    ax.text(2, ALPHA_FLOOR + 0.03,
            r"$\alpha_{\mathrm{floor}}=%.2f$" % ALPHA_FLOOR, fontsize=7)

    # end of stream: recovery is no longer possible, so the cleanup deletes it
    ax.axvline(n - 1, color="black", ls=":", lw=0.9)
    # clip_on=False: the marker sits on the floor line at the right-hand edge,
    # where the axes would otherwise cut most of it away
    ax.plot([n - 1], [A[-1]], marker="x", ms=8, mew=2.2, color=RED,
            zorder=6, clip_on=False)
    ax.annotate("deleted by\nfinal cleanup", xy=(n - 1, A[-1]),
                xytext=(-16, 30), textcoords="offset points", fontsize=6.5,
                color=RED, ha="right", va="bottom",
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.9,
                                shrinkA=0, shrinkB=4))
    ax.text(n - 4, 0.93, "stream ends", fontsize=6.5, ha="right",
            color="black", rotation=90, va="top")

    ax.text(20, 0.90, "visible", fontsize=7, ha="center", color=GREEN)
    ax.text(67, 0.52, "unobserved\n$\\rightarrow$ decay",
            fontsize=7, ha="center", color=RED)
    ax.text(112, 0.90, "re-observed", fontsize=7, ha="center", color=GREEN)

    ax.set_xlim(0, n + 8)          # room for the deletion marker at the edge
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("frame")
    ax.set_ylabel(r"opacity $\alpha_i$")
    ax.set_title("(b) reversible opacity, then deletion")


# ----------------------------------------------------------------------------
# Panel (c): spatial floater mask, side-view schematic
# ----------------------------------------------------------------------------
def panel_mask(ax):
    ax.set_xlim(0, 10.5)
    ax.set_ylim(-0.6, 6.6)
    ax.axis("off")

    lx, ly, fx, s = 1.15, 3.0, 8.6, 0.3356   # lens apex, far edge, frustum half-slope

    # camera
    ax.add_patch(Rectangle((0.4, 2.6), 0.7, 0.8, fc="none", ec=GRAY, lw=0.8))
    ax.add_patch(Ellipse((lx, ly), 0.3, 0.3, fc=GRAY, ec=GRAY))
    ax.text(0.75, 2.2, "camera", fontsize=7, ha="center", color=GRAY)

    # frustum
    ax.plot([lx, fx], [ly, ly + s * (fx - lx)], ls="--", color=GRAY, lw=0.6)
    ax.plot([lx, fx], [ly, ly - s * (fx - lx)], ls="--", color=GRAY, lw=0.6)

    # keep band (surface +/- gamma*D)
    ax.add_patch(Rectangle((6.45, 0.6), 1.1, 4.8, fc=BAND, ec="none", alpha=0.6))
    ax.plot([6.45, 6.45], [0.6, 5.4], ls=":", color=GRAY, lw=0.8)
    ax.plot([7.55, 7.55], [0.6, 5.4], ls=":", color=GRAY, lw=0.8)
    ax.annotate("reserved band", xy=(7.55, 4.3), xytext=(8.5, 4.7),
                fontsize=7, color="black", ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.6, ls=":"))

    # tissue surface (depth D)
    ys = np.linspace(0.7, 5.4, 60)
    xs = 7.0 + 0.15 * np.sin(ys * 2.2)
    ax.plot(xs, ys, color=GREEN, lw=2.2)
    ax.text(7.0, 5.9, "surface (depth $D$)", fontsize=7, ha="center", color=GREEN)

    # kept Gaussian on the surface (label to the clear right)
    ax.add_patch(Ellipse((7.0, 3.0), 0.5, 0.32, fc=GREEN, ec="none", alpha=0.9))
    ax.annotate("on surface\n$\\rightarrow$ keep", xy=(7.35, 2.9), xytext=(8.7, 2.4),
                fontsize=7, color=GREEN, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.6, ls=":"))

    # front floater (label in the clear upper-left wedge)
    ax.add_patch(Ellipse((4.0, 3.0), 0.6, 0.4, fc=RED, ec="none", alpha=0.5))
    ax.text(2.6, 5.3, "front floater", fontsize=7, ha="center", color=RED)
    ax.annotate(r"$|z_i-D|>\gamma D$", xy=(3.7, 3.3), xytext=(2.6, 4.7),
                fontsize=7, color="black", ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.6, ls=":"))

    # back floater (behind the surface; label kept clear of the band, to the right)
    ax.add_patch(Ellipse((8.2, 1.9), 0.46, 0.3, fc=RED, ec="none", alpha=0.5))
    ax.text(7.75, 1.1, "back floater", fontsize=7, ha="left", color=RED)

    # depth axis
    ax.add_patch(FancyArrowPatch((1.0, -0.1), (9.4, -0.1),
                                 arrowstyle="-|>", mutation_scale=8, color="black", lw=0.7))
    ax.text(5.0, -0.45, "depth", fontsize=7, ha="center")

    # mask attenuation (clear lower-left wedge)
    ax.text(3.4, 1.0, r"$\alpha_i\leftarrow %.1f\,\alpha_i$" % ETA_SPATIAL,
            fontsize=7, ha="center", color="black")

    ax.set_title("(c) spatial floater mask")


DESK = r"C:\Users\lenov\OneDrive\Desktop"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_group",
                    default=os.path.join(DESK, "experiment0.1", "Baseline"))
    ap.add_argument("--tvs_group", default=os.path.join(DESK, "experiment_mo15", "full"),
                    help="A TVS run WITHOUT the final cleanup; a cleaned run "
                         "has already deleted what panel (a) must show.")
    ap.add_argument("--cache", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "opacity_hist.npz"))
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.6))
    panel_hist(axes[0], args.baseline_group, args.tvs_group, args.cache)
    panel_opacity(axes[1])
    panel_mask(axes[2])
    fig.tight_layout(w_pad=1.4)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    png = os.path.join(out_dir, "tvs_mechanism.png")
    pdf = os.path.join(out_dir, "tvs_mechanism.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print("wrote", png)
    print("wrote", pdf)


if __name__ == "__main__":
    main()
