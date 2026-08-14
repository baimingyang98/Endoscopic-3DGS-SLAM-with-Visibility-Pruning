"""Draw the map-growth figure (paper Fig. 4) from per_frame_gauss.csv.

Three online curves -- Baseline, TV, TVS (TV+Spatial) -- plus the drop the
final cleanup produces once the stream ends. The curves come from the online
loop only: per_frame_gauss.csv is written before refinement, so the cleanup
cannot appear in them and is drawn separately as a terminal step. Labelling
that step as part of the online pass would be wrong.

Point --tvs at the *full* arm, not at TVS+Spatial. Their online phases are
configured identically, but CUDA nondeterminism compounds through the
opacity-decay feedback loop and they end 1.8% apart on this scene; taking the
curve from one arm and the cleanup endpoint from the other would put a step in
the figure that no single run produced. The full arm's online phase is exactly
TV+Spatial, so its curve and its cleanup endpoint belong together.

Usage
-----
    python docs/make_map_growth.py \\
        --baseline  <dir>/C3VD_baseline_growth/sigmoid_t1_a \\
        --tv        <dir>/experiment_mo15_v2/TVS/sigmoid_t1_a \\
        --tvs       <dir>/experiment_mo15_v2/full/sigmoid_t1_a \\
        --full      <dir>/experiment_mo15_v2/full/sigmoid_t1_a \\
        --out       pictures/map_growth

Pass --no-cleanup to draw the online curves alone.
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GRAY, BLUE, GREEN = "#6B6B6B", "#2E6DB4", "#1D9E75"

plt.rcParams.update({
    "font.size": 8, "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.titlesize": 9, "axes.labelsize": 8, "axes.linewidth": 0.6,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "lines.antialiased": True,
})


def smooth(y):
    """Savitzky-Golay: preserves peaks and edges better than a moving average.

    Only the drawn line is smoothed. Every percentage in the figure is computed
    from the raw endpoints, so the annotations match the tables exactly even
    where the filter shifts the tail by a few hundred Gaussians.
    """
    w = max(7, (len(y) // 25) | 1)                  # odd window, ~4% of length
    w = min(w, len(y) - 1 if len(y) % 2 == 0 else len(y) - 2)
    if w < 5:
        return y
    try:
        from scipy.signal import savgol_filter
        return savgol_filter(y, w, 3, mode="interp")
    except ImportError:                             # centred moving average
        pad = np.pad(y, w // 2, mode="edge")
        return np.convolve(pad, np.ones(w) / w, mode="valid")


def load_curve(d):
    a = np.loadtxt(os.path.join(d, "per_frame_gauss.csv"),
                   delimiter=",", skiprows=1)
    return a[:, 0], a[:, 1] / 1000.0          # frames, thousands of Gaussians


def final_count(d):
    p = os.path.join(d, "metrics.json")
    if os.path.exists(p):
        return json.load(open(p))["final_gauss_count"] / 1000.0
    import re
    t = open(os.path.join(d, "runtimes.txt")).read()
    return int(re.search(r"Final Gaussian count:\s*(\d+)", t).group(1)) / 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--tv", default=None,
                    help="Optional TV-only curve; omit to plot Baseline vs TVS.")
    ap.add_argument("--tvs", required=True)
    ap.add_argument("--full", required=True)
    ap.add_argument("--out", default="pictures/map_growth")
    ap.add_argument("--scene", default="sigmoid_t1_a")
    ap.add_argument("--no-cleanup", action="store_true")
    ap.add_argument("--no-smooth", action="store_true",
                    help="Draw the raw per-frame counts as the main line.")
    ap.add_argument("--raw", action="store_true", default=True,
                    help="Underlay the unsmoothed counts faintly (default on).")
    ap.add_argument("--no-raw", dest="raw", action="store_false")
    ap.add_argument("--width", type=float, default=3.45)   # IEEE single column
    ap.add_argument("--height", type=float, default=2.6)
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(args.width, args.height))

    # The TV curve is optional. On C3VD it lands within ~1% of TV+Spatial, well
    # inside run-to-run variation, so plotting both draws two lines on top of
    # each other and invites the reader to conclude the spatial term does
    # nothing. Table III carries that comparison with the numbers to support it.
    series = [("Baseline", args.baseline, GRAY)]
    if args.tv:
        series.append(("TV", args.tv, BLUE))
    series.append(("TVS (TV+Spatial)", args.tvs, GREEN))
    ends = {}
    for label, d, colour in series:
        x, y = load_curve(d)
        if args.raw:
            ax.plot(x, y, color=colour, lw=0.5, alpha=0.20, zorder=1)
        ax.plot(x, y if args.no_smooth else smooth(y), color=colour, lw=1.7,
                label=label, solid_capstyle="round", solid_joinstyle="round",
                zorder=2)
        ends[label] = (x[-1], y[-1])          # raw, so annotations match tables

    base_end = ends["Baseline"][1]
    tvs_x, tvs_end = ends["TVS (TV+Spatial)"]
    ax.annotate(f"−{100 * (1 - tvs_end / base_end):.0f}%",
                xy=(tvs_x, (base_end + tvs_end) / 2),
                xytext=(-6, 0), textcoords="offset points",
                ha="right", va="center", fontsize=8, color=GREEN)

    if not args.no_cleanup:
        clean = final_count(args.full)
        # Terminal step: happens after refinement, outside the online loop.
        ax.plot([tvs_x, tvs_x], [tvs_end, clean], color=GREEN, lw=1.4,
                ls=(0, (2, 1.5)), zorder=3)
        ax.plot([tvs_x], [clean], marker="o", ms=4, color=GREEN, zorder=4)
        ax.annotate(f"after cleanup\n−{100 * (1 - clean / base_end):.0f}%",
                    xy=(tvs_x, clean), xytext=(-8, -2),
                    textcoords="offset points", ha="right", va="top",
                    fontsize=8, color=GREEN)
        ax.set_ylim(0, base_end * 1.12)

    ax.set_xlabel("frame")
    ax.set_ylabel("Gaussians (k)")
    ax.set_title(f"Map growth ({args.scene})", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(x=0.02)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(f"{args.out}.png", dpi=400, bbox_inches="tight")
    fig.savefig(f"{args.out}.pdf", bbox_inches="tight")
    print(f"wrote {args.out}.png / .pdf")
    for k, (_, v) in ends.items():
        print(f"  {k:<20} online end {v * 1000:9.0f}  "
              f"({100 * (v / base_end - 1):+5.1f}% vs baseline)")
    if not args.no_cleanup:
        c = final_count(args.full)
        print(f"  {'full after cleanup':<20} {c * 1000:9.0f}  "
              f"({100 * (c / base_end - 1):+5.1f}% vs baseline)")


if __name__ == "__main__":
    main()
