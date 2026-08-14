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

GRAY, BLUE, GREEN = "#5a5a5a", "#2b6cb0", "#2a9d3a"


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
    ap.add_argument("--tv", required=True)
    ap.add_argument("--tvs", required=True)
    ap.add_argument("--full", required=True)
    ap.add_argument("--out", default="pictures/map_growth")
    ap.add_argument("--scene", default="sigmoid_t1_a")
    ap.add_argument("--no-cleanup", action="store_true")
    ap.add_argument("--width", type=float, default=3.4)   # IEEE single column
    ap.add_argument("--height", type=float, default=2.7)
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(args.width, args.height))

    series = [("Baseline", args.baseline, GRAY, 1.6),
              ("TV", args.tv, BLUE, 1.4),
              ("TVS (TV+Spatial)", args.tvs, GREEN, 1.6)]
    ends = {}
    for label, d, colour, lw in series:
        x, y = load_curve(d)
        ax.plot(x, y, color=colour, lw=lw, label=label, zorder=3)
        ends[label] = (x[-1], y[-1])

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
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=300)
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
