"""Assemble teaser Row 3: mini map-growth curve + baseline-vs-ours checkmark table.

Plots straight from the per_frame_gauss.csv files, the same sources Fig. 4 uses,
so the teaser and Fig. 4 cannot drift apart. The previous version digitised the
pixels of pictures/map_growth.png, which silently reproduced whatever curve
happened to be on disk -- that is how the teaser ended up still showing the old
sigmoid_t2_a run and $-12\\%$ after the results changed.

Axes carry no numbers: this is the glance figure, and Fig. 4 is where the reader
goes for values. The percentages in the annotations are computed from the raw
endpoints so they match Fig. 4 and the tables exactly.

Usage
-----
    python docs/make_teaser_row3.py \\
        --baseline <dir>/C3VD_baseline_growth/sigmoid_t1_a \\
        --tvs      <dir>/C3VD_full_mo15_v2/sigmoid_t1_a \\
        --full     <dir>/C3VD_full_mo15_v2/sigmoid_t1_a
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_map_growth import load_curve, final_count, smooth   # noqa: E402

OUT = r"D:\project26.01\new_eGSLAM\pictures"
RED, GREEN = "#c1121f", "#2a9d3a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--tvs", required=True)
    ap.add_argument("--full", required=True)
    ap.add_argument("--scene", default="sigmoid_t1_a")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    bx, by = load_curve(args.baseline)
    gx, gy = load_curve(args.tvs)
    clean = final_count(args.full)
    base_end, tvs_end = by[-1], gy[-1]
    d_online = 100 * (1 - tvs_end / base_end)
    d_clean = 100 * (1 - clean / base_end)

    fig = plt.figure(figsize=(7.2, 1.9), dpi=300)

    # ---- left: mini map-growth curve ----
    axc = fig.add_axes([0.06, 0.22, 0.40, 0.64])
    bys, gys = smooth(by), smooth(gy)
    axc.plot(bx, bys, color=RED, lw=1.8, label="baseline")
    axc.plot(gx, gys, color=GREEN, lw=1.8, label="ours (TVS)")
    axc.fill_between(gx, gys, np.interp(gx, bx, bys), color=RED, alpha=0.10, lw=0)

    # cleanup step: happens after refinement, outside the online loop
    axc.plot([gx[-1], gx[-1]], [gys[-1], clean], color=GREEN, lw=1.6,
             ls=(0, (2, 1.5)))
    axc.plot([gx[-1]], [clean], marker="o", ms=4.5, color=GREEN)

    axc.annotate(f"$-${d_online:.0f}%", xy=(gx[-1], (base_end + tvs_end) / 2),
                 xytext=(4, 0), textcoords="offset points", fontsize=8.5,
                 fontweight="bold", color=GREEN, ha="left", va="center",
                 annotation_clip=False)
    # inside the axes, below-left of the dot: the strip to the right of the
    # curve is only wide enough for one label before it runs into the table
    axc.annotate(f"$-${d_clean:.0f}%\nafter cleanup", xy=(gx[-1], clean),
                 xytext=(-7, -3), textcoords="offset points", fontsize=8.5,
                 fontweight="bold", color=GREEN, ha="right", va="top")

    axc.set_title(f"map growth ({args.scene})", fontsize=9,
                  fontweight="bold", pad=3)
    axc.set_xlabel("frame", fontsize=8, labelpad=1)
    axc.set_ylabel("#Gaussians", fontsize=8, labelpad=1)
    axc.set_xticks([]); axc.set_yticks([])
    for s in ("top", "right"):
        axc.spines[s].set_visible(False)
    axc.legend(fontsize=7.5, loc="lower left", frameon=False,
               handlelength=1.4, borderaxespad=0.1)
    axc.set_xlim(0, gx[-1] * 1.02)
    axc.set_ylim(0, base_end * 1.15)

    # ---- right: checkmark table ----
    axt = fig.add_axes([0.60, 0.02, 0.38, 0.96]); axt.axis("off")
    axt.set_xlim(0, 1); axt.set_ylim(0, 1)
    rows = ["floater suppression", "bounded map growth", "map rebalancing"]
    cB, cO = 0.62, 0.86
    axt.text(cB, 0.92, "baseline", fontsize=9, ha="center", fontweight="bold")
    axt.text(cO, 0.92, "ours", fontsize=9, ha="center", fontweight="bold")
    for i, name in enumerate(rows):
        y = 0.72 - i * 0.26
        axt.text(0.44, y, name, fontsize=9, ha="right", va="center")
        axt.text(cB, y, "✗", fontsize=12, ha="center", va="center", color=RED)
        axt.text(cO, y, "✓", fontsize=12, ha="center", va="center",
                 color=GREEN, fontweight="bold")
    axt.plot([0.06, 0.97], [0.855, 0.855], color="0.55", lw=0.8)

    os.makedirs(args.out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.out, f"teaser_row3.{ext}"),
                    bbox_inches="tight", facecolor="white")
    print(f"baseline end {base_end * 1000:,.0f}  online {tvs_end * 1000:,.0f} "
          f"({-d_online:+.1f}%)  cleaned {clean * 1000:,.0f} ({-d_clean:+.1f}%)")
    print("saved teaser_row3.png + .pdf")


if __name__ == "__main__":
    main()
