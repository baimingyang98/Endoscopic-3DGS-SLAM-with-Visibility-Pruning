"""Assemble teaser Row 2: baseline-vs-ours floater comparison on sigmoid_t1_a f0087.

Zoom box (265,125)-(395,235): baseline shows dark floater specks, ours is clean.
Output: pictures/teaser_row2.png / .pdf
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from PIL import Image

ROOT = r"D:\project26.01\new_eGSLAM\rendering_comparison"
OUT = r"D:\project26.01\new_eGSLAM\pictures"
ZB = (265, 125, 395, 235)          # zoom box x0,y0,x1,y1
ZOOM = 2.6                          # inset magnification
RED, GREEN = "#c1121f", "#2a9d3a"

panels = [
    ("EndoGSLAM: hard-threshold pruning",
     ROOT + r"\baseline\sigmoid_t1_a\color_0087.png",
     "floaters survive,  #G = 622k", RED),
    ("TVS-SLAM (ours): TVS soft pruning",
     ROOT + r"\Ours\sigmoid_t1_a\color_0087.png",
     "floaters faded out,  #G = 340k  ($-$45%)", GREEN),
]

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), dpi=300)
plt.subplots_adjust(left=0.005, right=0.995, top=0.90, bottom=0.10, wspace=0.03)

for ax, (title, path, gtext, gcol) in zip(axes, panels):
    im = np.asarray(Image.open(path).convert("RGB"))
    H, W = im.shape[:2]
    ax.imshow(im)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("0.4"); s.set_linewidth(0.8)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
    ax.set_xlabel(gtext, fontsize=10, fontweight="bold", color=gcol, labelpad=3)

    x0, y0, x1, y1 = ZB
    ax.add_patch(mp.Rectangle((x0, y0), x1 - x0, y1 - y0,
                              fill=False, edgecolor=RED, lw=1.6))
    # magnified inset, bottom-left corner
    crop = im[y0:y1, x0:x1]
    iw, ih = int((x1 - x0) * ZOOM), int((y1 - y0) * ZOOM)
    ix, iy = 6, H - ih - 6           # inset top-left in data coords
    ax.imshow(np.asarray(Image.fromarray(crop).resize((iw, ih), Image.LANCZOS)),
              extent=(ix, ix + iw, iy + ih, iy), zorder=3)
    ax.add_patch(mp.Rectangle((ix, iy), iw, ih, fill=False, edgecolor=RED,
                              lw=1.6, zorder=4))
    # connector from box to inset
    ax.plot([x0, ix + iw], [y1, iy], color=RED, lw=0.7, ls="--", zorder=2)
    ax.set_xlim(0, W); ax.set_ylim(H, 0)

for ext in ("png", "pdf"):
    fig.savefig(OUT + rf"\teaser_row2.{ext}", bbox_inches="tight",
                facecolor="white")
print("saved pictures/teaser_row2.png + .pdf")
