"""Teaser mechanism strip: how TVS soft pruning fades floaters.

Three stages: online map with floater -> TVS scoring/gate -> compacted map
(floater dormant, recoverable). Output: pictures/teaser_mech.png/.pdf
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp

OUT = r"D:\project26.01\new_eGSLAM\pictures"
RED, GREEN, BLUE = "#c1121f", "#2a9d3a", "#1f4bd8"
TISSUE, TEDGE = "#e8a68f", "#b4735c"

fig, ax = plt.subplots(figsize=(7.6, 2.0), dpi=300)
ax.set_xlim(0, 100); ax.set_ylim(0, 26); ax.axis("off")

def surface(x0, x1, y0=5.5, amp=1.6):
    x = np.linspace(x0, x1, 100)
    y = y0 + amp*np.sin((x - x0)/(x1 - x0)*np.pi*1.6 + 0.4)
    ax.plot(x, y, color=TEDGE, lw=2.2, zorder=1)
    ax.fill_between(x, 0.5, y, color=TISSUE, alpha=0.35, lw=0, zorder=0)
    return x, y

def gauss(x, y, w=3.4, h=1.7, ang=0, fc=TISSUE, ec=TEDGE, alpha=0.95, ls="-", lw=1.1):
    ax.add_patch(mp.Ellipse((x, y), w, h, angle=ang, facecolor=fc,
                            edgecolor=ec, alpha=alpha, ls=ls, lw=lw, zorder=3))

# ---------- stage 1: online map with floaters ----------
sx, sy = surface(3, 30)
for xx in (6, 11, 16.5, 22, 27):
    yy = np.interp(xx, sx, sy)
    gauss(xx, yy, ang=np.degrees(np.arctan2(np.gradient(sy)[int((xx-3)/27*99)], 27/99)))
# floater above surface + rarely-seen one
gauss(13, 15.5, fc="#d98f8f", ec=RED, lw=1.6)
gauss(24, 19.0, fc="#d98f8f", ec=RED, lw=1.6)
ax.annotate("floaters", xy=(20, 17.5), xytext=(6.2, 22.6), fontsize=8, color=RED,
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))
ax.text(16.5, 1.8, "Gaussian map", fontsize=8.4, ha="center", fontweight="bold")

# ---------- stage 2: the TVS gate ----------
ax.add_patch(mp.FancyArrow(32.5, 12, 4.2, 0, width=0.7, head_width=2.0,
                           head_length=1.6, color="0.35"))
bx, by, bw, bh = 39, 4.6, 27, 17.4
ax.add_patch(mp.FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.6",
                               facecolor="#f4f4f6", edgecolor="0.45", lw=1.1))
ax.text(bx+bw/2, by+bh-1.4, "TVS soft pruning", fontsize=9, ha="center",
        fontweight="bold")
ax.text(bx+1.6, by+11.2, "TV: rarely observed?", fontsize=7.6, ha="left")
ax.text(bx+1.6, by+8.0,  r"Spatial: $|z_i-D|>\gamma D$?", fontsize=7.6, ha="left")
ax.text(bx+1.6, by+3.6,  r"$\alpha_i \leftarrow \max(\alpha_i d_i,\ \alpha_{\rm floor})$",
        fontsize=8.6, ha="left", color=RED)
# small sigmoid gate icon (top-right corner, clear of the text lines)
gx = np.linspace(0, 1, 40)
gy = 1/(1+np.exp(-(gx-0.5)*12))
ax.plot(bx+bw-5.6+4.4*gx, by+10.6+3.8*gy, color=BLUE, lw=1.6)
ax.text(bx+bw-3.4, by+9.2, "gate", fontsize=7, ha="center", color=BLUE)

# ---------- stage 3: compacted map ----------
ax.add_patch(mp.FancyArrow(68.5, 12, 4.2, 0, width=0.7, head_width=2.0,
                           head_length=1.6, color="0.35"))
sx2, sy2 = surface(75, 99)
for xx in (78, 83, 88, 93.5, 97):
    yy = np.interp(xx, sx2, sy2)
    gauss(xx, yy)
# faded floaters (dormant, recoverable)
gauss(84, 16.0, fc="none", ec=RED, alpha=0.5, ls="--", lw=1.2)
gauss(94, 19.2, fc="none", ec=RED, alpha=0.5, ls="--", lw=1.2)
ax.annotate("faded, recoverable", xy=(88, 17.6), xytext=(75.5, 23.2), fontsize=8,
            color=GREEN, arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.0))
ax.text(87, 1.8, "compact map ($-$45% Gaussians)", fontsize=8.4, ha="center",
        fontweight="bold", color=GREEN)

for ext in ("png", "pdf"):
    fig.savefig(OUT + rf"\teaser_mech.{ext}", bbox_inches="tight",
                facecolor="white", pad_inches=0.04)
print("saved pictures/teaser_mech.png + .pdf")
