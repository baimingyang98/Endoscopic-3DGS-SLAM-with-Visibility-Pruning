"""
Generate a paper-style workflow figure for EndoGSLAM with Innovations.

This version shows a COMPLETE LINEAR PIPELINE with the two innovations
(Visibility-Aware Pruning + Periodic BA) highlighted inline within the
actual SLAM loop. Innovation 3 (deformation) is omitted because it was
not used in the C3VD experiments.

Layout:
- Top row (~30%): linear pipeline (Input -> Init -> Tracking -> Mapping -> Output)
- Bottom rows (~65%): two innovation spotlights with detailed sub-modules

Usage:
    python docs/make_workflow_figure.py
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from PIL import Image


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
PIC_DIR = Path(r"C:/Users/lenov/OneDrive/Desktop/pic4project")
OUTPUT = Path(__file__).parent / "workflow_figure.png"

# Compact canvas: width fits pipeline, height makes spotlights breathable
CANVAS_W = 20.0
CANVAS_H = 11.5

COLORS = {
    "input":  "#FFE8E0",
    "init":   "#FFE4E1",
    "track":  "#D4F1D4",
    "map":    "#EAF3FF",
    "out":    "#E8E8F4",
    "inno1":  "#FFF1B8",
    "inno2":  "#D6F0E8",
    "edge":   "#444444",
    "arrow":  "#555555",
}

BORDERS = {
    "input":  "#E08C70",
    "init":   "#E08CA0",
    "track":  "#7DBE7D",
    "map":    "#6FA0D7",
    "out":    "#9090B0",
    "inno1":  "#D4A700",
    "inno2":  "#5FB58E",
}


def load_img(path, max_side=320):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return np.array(img)


def add_module(ax, x, y, w, h, title, color_key, title_size=11,
               highlight=False):
    fill = COLORS[color_key]
    border = BORDERS[color_key]

    if highlight:
        linestyle = "solid"
        linewidth = 2.8
    else:
        linestyle = (0, (5, 3))
        linewidth = 1.8

    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        linewidth=linewidth, linestyle=linestyle,
        facecolor=fill, edgecolor=border,
        zorder=1,
    )
    ax.add_patch(box)

    char_w = 0.115 * (title_size / 11.0)
    title_w = char_w * len(title) + 0.6
    title_x = x + (w - title_w) / 2
    title_y = y + h - 0.30

    title_box = FancyBboxPatch(
        (title_x, title_y), title_w, 0.55,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.5,
        linestyle="solid" if highlight else (0, (4, 2)),
        facecolor="white", edgecolor=border,
        zorder=2,
    )
    ax.add_patch(title_box)
    ax.text(
        title_x + title_w / 2, title_y + 0.275, title,
        ha="center", va="center",
        fontsize=title_size, fontweight="bold", color="#222",
        zorder=3,
    )


def add_image(ax, img, x, y, zoom=0.30, label=None, label_size=9):
    imagebox = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(imagebox, (x, y), frameon=True,
                        bboxprops=dict(edgecolor="#666", linewidth=0.8),
                        pad=0.05, zorder=4)
    ax.add_artist(ab)
    if label:
        ax.text(x, y - 0.7, label, ha="center", va="top",
                fontsize=label_size, style="italic", zorder=5)


def add_arrow(ax, x1, y1, x2, y2, color=None, style="->", lw=2.0, zorder=3,
              dashed=False):
    color = color or COLORS["arrow"]
    ls = "dashed" if dashed else "solid"
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=20,
        linewidth=lw, color=color, zorder=zorder, linestyle=ls,
    )
    ax.add_patch(arrow)


def gaussian_schematic(ax, cx, cy, n=4, scale=0.25, alpha=0.55):
    rng = np.random.RandomState(7)
    palette = ["#F4A8C7", "#A8D8F4", "#F4D5A8", "#C4F4A8"]
    for i in range(n):
        dx = rng.uniform(-0.25, 0.25)
        dy = rng.uniform(-0.18, 0.18)
        w = scale * rng.uniform(0.8, 1.2)
        h = scale * rng.uniform(0.6, 1.0)
        ang = rng.uniform(0, 180)
        e = Ellipse((cx + dx, cy + dy), w, h, angle=ang,
                    facecolor=palette[i % len(palette)], edgecolor="#666",
                    alpha=alpha, linewidth=0.8, zorder=4)
        ax.add_patch(e)


def camera_frustum(ax, cx, cy, size=0.25, color="#FFA94D"):
    pts = np.array([
        [cx, cy],
        [cx - size, cy + size * 0.7],
        [cx + size, cy + size * 0.7],
        [cx, cy],
        [cx - size * 1.1, cy + size * 0.9],
        [cx + size * 1.1, cy + size * 0.9],
        [cx, cy],
    ])
    ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.6, zorder=4)


def snowflake(ax, x, y, size=0.3, color="#5BB3D6"):
    for ang in [0, 60, 120]:
        rad = np.deg2rad(ang)
        dx, dy = size * np.cos(rad), size * np.sin(rad)
        ax.plot([x - dx, x + dx], [y - dy, y + dy], color=color, lw=1.8, zorder=5)
    ax.add_patch(plt.Circle((x, y), size * 0.18, color=color, zorder=6))


def flame(ax, x, y, size=0.25, color="#E26A3A"):
    pts = np.array([
        [x, y + size],
        [x + size * 0.6, y + size * 0.2],
        [x + size * 0.4, y - size * 0.3],
        [x, y - size * 0.7],
        [x - size * 0.4, y - size * 0.3],
        [x - size * 0.6, y + size * 0.2],
        [x, y + size],
    ])
    ax.fill(pts[:, 0], pts[:, 1], color=color, alpha=0.85, zorder=5)
    inner = np.array([
        [x, y + size * 0.5],
        [x + size * 0.25, y - size * 0.05],
        [x, y - size * 0.4],
        [x - size * 0.25, y - size * 0.05],
        [x, y + size * 0.5],
    ])
    ax.fill(inner[:, 0], inner[:, 1], color="#FFD060", alpha=0.9, zorder=6)


def star_badge(ax, x, y, label, color="#D4A700"):
    pts = []
    for i in range(10):
        ang = -np.pi / 2 + i * np.pi / 5
        r = 0.20 if i % 2 == 0 else 0.09
        pts.append([x + r * np.cos(ang), y + r * np.sin(ang)])
    pts = np.array(pts)
    ax.fill(pts[:, 0], pts[:, 1], color=color, edgecolor="#8B7000",
            linewidth=1.0, zorder=10)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="white", zorder=11)


def step_circle(ax, x, y, num, size=0.18, color="#666"):
    """Draw a small numbered circle for sequence steps (①②③)."""
    ax.add_patch(plt.Circle((x, y), size, facecolor="white",
                            edgecolor=color, linewidth=1.5, zorder=10))
    ax.text(x, y, str(num), ha="center", va="center",
            fontsize=10, fontweight="bold", color=color, zorder=11)


# ============================================================
# Main figure
# ============================================================
def main():
    paths = {
        "rgb":      PIC_DIR / "215" / "215f_gt.png",
        "baseline": PIC_DIR / "215" / "215_bl.png",
        "ours":     PIC_DIR / "215" / "215_ours.png",
        "depth":    PIC_DIR / "depth.png",
    }
    imgs = {}
    for k, p in paths.items():
        if p.exists():
            imgs[k] = load_img(p, max_side=320)
        else:
            print(f"WARN: missing {p}")
            imgs[k] = np.ones((100, 100, 3), dtype=np.uint8) * 200

    if imgs["depth"].shape[1] > imgs["depth"].shape[0] * 2:
        h = imgs["depth"].shape[0]
        imgs["depth_crop"] = imgs["depth"][:, h:h*2]
    else:
        imgs["depth_crop"] = imgs["depth"]

    fig, ax = plt.subplots(figsize=(CANVAS_W, CANVAS_H), dpi=160)
    ax.set_xlim(0, CANVAS_W)
    ax.set_ylim(0, CANVAS_H)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Title
    ax.text(CANVAS_W / 2, CANVAS_H - 0.35,
            "Visibility-Driven Gaussian Map Management for Endoscopic SLAM",
            ha="center", va="center", fontsize=15, fontweight="bold")

    # =========================================================
    # ROW 1 (TOP): Linear pipeline
    # y range: 7.6 - 10.7
    # =========================================================
    R1_Y = 7.6
    R1_H = 3.1

    # ---- Input ----
    add_module(ax, 0.4, R1_Y, 3.4, R1_H, "RGB-D Input", "input")
    add_image(ax, imgs["rgb"], 1.4, R1_Y + 1.85, zoom=0.33, label="RGB")
    add_image(ax, imgs["depth_crop"], 2.85, R1_Y + 1.85, zoom=0.58, label="Depth")
    ax.text(2.1, R1_Y + 0.55, "Per-frame stream\n(C3VD / StereoMIS)",
            ha="center", va="center", fontsize=9.5, style="italic", color="#444")

    # ---- Initialization ----
    add_module(ax, 4.0, R1_Y, 3.0, R1_H, "Initialization", "init")
    ax.annotate("$\\mathcal{G}_0$", xy=(4.7, R1_Y + 1.55),
                ha="center", va="center",
                fontsize=17, fontweight="bold", color="#444", zorder=5)
    add_arrow(ax, 5.05, R1_Y + 1.55, 5.45, R1_Y + 1.55, lw=1.8)
    gaussian_schematic(ax, 5.95, R1_Y + 1.55, n=5, scale=0.5)
    ax.text(5.5, R1_Y + 2.4, "Frame 0 → Initial Gaussians",
            ha="center", va="center", fontsize=9.5, style="italic", color="#444")
    ax.text(5.5, R1_Y + 2, "Camera Pose → Identity",
            ha="center", va="center", fontsize=9.5, style="italic", color="#444")
    ax.text(5.5, R1_Y + 0.55, "(point cloud back-projection)",
            ha="center", va="center", fontsize=9, color="#666")

    # ---- Tracking ----
    add_module(ax, 7.2, R1_Y, 3.4, R1_H, "Tracking", "track")
    camera_frustum(ax, 7.7, R1_Y + 2.1, size=0.28, color="#E58C2A")
    ax.text(7.7, R1_Y + 2.45, "$\\hat{\\mathcal{T}}_t$",
            ha="center", va="bottom", fontsize=12, fontweight="bold")
    add_arrow(ax, 8.05, R1_Y + 2.15, 8.55, R1_Y + 2.15, lw=1.6)
    camera_frustum(ax, 8.95, R1_Y + 2.1, size=0.28, color="#E58C2A")
    ax.text(8.95, R1_Y + 2.45, "$\\hat{\\mathcal{T}}_{t+1}$",
            ha="center", va="bottom", fontsize=12, fontweight="bold")
    snowflake(ax, 9.85, R1_Y + 2.15, size=0.20, color="#5BB3D6")
    ax.text(9.85, R1_Y + 1.65, "Gaussians frozen ", ha="center", va="center",
            fontsize=8.5, color="#444")

    ax.text(8.85, R1_Y + 1.05,
            r"$\mathcal{L}_{tr} = w_d\|D-\hat D\|_1 + w_c\|C-\hat C\|_1$",
            ha="center", va="center", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#888"))
    ax.text(8.85, R1_Y + 0.45, "Camera params only",
            ha="center", va="center", fontsize=8.5, style="italic", color="#444")

    # ---- Mapping (per-frame) - widened to 6.0 with longer arrows ----
    MAP_X, MAP_W = 10.8, 6.0
    add_module(ax, MAP_X, R1_Y, MAP_W, R1_H, "Mapping (per-frame)", "map")

    sub_y = R1_Y + 1.75

    # Step ①: Add Gaussians (small icon, no box)
    step_circle(ax, MAP_X + 0.45, sub_y + 0.30, "1", size=0.18)
    ax.text(MAP_X + 0.45, sub_y - 0.15, "Add\nGaussians",
            ha="center", va="center", fontsize=8.5, color="#444")
    add_arrow(ax, MAP_X + 0.75, sub_y, MAP_X + 1.4, sub_y, lw=1.6)

    # Step ②: Modified CUDA Rasterizer (highlighted as Innovation 1)
    RAST_X, RAST_W = MAP_X + 1.4, 1.8
    rast_box = FancyBboxPatch(
        (RAST_X, sub_y - 0.40), RAST_W, 0.85,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=2.2, facecolor=COLORS["inno1"], edgecolor=BORDERS["inno1"],
        zorder=2,
    )
    ax.add_patch(rast_box)
    # ax.text(RAST_X + RAST_W/2, sub_y + 0.13, "Modified",
    #         ha="center", va="center", fontsize=9.5, fontweight="bold", color="#222")
    ax.text(RAST_X + RAST_W/2, sub_y + 0.13, "CUDA Rasterizer",
            ha="center", va="center", fontsize=9.5, fontweight="bold", color="#222")
    ax.text(RAST_X + RAST_W/2, sub_y - 0.18, "Visibility Calculation",
            ha="center", va="center", fontsize=9.5, fontweight="bold", color="#222")
    step_circle(ax, RAST_X + 0.20, sub_y + 0.50, "2", size=0.18, color="#8B6F00")

    add_arrow(ax, RAST_X + RAST_W, sub_y, RAST_X + RAST_W + 0.65, sub_y, lw=1.6)

    # Step ③: Visibility-Aware Pruning (highlighted as Innovation 1)
    PRUNE_X, PRUNE_W = RAST_X + RAST_W + 0.65, 1.8
    prune_box = FancyBboxPatch(
        (PRUNE_X, sub_y - 0.40), PRUNE_W, 0.85,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=2.2, facecolor=COLORS["inno1"], edgecolor=BORDERS["inno1"],
        zorder=2,
    )
    ax.add_patch(prune_box)
    ax.text(PRUNE_X + PRUNE_W/2, sub_y + 0.13, "Visibility-Aware",
            ha="center", va="center", fontsize=9.5, fontweight="bold", color="#222")
    ax.text(PRUNE_X + PRUNE_W/2, sub_y - 0.18, "Pruning",
            ha="center", va="center", fontsize=9.5, fontweight="bold", color="#222")
    step_circle(ax, PRUNE_X + 0.20, sub_y + 0.50, "3", size=0.18, color="#8B6F00")

    # Innovation 1 banner spanning both highlighted blocks
    banner_x = RAST_X - 0.05
    banner_w = (PRUNE_X + PRUNE_W) - banner_x + 0.05
    banner_y = sub_y + 0.95
    banner = FancyBboxPatch(
        (banner_x, banner_y - 0.20), banner_w, 0.36,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.5, facecolor="#FFF9E0", edgecolor=BORDERS["inno1"],
        zorder=2,
    )
    # ax.add_patch(banner)
    # ax.text(banner_x + banner_w/2, banner_y, "★ Innovation 1",
    #         ha="center", va="center", fontsize=9.5, fontweight="bold",
    #         color="#8B6F00", zorder=3)

    # V_i flow indicator below the rasterizer block
    ax.text(RAST_X + RAST_W/2, R1_Y + 0.55,
            r"per-Gaussian visibility $V_i$",
            ha="center", va="center", fontsize=9, style="italic", color="#8B6F00",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#FFF9E0",
                     edgecolor="#D4A700"))
    ax.text(PRUNE_X + PRUNE_W/2, R1_Y + 0.55, "backprop $\\to$ update",
            ha="center", va="center", fontsize=9, style="italic", color="#444")

    # ---- Output ---- (shifted right because Mapping is wider)
    add_module(ax, 17.0, R1_Y, 2.6, R1_H, "Output", "out")
    add_image(ax, imgs["ours"], 17.85, R1_Y + 1.85, zoom=0.28, label="Rendered")
    ax.text(18.3, R1_Y + 0.55,
            "+Camera trajectory\n+ Gaussian map (.npz)\n+ Per-frame metrics",
            ha="center", va="center", fontsize=8.5, style="italic", color="#444")

    # =========================================================
    # Pipeline arrows between top-row boxes (longer/clearer)
    # =========================================================
    pipeline_y = R1_Y + R1_H / 2
    pipeline_arrows = [
        (3.8, 4.0),     # Input -> Init
        (7.0, 7.2),     # Init -> Tracking
        (10.6, 10.8),   # Tracking -> Mapping
        (16.8, 17.0),   # Mapping -> Output (gap shifted)
    ]
    for x_start, x_end in pipeline_arrows:
        add_arrow(ax, x_start, pipeline_y, x_end, pipeline_y, lw=2.4, color="#333")

    # Loop-back arrow (next frame): clean dashed loop UNDER the pipeline
    # Goes from Output bottom -> down -> left -> up to Tracking bottom
    loop_y = R1_Y - 0.35
    ax.plot([17.8, 17.8, 8.85, 8.85],
            [R1_Y, loop_y, loop_y, R1_Y],
            color="#888", linestyle="dashed", linewidth=1.3, zorder=0)
    ax.annotate("", xy=(8.85, R1_Y), xytext=(8.85, R1_Y - 0.15),
                arrowprops=dict(arrowstyle="->", color="#666", lw=1.3))
    ax.text(13.3, loop_y + 0.05, "next frame", ha="center", va="center",
            fontsize=9.5, style="italic", color="#666",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                     edgecolor="none"))

    # =========================================================
    # ROW 2: Two innovation spotlights (shorter / more compact)
    # y range: 0.5 - 6.0 (height 5.5)
    # =========================================================
    R2_Y = 0.5
    R2_H = 5.5

    # Section header
    # ax.text(CANVAS_W / 2, R2_Y + R2_H + 0.15,
    #         "─── Innovation Spotlights ───",
    #         ha="center", va="center", fontsize=12, fontweight="bold",
    #         style="italic", color="#444")

    # ---- SPOTLIGHT 1: Visibility-Aware Pruning ----
    SP1_X, SP1_W = 0.4, 9.6
    add_module(ax, SP1_X, R2_Y, SP1_W, R2_H,
               "Innovation 1 — Visibility-Aware Pruning",
               "inno1", title_size=12, highlight=True)

    star_badge(ax, SP1_X + 0.4, R2_Y + R2_H - 0.05, "1")

    # Vertical positions for the 3 sub-sections (compact)
    y_a = R2_Y + R2_H - 1.05
    y_b = R2_Y + R2_H - 2.55
    y_c = R2_Y + R2_H - 3.95

    # ---- (a) CUDA-level extraction ----
    ax.text(SP1_X + 0.5, y_a, "(a)", ha="left", va="center",
            fontsize=11, fontweight="bold", color="#8B6F00")
    ax.text(SP1_X + 0.95, y_a, "CUDA-level visibility extraction (free)",
            ha="left", va="center", fontsize=11, fontweight="bold", color="#222")

    ax.text(SP1_X + SP1_W / 2, y_a - 0.6,
            r"$V_i = \sum_p \alpha_i^{(p)} \cdot T^{(p)}$  (one extra atomicAdd per pixel — zero training, zero extra params)",
            ha="center", va="center", fontsize=10.5, style="italic", color="#444",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#aaa"))

    # ---- (b) Visibility history buffer ----
    ax.text(SP1_X + 0.5, y_b, "(b)", ha="left", va="center",
            fontsize=11, fontweight="bold", color="#8B6F00")
    ax.text(SP1_X + 0.95, y_b, "Visibility history buffer (W=15 frames)",
            ha="left", va="center", fontsize=11, fontweight="bold", color="#222")

    buf_y = y_b - 0.65
    n_squares = 14
    sq_w = 0.45
    sq_gap = 0.50
    buf_start_x = SP1_X + 1.7
    for i, alpha in enumerate([0.3, 0.5, 0.7, 0.9, 1.0, 0.85, 0.7, 0.55,
                                0.4, 0.3, 0.45, 0.6, 0.7, 0.5][:n_squares]):
        ax.add_patch(mpatches.Rectangle((buf_start_x + i*sq_gap, buf_y - 0.21),
                                        sq_w, 0.42,
                                        facecolor=plt.cm.viridis(alpha),
                                        edgecolor="#444", linewidth=0.5, zorder=4))
    ax.text(SP1_X + 0.55, buf_y, "$V_i^{(t)}$:",
            ha="left", va="center", fontsize=10, color="#444")
    # Statistics label BELOW the buffer (not to the side, where it overlaps)
    ax.text(SP1_X + SP1_W / 2, buf_y - 0.55,
            r"→ mean $\bar V_i$ (visibility magnitude),  var $\sigma_V^2$ (consistency over frames)",
            ha="center", va="center", fontsize=9.5, style="italic", color="#444")

    # ---- (c) Dual-mask pruning ----
    # Move (c) up a bit to leave clear room for the highlighted bottom equation
    y_c_actual = y_c
    ax.text(SP1_X + 0.5, y_c_actual, "(c)", ha="left", va="center",
            fontsize=11, fontweight="bold", color="#8B6F00")
    ax.text(SP1_X + 0.95, y_c_actual, "Dual-mask pruning + opacity degeneration",
            ha="left", va="center", fontsize=11, fontweight="bold", color="#222")

    rule_y = y_c_actual - 0.65
    rule1 = FancyBboxPatch(
        (SP1_X + 0.6, rule_y - 0.30), 4.1, 0.6,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        linewidth=1.2, facecolor="white", edgecolor="#aaa", zorder=3,
    )
    ax.add_patch(rule1)
    ax.text(SP1_X + 2.65, rule_y + 0.07,
            r"Mask $V$:  $\bar V_i < \tau_v$  ∧  $age_i > 5$",
            ha="center", va="center", fontsize=10, color="#222")
    ax.text(SP1_X + 2.65, rule_y - 0.18, "(low visibility)",
            ha="center", va="center", fontsize=8.5, style="italic", color="#666")

    rule2 = FancyBboxPatch(
        (SP1_X + 4.9, rule_y - 0.30), 4.1, 0.6,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        linewidth=1.2, facecolor="white", edgecolor="#aaa", zorder=3,
    )
    ax.add_patch(rule2)
    ax.text(SP1_X + 6.95, rule_y + 0.07,
            r"Mask $D$:  $d_{obs} - z_i^{cam} > \gamma$",
            ha="center", va="center", fontsize=10, color="#222")
    ax.text(SP1_X + 6.95, rule_y - 0.18, "(distance / floater)",
            ha="center", va="center", fontsize=8.5, style="italic", color="#666")

    # Highlighted final equation at bottom (separated from rule boxes by gap)
    eff_y = R2_Y + 0.3
    ax.text(SP1_X + SP1_W / 2, eff_y,
            r"$\sigma_i \leftarrow \sigma_i \cdot \eta$  if  ($V \cup D$), where $\eta=0.90$, gentle decay",
            ha="center", va="center", fontsize=11, color="#222",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF9E0",
                     edgecolor=BORDERS["inno1"], linewidth=1.5))

    # ---- SPOTLIGHT 2: Periodic Bundle Adjustment ----
    SP2_X, SP2_W = 10.4, 9.2
    add_module(ax, SP2_X, R2_Y, SP2_W, R2_H,
               "Innovation 2 — Periodic Bundle Adjustment",
               "inno2", title_size=12, highlight=True)

    star_badge(ax, SP2_X + 0.4, R2_Y + R2_H - 0.05, "2", color="#3B9C73")

    y_a2 = R2_Y + R2_H - 1.05
    y_b2 = R2_Y + R2_H - 2.1
    y_c2 = R2_Y + R2_H - 3.95

    # ---- (a) Trigger ----
    ax.text(SP2_X + 0.5, y_a2, "(a)", ha="left", va="center",
            fontsize=11, fontweight="bold", color="#3B7A60")
    ax.text(SP2_X + 0.95, y_a2, "Trigger: every $M = 50$ frames",
            ha="left", va="center", fontsize=11, fontweight="bold", color="#222")

    ax.text(SP2_X + SP2_W / 2, y_a2 - 0.6,
            "To Reduce accumulated pose drift between widely-separated keyframes",
            ha="center", va="center", fontsize=10.5, style="italic", color="#444",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#aaa"))

    # ---- (b) Hybrid keyframe selection ----
    ax.text(SP2_X + 0.5, y_b2, "(b)", ha="left", va="center",
            fontsize=11, fontweight="bold", color="#3B7A60")
    ax.text(SP2_X + 0.95, y_b2, "Hybrid keyframe selection (3 recent + 2 random older)",
            ha="left", va="center", fontsize=11, fontweight="bold", color="#222")

    tl_y = y_b2 - 0.85
    ax.plot([SP2_X + 0.7, SP2_X + 8.5], [tl_y, tl_y], color="#888", lw=1.0, zorder=2)
    for x_off, label in [(1.0, r"$kf_1$"), (2.5, r"$kf_2$")]:
        camera_frustum(ax, SP2_X + x_off, tl_y, size=0.18, color="#3B9C73")
        ax.text(SP2_X + x_off, tl_y - 0.4, label, ha="center", va="top",
                fontsize=8.5, color="#444")
    for x_off, label in [(5.5, r"$kf_{3}$"), (6.7, r"$kf_{4}$"), (7.9, r"$kf_{5}$")]:
        camera_frustum(ax, SP2_X + x_off, tl_y, size=0.18, color="#3B9C73")
        ax.text(SP2_X + x_off, tl_y - 0.4, label, ha="center", va="top",
                fontsize=8.5, color="#444")
    ax.text(SP2_X + 1.7, tl_y + 0.45, "older", ha="center", va="bottom",
            fontsize=9, style="italic", color="#666")
    ax.text(SP2_X + 6.7, tl_y + 0.45, "recent", ha="center", va="bottom",
            fontsize=9, style="italic", color="#666")

    # ---- (c) Joint optimization (moved up to leave space for bottom highlight) ----
    y_c2_actual = y_c2 + 0.15
    ax.text(SP2_X + 0.5, y_c2_actual, "(c)", ha="left", va="center",
            fontsize=11, fontweight="bold", color="#3B7A60")
    ax.text(SP2_X + 0.95, y_c2_actual, "Joint optimization of camera poses + Gaussians",
            ha="left", va="center", fontsize=11, fontweight="bold", color="#222")

    opt_y = y_c2_actual - 0.7
    # flame(ax, SP2_X + 1.6, opt_y, size=0.22)
    camera_frustum(ax, SP2_X + 1.6, opt_y, size=0.18, color="#3B9C73")
    ax.text(SP2_X + 2.05, opt_y, "camera $\\hat{\\mathcal{T}}$",
            ha="left", va="center", fontsize=10.5, color="#222")
    ax.text(SP2_X + 3.5, opt_y, "+", ha="center", va="center",
            fontsize=14, fontweight="bold", color="#444")
    # flame(ax, SP2_X + 4.4, opt_y, size=0.22)
    gaussian_schematic(ax, SP2_X + 4.4, opt_y, n=3, scale=0.5)
    ax.text(SP2_X + 4.85, opt_y, "Gaussians",
            ha="left", va="center", fontsize=10.5, color="#222")
    ax.text(SP2_X + 7.15, opt_y, "20 iters,  ~1.5% overhead",
            ha="center", va="center", fontsize=10, style="italic", color="#666",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#bbb"))

    # Highlighted final note at bottom
    ax.text(SP2_X + SP2_W / 2, R2_Y + 0.45,
            r"Conservative camera LR during BA: $\frac{1}{2}\times$ tracking LR (avoid destabilizing converged poses)",
            ha="center", va="center", fontsize=11, color="#222",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F8F0",
                     edgecolor=BORDERS["inno2"], linewidth=1.5))

    # =========================================================
    # Connecting lines from pipeline (top) down to spotlights (bottom)
    # =========================================================
    # From Mapping highlighted blocks down to Spotlight 1
    ax.plot([MAP_X + 2.2, MAP_X + 2.2, SP1_X + SP1_W / 2, SP1_X + SP1_W / 2],
            [R1_Y - 0.05, R2_Y + R2_H + 0.4, R2_Y + R2_H + 0.4, R2_Y + R2_H + 0.05],
            color=BORDERS["inno1"], linestyle="dashed", linewidth=1.4, zorder=0)
    # From Mapping (BA branch) down to Spotlight 2
    ax.plot([MAP_X + 4.0, MAP_X + 4.0, SP2_X + SP2_W / 2, SP2_X + SP2_W / 2],
            [R1_Y - 0.05, R2_Y + R2_H + 0.4, R2_Y + R2_H + 0.4, R2_Y + R2_H + 0.05],
            color=BORDERS["inno2"], linestyle="dashed", linewidth=1.4, zorder=0)

    plt.tight_layout()
    OUTPUT.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(OUTPUT, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
