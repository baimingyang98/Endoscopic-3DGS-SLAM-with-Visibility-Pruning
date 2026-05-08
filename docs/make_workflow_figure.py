"""
Generate a paper-style workflow figure for EndoGSLAM with Innovations.

Produces docs/workflow_figure.png — a high-resolution figure with:
- Pastel-colored module boxes (similar to EndoFlow-SLAM / EndoGSLAM style)
- Embedded real images (RGB / Depth / GT / Baseline / Ours)
- Schematic icons (camera frustums, Gaussian ellipses)
- Three innovation modules clearly highlighted
- Labels and equations inside modules

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

# Canvas size (in matplotlib coordinates)
CANVAS_W = 20.0
CANVAS_H = 13.0

COLORS = {
    "init":   "#FFE4E1",
    "track":  "#D4F1D4",
    "expand": "#FFF4D6",
    "map":    "#EAF3FF",
    "inno1":  "#FFD6E1",
    "inno2":  "#D6F0E8",
    "inno3":  "#FFE0B5",
    "out":    "#E8E8F4",
    "edge":   "#444444",
    "arrow":  "#555555",
}

BORDERS = {
    "init":   "#E08CA0",
    "track":  "#7DBE7D",
    "expand": "#E0B85A",
    "map":    "#6FA0D7",
    "inno1":  "#D86A92",
    "inno2":  "#5FB58E",
    "inno3":  "#E08C4D",
    "out":    "#9090B0",
}


def load_img(path, max_side=320):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return np.array(img)


def add_module(ax, x, y, w, h, title, color_key, title_size=11):
    """Rounded module box with floating title pill on top."""
    fill = COLORS[color_key]
    border = BORDERS[color_key]

    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        linewidth=1.8, linestyle=(0, (5, 3)),
        facecolor=fill, edgecolor=border,
        zorder=1,
    )
    ax.add_patch(box)

    # Tight pill: width ~ text length * char-width + small horizontal padding
    char_w = 0.115 * (title_size / 11.0)  # scales with font size
    title_w = char_w * len(title) + 0.6
    title_x = x + (w - title_w) / 2
    title_y = y + h - 0.30
    title_box = FancyBboxPatch(
        (title_x, title_y), title_w, 0.6,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.5, linestyle=(0, (4, 2)),
        facecolor="white", edgecolor=border,
        zorder=2,
    )
    ax.add_patch(title_box)
    ax.text(
        title_x + title_w / 2, title_y + 0.30, title,
        ha="center", va="center",
        fontsize=title_size, fontweight="bold", color="#222",
        zorder=3,
    )


def add_image(ax, img, x, y, zoom=0.30, label=None, label_pos="below", label_size=9):
    imagebox = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(imagebox, (x, y), frameon=True,
                        bboxprops=dict(edgecolor="#666", linewidth=0.8),
                        pad=0.05, zorder=4)
    ax.add_artist(ab)
    if label:
        if label_pos == "below":
            ax.text(x, y - 0.75, label, ha="center", va="top",
                    fontsize=label_size, style="italic", zorder=5)
        elif label_pos == "above":
            ax.text(x, y + 0.75, label, ha="center", va="bottom",
                    fontsize=label_size, style="italic", zorder=5)


def add_arrow(ax, x1, y1, x2, y2, color=None, style="->", lw=2.0, zorder=3):
    color = color or COLORS["arrow"]
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=18,
        linewidth=lw, color=color, zorder=zorder,
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


# ============================================================
# Main figure
# ============================================================
def main():
    paths = {
        "rgb":      PIC_DIR / "215" / "215f_gt.png",
        "baseline": PIC_DIR / "215" / "215_bl.png",
        "ours":     PIC_DIR / "215" / "215_ours.png",
        "depth":    PIC_DIR / "depth_est.png",
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
    ax.text(CANVAS_W / 2, CANVAS_H - 0.4,
            "Visibility-Driven Gaussian Map Management for Endoscopic SLAM",
            ha="center", va="center", fontsize=16, fontweight="bold")

    # =========================================================
    # ROW 1 (top): Init -> Tracking -> Expansion -> Output
    # y range: 8.0 - 12.2
    # =========================================================
    R1_Y = 8.0
    R1_H = 4.2

    # ---- Initialization ----
    add_module(ax, 0.4, R1_Y, 4.4, R1_H, "Initialization", "init")
    add_image(ax, imgs["rgb"],   1.55, R1_Y + 2.5, zoom=0.30, label="RGB")
    add_image(ax, imgs["depth_crop"], 3.55, R1_Y + 2.5, zoom=0.26, label="Depth")
    ax.annotate("$\\mathcal{G}_0$", xy=(2.4, R1_Y + 1.0), ha="center", va="center",
                fontsize=16, fontweight="bold", color="#444", zorder=5)
    add_arrow(ax, 2.7, R1_Y + 1.0, 3.2, R1_Y + 1.0, lw=1.8, color=COLORS["arrow"])
    gaussian_schematic(ax, 3.7, R1_Y + 1.0, n=5, scale=0.6)
    ax.text(3.7, R1_Y + 0.3, "3D Gaussians", ha="center", va="top",
            fontsize=10, style="italic", color="#444")

    # ---- Tracking ----
    add_module(ax, 5.1, R1_Y, 5.4, R1_H, "Tracking Module", "track")
    camera_frustum(ax, 6.1, R1_Y + 2.7, size=0.36, color="#E58C2A")
    ax.text(6.1, R1_Y + 3.2, "$\\hat{\\mathcal{T}}_t$", ha="center", va="bottom",
            fontsize=14, fontweight="bold")
    add_arrow(ax, 6.6, R1_Y + 2.75, 7.4, R1_Y + 2.75, lw=1.8)
    camera_frustum(ax, 7.95, R1_Y + 2.7, size=0.36, color="#E58C2A")
    ax.text(7.95, R1_Y + 3.2, "$\\hat{\\mathcal{T}}_{t+1}$", ha="center", va="bottom",
            fontsize=14, fontweight="bold")

    snowflake(ax, 9.4, R1_Y + 2.75, size=0.28, color="#5BB3D6")
    ax.text(9.4, R1_Y + 2.15, "Frozen\nGaussians", ha="center", va="center",
            fontsize=9.5, color="#444")

    ax.text(7.8, R1_Y + 1.4, r"$\mathcal{L}_{tr} = w_d\|D-\hat D\|_1 + w_c\|C-\hat C\|_1$",
            ha="center", va="center", fontsize=11.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#888"))
    ax.text(7.8, 8.4, "Adam Optimizer (camera params only)",
            ha="center", va="center", fontsize=10, style="italic", color="#444")

    # ---- Gaussian Expansion ----
    add_module(ax, 10.8, R1_Y, 4.6, R1_H, "Gaussian Expansion", "expand")
    ax.add_patch(mpatches.Rectangle((11.45, R1_Y + 1.8), 1.6, 1.6, facecolor="#222",
                                    edgecolor="#666", linewidth=1.0, zorder=4))
    ax.text(12.25, R1_Y + 2.6, "$\\mathcal{G}_t$\nrendered", ha="center", va="center",
            fontsize=10, color="white", zorder=5)
    add_arrow(ax, 13.2, R1_Y + 2.6, 13.85, R1_Y + 2.6, lw=1.8)
    ax.add_patch(mpatches.Rectangle((13.85, R1_Y + 1.8), 1.6, 1.6, facecolor="#FFF4D6",
                                    edgecolor="#666", linewidth=1.0, zorder=4))
    gaussian_schematic(ax, 14.65, R1_Y + 2.6, n=4, scale=0.45)
    ax.text(12.25, R1_Y + 1.55, "Silhouette mask", ha="center", va="top",
            fontsize=9, style="italic")
    ax.text(14.65, R1_Y + 1.55, "Add new", ha="center", va="top",
            fontsize=9, style="italic")
    ax.text(13.1, R1_Y + 0.85,
            "Add Gaussians where rendered\nsilhouette < $\\tau$ or depth disagrees",
            ha="center", va="center", fontsize=10, color="#444")

    # ---- Output ----
    add_module(ax, 15.7, R1_Y, 4.0, R1_H, "Output (Comparison)", "out")
    add_image(ax, imgs["rgb"],      16.65, R1_Y + 3.0, zoom=0.16, label="GT")
    add_image(ax, imgs["baseline"], 18.65, R1_Y + 3.0, zoom=0.16, label="Baseline")
    add_image(ax, imgs["ours"],     17.7, R1_Y + 1.1, zoom=0.16, label="Ours ($\\eta$=0.90)")

    # =========================================================
    # ROW 1 -> ROW 2 connector
    # =========================================================
    add_arrow(ax, 7.8, R1_Y, 7.8, 7.7, lw=2.4, color="#444")

    # =========================================================
    # ROW 2 (bottom): Mapping Module containing 3 innovations
    # y range: 0.4 - 7.4
    # =========================================================
    R2_Y = 0.4
    R2_H = 7.0

    add_module(ax, 0.3, R2_Y, CANVAS_W - 0.6, R2_H,
               "Mapping Module with Innovations", "map", title_size=13)

    # Modified CUDA rasterizer (foundation, full-width inside the box)
    rast_y = R2_Y + R2_H - 1.5
    rast_box = FancyBboxPatch(
        (1.0, rast_y), CANVAS_W - 2.0, 1.0,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.5, facecolor="white", edgecolor=BORDERS["map"],
        zorder=2,
    )
    ax.add_patch(rast_box)
    ax.text(2.0, rast_y + 0.5, "⚙", ha="center", va="center",
            fontsize=22, color="#5577AA")
    ax.text(CANVAS_W / 2, rast_y + 0.7, "Modified CUDA Rasterizer",
            ha="center", va="center", fontsize=13, fontweight="bold", color="#222")
    ax.text(CANVAS_W / 2, rast_y + 0.25,
            r"output: per-Gaussian visibility  $V_i = \sum_p \alpha_i^{(p)} \cdot T^{(p)}$  (free piggyback on alpha-blending)",
            ha="center", va="center", fontsize=11, style="italic", color="#444")

    # Three downward arrows from rasterizer to innovations
    inno_centers_x = [3.0, 9.85, 16.7]
    for cx in inno_centers_x:
        add_arrow(ax, cx, rast_y, cx, rast_y - 0.5, lw=1.8, color=BORDERS["map"])

    # Innovation boxes (more spacious now)
    INO_Y = R2_Y + 0.4
    INO_H = R2_H - 2.5  # 4.5 high - lots of room

    # ---- Innovation 1: Visibility Pruning ----
    INO1_X, INO1_W = 0.7, 5.6
    add_module(ax, INO1_X, INO_Y, INO1_W, INO_H,
               "Innovation 1 — Visibility Pruning", "inno1")

    # Visibility buffer
    buf_y = INO_Y + INO_H - 1.4
    for i, alpha in enumerate([0.3, 0.5, 0.7, 0.9, 1.0, 0.85, 0.7, 0.5]):
        ax.add_patch(mpatches.Rectangle((INO1_X + 0.4 + i*0.55, buf_y), 0.5, 0.42,
                                        facecolor=plt.cm.viridis(alpha),
                                        edgecolor="#444", linewidth=0.5, zorder=4))
    ax.text(INO1_X + INO1_W/2, buf_y - 0.25,
            "Visibility history buffer ($W$=15 frames)",
            ha="center", va="top", fontsize=10, style="italic", color="#444")

    # Three-way classifier
    cls_label_y = INO_Y + 1.7
    ax.text(INO1_X + INO1_W/2, cls_label_y, "Three-way classifier:",
            ha="center", va="center", fontsize=11.5, fontweight="bold")

    cls_y = INO_Y + 1.05
    ax.text(INO1_X + 0.9, cls_y, "STATIC", ha="center", va="center",
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#A8E6A8", edgecolor="#5BA05B"))
    ax.text(INO1_X + INO1_W/2, cls_y, "DEFORM", ha="center", va="center",
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFD08A", edgecolor="#CC8030"))
    ax.text(INO1_X + INO1_W - 0.95, cls_y, "FLOATER", ha="center", va="center",
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFA8A8", edgecolor="#CC5050"))

    ax.text(INO1_X + INO1_W/2, INO_Y + 0.45,
            r"opacity degeneration:  $\sigma \leftarrow \sigma \cdot \eta$,  $\eta=0.90$",
            ha="center", va="center", fontsize=11, style="italic", color="#333",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#888"))

    # ---- Innovation 2: Periodic BA ----
    INO2_X, INO2_W = 7.1, 5.6
    add_module(ax, INO2_X, INO_Y, INO2_W, INO_H,
               "Innovation 2 — Periodic BA", "inno2")

    # 5 keyframes
    kf_y = INO_Y + INO_H - 1.3
    for i, x_off in enumerate([0.6, 1.15, 1.7, 2.25, 2.8]):
        camera_frustum(ax, INO2_X + x_off, kf_y, size=0.22, color="#3B9C73")
    ax.text(INO2_X + 1.7, kf_y - 0.5,
            "5 keyframes (hybrid: recent + random older)",
            ha="center", va="center", fontsize=9.5, style="italic", color="#444")

    # Joint optimization
    add_arrow(ax, INO2_X + 3.1, kf_y, INO2_X + 3.7, kf_y, lw=1.8, color=BORDERS["inno2"])
    flame(ax, INO2_X + 4.05, kf_y, size=0.28)
    ax.text(INO2_X + 5.0, kf_y, "Joint cam +\nGaussian opt",
            ha="center", va="center", fontsize=10, color="#333")

    # Stats
    stats_y = INO_Y + 1.5
    ax.text(INO2_X + INO2_W/2, stats_y,
            r"Trigger: every $M = 50$ frames",
            ha="center", va="center", fontsize=11, color="#333")
    ax.text(INO2_X + INO2_W/2, stats_y - 0.6,
            r"BA iterations: $20$    Overhead: $\sim 1.5\%$",
            ha="center", va="center", fontsize=11, color="#333")
    ax.text(INO2_X + INO2_W/2, stats_y - 1.2,
            r"Camera LR during BA: $\frac{1}{2}\times$ tracking LR (conservative)",
            ha="center", va="center", fontsize=10, style="italic", color="#444")

    # ---- Innovation 3: Deformation ----
    INO3_X, INO3_W = 13.6, 6.1
    add_module(ax, INO3_X, INO_Y, INO3_W, INO_H,
               "Innovation 3 — Deformation Modeling", "inno3")

    # Original -> deformed Gaussian
    e_static = Ellipse((INO3_X + 0.9, INO_Y + INO_H - 1.4), 0.55, 0.36,
                      angle=20, facecolor="#A8D8F4",
                      edgecolor="#446", linewidth=1.0, zorder=4)
    ax.add_patch(e_static)
    add_arrow(ax, INO3_X + 1.25, INO_Y + INO_H - 1.4,
              INO3_X + 2.05, INO_Y + INO_H - 1.65, lw=2.2, color="#E0682A")
    e_deform = Ellipse((INO3_X + 2.4, INO_Y + INO_H - 1.7), 0.55, 0.36,
                      angle=20, facecolor="#FFD08A",
                      edgecolor="#666", linewidth=1.0, zorder=4)
    ax.add_patch(e_deform)
    ax.text(INO3_X + 1.65, INO_Y + INO_H - 0.7,
            r"$\Delta_{xyz}$ offset",
            ha="center", va="bottom", fontsize=11, fontweight="bold", color="#E0682A")

    ax.text(INO3_X + 4.6, INO_Y + INO_H - 1.4,
            "Applied only to\nDEFORMING-class\nGaussians",
            ha="center", va="center", fontsize=10.5, color="#333")

    # Loss equation
    ax.text(INO3_X + INO3_W/2, INO_Y + 1.4,
            r"$\mathcal{L}_{def} = \lambda_m\|\Delta\|_2 + \lambda_t\|\Delta_t-\Delta_{t-1}\|_2$",
            ha="center", va="center", fontsize=11.5, style="italic", color="#333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                     edgecolor="#aaa", linewidth=0.7))
    ax.text(INO3_X + INO3_W/2, INO_Y + 0.55,
            "Magnitude + Temporal Smoothness Regularization",
            ha="center", va="center", fontsize=10, style="italic", color="#666")

    plt.tight_layout()
    OUTPUT.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(OUTPUT, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
