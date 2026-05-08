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

Required input images (defaults to C:/Users/lenov/OneDrive/Desktop/pic4project):
    215/215f_gt.png        # GT RGB
    215/215_bl.png         # baseline rendering
    215/215_ours.png       # our rendering
    depth_est.png          # depth example
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

# Pastel color palette (matched to common SLAM papers)
COLORS = {
    "init":   "#FFE4E1",  # misty rose
    "track":  "#D4F1D4",  # light green
    "expand": "#FFF4D6",  # cream
    "map":    "#EAF3FF",  # very light blue
    "inno1":  "#FFD6E1",  # light pink (Innovation 1)
    "inno2":  "#D6F0E8",  # mint   (Innovation 2)
    "inno3":  "#FFE0B5",  # peach  (Innovation 3)
    "out":    "#E8E8F4",  # lavender grey
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


def load_img(path, max_side=300):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return np.array(img)


def add_module(ax, x, y, w, h, title, color_key, title_size=11):
    """Draw a rounded module box with a title bar (title floats on top edge)."""
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

    # Title pill
    title_w = max(2.2, 0.18 * len(title) + 0.9)
    title_x = x + (w - title_w) / 2
    title_y = y + h - 0.28
    title_box = FancyBboxPatch(
        (title_x, title_y), title_w, 0.55,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.5, linestyle=(0, (4, 2)),
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


def add_image(ax, img, x, y, zoom=0.40, label=None, label_pos="below", label_size=8.5):
    imagebox = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(imagebox, (x, y), frameon=True,
                        bboxprops=dict(edgecolor="#666", linewidth=0.8),
                        pad=0.05, zorder=4)
    ax.add_artist(ab)
    if label:
        if label_pos == "below":
            ax.text(x, y - 0.65, label, ha="center", va="top",
                    fontsize=label_size, style="italic", zorder=5)
        elif label_pos == "above":
            ax.text(x, y + 0.65, label, ha="center", va="bottom",
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
    """Draw a simple snowflake-like icon (used for 'frozen' indicator)."""
    for ang in [0, 60, 120]:
        rad = np.deg2rad(ang)
        dx, dy = size * np.cos(rad), size * np.sin(rad)
        ax.plot([x - dx, x + dx], [y - dy, y + dy], color=color, lw=1.8, zorder=5)
    ax.add_patch(plt.Circle((x, y), size * 0.18, color=color, zorder=6))


def flame(ax, x, y, size=0.25, color="#E26A3A"):
    """Draw a simple flame-like teardrop (used for 'activate' indicator)."""
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

    # Larger figure to give breathing room
    fig, ax = plt.subplots(figsize=(18, 10), dpi=160)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Title
    ax.text(9, 9.65,
            "Visibility-Driven Gaussian Map Management for Endoscopic SLAM",
            ha="center", va="center", fontsize=14, fontweight="bold")

    # =========================================================
    # ROW 1 (top, y=5.4-9.0): Init -> Tracking -> Expansion -> Output preview
    # =========================================================

    # ---- Initialization ----
    add_module(ax, 0.3, 5.4, 4.0, 3.6, "Initialization", "init")
    add_image(ax, imgs["rgb"],   1.4, 7.6, zoom=0.4, label="RGB")
    add_image(ax, imgs["depth_crop"], 3.2, 7.6, zoom=0.4, label="Depth")
    ax.annotate("$\\mathcal{G}_0$", xy=(2.3, 6.3), ha="center", va="center",
                fontsize=15, fontweight="bold", color="#444", zorder=5)
    add_arrow(ax, 2.55, 6.3, 3.05, 6.3, lw=1.8, color=COLORS["arrow"])
    gaussian_schematic(ax, 3.5, 6.3, n=5, scale=0.55)
    ax.text(3.5, 5.65, "3D Gaussians", ha="center", va="top",
            fontsize=9, style="italic", color="#444")

    # ---- Tracking ----
    add_module(ax, 4.6, 5.4, 5.0, 3.6, "Tracking Module", "track")
    camera_frustum(ax, 5.4, 7.7, size=0.32, color="#E58C2A")
    ax.text(5.4, 8.1, "$\\hat{\\mathcal{T}}_t$", ha="center", va="bottom",
            fontsize=12, fontweight="bold")
    add_arrow(ax, 5.85, 7.75, 6.55, 7.75, lw=1.8)
    camera_frustum(ax, 7.0, 7.7, size=0.32, color="#E58C2A")
    ax.text(7.0, 8.1, "$\\hat{\\mathcal{T}}_{t+1}$", ha="center", va="bottom",
            fontsize=12, fontweight="bold")

    # Frozen Gaussians indicator
    snowflake(ax, 8.4, 7.75, size=0.25, color="#5BB3D6")
    ax.text(8.4, 7.25, "Frozen\nGaussians", ha="center", va="center",
            fontsize=8.5, color="#444")

    ax.text(7.1, 6.6, r"$\mathcal{L}_{tr} = w_d\|D-\hat D\|_1 + w_c\|C-\hat C\|_1$",
            ha="center", va="center", fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#888"))
    ax.text(7.1, 5.85, "Adam Optimizer (camera params only)",
            ha="center", va="center", fontsize=9, style="italic", color="#444")

    # ---- Gaussian Expansion ----
    add_module(ax, 9.9, 5.4, 4.2, 3.6, "Gaussian Expansion", "expand")
    ax.add_patch(mpatches.Rectangle((10.55, 6.7), 1.4, 1.4, facecolor="#222",
                                    edgecolor="#666", linewidth=1.0, zorder=4))
    ax.text(11.25, 7.4, "$\\mathcal{G}_t$\nrendered", ha="center", va="center",
            fontsize=9.5, color="white", zorder=5)
    add_arrow(ax, 12.05, 7.4, 12.65, 7.4, lw=1.8)
    ax.add_patch(mpatches.Rectangle((12.65, 6.7), 1.4, 1.4, facecolor="#FFF4D6",
                                    edgecolor="#666", linewidth=1.0, zorder=4))
    gaussian_schematic(ax, 13.35, 7.4, n=4, scale=0.4)
    ax.text(11.25, 6.55, "Silhouette mask", ha="center", va="top",
            fontsize=8.5, style="italic")
    ax.text(13.35, 6.55, "Add new", ha="center", va="top",
            fontsize=8.5, style="italic")
    ax.text(12.0, 6.0,
            "Add Gaussians where rendered\nsilhouette < $\\tau$ or depth disagrees",
            ha="center", va="center", fontsize=9.5, color="#444")

    # ---- Top-right: Output preview ----
    add_module(ax, 14.4, 5.4, 3.4, 3.6, "Output (Comparison)", "out")
    add_image(ax, imgs["rgb"],      15.2, 7.85, zoom=0.35, label="GT")
    add_image(ax, imgs["baseline"], 16.95, 7.85, zoom=0.35, label="Baseline")
    add_image(ax, imgs["ours"],     16.05, 6.15, zoom=0.35, label="Ours ($\\eta$=0.90)")

    # =========================================================
    # ROW 1 -> ROW 2 connecting arrow
    # =========================================================
    add_arrow(ax, 7.1, 5.4, 7.1, 5.05, lw=2.2, color="#444")

    # =========================================================
    # ROW 2 (bottom, y=0.3-5.0): Mapping Module containing 3 innovations
    # =========================================================
    
    add_module(ax, 0.3, 0.3, 17.5, 4.7, "Mapping Module with Innovations", "map",
               title_size=12)

    # Modified CUDA rasterizer (the technical foundation)
    rast_box = FancyBboxPatch(
        (1.0, 3.55), 15.5, 0.95,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.5, facecolor="white", edgecolor=BORDERS["map"],
        zorder=2,
    )
    ax.add_patch(rast_box)
    # Wrench icon (replaced with cogs symbol from matplotlib's default fonts)
    ax.text(2.0, 4.05, "⚙", ha="center", va="center", fontsize=20, color="#5577AA")
    ax.text(8.7, 4.20, "Modified CUDA Rasterizer",
            ha="center", va="center", fontsize=12, fontweight="bold", color="#222")
    ax.text(8.7, 3.78,
            r"output: per-Gaussian visibility  $V_i = \sum_p \alpha_i^{(p)} \cdot T^{(p)}$  (free piggyback on alpha-blending)",
            ha="center", va="center", fontsize=10, style="italic", color="#444")
    
    # Three arrows down to innovations
    for ax_x in [2.4, 8.7, 14.7]:
        add_arrow(ax, ax_x, 3.55, ax_x, 3.25, lw=1.6, color=BORDERS["map"])

    # ---- Innovation 1: Visibility Pruning ----
    add_module(ax, 0.6, 0.55, 5.2, 2.65, "Innovation 1 — Visibility Pruning", "inno1")
    
    # Visibility buffer schematic
    buf_y = 2.45
    for i, alpha in enumerate([0.3, 0.5, 0.7, 0.9, 1.0, 0.85, 0.7, 0.5]):
        ax.add_patch(mpatches.Rectangle((1.1 + i*0.42, buf_y), 0.4, 0.32,
                                        facecolor=plt.cm.viridis(alpha),
                                        edgecolor="#444", linewidth=0.5, zorder=4))
    ax.text(3.2, 2.18, "Visibility history buffer ($W$=15 frames)",
            ha="center", va="top", fontsize=8.5, style="italic", color="#444")

    # Three-way classifier
    ax.text(3.2, 1.55, "Three-way classifier:", ha="center", va="center",
            fontsize=10, fontweight="bold")
    
    cls_y = 1.05
    ax.text(1.4, cls_y, "STATIC", ha="center", va="center", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#A8E6A8", edgecolor="#5BA05B"))
    ax.text(3.2, cls_y, "DEFORM", ha="center", va="center", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#FFD08A", edgecolor="#CC8030"))
    ax.text(5.0, cls_y, "FLOATER", ha="center", va="center", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#FFA8A8", edgecolor="#CC5050"))

    ax.text(3.2, 0.7, r"opacity degeneration:  $\sigma \leftarrow \sigma \cdot \eta$,  $\eta=0.90$",
            ha="center", va="center", fontsize=10, style="italic", color="#333",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#888"))

    # ---- Innovation 2: Periodic BA ----
    add_module(ax, 6.05, 0.55, 5.5, 2.65, "Innovation 2 — Periodic BA", "inno2")

    # Five keyframes
    for i, x in enumerate([6.5, 7.05, 7.6, 8.15, 8.7]):
        camera_frustum(ax, x, 2.4, size=0.18, color="#3B9C73")
    ax.text(7.6, 2.85, "5 keyframes (hybrid: recent + random older)",
            ha="center", va="bottom", fontsize=9, style="italic", color="#444")

    # Joint optimization indicator
    add_arrow(ax, 8.95, 2.4, 9.5, 2.4, lw=1.8, color=BORDERS["inno2"])
    flame(ax, 9.85, 2.4, size=0.25)
    ax.text(10.65, 2.4, "Joint cam +\nGaussian opt",
            ha="center", va="center", fontsize=9, color="#333")

    # Triggers and stats
    ax.text(8.8, 1.55, r"Trigger: every $M = 50$ frames",
            ha="center", va="center", fontsize=10, color="#333")
    ax.text(8.8, 1.12, r"BA iterations: $20$    Overhead: $\sim 1.5\%$",
            ha="center", va="center", fontsize=10, color="#333")
    ax.text(8.8, 0.7,
            r"Camera LR during BA: $\frac{1}{2}\times$ tracking LR (conservative)",
            ha="center", va="center", fontsize=9, style="italic", color="#444")

    # ---- Innovation 3: Deformation Modeling ----
    add_module(ax, 11.75, 0.55, 5.7, 2.65, "Innovation 3 — Deformation Modeling", "inno3")

    # Original -> deformed Gaussian schematic
    e_static = Ellipse((12.4, 2.3), 0.5, 0.32, angle=20, facecolor="#A8D8F4",
                      edgecolor="#446", linewidth=1.0, zorder=4)
    ax.add_patch(e_static)
    add_arrow(ax, 12.7, 2.3, 13.4, 2.05, lw=2.0, color="#E0682A")
    e_deform = Ellipse((13.7, 1.95), 0.5, 0.32, angle=20, facecolor="#FFD08A",
                      edgecolor="#666", linewidth=1.0, zorder=4)
    ax.add_patch(e_deform)
    ax.text(13.05, 2.75, r"$\Delta_{xyz}$ offset",
            ha="center", va="bottom", fontsize=10, fontweight="bold", color="#E0682A")

    ax.text(15.7, 2.45, "Applied only to\nDEFORMING-class\nGaussians",
            ha="center", va="center", fontsize=9.5, color="#333")

    ax.text(14.55, 1.4,
            r"$\mathcal{L}_{def} = \lambda_m\|\Delta\|_2 + \lambda_t\|\Delta_t-\Delta_{t-1}\|_2$",
            ha="center", va="center", fontsize=10, style="italic", color="#333",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                     edgecolor="#aaa", linewidth=0.7))
    
    ax.text(14.55, 0.78, "Magnitude + Temporal Smoothness Regularization",
            ha="center", va="center", fontsize=9, style="italic", color="#666")

    # =========================================================
    # Loop-back arrow: keyframe -> next frame tracking
    # =========================================================
    arrow_back = FancyArrowPatch(
        (17.7, 0.6), (17.95, 0.6),
        arrowstyle="-", mutation_scale=15, linewidth=1.5,
        color="#888", linestyle="dashed", zorder=2,
    )
    # use a manual U-shaped arrow with line segments
    line_pts = [(7.0, 5.0), (7.0, 5.05)]  # placeholder; actual loop drawn below

    # Draw a faint loop-back from bottom to tracking
    ax.plot([17.6, 17.95, 17.95, 0.15, 0.15, 4.6],
            [2.5, 2.5, 9.4, 9.4, 7.2, 7.2],
            color="#aaa", linestyle="dashed", linewidth=1.2, zorder=0)
    add_arrow(ax, 0.4, 7.2, 4.6, 7.2, lw=0, color="#aaa")  # invisible (just to show direction)
    ax.annotate("", xy=(4.6, 7.2), xytext=(4.4, 7.2),
                arrowprops=dict(arrowstyle="->", color="#888", lw=1.4))
    ax.text(0.5, 9.55, "next frame", ha="left", va="center",
            fontsize=9, style="italic", color="#666")

    plt.tight_layout()
    OUTPUT.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(OUTPUT, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
