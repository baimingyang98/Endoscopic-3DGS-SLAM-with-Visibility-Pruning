"""
Generate a paper-style workflow figure for EndoGSLAM with Innovations.

This version shows a COMPLETE LINEAR PIPELINE with the two innovations
(Visibility-Aware Pruning + Periodic BA) highlighted inline within the
actual SLAM loop. Innovation 3 (deformation) is omitted because it was
not used in the C3VD experiments.

Top row: linear pipeline (Input -> Init -> Tracking -> Mapping -> Output)
Bottom row: zoom-in spotlights of the two innovations

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

CANVAS_W = 20.0
CANVAS_H = 12.0

COLORS = {
    "input":  "#FFE8E0",  # light salmon
    "init":   "#FFE4E1",
    "track":  "#D4F1D4",
    "map":    "#EAF3FF",
    "out":    "#E8E8F4",
    "inno1":  "#FFF1B8",  # warm yellow (highlight)
    "inno2":  "#D6F0E8",  # mint
    "edge":   "#444444",
    "arrow":  "#555555",
}

BORDERS = {
    "input":  "#E08C70",
    "init":   "#E08CA0",
    "track":  "#7DBE7D",
    "map":    "#6FA0D7",
    "out":    "#9090B0",
    "inno1":  "#D4A700",  # gold border (highlight)
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
               highlight=False, title_above=False):
    """Rounded module box with floating title pill on top.
    
    Args:
        highlight: if True, use a thicker solid border (for innovation modules)
        title_above: if True, place title fully above the box (not overlapping)
    """
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
    title_y = y + h - 0.30 if not title_above else y + h + 0.05

    title_box = FancyBboxPatch(
        (title_x, title_y), title_w, 0.6,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.5,
        linestyle="solid" if highlight else (0, (4, 2)),
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


def gaussian_schematic(ax, cx, cy, n=4, scale=0.25, alpha=0.55, palette=None):
    rng = np.random.RandomState(7)
    if palette is None:
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
    """Gold star badge for marking highlighted items (Innovation 1, 2)."""
    # Five-pointed star
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
    # ROW 1 (TOP): Linear pipeline
    # 5 boxes: Input -> Init -> Tracking -> Mapping -> Output
    # y range: 7.0 - 10.6
    # =========================================================
    R1_Y = 7.0
    R1_H = 3.6

    # Box geometry
    boxes = [
        ("input",  0.4,  3.4, "RGB-D Input"),
        ("init",   4.0,  3.0, "Initialization"),
        ("track",  7.2,  3.4, "Tracking"),
        ("map",    10.8, 5.0, "Mapping (per-frame)"),
        ("out",    16.0, 3.6, "Output"),
    ]
    
    # ---- Input ----
    add_module(ax, 0.4, R1_Y, 3.4, R1_H, "RGB-D Input", "input")
    add_image(ax, imgs["rgb"], 1.4, R1_Y + 2.1, zoom=0.28, label="RGB")
    add_image(ax, imgs["depth_crop"], 2.9, R1_Y + 2.1, zoom=0.22, label="Depth")
    ax.text(2.1, R1_Y + 0.65, "Per-frame stream\n(C3VD / StereoMIS)",
            ha="center", va="center", fontsize=10, style="italic", color="#444")

    # ---- Initialization ----
    add_module(ax, 4.0, R1_Y, 3.0, R1_H, "Initialization", "init")
    ax.annotate("$\\mathcal{G}_0$", xy=(4.7, R1_Y + 1.7),
                ha="center", va="center",
                fontsize=18, fontweight="bold", color="#444", zorder=5)
    add_arrow(ax, 5.05, R1_Y + 1.7, 5.45, R1_Y + 1.7, lw=1.8)
    gaussian_schematic(ax, 5.95, R1_Y + 1.7, n=5, scale=0.55)
    ax.text(5.5, R1_Y + 2.7, "Frame 0 → seed\n3D Gaussians",
            ha="center", va="center", fontsize=10, style="italic", color="#444")
    ax.text(5.5, R1_Y + 0.55, "(point cloud back-projection)",
            ha="center", va="center", fontsize=9, color="#666")

    # ---- Tracking ----
    add_module(ax, 7.2, R1_Y, 3.4, R1_H, "Tracking", "track")
    camera_frustum(ax, 7.7, R1_Y + 2.4, size=0.30, color="#E58C2A")
    ax.text(7.7, R1_Y + 2.85, "$\\hat{\\mathcal{T}}_t$",
            ha="center", va="bottom", fontsize=12, fontweight="bold")
    add_arrow(ax, 8.05, R1_Y + 2.45, 8.55, R1_Y + 2.45, lw=1.6)
    camera_frustum(ax, 8.95, R1_Y + 2.4, size=0.30, color="#E58C2A")
    ax.text(8.95, R1_Y + 2.85, "$\\hat{\\mathcal{T}}_{t+1}$",
            ha="center", va="bottom", fontsize=12, fontweight="bold")
    snowflake(ax, 9.85, R1_Y + 2.45, size=0.20, color="#5BB3D6")
    ax.text(9.85, R1_Y + 1.9, "frozen", ha="center", va="center",
            fontsize=8.5, color="#444")
    
    ax.text(8.85, R1_Y + 1.3,
            r"$\mathcal{L}_{tr} = w_d\|D-\hat D\|_1 + w_c\|C-\hat C\|_1$",
            ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#888"))
    ax.text(8.85, R1_Y + 0.55, "Camera params only",
            ha="center", va="center", fontsize=9, style="italic", color="#444")

    # ---- Mapping (per-frame) - this is where the innovations live ----
    MAP_X, MAP_W = 10.8, 5.0
    add_module(ax, MAP_X, R1_Y, MAP_W, R1_H, "Mapping (per-frame)", "map")
    
    # Inside mapping: 3 sub-steps with innovation 1 highlighted
    sub_y = R1_Y + 2.3
    
    # Step 1: Add new Gaussians (small icon)
    ax.text(MAP_X + 0.7, sub_y + 0.4, "①", ha="center", va="center",
            fontsize=14, fontweight="bold", color="#666")
    ax.text(MAP_X + 0.7, sub_y - 0.05, "Add\nGaussians",
            ha="center", va="center", fontsize=8.5, color="#444")
    add_arrow(ax, MAP_X + 1.1, sub_y, MAP_X + 1.4, sub_y, lw=1.4)
    
    # Step 2: Render with modified rasterizer (HIGHLIGHTED - Innovation 1A)
    rast_box = FancyBboxPatch(
        (MAP_X + 1.4, sub_y - 0.45), 1.6, 0.95,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=2.2, linestyle="solid",
        facecolor=COLORS["inno1"], edgecolor=BORDERS["inno1"],
        zorder=2,
    )
    ax.add_patch(rast_box)
    ax.text(MAP_X + 2.2, sub_y + 0.18, "Modified",
            ha="center", va="center", fontsize=9, fontweight="bold", color="#222")
    ax.text(MAP_X + 2.2, sub_y - 0.18, "CUDA Rasterizer",
            ha="center", va="center", fontsize=9, fontweight="bold", color="#222")
    star_badge(ax, MAP_X + 1.5, sub_y + 0.55, "1")
    
    add_arrow(ax, MAP_X + 3.0, sub_y, MAP_X + 3.3, sub_y, lw=1.4)
    
    # Step 3: Visibility-aware pruning (HIGHLIGHTED - Innovation 1B)
    prune_box = FancyBboxPatch(
        (MAP_X + 3.3, sub_y - 0.45), 1.5, 0.95,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=2.2, linestyle="solid",
        facecolor=COLORS["inno1"], edgecolor=BORDERS["inno1"],
        zorder=2,
    )
    ax.add_patch(prune_box)
    ax.text(MAP_X + 4.05, sub_y + 0.18, "Visibility-",
            ha="center", va="center", fontsize=9, fontweight="bold", color="#222")
    ax.text(MAP_X + 4.05, sub_y - 0.18, "Aware Pruning",
            ha="center", va="center", fontsize=9, fontweight="bold", color="#222")
    star_badge(ax, MAP_X + 3.4, sub_y + 0.55, "1")

    # Output: V_i flowing down
    ax.annotate("", xy=(MAP_X + 2.2, sub_y - 0.55), xytext=(MAP_X + 2.2, sub_y - 1.0),
                arrowprops=dict(arrowstyle="-", color=BORDERS["inno1"], lw=1.5,
                              linestyle="dashed"))
    ax.text(MAP_X + 2.2, R1_Y + 0.95, r"per-Gaussian visibility $V_i$",
            ha="center", va="center", fontsize=8.5, style="italic", color="#8B6F00",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#FFF9E0", edgecolor="#D4A700"))
    
    ax.text(MAP_X + MAP_W/2, R1_Y + 0.45,
            r"backprop $\to$ update Gaussians",
            ha="center", va="center", fontsize=9, style="italic", color="#444")

    # ---- Output ----
    add_module(ax, 16.0, R1_Y, 3.6, R1_H, "Output", "out")
    add_image(ax, imgs["ours"], 17.0, R1_Y + 2.3, zoom=0.18, label="Rendered")
    camera_frustum(ax, 18.7, R1_Y + 2.5, size=0.25, color="#888")
    ax.text(18.7, R1_Y + 2.0, "Trajectory", ha="center", va="center",
            fontsize=9.5, color="#444")
    ax.text(17.8, R1_Y + 0.55, "+ Gaussian map (.npz)\n+ Per-frame metrics",
            ha="center", va="center", fontsize=9, style="italic", color="#444")

    # =========================================================
    # Pipeline arrows (between top-row boxes)
    # =========================================================
    pipeline_y = R1_Y + R1_H / 2
    add_arrow(ax, 3.8, pipeline_y, 4.0, pipeline_y, lw=2.4, color="#333")
    add_arrow(ax, 7.0, pipeline_y, 7.2, pipeline_y, lw=2.4, color="#333")
    add_arrow(ax, 10.6, pipeline_y, 10.8, pipeline_y, lw=2.4, color="#333")
    add_arrow(ax, 15.8, pipeline_y, 16.0, pipeline_y, lw=2.4, color="#333")

    # =========================================================
    # Periodic BA branch (Innovation 2) - dashed branch from mapping
    # =========================================================
    # From bottom of Mapping box, branch out to BA, then back to keyframes
    ba_x, ba_y = MAP_X + 2.2, R1_Y - 0.3
    add_arrow(ax, MAP_X + 2.5, R1_Y, MAP_X + 2.5, 6.4,
              lw=2.0, color=BORDERS["inno2"], dashed=True)
    
    # Loop-back arrow (next frame): from output back to tracking input area
    # A long dashed arc at the bottom edge
    ax.plot([19.7, 19.7, 0.2, 0.2, 7.2],
            [R1_Y + 0.2, 6.3, 6.3, pipeline_y, pipeline_y],
            color="#888", linestyle="dashed", linewidth=1.4, zorder=0)
    ax.annotate("", xy=(7.2, pipeline_y), xytext=(7.0, pipeline_y),
                arrowprops=dict(arrowstyle="->", color="#666", lw=1.4))
    ax.text(10.0, 6.5, "next frame", ha="center", va="center",
            fontsize=10, style="italic", color="#666")

    # =========================================================
    # ROW 2 (BOTTOM): Two innovation spotlights
    # y range: 0.4 - 5.8
    # =========================================================
    R2_Y = 0.4
    R2_H = 5.4
    
    # Section header
    ax.text(CANVAS_W / 2, R2_Y + R2_H + 0.3,
            "Innovation Spotlights",
            ha="center", va="center", fontsize=13, fontweight="bold",
            style="italic", color="#444")

    # ---- SPOTLIGHT 1: Visibility-Aware Pruning (Innovation 1) ----
    SP1_X, SP1_W = 0.4, 9.6
    add_module(ax, SP1_X, R2_Y, SP1_W, R2_H,
               "Innovation 1 — Visibility-Aware Pruning",
               "inno1", title_size=12, highlight=True)
    
    # Star badge on title corner
    star_badge(ax, SP1_X + 0.4, R2_Y + R2_H - 0.05, "1")
    
    # ---- (a) per-Gaussian visibility extraction ----
    sec_a_y = R2_Y + R2_H - 1.4
    ax.text(SP1_X + 0.5, sec_a_y, "(a)", ha="left", va="center",
            fontsize=11, fontweight="bold", color="#8B6F00")
    ax.text(SP1_X + 0.95, sec_a_y, "CUDA-level visibility extraction (free)",
            ha="left", va="center", fontsize=10.5, fontweight="bold", color="#222")
    
    ax.text(SP1_X + 4.7, sec_a_y - 0.5,
            r"$V_i = \sum_p \alpha_i^{(p)} \cdot T^{(p)}$  (one extra atomicAdd per pixel)",
            ha="center", va="center", fontsize=10.5, style="italic", color="#444",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#aaa"))

    # ---- (b) visibility history buffer ----
    sec_b_y = R2_Y + R2_H - 2.6
    ax.text(SP1_X + 0.5, sec_b_y, "(b)", ha="left", va="center",
            fontsize=11, fontweight="bold", color="#8B6F00")
    ax.text(SP1_X + 0.95, sec_b_y, "Visibility history buffer (W=15 frames)",
            ha="left", va="center", fontsize=10.5, fontweight="bold", color="#222")
    
    buf_y = sec_b_y - 0.7
    for i, alpha in enumerate([0.3, 0.5, 0.7, 0.9, 1.0, 0.85, 0.7, 0.55,
                                0.4, 0.3, 0.45, 0.6, 0.7, 0.5, 0.3]):
        ax.add_patch(mpatches.Rectangle((SP1_X + 1.2 + i*0.50, buf_y), 0.45, 0.40,
                                        facecolor=plt.cm.viridis(alpha),
                                        edgecolor="#444", linewidth=0.5, zorder=4))
    ax.text(SP1_X + 0.5, buf_y + 0.2, "$V_i^{(t)}$:",
            ha="left", va="center", fontsize=10, color="#444")
    ax.text(SP1_X + 8.85, buf_y + 0.2, "→ mean $\\bar V_i$, var $\\sigma_V^2$",
            ha="left", va="center", fontsize=10, color="#444")

    # ---- (c) Dual-mask pruning rule ----
    sec_c_y = R2_Y + R2_H - 3.6
    ax.text(SP1_X + 0.5, sec_c_y, "(c)", ha="left", va="center",
            fontsize=11, fontweight="bold", color="#8B6F00")
    ax.text(SP1_X + 0.95, sec_c_y, "Dual-mask pruning + opacity degeneration",
            ha="left", va="center", fontsize=10.5, fontweight="bold", color="#222")

    # Two sub-rules side by side, positioned higher to leave room for the eq box
    rule_y = sec_c_y - 0.55

    rule1 = FancyBboxPatch(
        (SP1_X + 0.6, rule_y - 0.35), 4.0, 0.65,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        linewidth=1.2, facecolor="white", edgecolor="#aaa", zorder=3,
    )
    ax.add_patch(rule1)
    ax.text(SP1_X + 2.6, rule_y + 0.10,
            r"Mask $V$:  $\bar V_i < \tau_v$  ∧  $age_i > 5$",
            ha="center", va="center", fontsize=10, color="#222")
    ax.text(SP1_X + 2.6, rule_y - 0.20, "(low visibility)",
            ha="center", va="center", fontsize=8.5, style="italic", color="#666")

    rule2 = FancyBboxPatch(
        (SP1_X + 5.0, rule_y - 0.35), 4.0, 0.65,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        linewidth=1.2, facecolor="white", edgecolor="#aaa", zorder=3,
    )
    ax.add_patch(rule2)
    ax.text(SP1_X + 7.0, rule_y + 0.10,
            r"Mask $D$:  $z_i^{cam} < d_{obs} - \gamma$",
            ha="center", va="center", fontsize=10, color="#222")
    ax.text(SP1_X + 7.0, rule_y - 0.20, "(distance / floater)",
            ha="center", va="center", fontsize=8.5, style="italic", color="#666")

    # Combined effect (highlighted yellow box, separated from rules above)
    eff_y = R2_Y + 0.45
    ax.text(SP1_X + SP1_W/2, eff_y,
            r"$\sigma_i \leftarrow \sigma_i \cdot \eta$  if  ($V \cup D$) — $\eta=0.90$, gradual decay",
            ha="center", va="center", fontsize=11, color="#222",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF9E0",
                     edgecolor=BORDERS["inno1"], linewidth=1.5))

    # ---- SPOTLIGHT 2: Periodic BA (Innovation 2) ----
    SP2_X, SP2_W = 10.4, 9.2
    add_module(ax, SP2_X, R2_Y, SP2_W, R2_H,
               "Innovation 2 — Periodic Bundle Adjustment",
               "inno2", title_size=12, highlight=True)
    
    star_badge(ax, SP2_X + 0.4, R2_Y + R2_H - 0.05, "2", color="#3B9C73")

    # ---- (a) Trigger condition ----
    sec2a_y = R2_Y + R2_H - 1.4
    ax.text(SP2_X + 0.5, sec2a_y, "(a)", ha="left", va="center",
            fontsize=11, fontweight="bold", color="#3B7A60")
    ax.text(SP2_X + 0.95, sec2a_y, "Trigger: every $M = 50$ frames",
            ha="left", va="center", fontsize=10.5, fontweight="bold", color="#222")
    
    ax.text(SP2_X + SP2_W/2, sec2a_y - 0.55,
            "Reduces accumulated pose drift between widely-separated keyframes",
            ha="center", va="center", fontsize=10, style="italic", color="#444")

    # ---- (b) Hybrid keyframe selection ----
    sec2b_y = R2_Y + R2_H - 2.7
    ax.text(SP2_X + 0.5, sec2b_y, "(b)", ha="left", va="center",
            fontsize=11, fontweight="bold", color="#3B7A60")
    ax.text(SP2_X + 0.95, sec2b_y, "Hybrid keyframe selection (3 recent + 2 random older)",
            ha="left", va="center", fontsize=10.5, fontweight="bold", color="#222")
    
    # Visualize keyframes on a timeline
    tl_y = sec2b_y - 0.7
    ax.plot([SP2_X + 0.7, SP2_X + 8.5], [tl_y, tl_y], color="#888", lw=1.0, zorder=2)
    # Old random keyframes
    for x_off, label in [(1.0, "kf₃"), (2.5, "kf₇")]:
        camera_frustum(ax, SP2_X + x_off, tl_y, size=0.18, color="#3B9C73")
        ax.text(SP2_X + x_off, tl_y - 0.4, label, ha="center", va="top",
                fontsize=8.5, color="#444")
    # Recent keyframes
    for x_off, label in [(5.5, "kf₂₈"), (6.7, "kf₃₀"), (7.9, "kf₃₂")]:
        camera_frustum(ax, SP2_X + x_off, tl_y, size=0.18, color="#3B9C73")
        ax.text(SP2_X + x_off, tl_y - 0.4, label, ha="center", va="top",
                fontsize=8.5, color="#444")
    ax.text(SP2_X + 1.7, tl_y + 0.4, "older", ha="center", va="bottom",
            fontsize=9, style="italic", color="#666")
    ax.text(SP2_X + 6.7, tl_y + 0.4, "recent", ha="center", va="bottom",
            fontsize=9, style="italic", color="#666")

    # ---- (c) Joint optimization ----
    sec2c_y = R2_Y + R2_H - 3.6
    ax.text(SP2_X + 0.5, sec2c_y, "(c)", ha="left", va="center",
            fontsize=11, fontweight="bold", color="#3B7A60")
    ax.text(SP2_X + 0.95, sec2c_y, "Joint optimization of camera poses + Gaussians",
            ha="left", va="center", fontsize=10.5, fontweight="bold", color="#222")

    # Side-by-side: cam params (activate) AND gaussians (activate)
    opt_y = sec2c_y - 0.55
    flame(ax, SP2_X + 1.6, opt_y, size=0.22)
    ax.text(SP2_X + 2.05, opt_y, "camera ($\\hat{\\mathcal{T}}$)",
            ha="left", va="center", fontsize=10, color="#222")
    ax.text(SP2_X + 3.7, opt_y, "+", ha="center", va="center",
            fontsize=14, fontweight="bold", color="#444")
    flame(ax, SP2_X + 4.2, opt_y, size=0.22)
    ax.text(SP2_X + 4.65, opt_y, "Gaussians",
            ha="left", va="center", fontsize=10, color="#222")
    ax.text(SP2_X + 6.5, opt_y, "20 iters,  ~1.5% overhead",
            ha="left", va="center", fontsize=9.5, style="italic", color="#666")

    # Conservative LR note at bottom
    ax.text(SP2_X + SP2_W/2, R2_Y + 0.45,
            r"Conservative camera LR during BA: $\frac{1}{2}\times$ tracking LR (avoid destabilizing converged poses)",
            ha="center", va="center", fontsize=10.5, color="#222",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F8F0",
                     edgecolor=BORDERS["inno2"], linewidth=1.5))

    # =========================================================
    # Connection lines from top-row Mapping to bottom-row spotlights
    # =========================================================
    # From mapping highlighted blocks down to spotlight 1
    ax.plot([MAP_X + 2.2, MAP_X + 2.2, SP1_X + SP1_W/2],
            [R1_Y - 0.5, R2_Y + R2_H + 0.3, R2_Y + R2_H + 0.3],
            color=BORDERS["inno1"], linestyle="dashed", linewidth=1.5, zorder=0)
    
    # From mapping (BA branch) down to spotlight 2
    ax.plot([MAP_X + 2.5, MAP_X + 2.5, SP2_X + SP2_W/2, SP2_X + SP2_W/2],
            [R1_Y - 0.5, 6.5, 6.5, R2_Y + R2_H + 0.3],
            color=BORDERS["inno2"], linestyle="dashed", linewidth=1.5, zorder=0)

    plt.tight_layout()
    OUTPUT.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(OUTPUT, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
