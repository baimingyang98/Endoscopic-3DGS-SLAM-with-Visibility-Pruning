"""
Sanity check for the patched diff_gaussian_rasterization.

Verifies:
1. The package imports correctly.
2. GaussianRasterizer returns 4 outputs (color, radii, depth, gauss_vis).
3. gauss_vis has the expected shape [P] and values are non-negative.
4. A trivial scene renders without crashing.

Usage:
    python scripts/check_rasterizer.py

Exit code 0 = all checks pass; non-zero = something is wrong.
"""
import math
import sys
import traceback

import numpy as np
import torch


def fail(msg):
    print(f"  [FAIL] {msg}")
    sys.exit(1)


def ok(msg):
    print(f"  [PASS] {msg}")


def main():
    print("=== Rasterizer sanity check ===\n")

    # 1. Import
    try:
        from diff_gaussian_rasterization import (
            GaussianRasterizationSettings, GaussianRasterizer,
        )
        ok("Imported diff_gaussian_rasterization")
    except ImportError as e:
        fail(f"Could not import: {e}")
    except Exception as e:
        fail(f"Unexpected error on import: {e}")

    if not torch.cuda.is_available():
        fail("CUDA not available — cannot test rasterizer")
    device = torch.device("cuda")
    ok(f"CUDA available: {torch.cuda.get_device_name(0)}")

    # 2. Build a tiny scene: 5 Gaussians in front of a centered camera
    P = 5
    H, W = 64, 64
    fov = math.radians(60.0)
    fx = fy = W / (2.0 * math.tan(fov / 2.0))
    cx, cy = W / 2.0, H / 2.0

    # Camera at origin, looking down +Z
    w2c = torch.eye(4, device=device, dtype=torch.float32)
    cam_center = torch.zeros(3, device=device, dtype=torch.float32)

    # OpenGL projection (matches setup_camera in recon_helpers.py)
    near, far = 0.01, 100.0
    proj = torch.tensor([
        [2 * fx / W, 0.0, -(W - 2 * cx) / W, 0.0],
        [0.0, 2 * fy / H, -(H - 2 * cy) / H, 0.0],
        [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
        [0.0, 0.0, 1.0, 0.0],
    ], device=device, dtype=torch.float32)
    view = w2c.unsqueeze(0).transpose(1, 2)
    full_proj = view.bmm(proj.unsqueeze(0).transpose(1, 2))

    raster_settings = GaussianRasterizationSettings(
        image_height=H, image_width=W,
        tanfovx=W / (2 * fx), tanfovy=H / (2 * fy),
        bg=torch.zeros(3, device=device, dtype=torch.float32),
        scale_modifier=1.0,
        viewmatrix=view,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False,
    )

    means3D = torch.tensor([
        [0.0, 0.0, 1.0],
        [0.2, 0.0, 1.0],
        [-0.2, 0.0, 1.0],
        [0.0, 0.2, 1.0],
        [0.0, -0.2, 1.0],
    ], device=device, dtype=torch.float32, requires_grad=True)
    means2D = torch.zeros_like(means3D, requires_grad=True)
    colors = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
    ], device=device, dtype=torch.float32)
    opacities = torch.full((P, 1), 0.9, device=device, dtype=torch.float32)
    scales = torch.full((P, 3), 0.05, device=device, dtype=torch.float32)
    rotations = torch.zeros(P, 4, device=device, dtype=torch.float32)
    rotations[:, 0] = 1.0  # identity quaternion (w, x, y, z)

    rendervar = dict(
        means3D=means3D, means2D=means2D,
        opacities=opacities, scales=scales, rotations=rotations,
        colors_precomp=colors,
    )

    # 3. Run the rasterizer
    try:
        out = GaussianRasterizer(raster_settings=raster_settings)(**rendervar)
    except Exception:
        print("  [FAIL] Rasterizer raised an exception:")
        traceback.print_exc()
        sys.exit(1)

    if not isinstance(out, tuple):
        fail(f"Rasterizer returned {type(out)}, expected tuple")
    if len(out) == 3:
        fail("Rasterizer returned 3 outputs — this is the STOCK rasterizer.\n"
             "      The patched fork should return 4: (color, radii, depth, gauss_vis).\n"
             "      Did the patch_rasterizer.py step succeed and rebuild?")
    if len(out) != 4:
        fail(f"Rasterizer returned {len(out)} outputs, expected 4")
    ok("Rasterizer returned 4 outputs (color, radii, depth, gauss_vis)")

    color, radii, depth, gauss_vis = out

    # 4. Shape and value checks
    if color.shape != (3, H, W):
        fail(f"color shape {tuple(color.shape)}, expected (3, {H}, {W})")
    ok(f"color shape OK: {tuple(color.shape)}")

    if depth.dim() == 2:
        depth_shape_ok = depth.shape == (H, W)
    elif depth.dim() == 3:
        depth_shape_ok = depth.shape in [(1, H, W), (3, H, W)]
    else:
        depth_shape_ok = False
    if not depth_shape_ok:
        fail(f"depth shape {tuple(depth.shape)} unexpected")
    ok(f"depth shape OK: {tuple(depth.shape)}")

    if gauss_vis.shape != (P,):
        fail(f"gauss_vis shape {tuple(gauss_vis.shape)}, expected ({P},)")
    ok(f"gauss_vis shape OK: ({P},)")

    if (gauss_vis < 0).any():
        fail(f"gauss_vis has negative values: min={gauss_vis.min().item()}")
    ok(f"gauss_vis non-negative (range [{gauss_vis.min().item():.4f}, "
       f"{gauss_vis.max().item():.4f}])")

    # 5. Check that the 5 Gaussians actually contributed something
    n_visible = (gauss_vis > 1e-6).sum().item()
    if n_visible == 0:
        fail("No Gaussian was visible — rendering pipeline likely broken")
    ok(f"{n_visible}/{P} Gaussians have non-zero visibility")

    # 6. Check that the rendered image has color (i.e., the alpha-blending happened)
    color_sum = color.sum().item()
    if color_sum < 1.0:
        fail(f"Rendered color is essentially black (sum={color_sum:.4f})")
    ok(f"Rendered color non-trivial (sum={color_sum:.2f})")

    print("\n=== All checks passed. Rasterizer is properly patched. ===")
    sys.exit(0)


if __name__ == "__main__":
    main()
