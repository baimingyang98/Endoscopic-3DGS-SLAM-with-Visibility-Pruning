"""
Patch diff-gaussian-rasterization-w-depth to add per-Gaussian visibility output.

Usage:
    git clone https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth.git /content/rasterizer
    python patch_rasterizer.py /content/rasterizer
    pip install /content/rasterizer/

This adds a 4th return value (gaussian_visibility) to the forward rasterization pass.
The visibility score is the sum of (alpha * transmittance) across all pixels for each Gaussian.
"""
import os
import sys


def patch_file(fpath, replacements):
    """Apply a list of (old, new) string replacements to a file."""
    with open(fpath, "r") as f:
        src = f.read()
    for old, new in replacements:
        if old not in src:
            print(f"  WARNING: pattern not found in {os.path.basename(fpath)}:")
            print(f"    {repr(old[:80])}")
            continue
        src = src.replace(old, new)
    with open(fpath, "w") as f:
        f.write(src)
    print(f"  [OK] {fpath}")


def main(rast_dir):
    print(f"Patching rasterizer at: {rast_dir}\n")

    # 1. cuda_rasterizer/forward.h
    patch_file(os.path.join(rast_dir, "cuda_rasterizer/forward.h"), [
        (
            'float* out_depth);',
            'float* out_depth,\n\t\tfloat* out_gaussian_visibility);'
        ),
    ])

    # 2. cuda_rasterizer/forward.cu
    patch_file(os.path.join(rast_dir, "cuda_rasterizer/forward.cu"), [
        # 2a. Kernel signature
        (
            'float* __restrict__ out_depth)',
            'float* __restrict__ out_depth,\n\tfloat* __restrict__ out_gaussian_visibility)'
        ),
        # 2b. Add atomicAdd after color accumulation
        (
            'for (int ch = 0; ch < CHANNELS; ch++)\n\t\t\t\tC[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;',
            'for (int ch = 0; ch < CHANNELS; ch++)\n\t\t\t\tC[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;\n\n\t\t\t\t// Per-Gaussian visibility accumulation (Innovation 1)\n\t\t\t\tif (inside)\n\t\t\t\t\tatomicAdd(&out_gaussian_visibility[collected_id[j]], alpha * T);'
        ),
        # 2c. FORWARD::render wrapper signature
        (
            'const float* depth,\n\tfloat* out_depth)\n{\n\trenderCUDA<NUM_CHANNELS>',
            'const float* depth,\n\tfloat* out_depth,\n\tfloat* out_gaussian_visibility)\n{\n\trenderCUDA<NUM_CHANNELS>'
        ),
        # 2d. Kernel call arguments
        (
            'depth,\n\t\tout_depth);',
            'depth,\n\t\tout_depth,\n\t\tout_gaussian_visibility);'
        ),
    ])

    # 3. cuda_rasterizer/rasterizer.h
    patch_file(os.path.join(rast_dir, "cuda_rasterizer/rasterizer.h"), [
        (
            'int* radii = nullptr);',
            'int* radii = nullptr,\n\t\t\tfloat* out_gaussian_visibility = nullptr);'
        ),
    ])

    # 4. cuda_rasterizer/rasterizer_impl.cu
    patch_file(os.path.join(rast_dir, "cuda_rasterizer/rasterizer_impl.cu"), [
        # 4a. forward() signature
        (
            'float* out_depth,\n\tint* radii)',
            'float* out_depth,\n\tint* radii,\n\tfloat* out_gaussian_visibility)'
        ),
        # 4b. Pass to FORWARD::render
        (
            'geomState.depths,\n\t\tout_depth);',
            'geomState.depths,\n\t\tout_depth,\n\t\tout_gaussian_visibility);'
        ),
    ])

    # 5. rasterize_points.cu
    patch_file(os.path.join(rast_dir, "rasterize_points.cu"), [
        # 5a. Return type tuple
        (
            'std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>\nRasterizeGaussiansCUDA',
            'std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>\nRasterizeGaussiansCUDA'
        ),
        # 5b. Allocate visibility tensor
        (
            'torch::Tensor out_depth = torch::full({1, H, W}, 0.0, float_opts);',
            'torch::Tensor out_depth = torch::full({1, H, W}, 0.0, float_opts);\n  torch::Tensor out_gaussian_visibility = torch::zeros({P}, float_opts);'
        ),
        # 5c. Pass to Rasterizer::forward
        (
            'radii.contiguous().data<int>());',
            'radii.contiguous().data<int>(),\n\t\tout_gaussian_visibility.contiguous().data<float>());'
        ),
        # 5d. Return tuple
        (
            'return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer, out_depth);',
            'return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer, out_depth, out_gaussian_visibility);'
        ),
    ])

    # 6. diff_gaussian_rasterization/__init__.py
    patch_file(os.path.join(rast_dir, "diff_gaussian_rasterization/__init__.py"), [
        # 6a. Unpack 8 values
        (
            'num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, depth = _C.rasterize_gaussians(*args)',
            'num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, depth, gaussian_visibility = _C.rasterize_gaussians(*args)'
        ),
        # 6b. Return 4 values
        (
            'return color, radii, depth',
            'return color, radii, depth, gaussian_visibility'
        ),
        # 6c. Backward accepts 4 grad inputs
        (
            'def backward(ctx, grad_out_color, _, depth):',
            'def backward(ctx, grad_out_color, _, depth, __):',
        ),
    ])

    print("\n=== All patches applied successfully! ===")
    print(f"Now run: pip install {rast_dir}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python patch_rasterizer.py <path-to-rasterizer-repo>")
        sys.exit(1)
    main(sys.argv[1])
