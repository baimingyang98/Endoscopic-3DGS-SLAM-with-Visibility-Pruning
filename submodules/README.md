# Modified Gaussian Rasterizer

This directory should contain the modified `diff-gaussian-rasterization-w-depth` package
with per-Gaussian visibility output support.

## Setup

Clone the base rasterizer and apply our modifications:

```bash
cd submodules
git clone https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth.git
cd diff-gaussian-rasterization-w-depth
```

Then apply the patches described below, and install:

```bash
pip install .
```

## Modifications (Innovation 1)

We added a 4th return value `out_gaussian_visibility` to the forward rasterization pass.
This is a per-Gaussian buffer that accumulates the blending weight (alpha * transmittance)
across all pixels, providing a continuous visibility score for each Gaussian.

### Files modified (7 files, ~15 lines total):

#### 1. `cuda_rasterizer/forward.h`
Add parameter to `renderCUDA` declaration:
```cpp
// Add to renderCUDA signature:
float* out_gaussian_visibility  // [P] per-Gaussian visibility accumulator
```

#### 2. `cuda_rasterizer/forward.cu`
In the per-pixel alpha blending loop inside `renderCUDA`:
```cpp
// After computing: float alpha = min(0.99f, con_o.w * power);
// After computing: float test_T = T * (1 - alpha);
// Add this line to accumulate visibility:
atomicAdd(&out_gaussian_visibility[collected_id[j]], alpha * T);
```

Also add the parameter to the function signature to match the header.

#### 3. `cuda_rasterizer/rasterizer.h`
Add `float* out_gaussian_visibility` parameter to the `forward` method declaration.

#### 4. `cuda_rasterizer/rasterizer_impl.cu`
- Add parameter to `CudaRasterizer::Rasterizer::forward()` signature
- Allocate the buffer: `float* gaussian_visibility = ...`
- Initialize to zeros: `cudaMemset(gaussian_visibility, 0, P * sizeof(float))`
- Pass it through to `renderCUDA`
- Copy result to `out_gaussian_visibility`

#### 5. `rasterize_points.cu`
Add `out_gaussian_visibility` tensor allocation and passing through the C++/CUDA bridge.

#### 6. `rasterize_points.h`
Update the C++ function declaration to include the new output tensor.

#### 7. `diff_gaussian_rasterization/__init__.py`
In the `GaussianRasterizer.forward()` method:
- Allocate `out_gaussian_visibility = torch.zeros(P, device="cuda")`
- Pass it to the C++ call
- Return it as the 4th element: `return color, radii, depth, out_gaussian_visibility`

### Return value change

Before: `Renderer() -> (color, radii, depth)`
After:  `Renderer() -> (color, radii, depth, gaussian_visibility)`

All call sites must unpack 4 values instead of 3.

## Notes

- Only the forward pass is modified; backward pass is unchanged
- `atomicAdd` is used for thread safety (multiple pixels can see the same Gaussian)
- The visibility score V_i for each Gaussian i is in range [0, +inf) where:
  - V_i = 0 means the Gaussian is not visible in the current frame
  - Higher values mean more contribution to the rendered image
- No performance impact beyond the single `atomicAdd` per Gaussian per pixel
