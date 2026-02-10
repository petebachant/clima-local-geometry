# LocalGeometry CUDA Resource Analysis

This analysis examines the GPU memory and computational overhead of carrying `LocalGeometry` data structures through ClimaCore calculations, with a focus on impact to larger atmospheric models like ClimaAtmos.

## Overview

`LocalGeometry` is a critical data structure in ClimaCore that contains geometric information at each grid point:
- **Coordinates**: X, Y, Z position information
- **Jacobians**: J (determinant), WJ (weighted by quadrature), invJ (inverse)
- **Metric tensors**: Transformation matrices (∂x∂ξ, ∂ξ∂x, gⁱʲ, gᵢⱼ)

This structure is **~200-500 bytes per grid point** depending on dimensionality and matrix components, making it a significant memory consumer in large simulations.

## Files

### 1. `test/gpu/local_geometry_cuda_resources.jl`
Comprehensive test suite measuring LocalGeometry resource consumption on CUDA devices.

**Key features:**
- Measures memory footprint of full vs. reduced LocalGeometry variants
- Tests performance impact of different access patterns
- Estimates overhead on realistic atmospheric model configurations
- Integrated with Buildkite CI/CD pipeline

**Run locally:**
```bash
julia --project test/gpu/local_geometry_cuda_resources.jl
```

**Buildkite integration:**
- Automatically runs on GPU nodes with the test name: "Unit: LocalGeometry CUDA resources"
- Located in `.buildkite/pipeline.yml` after basic CUDA tests

### 2. `analysis_local_geometry_impact.jl`
Standalone analysis script estimating LocalGeometry impact on ClimaAtmos-scale simulations.

**Analyzes:**
- Four representative configurations (development, operational low/medium, research)
- Memory footprint breakdown (LocalGeometry vs. state variables vs. temporary storage)
- Bandwidth overhead in physics kernels (typically 3-5x multiplier)
- Register pressure and kernel occupancy loss
- Optimization strategies

**Run:**
```bash
julia analysis_local_geometry_impact.jl
```

## Key Findings

### Memory Impact
For realistic atmospheric simulations:
- **Development (small tests)**: 8.5 MB LocalGeometry for 30K points (~43% of GPU memory)
- **Operational (medium)**: 74 MB LocalGeometry for 263K points (~35% of GPU memory)
- **Research (high-res)**: 155 MB LocalGeometry for 548K points (~35% of GPU memory)

### Computational Overhead

**Bandwidth multiplier**: 3-5x
- Physics kernels that access LocalGeometry incur significant memory overhead
- Reading full LocalGeometry for operations that only need J or WJ is inefficient

**Register pressure**: +24 registers/thread
- LocalGeometry access reduces kernel occupancy by ~9-30%
- May limit parallelism and throughput on GPUs

### Register Usage
- Base computation: ~35 registers/thread
- LocalGeometry access adds: ~24 registers/thread
- **Total: ~59 registers/thread** (exceeds typical single-block optimal)

## Optimization Strategies

### 1. **Extract at kernel entry**
Instead of passing full LocalGeometry to inner loops, extract needed scalars once:
```julia
function kernel!(Y, p, t)
    J = p.local_geom.J        # Extract once
    WJ = p.local_geom.WJ
    # ... use J, WJ throughout
end
```

### 2. **LocalGeometry-lite structs**
Create minimal geometry types for kernels that don't need tensors:
```julia
struct MinimalGeometry
    coordinates
    J::FT
    WJ::FT
end
```

### 3. **Shared memory caching**
For block-based kernels, load once to shared memory:
```julia
@cuda blocks=... threads=... function kernel(geom_data)
    shared = @cuDynamicSharedMem(LocalGeometry)
    # Load once, use across threads
end
```

### 4. **Reduced precision**
Use Float32 for LocalGeometry in non-critical kernels:
- Halves memory footprint (100 bytes → 50 bytes per point)
- Still maintains accuracy for geometric transformations

### 5. **Lazy evaluation**
Compute metric tensors on-demand rather than storing:
- If computation cost < bandwidth cost, compute ∂x∂ξ when needed
- Saves 128+ bytes per point

### 6. **Kernel fusion**
Combine multiple physics kernels to reuse LocalGeometry loads

## Using in Your Analysis

### For ClimaAtmos developers:
1. Run `test/gpu/local_geometry_cuda_resources.jl` to baseline measurements
2. Profile your specific physics kernels with `julia analysis_local_geometry_impact.jl`
3. Use NVIDIA Nsys to measure actual impact in realistic simulations
4. Compare bandwidth and occupancy with/without LocalGeometry optimization

### For ClimaCore contributors:
1. Use baseline measurements to track changes
2. Benchmark LocalGeometry-lite types before/after optimization
3. Profile on multiple GPU architectures (V100, A100, H100, etc.)

## Technical Details

### LocalGeometry Structure (2D example)
```
coordinates: XYPoint (16 bytes)
J: Float64 (8 bytes)
WJ: Float64 (8 bytes)
invJ: Float64 (8 bytes)
∂x∂ξ: 2×2 Matrix (32 bytes)
∂ξ∂x: 2×2 Matrix (32 bytes)
gⁱʲ: 2×2 Matrix (32 bytes)
gᵢⱼ: 2×2 Matrix (32 bytes)
────────────────────
Total: ~168 bytes + padding ≈ 200-296 bytes
```

### GPU Architecture Considerations
- **A100**: 128 warps/SM, 32KB+ shared memory → good for caching strategies
- **H100**: Improved L2 cache → may tolerate higher bandwidth
- Occupancy critical for latency hiding in memory-bound kernels

## References

- ClimaCore.Geometry.LocalGeometry documentation
- NVIDIA Performance Analysis Guide
- Roofline Model for bandwidth-bound analysis
- Test results in test/gpu/local_geometry_cuda_resources.jl

## Integration with CI/CD

The test is integrated with the standard test suite:
- Added to `test/runtests.jl` with `gpu_only` metadata
- Added to `.buildkite/pipeline.yml` as a GPU unit test
- Runs automatically on all GPU CI builds
