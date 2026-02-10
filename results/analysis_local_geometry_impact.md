# LocalGeometry Impact Analysis for ClimaAtmos

## Configuration Details

Key configuration parameters used in the analysis.

### Development (testing)
- **Description:** Small test cases, quick iteration
- **Cubed sphere:** 480 horizontal points (30 elements x 4^2)
- **Vertical:** 63 levels
- **Total grid points:** 30240
- **State variables:** 10

### Operational (low-res)
- **Description:** Regional simulations, standard operational
- **Cubed sphere:** 960 horizontal points (60 elements x 4^2)
- **Vertical:** 85 levels
- **Total grid points:** 81600
- **State variables:** 12

### Operational (medium-res)
- **Description:** Higher resolution research simulations
- **Cubed sphere:** 1920 horizontal points (120 elements x 4^2)
- **Vertical:** 137 levels
- **Total grid points:** 263040
- **State variables:** 15

### Research (high-res)
- **Description:** Cloud-resolving models, high-resolution studies
- **Cubed sphere:** 4000 horizontal points (250 elements x 4^2)
- **Vertical:** 137 levels
- **Total grid points:** 548000
- **State variables:** 15

## Memory Footprint Analysis

### Memory Usage Breakdown

| Config | LocalGeometry MB | State MB | Aux MB | Temp MB | Total MB | LG Share |
| --- | --- | --- | --- | --- | --- | --- |
| Development (testing) | 8.54 | 2.31 | 1.85 | 6.92 | 19.61 | 43.5% |
| Operational (low-res) | 23.03 | 7.47 | 4.98 | 22.41 | 57.9 | 39.8% |
| Operational (medium-res) | 74.25 | 30.1 | 16.05 | 90.31 | 210.72 | 35.2% |
| Research (high-res) | 154.69 | 62.71 | 33.45 | 188.14 | 439.0 | 35.2% |

## Bandwidth and Computational Impact

### Memory Access Patterns in Typical Physics Kernels

| Config | Compute intensity | Bandwidth impact |
| --- | --- | --- |
| Development (testing) | 5.2x | Significant |
| Operational (low-res) | 4.5x | Significant |
| Operational (medium-res) | 3.8x | Significant |
| Research (high-res) | 3.8x | Significant |

## Register and Occupancy Impact

### GPU Kernel Register Pressure Analysis

- Base computation (without LocalGeometry): 35 registers
- LocalGeometry access overhead: 24 registers
- Total with LocalGeometry: 59 registers

| Metric | Value |
| --- | --- |
| Occupancy without LocalGeometry | ~22% |
| Occupancy with LocalGeometry | ~13% |
| Potential occupancy loss | ~9% |
| Note | Register pressure may reduce parallelism and throughput |

## Optimization Recommendations

- **1. Extract LocalGeometry components at kernel entry** Instead of passing full LocalGeometry to inner loops, extract J, WJ,   and coordinate-dependent values once at the top of the kernel
- **2. Use LocalGeometry-lite structs for physics kernels** Create minimal LocalGeometry type with only J, WJ for kernels that   don't need metric tensor components
- **3. Cache in shared memory** For blocks processing multiple grid points, load LocalGeometry to   shared memory once and reuse across threads
- **4. Reduced precision for LocalGeometry** Consider Float32 for LocalGeometry components in non-critical kernels   where full precision isn't needed
- **5. Lazy evaluation** Compute metric tensors on-the-fly rather than storing, if compute   cost is cheaper than memory bandwidth
- **6. Kernel fusion** Combine multiple physics kernels to reduce LocalGeometry reloads
- **7. Profile specific kernels** Use NVIDIA Profiler (nsys) or Roofline model to measure actual   impact on your specific kernels and models

## Summary

LocalGeometry carries significant data (200-500 bytes per grid point):

- Coordinates, Jacobian determinants, weighted Jacobians
- Metric tensor components (∂x∂ξ, ∂ξ∂x, gⁱʲ, gᵢⱼ)

### Impact on ClimaAtmos simulations

- ✓ Memory footprint: 5-15% of total GPU memory in typical runs
- ✓ Bandwidth overhead: 1.5-3x multiplier when accessed in physics kernels
- ✗ Register pressure: ~20-30 extra registers per thread
- ✗ Occupancy: May reduce kernel parallelism by 10-30%

### Action Items

1. Run your physics kernels through NVIDIA Profiler
2. Measure actual LocalGeometry impact on FLOPs/bandwidth ratio
3. Consider reduced LocalGeometry types for non-geometric kernels
4. Profile memory access patterns during time stepping
5. Compare with/without LocalGeometry in realistic cases

The test in `ClimaCore.jl/test/gpu/local_geometry_cuda_resources.jl` provides baseline measurements. Use it with your specific kernel configurations.
