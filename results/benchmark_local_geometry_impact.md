# LocalGeometry Performance Benchmark Results

## Execution Time Results

### Section 1: Basic Geometry Access

| Operation | Time (μs) | Overhead vs Baseline |
|-----------|-----------|----------------------|
| baseline simple | 14.91 | 0.0% |
| full lg jacobian | 16.11 | 8.0% |
| 2b pointwise lg j | 16.00 | 7.3% |
| 2c pointwise lg j stack | 16.12 | 8.1% |
| 2d pointwise lg j noinline | 17.58 | 17.9% |
| 2e fd localgeom constructor | 13.59 | -8.9% |
| 2f f x lg | 15.35 | 3.0% |
| 2g lambda f x lg | 16.04 | 7.6% |
| 2h f x lg noinline | 16.69 | 11.9% |
| 2i lambda f x lg noinline | 16.74 | 12.3% |
| full lg multiple | 18.63 | 24.9% |
| extracted j | 16.07 | 7.8% |
| simplified lg | 15.88 | 6.5% |

### Section 2: Struct Size Impact on Inlining

| Struct Type | Size (bytes) | Time (μs) | Overhead vs Baseline |
|-------------|--------------|-----------|----------------------|
| two field access | 16 | 16.02 | 7.4% |
| four field access | 32 | 16.02 | 7.4% |
| eight field access | 64 | 16.24 | 8.9% |
| sixteen field access | 128 | 16.17 | 8.5% |

### Section 3: Projection Operations

| Operation | Time (μs) | Overhead vs Vector Baseline |
|-----------|-----------|----------------------------|
| vector baseline | 16.09 | 0.0% |
| project full lg | 16.43 | 2.1% |
| multiple scalar access | 17.92 | 11.4% |

## Memory Footprint

| Structure | Total Size (MB) | Size per Point (bytes) | Ratio vs Scalar |
|-----------|-----------------|------------------------|-----------------|
| Scalar field | 0.000 | 128.0 | 1.0x |
| TwoFieldGeom | 0.000 | 256.0 | 2.0x |
| FourFieldGeom | 0.000 | 512.0 | 4.0x |
| EightFieldGeom | 0.001 | 1024.0 | 8.0x |
| SixteenFieldGeom | 0.002 | 2048.0 | 16.0x |
| Full LocalGeometry | 0.003 | 2688.0 | 21.0x |

## Key Findings

### Basic Geometry Access
- Full LocalGeometry (J only) overhead: 8.0%
- Extracted J overhead: 7.8%

### Struct Size Impact
- TwoFieldGeom (16 bytes): 7.4%
- FourFieldGeom (32 bytes): 7.4%
- EightFieldGeom (64 bytes): 8.9%
- SixteenFieldGeom (128 bytes): 8.5%

### Projection Operations
- Covariant to Contravariant: 2.1%

## Assessment

⚠️ **MODERATE OVERHEAD** - Consider optimization strategies

- Extract J/WJ at kernel entry for hot paths
- Use nsys profiling to identify bandwidth-bound kernels

## Next Steps

1. **Profile real physics kernels**: These synthetic benchmarks test individual operations. Real kernels may show different behavior.

2. **Use nsys for detailed analysis**:
   ```bash
   ./scripts/run-nsys.sh --output=results/nsys/benchmark_lg \
       julia --project scripts/benchmark_local_geometry_impact.jl
   nsys stats results/nsys/benchmark_lg.nsys-rep
   ```

3. **Verify inlining**: Use `@code_llvm` and `@device_code_ptx` to inspect compiler output.

4. **Test with realistic field operations**: Include projections in gradient/divergence operators.
