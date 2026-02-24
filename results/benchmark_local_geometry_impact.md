# LocalGeometry Performance Benchmark Results

## Execution Time Results

### Section 1: Basic Geometry Access

| Operation | Time (μs) | Overhead vs Baseline |
|-----------|-----------|----------------------|
| baseline simple | 15.07 | 0.0% |
| full lg jacobian | 15.80 | 4.9% |
| 2b pointwise lg j | 16.21 | 7.6% |
| 2c pointwise lg j stack | 16.19 | 7.4% |
| 2d pointwise lg j noinline | 18.09 | 20.0% |
| 2e fd localgeom constructor | 13.68 | -9.2% |
| 2f f x lg | 15.66 | 3.9% |
| 2g lambda f x lg | 15.40 | 2.2% |
| 2h f x lg noinline | 17.02 | 12.9% |
| 2i lambda f x lg noinline | 16.65 | 10.5% |
| full lg multiple | 18.58 | 23.3% |
| extracted j | 15.99 | 6.1% |
| simplified lg | 16.81 | 11.6% |

### Section 2: Struct Size Impact on Inlining

| Struct Type | Size (bytes) | Time (μs) | Overhead vs Baseline |
|-------------|--------------|-----------|----------------------|
| two field access | 16 | 16.75 | 11.2% |
| four field access | 32 | 15.90 | 5.5% |
| eight field access | 64 | 16.49 | 9.4% |
| sixteen field access | 128 | 16.97 | 12.6% |

### Section 3: Projection Operations

| Operation | Time (μs) | Overhead vs Vector Baseline |
|-----------|-----------|----------------------------|
| vector baseline | 16.42 | 0.0% |
| project full lg | 17.31 | 5.4% |
| multiple scalar access | 18.04 | 9.9% |

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
- Full LocalGeometry (J only) overhead: 4.9%
- Extracted J overhead: 6.1%

### Struct Size Impact
- TwoFieldGeom (16 bytes): 11.2%
- FourFieldGeom (32 bytes): 5.5%
- EightFieldGeom (64 bytes): 9.4%
- SixteenFieldGeom (128 bytes): 12.6%

### Projection Operations
- Covariant to Contravariant: 5.4%

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
