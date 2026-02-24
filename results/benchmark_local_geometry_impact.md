# LocalGeometry Performance Benchmark Results

## Execution Time Results

### Section 1: Basic Geometry Access

| Operation | Time (μs) | Overhead vs Baseline |
|-----------|-----------|----------------------|
| baseline simple | 15.86 | 0.0% |
| full lg jacobian | 16.85 | 6.2% |
| 2b pointwise lg j | 17.39 | 9.6% |
| 2c pointwise lg j stack | 17.29 | 9.0% |
| 2d pointwise lg j noinline | 18.45 | 16.3% |
| 2e fd localgeom constructor | 13.79 | -13.1% |
| 2f f x lg | 16.33 | 3.0% |
| 2g lambda f x lg | 16.85 | 6.2% |
| 2h f x lg noinline | 17.49 | 10.3% |
| 2i lambda f x lg noinline | 16.90 | 6.6% |
| full lg multiple | 19.36 | 22.1% |
| extracted j | 16.90 | 6.6% |
| simplified lg | 17.12 | 7.9% |

### Section 2: Struct Size Impact on Inlining

| Struct Type | Size (bytes) | Time (μs) | Overhead vs Baseline |
|-------------|--------------|-----------|----------------------|
| two field access | 16 | 16.98 | 7.1% |
| four field access | 32 | 16.73 | 5.5% |
| eight field access | 64 | 17.21 | 8.5% |
| sixteen field access | 128 | 17.17 | 8.3% |

### Section 3: Projection Operations

| Operation | Time (μs) | Overhead vs Vector Baseline |
|-----------|-----------|----------------------------|
| vector baseline | 17.16 | 0.0% |
| project full lg | 17.24 | 0.5% |
| multiple scalar access | 18.23 | 6.2% |

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
- Full LocalGeometry (J only) overhead: 6.2%
- Extracted J overhead: 6.6%

### Struct Size Impact
- TwoFieldGeom (16 bytes): 7.1%
- FourFieldGeom (32 bytes): 5.5%
- EightFieldGeom (64 bytes): 8.5%
- SixteenFieldGeom (128 bytes): 8.3%

### Projection Operations
- Covariant to Contravariant: 0.5%

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
