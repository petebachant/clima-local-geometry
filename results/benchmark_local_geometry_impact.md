# LocalGeometry Performance Benchmark Results

## Execution Time Results

### Section 1: Basic Geometry Access

| Operation | Time (μs) | Overhead vs Baseline |
|-----------|-----------|----------------------|
| baseline simple | 15.17 | 0.0% |
| full lg jacobian | 16.38 | 8.0% |
| 2b pointwise lg j | 16.41 | 8.2% |
| 2c pointwise lg j stack | 17.44 | 15.0% |
| 2d pointwise lg j noinline | 17.62 | 16.2% |
| 2e fd localgeom constructor | 13.91 | -8.3% |
| 2f f x lg | 15.70 | 3.5% |
| 2g lambda f x lg | 15.53 | 2.4% |
| 2h f x lg noinline | 17.50 | 15.4% |
| full lg multiple | 18.79 | 23.9% |
| extracted j | 16.08 | 6.0% |
| simplified lg | 16.26 | 7.2% |

### Section 2: Struct Size Impact on Inlining

| Struct Type | Size (bytes) | Time (μs) | Overhead vs Baseline |
|-------------|--------------|-----------|----------------------|
| two field access | 16 | 16.29 | 7.4% |
| four field access | 32 | 16.46 | 8.5% |
| eight field access | 64 | 16.24 | 7.1% |
| sixteen field access | 128 | 16.57 | 9.2% |

### Section 3: Projection Operations

| Operation | Time (μs) | Overhead vs Vector Baseline |
|-----------|-----------|----------------------------|
| vector baseline | 16.39 | 0.0% |
| project full lg | 16.08 | -1.9% |
| multiple scalar access | 17.35 | 5.9% |

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
- Extracted J overhead: 6.0%

### Struct Size Impact
- TwoFieldGeom (16 bytes): 7.4%
- FourFieldGeom (32 bytes): 8.5%
- EightFieldGeom (64 bytes): 7.1%
- SixteenFieldGeom (128 bytes): 9.2%

### Projection Operations
- Covariant to Contravariant: -1.9%

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
