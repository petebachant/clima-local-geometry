# LocalGeometry Performance Benchmark Results

## Execution Time Results

### Section 1: Basic Geometry Access

| Operation | Time (μs) | Overhead vs Baseline |
|-----------|-----------|----------------------|
| baseline simple | 16.07 | 0.0% |
| full lg jacobian | 17.27 | 7.5% |
| full lg multiple | 19.53 | 21.5% |
| extracted j | 17.13 | 6.6% |
| simplified lg | 17.21 | 7.1% |

### Section 2: Struct Size Impact on Inlining

| Struct Type | Size (bytes) | Time (μs) | Overhead vs Baseline |
|-------------|--------------|-----------|----------------------|
| two field access | 16 | 17.34 | 7.9% |
| four field access | 32 | 17.43 | 8.5% |
| eight field access | 64 | 16.91 | 5.2% |
| sixteen field access | 128 | 17.05 | 6.1% |

### Section 3: Projection Operations

| Operation | Time (μs) | Overhead vs Vector Baseline |
|-----------|-----------|----------------------------|
| vector baseline | 17.30 | 0.0% |
| project full lg | 17.61 | 1.8% |

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
- Full LocalGeometry (J only) overhead: 7.5%
- Extracted J overhead: 6.6%

### Struct Size Impact
- TwoFieldGeom (16 bytes): 7.9%
- FourFieldGeom (32 bytes): 8.5%
- EightFieldGeom (64 bytes): 5.2%
- SixteenFieldGeom (128 bytes): 6.1%

### Projection Operations
- Covariant to Contravariant: 1.8%

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
