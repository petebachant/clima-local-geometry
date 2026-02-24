# LocalGeometry Performance Benchmark Results

## Execution Time Results

### Section 1: Basic Geometry Access

| Operation | Time (μs) | Overhead vs Baseline |
|-----------|-----------|----------------------|
| baseline simple | 15.35 | 0.0% |
| full lg jacobian | 17.04 | 11.0% |
| 2b pointwise lg j | 16.64 | 8.4% |
| 2c pointwise lg j stack | 17.06 | 11.1% |
| 2d pointwise lg j noinline | 18.18 | 18.4% |
| 2e fd localgeom constructor | 13.64 | -11.1% |
| 2f f x lg | 16.05 | 4.6% |
| 2g lambda f x lg | 15.94 | 3.8% |
| 2h f x lg noinline | 17.88 | 16.5% |
| 2i lambda f x lg noinline | 17.21 | 12.1% |
| full lg multiple | 18.81 | 22.5% |
| extracted j | 16.68 | 8.7% |
| simplified lg | 15.84 | 3.2% |

### Section 2: Struct Size Impact on Inlining

| Struct Type | Size (bytes) | Time (μs) | Overhead vs Baseline |
|-------------|--------------|-----------|----------------------|
| two field access | 16 | 16.92 | 10.2% |
| four field access | 32 | 16.85 | 9.8% |
| eight field access | 64 | 17.10 | 11.4% |
| sixteen field access | 128 | 16.94 | 10.4% |

### Section 3: Projection Operations

| Operation | Time (μs) | Overhead vs Vector Baseline |
|-----------|-----------|----------------------------|
| vector baseline | 16.66 | 0.0% |
| project full lg | 16.77 | 0.7% |
| multiple scalar access | 17.17 | 3.1% |

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
- Full LocalGeometry (J only) overhead: 11.0%
- Extracted J overhead: 8.7%

### Struct Size Impact
- TwoFieldGeom (16 bytes): 10.2%
- FourFieldGeom (32 bytes): 9.8%
- EightFieldGeom (64 bytes): 11.4%
- SixteenFieldGeom (128 bytes): 10.4%

### Projection Operations
- Covariant to Contravariant: 0.7%

## Assessment

⚠️ **SIGNIFICANT OVERHEAD** - Refactoring recommended

- Consider extracting commonly-used fields at kernel entry
- Profile real physics kernels with nsys for detailed analysis

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
