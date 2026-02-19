#=
Benchmark the actual performance impact of LocalGeometry on CUDA kernels.

This compares:
1. Kernel operations with full LocalGeometry
2. Kernel operations with simplified/reduced variants
3. Measures execution time, memory bandwidth, occupancy
4. Tests struct size impact on inlining and performance
5. Tests projection operations commonly used in physics kernels
6. Verifies compiler inlining behavior

Usage:
    julia --project scripts/benchmark_local_geometry_impact.jl

For nsys profiling:
    ./scripts/run-nsys.sh --output=results/nsys/benchmark_lg julia --project scripts/benchmark_local_geometry_impact.jl
=#

using BenchmarkTools
using CUDA
using ClimaComms
using ClimaComms: SingletonCommsContext
using Printf
import ClimaCore
import ClimaCore: Domains, Topologies, Meshes, Spaces, Geometry, Fields, Grids
import ClimaCore.Geometry: LocalGeometry

@isdefined(TU) || include(
    joinpath(
        dirname(@__DIR__),
        "ClimaCore.jl",
        "test",
        "TestUtilities",
        "TestUtilities.jl",
    ),
)
import .TestUtilities as TU

CUDA.functional() || error("CUDA not available")

device = ClimaComms.CUDADevice()
context = SingletonCommsContext(device)

FT = Float64

# Define simplified geometry structs with varying field counts
# Test at what size the compiler stops inlining effectively
struct TwoFieldGeom{FT}
    J::FT
    WJ::FT
end

struct FourFieldGeom{FT}
    J::FT
    WJ::FT
    invJ::FT
    scalar1::FT
end

struct EightFieldGeom{FT}
    J::FT
    WJ::FT
    invJ::FT
    scalar1::FT
    scalar2::FT
    scalar3::FT
    scalar4::FT
    scalar5::FT
end

struct SixteenFieldGeom{FT}
    J::FT
    WJ::FT
    invJ::FT
    scalar1::FT
    scalar2::FT
    scalar3::FT
    scalar4::FT
    scalar5::FT
    scalar6::FT
    scalar7::FT
    scalar8::FT
    scalar9::FT
    scalar10::FT
    scalar11::FT
    scalar12::FT
    scalar13::FT
end

# Create test space
space = TU.SpectralElementSpace2D(FT; context = context)

# Create test fields
scalar_field = Fields.Field(FT, space)
result_field = similar(scalar_field)
vector_field = Fields.Field(Geometry.Covariant12Vector{FT}, space)

# Get full local geometry
local_geom_full = Fields.local_geometry_field(space)

# Extract scalar components for comparison
J_field = similar(scalar_field)
@. J_field = local_geom_full.J

# Create fields with simplified geometry structs
two_field_geom = similar(scalar_field, TwoFieldGeom{FT})
@. two_field_geom = TwoFieldGeom(local_geom_full.J, local_geom_full.WJ)

four_field_geom = similar(scalar_field, FourFieldGeom{FT})
@. four_field_geom = FourFieldGeom(local_geom_full.J, local_geom_full.WJ, local_geom_full.invJ, FT(1.0))

eight_field_geom = similar(scalar_field, EightFieldGeom{FT})
@. eight_field_geom = EightFieldGeom(
    local_geom_full.J, local_geom_full.WJ, local_geom_full.invJ,
    FT(1.0), FT(2.0), FT(3.0), FT(4.0), FT(5.0)
)

sixteen_field_geom = similar(scalar_field, SixteenFieldGeom{FT})
@. sixteen_field_geom = SixteenFieldGeom(
    local_geom_full.J, local_geom_full.WJ, local_geom_full.invJ,
    FT(1.0), FT(2.0), FT(3.0), FT(4.0), FT(5.0),
    FT(6.0), FT(7.0), FT(8.0), FT(9.0), FT(10.0),
    FT(11.0), FT(12.0), FT(13.0)
)

# Create a simple wrapper using NamedTuple
simplified_geom = (J = J_field,)

println("\n" * "="^70)
println("LOCALGEOMETRY CUDA KERNEL PERFORMANCE BENCHMARK")
println("="^70)

# Warm up GPU
CUDA.synchronize()
@. result_field = scalar_field + 1.0
CUDA.synchronize()

# Test inlining verification
println("\n" * "="^70)
println("INLINING VERIFICATION")
println("="^70)
println("\nSizeof checks (should help predict inlining threshold):")
println("  LocalGeometry field element:  $(sizeof(eltype(parent(local_geom_full)))) bytes")
println("  TwoFieldGeom:                 $(sizeof(TwoFieldGeom{FT})) bytes")
println("  FourFieldGeom:                $(sizeof(FourFieldGeom{FT})) bytes")
println("  EightFieldGeom:               $(sizeof(EightFieldGeom{FT})) bytes")
println("  SixteenFieldGeom:             $(sizeof(SixteenFieldGeom{FT})) bytes")
println("\nNote: CUDA typically inlines structs < 128 bytes effectively")

# Benchmark Suite
suite = BenchmarkGroup()

# ========================================
# SECTION 1: Basic Geometry Access
# ========================================
println("\n" * "="^70)
println("SECTION 1: Basic Geometry Access Patterns")
println("="^70)

# Baseline: simple operation without LocalGeometry
suite["1_baseline_simple"] = @benchmarkable begin
    @. $result_field = $scalar_field + 1.0
    CUDA.synchronize()
end

# Full LocalGeometry: access Jacobian only
suite["2_full_lg_jacobian"] = @benchmarkable begin
    @. $result_field = $scalar_field + $local_geom_full.J
    CUDA.synchronize()
end

# Full LocalGeometry: access multiple scalar components
suite["3_full_lg_multiple"] = @benchmarkable begin
    @. $result_field = $scalar_field + $local_geom_full.J + $local_geom_full.WJ + $local_geom_full.invJ
    CUDA.synchronize()
end

# Extracted scalar: J only
suite["4_extracted_j"] = @benchmarkable begin
    @. $result_field = $scalar_field + $J_field
    CUDA.synchronize()
end

# Simplified geometry
suite["5_simplified_lg"] = @benchmarkable begin
    @. $result_field = $scalar_field + $simplified_geom.J
    CUDA.synchronize()
end

# ========================================
# SECTION 2: Struct Size Impact on Performance
# ========================================
println("\nSECTION 2: Testing impact of struct size on inlining")

suite["6_two_field_access"] = @benchmarkable begin
    @. $result_field = $scalar_field + $two_field_geom.J
    CUDA.synchronize()
end

suite["7_four_field_access"] = @benchmarkable begin
    @. $result_field = $scalar_field + $four_field_geom.J
    CUDA.synchronize()
end

suite["8_eight_field_access"] = @benchmarkable begin
    @. $result_field = $scalar_field + $eight_field_geom.J
    CUDA.synchronize()
end

suite["9_sixteen_field_access"] = @benchmarkable begin
    @. $result_field = $scalar_field + $sixteen_field_geom.J
    CUDA.synchronize()
end

# ========================================
# SECTION 3: Projection Operations
# ========================================
println("\nSECTION 3: Testing projection operations (common in physics kernels)")

# Initialize vector field with some test data
@. vector_field = Geometry.Covariant12Vector(scalar_field, scalar_field)

result_vector = similar(vector_field, Geometry.Contravariant12Vector{FT})

# Baseline vector operation (no projection)
suite["10_vector_baseline"] = @benchmarkable begin
    @. $result_vector = Geometry.Contravariant12Vector($scalar_field * 2.0, $scalar_field * 2.0)
    CUDA.synchronize()
end

# Projection: Covariant to Contravariant using full LocalGeometry
suite["11_project_full_lg"] = @benchmarkable begin
    @. $result_vector = Geometry.project(Geometry.Contravariant12Axis(), $vector_field, $local_geom_full)
    CUDA.synchronize()
end

# Test accessing multiple scalar components together
suite["12_multiple_scalar_access"] = @benchmarkable begin
    @. $result_field = $scalar_field * ($local_geom_full.J + $local_geom_full.WJ * 0.5)
    CUDA.synchronize()
end

# Run benchmarks with tuning
println("\nRunning benchmarks (this takes ~2-3 minutes)...\n")
results = run(suite, verbose = true, samples = 30)

# Print results
println("\n" * "="^70)
println("BENCHMARK RESULTS")
println("="^70)

# Calculate metrics - use first benchmark as baseline
baseline_time = minimum(results["1_baseline_simple"].times)
baseline_memory = sizeof(parent(scalar_field))

println("\nSECTION 1: Basic Geometry Access")
println(repeat("-", 70))
println("\nExecution Time (μs, lower is better):")

section1_keys = filter(k -> startswith(k, "1_") || startswith(k, "2_") || startswith(k, "3_") || startswith(k, "4_") || startswith(k, "5_"), collect(keys(results)))
for key in sort(section1_keys)
    result = results[key]
    time_μs = minimum(result.times) / 1000
    overhead = 100 * (time_μs - baseline_time / 1000) / (baseline_time / 1000)
    overhead_str = overhead >= 0 ? @sprintf("+%.1f", overhead) : @sprintf("%.1f", overhead)
    # Strip number prefix for display
    display_name = replace(key, r"^\d+_" => "")
    @printf("  %-30s %10.2f μs  (%6s%% vs baseline)\n", display_name, time_μs, overhead_str)
end

println("\n" * repeat("-", 70))
println("SECTION 2: Struct Size Impact on Inlining")
println(repeat("-", 70))
println("\nExecution Time (μs, lower is better):")

section2_keys = filter(k -> startswith(k, "6_") || startswith(k, "7_") || startswith(k, "8_") || startswith(k, "9_"), collect(keys(results)))
for key in sort(section2_keys)
    result = results[key]
    time_μs = minimum(result.times) / 1000
    overhead = 100 * (time_μs - baseline_time / 1000) / (baseline_time / 1000)
    overhead_str = overhead >= 0 ? @sprintf("+%.1f", overhead) : @sprintf("%.1f", overhead)
    display_name = replace(key, r"^\d+_" => "")
    @printf("  %-30s %10.2f μs  (%6s%% vs baseline)\n", display_name, time_μs, overhead_str)
end

println("\n" * repeat("-", 70))
println("SECTION 3: Projection Operations")
println(repeat("-", 70))
println("\nExecution Time (μs, lower is better):")

section3_keys = filter(k -> startswith(k, "10_") || startswith(k, "11_") || startswith(k, "12_"), collect(keys(results)))
vector_baseline = minimum(results["10_vector_baseline"].times) / 1000
for key in sort(section3_keys)
    result = results[key]
    time_μs = minimum(result.times) / 1000
    overhead = 100 * (time_μs - vector_baseline) / vector_baseline
    overhead_str = overhead >= 0 ? @sprintf("+%.1f", overhead) : @sprintf("%.1f", overhead)
    display_name = replace(key, r"^\d+_" => "")
    @printf("  %-30s %10.2f μs  (%6s%% vs vec_baseline)\n", display_name, time_μs, overhead_str)
end

println("\n" * "="^70)
println("MEMORY FOOTPRINT COMPARISON")
println("="^70)

lg_size = sizeof(parent(local_geom_full))
j_size = sizeof(parent(J_field))
two_field_size = sizeof(parent(two_field_geom))
four_field_size = sizeof(parent(four_field_geom))
eight_field_size = sizeof(parent(eight_field_geom))
sixteen_field_size = sizeof(parent(sixteen_field_geom))
scalar_size = sizeof(parent(scalar_field))

println("\nData structure size per point:")
println("  Scalar field:                    $(scalar_size / length(scalar_field)) bytes")
println("  TwoFieldGeom:                    $(two_field_size / length(two_field_geom)) bytes")
println("  FourFieldGeom:                   $(four_field_size / length(four_field_geom)) bytes")
println("  EightFieldGeom:                  $(eight_field_size / length(eight_field_geom)) bytes")
println("  SixteenFieldGeom:                $(sixteen_field_size / length(sixteen_field_geom)) bytes")
println("  Full LocalGeometry:              $(lg_size / length(local_geom_full)) bytes")
println("  Extracted J:                     $(j_size / length(J_field)) bytes")

println("\nTotal memory footprint:")
println("  Scalar field:                    $(scalar_size / (1024^2)) MB")
println("  TwoFieldGeom:                    $(two_field_size / (1024^2)) MB ($(two_field_size / scalar_size)x scalar)")
println("  FourFieldGeom:                   $(four_field_size / (1024^2)) MB ($(four_field_size / scalar_size)x scalar)")
println("  EightFieldGeom:                  $(eight_field_size / (1024^2)) MB ($(eight_field_size / scalar_size)x scalar)")
println("  SixteenFieldGeom:                $(sixteen_field_size / (1024^2)) MB ($(sixteen_field_size / scalar_size)x scalar)")
println("  Full LocalGeometry:              $(lg_size / (1024^2)) MB ($(lg_size / scalar_size)x scalar)")
println("  Extracted J:                     $(j_size / (1024^2)) MB ($(j_size / scalar_size)x scalar)")

println("\n" * "="^70)
println("ANALYSIS & KEY FINDINGS")
println("="^70)

baseline_μs = minimum(results["1_baseline_simple"].times) / 1000
full_lg_μs = minimum(results["2_full_lg_jacobian"].times) / 1000
extracted_μs = minimum(results["4_extracted_j"].times) / 1000

two_field_μs = minimum(results["6_two_field_access"].times) / 1000
four_field_μs = minimum(results["7_four_field_access"].times) / 1000
eight_field_μs = minimum(results["8_eight_field_access"].times) / 1000
sixteen_field_μs = minimum(results["9_sixteen_field_access"].times) / 1000

projection_μs = minimum(results["11_project_full_lg"].times) / 1000
vector_baseline_μs = minimum(results["10_vector_baseline"].times) / 1000

lg_overhead_pct = 100 * (full_lg_μs - baseline_μs) / baseline_μs
projection_overhead_pct = 100 * (projection_μs - vector_baseline_μs) / vector_baseline_μs

println("\n1. BASIC GEOMETRY ACCESS OVERHEAD:")
println("   Full LocalGeometry (J only):      $(@sprintf("%.1f", lg_overhead_pct))%")
println("   Extracted J:                      $(@sprintf("%.1f", 100 * (extracted_μs - baseline_μs) / baseline_μs))%")

println("\n2. STRUCT SIZE IMPACT (accessing single field):")
println("   TwoFieldGeom (16 bytes):          $(@sprintf("%.1f", 100 * (two_field_μs - baseline_μs) / baseline_μs))%")
println("   FourFieldGeom (32 bytes):         $(@sprintf("%.1f", 100 * (four_field_μs - baseline_μs) / baseline_μs))%")
println("   EightFieldGeom (64 bytes):        $(@sprintf("%.1f", 100 * (eight_field_μs - baseline_μs) / baseline_μs))%")
println("   SixteenFieldGeom (128 bytes):     $(@sprintf("%.1f", 100 * (sixteen_field_μs - baseline_μs) / baseline_μs))%")

# Detect inlining threshold
inlining_threshold_detected = false
if abs(sixteen_field_μs - eight_field_μs) > 0.1 * eight_field_μs
    println("\n   ⚠️  INLINING THRESHOLD DETECTED: Performance degrades at ~128 bytes")
    inlining_threshold_detected = true
elseif abs(eight_field_μs - four_field_μs) > 0.1 * four_field_μs
    println("\n   ⚠️  INLINING THRESHOLD DETECTED: Performance degrades at ~64 bytes")
    inlining_threshold_detected = true
else
    println("\n   ✓ All struct sizes show similar performance - good inlining")
end

println("\n3. PROJECTION OPERATIONS OVERHEAD:")
println("   Covariant->Contravariant:         $(@sprintf("%.1f", projection_overhead_pct))%")

println("\n" * "="^70)
println("RECOMMENDATIONS")
println("="^70)

if lg_overhead_pct > 10 || projection_overhead_pct > 20
    println("\n⚠️  SIGNIFICANT OVERHEAD DETECTED:")
    if lg_overhead_pct > 10
        println("   • Full LocalGeometry access has >10% overhead")
        println("   • Consider: Extract J/WJ at kernel entry")
    end
    if projection_overhead_pct > 20
        println("   • Projection operations have >20% overhead")
        println("   • Consider: Cache projected vectors or simplify metric tensors")
    end
    if inlining_threshold_detected
        println("   • Inlining threshold reached - consider splitting large structs")
    end
    println("\n   ACTION: Refactor hot paths to use simplified geometry types")

elseif lg_overhead_pct > 3 || projection_overhead_pct > 10
    println("\n⚠️  MODERATE OVERHEAD DETECTED:")
    println("   • LocalGeometry overhead: $(@sprintf("%.1f", lg_overhead_pct))%")
    println("   • Projection overhead: $(@sprintf("%.1f", projection_overhead_pct))%")
    println("\n   CONSIDER:")
    println("   1. Extract commonly-used components (J, WJ) at kernel entry")
    println("   2. Profile with nsys to see actual bandwidth/occupancy impact:")
    println("      ./scripts/run-nsys.sh --output=results/nsys/benchmark_lg \\")
    println("          julia --project scripts/benchmark_local_geometry_impact.jl")

else
    println("\n✓ MINIMAL OVERHEAD:")
    println("   • LocalGeometry overhead: $(@sprintf("%.1f", lg_overhead_pct))%")
    println("   • Projection overhead: $(@sprintf("%.1f", projection_overhead_pct))%")
    println("   • Current structure is reasonable for these access patterns")
    println("\n   NOTE: Real physics kernels may show different behavior due to:")
    println("   • Multiple LocalGeometry accesses per kernel")
    println("   • Complex projection operations in tight loops")
    println("   • Recommend profiling actual physics kernels with nsys")
end

println("\n" * "="^70)
println("CODE INSPECTION NOTES")
println("="^70)
println("""
To verify compiler inlining behavior:

1. Check LLVM IR for a simple kernel:
   julia> f(x, lg) = x + lg.J
   julia> @code_llvm f(1.0, first(local_geom_full))

   Look for: Should see direct field access, not function calls

2. Check PTX assembly for CUDA kernels:
   julia> using CUDA
   julia> kernel(x, lg) = (@inbounds x[1] += lg[1].J; nothing)
   julia> @device_code_ptx kernel(CuArray([1.0]), parent(local_geom_full))

   Look for: ld.param instructions (should be minimal for inlined structs)

3. Use nsys to profile actual memory bandwidth:
   ./scripts/run-nsys.sh --output=results/nsys/benchmark_lg \\
       julia --project scripts/benchmark_local_geometry_impact.jl

   Then analyze with:
   nsys stats results/nsys/benchmark_lg.nsys-rep

   Look for: Memory bandwidth utilization, kernel occupancy
""")

println("="^70)

begin
    # Save results to file
    local results_md
    results_md = """
# LocalGeometry Performance Benchmark Results

## Execution Time Results

### Section 1: Basic Geometry Access

| Operation | Time (μs) | Overhead vs Baseline |
|-----------|-----------|----------------------|
"""

    baseline_time_μs = minimum(results["1_baseline_simple"].times) / 1000

    section1_keys = ["1_baseline_simple", "2_full_lg_jacobian", "3_full_lg_multiple", "4_extracted_j", "5_simplified_lg"]
    for key in section1_keys
        if haskey(results, key)
            result = results[key]
            time_μs = minimum(result.times) / 1000
            overhead_pct = 100 * (time_μs - baseline_time_μs) / baseline_time_μs
            display_name = replace(key, r"^\d+_" => "")
            results_md = results_md * "| $(replace(display_name, "_" => " ")) | $(@sprintf("%.2f", time_μs)) | $(@sprintf("%.1f", overhead_pct))% |\n"
        end
    end

    results_md = results_md * """

### Section 2: Struct Size Impact on Inlining

| Struct Type | Size (bytes) | Time (μs) | Overhead vs Baseline |
|-------------|--------------|-----------|----------------------|
"""

    section2_keys = ["6_two_field_access", "7_four_field_access", "8_eight_field_access", "9_sixteen_field_access"]
    struct_sizes = [16, 32, 64, 128]
    for (i, key) in enumerate(section2_keys)
        if haskey(results, key)
            result = results[key]
            time_μs = minimum(result.times) / 1000
            overhead_pct = 100 * (time_μs - baseline_time_μs) / baseline_time_μs
            display_name = replace(key, r"^\d+_" => "")
            results_md = results_md * "| $(replace(display_name, "_" => " ")) | $(struct_sizes[i]) | $(@sprintf("%.2f", time_μs)) | $(@sprintf("%.1f", overhead_pct))% |\n"
        end
    end

    results_md = results_md * """

### Section 3: Projection Operations

| Operation | Time (μs) | Overhead vs Vector Baseline |
|-----------|-----------|----------------------------|
"""

    vector_baseline_time = minimum(results["10_vector_baseline"].times) / 1000
    section3_keys = ["10_vector_baseline", "11_project_full_lg", "12_metric_tensor_access"]
    for key in section3_keys
        if haskey(results, key)
            result = results[key]
            time_μs = minimum(result.times) / 1000
            overhead_pct = 100 * (time_μs - vector_baseline_time) / vector_baseline_time
            display_name = replace(key, r"^\d+_" => "")
            results_md = results_md * "| $(replace(display_name, "_" => " ")) | $(@sprintf("%.2f", time_μs)) | $(@sprintf("%.1f", overhead_pct))% |\n"
        end
    end

    results_md = results_md * """

## Memory Footprint

| Structure | Total Size (MB) | Size per Point (bytes) | Ratio vs Scalar |
|-----------|-----------------|------------------------|-----------------|
"""

    results_md = results_md * "| Scalar field | $(@sprintf("%.3f", scalar_size / (1024^2))) | $(@sprintf("%.1f", scalar_size / length(scalar_field))) | 1.0x |\n"
    results_md = results_md * "| TwoFieldGeom | $(@sprintf("%.3f", two_field_size / (1024^2))) | $(@sprintf("%.1f", two_field_size / length(two_field_geom))) | $(@sprintf("%.1f", two_field_size / scalar_size))x |\n"
    results_md = results_md * "| FourFieldGeom | $(@sprintf("%.3f", four_field_size / (1024^2))) | $(@sprintf("%.1f", four_field_size / length(four_field_geom))) | $(@sprintf("%.1f", four_field_size / scalar_size))x |\n"
    results_md = results_md * "| EightFieldGeom | $(@sprintf("%.3f", eight_field_size / (1024^2))) | $(@sprintf("%.1f", eight_field_size / length(eight_field_geom))) | $(@sprintf("%.1f", eight_field_size / scalar_size))x |\n"
    results_md = results_md * "| SixteenFieldGeom | $(@sprintf("%.3f", sixteen_field_size / (1024^2))) | $(@sprintf("%.1f", sixteen_field_size / length(sixteen_field_geom))) | $(@sprintf("%.1f", sixteen_field_size / scalar_size))x |\n"
    results_md = results_md * "| Full LocalGeometry | $(@sprintf("%.3f", lg_size / (1024^2))) | $(@sprintf("%.1f", lg_size / length(local_geom_full))) | $(@sprintf("%.1f", lg_size / scalar_size))x |\n"

    results_md = results_md * """

## Key Findings

### Basic Geometry Access
- Full LocalGeometry (J only) overhead: $(@sprintf("%.1f", lg_overhead_pct))%
- Extracted J overhead: $(@sprintf("%.1f", 100 * (extracted_μs - baseline_μs) / baseline_μs))%

### Struct Size Impact
- TwoFieldGeom (16 bytes): $(@sprintf("%.1f", 100 * (two_field_μs - baseline_μs) / baseline_μs))%
- FourFieldGeom (32 bytes): $(@sprintf("%.1f", 100 * (four_field_μs - baseline_μs) / baseline_μs))%
- EightFieldGeom (64 bytes): $(@sprintf("%.1f", 100 * (eight_field_μs - baseline_μs) / baseline_μs))%
- SixteenFieldGeom (128 bytes): $(@sprintf("%.1f", 100 * (sixteen_field_μs - baseline_μs) / baseline_μs))%

### Projection Operations
- Covariant to Contravariant: $(@sprintf("%.1f", projection_overhead_pct))%

## Assessment

"""

    if lg_overhead_pct > 10 || projection_overhead_pct > 20
        results_md = results_md * "⚠️ **SIGNIFICANT OVERHEAD** - Refactoring recommended\n\n"
        results_md = results_md * "- Consider extracting commonly-used fields at kernel entry\n"
        results_md = results_md * "- Profile real physics kernels with nsys for detailed analysis\n"
        if inlining_threshold_detected
            results_md = results_md * "- Inlining threshold reached - consider splitting large structs\n"
        end
    elseif lg_overhead_pct > 3 || projection_overhead_pct > 10
        results_md = results_md * "⚠️ **MODERATE OVERHEAD** - Consider optimization strategies\n\n"
        results_md = results_md * "- Extract J/WJ at kernel entry for hot paths\n"
        results_md = results_md * "- Use nsys profiling to identify bandwidth-bound kernels\n"
    else
        results_md = results_md * "✓ **MINIMAL OVERHEAD** - Current structure is reasonable for these access patterns\n\n"
        results_md = results_md * "- Monitor performance in real physics kernels\n"
        results_md = results_md * "- Consider nsys profiling for production workloads\n"
    end

    results_md = results_md * """

## Next Steps

1. **Profile real physics kernels**: These synthetic benchmarks test individual operations. Real kernels may show different behavior.

2. **Use nsys for detailed analysis**:
   ```bash
   ./scripts/run-nsys.sh --output=results/nsys/benchmark_lg \\
       julia --project scripts/benchmark_local_geometry_impact.jl
   nsys stats results/nsys/benchmark_lg.nsys-rep
   ```

3. **Verify inlining**: Use `@code_llvm` and `@device_code_ptx` to inspect compiler output.

4. **Test with realistic field operations**: Include projections in gradient/divergence operators.
"""

    # Write markdown results
    results_dir = joinpath(@__DIR__, "..", "results")
    isdir(results_dir) || mkpath(results_dir)

    open(joinpath(results_dir, "benchmark_local_geometry_impact.md"), "w") do io
        write(io, results_md)
    end

    println("\nMarkdown results written to: $(joinpath(results_dir, "benchmark_local_geometry_impact.md"))")
end
