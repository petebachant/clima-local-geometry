#=
Benchmark the actual performance impact of LocalGeometry on CUDA kernels.

This compares:
1. Kernel operations with full LocalGeometry
2. Kernel operations with simplified/reduced variants
3. Measures execution time, memory bandwidth, occupancy

Usage:
    julia --project scripts/benchmark_local_geometry_impact.jl
=#

using BenchmarkTools
using CUDA
using ClimaComms
using ClimaComms: SingletonCommsContext
import ClimaCore
import ClimaCore: Domains, Topologies, Meshes, Spaces, Geometry, Fields, Grids
import ClimaCore.Geometry: LocalGeometry

@isdefined(TU) || include(
    joinpath(
        dirname(dirname(@__DIR__)),
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

# Create test space
space = TU.SpectralElementSpace2D(FT; context = context)

# Create test fields
scalar_field = Fields.Field(FT, space)
result_field = similar(scalar_field)

# Get full local geometry
local_geom_full = Fields.local_geometry_field(space)

# Extract scalar components for comparison
J_field = Fields.Field(
    similar(parent(scalar_field), FT),
    space,
)
@. J_field = local_geom_full.J

# Create a simplified geometry struct
struct SimplifiedGeometry{FT}
    J::FT
    coordinates::Any  # XYPoint or similar
end

# Create simplified geometry field
simplified_geom = Fields.Field(
    SimplifiedGeometry{FT}.(local_geom_full.J, Spaces.coordinates_data(space)),
    space,
)

println("\n" * "="^70)
println("LOCALGEOMETRY CUDA KERNEL PERFORMANCE BENCHMARK")
println("="^70)

# Warm up GPU
CUDA.synchronize()
@. result_field = scalar_field + 1.0
CUDA.synchronize()

# Benchmark Suite
suite = BenchmarkGroup()

# Baseline: simple operation without LocalGeometry
suite["baseline_simple"] = @benchmarkable begin
    @. $result_field = $scalar_field + 1.0
    CUDA.synchronize()
end

# Full LocalGeometry: access Jacobian
suite["full_lg_jacobian"] = @benchmarkable begin
    @. $result_field = $scalar_field + $local_geom_full.J
    CUDA.synchronize()
end

# Full LocalGeometry: access multiple components
suite["full_lg_multiple"] = @benchmarkable begin
    @. $result_field = $scalar_field + $local_geom_full.J + $local_geom_full.WJ + $local_geom_full.invJ
    CUDA.synchronize()
end

# Extracted scalar: J only
suite["extracted_j"] = @benchmarkable begin
    @. $result_field = $scalar_field + $J_field
    CUDA.synchronize()
end

# Simplified geometry
suite["simplified_lg"] = @benchmarkable begin
    @. $result_field = $scalar_field + $simplified_geom.J
    CUDA.synchronize()
end

# Run benchmarks with tuning
println("\nRunning benchmarks (this takes ~1-2 minutes)...\n")
results = run(suite, verbose = true, samples = 30)

# Print results
println("\n" * "="^70)
println("BENCHMARK RESULTS")
println("="^70)

# Calculate metrics
baseline_time = minimum(results["baseline_simple"].times)
baseline_memory = sizeof(parent(scalar_field))

println("\nExecution Time (μs, lower is better):")
println(repeat("-", 70))

for (name, result) in results
    time_μs = minimum(result.times) / 1000
    overhead = 100 * (time_μs - baseline_time / 1000) / (baseline_time / 1000)
    if overhead >= 0
        overhead_str = "+$overhead"
    else
        overhead_str = "$overhead"
    end
    @printf("  %-30s %10.2f μs  (%6.1f%% vs baseline)\n", name, time_μs, overhead)
end

println("\n" * "="^70)
println("MEMORY FOOTPRINT COMPARISON")
println("="^70)

lg_size = sizeof(parent(local_geom_full))
j_size = sizeof(parent(J_field))
simp_size = sizeof(parent(simplified_geom))
scalar_size = sizeof(parent(scalar_field))

println("\nData structure size per point:")
println("  Scalar field:                    $(scalar_size / length(scalar_field)) bytes")
println("  Full LocalGeometry:              $(lg_size / length(local_geom_full)) bytes")
println("  Extracted J:                     $(j_size / length(J_field)) bytes")
println("  Simplified geometry:             $(simp_size / length(simplified_geom)) bytes")

println("\nTotal memory footprint:")
println("  Scalar field:                    $(scalar_size / (1024^2)) MB")
println("  Full LocalGeometry:              $(lg_size / (1024^2)) MB ($(lg_size / scalar_size)x scalar)")
println("  Extracted J:                     $(j_size / (1024^2)) MB ($(j_size / scalar_size)x scalar)")
println("  Simplified geometry:             $(simp_size / (1024^2)) MB ($(simp_size / scalar_size)x scalar)")

println("\n" * "="^70)
println("ANALYSIS")
println("="^70)

baseline_μs = minimum(results["baseline_simple"].times) / 1000
full_lg_μs = minimum(results["full_lg_jacobian"].times) / 1000
extracted_μs = minimum(results["extracted_j"].times) / 1000
simplified_μs = minimum(results["simplified_lg"].times) / 1000

lg_overhead_pct = 100 * (full_lg_μs - baseline_μs) / baseline_μs

println("\nKey findings:")
println("  Full LocalGeometry overhead:     $(@sprintf("%.1f", lg_overhead_pct))%")
println("  Extracted J overhead:            $(@sprintf("%.1f", 100 * (extracted_μs - baseline_μs) / baseline_μs))%")
println("  Simplified geometry overhead:    $(@sprintf("%.1f", 100 * (simplified_μs - baseline_μs) / baseline_μs))%")

if lg_overhead_pct > 10
    println("\n⚠️  SIGNIFICANT OVERHEAD: LocalGeometry has >10% performance cost")
    println("   Refactoring to simplified structures may be worthwhile")
elseif lg_overhead_pct > 3
    println("\n⚠️  MODERATE OVERHEAD: LocalGeometry has ~3-10% performance cost")
    println("   Consider targeted optimizations (extract components, caching)")
else
    println("\n✓ MINIMAL OVERHEAD: LocalGeometry cost is <3%")
    println("   Current structure is reasonable; refactoring likely not priority")
end

println("\n" * "="^70)
println("INTERPRETATION")
println("="^70)

println("""
The overhead depends on:
  1. How often LocalGeometry is accessed per kernel call
  2. What proportion of LocalGeometry components are actually used
  3. Whether multiple threads compete for cache/bandwidth

In real ClimaAtmos kernels:
  • LocalGeometry is accessed at every grid point
  • Most kernels use only J or (J, WJ) - not full tensors
  • This can amplify overhead in bandwidth-bound operations

Recommendations:
  1. If overhead > 10%: Refactor to simplified geometry types
  2. If overhead 3-10%: Extract J/WJ at kernel entry, pass separately
  3. If overhead < 3%: Keep as-is; focus on other optimizations

Next steps:
  • Run this benchmark on real physics kernels (not just synthetic ops)
  • Profile with NVIDIA Nsight Compute to measure bandwidth efficiency
  • Test with actual ClimaAtmos configurations
""")

println("="^70)

# Save results to file
results_md = """
# LocalGeometry Performance Benchmark Results

## Execution Time Results

| Operation | Time (μs) | Overhead vs Baseline |
|-----------|-----------|----------------------|
"""

baseline_time_μs = minimum(results["baseline_simple"].times) / 1000

for (name, result) in results
    time_μs = minimum(result.times) / 1000
    overhead_pct = 100 * (time_μs - baseline_time_μs) / baseline_time_μs
    results_md *= "| $(replace(name, "_" => " ")) | $(@sprintf("%.2f", time_μs)) | $(@sprintf("%.1f", overhead_pct))% |\n"
end

results_md *= """

## Memory Footprint

- Full LocalGeometry: $(lg_size / (1024^2)) MB ($(lg_size / scalar_size)x scalar field)
- Extracted J: $(j_size / (1024^2)) MB ($(j_size / scalar_size)x scalar field)

## Key Finding

Full LocalGeometry overhead: $(@sprintf("%.1f", lg_overhead_pct))%

"""

if lg_overhead_pct > 10
    results_md *= "⚠️ **SIGNIFICANT OVERHEAD** - Refactoring recommended\n"
elseif lg_overhead_pct > 3
    results_md *= "⚠️ **MODERATE OVERHEAD** - Consider optimization strategies\n"
else
    results_md *= "✓ **MINIMAL OVERHEAD** - Current structure is reasonable\n"
end

# Write markdown results
results_dir = joinpath(@__DIR__, "..", "results")
isdir(results_dir) || mkpath(results_dir)

open(joinpath(results_dir, "benchmark_local_geometry_impact.md"), "w") do io
    write(io, results_md)
end

println("\nMarkdown results written to: $(joinpath(results_dir, "benchmark_local_geometry_impact.md"))")
