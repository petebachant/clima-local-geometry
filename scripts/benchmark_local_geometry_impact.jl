#=
Benchmark the actual performance impact of LocalGeometry on CUDA kernels.
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

@inline lg_j_direct(lg) = lg.J
@inline lg_j_stack1(lg) = lg_j_direct(lg)
@inline lg_j_stack2(lg) = lg_j_stack1(lg)
@noinline lg_j_noinline(lg) = lg_j_stack2(lg)

@inline f_from_lg(lg) = lg.J
@noinline f_from_lg_noinline(lg) = lg.J

struct WithLG{LG}
    lg::LG
end

function fd_localgeom_kernel!(out, input, space, nv)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= nv
        lg = Geometry.LocalGeometry(space, i, (1, 1, 1))
        @inbounds out[i, 1] = input[i, 1] + lg.J
    end
    return nothing
end

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

# Testing overhead of nothing fields versus simplified structs
# These structs mimic LocalGeometry with optional fields, but stay isbits
struct TwoFieldWithNothingGeom{FT}
    J::FT
    WJ::FT
    metadata1::Nothing
    metadata2::Nothing
end

struct FourFieldPartiallyNothingGeom{FT}
    J::FT
    WJ::FT
    invJ::Nothing  # Set to nothing
    extra1::Nothing  # Set to nothing
end

struct FullGeomWithOptionals{FT}
    J::FT
    WJ::FT
    invJ::FT
    scalar1::FT
    scalar2::FT
    scalar3::FT
end

struct MinimalGeomWithPadding{FT}
    J::FT
    WJ::FT
    padding::NTuple{6,Nothing}  # Simulate overhead of nothing fields
end

# Create test space
space = TU.SpectralElementSpace2D(FT; context=context)

# Finite-difference space for explicit LocalGeometry(space, idx, hidx)
fd_space = TU.ColumnCenterFiniteDifferenceSpace(FT; context=context, zelem=128)

# Create test fields
scalar_field = Fields.Field(FT, space)
result_field = similar(scalar_field)
vector_field = Fields.Field(Geometry.Covariant12Vector{FT}, space)

fd_scalar_field = Fields.Field(FT, fd_space)
fd_result_field = similar(fd_scalar_field)
fd_nv = Spaces.nlevels(fd_space)
fd_in = parent(Fields.field_values(fd_scalar_field))
fd_out = parent(Fields.field_values(fd_result_field))
fd_threads = 256
fd_blocks = cld(fd_nv, fd_threads)

# Get full local geometry
local_geom_full = Fields.local_geometry_field(space)

wrapper_lg_field = similar(scalar_field, WithLG{eltype(local_geom_full)})
@. wrapper_lg_field = WithLG(local_geom_full)

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

# Create fields with nothing-based geometry structs
two_field_with_nothing = similar(scalar_field, TwoFieldWithNothingGeom{FT})
@. two_field_with_nothing = TwoFieldWithNothingGeom(local_geom_full.J, local_geom_full.WJ, nothing, nothing)

four_field_partially_nothing = similar(scalar_field, FourFieldPartiallyNothingGeom{FT})
@. four_field_partially_nothing = FourFieldPartiallyNothingGeom(local_geom_full.J, local_geom_full.WJ, nothing, nothing)

full_geom_with_optionals = similar(scalar_field, FullGeomWithOptionals{FT})
@. full_geom_with_optionals = FullGeomWithOptionals(
    local_geom_full.J,
    local_geom_full.WJ,
    local_geom_full.invJ,
    FT(1.0),
    FT(NaN),
    FT(NaN),
)

minimal_geom_with_padding = similar(scalar_field, MinimalGeomWithPadding{FT})
make_padding() = (nothing, nothing, nothing, nothing, nothing, nothing)
@. minimal_geom_with_padding = MinimalGeomWithPadding(
    local_geom_full.J,
    local_geom_full.WJ,
    make_padding(),
)

# Create a simple wrapper using NamedTuple
simplified_geom = (J=J_field,)

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
println("\nNone field overhead testing:")
println("  TwoFieldWithNothingGeom:      $(sizeof(TwoFieldWithNothingGeom{FT})) bytes")
println("  FourFieldPartiallyNothingGeom: $(sizeof(FourFieldPartiallyNothingGeom{FT})) bytes")
println("  FullGeomWithOptionals:        $(sizeof(FullGeomWithOptionals{FT})) bytes (FT sentinel)")
println("  MinimalGeomWithPadding:       $(sizeof(MinimalGeomWithPadding{FT})) bytes")
println("\nNote: CUDA typically inlines structs < 128 bytes effectively")
println("Note: sizeof(Nothing) = $(sizeof(Nothing)) bytes")

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

# Pointwise function: access Jacobian via element LocalGeometry
suite["2b_pointwise_lg_j"] = @benchmarkable begin
    @. $result_field = $scalar_field + lg_j_direct($local_geom_full)
    CUDA.synchronize()
end

# Pointwise function: add a couple of stack frames
suite["2c_pointwise_lg_j_stack"] = @benchmarkable begin
    @. $result_field = $scalar_field + lg_j_stack2($local_geom_full)
    CUDA.synchronize()
end

# Pointwise function: force a non-inlined call boundary
suite["2d_pointwise_lg_j_noinline"] = @benchmarkable begin
    @. $result_field = $scalar_field + lg_j_noinline($local_geom_full)
    CUDA.synchronize()
end

# Broadcast f(x.lg) vs anonymous function wrapping
suite["2f_f_x_lg"] = @benchmarkable begin
    @. $result_field = f_from_lg($wrapper_lg_field.lg)
    CUDA.synchronize()
end

suite["2g_lambda_f_x_lg"] = @benchmarkable begin
    $result_field .= (x -> f_from_lg(x.lg)).($wrapper_lg_field)
    CUDA.synchronize()
end

suite["2h_f_x_lg_noinline"] = @benchmarkable begin
    @. $result_field = f_from_lg_noinline($wrapper_lg_field.lg)
    CUDA.synchronize()
end

suite["2i_lambda_f_x_lg_noinline"] = @benchmarkable begin
    $result_field .= (x -> f_from_lg_noinline(x.lg)).($wrapper_lg_field)
    CUDA.synchronize()
end

# Finite-difference: explicitly call LocalGeometry(space, idx, hidx)
suite["2e_fd_localgeom_constructor"] = @benchmarkable begin
    CUDA.@sync @cuda threads = $fd_threads blocks = $fd_blocks fd_localgeom_kernel!(
        $fd_out,
        $fd_in,
        $fd_space,
        $fd_nv,
    )
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
# SECTION 2B: Nothing Field Overhead Testing
# ========================================
println("\nSECTION 2B: Testing overhead of nothing fields vs simplified structs")

# Two fields with explicit nothing padding
suite["9b_two_field_with_nothing"] = @benchmarkable begin
    @. $result_field = $scalar_field + $two_field_with_nothing.J
    CUDA.synchronize()
end

# Four fields where some are nothing
suite["9c_four_field_partial_nothing"] = @benchmarkable begin
    @. $result_field = $scalar_field + $four_field_partially_nothing.J
    CUDA.synchronize()
end

# Full geometry with optional fields (using Union types)
suite["9d_full_geom_with_optionals"] = @benchmarkable begin
    @. $result_field = $scalar_field + $full_geom_with_optionals.J
    CUDA.synchronize()
end

# Minimal geometry with nothing padding tuple
suite["9e_minimal_with_padding"] = @benchmarkable begin
    @. $result_field = $scalar_field + $minimal_geom_with_padding.J
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

# ========================================
# SECTION 4: Fixed Expression, Varying Struct (isolate memory effects)
# ========================================
println("\nSECTION 4: Fixed Expression with Varying Struct Sizes (isolate memory effects)")

# We apply the EXACT SAME EXPRESSION to different struct types
# This isolates memory bandwidth effects from compilation effects
# All structs have a .J field, so @. result_field = scalar_field + struct.J is valid

suite["4_1_scalar_with_2field"] = @benchmarkable begin
    @. $result_field = $scalar_field + $two_field_geom.J
    CUDA.synchronize()
end

suite["4_2_scalar_with_4field"] = @benchmarkable begin
    @. $result_field = $scalar_field + $four_field_geom.J
    CUDA.synchronize()
end

suite["4_3_scalar_with_8field"] = @benchmarkable begin
    @. $result_field = $scalar_field + $eight_field_geom.J
    CUDA.synchronize()
end

suite["4_4_scalar_with_16field"] = @benchmarkable begin
    @. $result_field = $scalar_field + $sixteen_field_geom.J
    CUDA.synchronize()
end

suite["4_5_scalar_with_full_lg"] = @benchmarkable begin
    @. $result_field = $scalar_field + $local_geom_full.J
    CUDA.synchronize()
end

# ========================================
# SECTION 5: Fixed Struct, Varying Expression Complexity (isolate compilation effects)
# ========================================
println("\nSECTION 5: Fixed Struct (LocalGeometry) with Varying Expression Complexity")

# Start with full LocalGeometry, gradually add more terms
# This isolates broadcast compilation complexity from memory bandwidth

suite["5_1_expr_1term"] = @benchmarkable begin
    @. $result_field = $scalar_field + $local_geom_full.J
    CUDA.synchronize()
end

suite["5_2_expr_2terms"] = @benchmarkable begin
    @. $result_field = $scalar_field + $local_geom_full.J + $local_geom_full.WJ * 0.1
    CUDA.synchronize()
end

suite["5_3_expr_3terms"] = @benchmarkable begin
    @. $result_field = $scalar_field + $local_geom_full.J + $local_geom_full.WJ * 0.1 + $local_geom_full.invJ * 0.01
    CUDA.synchronize()
end

suite["5_4_expr_4terms"] = @benchmarkable begin
    @. $result_field = $scalar_field * ($local_geom_full.J + $local_geom_full.WJ * 0.1 + $local_geom_full.invJ * 0.01 + $scalar_field * 0.001)
    CUDA.synchronize()
end

suite["5_5_expr_6terms"] = @benchmarkable begin
    @. $result_field = ($scalar_field + $local_geom_full.J) *
                       ($local_geom_full.WJ + $local_geom_full.invJ) *
                       (1.0 + $scalar_field * 0.01)
    CUDA.synchronize()
end

# ========================================
# SECTION 6: Stencil-like Operations (chained geometry access)
# ========================================
println("\nSECTION 6: Stencil-like Operations (chained LocalGeometry operations)")

# These tests simulate what would happen in real stencil operations
# where LocalGeometry is accessed multiple times in nested operations

# Create a helper function to test recursive operations
@inline function apply_lg_twice_inlined(x, lg)
    return x + lg.J * lg.WJ
end

@noinline function apply_lg_twice_noinline(x, lg)
    return x + lg.J * lg.WJ
end

@inline function apply_lg_three_times_inlined(x, lg)
    return x + lg.J * lg.WJ * lg.invJ
end

@noinline function apply_lg_three_times_noinline(x, lg)
    return x + lg.J * lg.WJ * lg.invJ
end

# Simple chained access (both inlined)
suite["6_1_chained_lg_2accesses_inlined"] = @benchmarkable begin
    @. $result_field = $scalar_field + apply_lg_twice_inlined($scalar_field, $local_geom_full)
    CUDA.synchronize()
end

# Chained access with noinline boundary
suite["6_2_chained_lg_2accesses_noinline"] = @benchmarkable begin
    @. $result_field = $scalar_field + apply_lg_twice_noinline($scalar_field, $local_geom_full)
    CUDA.synchronize()
end

# Three-way chained access (inlined)
suite["6_3_chained_lg_3accesses_inlined"] = @benchmarkable begin
    @. $result_field = $scalar_field + apply_lg_three_times_inlined($scalar_field, $local_geom_full)
    CUDA.synchronize()
end

# Three-way chained access with noinline boundary
suite["6_4_chained_lg_3accesses_noinline"] = @benchmarkable begin
    @. $result_field = $scalar_field + apply_lg_three_times_noinline($scalar_field, $local_geom_full)
    CUDA.synchronize()
end

# Simulate the complex recursive pattern from Dennis and Teja's findings
# Multiple products combined (mimics the project_for_mul issue)
@inline function complex_lg_computation_inlined(lg::LocalGeometry{FT}) where {FT}
    # Simulate: diag1 * diag2 * diag3 + diag4 * diag5 * diag6 pattern
    # Using LG fields instead of actual matrices
    term1 = lg.J * lg.WJ * lg.invJ
    term2 = lg.J * lg.WJ * lg.invJ
    return term1 + term2
end

@noinline function complex_lg_computation_noinline(lg::LocalGeometry{FT}) where {FT}
    term1 = lg.J * lg.WJ * lg.invJ
    term2 = lg.J * lg.WJ * lg.invJ
    return term1 + term2
end

suite["6_5_complex_multi_product_inlined"] = @benchmarkable begin
    @. $result_field = $scalar_field - complex_lg_computation_inlined($local_geom_full)
    CUDA.synchronize()
end

suite["6_6_complex_multi_product_noinline"] = @benchmarkable begin
    @. $result_field = $scalar_field - complex_lg_computation_noinline($local_geom_full)
    CUDA.synchronize()
end

# Run benchmarks with tuning
println("\nRunning benchmarks (this takes ~3-5 minutes due to additional sections)...\n")
results = run(suite, verbose=true, samples=30)

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

section1_keys = filter(k -> startswith(k, "1_") || startswith(k, "2_") || startswith(k, "2b_") || startswith(k, "2c_") || startswith(k, "2d_") || startswith(k, "2e_") || startswith(k, "2f_") || startswith(k, "2g_") || startswith(k, "2h_") || startswith(k, "2i_") || startswith(k, "3_") || startswith(k, "4_") || startswith(k, "5_"), collect(keys(results)))
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

section2_keys = filter(k -> (startswith(k, "6_") || startswith(k, "7_") || startswith(k, "8_") || startswith(k, "9_")) && !startswith(k, "9b_") && !startswith(k, "9c_") && !startswith(k, "9d_") && !startswith(k, "9e_"), collect(keys(results)))
for key in sort(section2_keys)
    result = results[key]
    time_μs = minimum(result.times) / 1000
    overhead = 100 * (time_μs - baseline_time / 1000) / (baseline_time / 1000)
    overhead_str = overhead >= 0 ? @sprintf("+%.1f", overhead) : @sprintf("%.1f", overhead)
    display_name = replace(key, r"^\d+_" => "")
    @printf("  %-30s %10.2f μs  (%6s%% vs baseline)\n", display_name, time_μs, overhead_str)
end

println("\n" * repeat("-", 70))
println("SECTION 2B: Nothing Field Overhead Testing")
println(repeat("-", 70))
println("\nExecution Time (μs, lower is better):")

section2b_keys = filter(k -> startswith(k, "9b_") || startswith(k, "9c_") || startswith(k, "9d_") || startswith(k, "9e_"), collect(keys(results)))
for key in sort(section2b_keys)
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

println("\n" * repeat("-", 70))
println("SECTION 4: Fixed Expression, Varying Struct (isolate memory effects)")
println(repeat("-", 70))
println("\nExecution Time (μs, lower is better):")
println("(Same expression @. result_field = scalar_field + struct.J, varying struct size)")

section4_keys = filter(k -> startswith(k, "4_1_") || startswith(k, "4_2_") || startswith(k, "4_3_") || startswith(k, "4_4_") || startswith(k, "4_5_"), collect(keys(results)))
baseline_4 = minimum(results["4_1_scalar_with_2field"].times) / 1000
for key in sort(section4_keys)
    result = results[key]
    time_μs = minimum(result.times) / 1000
    overhead = 100 * (time_μs - baseline_4) / baseline_4
    overhead_str = overhead >= 0 ? @sprintf("+%.1f", overhead) : @sprintf("%.1f", overhead)
    display_name = replace(key, r"^\d+_" => "")
    # Extract struct size info
    struct_info = ""
    if contains(key, "2field")
        struct_info = " (2 fields, 16 bytes)"
    elseif contains(key, "4field")
        struct_info = " (4 fields, 32 bytes)"
    elseif contains(key, "8field")
        struct_info = " (8 fields, 64 bytes)"
    elseif contains(key, "16field")
        struct_info = " (16 fields, 128 bytes)"
    elseif contains(key, "full_lg")
        struct_info = " (Full LocalGeometry)"
    end
    @printf("  %-30s %10.2f μs  (%6s%% vs min)  %s\n", display_name, time_μs, overhead_str, struct_info)
end

println("\n" * repeat("-", 70))
println("SECTION 5: Fixed Struct (LocalGeometry), Varying Expression Complexity")
println(repeat("-", 70))
println("\nExecution Time (μs, lower is better):")
println("(Same LocalGeometry field, varying number of terms and operations)")

section5_keys = filter(k -> startswith(k, "5_1_") || startswith(k, "5_2_") || startswith(k, "5_3_") || startswith(k, "5_4_") || startswith(k, "5_5_"), collect(keys(results)))
baseline_5 = minimum(results["5_1_expr_1term"].times) / 1000
for key in sort(section5_keys)
    result = results[key]
    time_μs = minimum(result.times) / 1000
    overhead = 100 * (time_μs - baseline_5) / baseline_5
    overhead_str = overhead >= 0 ? @sprintf("+%.1f", overhead) : @sprintf("%.1f", overhead)
    display_name = replace(key, r"^\d+_" => "")
    # Extract complexity info
    complexity_info = ""
    if contains(key, "1term")
        complexity_info = " (1 term)"
    elseif contains(key, "2terms")
        complexity_info = " (2 terms)"
    elseif contains(key, "3terms")
        complexity_info = " (3 terms)"
    elseif contains(key, "4terms")
        complexity_info = " (4 terms, 1 multiply)"
    elseif contains(key, "6terms")
        complexity_info = " (6 terms, nested mults)"
    end
    @printf("  %-30s %10.2f μs  (%6s%% vs baseline)  %s\n", display_name, time_μs, overhead_str, complexity_info)
end

println("\n" * repeat("-", 70))
println("SECTION 6: Stencil-like Operations (chained LocalGeometry access)")
println(repeat("-", 70))
println("\nExecution Time (μs, lower is better):")
println("(Multiple accesses to LocalGeometry fields in single operation)")

section6_keys = filter(k -> startswith(k, "6_"), collect(keys(results)))
baseline_6 = minimum(results["1_baseline_simple"].times) / 1000
for key in sort(section6_keys)
    result = results[key]
    time_μs = minimum(result.times) / 1000
    overhead = 100 * (time_μs - baseline_6) / baseline_6
    overhead_str = overhead >= 0 ? @sprintf("+%.1f", overhead) : @sprintf("%.1f", overhead)
    display_name = replace(key, r"^\d+_" => "")
    @printf("  %-30s %10.2f μs  (%6s%% vs scalar baseline)\n", display_name, time_μs, overhead_str)
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

# Section 4 analysis: Same expression, varying struct
section4_times = [
    minimum(results["4_1_scalar_with_2field"].times) / 1000,
    minimum(results["4_2_scalar_with_4field"].times) / 1000,
    minimum(results["4_3_scalar_with_8field"].times) / 1000,
    minimum(results["4_4_scalar_with_16field"].times) / 1000,
    minimum(results["4_5_scalar_with_full_lg"].times) / 1000,
]
section4_min = minimum(section4_times)
section4_max = maximum(section4_times)
section4_memory_variance = 100 * (section4_max - section4_min) / section4_min

# Section 5 analysis: Same struct, varying expression
section5_times = [
    minimum(results["5_1_expr_1term"].times) / 1000,
    minimum(results["5_2_expr_2terms"].times) / 1000,
    minimum(results["5_3_expr_3terms"].times) / 1000,
    minimum(results["5_4_expr_4terms"].times) / 1000,
    minimum(results["5_5_expr_6terms"].times) / 1000,
]
section5_min = minimum(section5_times)
section5_max = maximum(section5_times)
section5_compilation_variance = 100 * (section5_max - section5_min) / section5_min

# Section 6 analysis: Stencil operations
section6_times = [
    minimum(results["6_1_chained_lg_2accesses_inlined"].times) / 1000,
    minimum(results["6_2_chained_lg_2accesses_noinline"].times) / 1000,
    minimum(results["6_3_chained_lg_3accesses_inlined"].times) / 1000,
    minimum(results["6_4_chained_lg_3accesses_noinline"].times) / 1000,
    minimum(results["6_5_complex_multi_product_inlined"].times) / 1000,
    minimum(results["6_6_complex_multi_product_noinline"].times) / 1000,
]

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

println("\n4. MEMORY EFFECT ANALYSIS (Section 4: Same expression, varying struct):")
println("   Memory variance across struct sizes: $(@sprintf("%.1f", section4_memory_variance))%")
if section4_memory_variance > 10
    println("   ⚠️  SIGNIFICANT MEMORY EFFECTS: struct size impacts performance")
    println("      Indicates inefficient memory access patterns or cache effects")
else
    println("   ✓ MINIMAL MEMORY EFFECTS: struct size has negligible impact")
    println("      Good memory access locality")
end

println("\n5. BROADCAST COMPILATION EFFECT (Section 5: Same struct, varying complexity):")
println("   Compilation variance across expression complexity: $(@sprintf("%.1f", section5_compilation_variance))%")
if section5_compilation_variance > 20
    println("   ⚠️  SIGNIFICANT COMPILATION EFFECTS: expression complexity matters")
    println("      More complex broadcasts → larger generated kernels")
    println("      May cause register pressure or reduced occupancy")
elseif section5_compilation_variance > 5
    println("   ⚠️  MODERATE COMPILATION EFFECTS: noticeable with complex expressions")
    println("      Keep broadcast expressions concise when possible")
else
    println("   ✓ MINIMAL COMPILATION EFFECTS: ClimaCore broadcasts compile efficiently")
end

println("\n6. STENCIL OPERATION ANALYSIS (Section 6: Chained operations):")
inlining_gap_2accesses = minimum(results["6_2_chained_lg_2accesses_noinline"].times) / minimum(results["6_1_chained_lg_2accesses_inlined"].times)
inlining_gap_3accesses = minimum(results["6_4_chained_lg_3accesses_noinline"].times) / minimum(results["6_3_chained_lg_3accesses_inlined"].times)
println("   Inlining impact (2 accesses):     $(@sprintf("%.2f", inlining_gap_2accesses))x slower with noinline")
println("   Inlining impact (3 accesses):     $(@sprintf("%.2f", inlining_gap_3accesses))x slower with noinline")
if maximum(section6_times) / minimum(section6_times) > 1.5
    println("   ⚠️  INLINING CRITICAL FOR STENCIL OPS: Large performance gap detected")
    println("      Ensure LocalGeometry access functions are inlined")
    println("      Avoid @noinline annotations on geometry-accessing functions")
else
    println("   ✓ Stencil operations handle LocalGeometry efficiently")
end

println("\n" * "="^70)
println("EFFECT DECOMPOSITION SUMMARY")
println("="^70)

println("\nThis benchmark isolates four distinct effects:")
println("\n1. MEMORY BANDWIDTH (Section 4):")
println("   Impact: $(@sprintf("%.1f", section4_memory_variance))%")
if section4_memory_variance > 10
    println("   → SIGNIFICANT: Memory is a limiting factor")
else
    println("   → MINIMAL: Computation dominates over memory")
end

println("\n2. BROADCAST COMPILATION (Section 5):")
println("   Impact: $(@sprintf("%.1f", section5_compilation_variance))%")
if section5_compilation_variance > 20
    println("   → SIGNIFICANT: Compiler struggles with complex expressions")
else
    println("   → MINIMAL: ClimaCore handles complex expressions well")
end

println("\n3. INLINING DECISIONS (Section 6):")
avg_inlining_impact = (inlining_gap_2accesses + inlining_gap_3accesses) / 2
println("   Impact: $(@sprintf("%.2f", avg_inlining_impact))x")
if avg_inlining_impact > 1.3
    println("   → CRITICAL: Inlining failures cause significant slowdowns")
else
    println("   → MANAGEABLE: Inlining works reasonably well")
end

println("\n4. OVERALL LOCALGEOMETRY OVERHEAD (Section 1 & 2):")
println("   Direct access impact: $(@sprintf("%.1f", lg_overhead_pct))%")
if lg_overhead_pct > 20
    println("   → HIGH: Consider design changes")
elseif lg_overhead_pct > 5
    println("   → MODERATE: Worth optimizing if in hot paths")
else
    println("   → LOW: No major optimization needed")
end

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

    section1_keys = [
        "1_baseline_simple",
        "2_full_lg_jacobian",
        "2b_pointwise_lg_j",
        "2c_pointwise_lg_j_stack",
        "2d_pointwise_lg_j_noinline",
        "2e_fd_localgeom_constructor",
        "2f_f_x_lg",
        "2g_lambda_f_x_lg",
        "2h_f_x_lg_noinline",
        "2i_lambda_f_x_lg_noinline",
        "3_full_lg_multiple",
        "4_extracted_j",
        "5_simplified_lg",
    ]
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
    section3_keys = ["10_vector_baseline", "11_project_full_lg", "12_multiple_scalar_access"]
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
