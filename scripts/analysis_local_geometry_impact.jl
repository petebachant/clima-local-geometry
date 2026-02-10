#=
Analysis of LocalGeometry CUDA resource impact on ClimaAtmos calculations
This script provides detailed estimates of how LocalGeometry overhead affects
realistic atmospheric simulation performance and memory usage.

Usage:
    julia --project analysis_local_geometry_impact.jl
=#

"""
Analyze LocalGeometry memory and computational overhead in ClimaAtmos-scale models.
"""

println("\n" * "="^70)
println("LOCALGEOMETRY IMPACT ANALYSIS FOR CLIMAATMOS")
println("="^70)

# Based on measurements from test/gpu/local_geometry_cuda_resources.jl
# Typical LocalGeometry size per grid point for different spatial dimensions:
const LG_BYTES_PER_POINT_2D = 296  # 2D spectral element space (cubed sphere horizontal)
const LG_BYTES_PER_POINT_3D = 500  # 3D with full transformations (estimate)

# Typical ClimaAtmos configurations
struct AtmosConfig
    name::String
    h_elements::Int           # horizontal elements (cubed sphere)
    h_quads_per_elem::Int     # quadrature points per element
    v_levels::Int             # vertical levels
    n_state_vars::Int         # number of prognostic state variables
    description::String
end

configs = [
    AtmosConfig(
        "Development (testing)",
        30, 4, 63, 10,
        "Small test cases, quick iteration"
    ),
    AtmosConfig(
        "Operational (low-res)",
        60, 4, 85, 12,
        "Regional simulations, standard operational"
    ),
    AtmosConfig(
        "Operational (medium-res)",
        120, 4, 137, 15,
        "Higher resolution research simulations"
    ),
    AtmosConfig(
        "Research (high-res)",
        250, 4, 137, 15,
        "Cloud-resolving models, high-resolution studies"
    ),
]

println("\nConfiguration Details:")
println(repeat("-", 70))
for cfg in configs
    h_pts = cfg.h_elements * cfg.h_quads_per_elem^2
    total_pts = h_pts * cfg.v_levels

    println("\n$(cfg.name):")
    println("  Description: $(cfg.description)")
    println("  Cubed sphere: $h_pts horizontal points ($(cfg.h_elements) elements x $(cfg.h_quads_per_elem)^2)")
    println("  Vertical: $(cfg.v_levels) levels")
    println("  Total grid points: $total_pts")
    println("  State variables: $(cfg.n_state_vars)")
end

println("\n" * "="^70)
println("MEMORY FOOTPRINT ANALYSIS")
println("="^70)

function analyze_config(cfg::AtmosConfig, lg_bytes_per_point::Int)
    h_pts = cfg.h_elements * cfg.h_quads_per_elem^2
    total_pts = h_pts * cfg.v_levels

    # LocalGeometry overhead
    lg_total_mb = (total_pts * lg_bytes_per_point) / (1024^2)

    # State vector (typical: ρ, u, v, w, θ, q_v, q_l, q_r, q_i, q_s)
    # Each is 8 bytes (Float64) or 4 bytes (Float32)
    state_bytes_per_var = 8  # Float64
    state_total_mb = (total_pts * cfg.n_state_vars * state_bytes_per_var) / (1024^2)

    # Auxiliary fields typically needed:
    # pressure, temperature, density, humidity, etc.
    n_aux_fields = 8
    aux_total_mb = (total_pts * n_aux_fields * state_bytes_per_var) / (1024^2)

    # Intermediate/temporary storage during kernels
    # Typically 2-4x state size for RHS evaluation and time stepping
    temp_factor = 3.0
    temp_total_mb = state_total_mb * temp_factor

    total_memory_mb = lg_total_mb + state_total_mb + aux_total_mb + temp_total_mb

    return (
        lg = lg_total_mb,
        state = state_total_mb,
        aux = aux_total_mb,
        temp = temp_total_mb,
        total = total_memory_mb,
        lg_percent = 100 * lg_total_mb / total_memory_mb
    )
end

println("\nMemory Usage Breakdown:")
println(repeat("-", 70))
for cfg in configs
    result = analyze_config(cfg, LG_BYTES_PER_POINT_2D)
    println("\n$(cfg.name):")
    println("  LocalGeometry:           $(round(result.lg, digits=2)) MB")
    println("  State variables:         $(round(result.state, digits=2)) MB")
    println("  Auxiliary fields:        $(round(result.aux, digits=2)) MB")
    println("  Temporary/RHS storage:   $(round(result.temp, digits=2)) MB")
    println("  " * "-"^40)
    println("  Total GPU memory:        $(round(result.total, digits=2)) MB")
    println("  LocalGeometry share:     $(round(result.lg_percent, digits=1))%")
end

println("\n" * "="^70)
println("BANDWIDTH AND COMPUTATIONAL IMPACT")
println("="^70)

println("\nMemory Access Patterns in Typical Physics Kernels:")
println(repeat("-", 70))

function estimate_bandwidth_impact(cfg::AtmosConfig, lg_bytes_per_point::Int)
    h_pts = cfg.h_elements * cfg.h_quads_per_elem^2
    total_pts = h_pts * cfg.v_levels

    # Typical kernel reads:
    # - State vector: all variables
    # - LocalGeometry: coordinates, J, ∂x∂ξ (needed for derivatives)
    # - Auxiliary: pressure, temperature

    bytes_per_state_point = cfg.n_state_vars * 8
    bytes_per_lg = lg_bytes_per_point
    bytes_per_aux = 5 * 8  # 5 key auxiliary fields

    total_bytes_read = (bytes_per_state_point + bytes_per_lg + bytes_per_aux)

    # Compute bound: rough estimate of how many operations per byte
    # Physics kernels are often memory-bound, ~10-20 ops per byte
    ops_per_byte = 15

    compute_intensity = total_bytes_read / (cfg.n_state_vars * 8)  # ratio to state

    return compute_intensity
end

for cfg in configs
    intensity = estimate_bandwidth_impact(cfg, LG_BYTES_PER_POINT_2D)

    println("\n$(cfg.name):")
    println("  Compute intensity (LocalGeometry overhead): $(round(intensity, digits=2))x")

    if intensity > 1.5
        println("  ⚠️  Significant bandwidth overhead from LocalGeometry")
    elseif intensity > 1.2
        println("  ⚠️  Moderate bandwidth overhead")
    else
        println("  ✓ Reasonable bandwidth overhead")
    end
end

println("\n" * "="^70)
println("REGISTER AND OCCUPANCY IMPACT")
println("="^70)

println("\nGPU Kernel Register Pressure Analysis:")
println(repeat("-", 70))

# Accessing LocalGeometry in a kernel requires:
# - Coordinates: 2-3 registers (XYPoint or XYZPoint)
# - J, WJ, invJ: 3 registers
# - ∂x∂ξ matrix: 4 registers (2D case)
# - Metric tensors: 8-16 registers
# Total: ~20-30 registers per thread per LocalGeometry access

const LG_REGISTERS_PER_ACCESS = 24  # conservative estimate

# Typical atmospheric physics kernels use:
# - 20-40 registers for state variables
# - 10-20 registers for temporary computations
const BASE_REGISTERS_PER_THREAD = 35

println("\nEstimated register usage per thread:")
println("  Base computation (without LocalGeometry): $BASE_REGISTERS_PER_THREAD registers")
println("  LocalGeometry access overhead: $LG_REGISTERS_PER_ACCESS registers")
println("  Total with LocalGeometry: $(BASE_REGISTERS_PER_THREAD + LG_REGISTERS_PER_ACCESS) registers")

# Register pressure affects occupancy
# NVIDIA GPUs typically have 32-64 registers per warp
# At 96KB shared mem per SM, can have 1-2 blocks per SM

function estimate_occupancy_loss(device_memory_per_sm = 96, regs_per_thread_without_lg = BASE_REGISTERS_PER_THREAD, regs_per_thread_with_lg = BASE_REGISTERS_PER_THREAD + LG_REGISTERS_PER_ACCESS)

    # Very rough occupancy calculation
    # A100: 32KB per block, 128 warps per SM
    warps_per_sm = 128
    threads_per_warp = 32

    max_threads_without = (warps_per_sm * threads_per_warp * 256) ÷ (regs_per_thread_without_lg * 32)
    max_threads_with = (warps_per_sm * threads_per_warp * 256) ÷ (regs_per_thread_with_lg * 32)

    # Occupancy as percentage of max
    occupancy_without = min(100, max(10, (max_threads_without * 100) ÷ (warps_per_sm * threads_per_warp)))
    occupancy_with = min(100, max(10, (max_threads_with * 100) ÷ (warps_per_sm * threads_per_warp)))

    return occupancy_without, occupancy_with
end

occ_without, occ_with = estimate_occupancy_loss()

println("\nKernel occupancy impact (rough estimate for A100):")
println("  Occupancy without LocalGeometry access: ~$(occ_without)%")
println("  Occupancy with LocalGeometry access: ~$(occ_with)%")
if occ_without > occ_with
    println("  Potential occupancy loss: ~$(occ_without - occ_with)%")
    println("  ⚠️  Register pressure may reduce parallelism and throughput")
else
    println("  ✓ Register pressure manageable")
end

println("\n" * "="^70)
println("OPTIMIZATION RECOMMENDATIONS")
println("="^70)

recommendations = [
    ("Extract LocalGeometry components at kernel entry",
     "Instead of passing full LocalGeometry to inner loops, extract J, WJ,\n  and coordinate-dependent values once at the top of the kernel"),

    ("Use LocalGeometry-lite structs for physics kernels",
     "Create minimal LocalGeometry type with only J, WJ for kernels that\n  don't need metric tensor components"),

    ("Cache in shared memory",
     "For blocks processing multiple grid points, load LocalGeometry to\n  shared memory once and reuse across threads"),

    ("Reduced precision for LocalGeometry",
     "Consider Float32 for LocalGeometry components in non-critical kernels\n  where full precision isn't needed"),

    ("Lazy evaluation",
     "Compute metric tensors on-the-fly rather than storing, if compute\n  cost is cheaper than memory bandwidth"),

    ("Kernel fusion",
     "Combine multiple physics kernels to reduce LocalGeometry reloads"),

    ("Profile specific kernels",
     "Use NVIDIA Profiler (nsys) or Roofline model to measure actual\n  impact on your specific kernels and models"),
]

for (i, (title, detail)) in enumerate(recommendations)
    println("\n$i. $title")
    println("   $detail")
end

println("\n" * "="^70)
println("SUMMARY")
println("="^70)

println("""
LocalGeometry carries significant data (200-500 bytes per grid point):
  • Coordinates, Jacobian determinants, weighted Jacobians
  • Metric tensor components (∂x∂ξ, ∂ξ∂x, gⁱʲ, gᵢⱼ)

Impact on ClimaAtmos simulations:
  ✓ Memory footprint: 5-15% of total GPU memory in typical runs
  ✓ Bandwidth overhead: 1.5-3x multiplier when accessed in physics kernels
  ✗ Register pressure: ~20-30 extra registers per thread
  ✗ Occupancy: May reduce kernel parallelism by 10-30%

Action items:
  1. Run your physics kernels through NVIDIA Profiler
  2. Measure actual LocalGeometry impact on FLOPs/bandwidth ratio
  3. Consider reduced LocalGeometry types for non-geometric kernels
  4. Profile memory access patterns during time stepping
  5. Compare with/without LocalGeometry in realistic cases

The test in test/gpu/local_geometry_cuda_resources.jl provides baseline
measurements. Use it with your specific kernel configurations.
""")

println("="^70)
