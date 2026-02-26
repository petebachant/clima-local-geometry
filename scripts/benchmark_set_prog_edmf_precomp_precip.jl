"""
GPU Kernel Optimization Script for Microphysics Precipitation

Rapidly iterate on optimizing the microphysics kernel in
set_prognostic_edmf_precomputed_quantities_precipitation!

Usage in REPL:
    julia> include("scripts/benchmark_set_prog_edmf_precomp_precip.jl")
    julia> bench()  # Run benchmark
    # Edit prognostic_edmf_precomputed_quantities.jl and save
    julia> bench()  # Re-run to see improvement

Use Revise.jl for auto-reloading on file changes.
"""

# Try to load Revise for auto-reloading (optional but highly recommended)
try
    using Revise
    @info "✅ Revise.jl loaded - your code edits will auto-reload!"
catch
    @warn "⚠️  Revise.jl not available. Install with: using Pkg; Pkg.add(\"Revise\")"
    @warn "    Without Revise, you'll need to restart Julia to see code changes."
end

ENV["CLIMACOMMS_DEVICE"] = "CUDA"

import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import YAML
using CUDA
using BenchmarkTools
using Printf

# Enable verbose CUDA output to see register usage during compilation
ENV["JULIA_CUDA_VERBOSE"] = "1"

println("\n" * "="^70)
println("GPU Kernel Optimization: Microphysics Precipitation")
println("="^70)

# Setup simulation (copied from run_progedmf_1m.jl)
config_file = abspath(joinpath(@__DIR__, "..", "config", "climaatmos_progedmf_1m.yml"))
if !isfile(config_file)
    error("Config file not found: $config_file")
end

job_id = "benchmark_precip_kernel"

function parse_duration_seconds(s)
    m = match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]+)\s*$", s)
    m === nothing && error("Unsupported duration format: $s")
    value = parse(Float64, m.captures[1])
    unit = lowercase(m.captures[2])
    if unit in ("sec", "secs", "second", "seconds")
        factor = 1
    elseif unit in ("min", "mins", "minute", "minutes")
        factor = 60
    elseif unit in ("hour", "hours", "hr", "hrs")
        factor = 3600
    elseif unit in ("day", "days")
        factor = 86400
    else
        error("Unsupported duration unit: $unit")
    end
    return value * factor
end

# Load config and setup simulation
println("\n📋 Loading configuration...")
config_dict = YAML.load_file(config_file)

# Use minimal time to setup quickly
dt_seconds = parse_duration_seconds(string(config_dict["dt"]))
config_dict["t_end"] = string(dt_seconds) * "secs"  # Just run one step

println("⚙️  Creating simulation...")
config = CA.AtmosConfig(config_dict; job_id)
simulation = CA.get_simulation(config)

# Extract state for benchmarking
Y = simulation.integrator.u
p = simulation.integrator.p
t = simulation.integrator.t

# Import the function we're benchmarking
import ClimaAtmos.set_prognostic_edmf_precomputed_quantities_precipitation! as precip_kernel

println("✅ Simulation setup complete")
println("\n" * "="^70)

"""
    benchmark_microphysics_kernel(; force_recompile=false, samples=10, seconds=30)

Benchmark the microphysics precipitation kernel.

# Keyword Arguments
- `force_recompile`: Clear CUDA cache and force kernel recompilation
- `samples`: Number of benchmark samples
- `seconds`: Maximum time to spend benchmarking

# Returns
BenchmarkTools.Trial object with timing statistics
"""
function benchmark_microphysics_kernel(; force_recompile=false, samples=10, seconds=30)
    println("\n" * "="^70)
    println("Benchmarking Microphysics Kernel")
    println("="^70)

    if force_recompile
        println("\n🔄 Forcing recompilation (clearing CUDA cache)...")
        CUDA.reclaim()
    end

    # Warm-up compilation
    println("\n🔥 Warming up (compiling kernel)...")
    println("    Look for 'ptxas info : Used N registers' in output above")
    precip_kernel(Y, p, p.atmos.microphysics_model)
    CUDA.synchronize()
    println("    ✓ Compilation complete")

    # Benchmark
    println("\n⏱️  Benchmarking ($(samples) samples, max $(seconds)s)...")
    result = @benchmark begin
        precip_kernel($Y, $p, $p.atmos.microphysics_model)
        CUDA.synchronize()
    end samples=samples seconds=seconds

    # Display results
    println("\n📊 Results:")
    println("  Median time:  $(BenchmarkTools.prettytime(median(result).time))")
    println("  Min time:     $(BenchmarkTools.prettytime(minimum(result).time))")
    println("  Mean time:    $(BenchmarkTools.prettytime(mean(result).time))")
    println("  Allocations:  $(result.allocs)")
    println("  Memory:       $(BenchmarkTools.prettymemory(result.memory))")

    # Calculate estimated occupancy info
    println("\n🔍 Optimization Tips:")
    println("  • Check compilation output above for 'Used N registers'")
    println("  • Target: <128 registers for good occupancy (>50%)")
    println("  • Current best: <64 registers for optimal occupancy (>75%)")
    println("  • If >200 registers: severe degradation (<25% occupancy)")

    return result
end

"""
    compare_runs(baseline_result, current_result)

Compare two benchmark results and show speedup.
"""
function compare_runs(baseline_result, current_result)
    baseline_time = median(baseline_result).time
    current_time = median(current_result).time
    speedup = baseline_time / current_time

    println("\n" * "="^70)
    println("Performance Comparison")
    println("="^70)
    println("\nBaseline: $(BenchmarkTools.prettytime(baseline_time))")
    println("Current:  $(BenchmarkTools.prettytime(current_time))")

    if speedup > 1.0
        improvement = (speedup - 1) * 100
        println("\n🎯 SPEEDUP: $(round(speedup, digits=2))x")
        println("   ✅ $(round(improvement, digits=1))% faster!")
    elseif speedup < 1.0
        degradation = (1 - speedup) * 100
        println("\n📉 SLOWDOWN: $(round(1/speedup, digits=2))x")
        println("   ❌ $(round(degradation, digits=1))% slower")
    else
        println("\n➡️  No significant change")
    end

    return speedup
end

"""
    profile_kernel(; duration=5.0)

Run CUDA profiler on the kernel for detailed analysis.
"""
function profile_kernel(; duration=5.0)
    println("\n" * "="^70)
    println("CUDA Profiling")
    println("="^70)

    println("\n🔬 Running profiler for $(duration)s...")

    CUDA.@profile begin
        t_start = time()
        n_iters = 0
        while time() - t_start < duration
            precip_kernel(Y, p, p.atmos.microphysics_model)
            CUDA.synchronize()
            n_iters += 1
        end
        println("   Completed $n_iters iterations")
    end

    println("\n✓ Profile complete. Check CUDA profiler output above.")
end

# Convenient aliases
const bench = benchmark_microphysics_kernel
const compare = compare_runs
const profile = profile_kernel

# Print usage instructions
println("""

Quick Commands
==============

bench()              - Run benchmark (default: 10 samples)
bench(samples=20)    - Run with more samples for accuracy
bench(force_recompile=true) - Force kernel recompilation

profile()            - Run CUDA profiler for detailed analysis
profile(duration=10) - Profile for 10 seconds

Workflow
========

1. Run initial benchmark:
   > result_baseline = bench()

2. Edit: src/cache/prognostic_edmf_precomputed_quantities.jl
   (If using Revise.jl, changes auto-reload on save)

3. Re-benchmark:
   > result_new = bench()

4. Compare:
   > compare(result_baseline, result_new)

5. Repeat steps 2-4 until satisfied

Tips
====

• Watch for "ptxas info : Used N registers" during compilation
• Target <128 registers for good GPU utilization
• Each iteration should show if you're improving
• Use profile() for detailed kernel analysis
• Look for "✅ microphysics_wrappers.jl loaded" to confirm code changes
• If Revise is working, you'll see load messages when you save edits

""")

println("🚀 Running initial benchmark...\n")
initial_result = benchmark_microphysics_kernel()

println("\n" * "="^70)
println("Ready for optimization! Edit code and run bench() to iterate.")
println("="^70)
