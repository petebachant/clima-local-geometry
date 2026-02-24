#!/bin/bash
#
# SLURM wrapper for running LocalGeometry benchmark on GPU cluster
#
# This script:
# - Sets up the Julia environment
# - Loads necessary modules (CUDA, Julia)
# - Runs the benchmark script
#

set -e  # Exit on error

echo "=========================================="
echo "LocalGeometry CUDA Benchmark - SLURM Job"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date)"
echo ""

module purge
module load climacommon/2025_05_15

# Set Julia project to the main environment subdirectory
export JULIA_PROJECT=".calkit/envs/main"

# CUDA device setup
export CLIMACOMMS_DEVICE="CUDA"

# Set environmental variable for julia to not use global packages for
# reproducibility
export JULIA_LOAD_PATH=@:@stdlib

# Instantiate julia environment, precompile, and build CUDA
julia --project=$JULIA_PROJECT -e 'using Pkg; Pkg.instantiate(;verbose=true); Pkg.precompile(;strict=true); using CUDA; CUDA.precompile_runtime(); Pkg.status()'

# Display environment info
echo "Julia version:"
julia --version

echo ""
echo "CUDA devices available:"
nvidia-smi --query-gpu=name,memory.total --format=csv

echo ""
echo "Julia project: ${JULIA_PROJECT}"

# Navigate to project directory
cd "${SLURM_SUBMIT_DIR}"

echo ""
echo "Instantiating Julia environment..."
echo "=========================================="
julia --project=${JULIA_PROJECT} -e 'using Pkg; Pkg.instantiate()'

echo ""
echo "Running benchmark..."
echo "=========================================="

# Run the benchmark
julia --project=${JULIA_PROJECT} scripts/benchmark_local_geometry_impact.jl

echo ""
echo "=========================================="
echo "Benchmark complete"
echo "End time: $(date)"
echo "=========================================="
