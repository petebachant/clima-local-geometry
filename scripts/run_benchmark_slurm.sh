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

# Load required modules (adjust for your cluster)
# Uncomment and modify as needed:
# module load julia/1.12
# module load cuda/12.0
# module load gcc/11

# Set Julia project to the ClimaCore.jl subdirectory
export JULIA_PROJECT="${SLURM_SUBMIT_DIR}/ClimaCore.jl"

# CUDA device setup
export CLIMACOMMS_DEVICE="CUDA"

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
echo "Running benchmark..."
echo "=========================================="

# Run the benchmark
julia --project=ClimaCore.jl scripts/benchmark_local_geometry_impact.jl

echo ""
echo "=========================================="
echo "Benchmark complete"
echo "End time: $(date)"
echo "=========================================="
