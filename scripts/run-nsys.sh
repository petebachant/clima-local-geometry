#!/bin/bash
#SBATCH --gpus=1

# Parse --output option
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 --output=<nsys_output_prefix> julia --project=<project_directory> [additional_args...]"
  exit 1
fi

if [[ $1 == --output=* ]]; then
  NSYS_OUTPUT_PREFIX="${1#--output=}"
  shift
else
  echo "Error: First argument must be --output=<nsys_output_prefix>"
  exit 1
fi

# Parse --project from remaining arguments for instantiation
PROJECT_DIR=""
for arg in "$@"; do
  if [[ $arg == --project=* ]]; then
    PROJECT_DIR="${arg#--project=}"
    break
  fi
done

if [ -z "$PROJECT_DIR" ]; then
  echo "Error: --project=<project_directory> is required"
  exit 1
fi

# Ensure the output prefix parent directory exists
OUTPUT_DIR=$(dirname "$NSYS_OUTPUT_PREFIX")
mkdir -p "$OUTPUT_DIR"

# Load modules
module purge
module load climacommon/2025_05_15

# Set environment variables for GPU usage
export CLIMACOMMS_DEVICE=CUDA
export CLIMA_NAME_CUDA_KERNELS_FROM_STACK_TRACE=true
export CLIMA_LOG_MIN_GEOM=1
export CLIMA_LOG_MIN_GEOM_LIMIT=1000
export CLIMA_LOG_MIN_GEOM_FULL_LIMIT=1000

# Set environmental variable for julia to not use global packages for
# reproducibility
export JULIA_LOAD_PATH=@:@stdlib

# Instantiate julia environment, precompile, and build CUDA
julia --project=$PROJECT_DIR -e 'using Pkg; Pkg.instantiate(;verbose=true); Pkg.precompile(;strict=true); using CUDA; CUDA.precompile_runtime(); Pkg.status()'

# Run nsys
nsys profile \
    --start-later=true \
    --capture-range=cudaProfilerApi \
    --kill=none \
    --trace=nvtx,mpi,cuda,osrt \
    --gpu-metrics-device=all \
    --cuda-memory-usage=true \
    --output=$NSYS_OUTPUT_PREFIX \
    "$@"

# Generate stats report
nsys stats $NSYS_OUTPUT_PREFIX.nsys-rep
