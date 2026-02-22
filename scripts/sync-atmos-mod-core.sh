#!/usr/bin/env bash
# Sync the atmos mod submodule buildkite project to the latest commit on the
# ClimaCore.jl submodule.

# Get the latest commit hash from the ClimaCore.jl submodule
latest_commit=$(cd ClimaCore.jl && git rev-parse HEAD)

# Update the Julia environment to use ClimaCore.jl at that commit
julia --project=ClimaAtmos.jl-mod/.buildkite/ \
    -e "using Pkg; Pkg.add(PackageSpec(url=\"https://github.com/CliMA/ClimaCore.jl\", rev=\"${latest_commit}\"))"

# Git commit and push
cd ClimaAtmos.jl-mod && git commit -am "Sync to ClimaCore.jl at ${latest_commit}"
git push
