# CliMA LocalGeometry

Assessing the impact of a simpler LocalGeometry struct on CUDA kernel
performance.

## Submodule branches

- `ClimaCore.jl`: `pb/localgeo`
- `ClimaAtmos.jl`: `pb/rm-nvtx`
- `ClimaAtmos.jl-mod`: `pb/precip-precomp`

## Running Julia Jupyter notebooks interactively on `clima` with a GPU

First, start an `srun` bash terminal with one GPU requested, then run:

```sh
calkit nb check-kernel -e main
calkit jupyter lab --notebook-dir notebooks --ip=0.0.0.0 --no-browser
```

Copy the server URL and paste into VS Code, then select the kernel with the
name `clima-local-geometry: main 1.11`.
This kernel will use the `main` environment with all packages up to date
and no global package loading allowed.
