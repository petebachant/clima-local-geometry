ENV["CLIMACOMMS_DEVICE"] = "CUDA"

import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import YAML

config_file = abspath(joinpath(@__DIR__, "..", "config", "climaatmos_progedmf_1m.yml"))
if !isfile(config_file)
    error("Config file not found: $config_file")
end

job_id = get(ENV, "CLIMAATMOS_JOB_ID", "progedmf_1m")

steps = 120

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

config_dict = YAML.load_file(config_file)
dt_seconds = parse_duration_seconds(string(config_dict["dt"]))
t_end_seconds = steps * dt_seconds
config_dict["t_end"] = string(t_end_seconds) * "secs"

# Convert relative toml paths to absolute paths relative to ClimaAtmos.jl-mod
if haskey(config_dict, "toml")
    climaatmos_dir = abspath(joinpath(@__DIR__, "..", "ClimaAtmos.jl-mod"))
    toml_paths = config_dict["toml"]
    if toml_paths isa Vector
        config_dict["toml"] = [joinpath(climaatmos_dir, p) for p in toml_paths]
    else
        config_dict["toml"] = joinpath(climaatmos_dir, toml_paths)
    end
end

config = CA.AtmosConfig(config_dict; job_id)
simulation = CA.get_simulation(config)
sol_res = CA.solve_atmos!(simulation)
CA.error_if_crashed(sol_res.ret_code)

if !isnothing(sol_res.sol)
    CA.verify_callbacks(sol_res.sol.t)
end

@info "Simulation complete" job_id = simulation.job_id output_dir = simulation.output_dir
