include(joinpath(@__DIR__, "..", "src", "Surge.jl"))
import .Surge
using JLD2, Random, Statistics, TOML
include(joinpath(@__DIR__, "..", "src", "experiment_diagnostics.jl"))
