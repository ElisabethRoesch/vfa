
push!(LOAD_PATH, "/Users/eroesch/github")
using vfa, GR
using  DifferentialEquations
using Plots, Optim, Dates, DiffEqParamEstim, Flux, DiffEqFlux, Statistics, LinearAlgebra, OrdinaryDiffEq
using BSON: @save, @load
