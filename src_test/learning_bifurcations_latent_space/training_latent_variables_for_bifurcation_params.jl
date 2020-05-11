push!(LOAD_PATH, "/Users/eroesch/github")
using vfa, Plots, Optim, Dates, DiffEqParamEstim, Flux, DiffEqFlux, Statistics, LinearAlgebra, OrdinaryDiffEq
using BSON: @save
using DataFrames


dudt = Chain(Dense(2,15,tanh),
       Dense(15,15,tanh),
       Dense(15,2))

tspan = (0.0f0, 3.f0)
t = range(tspan[1], tspan[2], length = st)
species = "Pitchfork Bifurcation"



# @save "src_test/pitchfork_bifur/constucted_latent_variables_pitchfork_bifur.bson" dudt
