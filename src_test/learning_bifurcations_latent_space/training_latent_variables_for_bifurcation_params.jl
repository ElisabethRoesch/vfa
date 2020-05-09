push!(LOAD_PATH, "/Users/eroesch/github")
using vfa, Plots, Optim, Dates, DiffEqParamEstim, Flux, DiffEqFlux, Statistics, LinearAlgebra, OrdinaryDiffEq
using BSON: @save
using DataFrames


dudt = Chain(Dense(1,15,tanh),
       Dense(15,15,tanh),
       Dense(15,1))
