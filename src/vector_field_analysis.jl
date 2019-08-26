module vfa
    using Plots, Optim, Dates, DiffEqParamEstim, Flux, DiffEqFlux, Statistics, LinearAlgebra
    include("vfa.jl")
    export hellowold;
end
