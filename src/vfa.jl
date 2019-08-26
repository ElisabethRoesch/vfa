module vfa #Plots, Optim, Dates, DiffEqParamEstim, Flux, DiffEqFlux, Statistics,
    export vfa_1, spindle_C
    include("vfa_methods.jl")
    include("../data/spindle_C.jl")
end
