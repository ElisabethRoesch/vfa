module vfa #Plots, Optim, Dates, DiffEqParamEstim, Flux, DiffEqFlux, Statistics,
#Plots
    using Flux, Plots
    export get_spindle_C_data, plot_path, get_coord_list, get_grad_and_end_point_list
    include("vfa_methods.jl")
    include("../data/spindle_C.jl")
    include("../visualize/pre_ana.jl")
end
