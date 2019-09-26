module vfa #Plots, Optim, Dates, DiffEqParamEstim, Flux, DiffEqFlux, Statistics
    using Flux, Plots
    export get_spindle_C_data, plot_path, get_coord_list, get_grad_and_end_point_list, plot_train, plot_quiver
    include("vfa_methods.jl")
    include("../data/spindle_C.jl")
    include("../visualize/pre_ana.jl")
    include("../visualize/quiver_plots.jl")
end
