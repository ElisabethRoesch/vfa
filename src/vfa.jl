module vfa #Plots, Optim, Dates, DiffEqParamEstim, Flux, DiffEqFlux, Statistics
    using Flux, Plots, CSV
    export dostuff, read_bifurfile, test_plots, get_spindle_C_data, get_saddle_data, plot_path_1d, plot_path_3d, get_coord_list, get_coord_list_1d, get_grad_and_end_point_list, get_grad_and_end_point_list_1d, plot_train, plot_train_1d, plot_quiver
    include("vfa_methods.jl")
    include("../data/spindle_C.jl")
    include("../data/bifur.jl")
    include("../visualize/pre_ana.jl")
    include("../visualize/quiver_plots.jl")
    include("../visualize/test_plots.jl")
end
