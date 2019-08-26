push!(LOAD_PATH, "/Users/eroesch/github")
using vfa, Plots, Optim, Dates, DiffEqParamEstim, Flux, DiffEqFlux, Statistics, LinearAlgebra, OrdinaryDiffEq
using BSON: @save, @load
##################
CCNB1, pMPS1, pENSA = get_spindle_C_data()
plt = plot_path(CCNB1, pMPS1, pENSA)
grid_ranges = [range(0, step=0.1, stop=1), range(0, step=0.1, stop=1), range(0, step=0.1, stop=1)]
coord_list = get_coord_list(grid_ranges)
##################
@load "../data/spindle_C.bson" dudt
pred = n_ode(u0)
scatter(t, ode_data[1,:], label = string("Observation: ", species[1]), grid = "off", ylim=[0,1.3])
scatter!(t, ode_data[2,:], label = string("Observation: ", species[2]))
scatter!(t, ode_data[3,:], label = string("Observation: ", species[3]))
plot!(t, Flux.data(pred[1,:]), label = string("Prediction: ", species[1]))
plot!(t, Flux.data(pred[2,:]), label = string("Prediction: ", species[2]))
plot!(t, Flux.data(pred[3,:]), label = string("Prediction: ", species[3]))
grad_list, end_coord_list = get_grad_and_end_point_list(coord_list, dudt)
##################
