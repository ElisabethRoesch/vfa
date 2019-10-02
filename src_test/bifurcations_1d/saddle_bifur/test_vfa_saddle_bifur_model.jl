push!(LOAD_PATH, "/Users/eroesch/github")
using vfa, Plots, Optim, Dates, DiffEqParamEstim, Flux, DiffEqFlux, Statistics, LinearAlgebra, OrdinaryDiffEq
using BSON: @save, @load
################## Get observed data.
saddle_bifur = get_saddle_data()
plt_path = plot_path_1d(saddle_bifur[:,2])
display(plt_path)
################## Set points to get gradients of neural ODE.
grid_range = [range(-1, step = 0.1, stop = 1)]
coord_list = get_coord_list_1d(grid_range)
################## Get neural ODE model.
@load "src_test/saddle_bifur/saddle_bifur.bson" dudt
species = ["System undergoing Saddle-node bifurcation"]
x = saddle_bifur[:,2]
st = length(x)
ode_data = transpose(hcat(x[1:st]))
u0 = [x[1]]
tspan = (0.0f0, 3.f0)
t = range(tspan[1], tspan[2], length = st)
n_ode = x->neural_ode(dudt, x, tspan, Rosenbrock23(autodiff=false), saveat=t, reltol=1e-7, abstol=1e-9)
pred = n_ode(u0)
grad_list, end_coord_list = get_grad_and_end_point_list_1d(coord_list, dudt)
################## Prediction on training data, species over time.
plt_train = plot_train_1d(t, ode_data, pred, species)
display(plt_train)
savefig("train_plot.pdf")
################## Visualise vector field via quiver with observed path (no time dim).
list_inits = [[0.0], [0.1], [0.2], [0.3]]
predics = dostuff(n_ode, coord_list)
ts = saddle_bifur[:,1]
test_plt = test_plots(ts, x, predics)
display(test_plt)
savefig("test_plot.pdf")
