push!(LOAD_PATH, "/Users/eroesch/github")
using vfa, Plots, Optim, Dates, DiffEqParamEstim, Flux, DiffEqFlux, Statistics, LinearAlgebra, OrdinaryDiffEq
using BSON: @save, @load
################## Get observed data.
# saddle_bifur = get_saddle_data()
# plt_path = plot_path_1d(saddle_bifur[:,2])
# display(plt_path)
################## Set points to get gradients of neural ODE.
grid_range = [range(-1, step = 0.1, stop = 1)]
coord_list = get_coord_list_1d(grid_range)
################## Get neural ODE model.
@load "src_test/pitchfork_bifur/pitchfork_bifur.bson" dudt
species = ["System undergoing Pitchfork bifurcation"]
bif_type, alpha, init = 3,9,2
alpha2, init2 = 9,7
x = read_bifurfile(bif_type, alpha, init)
x2 = read_bifurfile(bif_type, alpha2, init2)
st = length(x)
ode_data = x
ode_data2 = x2
u0 = [x[1]]
u02 = [x2[1]]
tspan = (0.0f0, 3.f0)
t = range(tspan[1], tspan[2], length = st)
n_ode = x->neural_ode(dudt, x, tspan, Rosenbrock23(autodiff=false), saveat=t, reltol=1e-7, abstol=1e-9)
pred = n_ode(u0)
pred2 = n_ode(u02)
################## Prediction on training data, species over time.
plt_train = plot_train_1d(t, ode_data, pred, species)
display(plt_train)
savefig("train_plot.pdf")
################## Visualise vector field via quiver with observed path (no time dim).
predics = dostuff(n_ode, coord_list)
saddle_bifur = get_saddle_data()
ts = saddle_bifur[:,1]
test_plt = test_plots(ts, x, predics)
#plot(ts, predics[2][1,:],color="blue", label= "test")
#display(test_plt)
#savefig("test_plot.pdf")
