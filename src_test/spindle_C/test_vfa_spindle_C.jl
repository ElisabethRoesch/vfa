push!(LOAD_PATH, "/Users/eroesch/github")
using vfa, Plots, Optim, Dates, DiffEqParamEstim, Flux, DiffEqFlux, Statistics, LinearAlgebra, OrdinaryDiffEq
using BSON: @save, @load
################## Get observed data.
CCNB1, pMPS1, pENSA = get_spindle_C_data()
plt_path = plot_path(CCNB1[:,2], pMPS1[:,2], pENSA[:,2])
display(plt_path)
################## Set points to get gradients of neural ODE.
grid_ranges = [range(0, step = 0.1, stop = 1), range(0, step = 0.1, stop = 1), range(0, step = 0.1, stop = 1)]
coord_list = get_coord_list(grid_ranges)
################## Get neural ODE model.
@load "src_test/spindle_C/spindle_C.bson" dudt
species = ["CCNB1", "pMPS1", "pENSA"]
tspan = (0.,1.)
a = CCNB1[:,2]
b = pMPS1[:,2]
c = pENSA[:,2]
ode_data = transpose(hcat(a, b, c))
u0 = ode_data[:,1]
if(CCNB1[1]>0)
    t = CCNB1[:,1].-(CCNB1[1])
else
    t = CCNB1[:,1].+abs(CCNB1[1])
end
t = t./t[end]
n_ode = x->neural_ode(dudt, x, tspan, Rosenbrock23(autodiff=false), saveat=t, reltol=1e-7, abstol=1e-9)
pred = n_ode(u0)
grad_list, end_coord_list = get_grad_and_end_point_list(coord_list, dudt)
################## Prediction on training data, species over time.
plt_train = plot_train(t, ode_data, pred, species)
display(plt_train)
savefig("train_plot.pdf")
################## Visualise vector field via quiver with observed path (no time dim).
plt_quiver = plot_quiver(coord_list, grad_list, species, a, b, c)
display(plt_quiver)
savefig("quiver_plot.pdf")
