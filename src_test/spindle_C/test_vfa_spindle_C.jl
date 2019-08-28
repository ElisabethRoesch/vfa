push!(LOAD_PATH, "/Users/eroesch/github")
using vfa, Plots, Optim, Dates, DiffEqParamEstim, Flux, DiffEqFlux, Statistics, LinearAlgebra, OrdinaryDiffEq
using BSON: @save, @load
##################
CCNB1, pMPS1, pENSA = get_spindle_C_data()
plt_path = plot_path(CCNB1, pMPS1, pENSA)
grid_ranges = [range(0, step=0.1, stop=1), range(0, step=0.1, stop=1), range(0, step=0.1, stop=1)]
coord_list = get_coord_list(grid_ranges)
##################
@load "src_test/spindle_C/spindle_C.bson" dudt
species = ["CCNB1", "pMPS1", "pENSA"]
tspan = (0.,1.)
a = CCNB1[:,2]
b = pMPS1[:,2]
c = pENSA[:,2]
ode_data = transpose(hcat( a,b,c))
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

print("h")
#plt = scatter(t, ode_data[1,:], label = string("Observation: ", species[1]), grid = "off", ylim=[0,1.3])
#scatter!(t, ode_data[2,:], label = string("Observation: ", species[2]))
#scatter!(t, ode_data[3,:], label = string("Observation: ", species[3]))
#plot!(t, Flux.data(pred[1,:]), label = string("Prediction: ", species[1]))
#plot!(t, Flux.data(pred[2,:]), label = string("Prediction: ", species[2]))
#plot!(t, Flux.data(pred[3,:]), label = string("Prediction: ", species[3]))
