push!(LOAD_PATH, "/Users/eroesch/github")
using vfa, Plots, Optim, Dates, DiffEqParamEstim, Flux, DiffEqFlux, Statistics, LinearAlgebra, OrdinaryDiffEq
using BSON: @save
CCNB1, pMPS1, pENSA = get_spindle_C_data()
a = CCNB1[:,2]
b = pMPS1[:,2]
c = pENSA[:,2]
ode_data = transpose(hcat( a,b,c))
u0 = ode_data[:,1]
tspan = (0.,1.)
if(CCNB1[1]>0)
    t = CCNB1[:,1].-(CCNB1[1])
else
    t = CCNB1[:,1].+abs(CCNB1[1])
end
t = t./t[end]
species = ["CCNB1", "pMPS1", "pENSA"];u0
mutable struct saver
    losses::Array{Float64,1}
    l2s::Array{Float64,1}
    times::Array{Dates.Time,1}
    count_epochs::Int128
end
function saver(n_epochs)
    losses = zeros(n_epochs)
    l2s = zeros(n_epochs)
    times = fill(Dates.Time(Dates.now()), n_epochs)
    count_epochs = 0
    return saver(losses, l2s, times, count_epochs)
end
function update_saver(saver, loss_i, l2_i, time_i)
    epoch_i = saver.count_epochs
    saver.losses[epoch_i] = loss_i
    saver.l2s[epoch_i] = l2_i
    saver.times[epoch_i] = time_i
end
dudt = Chain(Dense(3,80,tanh),
       Dense(80,80,tanh),
       Dense(80,80,tanh),
       Dense(80,3))
ps = Flux.params(dudt)
function node_two_stage_function(model, x, tspan, saveat, ode_data,
            args...; kwargs...)
  dudt_(du,u,p,t) = du .= model(u)
  prob_fly = ODEProblem(dudt_,x,tspan)
  two_stage_method(prob_fly, saveat, ode_data)
end
loss_n_ode = node_two_stage_function(dudt, u0, tspan, t, ode_data, Rosenbrock23(autodiff=false), reltol=1e-7, abstol=1e-9)
two_stage_loss_fct()=loss_n_ode.cost_function(ps)
n_ode = x->neural_ode(dudt, x, tspan, Rosenbrock23(autodiff=false), saveat=t, reltol=1e-7, abstol=1e-9)
n_epochs = 4000
verify = 10 # for <verify>th epoch the L2 is calculated
data1 = Iterators.repeated((), n_epochs)
opt1 = ADAM(0.0001)
L2_loss_fct() = sum(abs2,ode_data .- n_ode(u0))
cb1 = function ()
    sa.count_epochs = sa.count_epochs +  1
    if mod(sa.count_epochs-1, verify)==0
        update_saver(sa, Tracker.data(two_stage_loss_fct()), Tracker.data(L2_loss_fct()), Dates.Time(Dates.now()))
    else
        update_saver(sa, Tracker.data(two_stage_loss_fct()),0, Dates.Time(Dates.now()))
    end
end
sa = saver(n_epochs)
@time Flux.train!(two_stage_loss_fct, ps, data1, opt1, cb = cb1)
pred = n_ode(u0)
scatter(t, ode_data[1,:], label = string("Observation: ", species[1]), grid = "off", ylim=[0,1.3])
scatter!(t, ode_data[2,:], label = string("Observation: ", species[2]))
scatter!(t, ode_data[3,:], label = string("Observation: ", species[3]))
plot!(t, Flux.data(pred[1,:]), label = string("Prediction: ", species[1]))
plot!(t, Flux.data(pred[2,:]), label = string("Prediction: ", species[2]))
plot!(t, Flux.data(pred[3,:]), label = string("Prediction: ", species[3]))
#@save "../data/spindle_C.bson" dudt
