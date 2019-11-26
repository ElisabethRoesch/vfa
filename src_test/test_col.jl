using Flux, DiffEqFlux, OrdinaryDiffEq, DiffEqParamEstim, Plots, Optim, Dates, LinearAlgebra
using BSON: @save

mutable struct saver
    losses::Array{Float64,1}
    l2s::Array{Float64,1}
    times::Array{Dates.Time,1}
    count_epochs::Int128
end
function saver(n_epochs)
    losses = zeros(n_epochs)
    l2s = zeros(n_epochs)
    times = fill(Dates.Time(Dates.now()),n_epochs)
    count_epochs = 0
    return saver(losses,l2s,times,count_epochs)
end
function update_saver(saver, loss_i, l2_i, time_i)
    epoch_i = saver.count_epochs
    saver.losses[epoch_i] = loss_i
    saver.l2s[epoch_i] = l2_i
    saver.times[epoch_i] = time_i
end
u0 = Float32[2.; 0.]
datasize = 30
tspan = (0.0f0, 1.5f0)
t = range(tspan[1], tspan[2], length = datasize)
function trueODEfunc(du, u, p, t)
  true_A = [-0.1 2.0; -2.0 -0.1]
  du .= ((u.^3)'true_A)'
end
prob = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob,Tsit5(),saveat=t))
test_out = [1.,80.]
test_in = [1.,2.]
# du, u , p ,t

prob.f(test_out,test_in,1.,1.9)
scatter(t, ode_data[1,:], label="Observation: species 1", grid = "off")
scatter!(t, ode_data[2,:], label="Observation: species 2", xlab = "time", ylab="Species")
dudt = Chain(x -> x.^3,
       Dense(2,30,tanh),
       Dense(30,2))
ps = Flux.params(dudt)
function node_two_stage_function(model, x, tspan, saveat, ode_data,
            args0; kwargs0)
  dudt_(du,u,p,t) = du .= model(u)
  prob_fly = ODEProblem(dudt_,x,tspan)
  two_stage_method(prob_fly, saveat, ode_data)
end
loss_n_ode = node_two_stage_function(dudt, u0, tspan, t, ode_data, Tsit5(), reltol=1e-7, abstol=1e-9)
esti =loss_n_ode.estimated_solution
scatter(t, ode_data[1,:], label = "data", grid = "off")
scatter!(t, ode_data[2,:], label = "data")
scatter!(t, esti[1,:], label = "esti", grid = "off")
scatter!(t, esti[2,:], label = "esti")



#  loss function
two_stage_loss_fct()=loss_n_ode.cost_function(ps)
# Defining anonymous function for the neural ODE with the model. in: u0, out: solution with current params.
n_ode = x->neural_ode(dudt, x, tspan, Tsit5(), saveat=t, reltol=1e-7, abstol=1e-9)
n_epochs = 500
verify = 100 # for <verify>th epoch the L2 is calculated
data1 = Iterators.repeated((), n_epochs)
opt1 = Descent(0.01)
sa = saver(n_epochs)
L2_loss_fct() = sum(abs2,ode_data .- n_ode(u0))
# Callback function to observe two stage training.
cb1 = function ()
    sa.count_epochs = sa.count_epochs +  1
    if mod(sa.count_epochs-1, verify)==0
        #update_saver(sa, Tracker.data(two_stage_loss_fct()),Tracker.data(L2_loss_fct()),Dates.Time(Dates.now()))
        #println("\"",Tracker.data(two_stage_loss_fct()),"\" \"",Dates.Time(Dates.now()),"\";")
    else
        update_saver(sa, Tracker.data(two_stage_loss_fct()),0,Dates.Time(Dates.now()))
        println("\"",Tracker.data(two_stage_loss_fct()),"\" \"",Dates.Time(Dates.now()),"\";")
    end
end

# train n_ode with collocation method
@time Flux.train!(two_stage_loss_fct, ps, data1, opt1, cb = cb1)
pred = n_ode(u0)



scatter(t, ode_data[1,:], label = "data", grid = "off")
scatter!(t, ode_data[2,:], label = "data")
plot!(t, Flux.data(pred[1,:]), label = "prediction")
plot!(t, Flux.data(pred[2,:]), label = "prediction")


t = 0.2
tpoints = t
n = length(tpoints)
n=35
h = (n^(-1/5))*(n^(-3/35))*((log(n))^(-1/16))
function construct_t1(t,tpoints)
    T1 = []
    for i in 1:length(tpoints)
        push!(T1,[1 tpoints[i]-t])
    end
    foldl(vcat,T1)
end
function construct_w(t,tpoints,h,kernel_function)
    n = length(tpoints)
    W = zeros(n)
    for i in 1:n
        W[i] = kernel_function((tpoints[i]-t)/h)/h
    end
    Matrix(Diagonal(W))
end
W = construct_w(t,tpoints,h,Epanechnikov_kernel)
construct_t1()
