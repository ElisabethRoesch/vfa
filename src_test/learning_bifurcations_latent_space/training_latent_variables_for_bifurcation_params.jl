push!(LOAD_PATH, "/Users/eroesch/github")
using vfa, Plots, Optim, Dates, DiffEqParamEstim, Flux, DiffEqFlux, Statistics, LinearAlgebra, OrdinaryDiffEq
using BSON: @save
using DataFrames


dudt = Chain(Dense(2,15,tanh),
       Dense(15,15,tanh),
       Dense(15,2))

tspan = (0.0f0, 3.f0)
t = range(tspan[1], tspan[2], length = st)
species = "Pitchfork Bifurcation"

#old training code
ps = Flux.params(dudt)
n_ode = x->neural_ode(dudt, x, tspan, Tsit5(), saveat=t, reltol=1e-7, abstol=1e-9)
n_epochs = 500
data1 = Iterators.repeated((), n_epochs)
#opt1 = ADAM(0.0001)
opt1 = Descent(0.0001)
L2_loss_fct() = sum(abs2,ode_data .- n_ode(u0))+sum(abs2,ode_data2 .- n_ode(u02))
# Callback function to observe two stage training.
cb1 = function ()
    println(Tracker.data(L2_loss_fct()))
end
pred = n_ode(u0)
scatter(t, ode_data[1,:], label = string("Observation: ", species1), grid = "off")
plot!(t, Flux.data(pred[1,:]), label = string("Prediction: ", species1))

# train n_ode with collocation method
@time Flux.train!(L2_loss_fct, ps, data1, opt1, cb = cb1)


# @save "src_test/pitchfork_bifur/constucted_latent_variables_pitchfork_bifur.bson" dudt
