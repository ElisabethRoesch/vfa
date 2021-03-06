push!(LOAD_PATH, "/Users/eroesch/github")
using vfa, Plots, Optim, Dates, DiffEqParamEstim, Flux, DiffEqFlux, Statistics, LinearAlgebra, OrdinaryDiffEq
using BSON: @save
using DataFrames

#works via index
#true_alpha =  [-1.5,-.8,-.5,-.2,0.,.2,.5,.8,1.5]
#true_init = [ -1,-.6,-.4,-.1,0.,.1,.4,.6,1.]

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
species1 = "Pitchfork Bifurcation"


scatter(t, ode_data[1,:], grid = "off", xlab = "time", ylab = "Species U", label = "Observation: System undegoing Saddle node bifurcation")
scatter!(t, ode_data2[1,:], grid = "off", xlab = "time", ylab = "Species U", label = "Observation: System undegoing Saddle node bifurcation")

dudt = Chain(Dense(1,15,tanh),
       Dense(15,15,tanh),
       Dense(15,1))
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

pred = n_ode(u0)
pred2 = n_ode(u02)
scatter(t, ode_data2[1,:], label = string("Observation "), grid = "off", legend=:bottomright,xlab = "time", ylab = "Species U",)
scatter!(t, ode_data[1,:], label = string("Observation "), grid = "off", legend=:bottomright,xlab = "time", ylab = "Species U",)
plot!(t, Flux.data(pred[1,:]), label = string("Prediction "))
plot!(t, Flux.data(pred2[1,:]), label = string("Prediction "))

header = string("Collocation model")
scatter(range(1,stop = length(sa.l2s)),log.(sa.l2s),width  = 2, label = "L2 control", grid = "off")
plot!(range(1,stop = length(sa.losses)),log.(sa.losses), width  = 2, label = header, ylab = "loss", xlab = "training epoch")
# 5% of time even with l2s

pred = n_ode(u0)
plot(t, Flux.data(pred[1,:]), label = string("trianed pred "), grid = "off")
pred = n_ode([200.])
plot!(t, Flux.data(pred[1,:]), label = string("Prediction: ", species1))
# Delete readme for this!
 @save "src_test/pitchfork_bifur/pitchfork_bifur.bson" dudt
#1,20 . 20 20. 20 1, 0.0001 descent
