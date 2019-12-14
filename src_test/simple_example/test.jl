
using Flux, DiffEqFlux, OrdinaryDiffEq, DiffEqParamEstim, Plots, Optim, Dates
using BSON: @save

u0 = Float32[2.; 0.]
datasize = 30
tspan = (0.0f0, 3.f0)
t = range(tspan[1], tspan[2], length = datasize)
function trueODEfunc(du, u, p, t)
  true_A = [-0.1 2.0; -2.0 -0.1]
  du .= ((u.^3)'true_A)'
end
prob = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob,Tsit5(),saveat=t))
scatter(t, ode_data[1,:], label="Observation: species 1", grid = "off")
scatter!(t, ode_data[2,:], label="Observation: species 2", xlab = "time", ylab="Species")
