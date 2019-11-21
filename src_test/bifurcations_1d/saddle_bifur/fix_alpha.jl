push!(LOAD_PATH, "/home/elisabeth/github/")
using vfa
using  DifferentialEquations, Plots, Optim, Dates, DiffEqParamEstim, Flux, DiffEqFlux, Statistics, LinearAlgebra, OrdinaryDiffEq
using BSON: @save, @load
datasize = 35
alpha, tspan, solver = -2.0,(0,2.0),Tsit5()
t = range(tspan[1], tspan[2], length = datasize)
function run_saddle_one_u0(u0)
    x0 = [u0]
    function saddle_1d(dx, x, p, t)
        dx[1] = alpha-x[1]*x[1]
    end
    prob = ODEProblem(saddle_1d, x0 ,tspan)
    obs = Array(solve(prob, solver,saveat=t))
    return obs[1,:]
end

function run_saddle_multi_u0(u0s)
    obs =[]
    for i in u0s
        push!(obs,run_saddle_one_u0(i))
    end
    obs
end

train_u0s = [-2.,-1.,0.,1.0,2.0]
ode_data = run_saddle_multi_u0(train_u0s)
plot(Array(range(1,stop = length(ode_data[1]))),ode_data[1])
plot!(Array(range(1,stop = length(ode_data[2]))),ode_data[2])
plot!(Array(range(1,stop = length(ode_data[3]))),ode_data[3])
plot!(Array(range(1,stop = length(ode_data[4]))),ode_data[4])
plot!(Array(range(1,stop = length(ode_data[5]))),ode_data[5])




dudt = Chain(Dense(1,15,tanh),
       Dense(15,15,tanh),
       Dense(15,1))
ps = Flux.params(dudt)
n_ode = x->neural_ode(dudt, x, tspan, Tsit5(), saveat=t, reltol=1e-7, abstol=1e-9)
n_epochs = 200
data1 = Iterators.repeated((), n_epochs)
#opt1 = ADAM(0.0001)
opt1 = Descent(0.0005)
#L2_loss_fct() = sum(abs2,ode_data .- n_ode(u0))+sum(abs2,ode_data2 .- n_ode(u02))
function euclidean_distance_ode(x1::AbstractArray{<:AbstractArray, 1}, x2::AbstractArray{<:AbstractArray, 1})
    x1 = hcat(x1...)
    x2 = hcat(x2...)
    # println(length(x1))
    # &length(x1)==1001
    if (length(x1)==length(x2))
        sum([vecnorm(x1[i, :] - x2[i, :]) for i=1:size(x1, 1)])
    else
        Inf
    end
end
function L2_loss_fct()
    sum = 0.0
    counter = 0
    for i in train_u0s
        counter = counter+1
        s = LinearAlgebra.norm(ode_data[counter[1]].-n_ode([i]))
        sum=sum+s
    end
    return sum
end

cb1 = function ()
    println(Tracker.data(L2_loss_fct()))
end
test_u0s = [-3.,-2.5,-2.,-1.5,-1.,-0.5,0.,0.5,1.,1.5,2.,2.5,-3.]
preds = []
for i in test_u0s
    pred = Flux.data(n_ode([i]))
    push!(preds, pred[1,:])
end


plot(Array(range(1,stop = datasize)),preds[1])
plot!(Array(range(1,stop = datasize)),preds[2])
plot!(Array(range(1,stop = datasize)),preds[3])
plot!(Array(range(1,stop = datasize)),preds[4])
plot!(Array(range(1,stop = datasize)),preds[5])
plot!(Array(range(1,stop = datasize)),preds[6])
plot!(Array(range(1,stop = datasize)),preds[7])
plot!(Array(range(1,stop = datasize)),preds[8])
plot!(Array(range(1,stop = datasize)),preds[9])
plot!(Array(range(1,stop = datasize)),preds[10])
plot!(Array(range(1,stop = datasize)),preds[11])
plot!(Array(range(1,stop = datasize)),preds[12])
plot!(Array(range(1,stop = datasize)),preds[13])
# train n_ode with collocation method
@time Flux.train!(L2_loss_fct, ps, data1, opt1, cb = cb1)

derivs = []
for i in test_u0s
    d = dudt([i])
    push!(derivs,Flux.data(d)[1])
end
a = test_u0s.+ derivs
plot([1,2],[test_u0s[1], a[1]],label ="", color ="blue", grid =:off)
plot!([1,2],[test_u0s[2], a[2]],label ="", color ="blue")
plot!([1,2],[test_u0s[3], a[3]],label ="", color ="blue")
plot!([1,2],[test_u0s[4], a[4]],label ="", color ="blue")
plot!([1,2],[test_u0s[5], a[5]],label ="", color ="blue")
plot!([1,2],[test_u0s[6], a[6]],label ="", color ="blue")
plot!([1,2],[test_u0s[7], a[7]],label ="", color ="blue")
plot!([1,2],[test_u0s[8], a[8]],label ="", color ="blue")
plot!([1,2],[test_u0s[9], a[9]],label ="", color ="blue")
plot!([1,2],[test_u0s[10], a[10]],label ="", color ="blue")
plot!([1,2],[test_u0s[11], a[11]],label ="", color ="blue")
plot!([1,2],[test_u0s[12], a[12]],label ="", color ="blue")
plot!([1,2],[test_u0s[13], a[13]],label ="", color ="blue")
#hline!([-sqrt(alpha),sqrt(alpha)], label ="",color ="red")
#savefig("saddle_alpha_-2.pdf")
#@save "saddle_bifur_alpha_-2.bson" dudt
