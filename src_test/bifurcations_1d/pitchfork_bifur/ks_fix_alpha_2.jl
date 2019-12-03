
push!(LOAD_PATH, "/Users/eroesch/github")
using vfa
using  DifferentialEquations, StatsBase
using Plots, Optim, Dates, DiffEqParamEstim, Flux, DiffEqFlux, Statistics, LinearAlgebra, OrdinaryDiffEq
using BSON: @save, @load
datasize = 35

alpha, tspan, solver = 5.0,(0,2.0),Tsit5()
t = range(tspan[1], tspan[2], length = datasize)
function run_pfsuper_one_u0(u0)
    x0 = [u0]
    function pfsuper(dx, x, p, t)
        dx[1] =alpha*x[1]-x[1]*x[1]*x[1]
    end
    prob = ODEProblem(pfsuper, x0 ,tspan)
    obs = Array(solve(prob, solver,saveat=t))
    return obs[1,:]
end
function run_pfsuper_multi_u0(u0s)
    obs =[]
    for i in u0s
        push!(obs,run_pfsuper_one_u0(i))
    end
    obs
end

train_u0s = [-2.,-1.,0.,1.0,2.0]
ode_data = run_pfsuper_multi_u0(train_u0s)
plot(Array(range(1,stop = datasize)),ode_data[1])
plot!(Array(range(1,stop = datasize)),ode_data[2])
plot!(Array(range(1,stop = datasize)),ode_data[3])
plot!(Array(range(1,stop = datasize)),ode_data[4])
plot!(Array(range(1,stop = datasize)),ode_data[5])



dudt = Chain(Dense(1,15,tanh),
       Dense(15,15,tanh),
       Dense(15,1))
ps = Flux.params(dudt)
n_ode = x->neural_ode(dudt, x, tspan, Tsit5(), saveat=t, reltol=1e-7, abstol=1e-9)
n_epochs = 200
data1 = Iterators.repeated((), n_epochs)
opt1 = Descent(0.0005)
function kolmogorov_smirnov_distance(data1,data2)
            ecdf_func_1 = StatsBase.ecdf(data1)
            ecdf_func_2 = StatsBase.ecdf(data2)
            max = maximum([data1;data2])
            intervals = max/999
            ecdf_vals_1 = Array{Float64,1}(undef, 1000)
            for i in 1:1000
                        ecdf_vals_1[i]=ecdf_func_1(intervals*(i-1))
            end
            ecdf_vals_2 = Array{Float64,1}(undef, 1000)
            for i in 1:1000
                        ecdf_vals_2[i]=ecdf_func_2(intervals*(i-1))
            end
            dist = maximum(abs.(ecdf_vals_1-ecdf_vals_2))
            return dist
end


function loss_fct()
    sum = 0.0
    counter = 0
    for i in train_u0s
        counter = counter+1
        s = kolmogorov_smirnov_distance(ode_data[counter[1]], reshape(n_ode([i]),length(n_ode([i]))))
        sum=sum+s
    end
    return sum
end

cb1 = function ()
    println(Tracker.data(loss_fct()))
end
test_u0s = [-3.,-2.5,-2.,-1.5,-1.,-0.5,0.,0.5,1.,1.5,2.,2.5,-3.]
preds = []
for i in test_u0s
    pred = Flux.data(n_ode([i]))
    push!(preds, pred[1,:])
end

print(typeof(ode_data[1])

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
@time Flux.train!(loss_fct, ps, data1, opt1, cb = cb1)

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
hline!([-sqrt(alpha),0,sqrt(alpha)], label ="",color ="red")
savefig("alpha_ks_5.pdf")
@save "pitchfork_bifur_alpha_ks_5.bson" dudt
