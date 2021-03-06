using MultivariateStats, Flux, DiffEqFlux, OrdinaryDiffEq, DiffEqParamEstim, Plots, Optim, Dates
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
u0 = Float32[0; 2.]
datasize = 30
tspan = (0.0f0, 3f0)
t = range(tspan[1], tspan[2], length = datasize)
#label_plot = "enhance_strong"
#label_plot = "inhibit_strong"
label_plot = "negative_feedback"
function positive_feedback(du, u, p, t)
    my1, my2 = 1., 1.
    b1, b2 = 1., 1.
    v1, v2 = 1., 1.
    k12 = 1.
    k21 = 1.
    n = 1
    du[1] = b1 - u[1]my1+(v1/(1+(u[2]/k12)^n))
    du[2] = b2 - u[2]my2+(v2/(1+(u[1]/k21)^n))
    return du
end
function negative_feedback(du, u, p, t)
    my1, my2 = 1., 1.
    b1, b2 = 1., 1.
    v1, v2 = 1., 1.
    k12 = 1.
    k21 = 1.
    n = 1
    du[1] = b1 - u[1]my1-(v1/(1+(u[2]/k12)^n))
    du[2] = b2 - u[2]my2+(v2/(1+(u[1]/k21)^n))
    return du
end
prob = ODEProblem(negative_feedback, u0, tspan)
ode_data = Array(solve(prob,Tsit5(),saveat=t))
scatter(t, ode_data[1,:], label="Observation: Species 1", grid = "off",legend =:topleft)
scatter!(t, ode_data[2,:], label="Observation: Species 2", xlab = "time", ylab="Species")

as = range(-5, step = 0.8, stop = 5)
bs = range(-5, step = 0.8, stop = 5)
cords = Array{Tuple{Real,Float64},1}(undef,length(as)*length(bs))
 m = 1
for a in as
    for b in bs
        cords[m] = (a,b)
        global m = m+1
    end
end
grads = []
for i in cords
    cord = [i[1], i[2]]
    grad = positive_feedback([0.,0.], cord, 0.1, 0.1)

    tuple = (grad[1], grad[2])
    push!(grads, tuple)
end
quiv_plt=quiver(cords, size = (500,500), ylim = (-5,5) , xlim = (-5,5), title= "ositive feedback", quiver=grads, grid = :off,framestyle = :box)
# plot!(ode_data[1,:], ode_data[2,:],
#     linewidth =4, color = "red", xlab = "X", ylab = "Y", label = "",
#     legend=:bottomright, grid = "off")
display(quiv_plt)
savefig(string("plots_mech_learning/negative_feedback_quiver.pdf"))



savefig(string("plots_mech_learning/", label_plot,"_obs.pdf"))

dudt = Chain(Dense(2,50,tanh),
       Dense(50,2))
ps = Flux.params(dudt)
function node_two_stage_function(model, x, tspan, saveat, ode_data,
            args...; kwargs...)
  dudt_(du,u,p,t) = du .= model(u)
  prob_fly = ODEProblem(dudt_,x,tspan)
  two_stage_method(prob_fly, saveat, ode_data)
end
loss_n_ode = node_two_stage_function(dudt, u0, tspan, t, ode_data, Tsit5(), reltol=1e-7, abstol=1e-9)
two_stage_loss_fct()=loss_n_ode.cost_function(ps)
esti =loss_n_ode.estimated_solution
scatter(t, ode_data[1,:], label = "Observation: Species 1", grid = "off",legend =:topleft)
scatter!(t, ode_data[2,:], label = "Observation: Species 2")
scatter!(t, esti[1,:], label = "Estimation: Species 1")
scatter!(t, esti[2,:], label = "Estimation: Species 2")
# savefig(string("plots/",label_plot,"_esti.pdf"))
n_ode = x->neural_ode(dudt, x, tspan, Tsit5(), saveat=t, reltol=1e-7, abstol=1e-9)
n_ode(u0)
n_epochs = 2000
verify = 50
data1 = Iterators.repeated((), n_epochs)
opt1 = Descent(0.01)
sa = saver(n_epochs)
L2_loss_fct() = sum(abs2,ode_data .- n_ode(u0))
cb1 = function ()
    sa.count_epochs = sa.count_epochs +  1
    if mod(sa.count_epochs-1, verify)==0
        #update_saver(sa, Tracker.data(two_stage_loss_fct()),Tracker.data(L2_loss_fct()),Dates.Time(Dates.now()))
        as = range(0, step = 0.2, stop = 2)
        bs = range(0, step = 0.2, stop = 2)
        cords = Array{Tuple{Real,Float64},1}(undef,length(as)*length(bs))
        m = 1
        for a in as
            for b in bs
                cords[m] = (a,b)
                m = m+1
            end
        end
        grads = []
        for i in cords
            cord = [i[1], i[2]]
            grad = Flux.data(dudt(cord))
            tuple = (grad[1], grad[2])
            push!(grads, tuple)
        end
        quiv_plt=quiver(cords, size = (500,500), quiver=grads, grid = :off,framestyle = :box)
        plot!(ode_data[1,:], ode_data[2,:],
            linewidth =4, color = "red", xlab = "X", ylab = "Y", label = "",
            legend=:bottomright, grid = "off")
        display(quiv_plt)
        update_saver(sa, Tracker.data(two_stage_loss_fct()),0,Dates.Time(Dates.now()))
        # println("\"",Tracker.data(two_stage_loss_fct()),"\" \"",Dates.Time(Dates.now()),"\";")
    else
        update_saver(sa, Tracker.data(two_stage_loss_fct()),0,Dates.Time(Dates.now()))
        println("\"",Tracker.data(two_stage_loss_fct()),"\" \"",Dates.Time(Dates.now()),"\";")
    end
end
@time Flux.train!(two_stage_loss_fct, ps, data1, opt1, cb = cb1)

pred = n_ode(u0)
scatter(t, ode_data[1,:], label = "Observation: Species 1", grid = "off",legend =:topleft)
scatter!(t, ode_data[2,:], label = "Observation: Species 2")
plot!(t, Flux.data(pred[1,:]), label = "Prediction: Species 1")
plot!(t, Flux.data(pred[2,:]), label = "Prediction: Species 2")
# savefig(string("plots/",label_plot,"_pred.pdf"))

grid_form= Array(range(-3.,stop =3,step =0.1))
list_Ys = grid_form
list_Xs = grid_form
n =length(list_Xs)
m =length(u0)
list_dX_dYs =[]
for i in list_Xs
    tempX_list=[]
    tempY_list=[]
    for j in list_Ys
        tempX = Flux.data(dudt([i, j]))[1]
        tempY = Flux.data(dudt([i, j]))[2]
        push!(tempX_list,tempX)
        push!(tempY_list,tempY)
    end
    push!(list_dX_dYs,[tempX_list,tempY_list])
end
#get PCA
all = Array{Float64,2}(undef,2,n*n)
for i in 1:n
    r_t = hcat(convert(Array{Float64,1},list_dX_dYs[i][1]),convert(Array{Float64,1},list_dX_dYs[i][2]))
    global all[:,(i-1)*n+1:i*n] = r_t'
end
M = fit(PCA, all)
a= scatter(list_dX_dYs[1,1][1],list_dX_dYs[1,1][2], titlefontsize=7,
            grid = "off", xlab = "dX", label = "Gradients",
            title = string("PCA results: proj: ", round.(M.proj,digits=3), ", prinvars: ",
            round.(M.prinvars,digits=3), ", tprinvar: " ,round.(M.tprinvar,digits=3), " ,tvar: ",round.(M.tvar,digits=3),"."),
            ylab = "dY", legend =:bottomleft)
for i in range(2,stop=length(list_Xs))
    scatter!(list_dX_dYs[i,1][1],list_dX_dYs[i,1][2], xlab = "dX", ylab = "dY", label = "")
end
scatter!(all[1,:],all[2,:], linewidth = 3, markeralpha = 0.00006,line=:blue,reg = true, label = "with regression line")
savefig(string("plots_mech_learning/",label_plot,"_gradients.pdf"))

# Messy plotting in trasformed data space
# transformed_all = transform(M, all)
# transformed_all[1,:]
# b=scatter(all[1:61], label = "", grid=:off)
# for i in 2:n
#     rang = Array(range(1+(i-1)*n,stop=(i)*n))
#     scatter!(all[rang],label="")
# end
# display(b)
# plot!([start_p[1],end_p[1]], [start_p[2],end_p[2]],  line=:arrow,label = "PCA")
#
as = range(0, step = 0.2, stop = 2)
bs = range(0, step = 0.2, stop = 2)
cords = Array{Tuple{Real,Float64},1}(undef,length(as)*length(bs))
m = 1
for a in as
    for b in bs
        cords[m] = (a,b)
        global m = m+1
    end
end
grads = []
for i in cords
    cord = [i[1], i[2]]
    grad = Flux.data(dudt(cord))
    tuple = (grad[1], grad[2])
    push!(grads, tuple)
end
quiv_plt=quiver(cords, size = (500,500), quiver=grads, grid = :off,framestyle = :box)
plot!(ode_data[1,:], ode_data[2,:],
    linewidth =4, color = "red", xlab = "X", ylab = "Y", label = "",
    legend=:bottomright, grid = "off")
display(quiv_plt)
