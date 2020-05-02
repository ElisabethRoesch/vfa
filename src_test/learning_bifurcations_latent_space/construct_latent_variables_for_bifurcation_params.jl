
push!(LOAD_PATH, "/Users/eroesch/github")
using vfa
using DifferentialEquations
using Plots, Optim, Dates, DiffEqParamEstim, Flux, DiffEqFlux, Statistics, LinearAlgebra, OrdinaryDiffEq
using BSON: @save, @load

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

alphas = [-1.5,-.8,-.5,-.2,0.,.2,.5,.8,1.5]
inits = [-1.0, -0.6, -0.4, -0.1, 0.0, 0.1, 0.4, 0.6, 1.0]
ind_alphas = [4,5,6]
ind_inits = [1,2,3,4,5,6,7,8,9]
l_a, l_i = length(ind_alphas), length(ind_inits)

# Fill array xes which contains all training data
xes = Array{Array{Float64,2},2}(undef, l_a, l_i)
counter_a = [1]
counter_i = [1]
for temp_ind_alpha in ind_alphas
    for temp_ind_init in ind_inits
        print(temp_ind_alpha, temp_ind_init)
        temp_x = read_bifurfile(bif_type, temp_ind_alpha, temp_ind_init)
        l_temp_x = length(temp_x)
        latent_space = Array{Float64,1}(undef,l_temp_x)
        latent_space[:] = Float64.(Array(range(1,l_temp_x)))
        temp_x_latent_space = vcat(temp_x, latent_space')
        xes[counter_a[1], counter_i[1]] = temp_x_latent_space
        counter_i[1]=+1
    end
    counter_a[1]=+1
end

# plot all xes
counter_a2 = [1]
counter_i2 = [1]
plt_time = plot(grid = "off", xlab = "time", ylab = "Species U")
for temp_ind_alpha in ind_alphas
    for temp_ind_init in ind_inits
        scatter!(t, xes[counter_a2[1], counter_i2[1]][1,:], label = "")
        counter_i2=+1
    end
    counter_a2=+1
end
display(plt_time)

# plot xes for same init in 2d
plt_one_init_2d = plot(grid = "off", xlab = "time", ylab = "Species U", title = "One init, multiple alpha")
for temp_ind_alpha in ind_alphas
    temp_ind_init = 3
    scatter!(t, xes[temp_ind_alpha, temp_ind_init][1,:], label = "")
end
display(plt_one_init_2d)

# plot xes for same init in 3d
plt_one_init_3d = plot(grid = "off", xlab = "time", ylab = "Species U")
for temp_ind_alpha in ind_alphas
    temp_ind_init = 3
    plot!(t, xes[temp_ind_alpha, temp_ind_init][1,:], xes[temp_ind_alpha, temp_ind_init][2,:], label = "")
end
display(plt_one_init_3d)


plt_state_space = plot(grid = "off", xlab = "time", ylab = "Species U")
latet_variable = [5,7,8,4,1,6,3,2,9]
scatter!(inits, latet_space,label = "")
display(plt_state_space)
