#Plot observation in state space (no time dim).
function plot_path(a, b, c)
    plt = plot(a, b, c, xlab = "CCNB1", ylab = "pMPS1", zlab  = "pENSA", label = "", grid = "off", legend = :bottomright)
    scatter!(a, b, c, label = "Path", legend= :topleft)
    return plt
end
# Plot observed data and pred over time of training data.
function plot_train(t, ode_data, pred, species)
    plt_train = scatter(t, ode_data[1,:], label = string("Observation: ", species[1]), grid = "off", ylim=[0,1.3])
    scatter!(t, ode_data[2,:], label = string("Observation: ", species[2]))
    scatter!(t, ode_data[3,:], label = string("Observation: ", species[3]))
    plot!(t, Flux.data(pred[1,:]), label = string("Prediction: ", species[1]))
    plot!(t, Flux.data(pred[2,:]), label = string("Prediction: ", species[2]))
    plot!(t, Flux.data(pred[3,:]), label = string("Prediction: ", species[3]))
    return plt_train
end
