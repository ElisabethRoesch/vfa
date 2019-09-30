using Plots
# Here I am using Plots' 3d quiver plot.
function plot_quiver(coord_list, grad_list, species)
    plt = quiver(coord_list[1,:], coord_list[2,:], coord_list[3,:], quiver = (grad_list[1,:],grad_list[2,:], grad_list[3,:]), xlab = "CCNB1", ylab = "pMPS1", zlab  = "pENSA", projection="3d")
    return plt
end
# Same but with observed data.
function plot_quiver(coord_list, grad_list, species, a, b, c)
    plt = quiver(coord_list[1,:], coord_list[2,:], coord_list[3,:], quiver = (grad_list[1,:],grad_list[2,:], grad_list[3,:]), xlab = "CCNB1", ylab = "pMPS1", zlab  = "pENSA", projection="3d")
    plot!(a, b, c, label = "", grid = "off")
    scatter!(a, b, c, label = "Path", legend = :bottomleft)
    return plt
end
function plot_quiver(coord_list, grad_list, species, a, b, c)
    plt = quiver(coord_list, quiver = (grad_list[1]), xlab = "CCNB1", ylab = "pMPS1", zlab  = "pENSA", projection="3d")
    plot!(a, b, c, label = "", grid = "off")
    scatter!(a, b, c, label = "Path", legend = :bottomleft)
    return plt
end
