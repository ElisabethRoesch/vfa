function plot_path(a, b, c)
    plt=plot(a, b, c, xlab = "CCNB1", ylab = "pMPS1", zlab  = "pENSA", grid = "off", legend = :bottomright)
    scatter!(a, b, c)
    return plt
end
