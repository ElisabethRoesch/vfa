function test_plots(ts, x, predics)
    plt = plot(ts, x)
    for i in 1:length(predics)
        plot!(ts, predics[i][1,:])
    end
    return plt
end
