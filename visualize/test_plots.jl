function test_plots(ts, x, predics)
    plt = plot(ts, x, color = "red", label= "train")
    for i in 1:length(predics)
        if i==1
            plot!(ts, predics[i][1,:],color="blue", label= "test")
        else
            plot!(ts, predics[i][1,:],color="blue", label= "")

        end

    end
    return plt
end
