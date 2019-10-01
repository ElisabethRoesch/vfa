function test_plots(ts, trains, predics)
    plt = plot(ts, trains[1], color = "red", label= "train")
    for i in 1:length(predics)
         if i==1
            plot!(ts, predics[i][1,:],color="blue", label= "test")
         else
             plot!(ts, predics[i][1,:],color="blue", label= "")
         end
    end
    return plt
    println("lenb ", length(predics))
    for i in 1:length(predics)
         if i==1
            plot!(ts, predics[i][1,:],color="blue", label= "test")
         else
             plot!(ts, predics[i][1,:],color="blue", label= "")
         end
    end
    return plt
end
