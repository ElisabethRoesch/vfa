function get_coord_list(grid_ranges)
    coord_list = []
    for i in grid_ranges[1]
        for j in grid_ranges[2]
            for k in grid_ranges[3]
                push!(coord_list,[i,j,k])
            end
        end
    end
    return coord_list
end

function get_grad_and_end_point_list(coord_list, dudt)
    grad_list = []
    end_coord_list = []
    for i in 1:length(coord_list)
        push!(grad_list,Flux.data(dudt(coord_list[i])))
        push!(end_coord_list,coord_list[i]+grad_list[i])
    end
    return grad_list, end_coord_list
end
